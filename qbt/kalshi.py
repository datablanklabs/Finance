"""Point-in-time Kalshi event-contract prices: market-implied macro-event odds.

Kalshi (kalshi.com) is a CFTC-regulated exchange for binary/scalar event
contracts. Unlike :mod:`qbt.macro`, which reads a macro number off FRED
*after* it's released, a handful of Kalshi's series continuously price the
market's odds on that same release *before* it happens -- this module exists
to let a strategy react to the run-up, not just the aftermath.

Liquidity survey (confirmed against the live public API, no key needed,
2026-09-19 -- see ``DEFAULT_SERIES``):

| Series (key)   | Ticker         | What it is                              | Liquidity |
|----------------|----------------|------------------------------------------|-----------|
| ``cpi``        | ``KXCPI``      | Monthly CPI MoM-change ladder (~14 strikes) | Best of the four. A finalized month's strikes show 4k-130k contracts of volume each; the front (nearest-release) month routinely reaches the tens of thousands per strike. |
| ``fed_decision``| ``KXFEDDECISION``| Hike/hold/cut-bucket markets, per FOMC meeting | Good on the next 2-3 meetings (thousands to ~80k per bucket), thin past a year out. |
| ``payrolls``   | ``KXPAYROLLS`` | Monthly nonfarm-payrolls-change ladder    | Moderate: thousands to ~20k near release, low hundreds far out. |
| ``recession``  | ``KXRECSSNBER``| Single binary, "recession this calendar year" | Thin by contract count but the one number financial press actually quotes; no ladder, so it degrades to a plain yes-price read. |

Two were checked and dropped. The legacy **``KXFED``** rate-*level* ladder
(distinct from ``KXFEDDECISION``) was almost entirely zero-volume across
every strike checked -- superseded, not a live signal. **``KXGOVTSHUTLENGTH``**
is real and occasionally important, but it's a one-off contingent event, not
a recurring scheduled release with a ladder each period -- it doesn't fit
this module's shape and isn't attempted here.

Kalshi encodes each threshold as its own market (e.g. ``KXCPI-26SEP-T0.6``
settles "did September 2026 CPI rise more than 0.6%?"), not one
continuously-updating series the way a FRED indicator is. There is no
single "CPI" price to track -- what :meth:`KalshiPanel.snapshot` derives per
event instead is the ladder's *implied confidence*: the largest single-bucket
probability mass after taking successive differences across strikes (see
:meth:`KalshiPanel._confidence`). ``KXFEDDECISION``'s five per-meeting
buckets (hike >25bp / hike 25bp / hold / cut 25bp / cut >25bp) are already
mutually exclusive outcomes rather than a cumulative threshold ladder --
each one's own price already *is* a bucket mass, no differencing needed --
so the two shapes are detected and handled separately.

The two shapes concentrate very differently, which matters for calibrating
:class:`KalshiEventRegimeFilter`. Pulling the complete live ladder for
``KXCPI-26SEP`` eight days before its release (its 14 strikes span -0.4% to
0.9% in $0.10 increments) put its single largest bucket at ~35% -- a
$0.10-wide, 14-strike ladder spreads mass thin even once the market has
mostly converged. The same day's ``KXFEDDECISION-27DEC`` ladder (5 coarse
buckets) put "hold" at ~63%. A single shared confidence floor across both
series would either never fire on CPI or fire on every FOMC meeting;
``DEFAULT_MIN_CONFIDENCE`` picks a floor per series with that gap in mind,
not a single bar copied across all of them -- and, like the ``vix``
level in :class:`~qbt.signals.MacroRegimeFilter`'s live wiring, it's a
reasoned starting point from one snapshot, not a backtested-optimal value.

Field names (``floor_strike``, ``strike_type``, ``close_time``,
``event_ticker``, the candlestick ``price``/``yes_bid``/``yes_ask`` dollar
sub-fields) are confirmed against the live ``/markets`` and
``/.../candlesticks`` endpoints (2026-09-19), not guessed. What is *not*
confirmed: authenticated (RSA-signed) requests -- every check above was an
unauthenticated GET, which is all :class:`KalshiRepository` needs for
read-only market data -- and the ``min_close_ts`` list filter and cursor
pagination shape, both used in ``_fetch_series`` per Kalshi's documented API
but not individually exercised against a multi-page pull in this session.

Requires ``pip install requests`` (present already). ``pip install
cryptography`` only if you pass ``key_id``/``private_key_path`` for signed
requests -- unnecessary for anything in this module's own fetch path; it
exists for callers who want the higher authenticated rate limit on a large
backfill. Import of both is deferred to call time, same convention as
``openbb`` elsewhere in this package.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import prune_cache, touch_cache

__all__ = ["KalshiPanel", "KalshiRepository", "DEFAULT_SERIES", "DEFAULT_MIN_CONFIDENCE"]

_ID_COLUMNS = (
    "series", "event_ticker", "market_ticker", "strike",
    "snapshot_date", "close_time", "yes_price", "volume", "open_interest",
)

# friendly name -> Kalshi series ticker. See the module docstring's
# liquidity survey for why these four and not e.g. KXFED or KXGOVTSHUTLENGTH.
DEFAULT_SERIES: dict[str, str] = {
    "cpi": "KXCPI",
    "fed_decision": "KXFEDDECISION",
    "payrolls": "KXPAYROLLS",
    "recession": "KXRECSSNBER",
}

# Per-series confidence floor for KalshiEventRegimeFilter -- see the module
# docstring for the live-ladder numbers these are reasoned from.
DEFAULT_MIN_CONFIDENCE: dict[str, float] = {
    "cpi": 0.40,
    "fed_decision": 0.65,
    "payrolls": 0.45,
    "recession": 0.75,
}

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KalshiPanel:
    """Point-in-time Kalshi event-contract prices: one row per market snapshot.

    Parameters
    ----------
    frame:
        Long/tidy frame with columns ``series`` (the friendly name, e.g.
        ``"cpi"``), ``event_ticker`` (one scheduled release, e.g.
        ``"KXCPI-26SEP"``), ``market_ticker`` (one strike/bucket within it),
        ``strike`` (the ladder threshold, NaN for a market with no numeric
        threshold -- a plain binary or a categorical bucket; see the module
        docstring), ``snapshot_date``, ``close_time``, ``yes_price``,
        ``volume``, ``open_interest``.

    Point-in-time anchor is ``snapshot_date``, the day the quote was taken --
    like :class:`~qbt.options.OptionsPanel`, a live price carries no
    disclosure lag, so there's no lag to estimate the way there is for
    :class:`~qbt.fundamentals.FundamentalsPanel` or
    :class:`~qbt.macro.MacrosPanel`.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(_ID_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"kalshi frame missing columns: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["snapshot_date"]):
            raise TypeError("snapshot_date must be datetime64")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["close_time"]):
            raise TypeError("close_time must be datetime64")

    @property
    def series_names(self) -> list[str]:
        return sorted(self.frame["series"].unique())

    def __len__(self) -> int:
        return len(self.frame)

    # -- point-in-time access ---------------------------------------------

    def as_of(self, date: pd.Timestamp) -> "KalshiPanel":
        """Return a copy containing only snapshots taken at or before ``date``.

        The look-ahead firewall, same contract as the other panels: a copy
        of the same type, structurally incapable of holding a future row.
        """
        date = pd.Timestamp(date).normalize()
        return KalshiPanel(frame=self.frame[self.frame["snapshot_date"] <= date])

    def _nearest_open_event(
        self, series: str, date: pd.Timestamp, max_age_days: int | None
    ) -> tuple[pd.DataFrame | None, int | None]:
        """Latest-known ladder for the soonest-closing not-yet-closed event
        of ``series`` as of ``date``. ``(None, None)`` if there's no such
        event, or its latest reading is older than ``max_age_days``.
        """
        date = pd.Timestamp(date).normalize()
        known = self.as_of(date).frame
        known = known[(known["series"] == series) & (known["close_time"] > date)]
        if known.empty:
            return None, None
        target_event = known.sort_values("close_time")["event_ticker"].iloc[0]
        rows = known[known["event_ticker"] == target_event]
        latest_snap = rows["snapshot_date"].max()
        if max_age_days is not None and (date - latest_snap).days > max_age_days:
            return None, None
        rows = rows[rows["snapshot_date"] == latest_snap]
        rows = rows.drop_duplicates(subset=["market_ticker"], keep="last")
        close_time = rows["close_time"].iloc[0]
        days_to_close = (close_time.normalize() - date).days
        return rows, days_to_close

    @staticmethod
    def _confidence(rows: pd.DataFrame) -> float:
        """Largest single-bucket probability mass implied by one event's
        markets, normalised so noise (a stale quote making the ladder not
        quite monotonic, or bid/ask overround) can't push it outside [0, 1].

        Three shapes, detected from the data rather than assumed:

        * One market -> plain binary. Confidence is ``max(p, 1-p)``.
        * Every market has a numeric ``strike`` -> a cumulative
          ``P(X > strike)`` ladder (Kalshi's ``strike_type="greater"``
          markets, e.g. ``cpi``/``payrolls``). Bucket masses come from
          successive differences across strikes, plus both tails.
        * Otherwise -> already-mutually-exclusive categorical outcomes
          (e.g. ``fed_decision``'s five hike/hold/cut buckets). Each
          market's own ``yes_price`` *is* a bucket mass already.
        """
        rows = rows.drop_duplicates(subset=["market_ticker"], keep="last")
        if len(rows) == 1:
            p = float(rows["yes_price"].iloc[0])
            return max(p, 1.0 - p)
        if rows["strike"].notna().all():
            g = rows.sort_values("strike")
            p = g["yes_price"].to_numpy(dtype=float)
            buckets = np.concatenate([[1.0 - p[0]], -np.diff(p), [p[-1]]])
        else:
            buckets = rows["yes_price"].to_numpy(dtype=float)
        buckets = np.clip(buckets, 0.0, None)
        total = buckets.sum()
        if total <= 0:
            return float("nan")
        return float(buckets.max() / total)

    def snapshot(self, date: pd.Timestamp, max_age_days: int | None = None) -> pd.Series:
        """Implied confidence of the nearest still-open event, per series.

        Same shape as :meth:`~qbt.macro.MacrosPanel.snapshot`: a ``Series``
        indexed by series name. A series with no open event, or none within
        ``max_age_days``, is simply absent -- not zero, not NaN -- the same
        "unknown means absent, not a reading" contract
        :class:`KalshiEventRegimeFilter` relies on.
        """
        out: dict[str, float] = {}
        for series in self.series_names:
            rows, _ = self._nearest_open_event(series, date, max_age_days)
            if rows is None:
                continue
            out[series] = self._confidence(rows)
        return pd.Series(out, dtype=float)

    def days_to_close(self, date: pd.Timestamp, max_age_days: int | None = None) -> pd.Series:
        """Calendar days to the nearest still-open event's close, per series.

        Companion to :meth:`snapshot` the same way
        :meth:`~qbt.macro.MacrosPanel.snapshot_age_days` companions
        ``snapshot`` -- what :class:`KalshiEventRegimeFilter` reads to
        decide whether a confidence reading is for an event close enough to
        matter yet.
        """
        out: dict[str, float] = {}
        for series in self.series_names:
            rows, days = self._nearest_open_event(series, date, max_age_days)
            if rows is None:
                continue
            out[series] = float(days)
        return pd.Series(out, dtype=float)

    def describe(self) -> str:
        if self.frame.empty:
            return "KalshiPanel(0 series, 0 observations)"
        return (
            f"KalshiPanel({len(self.series_names)} series, "
            f"{len(self.frame)} observations, "
            f"{self.frame['snapshot_date'].min().date()} to "
            f"{self.frame['snapshot_date'].max().date()})"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class KalshiRepository:
    """Fetch point-in-time Kalshi event-contract prices from the public API.

    Every request this class makes is an unauthenticated GET against
    Kalshi's public market-data endpoints -- confirmed live (2026-09-19) to
    return real market/candlestick data with no key. ``key_id`` /
    ``private_key_path`` add Kalshi's documented RSA-PSS request signing on
    top of that, for the higher rate limit a full-universe historical
    backfill would eventually want; nothing here requires it. See the
    module docstring for exactly what was and wasn't checked against the
    live service.

    Results are cached on disk per (series, start, end), same convention as
    :class:`~qbt.macro.MacrosRepository`.
    """

    def __init__(
        self,
        series: dict[str, str] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        key_id: str | None = None,
        private_key_path: str | None = None,
        cache_dir: str | None = ".cache/kalshi",
        timeout: float = 10.0,
    ) -> None:
        self.series = dict(series) if series else dict(DEFAULT_SERIES)
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.private_key_path = private_key_path
        self._private_key = None
        self.cache_dir = cache_dir
        self.timeout = timeout

    # -- signed-request auth (optional) -------------------------------------

    def _load_private_key(self):
        if self._private_key is not None:
            return self._private_key
        try:
            from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Signed Kalshi requests need `pip install cryptography`."
            ) from exc
        with open(self.private_key_path, "rb") as fh:
            self._private_key = serialization.load_pem_private_key(fh.read(), password=None)
        return self._private_key

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """RSA-PSS-signed headers, or ``{}`` when no key was configured --
        every endpoint this class calls works fine unauthenticated.
        """
        if not self.key_id or not self.private_key_path:
            return {}
        from cryptography.hazmat.primitives import hashes  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric import padding  # noqa: PLC0415

        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode("utf-8")
        signature = self._load_private_key().sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        }

    # -- HTTP -----------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        """One GET, JSON in, JSON out. Kept as its own method (rather than
        inlined into the callers below) so tests can stub it and exercise
        the parsing logic with no network -- see test_kalshi.py.
        """
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install requests") from exc
        headers = self._auth_headers("GET", path)
        resp = requests.get(
            self.base_url + path, params=params, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _get_paginated(self, path: str, params: dict, key: str) -> list[dict]:
        out: list[dict] = []
        cursor = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            page = self._get(path, page_params)
            out.extend(page.get(key, []))
            cursor = page.get("cursor") or None
            if not cursor:
                break
        return out

    # -- cache --------------------------------------------------------------

    def _cache_path(self, name: str, ticker: str, start: str, end: str) -> str | None:
        if not self.cache_dir:
            return None
        key = "|".join([name, ticker, start, end])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    # -- fetch ----------------------------------------------------------

    def fetch(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> KalshiPanel:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        start_s, end_s = str(start_ts.date()), str(end_ts.date())

        if not self.series:
            raise ValueError("no series configured")

        prune_cache(self.cache_dir)
        frames = []
        for name, ticker in self.series.items():
            path = self._cache_path(name, ticker, start_s, end_s)
            if path and os.path.exists(path):
                touch_cache(path)
                long = pd.read_csv(path, parse_dates=["snapshot_date", "close_time"])
            else:
                long = self._fetch_series(name, ticker, start_ts, end_ts)
                if path:
                    long.to_csv(path, index=False)
            frames.append(long)

        frame = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=list(_ID_COLUMNS))
        )
        # A series that contributed zero rows (nothing in range, or every
        # candlestick request came back empty) leaves these as object dtype
        # after concat, which KalshiPanel's constructor rejects outright --
        # coerce explicitly rather than relying on concat to infer it, the
        # same reason MacrosRepository.fetch doesn't just trust read_csv.
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        frame["close_time"] = pd.to_datetime(frame["close_time"])
        frame = frame.sort_values(["series", "event_ticker", "snapshot_date"]).reset_index(
            drop=True
        )
        return KalshiPanel(frame=frame)

    def _fetch_series(
        self, name: str, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> pd.DataFrame:
        markets = self._get_paginated(
            "/markets",
            {
                "series_ticker": ticker,
                "min_close_ts": int(start_ts.timestamp()),
                "limit": 200,
            },
            "markets",
        )
        rows: list[dict] = []
        for m in markets:
            close_time = pd.Timestamp(m["close_time"]).tz_localize(None)
            open_time = pd.Timestamp(m["open_time"]).tz_localize(None)
            if close_time < start_ts or open_time > end_ts:
                continue
            strike = m.get("floor_strike") if m.get("strike_type") == "greater" else None

            candle_start = max(open_time, start_ts)
            candle_end = min(close_time, end_ts)
            candles = self._get(
                f"/series/{ticker}/markets/{m['ticker']}/candlesticks",
                {
                    "start_ts": int(candle_start.timestamp()),
                    "end_ts": int(candle_end.timestamp()),
                    "period_interval": 1440,
                },
            )
            for c in candles.get("candlesticks", []):
                yes_price = self._candle_yes_price(c)
                if yes_price is None:
                    continue
                rows.append(
                    {
                        "series": name,
                        "event_ticker": m["event_ticker"],
                        "market_ticker": m["ticker"],
                        "strike": float(strike) if strike is not None else np.nan,
                        "snapshot_date": pd.Timestamp(
                            c["end_period_ts"], unit="s"
                        ).normalize(),
                        "close_time": close_time,
                        "yes_price": yes_price,
                        "volume": float(c.get("volume_fp", 0.0) or 0.0),
                        "open_interest": float(c.get("open_interest_fp", 0.0) or 0.0),
                    }
                )
        return pd.DataFrame(rows, columns=list(_ID_COLUMNS))

    @staticmethod
    def _candle_yes_price(candle: dict) -> float | None:
        """A day with no trades still has bid/ask quotes but an empty
        ``price`` object (confirmed live) -- fall back to the bid/ask
        midpoint rather than dropping the day, since a wide-but-quoted
        market is still a real (if less certain) reading.
        """
        price = candle.get("price") or {}
        if "close_dollars" in price:
            return float(price["close_dollars"])
        bid = (candle.get("yes_bid") or {}).get("close_dollars")
        ask = (candle.get("yes_ask") or {}).get("close_dollars")
        if bid is None or ask is None:
            return None
        return (float(bid) + float(ask)) / 2.0
