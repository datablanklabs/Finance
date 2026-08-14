"""Point-in-time options-derived indicators.

Same per-symbol shape as :mod:`qbt.fundamentals`, but the point-in-time
story here is the opposite kind of problem from fundamentals, macro, or
filings. Those all need to correct for a *disclosure lag* -- the reading
existed before the market could see it. An options chain has no such lag:
it's a live quote, exactly like a price bar, so ``as_of_date`` is simply the
snapshot date itself.

The real constraint is different and more fundamental: **free options-chain
providers only return today's live chain.** ``obb.derivatives.options.
chains`` takes a ``date`` parameter for two providers (``intrinio``,
paid; ``tmx``, Canadian exchanges only) -- for the common US-equity, no-key
path (``yfinance``, ``cboe``), there is no way to ask for last Tuesday's
chain. This is a real limitation of the underlying data market, not
something this module can paper over. Concretely, that means:

* :meth:`OptionsRepository.fetch_snapshot` gets *today's* chain, derives a
  handful of scalar indicators from it, and appends the result to a
  per-symbol on-disk archive. Called once a day (e.g. from a cron job
  alongside live trading), this organically builds a real point-in-time
  history over time -- the only honest way to get one without a paid
  historical-options vendor.
* :meth:`OptionsRepository.fetch` reads that archive back as an
  :class:`OptionsPanel`. Ask it for a date range wider than what you've
  actually archived and you get back only what exists -- it does not
  fabricate history.

Indicators derived per snapshot, each computed at the expiration nearest a
target ``dte`` (days to expiration):

* ``iv_atm_{near,far}`` -- implied vol of the strike closest to the
  underlying price, at the near-term and far-term expirations. The
  near/far spread is a standard term-structure signal (inversion often
  precedes elevated realised vol).
* ``iv_skew`` -- IV of a downside-OTM put minus IV of an upside-OTM call at
  the near-term expiration, both roughly ``otm_pct`` away from spot. A
  cruder proxy than a true delta-based 25-delta risk reversal (the
  ``yfinance`` chain doesn't carry greeks), but directional put-skew vs.
  call-skew is still informative.
* ``put_call_volume_ratio`` / ``put_call_oi_ratio`` -- at the near-term
  expiration. Classic sentiment/positioning gauges.

Requires ``pip install openbb openbb-yfinance`` (or ``openbb-cboe``/
``openbb-tradier``). Import of ``openbb`` is deferred to call time so the
rest of this package works without it installed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["OptionsPanel", "OptionsRepository", "derive_indicators"]

_ID_COLUMNS = ("symbol", "metric", "period_end", "as_of_date", "value")


def _empty_frame() -> pd.DataFrame:
    """An empty but properly-typed frame -- ``pd.DataFrame(columns=...)``
    alone leaves the date columns as ``object`` dtype, which
    ``OptionsPanel``'s own validation then rejects."""
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "metric": pd.Series(dtype=str),
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "as_of_date": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionsPanel:
    """Point-in-time options indicators: one row per (symbol, metric, snapshot).

    Parameters
    ----------
    frame:
        Long/tidy frame with columns ``symbol``, ``metric``, ``period_end``
        (here, the same as ``as_of_date`` -- a chain snapshot describes a
        single instant, not a reporting period), ``as_of_date`` (the
        snapshot date), ``value``.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(_ID_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"options frame missing columns: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["as_of_date"]):
            raise TypeError("as_of_date must be datetime64")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["period_end"]):
            raise TypeError("period_end must be datetime64")

    @property
    def symbols(self) -> list[str]:
        return sorted(self.frame["symbol"].unique())

    @property
    def metrics(self) -> list[str]:
        return sorted(self.frame["metric"].unique())

    def __len__(self) -> int:
        return len(self.frame)

    # -- point-in-time access ---------------------------------------------

    def as_of(self, date: pd.Timestamp) -> "OptionsPanel":
        """Return a copy containing only snapshots taken at or before ``date``.

        The look-ahead firewall, same contract as
        :meth:`~qbt.data.PricePanel.as_of` and
        :meth:`~qbt.fundamentals.FundamentalsPanel.as_of`. This is what the
        backtester and live runner hand to strategies.

        Same tradeoff as ``FundamentalsPanel.as_of``: a full scan of
        ``frame``, not a binary search -- reasonable given
        :class:`OptionsRepository`'s own archive is naturally small (one
        snapshot per symbol per day you actually ran it), but worth
        revisiting first if this panel ever grows unusually large.
        """
        date = pd.Timestamp(date).normalize()
        return OptionsPanel(frame=self.frame[self.frame["as_of_date"] <= date])

    def snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        """Latest known value of every indicator, per symbol, as of ``date``.

        Safe to call on an already-truncated panel.
        """
        known = self.as_of(date).frame
        if known.empty:
            return pd.DataFrame(columns=self.metrics)
        latest = known.sort_values(["as_of_date", "period_end"]).drop_duplicates(
            subset=["symbol", "metric"], keep="last"
        )
        return latest.pivot(index="symbol", columns="metric", values="value")

    def to_daily(
        self,
        dates: pd.DatetimeIndex,
        symbols: Sequence[str] | None = None,
        metrics: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Forward-fill each metric onto a daily calendar.

        For building research features over a whole backtest window at
        once. This is *not* the live/backtest decision path -- that path
        calls :meth:`as_of` bar by bar. Note the ceiling this module's
        docstring describes: forward-filling can only carry a snapshot
        forward, it can't invent snapshots you never archived, so a gap in
        the underlying archive shows up as a gap (NaN) here, not a lie.
        """
        names = list(symbols) if symbols is not None else self.symbols
        wanted = list(metrics) if metrics is not None else self.metrics
        calendar = pd.DataFrame({"as_of_date": pd.DatetimeIndex(dates)})

        out: dict[str, pd.DataFrame] = {}
        for metric in wanted:
            sub = self.frame[
                (self.frame["metric"] == metric) & (self.frame["symbol"].isin(names))
            ]
            columns: dict[str, np.ndarray] = {}
            for sym, g in sub.groupby("symbol"):
                g = g.sort_values("as_of_date").drop_duplicates(
                    "as_of_date", keep="last"
                )
                merged = pd.merge_asof(
                    calendar, g[["as_of_date", "value"]], on="as_of_date",
                    direction="backward",
                )
                columns[sym] = merged["value"].to_numpy()
            wide = pd.DataFrame(columns, index=pd.DatetimeIndex(dates))
            # Reindex against the full requested list, not a subset already
            # filtered down to what's present -- filtering first meant
            # reindex could only reorder/subset existing columns, never add
            # a missing one, so a requested symbol with zero rows for this
            # metric (a brand-new IPO, or one added mid-backtest) was
            # silently dropped from the output entirely instead of coming
            # back as an all-NaN column. A caller indexing by symbol
            # (wide[symbol]) got a KeyError instead of NaN.
            wide = wide.reindex(columns=names)
            out[metric] = wide
        return out

    def describe(self) -> str:
        if self.frame.empty:
            return "OptionsPanel(0 symbols, 0 metrics, 0 snapshots)"
        return (
            f"OptionsPanel({len(self.symbols)} symbols, {len(self.metrics)} metrics, "
            f"{len(self.frame)} readings, "
            f"{self.frame['as_of_date'].min().date()} to "
            f"{self.frame['as_of_date'].max().date()})"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def _nearest_expiration(chain: pd.DataFrame, target_dte: int) -> pd.DataFrame:
    dte = chain["dte"].to_numpy()
    idx = int(np.argmin(np.abs(dte - target_dte)))
    chosen = chain["dte"].iloc[idx]
    return chain[chain["dte"] == chosen]

def _atm_iv(leg: pd.DataFrame, spot: float) -> float:
    if leg.empty:
        return float("nan")
    nearest_strike = leg["strike"].iloc[int(np.argmin(np.abs(leg["strike"].to_numpy() - spot)))]
    at_strike = leg[leg["strike"] == nearest_strike]
    iv = at_strike["implied_volatility"].astype(float)
    iv = iv[np.isfinite(iv) & (iv > 0)]
    return float(iv.mean()) if len(iv) else float("nan")


def _iv_near_strike(leg: pd.DataFrame, option_type: str, target_strike: float) -> float:
    side = leg[leg["option_type"].astype(str).str.lower() == option_type]
    if side.empty:
        return float("nan")
    nearest = side["strike"].iloc[int(np.argmin(np.abs(side["strike"].to_numpy() - target_strike)))]
    iv = side.loc[side["strike"] == nearest, "implied_volatility"].astype(float)
    iv = iv[np.isfinite(iv) & (iv > 0)]
    return float(iv.mean()) if len(iv) else float("nan")


def derive_indicators(
    chain: pd.DataFrame,
    near_dte: int = 30,
    far_dte: int = 90,
    otm_pct: float = 0.10,
) -> dict[str, float]:
    """Pure function from a raw chain snapshot to scalar indicators.

    Exposed at module level (not just inside the repository) so it's
    testable against a hand-built chain and reusable if you archive chains
    some other way than :class:`OptionsRepository`.
    """
    if chain.empty or "underlying_price" not in chain.columns:
        return {}
    spot = float(chain["underlying_price"].iloc[0])
    if not np.isfinite(spot) or spot <= 0:
        return {}

    near = _nearest_expiration(chain, near_dte)
    far = _nearest_expiration(chain, far_dte)

    out: dict[str, float] = {
        "iv_atm_near": _atm_iv(near, spot),
        "iv_atm_far": _atm_iv(far, spot),
    }

    put_iv = _iv_near_strike(near, "put", spot * (1.0 - otm_pct))
    call_iv = _iv_near_strike(near, "call", spot * (1.0 + otm_pct))
    if np.isfinite(put_iv) and np.isfinite(call_iv):
        out["iv_skew"] = put_iv - call_iv

    vol_by_side = near.groupby(near["option_type"].astype(str).str.lower())["volume"].sum()
    oi_by_side = near.groupby(near["option_type"].astype(str).str.lower())["open_interest"].sum()
    call_vol = float(vol_by_side.get("call", 0.0))
    put_vol = float(vol_by_side.get("put", 0.0))
    call_oi = float(oi_by_side.get("call", 0.0))
    put_oi = float(oi_by_side.get("put", 0.0))
    if call_vol > 0:
        out["put_call_volume_ratio"] = put_vol / call_vol
    if call_oi > 0:
        out["put_call_oi_ratio"] = put_oi / call_oi

    return {k: v for k, v in out.items() if np.isfinite(v)}


class OptionsRepository:
    """Archive daily options-chain snapshots and read them back as a panel.

    See the module docstring for why this is split into
    :meth:`fetch_snapshot` (call this daily, e.g. from live trading or a
    cron job, to grow the archive) and :meth:`fetch` (reads whatever the
    archive currently has for the requested range) rather than a single
    ``fetch`` like the other repositories in this package -- free chain
    providers simply don't offer historical chains to fetch in one shot.
    """

    def __init__(
        self,
        provider: str = "yfinance",
        near_dte: int = 30,
        far_dte: int = 90,
        otm_pct: float = 0.10,
        cache_dir: str = ".cache/options",
    ) -> None:
        self.provider = provider
        self.near_dte = near_dte
        self.far_dte = far_dte
        self.otm_pct = otm_pct
        self.cache_dir = cache_dir

    # -- archive --------------------------------------------------------

    def _archive_path(self, symbol: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        digest = hashlib.sha256(f"{self.provider}|{symbol}".encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    def fetch_snapshot(
        self, symbols: Sequence[str], as_of: pd.Timestamp | None = None
    ) -> OptionsPanel:
        """Fetch today's chain for each symbol, derive indicators, archive them.

        ``as_of`` labels the snapshot (defaults to today); it does not ask
        the provider for a historical chain -- see the module docstring.
        """
        symbols = list(dict.fromkeys(symbols))
        as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()

        rows = []
        for sym in symbols:
            chain = self._fetch_chain_remote(sym)
            indicators = derive_indicators(
                chain, near_dte=self.near_dte, far_dte=self.far_dte, otm_pct=self.otm_pct
            )
            if indicators:
                sym_rows = pd.DataFrame(
                    [
                        {
                            "symbol": sym,
                            "metric": metric,
                            "period_end": as_of_ts,
                            "as_of_date": as_of_ts,
                            "value": value,
                        }
                        for metric, value in indicators.items()
                    ]
                )
                sym_rows["period_end"] = pd.to_datetime(sym_rows["period_end"])
                sym_rows["as_of_date"] = pd.to_datetime(sym_rows["as_of_date"])
            else:
                sym_rows = _empty_frame()
            self._append_archive(sym, sym_rows)
            rows.append(sym_rows)

        frame = (
            pd.concat(rows, ignore_index=True)
            if rows
            else _empty_frame()
        )
        return OptionsPanel(frame=frame)

    def _append_archive(self, symbol: str, new_rows: pd.DataFrame) -> None:
        if new_rows.empty:
            return
        path = self._archive_path(symbol)
        if os.path.exists(path):
            existing = pd.read_csv(path, parse_dates=["period_end", "as_of_date"])
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "metric", "as_of_date"], keep="last"
            )
        else:
            combined = new_rows
        combined.to_csv(path, index=False)

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> OptionsPanel:
        """Read back whatever :meth:`fetch_snapshot` has already archived.

        Does not hit the network and does not fabricate history for dates
        never snapshotted -- see the module docstring.
        """
        symbols = list(dict.fromkeys(symbols))
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        frames = []
        for sym in symbols:
            path = self._archive_path(sym)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, parse_dates=["period_end", "as_of_date"])
            frames.append(df)

        frame = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
        if not frame.empty:
            frame = frame[
                (frame["as_of_date"] >= start_ts) & (frame["as_of_date"] <= end_ts)
            ].sort_values(["symbol", "metric", "as_of_date"]).reset_index(drop=True)
        return OptionsPanel(frame=frame)

    def _fetch_chain_remote(self, symbol: str) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb openbb-yfinance`."
            ) from exc

        out = obb.derivatives.options.chains(symbol=symbol, provider=self.provider)
        return out.to_dataframe().reset_index()
