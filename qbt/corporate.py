"""Point-in-time corporate-filings-derived indicators.

Same shape as :mod:`qbt.fundamentals`: per-symbol, long/tidy, keyed by
``as_of_date``. What's different from the financial-statement metrics in
:class:`~qbt.fundamentals.FundamentalsPanel` is the source and the nature of
the point-in-time anchor: these indicators are engineered off two SEC event
streams -- the filing index itself (10-K/10-Q/8-K cadence) and Form 4
insider transactions -- rather than off the numbers inside a financial
statement.

The point-in-time story here is actually the cleanest of the data modules
in this package. Fundamentals and macro both need an *estimated* lag
because the only thing on record is a reporting period, not a disclosure
date. A filing has no such ambiguity: the SEC's own ``accepted_date`` (for
the filing index) or ``filing_date`` (for insider transactions, which is
when the Form 4 hit EDGAR, not the earlier ``transaction_date`` the trade
actually happened -- Section 16 gives insiders up to two business days to
report) *is* the moment it became public. ``as_of_date`` below is that
field, unmodified -- no conservative fallback lag required.

Two engineered indicator families, both trailing-window counts/sums
recomputed at every underlying event (so between events the value is a
genuine flat "level," not an artifact of forward-filling a single reading):

* Filing cadence: ``filed_{form}_count_{window}d`` -- how many filings of a
  given form type a company made in the trailing window, as of each new
  filing. A spike in 8-Ks in particular is a well-known proxy for
  "something is happening" -- restatements, executive departures, M&A --
  independent of what the filing actually says.
* Insider activity: ``insider_net_shares_{window}d``,
  ``insider_buy_count_{window}d``, ``insider_sell_count_{window}d`` --
  trailing net insider buying, as of each new Form 4. Insider net buying is
  one of the more replicated slow-moving return predictors in the
  empirical literature (Lakonishok & Lee 2001 and follow-ons): executives
  buying their own stock on the open market is a costly signal a press
  release isn't.

Requires ``pip install openbb openbb-sec``. Both source endpoints work with
the free ``sec`` provider (a thin wrapper over EDGAR's own public APIs) --
no credential needed, unlike fundamentals (fmp/intrinio) or macro (fred).
Import of ``openbb`` is deferred to call time so the rest of this package
works without it installed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import as_merge_key

__all__ = ["CorpsPanel", "CorpsRepository"]

_ID_COLUMNS = ("symbol", "metric", "period_end", "as_of_date", "value")

DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K")
DEFAULT_WINDOW_DAYS = 90


def _empty_frame() -> pd.DataFrame:
    """An empty but properly-typed frame -- ``pd.DataFrame(columns=...)``
    alone leaves the date columns as ``object`` dtype, which
    ``CorpsPanel``'s own validation then rejects."""
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
class CorpsPanel:
    """Point-in-time corporate-filings indicators: one row per (symbol, metric, event).

    Parameters
    ----------
    frame:
        Long/tidy frame with columns ``symbol``, ``metric``, ``period_end``
        (here, the same as ``as_of_date`` -- these are windowed-as-of-today
        readings, not point-in-period fundamentals), ``as_of_date`` (when
        the underlying filing became public), ``value``.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(_ID_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"corps frame missing columns: {missing}")
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

    def as_of(self, date: pd.Timestamp) -> "CorpsPanel":
        """Return a copy containing only events known at or before ``date``.

        The look-ahead firewall, same contract as
        :meth:`~qbt.data.PricePanel.as_of` and
        :meth:`~qbt.fundamentals.FundamentalsPanel.as_of`: a copy of the
        same type, structurally incapable of holding a future row. This is
        what the backtester and live runner hand to strategies.

        Same tradeoff as ``FundamentalsPanel.as_of``: a full scan of
        ``frame``, not a binary search, on the assumption an event history
        (filings, insider trades) stays small relative to when that would
        start to matter.
        """
        date = pd.Timestamp(date).normalize()
        return CorpsPanel(frame=self.frame[self.frame["as_of_date"] <= date])

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
        calls :meth:`as_of` bar by bar so a strategy can never see an event
        before it was filed.
        """
        names = list(symbols) if symbols is not None else self.symbols
        wanted = list(metrics) if metrics is not None else self.metrics
        calendar = pd.DataFrame({"as_of_date": as_merge_key(pd.DatetimeIndex(dates))})

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
                right = g[["as_of_date", "value"]].copy()
                right["as_of_date"] = as_merge_key(right["as_of_date"])
                merged = pd.merge_asof(
                    calendar, right, on="as_of_date",
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
            return "CorpsPanel(0 symbols, 0 metrics, 0 events)"
        return (
            f"CorpsPanel({len(self.symbols)} symbols, {len(self.metrics)} metrics, "
            f"{len(self.frame)} events, "
            f"{self.frame['as_of_date'].min().date()} to "
            f"{self.frame['as_of_date'].max().date()})"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def _trailing_count(as_of: pd.Series, window_days: int) -> pd.Series:
    """Count of events in the trailing ``window_days`` ending at each event.

    Requires ``as_of`` sorted ascending. Uses a time-based rolling window,
    so it's correct even when events aren't evenly spaced.

    ``closed="both"``, not pandas' own default of ``"right"``: an
    offset-based rolling window is otherwise the half-open interval
    ``(t - window_days, t]``, which excludes an event landing *exactly*
    ``window_days`` before another one from that later event's own count.
    "Trailing N days" reads as inclusive of both ends in the ordinary
    sense, and there's no reason a same-instant boundary hit should be the
    one case silently dropped.
    """
    s = pd.Series(1.0, index=pd.DatetimeIndex(as_of))
    return s.rolling(f"{window_days}D", closed="both").sum().to_numpy()


def _trailing_sum(as_of: pd.Series, values: pd.Series, window_days: int) -> pd.Series:
    s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(as_of))
    return s.rolling(f"{window_days}D", closed="both").sum().to_numpy()


class CorpsRepository:
    """Fetch point-in-time corporate-filings indicators through OpenBB/SEC.

    Pulls each symbol's filing index (``obb.equity.fundamental.filings``)
    and Form 4 insider transactions (``obb.equity.ownership.insider_trading``),
    both via the free ``sec`` provider, and derives trailing-window
    indicators from each (see module docstring). Results are cached on disk
    per (provider, endpoint, symbol), same as the other repositories in
    this package -- the raw event stream is cached, not the derived
    indicators, so changing ``window_days`` doesn't require a re-fetch.
    """

    def __init__(
        self,
        provider: str = "sec",
        form_types: Sequence[str] = DEFAULT_FORM_TYPES,
        window_days: int = DEFAULT_WINDOW_DAYS,
        filings_limit: int = 250,
        insider_limit: int = 250,
        cache_dir: str | None = ".cache/corporate",
    ) -> None:
        self.provider = provider
        self.form_types = list(form_types)
        self.window_days = window_days
        self.filings_limit = filings_limit
        self.insider_limit = insider_limit
        self.cache_dir = cache_dir

    # -- cache --------------------------------------------------------------

    def _cache_path(self, symbol: str, kind: str) -> str | None:
        if not self.cache_dir:
            return None
        limit = self.filings_limit if kind == "filings" else self.insider_limit
        key = "|".join([self.provider, kind, symbol, str(limit)])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    def _fetch_raw(self, symbol: str, kind: str) -> pd.DataFrame:
        path = self._cache_path(symbol, kind)
        if path and os.path.exists(path):
            return pd.read_csv(path, parse_dates=["as_of_date"])
        raw = (
            self._fetch_filings_remote(symbol)
            if kind == "filings"
            else self._fetch_insider_remote(symbol)
        )
        if path:
            raw.to_csv(path, index=False)
        return raw

    # -- fetch ----------------------------------------------------------

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> CorpsPanel:
        symbols = list(dict.fromkeys(symbols))
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        # Per symbol *and* per kind, tolerating absence. "This issuer has no
        # Form 4 filings" is a fact about the issuer, not a failure of the
        # request -- and for a fund it is the only possible answer, since an
        # ETF has no officers or directors to file one. The SEC provider
        # signals that by raising ("No Form 4 data was returned for XLK"),
        # which previously aborted the whole multi-symbol loop: one ETF in a
        # 30-name universe meant zero rows for the 12 issuers that did have
        # real filings. A panel whose columns already tolerate a symbol with
        # no rows should tolerate the fetch that produces none.
        #
        # Deliberately narrow: only this symbol/kind is skipped, it is
        # recorded on `self.skipped` so a caller can see what was missing
        # rather than guess, and nothing here suppresses a failure that
        # would affect every symbol (a bad credential still surfaces on the
        # first one, because every subsequent symbol fails the same way and
        # the panel comes back empty).
        self.skipped: list[tuple[str, str, str]] = []
        frames = []
        for sym in symbols:
            for kind, derive in (("filings", self._derive_filing_indicators),
                                 ("insider", self._derive_insider_indicators)):
                try:
                    raw = self._fetch_raw(sym, kind)
                except Exception as exc:
                    self.skipped.append((sym, kind, f"{type(exc).__name__}: {exc}"))
                    continue
                frames.append(derive(sym, raw))

        frame = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
        if not frame.empty:
            frame = frame[
                (frame["as_of_date"] <= end_ts)
                & (frame["as_of_date"] >= start_ts - pd.Timedelta(days=self.window_days))
            ]
            frame = frame.sort_values(["symbol", "metric", "as_of_date"]).reset_index(
                drop=True
            )
        return CorpsPanel(frame=frame)

    def _derive_filing_indicators(self, symbol: str, filings: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for form in self.form_types:
            g = filings[filings["report_type"] == form].sort_values("as_of_date")
            if g.empty:
                continue
            counts = _trailing_count(g["as_of_date"], self.window_days)
            metric = f"filed_{form.lower().replace('-', '')}_count_{self.window_days}d"
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "metric": metric,
                        "period_end": g["as_of_date"].to_numpy(),
                        "as_of_date": g["as_of_date"].to_numpy(),
                        "value": counts,
                    }
                )
            )
        if not rows:
            return _empty_frame()
        return pd.concat(rows, ignore_index=True)

    def _derive_insider_indicators(self, symbol: str, insider: pd.DataFrame) -> pd.DataFrame:
        if insider.empty:
            return _empty_frame()
        g = insider.sort_values("as_of_date").copy()
        signed = np.where(
            g["acquisition_or_disposition"].astype(str).str.lower().eq("acquisition"),
            g["securities_transacted"].to_numpy(dtype=float),
            -g["securities_transacted"].to_numpy(dtype=float),
        )
        is_buy = g["acquisition_or_disposition"].astype(str).str.lower().eq("acquisition")
        is_sell = g["acquisition_or_disposition"].astype(str).str.lower().eq("disposition")

        net_shares = _trailing_sum(g["as_of_date"], pd.Series(signed), self.window_days)
        buy_count = _trailing_count(
            g.loc[is_buy, "as_of_date"], self.window_days
        ) if is_buy.any() else np.array([])
        sell_count = _trailing_count(
            g.loc[is_sell, "as_of_date"], self.window_days
        ) if is_sell.any() else np.array([])

        rows = [
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "metric": f"insider_net_shares_{self.window_days}d",
                    "period_end": g["as_of_date"].to_numpy(),
                    "as_of_date": g["as_of_date"].to_numpy(),
                    "value": net_shares,
                }
            )
        ]
        if is_buy.any():
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "metric": f"insider_buy_count_{self.window_days}d",
                        "period_end": g.loc[is_buy, "as_of_date"].to_numpy(),
                        "as_of_date": g.loc[is_buy, "as_of_date"].to_numpy(),
                        "value": buy_count,
                    }
                )
            )
        if is_sell.any():
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "metric": f"insider_sell_count_{self.window_days}d",
                        "period_end": g.loc[is_sell, "as_of_date"].to_numpy(),
                        "as_of_date": g.loc[is_sell, "as_of_date"].to_numpy(),
                        "value": sell_count,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def _fetch_filings_remote(self, symbol: str) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb openbb-sec`."
            ) from exc

        out = obb.equity.fundamental.filings(
            symbol=symbol, provider=self.provider, limit=self.filings_limit
        )
        df = out.to_dataframe().reset_index()
        lower = {c.lower(): c for c in df.columns}
        accepted_col = lower.get("accepted_date") or lower.get("filing_date")
        report_type_col = lower.get("report_type")
        if accepted_col is None or report_type_col is None:
            raise ValueError(f"unexpected filings schema for {symbol}: {list(df.columns)}")
        out_df = pd.DataFrame(
            {
                "as_of_date": pd.to_datetime(df[accepted_col], utc=True).dt.tz_localize(None),
                "report_type": df[report_type_col].astype(str),
            }
        )
        return out_df.dropna(subset=["as_of_date"]).reset_index(drop=True)

    def _fetch_insider_remote(self, symbol: str) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb openbb-sec`."
            ) from exc

        out = obb.equity.ownership.insider_trading(
            symbol=symbol, provider=self.provider, limit=self.insider_limit
        )
        df = out.to_dataframe().reset_index()
        lower = {c.lower(): c for c in df.columns}
        filing_col = lower.get("filing_date")
        aod_col = lower.get("acquisition_or_disposition")
        shares_col = lower.get("securities_transacted")
        if filing_col is None or aod_col is None or shares_col is None:
            raise ValueError(
                f"unexpected insider_trading schema for {symbol}: {list(df.columns)}"
            )
        out_df = pd.DataFrame(
            {
                "as_of_date": pd.to_datetime(df[filing_col], utc=True).dt.tz_localize(None),
                "acquisition_or_disposition": df[aod_col].astype(str),
                "securities_transacted": pd.to_numeric(df[shares_col], errors="coerce"),
            }
        )
        return out_df.dropna(subset=["as_of_date", "securities_transacted"]).reset_index(
            drop=True
        )
