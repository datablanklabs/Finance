"""Point-in-time fundamentals.

Mirrors the shape of :mod:`qbt.data`: a repository fetches raw data, a panel
holds it and enforces the look-ahead firewall. The one thing that's
different -- and the entire reason this is its own module instead of a few
extra columns on :class:`~qbt.data.PricePanel` -- is that fundamentals are
not a dense daily grid. A symbol reports a handful of times a year, and the
date that matters is not the fiscal period the numbers describe but the date
they became public.

Joining a Q1 income statement onto March 31st is the single most common
source of look-ahead bias in equity research: nobody could see that
statement on March 31st, they saw it on the 10-Q filing date, six to eight
weeks later. Every method on :class:`FundamentalsPanel` keys off
``as_of_date`` for exactly that reason -- it is the fundamentals analogue of
:meth:`PricePanel.as_of`.

Requires ``pip install openbb openbb-fmp`` or ``openbb-intrinio``. Import of
``openbb`` is deferred to call time so the rest of this package works
without it installed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["FundamentalsPanel", "FundamentalsRepository"]

_ID_COLUMNS = ("symbol", "metric", "period_end", "as_of_date", "value")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalsPanel:
    """Point-in-time fundamentals: one row per (symbol, metric, filing).

    Parameters
    ----------
    frame:
        Long/tidy frame with columns ``symbol``, ``metric``, ``period_end``
        (fiscal period the figure describes), ``as_of_date`` (when the figure
        became public), ``value``.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(_ID_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"fundamentals frame missing columns: {missing}")
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

    def as_of(self, date: pd.Timestamp) -> "FundamentalsPanel":
        """Return a copy containing only filings known at or before ``date``.

        This is the look-ahead firewall, and it mirrors
        :meth:`PricePanel.as_of` exactly: same name, same contract (a copy
        of the same type, structurally incapable of holding a future row).
        It is what the backtester and live runner hand to strategies --
        never the unfiltered panel.

        Unlike ``PricePanel.as_of``, which binary-searches a sorted date
        index, this is a full boolean-mask scan of ``frame`` -- a
        deliberate simplicity-over-asymptotics choice, since a realistic
        fundamentals history is thousands of rows, not millions. If a
        panel ever grows large enough for that to matter, switch to the
        same sorted-index-plus-``searchsorted`` pattern ``PricePanel.as_of``
        uses.
        """
        date = pd.Timestamp(date).normalize()
        return FundamentalsPanel(frame=self.frame[self.frame["as_of_date"] <= date])

    def snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        """Latest known value of every metric, per symbol, as of ``date``.

        Convenience wrapper around :meth:`as_of` for the common case of
        wanting one flat row per symbol rather than the raw filing history.
        Safe to call on an already-truncated panel (e.g. the one a strategy
        receives) -- it can only surface what's already there.
        """
        known = self.as_of(date).frame
        if known.empty:
            return pd.DataFrame(columns=self.metrics)
        latest = known.sort_values("as_of_date").drop_duplicates(
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
        once -- ``information_coefficient`` and friends in
        :mod:`qbt.research` expect exactly this shape. This is *not* the
        live/backtest decision path; that path calls :meth:`as_of` bar by
        bar so a strategy can never see a filing before it happened.
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
            return "FundamentalsPanel(0 symbols, 0 metrics, 0 filings)"
        return (
            f"FundamentalsPanel({len(self.symbols)} symbols, "
            f"{len(self.metrics)} metrics, {len(self.frame)} filings, "
            f"{self.frame['as_of_date'].min().date()} to "
            f"{self.frame['as_of_date'].max().date()})"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class FundamentalsRepository:
    """Fetch point-in-time fundamentals through OpenBB (FMP or Intrinio).

    Pulls one or more statements (``income``, ``balance``, ``cash``,
    ``ratios``, ...) per symbol from ``obb.equity.fundamental``, tags each
    row with the filing/acceptance date the provider reports, and melts
    everything into the long format :class:`FundamentalsPanel` expects.

    When a provider doesn't expose a true filing date for a statement, the
    fetch falls back to ``period_end + fallback_lag_days`` rather than
    silently joining on period end -- a wrong-but-conservative lag is much
    safer than an invisible look-ahead leak. Results are cached on disk per
    (provider, statement, symbol, period) since fundamentals endpoints are
    slow and rate-limited, same as :class:`~qbt.data.OpenBBRepository`.
    """

    def __init__(
        self,
        provider: str = "fmp",
        statements: Sequence[str] = ("income", "balance", "cash", "ratios"),
        period: str = "quarter",
        limit: int = 40,
        cache_dir: str | None = ".cache/fundamentals",
        fallback_lag_days: int = 45,
    ) -> None:
        self.provider = provider
        self.statements = list(statements)
        self.period = period
        self.limit = limit
        self.cache_dir = cache_dir
        self.fallback_lag_days = fallback_lag_days

    # -- cache --------------------------------------------------------------

    def _cache_path(self, symbol: str, statement: str, start: str, end: str) -> str | None:
        if not self.cache_dir:
            return None
        key = "|".join(
            [self.provider, statement, self.period, symbol, start, end]
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    # -- fetch ----------------------------------------------------------

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> FundamentalsPanel:
        symbols = list(dict.fromkeys(symbols))
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        start_s, end_s = str(start_ts.date()), str(end_ts.date())

        frames = []
        for sym in symbols:
            for stmt in self.statements:
                path = self._cache_path(sym, stmt, start_s, end_s)
                if path and os.path.exists(path):
                    long = pd.read_csv(
                        path, parse_dates=["period_end", "as_of_date"]
                    )
                else:
                    long = self._fetch_remote(sym, stmt)
                    if path:
                        long.to_csv(path, index=False)
                frames.append(long)

        if not frames:
            raise ValueError("no symbols requested")

        frame = pd.concat(frames, ignore_index=True)
        # A generous back-window: strategies commonly need a year or two of
        # trailing fundamentals before the backtest's own start date.
        frame = frame[
            (frame["as_of_date"] <= end_ts)
            & (frame["period_end"] >= start_ts - pd.Timedelta(days=730))
        ]
        frame = frame.sort_values(["symbol", "metric", "as_of_date"]).reset_index(
            drop=True
        )
        return FundamentalsPanel(frame=frame)

    def _fetch_remote(self, symbol: str, statement: str) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb "
                "openbb-fmp` (or `openbb-intrinio`)."
            ) from exc

        try:
            fn = getattr(obb.equity.fundamental, statement)
        except AttributeError as exc:  # pragma: no cover
            raise ValueError(f"unknown fundamentals statement: {statement!r}") from exc

        out = fn(symbol=symbol, period=self.period, provider=self.provider, limit=self.limit)
        df = out.to_dataframe().reset_index()
        lower = {c.lower(): c for c in df.columns}

        period_col = lower.get("period_ending") or lower.get("date")
        if period_col is None:
            raise ValueError(
                f"{statement} response for {symbol} has no period date column"
            )
        period_end = pd.to_datetime(df[period_col])

        filing_col = lower.get("filing_date") or lower.get("accepted_date")
        if filing_col is not None and df[filing_col].notna().any():
            as_of = pd.to_datetime(df[filing_col])
            as_of = as_of.fillna(period_end + pd.Timedelta(days=self.fallback_lag_days))
        else:
            lag = 90 if self.period == "annual" else self.fallback_lag_days
            as_of = period_end + pd.Timedelta(days=lag)

        exclude = {
            period_col,
            lower.get("filing_date"),
            lower.get("accepted_date"),
            lower.get("symbol"),
            lower.get("cik"),
            lower.get("fiscal_period"),
            lower.get("fiscal_year"),
            lower.get("index"),
        }
        exclude.discard(None)
        metric_cols = [c for c in df.columns if c not in exclude]

        long = df[metric_cols].copy()
        long["period_end"] = period_end
        long["as_of_date"] = as_of
        long["symbol"] = symbol
        long = long.melt(
            id_vars=["symbol", "period_end", "as_of_date"],
            var_name="metric",
            value_name="value",
        )
        long["metric"] = f"{statement}_" + long["metric"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        return long.dropna(subset=["value"]).reset_index(drop=True)
