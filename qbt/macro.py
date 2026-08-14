"""Point-in-time macroeconomic indicators.

Same shape as :mod:`qbt.fundamentals`, minus the symbol dimension: macro
series (CPI, Fed funds rate, unemployment, GDP, the yield curve, ...)
describe the whole economy, not one issuer, so :class:`MacrosPanel` has no
``symbol`` column and a strategy's whole universe shares the same reading.

The point-in-time problem is *worse* here than for fundamentals, not
better. Two things make it so:

* **Revisions.** GDP ships as an advance estimate and gets revised twice
  more over the following two months; payrolls and CPI get revised too. A
  naive backtest that reads "GDP for Q1" from a table fetched today is
  reading the *final*, twice-revised number -- which the market could not
  see until months after Q1 ended.
* **FRED's date convention.** FRED labels monthly and quarterly
  observations with the *first* day of the period (a January 2024 CPI
  reading is dated ``2024-01-01``, not ``2024-01-31``), not the last. A lag
  computed from that label has to account for the rest of the period too,
  or it silently understates itself by 30-90 days.

The OpenBB ``fred`` provider only surfaces the latest revised value --
it does not expose FRED's ALFRED vintage/realtime data -- so true
revision-aware point-in-time correctness isn't available through this path.
:class:`MacrosRepository` compensates the only honest way available: a
conservative fixed lag per indicator (period length + typical publication
delay), applied the same defensive way :class:`~qbt.fundamentals.
FundamentalsRepository` handles a missing filing date. The default lags
below are estimates from each series' known release cadence (BLS/BEA/Fed
publication schedules), not verified against a live pull -- this environment
has no FRED credential configured. Treat them as a starting point and
correct them once you can check real release dates (OpenBB's
``obb.economy.calendar`` carries an actual economic release calendar and
would let you replace the fixed lag with the true one; that's a reasonable
follow-up, not implemented here).

Requires ``pip install openbb openbb-fred`` and a free FRED API key
(https://fred.stlouisfed.org -- register, then set
``obb.user.credentials.fred_api_key``). Import of ``openbb`` is deferred to
call time so the rest of this package works without it installed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import prune_cache, touch_cache

__all__ = ["MacrosPanel", "MacrosRepository", "DEFAULT_INDICATORS"]

_ID_COLUMNS = ("metric", "period_end", "as_of_date", "value")

# name -> (FRED series ID, days from FRED's labeled date to public release).
# The lag already accounts for FRED's period-start labeling convention
# (see module docstring), so it's roughly period length + true publication
# delay from period *end*, not from the label itself.
DEFAULT_INDICATORS: dict[str, tuple[str, int]] = {
    "fed_funds_rate": ("FEDFUNDS", 33),
    "cpi": ("CPIAUCSL", 45),
    "core_cpi": ("CPILFESL", 45),
    "unemployment_rate": ("UNRATE", 37),
    "real_gdp": ("GDPC1", 120),
    "m2_money_supply": ("M2SL", 45),
    "yield_3m": ("DGS3MO", 1),
    "yield_2y": ("DGS2", 1),
    "yield_10y": ("DGS10", 1),
    "vix": ("VIXCLS", 1),
}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacrosPanel:
    """Point-in-time macro indicators: one row per (metric, release).

    Parameters
    ----------
    frame:
        Long/tidy frame with columns ``metric``, ``period_end`` (the period
        the reading describes), ``as_of_date`` (when it became public),
        ``value``. No ``symbol`` column -- every symbol in a strategy's
        universe sees the same macro reading.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = set(_ID_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"macros frame missing columns: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["as_of_date"]):
            raise TypeError("as_of_date must be datetime64")
        if not pd.api.types.is_datetime64_any_dtype(self.frame["period_end"]):
            raise TypeError("period_end must be datetime64")

    @property
    def metrics(self) -> list[str]:
        return sorted(self.frame["metric"].unique())

    def __len__(self) -> int:
        return len(self.frame)

    # -- point-in-time access ---------------------------------------------

    def as_of(self, date: pd.Timestamp) -> "MacrosPanel":
        """Return a copy containing only readings known at or before ``date``.

        The look-ahead firewall, same contract as
        :meth:`PricePanel.as_of` and :meth:`FundamentalsPanel.as_of`: a
        copy of the same type, structurally incapable of holding a future
        row. This is what the backtester and live runner hand to
        strategies.

        Same tradeoff as ``FundamentalsPanel.as_of``: a full scan of
        ``frame``, not a binary search, on the assumption a macro history
        stays small (a handful of indicators over years of daily readings)
        relative to when that would start to matter.
        """
        date = pd.Timestamp(date).normalize()
        return MacrosPanel(frame=self.frame[self.frame["as_of_date"] <= date])

    def snapshot(self, date: pd.Timestamp,
                 max_age_days: int | None = None) -> pd.Series:
        """Latest known value of every indicator as of ``date``.

        Returns a ``Series`` indexed by metric -- there's no symbol axis to
        pivot against. Safe to call on an already-truncated panel.

        ``max_age_days`` drops readings released longer ago than that,
        rather than returning a stale value as though it were current.
        There is no default, because the right answer is per indicator: VIX
        is daily and a week-old print is suspect, while CPI is monthly and
        a five-week-old print is simply the newest one that exists. Callers
        that act on a reading as a *current* regime read should set it --
        see :class:`~qbt.signals.MacroRegimeFilter`. Without it, a series
        that silently stopped updating (a dead API key, a provider outage,
        a cache that never refreshed) keeps answering with its last value
        forever, and nothing downstream can tell.
        """
        known = self.as_of(date).frame
        if known.empty:
            return pd.Series(dtype=float)
        latest = known.sort_values(["as_of_date", "period_end"]).drop_duplicates(
            "metric", keep="last"
        )
        if max_age_days is not None:
            cutoff = pd.Timestamp(date).normalize() - pd.Timedelta(days=max_age_days)
            latest = latest[latest["as_of_date"] >= cutoff]
        return latest.set_index("metric")["value"].rename(None)

    def snapshot_age_days(self, date: pd.Timestamp) -> pd.Series:
        """Days between each metric's latest release and ``date``.

        The diagnostic behind :meth:`snapshot`'s ``max_age_days``: how old
        is the number you are about to act on.
        """
        known = self.as_of(date).frame
        if known.empty:
            return pd.Series(dtype=float)
        latest = known.sort_values(["as_of_date", "period_end"]).drop_duplicates(
            "metric", keep="last"
        ).set_index("metric")
        age = (pd.Timestamp(date).normalize() - latest["as_of_date"]).dt.days
        return age.astype(float).rename(None)

    def to_daily(
        self,
        dates: pd.DatetimeIndex,
        metrics: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Forward-fill each indicator onto a daily calendar.

        For building research features over a whole backtest window at
        once. This is *not* the live/backtest decision path -- that path
        calls :meth:`as_of` bar by bar so a strategy can never see a
        reading before it was released.
        """
        wanted = list(metrics) if metrics is not None else self.metrics
        calendar = pd.DataFrame({"as_of_date": pd.DatetimeIndex(dates)})

        columns: dict[str, np.ndarray] = {}
        for metric in wanted:
            g = self.frame[self.frame["metric"] == metric].sort_values(
                "as_of_date"
            ).drop_duplicates("as_of_date", keep="last")
            merged = pd.merge_asof(
                calendar, g[["as_of_date", "value"]], on="as_of_date",
                direction="backward",
            )
            columns[metric] = merged["value"].to_numpy()
        return pd.DataFrame(columns, index=pd.DatetimeIndex(dates))

    def describe(self) -> str:
        if self.frame.empty:
            return "MacrosPanel(0 indicators, 0 observations)"
        return (
            f"MacrosPanel({len(self.metrics)} indicators, "
            f"{len(self.frame)} observations, "
            f"{self.frame['as_of_date'].min().date()} to "
            f"{self.frame['as_of_date'].max().date()})"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class MacrosRepository:
    """Fetch point-in-time macro indicators through OpenBB's FRED provider.

    Pulls each configured indicator via ``obb.economy.fred_series`` (one
    uniform endpoint covering essentially any published FRED series, which
    is why this doesn't need the per-statement dispatch
    :class:`~qbt.fundamentals.FundamentalsRepository` does), tags each
    observation with an estimated public-release date, and returns a single
    :class:`MacrosPanel`. See the module docstring for why that release date
    is an estimate rather than the true (revision-aware) one, and for the
    FRED period-start labeling quirk the default lags correct for.

    Results are cached on disk per (provider, indicator, date range), same
    as the other repositories in this package.
    """

    def __init__(
        self,
        provider: str = "fred",
        indicators: dict[str, tuple[str, int]] | None = None,
        cache_dir: str | None = ".cache/macro",
    ) -> None:
        self.provider = provider
        self.indicators = dict(indicators) if indicators else dict(DEFAULT_INDICATORS)
        self.cache_dir = cache_dir

    # -- cache --------------------------------------------------------------

    def _cache_path(self, name: str, series_id: str, start: str, end: str) -> str | None:
        if not self.cache_dir:
            return None
        key = "|".join([self.provider, name, series_id, start, end])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    # -- fetch ----------------------------------------------------------

    def fetch(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> MacrosPanel:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        start_s, end_s = str(start_ts.date()), str(end_ts.date())

        if not self.indicators:
            raise ValueError("no indicators configured")

        # Sweep dead entries before writing new ones -- see prune_cache.
        prune_cache(self.cache_dir)
        frames = []
        for name, (series_id, lag_days) in self.indicators.items():
            path = self._cache_path(name, series_id, start_s, end_s)
            if path and os.path.exists(path):
                touch_cache(path)      # a hit keeps it alive; see prune_cache
                long = pd.read_csv(path, parse_dates=["period_end", "as_of_date"])
            else:
                long = self._fetch_remote(name, series_id, lag_days, start_s, end_s)
                if path:
                    long.to_csv(path, index=False)
            frames.append(long)

        frame = pd.concat(frames, ignore_index=True)
        frame = frame[
            (frame["as_of_date"] <= end_ts)
            & (frame["period_end"] >= start_ts - pd.Timedelta(days=730))
        ]
        frame = frame.sort_values(["metric", "as_of_date"]).reset_index(drop=True)
        return MacrosPanel(frame=frame)

    def _fetch_remote(
        self, name: str, series_id: str, lag_days: int, start: str, end: str
    ) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb openbb-fred`."
            ) from exc

        out = obb.economy.fred_series(
            symbol=series_id, start_date=start, end_date=end, provider=self.provider
        )
        df = out.to_dataframe().reset_index()
        lower = {c.lower(): c for c in df.columns}
        date_col = lower.get("date") or df.columns[0]

        value_col = None
        for candidate in (series_id, series_id.lower(), "value"):
            if candidate in df.columns:
                value_col = candidate
                break
        if value_col is None:
            others = [c for c in df.columns if c != date_col]
            if not others:
                raise ValueError(f"no value column found for {name!r} ({series_id})")
            value_col = others[0]

        period_end = pd.to_datetime(df[date_col])
        long = pd.DataFrame(
            {
                "metric": name,
                "period_end": period_end,
                "as_of_date": period_end + pd.Timedelta(days=lag_days),
                "value": pd.to_numeric(df[value_col], errors="coerce"),
            }
        )
        return long.dropna(subset=["value"]).reset_index(drop=True)
