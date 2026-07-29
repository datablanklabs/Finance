"""Data layer.

The only object the signal engine ever sees is a :class:`PricePanel`. The
backtester hands strategies a panel that has been *sliced* to the decision
date, so a strategy is structurally unable to read the future. That is the
single most important property in this file.

Two repositories implement the same protocol:

* :class:`OpenBBRepository` -- live/research data via the OpenBB ODP package.
* :class:`SyntheticRepository` -- deterministic generated data, no network.
  Used by the test suite and for developing strategy logic offline.

Because both satisfy :class:`PriceRepository`, swapping between them changes
no strategy code. This is the seam where you would later plug in a
point-in-time store to eliminate survivorship bias.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PricePanel",
    "PriceRepository",
    "OpenBBRepository",
    "SyntheticRepository",
]


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricePanel:
    """Aligned wide price data.

    Parameters
    ----------
    close:
        Split/dividend adjusted closes. ``index`` is a tz-naive normalised
        ``DatetimeIndex``, ``columns`` are symbols.
    open_:
        Optional opens, same shape. Needed only if you execute at the open.
    volume:
        Optional volume, same shape. Used for liquidity filters.
    """

    close: pd.DataFrame
    open_: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.close.index, pd.DatetimeIndex):
            raise TypeError("close must be indexed by DatetimeIndex")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("close index must be sorted ascending")
        if self.close.index.has_duplicates:
            raise ValueError("close index has duplicate dates")
        for name in ("open_", "volume"):
            other = getattr(self, name)
            if other is None:
                continue
            if not other.index.equals(self.close.index):
                raise ValueError(f"{name} index does not match close")
            if list(other.columns) != list(self.close.columns):
                raise ValueError(f"{name} columns do not match close")

    # -- shape ------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close.index)

    # -- slicing ----------------------------------------------------------

    def as_of(self, date: pd.Timestamp) -> "PricePanel":
        """Return a copy containing only bars at or before ``date``.

        This is the look-ahead firewall. Strategies receive the output of
        this method and nothing else.
        """
        date = pd.Timestamp(date).normalize()
        pos = int(self.close.index.searchsorted(date, side="right"))
        return PricePanel(
            close=self.close.iloc[:pos],
            open_=None if self.open_ is None else self.open_.iloc[:pos],
            volume=None if self.volume is None else self.volume.iloc[:pos],
        )

    def tail(self, k: int) -> "PricePanel":
        """Return the last ``k`` bars. Useful for cheaper experiments."""
        return PricePanel(
            close=self.close.tail(k),
            open_=None if self.open_ is None else self.open_.tail(k),
            volume=None if self.volume is None else self.volume.tail(k),
        )

    def select(self, symbols: Sequence[str]) -> "PricePanel":
        """Restrict to ``symbols`` (order preserved, missing ones dropped)."""
        keep = [s for s in symbols if s in self.close.columns]
        return PricePanel(
            close=self.close[keep],
            open_=None if self.open_ is None else self.open_[keep],
            volume=None if self.volume is None else self.volume[keep],
        )

    def with_min_history(self, min_bars: int) -> "PricePanel":
        """Drop symbols with fewer than ``min_bars`` non-null observations."""
        keep = self.close.columns[self.close.notna().sum() >= min_bars]
        return self.select(list(keep))

    # -- derived ----------------------------------------------------------

    def returns(self) -> pd.DataFrame:
        """Simple daily returns from adjusted closes."""
        return self.close.pct_change(fill_method=None)

    def log_returns(self) -> pd.DataFrame:
        return np.log(self.close).diff()

    def trailing_return(self, lookback: int, skip: int = 0) -> pd.Series:
        """Return over ``lookback`` bars ending ``skip`` bars before the end.

        ``skip`` lets a momentum signal exclude the most recent bars, which
        is standard practice because short-horizon reversal contaminates the
        momentum signal at the near end.
        """
        if len(self.close) < lookback + skip + 1:
            return pd.Series(np.nan, index=self.close.columns, dtype=float)
        end = self.close.iloc[len(self.close) - 1 - skip]
        start = self.close.iloc[len(self.close) - 1 - skip - lookback]
        return (end / start) - 1.0

    def tail_returns(self, k: int) -> pd.DataFrame:
        """Returns over the last ``k`` bars, computed only on the tail.

        Computing ``pct_change`` over the full history just to keep the tail is
        O(history) per call, which turns a backtest loop into O(n^2).
        """
        return self.close.tail(k + 1).pct_change(fill_method=None).iloc[1:]

    def realised_vol(self, lookback: int = 63, annualise: bool = True) -> pd.Series:
        """Per-symbol realised volatility over the trailing window."""
        vol = self.tail_returns(lookback).std()
        return vol * np.sqrt(252.0) if annualise else vol

    def covariance(self, lookback: int = 126, annualise: bool = True) -> pd.DataFrame:
        r = self.tail_returns(lookback).dropna(axis=1, how="all")
        cov = r.cov()
        return cov * 252.0 if annualise else cov

    def last_close(self) -> pd.Series:
        return self.close.iloc[-1]

    def head_date(self) -> pd.Timestamp:
        return self.close.index[0]

    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]

    def dropna_symbols(self) -> "PricePanel":
        keep = self.close.columns[self.close.iloc[-1].notna()]
        return self.select(list(keep))

    def describe(self) -> str:
        return (
            f"PricePanel({len(self.symbols)} symbols, {len(self)} bars, "
            f"{self.head_date().date()} to {self.last_date().date()})"
        )


# ---------------------------------------------------------------------------
# Repository protocol
# ---------------------------------------------------------------------------


class PriceRepository(Protocol):
    """Anything that can produce a :class:`PricePanel`."""

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> PricePanel: ...


# ---------------------------------------------------------------------------
# OpenBB
# ---------------------------------------------------------------------------


class OpenBBRepository:
    """Fetch daily bars through the OpenBB ODP Python package.

    Requires ``pip install openbb openbb-yfinance`` (or another provider
    extension). Import of ``openbb`` is deferred to call time so the rest of
    this package works without it installed.

    Results are cached on disk keyed by the request, because pulling a wide
    universe repeatedly during research is slow and rate-limited.
    """

    def __init__(
        self,
        provider: str = "yfinance",
        cache_dir: str | None = ".cache/prices",
        include_open: bool = True,
        include_volume: bool = True,
    ) -> None:
        self.provider = provider
        self.cache_dir = cache_dir
        self.include_open = include_open
        self.include_volume = include_volume

    # -- cache ------------------------------------------------------------

    def _cache_path(self, symbols: Sequence[str], start: str, end: str) -> str | None:
        if not self.cache_dir:
            return None
        key = "|".join([self.provider, start, end, ",".join(sorted(symbols))])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}.csv.gz")

    # -- fetch ------------------------------------------------------------

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> PricePanel:
        symbols = list(dict.fromkeys(symbols))
        start_s = str(pd.Timestamp(start).date())
        end_s = str(pd.Timestamp(end).date())

        path = self._cache_path(symbols, start_s, end_s)
        if path and os.path.exists(path):
            tidy = pd.read_csv(path, parse_dates=["date"])
        else:
            tidy = self._fetch_remote(symbols, start_s, end_s)
            if path:
                tidy.to_csv(path, index=False)

        return self._to_panel(tidy, symbols)

    def _fetch_remote(
        self, symbols: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        try:
            from openbb import obb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenBB is not installed. Run `pip install openbb "
                "openbb-yfinance`, or use SyntheticRepository."
            ) from exc

        out = obb.equity.price.historical(
            symbol=symbols,
            start_date=start,
            end_date=end,
            interval="1d",
            provider=self.provider,
        )
        df = out.to_dataframe().reset_index()

        # Normalise the two shapes OpenBB returns: single-symbol frames have
        # no `symbol` column, multi-symbol frames do.
        lower = {c.lower(): c for c in df.columns}
        date_col = lower.get("date") or df.columns[0]
        df = df.rename(columns={date_col: "date"})
        if "symbol" not in [c.lower() for c in df.columns]:
            df["symbol"] = symbols[0]
        else:
            df = df.rename(columns={lower.get("symbol", "symbol"): "symbol"})

        rename = {}
        for want in ("open", "high", "low", "close", "volume", "adj_close"):
            if want in lower:
                rename[lower[want]] = want
        df = df.rename(columns=rename)

        # Prefer an explicitly adjusted close when the provider supplies one.
        if "adj_close" in df.columns and df["adj_close"].notna().any():
            df["close"] = df["adj_close"]

        keep = ["date", "symbol", "close"]
        if self.include_open and "open" in df.columns:
            keep.append("open")
        if self.include_volume and "volume" in df.columns:
            keep.append("volume")
        return df[keep]

    def _to_panel(self, tidy: pd.DataFrame, symbols: Sequence[str]) -> PricePanel:
        tidy = tidy.copy()
        tidy["date"] = pd.to_datetime(tidy["date"], utc=True).dt.tz_localize(None)
        tidy["date"] = tidy["date"].dt.normalize()
        tidy = tidy.drop_duplicates(subset=["date", "symbol"], keep="last")

        def pivot(col: str) -> pd.DataFrame | None:
            if col not in tidy.columns:
                return None
            wide = tidy.pivot(index="date", columns="symbol", values=col)
            wide = wide.reindex(columns=[s for s in symbols if s in wide.columns])
            wide.columns.name = None
            return wide.sort_index().astype(float)

        close = pivot("close")
        if close is None or close.empty:
            raise ValueError("no close data returned")
        return PricePanel(close=close, open_=pivot("open"), volume=pivot("volume"))


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


class SyntheticRepository:
    """Deterministic price generator with known, tunable structure.

    Returns are built as::

        r[i,t] = beta[i] * f[t] + mu[i,t] + e[i,t] - theta * e[i,t-1]

    where ``mu`` is a slow AR(1) process per symbol and ``e`` is idiosyncratic
    noise. The two knobs create the two effects the strategies in this
    package are designed to harvest:

    * ``mu_persistence`` near 1 produces **cross-sectional momentum** -- names
      whose expected return was high stay high for months.
    * ``reversal_theta`` above 0 produces **short-horizon reversal** -- a large
      idiosyncratic move is partly given back the next day.

    Set both to zero and you get an efficient market: every strategy in this
    package should then produce a Sharpe statistically indistinguishable from
    zero. That is a useful sanity test of the backtester itself -- if a
    strategy looks profitable on structureless data, the harness is leaking.
    """

    def __init__(
        self,
        n_symbols: int = 40,
        seed: int = 7,
        mu_persistence: float = 0.98,
        mu_scale: float = 0.0003,
        reversal_theta: float = 0.10,
        market_vol: float = 0.011,
        idio_vol: float = 0.014,
        drift: float = 0.00025,
        start_price: float = 100.0,
    ) -> None:
        self.n_symbols = n_symbols
        self.seed = seed
        self.mu_persistence = mu_persistence
        self.mu_scale = mu_scale
        self.reversal_theta = reversal_theta
        self.market_vol = market_vol
        self.idio_vol = idio_vol
        self.drift = drift
        self.start_price = start_price

    def universe(self) -> list[str]:
        return [f"SYN{i:03d}" for i in range(self.n_symbols)]

    def fetch(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp = "2015-01-01",
        end: str | pd.Timestamp = "2025-12-31",
    ) -> PricePanel:
        dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
        names = list(symbols) if symbols else self.universe()
        n, t = len(names), len(dates)

        rng = np.random.default_rng(self.seed)
        beta = rng.uniform(0.6, 1.4, size=n)
        factor = rng.normal(0.0, self.market_vol, size=t)
        eps = rng.normal(0.0, self.idio_vol, size=(n, t))

        mu = np.zeros((n, t))
        shock = rng.normal(0.0, self.mu_scale, size=(n, t))
        mu[:, 0] = shock[:, 0]
        for k in range(1, t):
            mu[:, k] = self.mu_persistence * mu[:, k - 1] + shock[:, k]

        lagged = np.concatenate([np.zeros((n, 1)), eps[:, :-1]], axis=1)
        rets = (
            beta[:, None] * factor[None, :]
            + mu
            + eps
            - self.reversal_theta * lagged
            + self.drift
        )

        closes = self.start_price * np.exp(np.cumsum(rets, axis=1))
        close = pd.DataFrame(closes.T, index=dates, columns=names)

        # Opens gap from the prior close by a fraction of the day's move, so
        # execution at the open is neither free nor identical to the close.
        gap = rng.normal(0.0, 0.004, size=(t, n))
        open_ = close.shift(1) * (1.0 + gap)
        open_.iloc[0] = close.iloc[0] * (1.0 + gap[0])

        volume = pd.DataFrame(
            rng.lognormal(13.5, 0.45, size=(t, n)).round(),
            index=dates,
            columns=names,
        )
        return PricePanel(close=close, open_=open_, volume=volume)


def align_panels(*panels: PricePanel) -> list[PricePanel]:
    """Reindex panels onto their common dates and symbols."""
    dates = panels[0].dates
    cols: Iterable[str] = panels[0].symbols
    for p in panels[1:]:
        dates = dates.intersection(p.dates)
        cols = [c for c in cols if c in p.symbols]
    out = []
    for p in panels:
        sub = p.select(list(cols))
        out.append(
            replace(
                sub,
                close=sub.close.loc[dates],
                open_=None if sub.open_ is None else sub.open_.loc[dates],
                volume=None if sub.volume is None else sub.volume.loc[dates],
            )
        )
    return out
