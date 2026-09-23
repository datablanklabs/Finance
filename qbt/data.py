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

import glob
import hashlib
import os
import time
import warnings
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PricePanel",
    "PriceRepository",
    "OpenBBRepository",
    "SyntheticRepository",
    "align_panels",
    "prune_cache",
    "touch_cache",
    "as_merge_key",
    "trim_incomplete_tail",
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
        if self.close.columns.has_duplicates:
            # Caught here rather than left to surface downstream: a
            # duplicated symbol column constructs fine, survives as_of()
            # and every strategy, then dies deep inside Backtester.run()
            # with pandas' own "cannot reindex on an axis with duplicate
            # labels" -- a message that names neither the panel nor the
            # symbol. Confirmed reachable by passing a bare string as
            # `symbols` to either repository (see their fetch()), which
            # used to split it into duplicate single-character names.
            dupes = sorted(set(self.close.columns[self.close.columns.duplicated()]))
            raise ValueError(f"close has duplicate symbol columns: {dupes}")
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


def trim_incomplete_tail(
    panel: PricePanel, max_bars: int = 3
) -> tuple[PricePanel, list[tuple[pd.Timestamp, list[str]]]]:
    """Drop trailing bars the provider hadn't finished filling in.

    A bar is incomplete when some symbol that closed on the bar before it
    has no close on it. Seen live (yfinance, 2026-09-23 08:19 ET): the
    previous session's row came back with ``open`` and ``volume`` but every
    ``close`` NaN. Left in place, that row becomes ``last_close()``, every
    held position is valued at nothing, and ``RiskGate``'s drawdown breaker
    trips on an account that hadn't lost a cent. Dropping it makes the cycle
    run on the last complete bar instead -- a day older, which
    ``LiveSignalRunner``'s own staleness warning already covers.

    Only the tail is checked. A gap mid-history (a halt, a listing date) is
    real data and stays as it is; and a symbol NaN on *both* of the last
    two bars (delisted, not yet listed) isn't evidence the newest bar is
    unfinished, so it never triggers a trim either.

    Returns the trimmed panel and ``[(dropped_date, [missing symbols]),
    ...]``, newest first. More than ``max_bars`` incomplete bars in a row is
    no longer "the provider is a few hours behind" -- raises ``ValueError``
    instead of quietly planning off data that old.
    """
    close = panel.close
    dropped: list[tuple[pd.Timestamp, list[str]]] = []
    while len(close) >= 2:
        missing = close.iloc[-1].isna() & close.iloc[-2].notna()
        if not missing.any():
            break
        dropped.append((close.index[-1], list(missing[missing].index)))
        close = close.iloc[:-1]
        if len(dropped) > max_bars:
            raise ValueError(
                f"price data incomplete on each of the last {len(dropped)} bars "
                f"(max_trim_bars={max_bars}); most recent: "
                f"{dropped[0][0].date()} missing {', '.join(dropped[0][1])}"
            )
    if not dropped:
        return panel, dropped
    for date, syms in dropped:
        warnings.warn(
            f"dropped incomplete price bar {date.date()}: no close for "
            f"{len(syms)} symbol(s) priced the bar before ({', '.join(syms)})",
            stacklevel=3,
        )
    return panel.as_of(close.index[-1]), dropped


def _remove_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def touch_cache(path: str | None) -> None:
    """Mark a cache entry as used, now.

    Called on every cache *hit*. Pruning below keys on mtime, and a plain
    read does not update mtime -- nor, on many modern filesystems, atime
    either: macOS APFS and Linux ``relatime``/``noatime`` mounts all skip
    atime updates on read, which was measured here (a genuine read left
    ``st_atime`` untouched). So "last used" has to be recorded explicitly
    rather than inferred from the filesystem, or a cache entry read every
    single day still looks untouched since the day it was written and gets
    swept for no reason.
    """
    if not path:
        return
    try:
        os.utime(path, None)
    except OSError:
        pass


def prune_cache(cache_dir: str | None, max_age_days: int = 30,
                pattern: str = "*.csv.gz") -> int:
    """Delete cache entries unused for ``max_age_days``. Returns the count.

    These caches key on the request, *including its end date*. That's
    correct for research -- the same fixed window re-fetched all afternoon
    hits cache every time -- but it means a daily scheduled run, whose end
    date is "today", writes a new entry every day and never reads it again.
    Left alone that grows without bound while its hit rate on the live path
    stays at zero, so the entries that are actually dead get swept.

    "Unused" means mtime, which :func:`touch_cache` refreshes on every hit
    -- see there for why the filesystem's own access time can't be trusted
    to answer this. A window you keep coming back to therefore survives
    however old it is; one written once by a scheduled run and never read
    again ages out. Failures are ignored on purpose: a cache is an
    optimisation, and being unable to tidy it must never take down a
    trading cycle.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    cutoff = time.time() - max_age_days * 86_400
    removed = 0
    for path in glob.glob(os.path.join(cache_dir, pattern)):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


def as_merge_key(values) -> pd.Series:
    """Coerce a datetime column to one fixed resolution for ``merge_asof``.

    ``pd.merge_asof`` requires its join keys to have *identical* dtypes and
    raises ``MergeError: incompatible merge keys ... must be the same type``
    otherwise -- it will not coerce, unlike most pandas joins. Datetime
    resolution now varies by how a column was produced: a price index read
    back from CSV lands on ``datetime64[s]``, while an ``as_of_date`` built
    from ``Timestamp.now()`` or arithmetic lands on ``datetime64[us]``.

    Confirmed (2026-08): every ``to_daily`` in this package joins a price
    calendar against a panel's ``as_of_date``, and the options panel tripped
    exactly this the first time it held live data rather than a synthetic
    stand-in whose dtypes happened to agree. Normalising both sides through
    here is what makes those joins independent of how either column was
    built.

    The input's index is preserved deliberately. Callers assign the result
    back onto an existing frame (``right["as_of_date"] = as_merge_key(...)``)
    whose index is a groupby subset, not ``0..n-1``; renumbering here would
    make that assignment align a fresh RangeIndex against the original
    labels and silently fill the column with NaN, which ``merge_asof`` then
    rejects as "Merge keys contain null values".
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    return pd.to_datetime(s).astype("datetime64[ns]")


def _require_symbol_sequence(symbols) -> list[str]:
    """Reject a bare string where a sequence of symbols is expected.

    ``str`` is itself an iterable of characters, so ``fetch("AAPL", ...)``
    silently became the four symbols ``['A', 'A', 'P', 'L']`` instead of
    raising -- and, being duplicated, then crashed deep inside
    ``Backtester.run()`` with pandas' opaque "cannot reindex on an axis with
    duplicate labels". Against :class:`SyntheticRepository` it was worse than
    a crash: it fabricated a plausible four-symbol panel of generated prices
    with no error at all. One wrong pair of brackets is an easy mistake and
    an expensive one to debug from either symptom.
    """
    if isinstance(symbols, str):
        raise TypeError(
            f"symbols must be a sequence of tickers, not a bare string "
            f"{symbols!r} -- a string iterates as characters, which would "
            f"silently request {list(dict.fromkeys(symbols))}. "
            f"Pass [{symbols!r}] for a single symbol."
        )
    return list(symbols)


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

    ``asset_class="crypto"`` calls OpenBB's separate ``obb.crypto.price.
    historical`` endpoint instead of ``obb.equity.price.historical`` --
    genuinely two different endpoints in OpenBB's own API, not a naming
    convenience, because a crypto ticker on ``yfinance`` (``"BTC-USD"``,
    not a bare ``"BTC"``) isn't resolvable through the equity one. This is
    plain OpenBB/yfinance behaviour, unrelated to and independent of
    anything about the Robinhood MCP surface -- it works (and is worth
    testing with ``USE_OPENBB=True`` / ``--consider-crypto`` without a
    broker at all) whether or not crypto trading itself is ever wired up.
    Two other differences follow from crypto trading 24/7: the resulting
    panel's dates include weekends, which is exactly why a crypto panel is
    never merged into the same :class:`PricePanel` as equities (see
    ``run_cycle.py``'s crypto pipeline) -- a shared date index would either
    invent fake weekend bars for equities or silently drop weekend crypto
    bars, and either corrupts every lookback-window calculation in
    :mod:`qbt.signals`.
    """

    def __init__(
        self,
        provider: str = "yfinance",
        cache_dir: str | None = ".cache/prices",
        include_open: bool = True,
        include_volume: bool = True,
        asset_class: str = "equity",
        max_trim_bars: int = 3,
    ) -> None:
        if asset_class not in ("equity", "crypto"):
            raise ValueError(f"asset_class must be 'equity' or 'crypto', got {asset_class!r}")
        self.provider = provider
        self.cache_dir = cache_dir
        self.include_open = include_open
        self.include_volume = include_volume
        self.asset_class = asset_class
        self.max_trim_bars = max_trim_bars
        # What the most recent fetch() dropped off the end -- see
        # trim_incomplete_tail. Exposed so a caller (run_cycle.py) can put
        # it in its audit log, not just the warning.
        self.last_trimmed: list[tuple[pd.Timestamp, list[str]]] = []

    # -- cache ------------------------------------------------------------

    def _cache_path(self, symbols: Sequence[str], start: str, end: str) -> str | None:
        if not self.cache_dir:
            return None
        key = "|".join([self.asset_class, self.provider, start, end,
                        ",".join(sorted(symbols))])
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
        symbols = _require_symbol_sequence(symbols)
        symbols = list(dict.fromkeys(symbols))
        start_s = str(pd.Timestamp(start).date())
        end_s = str(pd.Timestamp(end).date())

        path = self._cache_path(symbols, start_s, end_s)
        from_cache = bool(path and os.path.exists(path))
        if from_cache:
            touch_cache(path)          # a hit keeps it alive; see prune_cache
            tidy = pd.read_csv(path, parse_dates=["date"])
        else:
            tidy = self._fetch_remote(symbols, start_s, end_s)

        # Never keep a response the provider hadn't finished: written (or
        # left) in the cache, it would pin today's incomplete bar for every
        # rerun under the same key, long after the provider has filled the
        # close in -- so a trimmed fetch isn't written, and a trimmed (or
        # untrimmably broken) cache entry is removed.
        try:
            panel, trimmed = trim_incomplete_tail(
                self._to_panel(tidy, symbols), max_bars=self.max_trim_bars
            )
        except ValueError:
            if from_cache:
                _remove_quietly(path)
            raise
        self.last_trimmed = trimmed
        if trimmed:
            if from_cache:
                _remove_quietly(path)
        elif not from_cache:
            if path:
                # Sweep dead entries before adding a new one -- see
                # prune_cache's own docstring for why this matters
                # specifically for a daily scheduled run (whose end date is
                # "today," so it always misses and always writes a new
                # entry, which is exactly the growth prune_cache exists to
                # bound). Only on the miss path: a hit isn't adding
                # anything to the cache, so it has nothing to sweep before
                # -- paying an O(cache-directory-size) glob+stat on every
                # single fetch(), hit or miss, was strictly more sweeping
                # than the docstring's own stated reason for doing this.
                prune_cache(self.cache_dir)
                tidy.to_csv(path, index=False)

        return panel

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

        endpoint = (
            obb.crypto.price.historical if self.asset_class == "crypto"
            else obb.equity.price.historical
        )
        out = endpoint(
            symbol=symbols,
            start_date=start,
            end_date=end,
            interval="1d",
            provider=self.provider,
        )
        df = out.to_dataframe().reset_index()

        # Normalise the two shapes OpenBB returns: single-symbol frames have
        # no `symbol`/`ticker` column, multi-symbol frames do.
        lower = {c.lower(): c for c in df.columns}
        date_col = lower.get("date") or df.columns[0]
        df = df.rename(columns={date_col: "date"})
        symbol_col = lower.get("symbol") or lower.get("ticker")
        if symbol_col is not None:
            df = df.rename(columns={symbol_col: "symbol"})
        elif len(symbols) == 1:
            # No symbol-identifying column at all -- OpenBB's documented
            # single-symbol shape. Safe only because there's exactly one
            # requested symbol to attribute every row to.
            df["symbol"] = symbols[0]
        else:
            # More than one symbol was requested but the response has no
            # column identifying which row belongs to which. Blindly
            # labeling every row symbols[0] here -- the previous
            # behavior -- would silently merge unrelated price series
            # into one column with no error. Fail loudly instead of
            # fabricating an attribution the provider never gave us.
            raise ValueError(
                f"requested {len(symbols)} symbols but the response has no "
                f"'symbol' or 'ticker' column to attribute rows to -- "
                f"columns: {list(df.columns)}, provider={self.provider!r}"
            )

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
        names = _require_symbol_sequence(symbols) if symbols else self.universe()
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
    if not panels:
        raise ValueError("align_panels requires at least one panel")
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
