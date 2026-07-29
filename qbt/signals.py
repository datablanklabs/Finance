"""Signal engine.

Every strategy implements one method::

    target_weights(view: PricePanel) -> pd.Series

``view`` is already sliced to the decision bar, so the last row of
``view.close`` is "today". Strategies are pure: no I/O, no broker calls, no
mutation of the view, no access to the future. The returned Series is indexed
by symbol and holds *fractions of equity*; it need not sum to one -- anything
unallocated is cash, and the risk gate is what enforces gross limits.

That purity is the whole point of the design. The identical strategy object is
called by :mod:`qbt.engine` during a backtest and by :mod:`qbt.live` when
generating real orders, so there is no second implementation to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .data import PricePanel

__all__ = [
    "Strategy",
    "EqualWeightBuyHold",
    "CrossSectionalMomentum",
    "TimeSeriesMomentum",
    "ShortHorizonReversal",
    "TrendFilter",
    "Composite",
    "InverseVolWeighted",
    "PairsTrading",
    "MultiFactorCrossSectional",
    "CalendarSeasonality",
    "RiskParityAllocation",
]


@runtime_checkable
class Strategy(Protocol):
    """The contract. Implement these three members and the engine accepts it."""

    name: str

    @property
    def min_history(self) -> int:
        """Bars of history required before this strategy emits weights."""
        ...

    def target_weights(self, view: PricePanel) -> pd.Series:
        """Desired portfolio as fractions of equity, indexed by symbol."""
        ...


def _empty(view: PricePanel) -> pd.Series:
    return pd.Series(0.0, index=view.symbols, dtype=float)


def _tradeable(view: PricePanel, min_history: int) -> list[str]:
    """Symbols with a price today and enough history to score."""
    recent = view.close.tail(min_history)
    ok = recent.notna().sum() >= min_history
    live = view.close.iloc[-1].notna()
    return list(view.close.columns[ok & live])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class EqualWeightBuyHold:
    """Hold every tradeable name at equal weight. The bar to beat."""

    name: str = "equal_weight"
    gross: float = 1.0

    @property
    def min_history(self) -> int:
        return 2

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        names = _tradeable(view, self.min_history)
        if names:
            w.loc[names] = self.gross / len(names)
        return w


# ---------------------------------------------------------------------------
# Cross-sectional momentum
# ---------------------------------------------------------------------------


@dataclass
class CrossSectionalMomentum:
    """Rank the universe by trailing return; hold the top ``top_n``.

    Parameters
    ----------
    lookback:
        Formation window in bars.
    skip:
        Bars excluded at the recent end. Standard in the momentum literature
        because short-horizon reversal works against you in the last week or
        so of the formation window.
    top_n:
        Number of names held. ``None`` means hold every name with a positive
        rank score above the median.
    weighting:
        ``"equal"`` or ``"rank"`` (linear in cross-sectional rank).
    """

    lookback: int = 126
    skip: int = 5
    top_n: int | None = 10
    weighting: str = "equal"
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"xsmom_{self.lookback}_{self.skip}_{self.top_n}"
        if self.weighting not in ("equal", "rank"):
            raise ValueError("weighting must be 'equal' or 'rank'")

    @property
    def min_history(self) -> int:
        return self.lookback + self.skip + 2

    def score(self, view: PricePanel) -> pd.Series:
        """Exposed separately so research code can study the raw signal."""
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=float)
        return view.select(names).trailing_return(self.lookback, self.skip).dropna()

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        scores = self.score(view)
        if scores.empty:
            return w

        k = self.top_n if self.top_n is not None else max(1, len(scores) // 2)
        k = min(k, len(scores))
        picks = scores.nlargest(k)

        if self.weighting == "equal":
            w.loc[picks.index] = self.gross / k
        else:
            ranks = picks.rank()
            w.loc[picks.index] = self.gross * ranks / ranks.sum()
        return w


# ---------------------------------------------------------------------------
# Time-series momentum
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesMomentum:
    """Absolute momentum: hold each name only while it is in its own uptrend.

    A name qualifies when its last close is above its ``lookback``-bar moving
    average and its trailing return is positive. Qualifying names are held at
    equal weight; the unallocated remainder sits in cash. This rarely raises
    returns much and reliably cuts drawdown.
    """

    lookback: int = 200
    require_positive_return: bool = True
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"tsmom_{self.lookback}"

    @property
    def min_history(self) -> int:
        return self.lookback + 2

    def qualifies(self, view: PricePanel) -> pd.Series:
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=bool)
        sub = view.select(names)
        ma = sub.close.tail(self.lookback).mean()
        last = sub.close.iloc[-1]
        ok = last > ma
        if self.require_positive_return:
            ok &= sub.trailing_return(self.lookback) > 0.0
        return ok.fillna(False)

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        ok = self.qualifies(view)
        held = list(ok.index[ok])
        if held:
            w.loc[held] = self.gross / len(held)
        return w


# ---------------------------------------------------------------------------
# Short-horizon reversal
# ---------------------------------------------------------------------------


@dataclass
class ShortHorizonReversal:
    """Buy the most oversold names and hold for a few bars.

    Scores each name by the z-score of its ``lookback``-bar return relative to
    its own recent distribution, then buys the ``top_n`` most negative. This is
    the strategy family Robinhood uses as its own agentic-trading example. Note
    the risk profile: it is short volatility in disguise, so it prints a smooth
    curve until it doesn't. The ``min_z`` gate keeps it out of the market when
    nothing is genuinely stretched.
    """

    lookback: int = 5
    z_window: int = 63
    top_n: int = 5
    min_z: float = 1.0
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"revert_{self.lookback}_{self.top_n}"

    @property
    def min_history(self) -> int:
        return self.z_window + self.lookback + 2

    def score(self, view: PricePanel) -> pd.Series:
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=float)
        need = self.z_window + self.lookback + 2
        close = view.select(names).close.tail(need)
        roll = close.pct_change(self.lookback, fill_method=None)
        window = roll.tail(self.z_window)
        mu, sd = window.mean(), window.std()
        sd = sd.replace(0.0, np.nan)
        return ((roll.iloc[-1] - mu) / sd).dropna()

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        z = self.score(view)
        if z.empty:
            return w
        candidates = z[z <= -abs(self.min_z)]
        if candidates.empty:
            return w
        picks = candidates.nsmallest(min(self.top_n, len(candidates)))
        w.loc[picks.index] = self.gross / len(picks)
        return w


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class TrendFilter:
    """Wrap a strategy and zero out names not in their own uptrend.

    Composing :class:`CrossSectionalMomentum` with this gives you dual
    momentum: relative strength picks the names, absolute momentum decides
    whether to be invested at all. Weight removed by the filter becomes cash
    rather than being redistributed, which is the point -- the filter is meant
    to reduce exposure in bad regimes, not reshuffle it.
    """

    inner: Strategy
    lookback: int = 200
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}+trend{self.lookback}"

    @property
    def min_history(self) -> int:
        return max(self.inner.min_history, self.lookback + 2)

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = self.inner.target_weights(view)
        if w.abs().sum() == 0:
            return w
        ma = view.close.tail(self.lookback).mean()
        last = view.close.iloc[-1]
        blocked = (last <= ma).reindex(w.index).fillna(True)
        return w.mask(blocked, 0.0)


@dataclass
class Composite:
    """Blend several strategies by fixed capital shares.

    ``members`` is a sequence of ``(strategy, share)``. Shares are normalised.
    Because each member returns weights in the same units, blending is just a
    weighted sum -- which is why the interface returns fractions of equity
    rather than share counts.
    """

    members: Sequence[tuple[Strategy, float]]
    name: str = "composite"

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("Composite needs at least one member")
        total = sum(s for _, s in self.members)
        if total <= 0:
            raise ValueError("shares must sum to a positive number")
        self._shares = [s / total for _, s in self.members]

    @property
    def min_history(self) -> int:
        return max(s.min_history for s, _ in self.members)

    def target_weights(self, view: PricePanel) -> pd.Series:
        out = _empty(view)
        for (strat, _), share in zip(self.members, self._shares):
            out = out.add(strat.target_weights(view).reindex(out.index).fillna(0.0) * share)
        return out

    def contributions(self, view: PricePanel) -> pd.DataFrame:
        """Per-member weights, for attribution during research."""
        cols = {}
        for (strat, _), share in zip(self.members, self._shares):
            cols[strat.name] = strat.target_weights(view).reindex(view.symbols).fillna(0.0) * share
        return pd.DataFrame(cols)


@dataclass
class InverseVolWeighted:
    """Re-weight a strategy's picks inversely to their own volatility.

    Distinct from the portfolio-level vol targeting in :mod:`qbt.risk`: this
    changes the *relative* sizing inside the book, the risk gate changes the
    *total*. Keeping them separate means you can study each in isolation.
    """

    inner: Strategy
    vol_lookback: int = 63
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}+ivol"

    @property
    def min_history(self) -> int:
        return max(self.inner.min_history, self.vol_lookback + 2)

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = self.inner.target_weights(view)
        held = w[w.abs() > 0]
        if held.empty:
            return w
        vol = view.select(list(held.index)).realised_vol(self.vol_lookback)
        inv = (1.0 / vol.replace(0.0, np.nan)).dropna()
        if inv.empty:
            return w
        gross = held.abs().sum()
        out = _empty(view)
        out.loc[inv.index] = np.sign(held.reindex(inv.index)) * gross * inv / inv.sum()
        return out


# ---------------------------------------------------------------------------
# Pairs trading / statistical arbitrage
# ---------------------------------------------------------------------------


@dataclass
class PairsTrading:
    """Market-neutral stat-arb: trade the spread between correlated pairs.

    Each rebalance re-selects the ``n_pairs`` most correlated symbol pairs
    over the trailing ``formation`` window (greedily, each symbol used at
    most once), then scores every pair by the z-score of its log-price
    spread over ``z_window`` bars. A pair only trades once its spread is at
    least ``entry_z`` standard deviations from its own recent mean: long the
    laggard, short the leader, equal dollar legs. There is no separate exit
    threshold or held-position state -- like :class:`ShortHorizonReversal`,
    the position is a pure function of today's z-score, so it unwinds on its
    own once the spread reverts inside ``entry_z``.

    This is deliberately the simplest version: equal-dollar legs rather than
    a beta-hedged spread, and correlation rather than a cointegration test
    for pair selection. Both are reasonable first extensions.
    """

    formation: int = 126
    z_window: int = 21
    entry_z: float = 2.0
    n_pairs: int = 10
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"pairs_{self.formation}_{self.z_window}_{self.entry_z}"
        if self.formation < 2:
            raise ValueError("formation must be >= 2")
        if self.z_window < 2:
            raise ValueError("z_window must be >= 2")
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be >= 1")

    @property
    def min_history(self) -> int:
        return max(self.formation, self.z_window) + 2

    def select_pairs(self, view: PricePanel) -> list[tuple[str, str, float]]:
        """Greedy, non-overlapping pairs ranked by trailing correlation."""
        names = _tradeable(view, self.min_history)
        if len(names) < 2:
            return []
        rets = view.select(names).tail_returns(self.formation)
        corr = rets.corr()
        candidates = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                c = corr.loc[a, b]
                if np.isfinite(c):
                    candidates.append((a, b, float(c)))
        candidates.sort(key=lambda t: t[2], reverse=True)

        chosen: list[tuple[str, str, float]] = []
        used: set[str] = set()
        for a, b, c in candidates:
            if a in used or b in used:
                continue
            chosen.append((a, b, c))
            used.add(a)
            used.add(b)
            if len(chosen) >= self.n_pairs:
                break
        return chosen

    def pair_zscore(self, view: PricePanel, a: str, b: str) -> float:
        need = self.z_window + 1
        spread = (np.log(view.close[a]) - np.log(view.close[b])).tail(need)
        if spread.isna().any() or len(spread) < need:
            return float("nan")
        mu, sd = spread.mean(), spread.std()
        if not sd or sd == 0.0:
            return float("nan")
        return float((spread.iloc[-1] - mu) / sd)

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        pairs = self.select_pairs(view)
        if not pairs:
            return w

        active = []
        for a, b, _ in pairs:
            z = self.pair_zscore(view, a, b)
            if np.isfinite(z) and abs(z) >= self.entry_z:
                active.append((a, b, z))
        if not active:
            return w

        leg = self.gross / (2 * len(active))
        for a, b, z in active:
            # Spread rich (z > 0): a is expensive relative to b -- short a,
            # long b. Spread cheap (z < 0): the opposite.
            if z > 0:
                w[a] -= leg
                w[b] += leg
            else:
                w[a] += leg
                w[b] -= leg
        return w


# ---------------------------------------------------------------------------
# Multi-factor cross-sectional
# ---------------------------------------------------------------------------


@dataclass
class MultiFactorCrossSectional:
    """Blend several price-derived cross-sectional factors into one score.

    Fundamentals (value, quality) aren't available from a price-only panel
    (see the README's "no point-in-time data" gap), so every factor here is
    derived from price and volatility: ``momentum`` (trailing return over
    ``momentum_lookback``, skipping the most recent ``momentum_skip`` bars,
    same construction as :class:`CrossSectionalMomentum`), ``low_vol``
    (inverse realised volatility over ``vol_lookback``), and ``reversal``
    (negative short-horizon return over ``reversal_lookback`` -- the one
    factor here with real independent information, since a price-only stand-in
    for "quality" doesn't exist). Each factor is cross-sectionally z-scored
    before blending so factors on different natural scales don't dominate by
    magnitude alone; names missing any factor are dropped rather than
    silently scored on a subset.
    """

    momentum_lookback: int = 126
    momentum_skip: int = 5
    vol_lookback: int = 63
    reversal_lookback: int = 5
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {"momentum": 0.5, "low_vol": 0.3, "reversal": 0.2}
    )
    top_n: int | None = 10
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"multifactor_{self.top_n}"
        unknown = set(self.factor_weights) - {"momentum", "low_vol", "reversal"}
        if unknown:
            raise ValueError(f"unknown factor(s) in factor_weights: {unknown}")
        if not any(self.factor_weights.values()):
            raise ValueError("factor_weights must have at least one nonzero entry")

    @property
    def min_history(self) -> int:
        return (
            max(
                self.momentum_lookback + self.momentum_skip,
                self.vol_lookback,
                self.reversal_lookback,
            )
            + 2
        )

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        if not sd or not np.isfinite(sd) or sd == 0.0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd

    def score(self, view: PricePanel) -> pd.Series:
        """Exposed separately so research code can study the blended signal."""
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=float)
        sub = view.select(names)

        parts = []
        w = self.factor_weights.get("momentum", 0.0)
        if w:
            mom = sub.trailing_return(self.momentum_lookback, self.momentum_skip)
            parts.append(self._zscore(mom) * w)
        w = self.factor_weights.get("low_vol", 0.0)
        if w:
            vol = sub.realised_vol(self.vol_lookback)
            parts.append(-self._zscore(vol) * w)
        w = self.factor_weights.get("reversal", 0.0)
        if w:
            rev = sub.trailing_return(self.reversal_lookback)
            parts.append(-self._zscore(rev) * w)
        if not parts:
            return pd.Series(dtype=float)

        combined = pd.concat(parts, axis=1).dropna(how="any").sum(axis=1)
        return combined

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        scores = self.score(view)
        if scores.empty:
            return w

        k = self.top_n if self.top_n is not None else max(1, len(scores) // 2)
        k = min(k, len(scores))
        picks = scores.nlargest(k)
        w.loc[picks.index] = self.gross / k
        return w


# ---------------------------------------------------------------------------
# Calendar / seasonality
# ---------------------------------------------------------------------------


@dataclass
class CalendarSeasonality:
    """Hold the market only in the turn-of-month window; flat otherwise.

    Equity returns are empirically concentrated in the last ``pre_days``
    trading days of the calendar month and the first ``post_days`` of the
    next (Ariel 1987; Lakonishok & Smirlov 1988). The window is derived from
    the decision date's position in its own calendar month -- ``today.day``
    versus ``today.days_in_month`` -- which needs no price history and no
    knowledge of future bars, so it does not violate the look-ahead firewall.
    It's a calendar-day approximation of the trading-day window (it doesn't
    account for holidays shifting the actual month-end trading session by a
    day or two), which is a reasonable simplification for a signal already
    this coarse. ``avoid_weekdays`` optionally excludes specific weekdays
    (0=Monday ... 4=Friday) even inside the window, for the Monday effect.

    Needs a ``rebalance`` schedule finer than the window it's detecting --
    run the :class:`~qbt.engine.Backtester` with ``rebalance='D'`` or
    ``'W'``. Monthly rebalancing only ever decides on the last trading day
    of the month, which is inside the window on essentially every call, so
    the strategy would look permanently invested rather than doing what it
    says.
    """

    pre_days: int = 3
    post_days: int = 3
    avoid_weekdays: tuple[int, ...] = ()
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"calendar_tom_{self.pre_days}_{self.post_days}"
        if self.pre_days < 0 or self.post_days < 0:
            raise ValueError("pre_days and post_days must be >= 0")
        if any(d not in range(7) for d in self.avoid_weekdays):
            raise ValueError("avoid_weekdays entries must be 0 (Mon) .. 6 (Sun)")

    @property
    def min_history(self) -> int:
        return 2

    def in_window(self, view: PricePanel) -> bool:
        today = view.last_date()
        if today.weekday() in self.avoid_weekdays:
            return False
        near_month_end = today.day > today.days_in_month - self.pre_days
        near_month_start = today.day <= self.post_days
        return near_month_end or near_month_start

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        names = _tradeable(view, self.min_history)
        if not names or not self.in_window(view):
            return w
        w.loc[names] = self.gross / len(names)
        return w


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


def _risk_parity_weights(cov: pd.DataFrame, max_iter: int = 200, tol: float = 1e-9) -> pd.Series:
    """Equal risk contribution weights via cyclical fixed-point iteration.

    Solves the standard log-barrier risk-parity formulation's first-order
    condition, ``w_i * (Sigma w)_i = b_i`` for all ``i``, by iterating
    ``w <- b / (Sigma w)`` and renormalising to sum to one (Bruder & Roncalli
    2012). No external optimiser needed. Falls back to equal weight if the
    covariance diagonal is degenerate (e.g. too little history).
    """
    n = len(cov)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov.index)

    sigma = cov.to_numpy()
    diag = np.diag(sigma)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
        return pd.Series(1.0 / n, index=cov.index)

    budget = np.full(n, 1.0 / n)
    weights = 1.0 / np.sqrt(np.clip(diag, 1e-12, None))
    weights = weights / weights.sum()

    for _ in range(max_iter):
        marginal = np.clip(sigma @ weights, 1e-12, None)
        updated = budget / marginal
        updated = updated / updated.sum()
        if np.max(np.abs(updated - weights)) < tol:
            weights = updated
            break
        weights = updated

    return pd.Series(weights, index=cov.index)


@dataclass
class RiskParityAllocation:
    """Equal risk contribution across the full tradeable universe.

    Sizes every tradeable name so it contributes the same share of total
    portfolio variance, using the trailing covariance matrix -- not just
    inverse volatility, which ignores correlation and misjudges how much a
    genuinely diversifying (low-correlation) name should get. Distinct from
    :class:`InverseVolWeighted`, which re-weights *another strategy's picks*
    by vol alone; this both chooses the picks (the whole universe) and
    accounts for correlation. Also distinct from
    :class:`qbt.risk.RiskGate`'s vol targeting, which scales the whole book
    to a target level rather than balancing risk *within* it -- the two
    compose fine, same as :class:`InverseVolWeighted` does.

    ``max_names`` caps the universe before computing the covariance matrix,
    because estimation error grows with the number of names relative to
    ``cov_lookback``; the cap keeps the most liquid names (by trailing
    volume) rather than the lowest-vol ones, to avoid quietly turning this
    into a low-vol factor strategy.
    """

    cov_lookback: int = 126
    max_names: int | None = 30
    gross: float = 1.0
    max_iter: int = 200
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"riskparity_{self.cov_lookback}"
        if self.cov_lookback < 2:
            raise ValueError("cov_lookback must be >= 2")

    @property
    def min_history(self) -> int:
        return self.cov_lookback + 2

    def target_weights(self, view: PricePanel) -> pd.Series:
        w = _empty(view)
        names = _tradeable(view, self.min_history)
        if not names:
            return w
        if len(names) == 1:
            w.loc[names] = self.gross
            return w

        if self.max_names is not None and len(names) > self.max_names:
            if view.volume is not None:
                liquidity = view.select(names).volume.tail(21).mean()
                names = list(liquidity.nlargest(self.max_names).index)
            else:
                names = names[: self.max_names]

        cov = view.select(names).covariance(self.cov_lookback)
        cov = cov.dropna(how="all").dropna(how="all", axis=1)
        names = [n for n in names if n in cov.index]
        if len(names) < 2:
            if names:
                w.loc[names] = self.gross
            return w

        rp = _risk_parity_weights(cov.loc[names, names], max_iter=self.max_iter)
        w.loc[rp.index] = self.gross * rp
        return w
