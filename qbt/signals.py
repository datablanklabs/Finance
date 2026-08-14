"""Signal engine.

Every strategy implements one method::

    target_weights(
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series

``view`` is already sliced to the decision bar, so the last row of
``view.close`` is "today". ``fundamentals``, ``macros``, ``corps``, and
``options``, when the caller has them, have already been through the same
firewall -- see e.g. :meth:`~qbt.fundamentals.FundamentalsPanel.as_of` -- so
a strategy is free to call ``.snapshot(view.last_date())`` on any of them
without re-deriving the cutoff itself. ``fundamentals``, ``corps``, and
``options`` are per symbol; ``macros`` is economy-wide (no symbol axis --
every name in the universe sees the same reading). Most strategies here
don't use any of them and simply ignore the parameters; they exist so the
ones that do (or that you write) don't need a diverging method signature.
Strategies are pure: no I/O, no broker calls, no mutation of any input, no
access to the future. The returned Series is indexed by symbol and holds
*fractions of equity*; it need not sum to one -- anything unallocated is
cash, and the risk gate is what enforces gross limits.

That purity is the whole point of the design. The identical strategy object is
called by :mod:`qbt.engine` during a backtest and by :mod:`qbt.live` when
generating real orders, so there is no second implementation to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .corporate import DEFAULT_WINDOW_DAYS, CorpsPanel
from .data import PricePanel
from .fundamentals import FundamentalsPanel
from .macro import MacrosPanel
from .options import OptionsPanel

__all__ = [
    "Strategy",
    "EqualWeightBuyHold",
    "CrossSectionalMomentum",
    "TimeSeriesMomentum",
    "ShortHorizonReversal",
    "TrendFilter",
    "FundamentalsValueFilter",
    "MacroRegimeFilter",
    "Composite",
    "InverseVolWeighted",
    "PairsTrading",
    "MultiFactorCrossSectional",
    "CalendarSeasonality",
    "RiskParityAllocation",
    "OptionsMeanReversion",
    "InsiderEventDrift",
]


@runtime_checkable
class Strategy(Protocol):
    """The contract. Implement these three members and the engine accepts it."""

    name: str

    @property
    def min_history(self) -> int:
        """Bars of history required before this strategy emits weights."""
        ...

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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


def _trailing_ma(view: PricePanel, lookback: int) -> tuple[pd.Series, pd.Series]:
    """Trailing ``lookback``-bar mean, plus a mask of who it's actually valid for.

    Returns ``(ma, complete)`` where ``complete`` marks symbols whose
    trailing window is *fully* populated -- the same all-bars-present rule
    :func:`_tradeable` applies, so a filter built on this agrees with the
    strategies it wraps about who has enough history.

    Splitting the mask out is the whole point. ``DataFrame.mean()`` skips
    NaN, so a symbol with 20 real bars inside a 200-bar window returns a
    perfectly finite 20-bar average rather than NaN -- a "200-day moving
    average" that is nothing of the sort, and one that ``ma.isna()`` cannot
    detect. Confirmed (2026-08): a name listed 20 bars ago passed a 200-day
    trend gate outright, and counted as fully scoreable in a breadth read.
    Callers must consult ``complete``; the mean alone cannot tell you
    whether it means anything.
    """
    window = view.close.tail(lookback)
    ma = window.mean()
    complete = window.notna().sum() >= lookback
    return ma, complete


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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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
        Number of names held. ``None`` means hold the top half of the
        universe by score (``len(scores) // 2``, at least one name).

        Note this is *relative* momentum throughout: the top names are held
        whatever the sign of their trailing return, so in a universe where
        every name is falling this still runs fully invested in the ones
        falling least. That is the definition, not an oversight -- an
        absolute-momentum gate on top of it is what :class:`TrendFilter`
        is for, and composing the two is the usual dual-momentum setup.
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
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n must be >= 1 (or None)")

    @property
    def min_history(self) -> int:
        return self.lookback + self.skip + 2

    def score(self, view: PricePanel) -> pd.Series:
        """Exposed separately so research code can study the raw signal."""
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=float)
        return view.select(names).trailing_return(self.lookback, self.skip).dropna()

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = self.inner.target_weights(view, fundamentals, macros, corps, options)
        if w.abs().sum() == 0:
            return w
        ma, complete = _trailing_ma(view, self.lookback)
        last = view.close.iloc[-1]
        # A symbol whose trailing `lookback` window isn't fully populated --
        # a recently-added or short-history name that still cleared the
        # *inner* strategy's shorter min_history -- must block, not pass
        # through. Two distinct ways that used to leak, both fixed here:
        #
        # 1. `last <= ma` evaluates to False, not NaN, when `ma` is NaN, so
        #    a plain comparison silently granted a position zero trend
        #    confirmation -- exactly backwards for a filter whose entire job
        #    is requiring that confirmation.
        # 2. `ma` is usually not NaN at all in this case. mean() skips NaN,
        #    so a 20-of-200-bar window yields a finite 20-bar average and an
        #    isna() check sees nothing wrong with it. `complete` (see
        #    _trailing_ma) is what actually catches this, and it's the same
        #    all-bars-present rule _tradeable applies.
        #
        # Same fail-safe direction as TimeSeriesMomentum.qualifies and
        # FundamentalsValueFilter elsewhere in this module -- unknown means
        # blocked, not unblocked.
        insufficient_history = ma.isna() | last.isna() | ~complete
        blocked = ((last <= ma) | insufficient_history).reindex(w.index).fillna(True)
        return w.mask(blocked, 0.0)


@dataclass
class FundamentalsValueFilter:
    """Wrap a strategy and zero out names failing a fundamentals screen.

    Same role :class:`TrendFilter` plays for price trend, applied to a
    fundamentals ratio instead: momentum, reversal, whatever picks the
    names, this decides which of those picks are allowed to trade. Weight
    removed by the filter becomes cash rather than being redistributed to
    the names that pass -- same reasoning as :class:`TrendFilter`: this is
    meant to reduce exposure to picks that fail the screen, not reshuffle
    conviction among survivors.

    ``metric`` must be a column :class:`~qbt.fundamentals.
    FundamentalsRepository` actually produced (e.g. ``"ratios_pe_ratio"``)
    -- check ``fundamentals.metrics`` if unsure. Set ``max_value`` for a
    cheap-valuation screen (block names above it, e.g. a P/E ceiling),
    ``min_value`` for a floor (e.g. a minimum margin), or both for a band.

    Two different "we don't know" cases get two different answers, and the
    difference matters: a name with no reading for ``metric`` is blocked --
    the same conservative default :class:`TrendFilter` uses for names with
    too little price history, since trading a name you can't screen defeats
    the point of screening. But ``fundamentals`` being entirely absent
    (``None``, or a caller who never wired up a
    :class:`~qbt.fundamentals.FundamentalsRepository`) passes the inner
    strategy's weights through unchanged rather than blocking everything --
    the alternative would make this filter capable of silently flattening
    an entire book just because someone forgot the ``fundamentals=``
    keyword, which is a far more dangerous failure mode than a no-op.
    """

    inner: Strategy
    metric: str
    max_value: float | None = None
    min_value: float | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}+{self.metric}filter"
        if self.max_value is None and self.min_value is None:
            raise ValueError("set at least one of max_value/min_value")

    @property
    def min_history(self) -> int:
        return self.inner.min_history

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = self.inner.target_weights(view, fundamentals, macros, corps, options)
        if w.abs().sum() == 0:
            return w
        if fundamentals is None or len(fundamentals.frame) == 0:
            return w

        snap = fundamentals.snapshot(view.last_date())
        if self.metric not in snap.columns:
            return w

        values = snap[self.metric].reindex(w.index)
        blocked = values.isna()
        if self.max_value is not None:
            blocked = blocked | (values > self.max_value)
        if self.min_value is not None:
            blocked = blocked | (values < self.min_value)
        return w.mask(blocked, 0.0)


@dataclass
class MacroRegimeFilter:
    """Wrap a strategy and de-risk the whole book in an unfavourable macro regime.

    Different axis from :class:`TrendFilter`/:class:`FundamentalsValueFilter`
    on purpose: those decide *which names* trade, per symbol. Macro data has
    no symbol axis -- see :mod:`qbt.macro` -- so this decides *how much of
    the book* trades, as a single scalar applied to every position, the same
    role portfolio-level vol targeting plays in :class:`qbt.risk.RiskGate`.
    Compose them the same way :class:`InverseVolWeighted` composes with the
    risk gate: this changes total exposure, the picks underneath are
    untouched.

    Configure any combination of ``max_level``/``min_level`` (block outside
    an absolute range -- e.g. only trade while ``fed_funds_rate`` is below
    some level) and ``max_increase`` (block after too sharp a rise over
    ``lookback`` trading days -- e.g. a rate hiking cycle). ``metric`` must
    be a column :class:`~qbt.macro.MacrosRepository` actually produced --
    check ``macros.metrics`` if unsure.

    Same reasoning as :class:`FundamentalsValueFilter` for what happens when
    the data isn't there, with the same asymmetry and for the same reason:
    ``macros`` being entirely absent passes the inner strategy's weights
    through unchanged (a no-op, not a silent full flatten from a missed
    keyword argument). It's specifically the *regime read as unfavourable*
    that scales the book down -- absence of information is not itself
    treated as bad news.
    """

    inner: Strategy
    metric: str
    max_level: float | None = None
    min_level: float | None = None
    max_increase: float | None = None
    lookback: int = 63
    scale_when_blocked: float = 0.0
    max_age_days: int | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}+{self.metric}regime"
        if self.max_level is None and self.min_level is None and self.max_increase is None:
            raise ValueError(
                "set at least one of max_level/min_level/max_increase"
            )
        if not 0.0 <= self.scale_when_blocked <= 1.0:
            raise ValueError("scale_when_blocked must be in [0, 1]")
        if self.max_age_days is not None and self.max_age_days < 1:
            raise ValueError("max_age_days must be >= 1 (or None to disable)")

    @property
    def min_history(self) -> int:
        return max(self.inner.min_history, self.lookback + 2)

    def blocked(self, view: PricePanel, macros: MacrosPanel | None) -> bool:
        """Exposed separately so research code can study the regime read directly.

        A reading older than ``max_age_days`` is treated as no reading at
        all -- the same no-op pass-through as ``macros=None``, per this
        class's own "absence of information is not bad news" rule. Without
        it a series that quietly stopped updating keeps answering with its
        last value indefinitely, and this filter would go on gating a live
        book on a months-old number believing it current.
        """
        if macros is None or len(macros.frame) == 0:
            return False

        today = view.last_date()
        snap_now = macros.snapshot(today, max_age_days=self.max_age_days)
        if self.metric not in snap_now.index:
            return False
        level_now = float(snap_now[self.metric])

        if self.max_level is not None and level_now > self.max_level:
            return True
        if self.min_level is not None and level_now < self.min_level:
            return True
        if self.max_increase is not None and len(view.dates) > self.lookback:
            prior_date = view.dates[-(self.lookback + 1)]
            snap_then = macros.snapshot(prior_date)
            if self.metric in snap_then.index:
                change = level_now - float(snap_then[self.metric])
                if change > self.max_increase:
                    return True
        return False

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = self.inner.target_weights(view, fundamentals, macros, corps, options)
        if w.abs().sum() == 0:
            return w
        if self.blocked(view, macros):
            return w * self.scale_when_blocked
        return w


@dataclass
class BreadthRegimeFilter:
    """Wrap a strategy and de-risk the whole book when market breadth is weak.

    Same axis as :class:`MacroRegimeFilter`: this decides *how much of the
    book* trades, not which names -- but the regime read comes from the
    price panel's own cross-section, not an external data source. No new
    data dependency: every strategy already receives a
    :class:`~qbt.data.PricePanel`.

    "Breadth" here is participation within the *strategy's own tradeable
    universe*, not some broad-market index this system doesn't necessarily
    trade: the fraction of symbols currently trading above their own
    trailing ``lookback``-bar moving average. A handful of names carrying
    the whole basket higher while most lag behind is a classically
    fragile setup -- ``min_breadth`` scales the book down (or fully flat,
    the default) once that fraction falls below the floor.

    Same asymmetry as :class:`MacroRegimeFilter`, for the same reason: a
    breadth reading that can't be computed at all (every symbol lacking
    enough of its own history to score, which ordinarily shouldn't
    survive the panel's own ``min_history`` gate, but is handled
    defensively anyway) passes the inner strategy's weights through
    unchanged rather than blocking. It's specifically a *breadth read as
    weak* that scales the book down, not the absence of a read -- and,
    unlike :class:`TrendFilter`'s per-name NaN handling, a symbol that
    can't be scored here is excluded from *both* the numerator and the
    denominator, not silently treated as passing (or failing) the
    threshold.
    """

    inner: Strategy
    lookback: int = 200
    min_breadth: float = 0.4
    scale_when_blocked: float = 0.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}+breadth{self.lookback}"
        if not 0.0 <= self.min_breadth <= 1.0:
            raise ValueError("min_breadth must be in [0, 1]")
        if not 0.0 <= self.scale_when_blocked <= 1.0:
            raise ValueError("scale_when_blocked must be in [0, 1]")

    @property
    def min_history(self) -> int:
        return max(self.inner.min_history, self.lookback + 2)

    def breadth(self, view: PricePanel) -> float:
        """Fraction of the universe trading above its own trailing moving
        average. NaN if not a single symbol has enough history to score.
        Exposed separately so research code can study the raw signal, the
        same reason :meth:`MacroRegimeFilter.blocked` is its own method.

        "Enough history" means a *fully populated* trailing window, not
        merely a non-NaN average -- mean() skips NaN, so a name with 20 real
        bars inside a 200-bar window otherwise counts as fully scoreable
        while contributing a 20-bar average to a 200-bar breadth read. Such
        a name is excluded from numerator and denominator alike, which is
        the same exclusion this class's docstring already promised for
        unscoreable symbols; it just wasn't catching this case.
        """
        ma, complete = _trailing_ma(view, self.lookback)
        last = view.close.iloc[-1]
        scoreable = ma.notna() & last.notna() & complete
        if not scoreable.any():
            return float("nan")
        above = (last > ma) & scoreable
        return float(above.sum()) / float(scoreable.sum())

    def blocked(self, view: PricePanel) -> bool:
        """Exposed separately so research code can study the regime read directly."""
        b = self.breadth(view)
        return bool(np.isfinite(b) and b < self.min_breadth)

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = self.inner.target_weights(view, fundamentals, macros, corps, options)
        if w.abs().sum() == 0:
            return w
        if self.blocked(view):
            return w * self.scale_when_blocked
        return w


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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        out = _empty(view)
        for (strat, _), share in zip(self.members, self._shares):
            member_w = strat.target_weights(view, fundamentals, macros, corps, options)
            out = out.add(member_w.reindex(out.index).fillna(0.0) * share)
        return out

    def contributions(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.DataFrame:
        """Per-member weights, for attribution during research."""
        cols = {}
        for (strat, _), share in zip(self.members, self._shares):
            member_w = strat.target_weights(view, fundamentals, macros, corps, options)
            cols[strat.name] = member_w.reindex(view.symbols).fillna(0.0) * share
        return pd.DataFrame(cols)


@dataclass
class InverseVolWeighted:
    """Re-weight a strategy's picks inversely to their own volatility.

    Distinct from the portfolio-level vol targeting in :mod:`qbt.risk`: this
    changes the *relative* sizing inside the book, the risk gate changes the
    *total*. Keeping them separate means you can study each in isolation.

    A pick whose volatility can't be measured (too little history in the
    ``vol_lookback`` window) is dropped, and its weight becomes **cash** --
    it is not silently absorbed by the picks that could be measured. Gross
    is preserved across the *scoreable* subset only. Redistributing it
    would quietly lever up the remaining names on the strength of a name
    this wrapper just admitted it can't size, and it would make this the
    one wrapper in the module that grows a position in response to missing
    data -- see :class:`TrendFilter` and :class:`MacroRegimeFilter`, where
    removed weight always becomes cash.
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = self.inner.target_weights(view, fundamentals, macros, corps, options)
        held = w[w.abs() > 0]
        if held.empty:
            return w
        vol = view.select(list(held.index)).realised_vol(self.vol_lookback)
        inv = (1.0 / vol.replace(0.0, np.nan)).dropna()
        if inv.empty:
            return w
        # Gross of the scoreable names only, not of everything the inner
        # strategy picked -- an unscoreable pick's weight becomes cash
        # rather than being handed to its neighbours.
        scoreable = held.reindex(inv.index)
        gross = scoreable.abs().sum()
        out = _empty(view)
        out.loc[inv.index] = np.sign(scoreable) * gross * inv / inv.sum()
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n must be >= 1 (or None)")

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
        if not np.isfinite(sd):
            # A non-finite sd means the factor could not be measured at all
            # (every observation NaN), which is *not* the same as "measured,
            # and everyone scored alike". Returning zeros for it would let a
            # missing factor pass silently through the dropna(how="any")
            # below and contribute a confident nothing to every name's
            # score -- exactly the "silently scored on a subset" case this
            # class's docstring says it avoids.
            return pd.Series(np.nan, index=s.index, dtype=float)
        if sd == 0.0:
            # Genuinely no cross-sectional dispersion: the factor was
            # measured and simply does not separate anyone today. Zero for
            # everyone is the right contribution, and names missing an
            # individual reading still carry NaN through.
            return pd.Series(0.0, index=s.index).mask(s.isna())
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
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


# ---------------------------------------------------------------------------
# Options-derived mean reversion
# ---------------------------------------------------------------------------


def _zscore_last(daily: pd.DataFrame) -> pd.Series:
    """Per-column time-series z-score of the last row against its own window.

    Distinct from :meth:`MultiFactorCrossSectional._zscore`, which is
    *cross-sectional* (one snapshot, many symbols). This is the opposite
    axis: one symbol's own recent history, which is what "elevated relative
    to normal" has to mean for a signal like implied vol that has wildly
    different baseline levels across symbols.
    """
    mu = daily.mean()
    sd = daily.std(ddof=0).replace(0.0, np.nan)
    z = (daily.iloc[-1] - mu) / sd
    return z.dropna()


@dataclass
class OptionsMeanReversion:
    """Contrarian: buy names where options-implied fear is most stretched.

    Two options-market gauges function as fear indicators: elevated implied
    vol (``iv_atm_near``) and unusually heavy put buying relative to calls
    (``put_call_volume_ratio``). Both are scored *against each symbol's own
    trailing history* (a time-series z-score, not a cross-sectional one --
    a 40% IV name and an 80% IV name can both be "stretched" for themselves),
    blended by ``iv_weight``/``pcr_weight``, and the names with the highest
    combined score are bought at equal weight. The bet is the same one
    behind fading a VIX spike or extreme put/call ratio: elevated
    options-implied stress tends to fade faster than it resolves into an
    actual price decline, so the mean reversion is in the fear gauge, not
    necessarily in the price move that produced it.

    This is the one strategy in this module built for options data
    specifically because it's the one new data source in this package that
    is genuinely daily -- see :mod:`qbt.options`. It only works with names
    that have enough archived options history (``iv_window`` trading days);
    names without it are silently skipped rather than scored on partial
    data. Note the risk profile, same caveat as :class:`ShortHorizonReversal`:
    this is short-vol-shaped. Fear is sometimes right. A name that's
    "stretched" because something is genuinely deteriorating will keep
    getting more stretched, and this strategy has no way to tell the
    difference from here.
    """

    iv_window: int = 20
    iv_weight: float = 0.5
    pcr_weight: float = 0.5
    min_z: float = 1.0
    top_n: int = 5
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"optrevert_{self.iv_window}_{self.top_n}"
        if not (self.iv_weight or self.pcr_weight):
            raise ValueError("at least one of iv_weight/pcr_weight must be nonzero")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

    @property
    def min_history(self) -> int:
        return self.iv_window + 2

    def score(self, view: PricePanel, options: OptionsPanel | None) -> pd.Series:
        """Exposed separately so research code can study the raw signal."""
        if options is None or len(options.frame) == 0:
            return pd.Series(dtype=float)
        names = _tradeable(view, self.min_history)
        names = [n for n in names if n in options.symbols]
        if not names:
            return pd.Series(dtype=float)

        need = min(self.iv_window + 1, len(view.dates))
        dates = view.dates[-need:]
        if len(dates) < 2:
            return pd.Series(dtype=float)

        daily = options.to_daily(
            dates, symbols=names, metrics=["iv_atm_near", "put_call_volume_ratio"]
        )

        parts = []
        iv = daily.get("iv_atm_near")
        if iv is not None and self.iv_weight:
            parts.append(self.iv_weight * _zscore_last(iv))
        pcr = daily.get("put_call_volume_ratio")
        if pcr is not None and self.pcr_weight:
            parts.append(self.pcr_weight * _zscore_last(pcr))
        if not parts:
            return pd.Series(dtype=float)

        return pd.concat(parts, axis=1).dropna(how="any").sum(axis=1)

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = _empty(view)
        z = self.score(view, options)
        if z.empty:
            return w
        candidates = z[z >= abs(self.min_z)]
        if candidates.empty:
            return w
        picks = candidates.nlargest(min(self.top_n, len(candidates)))
        w.loc[picks.index] = self.gross / len(picks)
        return w


# ---------------------------------------------------------------------------
# Insider-buying / 8-K event drift
# ---------------------------------------------------------------------------


@dataclass
class InsiderEventDrift:
    """Long fresh insider buying that follows an 8-K, held for a short drift window.

    An 8-K alone is a coin flip -- it just means something happened, not
    whether it's good news. Insiders buying in the open market shortly
    after one is a much more specific signal: they've seen the same event
    the market has and are putting their own money on it being overreacted
    to (or simply fine). That combination -- a recent 8-K *and* fresh net
    insider buying -- is the entry condition here, which is a real, if
    modest, empirical effect (see Lakonishok & Lee 2001 on insider-trading
    drift, and the broader post-8-K/PEAD literature this rhymes with).

    The "held for a short drift window" part is deliberately *not*
    implemented with position-entry bookkeeping. Every other strategy in
    this module is a pure function of today's data that "unwinds on its
    own" as the underlying signal fades (see :class:`PairsTrading`'s
    docstring) rather than remembering when it opened a position, and this
    one follows the same discipline: it compares today's
    :meth:`~qbt.corporate.CorpsPanel.snapshot` against the snapshot from
    ``drift_days`` trading days ago, and only holds a name while its
    trailing insider-buy/8-K counts have *increased* since then -- i.e.
    while a qualifying event is still inside the window. Once the event
    ages out, the comparison stops firing and the position exits on the
    next rebalance, with no state carried between calls.

    ``window_days`` must match whatever
    :class:`~qbt.corporate.CorpsRepository` produced ``corps`` (it's baked
    into the metric names, e.g. ``filed_8k_count_90d``) -- a mismatch isn't
    an error, it just means every name silently fails to qualify, since a
    missing column reads as "no activity" rather than "unknown." Cheap to
    get wrong, easy to check: compare ``corps.metrics`` against
    ``qbt.corporate.DEFAULT_WINDOW_DAYS`` or whatever you configured.
    """

    drift_days: int = 10
    window_days: int = DEFAULT_WINDOW_DAYS
    require_8k: bool = True
    top_n: int | None = 10
    weighting: str = "equal"
    gross: float = 1.0
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"insiderdrift_{self.drift_days}_{self.top_n}"
        if self.weighting not in ("equal", "rank"):
            raise ValueError("weighting must be 'equal' or 'rank'")
        if self.drift_days < 1:
            raise ValueError("drift_days must be >= 1")
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n must be >= 1 (or None)")

    @property
    def min_history(self) -> int:
        return self.drift_days + 2

    def _metric_col(self, snap: pd.DataFrame, base: str, names: list[str]) -> pd.Series:
        metric = f"{base}_{self.window_days}d"
        if metric not in snap.columns:
            return pd.Series(0.0, index=names)
        return snap[metric].reindex(names).fillna(0.0)

    def score(self, view: PricePanel, corps: CorpsPanel | None) -> pd.Series:
        """Net insider shares for names with a fresh qualifying event.

        Exposed separately so research code can study the raw signal.
        Positive-valued only -- a name mid-drift with net insider *selling*
        doesn't qualify no matter how recent the 8-K was.
        """
        if corps is None or len(corps.frame) == 0:
            return pd.Series(dtype=float)
        names = _tradeable(view, self.min_history)
        if not names:
            return pd.Series(dtype=float)
        if len(view.dates) <= self.drift_days:
            return pd.Series(dtype=float)

        today = view.last_date()
        prior_date = view.dates[-(self.drift_days + 1)]
        now = corps.snapshot(today)
        then = corps.snapshot(prior_date)
        if now.empty:
            return pd.Series(dtype=float)

        buy_now = self._metric_col(now, "insider_buy_count", names)
        buy_then = self._metric_col(then, "insider_buy_count", names)
        fresh_buy = buy_now > buy_then

        if self.require_8k:
            eightk_now = self._metric_col(now, "filed_8k_count", names)
            eightk_then = self._metric_col(then, "filed_8k_count", names)
            fresh_buy &= eightk_now > eightk_then

        net_now = self._metric_col(now, "insider_net_shares", names)
        qualifies = fresh_buy & (net_now > 0)
        return net_now[qualifies]

    def target_weights(
        self,
        view: PricePanel,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> pd.Series:
        w = _empty(view)
        candidates = self.score(view, corps)
        if candidates.empty:
            return w

        k = len(candidates) if self.top_n is None else min(self.top_n, len(candidates))
        picks = candidates.nlargest(k)

        if self.weighting == "equal":
            w.loc[picks.index] = self.gross / len(picks)
        else:
            ranks = picks.rank()
            w.loc[picks.index] = self.gross * ranks / ranks.sum()
        return w
