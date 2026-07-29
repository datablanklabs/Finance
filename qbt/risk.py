"""Risk gate.

Everything that can veto or shrink a trade lives here, deliberately separate
from strategy logic. Two reasons that separation matters:

1. You can change risk policy without touching a strategy, and study either
   in isolation.
2. The gate is the *only* place with authority to reduce exposure, so there is
   a single audit point. Every decision returns the reasons it made, and the
   engine persists them -- when you ask "why was I flat in March", the answer
   is in the log rather than reconstructed by hand.

The gate is a pure function of ``(proposed_weights, RiskContext)``. All state
lives in the context, which the caller assembles. That keeps it trivially
testable and identical between backtest and live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["RiskContext", "RiskDecision", "RiskGate", "DayTradeLedger"]


# ---------------------------------------------------------------------------
# Pattern day trader accounting
# ---------------------------------------------------------------------------


@dataclass
class DayTradeLedger:
    """Rolling count of day trades under FINRA's pattern day trader rule.

    A day trade is opening and closing the same security on the same session.
    In a margin account under the equity threshold you get three per rolling
    five business days. Cash accounts are exempt from the rule but subject to
    settlement, which is a different constraint.

    Monthly or weekly rebalancing essentially never trips this. It binds when
    you shorten the holding period, which is exactly when you are least likely
    to be watching for it -- hence tracking it from day one.
    """

    limit: int = 3
    window_days: int = 5
    equity_threshold: float = 25_000.0
    events: list[pd.Timestamp] = field(default_factory=list)

    def count(self, asof: pd.Timestamp) -> int:
        if not self.events:
            return 0
        cutoff = pd.Timestamp(asof) - pd.tseries.offsets.BDay(self.window_days)
        return sum(1 for d in self.events if d > cutoff)

    def remaining(self, asof: pd.Timestamp, equity: float) -> int:
        if equity >= self.equity_threshold:
            return 10_000  # effectively unlimited
        return max(0, self.limit - self.count(asof))

    def record(self, date: pd.Timestamp) -> None:
        self.events.append(pd.Timestamp(date))


# ---------------------------------------------------------------------------
# Context and decision
# ---------------------------------------------------------------------------


@dataclass
class RiskContext:
    """Everything the gate needs to know about the current state of the world."""

    date: pd.Timestamp
    equity: float
    peak_equity: float
    prices: pd.Series
    returns: pd.DataFrame
    positions: pd.Series | None = None
    day_trades_remaining: int = 10_000

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity


@dataclass
class RiskDecision:
    """Result of the gate: final weights plus why they look like that."""

    weights: pd.Series
    vol_scale: float = 1.0
    gross_before: float = 0.0
    gross_after: float = 0.0
    forecast_vol: float = float("nan")
    halted: bool = False
    notes: list[str] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "vol_scale": self.vol_scale,
            "gross_before": self.gross_before,
            "gross_after": self.gross_after,
            "forecast_vol": self.forecast_vol,
            "halted": self.halted,
            "n_positions": int((self.weights.abs() > 1e-9).sum()),
            "notes": "; ".join(self.notes),
        }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass
class RiskGate:
    """Shrink or veto a proposed portfolio.

    Applied in order:

    1. **Per-name cap** -- clip any single weight to ``max_weight``.
    2. **Volatility target** -- scale the whole book so its forecast
       volatility is ``target_vol``. Forecast comes from the proposed weights
       and a trailing covariance matrix, not from a single asset's vol, so
       correlation is accounted for.
    3. **Gross cap** -- hard ceiling on total exposure. Leave at 1.0 for a
       cash account; above 1.0 requires margin and changes your risk profile
       more than the number suggests.
    4. **Drawdown breaker** -- go flat once peak-to-trough loss exceeds
       ``max_drawdown``, and stay flat until equity recovers past
       ``resume_at`` of the peak.

    Set ``target_vol=None`` to disable vol targeting and study the raw signal.
    """

    target_vol: float | None = 0.12
    vol_lookback: int = 126
    max_vol_scale: float = 1.0
    min_vol_scale: float = 0.0
    max_weight: float = 0.25
    max_gross: float = 1.0
    max_drawdown: float | None = 0.20
    resume_at: float = 0.90
    _halted: bool = field(default=False, init=False, repr=False)

    # -- helpers ----------------------------------------------------------

    def forecast_vol(self, weights: pd.Series, returns: pd.DataFrame) -> float:
        """Annualised forecast volatility of the proposed book."""
        held = weights[weights.abs() > 1e-9]
        if held.empty:
            return 0.0
        hist = returns[[c for c in held.index if c in returns.columns]]
        hist = hist.tail(self.vol_lookback).dropna(how="all")
        if hist.shape[0] < 20 or hist.shape[1] == 0:
            return float("nan")
        w = held.reindex(hist.columns).fillna(0.0).to_numpy()
        cov = hist.cov().to_numpy() * 252.0
        var = float(w @ cov @ w)
        return float(np.sqrt(max(var, 0.0)))

    # -- main -------------------------------------------------------------

    def apply(self, proposed: pd.Series, ctx: RiskContext) -> RiskDecision:
        notes: list[str] = []
        w = proposed.astype(float).fillna(0.0).copy()
        gross_before = float(w.abs().sum())

        # 1. per-name cap
        if self.max_weight is not None:
            over = w.abs() > self.max_weight + 1e-12
            if over.any():
                notes.append(
                    f"capped {int(over.sum())} name(s) at {self.max_weight:.0%}"
                )
                w = w.clip(-self.max_weight, self.max_weight)

        # 2. volatility target
        fvol = self.forecast_vol(w, ctx.returns)
        scale = 1.0
        if self.target_vol is not None and np.isfinite(fvol) and fvol > 1e-8:
            scale = self.target_vol / fvol
            scale = float(np.clip(scale, self.min_vol_scale, self.max_vol_scale))
            if abs(scale - 1.0) > 0.01:
                notes.append(
                    f"vol scale {scale:.2f} (forecast {fvol:.1%} vs "
                    f"target {self.target_vol:.1%})"
                )
            w = w * scale
        elif self.target_vol is not None and not np.isfinite(fvol):
            notes.append("vol forecast unavailable, no scaling applied")

        # 3. gross cap
        gross = float(w.abs().sum())
        if self.max_gross is not None and gross > self.max_gross + 1e-12:
            w = w * (self.max_gross / gross)
            notes.append(f"gross trimmed {gross:.2f} -> {self.max_gross:.2f}")

        # 4. drawdown breaker
        halted = False
        if self.max_drawdown is not None:
            dd = ctx.drawdown
            if self._halted:
                if ctx.equity >= self.resume_at * ctx.peak_equity:
                    self._halted = False
                    notes.append("drawdown breaker reset")
                else:
                    halted = True
            elif dd >= self.max_drawdown:
                self._halted = True
                halted = True
                notes.append(
                    f"drawdown breaker tripped at {dd:.1%} "
                    f"(limit {self.max_drawdown:.0%})"
                )
            if halted:
                w = w * 0.0
                if "breaker tripped" not in " ".join(notes):
                    notes.append(f"halted, drawdown {dd:.1%}")

        if ctx.day_trades_remaining <= 0:
            notes.append("no day trades remaining under PDT rule")

        return RiskDecision(
            weights=w,
            vol_scale=scale,
            gross_before=gross_before,
            gross_after=float(w.abs().sum()),
            forecast_vol=fvol,
            halted=halted,
            notes=notes,
        )

    def reset(self) -> None:
        """Clear latched state. Call between independent backtest runs."""
        self._halted = False
