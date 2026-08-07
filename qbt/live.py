"""Live signal generation.

This module is the payoff for keeping strategies pure. It calls the *same*
strategy object and the *same* risk gate the backtester used, on a panel whose
last bar is today, and emits order intents. There is no second copy of the
sizing logic to drift out of sync.

It deliberately stops short of execution. :class:`LiveSignalRunner` produces
:class:`OrderIntent` records and an audit record; submitting them is the order
manager's job, behind a separate interface with its own idempotency and
reconciliation. Keeping the boundary here means you can run this in a notebook
against your real positions and read the intended trades without any risk of
one escaping.

On the broker side: do not hard-code tool names. Enumerate the MCP server's
tools at runtime and bind to what it advertises, because the schema is the
server's to change and a stale hard-coded name fails at the worst moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .corporate import CorpsPanel
from .data import PricePanel
from .fundamentals import FundamentalsPanel
from .macro import MacrosPanel
from .options import OptionsPanel
from .risk import DayTradeLedger, RiskContext, RiskDecision, RiskGate
from .signals import Strategy

__all__ = ["OrderIntent", "LivePlan", "LiveSignalRunner", "PortfolioState"]


@dataclass
class PortfolioState:
    """Current holdings, as reported by the broker.

    Populate this from a broker read, never from memory. After a crash your
    in-process view of positions is a guess; the broker's view is the truth.
    """

    cash: float
    shares: pd.Series
    peak_equity: float | None = None

    def equity(self, prices: pd.Series) -> float:
        held = self.shares.reindex(prices.index).fillna(0.0)
        return float(self.cash + (held * prices).fillna(0.0).sum())

    def weights(self, prices: pd.Series) -> pd.Series:
        eq = self.equity(prices)
        if eq <= 0:
            return self.shares * 0.0
        held = self.shares.reindex(prices.index).fillna(0.0)
        return (held * prices).fillna(0.0) / eq


@dataclass
class OrderIntent:
    symbol: str
    side: str
    shares: float
    reference_price: float
    notional: float
    current_weight: float
    target_weight: float

    def to_payload(self, order_type: str = "market") -> dict:
        """Broker-agnostic order description.

        Map this onto whatever the MCP server's order tool actually accepts.
        Include your own ``client_order_id`` upstream of this call so a retry
        cannot double-fill.
        """
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": abs(round(self.shares, 6)),
            "order_type": order_type,
            "reference_price": round(self.reference_price, 4),
            "estimated_notional": round(abs(self.notional), 2),
        }


@dataclass
class LivePlan:
    asof: pd.Timestamp
    equity: float
    intents: list[OrderIntent]
    target_weights: pd.Series
    current_weights: pd.Series
    decision: RiskDecision | None
    warnings: list[str] = field(default_factory=list)

    @property
    def turnover(self) -> float:
        if self.equity <= 0:
            return 0.0
        return sum(abs(i.notional) for i in self.intents) / self.equity

    def to_frame(self) -> pd.DataFrame:
        if not self.intents:
            return pd.DataFrame(
                columns=[
                    "symbol", "side", "shares", "reference_price",
                    "notional", "current_weight", "target_weight",
                ]
            )
        return pd.DataFrame([vars(i) for i in self.intents]).sort_values(
            "notional", key=abs, ascending=False
        ).reset_index(drop=True)

    def audit_record(self) -> dict:
        rec = {
            "asof": str(self.asof),
            "equity": round(self.equity, 2),
            "n_intents": len(self.intents),
            "turnover": round(self.turnover, 4),
            "warnings": "; ".join(self.warnings),
        }
        if self.decision is not None:
            rec.update(self.decision.as_record())
        return rec

    def __repr__(self) -> str:
        return (
            f"LivePlan({self.asof.date()}, equity={self.equity:,.0f}, "
            f"{len(self.intents)} intents, turnover={self.turnover:.1%})"
        )


class LiveSignalRunner:
    """Produce today's order intents using backtest-identical logic."""

    def __init__(
        self,
        strategy: Strategy,
        risk_gate: RiskGate | None = None,
        min_trade_notional: float = 10.0,
        allow_fractional: bool = True,
        max_turnover: float | None = 0.67,
        allow_full_turnover_from_flat: bool = True,
        day_trade_ledger: DayTradeLedger | None = None,
    ) -> None:
        self.strategy = strategy
        self.gate = risk_gate
        self.min_trade_notional = min_trade_notional
        self.allow_fractional = allow_fractional
        self.max_turnover = max_turnover
        self.allow_full_turnover_from_flat = allow_full_turnover_from_flat
        self.ledger = day_trade_ledger or DayTradeLedger()

    def plan(
        self,
        panel: PricePanel,
        state: PortfolioState,
        asof: pd.Timestamp | None = None,
        tradeable: Sequence[str] | None = None,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
    ) -> LivePlan:
        """Produce today's plan.

        ``fundamentals``, ``macros``, ``corps``, and ``options``, if given,
        are each truncated here to ``as_of(asof_ts)`` before ever reaching
        the strategy, the same firewall the backtester applies.
        """
        warnings: list[str] = []
        view = panel.as_of(asof) if asof is not None else panel
        asof_ts = view.last_date()
        fview = fundamentals.as_of(asof_ts) if fundamentals is not None else None
        mview = macros.as_of(asof_ts) if macros is not None else None
        cview = corps.as_of(asof_ts) if corps is not None else None
        oview = options.as_of(asof_ts) if options is not None else None

        staleness = (pd.Timestamp.today().normalize() - asof_ts).days
        if staleness > 4:
            warnings.append(f"price data is {staleness} days stale")
        if len(view) < self.strategy.min_history:
            warnings.append(
                f"only {len(view)} bars, strategy wants {self.strategy.min_history}"
            )
            return LivePlan(
                asof=asof_ts,
                equity=state.equity(view.last_close()),
                intents=[],
                target_weights=pd.Series(dtype=float),
                current_weights=pd.Series(dtype=float),
                decision=None,
                warnings=warnings,
            )

        prices = view.last_close()
        equity = state.equity(prices)
        peak = state.peak_equity if state.peak_equity is not None else equity
        current_w = state.weights(prices)

        proposed = self.strategy.target_weights(view, fview, mview, cview, oview)
        if tradeable is not None:
            blocked = [s for s in proposed.index if s not in set(tradeable)]
            if any(abs(proposed.get(s, 0.0)) > 1e-9 for s in blocked):
                warnings.append("some targets are not in the tradeable list")
            proposed = proposed.copy()
            proposed.loc[blocked] = 0.0

        decision: RiskDecision | None = None
        if self.gate is not None:
            ctx = RiskContext(
                date=asof_ts,
                equity=equity,
                peak_equity=peak,
                prices=prices,
                returns=view.returns(),
                positions=state.shares,
                day_trades_remaining=self.ledger.remaining(asof_ts, equity),
            )
            decision = self.gate.apply(proposed, ctx)
            target_w = decision.weights
        else:
            target_w = proposed

        intents = self._to_intents(target_w, current_w, state, prices, equity)

        if self.max_turnover is not None and equity > 0:
            turnover = sum(abs(i.notional) for i in intents) / equity
            # A plan whose turnover lands exactly on the cap must be
            # allowed through unscaled -- "at the limit" is compliant, not
            # a breach. The tolerance exists because "exactly at the cap"
            # is a mathematical statement, not a floating-point one: this
            # is a sum of per-symbol notionals divided by equity, and
            # floating-point rounding can land that a hair above a cap
            # that was conceptually hit precisely (e.g. 0.6700000000000001
            # for a true 0.67), which would otherwise scale or reject a
            # plan for no real reason.
            if turnover > self.max_turnover + 1e-9:
                # A flat account's first-ever buildout is structurally ~100%
                # turnover -- current weights are all zero, so turnover and
                # gross exposure are the same number. Same mechanical
                # exemption as ExecutionPolicy.allow_full_turnover_from_flat
                # in qbt/orders.py: the account currently holds zero
                # positions, full stop, not a judgment call about whether
                # the trade looks reasonable.
                flat = state.shares.empty or bool((state.shares.abs() < 1e-9).all())
                if self.allow_full_turnover_from_flat and flat:
                    warnings.append(
                        f"turnover {turnover:.1%} exceeds cap "
                        f"{self.max_turnover:.0%} but the account holds no "
                        "positions -- the first buildout from cash is exempt "
                        "(allow_full_turnover_from_flat)"
                    )
                else:
                    # Rather than discard the whole plan, shrink every
                    # intent by the same factor so gross turnover lands
                    # exactly at the cap -- the largest plan, still moving
                    # in the direction the strategy actually wants, that
                    # respects the limit. Turnover scales linearly with a
                    # uniform size reduction (same equity denominator, same
                    # trade directions), so the exact factor is closed-form,
                    # not something to search for. current_weight/
                    # target_weight on each intent are left as the
                    # strategy's real read of the book -- only the size of
                    # the trade actually sent this cycle shrinks; the
                    # rest of the move happens on a later cycle once the
                    # book has caught up.
                    scale = self.max_turnover / turnover
                    n_before = len(intents)
                    scaled = []
                    for i in intents:
                        shares = i.shares * scale
                        notional = i.notional * scale
                        if (abs(notional) < self.min_trade_notional
                                or abs(shares) < 1e-9):
                            continue
                        scaled.append(replace(i, shares=shares, notional=notional))
                    intents = scaled
                    warnings.append(
                        f"turnover {turnover:.1%} exceeds cap "
                        f"{self.max_turnover:.0%}, every order scaled to "
                        f"{scale:.0%} of its target size to fit "
                        f"({len(scaled)} of {n_before} orders survive the "
                        "min-notional filter after scaling)"
                    )

        return LivePlan(
            asof=asof_ts,
            equity=equity,
            intents=intents,
            target_weights=target_w,
            current_weights=current_w,
            decision=decision,
            warnings=warnings,
        )

    def _to_intents(
        self,
        target_w: pd.Series,
        current_w: pd.Series,
        state: PortfolioState,
        prices: pd.Series,
        equity: float,
    ) -> list[OrderIntent]:
        if equity <= 0:
            return []
        px = prices.replace(0.0, np.nan)
        target_sh = (target_w.reindex(px.index).fillna(0.0) * equity / px).fillna(0.0)
        if not self.allow_fractional:
            target_sh = np.sign(target_sh) * np.floor(target_sh.abs())

        held = state.shares.reindex(px.index).fillna(0.0)
        delta = (target_sh - held).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        intents = []
        for sym, qty in delta.items():
            price = px.get(sym, np.nan)
            if not np.isfinite(price) or price <= 0:
                continue
            notional = float(qty) * float(price)
            if abs(notional) < self.min_trade_notional or abs(qty) < 1e-9:
                continue
            intents.append(
                OrderIntent(
                    symbol=str(sym),
                    side="buy" if qty > 0 else "sell",
                    shares=float(qty),
                    reference_price=float(price),
                    notional=notional,
                    current_weight=float(current_w.get(sym, 0.0)),
                    target_weight=float(target_w.get(sym, 0.0)),
                )
            )
        return intents
