"""Backtest engine and performance metrics.

Design decisions that matter more than the code:

**Signals on bar T, fills on bar T+1.** A strategy scores the market using
data through the close of the decision bar, and the resulting orders fill at
the *next* bar. Computing a signal from a close and filling at that same close
is the most common way a backtest invents returns that do not exist. Set
``delay_bars=0`` only if you know why you want it.

**Sizing happens at execution, not decision.** Target weights are converted to
share counts using the equity and prices at the fill, not those at the signal.

**The gate is consulted every rebalance and its reasoning is persisted.** The
``audit`` frame on the result is the record of why the book looked the way it
did, which is the artefact you actually want six months later.

**Costs are charged, always.** Zero commission is not zero cost: there is
spread, there is slippage, and on the sell side there are regulatory fees. A
strategy whose edge is smaller than its costs is a losing strategy, and the
only way to find out is to model them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .corporate import CorpsPanel
from .data import PricePanel
from .fundamentals import FundamentalsPanel
from .macro import MacrosPanel
from .options import OptionsPanel
from .risk import DayTradeLedger, RiskContext, RiskGate
from .signals import Strategy

__all__ = [
    "CostModel",
    "ExecutionConfig",
    "BacktestResult",
    "Backtester",
    "rebalance_dates",
    "summary_stats",
    "drawdown_series",
    "compare",
]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Costs and execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Transaction costs in basis points of traded notional.

    Defaults approximate a retail equity account at a zero-commission broker
    trading liquid large caps. ``slippage_bps`` is the one to stress: it is
    where half the spread plus market impact lives, and it scales with how
    illiquid and how large your orders are. For small caps or size, 5bps is
    optimistic by a wide margin.

    ``sell_fee_bps`` stands in for the SEC transaction fee and FINRA trading
    activity fee, which are charged on sales only and are small but real.
    """

    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    sell_fee_bps: float = 0.8
    fixed_per_trade: float = 0.0

    def fill_price(self, price: float | pd.Series, side: int | pd.Series):
        """Price after slippage. ``side`` is +1 to buy, -1 to sell."""
        return price * (1.0 + (self.slippage_bps / 1e4) * side)

    def fees(self, buy_notional: float, sell_notional: float, n_trades: int) -> float:
        gross = buy_notional + sell_notional
        return (
            gross * self.commission_bps / 1e4
            + sell_notional * self.sell_fee_bps / 1e4
            + n_trades * self.fixed_per_trade
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """How target weights become fills."""

    delay_bars: int = 1
    price: str = "open"  # "open" or "close"
    allow_fractional: bool = True
    min_trade_notional: float = 1.0

    def __post_init__(self) -> None:
        if self.price not in ("open", "close"):
            raise ValueError("price must be 'open' or 'close'")
        if self.delay_bars < 0:
            raise ValueError("delay_bars must be >= 0")
        if self.delay_bars == 0 and self.price == "open":
            # The signal is computed from the close of bar i. Filling at the
            # open of that same bar means trading on information that did not
            # exist yet. Refuse rather than silently produce fiction.
            raise ValueError(
                "delay_bars=0 with price='open' fills before the signal exists. "
                "Use delay_bars=1 (realistic) or price='close' (optimistic)."
            )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def rebalance_dates(index: pd.DatetimeIndex, freq: str | int) -> pd.DatetimeIndex:
    """Decision bars. ``freq`` is 'D', 'W', 'M', 'Q', 'A'/'Y', or every-n-bars as int.

    Implemented by period grouping rather than ``resample`` so it behaves the
    same across pandas versions, and so the returned dates are always real
    trading days present in ``index``. Matches on the first letter, so
    ``"Monthly"``/``"monthly"``/``"M"`` are all fine -- but an unrecognized
    string (a typo, or a pandas-style offset like ``"3M"`` that this
    function does not support) raises rather than silently rebalancing
    every single day, which is what a bad string used to fall back to.
    """
    if isinstance(freq, int):
        if freq < 1:
            raise ValueError("integer freq must be >= 1")
        return index[::freq]

    if not freq:
        freq = "M"
    first = freq.upper()[0]
    if first == "D":
        return index
    key = {"W": "W", "M": "M", "Q": "Q", "A": "Y", "Y": "Y"}.get(first)
    if key is None:
        raise ValueError(
            f"unrecognized rebalance freq {freq!r}; use 'D', 'W', 'M', 'Q', "
            "'A'/'Y', or an integer bar count"
        )
    periods = index.to_period(key)
    s = pd.Series(np.arange(len(index)), index=periods)
    last = s.groupby(level=0).last().to_numpy()
    return index[np.sort(last)]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    holdings: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    audit: pd.DataFrame
    costs: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def exposure(self) -> pd.Series:
        return self.holdings.abs().sum(axis=1)

    def summary(self) -> pd.Series:
        return summary_stats(self)

    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    def turnover(self) -> pd.Series:
        """One-way turnover per rebalance, as a fraction of equity."""
        if self.trades.empty:
            return pd.Series(dtype=float)
        notional = self.trades.groupby("date")["notional"].apply(
            lambda s: s.abs().sum()
        )
        eq = self.equity.reindex(notional.index).ffill()
        return (notional / eq).rename("turnover")

    def cost_drag(self) -> float:
        """Total costs as a fraction of starting equity."""
        if self.equity.empty:
            return float("nan")
        return float(self.costs.sum() / self.equity.iloc[0])

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BacktestResult(cagr={s['cagr']:.2%}, vol={s['ann_vol']:.2%}, "
            f"sharpe={s['sharpe']:.2f}, maxdd={s['max_drawdown']:.2%}, "
            f"trades={int(s['n_trades'])})"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Walk a panel bar by bar, rebalancing on schedule.

    Parameters
    ----------
    panel:
        Full history. The engine never hands a strategy more than the slice up
        to the current decision bar.
    strategy:
        Any object satisfying :class:`qbt.signals.Strategy`.
    fundamentals:
        Optional. Full filing history; the engine truncates it to
        ``fundamentals.as_of(date)`` before every decision, the same way it
        truncates ``panel``, so a strategy that reads it can't see a filing
        before its own filing date. Strategies that ignore the argument are
        unaffected either way -- pass ``None`` (or leave it out) and nothing
        about their behaviour changes.
    macros:
        Optional. Full macro-indicator history, truncated to
        ``macros.as_of(date)`` the same way. Economy-wide rather than
        per-symbol -- see :mod:`qbt.macro`.
    corps:
        Optional. Full corporate-filings-indicator history (filing cadence,
        insider activity), truncated the same way -- see
        :mod:`qbt.corporate`.
    options:
        Optional. Full options-indicator history, truncated the same way --
        see :mod:`qbt.options` for why this one is realistically only
        populated for backtests over a window you've personally archived.
    risk_gate:
        Optional. Omit to study raw strategy behaviour with no risk overlay.
    rebalance:
        'D', 'W', 'M', 'Q' or an integer bar count.
    """

    def __init__(
        self,
        panel: PricePanel,
        strategy: Strategy,
        fundamentals: FundamentalsPanel | None = None,
        macros: MacrosPanel | None = None,
        corps: CorpsPanel | None = None,
        options: OptionsPanel | None = None,
        risk_gate: RiskGate | None = None,
        cost_model: CostModel | None = None,
        execution: ExecutionConfig | None = None,
        initial_equity: float = 25_000.0,
        rebalance: str | int = "M",
        warmup: int | None = None,
        day_trade_ledger: DayTradeLedger | None = None,
    ) -> None:
        self.panel = panel
        self.strategy = strategy
        self.fundamentals = fundamentals
        self.macros = macros
        self.corps = corps
        self.options = options
        self.gate = risk_gate
        self.costs = cost_model or CostModel()
        self.exec = execution or ExecutionConfig()
        self.initial_equity = float(initial_equity)
        self.rebalance = rebalance
        self.warmup = warmup if warmup is not None else strategy.min_history
        self.ledger = day_trade_ledger or DayTradeLedger()

        if self.exec.price == "open" and panel.open_ is None:
            raise ValueError(
                "execution price 'open' requires a panel with open_ data; "
                "pass ExecutionConfig(price='close') instead"
            )

    # -- internals --------------------------------------------------------

    def _exec_prices(self, i: int) -> pd.Series:
        if self.exec.price == "open" and self.panel.open_ is not None:
            px = self.panel.open_.iloc[i]
            # Fall back to the close where a provider left the open blank.
            return px.where(px.notna(), self.panel.close.iloc[i])
        return self.panel.close.iloc[i]

    def _size(
        self, weights: pd.Series, equity: float, prices: pd.Series
    ) -> pd.Series:
        px = prices.replace(0.0, np.nan)
        raw = (weights.reindex(px.index).fillna(0.0) * equity) / px
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not self.exec.allow_fractional:
            raw = np.sign(raw) * np.floor(raw.abs())
        return raw

    # -- main loop --------------------------------------------------------

    def run(self) -> BacktestResult:
        if self.gate is not None:
            self.gate.reset()

        close = self.panel.close
        dates = self.panel.dates
        n = len(dates)
        symbols = self.panel.symbols

        decision_days = set(rebalance_dates(dates, self.rebalance))

        shares = pd.Series(0.0, index=symbols, dtype=float)
        cash = self.initial_equity
        peak = self.initial_equity
        opened_on: dict[str, pd.Timestamp] = {}

        # A queue, not a single slot -- a single `tuple | None` here used to
        # silently drop an entire rebalance whenever a new decision arrived
        # before the previous one's fill executed (any rebalance cadence
        # faster than delay_bars, e.g. rebalance="D" with delay_bars=2):
        # the second decision's assignment overwrote the first with no
        # error, no warning, and no trace, while target_rows/audit_rows
        # still recorded the dropped decision as if it were normal. A
        # decision's own fill_bar (i + delay_bars) is strictly increasing
        # in i, so two entries can never collide on the same bar -- this
        # queue never grows past what a single pending fill already
        # implied, it just stops erasing history.
        pending: list[tuple[int, pd.Series]] = []
        equity_curve = np.full(n, np.nan)
        cost_curve = np.zeros(n)
        holdings_rows: dict[pd.Timestamp, pd.Series] = {}
        target_rows: dict[pd.Timestamp, pd.Series] = {}
        audit_rows: list[dict] = []
        trade_rows: list[dict] = []

        # Hot-path arrays. The per-bar mark-to-market runs once per trading day,
        # so doing it in pandas turns a 4k-bar backtest into ~12k Series
        # constructions. numpy here, pandas only at decision and fill bars.
        close_np = close.to_numpy(dtype=float)
        holdings_np = np.zeros((n, len(symbols)), dtype=float)
        state = {"cash": self.initial_equity, "shares_np": np.zeros(len(symbols))}

        # Returns are a pure function of closes, so precompute once and slice.
        # Recomputing pct_change over a growing history at every rebalance is
        # the other O(n^2) trap.
        all_returns = self.panel.returns()
        ret_window = 504
        if self.gate is not None:
            ret_window = max(ret_window, getattr(self.gate, "vol_lookback", 0) + 10)

        def execute(bar: int, target_w: pd.Series) -> None:
            """Turn target weights into fills at ``bar``'s execution price."""
            date_ = dates[bar]
            px = self._exec_prices(bar)
            held = pd.Series(state["shares_np"], index=symbols)
            # Must use the execution price, not this bar's close: at an open
            # fill the close does not exist yet, and sizing against it would
            # reintroduce look-ahead through the back door.
            equity_now = state["cash"] + float(
                np.nansum(state["shares_np"] * px.to_numpy(dtype=float))
            )
            if equity_now <= 0:
                return

            target_sh = self._size(target_w, equity_now, px)
            delta = (target_sh - held).fillna(0.0)
            notional = (delta * px).fillna(0.0)
            delta = delta.mask(notional.abs() < self.exec.min_trade_notional, 0.0)
            traded = delta[delta.abs() > 0]
            if traded.empty:
                return

            side = np.sign(traded)
            ref_px = px.reindex(traded.index)
            fill_px = self.costs.fill_price(ref_px, side)
            cash_flow = float((traded * fill_px).sum())
            buys = traded[traded > 0]
            sells = traded[traded < 0]
            buy_notional = float((buys * fill_px.reindex(buys.index)).sum())
            sell_notional = float(-(sells * fill_px.reindex(sells.index)).sum())
            fees = self.costs.fees(buy_notional, sell_notional, len(traded))

            state["cash"] -= cash_flow + fees
            slip_cost = float((traded.abs() * ref_px).sum()) * self.costs.slippage_bps / 1e4
            cost_curve[bar] += fees + slip_cost

            for sym, qty in traded.items():
                prev = float(held.get(sym, 0.0))
                new = prev + float(qty)
                if prev == 0.0 and new != 0.0:
                    opened_on[sym] = date_
                if new == 0.0 and opened_on.get(sym) == date_:
                    self.ledger.record(date_)
                if new == 0.0:
                    opened_on.pop(sym, None)
                trade_rows.append(
                    {
                        "date": date_,
                        "symbol": sym,
                        "shares": float(qty),
                        "price": float(fill_px[sym]),
                        "notional": float(qty * fill_px[sym]),
                        "side": "buy" if qty > 0 else "sell",
                    }
                )
            state["shares_np"] = (
                held + traded.reindex(held.index).fillna(0.0)
            ).round(10).to_numpy()

        for i in range(n):
            date = dates[i]
            px_close = close.iloc[i]

            # --- 1. fill anything scheduled for this bar -----------------
            if pending:
                due = [queued for fill_bar, queued in pending if fill_bar == i]
                if due:
                    pending = [p for p in pending if p[0] != i]
                    for queued in due:
                        execute(i, queued)

            # --- 2. decide, using data through this bar's close ----------
            if date in decision_days and (i + 1) >= self.warmup:
                equity_at_decision = state["cash"] + float(
                    np.nansum(state["shares_np"] * close_np[i])
                )
                peak_at_decision = max(peak, equity_at_decision)
                view = self.panel.as_of(date)
                fview = (
                    self.fundamentals.as_of(date)
                    if self.fundamentals is not None
                    else None
                )
                mview = (
                    self.macros.as_of(date) if self.macros is not None else None
                )
                cview = (
                    self.corps.as_of(date) if self.corps is not None else None
                )
                oview = (
                    self.options.as_of(date) if self.options is not None else None
                )
                proposed = self.strategy.target_weights(
                    view, fview, mview, cview, oview
                )

                if self.gate is not None:
                    ctx = RiskContext(
                        date=date,
                        equity=equity_at_decision,
                        peak_equity=peak_at_decision,
                        prices=px_close,
                        returns=all_returns.iloc[max(0, i - ret_window + 1) : i + 1],
                        positions=pd.Series(state["shares_np"], index=symbols),
                        day_trades_remaining=self.ledger.remaining(
                            date, equity_at_decision
                        ),
                    )
                    decision = self.gate.apply(proposed, ctx)
                    final_w = decision.weights
                    audit_rows.append({"date": date, **decision.as_record()})
                else:
                    final_w = proposed
                    audit_rows.append(
                        {
                            "date": date,
                            "vol_scale": 1.0,
                            "gross_before": float(proposed.abs().sum()),
                            "gross_after": float(proposed.abs().sum()),
                            "forecast_vol": float("nan"),
                            "halted": False,
                            "n_positions": int((proposed.abs() > 1e-9).sum()),
                            "notes": "no risk gate",
                        }
                    )

                target_rows[date] = final_w
                if self.exec.delay_bars == 0:
                    execute(i, final_w)
                    audit_rows[-1]["scheduled"] = True
                elif i + self.exec.delay_bars < n:
                    pending.append((i + self.exec.delay_bars, final_w))
                    audit_rows[-1]["scheduled"] = True
                else:
                    # Near the panel's end: no bar left to schedule this
                    # decision's fill at (i + delay_bars would run past
                    # the last bar). The decision is still recorded here
                    # and in target_rows for transparency, but it was
                    # never actually executed -- flag it so a reader of
                    # the tail of these frames doesn't mistake "decided"
                    # for "filled." Not a look-ahead or return-fabrication
                    # bug, just an unavoidable edge effect of a finite
                    # panel; the flag exists so it's visible, not silent.
                    audit_rows[-1]["scheduled"] = False

            # --- 3. mark to market on the close --------------------------
            position_value = state["shares_np"] * close_np[i]
            equity = state["cash"] + float(np.nansum(position_value))
            equity_curve[i] = equity
            peak = max(peak, equity)
            if equity > 0:
                holdings_np[i] = np.nan_to_num(position_value) / equity

        shares = pd.Series(state["shares_np"], index=symbols)

        equity_s = pd.Series(equity_curve, index=dates, name="equity").ffill()
        returns_s = equity_s.pct_change(fill_method=None).fillna(0.0).rename("returns")

        holdings = pd.DataFrame(holdings_np, index=dates, columns=symbols)
        targets = (
            pd.DataFrame(target_rows).T if target_rows else pd.DataFrame(columns=symbols)
        )
        trades = (
            pd.DataFrame(trade_rows)
            if trade_rows
            else pd.DataFrame(columns=["date", "symbol", "shares", "price", "notional", "side"])
        )
        audit = (
            pd.DataFrame(audit_rows).set_index("date")
            if audit_rows
            else pd.DataFrame()
        )

        return BacktestResult(
            equity=equity_s,
            returns=returns_s,
            holdings=holdings,
            targets=targets,
            trades=trades,
            audit=audit,
            costs=pd.Series(cost_curve, index=dates, name="costs"),
            meta={
                "strategy": self.strategy.name,
                "rebalance": self.rebalance,
                "delay_bars": self.exec.delay_bars,
                "exec_price": self.exec.price,
                "initial_equity": self.initial_equity,
                "slippage_bps": self.costs.slippage_bps,
                "warmup": self.warmup,
                "day_trades": len(self.ledger.events),
            },
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def summary_stats(result: BacktestResult, rf_annual: float = 0.0) -> pd.Series:
    eq = result.equity.dropna()
    r = result.returns.reindex(eq.index).fillna(0.0)

    if len(eq) < 2:
        return pd.Series(dtype=float)

    years = len(eq) / TRADING_DAYS
    total = eq.iloc[-1] / eq.iloc[0]
    cagr = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)

    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r - rf_daily
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
        if excess.std() > 0
        else np.nan
    )
    downside = excess[excess < 0].std()
    sortino = (
        excess.mean() / downside * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan
    )

    dd = drawdown_series(eq)
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    active = r[r != 0.0]
    exposure = result.exposure.reindex(eq.index).fillna(0.0)
    to = result.turnover()

    return pd.Series(
        {
            "cagr": cagr,
            "ann_vol": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
            "best_day": float(r.max()),
            "worst_day": float(r.min()),
            "avg_exposure": float(exposure.mean()),
            "time_invested": float((exposure > 0.01).mean()),
            "turnover_ann": float(to.sum() / years) if len(to) else 0.0,
            "cost_drag": result.cost_drag(),
            "n_trades": float(len(result.trades)),
            "n_rebalances": float(len(result.targets)),
            "final_equity": float(eq.iloc[-1]),
            "years": years,
        }
    )


def compare(results: dict[str, BacktestResult], rf_annual: float = 0.0) -> pd.DataFrame:
    """Side-by-side summary table for several runs."""
    return pd.DataFrame(
        {name: summary_stats(res, rf_annual) for name, res in results.items()}
    ).T
