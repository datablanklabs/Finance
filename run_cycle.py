#!/usr/bin/env python3
"""One trading cycle. Run this on a schedule; never from a notebook.

    python run_cycle.py --dry-run            # default, sends nothing
    python run_cycle.py --live --max-order 50

Exit codes: 0 completed, 1 aborted at preflight, 2 unresolved in-flight order
(halt and investigate before the next cycle), 3 setup or connection failure.

Why this is a separate process from the notebook: it needs a durable journal, a
deterministic single pass, and crash recovery on startup. A notebook gives you
none of those, and re-running a cell would re-submit.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from qbt import (
    CrossSectionalMomentum, LiveSignalRunner, OpenBBRepository, PortfolioState,
    RiskGate, SyntheticRepository,
)
from qbt.broker import MockBroker, RobinhoodMCPBroker
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager

UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
            "XLV", "XLY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]

STRATEGY = CrossSectionalMomentum(lookback=63, skip=5, top_n=5)
GATE = dict(target_vol=0.12, max_weight=0.30, max_gross=1.0, max_drawdown=0.25)
PEAK_FILE = "state/peak_equity.txt"


def load_peak(current: float) -> float:
    """The running equity peak must survive restarts.

    If this is lost, the drawdown breaker forgets the high-water mark. If it is
    seeded too high, the breaker trips immediately and the bot silently never
    trades. Both failures are quiet, so persist it.
    """
    try:
        with open(PEAK_FILE) as fh:
            return max(float(fh.read().strip()), current)
    except (OSError, ValueError):
        return current


def save_peak(value: float) -> None:
    os.makedirs(os.path.dirname(PEAK_FILE) or ".", exist_ok=True)
    with open(PEAK_FILE, "w") as fh:
        fh.write(f"{value:.4f}")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually submit orders (default is dry run)")
    ap.add_argument("--max-order", type=float, default=250.0)
    ap.add_argument("--max-plan", type=float, default=5_000.0)
    ap.add_argument("--max-turnover", type=float, default=0.50)
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data and a mock broker")
    ap.add_argument("--ignore-market-hours", action="store_true")
    args = ap.parse_args()

    audit = AuditLog("audit/orders.jsonl")
    audit.emit("cycle_start", live=args.live, argv=" ".join(sys.argv[1:]))

    # ---- data -----------------------------------------------------------
    try:
        end = str(pd.Timestamp.today().normalize().date())
        if args.synthetic:
            panel = SyntheticRepository(n_symbols=18, seed=42).fetch(
                start="2010-01-01", end=end)
        else:
            panel = OpenBBRepository(provider="yfinance",
                                     cache_dir=".cache/prices").fetch(
                UNIVERSE, "2010-01-01", end)
    except Exception as exc:
        audit.emit("data_fetch_failed", error=repr(exc))
        return 3

    # ---- broker ---------------------------------------------------------
    try:
        if args.synthetic:
            broker = MockBroker(prices=panel.last_close(), cash=25_000.0, seed=0)
        else:
            token = os.environ.get("ROBINHOOD_MCP_TOKEN")
            if not token:
                audit.emit("setup_failed", error="ROBINHOOD_MCP_TOKEN not set")
                return 3
            broker = RobinhoodMCPBroker(token=token, require_agentic=True)
        broker.connect()
    except Exception as exc:
        audit.emit("broker_connect_failed", error=repr(exc))
        return 3

    policy = ExecutionPolicy(
        max_order_notional=args.max_order,
        max_plan_notional=args.max_plan,
        max_plan_turnover=args.max_turnover,
        symbol_allowlist=tuple(panel.symbols) if args.synthetic else tuple(UNIVERSE),
        require_review=True,
        require_market_open=not args.ignore_market_hours,
        dry_run=not args.live,
    )
    manager = OrderManager(broker=broker, policy=policy, audit=audit)

    # ---- recovery before anything else ----------------------------------
    try:
        unresolved = manager.recover()
    except Exception as exc:
        audit.emit("recover_failed", error=repr(exc))
        return 2
    if not unresolved.empty:
        lost = unresolved[unresolved["outcome"] == "not_at_broker"]
        if not lost.empty:
            audit.emit("halt_unresolved_orders", n=len(lost),
                       symbols=", ".join(lost["symbol"]))
            print("HALT: in-flight orders could not be accounted for.")
            print(unresolved.to_string(index=False))
            return 2

    # ---- plan -----------------------------------------------------------
    try:
        account = broker.get_account()
        peak = load_peak(account.equity)
        state = PortfolioState(
            cash=account.cash,
            shares=account.positions.reindex(panel.symbols).fillna(0.0),
            peak_equity=peak,
        )
        plan = LiveSignalRunner(strategy=STRATEGY, risk_gate=RiskGate(**GATE),
                                max_turnover=args.max_turnover * 1.5).plan(
            panel, state)
        report = manager.execute(plan, strategy_name=STRATEGY.name)
    except Exception as exc:
        audit.emit("cycle_error", error=repr(exc))
        broker.close()
        return 3

    save_peak(max(peak, account.equity))
    print(report)
    if not report.to_frame().empty:
        print(report.to_frame().to_string(index=False))
    if report.reconciliation is not None and report.reconciliation["breach"].any():
        print("\nRECONCILIATION DRIFT:")
        print(report.reconciliation[report.reconciliation["breach"]].to_string())

    audit.emit("cycle_end", aborted=report.aborted,
               submitted=len(report.submitted), skipped=len(report.skipped))
    broker.close()
    return 1 if report.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
