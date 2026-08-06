#!/usr/bin/env python3
"""One trading cycle. Run this on a schedule; never from a notebook.

    python run_cycle.py --dry-run            # default, sends nothing
    python run_cycle.py --live --max-order 50

Exit codes: 0 completed, 1 aborted at preflight, 2 unresolved in-flight order
(halt and investigate before the next cycle), 3 setup or connection failure.

Why this is a separate process from the notebook: it needs a durable journal, a
deterministic single pass, and crash recovery on startup. A notebook gives you
none of those, and re-running a cell would re-submit.

**First real (non-synthetic) run needs a browser.** Authentication against
Robinhood is OAuth 2.0 Authorization Code + PKCE (see :mod:`qbt.oauth`), not a
static token: the first connection opens a browser for you to log in and
authorize once, then persists a refresh token to ``ROBINHOOD_OAUTH_STATE``
(default ``state/robinhood_oauth.json``) so every later scheduled run
refreshes silently with no browser involved -- until that refresh token
itself expires or is revoked, at which point the browser flow fires again.
That means this cannot be the very first invocation on a truly headless
remote box with no way to reach ``127.0.0.1`` in a browser; run the first
login somewhere you can open a browser (or tunnel the callback port), then
copy the resulting state file over.
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
from qbt.oauth import build_robinhood_oauth
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager

UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
            "XLV", "XLY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]

STRATEGY = CrossSectionalMomentum(lookback=63, skip=5, top_n=5)
GATE = dict(target_vol=0.12, max_weight=0.30, max_gross=1.0, max_drawdown=0.25)


def mode_path(base: str, synthetic: bool) -> str:
    """Suffix a state/audit path by run mode -- e.g. "state/peak_equity.txt"
    -> "state/peak_equity_live.txt" or "..._synthetic.txt".

    Synthetic and real runs must never share persisted state. This was a
    real, confirmed bug: a ``--synthetic`` run's $25,000 MockBroker starting
    cash wrote to the same peak-equity file a real ~$1,000 account later
    read back. ``load_peak`` takes ``max(stored, current)``, which can only
    ratchet the peak up, never self-correct down -- so the real account's
    peak got stuck at $25,000, its apparent drawdown read as 96%, and the
    drawdown breaker inside :class:`~qbt.risk.RiskGate` silently zeroed the
    entire book on the very first live cycle. Applies to the order journal
    and audit log too, for the same reason: a synthetic run's fake fills
    have no business in the same journal `recover()` reads to decide what a
    real crash left in flight.
    """
    root, ext = os.path.splitext(base)
    return f"{root}_{'synthetic' if synthetic else 'live'}{ext}"


def load_peak(path: str, current: float) -> float:
    """The running equity peak must survive restarts.

    If this is lost, the drawdown breaker forgets the high-water mark. If it is
    seeded too high, the breaker trips immediately and the bot silently never
    trades. Both failures are quiet, so persist it -- see :func:`mode_path`
    for why ``path`` must never be shared between synthetic and real runs.
    """
    try:
        with open(path) as fh:
            return max(float(fh.read().strip()), current)
    except (OSError, ValueError):
        return current


def save_peak(path: str, value: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{value:.4f}")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually submit orders (default is dry run)")
    ap.add_argument("--max-order", type=float, default=250.0)
    ap.add_argument("--max-plan", type=float, default=5_000.0)
    ap.add_argument("--max-turnover", type=float, default=0.67)
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data and a mock broker")
    ap.add_argument("--ignore-market-hours", action="store_true")
    args = ap.parse_args()

    audit = AuditLog(mode_path("audit/orders.jsonl", args.synthetic))
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
            oauth = build_robinhood_oauth(
                storage_path=os.environ.get(
                    "ROBINHOOD_OAUTH_STATE", "state/robinhood_oauth.json"
                ),
                port=int(os.environ.get("ROBINHOOD_OAUTH_CALLBACK_PORT", "8765")),
            )
            broker = RobinhoodMCPBroker(auth=oauth, require_agentic=True)
        broker.connect()
        # Resolve the account now, before recovery -- recover() reads the
        # broker's orders (get_orders()), which requires account_number on
        # the real API the same as review/place/cancel do. get_account() is
        # what caches self.account_id on the broker; recover() running
        # first (it has to -- unresolved in-flight orders must be settled
        # before a new plan is built) needs that already done. The plan
        # step below reuses this same `account` rather than calling
        # get_account() a second time.
        account = broker.get_account()
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
    manager = OrderManager(
        broker=broker, policy=policy, audit=audit,
        journal_path=mode_path("audit/journal.jsonl", args.synthetic),
    )

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
    peak_file = mode_path("state/peak_equity.txt", args.synthetic)
    try:
        peak = load_peak(peak_file, account.equity)
        state = PortfolioState(
            cash=account.cash,
            shares=account.positions.reindex(panel.symbols).fillna(0.0),
            peak_equity=peak,
        )
        plan = LiveSignalRunner(strategy=STRATEGY, risk_gate=RiskGate(**GATE),
                                max_turnover=args.max_turnover * 1.5).plan(
            panel, state)

        # Surface *why* the plan looks the way it does -- these were
        # computed but never printed anywhere, which is exactly how the
        # peak-equity bug above went undetected: the risk gate's own
        # explanation ("drawdown breaker tripped at 96.0%") was sitting
        # right there in plan.decision.notes the whole time.
        if plan.warnings:
            for w in plan.warnings:
                print(f"  PLAN WARNING: {w}")
            audit.emit("plan_warnings", warnings="; ".join(plan.warnings))
        if plan.decision is not None and plan.decision.notes:
            for n in plan.decision.notes:
                print(f"  RISK GATE: {n}")
            audit.emit("risk_gate_notes", notes="; ".join(plan.decision.notes))

        report = manager.execute(plan, strategy_name=STRATEGY.name)
    except Exception as exc:
        audit.emit("cycle_error", error=repr(exc))
        broker.close()
        return 3

    save_peak(peak_file, max(peak, account.equity))
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
