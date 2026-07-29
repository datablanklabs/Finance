"""Validate the order path against MockBroker. No network, no account."""

import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from qbt import (
    CrossSectionalMomentum, LiveSignalRunner, PortfolioState, RiskGate,
    SyntheticRepository,
)
from qbt.broker import BrokerOrder, MockBroker
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager

FAILS = []
WORK = "/tmp/qbt_orders_test"


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def fresh(subdir):
    path = os.path.join(WORK, subdir)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


shutil.rmtree(WORK, ignore_errors=True)

# Tuesday 10:00 ET, inside the session window.
MARKET_HOURS = pd.Timestamp("2026-07-21 14:00", tz="UTC").to_pydatetime()
AFTER_HOURS = pd.Timestamp("2026-07-21 23:00", tz="UTC").to_pydatetime()

print("=" * 72)
print("setup: build a plan the same way the notebook does")
print("=" * 72)

panel = SyntheticRepository(n_symbols=12, seed=99).fetch(
    start="2018-01-01", end="2026-07-20")
strategy = CrossSectionalMomentum(lookback=63, skip=5, top_n=4)
gate = RiskGate(target_vol=0.12, max_weight=0.30, max_drawdown=0.25)
prices = panel.last_close()

# Start already holding two of the eventual targets so turnover is realistic.
runner = LiveSignalRunner(strategy=strategy, risk_gate=gate, max_turnover=None)
probe = runner.plan(panel, PortfolioState(cash=20_000.0,
                                          shares=pd.Series(0.0, index=panel.symbols)))
targets = probe.target_weights[probe.target_weights > 0].index.tolist()
seed_shares = pd.Series(
    {targets[0]: float(20_000 * 0.15 / prices[targets[0]]),
     targets[1]: float(20_000 * 0.12 / prices[targets[1]])})

state = PortfolioState(cash=14_600.0, shares=seed_shares, peak_equity=21_000.0)
plan = runner.plan(panel, state)
print(f"{plan!r}")
print(f"targets: {list(plan.target_weights[plan.target_weights > 0].round(3).items())}")
check("plan has intents", len(plan.intents) > 0, f"{len(plan.intents)}")


def make(subdir, **policy_kw):
    path = fresh(subdir)
    broker = MockBroker(prices=prices, cash=state.cash,
                        positions=state.shares.copy(), seed=4, **{
                            k: v for k, v in policy_kw.pop("broker", {}).items()})
    pol = ExecutionPolicy(
        max_order_notional=policy_kw.pop("max_order_notional", 4_000.0),
        max_plan_notional=policy_kw.pop("max_plan_notional", 40_000.0),
        max_plan_turnover=policy_kw.pop("max_plan_turnover", 1.50),
        kill_switch_path=os.path.join(path, "KILL"),
        dry_run=policy_kw.pop("dry_run", True),
        **policy_kw)
    om = OrderManager(broker=broker, policy=pol,
                      audit=AuditLog(os.path.join(path, "audit.jsonl"),
                                     stdout=False),
                      journal_path=os.path.join(path, "journal.jsonl"))
    broker.connect()
    return om, broker, path


print()
print("=" * 72)
print("1. Dry run submits nothing")
print("=" * 72)

om, broker, path = make("dry", dry_run=True)
rep = om.execute(plan, strategy_name="xsmom", now=MARKET_HOURS)
print(f"  {rep!r}")
check("dry run submits zero orders", len(rep.submitted) == 0)
check("dry run skips every intent", len(rep.skipped) == len(plan.intents))
check("dry run made no place_order calls",
      not any(c[0] == "place_order" for c in broker.call_log))
check("dry run still read the account",
      any(c[0] == "get_account" for c in broker.call_log))
check("dry run still ran review",
      any(c[0] == "review_order" for c in broker.call_log))
audit = om.audit.read()
check("audit log written", len(audit) > 0, f"{len(audit)} events")
check("dry-run events recorded",
      (audit["event"] == "order_not_sent_dry_run").sum() == len(plan.intents))

print()
print("=" * 72)
print("2. Live run submits, and a re-run deduplicates")
print("=" * 72)

om, broker, path = make("live", dry_run=False)
rep1 = om.execute(plan, strategy_name="xsmom", now=MARKET_HOURS)
print(f"  first run:  {rep1!r}")
n_sub = len(rep1.submitted)
check("live run submits orders", n_sub > 0, f"{n_sub}")
check("all submitted orders reached a broker state",
      all(o.state in ("filled", "partial", "pending", "rejected")
          for o in rep1.submitted))

placed_first = sum(1 for c in broker.call_log if c[0] == "place_order")
rep2 = om.execute(plan, strategy_name="xsmom", now=MARKET_HOURS)
placed_total = sum(1 for c in broker.call_log if c[0] == "place_order")
print(f"  second run: {rep2!r}")
check("re-running the same plan places no new orders",
      placed_total == placed_first,
      f"place_order calls stayed at {placed_first}")
check("re-run reports deduplication",
      any("already submitted" in r for _, r in rep2.skipped),
      f"{len(rep2.skipped)} skipped")

pid1 = om.plan_id(plan, "xsmom")
pid2 = om.plan_id(plan, "xsmom")
check("plan_id is deterministic", pid1 == pid2, pid1)
other = LiveSignalRunner(
    strategy=CrossSectionalMomentum(lookback=126, skip=5, top_n=2),
    risk_gate=gate, max_turnover=None).plan(panel, state)
check("a genuinely different target portfolio gets a different id",
      om.plan_id(other, "xsmom") != pid1)

print()
print("=" * 72)
print("3. Preflight gates")
print("=" * 72)

om, broker, path = make("kill", dry_run=False)
open(os.path.join(path, "KILL"), "w").close()
rep = om.execute(plan, now=MARKET_HOURS)
check("kill switch aborts the plan", rep.aborted and "kill switch" in rep.aborted_reason,
      rep.aborted_reason)
check("kill switch prevents any order",
      not any(c[0] == "place_order" for c in broker.call_log))

om, broker, path = make("closed", dry_run=False)
rep = om.execute(plan, now=AFTER_HOURS)
check("closed market aborts the plan",
      rep.aborted and "market closed" in rep.aborted_reason, rep.aborted_reason)

om, broker, path = make("turnover", dry_run=False, max_plan_turnover=0.01)
rep = om.execute(plan, now=MARKET_HOURS)
check("turnover cap aborts the plan",
      rep.aborted and "turnover" in rep.aborted_reason, rep.aborted_reason)

om, broker, path = make("notional", dry_run=False, max_plan_notional=100.0)
rep = om.execute(plan, now=MARKET_HOURS)
check("plan notional cap aborts",
      rep.aborted and "notional" in rep.aborted_reason, rep.aborted_reason)

om, broker, path = make("drift", dry_run=False)
broker.cash += 40_000.0          # broker equity now far from the plan's view
rep = om.execute(plan, now=MARKET_HOURS)
check("equity drift aborts the plan",
      rep.aborted and "drift" in rep.aborted_reason, rep.aborted_reason)

om, broker, path = make("nonagentic", dry_run=False)
_orig = broker.get_account
broker.get_account = lambda: (lambda a: (setattr(a, "is_agentic", False), a)[1])(_orig())
rep = om.execute(plan, now=MARKET_HOURS)
check("non-agentic account aborts the plan",
      rep.aborted and "agentic" in rep.aborted_reason, rep.aborted_reason)

om, broker, path = make("allowlist", dry_run=False,
                        symbol_allowlist=("NOTHING",))
rep = om.execute(plan, now=MARKET_HOURS)
check("allowlist blocks every off-list intent",
      len(rep.submitted) == 0 and all("allowlist" in r for _, r in rep.skipped))

om, broker, path = make("percap", dry_run=False, max_order_notional=1.0)
rep = om.execute(plan, now=MARKET_HOURS)
check("per-order notional cap blocks individually, without aborting",
      not rep.aborted and len(rep.submitted) == 0
      and all("per-order" in r for _, r in rep.skipped))

print()
print("=" * 72)
print("4. Review gate and broker-side failures")
print("=" * 72)

path = fresh("reject")
bad = MockBroker(prices=prices, cash=state.cash, positions=state.shares.copy(),
                 seed=1, fail_on_symbols=[i.symbol for i in plan.intents[:2]])
bad.connect()
om = OrderManager(bad, ExecutionPolicy(dry_run=False, max_plan_turnover=1.5,
                                       max_plan_notional=40_000,
                                       max_order_notional=4_000,
                                       kill_switch_path=os.path.join(path, "KILL")),
                  AuditLog(os.path.join(path, "a.jsonl"), stdout=False),
                  os.path.join(path, "j.jsonl"))
rep = om.execute(plan, now=MARKET_HOURS)
print(f"  {rep!r}")
check("review gate blocks untradeable symbols before submission",
      sum(1 for _, r in rep.skipped if "review" in r) == 2,
      f"{[r for _, r in rep.skipped]}")
check("untradeable symbols never reached place_order",
      not any(c[0] == "place_order" and c[1]["symbol"] in
              {i.symbol for i in plan.intents[:2]} for c in bad.call_log))

path = fresh("partial")
part = MockBroker(prices=prices, cash=state.cash, positions=state.shares.copy(),
                  seed=2, partial_rate=1.0)
part.connect()
om = OrderManager(part, ExecutionPolicy(dry_run=False, max_plan_turnover=1.5,
                                        max_plan_notional=40_000,
                                        max_order_notional=4_000,
                                        kill_switch_path=os.path.join(path, "KILL")),
                  AuditLog(os.path.join(path, "a.jsonl"), stdout=False),
                  os.path.join(path, "j.jsonl"))
rep = om.execute(plan, now=MARKET_HOURS)
check("partial fills are reported as partial",
      all(o.state == "partial" for o in rep.submitted), f"{len(rep.submitted)}")
check("partial fills show up as reconciliation drift",
      rep.reconciliation is not None and bool(rep.reconciliation["breach"].any()),
      f"max drift {rep.reconciliation['drift'].max():.3f}")

print()
print("=" * 72)
print("5. Crash recovery: unknown outcome is resolved by reading, not retrying")
print("=" * 72)

path = fresh("crash")
crash = MockBroker(prices=prices, cash=state.cash, positions=state.shares.copy(),
                   seed=3)
crash.connect()
jpath = os.path.join(path, "j.jsonl")
om = OrderManager(crash, ExecutionPolicy(dry_run=False,
                                         kill_switch_path=os.path.join(path, "KILL")),
                  AuditLog(os.path.join(path, "a.jsonl"), stdout=False), jpath)

# Simulate the fatal case: the journal says we were submitting, and the process
# died before recording the outcome. One order did reach the broker; one didn't.
victim = plan.intents[0]
ghost = plan.intents[1]
crash.place_order(victim.symbol, victim.side, abs(victim.shares))  # really landed
for intent in (victim, ghost):
    key = f"{intent.symbol}:{intent.side}"
    with open(jpath, "a") as fh:
        fh.write(json.dumps({"stage": "submitting", "plan_id": "crashplan",
                             "intent_key": key, "symbol": intent.symbol,
                             "side": intent.side,
                             "quantity": abs(intent.shares)}) + "\n")

placed_before = sum(1 for c in crash.call_log if c[0] == "place_order")
rec = om.recover()
placed_after = sum(1 for c in crash.call_log if c[0] == "place_order")
print(rec.to_string(index=False))
check("recover() places no orders", placed_after == placed_before,
      "never retries an unknown outcome")
check("recover() finds the order that landed",
      (rec["outcome"] == "found_at_broker").sum() == 1)
check("recover() flags the one that did not",
      (rec["outcome"] == "not_at_broker").sum() == 1)
check("recover() closes out the journal entries",
      om.recover().empty, "second call finds nothing unresolved")

print()
print("=" * 72)
print("6. Reconciliation and audit stream")
print("=" * 72)

om, broker, path = make("recon", dry_run=False)
rep = om.execute(plan, strategy_name="xsmom", now=MARKET_HOURS)
recon = rep.reconciliation
print(recon.head(6).to_string())
check("reconciliation covers targets and holdings", len(recon) > 0)
check("clean fills reconcile within tolerance",
      not recon["breach"].any(),
      f"max drift {recon['drift'].max():.4f}")

audit = om.audit.read()
seq = audit["event"].tolist()
print(f"\n  {len(audit)} audit events: {sorted(set(seq))}")
check("audit is valid JSONL, one object per line",
      all(isinstance(json.loads(l), dict)
          for l in open(om.audit.path) if l.strip()))
check("audit records the plan before the account read",
      seq.index("plan_received") < seq.index("account_read"))
check("preflight is recorded before any submission",
      seq.index("preflight_passed") < seq.index("order_submitting"))
check("every submitting event has a terminal counterpart",
      seq.count("order_submitting")
      == seq.count("order_submitted") + seq.count("order_rejected"))
check("audit carries stable SIEM fields",
      {"ts", "event", "host", "pid"}.issubset(set(audit.columns)))
check("reconciliation is audited", "reconciled" in seq)

rules = OrderManager.detection_rules()
check("detection rules provided", len(rules) >= 8, f"{len(rules)} rules")
print(f"\n  {len(rules)} SIEM detection rules, "
      f"{(rules.severity == 'critical').sum()} critical")

print()
print("=" * 72)
print("7. Tool discovery binds by schema, not by assumption")
print("=" * 72)

from qbt.broker import CAPABILITY_CANDIDATES, REQUIRED_CAPABILITIES, ToolBinding

documented = {
    "get_accounts", "get_portfolio", "get_equity_positions", "get_equity_quotes",
    "get_equity_orders", "search", "get_watchlists", "add_to_watchlist",
    "update_watchlist", "review_equity_order", "place_equity_order",
    "cancel_equity_order",
}
resolved = {}
for cap, cands in CAPABILITY_CANDIDATES.items():
    hit = next((c for c in cands if c in documented), None)
    if hit:
        resolved[cap] = hit
print("  capability -> tool, against the documented surface:")
for cap, tool in sorted(resolved.items()):
    print(f"    {cap:10s} -> {tool}")
check("every required capability resolves against the documented surface",
      all(c in resolved for c in REQUIRED_CAPABILITIES),
      f"missing {[c for c in REQUIRED_CAPABILITIES if c not in resolved]}")
check("review capability resolves (the safety-critical preflight)",
      resolved.get("review") == "review_equity_order")

# Argument-name resolution must tolerate a server that says "ticker".
b = ToolBinding("place", "place_equity_order", {
    "properties": {"ticker": {"type": "string"}, "direction": {"type": "string"},
                   "shares": {"type": "number"}, "type": {"type": "string"}},
    "required": ["ticker", "direction", "shares"]})
check("resolves symbol -> ticker",
      b.resolve_arg("symbol", ("symbol", "ticker", "instrument")) == "ticker")
check("resolves side -> direction",
      b.resolve_arg("side", ("side", "direction")) == "direction")
check("resolves quantity -> shares",
      b.resolve_arg("quantity", ("quantity", "shares", "qty")) == "shares")
check("returns None for an absent argument",
      b.resolve_arg("stop_price", ("stop_price",)) is None)

print()
print("=" * 72)
print(f"{len(FAILS)} FAILURE(S): {FAILS}" if FAILS else "ALL CHECKS PASSED")
print("=" * 72)
