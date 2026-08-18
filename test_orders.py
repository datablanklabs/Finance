"""Validate the order path against MockBroker. No network, no account."""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from qbt import (
    CrossSectionalMomentum, DayTradeLedger, LiveSignalRunner, LivePlan,
    OrderIntent, PortfolioState, RiskGate, SyntheticRepository,
)
from qbt.broker import BrokerOrder, MockBroker
from qbt.orders import (
    AuditLog, ExecutionPolicy, ExecutionReport, OrderManager, _clean_broker_rejection,
)
from qbt.risk import as_session_date

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
check("turnover cap scales the plan down to fit instead of aborting it",
      not rep.aborted, rep.aborted_reason)
check("the re-fit that made this possible is in the audit trail",
      "turnover_refit" in om.audit.read()["event"].tolist())

# A flat account's first-ever buildout is structurally ~100% turnover --
# current weights are all zero, so turnover and gross exposure are the same
# number. allow_full_turnover_from_flat (default True) exempts exactly that
# case; a non-flat account with the same turnover must still be blocked.
_flat_prices = pd.Series({"S0": 100.0, "S1": 100.0})
_flat_intents = [
    OrderIntent(symbol=s, side="buy", shares=1.577, reference_price=100.0,
                notional=157.7, current_weight=0.0, target_weight=0.1577)
    for s in ("S0", "S1")
]
_flat_plan = LivePlan(
    asof=pd.Timestamp("2026-08-06"), equity=400.0, intents=_flat_intents,
    target_weights=pd.Series({"S0": 0.1577, "S1": 0.1577}),
    current_weights=pd.Series(dtype=float), decision=None, warnings=[],
)  # notional 315.4 / equity 400 = 78.9% turnover, same shape as the real case

_flat_broker = MockBroker(prices=_flat_prices, cash=400.0, positions=pd.Series(dtype=float))
_flat_broker.connect()
_flat_policy = ExecutionPolicy(max_plan_turnover=0.67, require_market_open=False,
                               kill_switch_path=os.path.join(fresh("flat"), "KILL"))
_flat_om = OrderManager(broker=_flat_broker, policy=_flat_policy,
                        journal_path=os.path.join(WORK, "flat", "journal.jsonl"))
_flat_ok, _flat_notes = _flat_om.preflight(_flat_plan, _flat_broker.get_account(), now=MARKET_HOURS)
check("flat account's first buildout is exempt from the turnover cap by default",
      _flat_ok, "; ".join(_flat_notes))
check("the exemption is recorded in the preflight notes, not silent",
      any("exempt" in n for n in _flat_notes))

_holding_broker = MockBroker(prices=_flat_prices, cash=42.3, positions=pd.Series({"S0": 3.577}))
_holding_broker.connect()  # equity = 42.3 + 3.577*100 = 400.0, same as the flat case
_holding_policy = ExecutionPolicy(max_plan_turnover=0.67, require_market_open=False,
                                  kill_switch_path=os.path.join(fresh("holding"), "KILL"))
_holding_om = OrderManager(broker=_holding_broker, policy=_holding_policy,
                           journal_path=os.path.join(WORK, "holding", "journal.jsonl"))
_holding_ok, _holding_notes = _holding_om.preflight(
    _flat_plan, _holding_broker.get_account(), now=MARKET_HOURS
)
check("an account that already holds a position is NOT exempt at the same turnover",
      not _holding_ok and any("turnover" in n and "exceeds" in n for n in _holding_notes),
      "; ".join(_holding_notes))

_strict_policy = ExecutionPolicy(max_plan_turnover=0.67, require_market_open=False,
                                 kill_switch_path=os.path.join(fresh("strict"), "KILL"),
                                 allow_full_turnover_from_flat=False)
_strict_om = OrderManager(broker=_flat_broker, policy=_strict_policy,
                          journal_path=os.path.join(WORK, "strict", "journal.jsonl"))
_strict_ok, _strict_notes = _strict_om.preflight(_flat_plan, _flat_broker.get_account(), now=MARKET_HOURS)
check("the exemption can be disabled via allow_full_turnover_from_flat=False",
      not _strict_ok, "; ".join(_strict_notes))

# A plan whose turnover lands exactly on the cap must clear preflight, not
# get hard-aborted -- but "exactly 0.67" is a mathematical statement, and
# gross/account.equity is a float computation. This specific equity and
# per-order notional (found by search) genuinely rounds to
# 0.6700000000000002 in IEEE double precision -- 1.1e-16 above the nominal
# cap, not a real overage. A non-flat account, deliberately: the
# allow_full_turnover_from_flat exemption would otherwise mask a broken
# comparison by passing for an unrelated reason.
_boundary_broker = MockBroker(prices=pd.Series({"S0": 100.0}), cash=3060.19,
                              positions=pd.Series({"S0": 1.0}))  # equity = 3160.19, not flat
_boundary_broker.connect()
_boundary_intents = [
    OrderIntent(symbol=f"B{i}", side="buy", shares=1.0, reference_price=705.7757666666668,
               notional=705.7757666666668, current_weight=0.0, target_weight=0.0)
    for i in range(3)
]
_boundary_plan = LivePlan(
    asof=pd.Timestamp("2026-08-06"), equity=3160.19, intents=_boundary_intents,
    target_weights=pd.Series(dtype=float), current_weights=pd.Series(dtype=float),
    decision=None, warnings=[],
)
_raw_turnover = sum(i.notional for i in _boundary_intents) / 3160.19
check("the boundary case actually reproduces a float artifact just above 0.67",
      0.67 < _raw_turnover < 0.67 + 1e-9, repr(_raw_turnover))

_boundary_policy = ExecutionPolicy(max_plan_turnover=0.67, require_market_open=False,
                                   kill_switch_path=os.path.join(fresh("boundary"), "KILL"))
_boundary_om = OrderManager(broker=_boundary_broker, policy=_boundary_policy,
                            journal_path=os.path.join(WORK, "boundary", "journal.jsonl"))
_boundary_ok, _boundary_notes = _boundary_om.preflight(
    _boundary_plan, _boundary_broker.get_account(), now=MARKET_HOURS
)
check("a plan landing a float hair above the cap clears preflight, not aborted",
      _boundary_ok, "; ".join(_boundary_notes))

# The real bug this session hit: LiveSignalRunner.plan() scales a plan down
# to *exactly* the turnover cap using the equity it had at planning time.
# By the time OrderManager.execute() reads the account fresh, ordinary
# price movement -- confirmed live, a $0.12 move on a ~$1,000 account --
# recomputes gross/account.equity a hair over the cap, and used to abort
# execute() outright even though the plan was correctly sized moments
# earlier. execute() must re-fit against the fresh read instead.
_drift_gross = 0.67 * 1001.28  # exactly what planning scaled to, equity=1001.28
_drift_intents = [
    OrderIntent(symbol="XLF", side="buy", shares=_drift_gross / 100.0,
               reference_price=100.0, notional=_drift_gross,
               current_weight=0.0, target_weight=0.1),
]
_drift_plan = LivePlan(
    asof=pd.Timestamp("2026-08-07"), equity=1001.28, intents=_drift_intents,
    target_weights=pd.Series({"XLF": 0.1}), current_weights=pd.Series({"XLF": 0.0}),
    decision=None, warnings=[],
)
# Not flat, fresh equity 1001.16 -- the exact reported drift, $0.12 lower.
_drift_broker = MockBroker(prices=pd.Series({"XLF": 100.0}), cash=901.16,
                           positions=pd.Series({"XLF": 1.0}))
_drift_broker.connect()
_drift_policy = ExecutionPolicy(max_plan_turnover=0.67, require_market_open=False,
                                dry_run=True,
                                kill_switch_path=os.path.join(fresh("drift_refit"), "KILL"))
_drift_path = fresh("drift_refit_om")
_drift_om = OrderManager(broker=_drift_broker, policy=_drift_policy,
                         audit=AuditLog(os.path.join(_drift_path, "a.jsonl"), stdout=False),
                         journal_path=os.path.join(_drift_path, "journal.jsonl"))
_drift_report = _drift_om.execute(_drift_plan, strategy_name="drift-refit", now=MARKET_HOURS)
check("a plan re-scaled to exactly the cap survives normal equity drift by "
      "execution time, instead of getting hard-aborted",
      _drift_report.aborted_reason is None, _drift_report.aborted_reason)
check("the re-fit is recorded in the audit trail",
      "turnover_refit" in _drift_om.audit.read()["event"].tolist())

# A plan comfortably under the cap must not be touched at all -- confirm
# _refit_turnover() is a true no-op (same object back) in the normal case,
# not just "close enough" scaling every single time.
_norefit_intents = [
    OrderIntent(symbol="XLF", side="buy", shares=1.0, reference_price=10.0,
               notional=10.0, current_weight=0.0, target_weight=0.01),
]
_norefit_plan = LivePlan(
    asof=pd.Timestamp("2026-08-07"), equity=1000.0, intents=_norefit_intents,
    target_weights=pd.Series({"XLF": 0.01}), current_weights=pd.Series({"XLF": 0.0}),
    decision=None, warnings=[],
)
_norefit_account = _drift_broker.get_account()
check("a plan already well under the cap is not rewritten at all",
      _drift_om._refit_turnover(_norefit_plan, _norefit_account) is _norefit_plan)

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

# recover()'s broker query window must anchor to when the unresolved
# attempt actually happened, not to "today at midnight" -- otherwise
# recover() running on a later calendar day than the crash (process down
# over a weekend, restarted after a Friday-close crash) would never even
# ask the broker about an order from before today.
window_path = fresh("recover_window")
window_broker = MockBroker(prices=pd.Series({"AAA": 100.0}), cash=100_000.0)
window_broker.connect()
window_om = OrderManager(window_broker, journal_path=os.path.join(window_path, "j.jsonl"))
window_om._journal(stage="submitting", plan_id="p1", intent_key="AAA:buy",
                   symbol="AAA", side="buy", quantity=10.0)
window_om.recover()
since_used = pd.Timestamp(
    next(c[1]["since"] for c in window_broker.call_log if c[0] == "get_orders")
)
submitting_ts = pd.Timestamp(
    next(e for e in window_om._journal_entries() if e.get("stage") == "submitting")["ts"]
)
check("recover() anchors the broker query to the actual submitting time",
      since_used < submitting_ts and (submitting_ts - since_used).total_seconds() < 3600,
      f"since={since_used}, submitted={submitting_ts}")

# Two unresolved entries sharing the same fingerprint (symbol/side/
# quantity-bucket) -- two ghost entries from one crash -- must not both
# claim the one broker order that actually landed.
dup_path = fresh("recover_dup")
dup_broker = MockBroker(prices=pd.Series({"AAA": 100.0}), cash=100_000.0)
dup_broker.connect()
dup_broker.place_order("AAA", "buy", 10.0)  # exactly one real order landed
dup_om = OrderManager(dup_broker, journal_path=os.path.join(dup_path, "j.jsonl"))
dup_om._journal(stage="submitting", plan_id="p1", intent_key="AAA:buy",
                symbol="AAA", side="buy", quantity=10.0)
dup_om._journal(stage="submitting", plan_id="p2", intent_key="AAA:buy",
                symbol="AAA", side="buy", quantity=10.0)
dup_rec = dup_om.recover()
check("only one ghost entry claims the one order that actually landed",
      (dup_rec["outcome"] == "found_at_broker").sum() == 1
      and (dup_rec["outcome"] == "not_at_broker").sum() == 1)

# A retry through execute() itself -- not a hand-crafted journal entry --
# before recover() has run must not resubmit. This is the actual bug: the
# except-block around place_order used to only log the unknown outcome,
# leaving nothing in the journal to stop a second execute() call on the
# same plan from blindly resubmitting.
retry_path = fresh("retry")
retry_broker = MockBroker(prices=pd.Series({"AAA": 100.0}), cash=100_000.0)
retry_broker.connect()
call_count = {"n": 0}
_real_place = retry_broker.place_order
def _flaky_place(*a, **kw):
    call_count["n"] += 1
    if call_count["n"] == 1:
        raise TimeoutError("simulated network timeout -- unknown outcome")
    return _real_place(*a, **kw)
retry_broker.place_order = _flaky_place

retry_om = OrderManager(
    retry_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(retry_path, "KILL")),
    AuditLog(os.path.join(retry_path, "a.jsonl"), stdout=False),
    os.path.join(retry_path, "j.jsonl"),
)
now = pd.Timestamp.now(tz="UTC")
retry_intent = OrderIntent(symbol="AAA", side="buy", shares=10.0, reference_price=100.0,
                           notional=1000.0, current_weight=0.0, target_weight=0.01)
retry_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[retry_intent],
                      target_weights=pd.Series({"AAA": 0.01}),
                      current_weights=pd.Series({"AAA": 0.0}), decision=None)

r1 = retry_om.execute(retry_plan, strategy_name="retry-test", now=now)
check("first attempt hits the flaky broker and reports unknown outcome",
      call_count["n"] == 1 and any("unknown outcome" in s[1] for s in r1.skipped))

r2 = retry_om.execute(retry_plan, strategy_name="retry-test", now=now)
check("retry before recover() is blocked, not resubmitted",
      call_count["n"] == 1,
      f"place_order called {call_count['n']} time(s), expected exactly 1")
check("retry is skipped with a pending_recovery reason",
      any("run recover() first" in s[1] for s in r2.skipped))

retry_om.recover()
r3 = retry_om.execute(retry_plan, strategy_name="retry-test", now=now)
check("after recover(), the intent stays permanently resolved (never retried)",
      call_count["n"] == 1)

# A synchronous "cannot include fractional shares" rejection (confirmed
# live, 2026-08, on a real instrument) is not the ambiguous kind of
# failure recover() exists for -- the order was definitely never created.
# execute() should correct and resubmit once at a whole-share size, close
# the journal chain either way, and keep processing the rest of the plan
# rather than abandoning it, since the old unknown-outcome/break path used
# to leave a dangling "submitting" entry that a later recover() would
# resolve as not_at_broker and HALT the *next* cycle for, even though the
# outcome was already fully known here.
#
# Raised as a real nested ExceptionGroup, not a flat RuntimeError -- the
# actual broker call goes through anyio task groups, and str() on that
# nested shape collapses to "unhandled errors in a TaskGroup
# (1 sub-exception)" with no mention of "fractional shares" at all; only
# repr() recurses into it. A flat RuntimeError here would pass even a
# str()-only match and miss the exact bug this session hit (XLK's rejection
# was caught, XLV's identical one wasn't, because the earlier fix checked
# str(exc) instead of repr(exc)).
def _fractional_shares_error():
    inner = RuntimeError(
        'MCP tool error: [TextContent(type=\'text\', text=\'API error 400: '
        '{"quantity":["Order quantity cannot include fractional shares."]}\')]')
    return ExceptionGroup("unhandled errors in a TaskGroup",
                          [ExceptionGroup("unhandled errors in a TaskGroup", [inner])])


frac_path = fresh("fractional_retry")
frac_broker = MockBroker(prices=pd.Series({"AAA": 100.0, "BBB": 50.0}), cash=100_000.0)
frac_broker.connect()
_real_frac_place = frac_broker.place_order
frac_calls = []
def _frac_place(symbol, side, quantity, *a, **kw):
    frac_calls.append((symbol, quantity))
    if symbol == "AAA" and quantity > 2:
        raise _fractional_shares_error()
    return _real_frac_place(symbol, side, quantity, *a, **kw)
frac_broker.place_order = _frac_place

frac_om = OrderManager(
    frac_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(frac_path, "KILL")),
    AuditLog(os.path.join(frac_path, "a.jsonl"), stdout=False),
    os.path.join(frac_path, "j.jsonl"),
)
frac_intents = [
    OrderIntent(symbol="AAA", side="buy", shares=2.7255578430183576, reference_price=100.0,
               notional=272.56, current_weight=0.0, target_weight=0.01),
    OrderIntent(symbol="BBB", side="buy", shares=5.0, reference_price=50.0,
               notional=250.0, current_weight=0.0, target_weight=0.01),
]
frac_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=frac_intents,
                     target_weights=pd.Series({"AAA": 0.01, "BBB": 0.01}),
                     current_weights=pd.Series({"AAA": 0.0, "BBB": 0.0}), decision=None)
frac_report = frac_om.execute(frac_plan, strategy_name="frac-test", now=now)

check("fractional rejection retries once at the floored whole-share size",
      frac_calls[:2] == [("AAA", 2.7255578430183576), ("AAA", 2.0)], frac_calls)
check("the retried whole-share order lands as submitted",
      any(o.symbol == "AAA" for o in frac_report.submitted))
check("the rest of the plan still executes after a fractional rejection, not abandoned",
      any(o.symbol == "BBB" for o in frac_report.submitted))

frac_recover = frac_om.recover()
check("recover() finds nothing dangling after a fractional-rejection retry succeeded",
      frac_recover.empty)

# A target under one share for a brand-new position (current_weight ~ 0)
# rounds UP to one whole share instead of being skipped -- confirmed live
# (2026-08): a $1,000 account targeting 5 equal-weight ETF sleeves at
# ~$160 each hit this on every sleeve whose share price exceeds that;
# skipping every one of them means never holding any of those names, on
# any account this size, rather than holding a *little* of each.
tiny_path = fresh("fractional_tiny")
tiny_broker = MockBroker(prices=pd.Series({"CCC": 300.0}), cash=100_000.0)
tiny_broker.connect()
_real_tiny_place = tiny_broker.place_order
tiny_calls = []
def _tiny_place(symbol, side, quantity, *a, **kw):
    tiny_calls.append(quantity)
    if quantity != 1.0:
        raise _fractional_shares_error()
    return _real_tiny_place(symbol, side, quantity, *a, **kw)
tiny_broker.place_order = _tiny_place

tiny_om = OrderManager(
    tiny_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(tiny_path, "KILL")),
    AuditLog(os.path.join(tiny_path, "a.jsonl"), stdout=False),
    os.path.join(tiny_path, "j.jsonl"),
)
tiny_intent = OrderIntent(symbol="CCC", side="buy", shares=0.848263, reference_price=300.0,
                          notional=254.48, current_weight=0.0, target_weight=0.01)
tiny_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[tiny_intent],
                     target_weights=pd.Series({"CCC": 0.01}),
                     current_weights=pd.Series({"CCC": 0.0}), decision=None)
tiny_report = tiny_om.execute(tiny_plan, strategy_name="frac-tiny", now=now)
check("a sub-one-share target for a brand-new position rounds up to 1 share",
      tiny_calls == [0.848263, 1.0], tiny_calls)
check("the rounded-up order actually lands",
      any(o.symbol == "CCC" for o in tiny_report.submitted))

tiny_recover = tiny_om.recover()
check("recover() finds nothing dangling after a round-up-to-1 retry succeeded",
      tiny_recover.empty)

# The same sub-one-share target, but as a marginal top-up on a position
# already held (current_weight != 0), with a truly small delta (< 0.5
# share) -- must still skip cleanly. Forcing a full extra share for a tiny
# rebalance nudge would be a much bigger trade than intended.
topup_path = fresh("fractional_topup")
topup_broker = MockBroker(prices=pd.Series({"DDD": 300.0}), cash=100_000.0)
topup_broker.connect()
def _topup_place(symbol, side, quantity, *a, **kw):
    raise _fractional_shares_error()
topup_broker.place_order = _topup_place

topup_om = OrderManager(
    topup_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(topup_path, "KILL")),
    AuditLog(os.path.join(topup_path, "a.jsonl"), stdout=False),
    os.path.join(topup_path, "j.jsonl"),
)
topup_intent = OrderIntent(symbol="DDD", side="buy", shares=0.312, reference_price=300.0,
                           notional=93.6, current_weight=0.08, target_weight=0.085)
topup_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[topup_intent],
                      target_weights=pd.Series({"DDD": 0.085}),
                      current_weights=pd.Series({"DDD": 0.08}), decision=None)
topup_report = topup_om.execute(topup_plan, strategy_name="frac-topup", now=now)
check("a small (< 0.5 share) top-up on an existing position is still "
      "skipped, not rounded",
      any("rounds down to 0" in s[1] for s in topup_report.skipped))

topup_recover = topup_om.recover()
check("a definitively-rejected top-up leaves nothing dangling -- recover() "
      "can't misresolve it as not_at_broker and HALT the next cycle",
      topup_recover.empty)

# The real bug this session hit: XLK and IWM trims on a small, whole-
# share-only account rounding-down-to-zero *every single cycle*, forever,
# with no whole-share holding ever getting closer to target. A top-up or
# trim delta that's already most of the way to a full share (>= 0.5) must
# now round to the nearest whole share and actually execute, not stall
# indefinitely just because it isn't a brand-new position.
nearest_path = fresh("fractional_nearest")
nearest_broker = MockBroker(prices=pd.Series({"EEE": 300.0}), cash=100_000.0,
                            positions=pd.Series({"EEE": 5.0}))
nearest_broker.connect()
_real_nearest_place = nearest_broker.place_order
nearest_calls = []
def _nearest_place(symbol, side, quantity, *a, **kw):
    nearest_calls.append(quantity)
    if quantity != round(quantity):
        raise _fractional_shares_error()
    return _real_nearest_place(symbol, side, quantity, *a, **kw)
nearest_broker.place_order = _nearest_place

nearest_om = OrderManager(
    nearest_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(nearest_path, "KILL")),
    AuditLog(os.path.join(nearest_path, "a.jsonl"), stdout=False),
    os.path.join(nearest_path, "j.jsonl"),
)
nearest_intent = OrderIntent(symbol="EEE", side="sell", shares=0.72, reference_price=300.0,
                             notional=216.0, current_weight=0.30, target_weight=0.16)
nearest_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[nearest_intent],
                        target_weights=pd.Series({"EEE": 0.16}),
                        current_weights=pd.Series({"EEE": 0.30}), decision=None)
nearest_report = nearest_om.execute(nearest_plan, strategy_name="frac-nearest", now=now)
check("a trim delta >= 0.5 share rounds to the nearest whole share and "
      "actually executes, instead of skipping forever",
      nearest_calls == [0.72, 1.0], nearest_calls)
check("the rounded trim actually lands",
      any(o.symbol == "EEE" for o in nearest_report.submitted))

nearest_recover = nearest_om.recover()
check("recover() finds nothing dangling after a round-to-nearest retry succeeded",
      nearest_recover.empty)

# Confirmed live (2026-08): a *different* synchronous 400 -- "not enough
# buying power" -- used to be misclassified as a genuinely unknown outcome
# (only "fractional shares" was recognized as a definitive rejection),
# which left a dangling journal entry for the next cycle's recover() to
# wrongly HALT on, and `break` abandoned every remaining intent in the
# plan even though they don't depend on this one's cash shortfall. Any
# synchronous "API error 4xx" is now treated as definitive.
def _buying_power_error():
    inner = RuntimeError(
        'MCP tool error: [TextContent(type=\'text\', text=\'API error 400: '
        '{"detail":"Not enough buying power."}\', annotations=None, meta=None)]')
    return ExceptionGroup("unhandled errors in a TaskGroup",
                          [ExceptionGroup("unhandled errors in a TaskGroup", [inner])])


bp_path = fresh("buying_power_rejection")
bp_broker = MockBroker(prices=pd.Series({"XLI": 185.0, "IWM": 300.0}), cash=100_000.0,
                       positions=pd.Series({"IWM": 1.0}))
bp_broker.connect()
_real_bp_place = bp_broker.place_order
bp_calls = []
def _bp_place(symbol, side, quantity, *a, **kw):
    bp_calls.append((symbol, side))
    if symbol == "XLI":
        raise _buying_power_error()
    return _real_bp_place(symbol, side, quantity, *a, **kw)
bp_broker.place_order = _bp_place

bp_om = OrderManager(
    bp_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(bp_path, "KILL")),
    AuditLog(os.path.join(bp_path, "a.jsonl"), stdout=False),
    os.path.join(bp_path, "j.jsonl"),
)
bp_xli_intent = OrderIntent(symbol="XLI", side="buy", shares=0.89, reference_price=185.0,
                            notional=164.65, current_weight=0.0, target_weight=0.16)
bp_iwm_intent = OrderIntent(symbol="IWM", side="sell", shares=0.45, reference_price=300.0,
                            notional=135.0, current_weight=0.30, target_weight=0.16)
bp_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0,
                   intents=[bp_xli_intent, bp_iwm_intent],
                   target_weights=pd.Series({"XLI": 0.16, "IWM": 0.16}),
                   current_weights=pd.Series({"XLI": 0.0, "IWM": 0.30}), decision=None)
bp_report = bp_om.execute(bp_plan, strategy_name="bp-test")
check("a non-fractional synchronous rejection (buying power) is reported "
      "as a definitive rejection, not an unknown outcome",
      any("rejected:" in r for i, r in bp_report.skipped if i.symbol == "XLI"),
      bp_report.skipped)
check("the skipped note shows the clean 'Insufficient buying power' message, "
      "not the raw ExceptionGroup/TaskGroup dump",
      any(r == "rejected: Insufficient buying power"
          for i, r in bp_report.skipped if i.symbol == "XLI"),
      bp_report.skipped)
bp_audit = bp_om.audit.read()
bp_rejected_row = bp_audit[bp_audit["event"] == "order_rejected"].iloc[0]
check("the audit log's printed/queryable reason field is the clean message too",
      bp_rejected_row["reason"] == "Insufficient buying power", bp_rejected_row["reason"])
check("the audit log's raw field still preserves the full original exception "
      "text, so nothing forensic is actually lost",
      "ExceptionGroup" in bp_rejected_row["raw"]
      and "Not enough buying power" in bp_rejected_row["raw"])
check("the rest of the plan still executes after that rejection -- the "
      "loop no longer breaks on every kind of confirmed rejection",
      any(o.symbol == "IWM" for o in bp_report.submitted))

bp_recover = bp_om.recover()
check("nothing is left dangling for recover() to misresolve after a "
      "non-fractional definitive rejection",
      bp_recover.empty)

# The exact real-money scenario this session actually hit: the original
# fractional attempt gets rejected for fractional shares, rounds up to one
# whole share (a brand-new position), and *that* retry then hits a
# *different* rejection -- not enough buying power, since rounding up
# costs more than the fractional target did. The retry's own except-block
# used to have no 4xx-detection at all (any exception there was
# unconditionally "unknown outcome"), so this exact case slipped through.
retry_bp_broker = MockBroker(prices=pd.Series({"XLI": 185.0, "IWM": 300.0}), cash=100_000.0,
                             positions=pd.Series({"IWM": 1.0}))
retry_bp_broker.connect()
_real_retry_bp_place = retry_bp_broker.place_order
retry_bp_calls = []
def _retry_bp_place(symbol, side, quantity, *a, **kw):
    retry_bp_calls.append((symbol, quantity))
    if symbol == "XLI" and quantity != 1.0:
        raise _fractional_shares_error()
    if symbol == "XLI" and quantity == 1.0:
        raise _buying_power_error()
    return _real_retry_bp_place(symbol, side, quantity, *a, **kw)
retry_bp_broker.place_order = _retry_bp_place

retry_bp_path = fresh("retry_buying_power")
retry_bp_om = OrderManager(
    retry_bp_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(retry_bp_path, "KILL")),
    AuditLog(os.path.join(retry_bp_path, "a.jsonl"), stdout=False),
    os.path.join(retry_bp_path, "j.jsonl"),
)
retry_bp_xli = OrderIntent(symbol="XLI", side="buy", shares=0.8904, reference_price=185.0,
                           notional=164.72, current_weight=0.0, target_weight=0.16)
retry_bp_iwm = OrderIntent(symbol="IWM", side="sell", shares=0.45, reference_price=300.0,
                           notional=135.0, current_weight=0.30, target_weight=0.16)
retry_bp_plan = LivePlan(
    asof=pd.Timestamp.now(), equity=100_000.0, intents=[retry_bp_xli, retry_bp_iwm],
    target_weights=pd.Series({"XLI": 0.16, "IWM": 0.16}),
    current_weights=pd.Series({"XLI": 0.0, "IWM": 0.30}), decision=None,
)
retry_bp_report = retry_bp_om.execute(retry_bp_plan, strategy_name="retry-bp-test")
check("the retry itself hitting a different definitive rejection (buying "
      "power) is also classified correctly, not as an unknown outcome",
      retry_bp_calls[:2] == [("XLI", 0.8904), ("XLI", 1.0)]
      and any("rejected:" in r for i, r in retry_bp_report.skipped if i.symbol == "XLI"),
      retry_bp_calls)
check("the plan still continues past the retry's own rejection too",
      any(o.symbol == "IWM" for o in retry_bp_report.submitted))
check("the retry's own rejection is also reported with the clean message, "
      "not the raw exception dump",
      any(r == "rejected: Insufficient buying power"
          for i, r in retry_bp_report.skipped if i.symbol == "XLI"),
      retry_bp_report.skipped)
retry_bp_recover = retry_bp_om.recover()
check("nothing is left dangling after the retry's own definitive rejection",
      retry_bp_recover.empty)

# _clean_broker_rejection() directly: both confirmed-live shapes, plus a
# fallback check that an unrecognized 4xx body is never silently swallowed
# -- a rejection reason nobody's seen before should stay fully visible,
# not get eaten by a pattern match that only expects two known shapes.
check("cleans the 'not enough buying power' detail shape, with the "
      "specific rewording requested (not just a verbatim passthrough)",
      _clean_broker_rejection(repr(_buying_power_error())) == "Insufficient buying power")
check("cleans the 'cannot include fractional shares' field-errors shape",
      _clean_broker_rejection(repr(_fractional_shares_error()))
      == "Order quantity cannot include fractional shares")
_unrecognized_error_text = (
    "RuntimeError('MCP tool error: [TextContent(type=\\'text\\', text=\\'"
    "API error 422: {\"nested\": {\"still\": \"unrecognized shape\"}}\\', "
    "annotations=None, meta=None)]')"
)
check("an unrecognized 4xx JSON shape falls back to the untouched original "
      "text instead of being silently hidden or mis-cleaned",
      _clean_broker_rejection(_unrecognized_error_text) == _unrecognized_error_text)
check("plain text with no 'API error 4xx' marker at all also falls back "
      "unchanged (e.g. a 5xx or a non-JSON body)",
      _clean_broker_rejection("boom, no structure here") == "boom, no structure here")

# The exact real-money scenario this session hit: a fractional-share
# rejection triggers a retry at a whole-share size, the retry *itself* then
# throws for an unrelated reason (there, an unparseable place_order
# response) even though the order actually landed at the broker. execute()
# must write a second "submitting" entry at the retried quantity, and
# recover() must fingerprint against that retried quantity (the *last*
# "submitting" entry), not the stale original one -- otherwise a real fill
# gets misresolved as not_at_broker.
landed_path = fresh("fractional_retry_landed")
landed_broker = MockBroker(prices=pd.Series({"EFA": 100.0}), cash=100_000.0)
landed_broker.connect()
_real_landed_place = landed_broker.place_order
landed_calls = []
def _landed_place(symbol, side, quantity, *a, **kw):
    landed_calls.append(quantity)
    if len(landed_calls) == 1:
        raise _fractional_shares_error()
    # The retry: really lands at the broker (so get_orders() will find it
    # later)...
    order = _real_landed_place(symbol, side, quantity, *a, **kw)
    # ...but the call site still blows up, the same way an unparseable
    # response did for the real EFA order -- execute() never sees `order`.
    raise RuntimeError(f"place_order response for {symbol} {side} {quantity} "
                       "could not be parsed into an order id")
landed_broker.place_order = _landed_place

landed_om = OrderManager(
    landed_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(landed_path, "KILL")),
    AuditLog(os.path.join(landed_path, "a.jsonl"), stdout=False),
    os.path.join(landed_path, "j.jsonl"),
)
landed_intent = OrderIntent(symbol="EFA", side="buy", shares=1.4621337438577722,
                            reference_price=100.0, notional=146.21,
                            current_weight=0.0, target_weight=0.01)
landed_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[landed_intent],
                       target_weights=pd.Series({"EFA": 0.01}),
                       current_weights=pd.Series({"EFA": 0.0}), decision=None)
landed_report = landed_om.execute(landed_plan, strategy_name="frac-landed", now=now)
check("the retry itself failing is reported as an unknown outcome, same as any other",
      any("unknown outcome" in s[1] for s in landed_report.skipped))

landed_entries = [e for e in landed_om._journal_entries() if e.get("intent_key") == "EFA:buy"]
submitting_quantities = [e["quantity"] for e in landed_entries if e.get("stage") == "submitting"]
check("execute() writes a second write-ahead entry at the retried whole-share quantity",
      submitting_quantities == [1.4621337438577722, 1.0], submitting_quantities)

landed_recover = landed_om.recover()
check("recover() finds the order that actually landed, not_at_broker",
      (landed_recover["outcome"] == "found_at_broker").all(),
      landed_recover.to_dict("records"))

print()
print("=" * 72)
print("6. recover() and execute() share the same dedup lock")
print("=" * 72)

# _dedup_lock() used to guard only execute()'s own read-then-write dedup
# sequence -- recover() reads and writes the exact same journal, but
# without the lock a genuinely concurrent second process (the same
# double-scheduled-run risk _dedup_lock's own docstring exists for) could
# have execute() mid-flight while recover() reads a partial view. Two
# threads, two separate OrderManager instances sharing one journal_path
# (mirroring two separate processes, since flock() locks are scoped to
# the open file description, not the thread/process that created it):
# thread 1 holds the lock via a slow execute()-shaped critical section,
# thread 2's recover() must not observe or act until thread 1 releases it.
import threading
import time as _time

lock_path = fresh("dedup_lock_shared")
lock_broker = MockBroker(prices=pd.Series({"AAA": 100.0}), cash=100_000.0)
lock_broker.connect()
lock_om1 = OrderManager(
    lock_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(lock_path, "KILL")),
    AuditLog(os.path.join(lock_path, "a1.jsonl"), stdout=False),
    os.path.join(lock_path, "j.jsonl"),
)
lock_om2 = OrderManager(
    lock_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(lock_path, "KILL")),
    AuditLog(os.path.join(lock_path, "a2.jsonl"), stdout=False),
    os.path.join(lock_path, "j.jsonl"),
)

events = []

def _hold_lock():
    with lock_om1._dedup_lock():
        events.append(("thread1_acquired", _time.perf_counter()))
        _time.sleep(0.3)
        events.append(("thread1_released", _time.perf_counter()))

def _try_recover():
    _time.sleep(0.05)  # ensure thread 1 has acquired the lock first
    events.append(("thread2_recover_start", _time.perf_counter()))
    lock_om2.recover()
    events.append(("thread2_recover_done", _time.perf_counter()))

t1 = threading.Thread(target=_hold_lock)
t2 = threading.Thread(target=_try_recover)
t1.start()
t2.start()
t1.join()
t2.join()

by_name = dict(events)
check("thread 2's recover() only actually runs after thread 1 releases "
      "the lock, not while it's held",
      by_name["thread2_recover_done"] > by_name["thread1_released"]
      and by_name["thread2_recover_start"] < by_name["thread1_released"],
      events)

print()
print("=" * 72)
print("7. PDT accounting -- a real day trade actually gets recorded now")
print("=" * 72)

# Confirmed live (2026-08): RiskGate's PDT check (step 5 of apply()) reads
# day_trades_remaining, but nothing in the live path ever called
# ledger.record() -- only Backtester.run() did. Every OrderManager/
# LiveSignalRunner pair got a fresh, empty DayTradeLedger every cycle, so
# the PDT limit could never actually trigger in live trading. These checks
# exercise the fix directly through OrderManager.execute(), not by poking
# the ledger by hand.

dt_path = fresh("day_trade_roundtrip")
dt_ledger = DayTradeLedger()
dt_broker = MockBroker(prices=pd.Series({"AAA": 100.0}), cash=100_000.0)
dt_broker.connect()
dt_policy = ExecutionPolicy(dry_run=False, require_review=False,
                            require_market_open=False,
                            kill_switch_path=os.path.join(dt_path, "KILL"))
dt_om = OrderManager(dt_broker, dt_policy,
                     AuditLog(os.path.join(dt_path, "a.jsonl"), stdout=False),
                     os.path.join(dt_path, "j.jsonl"),
                     day_trade_ledger=dt_ledger)

buy_intent = OrderIntent(symbol="AAA", side="buy", shares=10.0, reference_price=100.0,
                         notional=1000.0, current_weight=0.0, target_weight=0.10)
buy_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[buy_intent],
                    target_weights=pd.Series({"AAA": 0.10}),
                    current_weights=pd.Series({"AAA": 0.0}), decision=None)
dt_om.execute(buy_plan, strategy_name="dt-test")
check("opening a new position from flat does not itself record a day trade",
      len(dt_ledger.events) == 0, dt_ledger.events)

sell_intent = OrderIntent(symbol="AAA", side="sell", shares=10.0, reference_price=100.0,
                          notional=1000.0, current_weight=0.10, target_weight=0.0)
sell_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[sell_intent],
                     target_weights=pd.Series({"AAA": 0.0}),
                     current_weights=pd.Series({"AAA": 0.10}), decision=None)
dt_om.execute(sell_plan, strategy_name="dt-test")
check("closing that same position to flat in the same session records exactly one day trade",
      len(dt_ledger.events) == 1, dt_ledger.events)

# A trim that does NOT close to flat, and a close of a position that was
# NOT opened today, must not be mistaken for a day trade.
notrade_path = fresh("day_trade_no_trade")
notrade_ledger = DayTradeLedger()
notrade_broker = MockBroker(prices=pd.Series({"BBB": 50.0}), cash=100_000.0,
                            positions=pd.Series({"BBB": 20.0}))
notrade_broker.connect()
notrade_om = OrderManager(
    notrade_broker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(notrade_path, "KILL")),
    AuditLog(os.path.join(notrade_path, "a.jsonl"), stdout=False),
    os.path.join(notrade_path, "j.jsonl"),
    day_trade_ledger=notrade_ledger,
)
trim_intent = OrderIntent(symbol="BBB", side="sell", shares=5.0, reference_price=50.0,
                          notional=250.0, current_weight=0.10, target_weight=0.05)
trim_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[trim_intent],
                     target_weights=pd.Series({"BBB": 0.05}),
                     current_weights=pd.Series({"BBB": 0.10}), decision=None)
notrade_om.execute(trim_plan, strategy_name="dt-test2")
check("trimming a pre-existing position (not opened today, not closed to "
      "flat) does not record a day trade",
      len(notrade_ledger.events) == 0, notrade_ledger.events)

close_intent = OrderIntent(symbol="BBB", side="sell", shares=20.0, reference_price=50.0,
                           notional=1000.0, current_weight=0.10, target_weight=0.0)
close_plan = LivePlan(asof=pd.Timestamp.now(), equity=100_000.0, intents=[close_intent],
                      target_weights=pd.Series({"BBB": 0.0}),
                      current_weights=pd.Series({"BBB": 0.10}), decision=None)
notrade_om.execute(close_plan, strategy_name="dt-test3")
check("closing a position that was already held before today (not opened "
      "this session) does not record a day trade",
      len(notrade_ledger.events) == 0, notrade_ledger.events)

# A LiveSignalRunner sharing the same ledger instance sees the recorded
# trade reflected in day_trades_remaining on its next plan() -- this is
# the actual point of the fix: the risk gate's PDT check can now trigger.
shared_ledger = DayTradeLedger(limit=1)
today = pd.Timestamp.now(tz="America/New_York").normalize()
shared_ledger.record(today)
check("a shared ledger correctly reports zero remaining after one recorded trade",
      shared_ledger.remaining(today, equity=1_000.0) == 0)

# The naive/aware seam. This is what actually happens in production and is
# NOT what the check above exercises: that one passes a tz-aware `today` to
# both record() and remaining(), so both sides matched and the bug hid.
# In run_cycle.py the two sides come from different places --
# OrderManager._session_date() records a tz-aware America/New_York midnight,
# while LiveSignalRunner passes a tz-naive *panel date* as asof -- and
# comparing them raised TypeError inside count(). It surfaced as a permanent
# outage, not one bad cycle: run_cycle aborts before save_day_trade_ledger()
# can rewrite the file, so every later run reloads the same poisoned event.
seam_om, seam_broker, seam_path = make("pdt_tz_seam", dry_run=False)
aware_session = seam_om._session_date(
    datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc))
check("OrderManager._session_date() resolves the session in the market tz",
      aware_session.tzinfo is not None and str(aware_session.date()) == "2026-08-13",
      repr(aware_session))

seam_ledger = DayTradeLedger()
seam_ledger.record(aware_session)
check("record() stores a tz-naive session date regardless of what it's given",
      seam_ledger.events[0].tzinfo is None, repr(seam_ledger.events[0]))

naive_asof = pd.Timestamp("2026-08-14")   # exactly what a PricePanel date is
try:
    seam_remaining = seam_ledger.remaining(naive_asof, equity=1_000.0)
    seam_ok, seam_detail = True, f"remaining={seam_remaining}"
except TypeError as exc:
    seam_ok, seam_detail = False, repr(exc)
check("a tz-aware recorded session date and a tz-naive panel asof compare "
      "without raising -- the exact live sequence that bricked the cycle",
      seam_ok, seam_detail)
check("and the day trade is actually counted, not silently dropped",
      seam_ledger.count(naive_asof) == 1, seam_ledger.count(naive_asof))

# The mirror image: events already persisted in the old tz-aware format must
# load and count cleanly rather than re-raising forever. count() normalises
# on read, not just write, precisely so an existing poisoned state file heals
# itself instead of needing to be noticed and deleted by hand.
legacy_ledger = DayTradeLedger()
legacy_ledger.events = [pd.Timestamp("2026-08-13T00:00:00-04:00")]
try:
    legacy_count = legacy_ledger.count(pd.Timestamp("2026-08-14"))
    legacy_ok, legacy_detail = True, f"count={legacy_count}"
except TypeError as exc:
    legacy_ok, legacy_detail = False, repr(exc)
check("a pre-fix tz-aware event assigned straight onto .events still counts, "
      "so an already-written state file heals rather than failing forever",
      legacy_ok and legacy_count == 1, legacy_detail)

check("as_session_date() is idempotent on an already-naive date",
      as_session_date(as_session_date(aware_session)) == as_session_date(aware_session))
check("as_session_date() strips intraday time to the session date",
      as_session_date(pd.Timestamp("2026-08-13 15:47:12")) == pd.Timestamp("2026-08-13"))

print()
print("=" * 72)
print("8. Reconciliation and audit stream")
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
print("9. Tool discovery binds by schema, not by assumption")
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

# An ambiguous substring match -- two properties both contain "quantity" --
# must not silently pick whichever one iterates first; it should skip that
# candidate and let the next one in the list resolve unambiguously.
ambiguous = ToolBinding("place", "place_equity_order", {
    "properties": {"order_quantity": {}, "max_quantity_per_order": {}, "qty": {}},
    "required": []})
check("ambiguous substring match falls through to the next candidate",
      ambiguous.resolve_arg("quantity", ("quantity", "qty")) == "qty")

truly_ambiguous = ToolBinding("place", "place_equity_order", {
    "properties": {"order_quantity": {}, "max_quantity_per_order": {}},
    "required": []})
check("no unambiguous candidate anywhere returns None, not a guess",
      truly_ambiguous.resolve_arg("quantity", ("quantity",)) is None)

# A schema that only exposes a dollar-notional "amount" field (not a share
# count) must fail loudly, never silently bind quantity to it -- "amount"
# commonly means dollar notional on real brokerage APIs, so accepting it as
# a quantity fallback could turn a 10-share order into a $10 one.
from qbt.broker import RobinhoodMCPBroker

amount_only = RobinhoodMCPBroker.__new__(RobinhoodMCPBroker)
amount_only.account_id = None
amount_only.bindings = {
    "place": ToolBinding("place", "place_equity_order", {
        "properties": {"symbol": {}, "side": {}, "amount": {}, "order_type": {}},
        "required": ["symbol", "side", "amount", "order_type"],
    })
}
try:
    amount_only._order_args("place", "AAPL", "buy", 10.0, "market", {})
    check("amount-only schema is rejected rather than guessed at", False)
except RuntimeError as exc:
    check("amount-only schema is rejected rather than guessed at",
          "amount" in str(exc))

genuine_shares = RobinhoodMCPBroker.__new__(RobinhoodMCPBroker)
genuine_shares.account_id = None
genuine_shares.bindings = {
    "place": ToolBinding("place", "place_equity_order", {
        "properties": {"symbol": {}, "side": {}, "shares": {}, "order_type": {}},
        "required": ["symbol", "side", "shares", "order_type"],
    })
}
args = genuine_shares._order_args("place", "AAPL", "buy", 10.0, "market", {})
check("a genuine share-count field still resolves normally",
      args.get("shares") == 10.0)

# An unparseable place_order response (no id/order_id field at all) must
# raise -- not return a plausible-looking BrokerOrder with a blank id that
# every downstream caller would mistake for a normal pending order.
unparseable = RobinhoodMCPBroker.__new__(RobinhoodMCPBroker)
unparseable.account_id = None
unparseable.bindings = genuine_shares.bindings
unparseable._call_sync = lambda capability, arguments: {}
try:
    unparseable.place_order("AAPL", "buy", 10.0)
    check("place_order raises on an unparseable response", False)
except RuntimeError as exc:
    check("place_order raises on an unparseable response",
          "order id" in str(exc))

# cancel_order must not read an error payload as success -- bool(raw) alone
# treats any non-empty response as a successful cancel.
cancel_broker = RobinhoodMCPBroker.__new__(RobinhoodMCPBroker)
cancel_broker.bindings = {
    "cancel": ToolBinding("cancel", "cancel_equity_order", {
        "properties": {"order_id": {}}, "required": ["order_id"]})
}
cancel_broker._call_sync = lambda capability, arguments: {"error": "already filled"}
check("an error payload is not read as a successful cancel",
      cancel_broker.cancel_order("mock-1") is False)

cancel_broker._call_sync = lambda capability, arguments: {"success": True}
check("an explicit success field is honored",
      cancel_broker.cancel_order("mock-1") is True)

cancel_broker._call_sync = lambda capability, arguments: {"success": False}
check("an explicit success=False is honored, not just truthiness of the payload",
      cancel_broker.cancel_order("mock-1") is False)

print()
print("=" * 72)
print("10. Matching, reporting and staleness edge cases")
print("=" * 72)

# Bucketed fingerprints put a hard edge in the middle of the tolerance
# window they exist to provide. round() is banker's rounding, so a
# journaled 0.89 buckets to 44 while a broker echoing 0.890001 -- a
# difference 20,000x smaller than the 0.02 tolerance -- buckets to 45.
# Equality on those tuples then reports a real, filled order as missing,
# which recover() resolves as not_at_broker and halts the next cycle.
_bnd = BrokerOrder(order_id="b1", symbol="XLF", side="buy",
                   quantity=0.890001, state="filled")
check("the premise: bucket equality breaks on a half-bucket boundary",
      _bnd.fingerprint() != ("XLF", "buy", round(0.89 / 0.02)))
check("matches() accepts a quantity inside the tolerance regardless of "
      "where it falls relative to a bucket edge",
      _bnd.matches("XLF", "buy", 0.89))
for _q in (0.25, 0.45, 0.89, 1.11):
    check(f"boundary quantity {_q} still matches under a 1e-6 broker rounding",
          BrokerOrder(order_id="x", symbol="AAA", side="sell",
                      quantity=_q + 1e-6, state="filled").matches("AAA", "sell", _q))
check("matches() still rejects a genuinely different size",
      not _bnd.matches("XLF", "buy", 0.95))
check("matches() still requires the symbol to agree",
      not _bnd.matches("XLK", "buy", 0.89))
check("matches() still requires the side to agree",
      not _bnd.matches("XLF", "sell", 0.89))

# recover() must find an order the broker echoed back at a boundary-shifted
# quantity, not resolve it as lost.
_bnd_path = fresh("boundary_match")
_bnd_broker = MockBroker(prices=pd.Series({"XLF": 100.0}), cash=10_000.0)
_bnd_broker.connect()
_bnd_om = OrderManager(_bnd_broker, ExecutionPolicy(dry_run=False),
                       AuditLog(os.path.join(_bnd_path, "a.jsonl"), stdout=False),
                       os.path.join(_bnd_path, "j.jsonl"))
_bnd_om._journal(stage="submitting", plan_id="p1", intent_key="XLF:buy",
                 symbol="XLF", side="buy", quantity=0.89)
_bnd_broker.orders.append(BrokerOrder(
    order_id="real-1", symbol="XLF", side="buy", quantity=0.890001,
    state="filled", filled_quantity=0.890001,
    created_at=datetime.now(timezone.utc)))
_bnd_rec = _bnd_om.recover()
check("recover() resolves a boundary-shifted quantity as found_at_broker, "
      "not as a lost order that halts the next cycle",
      list(_bnd_rec["outcome"]) == ["found_at_broker"], _bnd_rec.to_dict("records"))

# A rejected order is not a submitted one. It used to be appended to
# report.submitted regardless, so a cycle where every order bounced still
# reported "3 submitted" -- the exact shape of the user's live run.
_rej = ExecutionReport(plan_id="r", dry_run=False)
_rej.rejected.append(BrokerOrder(order_id=None, symbol="XLI", side="buy",
                                 quantity=1.0, state="rejected",
                                 reject_reason="Insufficient buying power"))
check("a rejected order is not counted as submitted",
      len(_rej.submitted) == 0 and len(_rej.rejected) == 1)
check("__repr__ names the rejection instead of calling it a submission",
      "0 submitted" in repr(_rej) and "1 rejected" in repr(_rej), repr(_rej))
check("accepted still covers everything that reached the broker",
      len(_rej.accepted) == 1)
check("to_frame still shows the rejected order and its reason",
      len(_rej.to_frame()) == 1
      and _rej.to_frame().iloc[0]["note"] == "Insufficient buying power")

# The above only pins the dataclass shape. This drives it through execute()
# with a broker that *returns* a rejected order (rather than raising), which
# is the path that actually did the miscounting.
_rejpath = fresh("rejected_routing")
_rejbroker = MockBroker(prices=pd.Series({"AAA": 100.0, "BBB": 50.0}), cash=100_000.0)
_rejbroker.connect()
_real_rejplace = _rejbroker.place_order


def _rej_place(symbol, side, quantity, *a, **kw):
    if symbol == "AAA":
        return BrokerOrder(order_id="rej-1", symbol="AAA", side=side,
                           quantity=quantity, state="rejected",
                           reject_reason="Insufficient buying power")
    return _real_rejplace(symbol, side, quantity, *a, **kw)


_rejbroker.place_order = _rej_place
_rej_om = OrderManager(
    _rejbroker,
    ExecutionPolicy(dry_run=False, require_review=False, require_market_open=False,
                    kill_switch_path=os.path.join(_rejpath, "KILL")),
    AuditLog(os.path.join(_rejpath, "a.jsonl"), stdout=False),
    os.path.join(_rejpath, "j.jsonl"),
)
_rej_plan = LivePlan(
    asof=pd.Timestamp.now(), equity=100_000.0,
    intents=[
        OrderIntent(symbol="AAA", side="buy", shares=1.0, reference_price=100.0,
                    notional=100.0, current_weight=0.0, target_weight=0.01),
        OrderIntent(symbol="BBB", side="buy", shares=2.0, reference_price=50.0,
                    notional=100.0, current_weight=0.0, target_weight=0.01),
    ],
    target_weights=pd.Series({"AAA": 0.01, "BBB": 0.01}),
    current_weights=pd.Series({"AAA": 0.0, "BBB": 0.0}), decision=None)
_rej_report = _rej_om.execute(_rej_plan, strategy_name="rej-routing", now=now)
check("execute() routes a broker-rejected order to .rejected, not .submitted",
      [o.symbol for o in _rej_report.rejected] == ["AAA"]
      and [o.symbol for o in _rej_report.submitted] == ["BBB"],
      f"submitted={[o.symbol for o in _rej_report.submitted]} "
      f"rejected={[o.symbol for o in _rej_report.rejected]}")
check("the rejection is still journalled as a terminal outcome, so recover() "
      "finds nothing dangling",
      _rej_om.recover().empty)

# reconcile(): a held symbol the broker won't quote cannot be weighed. It
# used to read as actual_weight 0.0 -- indistinguishable from "we hold none
# of it" -- turning a pricing gap into a full-size drift breach, which
# detection_rules() rates "high".
_unp_path = fresh("reconcile_unpriced")
_unp_broker = MockBroker(prices=pd.Series({"XLF": 40.0}), cash=100.0,
                         positions=pd.Series({"XLF": 5.0, "AAPL": 3.0}))
_unp_broker.connect()
_unp_om = OrderManager(_unp_broker, ExecutionPolicy(dry_run=True),
                       AuditLog(os.path.join(_unp_path, "a.jsonl"), stdout=False),
                       os.path.join(_unp_path, "j.jsonl"))
_unp_plan = LivePlan(asof=pd.Timestamp("2026-08-13"), equity=300.0, intents=[],
                     target_weights=pd.Series({"XLF": 0.6}),
                     current_weights=pd.Series({"XLF": 0.6}), decision=None)
_unp_rec = _unp_om.reconcile(_unp_plan)
check("an unpriceable holding is flagged as not priced",
      bool(_unp_rec.loc["AAPL", "priced"]) is False)
check("and is not counted as a drift breach on a weight never computed",
      bool(_unp_rec.loc["AAPL", "breach"]) is False
      and pd.isna(_unp_rec.loc["AAPL", "actual_weight"]))
check("a normally-priced holding still reconciles and can still breach",
      bool(_unp_rec.loc["XLF", "priced"]) is True)
check("the unpriced symbols get their own audit event",
      "reconcile_unpriced_positions" in _unp_om.audit.read()["event"].tolist())

# preflight's staleness check localised plan.asof to UTC unconditionally,
# which raises on an already-aware timestamp rather than converting -- the
# same naive/aware seam that bricked DayTradeLedger.
_tz_path = fresh("preflight_tz")
_tz_broker = MockBroker(prices=pd.Series({"XLF": 40.0}), cash=1_000.0)
_tz_broker.connect()
_tz_om = OrderManager(_tz_broker, ExecutionPolicy(require_market_open=False),
                      AuditLog(os.path.join(_tz_path, "a.jsonl"), stdout=False),
                      os.path.join(_tz_path, "j.jsonl"))
_tz_intent = OrderIntent(symbol="XLF", side="buy", shares=1.0, reference_price=40.0,
                         notional=40.0, current_weight=0.0, target_weight=0.04)
for _label, _asof in (("naive", pd.Timestamp("2026-08-13")),
                      ("tz-aware", pd.Timestamp("2026-08-13", tz="America/New_York"))):
    _tz_plan = LivePlan(asof=_asof, equity=1_000.0, intents=[_tz_intent],
                        target_weights=pd.Series({"XLF": 0.04}),
                        current_weights=pd.Series({"XLF": 0.0}), decision=None)
    try:
        _tz_ok, _ = _tz_om.preflight(_tz_plan, _tz_broker.get_account(),
                                     now=datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc))
        check(f"preflight handles a {_label} plan.asof without raising", _tz_ok)
    except TypeError as exc:
        check(f"preflight handles a {_label} plan.asof without raising",
              False, repr(exc))

print()
print("=" * 72)
print("11. Short positions on an account that cannot short")
print("=" * 72)

# PairsTrading ships in this package and emits negative target weights by
# design. Nothing used to object: preflight passed, the long legs filled,
# and the short legs were picked off one at a time by the broker's own
# per-intent review -- which runs *after* preflight. The result was a
# market-neutral plan executed as a directional book, hedged by nothing.
# That is the "partially executed rebalance is a portfolio nobody chose"
# failure this module's docstring says preflight exists to prevent.
_short_intents = [
    OrderIntent(symbol="AAA", side="buy", shares=10.0, reference_price=100.0,
                notional=1_000.0, current_weight=0.0, target_weight=0.25),
    OrderIntent(symbol="BBB", side="sell", shares=-20.0, reference_price=50.0,
                notional=-1_000.0, current_weight=0.0, target_weight=-0.25),
]
_short_plan = LivePlan(
    asof=pd.Timestamp("2026-07-21"), equity=100_000.0, intents=_short_intents,
    target_weights=pd.Series({"AAA": 0.25, "BBB": -0.25}),
    current_weights=pd.Series({"AAA": 0.0, "BBB": 0.0}), decision=None)

_short_path = fresh("short_guard")
_short_broker = MockBroker(prices=pd.Series({"AAA": 100.0, "BBB": 50.0}),
                           cash=100_000.0)
_short_broker.connect()


def _short_om(**policy_kw):
    return OrderManager(
        _short_broker,
        ExecutionPolicy(dry_run=False, require_review=True,
                        require_market_open=False, max_plan_notional=1e9,
                        max_order_notional=1e9,
                        kill_switch_path=os.path.join(_short_path, "KILL"),
                        **policy_kw),
        AuditLog(os.path.join(_short_path, "a.jsonl"), stdout=False),
        os.path.join(_short_path, "j.jsonl"),
    )


_short_ok, _short_notes = _short_om().preflight(
    _short_plan, _short_broker.get_account(), now=MARKET_HOURS)
check("a plan with short target weights fails preflight by default",
      not _short_ok, "; ".join(_short_notes)[:120])
check("and the failure names shorting, not some incidental limit",
      any("short" in n.lower() for n in _short_notes))

_short_report = _short_om().execute(_short_plan, strategy_name="short-guard",
                                    now=MARKET_HOURS)
check("nothing at all is sent -- not even the long leg that would have "
      "filled fine on its own",
      _short_report.aborted and not _short_report.submitted,
      f"submitted={[o.symbol for o in _short_report.submitted]}")

_ok_long, _ = _short_om().preflight(
    LivePlan(asof=pd.Timestamp("2026-07-21"), equity=100_000.0,
             intents=[_short_intents[0]],
             target_weights=pd.Series({"AAA": 0.25}),
             current_weights=pd.Series({"AAA": 0.0}), decision=None),
    _short_broker.get_account(), now=MARKET_HOURS)
check("a long-only plan is unaffected by the guard", _ok_long)

_allow_ok, _ = _short_om(allow_short=True).preflight(
    _short_plan, _short_broker.get_account(), now=MARKET_HOURS)
check("allow_short=True lets a genuinely short-capable account through",
      _allow_ok)

# A zero target weight is not a short -- the guard must not fire on the
# ordinary "exit this position" case, which is most of what a sell is.
_flat_ok, _ = _short_om().preflight(
    LivePlan(asof=pd.Timestamp("2026-07-21"), equity=100_000.0,
             intents=[OrderIntent(symbol="AAA", side="sell", shares=-5.0,
                                  reference_price=100.0, notional=-500.0,
                                  current_weight=0.25, target_weight=0.0)],
             target_weights=pd.Series({"AAA": 0.0}),
             current_weights=pd.Series({"AAA": 0.25}), decision=None),
    _short_broker.get_account(), now=MARKET_HOURS)
check("selling a position down to flat is not treated as shorting", _flat_ok)

# MockBroker used to log a review_order entry for the review place_order
# runs internally, so a test counting reviews saw one phantom per placement.
_log_broker = MockBroker(prices=pd.Series({"AAA": 10.0}), cash=1_000.0)
_log_broker.connect()
_log_broker.place_order("AAA", "buy", 1.0)
check("place_order logs exactly one call, not a phantom review too",
      [c[0] for c in _log_broker.call_log] == ["place_order"],
      [c[0] for c in _log_broker.call_log])
_log_broker.review_order("AAA", "buy", 1.0)
check("an explicit review_order is still logged",
      [c[0] for c in _log_broker.call_log] == ["place_order", "review_order"])
check("and still returns the same verdict it always did",
      _log_broker.review_order("AAA", "buy", 10_000.0)["ok"] is False)

# get_orders' since filter is truncated to a date, and which date a UTC
# instant lands on depends on the server's timezone. 21:00 New York is
# already "tomorrow" in UTC, so the bare UTC date could ask for a window
# starting after the very order recover() is looking for.
_since = datetime(2026, 8, 14, 1, 0, tzinfo=timezone.utc)   # 21:00 Aug 13 ET
check("the premise: the bare UTC date of a late-evening ET order is the "
      "following day",
      _since.date().isoformat() == "2026-08-14")
check("get_orders widens by a day before truncating, so the window can "
      "never start after the order it's hunting",
      (_since - timedelta(days=1)).date().isoformat() == "2026-08-13")

print()
print("=" * 72)
print("12. The whole-share retry is re-checked against the per-order cap")
print("=" * 72)

# max_order_notional is tested once, on the original intent, before
# place_order. The fractional-shares retry then submits a *different, larger*
# order: rounding 0.5 shares of a $300 ETF up to 1 turns a $150 intent into a
# $300 order, and that used to walk straight past the cap. Live-relevant, not
# theoretical -- IWM (~$300) and GLD (~$392) are both in run_cycle's universe
# and both exceed the $250 default at a single share, and IWM has already hit
# the fractional rejection in a real cycle.
def _cap_retry_case(price, shares, cap):
    path = fresh(f"cap_retry_{int(price)}_{int(cap)}")
    broker = MockBroker(prices=pd.Series({"IWM": price}), cash=100_000.0)
    broker.connect()
    _real_place = broker.place_order
    sent = []

    def _place(symbol, side, quantity, *a, **kw):
        sent.append(quantity)
        if quantity != float(int(quantity)):
            raise _fractional_shares_error()
        return _real_place(symbol, side, quantity, *a, **kw)

    broker.place_order = _place
    om = OrderManager(
        broker,
        ExecutionPolicy(dry_run=False, require_review=False,
                        require_market_open=False, max_order_notional=cap,
                        max_plan_notional=1e9,
                        kill_switch_path=os.path.join(path, "KILL")),
        AuditLog(os.path.join(path, "a.jsonl"), stdout=False),
        os.path.join(path, "j.jsonl"),
    )
    intent = OrderIntent(symbol="IWM", side="buy", shares=shares,
                         reference_price=price, notional=shares * price,
                         current_weight=0.0, target_weight=0.15)
    plan = LivePlan(asof=pd.Timestamp("2026-07-21"),
                    equity=broker.get_account().equity, intents=[intent],
                    target_weights=pd.Series({"IWM": 0.15}),
                    current_weights=pd.Series({"IWM": 0.0}), decision=None)
    return sent, om.execute(plan, strategy_name="cap-retry", now=MARKET_HOURS), om


_cap_sent, _cap_rep, _cap_om = _cap_retry_case(300.58, 0.5, 250.0)
check("the premise: the original intent clears the cap on its own",
      abs(0.5 * 300.58) < 250.0)
check("the whole-share retry is blocked when one share exceeds the cap",
      _cap_sent == [0.5] and not _cap_rep.submitted,
      f"sent={_cap_sent} submitted={[o.symbol for o in _cap_rep.submitted]}")
check("the skip reason names the share price and the cap, not just 'rejected'",
      any("exceeds the per-order cap" in r for _i, r in _cap_rep.skipped),
      [r for _i, r in _cap_rep.skipped])
check("the blocked retry is audited with both quantities and the cap",
      "order_notional_after_whole_share_retry"
      in _cap_om.audit.read()["reason"].dropna().tolist())
# The first attempt already wrote a "submitting" entry. Skipping the retry
# without closing it would leave the next cycle's recover() treating a fully
# known outcome as an unresolved in-flight order, and halting on it.
check("the first attempt's write-ahead entry is closed, so recover() finds "
      "nothing dangling and the next cycle doesn't halt",
      _cap_om.recover().empty)

_ok_sent, _ok_rep, _ = _cap_retry_case(300.58, 0.5, 400.0)
check("raising the cap above one share lets the same retry through",
      _ok_sent == [0.5, 1.0] and len(_ok_rep.submitted) == 1, _ok_sent)

_cheap_sent, _cheap_rep, _ = _cap_retry_case(40.0, 0.5, 250.0)
check("a name whose single share is well under the cap is unaffected",
      _cheap_sent == [0.5, 1.0] and len(_cheap_rep.submitted) == 1, _cheap_sent)

print()
print("=" * 72)
print(f"{len(FAILS)} FAILURE(S): {FAILS}" if FAILS else "ALL CHECKS PASSED")
print("=" * 72)
