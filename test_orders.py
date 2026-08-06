"""Validate the order path against MockBroker. No network, no account."""

import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from qbt import (
    CrossSectionalMomentum, LiveSignalRunner, LivePlan, OrderIntent,
    PortfolioState, RiskGate, SyntheticRepository,
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

# A target size that rounds down to zero whole shares can't be corrected --
# must skip cleanly (terminal "rejected" in the journal), not retry at 0.
tiny_path = fresh("fractional_tiny")
tiny_broker = MockBroker(prices=pd.Series({"CCC": 300.0}), cash=100_000.0)
tiny_broker.connect()
def _tiny_place(symbol, side, quantity, *a, **kw):
    raise _fractional_shares_error()
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
check("a sub-one-share fractional rejection is skipped cleanly, not retried at 0 shares",
      any("rounds down to 0" in s[1] for s in tiny_report.skipped))

tiny_recover = tiny_om.recover()
check("a definitively-rejected sub-share order leaves nothing dangling -- recover() "
      "can't misresolve it as not_at_broker and HALT the next cycle",
      tiny_recover.empty)

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
print(f"{len(FAILS)} FAILURE(S): {FAILS}" if FAILS else "ALL CHECKS PASSED")
print("=" * 72)
