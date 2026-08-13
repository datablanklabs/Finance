"""Validate run_cycle.py's live STRATEGY wiring (BreadthRegimeFilter + MacroRegimeFilter).

This is a script, not a package module, so it's loaded here via importlib
rather than a normal import. The point of this file is narrow: prove the
composition on disk is what the docstring/README claim (right nesting, right
thresholds) and that both overlays actually fire and compound correctly when
their regime read is unfavourable -- not to re-test BreadthRegimeFilter or
MacroRegimeFilter themselves (that's test_regime_filters.py's job).
"""

import importlib.util
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone

import pandas as pd

from qbt import (
    Backtester, BreadthRegimeFilter, CrossSectionalMomentum, DayTradeLedger,
    ExecutionConfig, LiveSignalRunner, MacroRegimeFilter, MacrosPanel,
    PortfolioState, ShortHorizonReversal, SyntheticRepository,
)
from qbt.broker import MockBroker
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


spec = importlib.util.spec_from_file_location(
    "run_cycle", pathlib.Path(__file__).parent / "run_cycle.py"
)
run_cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_cycle)

print("=" * 72)
print("1. STRATEGY composition -- matches the documented wiring")
print("=" * 72)

breadth = run_cycle.STRATEGY
macro = breadth.inner
core = macro.inner

check("outer wrapper is BreadthRegimeFilter", isinstance(breadth, BreadthRegimeFilter))
check("middle wrapper is MacroRegimeFilter", isinstance(macro, MacroRegimeFilter))
check("inner strategy is CrossSectionalMomentum", isinstance(core, CrossSectionalMomentum))

check("breadth lookback is 200", breadth.lookback == 200)
check("breadth min_breadth is 0.3", breadth.min_breadth == 0.3)
check("breadth scale_when_blocked is 0.5", breadth.scale_when_blocked == 0.5)

check("macro metric is vix", macro.metric == "vix")
check("macro max_level is 35.0", macro.max_level == 35.0)
check("macro max_increase is 15.0", macro.max_increase == 15.0)
check("macro lookback is 21", macro.lookback == 21)
check("macro scale_when_blocked is 0.5", macro.scale_when_blocked == 0.5)

check("core momentum lookback/skip/top_n untouched by the overlays",
      (core.lookback, core.skip, core.top_n) == (63, 5, 5))

print()
print("=" * 72)
print("2. Neither overlay changes which names get picked, only total exposure")
print("=" * 72)

repo = SyntheticRepository(n_symbols=18, seed=42)
panel = repo.fetch(start="2010-01-01", end="2020-06-30")
state = PortfolioState(cash=25_000.0, shares=pd.Series(dtype=float))

calm_macros = MacrosPanel(frame=pd.DataFrame(
    [("vix", d, d, 15.0) for d in panel.dates],
    columns=["metric", "period_end", "as_of_date", "value"],
))
runner = LiveSignalRunner(strategy=run_cycle.STRATEGY, risk_gate=None, max_turnover=None)
calm_plan = runner.plan(panel, state, macros=calm_macros)

bare_runner = LiveSignalRunner(strategy=core, risk_gate=None, max_turnover=None)
bare_plan = bare_runner.plan(panel, state)

calm_names = {i.symbol for i in calm_plan.intents if abs(i.target_weight) > 1e-9}
bare_names = {i.symbol for i in bare_plan.intents if abs(i.target_weight) > 1e-9}
check("calm regime picks the same names as the unwrapped core strategy",
      calm_names == bare_names, f"calm={calm_names} bare={bare_names}")

print()
print("=" * 72)
print("3. Elevated VIX alone scales the book to 50%, without touching breadth")
print("=" * 72)

high_vix_macros = MacrosPanel(frame=pd.DataFrame(
    [("vix", d, d, 45.0) for d in panel.dates],
    columns=["metric", "period_end", "as_of_date", "value"],
))
regime_view = panel.as_of(panel.dates[-1])
check("macro filter reads elevated VIX as blocked",
      macro.blocked(regime_view, high_vix_macros.as_of(regime_view.last_date())))

high_vix_plan = runner.plan(panel, state, macros=high_vix_macros)
ratio = high_vix_plan.turnover / bare_plan.turnover if bare_plan.turnover else None
check("turnover under elevated VIX is ~50% of the unscaled plan (macro-only trigger)",
      ratio is not None and abs(ratio - 0.5) < 0.02, f"ratio={ratio}")

print()
print("=" * 72)
print("4. macros=None (a fetch failure) is a no-op on the macro overlay, not a block")
print("=" * 72)

none_macro_plan = runner.plan(panel, state, macros=None)
check("macros=None leaves turnover matching the calm-VIX plan (documented pass-through)",
      abs(none_macro_plan.turnover - calm_plan.turnover) < 1e-9,
      f"none={none_macro_plan.turnover} calm={calm_plan.turnover}")

print()
print("=" * 72)
print("5. Day-trade ledger survives a save/load round trip and stays comparable")
print("=" * 72)

# The failure this guards against was permanent, not transient: a tz-aware
# session date recorded by OrderManager raised TypeError inside
# DayTradeLedger.count() when compared against a tz-naive panel asof, and
# run_cycle.py aborts the cycle (exit 3) *before* save_day_trade_ledger()
# can rewrite the file -- so every subsequent run reloaded the same poisoned
# event and failed identically, with only cycle_error in the audit log.
work = tempfile.mkdtemp()
dt_path = os.path.join(work, "day_trades_live.json")

broker = MockBroker(prices=pd.Series({"XLF": 100.0}), cash=1_000.0)
broker.connect()
om = OrderManager(
    broker, ExecutionPolicy(),
    AuditLog(os.path.join(work, "a.jsonl"), stdout=False),
    os.path.join(work, "j.jsonl"),
)
aware_session = om._session_date(datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc))
om.day_trade_ledger.record(aware_session)

run_cycle.save_day_trade_ledger(dt_path, om.day_trade_ledger)
with open(dt_path) as fh:
    persisted = json.load(fh)
check("save_day_trade_ledger writes tz-naive session dates",
      persisted["events"] == ["2026-08-13T00:00:00"], persisted)

reloaded = run_cycle.load_day_trade_ledger(dt_path)
naive_asof = pd.Timestamp("2026-08-14")
try:
    remaining = reloaded.remaining(naive_asof, equity=1_000.0)
    rt_ok, rt_detail = True, f"remaining={remaining}"
except TypeError as exc:
    rt_ok, rt_detail = False, repr(exc)
check("a reloaded ledger compares against a tz-naive panel asof without raising",
      rt_ok, rt_detail)
check("and the day trade survives the round trip", reloaded.count(naive_asof) == 1)

# save_day_trade_ledger's own trim is the mirror-image seam: once events are
# naive, a tz-aware UTC cutoff raises the same TypeError from the other
# direction -- and it would do it on the *write* path, immediately after a
# day trade was recorded.
try:
    run_cycle.save_day_trade_ledger(dt_path, reloaded)
    trim_ok, trim_detail = True, ""
except TypeError as exc:
    trim_ok, trim_detail = False, repr(exc)
check("re-saving an already-naive ledger doesn't raise on the trim cutoff",
      trim_ok, trim_detail)

# A state file written by the pre-fix code must heal, not fail forever.
legacy_path = os.path.join(work, "legacy.json")
with open(legacy_path, "w") as fh:
    json.dump({"events": ["2026-08-13T00:00:00-04:00"]}, fh)
legacy = run_cycle.load_day_trade_ledger(legacy_path)
try:
    legacy_count = legacy.count(naive_asof)
    legacy_ok, legacy_detail = True, f"count={legacy_count}"
except TypeError as exc:
    legacy_ok, legacy_detail = False, repr(exc)
check("a pre-fix tz-aware state file loads and counts instead of re-raising",
      legacy_ok and legacy_count == 1, legacy_detail)
run_cycle.save_day_trade_ledger(legacy_path, legacy)
with open(legacy_path) as fh:
    healed = json.load(fh)
check("and re-saving rewrites it in the clean naive format (self-healing)",
      healed["events"] == ["2026-08-13T00:00:00"], healed)

print()
print("=" * 72)
print("6. A daily backtest cannot produce a day trade -- and says so")
print("=" * 72)

# Backtester.run() has opened_on/ledger.record() bookkeeping, but on a daily
# panel it is structurally unreachable: execute() runs at most once per bar
# and a symbol appears once in `traded` per call, so nothing round-trips
# inside a session. This matters for *parity*, not correctness -- live
# genuinely can day-trade and trip RiskGate's PDT block, research never
# will, so a clean backtest is not evidence of staying inside the budget.
bt_panel = SyntheticRepository(n_symbols=8, seed=3).fetch(
    start="2021-01-01", end="2022-12-31")
for delay in (0, 1, 2):
    led = DayTradeLedger()
    ex = ExecutionConfig(delay_bars=delay, price="close" if delay == 0 else "open")
    res = Backtester(bt_panel, ShortHorizonReversal(lookback=3, top_n=2),
                     rebalance="D", initial_equity=25_000.0,
                     day_trade_ledger=led, execution=ex).run()
    multi = res.trades.groupby(["date", "symbol"]).size()
    check(f"delay_bars={delay}: max-churn daily rebalancing produces no "
          f"same-session round trip",
          len(led.events) == 0 and int((multi > 1).sum()) == 0,
          f"{len(res.trades)} trades, {len(led.events)} day trades, "
          f"{int((multi > 1).sum())} symbol-days with 2+ trades")

# The consumption side still works: a caller can seed a ledger to research
# an already-PDT-restricted account, and RiskGate will act on it.
seeded = DayTradeLedger(limit=3)
for _ in range(3):
    seeded.record(pd.Timestamp("2021-06-01"))
check("a seeded ledger still reports the budget as exhausted to RiskGate",
      seeded.remaining(pd.Timestamp("2021-06-02"), equity=1_000.0) == 0)
check("and reports it unrestricted above the $25k equity threshold",
      seeded.remaining(pd.Timestamp("2021-06-02"), equity=30_000.0) > 0)

print()
print("=" * 72)
print("7. Holdings outside the strategy universe are surfaced, not swallowed")
print("=" * 72)

# run_cycle reindexes account.positions onto panel.symbols, which it has to
# -- there's no price series to size or trade an off-universe name with. But
# that makes such a holding contribute nothing to plan.equity, produce no
# sell intent, and sit stranded forever, all silently. Worse, the missing
# value shows up indirectly as preflight equity drift against the broker's
# true equity, whose message ("Recompute, do not send") points at stale data
# rather than at the actual cause.
uni_prices = pd.Series({"XLF": 40.0, "XLK": 200.0})
positions = pd.Series({"XLF": 5.0, "XLK": 1.0, "AAPL": 3.0})   # AAPL off-universe

managed = PortfolioState(cash=100.0,
                         shares=positions.reindex(uni_prices.index).fillna(0.0))
check("the premise: an off-universe holding contributes nothing to plan equity",
      managed.equity(uni_prices) == 100.0 + 5 * 40.0 + 1 * 200.0,
      managed.equity(uni_prices))

held_all = positions[positions.abs() > 1e-9]
unmanaged = held_all[~held_all.index.isin(list(uni_prices.index))]
check("detection identifies exactly the off-universe names",
      list(unmanaged.index) == ["AAPL"], list(unmanaged.index))
clean_positions = pd.Series({"XLF": 5.0, "XLK": 1.0})
clean_held = clean_positions[clean_positions.abs() > 1e-9]
check("and reports nothing when every holding is in the universe",
      clean_held[~clean_held.index.isin(list(uni_prices.index))].empty)

# A fully-exited name (zero shares, still listed by the broker) is not an
# unmanaged holding -- reporting it would cry wolf every cycle.
stale_positions = pd.Series({"XLF": 5.0, "AAPL": 0.0})
stale_held = stale_positions[stale_positions.abs() > 1e-9]
check("a zero-share off-universe row is not reported as unmanaged",
      stale_held[~stale_held.index.isin(list(uni_prices.index))].empty)

# The understated equity is what trips preflight, and by how much scales
# with the unmanaged position -- small ones just size orders conservatively,
# large ones abort. Both are acceptable; being unexplained was not.
true_equity = 100.0 + 5 * 40.0 + 1 * 200.0 + 3 * 250.0
drift = abs(true_equity - managed.equity(uni_prices)) / true_equity
check("a large unmanaged holding drives equity drift past the 5% preflight "
      "limit, so the cycle aborts rather than trading on a wrong equity base",
      drift > ExecutionPolicy().max_equity_drift, f"drift={drift:.1%}")

check("run_cycle exposes the audit event name used to report this",
      "unmanaged_holdings" in pathlib.Path("run_cycle.py").read_text())

print()
print("=" * 72)
print(f"{len(FAILS)} failing check(s)" if FAILS else "All checks passed")
print("=" * 72)

if FAILS:
    raise SystemExit(1)
