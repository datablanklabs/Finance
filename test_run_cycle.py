"""Validate run_cycle.py's live STRATEGY wiring (BreadthRegimeFilter +
MacroRegimeFilter + KalshiEventRegimeFilter wrapped around a two-member
Composite core).

This is a script, not a package module, so it's loaded here via importlib
rather than a normal import. The point of this file is narrow: prove the
composition on disk is what the docstring/README claim (right nesting, right
capital shares, right thresholds), that all three overlays actually fire and
compound correctly when their regime read is unfavourable, and that the
Composite core's fundamentals-dependent member degrades to cash rather than
crashing when fundamentals aren't available -- not to re-test
BreadthRegimeFilter, MacroRegimeFilter, KalshiEventRegimeFilter, or
MultiFactorCrossSectional themselves (that's test_regime_filters.py's and
test_qbt.py's job).
"""

import importlib.util
import json
import re
import subprocess
import sys
import os
import pathlib
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from qbt import (
    Backtester, BreadthRegimeFilter, Composite, CrossSectionalMomentum,
    DayTradeLedger, ExecutionConfig, FundamentalsPanel, KalshiEventRegimeFilter,
    LiveSignalRunner, MacroRegimeFilter, MacrosPanel, MultiFactorCrossSectional,
    PortfolioState, PricePanel, RiskGate, ShortHorizonReversal,
    SyntheticRepository,
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
kalshi_filter = macro.inner
core = kalshi_filter.inner

check("outer wrapper is BreadthRegimeFilter", isinstance(breadth, BreadthRegimeFilter))
check("middle wrapper is MacroRegimeFilter", isinstance(macro, MacroRegimeFilter))
check("innermost overlay is KalshiEventRegimeFilter",
      isinstance(kalshi_filter, KalshiEventRegimeFilter))
check("core strategy is a two-member Composite", isinstance(core, Composite))
check("Composite has exactly two members", len(core.members) == 2)

check("breadth lookback is 200", breadth.lookback == 200)
check("breadth min_breadth is 0.3", breadth.min_breadth == 0.3)
check("breadth scale_when_blocked is 0.5", breadth.scale_when_blocked == 0.5)

check("macro metric is vix", macro.metric == "vix")
check("macro max_level is 35.0", macro.max_level == 35.0)
check("macro max_increase is 15.0", macro.max_increase == 15.0)
check("macro lookback is 21", macro.lookback == 21)
check("macro scale_when_blocked is 0.5", macro.scale_when_blocked == 0.5)

check("kalshi horizon_days is 3", kalshi_filter.horizon_days == 3)
check("kalshi scale_when_blocked is 0.5", kalshi_filter.scale_when_blocked == 0.5)
check("kalshi max_age_days is 3", kalshi_filter.max_age_days == 3)

mom_member, mom_share = core.members[0]
mf_member, mf_share = core.members[1]
check("first Composite member is CrossSectionalMomentum",
      isinstance(mom_member, CrossSectionalMomentum))
check("momentum lookback/skip/top_n untouched by the overlays",
      (mom_member.lookback, mom_member.skip, mom_member.top_n) == (63, 5, 5))
check("momentum member holds 60% of the core's capital share",
      abs(mom_share / (mom_share + mf_share) - 0.6) < 1e-9,
      f"share={mom_share}/{mom_share + mf_share}")

check("second Composite member is MultiFactorCrossSectional",
      isinstance(mf_member, MultiFactorCrossSectional))
check("multifactor member holds 40% of the core's capital share",
      abs(mf_share / (mom_share + mf_share) - 0.4) < 1e-9,
      f"share={mf_share}/{mom_share + mf_share}")
check("multifactor member's quality factor has a nonzero weight",
      mf_member.factor_weights.get("quality", 0.0) > 0.0,
      mf_member.factor_weights)
check("multifactor member reads the FMP ratios ROE column",
      mf_member.quality_metric == "ratios_return_on_equity",
      mf_member.quality_metric)

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
print("3. fundamentals=None degrades the quality sleeve to cash, not a crash "
      "-- and real fundamentals bring it back")
print("=" * 72)

# The multifactor member's quality factor has a nonzero weight (checked in
# section 1), which means -- per MultiFactorCrossSectional's own docstring
# -- fundamentals=None makes every name fail to qualify for that 40% share,
# not just the ones missing a reading. bare_plan above already exercised
# this (no fundamentals passed) without raising; this section makes it an
# explicit, permanent check rather than an implicit side effect of section 2.
check("no fundamentals -> the multifactor member alone contributes nothing",
      mf_member.target_weights(panel.as_of(panel.dates[-1])).abs().sum() == 0.0)
check("...but the momentum member is unaffected, so the core still trades",
      bare_plan.turnover > 0, f"turnover={bare_plan.turnover}")

# With real dispersion, the quality sleeve should actually pick names --
# proving the wiring moves data, not just that it fails to crash.
real_fundamentals = FundamentalsPanel(frame=pd.DataFrame(
    {
        "symbol": panel.symbols,
        "metric": "ratios_return_on_equity",
        "period_end": pd.DatetimeIndex([panel.dates[0]] * len(panel.symbols)),
        "as_of_date": pd.DatetimeIndex([panel.dates[0]] * len(panel.symbols)),
        "value": [10.0 + 5.0 * (i % 7) for i in range(len(panel.symbols))],
    }
))
mf_weights = mf_member.target_weights(panel.as_of(panel.dates[-1]), real_fundamentals)
check("real fundamentals -> the multifactor member picks names",
      mf_weights.abs().sum() > 0.0, f"gross={mf_weights.abs().sum()}")

funded_plan = LiveSignalRunner(strategy=run_cycle.STRATEGY, risk_gate=None,
                               max_turnover=None).plan(
    panel, state, fundamentals=real_fundamentals, macros=calm_macros)
funded_names = {i.symbol for i in funded_plan.intents if abs(i.target_weight) > 1e-9}
check("a funded plan's picks differ from the momentum-only core's "
      "(the quality sleeve is actually contributing, not just failing to crash)",
      funded_names != bare_names, f"funded={funded_names} bare={bare_names}")

print()
print("=" * 72)
print("4. Elevated VIX alone scales the book to 50%, without touching breadth")
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
print("5. macros=None (a fetch failure) is a no-op on the macro overlay, not a block")
print("=" * 72)

none_macro_plan = runner.plan(panel, state, macros=None)
check("macros=None leaves turnover matching the calm-VIX plan (documented pass-through)",
      abs(none_macro_plan.turnover - calm_plan.turnover) < 1e-9,
      f"none={none_macro_plan.turnover} calm={calm_plan.turnover}")

print()
print("=" * 72)
print("6. Day-trade ledger survives a save/load round trip and stays comparable")
print("=" * 72)

# The failure this guards against was permanent, not transient: a tz-aware
# session date recorded by OrderManager raised TypeError inside
# DayTradeLedger.count() when compared against a tz-naive panel asof, and
# run_cycle.py aborts the cycle (exit 3) *before* save_day_trade_ledger()
# can rewrite the file -- so every subsequent run reloaded the same poisoned
# event and failed identically, with only cycle_error in the audit log.
#
# Anchored to *now*, not a fixed calendar date -- save_day_trade_ledger's own
# trim cutoff (10 business days, see its docstring) is relative to real time,
# so a hardcoded event date here would eventually drift outside that window
# and start failing for reasons having nothing to do with the tz behaviour
# under test (confirmed: a 2026-08-13 literal written when this test still
# had headroom silently went stale and started failing once "today" passed
# roughly two business weeks past it). 2 calendar days back is comfortably
# inside both that 10-business-day save cutoff and DayTradeLedger.count()'s
# own 5-business-day window on any weekday this happens to run.
work = tempfile.mkdtemp()
dt_path = os.path.join(work, "day_trades_live.json")

broker = MockBroker(prices=pd.Series({"XLF": 100.0}), cash=1_000.0)
broker.connect()
om = OrderManager(
    broker, ExecutionPolicy(),
    AuditLog(os.path.join(work, "a.jsonl"), stdout=False),
    os.path.join(work, "j.jsonl"),
)
recent_utc = (datetime.now(timezone.utc) - pd.Timedelta(days=2)).replace(
    hour=14, minute=0, second=0, microsecond=0)
aware_session = om._session_date(recent_utc)
om.day_trade_ledger.record(aware_session)
# What record() actually stored -- as_session_date() of the same instant,
# not the recent_utc calendar date itself, since the market-tz conversion
# inside _session_date can shift the wall-clock date at the boundary (it
# doesn't at 14:00 UTC / 10:00 America/New_York, but deriving the expected
# value the same way the code under test does is the point, not assuming).
expected_event = run_cycle.as_session_date(aware_session)

run_cycle.save_day_trade_ledger(dt_path, om.day_trade_ledger)
with open(dt_path) as fh:
    persisted = json.load(fh)
check("save_day_trade_ledger writes tz-naive session dates",
      persisted["events"] == [expected_event.isoformat()], persisted)

reloaded = run_cycle.load_day_trade_ledger(dt_path)
naive_asof = expected_event + pd.Timedelta(days=1)
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
# Same relative anchor as above (expected_event), reformatted as a pre-fix
# tz-aware ISO string (America/New_York's UTC offset) rather than a second
# independent hardcoded date -- the two need to land on the same calendar
# day for this section's counts to mean anything.
legacy_path = os.path.join(work, "legacy.json")
with open(legacy_path, "w") as fh:
    json.dump({"events": [f"{expected_event.date().isoformat()}T00:00:00-04:00"]}, fh)
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
      healed["events"] == [expected_event.isoformat()], healed)

print()
print("=" * 72)
print("7. A daily backtest cannot produce a day trade -- and says so")
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
print("8. Holdings outside the strategy universe are surfaced, not swallowed")
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
print("9. --max-position drives both concentration caps, keeping them layered")
print("=" * 72)

# There are two position caps and they are layered, not redundant:
# RiskGate.max_weight *clips* proposed weights so no target normally reaches
# the backstop, while ExecutionPolicy.max_position_weight is a preflight fatal
# for when something upstream misbehaves -- plus the re-check on the
# whole-share retry, which preflight cannot see because it inspects target
# weights rather than retried quantities. One flag moves both so there is a
# single number to reason about; the margin keeps the backstop a backstop.
# GATE must describe the configured policy completely on its own. Filling
# max_weight in only inside main() meant RiskGate(**GATE) from a probe, a
# notebook or a test silently fell back to RiskGate's own 0.25 default rather
# than the 0.30 configured here -- a tighter cap than the operator set, and
# invisible.
check("GATE is complete at import, not half-filled by main()",
      "max_weight" in run_cycle.GATE, run_cycle.GATE)
check("RiskGate(**GATE) yields the configured cap, not RiskGate's own default",
      RiskGate(**run_cycle.GATE).max_weight == 0.30,
      RiskGate(**run_cycle.GATE).max_weight)
_helptext = subprocess.run(
    [sys.executable, "run_cycle.py", "--help"], capture_output=True, text=True,
    cwd=str(pathlib.Path(__file__).parent)).stdout
check("the flag's default is sourced from GATE rather than restated",
      f"(default: {run_cycle.GATE['max_weight']:.2f})" in _helptext,
      [l for l in _helptext.splitlines() if "default:" in l])
check("main() does not mutate the module-level GATE",
      run_cycle.GATE["max_weight"] == 0.30, run_cycle.GATE)
check("the backstop margin is a named constant, not a literal at the use site",
      run_cycle.POSITION_BACKSTOP_MARGIN == 0.05,
      run_cycle.POSITION_BACKSTOP_MARGIN)


class _AllIn:
    """Wants the whole book in one name, so the cap is the only thing binding."""
    name = "allin"
    min_history = 5

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None, kalshi=None):
        return pd.Series({view.symbols[0]: 1.0}).reindex(view.symbols).fillna(0.0)


_pos_idx = pd.bdate_range("2024-01-01", periods=300)
_pos_panel = PricePanel(close=pd.DataFrame(
    {"AAA": np.linspace(100, 200, 300), "BBB": np.linspace(100, 120, 300)},
    index=_pos_idx))
_pos_state = PortfolioState(cash=10_000.0, shares=pd.Series(dtype=float))

for _mp in (0.30, 0.45, 0.60):
    _gate = dict(run_cycle.GATE, max_weight=_mp, target_vol=None)
    _backstop = min(1.0, _mp + run_cycle.POSITION_BACKSTOP_MARGIN)
    _plan = LiveSignalRunner(strategy=_AllIn(), risk_gate=RiskGate(**_gate),
                             max_turnover=None).plan(_pos_panel, _pos_state)
    _got = float(_plan.target_weights.max())
    check(f"--max-position {_mp} clips the book to exactly {_mp}",
          abs(_got - _mp) < 1e-9, _got)
    check(f"and the backstop stays strictly looser ({_backstop:.2f} > {_mp:.2f})",
          _backstop > _mp)

# A weight is a fraction. Passing 30 meaning "30%" would otherwise disable the
# cap outright rather than tightening it, which is the worst possible
# direction for a typo in a live risk limit to fail.
_bad = subprocess.run(
    [sys.executable, "run_cycle.py", "--max-position", "30", "--synthetic"],
    capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
check("--max-position 30 is rejected rather than read as 3000%",
      _bad.returncode != 0 and "must be a weight in (0, 1]" in _bad.stderr,
      _bad.stderr.strip()[-90:])
for _v in ("0", "-0.1", "1.5"):
    _r = subprocess.run(
        [sys.executable, "run_cycle.py", "--max-position", _v, "--synthetic"],
        capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
    check(f"--max-position {_v} is rejected", _r.returncode != 0)

# argparse %-formats help strings, so an unescaped percent sign swallows the
# rest and prints the raw action dict in --help.
_help = subprocess.run([sys.executable, "run_cycle.py", "--help"],
                       capture_output=True, text=True,
                       cwd=str(pathlib.Path(__file__).parent)).stdout
check("--help renders the flag's help text rather than an action dict",
      "max single-name weight" in _help and "option_strings" not in _help)

print()
print("=" * 72)
print("10. Crypto sleeve -- CRYPTO_STRATEGY/CRYPTO_GATE wiring and "
      "--consider-crypto end to end")
print("=" * 72)

check("CRYPTOS is a non-empty list of yfinance-style tickers",
      len(run_cycle.CRYPTOS) > 0 and all("-" in s for s in run_cycle.CRYPTOS),
      run_cycle.CRYPTOS)

crypto_breadth = run_cycle.CRYPTO_STRATEGY
crypto_macro = crypto_breadth.inner
crypto_kalshi_filter = crypto_macro.inner
crypto_core = crypto_kalshi_filter.inner
check("crypto outer wrapper is BreadthRegimeFilter", isinstance(crypto_breadth, BreadthRegimeFilter))
check("crypto middle wrapper is MacroRegimeFilter", isinstance(crypto_macro, MacroRegimeFilter))
check("crypto innermost overlay is KalshiEventRegimeFilter",
      isinstance(crypto_kalshi_filter, KalshiEventRegimeFilter))
check("crypto core strategy is a two-member Composite", isinstance(crypto_core, Composite))
check("crypto Composite has exactly two members", len(crypto_core.members) == 2)

crypto_mom_member, crypto_mom_share = crypto_core.members[0]
crypto_mf_member, crypto_mf_share = crypto_core.members[1]
check("crypto first member is CrossSectionalMomentum",
      isinstance(crypto_mom_member, CrossSectionalMomentum))
check("crypto momentum member holds 60% of the core's capital share",
      abs(crypto_mom_share / (crypto_mom_share + crypto_mf_share) - 0.6) < 1e-9)
check("crypto second member is MultiFactorCrossSectional",
      isinstance(crypto_mf_member, MultiFactorCrossSectional))
check("crypto multifactor member has NO quality factor -- there is no such "
      "thing as a crypto fundamentals reading",
      "quality" not in crypto_mf_member.factor_weights, crypto_mf_member.factor_weights)
check("crypto multifactor's momentum/low_vol/reversal weights sum to 1.0 "
      "(quality dropped and the rest re-normalised, not just left as a "
      "60%-of-intended blend)",
      abs(sum(crypto_mf_member.factor_weights.values()) - 1.0) < 1e-9,
      crypto_mf_member.factor_weights)

check("CRYPTO_GATE has its own, tighter max_weight than the equity GATE "
      "(crypto concentration risk is a real, separate judgement call)",
      run_cycle.CRYPTO_GATE["max_weight"] < run_cycle.GATE["max_weight"],
      (run_cycle.CRYPTO_GATE["max_weight"], run_cycle.GATE["max_weight"]))

check("--consider-crypto and --max-crypto-* flags exist",
      "--consider-crypto" in _help and "--max-crypto-order" in _help and
      "--max-crypto-plan" in _help and "--max-crypto-position" in _help)

_bad_crypto = subprocess.run(
    [sys.executable, "run_cycle.py", "--max-crypto-position", "20", "--synthetic"],
    capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
check("--max-crypto-position 20 is rejected rather than read as 2000%",
      _bad_crypto.returncode != 0 and
      "must be a weight in (0, 1]" in _bad_crypto.stderr,
      _bad_crypto.stderr.strip()[-90:])

# End to end: both sleeves plan and place mock fills in one process, over
# genuinely independent synthetic universes (no symbol collision -- see
# run_crypto_pipeline's own comment on why it builds a second MockBroker
# rather than reusing the equity one), and the process exits 0 -- the
# strongest available offline evidence that main() keeps the shared broker
# connection open across both sleeves (run_crypto_pipeline's MockBroker
# path would raise "broker not connected" from MockBroker._require() if
# main() had already closed anything the crypto sleeve still needed).
#
# Journal/peak-equity files under the real audit/state/ dirs persist
# between runs by design (that's the whole point of the idempotency
# journal -- see qbt/orders.py) -- which makes a *second* run of this exact
# synthetic scenario correctly skip re-submitting ("already submitted for
# this plan") rather than fill again. Deleting the synthetic-suffixed files
# this test depends on first keeps the "submitted"/"filled" counts below
# deterministic regardless of what other manual or automated runs left
# behind; never touches the "_live" (real) files.
for _stale in (
    run_cycle.mode_path("audit/journal.jsonl", True),
    run_cycle.mode_path("audit/journal_crypto.jsonl", True),
    run_cycle.mode_path("state/peak_equity.txt", True),
    run_cycle.mode_path("state/peak_equity_crypto.txt", True),
    run_cycle.mode_path("state/day_trades.json", True),
):
    if os.path.exists(_stale):
        os.remove(_stale)

_e2e = subprocess.run(
    [sys.executable, "run_cycle.py", "--synthetic", "--ignore-market-hours",
     "--live", "--consider-crypto",
     "--max-plan", "20000", "--max-order", "3000",
     "--max-crypto-plan", "5000", "--max-crypto-order", "1000"],
    capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
check("a --consider-crypto cycle exits 0 when both sleeves complete cleanly",
      _e2e.returncode == 0, (_e2e.returncode, _e2e.stdout[-2000:], _e2e.stderr[-500:]))
check("both sleeves' sections appear in the output",
      "--- equity ---" in _e2e.stdout and "--- crypto ---" in _e2e.stdout)
_crypto_cycle_end = re.search(
    r"\[cycle_end\][^\n]*sleeve=crypto", _e2e.stdout.split("--- crypto ---")[-1])
_crypto_submitted = (
    int(re.search(r"submitted=(\d+)", _crypto_cycle_end.group()).group(1))
    if _crypto_cycle_end else None
)
check("the crypto sleeve actually placed and filled orders, not just planned",
      _crypto_submitted is not None and _crypto_submitted > 0,
      (_crypto_submitted, _e2e.stdout[-800:]))
check("crypto journal/peak-equity state land in their own crypto-suffixed "
      "files, not the equity ones",
      os.path.exists(run_cycle.mode_path("audit/journal_crypto.jsonl", True)) and
      os.path.exists(run_cycle.mode_path("state/peak_equity_crypto.txt", True)))

# A run without --consider-crypto must be completely unaffected -- no crypto
# section, no crypto capability lookup, no crypto state files touched. Sized
# the same generous --max-plan/--max-order as the e2e run above so this is
# actually exercising a completed cycle, not an unrelated preflight abort on
# the (much smaller) default caps.
_no_crypto = subprocess.run(
    [sys.executable, "run_cycle.py", "--synthetic", "--ignore-market-hours",
     "--max-plan", "20000", "--max-order", "3000"],
    capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
check("omitting --consider-crypto runs equity-only, no crypto section printed",
      _no_crypto.returncode == 0 and "--- crypto ---" not in _no_crypto.stdout,
      _no_crypto.stdout[-500:])

print()
print("=" * 72)
print("11. --max-price excludes expensive symbols from the panel entirely")
print("=" * 72)

_price_panel = SyntheticRepository(n_symbols=18, seed=42).fetch(
    start="2010-01-01", end="2020-01-01")
_price_last = _price_panel.last_close()
_price_cap = float(_price_last.median())
_price_work = tempfile.mkdtemp()
_price_audit = AuditLog(os.path.join(_price_work, "a.jsonl"), stdout=False)

check("max_price=None is a true no-op -- returns the same object, not a copy",
      run_cycle.apply_price_cap(_price_panel, None, "equity", _price_audit)
      is _price_panel)

_capped_panel = run_cycle.apply_price_cap(_price_panel, _price_cap, "equity", _price_audit)
check("every remaining symbol's last close is within the cap",
      bool((_price_last[_capped_panel.symbols] <= _price_cap).all()),
      dict(_price_last[_capped_panel.symbols]))
check("at least one symbol was actually excluded (the cap is the median, "
      "not a no-op by construction)",
      len(_capped_panel.symbols) < len(_price_panel.symbols),
      (len(_capped_panel.symbols), len(_price_panel.symbols)))
check("apply_price_cap() emits an audit event naming what it excluded",
      "price_cap_excluded" in _price_audit.read()["event"].values)

# A cap below every symbol's price empties the panel rather than raising --
# consistent with the rest of this module's "degrade gracefully, don't
# crash" philosophy (an empty PricePanel is a valid, if useless, one; the
# strategy just proposes no positions, same as any other data-starved cycle).
_empty_capped = run_cycle.apply_price_cap(_price_panel, 0.01, "equity", _price_audit)
check("a cap below every price empties the panel instead of raising",
      len(_empty_capped.symbols) == 0)

# End to end: the flag actually reaches the equity pipeline and the printed
# exclusion list matches what a direct apply_price_cap() call would produce.
_expensive = sorted(_price_last[_price_last > _price_cap].index)
_price_e2e = subprocess.run(
    [sys.executable, "run_cycle.py", "--synthetic", "--ignore-market-hours",
     "--max-price", str(_price_cap), "--max-plan", "20000", "--max-order", "3000"],
    capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent))
check("--max-price reaches the equity pipeline and excludes the same names "
      "a direct apply_price_cap() call would",
      _price_e2e.returncode == 0 and
      all(s in _price_e2e.stdout for s in _expensive),
      (_price_e2e.returncode, _expensive, _price_e2e.stdout[:400]))

check("--max-crypto-price flag exists and is independent of --max-price",
      "--max-crypto-price" in _help and "--max-price" in _help)

print()
print("=" * 72)
print(f"{len(FAILS)} failing check(s)" if FAILS else "All checks passed")
print("=" * 72)

if FAILS:
    raise SystemExit(1)
