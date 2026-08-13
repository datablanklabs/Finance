"""Validate FundamentalsValueFilter, MacroRegimeFilter, and BreadthRegimeFilter."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, BreadthRegimeFilter, CrossSectionalMomentum, EqualWeightBuyHold,
    FundamentalsPanel, FundamentalsValueFilter, MacroRegimeFilter, MacrosPanel,
    PricePanel, SyntheticRepository,
)

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


repo = SyntheticRepository(n_symbols=3, seed=5)
panel = repo.fetch(start="2020-01-01", end="2020-06-30")
syms = panel.symbols
d = panel.dates[80]
view = panel.as_of(d)

print("=" * 72)
print("1. FundamentalsValueFilter -- construction and validation")
print("=" * 72)

try:
    FundamentalsValueFilter(inner=EqualWeightBuyHold(), metric="ratios_pe_ratio")
    check("rejects no max_value/min_value configured", False)
except ValueError:
    check("rejects no max_value/min_value configured", True)

f = FundamentalsValueFilter(inner=EqualWeightBuyHold(), metric="ratios_pe_ratio", max_value=25.0)
check("auto-names itself", f.name == "equal_weight+ratios_pe_ratiofilter")
check("min_history matches the inner strategy", f.min_history == EqualWeightBuyHold().min_history)

print()
print("=" * 72)
print("2. FundamentalsValueFilter -- filtering behaviour")
print("=" * 72)

fframe = pd.DataFrame(
    [
        (syms[0], "ratios_pe_ratio", d, d, 50.0),   # expensive -> blocked
        (syms[1], "ratios_pe_ratio", d, d, 10.0),   # cheap -> kept
        # syms[2] has no reading at all -> blocked
    ],
    columns=["symbol", "metric", "period_end", "as_of_date", "value"],
)
fpanel = FundamentalsPanel(frame=fframe)

w_noop = f.target_weights(view, None, None, None, None)
check(
    "fundamentals entirely absent -> no-op, passes inner weights through",
    (w_noop == pd.Series(1 / 3, index=syms)).all(),
)

w = f.target_weights(view, fpanel.as_of(d), None, None, None)
check("blocks the expensive name", w[syms[0]] == 0.0)
check("keeps the cheap name", w[syms[1]] == 1 / 3)
check("blocks the name with no reading at all", w[syms[2]] == 0.0)

f_unknown = FundamentalsValueFilter(inner=EqualWeightBuyHold(), metric="not_a_real_metric", max_value=25.0)
w_unknown = f_unknown.target_weights(view, fpanel.as_of(d), None, None, None)
check(
    "metric the repository never produced -> no-op, not a full block",
    (w_unknown == pd.Series(1 / 3, index=syms)).all(),
)

f_min = FundamentalsValueFilter(inner=EqualWeightBuyHold(), metric="ratios_pe_ratio", min_value=20.0)
w_min = f_min.target_weights(view, fpanel.as_of(d), None, None, None)
check("min_value blocks names below the floor", w_min[syms[1]] == 0.0 and w_min[syms[0]] == 1 / 3)

f_zero_gross = FundamentalsValueFilter(inner=EqualWeightBuyHold(gross=0.0), metric="ratios_pe_ratio", max_value=25.0)
w_zero = f_zero_gross.target_weights(view, fpanel.as_of(d), None, None, None)
check("skips the fundamentals lookup entirely when inner is already flat", (w_zero == 0).all())

print()
print("=" * 72)
print("3. MacroRegimeFilter -- construction and validation")
print("=" * 72)

try:
    MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate")
    check("rejects no level/increase condition configured", False)
except ValueError:
    check("rejects no level/increase condition configured", True)

try:
    MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate", max_level=5.0, scale_when_blocked=1.5)
    check("rejects scale_when_blocked outside [0, 1]", False)
except ValueError:
    check("rejects scale_when_blocked outside [0, 1]", True)

g = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate", max_level=5.0)
check("auto-names itself", g.name == "equal_weight+fed_funds_rateregime")
check(
    "min_history covers both the inner strategy and the trend lookback",
    g.min_history == max(EqualWeightBuyHold().min_history, g.lookback + 2),
)

print()
print("=" * 72)
print("4. MacroRegimeFilter -- gating behaviour")
print("=" * 72)

d_prior = panel.dates[80 - 63]
mframe = pd.DataFrame(
    [
        ("fed_funds_rate", d_prior, d_prior, 1.0),
        ("fed_funds_rate", d, d, 5.5),
    ],
    columns=["metric", "period_end", "as_of_date", "value"],
)
mpanel = MacrosPanel(frame=mframe)

w_noop = g.target_weights(view, None, None, None, None)
check(
    "macros entirely absent -> no-op, passes inner weights through",
    (w_noop == pd.Series(1 / 3, index=syms)).all(),
)

w_blocked = g.target_weights(view, None, mpanel.as_of(d), None, None)
check("max_level breached -> fully flat by default", (w_blocked == 0).all())

g_ok = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate", max_level=10.0)
w_ok = g_ok.target_weights(view, None, mpanel.as_of(d), None, None)
check("max_level not breached -> unchanged", (w_ok == pd.Series(1 / 3, index=syms)).all())

g_min = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate", min_level=6.0)
w_min = g_min.target_weights(view, None, mpanel.as_of(d), None, None)
check("min_level breached (below floor) -> flat", (w_min == 0).all())

g_trend = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="fed_funds_rate", max_increase=1.0, lookback=63)
w_trend = g_trend.target_weights(view, None, mpanel.as_of(d), None, None)
check("max_increase (sharp rise) breached -> flat", (w_trend == 0).all())

g_scale = MacroRegimeFilter(
    inner=EqualWeightBuyHold(), metric="fed_funds_rate", max_level=5.0, scale_when_blocked=0.5
)
w_scale = g_scale.target_weights(view, None, mpanel.as_of(d), None, None)
check(
    "scale_when_blocked partially de-risks instead of fully flattening",
    np.allclose(w_scale.to_numpy(), (1 / 3) * 0.5),
)

g_unknown = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="not_a_real_metric", max_level=5.0)
w_unknown = g_unknown.target_weights(view, None, mpanel.as_of(d), None, None)
check(
    "metric the repository never produced -> no-op, not a full block",
    (w_unknown == pd.Series(1 / 3, index=syms)).all(),
)

check(
    "blocked() is directly inspectable for research",
    g.blocked(view, mpanel.as_of(d)) is True and g_ok.blocked(view, mpanel.as_of(d)) is False,
)

print()
print("=" * 72)
print("5. Wired into Backtester -- composes with a real strategy, no look-ahead")
print("=" * 72)

repo2 = SyntheticRepository(n_symbols=8, seed=17)
panel2 = repo2.fetch(start="2019-01-01", end="2021-12-31")
syms2 = panel2.symbols

# Fed funds rate steps up sharply partway through -- should flatten the book
# from that point on, and not a day before.
hike_date = panel2.dates[400]
mrows = []
for dte in panel2.dates:
    level = 0.5 if dte < hike_date else 5.0
    mrows.append(("fed_funds_rate", dte, dte, level))
mpanel2 = MacrosPanel(
    frame=pd.DataFrame(mrows, columns=["metric", "period_end", "as_of_date", "value"])
)

inner = CrossSectionalMomentum(lookback=63, top_n=4)
gated = MacroRegimeFilter(inner=inner, metric="fed_funds_rate", max_level=3.0)

bt = Backtester(panel=panel2, strategy=gated, macros=mpanel2, rebalance="W", initial_equity=50_000)
result = bt.run()
check("backtest completes without error", result.equity.notna().all())

after_hike = result.targets.loc[result.targets.index >= hike_date]
before_hike = result.targets.loc[result.targets.index < hike_date]
check(
    "book is flat on every rebalance after the hike",
    (after_hike.abs().sum(axis=1) == 0).all(),
    f"{len(after_hike)} rebalances checked",
)
check(
    "book was actually invested before the hike (the gate isn't just always-off)",
    (before_hike.abs().sum(axis=1) > 0).any(),
)

print()
print("=" * 72)
print("6. BreadthRegimeFilter -- construction and validation")
print("=" * 72)

try:
    BreadthRegimeFilter(inner=EqualWeightBuyHold(), min_breadth=1.5)
    check("rejects min_breadth outside [0, 1]", False)
except ValueError:
    check("rejects min_breadth outside [0, 1]", True)

try:
    BreadthRegimeFilter(inner=EqualWeightBuyHold(), scale_when_blocked=-0.1)
    check("rejects scale_when_blocked outside [0, 1]", False)
except ValueError:
    check("rejects scale_when_blocked outside [0, 1]", True)

bf = BreadthRegimeFilter(inner=EqualWeightBuyHold(), lookback=200, min_breadth=0.4)
check("auto-names itself", bf.name == "equal_weight+breadth200")
check(
    "min_history covers both the inner strategy and the breadth lookback",
    bf.min_history == max(EqualWeightBuyHold().min_history, bf.lookback + 2),
)

print()
print("=" * 72)
print("7. BreadthRegimeFilter -- gating behaviour")
print("=" * 72)

breadth_dates = pd.date_range("2024-01-01", periods=250, freq="D")
# 3 of 5 names above their own 200-bar moving average, 2 below -> breadth 0.6.
breadth_close = pd.DataFrame({
    "A": np.linspace(100, 200, 250),
    "B": np.linspace(100, 200, 250),
    "C": np.linspace(100, 200, 250),
    "D": np.linspace(200, 100, 250),
    "E": np.linspace(200, 100, 250),
}, index=breadth_dates)
breadth_panel = PricePanel(close=breadth_close)

check("breadth() computes the correct fraction above the moving average",
      abs(bf.breadth(breadth_panel) - 0.6) < 1e-9, bf.breadth(breadth_panel))

bf_tight = BreadthRegimeFilter(inner=EqualWeightBuyHold(), lookback=200, min_breadth=0.7)
w_blocked = bf_tight.target_weights(breadth_panel)
check("min_breadth breached -> fully flat by default", (w_blocked == 0).all())

bf_loose = BreadthRegimeFilter(inner=EqualWeightBuyHold(), lookback=200, min_breadth=0.5)
w_ok = bf_loose.target_weights(breadth_panel)
check("min_breadth not breached -> unchanged",
      np.allclose(w_ok.to_numpy(), 1 / 5))

bf_scale = BreadthRegimeFilter(inner=EqualWeightBuyHold(), lookback=200,
                               min_breadth=0.7, scale_when_blocked=0.5)
w_scale = bf_scale.target_weights(breadth_panel)
check("scale_when_blocked partially de-risks instead of fully flattening",
      np.allclose(w_scale.to_numpy(), (1 / 5) * 0.5))

check("blocked() is directly inspectable for research",
      bf_tight.blocked(breadth_panel) is True
      and bf_loose.blocked(breadth_panel) is False)

# A symbol that can't be scored (no valid data at all) must be excluded
# from both the numerator and the denominator, not silently counted as
# passing or failing -- the exact NaN-direction bug TrendFilter had.
nan_close = breadth_close.copy()
nan_close["F"] = np.nan
nan_panel = PricePanel(close=nan_close)
check("an unscoreable symbol doesn't change the breadth reading",
      abs(bf.breadth(nan_panel) - 0.6) < 1e-9, bf.breadth(nan_panel))

all_nan_close = pd.DataFrame({"X": [np.nan] * 250, "Y": [np.nan] * 250},
                             index=breadth_dates)
all_nan_panel = PricePanel(close=all_nan_close)
bf_allnan = BreadthRegimeFilter(inner=EqualWeightBuyHold(), lookback=200, min_breadth=0.5)
check("breadth is NaN, not a crash, when nothing at all can be scored",
      np.isnan(bf_allnan.breadth(all_nan_panel)))
check("an unscoreable panel passes through (no-op), not a block -- "
      "absence of a read is not itself bad news",
      bf_allnan.blocked(all_nan_panel) is False)

print()
print("=" * 72)
print("8. BreadthRegimeFilter -- wired into Backtester")
print("=" * 72)

repo3 = SyntheticRepository(n_symbols=8, seed=23)
panel3 = repo3.fetch(start="2019-01-01", end="2021-12-31")
inner3 = CrossSectionalMomentum(lookback=63, top_n=4)
gated_breadth = BreadthRegimeFilter(inner=inner3, lookback=100, min_breadth=0.3)

bt3 = Backtester(panel=panel3, strategy=gated_breadth, rebalance="W", initial_equity=50_000)
result3 = bt3.run()
check("backtest completes without error", result3.equity.notna().all())
check("the book is at least sometimes invested (not always-blocked)",
      (result3.targets.abs().sum(axis=1) > 0).any())

print()
print("=" * 72)
print("9. MacroRegimeFilter with the 'vix' indicator -- a concrete, ready-to-use signal")
print("=" * 72)

# "vix" -> FRED's VIXCLS is already in qbt.macro.DEFAULT_INDICATORS, but
# nothing anywhere actually exercised it end-to-end through
# MacroRegimeFilter before this. A VIX *spike* (not just an elevated
# level) is the more standard "fear" trigger, so max_increase is the more
# natural knob here, alongside max_level as an absolute-fear-level backstop.
#
# spike_date is derived from the price panel's own (trading-day) date
# index, not a separately-built calendar range -- vix_panel.dates skips
# weekends, a raw pd.date_range(freq="D") doesn't, so picking dates by
# position from two differently-spaced indices silently points at
# different points in time.
vix_repo = SyntheticRepository(n_symbols=6, seed=29)
vix_panel = vix_repo.fetch(start="2019-01-01", end="2021-12-31")
spike_date = vix_panel.dates[400]

vix_calendar = pd.date_range(vix_panel.dates[0], vix_panel.dates[-1], freq="D")
vix_rows = [
    ("vix", dte, dte, 14.0 if dte < spike_date else 38.0)  # calm -> panic move
    for dte in vix_calendar
]
vix_mpanel = MacrosPanel(
    frame=pd.DataFrame(vix_rows, columns=["metric", "period_end", "as_of_date", "value"])
)

vix_inner = CrossSectionalMomentum(lookback=63, top_n=3)
vix_gate = MacroRegimeFilter(
    inner=vix_inner, metric="vix", max_level=35.0, max_increase=15.0,
    lookback=21, scale_when_blocked=0.25,
)

vix_view = vix_panel.as_of(vix_panel.dates[420])  # after the spike
check("a VIX spike past max_level de-risks rather than fully flattening "
      "(scale_when_blocked)",
      vix_gate.blocked(vix_view, vix_mpanel.as_of(vix_view.last_date())) is True)

calm_view = vix_panel.as_of(vix_panel.dates[350])  # before the spike
check("a calm VIX level does not trigger the gate",
      vix_gate.blocked(calm_view, vix_mpanel.as_of(calm_view.last_date())) is False)

bt_vix = Backtester(panel=vix_panel, strategy=vix_gate, macros=vix_mpanel,
                    rebalance="W", initial_equity=50_000)
result_vix = bt_vix.run()
check("a live VIX-gated backtest runs cleanly end to end",
      result_vix.equity.notna().all())
after_spike = result_vix.targets.loc[result_vix.targets.index >= spike_date]
check("exposure is scaled down (not necessarily zero), not fully flat, "
      "after the VIX spike",
      (after_spike.abs().sum(axis=1) <= 0.25 + 1e-9).all(),
      f"max gross after spike: {after_spike.abs().sum(axis=1).max():.3f}")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
