"""Validate OptionsMeanReversion's scoring, gating, and look-ahead safety."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, OptionsMeanReversion, OptionsPanel, RiskGate,
    SyntheticRepository,
)

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 72)
print("1. Construction and validation")
print("=" * 72)

strat = OptionsMeanReversion()
check("auto-names itself", strat.name == "optrevert_20_5")
check("default min_history = iv_window + 2", strat.min_history == 22)

try:
    OptionsMeanReversion(iv_weight=0.0, pcr_weight=0.0)
    check("rejects both weights zero", False)
except ValueError:
    check("rejects both weights zero", True)

OptionsMeanReversion(iv_weight=1.0, pcr_weight=0.0)  # should not raise
check("allows a single-factor configuration", True)

try:
    OptionsMeanReversion(top_n=0)
    check("rejects top_n=0", False)
except ValueError:
    check("rejects top_n=0", True)

print()
print("=" * 72)
print("2. score() -- z-scored options stress, per symbol against its own history")
print("=" * 72)

repo = SyntheticRepository(n_symbols=3, seed=11)
panel = repo.fetch(start="2020-01-01", end="2020-06-30")
syms = panel.symbols
spike_date = panel.dates[80]

rows = []
for d in panel.dates[:100]:
    for sym in syms:
        iv, pcr = 0.25, 0.9
        if sym == syms[0] and d == spike_date:
            iv, pcr = 0.60, 2.5
        rows.append((sym, "iv_atm_near", d, d, iv))
        rows.append((sym, "put_call_volume_ratio", d, d, pcr))
opanel = OptionsPanel(
    frame=pd.DataFrame(rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)

s = OptionsMeanReversion(iv_window=20, top_n=1, min_z=1.0)

check("score returns empty when options is None", s.score(panel.as_of(spike_date), None).empty)

empty_opanel = OptionsPanel(frame=opanel.frame.iloc[0:0])
check("score returns empty when options panel is empty", s.score(panel.as_of(spike_date), empty_opanel).empty)

z = s.score(panel.as_of(spike_date), opanel.as_of(spike_date))
check("only the spiked symbol gets a finite score (constant series -> NaN, dropped)", list(z.index) == [syms[0]])
check("spiked symbol's z-score is strongly positive", z[syms[0]] > 3.0, f"z={z[syms[0]]:.2f}")

before = panel.dates[79]
z_before = s.score(panel.as_of(before), opanel.as_of(before))
check("no stress the day before the spike -> no finite scores", z_before.empty)

print()
print("=" * 72)
print("3. target_weights() -- gating and sizing")
print("=" * 72)

w_spike = s.target_weights(panel.as_of(spike_date), None, None, None, opanel.as_of(spike_date))
check("buys the stressed name at full gross on the spike date", w_spike[syms[0]] == 1.0)
check("no other names held", (w_spike.drop(syms[0]) == 0).all())

w_before = s.target_weights(panel.as_of(before), None, None, None, opanel.as_of(before))
check("holds nothing before any stress exists", (w_before == 0).all())

mild = OptionsMeanReversion(iv_window=20, top_n=1, min_z=10.0)  # deliberately unreachable
w_mild = mild.target_weights(panel.as_of(spike_date), None, None, None, opanel.as_of(spike_date))
check("min_z gate blocks a spike below threshold", (w_mild == 0).all())

# top_n cap: three symbols all spike together, only the configured top_n get bought.
rows3 = []
d3 = panel.dates[80]
for d in panel.dates[:100]:
    for sym in syms:
        iv = 0.60 if d == d3 else 0.25
        pcr = 2.5 if d == d3 else 0.9
        rows3.append((sym, "iv_atm_near", d, d, iv))
        rows3.append((sym, "put_call_volume_ratio", d, d, pcr))
opanel3 = OptionsPanel(
    frame=pd.DataFrame(rows3, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)
s_top2 = OptionsMeanReversion(iv_window=20, top_n=2, min_z=1.0)
w_top2 = s_top2.target_weights(panel.as_of(d3), None, None, None, opanel3.as_of(d3))
check("respects top_n cap when more names qualify than top_n", (w_top2 != 0).sum() == 2)
check("equal-weights the picks", np.isclose(w_top2[w_top2 != 0].to_numpy(), 0.5).all())

print()
print("=" * 72)
print("4. Wired into Backtester -- runs cleanly and never leaks a future spike")
print("=" * 72)

repo2 = SyntheticRepository(n_symbols=6, seed=21)
panel2 = repo2.fetch(start="2020-01-01", end="2021-12-31")
syms2 = panel2.symbols

rng = np.random.default_rng(0)
rows2 = []
spike_dates = {}
for sym in syms2:
    spike_dates[sym] = panel2.dates[rng.integers(60, len(panel2.dates) - 1)]
for d in panel2.dates:
    for sym in syms2:
        iv, pcr = 0.25, 0.9
        if d == spike_dates[sym]:
            iv, pcr = 0.60, 2.5
        rows2.append((sym, "iv_atm_near", d, d, iv))
        rows2.append((sym, "put_call_volume_ratio", d, d, pcr))
opanel2 = OptionsPanel(
    frame=pd.DataFrame(rows2, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)

strat2 = OptionsMeanReversion(iv_window=20, top_n=3, min_z=1.0)
bt = Backtester(
    panel=panel2, strategy=strat2, options=opanel2,
    risk_gate=RiskGate(target_vol=0.15), rebalance="D", initial_equity=50_000,
)
result = bt.run()
check("backtest completes without error", result.equity.notna().all())
check("strategy actually traded at least once", len(result.trades) > 0, f"{len(result.trades)} trades")

# Every day a name is bought, the spike that triggered it must have been on
# or before that decision date -- never a future one.
violations = 0
for date, row in result.targets.iterrows():
    held = row[row != 0].index
    for sym in held:
        if spike_dates[sym] > date:
            violations += 1
check("never buys a name before its own spike has happened", violations == 0, f"{violations} violation(s)")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
