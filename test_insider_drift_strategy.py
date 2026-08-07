"""Validate InsiderEventDrift's scoring, drift-window exit, and look-ahead safety."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, CorpsPanel, InsiderEventDrift, RiskGate, SyntheticRepository,
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

s = InsiderEventDrift()
check("auto-names itself", s.name == "insiderdrift_10_10")
check("default min_history = drift_days + 2", s.min_history == 12)

try:
    InsiderEventDrift(weighting="bogus")
    check("rejects bad weighting", False)
except ValueError:
    check("rejects bad weighting", True)

try:
    InsiderEventDrift(drift_days=0)
    check("rejects drift_days < 1", False)
except ValueError:
    check("rejects drift_days < 1", True)

try:
    InsiderEventDrift(top_n=0)
    check("rejects top_n=0", False)
except ValueError:
    check("rejects top_n=0", True)

InsiderEventDrift(top_n=None)  # a documented fallback, must not raise
check("top_n=None still constructs fine", True)

print()
print("=" * 72)
print("2. score()/target_weights() -- entry, hold-through-drift, and clean exit")
print("=" * 72)

repo = SyntheticRepository(n_symbols=3, seed=13)
panel = repo.fetch(start="2020-01-01", end="2020-06-30")
syms = panel.symbols
event_date = panel.dates[60]

rows = []
for d in panel.dates[:120]:
    for sym in syms:
        active = sym == syms[0] and d >= event_date
        rows.append((sym, "filed_8k_count_90d", d, d, 1.0 if active else 0.0))
        rows.append((sym, "insider_buy_count_90d", d, d, 1.0 if active else 0.0))
        rows.append((sym, "insider_net_shares_90d", d, d, 500.0 if active else 0.0))
frame = pd.DataFrame(rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
cpanel = CorpsPanel(frame=frame)

strat = InsiderEventDrift(drift_days=10, top_n=5)


def w_at(date):
    return strat.target_weights(panel.as_of(date), None, None, cpanel.as_of(date), None)


check("no position the day before the event", w_at(panel.dates[59])[syms[0]] == 0.0)
check("enters at full weight on the event day", w_at(event_date)[syms[0]] == 1.0)
check("still held mid-drift-window", w_at(panel.dates[65])[syms[0]] == 1.0)
check(
    "exits exactly once the drift window elapses",
    w_at(panel.dates[71])[syms[0]] == 0.0,
)
check(
    "stays flat long after, even though trailing counts are still elevated "
    "(the exit is real, not a fluke of window edges)",
    w_at(panel.dates[110])[syms[0]] == 0.0,
)
check("other symbols never triggered", (w_at(event_date).drop(syms[0]) == 0).all())

print()
print("=" * 72)
print("3. Signal quality gates")
print("=" * 72)

# require_8k=True (default): insider buying alone, with no 8-K, shouldn't qualify.
rows_no_8k = []
for d in panel.dates[:120]:
    for sym in syms:
        active = sym == syms[0] and d >= event_date
        rows_no_8k.append((sym, "insider_buy_count_90d", d, d, 1.0 if active else 0.0))
        rows_no_8k.append((sym, "insider_net_shares_90d", d, d, 500.0 if active else 0.0))
cpanel_no_8k = CorpsPanel(
    frame=pd.DataFrame(rows_no_8k, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)
strict = InsiderEventDrift(drift_days=10, top_n=5, require_8k=True)
loose = InsiderEventDrift(drift_days=10, top_n=5, require_8k=False)

check(
    "require_8k=True blocks insider buying with no accompanying 8-K",
    strict.target_weights(panel.as_of(event_date), None, None, cpanel_no_8k.as_of(event_date), None)[syms[0]] == 0.0,
)
check(
    "require_8k=False allows insider buying alone to qualify",
    loose.target_weights(panel.as_of(event_date), None, None, cpanel_no_8k.as_of(event_date), None)[syms[0]] == 1.0,
)

# Net insider *selling* shouldn't qualify even with a fresh 8-K + buy transaction.
rows_selling = []
for d in panel.dates[:120]:
    for sym in syms:
        active = sym == syms[0] and d >= event_date
        rows_selling.append((sym, "filed_8k_count_90d", d, d, 1.0 if active else 0.0))
        rows_selling.append((sym, "insider_buy_count_90d", d, d, 1.0 if active else 0.0))
        rows_selling.append((sym, "insider_net_shares_90d", d, d, -500.0 if active else 0.0))
cpanel_selling = CorpsPanel(
    frame=pd.DataFrame(rows_selling, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)
check(
    "net insider selling disqualifies even with a buy transaction + 8-K present",
    strat.target_weights(panel.as_of(event_date), None, None, cpanel_selling.as_of(event_date), None)[syms[0]] == 0.0,
)

# window_days mismatch: metric names won't match, should silently score empty rather than crash.
mismatched = InsiderEventDrift(drift_days=10, window_days=30)
w_mismatched = mismatched.target_weights(panel.as_of(event_date), None, None, cpanel.as_of(event_date), None)
check(
    "window_days mismatch degrades to no signal rather than raising",
    (w_mismatched == 0).all(),
)

check("score returns empty when corps is None", strat.score(panel.as_of(event_date), None).empty)

print()
print("=" * 72)
print("4. Wired into Backtester -- runs cleanly and never buys before its own event")
print("=" * 72)

repo2 = SyntheticRepository(n_symbols=6, seed=23)
panel2 = repo2.fetch(start="2020-01-01", end="2021-12-31")
syms2 = panel2.symbols

rng = np.random.default_rng(1)
event_dates = {sym: panel2.dates[rng.integers(60, len(panel2.dates) - 30)] for sym in syms2}

rows2 = []
for d in panel2.dates:
    for sym in syms2:
        active = d >= event_dates[sym]
        rows2.append((sym, "filed_8k_count_90d", d, d, 1.0 if active else 0.0))
        rows2.append((sym, "insider_buy_count_90d", d, d, 1.0 if active else 0.0))
        rows2.append((sym, "insider_net_shares_90d", d, d, 500.0 if active else 0.0))
cpanel2 = CorpsPanel(
    frame=pd.DataFrame(rows2, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)

strat2 = InsiderEventDrift(drift_days=10, top_n=3)
bt = Backtester(
    panel=panel2, strategy=strat2, corps=cpanel2,
    risk_gate=RiskGate(target_vol=0.15), rebalance="D", initial_equity=50_000,
)
result = bt.run()
check("backtest completes without error", result.equity.notna().all())
check("strategy actually traded at least once", len(result.trades) > 0, f"{len(result.trades)} trades")

violations = 0
holds_ever_fire = 0
for date, row in result.targets.iterrows():
    held = row[row != 0].index
    for sym in held:
        holds_ever_fire += 1
        if event_dates[sym] > date:
            violations += 1
check("never buys a name before its own event has happened", violations == 0, f"{violations} violation(s)")
check("positions actually fired at least once", holds_ever_fire > 0, f"{holds_ever_fire} name-days held")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
