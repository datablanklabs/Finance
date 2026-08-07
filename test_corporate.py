"""Validate CorpsPanel's point-in-time firewall and indicator derivation. No network."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, FundamentalsPanel, LiveSignalRunner, MacrosPanel,
    PortfolioState, SyntheticRepository,
)
from qbt.corporate import CorpsPanel, CorpsRepository, _trailing_count, _trailing_sum

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 72)
print("1. Construction and validation")
print("=" * 72)

rows = [
    # symbol, metric,                    period_end,   as_of_date,   value
    ("AAA", "filed_8k_count_90d", "2023-01-10", "2023-01-10", 1.0),
    ("AAA", "filed_8k_count_90d", "2023-02-15", "2023-02-15", 2.0),
    ("AAA", "insider_net_shares_90d", "2023-01-20", "2023-01-20", -500.0),
    ("BBB", "filed_8k_count_90d", "2023-01-05", "2023-01-05", 1.0),
]
frame = pd.DataFrame(rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
frame["period_end"] = pd.to_datetime(frame["period_end"])
frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])

panel = CorpsPanel(frame=frame)
print(panel.describe())
check("symbols", panel.symbols == ["AAA", "BBB"])
check("metrics", panel.metrics == ["filed_8k_count_90d", "insider_net_shares_90d"])
check("len", len(panel) == len(rows))

try:
    CorpsPanel(frame=frame.drop(columns=["as_of_date"]))
    check("rejects missing as_of_date column", False)
except ValueError:
    check("rejects missing as_of_date column", True)

try:
    bad = frame.copy()
    bad["as_of_date"] = bad["as_of_date"].astype(str)
    CorpsPanel(frame=bad)
    check("rejects non-datetime as_of_date", False)
except TypeError:
    check("rejects non-datetime as_of_date", True)

empty = CorpsPanel(frame=frame.iloc[0:0])
check("describe handles an empty panel", empty.describe().startswith("CorpsPanel(0"))

print()
print("=" * 72)
print("2. as_of -- the look-ahead firewall (truncates to a CorpsPanel)")
print("=" * 72)

trunc = panel.as_of("2023-01-15")
check("as_of returns a CorpsPanel", isinstance(trunc, CorpsPanel))
check(
    "as_of drops every event filed after the cutoff",
    (trunc.frame["as_of_date"] <= pd.Timestamp("2023-01-15")).all(),
)
check("as_of keeps events filed on/before the cutoff", len(trunc) == 2)
check(
    "as_of on an already-truncated panel is idempotent",
    trunc.as_of("2023-01-15").frame.equals(trunc.frame),
)

print()
print("=" * 72)
print("3. snapshot -- flat latest-known-value view")
print("=" * 72)

before_any = panel.snapshot("2023-01-01")
check("nothing known before first event", before_any.empty)

mid = panel.snapshot("2023-01-31")
check("sees AAA's first 8-K count after it happened", mid.loc["AAA", "filed_8k_count_90d"] == 1.0)
check("sees AAA's insider activity after its filing date", mid.loc["AAA", "insider_net_shares_90d"] == -500.0)

late = panel.snapshot("2023-12-31")
check("latest known value wins, not the first", late.loc["AAA", "filed_8k_count_90d"] == 2.0)
check(
    "symbols with no data for a metric don't contaminate others",
    "insider_net_shares_90d" not in late.columns or pd.isna(late.loc["BBB"].get("insider_net_shares_90d", np.nan)),
)

print()
print("=" * 72)
print("4. to_daily -- forward-fill for offline research")
print("=" * 72)

dates = pd.bdate_range("2023-01-01", "2023-03-01")
daily = panel.to_daily(dates, metrics=["filed_8k_count_90d"])
eightk = daily["filed_8k_count_90d"]

check("one column per symbol", list(eightk.columns) == ["AAA", "BBB"])
check(
    "forward-filled after AAA's first 8-K (2023-01-10)",
    eightk.loc["2023-01-11", "AAA"] == 1.0,
)
check(
    "steps up exactly on AAA's second 8-K, not before",
    eightk.loc["2023-02-14", "AAA"] == 1.0 and eightk.loc["2023-02-15", "AAA"] == 2.0,
)

# wide.reindex(columns=[s for s in names if s in wide.columns]) used to
# filter the column list down to what already existed *before*
# reindexing, so reindex could only reorder/subset -- never add -- a
# requested symbol with zero rows for this metric. A brand-new IPO or a
# symbol added mid-backtest got silently dropped from the output entirely
# instead of coming back as an all-NaN column.
daily_missing = panel.to_daily(dates, symbols=["AAA", "NEWIPO"],
                               metrics=["filed_8k_count_90d"])
eightk_missing = daily_missing["filed_8k_count_90d"]
check("a requested symbol with zero rows is still a column, not silently dropped",
      "NEWIPO" in eightk_missing.columns, eightk_missing.columns.tolist())
check("that column is all-NaN, not fabricated data",
      eightk_missing["NEWIPO"].isna().all())

print()
print("=" * 72)
print("5. Trailing-window derivation logic (offline, no network)")
print("=" * 72)

events = pd.to_datetime(
    ["2023-01-01", "2023-01-30", "2023-02-15", "2023-06-01"]
)
counts = _trailing_count(pd.Series(events), window_days=90)
check(
    "trailing count at day 1 is 1 (just itself)",
    counts[0] == 1.0,
)
check(
    "trailing count at day 2 includes both events within 90d",
    counts[1] == 2.0,
)
check(
    "trailing count at day 3 includes all three within 90d",
    counts[2] == 3.0,
)
check(
    "trailing count resets once earlier events fall outside the window",
    counts[3] == 1.0,  # 2023-06-01 is >90d after the first three
)

values = pd.Series([100.0, -50.0, 200.0, -10.0])
sums = _trailing_sum(pd.Series(events), values, window_days=90)
check("trailing sum accumulates within the window", sums[2] == 250.0)  # 100-50+200
check("trailing sum drops out-of-window events", sums[3] == -10.0)

# pandas' own default for an offset-based rolling window is closed="right"
# -- the half-open interval (t - window_days, t], which excludes an event
# landing *exactly* window_days before another one. "Trailing N days"
# reads as inclusive of both ends in the ordinary sense, so an
# exact-boundary hit shouldn't be the one case silently dropped.
boundary_events = pd.to_datetime(["2024-01-01", "2024-03-31"])  # exactly 90 days apart
boundary_counts = _trailing_count(pd.Series(boundary_events), window_days=90)
check("an event exactly window_days before another is included, not "
      "excluded by a half-open boundary",
      list(boundary_counts) == [1.0, 2.0], list(boundary_counts))

repo = CorpsRepository(window_days=90)

filings = pd.DataFrame(
    {
        "as_of_date": pd.to_datetime(
            ["2023-01-01", "2023-01-30", "2023-08-01", "2023-08-15"]
        ),
        "report_type": ["8-K", "8-K", "10-Q", "8-K"],
    }
)
derived = repo._derive_filing_indicators("AAA", filings)
check(
    "derives one metric row per form type per event",
    set(derived["metric"]) == {"filed_8k_count_90d", "filed_10q_count_90d"},
)
eightk_derived = derived[derived["metric"] == "filed_8k_count_90d"].sort_values("as_of_date")
check(
    "8-K trailing count accumulates then resets across the gap",
    list(eightk_derived["value"]) == [1.0, 2.0, 1.0],
)

insider = pd.DataFrame(
    {
        "as_of_date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
        "acquisition_or_disposition": ["Acquisition", "Disposition", "Disposition"],
        "securities_transacted": [1000.0, 300.0, 200.0],
    }
)
derived_insider = repo._derive_insider_indicators("AAA", insider)
net = derived_insider[derived_insider["metric"] == "insider_net_shares_90d"].sort_values("as_of_date")
check(
    "net shares = signed cumulative sum (buy positive, sell negative)",
    list(net["value"]) == [1000.0, 700.0, 500.0],
)
buy_count = derived_insider[derived_insider["metric"] == "insider_buy_count_90d"]
sell_count = derived_insider[derived_insider["metric"] == "insider_sell_count_90d"]
check("buy count only counts acquisitions", list(buy_count["value"]) == [1.0])
check("sell count only counts dispositions", list(sell_count["value"]) == [1.0, 2.0])

print()
print("=" * 72)
print("6. Wired into Backtester and LiveSignalRunner (with fundamentals/macros too)")
print("=" * 72)

price_repo = SyntheticRepository(n_symbols=3, seed=7)
price_panel = price_repo.fetch(start="2020-01-01", end="2021-06-30")
syms = price_panel.symbols

corps_rows = []
d = pd.Timestamp("2020-02-01")
while d < pd.Timestamp("2021-06-01"):
    for sym in syms:
        corps_rows.append((sym, "filed_8k_count_90d", d, d, 1.0))
    d += pd.Timedelta(days=45)
cframe = pd.DataFrame(
    corps_rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"]
)
cpanel = CorpsPanel(frame=cframe)

fund_rows = [
    (sym, "income_revenue", pd.Timestamp("2020-03-31"), pd.Timestamp("2020-05-05"), 100.0)
    for sym in syms
]
fpanel = FundamentalsPanel(
    frame=pd.DataFrame(fund_rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)
mpanel = MacrosPanel(
    frame=pd.DataFrame(
        [("fed_funds_rate", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), 1.0)],
        columns=["metric", "period_end", "as_of_date", "value"],
    )
)

seen_violations = []
seen_calls = {"n": 0}
seen_all_three = {"n": 0}


class ProbeStrategy:
    name = "probe"

    @property
    def min_history(self):
        return 30

    def target_weights(self, view, fundamentals=None, macros=None, corps=None, options=None):
        seen_calls["n"] += 1
        decision_date = view.last_date()
        if corps is not None and len(corps.frame):
            if (corps.frame["as_of_date"] > decision_date).any():
                seen_violations.append(decision_date)
        if fundamentals is not None and macros is not None and corps is not None:
            seen_all_three["n"] += 1
        return pd.Series(1.0 / len(view.symbols), index=view.symbols)


bt = Backtester(
    panel=price_panel,
    strategy=ProbeStrategy(),
    fundamentals=fpanel,
    macros=mpanel,
    corps=cpanel,
    rebalance="M",
    initial_equity=50_000,
)
bt.run()
check("backtester actually calls the strategy", seen_calls["n"] > 0, f"{seen_calls['n']} calls")
check(
    "backtester never leaks a future corp event into the strategy",
    not seen_violations, f"{len(seen_violations)} violation(s)",
)
check(
    "backtester hands fundamentals, macros, and corps to the same call",
    seen_all_three["n"] == seen_calls["n"],
)

runner = LiveSignalRunner(strategy=ProbeStrategy())
state = PortfolioState(cash=10_000.0, shares=pd.Series(0.0, index=syms))

seen_calls["n"] = 0
seen_violations.clear()
plan = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"), corps=cpanel)
check("live runner calls the strategy", seen_calls["n"] == 1)
check("live runner never leaks a future corp event", not seen_violations)

seen_calls["n"] = 0
plan2 = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"))
check(
    "live runner still works with no corps argument at all",
    seen_calls["n"] == 1 and plan2.asof == plan.asof,
)

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
