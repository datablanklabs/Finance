"""Validate FundamentalsPanel's point-in-time firewall. No network."""

import numpy as np
import pandas as pd

from qbt import Backtester, LiveSignalRunner, PortfolioState, SyntheticRepository
from qbt.fundamentals import FundamentalsPanel

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
    # symbol, metric,              period_end,   as_of_date,   value
    ("AAA", "income_revenue", "2023-03-31", "2023-05-05", 100.0),
    ("AAA", "income_revenue", "2023-06-30", "2023-08-04", 110.0),
    ("AAA", "income_revenue", "2023-09-30", "2023-11-03", 105.0),
    ("AAA", "ratios_pe_ratio", "2023-03-31", "2023-05-05", 18.0),
    ("AAA", "ratios_pe_ratio", "2023-06-30", "2023-08-04", 21.0),
    ("BBB", "income_revenue", "2023-03-31", "2023-05-12", 40.0),
    ("BBB", "income_revenue", "2023-06-30", "2023-08-11", 44.0),
]
frame = pd.DataFrame(
    rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"]
)
frame["period_end"] = pd.to_datetime(frame["period_end"])
frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])

panel = FundamentalsPanel(frame=frame)
print(panel.describe())
check("symbols", panel.symbols == ["AAA", "BBB"])
check("metrics", panel.metrics == ["income_revenue", "ratios_pe_ratio"])
check("len", len(panel) == len(rows))

try:
    FundamentalsPanel(frame=frame.drop(columns=["as_of_date"]))
    check("rejects missing as_of_date column", False)
except ValueError:
    check("rejects missing as_of_date column", True)

try:
    bad = frame.copy()
    bad["as_of_date"] = bad["as_of_date"].astype(str)
    FundamentalsPanel(frame=bad)
    check("rejects non-datetime as_of_date", False)
except TypeError:
    check("rejects non-datetime as_of_date", True)

# describe() used to call .min()/.max() on as_of_date unconditionally, the
# one panel of the four missing the empty-frame guard MacrosPanel/
# CorpsPanel/OptionsPanel all have -- inconsistent, and fragile beyond the
# specific pandas version where pd.NaT.date() happens not to raise.
empty = FundamentalsPanel(frame=frame.iloc[0:0])
empty_desc = empty.describe()
check("describe handles an empty panel", empty_desc.startswith("FundamentalsPanel(0"))
# The weaker startswith check above passes either way on this pandas
# version (pd.NaT.date() doesn't raise here) -- confirm the explicit
# empty-frame guard actually fired, not the min()/max() fallback path
# quietly degrading to "NaT to NaT".
check("the empty-frame guard produces a clean message, not a NaT fallback",
      "NaT" not in empty_desc, empty_desc)

print()
print("=" * 72)
print("2. as_of -- the look-ahead firewall (truncates to a FundamentalsPanel)")
print("=" * 72)

trunc = panel.as_of("2023-06-01")
check("as_of returns a FundamentalsPanel", isinstance(trunc, FundamentalsPanel))
check(
    "as_of drops every row filed after the cutoff",
    (trunc.frame["as_of_date"] <= pd.Timestamp("2023-06-01")).all(),
)
check(
    "as_of keeps rows filed on/before the cutoff",
    len(trunc) == 3,  # AAA Q1 revenue+pe, BBB Q1 revenue
)
check(
    "as_of on an already-truncated panel is idempotent",
    trunc.as_of("2023-06-01").frame.equals(trunc.frame),
)

print()
print("=" * 72)
print("3. snapshot -- flat latest-known-value view")
print("=" * 72)

before_any_filing = panel.snapshot("2023-01-01")
check(
    "nothing known before first filing",
    before_any_filing.empty or before_any_filing["income_revenue"].isna().all(),
)

mid = panel.snapshot("2023-06-01")
check(
    "sees Q1 revenue after its filing date",
    mid.loc["AAA", "income_revenue"] == 100.0,
)
check(
    "does not see Q2 revenue before its filing date (2023-08-04)",
    "income_revenue" not in mid.columns or pd.isna(mid.get("income_revenue", {}).get("AAA", np.nan))
    or mid.loc["AAA", "income_revenue"] == 100.0,
)

day_of_filing = panel.snapshot("2023-08-04")
check(
    "snapshot is inclusive of the filing date itself",
    day_of_filing.loc["AAA", "income_revenue"] == 110.0,
)

day_before_filing = panel.snapshot("2023-08-03")
check(
    "one day before filing still shows the prior quarter",
    day_before_filing.loc["AAA", "income_revenue"] == 100.0,
)

late = panel.snapshot("2024-01-01")
check(
    "latest known value wins, not the first",
    late.loc["AAA", "income_revenue"] == 105.0,
)
check(
    "symbols with no data for a metric don't contaminate others",
    "ratios_pe_ratio" not in late.columns or pd.isna(late.loc["BBB"].get("ratios_pe_ratio", np.nan)),
)
check(
    "snapshot on a pre-truncated panel matches snapshot on the full panel",
    panel.as_of("2023-08-04").snapshot("2023-08-04").equals(panel.snapshot("2023-08-04")),
)

print()
print("=" * 72)
print("4. to_daily -- forward-fill for offline research")
print("=" * 72)

dates = pd.bdate_range("2023-04-01", "2023-09-01")
daily = panel.to_daily(dates, metrics=["income_revenue"])
rev = daily["income_revenue"]

check("daily frame has one column per symbol", list(rev.columns) == ["AAA", "BBB"])
check(
    "NaN before the first filing is known",
    rev.loc["2023-04-03", "AAA"] != rev.loc["2023-04-03", "AAA"],  # NaN != NaN
)
check(
    "forward-filled the day after filing",
    rev.loc["2023-05-08", "AAA"] == 100.0,
)
check(
    "steps up exactly on the Q2 filing date, not before",
    rev.loc["2023-08-03", "AAA"] == 100.0 and rev.loc["2023-08-04", "AAA"] == 110.0,
)
check(
    "no look-ahead: forward-filled value never exceeds what as_of would show",
    (rev.loc["2023-08-01":"2023-08-03", "AAA"] == 100.0).all(),
)

# wide.reindex(columns=[s for s in names if s in wide.columns]) used to
# filter the column list down to what already existed *before*
# reindexing, so reindex could only reorder/subset -- never add -- a
# requested symbol with zero rows for this metric. A brand-new IPO or a
# symbol added mid-backtest got silently dropped from the output entirely
# instead of coming back as an all-NaN column, and a caller indexing by
# symbol (wide[symbol]) got a KeyError instead of NaN.
daily_missing = panel.to_daily(dates, symbols=["AAA", "NEWIPO"],
                               metrics=["income_revenue"])
rev_missing = daily_missing["income_revenue"]
check("a requested symbol with zero rows is still a column, not silently dropped",
      "NEWIPO" in rev_missing.columns, rev_missing.columns.tolist())
check("that column is all-NaN, not fabricated data",
      rev_missing["NEWIPO"].isna().all())

print()
print("=" * 72)
print("5. Wired into Backtester and LiveSignalRunner")
print("=" * 72)

repo = SyntheticRepository(n_symbols=3, seed=3)
price_panel = repo.fetch(start="2020-01-01", end="2021-06-30")
syms = price_panel.symbols

filing_rows = []
d = pd.Timestamp("2020-02-01")
while d < pd.Timestamp("2021-06-01"):
    for sym in syms:
        filing_rows.append((sym, "income_revenue", d - pd.Timedelta(days=45), d, 100.0))
    d += pd.Timedelta(days=91)
fframe = pd.DataFrame(
    filing_rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"]
)
fpanel = FundamentalsPanel(frame=fframe)
check(
    "fixture actually spans filings inside the backtest window",
    fpanel.frame["as_of_date"].min() < price_panel.dates[-1]
    and fpanel.frame["as_of_date"].max() > price_panel.dates[len(price_panel) // 2],
)

seen_violations = []
seen_calls = {"n": 0}


class ProbeStrategy:
    """Records whether it was ever handed a filing from its own future."""

    name = "probe"

    @property
    def min_history(self):
        return 30

    def target_weights(self, view, fundamentals=None, macros=None, corps=None, options=None):
        seen_calls["n"] += 1
        decision_date = view.last_date()
        if fundamentals is not None and len(fundamentals.frame):
            if (fundamentals.frame["as_of_date"] > decision_date).any():
                seen_violations.append(decision_date)
        return pd.Series(1.0 / len(view.symbols), index=view.symbols)


bt = Backtester(
    panel=price_panel,
    strategy=ProbeStrategy(),
    fundamentals=fpanel,
    rebalance="M",
    initial_equity=50_000,
)
bt.run()
check(
    "backtester actually calls the strategy",
    seen_calls["n"] > 0, f"{seen_calls['n']} calls",
)
check(
    "backtester never leaks a future filing into the strategy",
    not seen_violations, f"{len(seen_violations)} violation(s)",
)

runner = LiveSignalRunner(strategy=ProbeStrategy())
state = PortfolioState(cash=10_000.0, shares=pd.Series(0.0, index=syms))

seen_calls["n"] = 0
seen_violations.clear()
plan = runner.plan(
    price_panel, state, asof=pd.Timestamp("2020-08-15"), fundamentals=fpanel
)
check("live runner calls the strategy", seen_calls["n"] == 1)
check("live runner never leaks a future filing", not seen_violations)
check(
    "live runner's fundamentals cutoff tracks the requested asof",
    plan.asof <= pd.Timestamp("2020-08-15"),
)

seen_calls["n"] = 0
plan2 = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"))
check(
    "live runner still works with no fundamentals argument at all",
    seen_calls["n"] == 1 and plan2.asof == plan.asof,
)

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
