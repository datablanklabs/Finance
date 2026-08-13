"""Validate MacrosPanel's point-in-time firewall. No network."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, EqualWeightBuyHold, FundamentalsPanel, LiveSignalRunner,
    MacroRegimeFilter, PortfolioState, PricePanel, SyntheticRepository,
)
from qbt.macro import MacrosPanel, MacrosRepository

FAILS = []


def _raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return True
    return False


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 72)
print("1. Construction and validation")
print("=" * 72)

rows = [
    # metric,               period_end,   as_of_date,   value
    ("fed_funds_rate", "2023-01-01", "2023-02-03", 4.50),
    ("fed_funds_rate", "2023-02-01", "2023-03-03", 4.75),
    ("fed_funds_rate", "2023-03-01", "2023-04-03", 5.00),
    ("cpi", "2023-01-01", "2023-02-14", 300.0),
    ("cpi", "2023-02-01", "2023-03-14", 301.5),
]
frame = pd.DataFrame(rows, columns=["metric", "period_end", "as_of_date", "value"])
frame["period_end"] = pd.to_datetime(frame["period_end"])
frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])

panel = MacrosPanel(frame=frame)
print(panel.describe())
check("metrics", panel.metrics == ["cpi", "fed_funds_rate"])
check("len", len(panel) == len(rows))
check("no symbol column required", "symbol" not in panel.frame.columns)

try:
    MacrosPanel(frame=frame.drop(columns=["as_of_date"]))
    check("rejects missing as_of_date column", False)
except ValueError:
    check("rejects missing as_of_date column", True)

try:
    bad = frame.copy()
    bad["as_of_date"] = bad["as_of_date"].astype(str)
    MacrosPanel(frame=bad)
    check("rejects non-datetime as_of_date", False)
except TypeError:
    check("rejects non-datetime as_of_date", True)

empty = MacrosPanel(frame=frame.iloc[0:0])
check("describe handles an empty panel", empty.describe().startswith("MacrosPanel(0"))

print()
print("=" * 72)
print("2. as_of -- the look-ahead firewall (truncates to a MacrosPanel)")
print("=" * 72)

trunc = panel.as_of("2023-03-01")
check("as_of returns a MacrosPanel", isinstance(trunc, MacrosPanel))
check(
    "as_of drops every reading released after the cutoff",
    (trunc.frame["as_of_date"] <= pd.Timestamp("2023-03-01")).all(),
)
check(
    "as_of keeps readings released on/before the cutoff",
    len(trunc) == 2,  # Jan fed funds (rel. 2/3), Jan cpi (rel. 2/14)
)
check(
    "as_of on an already-truncated panel is idempotent",
    trunc.as_of("2023-03-01").frame.equals(trunc.frame),
)

print()
print("=" * 72)
print("3. snapshot -- flat latest-known-value view (a Series, no symbol axis)")
print("=" * 72)

before_any = panel.snapshot("2023-01-01")
check("nothing known before first release", before_any.empty)

mid = panel.snapshot("2023-02-20")
check("snapshot is a Series indexed by metric", isinstance(mid, pd.Series))
check("sees January fed funds after its release date", mid["fed_funds_rate"] == 4.50)
check("sees January cpi after its release date", mid["cpi"] == 300.0)
check(
    "does not yet see February fed funds (released 2023-03-03)",
    "fed_funds_rate" not in mid.index or mid["fed_funds_rate"] == 4.50,
)

late = panel.snapshot("2023-12-31")
check("latest known value wins, not the first", late["fed_funds_rate"] == 5.00)
check(
    "snapshot on a pre-truncated panel matches snapshot on the full panel",
    panel.as_of("2023-03-05").snapshot("2023-03-05").equals(panel.snapshot("2023-03-05")),
)

print()
print("=" * 72)
print("4. to_daily -- forward-fill for offline research")
print("=" * 72)

dates = pd.bdate_range("2023-01-15", "2023-04-15")
daily = panel.to_daily(dates, metrics=["fed_funds_rate"])
ffr = daily["fed_funds_rate"]

check("one column per requested metric", list(daily.columns) == ["fed_funds_rate"])
check(
    "NaN before the first release is known",
    ffr.loc["2023-01-17"] != ffr.loc["2023-01-17"],  # NaN != NaN
)
check(
    "forward-filled after the January release (2023-02-03)",
    ffr.loc["2023-02-06"] == 4.50,
)
check(
    "steps up exactly on the February release date, not before",
    ffr.loc["2023-03-02"] == 4.50 and ffr.loc["2023-03-03"] == 4.75,
)

print()
print("=" * 72)
print("5. MacrosRepository -- offline schema-parsing check (no network/key)")
print("=" * 72)


class _FakeResult:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


def _fake_fred_series(symbol, start_date, end_date, provider):
    # Mirrors the real openbb-fred output shape confirmed from its source
    # (openbb_fred/models/series.py): a 'date' column plus one column
    # named exactly after the series ID.
    idx = pd.date_range("2023-01-01", periods=3, freq="MS")
    return _FakeResult(pd.DataFrame({symbol: [1.0, 2.0, 3.0]}, index=idx).rename_axis("date"))


class _FakeEconomy:
    fred_series = staticmethod(_fake_fred_series)


class _FakeOBB:
    economy = _FakeEconomy()


import sys
import types

repo = MacrosRepository(
    indicators={"fed_funds_rate": ("FEDFUNDS", 33)}, cache_dir=None
)

fake_openbb = types.ModuleType("openbb")
fake_openbb.obb = _FakeOBB()
sys.modules["openbb"] = fake_openbb
try:
    long = repo._fetch_remote("fed_funds_rate", "FEDFUNDS", 33, "2023-01-01", "2023-06-01")
finally:
    del sys.modules["openbb"]

check("parses the series-ID-named value column", list(long["value"]) == [1.0, 2.0, 3.0])
check(
    "as_of_date = period_end + lag_days",
    (long["as_of_date"] - long["period_end"]).dt.days.eq(33).all(),
)
check("metric column is the friendly name, not the FRED ID", (long["metric"] == "fed_funds_rate").all())
check("period_end preserved from the FRED date column", long["period_end"].iloc[0] == pd.Timestamp("2023-01-01"))

print()
print("=" * 72)
print("6. Wired into Backtester and LiveSignalRunner (with fundamentals too)")
print("=" * 72)

price_repo = SyntheticRepository(n_symbols=3, seed=5)
price_panel = price_repo.fetch(start="2020-01-01", end="2021-06-30")
syms = price_panel.symbols

macro_rows = []
d = pd.Timestamp("2020-02-01")
while d < pd.Timestamp("2021-06-01"):
    macro_rows.append(("fed_funds_rate", d - pd.Timedelta(days=30), d, 1.0))
    d += pd.Timedelta(days=30)
mframe = pd.DataFrame(
    macro_rows, columns=["metric", "period_end", "as_of_date", "value"]
)
mpanel = MacrosPanel(frame=mframe)

fund_rows = [(sym, "income_revenue", pd.Timestamp("2020-03-31"), pd.Timestamp("2020-05-05"), 100.0) for sym in syms]
fframe = pd.DataFrame(
    fund_rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"]
)
fpanel = FundamentalsPanel(frame=fframe)

seen_violations = []
seen_calls = {"n": 0}
seen_both = {"n": 0}


class ProbeStrategy:
    name = "probe"

    @property
    def min_history(self):
        return 30

    def target_weights(self, view, fundamentals=None, macros=None, corps=None, options=None):
        seen_calls["n"] += 1
        decision_date = view.last_date()
        if macros is not None and len(macros.frame):
            if (macros.frame["as_of_date"] > decision_date).any():
                seen_violations.append(decision_date)
        if fundamentals is not None and macros is not None:
            seen_both["n"] += 1
        return pd.Series(1.0 / len(view.symbols), index=view.symbols)


bt = Backtester(
    panel=price_panel,
    strategy=ProbeStrategy(),
    fundamentals=fpanel,
    macros=mpanel,
    rebalance="M",
    initial_equity=50_000,
)
bt.run()
check("backtester actually calls the strategy", seen_calls["n"] > 0, f"{seen_calls['n']} calls")
check(
    "backtester never leaks a future macro reading into the strategy",
    not seen_violations, f"{len(seen_violations)} violation(s)",
)
check(
    "backtester hands both fundamentals and macros to the same call",
    seen_both["n"] == seen_calls["n"],
)

runner = LiveSignalRunner(strategy=ProbeStrategy())
state = PortfolioState(cash=10_000.0, shares=pd.Series(0.0, index=syms))

seen_calls["n"] = 0
seen_violations.clear()
plan = runner.plan(
    price_panel, state, asof=pd.Timestamp("2020-08-15"), macros=mpanel
)
check("live runner calls the strategy", seen_calls["n"] == 1)
check("live runner never leaks a future macro reading", not seen_violations)

seen_calls["n"] = 0
plan2 = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"))
check(
    "live runner still works with no macros argument at all",
    seen_calls["n"] == 1 and plan2.asof == plan.asof,
)

print()
print("=" * 72)
print("Staleness: a reading that stopped updating is not a current reading")
print("=" * 72)

# snapshot() returns the newest reading at *any* age. That's right for a
# monthly series (a five-week-old CPI print is simply the newest one that
# exists) and wrong for a daily one acted on as a current regime read: a
# dead API key, a provider outage or a frozen cache leaves VIX answering
# with its last value indefinitely, and nothing downstream can tell.
_stale_dates = pd.date_range("2026-01-01", periods=3)
_stale_panel = MacrosPanel(frame=pd.DataFrame(
    [("vix", d, d, 45.0) for d in _stale_dates],
    columns=["metric", "period_end", "as_of_date", "value"],
))
_late = pd.Timestamp("2026-06-01")

check("without an age bound, a five-month-old reading is still returned",
      float(_stale_panel.snapshot(_late)["vix"]) == 45.0)
check("snapshot_age_days reports how old that reading actually is",
      float(_stale_panel.snapshot_age_days(_late)["vix"]) == 149.0,
      _stale_panel.snapshot_age_days(_late).to_dict())
check("max_age_days drops it entirely rather than presenting it as current",
      "vix" not in _stale_panel.snapshot(_late, max_age_days=7).index)
check("a fresh reading inside the age bound is still returned",
      float(_stale_panel.snapshot(_stale_dates[-1], max_age_days=7)["vix"]) == 45.0)

# MacroRegimeFilter must treat an over-age reading as no reading at all --
# the same documented no-op as macros=None, not a block and not a gate on
# stale data.
_stale_idx = pd.bdate_range("2025-06-01", periods=300)
_stale_view = PricePanel(close=pd.DataFrame(
    {"A": np.linspace(100, 110, 300)}, index=_stale_idx))
_bounded = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="vix",
                             max_level=35.0, max_age_days=7)
_unbounded = MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="vix",
                               max_level=35.0)
check("unbounded, the filter gates the book on a five-month-old VIX=45",
      _unbounded.blocked(_stale_view, _stale_panel))
check("bounded by max_age_days, the same stale reading is a no-op",
      not _bounded.blocked(_stale_view, _stale_panel))

_fresh_panel = MacrosPanel(frame=pd.DataFrame(
    [("vix", d, d, 45.0) for d in pd.date_range(_stale_idx[-1], periods=1)],
    columns=["metric", "period_end", "as_of_date", "value"],
))
check("a genuinely current elevated reading still blocks with the bound on",
      _bounded.blocked(_stale_view, _fresh_panel))
check("max_age_days=0 is rejected as a construction error",
      _raises(lambda: MacroRegimeFilter(inner=EqualWeightBuyHold(), metric="vix",
                                        max_level=35.0, max_age_days=0), ValueError))

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
