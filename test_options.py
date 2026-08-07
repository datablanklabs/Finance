"""Validate OptionsPanel's point-in-time firewall and indicator derivation. No network."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, CorpsPanel, FundamentalsPanel, LiveSignalRunner, MacrosPanel,
    PortfolioState, SyntheticRepository,
)
from qbt.options import OptionsPanel, OptionsRepository, derive_indicators

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
    # symbol, metric,        period_end,   as_of_date,   value
    ("AAA", "iv_atm_near", "2023-01-10", "2023-01-10", 0.25),
    ("AAA", "iv_atm_near", "2023-01-11", "2023-01-11", 0.28),
    ("BBB", "iv_atm_near", "2023-01-10", "2023-01-10", 0.40),
]
frame = pd.DataFrame(rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
frame["period_end"] = pd.to_datetime(frame["period_end"])
frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])

panel = OptionsPanel(frame=frame)
print(panel.describe())
check("symbols", panel.symbols == ["AAA", "BBB"])
check("metrics", panel.metrics == ["iv_atm_near"])
check("len", len(panel) == len(rows))

try:
    OptionsPanel(frame=frame.drop(columns=["as_of_date"]))
    check("rejects missing as_of_date column", False)
except ValueError:
    check("rejects missing as_of_date column", True)

try:
    bad = frame.copy()
    bad["as_of_date"] = bad["as_of_date"].astype(str)
    OptionsPanel(frame=bad)
    check("rejects non-datetime as_of_date", False)
except TypeError:
    check("rejects non-datetime as_of_date", True)

empty = OptionsPanel(frame=frame.iloc[0:0])
check("describe handles an empty panel", empty.describe().startswith("OptionsPanel(0"))

print()
print("=" * 72)
print("2. as_of -- the look-ahead firewall (truncates to an OptionsPanel)")
print("=" * 72)

trunc = panel.as_of("2023-01-10")
check("as_of returns an OptionsPanel", isinstance(trunc, OptionsPanel))
check(
    "as_of drops every snapshot taken after the cutoff",
    (trunc.frame["as_of_date"] <= pd.Timestamp("2023-01-10")).all(),
)
check("as_of keeps snapshots taken on/before the cutoff", len(trunc) == 2)
check(
    "as_of on an already-truncated panel is idempotent",
    trunc.as_of("2023-01-10").frame.equals(trunc.frame),
)

print()
print("=" * 72)
print("3. snapshot -- flat latest-known-value view")
print("=" * 72)

before_any = panel.snapshot("2023-01-09")
check("nothing known before first snapshot", before_any.empty)

day1 = panel.snapshot("2023-01-10")
check("sees the first day's reading", day1.loc["AAA", "iv_atm_near"] == 0.25)
check("BBB unaffected by AAA's later update", day1.loc["BBB", "iv_atm_near"] == 0.40)

day2 = panel.snapshot("2023-01-11")
check("updates to the newer reading the next day", day2.loc["AAA", "iv_atm_near"] == 0.28)

print()
print("=" * 72)
print("4. to_daily -- forward-fill for offline research")
print("=" * 72)

dates = pd.bdate_range("2023-01-05", "2023-01-20")
daily = panel.to_daily(dates, metrics=["iv_atm_near"])
iv = daily["iv_atm_near"]

check("one column per symbol", list(iv.columns) == ["AAA", "BBB"])
check(
    "NaN before the first snapshot",
    iv.loc["2023-01-06", "AAA"] != iv.loc["2023-01-06", "AAA"],
)
check("forward-filled after the first snapshot", iv.loc["2023-01-10", "AAA"] == 0.25)
check(
    "steps to the new reading exactly on the second snapshot date",
    iv.loc["2023-01-11", "AAA"] == 0.28,
)
check(
    "BBB has no second snapshot -- stays flat rather than inventing one",
    (iv.loc["2023-01-10":"2023-01-20", "BBB"] == 0.40).all(),
)

# wide.reindex(columns=[s for s in names if s in wide.columns]) used to
# filter the column list down to what already existed *before*
# reindexing, so reindex could only reorder/subset -- never add -- a
# requested symbol with zero rows for this metric. A brand-new IPO or a
# symbol added mid-backtest got silently dropped from the output entirely
# instead of coming back as an all-NaN column.
daily_missing = panel.to_daily(dates, symbols=["AAA", "NEWIPO"],
                               metrics=["iv_atm_near"])
iv_missing = daily_missing["iv_atm_near"]
check("a requested symbol with zero rows is still a column, not silently dropped",
      "NEWIPO" in iv_missing.columns, iv_missing.columns.tolist())
check("that column is all-NaN, not fabricated data",
      iv_missing["NEWIPO"].isna().all())

print()
print("=" * 72)
print("5. derive_indicators -- pure derivation from a hand-built chain (offline)")
print("=" * 72)

def _leg(expiration, dte, strike, option_type, iv, volume, oi):
    return {
        "underlying_price": 100.0,
        "expiration": expiration,
        "dte": dte,
        "strike": strike,
        "option_type": option_type,
        "implied_volatility": iv,
        "volume": volume,
        "open_interest": oi,
    }

chain = pd.DataFrame(
    [
        # near-term (30 dte): ATM strike 100, plus 10% OTM put/call
        _leg("2023-02-01", 30, 100.0, "call", 0.20, 500, 1000),
        _leg("2023-02-01", 30, 100.0, "put", 0.22, 400, 900),
        _leg("2023-02-01", 30, 90.0, "put", 0.30, 100, 200),
        _leg("2023-02-01", 30, 110.0, "call", 0.18, 300, 600),
        # far-term (90 dte): ATM strike 100
        _leg("2023-04-01", 90, 100.0, "call", 0.24, 50, 300),
        _leg("2023-04-01", 90, 100.0, "put", 0.26, 40, 250),
    ]
)

indicators = derive_indicators(chain, near_dte=30, far_dte=90, otm_pct=0.10)
check(
    "iv_atm_near averages the ATM call+put IV at the near expiration",
    np.isclose(indicators["iv_atm_near"], (0.20 + 0.22) / 2),
)
check(
    "iv_atm_far averages the ATM call+put IV at the far expiration",
    np.isclose(indicators["iv_atm_far"], (0.24 + 0.26) / 2),
)
check(
    "iv_skew = OTM put IV - OTM call IV at the near expiration",
    np.isclose(indicators["iv_skew"], 0.30 - 0.18),
)
check(
    "put_call_volume_ratio uses near-expiration volume only",
    np.isclose(indicators["put_call_volume_ratio"], (400 + 100) / (500 + 300)),
)
check(
    "put_call_oi_ratio uses near-expiration open interest only",
    np.isclose(indicators["put_call_oi_ratio"], (900 + 200) / (1000 + 600)),
)

check("empty chain yields no indicators", derive_indicators(pd.DataFrame()) == {})

no_calls = chain[chain["option_type"] != "call"]
no_call_indicators = derive_indicators(no_calls, near_dte=30, far_dte=90)
check(
    "missing calls doesn't crash or divide by zero",
    "put_call_volume_ratio" not in no_call_indicators,
)

print()
print("=" * 72)
print("6. OptionsRepository -- archive/read-back cycle (offline, mocked chain)")
print("=" * 72)

import sys
import types

class _FakeResult:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


class _FakeDerivatives:
    class options:
        @staticmethod
        def chains(symbol, provider):
            return _FakeResult(chain.assign(underlying_symbol=symbol))


class _FakeOBB:
    derivatives = _FakeDerivatives()


fake_openbb = types.ModuleType("openbb")
fake_openbb.obb = _FakeOBB()
sys.modules["openbb"] = fake_openbb
try:
    repo = OptionsRepository(cache_dir="/tmp/qbt_options_test_archive")
    import shutil
    shutil.rmtree("/tmp/qbt_options_test_archive", ignore_errors=True)

    snap1 = repo.fetch_snapshot(["AAA"], as_of=pd.Timestamp("2023-01-10"))
    check("fetch_snapshot returns an OptionsPanel with derived metrics", len(snap1) > 0)

    snap2 = repo.fetch_snapshot(["AAA"], as_of=pd.Timestamp("2023-01-11"))
    check("second snapshot on a different date also archives", len(snap2) > 0)

    archived = repo.fetch(["AAA"], start="2023-01-01", end="2023-01-31")
    check(
        "fetch reads back both archived snapshots",
        archived.frame["as_of_date"].nunique() == 2,
    )

    narrow = repo.fetch(["AAA"], start="2023-01-01", end="2023-01-10")
    check(
        "fetch respects the date range -- doesn't return the later snapshot",
        (narrow.frame["as_of_date"] <= pd.Timestamp("2023-01-10")).all(),
    )

    unarchived = repo.fetch(["ZZZ"], start="2023-01-01", end="2023-01-31")
    check("fetch on a never-snapshotted symbol returns an empty panel, not an error", len(unarchived) == 0)

    shutil.rmtree("/tmp/qbt_options_test_archive", ignore_errors=True)
finally:
    del sys.modules["openbb"]

print()
print("=" * 72)
print("7. Wired into Backtester and LiveSignalRunner (all four panels together)")
print("=" * 72)

price_repo = SyntheticRepository(n_symbols=3, seed=9)
price_panel = price_repo.fetch(start="2020-01-01", end="2021-06-30")
syms = price_panel.symbols

opt_rows = []
d = pd.Timestamp("2020-02-01")
while d < pd.Timestamp("2021-06-01"):
    for sym in syms:
        opt_rows.append((sym, "iv_atm_near", d, d, 0.25))
    d += pd.Timedelta(days=1)
opanel = OptionsPanel(
    frame=pd.DataFrame(opt_rows, columns=["symbol", "metric", "period_end", "as_of_date", "value"])
)
fpanel = FundamentalsPanel(
    frame=pd.DataFrame(
        [(sym, "income_revenue", pd.Timestamp("2020-03-31"), pd.Timestamp("2020-05-05"), 100.0) for sym in syms],
        columns=["symbol", "metric", "period_end", "as_of_date", "value"],
    )
)
mpanel = MacrosPanel(
    frame=pd.DataFrame(
        [("fed_funds_rate", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), 1.0)],
        columns=["metric", "period_end", "as_of_date", "value"],
    )
)
cpanel = CorpsPanel(
    frame=pd.DataFrame(
        [(sym, "filed_8k_count_90d", pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-01"), 1.0) for sym in syms],
        columns=["symbol", "metric", "period_end", "as_of_date", "value"],
    )
)

seen_violations = []
seen_calls = {"n": 0}
seen_all_four = {"n": 0}


class ProbeStrategy:
    name = "probe"

    @property
    def min_history(self):
        return 30

    def target_weights(self, view, fundamentals=None, macros=None, corps=None, options=None):
        seen_calls["n"] += 1
        decision_date = view.last_date()
        if options is not None and len(options.frame):
            if (options.frame["as_of_date"] > decision_date).any():
                seen_violations.append(decision_date)
        if all(x is not None for x in (fundamentals, macros, corps, options)):
            seen_all_four["n"] += 1
        return pd.Series(1.0 / len(view.symbols), index=view.symbols)


bt = Backtester(
    panel=price_panel,
    strategy=ProbeStrategy(),
    fundamentals=fpanel,
    macros=mpanel,
    corps=cpanel,
    options=opanel,
    rebalance="M",
    initial_equity=50_000,
)
bt.run()
check("backtester actually calls the strategy", seen_calls["n"] > 0, f"{seen_calls['n']} calls")
check(
    "backtester never leaks a future options reading into the strategy",
    not seen_violations, f"{len(seen_violations)} violation(s)",
)
check(
    "backtester hands all four panels to the same call",
    seen_all_four["n"] == seen_calls["n"],
)

runner = LiveSignalRunner(strategy=ProbeStrategy())
state = PortfolioState(cash=10_000.0, shares=pd.Series(0.0, index=syms))

seen_calls["n"] = 0
seen_violations.clear()
plan = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"), options=opanel)
check("live runner calls the strategy", seen_calls["n"] == 1)
check("live runner never leaks a future options reading", not seen_violations)

seen_calls["n"] = 0
plan2 = runner.plan(price_panel, state, asof=pd.Timestamp("2020-08-15"))
check(
    "live runner still works with no options argument at all",
    seen_calls["n"] == 1 and plan2.asof == plan.asof,
)

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
