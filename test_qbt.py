"""Validation suite. The null test is the one that matters."""

import os
import tempfile
import time

import numpy as np
import pandas as pd

from qbt import (
    Backtester, BreadthRegimeFilter, Composite, CostModel,
    CrossSectionalMomentum, DayTradeLedger,
    EqualWeightBuyHold, ExecutionConfig, InverseVolWeighted, LiveSignalRunner,
    MultiFactorCrossSectional, OpenBBRepository, PortfolioState, PricePanel,
    RiskGate, ShortHorizonReversal, SyntheticRepository, TimeSeriesMomentum,
    TrendFilter, compare, ic_grid, ic_summary, information_coefficient,
    expected_max_sharpe, forward_returns, rebalance_dates,
    return_autocorrelation, trailing_signal, walk_forward_splits,
    ParameterSweep, sharpe_haircut,
)
from qbt.data import prune_cache
from qbt.risk import RiskContext

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 72)
print("1. Panel and look-ahead firewall")
print("=" * 72)

repo = SyntheticRepository(n_symbols=30, seed=11)
panel = repo.fetch(start="2015-01-01", end="2024-12-31")
print(panel.describe())

mid = panel.dates[1000]
view = panel.as_of(mid)
check("as_of truncates at date", view.last_date() == mid,
      f"{view.last_date().date()}")
check("as_of drops all future bars", len(view) == 1001, f"len={len(view)}")
check("as_of preserves symbols", view.symbols == panel.symbols)


class PeekingStrategy:
    """Tries to read tomorrow. Should be unable to."""
    name = "peeker"

    def __init__(self):
        self.max_seen = None

    @property
    def min_history(self):
        return 10

    def target_weights(self, view, fundamentals=None, macros=None, corps=None, options=None):
        last = view.last_date()
        self.max_seen = last if self.max_seen is None else max(self.max_seen, last)
        return pd.Series(1.0 / len(view.symbols), index=view.symbols)


peeker = PeekingStrategy()
bt = Backtester(panel=panel, strategy=peeker, risk_gate=None,
                rebalance="M", initial_equity=25_000)
res = bt.run()
last_decision = res.targets.index.max()
check("strategy never sees beyond its decision bar",
      peeker.max_seen == last_decision,
      f"max_seen={peeker.max_seen.date()} last_decision={last_decision.date()}")

# OpenBBRepository used to infer "this is a single-symbol response" solely
# from the *absence* of a column named "symbol" -- if a multi-symbol
# request came back with no symbol-identifying column at all (a provider
# naming it differently, or omitting it unexpectedly), every row across
# every requested symbol got silently mislabeled with symbols[0], merging
# unrelated price series into one column with no error. Deferred openbb
# import means the real dependency is mocked here via sys.modules, not a
# live network call.
import sys
from unittest.mock import MagicMock


def _fake_openbb(df):
    # _fetch_remote() does `from openbb import obb`, which reads the
    # `obb` *attribute* of the openbb module, not the module object
    # itself -- the mock has to live at .obb, not at the module's own
    # top level, or the real code path never touches it.
    fake_module = MagicMock()
    fake_module.obb.equity.price.historical.return_value.to_dataframe.return_value = df
    sys.modules["openbb"] = fake_module


_real_openbb_module = sys.modules.get("openbb")
try:
    obb_repo = OpenBBRepository(provider="fake", cache_dir=None)

    # 1. Multi-symbol response with an explicit "symbol" column -- normal
    #    case, must still work.
    multi_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "symbol": ["AAA", "BBB"],
        "close": [10.0, 20.0],
    })
    _fake_openbb(multi_df)
    out = obb_repo._fetch_remote(["AAA", "BBB"], "2024-01-01", "2024-01-02")
    check("a genuine multi-symbol response with a 'symbol' column still works",
          set(out["symbol"]) == {"AAA", "BBB"})

    # 2. Single-symbol response, no symbol column at all -- still safe,
    #    since there's exactly one symbol to attribute every row to.
    single_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]), "close": [10.0],
    })
    _fake_openbb(single_df)
    out = obb_repo._fetch_remote(["AAA"], "2024-01-01", "2024-01-02")
    check("a single-symbol response with no symbol column is still labeled correctly",
          list(out["symbol"]) == ["AAA"])

    # 3. The dangerous case: MULTIPLE symbols requested, but the response
    #    has no symbol/ticker column to attribute rows to. Must raise, not
    #    silently mislabel every row with symbols[0].
    ambiguous_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "close": [10.0, 20.0],
    })
    _fake_openbb(ambiguous_df)
    try:
        obb_repo._fetch_remote(["AAA", "BBB"], "2024-01-01", "2024-01-02")
        check("a multi-symbol request with no symbol column raises, doesn't "
              "silently mislabel every row", False)
    except ValueError as exc:
        check("a multi-symbol request with no symbol column raises, doesn't "
              "silently mislabel every row", "symbol" in str(exc).lower(), str(exc))

    # 4. A "ticker" column (not "symbol") must also be recognized, not
    #    treated as absent.
    ticker_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "ticker": ["AAA", "BBB"],
        "close": [10.0, 20.0],
    })
    _fake_openbb(ticker_df)
    out = obb_repo._fetch_remote(["AAA", "BBB"], "2024-01-01", "2024-01-02")
    check("a 'ticker' column is recognized the same as 'symbol'",
          set(out["symbol"]) == {"AAA", "BBB"})
finally:
    if _real_openbb_module is None:
        sys.modules.pop("openbb", None)
    else:
        sys.modules["openbb"] = _real_openbb_module

print()
print("=" * 72)
print("2. Null test: structureless data must produce no edge")
print("=" * 72)

null_repo = SyntheticRepository(
    n_symbols=40, seed=3, mu_persistence=0.0, mu_scale=0.0,
    reversal_theta=0.0, drift=0.0,
)
null_panel = null_repo.fetch(start="2010-01-01", end="2024-12-31")

null_results = {}
for strat in [
    CrossSectionalMomentum(lookback=126, top_n=10),
    ShortHorizonReversal(lookback=5, top_n=5),
]:
    r = Backtester(
        panel=null_panel, strategy=strat,
        risk_gate=RiskGate(target_vol=0.12, max_drawdown=None),
        cost_model=CostModel(slippage_bps=0.0, sell_fee_bps=0.0),
        rebalance="M", initial_equity=100_000,
    ).run()
    null_results[strat.name] = r
    s = r.summary()
    # t-stat of the mean daily return
    rr = r.returns[r.returns != 0]
    t = rr.mean() / rr.std() * np.sqrt(len(rr)) if len(rr) > 2 else np.nan
    print(f"  {strat.name:26s} sharpe={s['sharpe']:+.2f}  t={t:+.2f}  "
          f"cagr={s['cagr']:+.2%}")
    check(f"null: |sharpe| < 0.6 for {strat.name}", abs(s["sharpe"]) < 0.6,
          f"sharpe={s['sharpe']:.2f}")
    check(f"null: |t| < 2.6 for {strat.name}", abs(t) < 2.6, f"t={t:.2f}")

print()
print("=" * 72)
print("3. Engine mechanics")
print("=" * 72)

bh = Backtester(
    panel=panel, strategy=EqualWeightBuyHold(), risk_gate=None,
    cost_model=CostModel(slippage_bps=0.0, sell_fee_bps=0.0, commission_bps=0.0),
    execution=ExecutionConfig(delay_bars=1, price="close"),
    rebalance=252,  # roughly annual: buy and mostly hold
    initial_equity=50_000,
).run()
check("equity starts at initial capital",
      abs(bh.equity.iloc[0] - 50_000) < 1e-6, f"{bh.equity.iloc[0]:.2f}")
check("buy-and-hold equity grows", bh.equity.iloc[-1] > bh.equity.iloc[0],
      f"{bh.equity.iloc[-1]:,.0f}")
check("no NaNs in equity curve", bh.equity.notna().all())
check("holdings never exceed 100% gross without margin",
      bh.exposure.max() < 1.05, f"max={bh.exposure.max():.3f}")

# Costs must reduce returns, monotonically in slippage.
sharpes = {}
for slip in (0.0, 10.0, 50.0):
    r = Backtester(
        panel=panel, strategy=CrossSectionalMomentum(lookback=63, top_n=8),
        risk_gate=RiskGate(target_vol=0.12, max_drawdown=None),
        cost_model=CostModel(slippage_bps=slip),
        rebalance="W", initial_equity=50_000,
    ).run()
    sharpes[slip] = r.summary()["cagr"]
print("  CAGR by slippage:", {k: f"{v:.2%}" for k, v in sharpes.items()})
check("higher slippage lowers CAGR",
      sharpes[0.0] > sharpes[10.0] > sharpes[50.0])

# Execution delay must matter, and delay=0 should flatter the strategy.
delayed = {}
for d in (0, 1):
    r = Backtester(
        panel=panel, strategy=ShortHorizonReversal(lookback=3, top_n=6, min_z=0.5),
        risk_gate=RiskGate(target_vol=0.15, max_drawdown=None),
        cost_model=CostModel(slippage_bps=0.0, sell_fee_bps=0.0),
        execution=ExecutionConfig(delay_bars=d, price="close"),
        rebalance="D", initial_equity=50_000,
    ).run()
    delayed[d] = r.summary()["sharpe"]
print(f"  reversal sharpe: delay=0 -> {delayed[0]:.2f}, delay=1 -> {delayed[1]:.2f}")
check("delay=0 inflates a same-close signal", delayed[0] > delayed[1],
      "this is exactly the look-ahead bug the default guards against")

# `pending` used to be a single tuple|None slot, not a queue -- a new
# decision arriving before the previous one's fill executed silently
# overwrote it, with no error and no trace, dropping the entire rebalance.
# Only shows up when the rebalance cadence outpaces delay_bars (not the
# library defaults, rebalance='M'/delay_bars=1, which is why it went
# unnoticed). This strategy alternates fully between two disjoint sleeves
# every single decision, so a genuine trade should exist for nearly every
# decision -- anything otherwise is dropped, not a legitimate no-op.
class _AlternatingStrategy:
    name = "alternating"
    min_history = 1

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None):
        syms = view.symbols
        half = pd.Series({syms[0]: 0.5, syms[1]: 0.5}).reindex(syms).fillna(0.0)
        other_half = pd.Series({syms[2]: 0.5, syms[3]: 0.5}).reindex(syms).fillna(0.0)
        return half if len(view.dates) % 2 == 0 else other_half


alt_panel = SyntheticRepository(n_symbols=4, seed=2).fetch(
    start="2020-01-01", end="2020-04-01")
alt_result = Backtester(
    panel=alt_panel, strategy=_AlternatingStrategy(), risk_gate=None,
    rebalance="D", execution=ExecutionConfig(delay_bars=2, price="close"),
    initial_equity=100_000,
).run()
alt_n_decisions = len(alt_result.targets)
alt_n_fill_dates = (alt_result.trades["date"].nunique()
                    if not alt_result.trades.empty else 0)
check("a rebalance cadence faster than delay_bars does not silently drop "
      "decisions (a single-slot pending overwrite used to drop ~all of them)",
      alt_n_decisions - alt_n_fill_dates <= 3,
      f"{alt_n_decisions} decisions, only {alt_n_fill_dates} distinct fill dates")

# The last delay_bars decisions of any run have nowhere to schedule a fill
# (i + delay_bars would run past the panel's last bar) -- an unavoidable
# edge effect, not a bug. But target_rows/audit used to record those
# decisions with no marker distinguishing them from ones that actually
# executed, so a reader of the tail of those frames could mistake
# "decided" for "filled." audit["scheduled"] now flags exactly this.
tail_panel = SyntheticRepository(n_symbols=5, seed=1).fetch(
    start="2020-01-01", end="2020-03-01")
tail_result = Backtester(
    panel=tail_panel, strategy=EqualWeightBuyHold(), risk_gate=None,
    rebalance="D", execution=ExecutionConfig(delay_bars=3, price="close"),
    initial_equity=100_000,
).run()
tail_unscheduled = tail_result.audit[~tail_result.audit["scheduled"]]
check("exactly the last delay_bars decisions are flagged unscheduled",
      len(tail_unscheduled) == 3, len(tail_unscheduled))
check("every decision before the tail is flagged scheduled",
      tail_result.audit["scheduled"].iloc[:-3].all())
check("the unscheduled decisions are exactly the panel's final dates",
      list(tail_unscheduled.index) == list(tail_result.audit.index[-3:]))

print()
print("=" * 72)
print("4. Risk gate")
print("=" * 72)

gate = RiskGate(target_vol=0.10, max_weight=0.20, max_gross=1.0,
                max_drawdown=0.15)
r_gated = Backtester(
    panel=panel, strategy=CrossSectionalMomentum(lookback=126, top_n=3),
    risk_gate=gate, rebalance="M", initial_equity=50_000,
).run()
max_w = r_gated.targets.abs().to_numpy().max()
check("per-name cap enforced", max_w <= 0.20 + 1e-9, f"max target weight={max_w:.4f}")
check("gross cap enforced", r_gated.targets.abs().sum(axis=1).max() <= 1.0 + 1e-9)
check("audit log populated", len(r_gated.audit) == len(r_gated.targets),
      f"{len(r_gated.audit)} rows")
check("audit records reasons",
      r_gated.audit["notes"].astype(str).str.len().gt(0).any())
print("  sample audit notes:")
for note in r_gated.audit["notes"].replace("", np.nan).dropna().head(3):
    print(f"    - {note}")

# Vol targeting should compress realised vol toward the target.
vols = {}
for tv in (0.06, 0.12, None):
    r = Backtester(
        panel=panel, strategy=CrossSectionalMomentum(lookback=126, top_n=10),
        risk_gate=RiskGate(target_vol=tv, max_drawdown=None, max_vol_scale=1.0),
        rebalance="M", initial_equity=50_000,
    ).run()
    vols[tv] = r.summary()["ann_vol"]
print("  realised ann vol by target:",
      {str(k): f"{v:.2%}" for k, v in vols.items()})
check("lower vol target -> lower realised vol", vols[0.06] < vols[0.12])

# Drawdown breaker must be able to fire and flatten the book.
crash = SyntheticRepository(n_symbols=20, seed=5, drift=-0.0020,
                            mu_persistence=0.0, mu_scale=0.0).fetch(
    start="2018-01-01", end="2022-12-31")
r_crash = Backtester(
    panel=crash, strategy=EqualWeightBuyHold(),
    risk_gate=RiskGate(target_vol=None, max_drawdown=0.10, resume_at=0.95),
    rebalance="W", initial_equity=50_000,
).run()
check("drawdown breaker fires in a bear market",
      bool(r_crash.audit["halted"].any()),
      f"halted on {int(r_crash.audit['halted'].sum())} of "
      f"{len(r_crash.audit)} rebalances")
check("breaker flattens the book",
      r_crash.exposure.tail(200).min() < 0.02)

# The breaker must be able to reset, not just fire. A book that goes flat
# has frozen equity, so a real reset needs a "shadow" recovery signal --
# this is the scenario that used to be permanently broken (resume_at could
# never be reached because the real, halted equity can't move on its own).
_n = 6
_dates = pd.bdate_range("2020-01-01", periods=300)
_down = np.full(50, -0.007)   # steep crash, ~30% trough
_up = np.full(150, 0.0045)    # genuine, strong recovery afterward
_flat = np.zeros(len(_dates) - 200)
_daily_ret = np.concatenate([_down, _up, _flat])
_rng = np.random.default_rng(3)
_idio = _rng.normal(0, 0.003, size=(len(_dates), _n))
_close = pd.DataFrame(
    100.0 * np.exp(np.cumsum(_daily_ret[:, None] + _idio, axis=0)),
    index=_dates, columns=[f"V{i}" for i in range(_n)],
)
v_panel = PricePanel(close=_close)
v_gate = RiskGate(target_vol=None, max_drawdown=0.20, resume_at=0.90)
r_vshape = Backtester(
    panel=v_panel, strategy=EqualWeightBuyHold(), risk_gate=v_gate,
    rebalance="D", initial_equity=50_000,
    execution=ExecutionConfig(price="close"),
).run()
_reset_mask = r_vshape.audit["notes"].str.contains("breaker reset", na=False)
check("drawdown breaker actually resets after a genuine recovery",
      bool(_reset_mask.any()))
check("breaker was genuinely halted for a stretch, not the whole run",
      0 < int(r_vshape.audit["halted"].sum()) < len(r_vshape.audit))
if _reset_mask.any():
    check("reset happens strictly after the trip, not immediately",
          r_vshape.audit.index[_reset_mask][0] > r_vshape.audit.index[r_vshape.audit["halted"]][0])

# The other way the breaker could latch forever: tripping on a bar where
# the proposed book is empty. There are then no weights whose recovery to
# shadow, and a flat shadow book earns exactly 0% in perpetuity, so
# _shadow_ratio never moves and resume_at is never reached -- the same
# silent permanent halt as the stuck peak-equity bug, from another
# direction. The gate re-arms from the first proposal that has something
# in it instead.
_rearm_idx = pd.bdate_range("2024-01-01", periods=30)
_rearm_rets = pd.DataFrame({"A": [0.0] * 29 + [0.05]}, index=_rearm_idx)
_rearm_ctx = RiskContext(date=_rearm_idx[-1], equity=70.0, peak_equity=100.0,
                         prices=pd.Series({"A": 10.0}), returns=_rearm_rets)
_rearm_gate = RiskGate(target_vol=None, max_drawdown=0.20, resume_at=0.90)
_trip = _rearm_gate.apply(pd.Series(dtype=float), _rearm_ctx)
check("breaker trips on an empty proposed book",
      _trip.halted and _rearm_gate._shadow_weights.empty)

_rearm_bars = None
for _k in range(1, 60):
    _d = _rearm_gate.apply(pd.Series({"A": 1.0}), _rearm_ctx)
    if _k == 1:
        check("it re-arms the shadow book once there's something to track",
              any("re-armed" in n for n in _d.notes), _d.notes)
    if not _d.halted:
        _rearm_bars = _k
        break
check("and then genuinely recovers instead of latching forever",
      _rearm_bars is not None,
      f"recovered after {_rearm_bars} bars" if _rearm_bars
      else "still halted after 59 bars")

# PDT limit must actually block new entries at the gate, not just log a note.
pdt_gate = RiskGate(target_vol=None, max_weight=None, max_gross=None, max_drawdown=None)
_syms = ["PA", "PB", "PC"]
_proposed = pd.Series([0.3, 0.3, 0.3], index=_syms)
_positions_a_only = pd.Series([100.0, 0.0, 0.0], index=_syms)
_pdt_returns = pd.DataFrame({s: [0.001] * 30 for s in _syms})
_pdt_prices = pd.Series([50.0, 60.0, 70.0], index=_syms)

_ctx_no_budget = RiskContext(
    date=pd.Timestamp("2024-01-01"), equity=50_000, peak_equity=50_000,
    prices=_pdt_prices, returns=_pdt_returns, positions=_positions_a_only,
    day_trades_remaining=0,
)
_decision_blocked = pdt_gate.apply(_proposed, _ctx_no_budget)
check("PDT limit blocks new entries at the gate",
      _decision_blocked.weights["PB"] == 0.0 and _decision_blocked.weights["PC"] == 0.0)
check("PDT limit leaves the already-held name alone",
      _decision_blocked.weights["PA"] == 0.3)
check("PDT block is recorded in the audit notes",
      any("PDT limit reached" in n for n in _decision_blocked.notes))

_ctx_with_budget = RiskContext(
    date=pd.Timestamp("2024-01-01"), equity=50_000, peak_equity=50_000,
    prices=_pdt_prices, returns=_pdt_returns, positions=_positions_a_only,
    day_trades_remaining=3,
)
_decision_ok = pdt_gate.apply(_proposed, _ctx_with_budget)
check("PDT limit doesn't interfere when budget is available",
      (_decision_ok.weights == _proposed).all())

_decision_close = pdt_gate.apply(
    pd.Series([0.0, 0.0, 0.0], index=_syms), _ctx_no_budget
)
check("PDT limit still allows closing an existing position",
      (_decision_close.weights == 0.0).all())

print()
print("=" * 72)
print("5. Strategies and composition")
print("=" * 72)

strategies = [
    EqualWeightBuyHold(),
    CrossSectionalMomentum(lookback=126, skip=5, top_n=10),
    CrossSectionalMomentum(lookback=126, top_n=10, weighting="rank"),
    TimeSeriesMomentum(lookback=200),
    ShortHorizonReversal(lookback=5, top_n=5),
    TrendFilter(CrossSectionalMomentum(lookback=126, top_n=10), lookback=200),
    InverseVolWeighted(CrossSectionalMomentum(lookback=126, top_n=10)),
    Composite([
        (CrossSectionalMomentum(lookback=126, top_n=10), 0.6),
        (ShortHorizonReversal(lookback=5, top_n=5), 0.4),
    ]),
]

results = {}
for strat in strategies:
    r = Backtester(
        panel=panel, strategy=strat,
        risk_gate=RiskGate(target_vol=0.12, max_drawdown=0.25),
        cost_model=CostModel(slippage_bps=5.0),
        rebalance="M", initial_equity=50_000,
    ).run()
    results[strat.name] = r
    check(f"{strat.name} runs and produces finite equity",
          np.isfinite(r.equity).all() and (r.equity > 0).all())

tbl = compare(results)
print()
print(tbl[["cagr", "ann_vol", "sharpe", "max_drawdown", "turnover_ann",
           "avg_exposure", "n_trades"]].round(3).to_string())

check("trend filter reduces exposure vs unfiltered",
      results["xsmom_126_5_10+trend200"].summary()["avg_exposure"]
      < results["xsmom_126_5_10"].summary()["avg_exposure"])

# top_n=0 used to reach all the way into target_weights() before failing
# (a ZeroDivisionError on equal weighting, or silent NaN weights on rank
# weighting) instead of being rejected at construction, where a bad config
# value belongs.
for _name, _ctor in [
    ("CrossSectionalMomentum", lambda: CrossSectionalMomentum(top_n=0)),
    ("ShortHorizonReversal", lambda: ShortHorizonReversal(top_n=0)),
    ("MultiFactorCrossSectional", lambda: MultiFactorCrossSectional(top_n=0)),
]:
    try:
        _ctor()
        check(f"{_name} rejects top_n=0 at construction", False)
    except ValueError:
        check(f"{_name} rejects top_n=0 at construction", True)
check("top_n=None (a documented fallback to half the candidates) still "
      "constructs fine", CrossSectionalMomentum(top_n=None).top_n is None)

# TrendFilter's blocked = (last <= ma).reindex(...).fillna(True) used to get
# the fail-safe direction backwards for a NaN moving average: `last <= ma`
# evaluates to False, not NaN, when ma is NaN (a symbol with zero valid
# observations anywhere in its trailing lookback window -- confirmed via
# `np.nan <= np.nan` -> False), so an unscreenable name passed straight
# through with zero trend confirmation instead of being blocked -- exactly
# backwards for a filter whose whole job is requiring that confirmation.
class _TrendStubInner:
    """Doesn't itself gate on live/tradeable data -- the point is testing
    TrendFilter's own robustness, not relying on the inner strategy to
    have already excluded an unscoreable symbol."""
    name = "stub"
    min_history = 5

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None):
        return pd.Series({"OLD": 0.5, "NEW": 0.5}).reindex(view.symbols).fillna(0.0)


_trend_dates = pd.date_range("2024-01-01", periods=250, freq="D")
_trend_close = pd.DataFrame({
    "OLD": np.linspace(100, 200, 250),  # clean long uptrend -- always passes
    "NEW": [np.nan] * 250,              # zero valid data -- unscoreable
}, index=_trend_dates)
_trend_panel = PricePanel(close=_trend_close)
_trend_w = TrendFilter(inner=_TrendStubInner(), lookback=200).target_weights(_trend_panel)
check("a symbol with zero valid data in its trailing window is blocked, "
      "not passed through with zero trend confirmation",
      _trend_w["NEW"] == 0.0, _trend_w.to_dict())
check("a symbol with a genuine, fully-scoreable uptrend still passes",
      _trend_w["OLD"] == 0.5, _trend_w.to_dict())

# The *realistic* version of the same bug, which the all-NaN column above
# cannot catch. mean() skips NaN, so a name with 20 real bars inside a
# 200-bar window produces a finite 20-bar average -- not NaN -- and
# ma.isna() sees nothing wrong. It then gets compared against a "200-day
# moving average" that is really a 20-day one, and a recently-listed name
# in a short uptrend sails through the gate. TrendFilter's own comment
# already described exactly this case ("a recently-added or short-history
# name that still cleared the inner strategy's shorter min_history") as
# one that must block; only the all-NaN half was actually being caught.
class _PartialStubInner:
    name = "stub_partial"
    min_history = 5

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None):
        return pd.Series({"FULL": 0.5, "YOUNG": 0.5}).reindex(
            view.symbols).fillna(0.0)


_partial_young = np.full(250, np.nan)
_partial_young[-20:] = np.linspace(50, 60, 20)   # listed 20 bars ago, rising
_partial_close = pd.DataFrame({
    "FULL": np.linspace(100, 200, 250),          # full history, real uptrend
    "YOUNG": _partial_young,
}, index=_trend_dates)
_partial_panel = PricePanel(close=_partial_close)

_partial_ma = _partial_panel.close.tail(200).mean()
check("the premise: a 20-of-200-bar window yields a finite mean, so an "
      "isna() check alone cannot detect it",
      np.isfinite(_partial_ma["YOUNG"]),
      f"ma[YOUNG]={_partial_ma['YOUNG']}")

_partial_w = TrendFilter(inner=_PartialStubInner(), lookback=200).target_weights(
    _partial_panel)
check("a symbol whose trailing window is only partially populated is blocked, "
      "not scored against a silently-shortened moving average",
      _partial_w["YOUNG"] == 0.0, _partial_w.to_dict())
check("its fully-populated neighbour is unaffected",
      _partial_w["FULL"] == 0.5, _partial_w.to_dict())

# BreadthRegimeFilter shares the same trailing-MA rule and had the same
# hole: a partially-populated name counted as fully scoreable, landing in
# both the numerator and the denominator of a breadth read it has no
# business contributing to.
#
# YOUNG must trend *down* here, unlike the TrendFilter panel above. A
# rising YOUNG sits above its own short MA, so it lands in numerator and
# denominator alike and breadth reads 2/2 = 1.0 -- exactly the 1/1 = 1.0
# the fix produces, making the check pass either way and prove nothing.
# Falling, it's counted in the denominator only: 1/2 = 0.5 before the fix
# versus 1/1 = 1.0 after.
_breadth_young = np.full(250, np.nan)
_breadth_young[-20:] = np.linspace(60, 50, 20)   # listed 20 bars ago, falling
_breadth_close = pd.DataFrame({
    "FULL": np.linspace(100, 200, 250),
    "YOUNG": _breadth_young,
}, index=_trend_dates)
_breadth_panel = PricePanel(close=_breadth_close)
_breadth_partial = BreadthRegimeFilter(inner=_PartialStubInner(), lookback=200)
check("breadth excludes a partially-populated symbol from both numerator "
      "and denominator (1 of 1 fully-scoreable name above its MA -> 1.0, "
      "not 1 of 2 -> 0.5)",
      _breadth_partial.breadth(_breadth_panel) == 1.0,
      _breadth_partial.breadth(_breadth_panel))
check("breadth is NaN when no symbol has a fully-populated window",
      np.isnan(_breadth_partial.breadth(
          PricePanel(close=_breadth_close[["YOUNG"]]))))

# InverseVolWeighted used to preserve the *inner* strategy's full gross
# across only the vol-scoreable names, so a pick it couldn't size handed
# its weight to the picks it could -- the one wrapper in the module that
# grew a position in response to missing data.
class _IVolPicks:
    name = "ivol_picks"
    min_history = 5

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None):
        return pd.Series({"GOOD": 0.5, "NOVOL": 0.5}).reindex(
            view.symbols).fillna(0.0)


_ivol_n = 100
_ivol_idx = pd.bdate_range("2024-01-01", periods=_ivol_n)
_ivol_close = pd.DataFrame({
    "GOOD": 100 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.01, _ivol_n))),
    "NOVOL": [100.0] * _ivol_n,          # zero variance -> unscoreable
}, index=_ivol_idx)
_ivol_w = InverseVolWeighted(inner=_IVolPicks(), vol_lookback=63).target_weights(
    PricePanel(close=_ivol_close))
check("an unscoreable pick's weight becomes cash, not absorbed by its "
      "neighbours (gross 0.5, not the inner strategy's full 1.0)",
      abs(float(_ivol_w.abs().sum()) - 0.5) < 1e-9, _ivol_w.to_dict())
check("the unscoreable name itself is flat",
      _ivol_w["NOVOL"] == 0.0, _ivol_w.to_dict())

# CrossSectionalMomentum is *relative* momentum: top_n=None holds the top
# half whatever the sign of their trailing return. The docstring used to
# claim it filtered to positive scores, which it never did -- an
# absolute-momentum gate is TrendFilter's job.
_falling = pd.DataFrame(
    {f"S{i}": np.linspace(100, 100 - (i + 1) * 5, 200) for i in range(6)},
    index=pd.bdate_range("2020-01-01", periods=200))
_falling_panel = PricePanel(close=_falling)
_xsm_none = CrossSectionalMomentum(lookback=126, skip=5, top_n=None)
check("the premise: every trailing return in this universe is negative",
      bool((_xsm_none.score(_falling_panel) < 0).all()))
_xsm_w = _xsm_none.target_weights(_falling_panel)
check("top_n=None holds the top half regardless of sign, as documented",
      abs(float(_xsm_w.abs().sum()) - 1.0) < 1e-9
      and int((_xsm_w > 0).sum()) == 3, _xsm_w.to_dict())

print()
print("=" * 72)
print("6. Research diagnostics")
print("=" * 72)

sig = trailing_signal(panel, 126, skip=5)
fwd = forward_returns(panel, 21)
ic = information_coefficient(sig, fwd, step=21)
summ = ic_summary(ic)
print("  momentum IC (126d signal, 21d fwd, non-overlapping):")
print("   ", {k: round(v, 4) for k, v in summ.items()})
check("IC series produced", len(ic) > 20, f"{len(ic)} obs")
check("synthetic momentum is detected", summ["mean_ic"] > 0,
      f"mean_ic={summ['mean_ic']:.4f} t={summ['t_stat']:.2f}")

mean_ic, t_ic = ic_grid(panel, lookbacks=(5, 21, 126), horizons=(1, 5, 21))
print("\n  mean IC grid (rows=lookback, cols=horizon):")
print(mean_ic.round(4).to_string())
check("ic_grid shape correct", mean_ic.shape == (3, 3))
rev_only = SyntheticRepository(
    n_symbols=40, seed=21, mu_persistence=0.0, mu_scale=0.0,
    reversal_theta=0.25, drift=0.0,
).fetch(start="2012-01-01", end="2024-12-31")
rev_ic, rev_t = ic_grid(rev_only, lookbacks=(3, 5, 21), horizons=(1, 5, 21))
print("\n  mean IC grid on a reversal-only panel:")
print(rev_ic.round(4).to_string())
check("reversal-only panel shows negative short-horizon IC",
      rev_ic.loc[5, 1] < 0 and rev_t.loc[5, 1] < -2,
      f"IC(5,1)={rev_ic.loc[5, 1]:.4f} t={rev_t.loc[5, 1]:.2f}")
check("mixed panel shows reversal at short/short, momentum at long/long",
      mean_ic.loc[5, 1] < 0 < mean_ic.loc[126, 21],
      f"IC(5,1)={mean_ic.loc[5, 1]:+.4f}  IC(126,21)={mean_ic.loc[126, 21]:+.4f}")

ac = return_autocorrelation(panel, lags=[1, 2, 5, 10])
print("\n  pooled return autocorrelation:")
print(ac.round(4).to_string())
check("lag-1 autocorrelation is negative (reversal built in)",
      ac.loc[1, "autocorr"] < 0, f"{ac.loc[1, 'autocorr']:.4f}")
# n treats every (symbol, date) pair as an independent draw, which overstates
# the sample size by roughly the symbol count since returns are strongly
# cross-sectionally correlated -- n_effective (distinct dates) is what the
# t-stat should actually be built on.
check("n_effective is far smaller than the raw pooled n",
      ac.loc[1, "n_effective"] < ac.loc[1, "n"] / 10,
      f"n={ac.loc[1, 'n']:.0f}  n_effective={ac.loc[1, 'n_effective']:.0f}")
check("n / n_effective roughly matches the symbol count",
      abs(ac.loc[1, "n"] / ac.loc[1, "n_effective"] - len(panel.symbols)) < 1,
      f"ratio={ac.loc[1, 'n'] / ac.loc[1, 'n_effective']:.1f}  "
      f"n_symbols={len(panel.symbols)}")
check("t_stat is built from n_effective, not the inflated raw n",
      abs(ac.loc[1, "t_stat"] - ac.loc[1, "autocorr"] * np.sqrt(ac.loc[1, "n_effective"])) < 1e-6)

splits = walk_forward_splits(panel.dates, train_bars=756, test_bars=252)
check("walk-forward splits generated", len(splits) >= 2, f"{len(splits)} splits")
check("no train/test overlap",
      all(tr[-1] < te[0] for tr, te in splits))

ems = expected_max_sharpe(n_trials=48, sharpe_dispersion=0.35)
hc = sharpe_haircut(0.95, n_trials=48, sharpe_dispersion=0.35)
print(f"\n  expected max sharpe under null (48 trials, sd=0.35): {ems:.2f}")
print(f"  haircut of an observed 0.95: {hc}")
check("expected_max_sharpe is positive and sensible", 0.3 < ems < 1.5)

sweep = ParameterSweep(
    grid={"lookback": [63, 126], "top_n": [5, 10]},
    evaluate=lambda lookback, top_n: {
        "sharpe": Backtester(
            panel=panel,
            strategy=CrossSectionalMomentum(lookback=lookback, top_n=top_n),
            risk_gate=RiskGate(target_vol=0.12, max_drawdown=None),
            rebalance="M", initial_equity=50_000,
        ).run().summary()["sharpe"]
    },
)
sw = sweep.run()
print("\n  sweep results:")
print(sw.round(3).to_string(index=False))
check("sweep returns one row per combination", len(sw) == 4)
print("  stability:", ParameterSweep.stability(sw).round(3).to_dict())

print()
print("=" * 72)
print("7. Live parity: same objects, same numbers")
print("=" * 72)

strategy = CrossSectionalMomentum(lookback=126, skip=5, top_n=10)
live_gate = RiskGate(target_vol=0.12, max_weight=0.20, max_drawdown=0.25)

runner = LiveSignalRunner(strategy=strategy, risk_gate=live_gate,
                          min_trade_notional=10.0, max_turnover=None)
state = PortfolioState(cash=50_000.0,
                       shares=pd.Series(0.0, index=panel.symbols),
                       peak_equity=50_000.0)
plan = runner.plan(panel, state)
print(f"  {plan!r}")
print(plan.to_frame().head(6).round(4).to_string(index=False))
check("live plan produces intents", len(plan.intents) > 0)
check("all intents are buys from a flat book",
      all(i.side == "buy" for i in plan.intents))
check("live intents sum close to target gross",
      abs(sum(i.notional for i in plan.intents) / plan.equity
          - plan.target_weights.abs().sum()) < 0.02)

# Parity: the runner's target weights must equal what the gate produced in a
# backtest that decided on the same bar with the same equity and no history.
bare_gate = RiskGate(target_vol=0.12, max_weight=0.20, max_drawdown=0.25)
v = panel.as_of(panel.last_date())
ctx = RiskContext(date=v.last_date(), equity=50_000.0, peak_equity=50_000.0,
                  prices=v.last_close(), returns=v.returns(),
                  positions=state.shares, day_trades_remaining=3)
direct = bare_gate.apply(strategy.target_weights(v), ctx).weights
check("live runner weights match a direct gate call",
      np.allclose(plan.target_weights.reindex(direct.index).fillna(0.0).to_numpy(),
                  direct.to_numpy(), atol=1e-12),
      "single code path confirmed")

print("\n  audit record:", plan.audit_record())

# Turnover guard should scale an oversized plan down to fit the cap, not
# discard it -- except `state` here is a flat book (all-zero shares), and
# allow_full_turnover_from_flat defaults to True, so the exemption needs to
# be off to actually exercise the scaling path.
uncapped = LiveSignalRunner(strategy=strategy, risk_gate=live_gate,
                            max_turnover=None)
uplan = uncapped.plan(panel, state)

guarded = LiveSignalRunner(strategy=strategy, risk_gate=live_gate,
                           max_turnover=0.05,
                           allow_full_turnover_from_flat=False)
gplan = guarded.plan(panel, state)
check("turnover guard scales an oversized plan down instead of discarding it",
      len(gplan.intents) > 0 and any("scaled" in w for w in gplan.warnings),
      gplan.warnings[0] if gplan.warnings else "")
check("the scaled plan's turnover respects the cap",
      gplan.turnover <= guarded.max_turnover + 1e-9,
      f"turnover={gplan.turnover:.4f} cap={guarded.max_turnover}")

# Scaling must be uniform (every surviving order shrunk by the same factor,
# not a hand-picked subset kept at full size) and must land at the cap, not
# arbitrarily below it -- the largest plan that still respects the limit.
expected_scale = guarded.max_turnover / uplan.turnover
uplan_by_symbol = {i.symbol: i for i in uplan.intents}
ratios = [gi.notional / uplan_by_symbol[gi.symbol].notional for gi in gplan.intents]
check("every surviving order is scaled by the same factor",
      all(abs(r - expected_scale) < 1e-9 for r in ratios), ratios)
check("that factor is exactly cap / uncapped turnover",
      abs(gplan.turnover - guarded.max_turnover) < 1e-9,
      f"turnover={gplan.turnover:.6f} cap={guarded.max_turnover}")

# A scaled-down order can itself fall below min_trade_notional -- the same
# filter _to_intents() already applies pre-scaling has to run again after,
# not just once up front, or a $9 leftover order would still go out.
class _FixedWeights:
    name = "fixed"
    min_history = 1

    def __init__(self, weights):
        self._weights = weights

    def target_weights(self, view, fundamentals=None, macros=None,
                       corps=None, options=None):
        return pd.Series(self._weights).reindex(view.symbols).fillna(0.0)


tiny_dates = pd.date_range("2024-01-01", periods=5, freq="D")
tiny_panel = PricePanel(close=pd.DataFrame(
    {"A": [100.0] * 5, "B": [100.0] * 5}, index=tiny_dates))
tiny_state = PortfolioState(cash=1_000.0, shares=pd.Series({"A": 0.0, "B": 0.0}))
# Uncapped: A=70%, B=30% of $1,000 -> $700/$300, turnover 100%.
# Capped at 10% with allow_full_turnover_from_flat off -> scale=0.1 ->
# A=$70, B=$30. min_trade_notional=35 keeps A, drops B.
tiny_runner = LiveSignalRunner(
    strategy=_FixedWeights({"A": 0.7, "B": 0.3}), risk_gate=None,
    min_trade_notional=35.0, max_turnover=0.10,
    allow_full_turnover_from_flat=False,
)
tiny_plan = tiny_runner.plan(tiny_panel, tiny_state)
check("a scaled order that drops below min_trade_notional is filtered out",
      [i.symbol for i in tiny_plan.intents] == ["A"],
      [(i.symbol, i.notional) for i in tiny_plan.intents])

# A plan whose turnover lands exactly on the cap must clear it, not get
# scaled/withheld -- but "exactly 0.67" is a mathematical statement, and
# sum(notional) / equity is a float computation. This specific combination
# of 7 equal-weighted symbols (0.67 / 7 each) and prices was found by
# search to genuinely round to 0.6700000000000002 in IEEE double precision
# -- 1.1e-16 above the nominal cap, not a real overage -- reproducing
# exactly the class of bug a bare `turnover > max_turnover` comparison
# would hit on real data, not a contrived exact-equality case that would
# have passed even without the fix.
boundary_prices = {"S0": 117.2, "S1": 410.39, "S2": 320.76, "S3": 468.89,
                   "S4": 305.06, "S5": 46.26, "S6": 70.98}
boundary_panel = PricePanel(close=pd.DataFrame(
    {s: [p] * 5 for s, p in boundary_prices.items()}, index=tiny_dates))
boundary_state = PortfolioState(
    cash=1_000.0, shares=pd.Series({s: 0.0 for s in boundary_prices}))
boundary_runner = LiveSignalRunner(
    strategy=_FixedWeights({s: 0.67 / 7 for s in boundary_prices}), risk_gate=None,
    min_trade_notional=0.001, max_turnover=0.67,
)
boundary_plan = boundary_runner.plan(boundary_panel, boundary_state)
raw_turnover = sum(i.notional for i in boundary_plan.intents) / 1_000.0
check("the boundary case actually reproduces a float artifact just above 0.67",
      0.67 < raw_turnover < 0.67 + 1e-9, repr(raw_turnover))
check("a plan landing a float hair above the cap is not scaled or withheld",
      len(boundary_plan.intents) == 7
      and not any("exceeds" in w for w in boundary_plan.warnings),
      boundary_plan.warnings)
check("the surviving order still respects the turnover cap",
      tiny_plan.turnover <= tiny_runner.max_turnover + 1e-9,
      f"turnover={tiny_plan.turnover}")

# Same cap, same flat state, default settings this time: the first buildout
# from cash is exempt rather than withheld.
exempt_runner = LiveSignalRunner(strategy=strategy, risk_gate=live_gate,
                                 max_turnover=0.05)
exempt_plan = exempt_runner.plan(panel, state)
check("a flat account's first buildout is exempt from the turnover cap by default",
      len(exempt_plan.intents) > 0
      and any("exempt" in w for w in exempt_plan.warnings),
      "; ".join(exempt_plan.warnings))

print()
print("=" * 72)
print("8. Edge cases")
print("=" * 72)

ledger = DayTradeLedger(limit=3, equity_threshold=25_000)
d = pd.Timestamp("2024-06-10")
for k in range(3):
    ledger.record(d + pd.Timedelta(days=k))
check("PDT ledger counts within window", ledger.count(d + pd.Timedelta(days=2)) == 3)
check("PDT blocks at limit under threshold",
      ledger.remaining(d + pd.Timedelta(days=2), 10_000) == 0)
check("PDT unrestricted above equity threshold",
      ledger.remaining(d + pd.Timedelta(days=2), 30_000) > 3)
check("PDT window rolls off",
      ledger.count(d + pd.Timedelta(days=30)) == 0)

short = panel.as_of(panel.dates[50])
r_short = Backtester(panel=short,
                     strategy=CrossSectionalMomentum(lookback=126, top_n=5),
                     rebalance="M", initial_equity=10_000).run()
check("insufficient history yields no trades, no crash",
      len(r_short.trades) == 0 and abs(r_short.equity.iloc[-1] - 10_000) < 1e-6)

sparse = panel.close.copy()
sparse.iloc[:400, :5] = np.nan
gappy = PricePanel(close=sparse, open_=panel.open_, volume=panel.volume)
r_gappy = Backtester(panel=gappy,
                     strategy=CrossSectionalMomentum(lookback=126, top_n=10),
                     risk_gate=RiskGate(), rebalance="M",
                     initial_equity=50_000).run()
check("handles symbols that start late",
      np.isfinite(r_gappy.equity).all() and (r_gappy.equity > 0).all())

check("integer share mode works",
      Backtester(panel=panel, strategy=EqualWeightBuyHold(),
                 execution=ExecutionConfig(allow_fractional=False, price="close"),
                 rebalance="M", initial_equity=50_000
                 ).run().equity.iloc[-1] > 0)

rb_m = rebalance_dates(panel.dates, "M")
rb_w = rebalance_dates(panel.dates, "W")
check("monthly schedule ~= 12/yr",
      110 <= len(rb_m) <= 125, f"{len(rb_m)} over 10y")
check("weekly schedule ~= 52/yr", 490 <= len(rb_w) <= 530, f"{len(rb_w)}")
check("all rebalance dates are real trading days",
      set(rb_m).issubset(set(panel.dates)))

# An unrecognized freq string used to silently fall back to rebalancing
# every single day instead of raising -- a plausible typo ("3M", meaning
# "every 3 months") would produce far more trading than intended, silently.
for lenient in ("Monthly", "monthly", "Q1", "Daily"):
    check(f"still accepts {lenient!r} (first-letter match)",
          len(rebalance_dates(panel.dates, lenient)) > 0)
for bad_freq in ("3M", "biweekly", "X"):
    try:
        rebalance_dates(panel.dates, bad_freq)
        check(f"rejects unrecognized freq {bad_freq!r}", False)
    except ValueError:
        check(f"rejects unrecognized freq {bad_freq!r}", True)

# A duplicated symbol column used to construct fine, survive as_of() and
# every strategy, then die deep inside Backtester.run() with pandas' own
# "cannot reindex on an axis with duplicate labels" -- naming neither the
# panel nor the offending symbol. PricePanel already validated its index
# thoroughly (sorted, no duplicate dates) but never its columns.
_dup_idx = pd.bdate_range("2020-01-01", periods=30)
_dup_close = pd.DataFrame(np.ones((30, 3)), index=_dup_idx,
                          columns=["AAA", "BBB", "AAA"])
try:
    PricePanel(close=_dup_close)
    check("rejects duplicate symbol columns at construction", False)
except ValueError as exc:
    check("rejects duplicate symbol columns at construction",
          "duplicate symbol" in str(exc) and "AAA" in str(exc), str(exc))

# `str` is an iterable of characters, so a bare string silently became a
# panel of single-character symbols. Against SyntheticRepository that was
# worse than a crash: it fabricated a plausible 4-symbol panel of generated
# prices with no error at all.
for _repo_name, _repo in (("SyntheticRepository", SyntheticRepository(n_symbols=5)),
                          ("OpenBBRepository", OpenBBRepository(cache_dir=None))):
    try:
        _repo.fetch("AAPL", "2020-01-01", "2020-03-01")
        check(f"{_repo_name} rejects a bare string for symbols", False)
    except TypeError as exc:
        check(f"{_repo_name} rejects a bare string for symbols",
              "bare string" in str(exc), str(exc))

check("a single-symbol list is still accepted",
      SyntheticRepository().fetch(["AAPL"], "2020-01-01", "2020-03-01").symbols
      == ["AAPL"])

# These caches key on the request including its end date, so a daily
# scheduled run (end = "today") writes a new entry every day and never
# reads it again -- unbounded growth at a zero hit rate on the live path.
# Pruning is by access time, so a window you keep returning to survives.
_cache_dir = tempfile.mkdtemp()
for _name, _age_days in (("stale.csv.gz", 60), ("recent.csv.gz", 1)):
    _p = os.path.join(_cache_dir, _name)
    with open(_p, "w") as _fh:
        _fh.write("x")
    _t = time.time() - _age_days * 86_400
    os.utime(_p, (_t, _t))
check("prune_cache removes entries untouched past the age limit",
      prune_cache(_cache_dir, max_age_days=30) == 1)
check("and keeps recently-accessed ones",
      sorted(os.listdir(_cache_dir)) == ["recent.csv.gz"])
check("prune_cache is a safe no-op on a missing or disabled cache dir",
      prune_cache(None) == 0 and prune_cache("/nonexistent/path/xyz") == 0)
check("a tuple of symbols is still accepted",
      SyntheticRepository().fetch(("X", "Y"), "2020-01-01", "2020-03-01").symbols
      == ["X", "Y"])
check("omitting symbols entirely still yields the default universe",
      SyntheticRepository(n_symbols=4).fetch(
          start="2020-01-01", end="2020-03-01").symbols
      == ["SYN000", "SYN001", "SYN002", "SYN003"])

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
