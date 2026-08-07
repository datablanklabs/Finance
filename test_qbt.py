"""Validation suite. The null test is the one that matters."""

import numpy as np
import pandas as pd

from qbt import (
    Backtester, Composite, CostModel, CrossSectionalMomentum, DayTradeLedger,
    EqualWeightBuyHold, ExecutionConfig, InverseVolWeighted, LiveSignalRunner,
    PortfolioState, PricePanel, RiskGate, ShortHorizonReversal,
    SyntheticRepository, TimeSeriesMomentum, TrendFilter, compare, ic_grid,
    ic_summary, information_coefficient, expected_max_sharpe, forward_returns,
    rebalance_dates, return_autocorrelation, trailing_signal,
    walk_forward_splits, ParameterSweep, sharpe_haircut,
)
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

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
