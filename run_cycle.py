#!/usr/bin/env python3
"""One trading cycle. Run this on a schedule; never from a notebook.

    python run_cycle.py --dry-run            # default, sends nothing
    python run_cycle.py --live --max-order 50
    python run_cycle.py --check-portfolio    # read-only: print holdings, exit

Exit codes: 0 completed, 1 aborted at preflight, 2 unresolved in-flight order
(halt and investigate before the next cycle), 3 setup or connection failure.

Why this is a separate process from the notebook: it needs a durable journal, a
deterministic single pass, and crash recovery on startup. A notebook gives you
none of those, and re-running a cell would re-submit.

**First real (non-synthetic) run needs a browser.** Authentication against
Robinhood is OAuth 2.0 Authorization Code + PKCE (see :mod:`qbt.oauth`), not a
static token: the first connection opens a browser for you to log in and
authorize once, then persists a refresh token to ``ROBINHOOD_OAUTH_STATE``
(default ``state/robinhood_oauth.json``) so every later scheduled run
refreshes silently with no browser involved -- until that refresh token
itself expires or is revoked, at which point the browser flow fires again.
That means this cannot be the very first invocation on a truly headless
remote box with no way to reach ``127.0.0.1`` in a browser; run the first
login somewhere you can open a browser (or tunnel the callback port), then
copy the resulting state file over.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from qbt import (
    BreadthRegimeFilter, CrossSectionalMomentum, DayTradeLedger,
    LiveSignalRunner, MacroRegimeFilter, MacrosPanel, MacrosRepository,
    OpenBBRepository, PortfolioState, RiskGate, SyntheticRepository,
)
from qbt.broker import MockBroker, RobinhoodMCPBroker
from qbt.macro import DEFAULT_INDICATORS
from qbt.oauth import build_robinhood_oauth
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager
from qbt.risk import as_session_date

# Single source of truth: the same timezone OrderManager resolves trading
# sessions in, so a day trade recorded during execute() and the trim below
# agree on which calendar date a fill belongs to.
MARKET_TZ = ExecutionPolicy.market_tz

# 11 GICS sector ETFs, plus small-cap/intl/EM equity, duration, gold and
# broad commodities -- breadth across asset classes, and every sleeve
# internally diversified.
ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
        "XLV", "XLY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]

# Individual issuers, spread across sectors, all continuously listed since
# 2010. Two things change by including them, both worth holding in mind:
#
# 1. **Idiosyncratic risk.** A sector ETF diversifies away single-name
#    earnings and headline risk; these do not. One of these gapping 20% on
#    a print is a real outcome that no sector sleeve can produce.
# 2. **Volatility heterogeneity.** Measured on this data, single names run
#    ~1.5x the annualised vol of the ETFs (median 24.1% vs 16.4%).
#    CrossSectionalMomentum ranks on trailing *return*, not risk-adjusted
#    return, so the higher-vol names occupy both tails of the ranking
#    disproportionately and top_n will skew toward them. RiskGate's vol
#    targeting still controls total book volatility -- this changes *which
#    names get chosen*, not how much risk the book carries.
EQUITIES = ["AAPL", "MSFT", "CSCO", "JNJ", "PFE", "JPM",
            "XOM", "CVX", "PG", "KO", "WMT", "HD"]

UNIVERSE = ETFS + EQUITIES

# Two whole-book de-risking overlays around the same momentum core, neither
# changing which names get picked -- both just scale total exposure down
# (never up) when their own regime read is unfavourable, the same role
# vol targeting plays in RiskGate:
#
# - MacroRegimeFilter(metric="vix"): elevated or sharply-rising implied
#   volatility. max_level=35 is "real stress" territory (VIX's ordinary
#   range is roughly 12-20; 30+ is a genuine risk-off regime, not routine
#   noise). max_increase=15 over 21 trading days (~1 month) catches a fast
#   spike even before the absolute level crosses 35 -- 2018-Q4 and
#   2020-Q1 both moved VIX by more than that in under a month.
# - BreadthRegimeFilter: participation *within this strategy's own
#   18-symbol universe* -- fewer than 30% of the sleeves above their own
#   200-day average (the same lookback TrendFilter uses) is a narrow,
#   fragile tape, independent of what the VIX-based read says.
#
# scale_when_blocked=0.5 on both, not a full flatten -- de-risk, don't
# bet the regime read is certainly right, and if both fire at once the
# combined 0.5 x 0.5 = 25% exposure is a real, but not total, retreat.
STRATEGY = BreadthRegimeFilter(
    inner=MacroRegimeFilter(
        inner=CrossSectionalMomentum(lookback=63, skip=5, top_n=5),
        metric="vix", max_level=35.0, max_increase=15.0, lookback=21,
        scale_when_blocked=0.5,
        # VIX is a daily series with a 1-day publication lag, so anything
        # older than about a week means the feed is broken, not quiet.
        # Past that, treat it as no reading at all (a no-op pass-through)
        # rather than gating a live book on a stale number believing it
        # current -- a dead FRED key or a frozen cache would otherwise go
        # on de-risking, or not de-risking, on last month's volatility.
        max_age_days=7,
    ),
    lookback=200, min_breadth=0.3, scale_when_blocked=0.5,
)
GATE = dict(target_vol=0.12, max_weight=0.30, max_gross=1.0, max_drawdown=0.25)


def mode_path(base: str, synthetic: bool) -> str:
    """Suffix a state/audit path by run mode -- e.g. "state/peak_equity.txt"
    -> "state/peak_equity_live.txt" or "..._synthetic.txt".

    Synthetic and real runs must never share persisted state. This was a
    real, confirmed bug: a ``--synthetic`` run's $25,000 MockBroker starting
    cash wrote to the same peak-equity file a real ~$1,000 account later
    read back. ``load_peak`` takes ``max(stored, current)``, which can only
    ratchet the peak up, never self-correct down -- so the real account's
    peak got stuck at $25,000, its apparent drawdown read as 96%, and the
    drawdown breaker inside :class:`~qbt.risk.RiskGate` silently zeroed the
    entire book on the very first live cycle. Applies to the order journal
    and audit log too, for the same reason: a synthetic run's fake fills
    have no business in the same journal `recover()` reads to decide what a
    real crash left in flight.
    """
    root, ext = os.path.splitext(base)
    return f"{root}_{'synthetic' if synthetic else 'live'}{ext}"


def load_peak(path: str, current: float) -> float:
    """The running equity peak must survive restarts.

    If this is lost, the drawdown breaker forgets the high-water mark. If it is
    seeded too high, the breaker trips immediately and the bot silently never
    trades. Both failures are quiet, so persist it -- see :func:`mode_path`
    for why ``path`` must never be shared between synthetic and real runs.
    """
    try:
        with open(path) as fh:
            return max(float(fh.read().strip()), current)
    except (OSError, ValueError):
        return current


def save_peak(path: str, value: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{value:.4f}")
        fh.flush()
        os.fsync(fh.fileno())


def load_day_trade_ledger(path: str) -> DayTradeLedger:
    """Persisted rolling day-trade log -- survives restarts, the same
    reason peak_equity does (see :func:`load_peak`).

    Confirmed live (2026-08): without this, every process invocation
    constructs a fresh, empty ``DayTradeLedger``, so ``RiskGate``'s PDT
    check can never actually trigger in live trading -- nothing remembers
    a day trade past the process that recorded it. See :func:`mode_path`
    for why ``path`` must never be shared between synthetic and real runs.
    """
    ledger = DayTradeLedger()
    try:
        with open(path) as fh:
            data = json.load(fh)
        ledger.events = [pd.Timestamp(d) for d in data.get("events", [])]
    except (OSError, ValueError, TypeError):
        pass
    return ledger


def save_day_trade_ledger(path: str, ledger: DayTradeLedger) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Trim to what the rolling window could still matter for, so this file
    # doesn't grow forever -- a generous 10 business days; the precise
    # 5-business-day cutoff is DayTradeLedger.count()'s own job.
    #
    # Both sides go through as_session_date() so this comparison stays
    # naive-vs-naive. It used to build a tz-aware UTC cutoff, which is the
    # mirror image of the bug that as_session_date() exists to close: once
    # events are stored as tz-naive session dates, an aware cutoff here
    # raises the same TypeError from the other direction, and it would do
    # it on the *write* path, right after a day trade was recorded.
    cutoff = as_session_date(pd.Timestamp.now(tz=MARKET_TZ)) - pd.tseries.offsets.BDay(10)
    events = [as_session_date(d) for d in ledger.events]
    events = [d for d in events if d > cutoff]
    with open(path, "w") as fh:
        json.dump({"events": [d.isoformat() for d in events]}, fh)
        fh.flush()
        os.fsync(fh.fileno())


def check_portfolio(args: argparse.Namespace) -> int:
    """Read-only: connect, fetch the agentic account's current holdings,
    print them, exit. No price panel, no strategy, no risk gate, no
    journal or audit writes -- just the same broker connection and
    get_account() call every real cycle already makes, surfaced on its own
    so you can see what's actually held without running (or dry-running)
    a full cycle.
    """
    try:
        if args.synthetic:
            broker = MockBroker(prices=pd.Series({"XLB": 100.0}), cash=25_000.0, seed=0)
        else:
            oauth = build_robinhood_oauth(
                storage_path=os.environ.get(
                    "ROBINHOOD_OAUTH_STATE", "state/robinhood_oauth.json"
                ),
                port=int(os.environ.get("ROBINHOOD_OAUTH_CALLBACK_PORT", "8765")),
            )
            broker = RobinhoodMCPBroker(auth=oauth, require_agentic=True)
        broker.connect()
        account = broker.get_account()
    except Exception as exc:
        print(f"FAILED to connect or read the account: {exc!r}")
        return 3

    print(f"account:         {account.account_id} "
          f"({'agentic' if account.is_agentic else 'NOT agentic'})")
    print(f"equity:          ${account.equity:,.2f}")
    print(f"cash:            ${account.cash:,.2f}")
    print(f"buying power:    ${account.buying_power:,.2f}")
    if account.day_trades_used is not None:
        print(f"day trades used: {account.day_trades_used}")

    held = account.positions[account.positions.abs() > 1e-9]
    print()
    if held.empty:
        print("No open positions -- fully in cash.")
        broker.close()
        return 0

    try:
        prices = broker.get_quotes(list(held.index))
    except Exception as exc:
        print(f"(could not fetch quotes for position values: {exc!r})")
        prices = pd.Series(dtype=float)

    weights = account.weights(prices) if not prices.empty else pd.Series(dtype=float)
    rows = []
    for sym, shares in held.sort_values(ascending=False).items():
        price = prices.get(sym, float("nan"))
        rows.append({
            "symbol": sym,
            "shares": round(float(shares), 6),
            "price": f"${price:,.2f}" if pd.notna(price) else "n/a",
            "value": f"${shares * price:,.2f}" if pd.notna(price) else "n/a",
            "weight": f"{weights.get(sym, float('nan')):.1%}"
                     if sym in weights.index and pd.notna(weights.get(sym))
                     else "n/a",
        })
    print(pd.DataFrame(rows).set_index("symbol").to_string())
    broker.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually submit orders (default is dry run)")
    # $500 clears one whole share of every name in UNIVERSE (the tallest are
    # GLD ~$392 and IWM ~$301), and clears the largest single position the
    # risk layer will ask for on a ~$1k account -- GATE's max_weight=0.30 is
    # ~$304, ExecutionPolicy's max_position_weight=0.35 is ~$354. Below that
    # the cap and the weight limits disagree: a position the gate permits
    # can't be built in one trade, and a whole-share-only instrument whose
    # share costs more than the cap can't be entered at all (see the
    # whole-share retry's own re-check in OrderManager.execute).
    ap.add_argument("--max-order", type=float, default=500.0)
    ap.add_argument("--max-plan", type=float, default=5_000.0)
    ap.add_argument("--max-turnover", type=float, default=0.67)
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data and a mock broker")
    ap.add_argument("--ignore-market-hours", action="store_true")
    ap.add_argument("--check-portfolio", action="store_true",
                    help="fetch and print the agentic account's current "
                         "holdings, then exit -- no planning or trading")
    args = ap.parse_args()

    if args.check_portfolio:
        return check_portfolio(args)

    audit = AuditLog(mode_path("audit/orders.jsonl", args.synthetic))
    audit.emit("cycle_start", live=args.live, argv=" ".join(sys.argv[1:]))

    # ---- data -----------------------------------------------------------
    try:
        end = str(pd.Timestamp.today().normalize().date())
        if args.synthetic:
            panel = SyntheticRepository(n_symbols=18, seed=42).fetch(
                start="2010-01-01", end=end)
        else:
            panel = OpenBBRepository(provider="yfinance",
                                     cache_dir=".cache/prices").fetch(
                UNIVERSE, "2010-01-01", end)
    except Exception as exc:
        audit.emit("data_fetch_failed", error=repr(exc))
        return 3

    # ---- macro (vix, for the MacroRegimeFilter wrapped around STRATEGY) -
    # Deliberately its own try/except, separate from the price fetch above:
    # unlike price data, this is a risk *overlay*, not something the
    # strategy strictly needs to function -- MacroRegimeFilter already
    # treats macros=None as a no-op pass-through by design (see its own
    # docstring), so a FRED outage or a missing API key should degrade
    # today's cycle back to breadth-only risk management, not abort a
    # trading day over an optional signal.
    try:
        if args.synthetic:
            # No real VIX series to fetch offline -- a flat, calm-level
            # synthetic one exercises the wiring (MacroRegimeFilter
            # actually receiving and reading a panel) without claiming to
            # be real data.
            macros = MacrosPanel(frame=pd.DataFrame(
                [("vix", d, d, 15.0) for d in panel.dates],
                columns=["metric", "period_end", "as_of_date", "value"],
            ))
        else:
            macros = MacrosRepository(
                indicators={"vix": DEFAULT_INDICATORS["vix"]},
                cache_dir=".cache/macro",
            ).fetch("2010-01-01", end)
    except Exception as exc:
        audit.emit("macro_fetch_failed", error=repr(exc))
        print(f"  (macro/VIX fetch failed, continuing without it: {exc!r})")
        macros = None

    # ---- broker ---------------------------------------------------------
    broker = None
    try:
        if args.synthetic:
            broker = MockBroker(prices=panel.last_close(), cash=25_000.0, seed=0)
        else:
            oauth = build_robinhood_oauth(
                storage_path=os.environ.get(
                    "ROBINHOOD_OAUTH_STATE", "state/robinhood_oauth.json"
                ),
                port=int(os.environ.get("ROBINHOOD_OAUTH_CALLBACK_PORT", "8765")),
            )
            broker = RobinhoodMCPBroker(auth=oauth, require_agentic=True)
        broker.connect()
        # Resolve the account now, before recovery -- recover() reads the
        # broker's orders (get_orders()), which requires account_number on
        # the real API the same as review/place/cancel do. get_account() is
        # what caches self.account_id on the broker; recover() running
        # first (it has to -- unresolved in-flight orders must be settled
        # before a new plan is built) needs that already done. The plan
        # step below reuses this same `account` rather than calling
        # get_account() a second time.
        account = broker.get_account()
    except Exception as exc:
        audit.emit("broker_connect_failed", error=repr(exc))
        # broker may or may not exist yet -- oauth/RobinhoodMCPBroker
        # construction itself can raise before broker is ever assigned,
        # and `broker = None` above is exactly what makes this safe to
        # check rather than risking a NameError on top of the original
        # failure.
        if broker is not None:
            broker.close()
        return 3

    policy = ExecutionPolicy(
        max_order_notional=args.max_order,
        max_plan_notional=args.max_plan,
        max_plan_turnover=args.max_turnover,
        symbol_allowlist=tuple(panel.symbols) if args.synthetic else tuple(UNIVERSE),
        require_review=True,
        require_market_open=not args.ignore_market_hours,
        dry_run=not args.live,
    )
    day_trades_path = mode_path("state/day_trades.json", args.synthetic)
    ledger = load_day_trade_ledger(day_trades_path)
    manager = OrderManager(
        broker=broker, policy=policy, audit=audit,
        journal_path=mode_path("audit/journal.jsonl", args.synthetic),
        day_trade_ledger=ledger,
    )

    # ---- recovery before anything else ----------------------------------
    try:
        unresolved = manager.recover()
    except Exception as exc:
        audit.emit("recover_failed", error=repr(exc))
        broker.close()
        return 2
    if not unresolved.empty:
        lost = unresolved[unresolved["outcome"] == "not_at_broker"]
        if not lost.empty:
            audit.emit("halt_unresolved_orders", n=len(lost),
                       symbols=", ".join(lost["symbol"]))
            print("HALT: in-flight orders could not be accounted for.")
            print(unresolved.to_string(index=False))
            broker.close()
            return 2

    # ---- plan -----------------------------------------------------------
    peak_file = mode_path("state/peak_equity.txt", args.synthetic)
    try:
        peak = load_peak(peak_file, account.equity)

        # Anything held that the price panel doesn't cover is capital this
        # strategy does not manage: reindexing it away below (which we still
        # have to do -- there's no price series to size or trade it with)
        # makes it contribute nothing to plan.equity, produces no sell
        # intent, and leaves it stranded indefinitely. Silent in every
        # direction, so say it out loud here, while account.positions is
        # still the broker's complete view.
        #
        # Deliberately not fatal on its own. A small unmanaged holding just
        # means plan.equity understates the account, which sizes orders
        # conservatively -- safe. A large one trips preflight's existing
        # equity-drift check and aborts the cycle, which is the right
        # outcome; the point of this block is that the abort then has an
        # accurate explanation attached instead of surfacing as a bare
        # "equity drift ... Recompute, do not send" that points at stale
        # data rather than at an unmanaged position.
        held_all = account.positions[account.positions.abs() > 1e-9]
        unmanaged = held_all[~held_all.index.isin(panel.symbols)]
        if not unmanaged.empty:
            names = ", ".join(f"{s} x{q:g}" for s, q in unmanaged.items())
            print(f"  UNMANAGED HOLDINGS: {names}")
            print("    Not in the strategy universe -- excluded from equity, "
                  "never traded, and never sold by this bot.")
            print("    Sell or add to UNIVERSE; if large enough, preflight "
                  "will abort on equity drift until you do.")
            audit.emit("unmanaged_holdings", symbols=", ".join(unmanaged.index),
                       n=int(len(unmanaged)))

        state = PortfolioState(
            cash=account.cash,
            shares=account.positions.reindex(panel.symbols).fillna(0.0),
            peak_equity=peak,
        )
        # Same cap ExecutionPolicy enforces below, not a looser multiple of
        # it -- the two used to disagree (this one 1.5x looser), so a plan
        # that cleared this check unscaled would still turn around and get
        # hard-aborted by ExecutionPolicy's own turnover check in
        # OrderManager.preflight(), the exact thing the scale-down logic in
        # LiveSignalRunner.plan() exists to avoid. Keeping both aligned
        # means that check now does what it was actually meant to do: a
        # rarely-firing safety net for equity drift between planning and
        # submission, not the real enforcement point.
        plan = LiveSignalRunner(strategy=STRATEGY, risk_gate=RiskGate(**GATE),
                                max_turnover=args.max_turnover,
                                day_trade_ledger=ledger).plan(
            panel, state, macros=macros)

        # Surface *why* the plan looks the way it does -- these were
        # computed but never printed anywhere, which is exactly how the
        # peak-equity bug above went undetected: the risk gate's own
        # explanation ("drawdown breaker tripped at 96.0%") was sitting
        # right there in plan.decision.notes the whole time.
        if plan.warnings:
            for w in plan.warnings:
                print(f"  PLAN WARNING: {w}")
            audit.emit("plan_warnings", warnings="; ".join(plan.warnings))
        if plan.decision is not None and plan.decision.notes:
            for n in plan.decision.notes:
                print(f"  RISK GATE: {n}")
            audit.emit("risk_gate_notes", notes="; ".join(plan.decision.notes))

        # BreadthRegimeFilter/MacroRegimeFilter just scale target_weights()
        # down silently -- neither goes through plan.warnings, so without
        # this a de-risk from either would show up only as unexplained
        # smaller position sizes. Same view LiveSignalRunner.plan() used
        # internally, reconstructed here purely for this diagnostic.
        regime_view = panel.as_of(plan.asof)
        regime_macros = macros.as_of(plan.asof) if macros is not None else None
        breadth_filter = STRATEGY
        macro_filter = STRATEGY.inner
        if breadth_filter.blocked(regime_view):
            print(f"  REGIME: market breadth below {breadth_filter.min_breadth:.0%} "
                  f"-- book scaled to {breadth_filter.scale_when_blocked:.0%} "
                  f"(breadth={breadth_filter.breadth(regime_view):.0%})")
            audit.emit("breadth_regime_blocked",
                       breadth=round(breadth_filter.breadth(regime_view), 4),
                       scale=breadth_filter.scale_when_blocked)
        if macro_filter.blocked(regime_view, regime_macros):
            print(f"  REGIME: VIX regime unfavourable -- book scaled to "
                  f"{macro_filter.scale_when_blocked:.0%}")
            audit.emit("macro_regime_blocked", metric=macro_filter.metric,
                       scale=macro_filter.scale_when_blocked)

        report = manager.execute(plan, strategy_name=STRATEGY.name)
    except Exception as exc:
        audit.emit("cycle_error", error=repr(exc))
        broker.close()
        return 3

    save_peak(peak_file, max(peak, account.equity))
    save_day_trade_ledger(day_trades_path, ledger)
    print(report)
    if not report.to_frame().empty:
        print(report.to_frame().to_string(index=False))
    if report.reconciliation is not None and report.reconciliation["breach"].any():
        print("\nRECONCILIATION DRIFT:")
        print(report.reconciliation[report.reconciliation["breach"]].to_string())

    audit.emit("cycle_end", aborted=report.aborted,
               submitted=len(report.submitted), rejected=len(report.rejected),
               skipped=len(report.skipped))
    broker.close()
    return 1 if report.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
