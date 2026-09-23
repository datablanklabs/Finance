#!/usr/bin/env python3
"""One trading cycle. Run this on a schedule; never from a notebook.

    python run_cycle.py --dry-run            # default, sends nothing
    python run_cycle.py --live --max-order 50
    python run_cycle.py --check-portfolio    # read-only: print holdings, exit
    python run_cycle.py --consider-crypto    # also run the crypto sleeve

Exit codes: 0 completed, 1 aborted at preflight, 2 unresolved in-flight order
(halt and investigate before the next cycle), 3 setup or connection failure.
With ``--consider-crypto`` the equity and crypto sleeves each produce one of
these independently (see "Crypto support" in the README) and the worse of
the two becomes the process exit code, in that same 3 > 2 > 1 > 0 order --
one sleeve halting does not stop the other from running.

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
    BreadthRegimeFilter, Composite, CrossSectionalMomentum, DayTradeLedger,
    FundamentalsPanel, FundamentalsRepository, KalshiEventRegimeFilter,
    KalshiPanel, KalshiRepository, LiveSignalRunner,
    MacroRegimeFilter, MacrosPanel, MacrosRepository, MultiFactorCrossSectional,
    OpenBBRepository, PortfolioState, RiskGate, SyntheticRepository,
)
from qbt.live import LivePlan
from qbt.broker import MockBroker, RobinhoodMCPBroker
from qbt.macro import DEFAULT_INDICATORS
from qbt.oauth import build_robinhood_oauth
from qbt.orders import AuditLog, ExecutionPolicy, OrderManager, durable_write
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

# Robinhood-tradeable major coins, "BTC-USD" yfinance-ticker form (see
# OpenBBRepository's asset_class="crypto" docstring for why that form, not
# "BTC"). **Unverified against a live account** -- this is a conservative,
# widely-liquid starter list, not a confirmed enumeration of what Robinhood
# actually lists (there is no discovery for "which coins can this account
# trade" the way there is for MCP tool names; the closest thing is watching
# which of these a real crypto_quotes/crypto_positions call actually prices).
# A symbol Robinhood doesn't support gets no quote back and is simply never
# tradeable (_tradeable() in qbt/signals.py requires a real price) -- overly
# generous is safe, overly narrow just means missing an opportunity, so this
# leans generous. Trim or extend freely; nothing else depends on this list's
# exact membership.
CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "LTC-USD",
          "BCH-USD", "AVAX-USD", "SHIB-USD", "XRP-USD", "ADA-USD",
          "LINK-USD", "UNI-USD", "AAVE-USD", "ETC-USD", "XLM-USD"]

# Three whole-book de-risking overlays around the same momentum core, none
# changing which names get picked -- each just scales total exposure down
# (never up) when its own regime read is unfavourable, the same role
# vol targeting plays in RiskGate:
#
# - MacroRegimeFilter(metric="vix"): elevated or sharply-rising implied
#   volatility. max_level=35 is "real stress" territory (VIX's ordinary
#   range is roughly 12-20; 30+ is a genuine risk-off regime, not routine
#   noise). max_increase=15 over 21 trading days (~1 month) catches a fast
#   spike even before the absolute level crosses 35 -- 2018-Q4 and
#   2020-Q1 both moved VIX by more than that in under a month.
# - KalshiEventRegimeFilter: forward-looking counterpart to the VIX read --
#   de-risks heading into a scheduled CPI/FOMC/payrolls print the market
#   itself hasn't converged on, rather than reacting to volatility already
#   realised. See qbt/kalshi.py's module docstring for the live-liquidity
#   survey and why its per-series confidence floors differ.
# - BreadthRegimeFilter: participation *within this strategy's own
#   UNIVERSE* (18 ETFs + 12 single names, 30 symbols) -- fewer than 30% of
#   the sleeves above their own 200-day average (the same lookback
#   TrendFilter uses) is a narrow, fragile tape, independent of what either
#   the VIX- or Kalshi-based reads say.
#
# scale_when_blocked=0.5 on all three, not a full flatten -- de-risk, don't
# bet any one regime read is certainly right, and if more than one fires at
# once the combined scale (e.g. 0.5 x 0.5 = 25% exposure with two, 12.5%
# with all three) is a real, but not total, retreat.
#
# The core itself is a two-member Composite, not a bare CrossSectionalMomentum,
# for the same reason the overlays above scale rather than flatten: one
# alpha source is one bet on one regime working. 60% stays the momentum
# strategy that's actually been through the null test (test_qbt.py section
# 2) the longest and has run live the longest; 40% is
# MultiFactorCrossSectional blending its own price-derived momentum/low-vol/
# reversal factors with a fundamentals quality factor (return on equity),
# so the sleeve isn't just momentum measured a second way. Both members are
# now covered by the same null test as the momentum core (section 2 of
# test_qbt.py extends to all 10 strategies, this composition included) --
# see the README's "Known gaps" for what that test can and can't tell you.
# Capital share, not name overlap: nothing stops both members from picking
# the same symbol, in which case its total weight is just the sum of what
# each sleeve independently assigned it.
STRATEGY = BreadthRegimeFilter(
    inner=MacroRegimeFilter(
        inner=KalshiEventRegimeFilter(
            inner=Composite(
                members=[
                    (CrossSectionalMomentum(lookback=63, skip=5, top_n=5), 0.6),
                    (
                        MultiFactorCrossSectional(
                            momentum_lookback=126, momentum_skip=5,
                            vol_lookback=63, reversal_lookback=5,
                            factor_weights={
                                "momentum": 0.3, "low_vol": 0.3,
                                "reversal": 0.1, "quality": 0.3,
                            },
                            # Verify against fundamentals.metrics once openbb-fmp
                            # is actually pulling live data -- see the fundamentals
                            # fetch below and FundamentalsValueFilter's own
                            # docstring for the same caveat: this name is what the
                            # FMP `ratios` statement is expected to produce, not
                            # independently confirmed against a live response the
                            # way the Robinhood broker's response shapes are.
                            quality_metric="ratios_return_on_equity",
                            top_n=5,
                        ),
                        0.4,
                    ),
                ],
                name="momentum_quality_blend",
            ),
            # 3 trading days: close enough to a scheduled CPI/FOMC/payrolls
            # print that the de-risk actually covers the release itself
            # (these all close within a day or two of 8:25-8:30am ET on the
            # data day), not so wide that a routine month-long run-up to the
            # next print gets caught too. scale_when_blocked matches the VIX
            # overlay below rather than flattening outright, for the same
            # "don't bet the regime read is certainly right" reasoning.
            horizon_days=3, scale_when_blocked=0.5,
            # Kalshi prices continuously, so a reading older than a few days
            # means the feed died, not that nothing changed -- same
            # reasoning as max_age_days below, tighter because there's no
            # equivalent of a monthly release cadence excusing a stale read.
            max_age_days=3,
        ),
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
# The live risk policy, complete as written. --max-position overrides
# max_weight for a single run (and is what --help reports as its default), but
# this dict always describes the configured policy on its own: leaving
# max_weight out and filling it in only inside main() meant RiskGate(**GATE)
# from anywhere else -- a research probe, a notebook, a test -- silently fell
# back to RiskGate's own 0.25 default instead of the 0.30 configured here.
GATE = dict(target_vol=0.12, max_weight=0.30, max_gross=1.0, max_drawdown=0.25)

# How far back the live cycle pulls Kalshi quotes. This bounds *history*,
# not which events are seen: every still-open market (next month's ladders,
# future FOMC meetings) is fetched whatever this is, since the /markets
# filter only drops markets that closed before the window start.
# KalshiEventRegimeFilter reads just the latest quote of the nearest open
# event per series, and open markets carry a daily bid/ask row even on
# no-trade days, so a couple of weeks is ample. Longer only pulls settled
# ladders nothing live reads -- extra requests against Kalshi's rate limit.
KALSHI_LOOKBACK_DAYS = 14

# How much looser ExecutionPolicy's backstop sits than the gate that actually
# shapes the book. The two caps are not redundant, they are layered:
# RiskGate.max_weight *clips* proposed weights, so under normal operation no
# target ever reaches the backstop; ExecutionPolicy.max_position_weight is a
# preflight fatal that only fires if something upstream misbehaved, plus the
# re-check on the whole-share retry in OrderManager.execute (which preflight
# cannot see, because it inspects target weights rather than retried
# quantities). Moving them together preserves that gap -- setting the backstop
# equal to the gate would leave nothing to back up.
POSITION_BACKSTOP_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Crypto sleeve -- opt-in via --consider-crypto, independent of everything
# above. See "Crypto support" in the README for the full rationale; the
# short version of what's different from STRATEGY/GATE and why:
#
# - Same two-member Composite core (60% CrossSectionalMomentum, 40%
#   MultiFactorCrossSectional), same three regime overlays -- these are
#   price-derived and VIX is a general risk-off gauge, not equity-specific,
#   so both transfer. The one thing that does NOT transfer is the quality
#   factor: it reads FundamentalsPanel, and there is no such thing as an SEC
#   filing or a return-on-equity figure for a coin. Dropping it here (rather
#   than leaving factor_weights unchanged and letting fundamentals=None
#   silently zero it, the way MultiFactorCrossSectional already tolerates)
#   means the other three factors' weights actually sum to what they claim
#   to, instead of 60% of the intended weight quietly doing 100% of the work.
#   KalshiEventRegimeFilter is present in the structure below for the same
#   reason MacroRegimeFilter is -- but see run_crypto_pipeline(), which
#   passes kalshi=None here for now, same open question as macros=None:
#   whether a scheduled 8:30am ET CPI/FOMC print is a de-risk trigger for a
#   market that trades 24/7 the same way it is for equities is a second,
#   independent judgement call this file isn't taking a position on yet.
# - max_weight/max_drawdown are tighter than GATE's: crypto's realised vol
#   typically runs several times an equity sector ETF's, so target_vol=0.12
#   (unchanged -- let RiskGate's existing vol-target scaling do its job
#   proportionally, the same mechanism that already handles "this book is
#   too volatile for the target") will usually scale the book down hard on
#   its own, but max_weight/max_drawdown are a second, independent backstop
#   against concentration and gap risk that don't depend on the vol
#   forecast being right.
CRYPTO_STRATEGY = BreadthRegimeFilter(
    inner=MacroRegimeFilter(
        inner=KalshiEventRegimeFilter(
            inner=Composite(
                members=[
                    (CrossSectionalMomentum(lookback=63, skip=5, top_n=5), 0.6),
                    (
                        MultiFactorCrossSectional(
                            momentum_lookback=126, momentum_skip=5,
                            vol_lookback=63, reversal_lookback=5,
                            factor_weights={
                                "momentum": 0.5, "low_vol": 0.3, "reversal": 0.2,
                            },
                            top_n=5,
                        ),
                        0.4,
                    ),
                ],
                name="crypto_momentum_blend",
            ),
            horizon_days=3, scale_when_blocked=0.5, max_age_days=3,
        ),
        metric="vix", max_level=35.0, max_increase=15.0, lookback=21,
        scale_when_blocked=0.5, max_age_days=7,
    ),
    lookback=200, min_breadth=0.3, scale_when_blocked=0.5,
)
CRYPTO_GATE = dict(target_vol=0.12, max_weight=0.20, max_gross=1.0, max_drawdown=0.20)
CRYPTO_POSITION_BACKSTOP_MARGIN = 0.05


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
    durable_write(path, f"{value:.4f}")


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
    durable_write(path, json.dumps({"events": [d.isoformat() for d in events]}))


def apply_price_cap(panel, max_price: float | None, label: str, audit: AuditLog):
    """Drop symbols whose latest close exceeds ``max_price`` from ``panel``
    entirely, so the strategy never ranks or picks them -- see --max-price's
    own help text and the README's "The per-order cap and whole-share
    instruments" for the specific small-account failure mode this exists to
    prevent (one expensive name's forced whole-share buy crowding out the
    rest of the book).

    ``max_price=None`` (the default) is a true no-op -- returns ``panel``
    unchanged, not a copy, so every existing caller is unaffected.

    A NaN last close (a symbol with no live price yet) is kept, not
    excluded: ``price > max_price`` is False for NaN either way, so this
    would happen automatically, but the point is that "unknown" and
    "expensive" are different things, and only one of them is this
    function's job -- a symbol with no price is already excluded from
    trading by _tradeable()'s own history/liveness check in qbt/signals.py,
    for its own, different reason.
    """
    if max_price is None:
        return panel
    last = panel.last_close()
    expensive = sorted(last[last > max_price].index)
    if expensive:
        affordable = [s for s in panel.symbols if s not in expensive]
        print(f"  {label} price cap ${max_price:,.2f}: excluding "
              f"{', '.join(expensive)}")
        audit.emit("price_cap_excluded", sleeve=label, max_price=max_price,
                   symbols=", ".join(expensive), n=len(expensive))
        panel = panel.select(affordable)
    return panel


def report_trimmed_bars(repo, label: str, audit: AuditLog) -> None:
    """Surface any incomplete trailing bars ``repo``'s last fetch dropped
    (see qbt.data.trim_incomplete_tail) in the audit log and the console --
    otherwise the only trace a cycle ran on yesterday's closes would be a
    Python warning and a slightly older ``asof``.
    """
    for date, symbols in repo.last_trimmed:
        print(f"  {label} prices: dropped incomplete bar {date.date()} "
              f"(no close for {len(symbols)} symbol(s)) -- planning on the "
              f"last complete bar instead")
        audit.emit("price_bar_trimmed", sleeve=label, date=str(date.date()),
                   symbols=", ".join(symbols), n=len(symbols))


def _print_holdings(broker_view, label: str) -> None:
    """The read side of check_portfolio(), for one broker view (equity or
    crypto). Shared so --consider-crypto prints a second, clearly-labelled
    section rather than duplicating this block.
    """
    try:
        account = broker_view.get_account()
    except Exception as exc:
        print(f"FAILED to read the {label} account: {exc!r}")
        return

    print(f"--- {label} ---")
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
        return

    try:
        prices = broker_view.get_quotes(list(held.index))
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


def check_portfolio(args: argparse.Namespace) -> int:
    """Read-only: connect, fetch the agentic account's current holdings,
    print them, exit. No price panel, no strategy, no risk gate, no
    journal or audit writes -- just the same broker connection and
    get_account() call every real cycle already makes, surfaced on its own
    so you can see what's actually held without running (or dry-running)
    a full cycle. With --consider-crypto, also prints the crypto sleeve
    (see qbt.broker.RobinhoodMCPBroker.crypto_view) -- still read-only,
    still no journal or audit writes.
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
    except Exception as exc:
        print(f"FAILED to connect: {exc!r}")
        return 3

    _print_holdings(broker, "equity")
    if args.consider_crypto:
        print()
        if args.synthetic:
            # No crypto-specific MockBroker view to wrap (MockBroker is
            # already asset-class-agnostic -- see run_pipeline's own
            # synthetic-mode crypto setup) -- reuse the same instance, it
            # just reports the same (equity-labelled, for this read-only
            # check) holdings a second time rather than fabricating a
            # separate crypto position set with nothing behind it.
            _print_holdings(broker, "crypto (synthetic -- same mock account)")
        else:
            try:
                broker.require_crypto()
                _print_holdings(broker.crypto_view(), "crypto")
            except Exception as exc:
                print(f"--- crypto ---\nFAILED: {exc!r}")
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
    ap.add_argument("--max-position", type=float, default=GATE["max_weight"],
                    # argparse runs help through %-formatting, so a literal
                    # percent sign has to be doubled or it swallows the rest
                    # of the string and prints the raw action dict instead.
                    help="max single-name weight as a fraction, e.g. 0.30 for "
                         "30%%; the ExecutionPolicy backstop is set "
                         f"{POSITION_BACKSTOP_MARGIN:.2f} above it "
                         "(default: 0.30)")
    # Addresses a specific, documented failure mode of a small account
    # (see "The per-order cap and whole-share instruments" in the README):
    # a name whose *single share* costs a large fraction of the intended
    # position -- GLD ~$392, IWM ~$301 in UNIVERSE -- forces the whole-share
    # retry in OrderManager.execute() to round a brand-new position up to
    # one full share regardless of the target weight, which can eat most of
    # a small account's remaining capital and leave the other picks
    # unfunded. Filtering the symbol out of the panel entirely (not just
    # capping the order) is what actually fixes this: the strategy never
    # ranks or picks an unaffordable name in the first place, so nothing
    # gets crowded out by a forced oversized buy. None (the default) is a
    # no-op -- every existing behaviour is unchanged unless this is set.
    # An already-held position above the cap is not sold; it becomes an
    # "unmanaged holding" (same diagnostic and equity-drift backstop as a
    # position outside UNIVERSE) until sold by hand or the cap is raised.
    ap.add_argument("--max-price", type=float, default=None,
                    help="exclude any symbol whose latest close exceeds "
                         "this many dollars per share from the tradeable "
                         "universe entirely (default: no cap)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data and a mock broker")
    ap.add_argument("--ignore-market-hours", action="store_true")
    ap.add_argument("--check-portfolio", action="store_true",
                    help="fetch and print the agentic account's current "
                         "holdings, then exit -- no planning or trading")
    # Crypto sleeve -- see CRYPTO_STRATEGY/CRYPTO_GATE above and "Crypto
    # support" in the README. Off by default: the crypto MCP tool surface
    # is unverified (see qbt/broker.py's module docstring), so this only
    # ever runs when explicitly asked for, and every equity flag above is
    # completely unaffected by its presence or absence either way.
    # Smaller defaults than the equity flags on purpose -- this is a new,
    # unverified pipeline, so it starts at the small end of "Going live, in
    # order" (see the README) rather than inheriting the equity sizing that
    # took months of confirmed live behaviour to justify.
    ap.add_argument("--consider-crypto", action="store_true",
                    help="also run the crypto sleeve (CRYPTO_STRATEGY) as a "
                         "second, independent plan/execute pass")
    ap.add_argument("--max-crypto-order", type=float, default=100.0)
    ap.add_argument("--max-crypto-plan", type=float, default=1_000.0)
    ap.add_argument("--max-crypto-turnover", type=float, default=0.67)
    ap.add_argument("--max-crypto-position", type=float,
                    default=CRYPTO_GATE["max_weight"],
                    help="max single-coin weight as a fraction (default: "
                         f"{CRYPTO_GATE['max_weight']:.2f})")
    # Same idea as --max-price, for the crypto sleeve. Less likely to bind
    # in practice -- crypto orders are fractional-friendly, so there's no
    # whole-share-retry forcing an oversized buy the way there is for
    # equities -- but a per-coin price ceiling is still a reasonable thing
    # to want on a small account (e.g. excluding BTC's four-to-six-figure
    # per-coin price from a $1k crypto sleeve for the same "one name
    # shouldn't be able to dominate" reasoning, even without the forced-
    # rounding mechanism that makes it a hard requirement for equities).
    ap.add_argument("--max-crypto-price", type=float, default=None,
                    help="exclude any coin whose latest close exceeds this "
                         "many dollars from the crypto universe entirely "
                         "(default: no cap)")
    args = ap.parse_args()

    # Validate before anything else touches the broker. This is a risk limit
    # being widened from the command line, so a typo has to fail loudly here
    # rather than quietly become the new ceiling: --max-position 30 (meaning
    # 30%) would otherwise sail through as 3000% and disable the cap
    # entirely, and a negative one would make every plan fail preflight for
    # reasons that read like a strategy bug.
    if not 0.0 < args.max_position <= 1.0:
        ap.error(f"--max-position must be a weight in (0, 1], got "
                 f"{args.max_position} (30% is 0.30, not 30)")
    gate_kwargs = dict(GATE, max_weight=args.max_position)
    # Clamped, so an operator who sets --max-position 1.0 gets a backstop of
    # 1.0 rather than an impossible 1.05. Note that at exactly 1.0 the gap
    # closes and the backstop stops backing anything up -- which is the
    # honest consequence of asking for no concentration limit at all.
    max_position_weight = min(1.0, args.max_position + POSITION_BACKSTOP_MARGIN)

    if not 0.0 < args.max_crypto_position <= 1.0:
        ap.error(f"--max-crypto-position must be a weight in (0, 1], got "
                 f"{args.max_crypto_position} (20% is 0.20, not 20)")
    crypto_gate_kwargs = dict(CRYPTO_GATE, max_weight=args.max_crypto_position)
    crypto_max_position_weight = min(
        1.0, args.max_crypto_position + CRYPTO_POSITION_BACKSTOP_MARGIN)

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
            price_repo = OpenBBRepository(provider="yfinance",
                                          cache_dir=".cache/prices")
            panel = price_repo.fetch(UNIVERSE, "2010-01-01", end)
            report_trimmed_bars(price_repo, "equity", audit)
        panel = apply_price_cap(panel, args.max_price, "equity", audit)
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

    # ---- kalshi (forward-looking macro-event odds, for
    # KalshiEventRegimeFilter wrapped around STRATEGY) ----------------------
    # Same degrade-not-abort shape as the macro block above, for the same
    # reason: KalshiEventRegimeFilter already treats kalshi=None (or a
    # tracked series with no live reading) as a no-op pass-through by
    # design, so a Kalshi outage should degrade today's cycle back to
    # VIX+breadth-only risk management, not abort a trading day over the
    # most optional of the three overlays.
    try:
        if args.synthetic:
            # No real Kalshi ladder to fetch offline -- a single always-open,
            # always-confident synthetic reading exercises the wiring
            # (KalshiEventRegimeFilter actually receiving and reading a
            # panel) without claiming to be a real forecast. close_time is
            # fixed a year past the panel's last bar, so days_to_close never
            # falls inside horizon_days and this never actually fires --
            # the point is exercising the plumbing, not simulating an event.
            far_close = panel.dates[-1] + pd.Timedelta(days=365)
            kalshi = KalshiPanel(frame=pd.DataFrame(
                [
                    ("cpi", "SYN-CPI", "SYN-CPI-T0", float("nan"), d, far_close,
                     0.95, 1.0, 1.0)
                    for d in panel.dates
                ],
                columns=["series", "event_ticker", "market_ticker", "strike",
                         "snapshot_date", "close_time", "yes_price",
                         "volume", "open_interest"],
            ))
        else:
            # A recent window only, not 2010-onward like the other panels:
            # the filter reads just the nearest still-open event per series
            # (horizon_days out), so years of settled ladders are a few
            # hundred extra candlestick requests -- enough to trip Kalshi's
            # rate limit on their own -- for nothing the live cycle reads.
            kalshi_start = str((pd.Timestamp(end) - pd.Timedelta(
                days=KALSHI_LOOKBACK_DAYS)).date())
            kalshi = KalshiRepository(cache_dir=".cache/kalshi").fetch(
                kalshi_start, end)
    except Exception as exc:
        audit.emit("kalshi_fetch_failed", error=repr(exc))
        print(f"  (Kalshi fetch failed, continuing without it: {exc!r})")
        kalshi = None

    # ---- fundamentals (quality factor for the MultiFactorCrossSectional
    # sleeve of STRATEGY's Composite core) ---------------------------------
    # Same reasoning as the macro block above, and the same degrade-not-abort
    # shape -- but the failure mode is milder here: MultiFactorCrossSectional
    # treats a nonzero "quality" factor_weight with no fundamentals reading as
    # "this name doesn't qualify" for every name (see its own docstring), so
    # fundamentals=None doesn't zero the whole cycle, only that 40% capital
    # share of the core -- the momentum sleeve and both regime overlays are
    # unaffected either way.
    try:
        if args.synthetic:
            # No real filings to fetch offline -- one filing per symbol,
            # dated at the panel's first bar, with enough cross-sectional
            # dispersion to exercise the quality factor's ranking end to
            # end (picking *some* names over others) without claiming to
            # simulate real filing cadence.
            fundamentals = FundamentalsPanel(frame=pd.DataFrame(
                {
                    "symbol": panel.symbols,
                    "metric": "ratios_return_on_equity",
                    "period_end": pd.DatetimeIndex([panel.dates[0]] * len(panel.symbols)),
                    "as_of_date": pd.DatetimeIndex([panel.dates[0]] * len(panel.symbols)),
                    "value": [10.0 + 5.0 * (i % 7) for i in range(len(panel.symbols))],
                }
            ))
        else:
            fundamentals = FundamentalsRepository(
                provider="fmp", statements=("ratios",),
                cache_dir=".cache/fundamentals",
            ).fetch(UNIVERSE, "2010-01-01", end)
    except Exception as exc:
        audit.emit("fundamentals_fetch_failed", error=repr(exc))
        print(f"  (fundamentals fetch failed, continuing without it: {exc!r})")
        fundamentals = None

    # ---- broker -----------------------------------------------------------
    # One connection, shared by both sleeves -- run_pipeline() below is
    # handed a *view* of it (the broker itself for equity, broker.crypto_view()
    # for crypto), never connects or closes it, and every early-return path
    # for either sleeve leaves the connection open so the other sleeve can
    # still run. main() owns connect()/close() exclusively, via the
    # try/finally at the bottom of this function, precisely so one sleeve's
    # failure can never strand the connection the other one still needs.
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

    try:
        equity_rc = run_pipeline(
            label="equity", panel=panel, strategy=STRATEGY,
            gate_kwargs=gate_kwargs, macros=macros, kalshi=kalshi,
            fundamentals=fundamentals,
            broker_view=broker,
            policy=ExecutionPolicy(
                max_order_notional=args.max_order,
                max_plan_notional=args.max_plan,
                max_plan_turnover=args.max_turnover,
                max_position_weight=max_position_weight,
                # panel.symbols, not the raw UNIVERSE constant -- the two
                # can now differ (--max-price excludes symbols from the
                # panel entirely; see apply_price_cap()), and the allowlist
                # should describe what this cycle can actually propose, not
                # the configured universe before that filter ran.
                symbol_allowlist=tuple(panel.symbols),
                require_review=True,
                require_market_open=not args.ignore_market_hours,
                dry_run=not args.live,
            ),
            ledger=load_day_trade_ledger(
                mode_path("state/day_trades.json", args.synthetic)),
            day_trades_path=mode_path("state/day_trades.json", args.synthetic),
            peak_file=mode_path("state/peak_equity.txt", args.synthetic),
            journal_path=mode_path("audit/journal.jsonl", args.synthetic),
            max_turnover=args.max_turnover,
            audit=audit,
        )

        crypto_rc = 0
        if args.consider_crypto:
            print()
            crypto_rc = run_crypto_pipeline(
                args, broker, audit, crypto_gate_kwargs, crypto_max_position_weight)
    finally:
        broker.close()

    # Worst-wins: 3 (setup failure) > 2 (unresolved halt) > 1 (aborted) > 0.
    # One sleeve's outcome should never mask a worse outcome in the other --
    # a clean crypto run must not report success over an equity halt that
    # genuinely needs investigating before the next cycle.
    return max(equity_rc, crypto_rc)


def run_crypto_pipeline(
    args: argparse.Namespace, broker, audit: AuditLog,
    crypto_gate_kwargs: dict, crypto_max_position_weight: float,
) -> int:
    """Fetch the crypto price panel and run CRYPTO_STRATEGY through
    run_pipeline() as a second, independent pass over the same broker
    connection. Split out of main() so a crypto-specific data/capability
    failure (require_crypto() raising because the account or server has no
    crypto tools) is caught here, close to its own data fetch, rather than
    threading a second try/except shape through main() itself.

    ``crypto_gate_kwargs``/``crypto_max_position_weight`` come from main(),
    not recomputed here -- main() is where --max-crypto-position is
    validated (same place --max-position is), and this stays the only
    place that validation happens rather than a second, easy-to-forget copy.
    """
    try:
        end = str(pd.Timestamp.today().normalize().date())
        if args.synthetic:
            # A second, independent SyntheticRepository -- not a slice of
            # the equity one, and (see below) not sharing the equity
            # MockBroker either. SyntheticRepository always names symbols
            # SYN000.. regardless of seed (see qbt/data.py), so two
            # independent legs sharing one broker/price-series would
            # collide on identical names; separate repos and separate mock
            # accounts avoid that the same way Robinhood's own equity and
            # crypto books are two separate balance sheets, not one.
            crypto_panel = SyntheticRepository(n_symbols=8, seed=99).fetch(
                start="2010-01-01", end=end)
        else:
            crypto_repo = OpenBBRepository(
                provider="yfinance", cache_dir=".cache/prices",
                asset_class="crypto",
            )
            crypto_panel = crypto_repo.fetch(CRYPTOS, "2010-01-01", end)
            report_trimmed_bars(crypto_repo, "crypto", audit)
        crypto_panel = apply_price_cap(crypto_panel, args.max_crypto_price,
                                       "crypto", audit)
    except Exception as exc:
        audit.emit("crypto_data_fetch_failed", error=repr(exc))
        print(f"  (crypto data fetch failed, crypto sleeve skipped: {exc!r})")
        return 3

    # broker_view resolution differs by mode for the same reason the panel
    # fetch above does: --synthetic gets its own independent MockBroker
    # (own cash, own price series, own lifecycle -- connected and closed
    # right here, since main()'s connect/close only owns the real,
    # network-connected `broker`), never the equity MockBroker instance.
    # --live reuses the one real MCP connection via crypto_view() -- same
    # account, same session, just pinned to the crypto-bound tools (see
    # RobinhoodMCPBroker.crypto_view()'s own docstring).
    crypto_mock_broker = None
    try:
        if args.synthetic:
            crypto_mock_broker = MockBroker(
                prices=crypto_panel.last_close(), cash=5_000.0, seed=1)
            crypto_mock_broker.connect()
            broker_view = crypto_mock_broker
        else:
            broker.require_crypto()
            broker_view = broker.crypto_view()
    except Exception as exc:
        audit.emit("crypto_capability_missing", error=repr(exc))
        print(f"  (crypto sleeve skipped: {exc!r})")
        return 3

    try:
        return run_pipeline(
            label="crypto", panel=crypto_panel, strategy=CRYPTO_STRATEGY,
            gate_kwargs=crypto_gate_kwargs,
            # No macro/fundamentals threading here -- CRYPTO_STRATEGY's
            # MultiFactorCrossSectional member has no "quality" factor to
            # begin with (see CRYPTO_STRATEGY's own comment), and reusing
            # the equity run's real VIX panel here would be a second,
            # independent judgement call (crypto trades 24/7 -- does
            # yesterday's VIX regime reading even apply Saturday morning?)
            # that this file isn't taking a position on yet. macros=None is
            # MacroRegimeFilter's documented no-op pass-through either way,
            # so the practical effect is simply that this overlay does
            # nothing for crypto today, same as it silently does nothing
            # for equities too when the VIX fetch itself fails (see the
            # macro fetch block above).
            macros=None, kalshi=None, fundamentals=None,
            broker_view=broker_view,
            policy=ExecutionPolicy(
                max_order_notional=args.max_crypto_order,
                max_plan_notional=args.max_crypto_plan,
                max_plan_turnover=args.max_crypto_turnover,
                max_position_weight=crypto_max_position_weight,
                symbol_allowlist=tuple(crypto_panel.symbols),
                require_review=True,
                # The one hard behavioural difference from the equity
                # policy: crypto trades 24/7, so gating it on NYSE hours
                # would leave it unable to trade most of the week for a
                # reason that has nothing to do with crypto markets
                # actually being open.
                require_market_open=False,
                dry_run=not args.live,
            ),
            # Deliberately NOT persisted (day_trades_path=None below) and
            # deliberately NOT the equity ledger. equity_threshold=0.0
            # makes remaining() always report the "effectively unlimited"
            # 10_000 sentinel (see DayTradeLedger.remaining()) regardless
            # of any accumulated event count -- crypto trades through
            # Robinhood Crypto, which is not subject to FINRA's Pattern
            # Day Trader rule at all (that rule applies to securities), so
            # this isn't "this account crossed $25k," it's "this rule
            # doesn't apply here." RiskGate's own step 5 PDT check reads
            # exactly this field, so nothing else needs to know crypto is
            # exempt.
            ledger=DayTradeLedger(equity_threshold=0.0),
            day_trades_path=None,
            peak_file=mode_path("state/peak_equity_crypto.txt", args.synthetic),
            journal_path=mode_path("audit/journal_crypto.jsonl", args.synthetic),
            max_turnover=args.max_crypto_turnover,
            audit=audit,
        )
    finally:
        if crypto_mock_broker is not None:
            crypto_mock_broker.close()


def run_pipeline(
    *, label: str, panel, strategy, gate_kwargs: dict, macros, kalshi, fundamentals,
    broker_view, policy: ExecutionPolicy, ledger: DayTradeLedger,
    day_trades_path: str | None, peak_file: str, journal_path: str,
    max_turnover: float, audit: AuditLog,
) -> int:
    """One trading-cycle pipeline: account read, crash recovery, plan,
    diagnostics, execute, persist, report.

    Shared by both the equity and crypto sleeves (see main() and
    run_crypto_pipeline()) -- this used to be inlined once in main() before
    the crypto sleeve existed; extracted rather than duplicated because the
    duplicated logic here is exactly the safety-critical part (crash
    recovery, idempotent journaling, the drawdown breaker's persisted peak)
    where two copies drifting apart on a bug fix is the actual risk, not a
    style preference. Never connects or closes broker_view -- see main()'s
    own comment on why that lifecycle stays there.
    """
    print(f"--- {label} ---")
    try:
        account = broker_view.get_account()
    except Exception as exc:
        audit.emit(f"{label}_account_read_failed", error=repr(exc), sleeve=label)
        return 3

    manager = OrderManager(
        broker=broker_view, policy=policy, audit=audit,
        journal_path=journal_path, day_trade_ledger=ledger,
    )

    # ---- recovery before anything else ----------------------------------
    try:
        unresolved = manager.recover()
    except Exception as exc:
        audit.emit("recover_failed", error=repr(exc), sleeve=label)
        return 2
    if not unresolved.empty:
        lost = unresolved[unresolved["outcome"] == "not_at_broker"]
        if not lost.empty:
            audit.emit("halt_unresolved_orders", n=len(lost),
                       symbols=", ".join(lost["symbol"]), sleeve=label)
            print("HALT: in-flight orders could not be accounted for.")
            print(unresolved.to_string(index=False))
            return 2

    # ---- plan -------------------------------------------------------------
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
            print("    Sell or add to the universe; if large enough, "
                  "preflight will abort on equity drift until you do.")
            audit.emit("unmanaged_holdings", symbols=", ".join(unmanaged.index),
                       n=int(len(unmanaged)), sleeve=label)

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
        plan = LiveSignalRunner(strategy=strategy, risk_gate=RiskGate(**gate_kwargs),
                                max_turnover=max_turnover,
                                day_trade_ledger=ledger).plan(
            panel, state, fundamentals=fundamentals, macros=macros, kalshi=kalshi)

        # Surface *why* the plan looks the way it does -- these were
        # computed but never printed anywhere, which is exactly how the
        # peak-equity bug above went undetected: the risk gate's own
        # explanation ("drawdown breaker tripped at 96.0%") was sitting
        # right there in plan.decision.notes the whole time.
        if plan.warnings:
            for w in plan.warnings:
                print(f"  PLAN WARNING: {w}")
            audit.emit("plan_warnings", warnings="; ".join(plan.warnings), sleeve=label)
        if plan.decision is not None and plan.decision.notes:
            for n in plan.decision.notes:
                print(f"  RISK GATE: {n}")
            audit.emit("risk_gate_notes", notes="; ".join(plan.decision.notes), sleeve=label)

        # BreadthRegimeFilter/MacroRegimeFilter/KalshiEventRegimeFilter just
        # scale target_weights() down silently -- none of them go through
        # plan.warnings, so without this a de-risk from any of them would
        # show up only as unexplained smaller position sizes. Same view
        # LiveSignalRunner.plan() used internally, reconstructed here purely
        # for this diagnostic. Both STRATEGY and CRYPTO_STRATEGY share this
        # exact BreadthRegimeFilter(inner=MacroRegimeFilter(inner=
        # KalshiEventRegimeFilter(inner=Composite))) nesting, which is what
        # this diagnostic assumes.
        regime_view = panel.as_of(plan.asof)
        regime_macros = macros.as_of(plan.asof) if macros is not None else None
        regime_kalshi = kalshi.as_of(plan.asof) if kalshi is not None else None
        breadth_filter = strategy
        macro_filter = strategy.inner
        kalshi_filter = strategy.inner.inner
        if breadth_filter.blocked(regime_view):
            print(f"  REGIME: market breadth below {breadth_filter.min_breadth:.0%} "
                  f"-- book scaled to {breadth_filter.scale_when_blocked:.0%} "
                  f"(breadth={breadth_filter.breadth(regime_view):.0%})")
            audit.emit("breadth_regime_blocked",
                       breadth=round(breadth_filter.breadth(regime_view), 4),
                       scale=breadth_filter.scale_when_blocked, sleeve=label)
        if macro_filter.blocked(regime_view, regime_macros):
            print(f"  REGIME: VIX regime unfavourable -- book scaled to "
                  f"{macro_filter.scale_when_blocked:.0%}")
            audit.emit("macro_regime_blocked", metric=macro_filter.metric,
                       scale=macro_filter.scale_when_blocked, sleeve=label)
        if kalshi_filter.blocked(regime_view, regime_kalshi):
            print(f"  REGIME: an imminent Kalshi-tracked print is still "
                  f"unresolved -- book scaled to "
                  f"{kalshi_filter.scale_when_blocked:.0%}")
            audit.emit("kalshi_regime_blocked",
                       scale=kalshi_filter.scale_when_blocked, sleeve=label)

        report = manager.execute(plan, strategy_name=strategy.name)
    except Exception as exc:
        audit.emit("cycle_error", error=repr(exc), sleeve=label)
        return 3

    save_peak(peak_file, max(peak, account.equity))
    if day_trades_path is not None:
        save_day_trade_ledger(day_trades_path, ledger)
    print(report)
    if not report.to_frame().empty:
        print(report.to_frame().to_string(index=False))
    if report.reconciliation is not None and report.reconciliation["breach"].any():
        print("\nRECONCILIATION DRIFT:")
        print(report.reconciliation[report.reconciliation["breach"]].to_string())

    audit.emit("cycle_end", aborted=report.aborted,
               submitted=len(report.submitted), rejected=len(report.rejected),
               skipped=len(report.skipped), sleeve=label)
    return 1 if report.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
