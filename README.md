# qbt — signal engine + backtester

A strategy is a **pure function** from a price panel *sliced to the decision bar*
to target portfolio weights. The backtester and the live runner call that same
function, so there is one implementation of your logic and no path for research
and production to drift apart.

## Layout

| File | Role |
|---|---|
| `qbt/data.py` | `PricePanel` + `as_of()` look-ahead firewall; OpenBB (confirmed live via yfinance) and synthetic price sources |
| `qbt/fundamentals.py` | `FundamentalsPanel`: point-in-time financial-statement ratios, keyed by filing date, not fiscal period |
| `qbt/macro.py` | `MacrosPanel`: point-in-time macro series (CPI, Fed funds rate, unemployment, yield curve, ...) from FRED |
| `qbt/corporate.py` | `CorpsPanel`: point-in-time SEC filing cadence (10-K/10-Q/8-K) and Form 4 insider-transaction indicators |
| `qbt/options.py` | `OptionsPanel`: daily-archived options-chain indicators (ATM IV, put/call volume ratio) |
| `qbt/signals.py` | `Strategy` protocol, 10 strategies, 6 composer/filter wrappers |
| `qbt/risk.py` | `RiskGate` (vol target, caps, drawdown breaker), `DayTradeLedger` (PDT budget, persisted across cycles by `run_cycle.py`) |
| `qbt/engine.py` | `Backtester` (T+1 fills, costs), performance metrics |
| `qbt/research.py` | IC grid, autocorrelation, walk-forward, multiple-testing haircut |
| `qbt/live.py` | `LiveSignalRunner` → order intents (no execution) |
| `qbt/broker.py` | `BrokerAdapter` protocol, `MockBroker`, `RobinhoodMCPBroker` |
| `qbt/oauth.py` | OAuth 2.0 Authorization Code + PKCE for `RobinhoodMCPBroker`, via the MCP SDK's own client |
| `qbt/orders.py` | `OrderManager`: journal, preflight, reconciliation, audit |
| `research.ipynb` | 49-cell research notebook, 30 code cells |
| `run_cycle.py` | One trading cycle. Run on a schedule, never from the notebook |
| `test_qbt.py` | 129 engine validation checks (strategies, engine, risk gate, research, panel validation) |
| `test_orders.py` | 152 order-path checks, offline against `MockBroker` |
| `test_fundamentals.py` | 33 checks: `FundamentalsPanel` PIT semantics, `FundamentalsValueFilter` |
| `test_macro.py` | 42 checks: `MacrosPanel` PIT semantics, `MacroRegimeFilter`, reading staleness |
| `test_corporate.py` | 38 checks: `CorpsPanel` PIT semantics, filing/insider indicators |
| `test_options.py` | 39 checks: `OptionsPanel`, daily-archive semantics |
| `test_options_strategy.py` | 19 checks: `OptionsMeanReversion` |
| `test_insider_drift_strategy.py` | 21 checks: `InsiderEventDrift` |
| `test_regime_filters.py` | 43 checks: `MacroRegimeFilter` (incl. a `vix` example)/`FundamentalsValueFilter`/`BreadthRegimeFilter` end-to-end |
| `test_robinhood_broker.py` | 43 checks: response shapes confirmed against the live Robinhood MCP server |
| `test_oauth.py` | 14 checks: PKCE flow, loopback callback server, token-file permissions |
| `test_run_cycle.py` | 33 checks: `run_cycle.py`'s live `STRATEGY` composition and thresholds, regime triggering, day-trade ledger persistence, unmanaged holdings |
| `build_notebook.py` | Regenerates `research.ipynb` |
| `smoke_test.py` | Executes every notebook cell in one namespace |

## Data sources

`qbt/data.py` covers price only. The other four panels exist because a
strategy that reads fundamentals, macro, filings, or options data off a
table fetched *today* is reading information the market couldn't see on
the date it's being joined to — each panel fixes that by keying itself to
the date the data actually became public, not the date it describes.

| Panel | Point-in-time anchor | Provider |
|---|---|---|
| `FundamentalsPanel` | 10-Q/10-K filing date, not the fiscal period the statement describes | `pip install openbb openbb-fmp` (or `openbb-intrinio`) |
| `MacrosPanel` | FRED release date — release *and* revision, since GDP/CPI/payrolls get revised after their first print | `pip install openbb openbb-fred` + a free FRED API key |
| `CorpsPanel` | SEC's own `accepted_date`/`filing_date` — no estimated lag needed, unlike the two above | `pip install openbb openbb-sec` |
| `OptionsPanel` | Snapshot date (a live quote has no disclosure lag) | `pip install openbb openbb-yfinance` (or `openbb-cboe`) |

`OptionsPanel` is the one exception to "point-in-time": free options-chain
providers only return *today's* chain, not history, so it's built by
appending one snapshot per day to an on-disk archive (`fetch_snapshot()`,
meant to run once daily from a cron job) rather than backfilling — there is
no getting last Tuesday's chain from `yfinance`/`cboe`. See the module
docstring for the two paid providers (`intrinio`, `tmx`) that do support a
historical `date` parameter.

All four are optional: every strategy's `target_weights()` accepts
`fundamentals`/`macros`/`corps`/`options` as `None` and either passes
through unfiltered (`FundamentalsValueFilter`, `MacroRegimeFilter`) or
returns no positions for names it can't score (`OptionsMeanReversion`,
`InsiderEventDrift`) rather than raising.

## Run

```bash
pip install pandas numpy scipy matplotlib          # required
pip install openbb openbb-yfinance                # optional, for real price/options data
pip install openbb-fmp                            # optional, for FundamentalsPanel (or openbb-intrinio)
pip install openbb-fred                           # optional, for MacrosPanel (needs a free FRED key)
pip install openbb-sec                            # optional, for CorpsPanel
pip install mcp                                   # optional, for RobinhoodMCPBroker + oauth
python test_qbt.py                     # 129 engine checks, ~90s
python test_orders.py                  # 152 order-path checks, ~10s
python test_fundamentals.py            # 33 checks
python test_macro.py                   # 42 checks
python test_corporate.py               # 38 checks
python test_options.py                 # 39 checks
python test_options_strategy.py        # 19 checks
python test_insider_drift_strategy.py  # 21 checks
python test_regime_filters.py          # 43 checks
python test_robinhood_broker.py        # 43 checks, offline against confirmed live response shapes
python test_oauth.py                   # 14 checks
python test_run_cycle.py               # 33 checks, live STRATEGY wiring
python smoke_test.py                   # all 30 notebook cells
jupyter lab research.ipynb
```

The notebook runs offline on generated data by default. Set `USE_OPENBB = True`
in cell 1 for real ETF data; nothing else changes.

## Strategies

| Strategy | Idea |
|---|---|
| `EqualWeightBuyHold` | Benchmark: hold everything tradeable at equal weight. |
| `CrossSectionalMomentum` | Rank the universe by trailing return, hold the top `n`. |
| `TimeSeriesMomentum` | Hold each name only while it's above its own moving average. |
| `ShortHorizonReversal` | Buy the most oversold names by trailing-return z-score. |
| `PairsTrading` | Market-neutral stat-arb on the log-price spread of correlated pairs. |
| `MultiFactorCrossSectional` | Blend momentum, low-vol, and reversal into one cross-sectional score. |
| `CalendarSeasonality` | Invested only in the turn-of-month window, flat the rest of the month. |
| `RiskParityAllocation` | Equal risk contribution across the universe from the covariance matrix. |
| `OptionsMeanReversion` | Buy names where options-implied fear (IV, put/call ratio) is most stretched vs. their own history. |
| `InsiderEventDrift` | Long fresh open-market insider buying that follows an 8-K, held while the signal stays inside its drift window. |

Plus six composer/filter wrappers that wrap any strategy, each deciding a
different axis: `TrendFilter` (absolute-momentum gate, per name),
`FundamentalsValueFilter` (block names failing a fundamentals screen, per
name), `MacroRegimeFilter` (scale the *whole book* down in an unfavourable
macro regime — no symbol axis to gate on; `metric="vix"` is the concrete,
tested example — FRED's `VIXCLS` was already in `MacrosPanel`'s default
indicators but nothing exercised it end-to-end before), `BreadthRegimeFilter`
(scale the whole book down when too few names in the *strategy's own
universe* are above their own trailing moving average — no external data
source, computed straight from the price panel), `InverseVolWeighted`
(re-weight picks by inverse vol), and `Composite` (blend several strategies
by fixed capital share).

## The three tests worth keeping permanently

1. **Null test** (`test_qbt.py` §2). On data with no exploitable structure, every
   strategy must score a Sharpe indistinguishable from zero. Currently 0.41
   (t=1.60) and 0.10 (t=0.41) for `CrossSectionalMomentum` and
   `ShortHorizonReversal`, the only two the loop currently runs. If a
   strategy looks profitable here, the harness is leaking, not the market
   paying. The other 8 strategies aren't in this loop yet — extending it to
   all of them is a good next step, not something to assume passes by
   analogy.
2. **Look-ahead firewall** (§1). A strategy that records the furthest date it saw
   must never exceed its decision bar.
3. **Live parity** (`test_qbt.py` §7 and notebook §10). The live runner and a
   direct gate call must produce bit-identical weights. Currently exact —
   0.00e+00.

## Two calibrated demonstrations

**Execution timing.** The same reversal strategy: Sharpe **+0.98** with
same-close fills, **−0.08** with next-bar fills. Identical signal and costs; the
entire edge was the timing assumption. The engine defaults to `delay_bars=1` and
*refuses* `delay_bars=0` with `price='open'`, which would fill before the signal
exists.

**Multiple testing.** A 36-config sweep produced a best Sharpe of 0.90 against an
expected-max-under-null of 0.49. Median across the grid was 0.20 — closer to what
you should expect to live with than the maximum.

## Known gaps

- **`OpenBBRepository` is confirmed working against yfinance in production**
  (`run_cycle.py` fetches real price data through it every live cycle) —
  the "untested against a live provider" gap this used to list is closed.
- **Point-in-time data now exists for fundamentals, macro, and filings** —
  `FundamentalsPanel`, `MacrosPanel`, and `CorpsPanel` each key off the date
  their data actually became public (see "Data sources" above), not the
  date it describes. Two things still don't have it: `OptionsPanel` has no
  historical backfill on the free providers (documented in its own module,
  not a bug to fix — a data-market limitation), and
  `MultiFactorCrossSectional` deliberately stays price-only rather than
  reaching for the new panels (see its docstring) — a genuine "quality"
  factor from `FundamentalsPanel` is the natural next extension there.
- **Fill model is a fixed spread.** No partial fills, gaps, halts, or
  size-dependent impact.
- **Walk-forward retention ratio > 1 on synthetic data** (OOS 0.97 vs IS 0.60).
  That is an artifact of the generator's stationarity. Real data will not do
  this; treat a ratio above 1 as a signal your data is too well-behaved.
- **`PairsTrading` selects pairs by correlation, not cointegration.** Two
  correlated series can still drift apart permanently; a proper Engle-Granger
  or Johansen test is the next step before trusting this beyond synthetic
  data. Legs are also equal-dollar rather than beta-hedged.
- **`CalendarSeasonality` uses calendar days, not the trading calendar.**
  `today.day` vs `today.days_in_month` approximates "last N trading days of
  the month" without accounting for holidays. It also needs `rebalance='D'`
  or `'W'` to express its logic at all — see the class docstring.
- **`RiskParityAllocation`'s covariance estimate gets noisy fast.** 126 bars
  against up to 30 names is a thin sample for a full covariance matrix;
  `max_names` and `cov_lookback` are levers to pull before trusting the
  weights on real data.

## Suggested next steps

1. Set `USE_OPENBB = True`, rerun. Results will be worse — that is information.
2. Read the IC grid on real data **before** looking at any equity curve.
3. Move `test_qbt.py` into a real `tests/` directory with pytest.
4. Persist the `audit` frame from every run. It is the only thing that
   distinguishes "the strategy stopped working" from "I changed something".
5. Give `MultiFactorCrossSectional` a real quality factor from
   `FundamentalsPanel` now that one exists (see "Known gaps").

## Connecting to Robinhood

The notebook never places orders. Notebooks have hidden state and out-of-order
execution — re-running a cell re-submits, and a kernel restart loses the record
of what was already sent. `run_cycle.py` does the placing, as a separate
process with a write-ahead journal and crash recovery. What the notebook *is*
good for is the read side and the dry run — section 12 (below, and in the
notebook itself) replaces the hypothetical `PortfolioState` with real broker
positions and runs `OrderManager` with `dry_run=True`, so you see exactly
what would be sent without sending it.

```bash
python run_cycle.py --synthetic --ignore-market-hours   # offline, mock broker
python run_cycle.py                                     # dry run, real data
python run_cycle.py --live --max-order 50               # live, minimum size
python run_cycle.py --check-portfolio                   # read-only: print holdings, exit
```

Exit codes: `0` completed, `1` aborted at preflight, `2` unresolved in-flight
order (halt before the next cycle), `3` setup failure.

`--check-portfolio` skips everything else — no price panel, no strategy, no
risk gate, no journal or audit writes. It connects, calls the same
`get_account()` every real cycle already calls, and prints holdings with
current value and weight (via `get_quotes()`) so you can see what's actually
in the account without running or dry-running a full cycle.

### The live strategy

`run_cycle.py`'s `STRATEGY` is not just `CrossSectionalMomentum` on its own —
it's that core wrapped in two whole-book de-risking overlays, both scaling
exposure down (never up, never redistributing into fewer names) when their
own regime read is unfavourable:

```
BreadthRegimeFilter(lookback=200, min_breadth=0.3, scale_when_blocked=0.5)
  -> MacroRegimeFilter(metric="vix", max_level=35.0, max_increase=15.0,
                        lookback=21, scale_when_blocked=0.5)
       -> CrossSectionalMomentum(lookback=63, skip=5, top_n=5)
```

`MacroRegimeFilter` blocks on a VIX level above 35 or a 15-point rise inside
21 trading days — genuine risk-off territory, not routine noise.
`BreadthRegimeFilter` blocks when fewer than 30% of `run_cycle.py`'s own
18-symbol universe is above its own 200-day average. Neither changes which
names get picked; each independently scales total exposure to 50% when
triggered, so both firing at once compounds to 25% (see the comment above
`STRATEGY` in `run_cycle.py` for the full rationale). A cycle where either
overlay is actively scaling prints a `REGIME:` line and emits a
`breadth_regime_blocked` / `macro_regime_blocked` audit event — without that,
a de-risk would show up only as an unexplained smaller position size.
`test_run_cycle.py` (33 checks) is the permanent regression guard on this
wiring: the composition, the thresholds, and that elevated VIX alone scales
turnover to exactly 50%.

**The VIX overlay needs a working FRED fetch to do anything.** In `--live`
and dry-run (non-`--synthetic`) modes, `run_cycle.py` fetches VIX via
`MacrosRepository`/`openbb-fred`, cached under `.cache/macro`. If that fetch
fails for any reason — no `openbb-fred` installed, no FRED API key
configured, FRED itself down — the cycle does **not** abort; it emits
`macro_fetch_failed`, prints the exception, and continues with `macros=None`.
`MacroRegimeFilter` treats `macros=None` as a documented no-op pass-through,
so the practical effect is silent: the VIX overlay simply does nothing every
cycle until FRED access is actually configured, and only `BreadthRegimeFilter`
(which needs no external data — it's computed straight from the price panel)
is doing any regime-based de-risking in the meantime. Safe, but worth knowing
explicitly rather than assuming both overlays are live.

### The per-order cap and whole-share instruments

`--max-order` (default **$500**) caps a single order's notional. It is
enforced per intent, skipping that order and continuing rather than aborting
the plan.

It is also re-checked on the **whole-share retry**. When a broker rejects a
fractional size, `OrderManager` retries once at a whole-share quantity — a
different, larger order than the one that originally cleared the cap, so it
has to clear it again. The consequence is deliberate: any instrument whose
*single share* costs more than the cap becomes untradeable rather than being
bought over the limit.

$500 is chosen against this universe and account size rather than picked
round. It clears one whole share of every name in `UNIVERSE` (the tallest are
GLD ~$392 and IWM ~$301), and it clears the largest single position the risk
layer will ask for on ~$1,000 of equity: `GATE`'s `max_weight=0.30` is ~$304
and `max_position_weight=0.35` is ~$354. Below roughly $400 the cap and the
weight limits disagree — a position the gate permits cannot be built in one
trade, and it can only be reached incrementally across cycles. At the earlier
$250 default, IWM and GLD were not enterable at all.

Two things this does *not* loosen: `--max-plan` (default $5,000) still caps
the whole cycle, and `max_position_weight` still caps any single name's share
of the book. Raising the per-order cap changes how large one *trade* may be,
not how large a *position* may become.

### Short positions

`ExecutionPolicy.allow_short` defaults to **False**, and preflight fails the
whole plan if any target weight is negative. This is a plan-level abort on
purpose. Short legs are otherwise stopped one at a time by the broker's own
per-intent review, which runs *after* preflight — so a market-neutral plan
half-executes: the long legs fill, the shorts bounce, and you hold a
directional book nobody chose. Measured against `PairsTrading`, which ships
here and emits negative weights by design: before the guard it submitted
`['SYN004', 'SYN005']` long and skipped both hedges. Selling a position down
to flat is not shorting and is unaffected. Set `allow_short=True` only if the
account genuinely supports it.

### Idempotency without a broker-side key

`place_equity_order` does not document a client idempotency key, so repeatability
comes from a write-ahead journal plus read-back:

1. Append `submitting` to `audit/journal.jsonl`, fsync.
2. Call the broker.
3. Append the outcome.

If the process dies between 1 and 3, `OrderManager.recover()` reads the broker's
order list and matches fingerprints to determine what happened. It never retries
an *ambiguous* outcome — a blind retry after a timeout or 5xx is the most likely
way this loses money for reasons unrelated to the strategy. Deduplication keys
on `(symbol, side)` within a plan, deliberately *not* on quantity: keying on a
float would let a 1% equity drift between runs defeat it.

One deliberate exception to "never retries": the specific, synchronous
"cannot include fractional shares" rejection (see below) is not ambiguous,
since the broker refused the request outright and the order was definitively
never created. `OrderManager.execute()` matches on that one message,
corrects and resubmits once at a whole-share size, and writes a *second*
write-ahead journal entry so `recover()` fingerprints against what was
actually sent on retry, not the original request. Every other rejection —
including any other 4xx — still goes through the never-retry path above;
this isn't a general "retry on 4xx" rule.

### Tool discovery

`RobinhoodMCPBroker` enumerates the server's tools at connect time and resolves
argument names from each input schema, so a server that says `ticker` where you
assumed `symbol` still works. Missing capabilities raise at connect, not at the
first order. Run `broker.list_capabilities()` first.

### Confirmed against the live service (2026-08)

Everything below was verified against a real agentic Robinhood account, not
inferred — the OAuth handshake, order placement, and the schema quirks that
follow all come from actual live calls, not the original design-time guesses.

- **The OAuth handshake works end-to-end.** `qbt/oauth.py`'s RFC 8414 /
  RFC 7591 discovery flow, confirmed against the real MCP server.
- **`accounts` nests two levels deep**, `{"data": {"accounts": [...]}}`, and
  the real agentic-account flag is `agentic_allowed` (a plain, caller-relative
  bool) — not `is_agentic`/`agentic`/`type` as originally guessed.
- **`portfolio` returns one object, not a list**, `{"data": {...}}` one level
  deep, with `buying_power` itself nested one level further,
  `{"buying_power": {"buying_power": "1000.0000", ...}}`.
- **`place_equity_order`'s response is wrapped two levels deep**,
  `{"data": {"order": {...}}}` — the same `"data"` envelope every other
  endpoint uses, plus a resource-name key, the same shape `"accounts"` uses
  for its list (`{"data": {"accounts": [...]}}`). `_unwrap_object()` used to
  stop after peeling one layer, which happened to be enough for `portfolio`
  (flat fields directly under `"data"`, no resource-name key) but not for
  `place`/`order` — it now peels every consecutive layer, not just one.
  Missing this meant two separate real, filled orders came back unparseable
  before being caught (see "Idempotency" above for how that's now closed out
  safely each time it happens, rather than by getting the shape right once
  and hoping).
- **`review_equity_order`/`place_equity_order`/`cancel_equity_order`/
  `get_equity_orders` all require `account_number`** in their schema, resolved
  from `get_account()` and cached for the rest of the broker's lifetime.
- **`quantity` is declared as a JSON string**, not a number, in at least the
  review schema — values are coerced to match whatever type each tool's own
  schema actually declares, not assumed.
- **Orders must not exceed 8 decimal places**, a business rule enforced by the
  API itself, beyond and separate from anything the schema says.
- **Fractional-share orders are rejected outright on at least several ETFs**
  (observed on XLF, XLK, XLV, IWM, and EFA) — "Order quantity cannot include
  fractional shares." This looks like a broad account/instrument-class
  limitation, not a one-off quirk. On a small account this means most
  multi-name targets round to under a share; `execute()` rounds a sub-share
  target *up* to one whole share specifically when establishing a brand-new
  position (nothing held yet), since a floor to zero there means never
  holding that name at all, on any account this size. A marginal top-up or
  trim on a position already held still rounds down and skips — see the
  idempotency note above for the full retry logic either way.
- **Real errors arrive as nested `anyio` `ExceptionGroup`s.** `str()` on one of
  these collapses to `"unhandled errors in a TaskGroup (1 sub-exception)"` —
  the actual message only survives `repr()`. Anything matching on error text
  needs to search the `repr()`, not the `str()`.
- **`get_equity_quotes`'s real response is `{"data": {"results": [{"quote":
  {...}, "close": {...}}, ...]}}`** — each result bundles a *live* quote and a
  *stale* end-of-day close as sibling objects, with `symbol`/price fields
  inside `"quote"`, not at the top level of the result. A naive top-level
  field lookup found neither and `get_quotes()` silently returned an empty
  series for every real request — no exception, just no prices, which meant
  a portfolio summary showed shares held with no value or weight for any
  position. Falls back to `"close"` only if the live quote has no usable
  price field.

The full, dated list — including fixtures that reproduce each exact response
shape — lives in `test_robinhood_broker.py`'s header docstring; that file is
the thing to extend the next time a new quirk turns up. Two things remain
genuinely unconfirmed: **rate limits** (not yet hit), and whether
**`review_equity_order`/`cancel_equity_order` wrap the same way `place`
does** (inferred from the pattern, not independently observed —
`broker.call_raw("review", ...)` is the safe way to check directly). Trust
the *pattern* here (a "data" envelope plus a resource-name key) more than
any single confirmed depth, though — that's the second time a real order
response turned out to be wrapped one layer deeper than the last confirmed
sample suggested.

The entire order path is additionally tested against `MockBroker`, which
models partial fills, rejections, and untradeable symbols — so the logic is
verified offline too, not only against the one live account it's been run
against so far.

### Three stops, and a two-stage throttle

The `KILL` file (checked every cycle), the drawdown breaker in the risk gate,
and Robinhood's one-tap disconnect in the app are the genuine stops — each
one zeroes out the plan (or, for the disconnect, everything). Turnover isn't
among them: both places that check it scale the plan down to fit rather than
reject it outright, and neither is the "real" one alone — they're two
applications of the same idea, not a soft check backed by a hard one.

`LiveSignalRunner.plan()` scales first, against the equity it has at
planning time. `OrderManager._refit_turnover()` re-fits the same plan again
right before submission, against a freshly-read account — because scaling to
*exactly* the cap at planning time leaves no margin for ordinary price
movement between then and submission. Confirmed live (2026-08): a $0.12 move
on a ~$1,000 account was enough to push a plan already scaled to exactly 67%
back over it by the time `OrderManager` read the account fresh, and the
first version of this hard-aborted on that — rejecting a plan that had been
correctly sized moments earlier for no reason a human would find
convincing. Re-fitting at both ends means overage gets corrected wherever it
turns up, not just detected once and given up on (see "Idempotency" above
for the same instinct applied to order retries instead of trade sizing).

### PDT accounting

`RiskGate`'s PDT check (step 5 of `apply()`) reads `day_trades_remaining` and
blocks opening a *new* position once the rolling budget is exhausted — closes
and trims of positions already held are still allowed, same as a real
PDT-flagged margin account. This is only as good as what feeds it: a
`DayTradeLedger` that gets thrown away every cycle can never actually
accumulate a day trade, so the check would never fire no matter how many day
trades really happened. `run_cycle.py` persists the ledger to
`state/day_trades.json` (mode-suffixed like `peak_equity.txt` — synthetic and
live runs never share it) and passes the *same* instance to both
`LiveSignalRunner` and `OrderManager`, so a round trip `OrderManager.execute()`
detects — bought from flat, then sold back to flat, in the same market-tz
session — is immediately visible to the next cycle's `day_trades_remaining`.

**A backtest cannot tell you whether a strategy will trip this.**
`Backtester.run()` has matching `opened_on`/`ledger.record()` bookkeeping, but
on a daily panel it is structurally unreachable: one fill per symbol per bar
and one bar per session means nothing ever round-trips inside a session, so
the count is always zero (measured — 976 daily rebalances of a 3-day reversal
strategy, ~1,420 trades, zero symbol-days with more than one trade, at
`delay_bars` 0, 1 and 2 alike). Live trading genuinely can day-trade and so
can trip the block; research never will. Seed a `DayTradeLedger` explicitly if
you want to study how an already-restricted account behaves —
`ledger.remaining()` still feeds `RiskContext` either way.

Events are stored as **tz-naive session dates** via `qbt.risk.as_session_date`.
That normalisation is load-bearing, not cosmetic: `OrderManager` resolves
sessions in the market timezone (tz-aware) while `LiveSignalRunner` passes a
tz-naive panel date as `asof`, and comparing the two raised `TypeError` inside
`DayTradeLedger.count()`. In `run_cycle.py` that surfaced as a *permanent*
outage rather than a single failed cycle — the exception aborted the run before
`save_day_trade_ledger` could rewrite the file, so every later cycle reloaded
the same poisoned event and failed identically, with only `cycle_error` in the
audit log. `count()` normalises on read as well as write, so a state file
written before the fix heals itself instead of needing to be deleted by hand.

### Going live, in order

Each step should run for long enough to be boring before you take the next.

1. **Dry run**, daily, for a couple of weeks. You are testing the scheduler,
   the data fetch, and preflight — not the strategy.
2. **Shadow**: keep `dry_run=True` but record what *would* have filled and
   compare to the backtest's expectation. Divergence here is a bug, since
   nothing has traded yet.
3. **Live at minimum size.** `max_order_notional=50`. The goal is to observe
   real fills, real rejections, and real reconciliation drift.
4. **Scale** only after a full rebalance cycle has completed cleanly,
   including at least one rejection handled correctly.
