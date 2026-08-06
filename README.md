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
| `qbt/signals.py` | `Strategy` protocol, 10 strategies, 5 composer/filter wrappers |
| `qbt/risk.py` | `RiskGate` (vol target, caps, drawdown breaker), `DayTradeLedger` |
| `qbt/engine.py` | `Backtester` (T+1 fills, costs), performance metrics |
| `qbt/research.py` | IC grid, autocorrelation, walk-forward, multiple-testing haircut |
| `qbt/live.py` | `LiveSignalRunner` → order intents (no execution) |
| `qbt/broker.py` | `BrokerAdapter` protocol, `MockBroker`, `RobinhoodMCPBroker` |
| `qbt/oauth.py` | OAuth 2.0 Authorization Code + PKCE for `RobinhoodMCPBroker`, via the MCP SDK's own client |
| `qbt/orders.py` | `OrderManager`: journal, preflight, reconciliation, audit |
| `research.ipynb` | 49-cell research notebook, 30 code cells |
| `run_cycle.py` | One trading cycle. Run on a schedule, never from the notebook |
| `test_qbt.py` | 74 engine validation checks (strategies, engine, risk gate, research) |
| `test_orders.py` | 73 order-path checks, offline against `MockBroker` |
| `test_fundamentals.py` | 29 checks: `FundamentalsPanel` PIT semantics, `FundamentalsValueFilter` |
| `test_macro.py` | 31 checks: `MacrosPanel` PIT semantics, `MacroRegimeFilter` |
| `test_corporate.py` | 35 checks: `CorpsPanel` PIT semantics, filing/insider indicators |
| `test_options.py` | 37 checks: `OptionsPanel`, daily-archive semantics |
| `test_options_strategy.py` | 18 checks: `OptionsMeanReversion` |
| `test_insider_drift_strategy.py` | 19 checks: `InsiderEventDrift` |
| `test_regime_filters.py` | 25 checks: `MacroRegimeFilter`/`FundamentalsValueFilter` end-to-end |
| `test_robinhood_broker.py` | 30 checks: response shapes confirmed against the live Robinhood MCP server |
| `test_oauth.py` | 11 checks: PKCE flow, loopback callback server |
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
python test_qbt.py                     # 74 engine checks, ~90s
python test_orders.py                  # 73 order-path checks, ~10s
python test_fundamentals.py            # 29 checks
python test_macro.py                   # 31 checks
python test_corporate.py               # 35 checks
python test_options.py                 # 37 checks
python test_options_strategy.py        # 18 checks
python test_insider_drift_strategy.py  # 19 checks
python test_regime_filters.py          # 25 checks
python test_robinhood_broker.py        # 30 checks, offline against confirmed live response shapes
python test_oauth.py                   # 11 checks
python smoke_test.py                   # all 30 notebook cells, ~120s
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

Plus five composer/filter wrappers that wrap any strategy, each deciding a
different axis: `TrendFilter` (absolute-momentum gate, per name),
`FundamentalsValueFilter` (block names failing a fundamentals screen, per
name), `MacroRegimeFilter` (scale the *whole book* down in an unfavourable
macro regime — no symbol axis to gate on), `InverseVolWeighted` (re-weight
picks by inverse vol), and `Composite` (blend several strategies by fixed
capital share).

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
good for is the read side and the dry run — section 11 replaces the
hypothetical `PortfolioState` with real broker positions and runs
`OrderManager` with `dry_run=True`, so you see exactly what would be sent
without sending it.

```bash
python run_cycle.py --synthetic --ignore-market-hours   # offline, mock broker
python run_cycle.py                                     # dry run, real data
python run_cycle.py --live --max-order 50               # live, minimum size
```

Exit codes: `0` completed, `1` aborted at preflight, `2` unresolved in-flight
order (halt before the next cycle), `3` setup failure.

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
- **`place_equity_order`'s response wraps under `"order"`**, not `"data"` like
  `portfolio` — each tool's response wrapper key matches what it returns, not
  one shared envelope. Missing this meant a real, filled order was briefly
  unparseable (see "Idempotency" above for how that's now closed out safely).
- **`review_equity_order`/`place_equity_order`/`cancel_equity_order`/
  `get_equity_orders` all require `account_number`** in their schema, resolved
  from `get_account()` and cached for the rest of the broker's lifetime.
- **`quantity` is declared as a JSON string**, not a number, in at least the
  review schema — values are coerced to match whatever type each tool's own
  schema actually declares, not assumed.
- **Orders must not exceed 8 decimal places**, a business rule enforced by the
  API itself, beyond and separate from anything the schema says.
- **Fractional-share orders are rejected outright on at least several ETFs**
  (observed on XLF, XLK, XLV, and IWM) — "Order quantity cannot include
  fractional shares." This looks like a broad account/instrument-class
  limitation, not a one-off quirk; budget for whole-share sizing on a small
  account, since a target position under 1 share just won't fill. See the
  idempotency note above for how a rejection here is now handled automatically.
- **Real errors arrive as nested `anyio` `ExceptionGroup`s.** `str()` on one of
  these collapses to `"unhandled errors in a TaskGroup (1 sub-exception)"` —
  the actual message only survives `repr()`. Anything matching on error text
  needs to search the `repr()`, not the `str()`.

The full, dated list — including fixtures that reproduce each exact response
shape — lives in `test_robinhood_broker.py`'s header docstring; that file is
the thing to extend the next time a new quirk turns up. Two things remain
genuinely unconfirmed: **rate limits** (not yet hit), and whether
**`review_equity_order`/`cancel_equity_order` follow the same `"order"`-key
wrapper `place_equity_order` does** (inferred from the pattern, not
independently observed — `broker.call_raw("review", ...)` is the safe way to
check directly).

The entire order path is additionally tested against `MockBroker`, which
models partial fills, rejections, and untradeable symbols — so the logic is
verified offline too, not only against the one live account it's been run
against so far.

### Four independent kill switches

Fastest to slowest: the `KILL` file preflight checks every cycle; the drawdown
breaker in the risk gate; the turnover cap in `LiveSignalRunner`; and
Robinhood's one-tap disconnect in the app — the only one that does not depend on
your code being correct.

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
