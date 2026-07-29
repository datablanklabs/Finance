# qbt — signal engine + backtester

A strategy is a **pure function** from a price panel *sliced to the decision bar*
to target portfolio weights. The backtester and the live runner call that same
function, so there is one implementation of your logic and no path for research
and production to drift apart.

## Layout

| File | Role |
|---|---|
| `qbt/data.py` | `PricePanel` + `as_of()` look-ahead firewall; OpenBB and synthetic sources |
| `qbt/signals.py` | `Strategy` protocol, 8 strategies, `TrendFilter`/`Composite`/`InverseVolWeighted` wrappers |
| `qbt/risk.py` | `RiskGate` (vol target, caps, drawdown breaker), `DayTradeLedger` |
| `qbt/engine.py` | `Backtester` (T+1 fills, costs), performance metrics |
| `qbt/research.py` | IC grid, autocorrelation, walk-forward, multiple-testing haircut |
| `qbt/live.py` | `LiveSignalRunner` → order intents (no execution) |
| `qbt/broker.py` | `BrokerAdapter` protocol, `MockBroker`, `RobinhoodMCPBroker` |
| `qbt/orders.py` | `OrderManager`: journal, preflight, reconciliation, audit |
| `research.ipynb` | 41-cell research notebook, 26 code cells |
| `run_cycle.py` | One trading cycle. Run on a schedule, never from the notebook |
| `test_qbt.py` | 72 engine validation checks (strategies, engine, risk gate, research) |
| `test_orders.py` | 46 order-path checks, offline against `MockBroker` |
| `build_notebook.py` | Regenerates `research.ipynb` |
| `smoke_test.py` | Executes every notebook cell in one namespace |

## Run

```bash
pip install pandas numpy scipy matplotlib          # required
pip install openbb openbb-yfinance                # optional, for real data
pip install mcp                                   # optional, for RobinhoodMCPBroker
python test_qbt.py        # 72 engine checks, ~90s
python test_orders.py     # 46 order-path checks, ~10s
python smoke_test.py      # all 26 notebook cells, ~120s
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

Plus three composers that wrap any strategy: `TrendFilter` (absolute-momentum
gate), `InverseVolWeighted` (re-weight picks by inverse vol), and `Composite`
(blend several strategies by fixed capital share).

## The three tests worth keeping permanently

1. **Null test** (`test_qbt.py` §2). On data with no exploitable structure, every
   strategy must score a Sharpe indistinguishable from zero. Currently 0.41
   (t=1.60) and 0.10 (t=0.41) for the original two, and all four new
   strategies land in the same range (|sharpe| < 0.6, |t| < 2.6). If a
   strategy looks profitable here, the harness is leaking, not the market
   paying.
2. **Look-ahead firewall** (§1). A strategy that records the furthest date it saw
   must never exceed its decision bar.
3. **Live parity** (§7 and notebook cell 22). The live runner and a direct gate
   call must produce bit-identical weights. Currently exact — 0.00e+00.

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

- **`OpenBBRepository` is untested against a live provider.** It was written
  without network access. It handles both single- and multi-symbol response
  shapes and prefers `adj_close` where present, but expect to adjust
  `_fetch_remote` on first contact. This is the first thing to verify.
- **No point-in-time data.** Fine for price-only signals on ETFs. Extend to
  fundamentals and you need a PIT store — implement it as an OpenBB provider and
  the repository swap buys correctness, not just convenience.
- **Fill model is a fixed spread.** No partial fills, gaps, halts, or
  size-dependent impact.
- **Walk-forward retention ratio > 1 on synthetic data** (OOS 0.97 vs IS 0.60).
  That is an artifact of the generator's stationarity. Real data will not do
  this; treat a ratio above 1 as a signal your data is too well-behaved.
- **`PairsTrading` selects pairs by correlation, not cointegration.** Two
  correlated series can still drift apart permanently; a proper Engle-Granger
  or Johansen test is the next step before trusting this beyond synthetic
  data. Legs are also equal-dollar rather than beta-hedged.
- **`MultiFactorCrossSectional` has no real "quality" factor.** All three
  factors are derived from price and volatility alone (see "no
  point-in-time data" above); the `reversal` factor is the only one carrying
  information independent of momentum.
- **`CalendarSeasonality` uses calendar days, not the trading calendar.**
  `today.day` vs `today.days_in_month` approximates "last N trading days of
  the month" without accounting for holidays. It also needs `rebalance='D'`
  or `'W'` to express its logic at all — see the class docstring.
- **`RiskParityAllocation`'s covariance estimate gets noisy fast.** 126 bars
  against up to 30 names is a thin sample for a full covariance matrix;
  `max_names` and `cov_lookback` are levers to pull before trusting the
  weights on real data.

## Suggested next steps

1. Verify `OpenBBRepository` against yfinance, fix the response shape handling.
2. Set `USE_OPENBB = True`, rerun. Results will be worse — that is information.
3. Read the IC grid on real data **before** looking at any equity curve.
4. Move `test_qbt.py` into a real `tests/` directory with pytest.
5. Persist the `audit` frame from every run. It is the only thing that
   distinguishes "the strategy stopped working" from "I changed something".

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
— a blind retry after an unknown outcome is the most likely way this loses money
for reasons unrelated to the strategy. Deduplication keys on `(symbol, side)`
within a plan, deliberately *not* on quantity: keying on a float would let a 1%
equity drift between runs defeat it.

### Tool discovery

`RobinhoodMCPBroker` enumerates the server's tools at connect time and resolves
argument names from each input schema, so a server that says `ticker` where you
assumed `symbol` still works. Missing capabilities raise at connect, not at the
first order. Run `broker.list_capabilities()` first.

### What I could not verify

- **The OAuth handshake.** No account or network in the build environment. This
  is the first thing to confirm.
- **Exact parameter names and response shapes.** Discovery and the coercion
  helpers in `broker.py` are written to absorb variation, but expect to add
  names to `CAPABILITY_CANDIDATES` on first contact.
- **Rate limits and fractional-share support.** Undocumented as far as I found.

The entire order path is tested against `MockBroker`, which models partial
fills, rejections, and untradeable symbols — so the logic is verified even
though the transport is not.

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
