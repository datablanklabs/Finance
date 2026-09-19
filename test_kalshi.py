"""Validate KalshiPanel's point-in-time firewall, ladder-confidence math,
and KalshiRepository's parsing of the (live-confirmed) Kalshi API shapes.
No network -- KalshiRepository._get is stubbed throughout, the same pattern
test_corporate.py uses for CorpsRepository._fetch_raw.
"""

import pandas as pd

from qbt.kalshi import DEFAULT_MIN_CONFIDENCE, DEFAULT_SERIES, KalshiPanel, KalshiRepository

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


def _frame(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "series", "event_ticker", "market_ticker", "strike",
            "snapshot_date", "close_time", "yes_price", "volume", "open_interest",
        ],
    )
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    return df


print("=" * 72)
print("1. Construction and validation")
print("=" * 72)

rows = [
    # series, event_ticker,   market_ticker,      strike, snapshot_date, close_time,  yes_price, volume, oi
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.5", 0.5, "2026-10-06", "2026-10-14", 0.66, 38599.97, 100.0),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, "2026-10-06", "2026-10-14", 0.31, 73417.40, 200.0),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.7", 0.7, "2026-10-06", "2026-10-14", 0.09, 5533.00, 50.0),
]
frame = _frame(rows)

panel = KalshiPanel(frame=frame)
print(panel.describe())
check("series_names", panel.series_names == ["cpi"])
check("len", len(panel) == len(rows))

try:
    KalshiPanel(frame=frame.drop(columns=["close_time"]))
    check("rejects missing close_time column", False)
except ValueError:
    check("rejects missing close_time column", True)

try:
    bad = frame.copy()
    bad["snapshot_date"] = bad["snapshot_date"].astype(str)
    KalshiPanel(frame=bad)
    check("rejects non-datetime snapshot_date", False)
except TypeError:
    check("rejects non-datetime snapshot_date", True)

empty = KalshiPanel(frame=frame.iloc[0:0])
check("describe handles an empty panel", empty.describe().startswith("KalshiPanel(0"))
check("snapshot on an empty panel returns an empty Series",
      empty.snapshot(pd.Timestamp("2026-10-06")).empty)

print()
print("=" * 72)
print("2. Look-ahead firewall (as_of)")
print("=" * 72)

d1, d2, d3 = "2026-10-01", "2026-10-06", "2026-10-10"
history = _frame([
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, d1, "2026-10-14", 0.20, 1000.0, 500.0),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, d2, "2026-10-14", 0.31, 2000.0, 700.0),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, d3, "2026-10-14", 0.45, 3000.0, 900.0),
])
hist_panel = KalshiPanel(frame=history)

sliced = hist_panel.as_of(pd.Timestamp(d2))
check("as_of drops future snapshots", sliced.frame["snapshot_date"].max() == pd.Timestamp(d2))
check("as_of keeps prior snapshots", len(sliced) == 2)
check("as_of never returns a snapshot after the cutoff",
      (sliced.frame["snapshot_date"] <= pd.Timestamp(d2)).all())

print()
print("=" * 72)
print("3. Confidence math -- the three ladder shapes")
print("=" * 72)

# 3a. Plain binary (recession-style): one market, no strike.
binary = _frame([
    ("recession", "KXRECSSNBER-26", "KXRECSSNBER-26", float("nan"),
     "2026-09-01", "2026-12-31", 0.18, 500.0, 300.0),
])
bpanel = KalshiPanel(frame=binary)
conf = bpanel.snapshot(pd.Timestamp("2026-09-01"))
check("binary confidence is max(p, 1-p)", abs(conf["recession"] - 0.82) < 1e-9, conf.to_dict())

# 3b. Cumulative P(X>strike) ladder -- the real KXCPI-26SEP numbers fetched
# live (2026-09-19, 8 days before close): confidence should land on the
# 0.5-0.6 bucket (0.66-0.31=0.35), the largest mass in this ladder.
cpi_ladder = _frame([
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T-0.4", -0.4, "2026-10-06", "2026-10-14", 0.99, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.0", 0.0, "2026-10-06", "2026-10-14", 0.99, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.3", 0.3, "2026-10-06", "2026-10-14", 0.86, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.5", 0.5, "2026-10-06", "2026-10-14", 0.66, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, "2026-10-06", "2026-10-14", 0.31, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.7", 0.7, "2026-10-06", "2026-10-14", 0.09, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.9", 0.9, "2026-10-06", "2026-10-14", 0.01, 1, 1),
])
cpanel = KalshiPanel(frame=cpi_ladder)
conf = cpanel.snapshot(pd.Timestamp("2026-10-06"))
check("cumulative-ladder confidence picks the largest bucket (~0.35)",
      abs(conf["cpi"] - 0.35) < 0.02, conf.to_dict())

# A market split evenly across many strikes should read as low confidence.
flat_ladder = _frame([
    ("cpi", "KXCPI-26OCT", "KXCPI-26OCT-T0.1", 0.1, "2026-11-01", "2026-11-10", 0.80, 1, 1),
    ("cpi", "KXCPI-26OCT", "KXCPI-26OCT-T0.2", 0.2, "2026-11-01", "2026-11-10", 0.60, 1, 1),
    ("cpi", "KXCPI-26OCT", "KXCPI-26OCT-T0.3", 0.3, "2026-11-01", "2026-11-10", 0.40, 1, 1),
    ("cpi", "KXCPI-26OCT", "KXCPI-26OCT-T0.4", 0.4, "2026-11-01", "2026-11-10", 0.20, 1, 1),
])
flat_panel = KalshiPanel(frame=flat_ladder)
flat_conf = flat_panel.snapshot(pd.Timestamp("2026-11-01"))
check("an evenly-split ladder reads as low confidence",
      flat_conf["cpi"] < 0.25, flat_conf.to_dict())
check("evenly-split is less confident than the resolved-ish ladder",
      flat_conf["cpi"] < conf["cpi"])

# 3c. Mutually-exclusive categorical buckets (fed_decision-style): no
# strike at all, each market's own yes_price already is a bucket mass.
# Real KXFEDDECISION-27DEC numbers (last_price_dollars, 2026-09-19).
fed = _frame([
    ("fed_decision", "KXFEDDECISION-27DEC", "KXFEDDECISION-27DEC-H0", float("nan"),
     "2026-09-19", "2027-12-08", 0.58, 1, 1),
    ("fed_decision", "KXFEDDECISION-27DEC", "KXFEDDECISION-27DEC-H25", float("nan"),
     "2026-09-19", "2027-12-08", 0.08, 1, 1),
    ("fed_decision", "KXFEDDECISION-27DEC", "KXFEDDECISION-27DEC-H26", float("nan"),
     "2026-09-19", "2027-12-08", 0.08, 1, 1),
    ("fed_decision", "KXFEDDECISION-27DEC", "KXFEDDECISION-27DEC-C25", float("nan"),
     "2026-09-19", "2027-12-08", 0.10, 1, 1),
    ("fed_decision", "KXFEDDECISION-27DEC", "KXFEDDECISION-27DEC-C26", float("nan"),
     "2026-09-19", "2027-12-08", 0.08, 1, 1),
])
fed_panel = KalshiPanel(frame=fed)
fed_conf = fed_panel.snapshot(pd.Timestamp("2026-09-19"))
check("categorical confidence normalises and picks the dominant bucket (~0.63)",
      abs(fed_conf["fed_decision"] - 0.63) < 0.02, fed_conf.to_dict())
check("a coarse 5-bucket ladder reads far more confident than a 14-strike one",
      fed_conf["fed_decision"] > conf["cpi"] + 0.2)

print()
print("=" * 72)
print("4. snapshot()/days_to_close() -- nearest open event, staleness, absence")
print("=" * 72)

# Two events for the same series: one already closed, one still open --
# only the open one should ever surface.
two_events = _frame([
    ("cpi", "KXCPI-26AUG", "KXCPI-26AUG-T0.5", 0.5, "2026-09-01", "2026-09-11", 0.90, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.5", 0.5, "2026-09-01", "2026-10-14", 0.60, 1, 1),
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.6", 0.6, "2026-09-01", "2026-10-14", 0.20, 1, 1),
])
te_panel = KalshiPanel(frame=two_events)
asof = pd.Timestamp("2026-09-15")  # after AUG closes, before SEP closes
snap = te_panel.snapshot(asof)
days = te_panel.days_to_close(asof)
check("snapshot ignores an already-closed event",
      "cpi" in snap.index and days["cpi"] == (pd.Timestamp("2026-10-14") - asof).days,
      (snap.to_dict(), days.to_dict()))

# Staleness: the only reading is from 10 days ago; max_age_days=3 should drop it.
stale = _frame([
    ("cpi", "KXCPI-26SEP", "KXCPI-26SEP-T0.5", 0.5, "2026-09-01", "2026-10-14", 0.60, 1, 1),
])
stale_panel = KalshiPanel(frame=stale)
fresh_read = stale_panel.snapshot(pd.Timestamp("2026-09-05"), max_age_days=10)
stale_read = stale_panel.snapshot(pd.Timestamp("2026-09-20"), max_age_days=3)
check("a fresh-enough reading passes max_age_days", "cpi" in fresh_read.index)
check("a reading older than max_age_days is treated as absent, not returned",
      "cpi" not in stale_read.index)

# A series with no data at all is simply absent from the Series, not NaN/0.
check("an untracked series is absent from snapshot()", "payrolls" not in snap.index)
check("an untracked series is absent from days_to_close()", "payrolls" not in days.index)

print()
print("=" * 72)
print("5. KalshiRepository -- parsing the confirmed live API shapes (stubbed HTTP)")
print("=" * 72)

_MARKET = {
    "ticker": "KXCPI-26SEP-T0.6",
    "event_ticker": "KXCPI-26SEP",
    "open_time": "2026-07-20T18:00:00Z",
    "close_time": "2026-10-14T12:25:00Z",
    "strike_type": "greater",
    "floor_strike": 0.6,
}
_CANDLES = {
    "ticker": "KXCPI-26SEP-T0.6",
    "candlesticks": [
        {  # a real trade that day -- confirmed shape
            "end_period_ts": int(pd.Timestamp("2026-09-01").timestamp()),
            "price": {"close_dollars": "0.3100"},
            "yes_bid": {"close_dollars": "0.2700"},
            "yes_ask": {"close_dollars": "0.3100"},
            "volume_fp": "73386.63",
            "open_interest_fp": "12755.25",
        },
        {  # no trade that day -- price is empty (confirmed live shape);
           # must fall back to the bid/ask midpoint, not drop the day.
            "end_period_ts": int(pd.Timestamp("2026-09-02").timestamp()),
            "price": {},
            "yes_bid": {"close_dollars": "0.0100"},
            "yes_ask": {"close_dollars": "0.0500"},
            "volume_fp": "0.00",
            "open_interest_fp": "0.00",
        },
    ],
}


class _StubRepo(KalshiRepository):
    """Replays canned market-list/candlestick pages -- no network, and no
    reliance on cursor pagination actually looping more than once (that's
    exercised separately below)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def _get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/markets":
            return {"markets": [_MARKET], "cursor": ""}
        if path.endswith("/candlesticks"):
            return _CANDLES
        raise AssertionError(f"unexpected path: {path}")


repo = _StubRepo(series={"cpi": "KXCPI"}, cache_dir=None)
kpanel = repo.fetch("2026-07-01", "2026-12-01")
check("fetch returns a non-empty KalshiPanel", len(kpanel) == 2, kpanel.describe())
check("series/event/market fields parsed", set(kpanel.frame["series"]) == {"cpi"}
      and set(kpanel.frame["event_ticker"]) == {"KXCPI-26SEP"}
      and set(kpanel.frame["market_ticker"]) == {"KXCPI-26SEP-T0.6"})
check("floor_strike parsed as a numeric strike (strike_type='greater')",
      (kpanel.frame["strike"] == 0.6).all())
check("close_time parsed and tz-stripped to a plain Timestamp",
      kpanel.frame["close_time"].iloc[0] == pd.Timestamp("2026-10-14T12:25:00"))

by_date = kpanel.frame.set_index("snapshot_date")["yes_price"]
check("a traded day uses price.close_dollars",
      abs(by_date[pd.Timestamp("2026-09-01")] - 0.31) < 1e-9, by_date.to_dict())
check("a no-trade day falls back to the yes_bid/yes_ask midpoint",
      abs(by_date[pd.Timestamp("2026-09-02")] - 0.03) < 1e-9, by_date.to_dict())

check("candlestick request used the market's own ticker path",
      any(p == "/series/KXCPI/markets/KXCPI-26SEP-T0.6/candlesticks" for p, _ in repo.calls),
      repo.calls)
check("the /markets list request filtered by series_ticker",
      repo.calls[0] == ("/markets", {"series_ticker": "KXCPI", "min_close_ts": repo.calls[0][1]["min_close_ts"], "limit": 200}),
      repo.calls[0])


class _PagedRepo(KalshiRepository):
    """Two-page /markets response -- exercises the cursor loop."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_pages = [
            {"markets": [dict(_MARKET, ticker="KXCPI-26SEP-T0.6")], "cursor": "page2"},
            {"markets": [dict(_MARKET, ticker="KXCPI-26SEP-T0.7", floor_strike=0.7)], "cursor": ""},
        ]

    def _get(self, path, params=None):
        if path == "/markets":
            return self.market_pages.pop(0)
        return {"ticker": "x", "candlesticks": []}


paged = _PagedRepo(series={"cpi": "KXCPI"}, cache_dir=None)
paged_panel = paged.fetch("2026-07-01", "2026-12-01")
check("cursor pagination is followed across both pages",
      not paged.market_pages, "pages remaining after fetch: " + str(len(paged.market_pages)))
# both markets had empty candlesticks, so no rows -- but no crash either.
check("a market with no candlesticks in range contributes no rows, not an error",
      len(paged_panel) == 0)


class _RaisingRepo(KalshiRepository):
    def _get(self, path, params=None):
        raise RuntimeError("boom")


try:
    _RaisingRepo(series={"cpi": "KXCPI"}, cache_dir=None).fetch("2026-07-01", "2026-12-01")
    check("a hard failure propagates rather than looking like an empty-but-successful fetch", False)
except RuntimeError:
    check("a hard failure propagates rather than looking like an empty-but-successful fetch", True)

print()
print("=" * 72)
print("6. Disk cache round-trip")
print("=" * 72)

import shutil

cache_dir = "/tmp/qbt_kalshi_test_cache"
shutil.rmtree(cache_dir, ignore_errors=True)
try:
    cached_repo = _StubRepo(series={"cpi": "KXCPI"}, cache_dir=cache_dir)
    p1 = cached_repo.fetch("2026-07-01", "2026-12-01")
    calls_after_first = len(cached_repo.calls)
    p2 = cached_repo.fetch("2026-07-01", "2026-12-01")
    check("a cache hit does not touch the network again",
          len(cached_repo.calls) == calls_after_first, cached_repo.calls)
    check("the cached read-back matches the original fetch",
          len(p1) == len(p2) and len(p1) == 2)
finally:
    shutil.rmtree(cache_dir, ignore_errors=True)

print()
print("=" * 72)
print("7. Defaults are sane")
print("=" * 72)

check("DEFAULT_SERIES covers the four liquid series from the liquidity survey",
      set(DEFAULT_SERIES) == {"cpi", "fed_decision", "payrolls", "recession"}, DEFAULT_SERIES)
check("DEFAULT_MIN_CONFIDENCE has a floor for every default series",
      set(DEFAULT_MIN_CONFIDENCE) == set(DEFAULT_SERIES), DEFAULT_MIN_CONFIDENCE)
check("fed_decision's floor is higher than cpi's -- coarser ladders concentrate more",
      DEFAULT_MIN_CONFIDENCE["fed_decision"] > DEFAULT_MIN_CONFIDENCE["cpi"])

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for name in FAILS:
        print(f"  - {name}")
    raise SystemExit(1)
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
