"""Validate KalshiPanel's point-in-time firewall, ladder-confidence math,
and KalshiRepository's parsing of the (live-confirmed) Kalshi API shapes.
No network -- KalshiRepository._get is stubbed throughout, the same pattern
test_corporate.py uses for CorpsRepository._fetch_raw.
"""

import pandas as pd

import os

import numpy as np

from qbt.kalshi import (
    DEFAULT_MIN_CONFIDENCE, DEFAULT_SERIES, KalshiFetchTimeout, KalshiPanel, KalshiRepository,
)

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

# 3d. Cumulative ladder, second series -- the real KXU3-26SEP numbers fetched
# live (2026-09-20, 12 days before close): same 14-strike/$0.10-increment
# shape as cpi, largest bucket lands around 0.29.
u3_ladder = _frame([
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T3.7", 3.7, "2026-09-20", "2026-10-02", 0.99, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T3.8", 3.8, "2026-09-20", "2026-10-02", 0.97, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T3.9", 3.9, "2026-09-20", "2026-10-02", 0.94, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.0", 4.0, "2026-09-20", "2026-10-02", 0.69, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.1", 4.1, "2026-09-20", "2026-10-02", 0.32, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.2", 4.2, "2026-09-20", "2026-10-02", 0.12, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.3", 4.3, "2026-09-20", "2026-10-02", 0.05, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.4", 4.4, "2026-09-20", "2026-10-02", 0.02, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.5", 4.5, "2026-09-20", "2026-10-02", 0.02, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.6", 4.6, "2026-09-20", "2026-10-02", 0.03, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.7", 4.7, "2026-09-20", "2026-10-02", 0.09, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.8", 4.8, "2026-09-20", "2026-10-02", 0.06, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T4.9", 4.9, "2026-09-20", "2026-10-02", 0.08, 1, 1),
    ("unemployment", "KXU3-26SEP", "KXU3-26SEP-T5.0", 5.0, "2026-09-20", "2026-10-02", 0.25, 1, 1),
])
u3_panel = KalshiPanel(frame=u3_ladder)
u3_conf = u3_panel.snapshot(pd.Timestamp("2026-09-20"))
check("unemployment's live ladder picks the largest bucket (~0.29)",
      abs(u3_conf["unemployment"] - 0.294) < 0.01, u3_conf.to_dict())

# 3e. Cumulative ladder, third series -- the real KXPCECORE-26AUG numbers
# fetched live (2026-09-20, 10 days before close): a coarser 8-strike
# ladder than cpi's, largest bucket lands around 0.59.
pce_ladder = _frame([
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.0", 0.0, "2026-09-20", "2026-09-30", 0.99, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.1", 0.1, "2026-09-20", "2026-09-30", 0.99, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.2", 0.2, "2026-09-20", "2026-09-30", 0.66, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.3", 0.3, "2026-09-20", "2026-09-30", 0.02, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.4", 0.4, "2026-09-20", "2026-09-30", 0.11, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.5", 0.5, "2026-09-20", "2026-09-30", 0.01, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.6", 0.6, "2026-09-20", "2026-09-30", 0.01, 1, 1),
    ("pce_core", "KXPCECORE-26AUG", "KXPCECORE-26AUG-T0.7", 0.7, "2026-09-20", "2026-09-30", 0.01, 1, 1),
])
pce_panel = KalshiPanel(frame=pce_ladder)
pce_conf = pce_panel.snapshot(pd.Timestamp("2026-09-20"))
check("pce_core's live ladder picks the largest bucket (~0.59), and the "
      "non-monotonic T0.3/T0.4 quotes (stale-quote noise) don't break it",
      abs(pce_conf["pce_core"] - 0.587) < 0.01, pce_conf.to_dict())

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
print("6b. Throttle + retry on 429 / 5xx")
print("=" * 72)


class _FakeResp:
    def __init__(self, status, body=None, headers=None, json_error=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


class _FlakyRepo(KalshiRepository):
    """Replays a scripted sequence of responses through the real _get."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = list(responses)
        self.sends = 0
        self.sleeps = []
        self._sleep = self.sleeps.append

    def _send(self, path, params, headers):
        self.sends += 1
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


flaky = _FlakyRepo([_FakeResp(429), _FakeResp(503), _FakeResp(200, {"ok": 1})],
                   cache_dir=None, min_request_interval=0.0)
check("a 429 then a 503 are retried until the 200 arrives",
      flaky._get("/markets") == {"ok": 1} and flaky.sends == 3, flaky.sends)
check("retries back off exponentially (1s, 2s)", flaky.sleeps == [1.0, 2.0], flaky.sleeps)

honours = _FlakyRepo([_FakeResp(429, headers={"Retry-After": "7"}), _FakeResp(200, {})],
                     cache_dir=None, min_request_interval=0.0)
honours._get("/markets")
check("Retry-After is honoured when Kalshi sends it", honours.sleeps == [7.0], honours.sleeps)

exhausted = _FlakyRepo([_FakeResp(429)] * 3, cache_dir=None,
                       min_request_interval=0.0, max_retries=2)
try:
    exhausted._get("/markets")
    check("a persistent 429 eventually raises instead of looping forever", False)
except RuntimeError:
    check("a persistent 429 eventually raises instead of looping forever",
          exhausted.sends == 3, exhausted.sends)

not_retried = _FlakyRepo([_FakeResp(404)], cache_dir=None, min_request_interval=0.0)
try:
    not_retried._get("/markets")
    check("a 404 is not retried", False)
except RuntimeError:
    check("a 404 is not retried", not_retried.sends == 1 and not not_retried.sleeps)

spaced = _FlakyRepo([_FakeResp(200, {}), _FakeResp(200, {})], cache_dir=None,
                    min_request_interval=0.5)
spaced._get("/markets")
spaced._get("/markets")
check("back-to-back requests are spaced by min_request_interval",
      len(spaced.sleeps) == 1 and 0.4 < spaced.sleeps[0] <= 0.5, spaced.sleeps)

long_wait = _FlakyRepo([_FakeResp(429, headers={"Retry-After": "60"}), _FakeResp(200, {})],
                       cache_dir=None, min_request_interval=0.0)
long_wait._get("/markets")
check("a Retry-After longer than backoff_max is still honoured in full",
      long_wait.sleeps == [60.0], long_wait.sleeps)

import json

import requests

transient = _FlakyRepo(
    [
        requests.ConnectionError("reset"),
        requests.Timeout("slow"),
        requests.exceptions.ChunkedEncodingError("cut off"),
        _FakeResp(200, json_error=json.JSONDecodeError("not json", "<html>", 0)),
        _FakeResp(200, {"ok": 1}),
    ],
    cache_dir=None, min_request_interval=0.0,
)
check("connection / timeout / truncated-body / non-JSON-200 failures are all retried",
      transient._get("/markets") == {"ok": 1} and transient.sends == 5, transient.sends)

conn_exhausted = _FlakyRepo([requests.ConnectionError("down")] * 2, cache_dir=None,
                            min_request_interval=0.0, max_retries=1)
try:
    conn_exhausted._get("/markets")
    check("a persistent connection failure raises after max_retries", False)
except requests.ConnectionError:
    check("a persistent connection failure raises after max_retries",
          conn_exhausted.sends == 2, conn_exhausted.sends)


class _SlowRepo(_StubRepo):
    """Every candlestick request 429s; the fake clock advances only by
    what the retry loop sleeps, so the deadline is hit deterministically."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.now = 0.0
        self._clock = lambda: self.now
        self._sleep = self._advance

    def _advance(self, seconds):
        self.now += seconds

    def _send(self, path, params, headers):
        self.calls.append((path, dict(params or {})))
        if path == "/markets":
            return _FakeResp(200, {"markets": [_MARKET], "cursor": ""})
        return _FakeResp(429)

    _get = KalshiRepository._get   # the real retry loop, not _StubRepo's canned one


slow = _SlowRepo(series={"cpi": "KXCPI"}, cache_dir=None, fetch_timeout=5.0,
                 max_retries=10)
try:
    slow.fetch("2026-07-01", "2026-12-01")
    check("fetch_timeout bounds a whole fetch, retries included", False)
except KalshiFetchTimeout:
    check("fetch_timeout bounds a whole fetch, retries included",
          slow.now <= 5.0, f"fake clock at {slow.now}s")
check("the deadline is cleared once fetch returns or raises", slow._deadline is None)

print()
print("=" * 72)
print("6c. Finalized markets cached per market, independent of the window")
print("=" * 72)

_CLOSED_MARKET = dict(
    _MARKET,
    ticker="KXCPI-26JUL-T0.6", event_ticker="KXCPI-26JUL",
    open_time="2026-07-01T14:00:00Z", close_time="2026-09-05T12:25:00Z",
)


class _ClosedRepo(_StubRepo):
    def _get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/markets":
            return {"markets": [_CLOSED_MARKET], "cursor": ""}
        return _CANDLES


cache_dir = "/tmp/qbt_kalshi_test_market_cache"
shutil.rmtree(cache_dir, ignore_errors=True)
try:
    closed = _ClosedRepo(series={"cpi": "KXCPI"}, cache_dir=cache_dir)
    first = closed.fetch("2026-08-01", "2026-09-10")
    candle_calls = [c for c in closed.calls if c[0].endswith("/candlesticks")]
    check("a finalized market's candles are requested over its whole life",
          candle_calls[0][1]["start_ts"] == int(pd.Timestamp("2026-07-01T14:00:00").timestamp()),
          candle_calls)
    closed.calls.clear()
    # A different window (next day's run) misses the per-window cache but
    # must hit the per-market one: no candlestick request at all.
    second = closed.fetch("2026-08-02", "2026-09-11")
    check("next day's window re-reads the finalized market from cache, not the network",
          not any(p.endswith("/candlesticks") for p, _ in closed.calls), closed.calls)
    check("the cached read-back returns the same rows as the fresh fetch",
          len(first) == len(second) == 2
          and np.allclose(first.frame["yes_price"], second.frame["yes_price"]),
          second.frame.to_string())
    early = closed.fetch("2026-08-03", "2026-09-01")
    check("rows after the window end are dropped",
          list(early.frame["snapshot_date"]) == [pd.Timestamp("2026-09-01")],
          early.frame.to_string())
    narrow = closed.fetch("2026-09-02", "2026-09-11")
    check("rows before the window start are dropped",
          list(narrow.frame["snapshot_date"]) == [pd.Timestamp("2026-09-02")],
          narrow.frame.to_string())

    # A truncated entry (a run killed mid-write, before writes were atomic)
    # must be dropped and refetched, not fail every later fetch.
    market_files = [f for f in os.listdir(cache_dir) if f.startswith("market-")]
    check("exactly one per-market entry was written", len(market_files) == 1, market_files)
    with open(os.path.join(cache_dir, market_files[0]), "wb") as fh:
        fh.write(b"\x1f\x8b\x08 truncated")
    closed.calls.clear()
    healed = closed.fetch("2026-08-04", "2026-09-11")
    check("a corrupt per-market entry is refetched rather than failing the fetch",
          len(healed) == 2 and any(p.endswith("/candlesticks") for p, _ in closed.calls),
          closed.calls)
    check("no temp files are left behind by the atomic write",
          not [f for f in os.listdir(cache_dir) if f.endswith(".tmp")], os.listdir(cache_dir))

    # Same ticker under a different friendly name must not be served rows
    # stamped with the other repository's label.
    renamed = _ClosedRepo(series={"inflation": "KXCPI"}, cache_dir=cache_dir)
    relabelled = renamed.fetch("2026-08-01", "2026-09-10")
    check("the per-market cache is keyed on the series name, not just the ticker",
          set(relabelled.frame["series"]) == {"inflation"}
          and any(p.endswith("/candlesticks") for p, _ in renamed.calls),
          relabelled.frame["series"].unique())
finally:
    shutil.rmtree(cache_dir, ignore_errors=True)


print()
print("=" * 72)
print("6d. One window rule on every path; no wasted history without a cache")
print("=" * 72)

# Pinned clock: _MARKET (closing 2026-10-14) must read as still open here
# whatever day this suite actually runs on.
_PINNED_NOW = pd.Timestamp("2026-09-23T15:00:00")

open_repo = _StubRepo(series={"cpi": "KXCPI"}, cache_dir=None)
open_repo._utcnow = lambda: _PINNED_NOW
open_repo.fetch("2026-08-01", "2026-09-10")
(open_params,) = [p for path, p in open_repo.calls if path.endswith("/candlesticks")]
check("an open market's request runs to the end of end's calendar day",
      open_params["end_ts"] == int(pd.Timestamp("2026-09-10T23:59:59").timestamp()),
      open_params)

future_repo = _StubRepo(series={"cpi": "KXCPI"}, cache_dir=None)
future_repo._utcnow = lambda: _PINNED_NOW
future_repo.fetch("2026-08-01", "2026-12-01")
(future_params,) = [p for path, p in future_repo.calls if path.endswith("/candlesticks")]
check("...but never past now, even for a window ending in the future",
      future_params["end_ts"] == int(_PINNED_NOW.timestamp()), future_params)

uncached = _ClosedRepo(series={"cpi": "KXCPI"}, cache_dir=None)
uncached_panel = uncached.fetch("2026-08-15", "2026-09-01")
(uncached_params,) = [p for path, p in uncached.calls if path.endswith("/candlesticks")]
check("with no cache, a finalized market is fetched for the window, not its whole life",
      uncached_params["start_ts"] == int(pd.Timestamp("2026-08-15").timestamp()),
      uncached_params)

cache_dir = "/tmp/qbt_kalshi_test_window_rule"
shutil.rmtree(cache_dir, ignore_errors=True)
try:
    cached_panel = _ClosedRepo(series={"cpi": "KXCPI"}, cache_dir=cache_dir).fetch(
        "2026-08-15", "2026-09-01")
    check("cached (whole-life) and uncached (windowed) paths return the same rows",
          uncached_panel.frame["snapshot_date"].tolist()
          == cached_panel.frame["snapshot_date"].tolist()
          == [pd.Timestamp("2026-09-01")],
          (uncached_panel.frame["snapshot_date"].tolist(),
           cached_panel.frame["snapshot_date"].tolist()))
finally:
    shutil.rmtree(cache_dir, ignore_errors=True)

import qbt.kalshi as _kalshi_mod


class _FakeSession:
    created = 0

    def __init__(self):
        type(self).created += 1
        self.gets = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets += 1
        return _FakeResp(200, {"ok": 1})


class _FakeRequests:
    Session = _FakeSession


_real_requests = _kalshi_mod._requests
_kalshi_mod._requests = lambda: _FakeRequests
try:
    session_repo = KalshiRepository(cache_dir=None, min_request_interval=0.0)
    for _ in range(3):
        session_repo._send("/markets", {}, {})
    check("one HTTP session is created and reused across requests",
          _FakeSession.created == 1 and session_repo._session.gets == 3,
          (_FakeSession.created, session_repo._session.gets))
finally:
    _kalshi_mod._requests = _real_requests


class _EmptyClosedRepo(_ClosedRepo):
    def _get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/markets":
            return {"markets": [_CLOSED_MARKET], "cursor": ""}
        return {"candlesticks": []}


cache_dir = "/tmp/qbt_kalshi_test_empty_cache"
shutil.rmtree(cache_dir, ignore_errors=True)
try:
    _EmptyClosedRepo(series={"cpi": "KXCPI"}, cache_dir=cache_dir).fetch(
        "2026-08-01", "2026-09-10")
    check("an empty candlestick answer for a finalized market is not cached for good",
          not [f for f in os.listdir(cache_dir) if f.startswith("market-")],
          os.listdir(cache_dir))
finally:
    shutil.rmtree(cache_dir, ignore_errors=True)

print()
print("=" * 72)
print("7. Defaults are sane")
print("=" * 72)

check("DEFAULT_SERIES covers the six liquid series from the liquidity survey",
      set(DEFAULT_SERIES) == {
          "cpi", "fed_decision", "payrolls", "recession", "unemployment", "pce_core",
      }, DEFAULT_SERIES)
check("DEFAULT_MIN_CONFIDENCE has a floor for every default series",
      set(DEFAULT_MIN_CONFIDENCE) == set(DEFAULT_SERIES), DEFAULT_MIN_CONFIDENCE)
check("fed_decision's floor is higher than cpi's -- coarser ladders concentrate more",
      DEFAULT_MIN_CONFIDENCE["fed_decision"] > DEFAULT_MIN_CONFIDENCE["cpi"])
check("unemployment's floor sits close to cpi's -- same 14-strike ladder shape",
      abs(DEFAULT_MIN_CONFIDENCE["unemployment"] - DEFAULT_MIN_CONFIDENCE["cpi"]) < 0.06)
check("pce_core's floor sits between payrolls' and fed_decision's -- an "
      "8-strike ladder, coarser than cpi's 14 but finer than fed_decision's 5",
      DEFAULT_MIN_CONFIDENCE["payrolls"]
      < DEFAULT_MIN_CONFIDENCE["pce_core"]
      < DEFAULT_MIN_CONFIDENCE["fed_decision"])

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
