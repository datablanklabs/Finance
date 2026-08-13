"""Order management: the layer between intents and the broker.

:class:`qbt.live.LiveSignalRunner` produces intents. This turns them into
orders without lying to you about what happened.

The hard problem is **idempotency**, and it is harder here than usual. The
Robinhood MCP's order tool does not document a client-supplied idempotency key,
so you cannot make submission safely repeatable by asking the broker to
deduplicate. The alternative is a write-ahead journal plus read-back:

1. Append ``planned`` to an append-only journal on disk. Flush and fsync.
2. Append ``submitting`` for a specific intent. Flush and fsync.
3. Call the broker.
4. Append the outcome.

If the process dies between 2 and 4, the journal shows an intent whose fate is
unknown. On restart, :meth:`OrderManager.recover` finds those entries and
resolves them by *reading the broker's order list* and matching fingerprints --
never by retrying. A blind retry after an unknown outcome is how you end up
with double the position you intended, and it is the single most likely way a
retail bot loses money for reasons unrelated to its strategy.

The second hard problem is that a partially executed rebalance is a portfolio
nobody chose. Preflight runs over the *whole plan*: if the plan as a set fails a
check, nothing is sent.

Audit events are emitted as one JSON object per line, with stable field names,
so the log is directly ingestible by a SIEM. See
:meth:`OrderManager.detection_rules` for the alerts worth wiring up.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .broker import BrokerAccount, BrokerAdapter, BrokerOrder
from .live import LivePlan, OrderIntent
from .risk import DayTradeLedger, as_session_date

try:
    import fcntl  # POSIX advisory file locking; unavailable on Windows.
except ImportError:  # pragma: no cover
    fcntl = None

__all__ = ["ExecutionPolicy", "OrderManager", "ExecutionReport", "AuditLog"]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only JSONL audit log, fsynced on every write.

    fsync costs a millisecond and buys you a journal that survives a hard kill.
    Given the journal is what prevents double submission, that trade is not
    close.
    """

    def __init__(self, path: str = "audit/orders.jsonl", stdout: bool = True) -> None:
        self.path = path
        self.stdout = stdout
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.host = socket.gethostname()
        self.pid = os.getpid()

    def emit(self, event: str, **fields) -> dict:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "host": self.host,
            "pid": self.pid,
            **fields,
        }
        line = json.dumps(rec, default=str, sort_keys=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self.stdout:
            print(f"  [{event}] " + " ".join(
                f"{k}={v}" for k, v in fields.items() if k != "raw"))
        return rec

    def read(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        rows = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class ExecutionPolicy:
    """Hard limits enforced outside the strategy.

    These are deliberately dumb, absolute, and checked on every cycle. The
    strategy and the risk gate are the intelligent layer; this is the layer that
    does not care how convincing the reasoning was.
    """

    max_order_notional: float = 2_500.0
    max_plan_notional: float = 10_000.0
    max_plan_turnover: float = 0.67
    # A flat account's first-ever buildout is structurally ~100% turnover --
    # current weights are all zero, so turnover and gross exposure are the
    # same number. A cap meant to catch excess *rebalancing* churn would
    # otherwise make it impossible to ever place the first trade. This is a
    # second dumb, mechanical rule (account currently holds zero positions,
    # full stop), not a judgment call about whether the trade "looks"
    # reasonable -- consistent with this class's own design.
    allow_full_turnover_from_flat: bool = True
    max_orders_per_cycle: int = 12
    max_position_weight: float = 0.35
    symbol_allowlist: tuple[str, ...] = ()
    require_review: bool = True
    allow_review_warnings: bool = False
    max_data_staleness_days: int = 4
    max_equity_drift: float = 0.05
    kill_switch_path: str = "KILL"
    require_market_open: bool = True
    session_open: dtime = dtime(9, 35)          # avoid the opening auction
    session_close: dtime = dtime(15, 50)        # and the closing one
    market_tz: str = "America/New_York"
    dry_run: bool = True                        # must be set False explicitly


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """Outcome of one :meth:`OrderManager.execute` pass.

    Three genuinely distinct outcomes, kept in three places rather than
    two. ``rejected`` used to be folded into ``submitted`` -- an order the
    broker refused was counted, and printed, as one we successfully placed,
    so a cycle where every single order bounced still reported "3
    submitted". The broker order object is worth keeping for those (it
    carries the order id, state and reject reason, which a plain intent
    does not), which is why they don't simply go into ``skipped``.
    """

    plan_id: str
    dry_run: bool
    submitted: list[BrokerOrder] = field(default_factory=list)
    skipped: list[tuple[OrderIntent, str]] = field(default_factory=list)
    rejected: list[BrokerOrder] = field(default_factory=list)
    aborted_reason: str | None = None
    preflight_notes: list[str] = field(default_factory=list)
    reconciliation: pd.DataFrame | None = None

    @property
    def aborted(self) -> bool:
        return self.aborted_reason is not None

    @property
    def accepted(self) -> list[BrokerOrder]:
        """Everything that reached the broker, accepted or refused.

        The old meaning of ``submitted``, kept for the "did this order
        leave the building" question, which is what recovery and
        reconciliation care about.
        """
        return [*self.submitted, *self.rejected]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for o in self.accepted:
            rows.append({"symbol": o.symbol, "side": o.side,
                         "quantity": round(o.quantity, 6), "state": o.state,
                         "filled": round(o.filled_quantity, 6),
                         "avg_price": o.average_price,
                         "order_id": o.order_id, "note": o.reject_reason or ""})
        for intent, reason in self.skipped:
            rows.append({"symbol": intent.symbol, "side": intent.side,
                         "quantity": round(abs(intent.shares), 6),
                         "state": "skipped", "filled": 0.0, "avg_price": None,
                         "order_id": "", "note": reason})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        if self.aborted:
            return f"ExecutionReport(ABORTED: {self.aborted_reason})"
        mode = "DRY RUN" if self.dry_run else "LIVE"
        filled = sum(1 for o in self.submitted if o.state == "filled")
        rejected = (f", {len(self.rejected)} rejected" if self.rejected else "")
        return (f"ExecutionReport({mode}, {len(self.submitted)} submitted, "
                f"{filled} filled{rejected}, {len(self.skipped)} skipped)")


def _clean_broker_rejection(text: str) -> str:
    """Pull the broker's actual rejection detail out of the exception soup.

    The real broker raises through nested anyio TaskGroups (see the comment
    at the ``text = repr(exc)`` call site below), so the one useful thing --
    the API's own JSON validation-error body -- is buried three exception
    layers deep inside a giant repr(). Confirmed live (2026-08), two shapes:
    ``{"detail": "..."}`` and ``{"<field>": ["...", ...]}``. Anything that
    doesn't match either shape falls back to the untouched input, so a new
    or unrecognized rejection is never silently hidden -- only the two
    known-clean cases get shortened.
    """
    match = re.search(r"API error 4\d\d: (\{[^}]*\})", text)
    if match is None:
        return text
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and "detail" in payload:
        detail = str(payload["detail"]).rstrip(".")
        if detail.lower() == "not enough buying power":
            return "Insufficient buying power"
        return detail
    if isinstance(payload, dict) and len(payload) == 1:
        (value,) = payload.values()
        if isinstance(value, list) and value:
            return str(value[0]).rstrip(".")
    return text


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class OrderManager:
    """Submits a :class:`LivePlan` through a broker adapter, or refuses to."""

    def __init__(
        self,
        broker: BrokerAdapter,
        policy: ExecutionPolicy | None = None,
        audit: AuditLog | None = None,
        journal_path: str = "audit/journal.jsonl",
        day_trade_ledger: DayTradeLedger | None = None,
    ) -> None:
        self.broker = broker
        self.policy = policy or ExecutionPolicy()
        self.audit = audit or AuditLog()
        self.journal_path = journal_path
        # Confirmed live (2026-08): RiskGate's PDT check (step 5 of
        # apply()) reads day_trades_remaining, but nothing in the live
        # path ever called .record(). Every LiveSignalRunner got a fresh,
        # empty DayTradeLedger every cycle, so the PDT limit could never
        # actually trigger in live trading. Pass the *same* ledger instance
        # the caller gives LiveSignalRunner (see run_cycle.py) so a day
        # trade recorded here during execute() is visible to the next
        # cycle's plan() via .remaining().
        #
        # This is the *only* place day trades are ever recorded. A daily
        # backtest cannot produce one at all -- one fill per symbol per
        # bar, one bar per session, so nothing round-trips inside a session
        # (see the opened_on comment in qbt/engine.py's Backtester.run()).
        # So a backtest showing no PDT pressure is not evidence that live
        # trading will stay inside the budget; this ledger, persisted
        # across cycles, is the only thing that actually knows.
        self.day_trade_ledger = day_trade_ledger or DayTradeLedger()
        os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)

    # -- concurrency --------------------------------------------------------

    @contextmanager
    def _dedup_lock(self) -> Iterator[None]:
        """Advisory exclusive lock scoped to ``journal_path``.

        Guards the read-then-write dedup sequence in :meth:`execute` --
        read what's already been submitted, decide what's left, submit it --
        against a second :class:`OrderManager` racing the same journal, e.g.
        an accidentally double-scheduled run. A per-instance lock (like
        ``threading.Lock``) can't help here: the risk is two separate
        instances, often in two separate processes, not two threads sharing
        one object. Best-effort: POSIX only (``fcntl``); on platforms
        without it this is a silent no-op, same as running without the lock
        at all.
        """
        if fcntl is None:  # pragma: no cover
            yield
            return
        lock_path = self.journal_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "a") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- journal ----------------------------------------------------------

    def _journal(self, **fields) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        with open(self.journal_path, "a") as fh:
            fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _journal_entries(self) -> list[dict]:
        if not os.path.exists(self.journal_path):
            return []
        out = []
        with open(self.journal_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        return out

    @staticmethod
    def plan_id(plan: LivePlan, strategy_name: str = "") -> str:
        """Deterministic id: the same plan on the same day yields the same id.

        This is what lets a re-run recognise itself rather than duplicating.
        """
        w = plan.target_weights.round(6)
        payload = "|".join([
            str(pd.Timestamp(plan.asof).date()), strategy_name,
            ",".join(f"{s}:{v}" for s, v in w[w.abs() > 0].sort_index().items()),
        ])
        return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex[:16]

    # -- recovery ---------------------------------------------------------

    def recover(self) -> pd.DataFrame:
        """Resolve intents whose outcome is unknown after an unclean shutdown.

        Never retries. Reads the broker's orders and matches fingerprints to
        decide what actually happened, then closes out the journal entry.

        Wrapped in the same lock :meth:`execute` takes around its own
        read-then-write dedup sequence -- recover() reads and writes this
        exact journal too, and without the lock a genuinely concurrent
        second process (the same double-scheduled-run risk _dedup_lock's
        own docstring calls out) could have its execute() mid-flight while
        this read a partial view: a "submitting" entry that's actually
        just in progress, not orphaned by a crash, queried against the
        broker and resolved before the other process's own attempt ever
        got there.
        """
        with self._dedup_lock():
            entries = self._journal_entries()
            by_key: dict[str, list[dict]] = {}
            for e in entries:
                key = f"{e.get('plan_id')}::{e.get('intent_key')}"
                by_key.setdefault(key, []).append(e)

            unresolved = [
                (k, es) for k, es in by_key.items()
                if any(e.get("stage") == "submitting" for e in es)
                and not any(e.get("stage") in ("submitted", "rejected", "resolved",
                                               "skipped")
                            for e in es)
            ]
            if not unresolved:
                self.audit.emit("recover_clean", unresolved=0)
                return pd.DataFrame()

            # The *first* "submitting" entry per key anchors the broker-query
            # window (the earliest moment something might have been sent); the
            # *last* one is what was actually most recently sent and is what
            # fingerprinting must match against. These differ whenever execute()
            # retried at a corrected quantity -- e.g. a fractional-share
            # rejection followed by a whole-share retry (see execute()) writes
            # a second "submitting" entry at the corrected size. Confirmed live
            # (2026-08): fingerprinting against the first entry's now-stale
            # original quantity searched for an order that was never actually
            # requested, and never found the one that was -- a real fill got
            # misresolved as not_at_broker.
            first_submitting = {
                key: next(e for e in es if e.get("stage") == "submitting")
                for key, es in unresolved
            }
            latest_submitting = {
                key: [e for e in es if e.get("stage") == "submitting"][-1]
                for key, es in unresolved
            }

            # Anchor the broker query to when the earliest unresolved attempt
            # actually happened, not to "today". recover() running on a later
            # calendar day than the crash -- the process was down over a
            # weekend, say, and comes back up Monday after a Friday-close
            # crash -- would otherwise never even ask the broker about an
            # order from before today's midnight, and a genuinely-placed order
            # gets wrongly resolved as not_at_broker.
            sub_times = []
            for sub in first_submitting.values():
                try:
                    sub_times.append(datetime.fromisoformat(sub["ts"]))
                except (KeyError, ValueError, TypeError):
                    continue
            since = (
                min(sub_times) - timedelta(minutes=5)
                if sub_times
                else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                                         microsecond=0)
            )
            broker_orders = list(self.broker.get_orders(since=since))

            rows = []
            for key, sub in first_submitting.items():
                latest = latest_submitting[key]
                sym = str(latest.get("symbol", "")).upper()
                side = str(latest.get("side", "")).lower()
                qty = float(latest.get("quantity", 0.0))
                # Tolerance comparison, not bucket equality. Bucketing put a
                # hard edge in the middle of the tolerance window: a journaled
                # 0.89 buckets to 44 while a broker echoing 0.890001 buckets
                # to 45, so a real filled order resolved as not_at_broker and
                # halted the next cycle over nothing. See BrokerOrder.matches.
                #
                # Consume the match so a second unresolved entry with the same
                # (symbol, side, quantity) -- two ghost entries from one crash,
                # say -- can't also claim the same one broker order.
                match_idx = next(
                    (i for i, o in enumerate(broker_orders) if o.matches(sym, side, qty)),
                    None,
                )
                match = broker_orders.pop(match_idx) if match_idx is not None else None
                outcome = "found_at_broker" if match else "not_at_broker"
                self._journal(stage="resolved", plan_id=sub.get("plan_id"),
                              intent_key=sub.get("intent_key"), outcome=outcome,
                              order_id=match.order_id if match else None)
                self.audit.emit("recover_resolved", intent=key, outcome=outcome,
                                order_id=match.order_id if match else None,
                                state=match.state if match else None)
                rows.append({"intent": key, "symbol": sym, "side": side,
                             "outcome": outcome,
                             "order_id": match.order_id if match else "",
                             "state": match.state if match else ""})
            return pd.DataFrame(rows)

    # -- preflight --------------------------------------------------------

    def _market_open(self, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.policy.market_tz)
        if local.weekday() >= 5:
            return False, f"weekend ({local.strftime('%a %H:%M %Z')})"
        t = local.time()
        if t < self.policy.session_open or t > self.policy.session_close:
            return False, f"outside session window ({local.strftime('%H:%M %Z')})"
        # NOTE: this does not know about market holidays or half days. Wire in
        # exchange_calendars before trusting it unattended.
        return True, f"open ({local.strftime('%a %H:%M %Z')})"

    def _session_date(self, now: datetime | None = None) -> pd.Timestamp:
        """The trading-session calendar date for ``now``, in the policy's
        market timezone -- not the caller's local time or UTC.

        Needed to detect a same-session day trade correctly: a trade at
        11pm UTC and one at 2am UTC the next day can be the same New York
        session or a different one depending which side of midnight ET
        they land on, and a naive UTC-date comparison gets that wrong
        right around the boundary.
        """
        now = now or datetime.now(timezone.utc)
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(self.policy.market_tz).normalize()

    def _opened_from_flat_today(self, symbol: str, session_date: pd.Timestamp) -> bool:
        """Was ``symbol`` bought into a new (from-flat) position in *this*
        trading session?

        Scans the journal for a successfully-submitted buy of this symbol
        timestamped in the same market-tz session -- the "opened" half of a
        flat -> position -> flat round trip. Uses that round-trip
        definition rather than the broader FINRA rule (any buy-then-sell
        pair on the session, not just a return to flat), matching the
        shape of :meth:`~qbt.engine.Backtester.run`'s own
        ``opened_on``/``ledger.record()`` bookkeeping.

        Note this is deliberately the *looser* of the two readings, so it
        under-counts rather than over-counts against the real FINRA rule.
        It is also the only place a day trade is ever actually recorded:
        the backtester's matching code cannot fire on a daily panel (one
        fill per symbol per bar -- see its own comment), so live is
        strictly more likely to trip the PDT block than research is, never
        less. Erring toward under-counting here is a deliberate choice to
        keep this from flattening the book on a definition the backtester
        was never checked against; the broker's own
        ``account.day_trades_used`` is the authority if you need the exact
        regulatory count.
        """
        key = f"{symbol}:buy"
        for e in self._journal_entries():
            if e.get("intent_key") != key or e.get("stage") != "submitted":
                continue
            ts = e.get("ts")
            if not ts:
                continue
            try:
                entry_ts = pd.Timestamp(ts)
            except (ValueError, TypeError):
                continue
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            if entry_ts.tz_convert(self.policy.market_tz).normalize() == session_date:
                return True
        return False

    def _refit_turnover(self, plan: LivePlan, account: BrokerAccount) -> LivePlan:
        """Re-scale the plan against *this* account read, right before
        preflight sees it -- not the equity LiveSignalRunner used to build
        it minutes or seconds ago.

        LiveSignalRunner.plan() already scales an over-cap plan down to
        exactly the turnover cap using the equity it had at planning time.
        "Exactly" is the problem: ordinary price movement between planning
        and this read -- confirmed live (2026-08), a $0.12 move on a
        ~$1,000 account was enough -- recomputes gross/account.equity a
        hair above the cap and used to hard-abort here, in
        preflight()'s own turnover check, even though the plan was
        correctly sized against reality as of a few seconds earlier. Same
        closed-form scale as LiveSignalRunner, just re-applied against
        fresher equity, so a plan that already fit only needs a
        vanishingly small correction, not a fresh rejection.
        """
        if account.equity <= 0 or not plan.intents:
            return plan
        gross = sum(abs(i.notional) for i in plan.intents)
        turnover = gross / account.equity
        if turnover <= self.policy.max_plan_turnover + 1e-9:
            return plan
        flat = account.positions.empty or bool(
            (account.positions.abs() < 1e-9).all()
        )
        if self.policy.allow_full_turnover_from_flat and flat:
            return plan
        scale = self.policy.max_plan_turnover / turnover
        scaled = [
            replace(i, shares=i.shares * scale, notional=i.notional * scale)
            for i in plan.intents
        ]
        scaled = [i for i in scaled if abs(i.notional) > 1e-9]
        return replace(
            plan,
            intents=scaled,
            warnings=[
                *plan.warnings,
                f"turnover {turnover:.1%} exceeds cap "
                f"{self.policy.max_plan_turnover:.0%} against the account's "
                f"current equity -- every order rescaled to {scale:.0%} of "
                "its already-planned size to fit (re-fit at submission "
                "time, not the equity the plan was originally sized against)",
            ],
        )

    def preflight(
        self, plan: LivePlan, account: BrokerAccount, now: datetime | None = None
    ) -> tuple[bool, list[str]]:
        """Whole-plan checks. Any failure aborts everything, not one order."""
        notes: list[str] = []
        fatal: list[str] = []

        if os.path.exists(self.policy.kill_switch_path):
            fatal.append(f"kill switch present at {self.policy.kill_switch_path}")

        if self.policy.require_market_open:
            ok, why = self._market_open(now)
            notes.append(f"session: {why}")
            if not ok:
                fatal.append(f"market closed: {why}")

        # Compare like with like on the naive calendar date. plan.asof is a
        # panel date (tz-naive today), but tz_localize("UTC") raises on an
        # already-aware timestamp rather than converting -- the same
        # naive/aware seam that broke DayTradeLedger, so it's closed the
        # same way rather than left as a latent TypeError.
        ref = pd.Timestamp(now or datetime.now(timezone.utc))
        stale = (as_session_date(ref) - as_session_date(plan.asof)).days
        if stale > self.policy.max_data_staleness_days:
            fatal.append(f"plan data {stale}d stale "
                         f"(limit {self.policy.max_data_staleness_days}d)")

        if account.equity > 0 and plan.equity > 0:
            drift = abs(account.equity - plan.equity) / account.equity
            notes.append(f"equity drift {drift:.2%}")
            if drift > self.policy.max_equity_drift:
                fatal.append(
                    f"equity drift {drift:.2%} exceeds "
                    f"{self.policy.max_equity_drift:.0%}: plan {plan.equity:,.0f} "
                    f"vs broker {account.equity:,.0f}. Recompute, do not send.")

        gross = sum(abs(i.notional) for i in plan.intents)
        notes.append(f"plan notional {gross:,.0f}")
        if gross > self.policy.max_plan_notional:
            fatal.append(f"plan notional {gross:,.0f} exceeds "
                         f"{self.policy.max_plan_notional:,.0f}")
        if account.equity > 0:
            turnover = gross / account.equity
            notes.append(f"turnover {turnover:.1%}")
            # Same tolerance as LiveSignalRunner.plan()'s turnover check --
            # a plan at exactly the cap must clear this, not get hard-
            # aborted because gross/account.equity landed a floating-point
            # hair above it. Genuinely matters here specifically: this
            # check runs against a freshly-read account.equity that can
            # differ in its last few bits from what LiveSignalRunner used
            # to size the plan, even when nothing meaningful drifted.
            if turnover > self.policy.max_plan_turnover + 1e-9:
                flat = account.positions.empty or bool(
                    (account.positions.abs() < 1e-9).all()
                )
                if self.policy.allow_full_turnover_from_flat and flat:
                    notes.append(
                        f"turnover {turnover:.1%} exceeds "
                        f"{self.policy.max_plan_turnover:.0%} but the account "
                        "holds no positions -- the first buildout from cash is "
                        "exempt (allow_full_turnover_from_flat)"
                    )
                else:
                    fatal.append(f"turnover {turnover:.1%} exceeds "
                                 f"{self.policy.max_plan_turnover:.0%}")

        if len(plan.intents) > self.policy.max_orders_per_cycle:
            fatal.append(f"{len(plan.intents)} orders exceeds "
                         f"{self.policy.max_orders_per_cycle}")

        if not account.is_agentic:
            fatal.append("account is not the agentic account; refusing to trade")

        heavy = plan.target_weights[
            plan.target_weights.abs() > self.policy.max_position_weight]
        if not heavy.empty:
            fatal.append(f"target weight exceeds "
                         f"{self.policy.max_position_weight:.0%}: "
                         f"{dict(heavy.round(3))}")

        for w in plan.warnings:
            notes.append(f"plan warning: {w}")

        return (not fatal), (fatal + notes)

    # -- execution --------------------------------------------------------

    def execute(
        self,
        plan: LivePlan,
        strategy_name: str = "",
        dry_run: bool | None = None,
        now: datetime | None = None,
    ) -> ExecutionReport:
        dry = self.policy.dry_run if dry_run is None else dry_run
        pid = self.plan_id(plan, strategy_name)
        report = ExecutionReport(plan_id=pid, dry_run=dry)
        session_date = self._session_date(now)

        self.audit.emit("plan_received", plan_id=pid, asof=str(plan.asof),
                        n_intents=len(plan.intents), equity=round(plan.equity, 2),
                        turnover=round(plan.turnover, 4), dry_run=dry,
                        strategy=strategy_name)

        if not plan.intents:
            report.aborted_reason = "plan contains no intents"
            self.audit.emit("plan_empty", plan_id=pid)
            return report

        account = self.broker.get_account()
        self.audit.emit("account_read", plan_id=pid, account=account.account_id,
                        equity=round(account.equity, 2),
                        cash=round(account.cash, 2),
                        n_positions=int(len(account.positions)),
                        is_agentic=account.is_agentic)

        # Re-fit against this fresh equity read before preflight ever sees
        # the plan -- see _refit_turnover's docstring. A no-op (returns the
        # same object) whenever the plan already fits, which is the normal
        # case; identity comparison is how we know whether it did anything.
        refit_plan = self._refit_turnover(plan, account)
        if refit_plan is not plan:
            self.audit.emit(
                "turnover_refit", plan_id=pid,
                n_intents_before=len(plan.intents),
                n_intents_after=len(refit_plan.intents),
                account_equity=round(account.equity, 2),
            )
            plan = refit_plan

        ok, notes = self.preflight(plan, account, now=now)
        report.preflight_notes = notes
        if not ok:
            report.aborted_reason = notes[0]
            self.audit.emit("preflight_failed", plan_id=pid, reason=notes[0],
                            all_notes="; ".join(notes))
            return report
        self.audit.emit("preflight_passed", plan_id=pid, notes="; ".join(notes))

        # The read-then-write dedup sequence below (read what's already
        # submitted, decide what's left, submit it) has to be one atomic
        # section w.r.t. any other OrderManager -- another thread, or another
        # process from an accidentally double-scheduled run -- racing the
        # same journal_path, or both can read the same "nothing submitted
        # yet" state and both place the same order.
        with self._dedup_lock():
            plan_entries = [
                e for e in self._journal_entries() if e.get("plan_id") == pid
            ]
            # Everything already submitted for this plan, so a re-run is a no-op.
            done = {
                e.get("intent_key") for e in plan_entries
                if e.get("stage") in ("submitted", "rejected", "resolved")
            }
            # An intent left dangling at "submitting" with no terminal stage
            # means the *previous* attempt's outcome is unknown -- the
            # broker call may have gone through, may not have (this is
            # exactly the case the except-block below produces). Silently
            # resubmitting here is how you get a real double-fill; recover()
            # has to check the broker directly and close the loop (into
            # "resolved") before this intent is safe to touch again.
            pending_recovery = {
                e.get("intent_key") for e in plan_entries
                if e.get("stage") == "submitting"
            } - done

            allow = {s.upper() for s in self.policy.symbol_allowlist}
            self._journal(stage="planned", plan_id=pid, n_intents=len(plan.intents))

            for intent in plan.intents:
                # Deliberately excludes quantity: within one plan there should be
                # at most one order per symbol per side, and keying on a float
                # would let a 1% equity drift defeat deduplication.
                key = f"{intent.symbol}:{intent.side}"

                if key in done:
                    report.skipped.append((intent, "already submitted for this plan"))
                    self.audit.emit("intent_deduplicated", plan_id=pid, intent=key)
                    continue
                if key in pending_recovery:
                    report.skipped.append(
                        (intent, "unresolved outcome from a prior attempt -- "
                                 "run recover() first")
                    )
                    self.audit.emit("intent_blocked", plan_id=pid, intent=key,
                                    reason="pending_recovery")
                    continue
                if allow and intent.symbol.upper() not in allow:
                    report.skipped.append((intent, "not on allowlist"))
                    self.audit.emit("intent_blocked", plan_id=pid, intent=key,
                                    reason="allowlist")
                    continue
                if abs(intent.notional) > self.policy.max_order_notional:
                    report.skipped.append((intent, "exceeds per-order notional cap"))
                    self.audit.emit("intent_blocked", plan_id=pid, intent=key,
                                    reason="order_notional",
                                    notional=round(abs(intent.notional), 2))
                    continue

                if self.policy.require_review:
                    review = self.broker.review_order(
                        intent.symbol, intent.side, abs(intent.shares))
                    warns = review.get("warnings") or []
                    # review.get("ok") is authoritative on its own -- both
                    # broker adapters define ok = not warnings today, but an
                    # explicit ok=False shouldn't be overridable by
                    # allow_review_warnings, which is only meant to tolerate
                    # non-fatal warnings.
                    not_ok = review.get("ok", True) is False
                    self.audit.emit("order_reviewed", plan_id=pid, intent=key,
                                    ok=review.get("ok"),
                                    est_price=review.get("estimated_price"),
                                    warnings="; ".join(map(str, warns)))
                    if not_ok:
                        reason = ("; ".join(map(str, warns))
                                  or "broker marked the order not ok")
                        report.skipped.append((intent, f"review: {reason}"))
                        continue
                    if warns and not self.policy.allow_review_warnings:
                        report.skipped.append((intent, f"review: {'; '.join(map(str, warns))}"))
                        continue

                if dry:
                    report.skipped.append((intent, "dry run"))
                    self.audit.emit("order_not_sent_dry_run", plan_id=pid, intent=key,
                                    symbol=intent.symbol, side=intent.side,
                                    quantity=round(abs(intent.shares), 6),
                                    notional=round(abs(intent.notional), 2))
                    continue

                # Write-ahead: the intent to submit is durable before the call.
                self._journal(stage="submitting", plan_id=pid, intent_key=key,
                              symbol=intent.symbol, side=intent.side,
                              quantity=abs(intent.shares))
                self.audit.emit("order_submitting", plan_id=pid, intent=key,
                                symbol=intent.symbol, side=intent.side,
                                quantity=round(abs(intent.shares), 6))
                def _finalize(order):
                    stage = "rejected" if order.state == "rejected" else "submitted"
                    self._journal(stage=stage, plan_id=pid, intent_key=key,
                                  order_id=order.order_id, state=order.state)
                    self.audit.emit("order_" + stage, plan_id=pid, intent=key,
                                    order_id=order.order_id, state=order.state,
                                    filled=round(order.filled_quantity, 6),
                                    avg_price=order.average_price,
                                    reason=order.reject_reason)
                    if stage == "rejected":
                        report.rejected.append(order)
                    else:
                        report.submitted.append(order)
                    # PDT accounting: a sell that closes a position to
                    # flat, where that same position was opened (bought
                    # from flat) earlier in this same session, is a day
                    # trade. Same round-trip shape Backtester.run() tracks
                    # via opened_on/ledger.record() (qbt/engine.py), but
                    # this is the only place it can actually fire -- a
                    # daily backtest gets one fill per symbol per bar and
                    # so never round-trips inside a session. The ledger is
                    # persisted across cycles by run_cycle.py; without that
                    # persistence a recorded day trade dies with the
                    # process and the PDT limit can never bind.
                    if (stage == "submitted" and intent.side == "sell"
                            and abs(intent.current_weight) > 1e-9
                            and abs(intent.target_weight) < 1e-9
                            and self._opened_from_flat_today(intent.symbol, session_date)):
                        self.day_trade_ledger.record(session_date)
                        self.audit.emit("day_trade_recorded", plan_id=pid, intent=key,
                                        symbol=intent.symbol,
                                        session_date=str(session_date.date()))

                try:
                    order = self.broker.place_order(
                        intent.symbol, intent.side, abs(intent.shares))
                except Exception as exc:
                    # repr(), not str(): the real broker raises through
                    # anyio task groups, so exc is often a nested
                    # ExceptionGroup whose str() collapses to "unhandled
                    # errors in a TaskGroup (1 sub-exception)" -- the
                    # actual message only survives in the recursive
                    # repr() of the whole tree. Confirmed live (2026-08):
                    # str()-only matching silently missed a rejection on a
                    # second symbol with the identical underlying error.
                    text = repr(exc)
                    # Confirmed live (2026-08), across three distinct
                    # rejection reasons now ("no more than 8 decimal
                    # places", "cannot include fractional shares", "not
                    # enough buying power"): every synchronous "API error
                    # 4xx" the broker has ever returned meant the order was
                    # definitely never created -- not the ambiguous case
                    # recover() exists for. Matching on this general
                    # pattern, not just the "fractional shares" text,
                    # matters because the *narrower* match used to
                    # misclassify a genuinely-known rejection (buying
                    # power) as an unknown outcome: it left a dangling
                    # "submitting" journal entry for the *next* cycle's
                    # recover() to wrongly HALT on, and `break` abandoned
                    # every remaining intent in the plan even though they
                    # don't depend on this one's cash shortfall.
                    definitive_rejection = re.search(r"API error 4\d\d", text) is not None
                    if not definitive_rejection:
                        # Genuinely unknown outcome (a timeout, 5xx, or
                        # anything else that doesn't match a known
                        # synchronous rejection). Do not retry; leave it
                        # for recover().
                        self.audit.emit("order_unknown_outcome", plan_id=pid,
                                        intent=key, error=repr(exc))
                        report.skipped.append((intent, f"unknown outcome: {exc!r}"))
                        break

                    if "fractional shares" not in text.lower():
                        # A confirmed rejection with no mechanical
                        # correction to apply (e.g. insufficient buying
                        # power) -- unlike the fractional-shares case
                        # below, there's no single well-defined retry
                        # that doesn't second-guess the strategy's own
                        # sizing. Close the journal entry now that the
                        # outcome is fully known, and keep going: the rest
                        # of the plan's intents are independent of this
                        # one's rejection.
                        self._journal(stage="rejected", plan_id=pid,
                                      intent_key=key, order_id=None,
                                      state="rejected")
                        clean_reason = _clean_broker_rejection(text)
                        self.audit.emit("order_rejected", plan_id=pid, intent=key,
                                        order_id=None, state="rejected",
                                        filled=0.0, avg_price=None,
                                        reason=clean_reason, raw=text)
                        report.skipped.append(
                            (intent, f"rejected: {clean_reason}"))
                        continue

                    # Confirmed live (2026-08): some instruments reject a
                    # fractional order size with a synchronous 400 ("Order
                    # quantity cannot include fractional shares"). A single
                    # corrected retry at a whole-share size is a
                    # deliberate, bounded correction (not a blind retry of
                    # the identical request).
                    if intent.side == "buy" and abs(intent.current_weight) < 1e-6:
                        # Establishing a brand-new position: "buy nothing"
                        # isn't a substitute for "buy a little," it's just
                        # never holding this name, ever, on any account
                        # small enough that a single share already costs
                        # more than the target dollar allocation --
                        # confirmed live (2026-08): a $1,000 account
                        # targeting 5 equal-weight ETF sleeves at ~$160 each
                        # hit this on every sleeve whose share price exceeds
                        # that. One whole share is the smallest unit that
                        # gets any exposure at all; downstream limits
                        # (max_order_notional, max_position_weight) still
                        # catch it if that one share is unreasonably large
                        # for the account. A magnitude already >= 1 floors
                        # normally -- this exception only kicks in when
                        # flooring would otherwise leave the position at
                        # zero forever.
                        whole_qty = math.floor(abs(intent.shares))
                        if whole_qty < 1:
                            whole_qty = 1
                    else:
                        # A top-up or trim on a position already held (or
                        # any sell): round the fractional delta to the
                        # *nearest* whole share instead of always flooring
                        # to zero. Flooring unconditionally used to mean any
                        # account whose per-share price makes a full share a
                        # large move relative to equity -- confirmed live
                        # (2026-08), a $1,000 account holding one ~$300 ETF
                        # share against a 16% target weight -- would skip
                        # that trim every single cycle, forever, with no
                        # whole-share holding between 0 and 1 ever getting
                        # closer to target. Rounding to nearest picks
                        # whichever whole-share state (unchanged vs. one
                        # share closer to target) is actually closer: a
                        # small nudge (< 0.5 share) still resolves to "do
                        # nothing," but a delta already most of the way to a
                        # full share now executes instead of stalling
                        # indefinitely. This can never oversell past what's
                        # held -- target weights are never negative in this
                        # codebase, so a sell's delta magnitude is always
                        # <= current holdings.
                        whole_qty = math.floor(abs(intent.shares) + 0.5)
                        if whole_qty < 1:
                            self._journal(stage="rejected", plan_id=pid,
                                          intent_key=key, order_id=None,
                                          state="rejected")
                            self.audit.emit("order_rejected", plan_id=pid, intent=key,
                                            order_id=None, state="rejected",
                                            filled=0.0, avg_price=None, reason=text)
                            report.skipped.append(
                                (intent, f"rejected: {intent.symbol} requires whole "
                                         f"shares and the target size "
                                         f"({abs(intent.shares):.4f}) rounds down "
                                         "to 0"))
                            continue

                    # A second write-ahead entry, at the corrected quantity
                    # -- recover() reads the *last* "submitting" entry per
                    # intent for fingerprint matching precisely so this
                    # retried size (not the original fractional one) is
                    # what gets searched for if this attempt itself throws
                    # below.
                    self._journal(stage="submitting", plan_id=pid, intent_key=key,
                                  symbol=intent.symbol, side=intent.side,
                                  quantity=float(whole_qty))
                    try:
                        order = self.broker.place_order(
                            intent.symbol, intent.side, float(whole_qty))
                    except Exception as exc2:
                        # Confirmed live (2026-08): rounding up to a whole
                        # share can itself push the cost past available
                        # buying power, so the retry needs the same
                        # definitive-rejection check the original attempt
                        # gets -- an unconditional "any exception here is
                        # unknown" used to misclassify this exact case.
                        text2 = repr(exc2)
                        if re.search(r"API error 4\d\d", text2) is not None:
                            self._journal(stage="rejected", plan_id=pid,
                                          intent_key=key, order_id=None,
                                          state="rejected")
                            clean_reason2 = _clean_broker_rejection(text2)
                            self.audit.emit("order_rejected", plan_id=pid, intent=key,
                                            order_id=None, state="rejected",
                                            filled=0.0, avg_price=None,
                                            reason=clean_reason2, raw=text2)
                            report.skipped.append((intent, f"rejected: {clean_reason2}"))
                            continue
                        self.audit.emit("order_unknown_outcome", plan_id=pid,
                                        intent=key, error=repr(exc2))
                        report.skipped.append((intent, f"unknown outcome: {exc2!r}"))
                        break

                    self.audit.emit("order_fractional_rejected_retried_whole",
                                    plan_id=pid, intent=key,
                                    original_quantity=round(abs(intent.shares), 6),
                                    retried_quantity=whole_qty)
                    _finalize(order)
                    continue

                _finalize(order)

        report.reconciliation = self.reconcile(plan, pid)
        return report

    # -- reconciliation ---------------------------------------------------

    def reconcile(self, plan: LivePlan, plan_id: str = "",
                  tolerance: float = 0.02) -> pd.DataFrame:
        """Compare realised weights against the plan's targets.

        Run this after every cycle and alert on it. Drift is how you learn that
        an order was rejected, partially filled, or silently never sent -- none
        of which announce themselves.
        """
        account = self.broker.get_account()
        symbols = sorted(set(plan.target_weights.index)
                         | set(account.positions.index))
        prices = self.broker.get_quotes(symbols)
        actual = account.weights(prices).reindex(symbols).fillna(0.0)
        target = plan.target_weights.reindex(symbols).fillna(0.0)
        drift = (actual - target).abs()

        # A held symbol the broker returned no quote for cannot be weighed,
        # and `actual` above reads it as 0.0 -- indistinguishable from "we
        # hold none of it". That turns a pricing gap into a full-size drift
        # breach, and `detection_rules()` rates any breach "high", so a
        # missing quote pages someone about a position that may be exactly
        # where it should be. Report it as its own state instead of
        # asserting a weight we never actually computed.
        held = account.positions.reindex(symbols).fillna(0.0)
        unpriced = (held.abs() > 1e-9) & ~pd.Series(
            [s in prices.index and pd.notna(prices.get(s)) for s in symbols],
            index=symbols,
        )

        out = pd.DataFrame({
            "target_weight": target.round(4),
            "actual_weight": actual.round(4),
            "drift": drift.round(4),
            "breach": drift > tolerance,
            "priced": ~unpriced,
        }).sort_values("drift", ascending=False)
        out = out[(out.target_weight.abs() > 1e-9)
                  | (out.actual_weight.abs() > 1e-9)
                  | out.index.isin(unpriced.index[unpriced])]
        # Unweighable rows are surfaced, never counted as drift -- their
        # actual_weight is unknown, not zero.
        out.loc[out.index.isin(unpriced.index[unpriced]),
                ["actual_weight", "drift", "breach"]] = [float("nan"),
                                                         float("nan"), False]

        n_breach = int(out["breach"].sum())
        missing = sorted(unpriced.index[unpriced])
        self.audit.emit("reconciled", plan_id=plan_id,
                        max_drift=float(out["drift"].max())
                        if len(out) and out["drift"].notna().any() else 0.0,
                        n_breaches=n_breach,
                        breached=", ".join(out.index[out["breach"]]),
                        unpriced=", ".join(missing))
        if missing:
            self.audit.emit("reconcile_unpriced_positions", plan_id=plan_id,
                            symbols=", ".join(missing), n=len(missing))
        return out

    # -- SIEM -------------------------------------------------------------

    @staticmethod
    def detection_rules() -> pd.DataFrame:
        """Alerts worth wiring into a SIEM against the JSONL audit stream."""
        rules = [
            ("order_submitted outside 09:35-15:50 America/New_York",
             "critical", "clock or scheduler failure, or a compromised process"),
            ("order_unknown_outcome present without a later recover_resolved",
             "critical", "position of record is unknown; halt before next cycle"),
            ("recover_resolved with outcome=not_at_broker",
             "high", "submission was lost in flight; verify no partial state"),
            ("reconciled where n_breaches > 0",
             "high", "realised book diverged from intent"),
            ("preflight_failed with reason containing 'equity drift'",
             "high", "broker equity moved from plan assumption; stale or wrong account"),
            ("count(order_submitting) > count(order_submitted|order_rejected)",
             "critical", "in-flight orders never closed out"),
            ("plan_received where dry_run=false and no preflight_passed within 5s",
             "critical", "preflight bypassed"),
            ("account_read where is_agentic=false",
             "critical", "wrong account resolved; blast radius is the whole portfolio"),
            ("order_submitted rate > 20 per hour",
             "medium", "runaway loop or duplicated scheduler"),
            ("intent_blocked reason=allowlist",
             "medium", "strategy proposed an unexpected symbol"),
        ]
        return pd.DataFrame(rules, columns=["condition", "severity", "why"])
