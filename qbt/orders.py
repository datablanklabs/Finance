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
    plan_id: str
    dry_run: bool
    submitted: list[BrokerOrder] = field(default_factory=list)
    skipped: list[tuple[OrderIntent, str]] = field(default_factory=list)
    aborted_reason: str | None = None
    preflight_notes: list[str] = field(default_factory=list)
    reconciliation: pd.DataFrame | None = None

    @property
    def aborted(self) -> bool:
        return self.aborted_reason is not None

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for o in self.submitted:
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
        return (f"ExecutionReport({mode}, {len(self.submitted)} submitted, "
                f"{filled} filled, {len(self.skipped)} skipped)")


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
    ) -> None:
        self.broker = broker
        self.policy = policy or ExecutionPolicy()
        self.audit = audit or AuditLog()
        self.journal_path = journal_path
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
        """
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
            fp = (str(latest.get("symbol", "")).upper(),
                  str(latest.get("side", "")).lower(),
                  round(float(latest.get("quantity", 0.0)) / 0.02))
            # Consume the match so a second unresolved entry with the same
            # (symbol, side, quantity-bucket) -- two ghost entries from one
            # crash, say -- can't also claim the same one broker order.
            match_idx = next(
                (i for i, o in enumerate(broker_orders) if o.fingerprint() == fp),
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
            rows.append({"intent": key, "symbol": fp[0], "side": fp[1],
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

        ref = pd.Timestamp(now or datetime.now(timezone.utc))
        if ref.tzinfo is None:
            ref = ref.tz_localize("UTC")
        stale = (ref.normalize() - pd.Timestamp(plan.asof).tz_localize("UTC")).days
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
                    report.submitted.append(order)

                try:
                    order = self.broker.place_order(
                        intent.symbol, intent.side, abs(intent.shares))
                except Exception as exc:
                    # repr(), not str(): the real broker raises through
                    # anyio task groups, so exc is often a nested
                    # ExceptionGroup whose str() collapses to "unhandled
                    # errors in a TaskGroup (1 sub-exception)" -- the actual
                    # "fractional shares" message only survives in the
                    # recursive repr() of the whole tree. Confirmed live
                    # (2026-08): str()-only matching silently missed this
                    # exact rejection on a second symbol (XLK caught it,
                    # XLV fell through to the generic unknown-outcome path
                    # right below with the identical underlying error).
                    text = repr(exc)
                    if "fractional shares" not in text.lower():
                        # Genuinely unknown outcome. Do not retry; leave it
                        # for recover().
                        self.audit.emit("order_unknown_outcome", plan_id=pid,
                                        intent=key, error=repr(exc))
                        report.skipped.append((intent, f"unknown outcome: {exc!r}"))
                        break

                    # Confirmed live (2026-08): some instruments reject a
                    # fractional order size with a synchronous 400 ("Order
                    # quantity cannot include fractional shares"). Unlike a
                    # timeout or 5xx, this means the order was definitely
                    # never created -- not the ambiguous case recover()
                    # exists for. A single corrected retry at a whole-share
                    # size is a deliberate, bounded correction (not a blind
                    # retry of the identical request). Routing this through
                    # the unknown-outcome path instead used to leave a
                    # dangling "submitting" journal entry that the *next*
                    # cycle's recover() would resolve as not_at_broker and
                    # HALT on, even though the outcome was already fully
                    # known right here.
                    whole_qty = math.floor(abs(intent.shares))
                    if whole_qty < 1:
                        # A target under one share can't be filled at all on
                        # an instrument that requires whole shares -- except
                        # when this is establishing a brand-new position
                        # (current_weight ~ 0, nothing held yet). "Buy
                        # nothing" there isn't a substitute for "buy a
                        # little," it's just never holding this name, ever,
                        # on any account small enough that a single share
                        # already costs more than the target dollar
                        # allocation -- confirmed live (2026-08): a $1,000
                        # account targeting 5 equal-weight ETF sleeves at
                        # ~$160 each hit this on every single sleeve whose
                        # share price exceeds that. One whole share is the
                        # smallest unit that gets any exposure at all;
                        # downstream limits (max_order_notional,
                        # max_position_weight) still catch it if that one
                        # share is unreasonably large for the account. A
                        # marginal top-up or trim on a position already
                        # held, or a sell, gets no such exception --
                        # rounding a small rebalance nudge up to a full
                        # share would be a much bigger trade than intended,
                        # not a substitute for a small one.
                        if intent.side == "buy" and abs(intent.current_weight) < 1e-6:
                            whole_qty = 1
                        else:
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

        out = pd.DataFrame({
            "target_weight": target.round(4),
            "actual_weight": actual.round(4),
            "drift": drift.round(4),
            "breach": drift > tolerance,
        }).sort_values("drift", ascending=False)
        out = out[(out.target_weight.abs() > 1e-9) | (out.actual_weight.abs() > 1e-9)]

        n_breach = int(out["breach"].sum())
        self.audit.emit("reconciled", plan_id=plan_id,
                        max_drift=float(out["drift"].max()) if len(out) else 0.0,
                        n_breaches=n_breach,
                        breached=", ".join(out.index[out["breach"]]))
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
