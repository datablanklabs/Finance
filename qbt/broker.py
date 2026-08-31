"""Broker adapters.

Same pattern as the data layer: one protocol, two implementations, and the
strategy/order code cannot tell which it is talking to.

* :class:`MockBroker` -- deterministic, offline, models partial fills and
  rejections. Everything in :mod:`qbt.orders` is testable against it without a
  network or an account.
* :class:`RobinhoodMCPBroker` -- the real thing, over the Robinhood Trading MCP
  at ``https://agent.robinhood.com/mcp/trading``.

**Tool names are discovered, not hard-coded.** The adapter enumerates the
server's advertised tools at connect time and binds capabilities to whatever it
finds, by matching against candidate name sets and then inspecting the input
schema. If a required capability is absent it raises immediately rather than
failing on the first order. The reason is not fussiness: the schema belongs to
the server, the product is in beta, and a name baked into your code becomes
wrong silently. As of mid-2026 the surface is reported as::

    read        get_accounts, get_portfolio, get_equity_positions,
                get_equity_quotes, get_equity_orders, search
    watchlists  get_watchlists, add_to_watchlist, update_watchlist
    trade       review_equity_order, place_equity_order, cancel_equity_order

That list comes from third-party documentation, so treat it as a hint for the
matcher and let discovery decide. ``list_capabilities()`` prints what your
server actually offers; run it first.

Note what is *not* in that surface: any documented idempotency key on
``place_equity_order``. :mod:`qbt.orders` therefore achieves idempotency with a
write-ahead journal plus read-back reconciliation instead of a key. See there.

**Crypto is opt-in and unverified.** Robinhood added crypto trading to the MCP
server after the equity surface above was confirmed against a live account
(see the README's "Confirmed against the live service"), and nothing in this
codebase has connected to a server that actually advertises crypto tools yet.
``crypto_view()`` and the ``crypto_*`` entries in ``CAPABILITY_CANDIDATES``
are a same-convention guess (``get_equity_X`` -> ``get_crypto_X``), not a
confirmed schema -- run ``debug_robinhood_crypto.py`` against a real,
crypto-enabled agentic account before trusting ``--consider-crypto --live``,
the same way ``debug_robinhood_accounts.py`` was what actually confirmed the
equity shapes below rather than the original guesses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

__all__ = [
    "BrokerAccount",
    "BrokerOrder",
    "BrokerAdapter",
    "BrokerRejection",
    "MockBroker",
    "RobinhoodMCPBroker",
    "ToolBinding",
]


class BrokerRejection(RuntimeError):
    """A synchronous, definitive order rejection the broker adapter raised
    *itself*, before the order ever reached the venue.

    Today the one case is a crypto quantity that, once snapped to its
    trading pair's increment, falls below that pair's ``min_order_size``
    (see :meth:`RobinhoodMCPBroker._crypto_order_quantity`) -- there is no
    point sending it, the venue would 400 it. :meth:`OrderManager.execute`
    treats this the same bounded way it treats a live ``API error 4xx``
    from review/place: skip this one intent, keep the rest of the plan.
    Not for ambiguous outcomes (timeouts, 5xx) -- those must stay
    unclassified so ``recover()`` checks the broker directly.
    """


class _SuppressSessionTerminationNoise(logging.Filter):
    """Confirmed live (2026-08): Robinhood's MCP server 400s on the client's
    session-termination DELETE at disconnect, every time, regardless of
    whether anything actually went wrong -- the `mcp` SDK logs it as a bare
    ``logger.warning(...)`` from :mod:`mcp.client.streamable_http` with no
    caller-facing option to disable it. It is unrelated to trading logic and
    shows up on every connect/disconnect cycle, drowning out real output.
    Filtering the one specific message, rather than raising this logger's
    level wholesale, leaves any other (currently hypothetical, but real)
    warning from that module visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Session termination failed" not in record.getMessage()


logging.getLogger("mcp.client.streamable_http").addFilter(
    _SuppressSessionTerminationNoise()
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass
class BrokerAccount:
    """Snapshot of an account as the broker reports it. The source of truth."""

    account_id: str
    cash: float
    equity: float
    buying_power: float
    positions: pd.Series          # symbol -> shares
    is_agentic: bool = False
    day_trades_used: int | None = None
    raw: dict = field(default_factory=dict)

    def weights(self, prices: pd.Series) -> pd.Series:
        if self.equity <= 0:
            return self.positions * 0.0
        held = self.positions.reindex(prices.index).fillna(0.0)
        return (held * prices).fillna(0.0) / self.equity


@dataclass
class BrokerOrder:
    """An order as the broker reports it."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    state: str                    # pending | filled | partial | cancelled | rejected
    filled_quantity: float = 0.0
    average_price: float | None = None
    created_at: datetime | None = None
    reject_reason: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.state in ("pending", "partial", "queued", "confirmed")

    @property
    def is_terminal(self) -> bool:
        return self.state in ("filled", "cancelled", "rejected", "failed")

    def fingerprint(self, qty_tolerance: float = 0.02) -> tuple:
        """Coarse bucketed identity, for grouping and display only.

        **Not for deciding whether two orders are the same** -- use
        :meth:`matches` for that. Bucketing quantity makes this cheap to
        hash but puts hard edges in the middle of the tolerance window: a
        quantity landing exactly on a half-bucket boundary flips buckets
        under a rounding smaller than the tolerance is meant to absorb.
        ``round`` is banker's rounding, so ``0.89 -> 44`` while a broker
        echoing ``0.890001`` gives ``45``, and an equality test on these
        tuples then reports a genuinely-placed order as missing.
        """
        bucket = round(self.quantity / max(qty_tolerance, 1e-9))
        return (self.symbol.upper(), self.side.lower(), bucket)

    def matches(self, symbol: str, side: str, quantity: float,
                qty_tolerance: float = 0.02) -> bool:
        """Is this the order described by ``(symbol, side, quantity)``?

        Symbol and side must match exactly; quantity only within
        ``qty_tolerance``, because a broker may round or renormalise a
        fractional share count on the way in. Comparing the distance
        directly is what :meth:`fingerprint`'s bucketing was approximating,
        without the boundary artefact: every quantity within the tolerance
        matches, and nothing outside it does, regardless of where the value
        happens to fall relative to a bucket edge.

        This is the comparison :meth:`qbt.orders.OrderManager.recover` uses.
        Getting it wrong is expensive in one specific direction -- a missed
        match resolves a real, filled order as ``not_at_broker``, which
        halts the next cycle for an order that was never actually lost.
        """
        return (
            self.symbol.upper() == str(symbol).upper()
            and self.side.lower() == str(side).lower()
            and abs(self.quantity - float(quantity)) <= max(qty_tolerance, 1e-9)
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BrokerAdapter(Protocol):
    """Minimum surface :mod:`qbt.orders` needs. Deliberately small."""

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def get_account(self) -> BrokerAccount: ...
    def get_quotes(self, symbols: Sequence[str]) -> pd.Series: ...
    def get_orders(self, since: datetime | None = None) -> list[BrokerOrder]: ...
    def review_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", **kw) -> dict: ...
    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", **kw) -> BrokerOrder: ...
    def cancel_order(self, order_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockBroker:
    """Deterministic broker for testing the order path offline.

    Models the things that actually break order managers: partial fills,
    rejections, slippage against the reference price, and orders that stay open
    across cycles. Seeded, so a failing test reproduces.
    """

    def __init__(
        self,
        prices: pd.Series,
        cash: float = 25_000.0,
        positions: pd.Series | None = None,
        seed: int = 0,
        reject_rate: float = 0.0,
        partial_rate: float = 0.0,
        slippage_bps: float = 3.0,
        fail_on_symbols: Sequence[str] = (),
    ) -> None:
        self.prices = prices.astype(float)
        self.cash = float(cash)
        self.positions = (
            positions.astype(float)
            if positions is not None
            else pd.Series(0.0, index=prices.index)
        )
        self.rng = np.random.default_rng(seed)
        self.reject_rate = reject_rate
        self.partial_rate = partial_rate
        self.slippage_bps = slippage_bps
        self.fail_on_symbols = {s.upper() for s in fail_on_symbols}
        self.orders: list[BrokerOrder] = []
        self._seq = 0
        self.connected = False
        self.call_log: list[tuple[str, dict]] = []

    # -- lifecycle --------------------------------------------------------

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def _require(self) -> None:
        if not self.connected:
            raise RuntimeError("broker not connected")

    # -- reads ------------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        self._require()
        self.call_log.append(("get_account", {}))
        held = self.positions.reindex(self.prices.index).fillna(0.0)
        equity = self.cash + float((held * self.prices).sum())
        return BrokerAccount(
            account_id="MOCK-AGENTIC-1",
            cash=self.cash,
            equity=equity,
            buying_power=self.cash,
            positions=self.positions[self.positions.abs() > 0].copy(),
            is_agentic=True,
            day_trades_used=0,
        )

    def get_quotes(self, symbols: Sequence[str]) -> pd.Series:
        self._require()
        self.call_log.append(("get_quotes", {"symbols": list(symbols)}))
        return self.prices.reindex([s for s in symbols]).dropna()

    def get_orders(self, since: datetime | None = None) -> list[BrokerOrder]:
        self._require()
        self.call_log.append(("get_orders", {"since": str(since)}))
        if since is None:
            return list(self.orders)
        return [o for o in self.orders
                if o.created_at is not None and o.created_at >= since]

    # -- writes -----------------------------------------------------------

    def review_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", **kw) -> dict:
        self._require()
        self.call_log.append(("review_order", {"symbol": symbol,
                                               "side": side.lower(),
                                               "quantity": quantity}))
        return self._review(symbol, side, quantity)

    def _review(self, symbol: str, side: str, quantity: float) -> dict:
        """The review itself, without the call-log entry.

        ``place_order`` needs this check but must not journal a
        ``review_order`` call for it: a test counting how many times the
        manager reviewed an order would otherwise see one phantom entry per
        placement and conclude the review gate ran twice.
        """
        side = side.lower()
        px = float(self.prices.get(symbol, np.nan))
        warnings = []
        if symbol.upper() in self.fail_on_symbols:
            warnings.append("symbol not tradeable")
        if not np.isfinite(px):
            warnings.append("no quote available")
        # Estimate against the same slippage-adjusted fill price place_order
        # will actually use -- checking buying power against the raw quote
        # lets an order sized to exactly self.cash pass review and then push
        # cash negative once slippage is applied on the real fill.
        sign = 1.0 if side == "buy" else -1.0
        fill_px = px * (1.0 + sign * self.slippage_bps / 1e4) if np.isfinite(px) else 0.0
        est = abs(quantity) * fill_px
        if side == "buy" and est > self.cash:
            warnings.append("insufficient buying power")
        held = float(self.positions.get(symbol, 0.0))
        if side == "sell" and quantity > held + 1e-9:
            warnings.append("sell exceeds position")
        return {
            "ok": not warnings,
            "estimated_price": fill_px,
            "estimated_notional": est,
            "warnings": warnings,
        }

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", **kw) -> BrokerOrder:
        self._require()
        side = side.lower()
        self.call_log.append(("place_order", {"symbol": symbol, "side": side,
                                              "quantity": quantity}))
        self._seq += 1
        oid = f"mock-{self._seq:05d}"
        now = datetime.now(timezone.utc)

        review = self._review(symbol, side, quantity)
        if not review["ok"] or self.rng.random() < self.reject_rate:
            order = BrokerOrder(
                order_id=oid, symbol=symbol, side=side, quantity=quantity,
                state="rejected", created_at=now,
                reject_reason="; ".join(review["warnings"]) or "simulated rejection",
            )
            self.orders.append(order)
            return order

        fill_frac = 1.0
        if self.rng.random() < self.partial_rate:
            fill_frac = float(self.rng.uniform(0.3, 0.8))

        # Reuse the exact price review_order already computed rather than
        # recomputing the same slippage formula a second time -- keeps the
        # two from ever drifting apart.
        sign = 1.0 if side == "buy" else -1.0
        fill_px = review["estimated_price"]
        filled = quantity * fill_frac

        self.cash -= sign * filled * fill_px
        self.positions[symbol] = self.positions.get(symbol, 0.0) + sign * filled

        order = BrokerOrder(
            order_id=oid, symbol=symbol, side=side, quantity=quantity,
            state="filled" if fill_frac >= 1.0 else "partial",
            filled_quantity=filled, average_price=fill_px, created_at=now,
        )
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._require()
        self.call_log.append(("cancel_order", {"order_id": order_id}))
        for o in self.orders:
            if o.order_id == order_id and o.is_open:
                o.state = "cancelled"
                return True
        return False


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _schema_types(prop: dict) -> set[str]:
    """The set of JSON types a schema property declares as acceptable.

    JSON Schema allows ``"type"`` to be either a single string
    (``"string"``) or a list of strings for a union (``["null",
    "array"]``) -- confirmed live (2026-08): Robinhood's real
    ``get_equity_quotes`` schema uses the list form for ``symbols``
    (nullable array), which a plain ``prop.get("type") == "array"``
    comparison can never match, since the value is a list, not the string
    ``"array"``. Normalising both shapes into a set here means every
    caller checks membership the same way instead of each reimplementing
    (and each potentially getting wrong) its own parsing of this field.
    """
    t = (prop or {}).get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, (list, tuple, set)):
        return {x for x in t if isinstance(x, str)}
    return set()


@dataclass
class ToolBinding:
    """A capability bound to a concrete tool the server advertises."""

    capability: str
    tool_name: str
    input_schema: dict = field(default_factory=dict)

    def properties(self) -> dict:
        return (self.input_schema or {}).get("properties", {}) or {}

    def required(self) -> list[str]:
        return list((self.input_schema or {}).get("required", []) or [])

    def resolve_arg(self, logical: str, candidates: Sequence[str]) -> str | None:
        """Find the server's parameter name for a logical argument.

        Handles the common case where a server calls it ``ticker`` and you
        assumed ``symbol``. Matching is exact-then-substring over the
        schema's declared properties. If a candidate's substring appears in
        more than one property (e.g. both ``order_quantity`` and
        ``max_quantity_per_order`` for candidate ``"quantity"``), that's
        ambiguous -- picking whichever one happened to iterate first is how
        you silently bind to the wrong field, so it's treated as no match
        for that candidate and the next candidate is tried instead.
        """
        props = {k.lower(): k for k in self.properties()}
        for c in candidates:
            if c.lower() in props:
                return props[c.lower()]
        for c in candidates:
            matches = [orig for low, orig in props.items() if c.lower() in low]
            if len(matches) == 1:
                return matches[0]
        return None

    def coerce(self, key: str, value: Any) -> Any:
        """Coerce a Python value to match this tool's declared JSON type for ``key``.

        Confirmed live (2026-08), twice, in two different shapes:
        Robinhood's order tools declare ``quantity`` as a JSON *string*,
        not a number (``... has type "number", want "string"``); and
        ``get_equity_quotes`` declares ``symbols`` as a nullable *array*
        (``"type": ["null", "array"]``) -- see :func:`_schema_types` for
        why a naive ``.get("type") == "array"`` check can't see that form.
        Rather than hardcode either shape, this reads the tool's *actual*
        declared type(s) for whichever field ``key`` resolved to and
        coerces to match, so it keeps working if a field's declared type
        differs across tools or changes later.
        """
        types = _schema_types(self.properties().get(key, {}))
        if isinstance(value, (list, tuple)):
            if "array" in types:
                return list(value)
            if "string" in types:
                return ",".join(str(v) for v in value)
            return value
        if "string" in types and not isinstance(value, str):
            return str(value)
        if "number" in types and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if "integer" in types and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value


# Candidate names per capability, most likely first. Discovery matches against
# these; if your server uses something else, add it here rather than editing
# call sites.
#
# The crypto_* tool *names* below are confirmed (2026-08-30,
# debug_robinhood_crypto.py connected to a live server advertising them);
# their response *shapes* are only partially cross-checked, so parsing is
# still somewhat a guess -- see the module docstring and the README's
# "Crypto support" list. Discovery degrades the same way it always has if a
# name is wrong: `require_crypto()` raises with the full advertised tool
# list rather than silently binding nothing, and `debug_robinhood_crypto.py`
# dumps the raw responses so the real names/shapes can replace these
# candidates the same way the equity ones were confirmed and corrected
# (nested envelopes, string-typed quantity, etc.) -- run it before trusting
# --consider-crypto --live.
CAPABILITY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "accounts": ("get_accounts", "list_accounts", "accounts"),
    "portfolio": ("get_portfolio", "portfolio", "get_portfolio_summary"),
    "positions": ("get_equity_positions", "get_positions", "positions"),
    "quotes": ("get_equity_quotes", "get_quotes", "quote", "get_quote"),
    "orders": ("get_equity_orders", "get_orders", "list_orders", "orders"),
    "review": ("review_equity_order", "review_order", "preview_order",
               "preview_equity_order"),
    "place": ("place_equity_order", "place_order", "submit_order",
              "create_equity_order"),
    "cancel": ("cancel_equity_order", "cancel_order"),
    "search": ("search", "search_instruments", "find_symbol"),
    # -- crypto (see the block comment above) ----------------------------
    # Tool *names* confirmed live (2026-08-30, debug_robinhood_crypto.py):
    # the server advertises get_crypto_positions / get_crypto_quotes /
    # get_crypto_orders / preview_crypto_order / place_crypto_order /
    # cancel_crypto_order / get_currency_pairs. Response *shapes* are still
    # only partially cross-checked -- see the module docstring.
    "crypto_positions": ("get_crypto_positions", "get_crypto_holdings",
                         "crypto_positions"),
    "crypto_quotes": ("get_crypto_quotes", "get_crypto_quote",
                      "crypto_quotes", "crypto_quote"),
    "crypto_orders": ("get_crypto_orders", "list_crypto_orders",
                      "crypto_orders"),
    "crypto_review": ("review_crypto_order", "preview_crypto_order"),
    "crypto_place": ("place_crypto_order", "create_crypto_order",
                     "submit_crypto_order"),
    "crypto_cancel": ("cancel_crypto_order",),
    # Pair catalog: min_order_size / min_order_quantity_increment /
    # min_order_price_increment / halted per pair. Not required (kept out of
    # REQUIRED_CRYPTO_CAPABILITIES); used to source CRYPTO_QUANTITY_DECIMALS.
    "crypto_pairs": ("get_currency_pairs", "get_crypto_currency_pairs",
                     "get_crypto_trading_pairs", "crypto_pairs"),
}

REQUIRED_CAPABILITIES = ("accounts", "positions", "quotes", "orders", "place")

# Checked only by require_crypto(), never by connect() -- a server with no
# crypto tools at all (an older account, or crypto genuinely unsupported on
# it) must not break plain equity trading, which is why this is a separate,
# opt-in check rather than an addition to REQUIRED_CAPABILITIES.
REQUIRED_CRYPTO_CAPABILITIES = ("crypto_positions", "crypto_quotes",
                                "crypto_orders", "crypto_place")


# Order-quantity precision. Two *different* downstream rules, both enforced
# by the API after MCP schema validation has already passed:
#
# Equities: "no more than 8 decimal places" on quantity. Computed share
#   counts (weight * equity / price) are ordinary floats at full binary
#   precision (e.g. 2.7255578430183576); sent as-is they 400 after passing
#   review. Confirmed live (2026-08); see _order_args().
# Crypto: each trading pair has its own quantity increment, coarser than
#   1e-8 and varying widely by pair (1e-8 for BTC, 0.01 for ADA, 1 for
#   SHIB), so the 8-dp equity ceiling is not enough -- a finer quantity is
#   rejected with 'API error 400 {"quantity": ["Your order quantity has too
#   much precision. Please round the quantity to an appropriate increment
#   and try placing your order again."]}'. Confirmed live (2026-08) on the
#   first real review_crypto_order call.
#
#   These increments ARE discoverable, and at run time the broker now does:
#   _load_crypto_pairs() pulls min_order_quantity_increment + min_order_size
#   per pair from get_currency_pairs (bound as "crypto_pairs"), once per
#   cycle, and _crypto_order_quantity() snaps each order to that increment
#   and raises BrokerRejection below the pair minimum.
#
#   CRYPTO_QUANTITY_DECIMALS is the *static fallback* for when that lookup
#   can't answer -- catalog tool unbound, the call failed, or the coin is
#   absent from the response. It is min_order_quantity_increment as decimal
#   places for every coin in run_cycle.py's CRYPTOS, captured from a live
#   get_currency_pairs response (2026-08-30) and cross-checked against live
#   accept/reject behaviour: ETH/BCH at 6 dp were accepted; ADA 71.108246,
#   LINK 2.100654 and AAVE 0.191821 at 6 dp were the three rejections that
#   first exposed this. Refresh from debug_robinhood_crypto.py if Robinhood
#   retunes a pair or CRYPTOS grows. A coin covered by neither the live
#   lookup nor this table falls back to _DEFAULT_CRYPTO_QUANTITY_DECIMALS,
#   and if that is still too fine it surfaces as a clean "review rejected:
#   ...too much precision" skip in OrderManager.execute(), not a crash.
_EQUITY_QUANTITY_DECIMALS = 8
_DEFAULT_CRYPTO_QUANTITY_DECIMALS = 6
# min_order_quantity_increment -> decimal places (see the block comment:
# static fallback for the live get_currency_pairs lookup). Every increment
# is an exact power of ten, so rounding to this many places lands on grid.
CRYPTO_QUANTITY_DECIMALS: dict[str, int] = {
    "BTC-USD": 8,    # increment 0.00000001
    "ETH-USD": 6,    # increment 0.000001
    "SOL-USD": 5,    # increment 0.00001
    "DOGE-USD": 2,   # increment 0.01
    "LTC-USD": 8,    # increment 0.00000001
    "BCH-USD": 8,    # increment 0.00000001
    "AVAX-USD": 4,   # increment 0.0001
    "SHIB-USD": 0,   # increment 1
    "XRP-USD": 3,    # increment 0.001
    "ADA-USD": 2,    # increment 0.01
    "LINK-USD": 4,   # increment 0.0001
    "UNI-USD": 4,    # increment 0.0001
    "AAVE-USD": 5,   # increment 0.00001
    "ETC-USD": 6,    # increment 0.000001
    "XLM-USD": 2,    # increment 0.01
}


def _quantity_decimals(capability: str, symbol: str) -> int:
    """Decimal places to round an order quantity to before it is sent.

    Crypto capabilities (``crypto_*``) round to the trading pair's own
    increment; everything else to Robinhood's equity 8-dp limit. See
    ``CRYPTO_QUANTITY_DECIMALS`` above.
    """
    if capability.startswith("crypto_"):
        return CRYPTO_QUANTITY_DECIMALS.get(
            symbol.upper(), _DEFAULT_CRYPTO_QUANTITY_DECIMALS)
    return _EQUITY_QUANTITY_DECIMALS


# ---------------------------------------------------------------------------
# Robinhood MCP
# ---------------------------------------------------------------------------


class RobinhoodMCPBroker:
    """Deterministic MCP client for the Robinhood Trading MCP.

    There is no language model anywhere in this class. MCP is usually driven by
    one, but the protocol is just JSON-RPC over HTTP -- a plain client can call
    tools directly. That matters for more than determinism: it removes the
    entire prompt-injection attack surface. A model that reads market
    commentary can be talked into a trade; a function that consumes a float
    cannot.

    Auth is OAuth against Robinhood. Pass an ``auth`` provider from the MCP SDK
    (``mcp.client.auth.OAuthClientProvider``) or a bearer ``token`` obtained by
    letting an established client complete the handshake once. **This is the
    part of this file I could not verify against the live service** -- the
    handshake specifics and token lifetime are what to confirm first.

    Every method is synchronous and wraps an async MCP call, because the order
    manager is a synchronous state machine and mixing the two invites the
    reentrancy bugs that make crash recovery unprovable.
    """

    URL = "https://agent.robinhood.com/mcp/trading"

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        auth: Any = None,
        account_id: str | None = None,
        timeout: float = 30.0,
        require_agentic: bool = True,
    ) -> None:
        self.url = url or self.URL
        self.token = token
        self.auth = auth
        self.account_id = account_id
        self.timeout = timeout
        self.require_agentic = require_agentic
        self.bindings: dict[str, ToolBinding] = {}
        self._all_tools: list[dict] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        # The persistent MCP session -- see connect()/_open_session(). None
        # until connect() succeeds; call_tool()/list_tools() go through
        # self._mcp_session, not a fresh session per call, once it's set.
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._mcp_session: Any = None
        # {SYMBOL: {"qty_increment": Decimal|None, "min_order_size":
        # Decimal|None}} from get_currency_pairs, loaded lazily once per
        # broker lifetime (== once per cycle -- run_cycle.py builds one
        # broker per cycle). None = not loaded yet; {} = loaded/attempted
        # and the static CRYPTO_QUANTITY_DECIMALS fallback is in effect.
        # See _load_crypto_pairs() / _crypto_order_quantity().
        self._crypto_pairs: dict[str, dict] | None = None

    # -- plumbing ---------------------------------------------------------

    def _run(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _open_session(self) -> None:
        """Open the HTTP transport and the MCP session once, and keep both
        open (via an ``AsyncExitStack``, tracked as ``self._exit_stack``)
        for the rest of this broker's lifetime -- ``call_tool()``/
        ``list_tools()`` reuse ``self._mcp_session`` from here on, rather
        than each opening and tearing down their own transport + session +
        handshake. A single trading cycle previously made roughly a dozen
        separate TCP+streamable-HTTP+``session.initialize()`` round trips
        (one per ``get_account``/``get_quotes``/``place_order``/... call)
        instead of reusing one open session for the cycle's lifetime --
        real, avoidable latency (and failure surface, from simply
        attempting the handshake more times) on every scheduled run.

        **Trade-off, stated plainly (this specific piece of connection
        lifecycle is NOT confirmed against the live service the way the
        rest of this file's behaviour is -- see the module docstring):**
        the old per-call-reconnect design was, as an accidental side
        effect of being wasteful, self-healing against a mid-cycle network
        blip -- a failure on one call couldn't affect the next, since the
        next call got its own fresh connection regardless. Reusing one
        session trades that away: if the persistent session breaks partway
        through a cycle, every remaining call in that cycle fails with it,
        rather than just the one call that hit the blip. This is
        deliberately NOT patched over with an automatic reconnect-and-retry
        here -- retrying a place_order/review_order/cancel_order call whose
        outcome is genuinely ambiguous (did the order go through before the
        connection died, or not?) is exactly the "blind retry after an
        unknown outcome" this package's write-ahead journal exists to rule
        out (see qbt/orders.py's module docstring) -- a broker-level
        auto-retry would submit a second order without OrderManager's
        journal ever knowing there'd been a first attempt. A cycle that
        dies partway through from a broken connection surfaces as an
        ordinary exception, exactly as it already could for any other
        reason (a bad response, a timeout, ...) -- the same
        crash-recovery-on-next-cycle path (OrderManager.recover(), see its
        own docstring) that already has to handle "the process died
        mid-cycle" for other reasons handles this one too, rather than this
        method inventing a second, riskier way to paper over it.
        """
        try:
            from mcp import ClientSession                        # noqa: PLC0415
            from mcp.client.streamable_http import (             # noqa: PLC0415
                streamablehttp_client,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the MCP SDK is required: pip install mcp"
            ) from exc

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.auth is not None:
            kwargs["auth"] = self.auth
        elif self.token:
            kwargs["headers"] = {"Authorization": f"Bearer {self.token}"}

        # AsyncExitStack, not two nested `async with` blocks: those only
        # stay entered for the duration of one `async with` statement, and
        # this needs both the transport and the session to stay open
        # across many separate self._run(...) calls afterward. The stack
        # is what makes close() able to unwind both, in the right order,
        # exception-safely, without connect() itself having to stay on the
        # call stack the whole time.
        stack = contextlib.AsyncExitStack()
        try:
            streams = await stack.enter_async_context(
                streamablehttp_client(self.url, **kwargs))
            read, write = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._mcp_session = session

    async def _call(self, tool_name: str, arguments: dict) -> Any:
        result = await self._mcp_session.call_tool(tool_name, arguments)
        return _unwrap_tool_result(result)

    async def _discover(self) -> list[dict]:
        listing = await self._mcp_session.list_tools()
        out = []
        for t in listing.tools:
            out.append({
                "name": t.name,
                "description": (t.description or "")[:400],
                "input_schema": getattr(t, "inputSchema", None) or {},
            })
        return out

    # -- lifecycle --------------------------------------------------------

    def connect(self) -> None:
        self._run(self._open_session())
        self._all_tools = self._run(self._discover())
        by_name = {t["name"].lower(): t for t in self._all_tools}

        for capability, candidates in CAPABILITY_CANDIDATES.items():
            chosen = None
            for cand in candidates:
                if cand.lower() in by_name:
                    chosen = by_name[cand.lower()]
                    break
            if chosen is None:  # fall back to fuzzy match on the tool name
                pattern = re.compile("|".join(re.escape(c) for c in candidates))
                for name, t in by_name.items():
                    if pattern.search(name):
                        chosen = t
                        break
            if chosen is not None:
                self.bindings[capability] = ToolBinding(
                    capability=capability,
                    tool_name=chosen["name"],
                    input_schema=chosen["input_schema"],
                )

        missing = [c for c in REQUIRED_CAPABILITIES if c not in self.bindings]
        if missing:
            raise RuntimeError(
                "the MCP server does not advertise required capabilities: "
                f"{missing}. Advertised tools: "
                f"{[t['name'] for t in self._all_tools]}. "
                "Add the server's names to CAPABILITY_CANDIDATES."
            )
        if "review" not in self.bindings:
            # Not fatal, but review is the cheap preflight that catches
            # untradeable symbols and buying-power problems before you commit.
            print("WARNING: no review/preview tool found; preflight is degraded")

    def close(self) -> None:
        if self._exit_stack is not None:
            try:
                # Best-effort: confirmed live (2026-08, see
                # _SuppressSessionTerminationNoise above) that the server
                # 400s on the client's own session-termination DELETE
                # every time regardless of whether anything actually went
                # wrong, and that's already filtered at the logger level.
                # A close() that raised here would turn "shut down a
                # connection we're done with" into a reason a cycle's own
                # cleanup could fail, which is the wrong thing for a
                # teardown step to do.
                self._run(self._exit_stack.aclose())
            except Exception:
                pass
            self._exit_stack = None
            self._mcp_session = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def list_capabilities(self) -> pd.DataFrame:
        """What the server actually offers. Run this before anything else."""
        rows = []
        bound = {b.tool_name for b in self.bindings.values()}
        cap_by_tool = {b.tool_name: b.capability for b in self.bindings.values()}
        for t in self._all_tools:
            rows.append({
                "tool": t["name"],
                "bound_to": cap_by_tool.get(t["name"], ""),
                "used": t["name"] in bound,
                "params": ", ".join(list(
                    (t["input_schema"] or {}).get("properties", {}) or {})[:6]),
                "description": t["description"][:80],
            })
        return pd.DataFrame(rows).sort_values(
            ["used", "tool"], ascending=[False, True]).reset_index(drop=True)

    def _binding(self, capability: str) -> ToolBinding:
        b = self.bindings.get(capability)
        if b is None:
            raise RuntimeError(f"capability {capability!r} not bound; call connect()")
        return b

    def _account_arg(self, b: ToolBinding, account_id: str | None) -> dict:
        """Resolve the schema's account-number-equivalent field and fill it
        from ``account_id``, raising clearly if the schema requires it but
        no id is available.

        The one piece of request-building logic every account-scoped call
        needs -- get_orders(), _order_args() (review/place/cancel's shared
        arg builder), cancel_order(), _positions(), and
        _portfolio_figures() each used to carry their own copy of this.
        That drifted: cancel_order()'s copy silently omitted the required-
        field check the other four had, so a cancel issued before
        get_account() had ever resolved an account hit an opaque MCP
        schema-validation error instead of the same clear RuntimeError
        every other account-scoped call already gave for the identical
        situation. One copy can't drift from itself.
        """
        key = b.resolve_arg("account", ("account_number", "account_id", "account"))
        if key is None:
            return {}
        if account_id:
            return {key: b.coerce(key, account_id)}
        if key in b.required():
            raise RuntimeError(
                f"{b.tool_name} requires {key!r} but no account_id is "
                "available. Call get_account() at least once first -- it's "
                "what resolves the agentic account."
            )
        return {}

    def require_crypto(self) -> None:
        """Raise unless the server advertises the crypto trading tools.

        Not part of ``connect()`` -- see ``REQUIRED_CAPABILITIES`` vs.
        ``REQUIRED_CRYPTO_CAPABILITIES``'s own comment for why. Call this
        explicitly (``run_cycle.py`` does, only when ``--consider-crypto``
        is passed) before touching ``crypto_view()`` or any ``asset_class=
        "crypto"`` argument below, so a server or account without crypto
        support fails with a clear message naming exactly what's missing
        rather than a confusing downstream KeyError from an unbound
        capability.
        """
        missing = [c for c in REQUIRED_CRYPTO_CAPABILITIES if c not in self.bindings]
        if missing:
            raise RuntimeError(
                "the MCP server does not advertise required crypto "
                f"capabilities: {missing}. Advertised tools: "
                f"{[t['name'] for t in self._all_tools]}. Run "
                "debug_robinhood_crypto.py to inspect what's actually there, "
                "and add the server's real names to CAPABILITY_CANDIDATES if "
                "they differ from the get_crypto_X / X_crypto_order guess."
            )
        if "crypto_review" not in self.bindings:
            print("WARNING: no crypto review/preview tool found; "
                  "crypto preflight is degraded")

    def crypto_view(self) -> "_AssetClassView":
        """A ``BrokerAdapter``-shaped view of this same connection, pinned to
        crypto capabilities.

        Same account, same MCP session, same tool-discovery result --
        ``connect()`` and ``get_account()`` (equity side) are not called
        again. This exists so :class:`~qbt.orders.OrderManager` and
        :class:`~qbt.live.LiveSignalRunner`, which both call the plain
        ``BrokerAdapter`` methods with no ``asset_class`` argument, can run a
        second, independent pass over the crypto sleeve without knowing
        anything asset-class-specific happened -- same reason
        :class:`MockBroker` and this class already share one interface.
        """
        return _AssetClassView(self, "crypto")

    # -- reads ------------------------------------------------------------

    def get_account(self, asset_class: str = "equity") -> BrokerAccount:
        accounts = _as_records(self._call_sync("accounts", {}))
        chosen = None
        # Set once self.account_id is pinned (by the caller at construction,
        # or by a prior get_account() call caching its own resolved account
        # -- see the assignment below) *and* actually matched against this
        # response, so a second call against the same account (e.g. the
        # crypto sleeve's asset_class="crypto" pass, reusing the id the
        # equity pass already resolved) doesn't get flagged as a mismatch.
        requested_account_matched = False
        for a in accounts:
            aid = str(_pick(a, "account_number", "account_id", "id", default=""))
            # Confirmed against a live response (2026-08): Robinhood's real
            # field is "agentic_allowed", a plain bool, and it's
            # caller-relative -- true means *this* agent can act on the
            # account, not that the account is agentic in general. "type"
            # here is the trading type (margin/cash), not agentic-ness --
            # this code used to guess "type" as a fallback, which matched
            # nothing on either account and is exactly why this raised for
            # every account. Do not add "nickname" as a candidate either:
            # Robinhood's own API guidance is explicit that nickname must
            # not be used to determine agentic-account eligibility.
            agentic = _truthy(_pick(a, "agentic_allowed", "is_agentic", "agentic",
                                    default=False))
            if self.account_id and aid == self.account_id:
                chosen = a
                requested_account_matched = True
                break
            if self.require_agentic and agentic and chosen is None:
                chosen = a
        if self.account_id and not requested_account_matched:
            # A caller who pinned account_id explicitly asked for *that*
            # account, not "whichever agentic one happens to come first" --
            # the loop above would otherwise silently fall through to the
            # latter, and every later order call goes to an account nobody
            # chose. Fail loudly instead; the alternative is real money
            # moving on the wrong account with no error anywhere.
            found = [str(_pick(a, "account_number", "account_id", "id", default=""))
                    for a in accounts]
            raise RuntimeError(
                f"account_id={self.account_id!r} was explicitly set but does "
                f"not match any account returned by 'accounts' ({found}). "
                "Refusing to silently substitute a different account."
            )
        if chosen is None:
            if self.require_agentic:
                raise RuntimeError(
                    "no agentic account found. Trade placement is confined to "
                    "the Agentic account; refusing to proceed against another."
                )
            chosen = accounts[0] if accounts else {}

        aid = str(_pick(chosen, "account_number", "account_id", "id", default=""))
        # Cache the resolved account for order-placement calls later in this
        # broker's lifetime. Confirmed live (2026-08): get_account() resolves
        # the right account for reads (positions, portfolio) via this local
        # `aid`, but review_order/place_order/cancel_order build their
        # request args from self.account_id -- which nothing populated
        # before this line existed, so every order call failed with
        # "still requires ['account_number']" even though the right account
        # had already been found. Means get_account() must be called at
        # least once before the first order call, which is already the
        # natural order everywhere in this codebase.
        self.account_id = aid
        positions = self._positions(aid, asset_class)
        cash, equity, buying_power = self._portfolio_figures(aid, chosen, asset_class)
        return BrokerAccount(
            account_id=aid,
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            positions=positions,
            is_agentic=True,
            # PDT accounting describes the equity/margin side of the
            # account; crypto is not subject to FINRA's Pattern Day Trader
            # rule at all (it trades through Robinhood Crypto, not the
            # broker-dealer the rule applies to), so reusing the equity
            # figure here would misleadingly suggest a crypto-specific PDT
            # count exists. run_cycle.py's crypto pass doesn't consult this
            # field either way -- see its DayTradeLedger(equity_threshold=0.0).
            day_trades_used=(
                None if asset_class == "crypto" else
                _maybe_int(_pick(chosen, "day_trades_used", "day_trade_count",
                                 default=None))
            ),
            raw=chosen,
        )

    def _portfolio_figures(
        self, account_id: str, account_rec: dict, asset_class: str = "equity"
    ) -> tuple[float, float, float]:
        """Cash, equity, buying power -- from the 'portfolio' tool, not 'accounts'.

        Confirmed against a live response (2026-08): the 'accounts' tool
        (``account_rec`` here) carries no cash/equity/buying_power field at
        all. 'portfolio' is a separately bound capability whose own
        description says "market value breakdown by asset type and buying
        power", so this calls it -- but its exact response field names are
        still an unverified guess (see the class docstring). If none of the
        guesses match, this reports 0.0 and warns loudly rather than
        silently returning a wrong number to something that sizes real
        trades and feeds the drawdown breaker.

        ``asset_class="crypto"`` reads the same confirmed response's
        ``crypto_value`` field for "equity" instead of ``equity_value``/
        ``total_value`` -- confirmed live (2026-08) that the field exists
        (see ``REAL_PORTFOLIO_PAYLOAD`` in test_robinhood_broker.py) but
        *not* confirmed that it's actually what should drive crypto sizing
        (vs. e.g. a `get_crypto_positions`-derived market value) -- cross-
        check the two once crypto_positions is confirmed live. cash/
        buying_power are the same account-wide figures either way: nothing
        here has confirmed whether Robinhood tracks crypto buying power
        separately from cash, so this assumes not (crypto purchases draw
        from the same settled cash) rather than guessing at a second,
        unconfirmed field name.
        """
        if "portfolio" not in self.bindings:
            cash = _to_float(_pick(account_rec, "cash", "buying_power",
                                   "cash_available_for_withdrawal", default=0.0))
            return cash, cash, cash

        b = self._binding("portfolio")
        args = self._account_arg(b, account_id)
        # 'portfolio' returns one object, not a collection -- _as_records is
        # for list-shaped responses (accounts, positions, orders) and falls
        # back to wrapping the whole raw payload as a single opaque "record"
        # when it can't find a list anywhere, which silently breaks _pick
        # below. Confirmed live (2026-08): the real shape is a single-level
        # wrapper, {"data": {"cash": ..., "total_value": ..., ...}}.
        rec = _unwrap_object(self._call_sync("portfolio", args))

        cash = _to_float(_pick(rec, "cash", "cash_balance",
                               "cash_available_for_withdrawal", default=np.nan))
        if asset_class == "crypto":
            equity = _to_float(_pick(rec, "crypto_value", default=np.nan))
        else:
            equity = _to_float(_pick(rec, "equity", "total_equity", "market_value",
                                     "portfolio_value", "total_value", default=np.nan))
        # buying_power is itself a nested object on the real response, not a
        # scalar -- confirmed live (2026-08): {"buying_power":
        # {"buying_power": "1000.0000", "unleveraged_buying_power": ..., ...}}.
        # _to_float(<dict>) would raise inside its own try/except and quietly
        # return 0.0 rather than surface that the shape was wrong.
        bp_value = _pick(rec, "buying_power", "cash_available_for_withdrawal",
                         default=None)
        if isinstance(bp_value, dict):
            bp_value = _pick(bp_value, "buying_power", "unleveraged_buying_power",
                             default=None)
        buying_power = _to_float(bp_value, default=np.nan)

        if not np.isfinite(cash) and not np.isfinite(equity):
            print(
                "WARNING: could not find cash/equity in the 'portfolio' "
                f"response: {rec!r}. Reporting 0.0 -- this is almost "
                "certainly wrong. Check the raw 'portfolio' response and "
                "fix _portfolio_figures's candidate field names."
            )
            return 0.0, 0.0, 0.0

        cash = cash if np.isfinite(cash) else 0.0
        equity = equity if np.isfinite(equity) else cash
        buying_power = buying_power if np.isfinite(buying_power) else cash
        return cash, equity, buying_power

    def _positions(self, account_id: str, asset_class: str = "equity") -> pd.Series:
        cap = "crypto_positions" if asset_class == "crypto" else "positions"
        b = self._binding(cap)
        args = self._account_arg(b, account_id)
        recs = _as_records(self._call_sync(cap, args))
        out: dict[str, float] = {}
        for r in recs:
            # "currency_code"/"asset_code" added as crypto-flavoured guesses
            # alongside the confirmed equity candidates -- unverified, see
            # this module's docstring.
            sym = _pick(r, "symbol", "ticker", "instrument_symbol",
                       "currency_code", "asset_code", default=None)
            qty = _to_float(_pick(r, "quantity", "shares", "qty",
                                  "amount", default=0.0))
            if sym and abs(qty) > 0:
                out[str(sym).upper()] = out.get(str(sym).upper(), 0.0) + qty
        return pd.Series(out, dtype=float)

    def get_quotes(self, symbols: Sequence[str], asset_class: str = "equity") -> pd.Series:
        cap = "crypto_quotes" if asset_class == "crypto" else "quotes"
        b = self._binding(cap)
        key = b.resolve_arg("symbols", ("symbols", "symbol", "tickers", "ticker"))
        if key is None:
            raise RuntimeError(f"cannot find symbol argument on {b.tool_name}")
        payload = b.coerce(key, list(symbols))
        recs = _as_records(self._call_sync(cap, {key: payload}))
        out: dict[str, float] = {}
        for r in recs:
            # Confirmed live (2026-08) for the equity 'quotes' tool: each
            # record bundles a live "quote" sub-object and a stale
            # end-of-day "close" sub-object as siblings, {"quote": {...},
            # "close": {...}} -- symbol/price fields live inside "quote",
            # not at the top level of the record. A naive top-level _pick
            # found neither and silently returned an empty series for every
            # request. Fall back to the record itself for a server that
            # returns a flatter shape -- which crypto_quotes may well do,
            # since it's unconfirmed (see this module's docstring); the
            # fallback exists precisely so this keeps working either way.
            q = r.get("quote") if isinstance(r.get("quote"), dict) else r
            sym = (_pick(q, "symbol", "ticker", default=None)
                   or _pick(r, "symbol", "ticker", default=None))
            px = _to_float(_pick(q, "last_trade_price", "last_price", "price",
                                 "mark_price", "ask_price", default=np.nan))
            if not np.isfinite(px):
                # "close" is yesterday's price, not live, but still better
                # than nothing when the market's closed or a live field
                # didn't parse. Crypto trades 24/7 so this fallback should
                # rarely matter there, unlike equities after hours.
                c = r.get("close") if isinstance(r.get("close"), dict) else {}
                px = _to_float(_pick(c, "price", default=np.nan))
            if sym and np.isfinite(px):
                out[str(sym).upper()] = px
        return pd.Series(out, dtype=float)

    def get_orders(self, since: datetime | None = None,
                   asset_class: str = "equity") -> list[BrokerOrder]:
        cap = "crypto_orders" if asset_class == "crypto" else "orders"
        b = self._binding(cap)
        # Confirmed live (2026-08) via the recover() path: the 'orders'
        # tool's schema requires account_number too. get_account() must
        # have already run to populate self.account_id (true everywhere
        # recover() is called from run_cycle.py) -- _account_arg() raises
        # clearly if it hasn't.
        args = self._account_arg(b, self.account_id)
        if since is not None:
            since_key = b.resolve_arg("since", ("start_date", "since", "after",
                                                 "created_after", "start"))
            if since_key:
                # Widen by a day before truncating to a date. `since` is a
                # UTC instant, and which calendar date that lands on depends
                # on the server's timezone: an order placed at 20:00 New York
                # is already "tomorrow" in UTC, so sending the bare UTC date
                # can ask for a window that starts *after* the order we're
                # looking for. Erring wide is free -- recover() fingerprints
                # every returned order anyway and consumes at most one match
                # -- whereas erring narrow resolves a real fill as
                # not_at_broker and halts the next cycle.
                args[since_key] = (since - timedelta(days=1)).date().isoformat()
        recs = _as_records(self._call_sync(cap, args))
        orders = []
        for r in recs:
            orders.append(BrokerOrder(
                order_id=str(_pick(r, "id", "order_id", default="")),
                symbol=str(_pick(r, "symbol", "ticker", default="")).upper(),
                side=str(_pick(r, "side", "direction", default="")).lower(),
                quantity=_to_float(_pick(r, "quantity", "shares", default=0.0)),
                state=_normalise_state(_pick(r, "state", "status", default="")),
                filled_quantity=_to_float(
                    _pick(r, "filled_quantity", "cumulative_quantity", default=0.0)),
                average_price=_maybe_float(
                    _pick(r, "average_price", "avg_price", default=None)),
                created_at=_maybe_dt(_pick(r, "created_at", "timestamp",
                                           "updated_at", default=None)),
                reject_reason=_pick(r, "reject_reason", "cancel_reason",
                                    default=None),
                raw=r,
            ))
        return orders

    # -- writes -----------------------------------------------------------

    def _load_crypto_pairs(self) -> None:
        """Populate ``self._crypto_pairs`` from ``get_currency_pairs`` once
        per broker lifetime (== once per cycle; ``run_cycle.py`` builds one
        broker per cycle).

        Best-effort. If the catalog tool isn't bound, or the call fails, or
        a coin just isn't in the response, callers fall back to the static
        ``CRYPTO_QUANTITY_DECIMALS`` table -- see the block comment there and
        ``_crypto_order_quantity()``. Paginates defensively (the full USD
        catalog was 91 pairs on 2026-08-30, one page at ``limit=700``, but
        the response does carry a ``next`` cursor at smaller limits).
        """
        if self._crypto_pairs is not None:
            return
        self._crypto_pairs = {}
        if "crypto_pairs" not in self.bindings:
            return
        try:
            specs: dict[str, dict] = {}
            cursor: str | None = None
            for _ in range(25):  # hard stop; the catalog is ~100 pairs
                args = {"limit": 700}
                if cursor:
                    args["cursor"] = cursor
                raw = self._call_sync("crypto_pairs", args)
                for r in _as_records(raw):
                    sym = str(_pick(r, "symbol", "display_symbol", "id",
                                    default="")).upper()
                    if not sym:
                        continue
                    specs[sym] = {
                        "qty_increment": _positive_decimal(_pick(
                            r, "min_order_quantity_increment", "asset_increment",
                            "quantity_increment")),
                        "min_order_size": _positive_decimal(_pick(
                            r, "min_order_size", "min_order_quantity")),
                    }
                cursor = _next_cursor(raw)
                if not cursor:
                    break
            self._crypto_pairs = specs
        except Exception as exc:  # noqa: BLE001 -- best-effort; static fallback
            logging.getLogger(__name__).warning(
                "get_currency_pairs lookup failed (%r); crypto quantity "
                "rounding falls back to the static CRYPTO_QUANTITY_DECIMALS "
                "table", exc)
            self._crypto_pairs = {}

    def _crypto_order_quantity(self, symbol: str, quantity: float) -> str:
        """``|quantity|`` snapped to the trading pair's own quantity
        increment, as a plain fixed-point string ready for the wire.

        The increment and ``min_order_size`` come from a live
        ``get_currency_pairs`` lookup (cached once per cycle by
        ``_load_crypto_pairs()``); on any miss this falls back to the
        static ``CRYPTO_QUANTITY_DECIMALS`` decimal-place table. Raises
        :class:`BrokerRejection` when the snapped size is zero or below the
        pair's ``min_order_size`` -- ``OrderManager.execute()`` turns that
        into a clean per-intent skip rather than a doomed round trip (the
        venue would 400 it) or an order that silently never fills.
        """
        self._load_crypto_pairs()
        spec = (self._crypto_pairs or {}).get(symbol.upper(), {})
        q = Decimal(str(abs(float(quantity))))

        inc = spec.get("qty_increment")
        if inc is not None:
            snapped = (q / inc).to_integral_value(ROUND_HALF_EVEN) * inc
        else:
            dp = _quantity_decimals("crypto_review", symbol)
            snapped = q.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_EVEN)

        min_size = spec.get("min_order_size")
        if snapped <= 0 or (min_size is not None and snapped < min_size):
            floor_txt = (f"{symbol.upper()} min_order_size "
                         f"{_plain_decimal(min_size)}" if min_size is not None
                         else "a positive size")
            raise BrokerRejection(
                f"{symbol.upper()} order size {_plain_decimal(q)} rounds to "
                f"{_plain_decimal(snapped) if snapped > 0 else '0'} at this "
                f"pair's quantity increment, below {floor_txt} -- nothing sent"
            )
        return _plain_decimal(snapped)

    def _order_args(self, capability: str, symbol: str, side: str,
                    quantity: float, order_type: str, extra: dict) -> dict:
        b = self._binding(capability)
        args: dict[str, Any] = {}

        def put(logical, candidates, value):
            key = b.resolve_arg(logical, candidates)
            if key is not None:
                args[key] = b.coerce(key, value)
            elif logical in b.required():
                raise RuntimeError(
                    f"{b.tool_name} requires {logical} but no matching "
                    f"parameter was found in its schema: {list(b.properties())}"
                )

        put("symbol", ("symbol", "ticker", "instrument"), symbol.upper())
        put("side", ("side", "direction"), side.lower())
        # "amount" is deliberately not a candidate here. On many real
        # brokerage APIs it denotes dollar notional, not a share count -- if
        # it were accepted as a stand-in for quantity, a schema exposing
        # only "amount" would silently turn a 10-share order into a $10
        # one. Better to leave quantity unresolved and let the required-
        # field check below fail loudly than guess at an ambiguous unit.
        # Confirmed live (2026-08): the API itself (not the MCP schema --
        # this passes schema validation and is rejected downstream)
        # enforces a quantity-precision limit. Computed share counts
        # (weight * equity / price) are ordinary floats with full binary
        # precision, e.g. 2.7255578430183576 -- sent as-is, every
        # fractional order fails with a 400 after already passing review.
        # Round once, here, so both review_order and place_order (both go
        # through this method) get a value that can actually be accepted --
        # the sub-unit difference this introduces is immaterial at any real
        # order size. The limit differs by asset class: 8 dp for equities;
        # crypto snaps to the trading pair's own (coarser) quantity
        # increment and can raise BrokerRejection for a sub-minimum size --
        # see _crypto_order_quantity() / _load_crypto_pairs().
        if capability.startswith("crypto_"):
            put("quantity", ("quantity", "shares", "qty"),
                self._crypto_order_quantity(symbol, quantity))
        else:
            put("quantity", ("quantity", "shares", "qty"),
                round(abs(quantity), _quantity_decimals(capability, symbol)))
        put("order_type", ("order_type", "type"), order_type)
        # Not routed through put(): that function's required-field check
        # only catches a missing *field name* in the schema, not a missing
        # *value* -- self.account_id being unset is a value problem, not a
        # schema problem, and _account_arg() gives it its own clearer error
        # rather than falling through to the generic belt-and-braces message
        # below, which used to make this look like a candidate-name
        # guessing problem when it was actually "get_account() was never
        # called."
        args.update(self._account_arg(b, self.account_id))
        for k, v in extra.items():
            key = b.resolve_arg(k, (k,))
            if key is not None:
                args[key] = b.coerce(key, v)

        # Belt-and-braces on top of put()'s own per-field check: that check
        # only catches a required field whose *logical* name happens to
        # match the server's field name (e.g. both spelled "quantity"). A
        # server that spells its required quantity-equivalent field
        # "amount" -- exactly the case the exclusion above is guarding
        # against -- would otherwise sail through with that field silently
        # absent from args.
        missing_required = [r for r in b.required() if r not in args]
        if missing_required:
            raise RuntimeError(
                f"{b.tool_name} still requires {missing_required} after "
                f"binding symbol/side/quantity/order_type -- schema: "
                f"{list(b.properties())}. Add the server's actual field "
                "name to the relevant candidate tuple rather than guessing "
                "at an ambiguous one."
            )
        return args

    def review_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", asset_class: str = "equity",
                     **kw) -> dict:
        cap = "crypto_review" if asset_class == "crypto" else "review"
        if cap not in self.bindings:
            return {"ok": True, "warnings": ["no review tool available"],
                    "estimated_price": np.nan, "estimated_notional": np.nan}
        raw = self._call_sync(
            cap, self._order_args(cap, symbol, side, quantity,
                                  order_type, kw))
        # A review response is one result, not a collection -- the same
        # shape as 'portfolio' (single object under "data"), not the same
        # shape as 'accounts' (list under "data"). _as_records is built for
        # the list case; handed a single object, it can't find one and
        # falls back to treating the whole {"data": {...}} wrapper as one
        # opaque record, which silently breaks every _pick call below (a
        # response *with* real warnings would read as none, defeating the
        # one thing this method exists to catch). Inferred from the
        # confirmed accounts/portfolio pattern, not independently verified
        # against a live review response -- call_raw("review", ...) to
        # check directly if this ever looks wrong.
        rec = _unwrap_object(raw)
        warnings = _pick(rec, "warnings", "alerts", "messages", default=[]) or []
        if isinstance(warnings, str):
            warnings = [warnings]
        return {
            "ok": not warnings,
            "estimated_price": _to_float(
                _pick(rec, "estimated_price", "price", default=np.nan)),
            "estimated_notional": _to_float(
                _pick(rec, "estimated_notional", "notional", "total",
                      default=np.nan)),
            "warnings": list(warnings),
            "raw": rec,
        }

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", asset_class: str = "equity",
                    **kw) -> BrokerOrder:
        cap = "crypto_place" if asset_class == "crypto" else "place"
        raw = self._call_sync(
            cap, self._order_args(cap, symbol, side, quantity,
                                  order_type, kw))
        # Same reasoning as review_order(): a place response is one order,
        # not a collection, so this needs _unwrap_object (single object
        # under "data"), not _as_records (list under "data") -- inferred
        # from the confirmed accounts/portfolio pattern, not independently
        # verified against a live place response.
        rec = _unwrap_object(raw)
        order_id = _pick(rec, "id", "order_id", default=None)
        if order_id is None or not str(order_id).strip():
            # The call went out -- for all we know the order is now live --
            # but the response can't be parsed into anything we can track or
            # cancel. Returning a BrokerOrder with a blank id would look
            # like an ordinary "pending" order to every downstream caller,
            # including OrderManager, which would then have no way to tell
            # "definitely never placed" from "placed but unreferenceable."
            # Raise instead, the same as this module's own stated
            # philosophy for a missing capability at connect time: fail
            # loudly rather than fabricate. OrderManager.execute() treats
            # this exception as an unknown outcome and refuses to retry it
            # until recover() has checked the broker directly.
            raise RuntimeError(
                f"place_order response for {symbol} {side} {quantity} could "
                f"not be parsed into an order id -- raw response: {rec!r}. "
                "The order may have gone through; check the broker directly "
                "before retrying."
            )
        return BrokerOrder(
            order_id=str(order_id),
            symbol=symbol.upper(), side=side.lower(), quantity=abs(quantity),
            state=_normalise_state(_pick(rec, "state", "status",
                                         default="pending")),
            filled_quantity=_to_float(
                _pick(rec, "filled_quantity", default=0.0)),
            average_price=_maybe_float(
                _pick(rec, "average_price", "price", default=None)),
            created_at=_maybe_dt(_pick(rec, "created_at", default=None))
            or datetime.now(timezone.utc),
            reject_reason=_pick(rec, "reject_reason", default=None),
            raw=rec,
        )

    def cancel_order(self, order_id: str, asset_class: str = "equity") -> bool:
        cap = "crypto_cancel" if asset_class == "crypto" else "cancel"
        if cap not in self.bindings:
            return False
        b = self._binding(cap)
        key = b.resolve_arg("order_id", ("order_id", "id"))
        order_id_key = key or "order_id"
        args = {order_id_key: b.coerce(order_id_key, order_id)}
        # Same account_number requirement as review/place -- cancel_equity_order's
        # schema requires it too, confirmed live (2026-08). This call doesn't
        # go through _order_args(), so it needs _account_arg() called
        # separately, same as get_orders() does.
        args.update(self._account_arg(b, self.account_id))
        raw = self._call_sync(cap, args)
        # bool(raw) alone treats any non-empty response as success, which
        # includes an error payload like {"error": "already filled"} --
        # exactly the case where the cancel did *not* happen. An explicit
        # error field means failure regardless of anything else in the
        # response; an explicit success-ish field is authoritative when
        # present. Absent either, fall back to the old non-empty-response
        # heuristic -- this adapter's response schema isn't verified
        # against the live service (see the class docstring), so a
        # confirmed error is the one thing worth being sure about.
        # Same single-object-under-"data" shape as review/place, inferred
        # from the confirmed accounts/portfolio pattern -- see review_order().
        rec = _unwrap_object(raw)
        if _pick(rec, "error", "error_message", default=None):
            return False
        success = _pick(rec, "success", "ok", "cancelled", "canceled", default=None)
        if success is not None:
            return bool(success)
        return bool(raw)

    def _call_sync(self, capability: str, arguments: dict) -> Any:
        return self._run(self._call(self._binding(capability).tool_name, arguments))

    def call_raw(self, capability: str, arguments: dict | None = None) -> Any:
        """Call a bound capability and return the raw, unparsed MCP response.

        For checking this adapter's field-name guesses (account type,
        agentic flag, position/order field names, ...) against what the
        real server actually returns -- see the module docstring and
        ``RobinhoodMCPBroker``'s own admission that none of this has been
        verified against the live service. Every parsing method above this
        one (``get_account``, ``_positions``, ``get_quotes``, ``get_orders``)
        is a guess at field names; this is how you check the guess instead
        of taking it on faith.
        """
        return self._call_sync(capability, arguments or {})


class _AssetClassView:
    """Pins a connected :class:`RobinhoodMCPBroker` to one ``asset_class``.

    :class:`~qbt.orders.OrderManager` and :class:`~qbt.live.LiveSignalRunner`
    call the plain ``BrokerAdapter`` methods with no ``asset_class``
    argument -- they run one plan against one broker. Running the crypto
    sleeve as a *second*, independent plan (see ``run_cycle.py``) over the
    *same* MCP connection needs something that looks like an ordinary
    broker to that code while always calling the crypto-bound tools
    underneath, without those two classes having to know anything
    asset-class-specific happened. Not a second connection: ``connect()``/
    ``close()`` are no-ops here on purpose, since the parent
    :class:`RobinhoodMCPBroker` owns that lifecycle and is expected to
    already be connected by the time this view is constructed (see
    :meth:`RobinhoodMCPBroker.crypto_view`).
    """

    def __init__(self, parent: "RobinhoodMCPBroker", asset_class: str) -> None:
        self._parent = parent
        self._asset_class = asset_class

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_account(self) -> BrokerAccount:
        return self._parent.get_account(asset_class=self._asset_class)

    def get_quotes(self, symbols: Sequence[str]) -> pd.Series:
        return self._parent.get_quotes(symbols, asset_class=self._asset_class)

    def get_orders(self, since: datetime | None = None) -> list[BrokerOrder]:
        return self._parent.get_orders(since=since, asset_class=self._asset_class)

    def review_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", **kw) -> dict:
        return self._parent.review_order(symbol, side, quantity, order_type,
                                         asset_class=self._asset_class, **kw)

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", **kw) -> BrokerOrder:
        return self._parent.place_order(symbol, side, quantity, order_type,
                                        asset_class=self._asset_class, **kw)

    def cancel_order(self, order_id: str) -> bool:
        return self._parent.cancel_order(order_id, asset_class=self._asset_class)


# ---------------------------------------------------------------------------
# Response coercion
# ---------------------------------------------------------------------------
# MCP tool results are content blocks, usually a JSON string in a text block.
# Shapes vary by server and by tool, so normalise once here rather than at
# twelve call sites.


def _unwrap_tool_result(result: Any) -> Any:
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool error: {getattr(result, 'content', result)}")
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured
    chunks = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    if not chunks:
        return {}
    joined = "\n".join(chunks)
    try:
        return json.loads(joined)
    except (ValueError, TypeError):
        return {"text": joined}


_LIST_WRAPPER_KEYS = ("results", "data", "items", "accounts", "positions",
                      "orders", "quotes")


def _find_wrapped_list(d: dict) -> list | None:
    for key in _LIST_WRAPPER_KEYS:
        value = d.get(key)
        if isinstance(value, list):
            return value
    return None


def _as_records(payload: Any) -> list[dict]:
    """Coerce any of the shapes a tool might return into a list of dicts.

    Handles both a flat wrapper (``{"accounts": [...]}``) and Robinhood's
    real one-level-deeper nesting, confirmed live (2026-08):
    ``{"data": {"accounts": [...]}}``. Checks one level of nesting under
    *any* dict-valued top-level key, not just ``"data"`` specifically,
    since a different tool could wrap the same way under a different name.
    """
    if payload is None:
        return []
    if isinstance(payload, dict):
        found = _find_wrapped_list(payload)
        if found is None:
            for value in payload.values():
                if isinstance(value, dict):
                    found = _find_wrapped_list(value)
                    if found is not None:
                        break
        if found is not None:
            return [r for r in found if isinstance(r, dict)]
        return [payload]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


# "order" confirmed live (2026-08): place_equity_order's real response is
# {"order": {"id": ..., "state": "unconfirmed", ...}}, not {"data": {...}}
# like portfolio -- each tool wraps its single-object response under a key
# named for what it returns, not a single generic envelope. Discovered when
# an EFA order that actually filled couldn't be parsed into an order id
# (the "order" key wasn't a candidate here, so _unwrap_object fell through
# to returning the whole {"order": {...}} wrapper as the "record", and
# _pick found no top-level "id" on it) -- place_order() correctly refused
# to fabricate an order id rather than silently return a wrong one, but
# that meant OrderManager treated a real, filled order as an unknown
# outcome. review_equity_order/cancel_equity_order are not yet confirmed
# to use "order" too (still inferred from the accounts/portfolio pattern,
# same caveat as before) -- call_raw(...) to check directly if one of
# those ever looks wrong.
_OBJECT_WRAPPER_KEYS = ("data", "result", "portfolio", "account", "order")


def _unwrap_object(payload: Any) -> dict:
    """Unwrap a single-object response, e.g. ``{"data": {...}} -> {...}``.

    Complementary to :func:`_as_records`/:func:`_find_wrapped_list`, which
    look for a *list*. Some tools -- Robinhood's ``portfolio`` confirmed
    live (2026-08) -- return one object, not a collection, still wrapped
    under a key like ``"data"``. Passing that straight to ``_as_records``
    finds no list anywhere and falls back to treating the whole wrapper as
    one opaque record, which silently breaks every ``_pick`` call against
    it -- the field it's looking for is one level too deep to see.

    Peels through *every* consecutive layer of single-key wrapping, not
    just one. Confirmed live (2026-08): place_equity_order's real response
    is two levels deep, {"data": {"order": {...}}} -- the same "data"
    envelope every other endpoint uses, plus a resource-name key, the same
    shape "accounts" uses for its list ({"data": {"accounts": [...]}}).
    An earlier version of this function stopped after unwrapping "data"
    once and never got to "order", which is exactly how a real, filled
    order (state="filled", not a rejection) came back unparseable and got
    treated as an unknown outcome a second time -- the first fix confirmed
    a one-level {"order": {...}} shape by reading an error message that
    had, itself, already been through one round of incomplete unwrapping.
    """
    if not isinstance(payload, dict):
        return {}
    depth = 0
    while depth < 5:
        key = next(
            (k for k in _OBJECT_WRAPPER_KEYS if isinstance(payload.get(k), dict)),
            None,
        )
        if key is None:
            break
        payload = payload[key]
        depth += 1
    return payload


def _pick(rec: dict, *keys, default=None):
    lowered = {k.lower(): v for k, v in rec.items()}
    for k in keys:
        if k.lower() in lowered and lowered[k.lower()] is not None:
            return lowered[k.lower()]
    return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_decimal(value):
    """Parse a catalog number (often a JSON string like ``"0.01"``) to a
    positive :class:`~decimal.Decimal`, or ``None`` if absent, unparseable,
    or non-positive -- so callers can tell "known" from "not known"."""
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d > 0 else None


def _plain_decimal(d: Decimal) -> str:
    """A Decimal as a plain fixed-point string -- no scientific notation
    (``format(x, "f")``) and no trailing-zero noise (``.normalize()``).
    ``Decimal("3.7651E-4")`` -> ``"0.00037651"``, ``Decimal("71.00")`` ->
    ``"71"``. Robinhood's crypto order tools declare ``quantity`` as a JSON
    string, so this is what actually goes on the wire."""
    d = d.normalize()
    if d == 0:
        return "0"
    return format(d, "f")


def _next_cursor(payload: Any) -> str | None:
    """The pagination cursor from a Robinhood list response, if any.

    Confirmed live (2026-08-30) for get_currency_pairs: the envelope is
    ``{"data": {"results": [...], "next": "http://.../?cursor=<c>&limit=5"}}``
    -- ``next`` is a full URL, the cursor is its ``cursor`` query param
    (absent entirely on the last page). Also accepts a bare cursor string
    under ``next``/``cursor``/``next_cursor`` in case another tool differs.
    """
    if not isinstance(payload, dict):
        return None
    for holder in (payload, *(v for v in payload.values() if isinstance(v, dict))):
        raw = _pick(holder, "next", "next_cursor", "cursor")
        if not raw:
            continue
        text = str(raw)
        if "://" in text or "cursor=" in text:
            qs = parse_qs(urlsplit(text).query)
            if qs.get("cursor"):
                return qs["cursor"][0]
            continue
        return text
    return None


def _maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_dt(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        return ts.to_pydatetime() if ts.tzinfo else ts.tz_localize("UTC").to_pydatetime()
    except (ValueError, TypeError):
        return None


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        # Fallback for the unconfirmed "is_agentic"/"agentic" field names --
        # the confirmed real field, "agentic_allowed" (see get_account()),
        # is a plain bool and never reaches this branch. A bare substring
        # check alone would read a negated value like "non_agentic" or
        # "not-agentic" as truthy just because it contains the letters
        # "agentic"; checking for a negation marker immediately before it
        # closes that without having to enumerate every real field spelling.
        if re.search(r"(?:^|[^a-z])(?:non|not)[-_]?agentic", v):
            return False
        return "agentic" in v
    return bool(value)


_STATE_MAP = {
    "filled": "filled", "complete": "filled", "completed": "filled",
    "partially_filled": "partial", "partial": "partial",
    "pending": "pending", "queued": "pending", "confirmed": "pending",
    "unconfirmed": "pending", "new": "pending",
    "cancelled": "cancelled", "canceled": "cancelled",
    "rejected": "rejected", "failed": "rejected",
}


def _normalise_state(value) -> str:
    return _STATE_MAP.get(str(value or "").strip().lower(), "pending")
