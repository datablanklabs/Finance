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
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "BrokerAccount",
    "BrokerOrder",
    "BrokerAdapter",
    "MockBroker",
    "RobinhoodMCPBroker",
    "ToolBinding",
]


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
        """Coarse identity for matching against an intent we may have sent.

        Quantity is bucketed because a broker may round fractional shares, and
        an exact float match would make reconciliation miss its own orders --
        which is the failure that causes double submission.
        """
        bucket = round(self.quantity / max(qty_tolerance, 1e-9))
        return (self.symbol.upper(), self.side.lower(), bucket)


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
        self.call_log.append(("review_order", {"symbol": symbol, "side": side,
                                               "quantity": quantity}))
        px = float(self.prices.get(symbol, np.nan))
        warnings = []
        if symbol.upper() in self.fail_on_symbols:
            warnings.append("symbol not tradeable")
        if not np.isfinite(px):
            warnings.append("no quote available")
        est = abs(quantity) * (px if np.isfinite(px) else 0.0)
        if side == "buy" and est > self.cash:
            warnings.append("insufficient buying power")
        held = float(self.positions.get(symbol, 0.0))
        if side == "sell" and quantity > held + 1e-9:
            warnings.append("sell exceeds position")
        return {
            "ok": not warnings,
            "estimated_price": px,
            "estimated_notional": est,
            "warnings": warnings,
        }

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", **kw) -> BrokerOrder:
        self._require()
        self.call_log.append(("place_order", {"symbol": symbol, "side": side,
                                              "quantity": quantity}))
        self._seq += 1
        oid = f"mock-{self._seq:05d}"
        now = datetime.now(timezone.utc)

        review = self.review_order(symbol, side, quantity, order_type)
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

        px = float(self.prices[symbol])
        sign = 1.0 if side == "buy" else -1.0
        fill_px = px * (1.0 + sign * self.slippage_bps / 1e4)
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
        assumed ``symbol``. Matching is exact-then-substring over the schema's
        declared properties.
        """
        props = {k.lower(): k for k in self.properties()}
        for c in candidates:
            if c.lower() in props:
                return props[c.lower()]
        for c in candidates:
            for low, orig in props.items():
                if c.lower() in low:
                    return orig
        return None


# Candidate names per capability, most likely first. Discovery matches against
# these; if your server uses something else, add it here rather than editing
# call sites.
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
}

REQUIRED_CAPABILITIES = ("accounts", "positions", "quotes", "orders", "place")


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

    # -- plumbing ---------------------------------------------------------

    def _run(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _session(self):
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
        return streamablehttp_client(self.url, **kwargs), ClientSession

    async def _call(self, tool_name: str, arguments: dict) -> Any:
        ctx, ClientSession = await self._session()
        async with ctx as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _unwrap_tool_result(result)

    async def _discover(self) -> list[dict]:
        ctx, ClientSession = await self._session()
        async with ctx as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                listing = await session.list_tools()
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

    # -- reads ------------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        accounts = _as_records(self._call_sync("accounts", {}))
        chosen = None
        for a in accounts:
            aid = str(_pick(a, "account_number", "account_id", "id", default=""))
            agentic = _truthy(_pick(a, "is_agentic", "agentic", "type",
                                    default=False))
            if self.account_id and aid == self.account_id:
                chosen = a
                break
            if self.require_agentic and agentic and chosen is None:
                chosen = a
        if chosen is None:
            if self.require_agentic:
                raise RuntimeError(
                    "no agentic account found. Trade placement is confined to "
                    "the Agentic account; refusing to proceed against another."
                )
            chosen = accounts[0] if accounts else {}

        aid = str(_pick(chosen, "account_number", "account_id", "id", default=""))
        positions = self._positions(aid)
        cash = _to_float(_pick(chosen, "cash", "buying_power",
                               "cash_available_for_withdrawal", default=0.0))
        equity = _to_float(_pick(chosen, "equity", "total_equity",
                                 "market_value", default=np.nan))
        return BrokerAccount(
            account_id=aid,
            cash=cash,
            equity=equity if np.isfinite(equity) else cash,
            buying_power=_to_float(_pick(chosen, "buying_power", default=cash)),
            positions=positions,
            is_agentic=True,
            day_trades_used=_maybe_int(_pick(chosen, "day_trades_used",
                                             "day_trade_count", default=None)),
            raw=chosen,
        )

    def _positions(self, account_id: str) -> pd.Series:
        b = self._binding("positions")
        args = {}
        key = b.resolve_arg("account", ("account_number", "account_id", "account"))
        if key:
            args[key] = account_id
        recs = _as_records(self._call_sync("positions", args))
        out: dict[str, float] = {}
        for r in recs:
            sym = _pick(r, "symbol", "ticker", "instrument_symbol", default=None)
            qty = _to_float(_pick(r, "quantity", "shares", "qty", default=0.0))
            if sym and abs(qty) > 0:
                out[str(sym).upper()] = out.get(str(sym).upper(), 0.0) + qty
        return pd.Series(out, dtype=float)

    def get_quotes(self, symbols: Sequence[str]) -> pd.Series:
        b = self._binding("quotes")
        key = b.resolve_arg("symbols", ("symbols", "symbol", "tickers", "ticker"))
        if key is None:
            raise RuntimeError(f"cannot find symbol argument on {b.tool_name}")
        prop = b.properties().get(key, {})
        payload = list(symbols) if prop.get("type") == "array" else ",".join(symbols)
        recs = _as_records(self._call_sync("quotes", {key: payload}))
        out: dict[str, float] = {}
        for r in recs:
            sym = _pick(r, "symbol", "ticker", default=None)
            px = _to_float(_pick(r, "last_trade_price", "last_price", "price",
                                 "mark_price", "ask_price", default=np.nan))
            if sym and np.isfinite(px):
                out[str(sym).upper()] = px
        return pd.Series(out, dtype=float)

    def get_orders(self, since: datetime | None = None) -> list[BrokerOrder]:
        b = self._binding("orders")
        args = {}
        if since is not None:
            key = b.resolve_arg("since", ("start_date", "since", "after",
                                          "created_after", "start"))
            if key:
                args[key] = since.date().isoformat()
        recs = _as_records(self._call_sync("orders", args))
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

    def _order_args(self, capability: str, symbol: str, side: str,
                    quantity: float, order_type: str, extra: dict) -> dict:
        b = self._binding(capability)
        args: dict[str, Any] = {}

        def put(logical, candidates, value):
            key = b.resolve_arg(logical, candidates)
            if key is not None:
                args[key] = value
            elif logical in b.required():
                raise RuntimeError(
                    f"{b.tool_name} requires {logical} but no matching "
                    f"parameter was found in its schema: {list(b.properties())}"
                )

        put("symbol", ("symbol", "ticker", "instrument"), symbol.upper())
        put("side", ("side", "direction"), side.lower())
        put("quantity", ("quantity", "shares", "qty", "amount"), abs(quantity))
        put("order_type", ("order_type", "type"), order_type)
        if self.account_id:
            put("account", ("account_number", "account_id", "account"),
                self.account_id)
        for k, v in extra.items():
            key = b.resolve_arg(k, (k,))
            if key is not None:
                args[key] = v
        return args

    def review_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "market", **kw) -> dict:
        if "review" not in self.bindings:
            return {"ok": True, "warnings": ["no review tool available"],
                    "estimated_price": np.nan, "estimated_notional": np.nan}
        raw = self._call_sync(
            "review", self._order_args("review", symbol, side, quantity,
                                       order_type, kw))
        recs = _as_records(raw)
        rec = recs[0] if recs else {}
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
                    order_type: str = "market", **kw) -> BrokerOrder:
        raw = self._call_sync(
            "place", self._order_args("place", symbol, side, quantity,
                                      order_type, kw))
        recs = _as_records(raw)
        rec = recs[0] if recs else {}
        return BrokerOrder(
            order_id=str(_pick(rec, "id", "order_id", default="")),
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

    def cancel_order(self, order_id: str) -> bool:
        if "cancel" not in self.bindings:
            return False
        b = self._binding("cancel")
        key = b.resolve_arg("order_id", ("order_id", "id"))
        raw = self._call_sync("cancel", {key or "order_id": order_id})
        return bool(raw)

    def _call_sync(self, capability: str, arguments: dict) -> Any:
        return self._run(self._call(self._binding(capability).tool_name, arguments))


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


def _as_records(payload: Any) -> list[dict]:
    """Coerce any of the shapes a tool might return into a list of dicts."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "accounts", "positions",
                    "orders", "quotes"):
            if key in payload and isinstance(payload[key], list):
                return [r for r in payload[key] if isinstance(r, dict)]
        return [payload]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


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
        return "agentic" in value.lower() or value.lower() in ("true", "1", "yes")
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
