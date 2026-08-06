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
        side = side.lower()
        self.call_log.append(("review_order", {"symbol": symbol, "side": side,
                                               "quantity": quantity}))
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
        positions = self._positions(aid)
        cash, equity, buying_power = self._portfolio_figures(aid, chosen)
        return BrokerAccount(
            account_id=aid,
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            positions=positions,
            is_agentic=True,
            day_trades_used=_maybe_int(_pick(chosen, "day_trades_used",
                                             "day_trade_count", default=None)),
            raw=chosen,
        )

    def _portfolio_figures(
        self, account_id: str, account_rec: dict
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
        """
        if "portfolio" not in self.bindings:
            cash = _to_float(_pick(account_rec, "cash", "buying_power",
                                   "cash_available_for_withdrawal", default=0.0))
            return cash, cash, cash

        b = self._binding("portfolio")
        args = {}
        key = b.resolve_arg("account", ("account_number", "account_id", "account"))
        if key:
            args[key] = account_id
        # 'portfolio' returns one object, not a collection -- _as_records is
        # for list-shaped responses (accounts, positions, orders) and falls
        # back to wrapping the whole raw payload as a single opaque "record"
        # when it can't find a list anywhere, which silently breaks _pick
        # below. Confirmed live (2026-08): the real shape is a single-level
        # wrapper, {"data": {"cash": ..., "total_value": ..., ...}}.
        rec = _unwrap_object(self._call_sync("portfolio", args))

        cash = _to_float(_pick(rec, "cash", "cash_balance",
                               "cash_available_for_withdrawal", default=np.nan))
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
        payload = b.coerce(key, list(symbols))
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
        put("quantity", ("quantity", "shares", "qty"), abs(quantity))
        put("order_type", ("order_type", "type"), order_type)
        # Not routed through put(): that function's required-field check
        # only catches a missing *field name* in the schema, not a missing
        # *value* -- self.account_id being unset is a value problem, not a
        # schema problem, and deserves its own clearer error rather than
        # falling through to the generic belt-and-braces message below,
        # which used to make this look like a candidate-name guessing
        # problem when it was actually "get_account() was never called."
        account_key = b.resolve_arg("account", ("account_number", "account_id", "account"))
        if account_key is not None:
            if self.account_id:
                args[account_key] = b.coerce(account_key, self.account_id)
            elif account_key in b.required():
                raise RuntimeError(
                    f"{b.tool_name} requires {account_key!r} but this broker "
                    "has no account_id set. Call get_account() at least once "
                    "before reviewing/placing/cancelling an order -- it's "
                    "what resolves the agentic account."
                )
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

    def cancel_order(self, order_id: str) -> bool:
        if "cancel" not in self.bindings:
            return False
        b = self._binding("cancel")
        key = b.resolve_arg("order_id", ("order_id", "id"))
        order_id_key = key or "order_id"
        args = {order_id_key: b.coerce(order_id_key, order_id)}
        # Same account_number requirement as review/place -- cancel_equity_order's
        # schema requires it too, confirmed live (2026-08). This call doesn't go
        # through _order_args(), so it needs its own copy of the same handling.
        account_key = b.resolve_arg("account", ("account_number", "account_id", "account"))
        if account_key is not None and self.account_id:
            args[account_key] = b.coerce(account_key, self.account_id)
        raw = self._call_sync("cancel", args)
        # bool(raw) alone treats any non-empty response as success, which
        # includes an error payload like {"error": "already filled"} --
        # exactly the case where the cancel did *not* happen. An explicit
        # error field means failure regardless of anything else in the
        # response; an explicit success-ish field is authoritative when
        # present. Absent either, fall back to the old non-empty-response
        # heuristic -- this adapter's response schema isn't verified
        # against the live service (see the class docstring), so a
        # confirmed error is the one thing worth being sure about.
        recs = _as_records(raw)
        rec = recs[0] if recs else {}
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


_OBJECT_WRAPPER_KEYS = ("data", "result", "portfolio", "account")


def _unwrap_object(payload: Any) -> dict:
    """Unwrap a single-object response, e.g. ``{"data": {...}} -> {...}``.

    Complementary to :func:`_as_records`/:func:`_find_wrapped_list`, which
    look for a *list*. Some tools -- Robinhood's ``portfolio`` confirmed
    live (2026-08) -- return one object, not a collection, still wrapped
    under a key like ``"data"``. Passing that straight to ``_as_records``
    finds no list anywhere and falls back to treating the whole wrapper as
    one opaque record, which silently breaks every ``_pick`` call against
    it -- the field it's looking for is one level too deep to see.
    """
    if not isinstance(payload, dict):
        return {}
    for key in _OBJECT_WRAPPER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
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
