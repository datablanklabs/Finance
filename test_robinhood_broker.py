"""Validate RobinhoodMCPBroker.get_account() against response shapes
confirmed from the live service (2026-08), so a future change can't
silently regress any of the bugs this session found:

* The real agentic-account flag is "agentic_allowed" (caller-relative,
  bool), not "is_agentic"/"agentic"/"type" -- "type" in the real schema
  means the account's trading type (margin/cash), an unrelated field that
  happened to never match, which is why every account failed the old
  check.
* "accounts" nests two levels deep, {"data": {"accounts": [...]}}, not one.
* "portfolio" returns a single object one level deep, {"data": {...}}, not
  a list -- a shape _as_records can't unwrap at all, since it's built for
  list-shaped responses.
* Within that object, "buying_power" is itself a nested object,
  {"buying_power": {"buying_power": "1000.0000", ...}}, not a scalar.
* review_equity_order/place_equity_order/cancel_equity_order all require
  account_number in their schema, but self.account_id (what _order_args()
  reads to fill it) was never populated by anything -- get_account()
  resolved the right account into a local variable and never cached it on
  the instance, so every order call failed with "still requires
  ['account_number']" even after the account itself was correctly found.
* review_equity_order declares "quantity" as a JSON *string* in its schema,
  not a number -- sending the raw Python float (e.g. 2.7255578430183576)
  failed MCP-side schema validation ('type: 2.72... has type "number",
  want "string"') before the order was even considered.
* get_equity_quotes declares "symbols" with a JSON Schema *union* type,
  ["null", "array"], not the bare string "array" -- a naive
  ``prop.get("type") == "array"`` check can never match a list, so
  get_quotes() fell through to comma-joining a list of 18 symbols into one
  string and got 'has type "string", want one of "null, array"' back.
* get_orders declares "account_number" as required too, same as
  review/place/cancel -- but get_orders() built its request args without
  ever resolving/including an account key at all (unlike _positions() and
  _portfolio_figures(), which both do), so OrderManager.recover()'s call to
  it failed MCP-side with 'missing properties: ["account_number"]'.
* place_equity_order's real response is {"order": {"id": ..., "state":
  "unconfirmed", ...}} -- a single object wrapped under "order", not under
  "data" like portfolio. "order" wasn't a candidate key in
  _unwrap_object(), so a real, filled order came back unparseable and
  place_order() (correctly) raised rather than fabricate an id -- which
  meant a genuinely successful live order got treated as an unknown
  outcome.
* Confirmed live (2026-08): the server 400s on the client's own
  session-termination DELETE at disconnect, every time, regardless of
  whether anything actually went wrong. The `mcp` SDK logs this as a bare
  warning from mcp.client.streamable_http with no caller-facing switch to
  turn it off, so importing qbt.broker installs a logging.Filter on that
  one logger that drops only this specific message -- not the whole
  logger, so a real warning from that module would still surface.


Account numbers and dollar values below are fabricated, not the real ones
from that session -- only the field names/nesting are verbatim.
"""

import io
import logging

import numpy as np

from qbt.broker import RobinhoodMCPBroker, ToolBinding, _schema_types, _unwrap_object

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


# The exact shape of a real 'accounts' response, field names verbatim,
# values fabricated.
REAL_ACCOUNTS_PAYLOAD = {
    "data": {
        "accounts": [
            {
                "account_number": "100000001",
                "type": "margin",
                "brokerage_account_type": "individual",
                "is_default": True,
                "agentic_allowed": False,
            },
            {
                "account_number": "100000002",
                "type": "cash",
                "brokerage_account_type": "individual",
                "nickname": "Agentic",
                "is_default": False,
                "agentic_allowed": True,
            },
        ]
    },
}

# The exact shape of a real 'portfolio' response, field names and nesting
# verbatim, dollar values fabricated.
REAL_PORTFOLIO_PAYLOAD = {
    "data": {
        "total_value": "1234",
        "equity_value": "0",
        "options_value": "0",
        "futures_value": "0",
        "event_contracts_value": "0",
        "crypto_value": "0",
        "cash": "1234",
        "pending_deposits": "0",
        "mutual_funds_value": "0",
        "fixed_income_value": "0",
        "currency": "USD",
        "buying_power": {
            "buying_power": "1234.0000",
            "unleveraged_buying_power": "1234.0000",
            "display_currency": "USD",
        },
    }
}


def _broker(portfolio_response=None, positions_response=None, with_portfolio_binding=True):
    b = RobinhoodMCPBroker(token="fake")
    b.bindings = {
        "accounts": ToolBinding("accounts", "get_accounts", {}),
        "positions": ToolBinding("positions", "get_equity_positions", {}),
    }
    if with_portfolio_binding:
        b.bindings["portfolio"] = ToolBinding("portfolio", "get_portfolio", {})

    def fake_call_sync(capability, arguments):
        if capability == "accounts":
            return REAL_ACCOUNTS_PAYLOAD
        if capability == "portfolio":
            return portfolio_response if portfolio_response is not None else {}
        if capability == "positions":
            return positions_response if positions_response is not None else {"data": {"positions": []}}
        raise AssertionError(f"unexpected capability {capability!r}")

    b._call_sync = fake_call_sync
    return b


print("=" * 72)
print("1. Agentic-account selection (the confirmed bug)")
print("=" * 72)

broker = _broker(portfolio_response={"equity": 1234.56, "cash": 500.0, "buying_power": 500.0})
account = broker.get_account()
check("selects the account with agentic_allowed=True", account.account_id == "100000002")
check("does not select the default (non-agentic) account", account.account_id != "100000001")
check("reports is_agentic=True on the returned account", account.is_agentic is True)

# All accounts non-agentic -> must raise, not silently pick the default.
all_non_agentic_payload = {
    "data": {"accounts": [dict(a, agentic_allowed=False) for a in REAL_ACCOUNTS_PAYLOAD["data"]["accounts"]]}
}
broker_none = RobinhoodMCPBroker(token="fake", require_agentic=True)
broker_none.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
}
broker_none._call_sync = lambda capability, arguments: (
    all_non_agentic_payload if capability == "accounts" else {"data": {"positions": []}}
)
try:
    broker_none.get_account()
    check("still raises when genuinely no agentic account exists", False)
except RuntimeError as exc:
    check("still raises when genuinely no agentic account exists", "no agentic account" in str(exc))

print()
print("=" * 72)
print("2. Cash/equity/buying_power parsed from the real 'portfolio' shape")
print("=" * 72)

broker2 = _broker(portfolio_response=REAL_PORTFOLIO_PAYLOAD)
account2 = broker2.get_account()
check("cash parses from the nested {data: {cash: ...}} shape", account2.cash == 1234.0)
check("equity uses total_value (whole account), not equity_value (stocks only)",
      account2.equity == 1234.0)
check("buying_power unwraps the nested buying_power.buying_power field",
      account2.buying_power == 1234.0)

# equity_value (stock positions only) must never be picked over total_value
# (whole account, including cash) -- they're deliberately different numbers
# in the fixture (0 vs 1234) so a wrong candidate order would be caught here.
broker2b = _broker(portfolio_response={
    "data": {**REAL_PORTFOLIO_PAYLOAD["data"], "equity_value": "999999"}
})
account2b = broker2b.get_account()
check("does not accidentally pick equity_value over total_value",
      account2b.equity == 1234.0)

broker3 = _broker(portfolio_response={"data": {"some_other_field": "x"}})
account3 = broker3.get_account()
check("unrecognized portfolio shape degrades to 0.0, doesn't crash",
      account3.cash == 0.0 and account3.equity == 0.0)

broker4 = _broker(portfolio_response={"data": {"cash": "500", "buying_power": "not-a-number"}})
account4 = broker4.get_account()
check("an unparseable buying_power value falls back to cash, doesn't crash",
      account4.buying_power == 500.0)

broker4 = _broker(with_portfolio_binding=False)
account4 = broker4.get_account()
check("falls back to the accounts record if 'portfolio' isn't bound at all",
      account4.cash == 0.0)  # accounts record here has no cash field either

print()
print("=" * 72)
print("3. account_number reaches review/place/cancel after get_account()")
print("=" * 72)

# The exact required-field shape confirmed live (2026-08): review_equity_order
# requires account_number alongside symbol/side/type/quantity.
REVIEW_SCHEMA = {
    "properties": {
        "account_number": {"type": "string"}, "symbol": {"type": "string"},
        "side": {"type": "string"}, "type": {"type": "string"},
        # Confirmed live (2026-08): quantity is declared as a JSON string
        # here, not a number -- this is the exact shape that failed.
        "quantity": {"type": "string"}, "dollar_amount": {"type": "string"},
    },
    "required": ["account_number", "symbol", "side", "type", "quantity"],
}
CANCEL_SCHEMA = {
    "properties": {"account_number": {"type": "string"}, "order_id": {"type": "string"}},
    "required": ["account_number", "order_id"],
}


def _order_broker():
    b = RobinhoodMCPBroker(token="fake")
    b.bindings = {
        "accounts": ToolBinding("accounts", "get_accounts", {}),
        "positions": ToolBinding("positions", "get_equity_positions", {}),
        "review": ToolBinding("review", "review_equity_order", REVIEW_SCHEMA),
        "cancel": ToolBinding("cancel", "cancel_equity_order", CANCEL_SCHEMA),
    }
    captured = {}

    def fake_call_sync(capability, arguments):
        if capability == "accounts":
            return REAL_ACCOUNTS_PAYLOAD
        if capability == "positions":
            return {"data": {"positions": []}}
        if capability in ("review", "cancel"):
            captured[capability] = arguments
            return {"data": {"success": True}}
        raise AssertionError(f"unexpected capability {capability!r}")

    b._call_sync = fake_call_sync
    return b, captured


ob, captured = _order_broker()
check("account_id is unset before get_account() is ever called", ob.account_id is None)
ob.get_account()
check("get_account() caches the resolved agentic account_id",
      ob.account_id == "100000002")

ob.review_order("XLF", "buy", 2.5)
check("review_order includes account_number once get_account() has run",
      captured.get("review", {}).get("account_number") == "100000002")

ob.cancel_order("order-123")
check("cancel_order includes account_number too, same fix",
      captured.get("cancel", {}).get("account_number") == "100000002")

fresh_broker = RobinhoodMCPBroker(token="fake")
fresh_broker.bindings = {"review": ToolBinding("review", "review_equity_order", REVIEW_SCHEMA)}
try:
    fresh_broker.review_order("XLF", "buy", 2.5)
    check("calling review_order before get_account() raises, doesn't send a null account_number", False)
except RuntimeError as exc:
    check("calling review_order before get_account() raises, doesn't send a null account_number",
          "get_account()" in str(exc), str(exc))

print()
print("=" * 72)
print("4. Values are coerced to the schema's declared type, not sent raw")
print("=" * 72)

ob2, captured2 = _order_broker()
ob2.get_account()
# The exact quantity from the reported error, so the regression test locks
# in the exact scenario that failed, not a rounder stand-in value.
ob2.review_order("XLF", "buy", 2.7255578430183576)
sent_quantity = captured2["review"]["quantity"]
check("quantity is coerced to a string to match the schema's declared type",
      isinstance(sent_quantity, str))
check("the coerced string is rounded to Robinhood's 8-decimal-place limit",
      sent_quantity == "2.72555784")

# The schema-declared-type check must be genuinely schema-driven, not a
# hardcoded "quantity is always a string" special case -- swap the schema
# to declare quantity as a number and confirm coercion follows suit.
ob3 = RobinhoodMCPBroker(token="fake")
ob3.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
    "review": ToolBinding("review", "review_equity_order", {
        **REVIEW_SCHEMA, "properties": {**REVIEW_SCHEMA["properties"], "quantity": {"type": "number"}},
    }),
}
captured3 = {}
def _fake_call_sync_numeric(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    if capability == "review":
        captured3["args"] = arguments
        return {"data": {"success": True}}
    raise AssertionError(capability)
ob3._call_sync = _fake_call_sync_numeric
ob3.get_account()
ob3.review_order("XLF", "buy", 2.5)
check("a schema declaring quantity as a number is left as a number, not forced to a string",
      isinstance(captured3["args"]["quantity"], float) and captured3["args"]["quantity"] == 2.5)

print()
print("=" * 72)
print("5. get_quotes() handles a union-typed array schema, not just a bare 'array'")
print("=" * 72)

check("_schema_types normalises the union form",
      _schema_types({"type": ["null", "array"]}) == {"null", "array"})
check("_schema_types normalises the bare-string form",
      _schema_types({"type": "array"}) == {"array"})
check("_schema_types on an absent 'type' is empty, not a guess",
      _schema_types({}) == set())

UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
            "XLV", "XLY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]
quotes_broker = RobinhoodMCPBroker(token="fake")
quotes_broker.bindings = {
    # The exact confirmed shape: a nullable array, not the bare string "array".
    "quotes": ToolBinding("quotes", "get_equity_quotes", {
        "properties": {"symbols": {"type": ["null", "array"], "items": {"type": "string"}}},
    }),
}
quotes_captured = {}
def _fake_quotes_call(capability, arguments):
    quotes_captured["args"] = arguments
    return {"data": {"quotes": [{"symbol": "XLF", "last_trade_price": "57.83"}]}}
quotes_broker._call_sync = _fake_quotes_call
quotes_broker.get_quotes(UNIVERSE)
check("symbols is sent as a real list, not a comma-joined string",
      isinstance(quotes_captured["args"]["symbols"], list))
check("the full symbol list survives, all 18 names",
      quotes_captured["args"]["symbols"] == UNIVERSE)

# A server that genuinely only accepts a comma-joined string must still get one.
string_quotes_broker = RobinhoodMCPBroker(token="fake")
string_quotes_broker.bindings = {
    "quotes": ToolBinding("quotes", "get_equity_quotes", {
        "properties": {"symbols": {"type": "string"}},
    }),
}
string_captured = {}
def _fake_string_quotes_call(capability, arguments):
    string_captured["args"] = arguments
    return {"data": {"quotes": []}}
string_quotes_broker._call_sync = _fake_string_quotes_call
string_quotes_broker.get_quotes(["AAPL", "MSFT"])
check("a schema that genuinely wants a string still gets one, comma-joined",
      string_captured["args"]["symbols"] == "AAPL,MSFT")

print()
print("=" * 72)
print("6. get_orders() includes account_number, same as review/place/cancel")
print("=" * 72)

ORDERS_SCHEMA = {
    "properties": {
        "account_number": {"type": "string"},
        "created_after": {"type": "string"},
    },
    "required": ["account_number"],
}


def _orders_broker():
    b = RobinhoodMCPBroker(token="fake")
    b.bindings = {
        "accounts": ToolBinding("accounts", "get_accounts", {}),
        "positions": ToolBinding("positions", "get_equity_positions", {}),
        "orders": ToolBinding("orders", "get_equity_orders", ORDERS_SCHEMA),
    }
    captured = {}

    def fake_call_sync(capability, arguments):
        if capability == "accounts":
            return REAL_ACCOUNTS_PAYLOAD
        if capability == "positions":
            return {"data": {"positions": []}}
        if capability == "orders":
            captured["orders"] = arguments
            return {"data": {"orders": []}}
        raise AssertionError(f"unexpected capability {capability!r}")

    b._call_sync = fake_call_sync
    return b, captured


orders_broker, orders_captured = _orders_broker()
orders_broker.get_account()
orders_broker.get_orders()
check("get_orders includes account_number once get_account() has run",
      orders_captured.get("orders", {}).get("account_number") == "100000002")

fresh_orders_broker = RobinhoodMCPBroker(token="fake")
fresh_orders_broker.bindings = {"orders": ToolBinding("orders", "get_equity_orders", ORDERS_SCHEMA)}
try:
    fresh_orders_broker.get_orders()
    check("calling get_orders before get_account() raises, doesn't send a null account_number", False)
except RuntimeError as exc:
    check("calling get_orders before get_account() raises, doesn't send a null account_number",
          "get_account()" in str(exc), str(exc))

print()
print("=" * 72)
print("7. place_order() parses the real, doubly-wrapped {\"data\": {\"order\": {...}}} shape")
print("=" * 72)

PLACE_SCHEMA = {
    "properties": {
        "account_number": {"type": "string"}, "symbol": {"type": "string"},
        "side": {"type": "string"}, "type": {"type": "string"},
        "quantity": {"type": "string"},
    },
    "required": ["account_number", "symbol", "side", "type", "quantity"],
}
# The exact shape confirmed live (2026-08): TWO levels deep,
# {"data": {"order": {...}}} -- the same "data" envelope every other
# endpoint uses, plus a resource-name key, the same pattern "accounts"
# uses for its list ({"data": {"accounts": [...]}}). Field names verbatim,
# id/timestamps fabricated. An earlier fix confirmed only a one-level
# {"order": {...}} shape (REAL_PLACE_RESPONSE_LEGACY_SHAPE below) by
# reading a RuntimeError's already-partly-unwrapped `rec!r` text rather
# than the true raw payload -- a second real, filled order (XLF) came back
# unparseable a second time because _unwrap_object stopped after peeling
# "data" and never got to "order".
REAL_PLACE_RESPONSE = {
    "data": {
        "order": {
            "id": "6a75fb46-4005-4062-b3d5-3a66c7b0058a",
            "instrument_id": "f25b2d63-0372-4827-9907-e7e9e37a10f1",
            "symbol": "",
            "side": "buy",
            "type": "market",
            "state": "filled",
            "quantity": "2.000000",
            "cumulative_quantity": "2.000000",
            "price": "57.520000",
            "average_price": "57.519900",
            "created_at": "2026-08-07T15:35:34.713562Z",
        }
    }
}
# The one-level shape the first fix (incorrectly) assumed was the whole
# story -- still a real possibility (maybe some other tool really does
# only wrap once), so _unwrap_object must keep handling it too.
REAL_PLACE_RESPONSE_LEGACY_SHAPE = {"order": dict(REAL_PLACE_RESPONSE["data"]["order"],
                                                   id="legacy-shape-id")}

place_broker = RobinhoodMCPBroker(token="fake")
place_broker.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
    "place": ToolBinding("place", "place_equity_order", PLACE_SCHEMA),
}
def _fake_place_call(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    if capability == "place":
        return REAL_PLACE_RESPONSE
    raise AssertionError(f"unexpected capability {capability!r}")
place_broker._call_sync = _fake_place_call
place_broker.get_account()

placed = place_broker.place_order("XLF", "buy", 2.0)
check("place_order parses the id out of the {\"data\": {\"order\": {...}}} "
      "double wrapper, doesn't raise",
      placed.order_id == "6a75fb46-4005-4062-b3d5-3a66c7b0058a")
check("state is read from inside the fully-unwrapped 'order' object",
      placed.state == "filled")
check("average_price is read from inside the fully-unwrapped 'order' object",
      placed.average_price == 57.5199)

legacy_broker = RobinhoodMCPBroker(token="fake")
legacy_broker.bindings = dict(place_broker.bindings)
def _fake_legacy_call(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    if capability == "place":
        return REAL_PLACE_RESPONSE_LEGACY_SHAPE
    raise AssertionError(f"unexpected capability {capability!r}")
legacy_broker._call_sync = _fake_legacy_call
legacy_broker.get_account()
legacy_placed = legacy_broker.place_order("XLF", "buy", 2.0)
check("a one-level {\"order\": {...}} response (no \"data\" wrapper) still parses too",
      legacy_placed.order_id == "legacy-shape-id")

print()
print("=" * 72)
print("8. Session-termination noise is filtered, not the whole logger")
print("=" * 72)

_mcp_logger = logging.getLogger("mcp.client.streamable_http")
_capture = io.StringIO()
_handler = logging.StreamHandler(_capture)
_mcp_logger.addHandler(_handler)
try:
    _mcp_logger.warning("Session termination failed: 400")
    _mcp_logger.warning("Session termination failed: some other exception text")
    _mcp_logger.warning("a genuinely different warning that should still surface")
finally:
    _mcp_logger.removeHandler(_handler)

_captured_text = _capture.getvalue()
check("the status-code variant of the noise is filtered",
      "Session termination failed: 400" not in _captured_text)
check("the exception-text variant of the noise is filtered too",
      "some other exception text" not in _captured_text)
check("an unrelated warning from the same logger still surfaces",
      "genuinely different warning" in _captured_text, repr(_captured_text))

print()
print("=" * 72)
print("9. _unwrap_object() peels every layer of wrapping, not just one")
print("=" * 72)

check("a two-level {\"data\": {\"order\": {...}}} response is fully unwrapped",
      _unwrap_object({"data": {"order": {"id": "x"}}}) == {"id": "x"})
check("a one-level {\"order\": {...}} response still works",
      _unwrap_object({"order": {"id": "x"}}) == {"id": "x"})
check("an already-flat response is returned unchanged",
      _unwrap_object({"id": "x"}) == {"id": "x"})
check("a non-dict payload returns an empty dict, doesn't crash",
      _unwrap_object([1, 2, 3]) == {})
# The critical non-regression: portfolio's real shape has a legitimate
# nested field named "buying_power" that must NOT be mistaken for another
# layer of wrapping just because unwrapping is now multi-level.
_portfolio_like = {"data": {"cash": "500", "buying_power": {"buying_power": "500"}}}
check("multi-level unwrapping does not over-unwrap into unrelated nested "
      "fields (portfolio's real shape)",
      _unwrap_object(_portfolio_like) == {"cash": "500",
                                          "buying_power": {"buying_power": "500"}})

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
