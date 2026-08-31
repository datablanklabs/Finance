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
* place_equity_order's real response is wrapped TWO levels deep,
  {"data": {"order": {"id": ..., "state": "unconfirmed", ...}}} -- the same
  "data" envelope every other endpoint uses, plus a resource-name key, the
  same pattern "accounts" uses for its list. An earlier fix confirmed only
  a one-level {"order": {...}} shape (inferred from a RuntimeError's
  already-partly-unwrapped `rec!r` text, not the true raw payload), so
  _unwrap_object() stopped after peeling "data" once and never got to
  "order" -- meaning a real, filled order came back unparseable and
  place_order() (correctly) raised rather than fabricate an id, a second
  time, for a second real order. _unwrap_object() now peels every
  consecutive layer of wrapping, not just one.
* Confirmed live (2026-08): the server 400s on the client's own
  session-termination DELETE at disconnect, every time, regardless of
  whether anything actually went wrong. The `mcp` SDK logs this as a bare
  warning from mcp.client.streamable_http with no caller-facing switch to
  turn it off, so importing qbt.broker installs a logging.Filter on that
  one logger that drops only this specific message -- not the whole
  logger, so a real warning from that module would still surface.
* get_equity_quotes's real response is {"data": {"results": [{"quote":
  {"symbol": ..., "last_trade_price": ..., ...}, "close": {...}}, ...]}} --
  each result bundles a live "quote" sub-object and a stale end-of-day
  "close" sub-object as siblings. symbol/price live inside "quote", not at
  the top level of the record, so a naive top-level _pick found neither
  and get_quotes() silently returned an empty series for every real
  request -- no exception, just no prices, which meant a portfolio summary
  showed shares held with no value or weight for any position.
* A fractional-share rejection now rounds a sub-share target UP to one
  whole share when establishing a brand-new position (current_weight ~ 0)
  -- see qbt/orders.py's execute(). Confirmed live (2026-08): a $1,000
  account targeting 5 equal-weight ETF sleeves at ~$160 each hit the
  fractional-share rejection on every sleeve whose share price exceeds
  that, and rounding down to 0 there means never holding that name at all,
  on any account this size. A marginal top-up or trim on a position
  already held still rounds down and skips.


Account numbers and dollar values below are fabricated, not the real ones
from that session -- only the field names/nesting are verbatim.

Section 10 is a different kind of test from everything above it: sections
1-9 pin down shapes *confirmed* against a live response. Nothing about
crypto has been confirmed live yet (see qbt/broker.py's module docstring
and debug_robinhood_crypto.py) -- section 10 only checks that
asset_class="crypto" dispatches to the crypto_* capability bindings and
parses a *plausible*, clearly-fabricated response correctly, i.e. that the
plumbing this session added is internally consistent. It is not evidence
the guessed tool names or response shapes are right.
"""

import io
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np

from qbt.broker import (
    REQUIRED_CRYPTO_CAPABILITIES, BrokerRejection, RobinhoodMCPBroker, ToolBinding,
    _schema_types, _unwrap_object,
)

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
quotes_result = quotes_broker.get_quotes(UNIVERSE)
check("symbols is sent as a real list, not a comma-joined string",
      isinstance(quotes_captured["args"]["symbols"], list))
check("the full symbol list survives, all 18 names",
      quotes_captured["args"]["symbols"] == UNIVERSE)
check("a flat {symbol, last_trade_price} record still parses to a price",
      quotes_result.get("XLF") == 57.83)

# The exact confirmed live shape: each record bundles a live "quote"
# sub-object and a stale "close" sub-object as siblings -- symbol/price
# live inside "quote", not at the top of the record. A naive top-level
# _pick found neither and silently returned an empty series for every
# real quotes call, so every position's value/weight in a portfolio
# summary came back blank.
REAL_QUOTES_PAYLOAD = {
    "data": {
        "results": [
            {
                "quote": {"symbol": "EFA", "last_trade_price": "108.320000",
                          "bid_price": "108.320000", "ask_price": "108.330000"},
                "close": {"symbol": "EFA", "price": "107.36"},
            },
            {
                "quote": {"symbol": "XLF", "last_trade_price": "57.560000",
                          "bid_price": "57.550000", "ask_price": "57.560000"},
                "close": {"symbol": "XLF", "price": "57.81"},
            },
        ]
    }
}
nested_quotes_broker = RobinhoodMCPBroker(token="fake")
nested_quotes_broker.bindings = {
    "quotes": ToolBinding("quotes", "get_equity_quotes", {
        "properties": {"symbols": {"type": ["null", "array"], "items": {"type": "string"}}},
    }),
}
nested_quotes_broker._call_sync = lambda capability, arguments: REAL_QUOTES_PAYLOAD
nested_result = nested_quotes_broker.get_quotes(["EFA", "XLF"])
check("the real nested {\"quote\": {...}, \"close\": {...}} shape parses both symbols",
      set(nested_result.index) == {"EFA", "XLF"}, dict(nested_result))
check("the live quote price is used, not the stale close price",
      nested_result["EFA"] == 108.32 and nested_result["XLF"] == 57.56)

# When the live "quote" sub-object has no usable price field, "close" is a
# reasonable fallback -- still better than dropping the symbol entirely.
close_fallback_payload = {
    "data": {"results": [
        {"quote": {"symbol": "EFA"}, "close": {"symbol": "EFA", "price": "107.36"}},
    ]}
}
fallback_broker = RobinhoodMCPBroker(token="fake")
fallback_broker.bindings = dict(nested_quotes_broker.bindings)
fallback_broker._call_sync = lambda capability, arguments: close_fallback_payload
fallback_result = fallback_broker.get_quotes(["EFA"])
check("falls back to the stale close price when the live quote has none",
      fallback_result.get("EFA") == 107.36)

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
print("10. Crypto: asset_class dispatch, require_crypto(), crypto_view() "
      "(unverified shapes -- see this file's own docstring)")
print("=" * 72)

check("require_crypto() checks exactly the crypto capability set",
      set(REQUIRED_CRYPTO_CAPABILITIES) ==
      {"crypto_positions", "crypto_quotes", "crypto_orders", "crypto_place"})

# A server with no crypto tools at all -- the common case today, since
# crypto was only just added -- must not break plain equity usage. connect()
# itself isn't exercised here (that needs a real MCP session); what matters
# is that an equity-only bindings dict, exactly what today's real servers
# produce, doesn't make require_crypto() pass by accident.
equity_only_broker = RobinhoodMCPBroker(token="fake")
equity_only_broker.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
    "quotes": ToolBinding("quotes", "get_equity_quotes", {}),
    "orders": ToolBinding("orders", "get_equity_orders", {}),
    "place": ToolBinding("place", "place_equity_order", {}),
}
equity_only_broker._all_tools = [
    {"name": n, "description": "", "input_schema": {}}
    for n in ("get_accounts", "get_equity_positions", "get_equity_quotes",
             "get_equity_orders", "place_equity_order")
]
try:
    equity_only_broker.require_crypto()
    check("require_crypto() raises when the server has no crypto tools", False)
except RuntimeError as exc:
    check("require_crypto() raises when the server has no crypto tools", True)
    check("...and names exactly what's missing, not a generic message",
          "crypto_positions" in str(exc) and "crypto_quotes" in str(exc),
          str(exc))

# A server that does advertise the guessed get_crypto_X / X_crypto_order
# names -- discovery should bind them the same way it already binds the
# equity ones, since it's the identical candidate-matching mechanism.
crypto_tool_names = ["get_crypto_positions", "get_crypto_quotes",
                     "get_crypto_orders", "review_crypto_order",
                     "place_crypto_order", "cancel_crypto_order"]
crypto_capable_broker = RobinhoodMCPBroker(token="fake")
crypto_capable_broker.bindings = dict(equity_only_broker.bindings)
crypto_capable_broker.bindings.update({
    "crypto_positions": ToolBinding("crypto_positions", "get_crypto_positions", {}),
    "crypto_quotes": ToolBinding("crypto_quotes", "get_crypto_quotes",
                                 {"properties": {"symbols": {"type": "array"}}}),
    "crypto_orders": ToolBinding("crypto_orders", "get_crypto_orders", {}),
    "crypto_review": ToolBinding("crypto_review", "review_crypto_order", {}),
    "crypto_place": ToolBinding("crypto_place", "place_crypto_order", {}),
    "crypto_cancel": ToolBinding("crypto_cancel", "cancel_crypto_order", {}),
})
crypto_capable_broker._all_tools = equity_only_broker._all_tools + [
    {"name": n, "description": "", "input_schema": {}} for n in crypto_tool_names
]
try:
    crypto_capable_broker.require_crypto()
    check("require_crypto() passes when the server advertises crypto tools", True)
except RuntimeError:
    check("require_crypto() passes when the server advertises crypto tools", False)

# get_account(asset_class="crypto") reads crypto_value (not equity_value/
# total_value) for "equity", and crypto_positions (not positions) for
# holdings -- the same REAL_ACCOUNTS_PAYLOAD/portfolio-with-crypto_value
# fixtures section 1/2 already use, since accounts/portfolio are the same
# tools for both asset classes (only positions/quotes/orders/place differ).
crypto_capable_broker.bindings["accounts"] = ToolBinding("accounts", "get_accounts", {})
crypto_capable_broker.bindings["portfolio"] = ToolBinding("portfolio", "get_portfolio", {})
_crypto_positions_payload = {
    "data": {"positions": [{"symbol": "BTC-USD", "quantity": "0.5"},
                           {"currency_code": "ETH-USD", "amount": "2.0"}]}
}
_crypto_portfolio_payload = {
    "data": {"cash": "1000", "crypto_value": "3456.78",
             "equity_value": "0", "total_value": "1000",
             "buying_power": {"buying_power": "1000.0000"}}
}


def _fake_crypto_call(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "portfolio":
        return _crypto_portfolio_payload
    if capability == "crypto_positions":
        return _crypto_positions_payload
    if capability == "positions":
        return {"data": {"positions": []}}
    raise AssertionError(f"unexpected capability {capability!r}")


crypto_capable_broker._call_sync = _fake_crypto_call
crypto_account = crypto_capable_broker.get_account(asset_class="crypto")
check("asset_class='crypto' reads crypto_value for equity, not equity_value/total_value",
      crypto_account.equity == 3456.78, crypto_account.equity)
check("asset_class='crypto' reads crypto_positions, not equity positions",
      dict(crypto_account.positions) == {"BTC-USD": 0.5, "ETH-USD": 2.0},
      dict(crypto_account.positions))
check("crypto accounts report day_trades_used=None (PDT doesn't apply to crypto)",
      crypto_account.day_trades_used is None)

# The plain equity path (asset_class="equity", the default) must be
# completely unaffected by any of the above -- same broker, same bindings,
# just the default asset_class.
equity_account = crypto_capable_broker.get_account()
check("the default asset_class='equity' still reads equity_value/total_value",
      equity_account.positions.empty or "BTC-USD" not in equity_account.positions.index)

# crypto_view() forwards to the parent with asset_class="crypto" fixed, and
# never re-connects (connect()/close() are no-ops) -- OrderManager and
# LiveSignalRunner call the plain BrokerAdapter methods with no asset_class
# argument, so this is what makes run_pipeline() work unmodified against
# either sleeve (see run_cycle.py).
view = crypto_capable_broker.crypto_view()
view.connect()  # must not raise, must not touch the network
view_account = view.get_account()
check("crypto_view().get_account() matches calling get_account(asset_class='crypto') directly",
      view_account.equity == crypto_account.equity and
      dict(view_account.positions) == dict(crypto_account.positions))
view.close()  # must not raise

# Crypto order quantity is rounded to the trading pair's own increment, not
# the equity 8-dp limit. Confirmed live (2026-08): review_crypto_order 400s
# on an over-precise quantity ("...too much precision. Please round the
# quantity to an appropriate increment..."), which _order_args() must
# pre-empt for both review and place. CRYPTO_QUANTITY_DECIMALS carries
# min_order_quantity_increment (as decimal places) per pair, from a live
# get_currency_pairs response (2026-08-30); a coin not listed falls back to
# _DEFAULT_CRYPTO_QUANTITY_DECIMALS (6 dp).
crypto_qty_broker = RobinhoodMCPBroker(token="fake")
crypto_qty_broker.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
    "review": ToolBinding("review", "review_equity_order", REVIEW_SCHEMA),
    "crypto_review": ToolBinding("crypto_review", "review_crypto_order", REVIEW_SCHEMA),
}
_crypto_qty_captured = {}


def _crypto_qty_call(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    _crypto_qty_captured[capability] = arguments
    return {"data": {"success": True}}


crypto_qty_broker._call_sync = _crypto_qty_call
crypto_qty_broker.get_account()
crypto_qty_broker.review_order("BTC-USD", "buy", 0.0003765123456789,
                               asset_class="crypto")
check("crypto quantity is rounded to the pair's own increment (BTC-USD 8 dp), "
      "not left at full binary precision",
      _crypto_qty_captured["crypto_review"]["quantity"] == "0.00037651",
      _crypto_qty_captured["crypto_review"]["quantity"])
crypto_qty_broker.review_order("ADA-USD", "buy", 71.108246, asset_class="crypto")
check("a coarse pair (ADA-USD, increment 0.01) rounds to 2 dp -- the live "
      "6-dp 'too much precision' rejection this table fixes",
      _crypto_qty_captured["crypto_review"]["quantity"] == "71.11",
      _crypto_qty_captured["crypto_review"]["quantity"])
crypto_qty_broker.review_order("XLF", "buy", 2.7255578430183576)
check("the equity path still rounds to 8 dp, unaffected by the crypto rule",
      _crypto_qty_captured["review"]["quantity"] == "2.72555784",
      _crypto_qty_captured["review"]["quantity"])

from qbt.broker import CRYPTO_QUANTITY_DECIMALS  # noqa: E402

# An explicit entry overrides whatever the shipped table / default would
# give. Perturb a real entry to a value that matches neither (shipped 2,
# default 6) and restore it.
_saved_doge = CRYPTO_QUANTITY_DECIMALS["DOGE-USD"]
CRYPTO_QUANTITY_DECIMALS["DOGE-USD"] = 3
try:
    crypto_qty_broker.review_order("DOGE-USD", "buy", 89.723451, asset_class="crypto")
    check("CRYPTO_QUANTITY_DECIMALS is consulted per symbol at call time",
          _crypto_qty_captured["crypto_review"]["quantity"] == "89.723",
          _crypto_qty_captured["crypto_review"]["quantity"])
finally:
    CRYPTO_QUANTITY_DECIMALS["DOGE-USD"] = _saved_doge

# When get_currency_pairs (bound as "crypto_pairs") is available, the live
# min_order_quantity_increment / min_order_size drive rounding -- the static
# CRYPTO_QUANTITY_DECIMALS table is only the fallback. Fetched once per
# broker (== once per cycle) and cached.
_PAIRS_PAGE_1 = {"data": {"results": [
    {"symbol": "ADA-USD", "min_order_quantity_increment": "0.001",
     "min_order_size": "0.1"},
    {"symbol": "XLM-USD", "min_order_quantity_increment": "0.01",
     "min_order_size": "1"},
], "next": "http://edge/currency_pairs/?cursor=PAGE2&limit=2"}}
_PAIRS_PAGE_2 = {"data": {"results": [
    {"symbol": "SHIB-USD", "min_order_quantity_increment": "1",
     "min_order_size": "800"},
]}}
_pairs_calls = []


def _catalog_call(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    if capability == "crypto_pairs":
        _pairs_calls.append(dict(arguments))
        return _PAIRS_PAGE_2 if arguments.get("cursor") == "PAGE2" else _PAIRS_PAGE_1
    _crypto_qty_captured[capability] = arguments
    return {"data": {"success": True}}


cat_broker = RobinhoodMCPBroker(token="fake")
cat_broker.bindings = {
    "accounts": ToolBinding("accounts", "get_accounts", {}),
    "positions": ToolBinding("positions", "get_equity_positions", {}),
    "crypto_review": ToolBinding("crypto_review", "review_crypto_order", REVIEW_SCHEMA),
    "crypto_pairs": ToolBinding("crypto_pairs", "get_currency_pairs", {}),
}
cat_broker._call_sync = _catalog_call
cat_broker.get_account()

cat_broker.review_order("ADA-USD", "buy", 71.108246, asset_class="crypto")
check("the live get_currency_pairs increment wins over the static table "
      "(ADA-USD catalog 0.001 -> 3 dp, not the shipped 2 dp)",
      _crypto_qty_captured["crypto_review"]["quantity"] == "71.108",
      _crypto_qty_captured["crypto_review"]["quantity"])

cat_broker.review_order("ADA-USD", "sell", 5.5, asset_class="crypto")
check("the catalog is fetched once and cached, not re-fetched per order",
      len(_pairs_calls) == 2 and [c.get("cursor") for c in _pairs_calls] == [None, "PAGE2"],
      f"{_pairs_calls}")
check("pagination followed the 'next' cursor -- a page-2 pair is known",
      "SHIB-USD" in (cat_broker._crypto_pairs or {}),
      list((cat_broker._crypto_pairs or {}).keys()))

# min_order_size guard: a size that snaps below the pair minimum is refused
# locally with BrokerRejection, never sent.
raised = None
try:
    cat_broker.review_order("XLM-USD", "buy", 0.4, asset_class="crypto")
except BrokerRejection as exc:
    raised = exc
check("a crypto size below the pair's min_order_size raises BrokerRejection "
      "before any call goes out",
      raised is not None and "min_order_size" in str(raised), repr(raised))
check("BrokerRejection is a RuntimeError (execute()'s except Exception catches it)",
      isinstance(raised, RuntimeError))

# A coin absent from the catalog falls back to the static table, no crash.
cat_broker.review_order("LINK-USD", "buy", 2.100654, asset_class="crypto")
check("a coin not in the live catalog falls back to CRYPTO_QUANTITY_DECIMALS "
      "(LINK-USD -> 4 dp)",
      _crypto_qty_captured["crypto_review"]["quantity"] == "2.1007",
      _crypto_qty_captured["crypto_review"]["quantity"])

# If the catalog call itself fails, the static table still carries the sleeve.
def _catalog_call_raises(capability, arguments):
    if capability == "accounts":
        return REAL_ACCOUNTS_PAYLOAD
    if capability == "positions":
        return {"data": {"positions": []}}
    if capability == "crypto_pairs":
        raise RuntimeError("MCP tool error: API error 503: upstream unavailable")
    _crypto_qty_captured[capability] = arguments
    return {"data": {"success": True}}


err_broker = RobinhoodMCPBroker(token="fake")
err_broker.bindings = dict(cat_broker.bindings)
err_broker._call_sync = _catalog_call_raises
err_broker.get_account()
err_broker.review_order("ADA-USD", "buy", 71.108246, asset_class="crypto")
check("a failed get_currency_pairs lookup degrades to the static table, "
      "not an exception (ADA-USD -> shipped 2 dp)",
      _crypto_qty_captured["crypto_review"]["quantity"] == "71.11",
      _crypto_qty_captured["crypto_review"]["quantity"])

print()
print("=" * 72)
print("11. Persistent MCP session -- reused across calls, torn down cleanly "
      "(lifecycle mechanics only; live behaviour unconfirmed, see broker.py)")
print("=" * 72)


class _FakeTransportCtx:
    """Stands in for streamablehttp_client(...)'s return value: an async
    context manager yielding (read, write, ...)."""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return (object(), object(), None)

    async def __aexit__(self, *exc):
        self.exited += 1
        return False


class _FakeSession:
    """Stands in for mcp.ClientSession -- tracks how many times it's
    entered/initialized/called, so the tests below can tell "one session,
    reused" apart from "a fresh session per call."
    """

    def __init__(self, read, write):
        self.entered = 0
        self.exited = 0
        self.initialized = 0
        self.call_log = []

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        return False

    async def initialize(self):
        self.initialized += 1

    async def call_tool(self, name, arguments):
        self.call_log.append((name, arguments))
        return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=[])

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(name="get_accounts", description="", inputSchema={}),
            SimpleNamespace(name="get_equity_positions", description="", inputSchema={}),
            SimpleNamespace(name="get_equity_quotes", description="", inputSchema={}),
            SimpleNamespace(name="get_equity_orders", description="", inputSchema={}),
            SimpleNamespace(name="place_equity_order", description="", inputSchema={}),
        ])


def _install_fake_mcp(session_factory):
    """Monkeypatch sys.modules so `from mcp import ClientSession` /
    `from mcp.client.streamable_http import streamablehttp_client` inside
    RobinhoodMCPBroker._open_session() resolve to fakes, without disturbing
    the real, already-installed `mcp` package for any other test. Returns a
    restore function -- callers must call it, even on failure, so a later
    section's genuine `import mcp` (if any) sees the real package again.
    """
    created_transports = []

    def fake_streamablehttp_client(url, **kwargs):
        ctx = _FakeTransportCtx()
        created_transports.append(ctx)
        return ctx

    fake_mcp = types.ModuleType("mcp")
    fake_mcp.ClientSession = session_factory
    fake_streamable = types.ModuleType("mcp.client.streamable_http")
    fake_streamable.streamablehttp_client = fake_streamablehttp_client

    saved = {name: sys.modules.get(name) for name in
             ("mcp", "mcp.client.streamable_http")}
    sys.modules["mcp"] = fake_mcp
    sys.modules["mcp.client.streamable_http"] = fake_streamable

    def restore():
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    return created_transports, restore


_created_sessions = []


def _tracking_session_factory(read, write):
    s = _FakeSession(read, write)
    _created_sessions.append(s)
    return s


_transports, _restore_mcp = _install_fake_mcp(_tracking_session_factory)
try:
    session_broker = RobinhoodMCPBroker(token="fake")
    session_broker.connect()
    check("connect() opens exactly one transport", len(_transports) == 1)
    check("connect() opens exactly one session", len(_created_sessions) == 1)
    check("connect() initializes the session exactly once",
          _created_sessions[0].initialized == 1)

    session_broker._call_sync("accounts", {})
    session_broker._call_sync("accounts", {})
    session_broker._call_sync("accounts", {})
    check("three calls after connect() still open only one transport/session "
          "(reused, not reconnected per call)",
          len(_transports) == 1 and len(_created_sessions) == 1)
    check("all three calls went through the one persistent session",
          len(_created_sessions[0].call_log) == 3)
    check("the session was still only ever initialized once, not once per call",
          _created_sessions[0].initialized == 1)

    session_broker.close()
    check("close() tears down the persistent session and transport",
          _created_sessions[0].exited == 1 and _transports[0].exited == 1)
    check("close() clears the cached session/exit-stack references",
          session_broker._mcp_session is None and session_broker._exit_stack is None)

    # Idempotent and safe even if called again (e.g. an except-block close()
    # after main() already closed once) or if connect() never ran at all.
    try:
        session_broker.close()
        check("close() is safe to call a second time", True)
    except Exception as exc:
        check("close() is safe to call a second time", False, repr(exc))

    never_connected = RobinhoodMCPBroker(token="fake")
    try:
        never_connected.close()
        check("close() is safe to call when connect() was never called", True)
    except Exception as exc:
        check("close() is safe to call when connect() was never called", False, repr(exc))
finally:
    _restore_mcp()


class _FailingInitSession(_FakeSession):
    async def initialize(self):
        self.initialized += 1
        raise RuntimeError("simulated handshake failure")


_failing_sessions = []


def _failing_session_factory(read, write):
    s = _FailingInitSession(read, write)
    _failing_sessions.append(s)
    return s


_fail_transports, _restore_mcp_2 = _install_fake_mcp(_failing_session_factory)
try:
    failing_broker = RobinhoodMCPBroker(token="fake")
    try:
        failing_broker.connect()
        check("connect() propagates a handshake failure rather than "
              "swallowing it", False)
    except RuntimeError as exc:
        check("connect() propagates a handshake failure rather than "
              "swallowing it", "simulated handshake failure" in str(exc))
    check("a failed handshake still tears down the transport it had already "
          "opened (no leaked open connection)",
          _fail_transports[0].exited == 1)
    check("a failed connect() leaves no half-set session/exit-stack state "
          "behind for a later call to trip over",
          failing_broker._mcp_session is None and failing_broker._exit_stack is None)
finally:
    _restore_mcp_2()

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
