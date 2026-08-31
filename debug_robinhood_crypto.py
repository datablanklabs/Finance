#!/usr/bin/env python3
"""Diagnostic: dump exactly what Robinhood's MCP server returns for the
crypto tools (crypto_positions, crypto_quotes, crypto_orders), so
qbt/broker.py's crypto_* capability candidates and field-name guesses in
_positions()/get_quotes()/get_orders() can be corrected against ground
truth instead of more guessing -- the same role debug_robinhood_accounts.py
played for the equity surface (see its own docstring and the README's
"Confirmed against the live service" for what that process actually found:
nested envelopes, string-typed quantities, and other surprises no amount of
reading documentation would have caught).

Read-only -- calls list_capabilities(), require_crypto(), and the raw
crypto_positions/crypto_quotes/crypto_orders tools for whichever account has
agentic_allowed=true. Nothing that places, reviews, or cancels an order,
same restraint debug_robinhood_accounts.py uses. Reuses the OAuth token
already stored from a prior run_cycle.py login (same storage path/port), so
this should not need a new browser login unless the refresh token has
expired or was revoked.

    python3 debug_robinhood_crypto.py

Run this BEFORE trusting `run_cycle.py --consider-crypto --live` against a
real account -- everything it prints is either confirming or correcting an
unverified guess (see qbt/broker.py's module docstring and the README's
"Crypto support" section for the full list of what's still open).
"""

from __future__ import annotations

import json
import os
from decimal import Decimal

from qbt.broker import REQUIRED_CRYPTO_CAPABILITIES, RobinhoodMCPBroker, _as_records, _pick, _truthy
from qbt.oauth import build_robinhood_oauth

# A couple of the starter list from run_cycle.py's CRYPTOS -- enough to see
# a real crypto_quotes response shape without guessing at the account's
# actual tradeable coin list (there's no discovery for that; see the
# README's "Crypto support" section, point 3).
_SAMPLE_SYMBOLS = ["BTC-USD", "ETH-USD"]

# run_cycle.py's CRYPTOS, kept in sync by hand -- used only to slice the
# get_currency_pairs catalog down to the coins the crypto sleeve trades so
# CRYPTO_QUANTITY_DECIMALS (qbt/broker.py) can be sourced/refreshed from it.
_CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "LTC-USD",
                    "BCH-USD", "AVAX-USD", "SHIB-USD", "XRP-USD", "ADA-USD",
                    "LINK-USD", "UNI-USD", "AAVE-USD", "ETC-USD", "XLM-USD"]


def _dump(title: str, payload) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    oauth = build_robinhood_oauth(
        storage_path=os.environ.get(
            "ROBINHOOD_OAUTH_STATE", "state/robinhood_oauth.json"
        ),
        port=int(os.environ.get("ROBINHOOD_OAUTH_CALLBACK_PORT", "8765")),
    )
    broker = RobinhoodMCPBroker(auth=oauth, require_agentic=True)
    broker.connect()

    print("=" * 72)
    print("Discovered tools -> bound capabilities")
    print("=" * 72)
    print(broker.list_capabilities().to_string(index=False))

    print()
    try:
        broker.require_crypto()
        print(f"require_crypto() PASSED -- all of {REQUIRED_CRYPTO_CAPABILITIES} "
              "are bound.")
    except RuntimeError as exc:
        print(f"require_crypto() FAILED: {exc}")
        print()
        print("Nothing further to check -- correct CAPABILITY_CANDIDATES's "
              "crypto_* entries in qbt/broker.py against the tool list "
              "above, or this account/server genuinely has no crypto "
              "trading enabled yet.")
        broker.close()
        return

    accounts_raw = broker.call_raw("accounts")
    agentic_account_id = None
    for rec in _as_records(accounts_raw):
        if _truthy(_pick(rec, "agentic_allowed", "is_agentic", "agentic", default=False)):
            agentic_account_id = str(
                _pick(rec, "account_number", "account_id", "id", default="")
            )
            break

    if not agentic_account_id:
        print()
        print("Could not identify an agentic_allowed=true account -- "
              "skipping crypto_positions/crypto_orders lookups (same "
              "'accounts' response debug_robinhood_accounts.py already "
              "dumps in full; run that first if this is unexpected).")
        broker.close()
        return

    print()
    print(f"Identified agentic account: {agentic_account_id}")

    # 'portfolio' is the same tool the equity path already confirmed --
    # crypto_value's presence there was itself confirmed live (2026-08, see
    # test_robinhood_broker.py's REAL_PORTFOLIO_PAYLOAD), but whether that's
    # the *right* figure to size the crypto sleeve against (vs. something
    # derived from crypto_positions) is still open -- see the README.
    if "portfolio" in broker.bindings:
        binding = broker._binding("portfolio")
        key = binding.resolve_arg("account", ("account_number", "account_id", "account"))
        args = {key: agentic_account_id} if key else {}
        raw = broker.call_raw("portfolio", args)
        _dump("Raw response from the 'portfolio' tool (check crypto_value here)", raw)

    for capability, title, extra_args in (
        ("crypto_positions", "crypto_positions", {}),
        ("crypto_orders", "crypto_orders", {}),
    ):
        if capability not in broker.bindings:
            print(f"\n'{capability}' capability not bound -- skipping.")
            continue
        binding = broker._binding(capability)
        key = binding.resolve_arg("account", ("account_number", "account_id", "account"))
        args = dict(extra_args)
        if key:
            args[key] = agentic_account_id
        raw = broker.call_raw(capability, args)
        _dump(f"Raw response from the '{title}' tool", raw)

    if "crypto_quotes" in broker.bindings:
        binding = broker._binding("crypto_quotes")
        key = binding.resolve_arg("symbols", ("symbols", "symbol", "tickers", "ticker"))
        if key is None:
            print("\n'crypto_quotes' is bound but no symbol-like argument "
                  f"could be resolved from its schema: {list(binding.properties())}")
        else:
            payload = binding.coerce(key, _SAMPLE_SYMBOLS)
            raw = broker.call_raw("crypto_quotes", {key: payload})
            _dump(f"Raw response from the 'crypto_quotes' tool "
                  f"(requested {_SAMPLE_SYMBOLS})", raw)
    else:
        print("\n'crypto_quotes' capability not bound -- skipping.")

    # get_currency_pairs -- the pair catalog. min_order_quantity_increment
    # here is the source of truth for qbt/broker.py's CRYPTO_QUANTITY_DECIMALS
    # (an over-precise quantity 400s at review with "...too much precision.
    # Please round the quantity to an appropriate increment..."). Print the
    # sizing fields for every coin in _CRYPTO_UNIVERSE so the table can be
    # refreshed by hand when Robinhood retunes a pair or CRYPTOS grows.
    if "crypto_pairs" in broker.bindings:
        raw = broker.call_raw("crypto_pairs", {"limit": 700})
        _dump("Raw response from the 'get_currency_pairs' tool (first page)", raw)
        recs = _as_records(raw)
        by_sym = {}
        for r in recs:
            sym = str(_pick(r, "symbol", "display_symbol", "id", default="")).upper()
            by_sym[sym] = r
            by_sym.setdefault(sym.replace("-USD", ""), r)
        print()
        print("=" * 72)
        print("min_order_quantity_increment / min_order_size per _CRYPTO_UNIVERSE coin")
        print("(-> CRYPTO_QUANTITY_DECIMALS in qbt/broker.py)")
        print("=" * 72)
        for sym in _CRYPTO_UNIVERSE:
            r = by_sym.get(sym) or by_sym.get(sym.replace("-USD", ""))
            if r is None:
                print(f"{sym:10s}  <not in catalog>")
                continue
            inc = _pick(r, "min_order_quantity_increment", "asset_increment",
                        "quantity_increment", default=None)
            try:
                dp = max(0, -Decimal(str(inc)).normalize().as_tuple().exponent) \
                    if inc is not None else "?"
            except (ArithmeticError, ValueError):
                dp = "?"
            print(f"{sym:10s}  increment={inc!s:14s} -> {dp} dp   "
                  f"min_order_size={_pick(r, 'min_order_size', default='?')}   "
                  f"halted={_pick(r, 'halted', 'trading_halted', default='?')}")
    else:
        print("\n'crypto_pairs' (get_currency_pairs) capability not bound -- "
              "skipping; CRYPTO_QUANTITY_DECIMALS cannot be auto-sourced.")

    broker.close()


if __name__ == "__main__":
    main()
