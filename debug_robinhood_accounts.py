#!/usr/bin/env python3
"""Diagnostic: dump exactly what Robinhood's MCP server returns for the
'accounts', 'portfolio', and 'positions' tools, so qbt/broker.py's
field-name guesses can be corrected against ground truth instead of more
guessing.

Read-only -- calls list_capabilities() and the raw 'accounts'/'portfolio'/
'positions' tools for whichever account has agentic_allowed=true. Nothing
that places, reviews, or cancels an order. Reuses the OAuth token already
stored from a prior run_cycle.py login (same storage path/port), so this
should not need a new browser login unless the refresh token has expired or
was revoked.

    python3 debug_robinhood_accounts.py
"""

from __future__ import annotations

import json
import os

from qbt.broker import RobinhoodMCPBroker, _as_records, _pick, _truthy
from qbt.oauth import build_robinhood_oauth


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

    accounts_raw = broker.call_raw("accounts")
    _dump("Raw response from the 'accounts' tool", accounts_raw)

    agentic_account_id = None
    for rec in _as_records(accounts_raw):
        if _truthy(_pick(rec, "agentic_allowed", "is_agentic", "agentic", default=False)):
            agentic_account_id = str(
                _pick(rec, "account_number", "account_id", "id", default="")
            )
            break

    if not agentic_account_id:
        print()
        print("Could not identify an agentic_allowed=true account from the "
              "response above -- skipping portfolio/positions lookups.")
        broker.close()
        return

    print()
    print(f"Identified agentic account: {agentic_account_id}")

    for capability, title in (("portfolio", "portfolio"), ("positions", "positions")):
        if capability not in broker.bindings:
            print(f"\n'{capability}' capability not bound -- skipping.")
            continue
        binding = broker._binding(capability)
        key = binding.resolve_arg("account", ("account_number", "account_id", "account"))
        args = {key: agentic_account_id} if key else {}
        raw = broker.call_raw(capability, args)
        _dump(f"Raw response from the '{title}' tool", raw)

    broker.close()


if __name__ == "__main__":
    main()
