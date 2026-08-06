"""Validate the local, offline mechanics of qbt.oauth: token storage and the
loopback redirect-capture server. Nothing here talks to Robinhood -- that
part can only be verified by an actual first login against the live service.
"""

import asyncio
import json
import os
import stat
import shutil
import threading
import time
import urllib.request

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from qbt.oauth import FileTokenStorage, _make_handlers

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


WORK = "/tmp/qbt_oauth_test"
shutil.rmtree(WORK, ignore_errors=True)

print("=" * 72)
print("1. FileTokenStorage -- round trip and permissions")
print("=" * 72)


async def _storage_checks():
    storage = FileTokenStorage(os.path.join(WORK, "state.json"))

    check("no tokens before anything is written", await storage.get_tokens() is None)
    check("no client_info before anything is written",
          await storage.get_client_info() is None)

    tokens = OAuthToken(access_token="at-1", refresh_token="rt-1", expires_in=3600)
    await storage.set_tokens(tokens)
    back = await storage.get_tokens()
    check("tokens round-trip", back.access_token == "at-1" and back.refresh_token == "rt-1")

    info = OAuthClientInformationFull(
        redirect_uris=["http://127.0.0.1:8765/callback"],
        client_id="client-abc",
        client_secret=None,
    )
    await storage.set_client_info(info)
    back_info = await storage.get_client_info()
    check("client_info round-trip", back_info.client_id == "client-abc")

    # Setting one must not clobber the other -- they're read-modify-written
    # against the same underlying file.
    still_there = await storage.get_tokens()
    check("setting client_info doesn't clobber previously-stored tokens",
          still_there is not None and still_there.access_token == "at-1")

    mode = stat.S_IMODE(os.stat(storage.path).st_mode)
    check("token file is owner-only (0600)", mode == 0o600, oct(mode))

    tokens2 = OAuthToken(access_token="at-2", refresh_token="rt-2")
    await storage.set_tokens(tokens2)
    back2 = await storage.get_tokens()
    check("overwriting tokens actually updates them", back2.access_token == "at-2")
    still_info = await storage.get_client_info()
    check("overwriting tokens doesn't clobber client_info",
          still_info is not None and still_info.client_id == "client-abc")


asyncio.run(_storage_checks())

print()
print("=" * 72)
print("2. Loopback callback server -- captures a real redirect")
print("=" * 72)

PORT = 8765


async def _callback_checks():
    redirect_handler, callback_handler = _make_handlers(PORT)

    result_holder = {}

    async def run_callback():
        result_holder["result"] = await callback_handler()

    task = asyncio.create_task(run_callback())
    await asyncio.sleep(0.2)  # let the loopback server actually bind

    def fire_redirect():
        urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/callback?code=AUTHCODE123&state=STATE456",
            timeout=5,
        )

    await asyncio.get_event_loop().run_in_executor(None, fire_redirect)
    await asyncio.wait_for(task, timeout=5)

    code, state = result_holder["result"]
    check("callback_handler captures the authorization code", code == "AUTHCODE123")
    check("callback_handler captures the state param", state == "STATE456")


asyncio.run(_callback_checks())

print()
print("=" * 72)
print("3. Loopback callback server -- redirect missing a code raises clearly")
print("=" * 72)

PORT2 = 8766


async def _callback_missing_code():
    redirect_handler, callback_handler = _make_handlers(PORT2)

    async def run_callback():
        return await callback_handler()

    task = asyncio.create_task(run_callback())
    await asyncio.sleep(0.2)

    def fire_redirect_no_code():
        urllib.request.urlopen(f"http://127.0.0.1:{PORT2}/callback?error=access_denied",
                                timeout=5)

    await asyncio.get_event_loop().run_in_executor(None, fire_redirect_no_code)
    try:
        await asyncio.wait_for(task, timeout=5)
        check("missing code raises RuntimeError", False)
    except RuntimeError as exc:
        check("missing code raises RuntimeError", True, str(exc)[:60])


asyncio.run(_callback_missing_code())

shutil.rmtree(WORK, ignore_errors=True)

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
