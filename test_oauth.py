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
import urllib.error
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

    # The check above only proves the *final* mode is right -- true even
    # under the old write-then-chmod code, since chmod eventually ran.
    # That code created the file via Path.write_text() first (the OS
    # default creation mode, typically 0o644-0o666 depending on umask --
    # world/group readable) and only *then* chmod'd it to 0600, leaving a
    # real transient exposure window for a live refresh token. Confirmed
    # live (2026-08) under a permissive umask (0o000): write_text() alone
    # produces 0o666. Prove the fix is atomic at creation, not
    # write-then-fix, two ways: chmod is never called at all, and the
    # file is 0600 even under a umask that would unmask everything else.
    chmod_calls = []
    _real_chmod = os.chmod
    os.chmod = lambda *a, **kw: chmod_calls.append((a, kw)) or _real_chmod(*a, **kw)
    old_umask = os.umask(0o000)
    try:
        perm_storage = FileTokenStorage(os.path.join(WORK, "perm_check.json"))
        await perm_storage.set_tokens(
            OAuthToken(access_token="perm-at", refresh_token="perm-rt"))
    finally:
        os.umask(old_umask)
        os.chmod = _real_chmod
    perm_mode = stat.S_IMODE(os.stat(perm_storage.path).st_mode)
    check("token file is created at 0600 atomically, even under a "
          "permissive umask that would unmask everything else",
          perm_mode == 0o600, oct(perm_mode))
    check("permissions are set at creation, not via a separate chmod call "
          "after the file already exists",
          len(chmod_calls) == 0, chmod_calls)

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

print()
print("=" * 72)
print("4. Loopback callback server -- a stray wrong-path request doesn't "
      "eat the real callback's turn")
print("=" * 72)

PORT3 = 8767


async def _callback_stray_request():
    redirect_handler, callback_handler = _make_handlers(PORT3)

    result_holder = {}

    async def run_callback():
        result_holder["result"] = await callback_handler()

    task = asyncio.create_task(run_callback())
    await asyncio.sleep(0.2)

    def fire_stray_then_real():
        # A request to some other path -- e.g. a browser or stray local
        # process probing the port -- used to consume the server's
        # one-shot handle_request() call before the real redirect ever
        # arrived, since the original handler accepted any GET
        # regardless of path. The fixed handler responds 404 to it, which
        # urlopen raises as HTTPError -- expected, not a test failure.
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT3}/favicon.ico", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, f"expected 404 for the stray path, got {exc.code}"
        urllib.request.urlopen(
            f"http://127.0.0.1:{PORT3}/callback?code=AUTHCODE789&state=STATE000",
            timeout=5,
        )

    await asyncio.get_event_loop().run_in_executor(None, fire_stray_then_real)
    await asyncio.wait_for(task, timeout=5)

    code, state = result_holder["result"]
    check("a stray request to the wrong path is ignored, not captured as "
          "the OAuth result",
          code == "AUTHCODE789" and state == "STATE000", (code, state))


asyncio.run(_callback_stray_request())

shutil.rmtree(WORK, ignore_errors=True)

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
else:
    print("ALL CHECKS PASSED")
print("=" * 72)
