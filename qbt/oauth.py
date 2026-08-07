"""OAuth 2.0 Authorization Code + PKCE for :class:`~qbt.broker.RobinhoodMCPBroker`.

Builds an ``mcp.client.auth.OAuthClientProvider`` -- the officially
maintained MCP Python SDK's own OAuth client, not a hand-rolled one -- wired
up with local loopback redirect handling and on-disk token persistence, so
``RobinhoodMCPBroker(auth=build_robinhood_oauth())`` can authenticate
without a static bearer token.

Why this discovers endpoints instead of hardcoding them: ``RobinhoodMCPBroker``'s
own docstring already flags that its auth handshake was never verified
against the live service. Third-party write-ups describing a specific token
endpoint for this (typically ``api.robinhood.com/oauth2/token/``) turn out,
on inspection, to cite sources that don't actually support that claim -- an
unrelated GitHub issue, a generic Auth0 PKCE tutorial, YouTube videos -- and
that exact endpoint is publicly documented elsewhere as belonging to the
older, unofficial, non-agentic Robinhood API (a resource-owner-password
flow, not PKCE, not this product). Rather than hardcode a guessed endpoint
into code that authenticates a real brokerage account, this uses the MCP
SDK's own support for RFC 8414 (authorization server metadata discovery) and
RFC 7591 (dynamic client registration): everything is derived from the MCP
resource URL itself by asking the server what its endpoints and
registration process actually are.

**Still unverified**: whether ``agent.robinhood.com/mcp/trading`` actually
implements RFC 8414 discovery and RFC 7591 registration at all -- that's
what running this for the first time actually tests. If it doesn't, the SDK
will raise a clear discovery/registration error rather than silently
succeeding against a wrong endpoint.

**Deployment note**: the callback listener binds to ``127.0.0.1`` on the
machine running this process. The first login (and any later re-login, if
the refresh token itself expires or is revoked) needs a browser that can
reach that loopback address -- fine on a laptop, not fine on a truly
headless remote box without an SSH tunnel forwarding the callback port.
Every other run reuses the stored refresh token silently, no browser
needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

__all__ = ["FileTokenStorage", "build_robinhood_oauth", "ROBINHOOD_MCP_URL"]

ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------


class FileTokenStorage(TokenStorage):
    """Persists OAuth client registration + tokens to one local JSON file.

    Implements the ``TokenStorage`` protocol from ``mcp.client.auth.oauth2``.
    The file holds a live refresh token -- exactly as sensitive as a
    password for as long as it's valid -- so it's written owner-only
    (``0600``) via a temp-file-then-rename, never world- or group-readable
    even transiently.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        # Confirmed live (2026-08): write-then-chmod leaves the file at
        # whatever the OS default creation mode is (typically 0o644 under
        # a standard umask -- world- and group-readable) for the entire
        # duration of the write, directly contradicting this class's own
        # docstring claim of "never world- or group-readable even
        # transiently." os.open() with an explicit mode applies owner-only
        # permissions atomically at creation, before the live refresh
        # token (as sensitive as a password) is ever written to disk.
        mode = stat.S_IRUSR | stat.S_IWUSR
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data))
        tmp.replace(self.path)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = json.loads(tokens.model_dump_json())
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = json.loads(client_info.model_dump_json())
        self._write(data)


# ---------------------------------------------------------------------------
# Loopback redirect capture
# ---------------------------------------------------------------------------


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures exactly one OAuth redirect GET to the registered callback
    path, then the server stops.

    Any other request -- a stray local probe, a browser doing something
    unrelated while the loopback port happens to be listening -- gets a
    404 and does not consume the one-shot slot. The original version
    accepted the *first* GET regardless of path: a wrong request arriving
    before the real redirect would eat that one `handle_request()` call,
    and the genuine callback would then hit a closed port. Not a
    CSRF/token-injection risk either way -- the `state` parameter is
    validated by the MCP SDK itself, independent of this handler -- this
    is a reliability gap in the happy path, not a security one.
    """

    def do_GET(self) -> None:  # noqa: N802 -- fixed name from BaseHTTPRequestHandler
        if urlparse(self.path).path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(urlparse(self.path).query)
        self.server.oauth_result = (  # type: ignore[attr-defined]
            query.get("code", [None])[0],
            query.get("state", [None])[0],
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h3>Robinhood authorization complete.</h3>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # don't spam stderr with one-line-per-request access logs


def _make_handlers(
    port: int,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]]:
    """Build the ``redirect_handler``/``callback_handler`` pair the SDK's
    ``OAuthClientProvider`` expects: one to send the user to log in, one to
    receive the authorization code afterward.
    """

    async def redirect_handler(authorization_url: str) -> None:
        print(f"Opening browser for Robinhood authorization:\n  {authorization_url}")
        print("(if a browser doesn't open automatically, or this is a remote "
              "machine, open that URL yourself)")
        webbrowser.open(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        server.timeout = 300  # abandon the login attempt after 5 minutes idle
        loop = asyncio.get_event_loop()
        # A stray request to the wrong path no longer consumes the
        # one-shot handle_request() call (see _CallbackHandler), which
        # means more than one call may be needed to reach the real
        # callback -- loop against the same overall 5-minute budget
        # instead of a single call, so a stray hit can't either eat the
        # real callback's turn or extend the wait past what was intended.
        deadline = loop.time() + 300
        while getattr(server, "oauth_result", None) is None and loop.time() < deadline:
            await loop.run_in_executor(None, server.handle_request)
        code, state = getattr(server, "oauth_result", (None, None))
        server.server_close()
        if not code:
            raise RuntimeError(
                "no authorization code received on the loopback callback -- "
                "the browser flow was cancelled, timed out, or never reached "
                f"http://127.0.0.1:{port}/callback"
            )
        return code, state

    return redirect_handler, callback_handler


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_robinhood_oauth(
    storage_path: str = "state/robinhood_oauth.json",
    port: int = 8765,
    scope: str | None = None,
) -> OAuthClientProvider:
    """Build the OAuth PKCE auth provider :class:`RobinhoodMCPBroker` expects.

    Pass the result as ``RobinhoodMCPBroker(auth=build_robinhood_oauth())``
    instead of a static ``token=``. On first use this opens a browser for
    you to log into Robinhood and authorize the connection once; the
    resulting tokens (and refresh token) persist to ``storage_path``, so
    later runs refresh silently -- until the refresh token itself expires
    or is revoked, at which point the browser flow fires again.

    See the module docstring for why endpoints are discovered rather than
    hardcoded, and for the headless-deployment caveat on the loopback
    callback.
    """
    redirect_handler, callback_handler = _make_handlers(port)
    metadata = OAuthClientMetadata(
        redirect_uris=[f"http://127.0.0.1:{port}/callback"],
        token_endpoint_auth_method="none",  # public client: PKCE, no secret
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name="qbt-run-cycle",
        scope=scope,
    )
    return OAuthClientProvider(
        server_url=ROBINHOOD_MCP_URL,
        client_metadata=metadata,
        storage=FileTokenStorage(storage_path),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
