"""terradev login — browser-based token handshake for Telinea.

Spins up a temporary local callback server, opens the user's default browser
to terradev.cloud/auth/cli, and writes the returned token to the encrypted
local vault under ``telinea/api_key``.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import click

from . import cli
from terradev_cli.core.output import get_output
from terradev_cli.core.vault_adapter import VaultAdapter

logger = logging.getLogger(__name__)


class _CallbackHandler(BaseHTTPRequestHandler):
    """Minimal handler that captures the token from the browser redirect."""

    token: Optional[str] = None
    error: Optional[str] = None
    received = threading.Event()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if "token" in query:
            _CallbackHandler.token = query["token"][0]
            _CallbackHandler.error = None
            _CallbackHandler.received.set()
        elif "error" in query:
            _CallbackHandler.error = query["error"][0]
            _CallbackHandler.received.set()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if _CallbackHandler.token:
            body = (
                "<h1>Terradev CLI authenticated</h1>"
                "<p>You can close this tab and return to the terminal.</p>"
            )
        elif _CallbackHandler.error:
            body = (
                f"<h1>Authentication failed</h1>"
                f"<p>{_CallbackHandler.error}</p>"
            )
        else:
            body = (
                "<h1>Waiting for Terradev CLI...</h1>"
                "<p>If you see this page, the CLI may not have captured the "
                "callback correctly. Check your terminal.</p>"
            )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass


@cli.command("login")
@click.option(
    "--port",
    type=int,
    default=0,
    help="Local callback port (0 = ephemeral)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Print the URL instead of opening a browser",
)
@click.option(
    "--endpoint",
    default="https://terradev.cloud",
    help="Telinea cloud auth endpoint",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Seconds to wait for the browser callback",
)
def login(port: int, no_browser: bool, endpoint: str, timeout: int):
    """Log in to Telinea via a browser-based token exchange.

    The command starts a temporary local HTTP server, opens your default
    browser to the Terradev cloud auth page, and writes the returned API key
    to the encrypted local vault.
    """
    output = get_output()

    state = secrets.token_urlsafe(16)
    _CallbackHandler.token = None
    _CallbackHandler.error = None
    _CallbackHandler.received.clear()

    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    actual_port = server.server_address[1]
    callback_url = f"http://127.0.0.1:{actual_port}/callback"

    auth_url = (
        f"{endpoint.rstrip('/')}/auth/cli"
        f"?callback={callback_url}"
        f"&state={state}"
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        if no_browser:
            output.info(f"Open this URL in your browser:\n  {auth_url}")
        else:
            output.info(f"Opening browser for Telinea login: {auth_url}")
            try:
                webbrowser.open(auth_url)
            except Exception:  # noqa: BLE001
                output.warning("Could not open browser. Visit the URL above.")

        output.info(f"Waiting up to {timeout}s for the browser callback...")
        received = _CallbackHandler.received.wait(timeout=timeout)

        if not received:
            output.error("Timed out waiting for browser authentication")
            output.set_result({"status": "timeout"})
            return

        if _CallbackHandler.error:
            output.error(f"Authentication failed: {_CallbackHandler.error}")
            output.set_result({"status": "failed", "error": _CallbackHandler.error})
            return

        if not _CallbackHandler.token:
            output.error("No token received from browser")
            output.set_result({"status": "failed", "error": "missing token"})
            return

        # Persist to vault
        vault = VaultAdapter(Path.home() / ".terradev")
        vault.set("telinea", "api_key", _CallbackHandler.token)

        masked = _CallbackHandler.token[:8] + "•" * 8
        output.success(f"Telinea API key saved ({masked})")
        output.set_result({"status": "authenticated", "masked_key": masked})

    finally:
        server.shutdown()
        server.server_close()
