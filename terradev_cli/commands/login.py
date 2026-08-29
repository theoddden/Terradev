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

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        cb = self.server.callback_state
        expected_state = self.server.expected_state

        state = query.get("state", [None])[0]
        if state != expected_state:
            cb["error"] = "Invalid or missing state parameter"
            cb["received"].set()
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                "<h1>Invalid callback state</h1>"
                "<p>Close this tab and try again.</p>".encode("utf-8")
            )
            return

        if "token" in query:
            cb["token"] = query["token"][0]
            cb["error"] = None
            cb["received"].set()
        elif "error" in query:
            cb["error"] = query["error"][0]
            cb["received"].set()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if cb["token"]:
            body = (
                "<h1>Terradev CLI authenticated</h1>"
                "<p>You can close this tab and return to the terminal.</p>"
            )
        elif cb["error"]:
            body = (
                "<h1>Authentication failed</h1>"
                f"<p>{cb['error']}</p>"
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
    type=click.IntRange(0, 65535),
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
    type=click.IntRange(1, 3600),
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
    callback_state = {"token": None, "error": None, "received": threading.Event()}

    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.expected_state = state
    server.callback_state = callback_state
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
            raise SystemExit(1)

        if callback_state["error"]:
            output.error(f"Authentication failed: {callback_state['error']}")
            output.set_result({"status": "failed", "error": callback_state["error"]})
            raise SystemExit(1)

        if not callback_state["token"]:
            output.error("No token received from browser")
            output.set_result({"status": "failed", "error": "missing token"})
            raise SystemExit(1)

        # Persist to vault
        vault = VaultAdapter(Path.home() / ".terradev")
        try:
            vault.set("telinea", "api_key", callback_state["token"])
        except Exception as exc:  # noqa: BLE001
            output.error(f"Could not save API key to vault: {exc}")
            output.set_result({"status": "failed", "error": str(exc)})
            raise SystemExit(1)

        masked = callback_state["token"][:8] + "•" * 8
        output.success(f"Telinea API key saved ({masked})")
        output.set_result({"status": "authenticated", "masked_key": masked})

    finally:
        server.shutdown()
        server.server_close()
