"""terradev telinea — observability connector commands."""

from __future__ import annotations

import os

import click

from . import cli
from terradev_cli.core.output import get_output
from terradev_cli.core.telinea import (
    TelineaClient,
    TelineaConfig,
    resolve_telinea_api_key,
)
from terradev_cli.core.vault_adapter import VaultAdapter


@cli.group("telinea")
def telinea():
    """Telinea observability dashboard and telemetry."""
    pass


@telinea.command("status")
def telinea_status():
    """Show Telinea connection status and resolved configuration."""
    output = get_output()
    config = TelineaConfig()
    output.info(f"Telinea enabled: {config.is_configured}")
    output.info(f"Endpoint: {config.base_url}")
    output.info(f"Project ID: {config.project_id or '(not set)'}")
    output.info(f"Workspace ID: {config.workspace_id or '(not set)'}")
    masked = ""
    if config.api_key:
        masked = config.api_key[:8] + "•" * 8
    output.info(f"API key: {masked or '(not set)'}")
    output.set_result(
        {
            "enabled": config.is_configured,
            "endpoint": config.base_url,
            "project_id": config.project_id,
            "workspace_id": config.workspace_id,
            "has_api_key": bool(config.api_key),
        }
    )


@telinea.command("set-key")
@click.option("--value", help="API key value (not recommended in shell history)")
@click.option("--from-env", help="Read the API key from an environment variable")
@click.option("--from-stdin", is_flag=True, help="Read the API key from stdin")
def set_key(value: str, from_env: str, from_stdin: bool):
    """Store the Telinea API key in the encrypted local vault."""
    output = get_output()
    source = ""
    secret = ""

    if from_env:
        if from_env not in os.environ:
            raise click.ClickException(f"Environment variable {from_env} is not set")
        source = f"env:{from_env}"
        secret = os.environ[from_env]
    elif from_stdin:
        source = "<stdin>"
        secret = (sys.stdin.buffer.read().decode("utf-8")).strip()
    elif value:
        source = "<value>"
        secret = value
    else:
        raise click.ClickException(
            "One of --value, --from-env, or --from-stdin is required"
        )

    vault = VaultAdapter()
    vault.set("telinea", "api_key", secret)
    del secret

    output.success("Telinea API key stored")
    output.set_result({"source": source, "provider": "telinea", "key": "api_key"})


@telinea.command("flush")
def flush():
    """Flush any queued Telinea telemetry synchronously."""
    output = get_output()
    client = TelineaClient(TelineaConfig())
    client.flush()
    output.success("Telinea telemetry flushed")
    output.set_result({"status": "flushed"})


@telinea.command("test")
def test_connection():
    """Test the Telinea API key by sending a small heartbeat payload."""
    output = get_output()
    config = TelineaConfig()
    if not config.is_configured:
        output.error("Telinea is not configured. Set TELINEA_API_KEY or run 'terradev telinea set-key'.")
        output.set_result({"status": "not_configured"})
        return

    try:
        import requests
        response = requests.get(
            f"{config.base_url}/v1/health",
            headers={"Authorization": config.auth_header},
            timeout=(config.connect_timeout, config.read_timeout),
        )
        if response.status_code == 200:
            output.success("Telinea API key is valid")
            output.set_result({"status": "ok", "http_status": response.status_code})
        else:
            output.error(f"Telinea API returned {response.status_code}")
            output.set_result({"status": "error", "http_status": response.status_code})
    except Exception as exc:  # noqa: BLE001
        output.error(f"Telinea connection test failed: {exc}")
        output.set_result({"status": "error", "error": str(exc)})


@telinea.command("login")
@click.pass_context
def telinea_login(ctx):
    """Alias for 'terradev login'."""
    from .login import login
    ctx.forward(login)
