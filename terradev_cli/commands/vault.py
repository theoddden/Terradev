#!/usr/bin/env python3
"""Vault command for secure, CI/CD-friendly credential management."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import click

from . import cli
from terradev_cli.core.output import get_output
from terradev_cli.core.vault_adapter import (
    VaultAdapter,
    read_secret_from_stdin,
)


def _get_vault() -> VaultAdapter:
    """Return a VaultAdapter for the default config directory."""
    return VaultAdapter(Path.home() / ".terradev")


def _secret_value(
    value: Optional[str],
    from_env: Optional[str],
    from_stdin: bool,
) -> Tuple[str, str]:
    """Resolve a secret from explicit value, env var, or stdin."""
    sources = [bool(value), bool(from_env), from_stdin]
    if sum(sources) > 1:
        raise click.ClickException(
            "Use only one of --value, --from-env, or --from-stdin"
        )

    if from_env:
        if from_env not in os.environ:
            raise click.ClickException(f"Environment variable {from_env} is not set")
        return from_env, os.environ[from_env]

    if from_stdin:
        return "<stdin>", read_secret_from_stdin()

    if value:
        return "<value>", value

    raise click.ClickException(
        "One of --value, --from-env, or --from-stdin is required"
    )


@cli.group("vault")
def vault():
    """Secure secret storage for CI/CD pipelines and local development."""
    pass


@vault.command("set")
@click.argument("provider")
@click.argument("key")
@click.option("--value", help="Secret value (not recommended for shell history)")
@click.option("--from-env", help="Read the secret from an environment variable")
@click.option(
    "--from-stdin",
    is_flag=True,
    help="Read the secret from stdin (best for CI and scripts)",
)
@click.option(
    "--no-persist",
    is_flag=True,
    help="Do not write to disk; keeps the secret in env/session only",
)
def vault_set(
    provider: str,
    key: str,
    value: Optional[str],
    from_env: Optional[str],
    from_stdin: bool,
    no_persist: bool,
):
    """Store a secret for a provider.

    Examples:
      terradev vault set runpod api_key --from-env RUNPOD_API_KEY
      terradev vault set aws secret_key --from-stdin
      cat key.txt | terradev vault set vastai api_key --from-stdin
      terradev vault set runpod api_key --value rpa_xxx --no-persist
    """
    output = get_output()
    source, secret = _secret_value(value, from_env, from_stdin)

    if no_persist:
        # In no-persist mode we honor the env but skip file writes.
        os.environ[f"TERRADEV_{provider.upper()}_{key.upper()}"] = secret
        output.success(
            f"{provider}.{key} set in environment (not persisted to disk)"
        )
        output.set_result({"provider": provider, "key": key, "source": source})
        return

    vault = _get_vault()
    vault.set(provider, key, secret)

    # Zeroize the local variable as best as pure Python can.
    del secret

    output.success(f"{provider}.{key} stored")
    output.set_result({"provider": provider, "key": key, "source": source})


@vault.command("sync")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be imported without persisting",
)
@click.option(
    "--no-persist",
    is_flag=True,
    help="Keep imported secrets in env only; do not write the vault file",
)
@click.option(
    "--all",
    "include_all",
    is_flag=True,
    help="Import every TERRADEV_* variable, not just supported cloud providers",
)
def vault_sync(dry_run: bool, no_persist: bool, include_all: bool):
    """Import TERRADEV_* environment variables for supported cloud providers.

    This is the recommended command for CI/CD pipelines using GitHub Secrets:

      env:
        TERRADEV_RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
        TERRADEV_AWS_SECRET_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: terradev vault sync

    Use --dry-run in a workflow to verify mapping before a real run.
    Use --all to also import non-provider/custom TERRADEV_* variables.
    """
    output = get_output()
    v = _get_vault()
    env_creds = v.load_env_credentials(known_only=not include_all)

    imported = []
    for provider, creds in env_creds.items():
        for key, value in creds.items():
            imported.append({"provider": provider, "key": key})

    if dry_run:
        output.info(f"Would import {len(imported)} secret(s)")
        output.set_result({"dry_run": True, "imported": imported})
        return

    if no_persist:
        output.info(f"Loaded {len(imported)} secret(s) from environment (no persistence)")
        output.set_result({"no_persist": True, "imported": imported})
        return

    if not imported:
        output.warning("No TERRADEV_* environment variables found")
        output.set_result({"imported": []})
        return

    base = v.load_credentials()
    merged = v.load_env_credentials(base, known_only=not include_all)
    v.save_credentials(merged)

    output.success(f"Synced {len(imported)} secret(s)")
    output.set_result({"imported": imported})


@vault.command("get")
@click.argument("provider")
@click.argument("key")
@click.option(
    "--raw",
    is_flag=True,
    help="Print the raw secret (disabled in non-TTY / CI by default)",
)
def vault_get(provider: str, key: str, raw: bool):
    """Retrieve a secret. By default the value is masked."""
    output = get_output()
    v = _get_vault()
    value = v.get(provider, key)

    if value is None:
        output.error(f"No secret found for {provider}.{key}")
        output.set_result({"found": False})
        raise SystemExit(1)

    if raw:
        if output.format == "human":
            output._write_raw(value + "\n")
        output.set_result({"provider": provider, "key": key, "value": value})
    else:
        masked = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
        output.info(f"{provider}.{key}: {masked}")
        output.set_result(
            {"provider": provider, "key": key, "value": masked, "masked": True}
        )


@vault.command("list")
def vault_list():
    """List stored provider and key names (values are never shown)."""
    output = get_output()
    v = _get_vault()
    creds = v.load_credentials()

    providers = sorted(creds.keys())
    listing = {}
    for provider in providers:
        listing[provider] = sorted(creds[provider].keys())

    if not providers:
        output.info("No secrets stored")
    else:
        for provider, keys in listing.items():
            output.info(f"{provider}: {', '.join(keys)}")

    output.set_result({"providers": listing})


@vault.command("remove")
@click.argument("provider")
@click.argument("key", required=False)
def vault_remove(provider: str, key: Optional[str]):
    """Remove a provider or a single key from the vault."""
    output = get_output()
    v = _get_vault()
    removed = v.remove(provider, key)

    if not removed:
        output.error(f"No secret found for {provider}{f'.{key}' if key else ''}")
        output.set_result({"removed": False})
        raise SystemExit(1)

    output.success(f"Removed {provider}{f'.{key}' if key else ''}")
    output.set_result({"removed": True, "provider": provider, "key": key})


@vault.command("verify")
def vault_verify():
    """Check which providers are fully configured and which keys are missing."""
    output = get_output()
    v = _get_vault()
    status = v.verify()

    for provider in status["configured"]:
        output.success(f"{provider}: configured")

    for provider, missing in status["missing"].items():
        output.warning(f"{provider}: missing {', '.join(missing)}")

    output.set_result(status)


@vault.command("env")
@click.argument("provider")
@click.option(
    "--raw",
    is_flag=True,
    help="Print raw values as shell export statements",
)
def vault_env(provider: str, raw: bool):
    """Print environment-style export lines for a provider."""
    output = get_output()
    v = _get_vault()
    env = v.to_env(provider)

    if not env:
        output.warning(f"No credentials for {provider}")
        output.set_result({"exports": {}})
        return

    if not raw:
        masked = {k: (v[:4] + "***" + v[-4:]) for k, v in env.items()}
        output.info("Use --raw to expose values in shell export format")
        output.set_result({"exports": masked, "masked": True})
        return

    for name, value in env.items():
        # shlex.quote is safe, but the value itself is secret.
        line = f'export {name}={shlex.quote(value)}'
        if output.format == "human":
            output._write_raw(line + "\n")

    output.set_result({"exports": env})


@vault.command("run")
@click.argument("command", nargs=-1, required=True)
@click.option("--provider", "-p", help="Only inject secrets for this provider")
@click.option(
    "--no-exec",
    is_flag=True,
    help="Build the env and print export lines without running",
)
def vault_run(command: Tuple[str, ...], provider: Optional[str], no_exec: bool):
    """Run a shell command with vault secrets injected into the environment.

    Examples:
      terradev vault run -- terradev up --job train
      terradev vault run --provider runpod -- python train.py
    """
    output = get_output()
    v = _get_vault()

    if no_exec:
        env = v.build_run_env(provider)
        for name, value in env.items():
            if name.startswith(v.ENV_PREFIX):
                if output.format == "human":
                    output._write_raw(f'export {name}={shlex.quote(value)}\n')
        output.set_result({"exports": env})
        return

    run_env = v.build_run_env(provider)

    # Copy current env and merge vault secrets, then clear the local copy.
    proc_env = dict(os.environ)
    proc_env.update(run_env)
    del run_env

    try:
        proc = subprocess.run(
            list(command),
            env=proc_env,
            stdout=None,
            stderr=None,
            stdin=None,
        )
    except FileNotFoundError as exc:
        output.error(f"Command not found: {command[0]} ({exc})")
        output.set_result({"exit_code": 127, "command": list(command)})
        return
    except Exception as exc:  # noqa: BLE001
        output.error(f"Failed to run command: {exc}")
        output.set_result({"exit_code": 1, "command": list(command)})
        return
    finally:
        # Best-effort zeroization of the merged env.
        for name in list(proc_env.keys()):
            if name.startswith(v.ENV_PREFIX):
                proc_env[name] = ""
        del proc_env

    output.set_result({"exit_code": proc.returncode, "command": list(command)})
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
