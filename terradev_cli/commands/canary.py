#!/usr/bin/env python3
"""Canary reporting commands for the Terradev CLI.

Reads canary result telemetry from ~/.terradev/canary-results.jsonl
and produces a human-readable or JSON summary.
"""

import base64
import binascii
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import yaml

from . import cli
from terradev_cli.core.output import get_output
from terradev_cli.drift_monitor.agent import DriftMonitor


def _default_canary_output() -> Path:
    """Default canary results file path."""
    return Path.home() / ".terradev" / "canary-results.jsonl"


def _parse_canary_record(line: str) -> Dict[str, Any]:
    """Parse a single JSONL record, returning an empty dict on error."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def _load_canary_records(path: Path) -> List[Dict[str, Any]]:
    """Load all readable canary records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _parse_canary_record(line)
            if rec:
                records.append(rec)
    return records


def _summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for canary records."""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "providers": {},
            "regions": {},
            "gpu_types": {},
            "tpu_types": {},
            "tpu_chips": {},
            "first_seen": None,
            "last_seen": None,
            "duration_ms": {"min": None, "max": None, "avg": None},
        }

    passed = sum(1 for r in records if r.get("status") == "passed")
    failed = sum(1 for r in records if r.get("status") == "failed")
    skipped = sum(1 for r in records if r.get("status") == "skipped")

    providers: Dict[str, int] = {}
    regions: Dict[str, int] = {}
    gpu_types: Dict[str, int] = {}
    tpu_types: Dict[str, int] = {}
    tpu_chips: Dict[str, int] = {}
    durations = []
    timestamps = []

    for r in records:
        providers[r.get("provider", "unknown")] = providers.get(r.get("provider", "unknown"), 0) + 1
        regions[r.get("region", "unknown")] = regions.get(r.get("region", "unknown"), 0) + 1
        gpu_types[r.get("gpu_type", "unknown")] = gpu_types.get(r.get("gpu_type", "unknown"), 0) + 1
        if r.get("tpu_type"):
            tpu_types[r["tpu_type"]] = tpu_types.get(r["tpu_type"], 0) + 1
        if r.get("tpu_chips"):
            key = f"{r['tpu_chips']} chips"
            tpu_chips[key] = tpu_chips.get(key, 0) + 1

        if "duration_ms" in r and isinstance(r["duration_ms"], (int, float)):
            durations.append(r["duration_ms"])

        ts = r.get("timestamp") or r.get("recorded_at")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(str(ts)))
            except ValueError:
                pass

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "providers": providers,
        "regions": regions,
        "gpu_types": gpu_types,
        "tpu_types": tpu_types,
        "tpu_chips": tpu_chips,
        "first_seen": min(timestamps).isoformat() if timestamps else None,
        "last_seen": max(timestamps).isoformat() if timestamps else None,
        "duration_ms": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "avg": sum(durations) / len(durations) if durations else None,
        },
    }


@cli.group("canary")
def canary():
    """Run and report canary health checks."""
    pass


@canary.command("report")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=False),
    default=None,
    help="Path to canary results JSONL file",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Filter results by provider",
)
@click.option(
    "--gpu",
    "-g",
    default=None,
    help="Filter results by GPU type",
)
def canary_report(output, file, provider, gpu):
    """Show a summary of recent canary test results."""
    path = Path(file) if file else _default_canary_output()
    records = _load_canary_records(path)

    if provider:
        records = [r for r in records if r.get("provider") == provider]
    if gpu:
        records = [r for r in records if r.get("gpu_type") == gpu]

    summary = _summarize_records(records)

    if output == "json":
        # Canary already produces stable JSON; suppress the global wrapper and
        # emit the raw canary report so existing scripts/pipes keep working.
        out = get_output()
        if out is not None:
            out._closed = True
        click.echo(json.dumps(summary, indent=2, default=str))
        return

    click.echo("=" * 60)
    click.echo("TERRADEV CANARY REPORT".center(60))
    click.echo("=" * 60)

    if not records:
        click.echo("No canary records found.")
        click.echo(f"Run canary tests first or point to a results file with --file.")
        return

    click.echo(f"Total runs:     {summary['total']}")
    click.echo(f"Passed:         {summary['passed']}")
    click.echo(f"Failed:         {summary['failed']}")
    click.echo(f"Skipped:        {summary['skipped']}")
    if summary["first_seen"]:
        click.echo(f"First seen:     {summary['first_seen']}")
    if summary["last_seen"]:
        click.echo(f"Last seen:      {summary['last_seen']}")

    dur = summary["duration_ms"]
    if dur["avg"] is not None:
        click.echo(f"Duration (ms):  min={dur['min']}, max={dur['max']}, avg={dur['avg']:.1f}")

    click.echo("-" * 60)
    click.echo("By provider:")
    for p, count in sorted(summary["providers"].items(), key=lambda x: -x[1]):
        click.echo(f"  {p:<20} {count}")

    if summary["regions"]:
        click.echo("-" * 60)
        click.echo("By region:")
        for r, count in sorted(summary["regions"].items(), key=lambda x: -x[1]):
            click.echo(f"  {r:<20} {count}")

    if summary["gpu_types"]:
        click.echo("-" * 60)
        click.echo("By GPU type:")
        for g, count in sorted(summary["gpu_types"].items(), key=lambda x: -x[1]):
            click.echo(f"  {g:<20} {count}")

    if summary["tpu_types"]:
        click.echo("-" * 60)
        click.echo("By TPU type:")
        for t, count in sorted(summary["tpu_types"].items(), key=lambda x: -x[1]):
            click.echo(f"  {t:<20} {count}")

    if summary["tpu_chips"]:
        click.echo("-" * 60)
        click.echo("By TPU chip count:")
        for t, count in sorted(summary["tpu_chips"].items(), key=lambda x: -x[1]):
            click.echo(f"  {t:<20} {count}")

    click.echo("=" * 60)


@canary.command("tail")
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=False),
    default=None,
    help="Path to canary results JSONL file",
)
@click.option(
    "--limit",
    "-n",
    default=10,
    type=click.IntRange(1, 1000),
    help="Number of recent records to show",
)
def canary_tail(file, limit):
    """Show the most recent canary records."""
    path = Path(file) if file else _default_canary_output()
    records = _load_canary_records(path)
    recent = records[-limit:]
    if not recent:
        click.echo("No canary records found.")
        return
    for rec in recent:
        click.echo(json.dumps(rec, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════════
# Provider API Drift Monitor
# ═══════════════════════════════════════════════════════════════════════════════


_DRIFT_PROVIDER_ALIASES = {
    "aws": ["aws"],
    "gcp": ["gcp"],
    "azure": ["azure"],
    "runpod": ["runpod"],
    "vastai": ["vastai", "vast"],
    "tensordock": ["tensordock"],
    "huggingface": ["huggingface", "hf"],
    "baseten": ["baseten", "basenten"],
    "crusoe": ["crusoe"],
    "hyperstack": ["hyperstack"],
    "digitalocean": ["digitalocean", "digital_ocean"],
    "siliconflow": ["siliconflow"],
    "inferx": ["inferx"],
    "latitude": ["latitude"],
    "yottalabs": ["yottalabs", "yotta"],
    "e2enetworks": ["e2enetworks", "e2e"],
    "gcore": ["gcore"],
}

# Provider-specific primary env names (full names, not alias-derived).
_DRIFT_PROVIDER_KEY_ENVS: Dict[str, List[str]] = {
    "aws": ["TERRADEV_AWS_ACCESS_KEY_ID"],
    "gcp": ["TERRADEV_GCP_CREDENTIALS"],
    "inferx": ["TERRADEV_INFERX_API_KEY"],
    "langfuse": ["TERRADEV_LANGFUSE_PUBLIC_KEY"],
    "letta": ["TERRADEV_LETTA_API_KEY"],
    "weaviate": ["TERRADEV_WEAVIATE_API_KEY"],
}


def _default_contracts_dir() -> Path:
    """Return the bundled provider contract directory."""
    return Path(__file__).resolve().parent.parent / "providers" / "contracts"


def _drift_provider_aliases(provider: str) -> List[str]:
    return _DRIFT_PROVIDER_ALIASES.get(provider, [provider])


def _load_drift_env_key(provider: str) -> Optional[Union[str, Dict[str, Any]]]:
    """Look for a raw API key in the environment."""
    specific = _DRIFT_PROVIDER_KEY_ENVS.get(provider, [])
    for alias in _drift_provider_aliases(provider):
        name = alias.upper().replace("-", "_")
        envs = list(specific) + [
            f"TERRADEV_{name}_KEY",
            f"TERRADEV_{name}_API_KEY",
            f"CANARY_{name}_KEY",
        ]
        for env in envs:
            value = os.environ.get(env)
            if value:
                extras = _load_drift_env_extras(alias, value)
                return extras if extras else value

    # Azure can also be configured via a service principal without a dedicated
    # bearer token / key. If the four service-principal values are present,
    # build the credential dict directly from them.
    if provider == "azure":
        subscription_id = (os.environ.get("TERRADEV_AZURE_SUBSCRIPTION_ID") or "").strip()
        tenant_id = (os.environ.get("TERRADEV_AZURE_TENANT_ID") or "").strip()
        client_id = (os.environ.get("TERRADEV_AZURE_CLIENT_ID") or "").strip()
        client_secret = (os.environ.get("TERRADEV_AZURE_CLIENT_SECRET") or "").strip()
        if subscription_id and tenant_id and client_id and client_secret:
            return _load_drift_env_extras("azure", "")

    return None


def _load_drift_env_extras(alias: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Look for provider extras (project_id, location) and build a credential dict."""
    name = alias.upper().replace("-", "_")
    project_id = os.environ.get(f"TERRADEV_{name}_PROJECT_ID") or ""
    location = os.environ.get(f"TERRADEV_{name}_LOCATION") or "Delhi"
    extras: Dict[str, Any] = {"api_key": api_key}

    # E2E Networks uses two distinct values: the apikey query param (API key)
    # and the Authorization: Bearer header (Auth Token / Bearer token).
    if alias == "e2enetworks":
        query_api_key = os.environ.get(f"TERRADEV_{name}_API_KEY") or ""
        bearer_token = (
            os.environ.get(f"TERRADEV_{name}_BEARER_TOKEN")
            or os.environ.get(f"TERRADEV_{name}_AUTH_TOKEN")
            or os.environ.get(f"TERRADEV_{name}_TOKEN")
            or ""
        )
        if query_api_key and bearer_token:
            extras["api_key"] = query_api_key
            extras["bearer_token"] = bearer_token
        elif query_api_key and not bearer_token:
            # The passed-in api_key is likely the bearer token; the API_KEY is the query key.
            extras["api_key"] = query_api_key
            extras["bearer_token"] = api_key
        elif bearer_token and not query_api_key:
            # The passed-in api_key is likely the query API key.
            extras["bearer_token"] = bearer_token
            extras["api_key"] = api_key

    elif alias == "aws":
        extras["aws_access_key_id"] = api_key.strip()
        extras["aws_secret_access_key"] = (
            os.environ.get("TERRADEV_AWS_SECRET_ACCESS_KEY", "") or ""
        ).strip()
        extras["aws_region"] = (
            os.environ.get("TERRADEV_AWS_REGION") or "us-east-1"
        ).strip()
        extras["aws_session_token"] = (
            os.environ.get("TERRADEV_AWS_SESSION_TOKEN", "") or ""
        ).strip()

    elif alias == "gcp":
        raw = api_key.strip()
        gcp_creds = None
        # Accept a plain JSON credential blob.
        if raw.startswith("{"):
            try:
                gcp_creds = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                gcp_creds = None
        # Otherwise try base64-encoded JSON.
        if gcp_creds is None:
            try:
                decoded = base64.b64decode(raw).decode("utf-8")
                gcp_creds = json.loads(decoded)
            except (ValueError, json.JSONDecodeError, binascii.Error):
                gcp_creds = None
        if gcp_creds:
            extras["gcp_credentials"] = gcp_creds
            extras["project_id"] = gcp_creds.get("project_id") or ""
        else:
            extras["gcp_credentials"] = raw

    elif alias == "azure":
        # Azure drift can use either a bearer token or a service principal.
        subscription_id = (
            os.environ.get("TERRADEV_AZURE_SUBSCRIPTION_ID")
            or os.environ.get("TERRADEV_AZURE_PROJECT_ID")
            or ""
        ).strip()
        project_id = subscription_id
        location = os.environ.get("TERRADEV_AZURE_LOCATION") or "eastus"
        extras["subscription_id"] = subscription_id
        extras["tenant_id"] = (os.environ.get("TERRADEV_AZURE_TENANT_ID") or "").strip()
        extras["client_id"] = (os.environ.get("TERRADEV_AZURE_CLIENT_ID") or "").strip()
        client_secret = (
            os.environ.get("TERRADEV_AZURE_CLIENT_SECRET") or ""
        ).strip()
        extras["client_secret"] = client_secret
        # Prefer a dedicated bearer token; fall back to the legacy key only when
        # it is not just the client secret being reused as the key placeholder.
        bearer_token = (
            os.environ.get("TERRADEV_AZURE_BEARER_TOKEN")
            or ("" if api_key == client_secret else api_key)
            or ""
        ).strip()
        extras["bearer_token"] = bearer_token

    elif alias == "inferx":
        extras["api_key"] = api_key.strip()
        extras["bearer_token"] = api_key.strip()
        extras["region"] = os.environ.get("TERRADEV_INFERX_REGION") or "us-west-2"
        extras["api_endpoint"] = (
            os.environ.get("TERRADEV_INFERX_API_ENDPOINT")
            or "https://model.inferx.net/endpoints/v1"
        ).strip()
        extras["model"] = os.environ.get("TERRADEV_INFERX_MODEL") or "Qwen3.8-27B-FP8"

    elif alias == "langfuse":
        public_key = api_key.strip()
        secret_key = (os.environ.get(f"TERRADEV_{name}_SECRET_KEY") or "").strip()
        if public_key and secret_key:
            extras["bearer_token"] = f"{public_key}:{secret_key}"
        extras["public_key"] = public_key
        extras["secret_key"] = secret_key
        extras["api_key"] = public_key

    elif alias == "letta":
        extras["api_key"] = api_key.strip()
        extras["bearer_token"] = api_key.strip()

    elif alias == "weaviate":
        extras["api_key"] = api_key.strip()
        extras["bearer_token"] = api_key.strip()

    elif alias == "gcore":
        extras["api_key"] = api_key.strip()
        # Project and region are optional; Gcore can discover them via API.
        # Add TERRADEV_GCORE_PROJECT_ID / TERRADEV_GCORE_REGION_ID to skip discovery.
        extras["project_id"] = (
            os.environ.get(f"TERRADEV_{name}_PROJECT_ID") or project_id or ""
        ).strip()
        extras["region_id"] = (
            os.environ.get(f"TERRADEV_{name}_REGION_ID") or ""
        ).strip()
        extras["region"] = (
            os.environ.get(f"TERRADEV_{name}_REGION") or ""
        ).strip()

    if project_id:
        extras["project_id"] = project_id
    if location:
        extras["location"] = location
    return extras


def _load_drift_env_creds(provider: str) -> Optional[Union[str, Dict[str, Any]]]:
    """Look for a JSON credential object in the environment."""
    for alias in _drift_provider_aliases(provider):
        name = alias.upper().replace("-", "_")
        for env in (f"TERRADEV_{name}_CREDS", f"CANARY_{name}_CREDS"):
            raw = os.environ.get(env)
            if raw:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and "api_key" in data:
                        return data
                except json.JSONDecodeError:
                    pass
    return None


def _load_drift_credentials_file(provider: str, data: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any]]]:
    """Extract a provider key from ``~/.terradev/credentials.json``."""
    for alias in _drift_provider_aliases(provider):
        for key in (alias, f"{alias}_api_key"):
            if key in data:
                value = data[key]
                if isinstance(value, dict) and "api_key" in value:
                    return value
                if isinstance(value, str):
                    return value
    return None


def _load_drift_credentials(providers: List[str]) -> Dict[str, Union[str, Dict[str, Any]]]:
    """Resolve API keys for the requested providers."""
    creds: Dict[str, Union[str, Dict[str, Any]]] = {}

    creds_path = Path.home() / ".terradev" / "credentials.json"
    file_data: Dict[str, Any] = {}
    if creds_path.exists():
        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            click.echo(f"WARNING: Failed to load credentials file {creds_path}: {exc}", err=True)
            file_data = {}

    for provider in providers:
        key = _load_drift_env_key(provider)
        if key:
            creds[provider] = key
            continue
        key = _load_drift_env_creds(provider)
        if key:
            creds[provider] = key
            continue
        key = _load_drift_credentials_file(provider, file_data)
        if key:
            creds[provider] = key

    return creds


def _render_drift_human(summary: Dict[str, Any]) -> None:
    """Print a human-readable drift report."""
    date = summary["checked_at"][:10]
    width = 50
    click.echo(f"Provider Drift Report — {date}")
    click.echo("─" * width)

    for p in summary["providers"]:
        if p["drift_detected"]:
            symbol, status = "✗", "DRIFT"
        elif p["status"] == "skipped_no_credentials":
            symbol, status = "⊘", "skipped"
        else:
            symbol, status = "✓", "healthy"

        total = len(p.get("endpoints", []))
        ok = sum(1 for ep in p.get("endpoints", []) if not ep.get("drift"))

        if status == "DRIFT":
            reasons = "; ".join(
                {
                    ep.get("drift_summary") or ep.get("error") or "drift"
                    for ep in p.get("endpoints", [])
                    if ep.get("drift")
                }
            )
            detail = reasons
        elif status == "skipped":
            detail = "no credentials"
        else:
            detail = f"{ok}/{total} endpoints"

        click.echo(f"{symbol} {p['provider']:<15} {status:<10} {detail}")

    click.echo("─" * width)
    click.echo(f"{summary['healthy']} healthy  |  {summary['drifted']} drifted  |  {summary['skipped']} skipped")


@canary.command("drift")
@click.option(
    "--all",
    "drift_all",
    is_flag=True,
    help="Check all provider contracts.",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Check a specific provider contract (matches contract filename or provider field).",
)
@click.option(
    "--contracts-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Directory containing provider contract YAML files.",
)
@click.option(
    "--format",
    "drift_format",
    type=click.Choice(["human", "json", "jsonl"]),
    default=None,
    help="Output format for the drift report.",
)
@click.option(
    "--timeout",
    "drift_timeout",
    type=click.IntRange(1, 600),
    default=30,
    help="HTTP timeout in seconds for each drift endpoint call.",
)
@click.option(
    "--no-credentials",
    "no_credentials",
    is_flag=True,
    default=False,
    help="Run drift checks without loading any credentials (only public/no-auth endpoints).",
)
@click.pass_context
def canary_drift(ctx, drift_all, provider, contracts_dir, drift_format, drift_timeout, no_credentials):
    """Run a provider API drift check against live endpoints."""
    if not drift_all and not provider:
        get_output(ctx.find_root()).error("Specify --all or --provider <name>")
        ctx.exit(2)

    contracts_path = Path(contracts_dir) if contracts_dir else _default_contracts_dir()
    if not contracts_path.exists():
        get_output(ctx.find_root()).error(f"Contracts directory not found: {contracts_path}")
        ctx.exit(2)

    contract_files = sorted(contracts_path.glob("*.yaml"))
    if provider:
        provider_lower = provider.lower()
        matching_files = []
        for p in contract_files:
            with open(p, "r", encoding="utf-8") as f:
                contract = yaml.safe_load(f) or {}
            contract_provider = (contract.get("provider") or p.stem).lower()
            if (
                p.stem.lower() == provider_lower
                or provider_lower in p.stem.lower()
                or contract_provider == provider_lower
                or provider_lower in contract_provider
            ):
                matching_files.append(p)
        contract_files = matching_files

    if not contract_files:
        get_output(ctx.find_root()).error(f"No matching provider contracts in {contracts_path}")
        ctx.exit(2)

    providers = []
    for p in contract_files:
        with open(p, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f) or {}
        providers.append(contract.get("provider") or p.stem)
    credentials = {} if no_credentials else _load_drift_credentials(providers)
    monitor = DriftMonitor(str(contracts_path), credentials, timeout=drift_timeout)
    monitor.results = []
    for p in contract_files:
        monitor.results.append(monitor.check_provider(p))
    summary = monitor.summary()

    out = get_output(ctx.find_root())
    if drift_format:
        out.set_format(drift_format)

    if out.format == "json":
        out._closed = True  # noqa: SLF001
        click.echo(json.dumps(summary, indent=2, default=str))
        if summary["drifted"]:
            ctx.exit(1)
        return

    if out.format == "jsonl":
        out._closed = True  # noqa: SLF001
        for r in summary["providers"]:
            click.echo(json.dumps(r, default=str))
        if summary["drifted"]:
            ctx.exit(1)
        return

    _render_drift_human(summary)
    if summary["drifted"]:
        ctx.exit(1)


def _default_ml_contracts_dir() -> Path:
    """Return the bundled ML service contract directory."""
    return Path(__file__).resolve().parent.parent / "drift_monitor" / "ml_service_contracts"


@canary.command("ml-drift")
@click.option(
    "--all",
    "ml_all",
    is_flag=True,
    help="Check all ML service contracts.",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Check a specific ML service contract (matches filename or provider field).",
)
@click.option(
    "--format",
    "drift_format",
    type=click.Choice(["human", "json", "jsonl"]),
    default=None,
    help="Output format for the drift report.",
)
@click.option(
    "--timeout",
    "drift_timeout",
    type=click.IntRange(1, 600),
    default=30,
    help="HTTP timeout in seconds for each drift endpoint call.",
)
@click.option(
    "--base-url",
    "base_url",
    default=None,
    help="Override the base URL for all selected ML service contracts.",
)
@click.option(
    "--no-credentials",
    "no_credentials",
    is_flag=True,
    default=False,
    help="Run drift checks without loading any credentials (only public/no-auth endpoints).",
)
@click.pass_context
def canary_ml_drift(ctx, ml_all, provider, drift_format, drift_timeout, base_url, no_credentials):
    """Run drift checks for self-hosted ML/inference services."""
    if not ml_all and not provider:
        get_output(ctx.find_root()).error("Specify --all or --provider <name>")
        ctx.exit(2)

    contracts_path = _default_ml_contracts_dir()
    if not contracts_path.exists():
        get_output(ctx.find_root()).error(f"Contracts directory not found: {contracts_path}")
        ctx.exit(2)

    contract_files = sorted(contracts_path.glob("*.yaml"))
    if provider:
        provider_lower = provider.lower()
        matching_files = []
        for p in contract_files:
            with open(p, "r", encoding="utf-8") as f:
                contract = yaml.safe_load(f) or {}
            contract_provider = (contract.get("provider") or p.stem).lower()
            if (
                p.stem.lower() == provider_lower
                or provider_lower in p.stem.lower()
                or contract_provider == provider_lower
                or provider_lower in contract_provider
            ):
                matching_files.append(p)
        contract_files = matching_files

    if not contract_files:
        get_output(ctx.find_root()).error(f"No matching ML service contracts in {contracts_path}")
        ctx.exit(2)

    providers = []
    for p in contract_files:
        with open(p, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f) or {}
        providers.append(contract.get("provider") or p.stem)

    credentials = {} if no_credentials else _load_drift_credentials(providers)
    overrides = {"*": base_url} if base_url else {}
    monitor = DriftMonitor(
        str(contracts_path),
        credentials,
        timeout=drift_timeout,
        base_url_overrides=overrides,
    )
    monitor.results = []
    try:
        for p in contract_files:
            monitor.results.append(monitor.check_provider(p))
        summary = monitor.summary()
    except Exception as exc:
        get_output(ctx.find_root()).error(f"ML drift check failed: {exc}")
        ctx.exit(1)

    out = get_output(ctx.find_root())
    if drift_format:
        out.set_format(drift_format)

    if out.format == "json":
        out._closed = True  # noqa: SLF001
        click.echo(json.dumps(summary, indent=2, default=str))
        if summary["drifted"]:
            ctx.exit(1)
        return

    if out.format == "jsonl":
        out._closed = True  # noqa: SLF001
        for r in summary["providers"]:
            click.echo(json.dumps(r, default=str))
        if summary["drifted"]:
            ctx.exit(1)
        return

    _render_drift_human(summary)
    if summary["drifted"]:
        ctx.exit(1)
