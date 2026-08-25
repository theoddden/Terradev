#!/usr/bin/env python3
"""Canary reporting commands for the Terradev CLI.

Reads canary result telemetry from ~/.terradev/canary-results.jsonl
and produces a human-readable or JSON summary.
"""

import base64
import binascii
import json
import os
import sys
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
        sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
        return

    print("=" * 60)
    print("TERRADEV CANARY REPORT".center(60))
    print("=" * 60)

    if not records:
        print("No canary records found.")
        print(f"Run canary tests first or point to a results file with --file.")
        return

    print(f"Total runs:     {summary['total']}")
    print(f"Passed:         {summary['passed']}")
    print(f"Failed:         {summary['failed']}")
    print(f"Skipped:        {summary['skipped']}")
    if summary["first_seen"]:
        print(f"First seen:     {summary['first_seen']}")
    if summary["last_seen"]:
        print(f"Last seen:      {summary['last_seen']}")

    dur = summary["duration_ms"]
    if dur["avg"] is not None:
        print(f"Duration (ms):  min={dur['min']}, max={dur['max']}, avg={dur['avg']:.1f}")

    print("-" * 60)
    print("By provider:")
    for p, count in sorted(summary["providers"].items(), key=lambda x: -x[1]):
        print(f"  {p:<20} {count}")

    if summary["regions"]:
        print("-" * 60)
        print("By region:")
        for r, count in sorted(summary["regions"].items(), key=lambda x: -x[1]):
            print(f"  {r:<20} {count}")

    if summary["gpu_types"]:
        print("-" * 60)
        print("By GPU type:")
        for g, count in sorted(summary["gpu_types"].items(), key=lambda x: -x[1]):
            print(f"  {g:<20} {count}")

    if summary["tpu_types"]:
        print("-" * 60)
        print("By TPU type:")
        for t, count in sorted(summary["tpu_types"].items(), key=lambda x: -x[1]):
            print(f"  {t:<20} {count}")

    if summary["tpu_chips"]:
        print("-" * 60)
        print("By TPU chip count:")
        for t, count in sorted(summary["tpu_chips"].items(), key=lambda x: -x[1]):
            print(f"  {t:<20} {count}")

    print("=" * 60)


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
    type=int,
    help="Number of recent records to show",
)
def canary_tail(file, limit):
    """Show the most recent canary records."""
    path = Path(file) if file else _default_canary_output()
    records = _load_canary_records(path)
    recent = records[-limit:]
    if not recent:
        print("No canary records found.")
        return
    for rec in recent:
        print(json.dumps(rec, indent=2, default=str))


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
    "oracle": ["oracle", "oci"],
    "crusoe": ["crusoe"],
    "hyperstack": ["hyperstack"],
    "digitalocean": ["digitalocean", "digital_ocean"],
    "alibaba": ["alibaba", "ali"],
    "ovhcloud": ["ovhcloud", "ovh"],
    "siliconflow": ["siliconflow"],
    "inferx": ["inferx"],
    "latitude": ["latitude"],
    "yottalabs": ["yottalabs", "yotta"],
    "e2enetworks": ["e2enetworks", "e2e"],
}

# Provider-specific primary env names (full names, not alias-derived).
_DRIFT_PROVIDER_KEY_ENVS: Dict[str, List[str]] = {
    "aws": ["TERRADEV_AWS_ACCESS_KEY_ID"],
    "gcp": ["TERRADEV_GCP_CREDENTIALS"],
    "oracle": ["TERRADEV_OCI_PRIVATE_KEY", "TERRADEV_OCI_TENANCY"],
    "inferx": ["TERRADEV_INFERX_API_KEY"],
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
        extras["aws_access_key_id"] = api_key
        extras["aws_secret_access_key"] = os.environ.get(
            "TERRADEV_AWS_SECRET_ACCESS_KEY", ""
        )
        extras["aws_region"] = os.environ.get("TERRADEV_AWS_REGION") or "us-east-1"

    elif alias == "gcp":
        try:
            raw = base64.b64decode(api_key).decode("utf-8")
            gcp_creds = json.loads(raw)
            extras["gcp_credentials"] = gcp_creds
            extras["project_id"] = gcp_creds.get("project_id") or ""
        except (ValueError, json.JSONDecodeError):
            extras["gcp_credentials"] = api_key

    elif alias == "oracle":
        extras["oci_tenancy"] = os.environ.get("TERRADEV_OCI_TENANCY") or ""
        extras["oci_user"] = os.environ.get("TERRADEV_OCI_USER") or ""
        extras["oci_fingerprint"] = os.environ.get("TERRADEV_OCI_FINGERPRINT") or ""
        extras["oci_region"] = os.environ.get("TERRADEV_OCI_REGION") or "us-ashburn-1"
        # The primary env value may be the private key (base64) or the tenancy.
        # If it decodes to a PEM private key, use it; otherwise read the dedicated secret.
        private_key = api_key
        if not private_key.startswith("-----BEGIN"):
            try:
                private_key = base64.b64decode(private_key).decode("utf-8")
            except (ValueError, binascii.Error):
                pass
        extras["oci_private_key"] = private_key
        # If the primary value was the tenancy, fall back to the dedicated private key secret.
        fallback_key = os.environ.get("TERRADEV_OCI_PRIVATE_KEY")
        if fallback_key:
            try:
                extras["oci_private_key"] = base64.b64decode(fallback_key).decode("utf-8")
            except (ValueError, binascii.Error):
                extras["oci_private_key"] = fallback_key

    elif alias == "inferx":
        extras["region"] = os.environ.get("TERRADEV_INFERX_REGION") or "us-west-2"

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
        except (json.JSONDecodeError, OSError):
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
    print(f"Provider Drift Report — {date}")
    print("─" * width)

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

        print(f"{symbol} {p['provider']:<15} {status:<10} {detail}")

    print("─" * width)
    print(f"{summary['healthy']} healthy  |  {summary['drifted']} drifted  |  {summary['skipped']} skipped")


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
    type=int,
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
        get_output(ctx).error("Specify --all or --provider <name>")
        ctx.exit(2)
        return

    contracts_path = Path(contracts_dir) if contracts_dir else _default_contracts_dir()
    if not contracts_path.exists():
        get_output(ctx).error(f"Contracts directory not found: {contracts_path}")
        ctx.exit(2)
        return

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
        get_output(ctx).error(f"No matching provider contracts in {contracts_path}")
        ctx.exit(2)
        return

    providers = []
    for p in contract_files:
        with open(p, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f) or {}
        providers.append(contract.get("provider") or p.stem)
    credentials = {} if no_credentials else _load_drift_credentials(providers)
    monitor = DriftMonitor(str(contracts_path), credentials, timeout=drift_timeout)
    monitor.run_all()
    summary = monitor.summary()

    out = get_output(ctx)
    if drift_format:
        out.set_format(drift_format)

    if out.format == "json":
        out._closed = True  # noqa: SLF001
        sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
        if summary["drifted"]:
            ctx.exit(1)
        return

    if out.format == "jsonl":
        out._closed = True  # noqa: SLF001
        for r in summary["providers"]:
            sys.stdout.write(json.dumps(r, default=str) + "\n")
        if summary["drifted"]:
            ctx.exit(1)
        return

    _render_drift_human(summary)
    if summary["drifted"]:
        ctx.exit(1)
