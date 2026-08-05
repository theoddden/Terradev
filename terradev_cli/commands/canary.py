#!/usr/bin/env python3
"""Canary reporting commands for the Terradev CLI.

Reads canary result telemetry from ~/.terradev/canary-results.jsonl
and produces a human-readable or JSON summary.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click

from . import cli
from terradev_cli.core.output import get_output


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
    durations = []
    timestamps = []

    for r in records:
        providers[r.get("provider", "unknown")] = providers.get(r.get("provider", "unknown"), 0) + 1
        regions[r.get("region", "unknown")] = regions.get(r.get("region", "unknown"), 0) + 1
        gpu_types[r.get("gpu_type", "unknown")] = gpu_types.get(r.get("gpu_type", "unknown"), 0) + 1

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
