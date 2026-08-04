#!/usr/bin/env python3
"""Universal database / vector store subcommand."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import click

from . import cli
from terradev_cli.core.adapters.orchestrator import UniversalOrchestrator
from terradev_cli.core.universal_manifest import UniversalManifest, Component
from terradev_cli.core.output import get_output


def _load_or_build_manifest(
    manifest: Optional[str],
    adapter: str,
    name: str,
    config: str,
    component_kind: str = "database",
) -> UniversalManifest:
    """Load a manifest or build one from CLI options."""
    if manifest:
        return UniversalManifest.load(Path(manifest))

    cfg = json.loads(config) if config else {}
    component = Component(
        kind=component_kind,
        name=name,
        adapter=adapter,
        config=cfg,
    )
    return UniversalManifest(
        name=f"database-{name}",
        version="0.0.0",
        components=[component],
    )


def _run(coro):
    """Run a single coroutine and return its result."""
    return asyncio.run(coro)


@cli.group("database")
def database():
    """Universal database and vector store operations."""
    pass


@database.command("up")
@click.option("--manifest", "-m", type=click.Path(exists=False), help="Path to universal manifest")
@click.option("--adapter", "-a", default="sqlite", help="Database adapter name")
@click.option("--name", "-n", default="db", help="Component name")
@click.option("--config", "-c", default="{}", help="JSON adapter config")
def database_up(manifest, adapter, name, config):
    """Initialize a database or vector store component."""
    output = get_output()
    m = _load_or_build_manifest(manifest, adapter, name, config)
    orchestrator = UniversalOrchestrator(m)

    async def _main():
        await orchestrator.initialize()
        return orchestrator.to_result().result

    output.set_result(_run(_main()))


@database.command("down")
@click.option("--manifest", "-m", type=click.Path(exists=False), required=True)
def database_down(manifest):
    """Teardown a database stack."""
    output = get_output()
    m = UniversalManifest.load(Path(manifest))
    orchestrator = UniversalOrchestrator(m)
    _run(orchestrator.teardown())
    output.success("Database stack torn down")


@database.command("crud")
@click.option("--manifest", "-m", type=click.Path(exists=False), help="Path to universal manifest")
@click.option("--adapter", "-a", default="sqlite")
@click.option("--name", "-n", default="db")
@click.option("--config", "-c", default="{}")
@click.option("--operation", required=True, type=click.Choice(["insert", "select", "update", "delete"]))
@click.option("--table", required=True)
@click.option("--data", default="{}", help="JSON data payload")
@click.option("--filters", default="{}", help="JSON filter payload")
def database_crud(manifest, adapter, name, config, operation, table, data, filters):
    """Run a CRUD operation on a database component."""
    output = get_output().set_format("json")
    m = _load_or_build_manifest(manifest, adapter, name, config)
    orchestrator = UniversalOrchestrator(m)

    async def _main():
        await orchestrator.initialize()
        try:
            return await orchestrator.execute(
                "database",
                name,
                "crud",
                {
                    "operation": operation,
                    "table": table,
                    "data": json.loads(data),
                    "filters": json.loads(filters),
                },
            )
        finally:
            await orchestrator.teardown()

    output.set_result(_run(_main()))


@database.command("search")
@click.option("--manifest", "-m", type=click.Path(exists=False), help="Path to universal manifest")
@click.option("--adapter", "-a", default="redis")
@click.option("--name", "-n", default="db")
@click.option("--config", "-c", default="{}")
@click.option("--table", required=True)
@click.option("--vector", required=True, help="JSON array of floats")
@click.option("--top-k", default=10, type=int)
@click.option("--filters", default="{}")
def database_search(manifest, adapter, name, config, table, vector, top_k, filters):
    """Run vector similarity search on a vector store component."""
    output = get_output().set_format("json")
    m = _load_or_build_manifest(manifest, adapter, name, config, component_kind="vector_store")
    orchestrator = UniversalOrchestrator(m)

    async def _main():
        await orchestrator.initialize()
        try:
            return await orchestrator.execute(
                "vector_store",
                name,
                "vector_search",
                {
                    "table": table,
                    "vector": json.loads(vector),
                    "top_k": top_k,
                    "filters": json.loads(filters),
                },
            )
        finally:
            await orchestrator.teardown()

    output.set_result(_run(_main()))
