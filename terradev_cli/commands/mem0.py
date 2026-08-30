#!/usr/bin/env python3
"""Mem0 (agentic memory) integration for the Terradev CLI."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import click

from terradev_cli.commands._base import (
    TerradevGroup as Mem0Group,
    get_api as _get_api,
    run_with_timeout as _run_with_timeout,
)


def _safe_json(raw: str, option: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        click.echo(f"ERROR: Invalid JSON in {option}: {exc}", err=True)
        raise SystemExit(1) from exc


@click.group(cls=Mem0Group)
def mem0():
    """Mem0 agentic memory: persistent, personalized memory for AI agents."""
    pass


@mem0.command("configure")
@click.option("--api-key", help="Mem0 API key (hosted mode)")
@click.option(
    "--mode",
    type=click.Choice(["hosted", "self_hosted"]),
    default="hosted",
    help="Mem0 mode",
)
@click.option("--host", default="https://api.mem0.ai", help="Mem0 Platform host")
@click.option("--org-id", help="Mem0 organization ID")
@click.option("--project-id", help="Mem0 project ID")
@click.option("--vector-store", help='Self-hosted vector store JSON, e.g. {"provider":"qdrant"}')
@click.option("--llm", help='Self-hosted LLM JSON, e.g. {"provider":"openai"}')
@click.option("--embedder", help='Self-hosted embedder JSON')
@click.option("--graph-store", help='Self-hosted graph store JSON')
@click.option("--custom-instructions", help="Custom instructions for memory extraction")
@click.option("--default-user-id", help="Default user_id for entity scoping")
@click.option("--default-agent-id", help="Default agent_id for entity scoping")
@click.option("--default-app-id", help="Default app_id for entity scoping")
@click.option("--default-run-id", help="Default run_id for entity scoping")
def mem0_configure(
    api_key,
    mode,
    host,
    org_id,
    project_id,
    vector_store,
    llm,
    embedder,
    graph_store,
    custom_instructions,
    default_user_id,
    default_agent_id,
    default_app_id,
    default_run_id,
):
    """Configure Mem0 credentials and defaults."""
    api = _get_api()

    resolved_key = api_key or os.environ.get("MEM0_API_KEY", "")
    if mode == "hosted" and not resolved_key:
        click.echo(
            "ERROR: --api-key or MEM0_API_KEY is required for hosted mode.",
            err=True,
        )
        raise SystemExit(1)

    creds: Dict[str, str] = {
        "mode": mode,
        "api_key": resolved_key,
        "host": host,
    }
    if org_id:
        creds["org_id"] = org_id
    if project_id:
        creds["project_id"] = project_id
    if vector_store:
        _safe_json(vector_store, "--vector-store")
        creds["vector_store"] = vector_store
    if llm:
        _safe_json(llm, "--llm")
        creds["llm"] = llm
    if embedder:
        _safe_json(embedder, "--embedder")
        creds["embedder"] = embedder
    if graph_store:
        _safe_json(graph_store, "--graph-store")
        creds["graph_store"] = graph_store
    if custom_instructions:
        creds["custom_instructions"] = custom_instructions
    if default_user_id:
        creds["default_user_id"] = default_user_id
    if default_agent_id:
        creds["default_agent_id"] = default_agent_id
    if default_app_id:
        creds["default_app_id"] = default_app_id
    if default_run_id:
        creds["default_run_id"] = default_run_id

    api._save_provider_creds("mem0", creds)
    click.echo(f"OK: Mem0 configured ({mode})")


@mem0.command("test")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_test(fmt):
    """Test connection to Mem0."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
            get_mem0_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")

        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo(get_mem0_setup_instructions())
            return

        service, _ = create_mem0_service_from_credentials(creds)
        click.echo(" Testing Mem0 connection...")
        result = _run_with_timeout(service.test_connection_async())

        if result["status"] == "connected":
            if fmt == "json":
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo("OK: Mem0 connected successfully")
                click.echo(f"   Mode: {result.get('mode', 'hosted')}")
                if "host" in result:
                    click.echo(f"   Host: {result['host']}")
        else:
            click.echo(f"ERROR: Mem0 connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("add")
@click.option("--text", "-t", help="Text to remember")
@click.option("--messages", "-m", help='JSON list of message dicts, e.g. [{"role":"user","content":"..."}]')
@click.option("--user", "user_id", help="User ID")
@click.option("--agent", "agent_id", help="Agent ID")
@click.option("--app", "app_id", help="App ID")
@click.option("--run", "run_id", help="Run/session ID")
@click.option("--metadata", help="JSON metadata to attach")
@click.option("--no-infer", is_flag=True, default=False, help="Store raw text without LLM fact extraction")
@click.option("--category", multiple=True, help="Custom categories")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_add(
    text,
    messages,
    user_id,
    agent_id,
    app_id,
    run_id,
    metadata,
    no_infer,
    category,
    fmt,
):
    """Add a memory for an agent or user."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)

        if messages:
            msgs = _safe_json(messages, "--messages")
        elif text:
            msgs = [{"role": "user", "content": text}]
        else:
            click.echo("ERROR: --text or --messages is required.", err=True)
            raise SystemExit(1)

        meta = _safe_json(metadata, "--metadata") if metadata else None
        cats = list(category) if category else None

        result = _run_with_timeout(
            service.add_async(
                msgs,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
                metadata=meta,
                infer=not no_infer,
                custom_categories=cats,
            )
        )

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            click.echo("OK: Memory added")
            _print_memories(result.get("results", []))
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("search")
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--user", "user_id", help="User ID")
@click.option("--agent", "agent_id", help="Agent ID")
@click.option("--app", "app_id", help="App ID")
@click.option("--run", "run_id", help="Run/session ID")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--threshold", type=float, help="Minimum similarity score")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_search(query, user_id, agent_id, app_id, run_id, top_k, threshold, fmt):
    """Search agent/user memories."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)
        result = _run_with_timeout(
            service.search_async(
                query,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
                top_k=top_k,
                threshold=threshold,
            )
        )

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            memories = result.get("results", [])
            if not memories:
                click.echo("No memories found.")
                return
            click.echo(f"Memories ({len(memories)}):")
            _print_memories(memories)
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("get")
@click.option("--memory-id", "-i", required=True, help="Memory ID")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_get(memory_id, fmt):
    """Get a single memory by ID."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)
        try:
            result = _run_with_timeout(service.get(memory_id))
        except Exception as e:  # noqa: BLE001
            click.echo(f"ERROR: Memory not found: {memory_id} ({e})", err=True)
            raise SystemExit(1)

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            _print_memories([result])
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("list")
@click.option("--user", "user_id", help="User ID")
@click.option("--agent", "agent_id", help="Agent ID")
@click.option("--app", "app_id", help="App ID")
@click.option("--run", "run_id", help="Run/session ID")
@click.option("--page", default=1, help="Page number")
@click.option("--page-size", default=50, help="Results per page")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_list(user_id, agent_id, app_id, run_id, page, page_size, fmt):
    """List memories for an entity scope."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)
        result = _run_with_timeout(
            service.get_all_async(
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
                page=page,
                page_size=page_size,
            )
        )

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            memories = result.get("results", [])
            if not memories:
                click.echo("No memories found.")
                return
            click.echo(f"Memories ({len(memories)} of {result.get('count', '?')}):")
            _print_memories(memories)
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("update")
@click.option("--memory-id", "-i", required=True, help="Memory ID")
@click.option("--text", "-t", help="Updated memory text")
@click.option("--metadata", help="JSON metadata")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def mem0_update(memory_id, text, metadata, fmt):
    """Update a memory by ID."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)
        meta = _safe_json(metadata, "--metadata") if metadata else None
        result = _run_with_timeout(service.update(memory_id, text=text, metadata=meta))

        if fmt == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            click.echo(f"OK: Updated memory {memory_id}")
            _print_memories([result] if isinstance(result, dict) else [])
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("delete")
@click.option("--memory-id", "-i", required=True, help="Memory ID")
def mem0_delete(memory_id):
    """Delete a memory by ID."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        service, _ = create_mem0_service_from_credentials(creds)
        _run_with_timeout(service.delete(memory_id))
        click.echo(f"OK: Deleted memory {memory_id}")
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


@mem0.command("forget")
@click.option("--user", "user_id", help="User ID")
@click.option("--agent", "agent_id", help="Agent ID")
@click.option("--app", "app_id", help="App ID")
@click.option("--run", "run_id", help="Run/session ID")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def mem0_forget(user_id, agent_id, app_id, run_id, yes):
    """Delete all memories matching the given entity scope."""
    try:
        from terradev_cli.ml_services.mem0_service import (
            create_mem0_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("mem0")
        if not creds.get("api_key") and creds.get("mode", "hosted") == "hosted":
            click.echo("ERROR: Mem0 not configured. Run 'terradev agent mem0 configure' first.", err=True)
            raise SystemExit(1)

        scope = [f"{k}={v}" for k, v in {
            "user_id": user_id,
            "agent_id": agent_id,
            "app_id": app_id,
            "run_id": run_id,
        }.items() if v]
        if not scope:
            click.echo("ERROR: At least one of --user/--agent/--app/--run is required.", err=True)
            raise SystemExit(1)

        if not yes:
            click.confirm(
                f"Delete all memories for {', '.join(scope)}?",
                abort=True,
            )

        service, _ = create_mem0_service_from_credentials(creds)
        result = _run_with_timeout(
            service.delete_all(
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )
        )
        click.echo(f"OK: Deleted memories: {result}")
    except ImportError:
        click.echo("ERROR: Mem0 service not available. Install with: pip install mem0ai", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_memories(memories: List[Dict[str, Any]]) -> None:
    for mem in memories:
        mid = mem.get("id", "<no-id>")
        text = mem.get("memory", mem.get("text", mem.get("content", "")))
        score = mem.get("score")
        cat = mem.get("category", "")
        out = f"  {mid}: {text}"
        if score is not None:
            out += f" (score: {score:.3f})"
        if cat:
            out += f" [cat: {cat}]"
        click.echo(out)
