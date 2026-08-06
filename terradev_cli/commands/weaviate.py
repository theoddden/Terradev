#!/usr/bin/env python3
"""Weaviate integration for the Terradev CLI database command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import click

from terradev_cli.core.universal_manifest import UniversalManifest, Component
from terradev_cli.core.adapters.orchestrator import UniversalOrchestrator


weaviate = click.Group("weaviate", help="Weaviate vector database operations.")


def _weaviate_manifest(
    environment: str,
    host: str,
    http_port: int,
    grpc_port: int,
    secure: bool,
    cluster_url: str,
    api_key: str,
    headers: str,
) -> UniversalManifest:
    """Build a universal manifest for a Weaviate component."""
    config: Dict[str, Any] = {"environment": environment}
    if host:
        config["host"] = host
    if http_port:
        config["http_port"] = http_port
    if grpc_port:
        config["grpc_port"] = grpc_port
    if secure:
        config["secure"] = True
    if cluster_url:
        config["cluster_url"] = cluster_url
    if api_key:
        config["api_key"] = api_key
    if headers:
        config["headers"] = json.loads(headers)

    component = Component(
        kind="vector_store",
        name="weaviate",
        adapter="weaviate",
        config=config,
    )
    return UniversalManifest(
        name="weaviate",
        version="0.1.0",
        components=[component],
    )


def _run(coro):
    """Run a single coroutine and return its result."""
    return asyncio.run(coro)


def _weaviate_exec(operation: str, args: Dict[str, Any], manifest: UniversalManifest):
    """Initialize the Weaviate adapter and execute an operation."""
    orchestrator = UniversalOrchestrator(manifest)

    async def _main():
        await orchestrator.initialize()
        try:
            return await orchestrator.execute("vector_store", "weaviate", operation, args)
        finally:
            await orchestrator.teardown()

    return _run(_main())


@weaviate.command("up")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost", help="Weaviate HTTP host")
@click.option("--http-port", "-p", default=8080, type=int, help="Weaviate HTTP port")
@click.option("--grpc-port", default=50051, type=int, help="Weaviate gRPC port")
@click.option("--secure", is_flag=True, help="Use HTTPS/gRPC TLS")
@click.option("--cluster-url", help="Weaviate Cloud cluster URL")
@click.option("--api-key", help="Weaviate API key (or WEAVIATE_API_KEY env)")
@click.option("--headers", default="{}", help="JSON headers for the client")
def weaviate_up(
    environment,
    host,
    http_port,
    grpc_port,
    secure,
    cluster_url,
    api_key,
    headers,
):
    """Initialize a Weaviate connection."""
    manifest = _weaviate_manifest(
        environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers
    )
    result = _weaviate_exec("health", {}, manifest)
    if result.get("healthy"):
        click.echo(f"Weaviate {environment} connection ready")
        click.echo(f"  Healthy: {result.get('healthy')}")
        click.echo(f"  Message: {result.get('message')}")
    else:
        click.echo(f"ERROR: {result.get('message')}", err=True)
        raise SystemExit(1)


@weaviate.command("list-collections")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_list_collections(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers):
    """List Weaviate collections."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec("list_collections", {}, manifest)
    if result.get("status") == "ok":
        cols = result.get("collections", [])
        if cols:
            click.echo("Weaviate collections:")
            for name in cols:
                click.echo(f"  {name}")
        else:
            click.echo("No Weaviate collections found.")
    else:
        click.echo(f"ERROR: {result.get('error')}", err=True)
        raise SystemExit(1)


@weaviate.command("create-collection")
@click.option("--name", "-n", required=True, help="Collection name")
@click.option("--vector-size", type=int, help="Vector dimension (omit when using vectorizer)")
@click.option("--vectorizer", type=click.Choice(["openai", "cohere", "huggingface", "ollama"]), help="Built-in vectorizer module")
@click.option("--properties", default="[]", help="JSON list of properties [{name, data_type}]")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_create_collection(
    name,
    vector_size,
    vectorizer,
    properties,
    environment,
    host,
    http_port,
    grpc_port,
    secure,
    cluster_url,
    api_key,
    headers,
):
    """Create a Weaviate collection."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec(
        "create_collection",
        {
            "collection": name,
            "vector_size": vector_size,
            "vectorizer": vectorizer,
            "properties": json.loads(properties),
        },
        manifest,
    )
    if result.get("status") == "created":
        click.echo(f"Created Weaviate collection: {name}")
    else:
        click.echo(f"ERROR: {result.get('error')}", err=True)
        raise SystemExit(1)


@weaviate.command("delete-collection")
@click.option("--name", "-n", required=True, help="Collection name")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_delete_collection(name, environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers):
    """Delete a Weaviate collection."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec("delete_collection", {"collection": name}, manifest)
    if result.get("status") == "deleted":
        click.echo(f"Deleted Weaviate collection: {name}")
    else:
        click.echo(f"ERROR: {result.get('error')}", err=True)
        raise SystemExit(1)


@weaviate.command("insert")
@click.option("--collection", "-c", required=True)
@click.option("--objects", required=True, help="JSON list of objects [{properties: {...}, vector: [...]}]")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_insert(collection, objects, environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers):
    """Insert objects into a Weaviate collection."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec(
        "insert",
        {"collection": collection, "objects": json.loads(objects)},
        manifest,
    )
    if result.get("status") == "ok":
        click.echo(f"Inserted {result.get('count', 0)} objects into {collection}")
    else:
        click.echo(f"ERROR: {result.get('error')}", err=True)
        raise SystemExit(1)


@weaviate.command("query")
@click.option("--collection", "-c", required=True)
@click.option("--vector", required=True, help="JSON array of floats")
@click.option("--top-k", default=10, type=int)
@click.option("--filters", default="{}", help="JSON filter payload")
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_query(collection, vector, top_k, filters, environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers):
    """Vector similarity search in a Weaviate collection."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec(
        "vector_search",
        {
            "table": collection,
            "vector": json.loads(vector),
            "top_k": top_k,
            "filters": json.loads(filters),
        },
        manifest,
    )
    click.echo(json.dumps(result, indent=2, default=str))


@weaviate.command("hybrid-search")
@click.option("--collection", "-c", required=True)
@click.option("--query", required=True, help="Text query")
@click.option("--alpha", default=0.7, type=float, help="Balance between vector (1.0) and keyword (0.0)")
@click.option("--top-k", default=10, type=int)
@click.option("--environment", "-e", default="local", type=click.Choice(["local", "embedded", "cloud", "custom"]))
@click.option("--host", "-H", default="localhost")
@click.option("--http-port", "-p", default=8080, type=int)
@click.option("--grpc-port", default=50051, type=int)
@click.option("--secure", is_flag=True)
@click.option("--cluster-url")
@click.option("--api-key")
@click.option("--headers", default="{}")
def weaviate_hybrid_search(collection, query, alpha, top_k, environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers):
    """Hybrid vector + BM25 search in a Weaviate collection."""
    manifest = _weaviate_manifest(environment, host, http_port, grpc_port, secure, cluster_url, api_key, headers)
    result = _weaviate_exec(
        "hybrid_search",
        {
            "table": collection,
            "query": query,
            "alpha": alpha,
            "top_k": top_k,
        },
        manifest,
    )
    click.echo(json.dumps(result, indent=2, default=str))
