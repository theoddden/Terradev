#!/usr/bin/env python3
"""Tests for ``terradev agent mcp``."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from terradev_cli.commands.agent_infra.mcp import (
    JsonRpcCodec,
    McpBridge,
    McpRegistry,
    McpServer,
    McpServerConnection,
    McpServerDefinition,
    SchemaCompiler,
    mcp,
)
from terradev_cli.commands.agent_infra.core import JsonRpcMessage


def _run(coro):
    return asyncio.run(coro)


def test_json_rpc_codec():
    msg = JsonRpcMessage(id=1, method="tools/list", params={})
    raw = JsonRpcCodec.encode(msg)
    decoded = JsonRpcCodec.decode(raw)
    assert decoded.method == "tools/list"
    assert decoded.id == 1


def test_schema_compiler_basic():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    model = SchemaCompiler.compile("search", schema)
    m = model(query="hello", limit=5)
    assert m.query == "hello"
    assert m.limit == 5


def test_schema_compiler_validates():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    with pytest.raises(ValidationError):
        SchemaCompiler.validate("search", schema, {"limit": 5})


def test_mcp_registry_roundtrip(tmp_path: Path):
    path = tmp_path / "mcp_registry.json"
    reg = McpRegistry()
    reg.add(McpServerDefinition(name="echo", command="python3", args=["-c", "print('hi')"]))
    reg.save(path)

    loaded = McpRegistry.load(path)
    assert len(loaded.servers) == 1
    assert loaded.servers[0].name == "echo"


class FakeTransport:
    """In-memory transport that mimics an MCP server with one tool."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.closed = False

    async def start(self):
        pass

    async def send(self, message: JsonRpcMessage) -> JsonRpcMessage:
        if message.method == "initialize":
            return JsonRpcMessage(id=message.id, result={
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "fake"},
            })
        if message.method == "tools/list":
            return JsonRpcMessage(
                id=message.id,
                result={"tools": [t.model_dump() for t in self.tools]},
            )
        if message.method == "tools/call":
            name = message.params.get("name")
            return JsonRpcMessage(
                id=message.id,
                result={"content": [{"type": "text", "text": f"called {name}"}]},
            )
        return JsonRpcMessage(id=message.id, result={})

    async def send_notification(self, message: JsonRpcMessage) -> None:
        pass

    async def close(self):
        self.closed = True


def test_mcp_server_connection():
    async def _main():
        tool = JsonRpcMessage()  # placeholder
        from terradev_cli.commands.agent_infra.core import McpTool
        t = McpTool(name="echo", inputSchema={"type": "object", "properties": {}})
        transport = FakeTransport(tools=[t])
        definition = McpServerDefinition(name="fake", command="echo")
        conn = McpServerConnection("fake", definition, transport)
        await conn.start()
        assert "echo" in conn.tools
        result = await conn.call_tool("echo", {})
        assert "called echo" in result["content"][0]["text"]
        await conn.close()
        assert transport.closed

    _run(_main())


def test_mcp_bridge_aggregates_tools():
    async def _main():
        from terradev_cli.commands.agent_infra.core import McpTool
        definition = McpServerDefinition(name="fake", command="echo")
        conn = McpServerConnection(
            "fake",
            definition,
            FakeTransport(tools=[McpTool(name="add", inputSchema={"type": "object"})]),
        )
        await conn.start()

        bridge = McpBridge()
        bridge.servers["fake"] = conn

        tools = await bridge.list_all_tools()
        assert any(t["name"] == "add" for t in tools)

        result = await bridge.call_tool("fake.add", {"x": 1})
        assert "called add" in result["content"][0]["text"]

        await bridge.close()

    _run(_main())


def test_mcp_bridge_handle_initialize():
    async def _main():
        bridge = McpBridge()
        msg = JsonRpcMessage(id=1, method="initialize", params={})
        resp = await bridge.handle_message(msg)
        assert resp.error is None
        assert resp.result["protocolVersion"] == "2025-03-26"

    _run(_main())


def test_mcp_server_http():
    async def _main():
        bridge = McpBridge()
        server = McpServer(bridge)
        ready = asyncio.Event()
        task = asyncio.create_task(server.serve_http(port=0, host="127.0.0.1", ready_event=ready))
        await asyncio.wait_for(ready.wait(), timeout=2.0)

        import aiohttp
        port = getattr(server, "_http_port", 8080)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/mcp") as resp:
                data = await resp.json()
                assert data["status"] == "ok"

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(_main())


def test_mcp_cli_help(runner: CliRunner):
    result = runner.invoke(mcp, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "call" in result.output


def test_mcp_registry_cli(runner: CliRunner, tmp_path: Path):
    path = tmp_path / "mcp_registry.json"
    result = runner.invoke(
        mcp,
        [
            "registry",
            "add",
            "--config",
            str(path),
            "--name",
            "echo",
            "--transport",
            "stdio",
            "--command",
            "python3",
            "--args",
            "-m,http.server",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "echo" in result.output

    result = runner.invoke(mcp, ["registry", "list", "--config", str(path)])
    assert result.exit_code == 0
    assert "echo" in result.output
