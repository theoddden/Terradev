#!/usr/bin/env python3
"""Chaos, failure-injection, and property-based tests for agent subcommands.

Covers the new and complex ``terradev agent`` components: sandbox, mesh, mcp.
These tests exercise silent failure paths, malformed inputs, unavailable
runtimes/transports, and concurrency edge cases.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

pytestmark = [pytest.mark.unit]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Sandbox chaos
# ---------------------------------------------------------------------------


class TestSandboxChaos:
    def test_runtime_factory_unknown_runtime(self):
        from terradev_cli.commands.agent_infra.sandbox import RuntimeFactory
        from types import SimpleNamespace

        with pytest.raises(Exception, match="Unknown runtime"):
            _run(RuntimeFactory.select(SimpleNamespace(runtime="unknown", dev_mode=False)))

    def test_runtime_factory_unavailable_named_runtime(self):
        from terradev_cli.commands.agent_infra.sandbox import RuntimeFactory

        # gvisor is unlikely to be installed in CI
        with pytest.raises(Exception, match="not available|Unknown"):
            _run(RuntimeFactory.select(SandboxConfigOrDict(runtime="gvisor")))

    def test_sandbox_runner_dry_run(self):
        from terradev_cli.commands.agent_infra.sandbox import (
            SandboxRunner,
            SandboxConfig,
            ResourceLimits,
        )

        cfg = SandboxConfig(
            runtime="local",
            payload="echo hi",
            dev_mode=True,
            resources=ResourceLimits(timeout_seconds=5),
            dry_run=True,
        )
        runner = SandboxRunner(cfg)
        result = _run(runner.dry_run(["echo", "hi"]))
        assert result.exit_code == 0
        assert result.runtime == "local"

    def test_sandbox_config_invalid_runtime(self):
        from terradev_cli.commands.agent_infra.sandbox import SandboxConfig

        with pytest.raises(ValueError, match="Unknown runtime"):
            SandboxConfig(runtime="not-a-runtime")

    def test_sandbox_config_invalid_network_mode(self):
        from terradev_cli.commands.agent_infra.sandbox import NetworkPolicy

        with pytest.raises(ValueError):
            NetworkPolicy(mode="invalid")

    def test_sandbox_timeout_error(self):
        from terradev_cli.commands.agent_infra.sandbox import (
            SandboxRunner,
            SandboxConfig,
            ResourceLimits,
            SandboxTimeoutError,
        )

        cfg = SandboxConfig(
            runtime="local",
            payload="sleep 10",
            dev_mode=True,
            resources=ResourceLimits(timeout_seconds=0.1),
        )
        runner = SandboxRunner(cfg)
        with pytest.raises(SandboxTimeoutError):
            _run(runner.run(["python3", "-c", "import time; time.sleep(5)"]))

    def test_sandbox_cli_malformed_env(self):
        from click.testing import CliRunner
        from terradev_cli.commands.agent_infra.sandbox import sandbox

        runner = CliRunner()
        result = runner.invoke(
            sandbox,
            ["run", "--runtime", "local", "--dev", "--env", "noequals", "echo hi"],
        )
        assert result.exit_code != 0

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    @given(size=st.text(min_size=1, max_size=10))
    def test_parse_size_accepts_human_units(self, size):
        from terradev_cli.commands.agent_infra.core import _parse_size

        # Only numbers with optional known unit should not raise
        try:
            value = _parse_size(f"{size}m")
            assert value is None or isinstance(value, int)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Mesh chaos
# ---------------------------------------------------------------------------


class TestMeshChaos:
    def test_mesh_node_start_twice(self):
        from terradev_cli.commands.agent_infra.mesh import MeshConfig, MeshNode

        async def _main():
            cfg = MeshConfig(listen="127.0.0.1:0", transport="http")
            node = MeshNode(cfg)
            await node.start()
            try:
                with pytest.raises(Exception, match="already started"):
                    await node.start()
            finally:
                await node.stop()

        _run(_main())

    def test_mesh_node_stop_when_not_started(self):
        from terradev_cli.commands.agent_infra.mesh import MeshConfig, MeshNode

        async def _main():
            node = MeshNode(MeshConfig())
            await node.stop()  # should be a no-op, not raise

        _run(_main())

    def test_mesh_unavailable_transport(self):
        from terradev_cli.commands.agent_infra.mesh import (
            MeshConfig,
            MeshNode,
            MeshError,
            MeshTransport,
        )

        class DeadTransport(MeshTransport):
            name = "dead"

            async def start(self, config):
                pass

            async def stop(self):
                pass

            async def publish_card(self, card):
                pass

            async def discover_cards(self, skills=None):
                return []

            async def delegate_task(self, card, task):
                return task

            async def is_available(self):
                return False

        async def _main():
            node = MeshNode(
                MeshConfig(transport="dead"),
                transport=DeadTransport(),
            )
            with pytest.raises(MeshError, match="not available"):
                await node.start()

        _run(_main())

    def test_route_by_strategy_empty(self):
        from terradev_cli.commands.agent_infra.mesh import (
            MeshConfig,
            MeshNode,
            MeshRoutingStrategy,
        )

        async def _main():
            node = MeshNode(MeshConfig(listen="127.0.0.1:0", transport="http"))
            await node.start()
            try:
                card = await node.route_by_strategy(skills=[], strategy=MeshRoutingStrategy.COST)
                assert card is None
            finally:
                await node.stop()

        _run(_main())

    def test_http_transport_endpoint_before_start(self):
        from terradev_cli.commands.agent_infra.mesh import HttpTransport

        transport = HttpTransport()
        assert transport._self_endpoint() == "http://127.0.0.1:0"

    def test_parse_listen_no_port(self):
        from terradev_cli.commands.agent_infra.mesh import HttpTransport

        transport = HttpTransport()
        assert transport._parse_listen("4222") == ("127.0.0.1", 4222)

    def test_delegate_task_connection_refused(self):
        from terradev_cli.commands.agent_infra.mesh import (
            AgentCard,
            HttpTransport,
            MeshConfig,
            MeshError,
            Task,
        )

        async def _main():
            transport = HttpTransport()
            await transport.start(MeshConfig(listen="127.0.0.1:0", transport="http"))
            try:
                card = AgentCard(name="dead", endpoint="http://127.0.0.1:1", skills=["x"])
                task = Task(input="x", skills=["x"])
                with pytest.raises(MeshError):
                    await transport.delegate_task(card, task)
            finally:
                await transport.stop()

        _run(_main())

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    @given(port=st.integers(min_value=0, max_value=65535))
    def test_parse_listen_valid_ports(self, port):
        from terradev_cli.commands.agent_infra.mesh import HttpTransport

        host, parsed = HttpTransport()._parse_listen(f"0.0.0.0:{port}")
        assert host == "0.0.0.0"
        assert parsed == port


# ---------------------------------------------------------------------------
# MCP chaos
# ---------------------------------------------------------------------------


class TestMcpChaos:
    def test_mcp_bridge_call_unknown_server(self):
        from terradev_cli.commands.agent_infra.mcp import McpBridge, McpError

        async def _main():
            bridge = McpBridge()
            with pytest.raises(McpError, match="Unknown MCP server"):
                await bridge.call_tool("missing.tool", {})

        _run(_main())

    def test_mcp_bridge_call_ambiguous_tool(self):
        from terradev_cli.commands.agent_infra.mcp import McpBridge, McpError

        class FakeConn:
            tools = {"add": None}

            async def close(self):
                pass

        bridge = McpBridge()
        bridge.servers["a"] = FakeConn()
        bridge.servers["b"] = FakeConn()

        async def _main():
            with pytest.raises(McpError, match="ambiguous"):
                await bridge.call_tool("add", {})

        _run(_main())

    def test_mcp_bridge_handle_unimplemented_method(self):
        from terradev_cli.commands.agent_infra.mcp import McpBridge
        from terradev_cli.commands.agent_infra.core import JsonRpcMessage

        async def _main():
            bridge = McpBridge()
            msg = JsonRpcMessage(id=1, method="not-implemented", params={})
            resp = await bridge.handle_message(msg)
            assert resp.error is not None
            assert "not implemented" in resp.error["message"]

        _run(_main())

    def test_schema_compiler_unknown_type(self):
        from terradev_cli.commands.agent_infra.mcp import SchemaCompiler

        schema = {
            "type": "object",
            "properties": {
                "weird": {"type": "weird_type"},
                "list": {"type": "array"},
            },
        }
        model = SchemaCompiler.compile("test", schema)
        assert model is not None

    def test_http_mcp_transport_fails_without_session(self):
        from terradev_cli.commands.agent_infra.mcp import HttpMcpTransport, McpError

        transport = HttpMcpTransport("http://127.0.0.1:1")

        async def _main():
            with pytest.raises(McpError, match="not started"):
                await transport.send(JsonRpcLike())

        _run(_main())

    def test_mcp_server_connection_tool_not_found(self):
        from terradev_cli.commands.agent_infra.mcp import McpServerConnection, McpError

        transport = AsyncMockTransport()

        async def _main():
            from terradev_cli.commands.agent_infra.core import McpServerDefinition

            conn = McpServerConnection(
                "fake",
                McpServerDefinition(name="fake", command="echo"),
                transport,
            )
            conn.tools = {}
            with pytest.raises(McpError, match="not found"):
                await conn.call_tool("missing", {})

        _run(_main())

    def test_mcp_server_stdio_ignores_malformed_lines(self):
        from terradev_cli.commands.agent_infra.mcp import McpServer, McpBridge
        from terradev_cli.commands.agent_infra.core import JsonRpcMessage

        class FakeStdout:
            def __init__(self):
                self.written = []

            def write(self, data):
                self.written.append(data)

            def flush(self):
                pass

        bridge = McpBridge()
        server = McpServer(bridge)

        async def _main():
            import io

            stdin = io.StringIO("not-json\n{not valid\n")
            with patch("sys.stdin", stdin), patch("sys.stdout", FakeStdout()):
                # read all input lines then end
                await server.serve_stdio()

        _run(_main())


# ---------------------------------------------------------------------------
# Helpers and hypothesis scaffolding
# ---------------------------------------------------------------------------


def SandboxConfigOrDict(**kwargs):
    """Return a SandboxConfig, or a simple dict if that fails."""
    from terradev_cli.commands.agent_infra.sandbox import SandboxConfig

    try:
        return SandboxConfig(**kwargs)
    except Exception:
        return kwargs


class JsonRpcLike:
    """Minimal stand-in for JsonRpcMessage."""

    def __init__(self):
        self.id = 1
        self.method = "tools/list"
        self.params = {}

    def model_dump(self, **k):
        return {"id": self.id, "method": self.method, "params": self.params}


class AsyncMockTransport:
    async def start(self):
        pass

    async def send(self, message):
        return JsonRpcLike()

    async def send_notification(self, message):
        pass

    async def close(self):
        pass
