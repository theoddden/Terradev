#!/usr/bin/env python3
"""End-to-end test for the agentic pipeline.

MCP tool request -> schema validation -> sandbox execution -> mesh task
delegation.  The test wires the three subsystems together in memory.
"""

from __future__ import annotations

import asyncio

import pytest

from terradev_cli.commands.agent_infra.core import (
    AgentCard,
    JsonRpcMessage,
    McpServerDefinition,
    McpTool,
    NetworkPolicy,
    ResourceLimits,
    SandboxConfig,
    Task,
)
from terradev_cli.commands.agent_infra.mesh import (
    HttpTransport,
    MeshConfig,
    MeshNode,
    TransportFactory,
)
from terradev_cli.commands.agent_infra.mcp import (
    McpBridge,
    McpServerConnection,
    SchemaCompiler,
)
from terradev_cli.commands.agent_infra.sandbox import LocalRuntime, SandboxRunner


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _enable_local_sandbox(monkeypatch):
    """Allow the local sandbox runtime to execute during e2e tests."""
    monkeypatch.setenv("TERRADEV_AGENT_SANDBOX_LOCAL", "1")


class FakeTransport:
    """Minimal in-memory MCP transport for tests."""

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


class AgentToolServer:
    """A fake MCP server that exposes a single `execute` tool."""

    def __init__(self):
        self.runtime = LocalRuntime()

    async def execute(self, payload: str, skills: list) -> dict:
        """Run payload in the local sandbox, then delegate to the mesh."""
        cfg = SandboxConfig(
            runtime="local",
            payload=payload,
            dev_mode=True,
            network=NetworkPolicy(mode="none"),
            resources=ResourceLimits(timeout_seconds=10),
        )
        runner = SandboxRunner(cfg)
        command = ["python3", "-c", payload]
        result = await runner.run(command)

        # After sandbox execution, delegate a mesh task with the result.
        mesh_cfg = MeshConfig(listen="127.0.0.1:0", transport="http")
        node = MeshNode(mesh_cfg, transport=HttpTransport())
        await node.start()
        try:
            card = AgentCard(
                name="worker",
                endpoint="http://127.0.0.1:0",
                skills=skills,
            )
            await node.publish_card(card)
            task = Task(
                input=f"sandbox result: {result.stdout.strip()}",
                skills=skills,
            )
            completed = await node.delegate(card, task)
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "task_id": completed.id,
                "task_status": completed.status,
                "task_input": completed.input,
            }
        finally:
            await node.stop()


class ExecuteToolTransport(FakeTransport):
    """Fake transport that routes `tools/call` to `AgentToolServer.execute`."""

    def __init__(self, server: AgentToolServer):
        super().__init__(tools=[
            McpTool(
                name="execute",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "payload": {"type": "string"},
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["echo"],
                        },
                    },
                    "required": ["payload"],
                },
            )
        ])
        self.server = server

    async def send(self, message: JsonRpcMessage) -> JsonRpcMessage:
        if message.method == "tools/call":
            args = message.params.get("arguments", {})
            SchemaCompiler.validate("execute", self.tools[0].inputSchema, args)
            result = await self.server.execute(args["payload"], args.get("skills", ["echo"]))
            return JsonRpcMessage(id=message.id, result=result)
        return await super().send(message)


def test_mcp_sandbox_mesh_pipeline():
    async def _main():
        server = AgentToolServer()
        definition = McpServerDefinition(name="agent-tools", command="echo")
        conn = McpServerConnection(
            "agent-tools",
            definition,
            ExecuteToolTransport(server),
        )
        await conn.start()

        bridge = McpBridge()
        bridge.servers["agent-tools"] = conn

        result = await bridge.call_tool(
            "execute",
            {
                "payload": "print('hello mesh')",
                "skills": ["echo"],
            },
        )
        assert result["exit_code"] == 0
        assert "hello mesh" in result["stdout"]
        assert result["task_status"] == "in_progress"
        assert "sandbox result: hello mesh" in result["task_input"]

        await bridge.close()

    _run(_main())
