"""MCP contract tests for core tool dispatch.

These tests verify that the MCP server's handle_call_tool correctly translates
registered tool requests into the expected terradev CLI command-line
invocations. This is the critical contract between the MCP schema and the Click
commands. We test a representative set of core tools covering the common
execution fallthrough, branch-specific execution, and direct dispatch patterns.
"""
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolRequest


class TestMCPToolDispatchContract:
    """Dynamic contract: schema arguments map to the right CLI args."""

    @pytest.fixture
    def call_tool(self):
        from terradev_cli.mcp.server import handle_call_tool

        async def _call(name, arguments):
            captured = {}

            async def fake_execute(args):
                captured["args"] = args
                return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

            request = CallToolRequest(
                method="tools/call", params={"name": name, "arguments": arguments}
            )

            with patch(
                "terradev_cli.mcp.server.execute_terradev_command", fake_execute
            ):
                with patch(
                    "terradev_cli.mcp.server._ensure_tools_loaded", AsyncMock()
                ):
                    result = await handle_call_tool(request)

            return result, captured.get("args")

        return _call

    @pytest.mark.asyncio
    async def test_quote_gpu_translates_schema_to_quote_args(self, call_tool):
        _, args = await call_tool("quote_gpu", {"gpu_type": "H100"})
        assert args == ["quote", "-g", "H100"]

    @pytest.mark.asyncio
    async def test_quote_gpu_with_providers_and_quick(self, call_tool):
        _, args = await call_tool(
            "quote_gpu",
            {"gpu_type": "A100", "providers": "aws,gcp", "quick": True},
        )
        assert args == ["quote", "-g", "A100", "-p", "aws,gcp", "--quick"]

    @pytest.mark.asyncio
    async def test_status_defaults_to_status_command(self, call_tool):
        _, args = await call_tool("status", {})
        assert args == ["status"]

    @pytest.mark.asyncio
    async def test_status_appends_live_when_requested(self, call_tool):
        _, args = await call_tool("status", {"live": True})
        assert args == ["status", "--live"]

    @pytest.mark.asyncio
    async def test_manage_instance_translates_required_fields(self, call_tool):
        _, args = await call_tool(
            "manage_instance", {"instance_id": "i-123", "action": "stop"}
        )
        assert args == ["manage", "-i", "i-123", "-a", "stop"]

    @pytest.mark.asyncio
    async def test_train_status_translates_job_id(self, call_tool):
        _, args = await call_tool("train_status", {"job_id": "job-42"})
        assert args == ["train", "status", "--job", "job-42"]

    @pytest.mark.asyncio
    async def test_checkpoint_list_translates_job_id(self, call_tool):
        _, args = await call_tool("checkpoint_list", {"job_id": "job-7"})
        assert args == ["checkpoint", "list", "--job", "job-7"]
