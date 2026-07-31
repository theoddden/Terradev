"""Chaos and subprocess-routing tests for the MCP execution layer."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")


class TestExecuteTerradevCommandChaos:
    """Exercise execute_terradev_command failure modes."""

    @pytest.fixture
    async def execute(self):
        from terradev_cli.mcp.server import execute_terradev_command
        return execute_terradev_command

    @pytest.mark.asyncio
    async def test_successful_command(self, execute):
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"ok output\n", b"")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            result = await execute(["quote", "gpu"])

        assert result["success"] is True
        assert result["stdout"] == "ok output"
        assert result["returncode"] == 0
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_zero_returncode(self, execute):
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate.return_value = (b"", b"something failed")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await execute(["provision"])

        assert result["success"] is False
        assert result["returncode"] == 1
        assert "something failed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_command_not_found(self, execute):
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            result = await execute(["missing"])

        assert result["success"] is False
        assert result["returncode"] == -1
        assert "not found" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_killed_process(self, execute):
        proc = AsyncMock()
        proc.returncode = -9
        proc.communicate.return_value = (b"", b"Killed")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await execute(["train"])

        assert result["success"] is False
        assert result["returncode"] == -9

    @pytest.mark.asyncio
    async def test_malformed_stdout(self, execute):
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"\xff\xfe", b"")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await execute(["status"])

        assert result["success"] is False
        assert result["returncode"] == -1
        assert "Unexpected error" in result["stderr"]


class TestToolSubprocessRouting:
    """Verify tool calls route to the expected terradev commands."""

    @pytest.mark.asyncio
    async def test_quote_gpu_routes_to_quote_command(self):
        from terradev_cli.mcp.server import handle_call_tool, _ensure_tools_loaded
        from mcp.types import CallToolRequest

        _ensure_tools_loaded = AsyncMock()
        captured = {}

        async def fake_execute(args):
            captured["args"] = args
            return {"success": True, "stdout": "", "stderr": "", "returncode": 0}

        request = CallToolRequest(
            method="tools/call",
            params={"name": "quote_gpu", "arguments": {"gpu_type": "A100"}},
        )

        with patch("terradev_cli.mcp.server.execute_terradev_command", fake_execute):
            with patch("terradev_cli.mcp.server._ensure_tools_loaded", _ensure_tools_loaded):
                await handle_call_tool(request)

        assert captured["args"][:2] == ["quote", "-g"]
        assert captured["args"][2] == "A100"
