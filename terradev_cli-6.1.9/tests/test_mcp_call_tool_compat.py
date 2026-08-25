"""MCP call_tool handler compatibility test.

The handler must accept both the old CallToolRequest signature used by the
existing test suite and the v1.x MCP low-level decorator (name, arguments)
signature used at runtime by clients.
"""

import pytest

pytest.importorskip("mcp")

from unittest.mock import AsyncMock, patch

from mcp.types import CallToolRequest, CallToolResult

from terradev_cli.mcp.server import handle_call_tool


@pytest.fixture
def mocked_execute():
    async def _execute(args):
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    with patch("terradev_cli.mcp.server.execute_terradev_command", _execute):
        with patch("terradev_cli.mcp.server._ensure_tools_loaded", new=AsyncMock()):
            yield _execute


@pytest.mark.asyncio
async def test_handle_call_tool_with_request_object(mocked_execute):
    """Existing tests pass a CallToolRequest directly."""
    request = CallToolRequest(
        method="tools/call", params={"name": "status", "arguments": {"live": True}}
    )
    result = await handle_call_tool(request)
    assert isinstance(result, CallToolResult)


@pytest.mark.asyncio
async def test_handle_call_tool_with_name_arguments(mocked_execute):
    """mcp v1.x low-level decorator passes (name, arguments)."""
    result = await handle_call_tool("status", {"live": True})
    assert isinstance(result, CallToolResult)
