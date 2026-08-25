"""Fast O(1) MCP tool dispatch router.

This module replaces the monolithic if/elif chain in server.py with a
registry-based dispatch table.  It executes the resulting Terradev CLI
commands using the helper from server.py.
"""

from typing import Any

try:
    from mcp.types import CallToolResult, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    CallToolResult = None
    TextContent = None

try:
    from .new_feature_tools import COMMAND_MAP as NEW_COMMAND_MAP
except ImportError:
    NEW_COMMAND_MAP = {}

from .registry_data import TERRADEV_COMMAND_MAP
from .handlers import HANDLERS


async def _execute_base(tool_name, execute_terradev_command):
    """Run the base terradev command for a tool with no custom handler."""
    cmd_args = TERRADEV_COMMAND_MAP.get(tool_name, []).copy()
    result = await execute_terradev_command(cmd_args)
    if result["success"]:
        return CallToolResult(
            content=[TextContent(type="text", text=result["stdout"])]
        )
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {result['stderr'] or 'Command failed'}")],
            isError=True,
        )


async def dispatch(tool_name: str, arguments: dict, execute_terradev_command) -> Any:
    """Dispatch a single MCP tool call.

    - Database / new-feature tools are handled by NEW_COMMAND_MAP.
    - Domain handlers return either a CallToolResult or the command args
      to be executed by the shared execute_terradev_command helper.
    - Tools with no custom handler fall through to the base command in
      TERRADEV_COMMAND_MAP.
    """
    # New-feature / database tools have their own handlers
    if tool_name in NEW_COMMAND_MAP:
        result = await NEW_COMMAND_MAP[tool_name](arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=result[0]["text"])]
        )

    handler = HANDLERS.get(tool_name)
    if handler is None:
        return await _execute_base(tool_name, execute_terradev_command)

    cmd_args = TERRADEV_COMMAND_MAP.get(tool_name, []).copy()
    returned = await handler(arguments, cmd_args, tool_name, execute_terradev_command)

    if isinstance(returned, CallToolResult):
        return returned

    if returned is None:
        returned = cmd_args

    result = await execute_terradev_command(returned)

    if result["success"]:
        return CallToolResult(
            content=[TextContent(type="text", text=result["stdout"])]
        )
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {result['stderr'] or 'Command failed'}")],
            isError=True,
        )
