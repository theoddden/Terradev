"""MCP tool handlers for the unsloth domain."""

import logging

try:
    from mcp.types import CallToolResult, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    CallToolResult = None
    TextContent = None

from .. import executor

logger = logging.getLogger(__name__)

HANDLERS = {}

ARGUMENTS_BY_TOOL = {
    'train_unsloth_run': [],
    'train_unsloth_start': ["agent"],
    'train_unsloth_stop': []
}


async def _handle(arguments, cmd_args, tool_name, execute_terradev_command):
    positional = ARGUMENTS_BY_TOOL.get(tool_name, [])
    return executor.build_cli_args(arguments, cmd_args, positional)

HANDLERS['train_unsloth_run'] = _handle
HANDLERS['train_unsloth_start'] = _handle
HANDLERS['train_unsloth_stop'] = _handle