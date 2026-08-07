"""MCP tool handlers for the vllm_lora domain."""

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
    'ml_vllm_lora_link': [],
    'ml_vllm_lora_list': [],
    'ml_vllm_lora_load': [],
    'ml_vllm_lora_sync': [],
    'ml_vllm_lora_unload': []
}


async def _handle(arguments, cmd_args, tool_name, execute_terradev_command):
    positional = ARGUMENTS_BY_TOOL.get(tool_name, [])
    return executor.build_cli_args(arguments, cmd_args, positional)

HANDLERS['ml_vllm_lora_link'] = _handle
HANDLERS['ml_vllm_lora_list'] = _handle
HANDLERS['ml_vllm_lora_load'] = _handle
HANDLERS['ml_vllm_lora_sync'] = _handle
HANDLERS['ml_vllm_lora_unload'] = _handle