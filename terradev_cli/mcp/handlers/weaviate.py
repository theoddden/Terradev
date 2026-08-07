"""MCP tool handlers for the weaviate domain."""

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
    'database_weaviate_create_collection': [],
    'database_weaviate_delete_collection': [],
    'database_weaviate_hybrid_search': [],
    'database_weaviate_insert': [],
    'database_weaviate_list_collections': [],
    'database_weaviate_query': [],
    'database_weaviate_up': []
}


async def _handle(arguments, cmd_args, tool_name, execute_terradev_command):
    positional = ARGUMENTS_BY_TOOL.get(tool_name, [])
    return executor.build_cli_args(arguments, cmd_args, positional)

HANDLERS['database_weaviate_create_collection'] = _handle
HANDLERS['database_weaviate_delete_collection'] = _handle
HANDLERS['database_weaviate_hybrid_search'] = _handle
HANDLERS['database_weaviate_insert'] = _handle
HANDLERS['database_weaviate_list_collections'] = _handle
HANDLERS['database_weaviate_query'] = _handle
HANDLERS['database_weaviate_up'] = _handle