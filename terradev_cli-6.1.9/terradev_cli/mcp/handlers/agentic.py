"""MCP tool handlers for the agentic domain."""

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
    'agent_agentic_serving_configure': [],
    'agent_agentic_serving_helm_values': [],
    'agent_agentic_serving_k8s': [],
    'agent_agentic_serving_launch_args': [],
    'agent_agentic_serving_lmcache_env': [],
    'agent_agentic_serving_show_config': [],
    'agent_cost': [],
    'agent_deploy': [],
    'agent_langchain_create_langgraph': ["graph_name"],
    'agent_langchain_create_pipeline': ["pipeline_name"],
    'agent_langchain_create_workflow': ["workflow_name"],
    'agent_langchain_test': [],
    'agent_langgraph_create_workflow': ["workflow_name"],
    'agent_langgraph_deploy': ["workflow_name"],
    'agent_langgraph_status': ["workflow_id"],
    'agent_langgraph_test': [],
    'agent_letta_chat': [],
    'agent_letta_create': [],
    'agent_letta_delete': [],
    'agent_letta_list': [],
    'agent_letta_remember': [],
    'agent_letta_status': [],
    'agent_list': [],
    'agent_plan': [],
    'agent_scale': [],
    'agent_skill_attach': [],
    'agent_skill_init': [],
    'agent_status': [],
    'agent_teardown': [],
    'agent_vector_db_down': [],
    'agent_vector_db_up': []
}


async def _handle(arguments, cmd_args, tool_name, execute_terradev_command):
    positional = ARGUMENTS_BY_TOOL.get(tool_name, [])
    return executor.build_cli_args(arguments, cmd_args, positional)

HANDLERS['agent_agentic_serving_configure'] = _handle
HANDLERS['agent_agentic_serving_helm_values'] = _handle
HANDLERS['agent_agentic_serving_k8s'] = _handle
HANDLERS['agent_agentic_serving_launch_args'] = _handle
HANDLERS['agent_agentic_serving_lmcache_env'] = _handle
HANDLERS['agent_agentic_serving_show_config'] = _handle
HANDLERS['agent_cost'] = _handle
HANDLERS['agent_deploy'] = _handle
HANDLERS['agent_langchain_create_langgraph'] = _handle
HANDLERS['agent_langchain_create_pipeline'] = _handle
HANDLERS['agent_langchain_create_workflow'] = _handle
HANDLERS['agent_langchain_test'] = _handle
HANDLERS['agent_langgraph_create_workflow'] = _handle
HANDLERS['agent_langgraph_deploy'] = _handle
HANDLERS['agent_langgraph_status'] = _handle
HANDLERS['agent_langgraph_test'] = _handle
HANDLERS['agent_letta_chat'] = _handle
HANDLERS['agent_letta_create'] = _handle
HANDLERS['agent_letta_delete'] = _handle
HANDLERS['agent_letta_list'] = _handle
HANDLERS['agent_letta_remember'] = _handle
HANDLERS['agent_letta_status'] = _handle
HANDLERS['agent_list'] = _handle
HANDLERS['agent_plan'] = _handle
HANDLERS['agent_scale'] = _handle
HANDLERS['agent_skill_attach'] = _handle
HANDLERS['agent_skill_init'] = _handle
HANDLERS['agent_status'] = _handle
HANDLERS['agent_teardown'] = _handle
HANDLERS['agent_vector_db_down'] = _handle
HANDLERS['agent_vector_db_up'] = _handle