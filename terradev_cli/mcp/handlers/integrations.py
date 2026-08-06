"""MCP tool handlers for the integrations domain."""

import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

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

async def _handle_langfuse_configure(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(
        [
            "--public-key",
            arguments["public_key"],
            "--secret-key",
            arguments["secret_key"],
        ]
    )
    if arguments.get("host"):
        cmd_args.extend(["--host", arguments["host"]])
    return cmd_args

HANDLERS['langfuse_configure'] = _handle_langfuse_configure

async def _handle_langfuse_traces(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--format", "json"])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    return cmd_args

HANDLERS['langfuse_traces'] = _handle_langfuse_traces

async def _handle_langfuse_trace(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.append(arguments["trace_id"])
    cmd_args.extend(["--format", "json"])
    return cmd_args

HANDLERS['langfuse_trace'] = _handle_langfuse_trace

async def _handle_langfuse_scores(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--format", "json"])
    if arguments.get("trace_id"):
        cmd_args.extend(["--trace-id", arguments["trace_id"]])
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['langfuse_scores'] = _handle_langfuse_scores

async def _handle_langfuse_score(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(
        [
            "--trace-id",
            arguments["trace_id"],
            "--name",
            arguments["name"],
            "--value",
            str(arguments["value"]),
        ]
    )
    if arguments.get("observation_id"):
        cmd_args.extend(["--observation-id", arguments["observation_id"]])
    if arguments.get("comment"):
        cmd_args.extend(["--comment", arguments["comment"]])
    return cmd_args

HANDLERS['langfuse_score'] = _handle_langfuse_score

async def _handle_langfuse_datasets(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--format", "json"])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['langfuse_datasets'] = _handle_langfuse_datasets

async def _handle_langfuse_export_training_data(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("min_score") is not None:
        cmd_args.extend(["--min-score", str(arguments["min_score"])])
    if arguments.get("score_name"):
        cmd_args.extend(["--score-name", arguments["score_name"]])
    return cmd_args

HANDLERS['langfuse_export_training_data'] = _handle_langfuse_export_training_data

async def _handle_langfuse_quality(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--format", "json"])
    if arguments.get("score_name"):
        cmd_args.extend(["--score-name", arguments["score_name"]])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['langfuse_quality'] = _handle_langfuse_quality

async def _handle_langfuse_otel_env(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("project"):
        cmd_args.extend(["--project", arguments["project"]])
    return cmd_args

HANDLERS['langfuse_otel_env'] = _handle_langfuse_otel_env

async def _handle_langfuse_k8s(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    return cmd_args

HANDLERS['langfuse_k8s'] = _handle_langfuse_k8s

