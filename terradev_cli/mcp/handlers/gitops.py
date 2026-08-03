"""MCP tool handlers for the gitops domain."""

import logging
import asyncio, base64, json, os, re

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

_load_datadog_creds = executor._load_datadog_creds
execute_safe_command = executor.execute_safe_command

HANDLERS = {}


async def _handle_gitops_init(arguments, cmd_args, tool_name, execute_terradev_command):
    repo = arguments["repo"]
    tool = arguments.get("tool", "argocd")
    provider = arguments.get("provider", "github")
    cluster = arguments.get("cluster", "production")

    cmd_args = [
        "gitops",
        "init",
        "--provider",
        provider,
        "--repo",
        repo,
        "--tool",
        tool,
        "--cluster",
        cluster,
    ]

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "🔧 **GitOps Repository Initialized**\n\n"
    if result["success"]:
        output_text += f"**Repository:** {repo}\n"
        output_text += f"**Tool:** {tool}\n"
        output_text += f"**Provider:** {provider}\n"
        output_text += f"**Cluster:** {cluster}\n\n"
        output_text += output
        output_text += "\n\n**Next steps:**\n"
        output_text += f"1. `terradev gitops bootstrap --tool {tool} --cluster {cluster}`\n"
        output_text += f"2. `terradev gitops sync --cluster {cluster} --environment prod`\n"
        output_text += (
            f"3. `terradev gitops validate --dry-run --cluster {cluster}`"
        )
    else:
        output_text += f"⚠️ {output}"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['gitops_init'] = _handle_gitops_init


async def _handle_gitops_bootstrap(arguments, cmd_args, tool_name, execute_terradev_command):
    tool = arguments["tool"]
    cluster = arguments["cluster"]
    cmd_args = ["gitops", "bootstrap", "--tool", tool, "--cluster", cluster]
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = f"🔧 **GitOps Bootstrap ({tool})**\n\n"
    if result["success"]:
        output_text += output
        output_text += f"\n\n**suggest_action:** Sync with `gitops_sync --cluster {cluster}`. Validate with `gitops_validate`."
    else:
        output_text += f"⚠️ {output}\n\n💡 Initialize first: `gitops_init`"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['gitops_bootstrap'] = _handle_gitops_bootstrap


async def _handle_gitops_sync(arguments, cmd_args, tool_name, execute_terradev_command):
    cluster = arguments["cluster"]
    cmd_args = ["gitops", "sync", "--cluster", cluster]
    if arguments.get("environment"):
        cmd_args.extend(["--environment", arguments["environment"]])
    if arguments.get("tool"):
        cmd_args.extend(["--tool", arguments["tool"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = f"🔄 **GitOps Sync — {cluster}**\n\n"
    if result["success"]:
        output_text += output
        output_text += (
            "\n\n**suggest_action:** Validate sync with `gitops_validate`."
        )
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['gitops_sync'] = _handle_gitops_sync


async def _handle_gitops_validate(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["gitops", "validate"]
    if arguments.get("cluster"):
        cmd_args.extend(["--cluster", arguments["cluster"]])
    if arguments.get("dry_run", True):
        cmd_args.append("--dry-run")
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "✅ **GitOps Validation**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['gitops_validate'] = _handle_gitops_validate