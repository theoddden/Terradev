"""MCP tool handlers for the networking domain."""

import logging
import asyncio, json, os

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

HANDLERS = {}


async def _handle_egress_cheapest_route(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.egress_optimizer import EgressOptimizer

        egress_optimizer = EgressOptimizer()
        src = f"{arguments['source_provider']}:{arguments['source_region']}"
        dst = f"{arguments['dest_provider']}:{arguments['dest_region']}"
        size_gb = arguments["size_gb"]
        route = egress_optimizer.find_cheapest_route(src, dst, size_gb)
        output_text = "🌐 **Cheapest Egress Route**\n\n"
        output_text += (
            f"**From:** {src}\n**To:** {dst}\n**Size:** {size_gb}GB\n\n"
        )
        output_text += f"```json\n{json.dumps(route, indent=2, default=str)[:2000]}\n```\n\n"
        output_text += "**suggest_action:** Use `stage` or `egress_optimize_staging` to execute the transfer."
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['egress_cheapest_route'] = _handle_egress_cheapest_route


async def _handle_egress_optimize_staging(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.egress_optimizer import EgressOptimizer

        egress_optimizer = EgressOptimizer()
        source_uri = arguments["source_uri"]
        targets = arguments["target_regions"]
        size_gb = arguments["size_gb"]
        plan = egress_optimizer.optimize_transfer_plan(
            source_uri, targets, size_gb
        )
        output_text = "🌐 **Optimized Staging Plan**\n\n"
        output_text += f"**Source:** {source_uri}\n**Targets:** {', '.join(targets)}\n**Size:** {size_gb}GB\n\n"
        output_text += f"```json\n{json.dumps(plan, indent=2, default=str)[:2000]}\n```\n\n"
        output_text += "**suggest_action:** Execute with `stage` tool."
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['egress_optimize_staging'] = _handle_egress_optimize_staging