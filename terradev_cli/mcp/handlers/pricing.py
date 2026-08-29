"""MCP tool handlers for the pricing domain."""

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


async def _handle_price_intel(arguments, cmd_args, tool_name, execute_terradev_command):
    gpu_type = arguments["gpu_type"]
    days = arguments.get("days", 7)
    cmd_args = [
        "analytics",
        "--price-intel",
        "--gpu",
        gpu_type,
        "--days",
        str(days),
    ]
    if "provider" in arguments:
        cmd_args.extend(["--provider", arguments["provider"]])

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = f"📈 **GPU Price Intelligence — {gpu_type}**\n\n"
    if result["success"]:
        output_text += f"**Period:** {days} days\n"
        output_text += (
            "**Metrics:** delta (δ), gamma (γ), realized volatility (σ)\n\n"
        )
        output_text += output
    else:
        # Still useful — run a fresh quote to seed the price tick db
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 **Tip:** Price intelligence requires historical data. Seed it with:\n"
        output_text += f"   `terradev quote -g {gpu_type}` (run periodically to build history)\n\n"
        output_text += "**Metrics available after seeding:**\n"
        output_text += "- **Delta (δ):** Rate of price change ($/hr/day)\n"
        output_text += "- **Gamma (γ):** Acceleration of price change\n"
        output_text += (
            "- **Realized Volatility (σ):** Annualized price volatility\n"
        )
        output_text += "- **Cheapest Window:** Best time to provision\n"
        output_text += (
            "- **Arbitrage Spread:** Max price difference across providers"
        )

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['price_intel'] = _handle_price_intel


async def _handle_cost_analyze(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.cost_optimizer import CostOptimizer

        cost_optimizer = CostOptimizer()
        days = arguments.get("days", 30)
        result = await cost_optimizer.analyze(days=days)
        output_text = f"💰 **Cost Analysis — Last {days} Days**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
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

HANDLERS['cost_analyze'] = _handle_cost_analyze


async def _handle_cost_optimize_recommend(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.cost_optimizer import CostOptimizer

        cost_optimizer = CostOptimizer()
        result = await cost_optimizer.recommend(
            target_savings=arguments.get("target_savings"),
            constraints=arguments.get("constraints"),
        )
        output_text = "💡 **Cost Optimization Recommendations**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
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

HANDLERS['cost_optimize_recommend'] = _handle_cost_optimize_recommend


async def _handle_cost_simulate(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.cost_optimizer import CostOptimizer

        cost_optimizer = CostOptimizer()
        result = await cost_optimizer.simulate(
            scenario=arguments["scenario"],
            compare_with=arguments.get("compare_with"),
        )
        output_text = "🔮 **Cost Simulation**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
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

HANDLERS['cost_simulate'] = _handle_cost_simulate


async def _handle_price_trends(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.price_intelligence import PriceIntelligence

        intel = PriceIntelligence()
        gpu_type = arguments["gpu_type"]
        hours = arguments.get("hours", 24)
        result = await intel.get_trends(gpu_type=gpu_type, hours=hours)
        output_text = (
            f"📈 **Price Trends — {gpu_type} (last {hours}h)**\n\n"
        )
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
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

HANDLERS['price_trends'] = _handle_price_trends


async def _handle_price_spot_risk(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.price_intelligence import PriceIntelligence

        intel = PriceIntelligence()
        result = await intel.spot_risk(
            gpu_type=arguments["gpu_type"],
            provider=arguments.get("provider", "all"),
        )
        output_text = (
            f"⚠️ **Spot Risk Assessment — {arguments['gpu_type']}**\n\n"
        )
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
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

HANDLERS['price_spot_risk'] = _handle_price_spot_risk