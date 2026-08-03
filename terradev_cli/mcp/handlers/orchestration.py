"""MCP tool handlers for the orchestration domain."""

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


async def _handle_orchestrator_start(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["orchestrator-start"]
    if arguments.get("gpu_id") is not None:
        cmd_args.extend(["--gpu-id", str(arguments["gpu_id"])])
    if arguments.get("memory_gb"):
        cmd_args.extend(["--memory-gb", str(arguments["memory_gb"])])
    if arguments.get("policy"):
        cmd_args.extend(["--policy", arguments["policy"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🎛️ **Model Orchestrator Started**\n\n"
    if result["success"]:
        output_text += output
        gpu_id = arguments.get("gpu_id", 0)
        memory_gb = arguments.get("memory_gb", 80)
        policy = arguments.get("policy", "billing_optimized")
        output_text += f"\n\n**suggest_action:** Orchestrator is enforcing memory invariants on GPU {gpu_id} ({memory_gb}GB, {policy} policy). Register models to enter the scheduling graph: `orchestrator_register`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_start'] = _handle_orchestrator_start


async def _handle_orchestrator_register(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "orchestrator-register",
        arguments["model_id"],
        arguments["model_path"],
    ]
    if arguments.get("framework"):
        cmd_args.extend(["--framework", arguments["framework"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = f"📝 **Model Registered: {arguments['model_id']}**\n\n"
    if result["success"]:
        output_text += output
        output_text += f"\n\n**suggest_action:** Model `{arguments['model_id']}` is now in the scheduling graph. The orchestrator will enforce memory and cost constraints on load. Next: `orchestrator_load`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_register'] = _handle_orchestrator_register


async def _handle_orchestrator_load(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["orchestrator-load", arguments["model_id"]]
    if arguments.get("force"):
        cmd_args.append("--force")
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = f"📥 **Model Loaded: {arguments['model_id']}**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Model loaded within memory budget. The orchestrator will auto-evict if idle >15min under billing-optimized policy. Verify inference: `orchestrator_infer`."
    else:
        output_text += f"⚠️ Load blocked: {output}\n\nThe cost scaler or memory invariant rejected this load. Use `--force` to override constraints, or free memory with `orchestrator_evict`."
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_load'] = _handle_orchestrator_load


async def _handle_orchestrator_evict(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["orchestrator-evict", arguments["model_id"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = (
        f"📤 **Model Evicted: {arguments['model_id']}**\n\n" + output
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_evict'] = _handle_orchestrator_evict


async def _handle_orchestrator_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["orchestrator-status"]
    if arguments.get("model_id"):
        cmd_args.append(arguments["model_id"])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🎛️ **Orchestrator Status**\n\n"
    if result["success"]:
        output_text += output
        # Agent recommendations based on output
        if "utilization" in output.lower():
            output_text += "\n\n**recommend:** "
            if "90%" in output or "95%" in output or "100%" in output:
                output_text += "Memory invariant near threshold. Eviction policy will auto-reclaim from lowest-priority idle models. Manual override: `orchestrator_evict`."
            elif "10%" in output or "15%" in output or "20%" in output:
                output_text += "Memory underutilized — scheduling graph has capacity. Load more models with `orchestrator_load` to increase warm pool coverage."
            else:
                output_text += "Memory utilization within policy bounds. The orchestrator is maintaining headroom for burst loads."
    else:
        output_text += f"⚠️ {output}\n\n💡 Start orchestrator first: `orchestrator_start`"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_status'] = _handle_orchestrator_status


async def _handle_orchestrator_infer(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["orchestrator-infer", arguments["model_id"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = (
        f"⚡ **Inference Test: {arguments['model_id']}**\n\n" + output
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['orchestrator_infer'] = _handle_orchestrator_infer


async def _handle_warm_pool_start(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["warm-pool-start"]
    if arguments.get("strategy"):
        cmd_args.extend(["--strategy", arguments["strategy"]])
    if arguments.get("max_warm"):
        cmd_args.extend(["--max-warm", str(arguments["max_warm"])])
    if arguments.get("min_warm"):
        cmd_args.extend(["--min-warm", str(arguments["min_warm"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🔥 **Warm Pool Started**\n\n"
    if result["success"]:
        output_text += output
        strategy = arguments.get("strategy", "traffic_based")
        max_warm = arguments.get("max_warm", 10)
        min_warm = arguments.get("min_warm", 3)
        output_text += f"\n\n**suggest_action:** Warm pool enforcing [{min_warm}, {max_warm}] model bounds under {strategy} policy. Register models to enter the warming graph: `warm_pool_register`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['warm_pool_start'] = _handle_warm_pool_start


async def _handle_warm_pool_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["warm-pool-status"]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🔥 **Warm Pool Status**\n\n"
    if result["success"]:
        output_text += output
        # Agent recommendation
        output_text += "\n\n**recommend:** "
        if "hit rate" in output.lower():
            output_text += "The warm pool enforces model bounds and eviction policy. If hit rate is below 80%, the pool's constraints may be too aggressive — consider increasing `max_warm` or switching strategy."
        else:
            output_text += "Warm pool is enforcing its scheduling invariants. Cold starts are being minimized within the configured bounds."
    else:
        output_text += (
            f"⚠️ {output}\n\n💡 Start warm pool first: `warm_pool_start`"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['warm_pool_status'] = _handle_warm_pool_status


async def _handle_cost_scaler_start(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["cost-scaler-start"]
    if arguments.get("strategy"):
        cmd_args.extend(["--strategy", arguments["strategy"]])
    if arguments.get("budget"):
        cmd_args.extend(["--budget", str(arguments["budget"])])
    if arguments.get("cost_per_gb"):
        cmd_args.extend(["--cost-per-gb", str(arguments["cost_per_gb"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💰 **Cost Scaler Started**\n\n"
    if result["success"]:
        output_text += output
        strategy = arguments.get("strategy", "balance_cost_latency")
        budget = arguments.get("budget", 15.0)
        output_text += f"\n\n**suggest_action:** Cost scaler enforcing ${budget:.2f}/hr budget under {strategy} policy. The scaler will block loads that exceed budget constraints. Monitor: `cost_scaler_status`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['cost_scaler_start'] = _handle_cost_scaler_start


async def _handle_cost_scaler_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["cost-scaler-status"]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💰 **Cost Scaler Status**\n\n"
    if result["success"]:
        output_text += output
        # Agent recommendations
        output_text += "\n\n**recommend:** "
        if "budget" in output.lower() and (
            "exceed" in output.lower() or "over" in output.lower()
        ):
            output_text += "⚠️ Budget constraint active: the scaler will block new model loads until utilization drops below 80%. Reduce spend with `orchestrator_evict` or switch to `minimize_cost` strategy."
        elif "under" in output.lower():
            output_text += "Budget constraint has headroom. The scaler permits new loads within the remaining budget envelope."
        else:
            output_text += "Cost invariants holding. The scaler is maintaining spend within configured bounds."
    else:
        output_text += (
            f"⚠️ {output}\n\n💡 Start cost scaler first: `cost_scaler_start`"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['cost_scaler_status'] = _handle_cost_scaler_status