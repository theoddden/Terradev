"""MCP tool handlers for the compute domain."""

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

_get_tf_workspace = executor._get_tf_workspace
_load_datadog_creds = executor._load_datadog_creds
_validate_config_dir = executor._validate_config_dir
execute_safe_command = executor.execute_safe_command
execute_terraform_command = executor.execute_terraform_command
execute_terraform_parallel = executor.execute_terraform_parallel
generate_inference_terraform_config = executor.generate_inference_terraform_config
generate_k8s_terraform_config = executor.generate_k8s_terraform_config

HANDLERS = {}


async def _handle_quote_gpu(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["-g", arguments["gpu_type"]])
    if "providers" in arguments:
        cmd_args.extend(["-p", arguments["providers"]])
    if arguments.get("quick"):
        cmd_args.append("--quick")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    return CallToolResult(
        content=[TextContent(type="text", text=output)],
        isError=not result["success"],
    )

HANDLERS['quote_gpu'] = _handle_quote_gpu


async def _handle_preflight_provision(arguments, cmd_args, tool_name, execute_terradev_command):
    gpu_type = arguments["gpu_type"]
    provider = arguments["provider"]
    region = arguments.get("region", "us-east-1")

    cmd_args = [
        "provision",
        "-g", gpu_type,
        "-p", provider,
        "--region", region,
        "--dry-run",
        "--auto",
    ]

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    return CallToolResult(
        content=[TextContent(type="text", text=output)],
        isError=not result["success"],
    )

HANDLERS['preflight_provision'] = _handle_preflight_provision


async def _handle_provision_gpu(arguments, cmd_args, tool_name, execute_terradev_command):
    gpu_type = arguments["gpu_type"]
    count = arguments.get("count", 1)
    providers = arguments.get(
        "providers", ["runpod", "vastai", "aws", "gcp", "azure", "tensordock", "crusoe", "digitalocean", "hyperstack", "ovhcloud", "siliconflow", "latitude", "e2enetworks", "yottalabs", "gcore"]
    )
    max_price = arguments.get("max_price")
    arguments.get("plan_only", False)

    result = await execute_terraform_parallel(
        gpu_type, count, providers, max_price
    )

    if result["success"]:
        output_text = "✅ GPU provisioning successful!\n\n"
        output_text += f"**GPU Type:** {gpu_type}\n"
        output_text += f"**Count:** {count}\n"
        output_text += f"**Providers:** {', '.join(providers)}\n"

        if result.get("terraform_outputs"):
            outputs = result["terraform_outputs"]
            if "instance_ids" in outputs:
                output_text += (
                    f"\n**Instance IDs:** {outputs['instance_ids']}\n"
                )
            if "instance_ips" in outputs:
                output_text += (
                    f"**Instance IPs:** {outputs['instance_ips']}\n"
                )
            if "provider_costs" in outputs:
                output_text += (
                    f"**Provider Costs:** {outputs['provider_costs']}\n"
                )

        output_text += "\n**State:** Managed\n"
        output_text += f"**Full Output:**\n{result['stdout']}"

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Provisioning failed: {result['stderr']}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['provision_gpu'] = _handle_provision_gpu


async def _handle_terraform_plan(arguments, cmd_args, tool_name, execute_terradev_command):
    config_dir = arguments["config_dir"]
    var_file = arguments.get("var_file")
    destroy = arguments.get("destroy", False)

    cmd = ["terraform", "plan"]
    if destroy:
        cmd.append("-destroy")
    if var_file:
        cmd.extend(["-var-file", var_file])
    cmd.append("-out=tfplan")

    result = await execute_terraform_command(cmd, config_dir)

    if result["success"]:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"✅ Plan generated:\n\n{result['stdout']}",
                )
            ]
        )
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Plan failed: {result['stderr']}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['terraform_plan'] = _handle_terraform_plan


async def _handle_terraform_apply(arguments, cmd_args, tool_name, execute_terradev_command):
    config_dir = arguments["config_dir"]
    try:
        config_dir = _validate_config_dir(config_dir)
    except ValueError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {str(e)}")],
            isError=True,
        )
    plan_file = arguments.get("plan_file", "tfplan")
    var_file = arguments.get("var_file")
    auto_approve = arguments.get("auto_approve", True)

    cmd = ["terraform", "apply"]
    if auto_approve:
        cmd.append("-auto-approve")
    if plan_file:
        cmd.append(plan_file)
    if var_file:
        cmd.extend(["-var-file", var_file])

    result = await execute_terraform_command(cmd, config_dir)

    if result["success"]:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"✅ Apply successful:\n\n{result['stdout']}",
                )
            ]
        )
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Apply failed: {result['stderr']}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['terraform_apply'] = _handle_terraform_apply


async def _handle_terraform_destroy(arguments, cmd_args, tool_name, execute_terradev_command):
    config_dir = arguments["config_dir"]
    try:
        config_dir = _validate_config_dir(config_dir)
    except ValueError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {str(e)}")],
            isError=True,
        )
    var_file = arguments.get("var_file")
    auto_approve = arguments.get("auto_approve", True)

    cmd = ["terraform", "destroy"]
    if auto_approve:
        cmd.append("-auto-approve")
    if var_file:
        cmd.extend(["-var-file", var_file])

    result = await execute_terraform_command(cmd, config_dir)

    if result["success"]:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"✅ Destroy successful:\n\n{result['stdout']}",
                )
            ]
        )
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Destroy failed: {result['stderr']}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['terraform_destroy'] = _handle_terraform_destroy


async def _handle_terraform_status(arguments, cmd_args, tool_name, execute_terradev_command):
    config_dir = arguments["config_dir"]
    show_outputs = arguments.get("show_outputs", True)

    # Fast status query using state
    output_result = await execute_terraform_command(
        ["terraform", "output", "-json"], config_dir
    )

    if output_result["success"] and show_outputs:
        try:
            outputs = json.loads(output_result["stdout"])
            output_text = "✅ Status (from state):\n\n"

            for key, value in outputs.items():
                if isinstance(value, dict) and "value" in value:
                    output_text += f"**{key}:** {value['value']}\n"

            # Also show state summary
            state_result = await execute_terraform_command(
                ["terraform", "show", "-json"], config_dir
            )
            if state_result["success"]:
                state_data = json.loads(state_result["stdout"])
                resource_count = len(
                    state_data.get("values", {})
                    .get("root_module", {})
                    .get("resources", [])
                )
                output_text += (
                    f"\n**Resources Managed:** {resource_count}\n"
                )
                output_text += "**State File:** Managed\n"

            return CallToolResult(
                content=[TextContent(type="text", text=output_text)]
            )
        except json.JSONDecodeError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"✅ Status:\n\n{output_result['stdout']}",
                    )
                ]
            )
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Status query failed: {output_result['stderr']}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['terraform_status'] = _handle_terraform_status


async def _handle_preflight_report(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.preflight_validator import PreflightValidator

        validator = PreflightValidator()
        result = await validator.full_report(
            nodes=arguments.get("nodes"),
            from_provision=arguments.get("from_provision"),
            checks=arguments.get("checks"),
        )
        output_text = "🔍 **Preflight Report**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:5000]}\n```"
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

HANDLERS['preflight_report'] = _handle_preflight_report


async def _handle_preflight_gpu_check(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.preflight_validator import (
            PreflightValidator,
            _NCU_STALL_SIGNATURES,
        )

        inference_cfg = {
            k: arguments[k]
            for k in (
                "tensor_parallel_size",
                "model_precision",
                "fp8_quant_scheme",
                "gpu_arch",
                "max_batch_size",
            )
            if k in arguments
        }
        validator = PreflightValidator(
            nodes=arguments.get("nodes"),
            inference_config=inference_cfg,
        )
        result = validator.run_quick()
        taxonomy = {
            k: {"label": v["label"], "category": v["category"]}
            for k, v in _NCU_STALL_SIGNATURES.items()
        }
        output_text = "🖥️ **GPU Preflight Check**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3500]}\n```\n\n"
        output_text += "**NCU Stall-Signature Taxonomy** (three-category framework — Khera 2026):\n"
        for stall, info in taxonomy.items():
            output_text += (
                f"- `{stall}` → **{info['label']}** [{info['category']}]\n"
            )
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

HANDLERS['preflight_gpu_check'] = _handle_preflight_gpu_check


async def _handle_preflight_network_check(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.preflight_validator import PreflightValidator

        validator = PreflightValidator()
        result = await validator.network_check(
            nodes=arguments.get("nodes"),
            from_provision=arguments.get("from_provision"),
        )
        output_text = "🌐 **Network Preflight Check**\n\n"
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

HANDLERS['preflight_network_check'] = _handle_preflight_network_check