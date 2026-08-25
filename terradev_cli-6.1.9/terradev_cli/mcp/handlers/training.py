"""MCP tool handlers for the training domain."""

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


async def _handle_moe_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    model_id = arguments["model_id"]
    gpu_type = arguments["gpu_type"]
    tp_size = arguments.get("tp_size", 8)
    backend = arguments.get("backend", "vllm")
    quantization = arguments.get("quantization", "fp8")
    dry_run = arguments.get("dry_run", False)

    cmd_args = [
        "provision",
        "--task",
        "clusters/moe-template/task.yaml",
        "--set",
        f"model_id={model_id}",
        "--set",
        f"tp_size={tp_size}",
        "--set",
        f"gpu_type={gpu_type}",
        "--set",
        f"backend={backend}",
        "--set",
        f"quantization={quantization}",
    ]
    if dry_run:
        cmd_args.append("--dry-run")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "🧬 **MoE Cluster Deployment**\n\n"
    output_text += f"**Model:** {model_id}\n"
    output_text += f"**GPU:** {gpu_type} × {tp_size} (TP={tp_size})\n"
    output_text += f"**Backend:** {backend}\n"
    output_text += f"**Quantization:** {quantization}\n"
    output_text += f"**Dry Run:** {dry_run}\n\n"
    output_text += "💰 **Auto-Applied Cost Optimizations:**\n"
    output_text += (
        "• KV Cache Offloading → CPU DRAM (up to 9x throughput)\n"
    )
    output_text += (
        "• MTP Speculative Decoding (up to 2.8x generation speed)\n"
    )
    output_text += (
        "• Sleep Mode (18-200x faster than cold restart on idle)\n"
    )
    output_text += "• Expert Load Balancing + DeepEP/DeepGEMM kernels\n\n"

    if result["success"]:
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 **Manual deployment:**\n"
        output_text += "```bash\n"
        output_text += "terradev provision --task clusters/moe-template/task.yaml \\\n"
        output_text += (
            f"  --set model_id={model_id} --set tp_size={tp_size}\n"
        )
        output_text += "```\n\n"
        output_text += "**Or via Kubernetes:**\n"
        output_text += "```bash\n"
        output_text += "kubectl apply -f clusters/moe-template/k8s/\n"
        output_text += "```"

    output_text += "\n\n🔗 **Next:** Use `lora_add` to hot-load fine-tuned adapters onto this endpoint."

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['moe_deploy'] = _handle_moe_deploy


async def _handle_lora_list(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"]
    api_key = arguments.get("api_key", "")
    cmd_args = ["lora", "list", "-e", endpoint]
    if api_key:
        cmd_args.extend(["--api-key", api_key])

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = f"🔍 **LoRA Adapters on {endpoint}**\n\n"
    output_text += output if output.strip() else "No adapters loaded.\n"
    output_text += "\n💡 Use `lora_add` to hot-load a fine-tuned adapter."

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['lora_list'] = _handle_lora_list


async def _handle_lora_add(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"]
    name = arguments["name"]
    path = arguments["path"]
    api_key = arguments.get("api_key", "")
    cmd_args = ["lora", "add", "-e", endpoint, "-n", name, "-p", path]
    if api_key:
        cmd_args.extend(["--api-key", api_key])

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    if result["success"]:
        output_text = f"✅ **Adapter '{name}' loaded on {endpoint}**\n\n"
        output_text += f'Use in API requests: `"model": "{name}"`\n\n'
        output_text += "```bash\n"
        output_text += f"curl {endpoint}/v1/chat/completions \\\n"
        output_text += (
            f'  -d \'{{"model": "{name}", "messages": [...]}}\' \n'
        )
        output_text += "```"
    else:
        output_text = f"❌ **Failed to load adapter '{name}'**\n\n{output}"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['lora_add'] = _handle_lora_add


async def _handle_lora_remove(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"]
    name = arguments["name"]
    api_key = arguments.get("api_key", "")
    cmd_args = ["lora", "remove", "-e", endpoint, "-n", name]
    if api_key:
        cmd_args.extend(["--api-key", api_key])

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    if result["success"]:
        output_text = f"✅ **Adapter '{name}' unloaded from {endpoint}**\n"
        output_text += "GPU memory freed for other adapters."
    else:
        output_text = (
            f"❌ **Failed to unload adapter '{name}'**\n\n{output}"
        )

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['lora_remove'] = _handle_lora_remove


async def _handle_train_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["train", "status"]
    if "job_id" in arguments and arguments["job_id"]:
        cmd_args.extend(["--job", arguments["job_id"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "📋 **Training Jobs**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['train_status'] = _handle_train_status


async def _handle_train_monitor(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["monitor", "--job", arguments["job_id"]]
    if "cost_rate" in arguments:
        cmd_args.extend(["--cost-rate", str(arguments["cost_rate"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "📊 **GPU Monitor**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['train_monitor'] = _handle_train_monitor


async def _handle_checkpoint_list(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["checkpoint", "list", "--job", arguments["job_id"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💾 **Checkpoints**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['checkpoint_list'] = _handle_checkpoint_list


async def _handle_checkpoint_save(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["checkpoint", "save", "--job", arguments["job_id"]]
    if "step" in arguments:
        cmd_args.extend(["--step", str(arguments["step"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💾 **Checkpoint Save**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['checkpoint_save'] = _handle_checkpoint_save


async def _handle_train_stop(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["train", "stop", "--job-id", arguments["job_id"], "-f", "json"]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "⏹️ **Training Stop**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Check final status with `train_status`, then optionally `train_resume` later."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['train_stop'] = _handle_train_stop


async def _handle_train_resume(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "train",
        "resume",
        "--job-id",
        arguments["job_id"],
        "-f",
        "json",
    ]
    if arguments.get("checkpoint_id"):
        cmd_args.extend(["--checkpoint-id", arguments["checkpoint_id"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "▶️ **Training Resume**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Monitor progress with `train_monitor`. Check `train_status` for ETA."
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += (
            "💡 **Tip:** Ensure the job has checkpoints: `checkpoint_list`"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['train_resume'] = _handle_train_resume


async def _handle_checkpoint_restore(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "checkpoint",
        "restore",
        "--job-id",
        arguments["job_id"],
        "-f",
        "json",
    ]
    if arguments.get("step"):
        cmd_args.extend(["--step", str(arguments["step"])])
    if arguments.get("checkpoint_id"):
        cmd_args.extend(["--checkpoint-id", arguments["checkpoint_id"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💾 **Checkpoint Restore**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Resume training with `train_resume` or promote with `checkpoint_promote`."
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 List available checkpoints: `checkpoint_list`"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['checkpoint_restore'] = _handle_checkpoint_restore


async def _handle_checkpoint_promote(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "checkpoint",
        "promote",
        "--job-id",
        arguments["job_id"],
        "--checkpoint-id",
        arguments["checkpoint_id"],
        "--dest",
        arguments["dest"],
        "-f",
        "json",
    ]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🏆 **Checkpoint Promoted**\n\n"
    if result["success"]:
        output_text += f"**Destination:** {arguments['dest']}\n\n"
        output_text += output
        output_text += "\n\n**suggest_action:** Deploy for inference with `infer_deploy` or `inferx_deploy`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['checkpoint_promote'] = _handle_checkpoint_promote


async def _handle_checkpoint_delete(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "checkpoint",
        "delete",
        "--job-id",
        arguments["job_id"],
        "--checkpoint-id",
        arguments["checkpoint_id"],
        "-f",
        "json",
    ]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🗑️ **Checkpoint Deleted**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['checkpoint_delete'] = _handle_checkpoint_delete


async def _handle_training_config_generate(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.training_orchestrator import (
            TrainingOrchestrator,
        )

        orch = TrainingOrchestrator()
        result = await orch.generate_config(
            name=arguments["name"],
            script=arguments["script"],
            framework=arguments.get("framework", "torchrun"),
            nodes=arguments.get("nodes"),
            gpus_per_node=arguments.get("gpus_per_node", 8),
            from_provision=arguments.get("from_provision"),
            deepspeed_config=arguments.get("deepspeed_config"),
            script_args=arguments.get("script_args"),
        )
        output_text = f"⚙️ **Training Config — {arguments['name']}**\n\n"
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

HANDLERS['training_config_generate'] = _handle_training_config_generate


async def _handle_training_launch_distributed(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.training_orchestrator import (
            TrainingOrchestrator,
        )

        orch = TrainingOrchestrator()
        skip_preflight = arguments.get("skip_preflight", False)
        # Parallel: preflight + config generation (if not skipping)
        if not skip_preflight and (
            arguments.get("nodes") or arguments.get("from_provision")
        ):
            from terradev_cli.core.preflight_validator import (
                PreflightValidator,
            )

            validator = PreflightValidator()
            config_coro = orch.generate_config(
                name=arguments["name"],
                script=arguments["script"],
                framework=arguments.get("framework", "torchrun"),
                nodes=arguments.get("nodes"),
                gpus_per_node=arguments.get("gpus_per_node", 8),
                from_provision=arguments.get("from_provision"),
            )
            preflight_coro = validator.validate(
                nodes=arguments.get("nodes"),
                from_provision=arguments.get("from_provision"),
            )
            config_result, preflight_result = await asyncio.gather(
                config_coro, preflight_coro, return_exceptions=True
            )
            output_text = f"🚀 **Distributed Training Launch — {arguments['name']}**\n\n"
            if isinstance(preflight_result, Exception):
                output_text += f"**Preflight:** ⚠️ {preflight_result}\n"
            else:
                passed = (
                    preflight_result.get("passed", True)
                    if isinstance(preflight_result, dict)
                    else True
                )
                output_text += f"**Preflight:** {'✅ Passed' if passed else '⚠️ Warnings'}\n"
            if isinstance(config_result, Exception):
                output_text += f"**Config:** ❌ {config_result}\n"
            else:
                output_text += "**Config:** ✅ Generated\n"
                output_text += f"```json\n{json.dumps(config_result, indent=2, default=str)[:3000]}\n```"
        else:
            result = await orch.launch_distributed(
                name=arguments["name"],
                script=arguments["script"],
                framework=arguments.get("framework", "torchrun"),
                nodes=arguments.get("nodes"),
                gpus_per_node=arguments.get("gpus_per_node", 8),
                from_provision=arguments.get("from_provision"),
            )
            output_text = f"🚀 **Distributed Training Launched — {arguments['name']}**\n\n"
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

HANDLERS['training_launch_distributed'] = _handle_training_launch_distributed


async def _handle_train_snapshot(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.training_monitor import TrainingMonitor

        monitor = TrainingMonitor()
        job_id = arguments["job_id"]
        cost_rate = arguments.get("cost_rate", 2.0)
        result = await monitor.snapshot(job_id=job_id, cost_rate=cost_rate)
        output_text = f"📊 **Training Snapshot — {job_id}**\n\n"
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

HANDLERS['train_snapshot'] = _handle_train_snapshot


async def _handle_train_detect_stragglers(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.training_monitor import TrainingMonitor

        monitor = TrainingMonitor()
        job_id = arguments["job_id"]
        threshold = arguments.get("threshold", 0.7)
        result = await monitor.detect_stragglers(
            job_id=job_id, threshold=threshold
        )
        stragglers = (
            result.get("stragglers", []) if isinstance(result, dict) else []
        )
        output_text = f"🐢 **Straggler Detection — {job_id}**\n\n"
        if stragglers:
            output_text += (
                f"⚠️ **{len(stragglers)} straggler(s) detected!**\n\n"
            )
        else:
            output_text += "✅ **No stragglers detected.**\n\n"
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

HANDLERS['train_detect_stragglers'] = _handle_train_detect_stragglers