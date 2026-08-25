"""MCP tool handlers for the core domain."""

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
discover_local_gpus = executor.discover_local_gpus
execute_safe_command = executor.execute_safe_command

HANDLERS = {}


async def _handle_local_scan(arguments, cmd_args, tool_name, execute_terradev_command):
    local_info = await discover_local_gpus()

    output_text = "🔍 **Local GPU Scan Results**\n\n"

    if local_info["has_local_gpu"]:
        output_text += (
            f"✅ **Found {local_info['device_count']} local GPU(s)**\n"
        )
        output_text += (
            f"📊 **Total VRAM Pool:** {local_info['total_vram_gb']} GB\n\n"
        )

        output_text += "**Devices:**\n"
        for device in local_info["local_devices"]:
            output_text += f"\n• **{device['name']}**\n"
            output_text += f"  - Type: {device['type'].upper()}\n"
            output_text += f"  - VRAM: {device['vram_gb']} GB\n"
            if "compute_capability" in device:
                output_text += (
                    f"  - Compute: {device['compute_capability']}\n"
                )
            if "platform" in device:
                output_text += f"  - Platform: {device['platform']}\n"
    else:
        output_text += "❌ **No local GPUs detected**\n\n"
        output_text += (
            "💡 **Tip:** Install PyTorch or nvidia-smi for GPU detection\n"
        )
        output_text += "   - CUDA: `pip install torch`\n"
        output_text += "   - Apple Silicon: PyTorch with MPS support\n"

    output_text += "\n\n💡 **Usage:**\n"
    output_text += (
        "• Use `provision_gpu` with `--local-first` to prefer local GPUs\n"
    )
    output_text += (
        "• Cloud overflow will be used if local pool is insufficient\n"
    )

    return CallToolResult(content=[TextContent(type="text", text=output_text)])

HANDLERS['local_scan'] = _handle_local_scan


async def _handle_status(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("live"):
        cmd_args.append("--live")
    return cmd_args

HANDLERS['status'] = _handle_status


async def _handle_manage_instance(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["-i", arguments["instance_id"]])
    cmd_args.extend(["-a", arguments["action"]])
    return cmd_args

HANDLERS['manage_instance'] = _handle_manage_instance


async def _handle_analytics(arguments, cmd_args, tool_name, execute_terradev_command):
    if "days" in arguments:
        cmd_args.extend(["--days", str(arguments["days"])])
    return cmd_args

HANDLERS['analytics'] = _handle_analytics


async def _handle_setup_provider(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.append(arguments["provider"])
    if arguments.get("quick"):
        cmd_args.append("--quick")
    return cmd_args

HANDLERS['setup_provider'] = _handle_setup_provider


async def _handle_configure_provider(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--provider", arguments["provider"]])
    return cmd_args

HANDLERS['configure_provider'] = _handle_configure_provider


async def _handle_train(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["train", "start", "--script", arguments["script"]]
    if "framework" in arguments:
        cmd_args.extend(["--framework", arguments["framework"]])
    if "from_provision" in arguments:
        cmd_args.extend(["--from-provision", arguments["from_provision"]])
    elif "nodes" in arguments:
        for node in arguments["nodes"]:
            cmd_args.extend(["--node", node])
    if "gpus_per_node" in arguments:
        cmd_args.extend(
            ["--gpus-per-node", str(arguments["gpus_per_node"])]
        )
    if "script_args" in arguments:
        cmd_args.extend(["--", arguments["script_args"]])

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🚀 **Training Launch**\n\n"
    if result["success"]:
        output_text += f"**Script:** {arguments['script']}\n"
        output_text += (
            f"**Framework:** {arguments.get('framework', 'torchrun')}\n\n"
        )
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 **Tip:** Provision GPU nodes first:\n"
        output_text += "   `terradev provision -g H100 -n 4`\n"
        output_text += "   Then: `terradev train --script train.py --from-provision latest`"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['train'] = _handle_train


async def _handle_preflight(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["preflight"]
    if "from_provision" in arguments:
        cmd_args.extend(["--from-provision", arguments["from_provision"]])
    elif "nodes" in arguments:
        for node in arguments["nodes"]:
            cmd_args.extend(["--node", node])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "✅ **Preflight Validation**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['preflight'] = _handle_preflight


async def _handle_stage(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["stage", "--dataset", arguments["dataset"]]
    if arguments.get("target_regions"):
        cmd_args.extend(["--target-regions", arguments["target_regions"]])
    if arguments.get("compression"):
        cmd_args.extend(["--compression", arguments["compression"]])
    if arguments.get("plan_only"):
        cmd_args.append("--plan-only")
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "📦 **Data Staging**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Data is staged. Proceed with `train` to start training or `preflight` to validate nodes."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['stage'] = _handle_stage


async def _handle_up(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["up", "--job", arguments["job"]]
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("gpu_count"):
        cmd_args.extend(["--count", str(arguments["gpu_count"])])
    if arguments.get("ttl"):
        cmd_args.extend(["--ttl", arguments["ttl"]])
    if arguments.get("budget"):
        cmd_args.extend(["--budget", str(arguments["budget"])])
    if arguments.get("region"):
        cmd_args.extend(["--region", arguments["region"]])
    if arguments.get("fix_drift"):
        cmd_args.append("--fix-drift")
    # Cost guardrail
    gpu_type = arguments.get("gpu_type", "A100")
    gpu_count = arguments.get("gpu_count", 1)
    gpu_costs = {
        "H100": 3.50,
        "A100": 2.20,
        "A10G": 1.10,
        "L40S": 1.80,
        "L4": 0.80,
        "T4": 0.50,
    }
    est_hourly = gpu_costs.get(gpu_type, 2.00) * gpu_count
    hours = arguments.get("hours", 1.0)
    est_total = est_hourly * hours

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "⬆️ **Manifest-Cached Provision**\n\n"
    if result["success"]:
        output_text += output
        output_text += f"\n\n**estimated_cost:** ${est_hourly:.2f}/hr × {hours}h = ${est_total:.2f}\n"
        if est_total > 50:
            output_text += "⚠️ **Cost Warning:** Estimated spend exceeds $50. Monitor with `status`.\n"
        output_text += f"**suggest_action:** Infrastructure provisioned via manifest-cached DAG ({gpu_count}× {gpu_type}, ${est_hourly:.2f}/hr). Drift detection is active. Next: `preflight` to validate nodes, then `train` to launch."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['up'] = _handle_up


async def _handle_rollback(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["rollback", arguments["job_version"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "⏪ **Rollback**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Check current state with `manifests` and verify with `status`."
    else:
        output_text += f"⚠️ {output}\n\n💡 List versions: `manifests`"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['rollback'] = _handle_rollback


async def _handle_manifests(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["manifests"]
    if arguments.get("job"):
        cmd_args.extend(["--job", arguments["job"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "📋 **Cached Manifests**\n\n" + output
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['manifests'] = _handle_manifests


async def _handle_smart_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "smart-deploy",
        "--image",
        arguments["image"],
        "--workload",
        arguments["workload"],
    ]
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("budget"):
        cmd_args.extend(["--budget", str(arguments["budget"])])
    if arguments.get("option") is not None:
        cmd_args.extend(["--option", str(arguments["option"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🧠 **Smart Deployment**\n\n"
    if result["success"]:
        output_text += output
        if arguments.get("option") is None:
            output_text += "\n\n**requires_confirmation:** true\n"
        output_text += "**suggest_action:** Options ranked by cost/risk. Selection requires confirmation — the deployment graph enforces manifest checksums and drift detection before applying. Execute with `smart_deploy` and `option` parameter."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['smart_deploy'] = _handle_smart_deploy


async def _handle_run_workflow(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("template"):
        cmd_args = ["workflow", "run", "--template", arguments["template"]]
    elif arguments.get("workflow"):
        cmd_args = ["workflow", "run", arguments["workflow"]]
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="⚠️ Provide either `workflow` (YAML path) or `template` (built-in).",
                )
            ],
            isError=True,
        )
    if arguments.get("dry_run"):
        cmd_args.append("--dry-run")
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🔄 **Workflow Execution**\n\n"
    if result["success"]:
        output_text += output
        if arguments.get("dry_run"):
            output_text += "\n\n**requires_confirmation:** true\n"
            output_text += "**suggest_action:** Review the plan above. Run again without `dry_run` to execute."
        else:
            output_text += "\n\n**suggest_action:** Monitor progress with `active_context` or `train_status`."
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "**Available templates:** finetune-llama, inference-deploy, benchmark-gpu, cost-optimize"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['run_workflow'] = _handle_run_workflow


async def _handle_active_context(arguments, cmd_args, tool_name, execute_terradev_command):
    context_parts = []

    # 1. Running training jobs
    jobs_result = await execute_terradev_command(
        ["train-status", "-f", "json"]
    )
    if jobs_result["success"]:
        context_parts.append(f"**Training Jobs:**\n{jobs_result['stdout']}")
    else:
        context_parts.append("**Training Jobs:** None running")

    # 2. Active instances
    status_result = await execute_terradev_command(["status", "-f", "json"])
    if status_result["success"]:
        context_parts.append(
            f"\n**Active Instances:**\n{status_result['stdout']}"
        )
    else:
        context_parts.append("\n**Active Instances:** None")

    # 3. Cost analytics (last 7 days)
    analytics_result = await execute_terradev_command(
        ["analytics", "--days", "7", "-f", "json"]
    )
    if analytics_result["success"]:
        context_parts.append(
            f"\n**Spend (7 days):**\n{analytics_result['stdout']}"
        )
    else:
        context_parts.append("\n**Spend:** No data")

    output_text = "🏠 **Active Context — Terradev State**\n\n"
    output_text += "\n".join(context_parts)
    output_text += "\n\n**suggest_action:** "
    if (
        jobs_result["success"]
        and "running" in jobs_result["stdout"].lower()
    ):
        output_text += "You have running jobs. Monitor with `train_monitor` or check `train_status`."
    elif (
        status_result["success"]
        and status_result["stdout"].strip()
        and status_result["stdout"].strip() != "[]"
    ):
        output_text += "You have active instances. Consider `optimize` to find cheaper alternatives."
    else:
        output_text += "No active workloads. Start with `quote_gpu` to compare prices, then `provision_gpu` or `up`."
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)]
    )

HANDLERS['active_context'] = _handle_active_context


async def _handle_karpenter_install(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("version"):
        cmd_args.extend(["--version", arguments["version"]])
    if arguments.get("cluster_name"):
        cmd_args.extend(["--cluster", arguments["cluster_name"]])
    return cmd_args

HANDLERS['karpenter_install'] = _handle_karpenter_install


async def _handle_karpenter_status(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    return cmd_args

HANDLERS['karpenter_status'] = _handle_karpenter_status


async def _handle_karpenter_nodepools(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    return cmd_args

HANDLERS['karpenter_nodepools'] = _handle_karpenter_nodepools


async def _handle_karpenter_create_nodepool(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("cpu_limit"):
        cmd_args.extend(["--cpu-limit", arguments["cpu_limit"]])
    if arguments.get("memory_limit"):
        cmd_args.extend(["--memory-limit", arguments["memory_limit"]])
    return cmd_args

HANDLERS['karpenter_create_nodepool'] = _handle_karpenter_create_nodepool


async def _handle_karpenter_delete_nodepool(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend([arguments["name"]])
    if arguments.get("yes"):
        cmd_args.extend(["--yes"])
    return cmd_args

HANDLERS['karpenter_delete_nodepool'] = _handle_karpenter_delete_nodepool


async def _handle_karpenter_events(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['karpenter_events'] = _handle_karpenter_events


async def _handle_karpenter_logs(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("lines"):
        cmd_args.extend(["--lines", str(arguments["lines"])])
    return cmd_args

HANDLERS['karpenter_logs'] = _handle_karpenter_logs


async def _handle_karpenter_gpu_nodes(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    return cmd_args

HANDLERS['karpenter_gpu_nodes'] = _handle_karpenter_gpu_nodes


async def _handle_karpenter_resources(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    return cmd_args

HANDLERS['karpenter_resources'] = _handle_karpenter_resources


async def _handle_triggers_create(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("type"):
        cmd_args.extend(["--type", arguments["type"]])
    if arguments.get("source"):
        cmd_args.extend(["--source", arguments["source"]])
    if arguments.get("schedule"):
        cmd_args.extend(["--schedule", arguments["schedule"]])
    if arguments.get("condition"):
        cmd_args.extend(["--condition", arguments["condition"]])
    if arguments.get("action"):
        cmd_args.extend(["--action", arguments["action"]])
    if arguments.get("target"):
        cmd_args.extend(["--target", arguments["target"]])
    if arguments.get("enabled") is not None:
        cmd_args.extend(
            ["--enabled" if arguments["enabled"] else "--disabled"]
        )
    return cmd_args

HANDLERS['triggers_create'] = _handle_triggers_create


async def _handle_triggers_list(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("type"):
        cmd_args.extend(["--type", arguments["type"]])
    if arguments.get("status"):
        cmd_args.extend(["--status", arguments["status"]])
    return cmd_args

HANDLERS['triggers_list'] = _handle_triggers_list


async def _handle_triggers_enable(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend([arguments["name"]])
    if arguments.get("enabled") is not None:
        cmd_args.append("--enable" if arguments["enabled"] else "--disable")
    return cmd_args

HANDLERS['triggers_enable'] = _handle_triggers_enable


async def _handle_triggers_execute(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend([arguments["name"]])
    if arguments.get("dry_run"):
        cmd_args.append("--dry-run")
    return cmd_args

HANDLERS['triggers_execute'] = _handle_triggers_execute


async def _handle_triggers_history(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['triggers_history'] = _handle_triggers_history


async def _handle_environments_create(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("type"):
        cmd_args.extend(["--type", arguments["type"]])
    if arguments.get("cluster"):
        cmd_args.extend(["--cluster", arguments["cluster"]])
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    if arguments.get("approval_required") is not None:
        cmd_args.extend(
            [
                (
                    "--approval-required"
                    if arguments["approval_required"]
                    else "--no-approval"
                )
            ]
        )
    if arguments.get("policies"):
        cmd_args.extend(["--policies", ",".join(arguments["policies"])])
    return cmd_args

HANDLERS['environments_create'] = _handle_environments_create


async def _handle_environments_promote(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("artifact"):
        cmd_args.extend(["--artifact", arguments["artifact"]])
    if arguments.get("from_env"):
        cmd_args.extend(["--from", arguments["from_env"]])
    if arguments.get("to_env"):
        cmd_args.extend(["--to", arguments["to_env"]])
    if arguments.get("version"):
        cmd_args.extend(["--version", arguments["version"]])
    if arguments.get("approval_comment"):
        cmd_args.extend(["--comment", arguments["approval_comment"]])
    if arguments.get("dry_run"):
        cmd_args.append("--dry-run")
    return cmd_args

HANDLERS['environments_promote'] = _handle_environments_promote


async def _handle_environments_list(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("cluster"):
        cmd_args.extend(["--cluster", arguments["cluster"]])
    return cmd_args

HANDLERS['environments_list'] = _handle_environments_list


async def _handle_environments_approve(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("request_id"):
        cmd_args.extend([arguments["request_id"]])
    if arguments.get("action"):
        cmd_args.extend(["--action", arguments["action"]])
    if arguments.get("comment"):
        cmd_args.extend(["--comment", arguments["comment"]])
    return cmd_args

HANDLERS['environments_approve'] = _handle_environments_approve


async def _handle_environments_history(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("environment"):
        cmd_args.extend(["--environment", arguments["environment"]])
    if arguments.get("artifact"):
        cmd_args.extend(["--artifact", arguments["artifact"]])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['environments_history'] = _handle_environments_history


async def _handle_lineage_trace(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("artifact"):
        cmd_args.extend([arguments["artifact"]])
    if arguments.get("depth"):
        cmd_args.extend(["--depth", str(arguments["depth"])])
    if arguments.get("direction"):
        cmd_args.extend(["--direction", arguments["direction"]])
    return cmd_args

HANDLERS['lineage_trace'] = _handle_lineage_trace


async def _handle_lineage_diff(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("from_artifact"):
        cmd_args.extend(["--from", arguments["from_artifact"]])
    if arguments.get("to_artifact"):
        cmd_args.extend(["--to", arguments["to_artifact"]])
    if arguments.get("show_changes") is not None:
        cmd_args.extend(
            [
                (
                    "--show-changes"
                    if arguments["show_changes"]
                    else "--no-changes"
                )
            ]
        )
    return cmd_args

HANDLERS['lineage_diff'] = _handle_lineage_diff


async def _handle_lineage_export(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("artifact"):
        cmd_args.extend([arguments["artifact"]])
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    if arguments.get("include_metadata") is not None:
        cmd_args.extend(
            [
                (
                    "--metadata"
                    if arguments["include_metadata"]
                    else "--no-metadata"
                )
            ]
        )
    return cmd_args

HANDLERS['lineage_export'] = _handle_lineage_export


async def _handle_lineage_checkpoint(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("checkpoint_id"):
        cmd_args.extend([arguments["checkpoint_id"]])
    if arguments.get("show_dependencies") is not None:
        cmd_args.extend(
            [
                (
                    "--show-deps"
                    if arguments["show_dependencies"]
                    else "--no-deps"
                )
            ]
        )
    return cmd_args

HANDLERS['lineage_checkpoint'] = _handle_lineage_checkpoint


async def _handle_lineage_graph(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("artifact"):
        cmd_args.extend([arguments["artifact"]])
    if arguments.get("format"):
        cmd_args.extend(["--format", arguments["format"]])
    if arguments.get("include_versions") is not None:
        cmd_args.extend(
            [
                (
                    "--versions"
                    if arguments["include_versions"]
                    else "--no-versions"
                )
            ]
        )
    return cmd_args

HANDLERS['lineage_graph'] = _handle_lineage_graph


async def _handle_migrate_plan(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("from_provider"):
        cmd_args.extend(["--from", arguments["from_provider"]])
    if arguments.get("to_provider"):
        cmd_args.extend(["--to", arguments["to_provider"]])
    if arguments.get("instances"):
        cmd_args.extend(["--instances", ",".join(arguments["instances"])])
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("region"):
        cmd_args.extend(["--region", arguments["region"]])
    if arguments.get("dry_run") is not None:
        cmd_args.extend(
            ["--dry-run" if arguments["dry_run"] else "--execute"]
        )
    return cmd_args

HANDLERS['migrate_plan'] = _handle_migrate_plan


async def _handle_migrate_execute(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("plan_id"):
        cmd_args.extend(["--plan", arguments["plan_id"]])
    if arguments.get("batch_size"):
        cmd_args.extend(["--batch-size", str(arguments["batch_size"])])
    if arguments.get("wait_for_health") is not None:
        cmd_args.extend(
            [
                (
                    "--wait-health"
                    if arguments["wait_for_health"]
                    else "--no-wait"
                )
            ]
        )
    if arguments.get("rollback_on_failure") is not None:
        cmd_args.extend(
            [
                (
                    "--rollback"
                    if arguments["rollback_on_failure"]
                    else "--no-rollback"
                )
            ]
        )
    return cmd_args

HANDLERS['migrate_execute'] = _handle_migrate_execute


async def _handle_migrate_status(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("plan_id"):
        cmd_args.extend(["--plan", arguments["plan_id"]])
    if arguments.get("show_details") is not None:
        cmd_args.extend(
            ["--details" if arguments["show_details"] else "--summary"]
        )
    return cmd_args

HANDLERS['migrate_status'] = _handle_migrate_status


async def _handle_migrate_rollback(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("plan_id"):
        cmd_args.extend(["--plan", arguments["plan_id"]])
    if arguments.get("force"):
        cmd_args.append("--force")
        if arguments.get("yes"):
            cmd_args.append("--yes")
    return cmd_args

HANDLERS['migrate_rollback'] = _handle_migrate_rollback


async def _handle_migrate_compare(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("plan_id"):
        cmd_args.extend(["--plan", arguments["plan_id"]])
    if arguments.get("metrics"):
        cmd_args.extend(["--metrics", ",".join(arguments["metrics"])])
    if arguments.get("duration"):
        cmd_args.extend(["--duration", arguments["duration"]])
    return cmd_args

HANDLERS['migrate_compare'] = _handle_migrate_compare