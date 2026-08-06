"""MCP tool handlers for the inference domain."""

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
execute_safe_command = executor.execute_safe_command
execute_terraform_command = executor.execute_terraform_command
generate_inference_terraform_config = executor.generate_inference_terraform_config

HANDLERS = {}


async def _handle_inferx_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    model = arguments["model"]
    gpu_type = arguments["gpu_type"]
    endpoint_name = arguments.get("endpoint_name")
    use_terraform = arguments.get("use_terraform", True)

    if use_terraform:
        # Use persistent workspace so state survives for endpoint teardown
        safe_ep = (
            (endpoint_name or model).replace("/", "-").replace(":", "-")
        )
        ws_dir = _get_tf_workspace(f"infer-{safe_ep}")
        try:
            # Generate inference Terraform configuration
            inference_config = generate_inference_terraform_config(
                model, gpu_type, endpoint_name
            )

            # Write configuration files
            main_tf_path = os.path.join(ws_dir, "main.tf")
            with open(main_tf_path, "w") as f:
                f.write(inference_config)

            # Initialize and apply Terraform
            init_result = await execute_terraform_command(
                ["terraform", "init"], ws_dir
            )
            if not init_result["success"]:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Terraform init failed: {init_result['stderr']}",
                        )
                    ],
                    isError=True,
                )

            apply_result = await execute_terraform_command(
                ["terraform", "apply", "-auto-approve"], ws_dir
            )

            if apply_result["success"]:
                output_text = (
                    "✅ Inference endpoint deployed via Terraform!\n\n"
                )
                output_text += f"**Model:** {model}\n"
                output_text += f"**GPU Type:** {gpu_type}\n"
                output_text += f"**Endpoint Name:** {endpoint_name or 'auto-generated'}\n"
                output_text += (
                    f"\n**Terraform State:** Persisted at {ws_dir}\n"
                )
                output_text += f"**Full Output:**\n{apply_result['stdout']}"

                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Terraform apply failed: {apply_result['stderr']}",
                        )
                    ],
                    isError=True,
                )
        except Exception as e:  # noqa: BLE001
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Inference Terraform deployment failed: {str(e)}",
                    )
                ],
                isError=True,
            )
    else:
        # Fall back to regular terradev command
        cmd_args.extend(["--model", model])
        cmd_args.extend(["--gpu-type", gpu_type])
    return cmd_args

HANDLERS['inferx_deploy'] = _handle_inferx_deploy


async def _handle_infer_route(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inference", "route"]
    if "model" in arguments:
        cmd_args.extend(["--model", arguments["model"]])
    strategy = arguments.get("strategy", "latency")
    cmd_args.extend(["--strategy", strategy])
    if arguments.get("measure"):
        cmd_args.append("--measure")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "🧠 **Semantic Inference Routing**\n\n"
    if result["success"]:
        output_text += f"**Strategy:** {strategy}\n"
        output_text += "**Signals:** modality, complexity, domain, language, safety, keywords\n"
        output_text += "**NUMA scoring:** enabled\n\n"
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += (
            "💡 **Tip:** Register inference endpoints first with:\n"
        )
        output_text += "   `terradev inference deploy --provider runpod --model <model>`\n"
        output_text += "   Then route with: `terradev inference route --strategy latency`"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['infer_route'] = _handle_infer_route


async def _handle_infer_route_disagg(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inference", "route", "--disagg"]
    cmd_args.extend(["--model", arguments["model"]])
    if arguments.get("check_health", True):
        cmd_args.append("--check")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = (
        "⚡ **Disaggregated Prefill/Decode Routing (DistServe)**\n\n"
    )
    if result["success"]:
        output_text += f"**Model:** {arguments['model']}\n"
        output_text += "**Architecture:** DistServe — PREFILL (compute-bound) → DECODE (memory-bound)\n"
        output_text += (
            "**KV Cache Handoff:** tracked via PrefillDecodeTracker\n\n"
        )
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 **Tip:** Disaggregated routing requires endpoints tagged with phase:\n"
        output_text += "   PREFILL endpoints: high-FLOPS GPUs (H100 SXM)\n"
        output_text += (
            "   DECODE endpoints: high-bandwidth GPUs (H200, MI300X)\n"
        )
        output_text += "   Register with: `terradev inference deploy --phase prefill --gpu H100`"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['infer_route_disagg'] = _handle_infer_route_disagg


async def _handle_infer_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inference", "status"]
    if arguments.get("check"):
        cmd_args.append("--check")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "📊 **Inference Endpoint Status**\n\n"
    if result["success"]:
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += (
            "💡 No inference endpoints registered. Deploy one with:\n"
        )
        output_text += "   `terradev inference deploy --provider runpod --model <model> --gpu H100`"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['infer_status'] = _handle_infer_status


async def _handle_infer_failover(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inference", "failover"]
    if arguments.get("dry_run"):
        cmd_args.append("--dry-run")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "🔄 **Inference Auto-Failover**\n\n"
    if result["success"]:
        output_text += output
    else:
        output_text += f"⚠️ {output}\n\n"
        output_text += "💡 Register backup endpoints with:\n"
        output_text += "   `terradev inference deploy --provider <backup> --model <model> --backup`"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['infer_failover'] = _handle_infer_failover


async def _handle_gpu_topology(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inference", "topology"]
    gpu_arch = arguments.get("gpu_arch", "auto")
    if gpu_arch and gpu_arch != "auto":
        cmd_args.extend(["--arch", gpu_arch])
    if arguments.get("generate_env", True):
        cmd_args.append("--env")

    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]

    output_text = "🔬 **GPU NUMA Topology Report**\n\n"
    if result["success"]:
        output_text += output
        if arguments.get("generate_env", True):
            output_text += (
                "\n\n**XCD-Aware Environment Variables Generated**\n"
            )
            output_text += "Apply these to your vLLM/SGLang process for optimal attention kernel performance.\n"
    else:
        # Provide useful topology info even without live GPUs
        output_text += f"⚠️ {output}\n\n"
        output_text += "📋 **Reference: Intra-GPU NUMA Topology**\n\n"
        output_text += "| GPU | XCDs | HBM | Architecture |\n"
        output_text += "|-----|------|-----|-------------|\n"
        output_text += "| MI300X | 8 XCDs | 192GB HBM3 | CDNA3 chiplet |\n"
        output_text += "| MI300A | 6 XCDs | 128GB HBM3 | CDNA3 APU |\n"
        output_text += "| H200 | 1 (unified) | 141GB HBM3e | Hopper |\n"
        output_text += "| H100 SXM | 1 (unified) | 80GB HBM3 | Hopper |\n"
        output_text += "| A100 | 1 (unified) | 80GB HBM2e | Ampere |\n\n"
        output_text += "💡 **XCD-aware env vars for MI300X:**\n"
        output_text += "```\n"
        output_text += "AITER_XCD_AWARE_ATTENTION=1\n"
        output_text += "CK_BLOCK_MAPPING_POLICY=xcd_aware\n"
        output_text += "NCCL_INTRA_GPU_NUMA=1\n"
        output_text += "```"

    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['gpu_topology'] = _handle_gpu_topology


async def _handle_infer_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    model_path = arguments["model_path"]
    name = arguments["name"]
    # Cost guardrail: estimate and warn
    est_cost = 0.0
    gpu_type = arguments.get("gpu_type", "A100")
    gpu_costs = {
        "H100": 3.50,
        "A100": 2.20,
        "A10G": 1.10,
        "L40S": 1.80,
        "L4": 0.80,
        "T4": 0.50,
    }
    est_cost = gpu_costs.get(gpu_type, 2.00) * arguments.get(
        "max_workers", 3
    )

    if arguments.get("dry_run"):
        output_text = "📋 **Inference Deployment Plan (Dry Run)**\n\n"
        output_text += f"**Model:** {model_path}\n"
        output_text += f"**Endpoint:** {name}\n"
        output_text += f"**GPU:** {gpu_type}\n"
        output_text += f"**Workers:** {arguments.get('min_workers', 0)}-{arguments.get('max_workers', 3)}\n"
        output_text += (
            f"**Idle Timeout:** {arguments.get('idle_timeout', 300)}s\n"
        )
        output_text += f"**Estimated Max Cost:** ${est_cost:.2f}/hr\n\n"
        output_text += "**requires_confirmation:** true\n"
        output_text += f"**estimated_cost:** ${est_cost:.2f}/hr (max {arguments.get('max_workers', 3)} workers × ${gpu_costs.get(gpu_type, 2.00):.2f}/hr)\n\n"
        budget_rate = gpu_costs.get(gpu_type, 2.00)
        output_text += f"**suggest_action:** Dry run complete: ${est_cost:.2f}/hr for {arguments.get('max_workers', 3)} workers. This requires confirmation — the cost scaler enforces a ${budget_rate:.2f}/hr-per-worker guardrail. Call `infer_deploy` without `dry_run` to execute."
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )

    cmd_args = ["infer-deploy", model_path, "--name", name]
    if arguments.get("provider"):
        cmd_args.extend(["--provider", arguments["provider"]])
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if "idle_timeout" in arguments:
        cmd_args.extend(["--idle-timeout", str(arguments["idle_timeout"])])
    if arguments.get("cost_optimize"):
        cmd_args.append("--cost-optimize")
    if "min_workers" in arguments:
        cmd_args.extend(["--min-workers", str(arguments["min_workers"])])
    if "max_workers" in arguments:
        cmd_args.extend(["--max-workers", str(arguments["max_workers"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🚀 **Inference Deployment**\n\n"
    if result["success"]:
        output_text += output
        output_text += f"\n\n**estimated_cost:** ${est_cost:.2f}/hr (max)\n"
        output_text += f"**suggest_action:** Deployment active at ${est_cost:.2f}/hr (max). The orchestrator will enforce idle timeout and auto-scale constraints. Monitor: `infer_status`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['infer_deploy'] = _handle_infer_deploy


async def _handle_inferx_configure(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inferx", "configure", "--api-key", arguments["api_key"]]
    if arguments.get("endpoint"):
        cmd_args.extend(["--endpoint", arguments["endpoint"]])
    if arguments.get("region"):
        cmd_args.extend(["--region", arguments["region"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "🔑 **InferX Configured**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Deploy a model with `inferx_deploy` or check quotes with `inferx_quote`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['inferx_configure'] = _handle_inferx_configure


async def _handle_inferx_delete(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inferx", "delete", "--model-id", arguments["model_id"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = (
        f"🗑️ **InferX Deployment Deleted: {arguments['model_id']}**\n\n"
        + output
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['inferx_delete'] = _handle_inferx_delete


async def _handle_inferx_usage(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inferx", "usage"]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "📊 **InferX Usage**\n\n"
    if result["success"]:
        output_text += output
        output_text += (
            "\n\n**suggest_action:** Optimize costs with `inferx_optimize`."
        )
    else:
        output_text += (
            f"⚠️ {output}\n\n💡 Configure InferX first: `inferx_configure`"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['inferx_usage'] = _handle_inferx_usage


async def _handle_inferx_quote(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["inferx", "quote"]
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("region"):
        cmd_args.extend(["--region", arguments["region"]])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "💰 **InferX Pricing Quote**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Deploy with `inferx_deploy` at these rates."
    else:
        output_text += (
            f"⚠️ {output}\n\n💡 Configure InferX first: `inferx_configure`"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['inferx_quote'] = _handle_inferx_quote


async def _handle_vllm_start(arguments, cmd_args, tool_name, execute_terradev_command):
    ip = arguments["instance_ip"]
    model = arguments["model"]
    port = arguments.get("port", 8000)
    tp = arguments.get("tp_size", 1)
    mem = arguments.get("gpu_memory_utilization", 0.9)
    user = arguments.get("ssh_user", "root")
    key = arguments.get("ssh_key")
    api_key = arguments.get("api_key")

    # Validate inputs to prevent injection
    if not re.match(r"^[a-zA-Z0-9_\-\.\/:]+$", model):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid model name")
            ],
            isError=True,
        )
    if not re.match(r"^[0-9]+$", str(port)):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid port")],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ip):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid IP address")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid username")],
            isError=True,
        )

    cmd_parts = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(mem),
        "--tensor-parallel-size",
        str(tp),
        "--enable-sleep-mode",
        "--kv-connector",
        "offloading",
    ]
    if api_key:
        cmd_parts.extend(["--api-key", api_key])
    exec_line = " ".join(cmd_parts)
    service = f"""[Unit]\nDescription=vLLM {model}\nAfter=network.target\n[Service]\nType=simple\nExecStart={exec_line}\nRestart=always\nRestartSec=10\nEnvironment=VLLM_SERVER_DEV_MODE=1\n[Install]\nWantedBy=multi-user.target"""
    service_b64 = base64.b64encode(service.encode()).decode()

    # Build SSH command safely using shell=True only for base64 decode chain
    setup = f"echo {service_b64} | base64 -d > /etc/systemd/system/vllm.service && systemctl daemon-reload && systemctl enable vllm && systemctl start vllm && sleep 5 && systemctl status vllm"
    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if key:
        ssh_args.extend(["-i", key])
    ssh_args.extend([f"{user}@{ip}", setup])

    result = await execute_safe_command(ssh_args)
    if result["success"]:
        output_text = f"✅ **vLLM Server Started**\n\n**Model:** {model}\n**Endpoint:** http://{ip}:{port}/v1\n**TP:** {tp}\n**Sleep Mode:** enabled\n**KV Offloading:** enabled\n\n{result['stdout']}\n\n"
        output_text += "**suggest_action:** Test with `vllm_inference`. Manage power with `vllm_sleep`/`vllm_wake`."
    else:
        output_text = f"❌ **Failed to start vLLM**\n\n{result['stderr']}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['vllm_start'] = _handle_vllm_start


async def _handle_vllm_stop(arguments, cmd_args, tool_name, execute_terradev_command):
    ip = arguments["instance_ip"]
    user = arguments.get("ssh_user", "root")
    key = arguments.get("ssh_key")

    # Validate inputs
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ip):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid IP address")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid username")],
            isError=True,
        )

    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if key:
        ssh_args.extend(["-i", key])
    ssh_args.extend(
        [
            f"{user}@{ip}",
            "systemctl stop vllm && systemctl disable vllm && rm -f /etc/systemd/system/vllm.service && systemctl daemon-reload",
        ]
    )

    result = await execute_safe_command(ssh_args)
    output_text = (
        f"⏹️ **vLLM Server Stopped** on {ip}\n\n{result['stdout']}"
        if result["success"]
        else f"❌ {result['stderr']}"
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['vllm_stop'] = _handle_vllm_stop


async def _handle_vllm_inference(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"].rstrip("/")
    model = arguments["model"]
    max_tokens = arguments.get("max_tokens", 100)
    api_key = arguments.get("api_key")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if arguments.get("messages"):
        url = f"{endpoint}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": arguments["messages"],
            "max_tokens": max_tokens,
            "stream": False,
        }
    elif arguments.get("prompt"):
        url = f"{endpoint}/v1/completions"
        payload = {
            "model": model,
            "prompt": arguments["prompt"],
            "max_tokens": max_tokens,
            "stream": False,
        }
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="⚠️ Provide either `prompt` or `messages`.",
                )
            ],
            isError=True,
        )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        if "message" in data["choices"][0]:
                            text = data["choices"][0]["message"]["content"]
                        else:
                            text = data["choices"][0].get("text", "")
                        output_text = (
                            f"⚡ **vLLM Inference — {model}**\n\n{text}\n\n"
                        )
                        if data.get("usage"):
                            u = data["usage"]
                            output_text += f"**Tokens:** {u.get('prompt_tokens', '?')} in → {u.get('completion_tokens', '?')} out"
                    else:
                        output_text = f"⚡ **Response:**\n```json\n{json.dumps(data, indent=2)}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ vLLM returned {resp.status}: {body}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"❌ Inference failed: {e}")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['vllm_inference'] = _handle_vllm_inference


async def _handle_vllm_info(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"].rstrip("/")
    api_key = arguments.get("api_key")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    output_text = f"ℹ️ **vLLM Server Info — {endpoint}**\n\n"
                    output_text += f"**Models loaded:** {len(models)}\n"
                    for m in models:
                        parent = m.get("parent")
                        tag = " (LoRA)" if parent else " (base)"
                        output_text += (
                            f"  - **{m.get('id', 'unknown')}**{tag}\n"
                        )
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Server returned {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['vllm_info'] = _handle_vllm_info


async def _handle_vllm_sleep(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"].rstrip("/")
    level = arguments.get("level", 1)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/sleep?level={level}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"😴 **vLLM Server Sleeping** (level {level})\n\nGPU memory freed. Wake with `vllm_wake`.",
                            )
                        ]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Sleep failed: {resp.status} {body}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['vllm_sleep'] = _handle_vllm_sleep


async def _handle_vllm_wake(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"].rstrip("/")
    level = arguments.get("sleep_level", 1)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/wake_up",
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Wake failed: {resp.status} {body}",
                            )
                        ],
                        isError=True,
                    )
                if level == 2:
                    async with session.post(
                        f"{endpoint}/collective_rpc",
                        json={"method": "reload_weights"},
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status != 200:
                            return CallToolResult(
                                content=[
                                    TextContent(
                                        type="text",
                                        text="❌ reload_weights failed",
                                    )
                                ],
                                isError=True,
                            )
                    async with session.post(
                        f"{endpoint}/reset_prefix_cache",
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        pass
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"☀️ **vLLM Server Awake** (from level {level})\n\nReady for inference.",
                        )
                    ]
                )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['vllm_wake'] = _handle_vllm_wake


async def _handle_sglang(arguments, cmd_args, tool_name, execute_terradev_command):
    action = arguments.get("action")
    if not action:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ 'action' parameter required. Use: optimize, detect, router, install, start, stop, inference, metrics, test",
                )
            ],
            isError=True,
        )

    # Import SGLang service

    from terradev_cli.ml_services.sglang_service import SGLangService, WorkloadType

    service = SGLangService()

    if action == "optimize":
        model_path = arguments.get("model_path")
        if not model_path:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'model_path' required for optimize action",
                    )
                ],
                isError=True,
            )

        workload_type_str = arguments.get("workload_type")
        user_description = arguments.get("user_description")
        host = arguments.get("host", "0.0.0.0")
        port = arguments.get("port", 8000)
        dry_run = arguments.get("dry_run", False)

        # Convert string to enum if provided
        workload_type = None
        if workload_type_str:
            workload_type = WorkloadType(workload_type_str)

        # Create optimized configuration
        config = service.create_optimized_config(
            model_path=model_path,
            workload_type=workload_type,
            user_description=user_description,
            host=host,
            port=port,
        )

        # Get optimization summary
        summary = service.get_optimization_summary(config)

        output_text = "🚀 **SGLang Optimization Configuration**\n\n"
        output_text += f"**Model:** {model_path}\n"
        output_text += f"**Workload Type:** {summary['workload_type']}\n"
        output_text += (
            f"**Hardware Detected:** {summary['hardware_detected']}\n"
        )
        output_text += (
            f"**Schedule Policy:** {summary['schedule_policy']}\n"
        )
        output_text += (
            f"**Attention Backend:** {summary['attention_backend']}\n\n"
        )

        output_text += "**Applied Optimizations:**\n"
        for opt in summary["optimizations_applied"]:
            output_text += f"  ✅ {opt}\n"
        output_text += "\n"

        if summary["performance_expectations"]:
            output_text += "**Performance Expectations:**\n"
            for key, value in summary["performance_expectations"].items():
                output_text += (
                    f"  📊 {key.replace('_', ' ').title()}: {value}\n"
                )
            output_text += "\n"

        if summary["hardware_tuned"]:
            output_text += (
                "🔧 **Hardware-specific optimizations applied**\n\n"
            )

        # Validate configuration
        warnings = service.validate_config(config)
        if warnings:
            output_text += "⚠️ **Configuration Warnings:**\n"
            for warning in warnings:
                output_text += f"  ⚠️ {warning}\n"
        output_text += "\n"

        if dry_run:
            output_text += "🔍 **Dry run** - configuration generated but not launched\n"
        else:
            # Generate and display launch command
            launch_cmd = service.generate_launch_command(config)
            output_text += (
                "**🚀 Launch Command:**\n```\n" + launch_cmd + "\n```\n\n"
            )
            output_text += (
                "**💡 To start the server:**\n```\n" + launch_cmd + "\n```"
            )

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )

    elif action == "detect":
        model_path = arguments.get("model_path")
        if not model_path:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'model_path' required for detect action",
                    )
                ],
                isError=True,
            )

        workload_type_str = arguments.get("workload_type")
        user_description = arguments.get("user_description")

        # Detect workload type
        detected_type = service.detect_workload_type(
            model_path, user_description
        )

        output_text = "🔍 **Workload Detection Results**\n\n"
        output_text += f"**Model:** {model_path}\n"
        output_text += (
            f"**Detected Workload Type:** {detected_type.value}\n"
        )

        if workload_type_str:
            manual_type = WorkloadType(workload_type_str)
            output_text += (
                f"**Manual Workload Type:** {manual_type.value}\n"
            )
            if detected_type != manual_type:
                output_text += "⚠️ Manual and detected types differ - using manual specification\n"
                final_type = manual_type
            else:
                output_text += "✅ Manual and detected types match\n"
                final_type = detected_type
        else:
            final_type = detected_type

        output_text += "\n"

        # Show optimization recommendations
        config = service.create_optimized_config(
            model_path=model_path,
            workload_type=final_type,
            user_description=user_description,
        )

        summary = service.get_optimization_summary(config)

        output_text += "**🎯 Optimization Recommendations:**\n"
        for opt in summary["optimizations_applied"]:
            output_text += f"  ✅ {opt}\n"
        output_text += "\n"

        if summary["performance_expectations"]:
            output_text += "**📊 Expected Performance:**\n"
            for key, value in summary["performance_expectations"].items():
                output_text += (
                    f"  📈 {key.replace('_', ' ').title()}: {value}\n"
                )

        output_text += "\n💡 **Next step:** Use sglang action='optimize' to generate the full launch command"

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )

    elif action == "router":
        model_path = arguments.get("model_path")
        if not model_path:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'model_path' required for router action",
                    )
                ],
                isError=True,
            )

        dp_size = arguments.get("dp_size", 8)
        workload_type_str = arguments.get("workload_type")

        # Convert string to enum if provided
        workload_type = None
        if workload_type_str:
            workload_type = WorkloadType(workload_type_str)

        # Create optimized configuration
        config = service.create_optimized_config(
            model_path=model_path, workload_type=workload_type
        )

        # Generate router command
        router_cmd = service.generate_multi_replica_command(config, dp_size)

        output_text = "🔄 **Cache-Aware Router Configuration**\n\n"
        output_text += f"**Model:** {model_path}\n"
        output_text += f"**DP Size:** {dp_size}\n"
        output_text += (
            f"**Workload Type:** {config.workload_type.value}\n\n"
        )
        output_text += (
            "**🚀 Router Launch Command:**\n```\n"
            + router_cmd
            + "\n```\n\n"
        )
        output_text += "**💡 This router provides:**\n"
        output_text += "  📈 Up to 1.9x throughput increase\n"
        output_text += "  🎯 3.8x higher cache hit rate\n"
        output_text += (
            "  🧠 Intelligent request routing based on cache predictions"
        )

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )

    elif action == "install":
        instance_ip = arguments.get("instance_ip")
        ssh_user = arguments.get("ssh_user", "root")
        ssh_key = arguments.get("ssh_key")

        if instance_ip:
            # Remote installation
            result = await service.install_on_instance(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )

            if result["status"] == "installed":
                output_text = f"✅ **SGLang installed successfully** on {instance_ip}\n\n📋 **Output:**\n{result['output']}"
            else:
                output_text = (
                    f"❌ **Installation failed:** {result['error']}"
                )
        else:
            # Local installation
            output_text = "📦 **Installing SGLang locally...**\n\n"
            output_text += "**Command to run:**\n"
            output_text += "```bash\n"
            output_text += 'pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python\n'
            output_text += "```"

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError="failed" in result.get("status", ""),
        )

    elif action == "start":
        model_path = arguments.get("model_path")
        if not model_path:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'model_path' required for start action",
                    )
                ],
                isError=True,
            )

        instance_ip = arguments.get("instance_ip")
        ssh_user = arguments.get("ssh_user", "root")
        ssh_key = arguments.get("ssh_key")
        workload_type_str = arguments.get("workload_type")
        port = arguments.get("port", 8000)

        # Create optimized configuration
        workload_type = None
        if workload_type_str:
            workload_type = WorkloadType(workload_type_str)

        config = service.create_optimized_config(
            model_path=model_path, workload_type=workload_type, port=port
        )

        if instance_ip:
            # Remote deployment
            result = await service.start_server(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )

            if result["status"] == "started":
                output_text = f"✅ **SGLang server started successfully**\n\n🌐 **Endpoint:** http://{instance_ip}:{port}"
            else:
                output_text = (
                    f"❌ **Failed to start server:** {result['error']}"
                )
        else:
            # Local launch
            launch_cmd = service.generate_launch_command(config)
            output_text = "🚀 **Starting SGLang server locally...**\n\n"
            output_text += f"🌐 **Endpoint:** http://localhost:{port}\n\n"
            output_text += (
                "**💡 Launch command:**\n```\n" + launch_cmd + "\n```\n\n"
            )
            output_text += "⚠️ **Run the command above to start the server**"

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError="failed" in result.get("status", ""),
        )

    elif action == "test":
        output_text = "🔍 **Testing SGLang installation...**\n\n"
        result = await service.test_connection()

        if result["status"] == "connected":
            output_text += f"✅ **SGLang is installed and available**\n\n📦 **Version:** {result['sglang_version']}"
        else:
            output_text += f"❌ **SGLang test failed:** {result['error']}\n\n💡 **Run:** `sglang action='install'` to install SGLang"

        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result["status"] != "connected",
        )

    elif action == "stop":
        # Legacy stop functionality
        instance_ip = arguments.get("instance_ip")
        if not instance_ip:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'instance_ip' required for stop action",
                    )
                ],
                isError=True,
            )

        ssh_user = arguments.get("ssh_user", "root")
        ssh_key = arguments.get("ssh_key")

        # Validate inputs
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", instance_ip):
            return CallToolResult(
                content=[
                    TextContent(type="text", text="❌ Invalid IP address")
                ],
                isError=True,
            )
        if not re.match(r"^[a-zA-Z0-9_\-]+$", ssh_user):
            return CallToolResult(
                content=[
                    TextContent(type="text", text="❌ Invalid username")
                ],
                isError=True,
            )

        ssh_args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
        ]
        if ssh_key:
            ssh_args.extend(["-i", ssh_key])
        ssh_args.extend(
            [
                f"{ssh_user}@{instance_ip}",
                "systemctl stop sglang && systemctl disable sglang && rm -f /etc/systemd/system/sglang.service && systemctl daemon-reload",
            ]
        )
        result = await execute_safe_command(ssh_args)
        output_text = (
            f"⏹️ **SGLang Server Stopped** on {instance_ip}"
            if result["success"]
            else f"❌ {result['stderr']}"
        )
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"],
        )

    elif action == "inference":
        # Legacy inference functionality
        endpoint = arguments.get("endpoint")
        if not endpoint:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'endpoint' required for inference action",
                    )
                ],
                isError=True,
            )

        model = arguments.get("model")
        if not model:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'model' required for inference action",
                    )
                ],
                isError=True,
            )

        max_tokens = arguments.get("max_tokens", 100)
        api_key = arguments.get("api_key")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if arguments.get("messages"):
            url = f"{endpoint.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": arguments["messages"],
                "max_tokens": max_tokens,
                "stream": False,
            }
        elif arguments.get("prompt"):
            url = f"{endpoint.rstrip('/')}/v1/completions"
            payload = {
                "model": model,
                "prompt": arguments["prompt"],
                "max_tokens": max_tokens,
                "stream": False,
            }
        else:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="⚠️ Provide either `prompt` or `messages`.",
                    )
                ],
                isError=True,
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "choices" in data and data["choices"]:
                            if "message" in data["choices"][0]:
                                text = data["choices"][0]["message"][
                                    "content"
                                ]
                            else:
                                text = data["choices"][0].get("text", "")
                            output_text = f"⚡ **SGLang Inference — {model}**\n\n{text}\n\n"
                            if data.get("usage"):
                                u = data["usage"]
                                output_text += f"**Tokens:** {u.get('prompt_tokens', '?')} in → {u.get('completion_tokens', '?')} out"
                        else:
                            output_text = f"⚡ **Response:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        return CallToolResult(
                            content=[
                                TextContent(type="text", text=output_text)
                            ]
                        )
                    else:
                        body = await resp.text()
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"❌ SGLang returned {resp.status}: {body}",
                                )
                            ],
                            isError=True,
                        )
        except Exception as e:  # noqa: BLE001
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"❌ Inference failed: {e}"
                    )
                ],
                isError=True,
            )

    elif action == "metrics":
        # Legacy metrics functionality
        endpoint = arguments.get("endpoint")
        if not endpoint:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="❌ 'endpoint' required for metrics action",
                    )
                ],
                isError=True,
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint.rstrip('/')}/metrics",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.text()
                        output_text = f"📊 **SGLang Metrics — {endpoint}**\n\n```\n{raw[:3000]}\n```"
                        return CallToolResult(
                            content=[
                                TextContent(type="text", text=output_text)
                            ]
                        )
                    else:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"❌ Metrics endpoint returned {resp.status}",
                                )
                            ],
                            isError=True,
                        )
        except Exception as e:  # noqa: BLE001
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ {e}")],
                isError=True,
            )

    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Unknown action '{action}'. Use: optimize, detect, router, install, start, stop, inference, metrics, test",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['sglang'] = _handle_sglang


async def _handle_sglang_start(arguments, cmd_args, tool_name, execute_terradev_command):
    ip = arguments["instance_ip"]
    model = arguments["model"]
    port = arguments.get("port", 8000)
    tp = arguments.get("tp_size", 1)
    dp = arguments.get("dp_size", 8)
    ep = arguments.get("enable_expert_parallel", False)
    user = arguments.get("ssh_user", "root")
    key = arguments.get("ssh_key")

    # Validate inputs
    if not re.match(r"^[a-zA-Z0-9_\-\.\/:]+$", model):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid model name")
            ],
            isError=True,
        )
    if not re.match(r"^[0-9]+$", str(port)):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid port")],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ip):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid IP address")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid username")],
            isError=True,
        )

    cmd_parts = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp-size",
        str(tp),
        "--dp-size",
        str(dp),
        "--trust-remote-code",
    ]
    if ep:
        cmd_parts.append("--enable-expert-parallel")
    exec_line = " ".join(cmd_parts)
    service = f"""[Unit]\nDescription=SGLang {model}\nAfter=network.target\n[Service]\nType=simple\nExecStart={exec_line}\nRestart=always\nRestartSec=10\nEnvironment=VLLM_USE_DEEP_GEMM=1\nEnvironment=VLLM_ALL2ALL_BACKEND=deepep_low_latency\n[Install]\nWantedBy=multi-user.target"""
    service_b64 = base64.b64encode(service.encode()).decode()
    setup = f"echo {service_b64} | base64 -d > /etc/systemd/system/sglang.service && systemctl daemon-reload && systemctl enable sglang && systemctl start sglang && sleep 5 && systemctl status sglang"
    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if key:
        ssh_args.extend(["-i", key])
    ssh_args.extend([f"{user}@{ip}", setup])

    result = await execute_safe_command(ssh_args)
    if result["success"]:
        output_text = f"✅ **SGLang Server Started**\n\n**Model:** {model}\n**Endpoint:** http://{ip}:{port}/v1\n**TP:** {tp}, **DP:** {dp}\n**Expert Parallel:** {ep}\n\n{result['stdout']}\n\n"
        output_text += "**suggest_action:** Test with `sglang_inference`."
    else:
        output_text = f"❌ **Failed to start SGLang**\n\n{result['stderr']}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['sglang_start'] = _handle_sglang_start


async def _handle_sglang_stop(arguments, cmd_args, tool_name, execute_terradev_command):
    ip = arguments["instance_ip"]
    user = arguments.get("ssh_user", "root")
    key = arguments.get("ssh_key")

    # Validate inputs
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ip):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid IP address")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid username")],
            isError=True,
        )

    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if key:
        ssh_args.extend(["-i", key])
    ssh_args.extend(
        [
            f"{user}@{ip}",
            "systemctl stop sglang && systemctl disable sglang && rm -f /etc/systemd/system/sglang.service && systemctl daemon-reload",
        ]
    )

    result = await execute_safe_command(ssh_args)
    output_text = (
        f"⏹️ **SGLang Server Stopped** on {ip}"
        if result["success"]
        else f"❌ {result['stderr']}"
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['sglang_stop'] = _handle_sglang_stop


async def _handle_sglang_inference(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments["endpoint"].rstrip("/")
    model = arguments["model"]
    max_tokens = arguments.get("max_tokens", 100)
    api_key = arguments.get("api_key")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if arguments.get("messages"):
        url = f"{endpoint}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": arguments["messages"],
            "max_tokens": max_tokens,
            "stream": False,
        }
    elif arguments.get("prompt"):
        url = f"{endpoint}/v1/completions"
        payload = {
            "model": model,
            "prompt": arguments["prompt"],
            "max_tokens": max_tokens,
            "stream": False,
        }
    else:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="⚠️ Provide either `prompt` or `messages`.",
                )
            ],
            isError=True,
        )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        if "message" in data["choices"][0]:
                            text = data["choices"][0]["message"]["content"]
                        else:
                            text = data["choices"][0].get("text", "")
                        output_text = f"⚡ **SGLang Inference — {model}**\n\n{text}\n\n"
                        if data.get("usage"):
                            u = data["usage"]
                            output_text += f"**Tokens:** {u.get('prompt_tokens', '?')} in → {u.get('completion_tokens', '?')} out"
                    else:
                        output_text = f"⚡ **Response:**\n```json\n{json.dumps(data, indent=2)}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ SGLang returned {resp.status}: {body}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"❌ Inference failed: {e}")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['sglang_inference'] = _handle_sglang_inference




async def _handle_ollama_list(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments.get("endpoint", "http://localhost:11434")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    output_text = f"🦙 **Ollama Models — {endpoint}**\n\n"
                    if models:
                        for m in models:
                            size_gb = m.get("size", 0) / (1024**3)
                            output_text += (
                                f"  - **{m['name']}** ({size_gb:.1f}GB)\n"
                            )
                    else:
                        output_text += (
                            "No models found. Pull one with `ollama_pull`."
                        )
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Ollama returned {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"❌ Cannot connect to Ollama: {e}"
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ollama_list'] = _handle_ollama_list


async def _handle_ollama_pull(arguments, cmd_args, tool_name, execute_terradev_command):
    model = arguments["model"]
    ip = arguments["instance_ip"]
    user = arguments.get("ssh_user", "root")
    key = arguments.get("ssh_key")

    # Validate inputs
    if not re.match(r"^[a-zA-Z0-9_\-\.\/:]+$", model):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid model name")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", ip):
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Invalid IP address")
            ],
            isError=True,
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ Invalid username")],
            isError=True,
        )

    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if key:
        ssh_args.extend(["-i", key])
    ssh_args.extend([f"{user}@{ip}", f"ollama pull {model}"])

    result = await execute_safe_command(ssh_args, timeout=600)
    if result["success"]:
        output_text = f"📥 **Model Pulled: {model}** on {ip}\n\n{result['stdout']}\n\n"
        output_text += "**suggest_action:** Generate with `ollama_generate` or chat with `ollama_chat`."
    else:
        output_text = f"❌ **Pull failed**\n\n{result['stderr']}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['ollama_pull'] = _handle_ollama_pull


async def _handle_ollama_generate(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments.get("endpoint", "http://localhost:11434")
    model = arguments["model"]
    prompt = arguments["prompt"]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output_text = f"🦙 **Ollama Generate — {model}**\n\n{data.get('response', '')}"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ {resp.status}: {body}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['ollama_generate'] = _handle_ollama_generate


async def _handle_ollama_chat(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments.get("endpoint", "http://localhost:11434")
    model = arguments["model"]
    messages = arguments["messages"]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reply = data.get("message", {}).get("content", "")
                    output_text = f"🦙 **Ollama Chat — {model}**\n\n{reply}"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ {resp.status}: {body}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['ollama_chat'] = _handle_ollama_chat


async def _handle_ollama_model_info(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments.get("endpoint", "http://localhost:11434")
    model = arguments["model"]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/api/show",
                json={"name": model},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output_text = f"ℹ️ **Ollama Model Info — {model}**\n\n```json\n{json.dumps(data, indent=2)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Model not found: {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['ollama_model_info'] = _handle_ollama_model_info


async def _handle_vllm_auto_optimize(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--model", arguments["model"]])
    if arguments.get("endpoint"):
        cmd_args.extend(["--endpoint", arguments["endpoint"]])
    if arguments.get("gpu_count"):
        cmd_args.extend(["--gpu-count", str(arguments["gpu_count"])])
    if arguments.get("output"):
        cmd_args.extend(["--output", arguments["output"]])
    return cmd_args

HANDLERS['vllm_auto_optimize'] = _handle_vllm_auto_optimize


async def _handle_vllm_analyze(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--endpoint", arguments["endpoint"]])
    if arguments.get("duration"):
        cmd_args.extend(["--duration", str(arguments["duration"])])
    return cmd_args

HANDLERS['vllm_analyze'] = _handle_vllm_analyze


async def _handle_vllm_benchmark(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--endpoint", arguments["endpoint"]])
    if arguments.get("concurrent"):
        cmd_args.extend(["--concurrent", str(arguments["concurrent"])])
    if arguments.get("prompt"):
        cmd_args.extend(["--prompt", arguments["prompt"]])
    if arguments.get("api_key"):
        cmd_args.extend(["--api-key", arguments["api_key"]])
    return cmd_args

HANDLERS['vllm_benchmark'] = _handle_vllm_benchmark


async def _handle_ollama_ps(arguments, cmd_args, tool_name, execute_terradev_command):
    endpoint = arguments.get("endpoint", "http://localhost:11434")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/api/ps",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    output_text = f"🦙 **Ollama Running Models — {endpoint}**\n\n"
                    if models:
                        for m in models:
                            size_gb = m.get("size", 0) / (1024**3)
                            until = m.get("expires_at", "unknown")
                            output_text += f"  - **{m['name']}** ({size_gb:.1f}GB, expires {until})\n"
                    else:
                        output_text += "No Ollama models are currently running."
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Ollama returned {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ Cannot connect to Ollama: {e}")],
            isError=True,
        )
    return cmd_args


HANDLERS['ollama_ps'] = _handle_ollama_ps