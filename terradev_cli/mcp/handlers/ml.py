"""MCP tool handlers for the ml domain."""

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
execute_terraform_command = executor.execute_terraform_command

HANDLERS = {}

async def _handle_hf_space_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.append(arguments["space_name"])
    cmd_args.extend(["--model-id", arguments["model_id"]])
    cmd_args.extend(["--template", arguments["template"]])
    if "hardware" in arguments:
        cmd_args.extend(["--hardware", arguments["hardware"]])
    if "sdk" in arguments:
        cmd_args.extend(["--sdk", arguments["sdk"]])
    return cmd_args

HANDLERS['hf_space_deploy'] = _handle_hf_space_deploy

async def _handle_hf_space_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = ["hf-spaces", "info", arguments["space_name"]]
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = (
        f"🤗 **HF Space Status: {arguments['space_name']}**\n\n" + output
    )
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['hf_space_status'] = _handle_hf_space_status

async def _handle_ray_status(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd = ["ray", "status"]
    if arguments.get("detailed", True):
        cmd.append("--details")
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=15
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        output_text = "🔵 **Ray Cluster Status**\n\n"
        if result.returncode == 0:
            output_text += output
        else:
            output_text += (
                f"⚠️ {output}\n\n💡 Start a cluster with `ray_start`."
            )
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ Ray not installed. Run: `pip install ray[default]`",
                )
            ],
            isError=True,
        )
    except asyncio.TimeoutError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Ray status timed out.")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_status'] = _handle_ray_status

async def _handle_ray_start(arguments, cmd_args, tool_name, execute_terradev_command):
    head = arguments.get("head", True)
    port = arguments.get("port", 6379)
    if head:
        cmd = [
            "ray",
            "start",
            "--head",
            "--port",
            str(port),
            "--dashboard-host",
            "0.0.0.0",
        ]
        if arguments.get("num_gpus"):
            cmd.extend(["--num-gpus", str(arguments["num_gpus"])])
    else:
        addr = arguments.get("head_address", f"localhost:{port}")
        cmd = ["ray", "start", "--address", addr]
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=30
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        mode = "head" if head else "worker"
        if result.returncode == 0:
            output_text = f"✅ **Ray {mode} node started**\n\n{output}\n\n"
            output_text += "**suggest_action:** Check cluster with `ray_status`. Submit jobs with `ray_submit_job`."
        else:
            output_text = (
                f"❌ **Failed to start Ray {mode} node**\n\n{output}"
            )
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ Ray not installed. Run: `pip install ray[default]`",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_start'] = _handle_ray_start

async def _handle_ray_stop(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        result = await asyncio.create_subprocess_exec(
            "ray",
            "stop",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=15
        )
        output = stdout.decode()
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"⏹️ **Ray Cluster Stopped**\n\n{output}",
                )
            ]
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Ray not installed.")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_stop'] = _handle_ray_stop

async def _handle_ray_submit_job(arguments, cmd_args, tool_name, execute_terradev_command):
    script = arguments["script"]
    # M-A: Validate the script path to prevent path injection.
    # Only accept .py files that resolve within the user home or CWD.
    import pathlib as _pl
    _script_path = _pl.Path(script).resolve()
    _allowed_roots = (_pl.Path.home(), _pl.Path.cwd())
    if not str(_script_path).endswith(".py"):
        return CallToolResult(
            content=[TextContent(type="text", text="❌ script must be a .py file.")],
            isError=True,
        )
    if not any(
        _script_path.is_relative_to(r) for r in _allowed_roots
    ):
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"❌ script path must be inside home or working directory: {script!r}",
            )],
            isError=True,
        )
    cmd = ["ray", "job", "submit", "--", "python", str(_script_path)]
    if arguments.get("job_name"):
        cmd = [
            "ray",
            "job",
            "submit",
            "--submission-id",
            arguments["job_name"],
            "--",
            "python",
            str(_script_path),
        ]
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=60
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        if result.returncode == 0:
            output_text = f"🚀 **Ray Job Submitted**\n\n**Script:** {script}\n\n{output}\n\n"
            output_text += "**suggest_action:** Monitor with `ray_list_jobs` or check dashboard with `ray_status`."
        else:
            output_text = f"❌ **Job submission failed**\n\n{output}"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Ray not installed.")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_submit_job'] = _handle_ray_submit_job

async def _handle_ray_list_jobs(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        result = await asyncio.create_subprocess_exec(
            "ray",
            "job",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=15
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        output_text = "📋 **Ray Jobs**\n\n" + (output or "No jobs found.")
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Ray not installed.")
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_list_jobs'] = _handle_ray_list_jobs

async def _handle_ray_wide_ep_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    model_id = arguments["model_id"]
    tp = arguments.get("tp_size", 1)
    dp = arguments.get("dp_size", 8)
    mem_util = arguments.get("gpu_memory_utilization", 0.85)
    max_len = arguments.get("max_model_len", 32768)
    try:
        from terradev_cli.ml_services.ray_enhanced import (
            EnhancedRayService,
            EnhancedRayConfig,
        )

        svc = EnhancedRayService(
            EnhancedRayConfig(
                model_id=model_id,
                tp_size=tp,
                dp_size=dp,
                gpu_memory_utilization=mem_util,
                max_model_len=max_len,
            )
        )
        config = svc.generate_wide_ep_deployment(
            model_id, tp, dp, mem_util, max_len
        )
        output_text = f"🧬 **Wide-EP Deployment Config — {model_id}**\n\n"
        output_text += "**Pattern:** Wide Expert Parallelism\n"
        output_text += f"**TP:** {config['engine_config']['tensor_parallel_size']}, **DP:** {config['engine_config']['data_parallel_size']}\n"
        output_text += f"**Experts/rank:** {config['model_profile']['experts_per_rank']}\n"
        output_text += f"**EPLB:** {config['engine_config']['enable_eplb']}, **DBO:** {config['engine_config']['enable_dbo']}\n\n"
        output_text += f"**Engine Config:**\n```json\n{json.dumps(config['engine_config'], indent=2)}\n```\n\n"
        output_text += f"**Env Vars:**\n```json\n{json.dumps(config['env_vars'], indent=2)}\n```\n"
        if arguments.get("generate_script", True):
            script = svc.generate_wide_ep_script(model_id, tp, dp)
            output_text += (
                f"\n**Executable Script:**\n```python\n{script}\n```\n"
            )
        output_text += "\n**suggest_action:** Save the script and run it on a Ray cluster with `ray_submit_job`."
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ Terradev CLI not found in path. Ensure terradev_cli is installed.",
                )
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"❌ Wide-EP generation failed: {e}"
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_wide_ep_deploy'] = _handle_ray_wide_ep_deploy

async def _handle_ray_disagg_pd_deploy(arguments, cmd_args, tool_name, execute_terradev_command):
    model_id = arguments["model_id"]
    try:
        from terradev_cli.ml_services.ray_enhanced import (
            EnhancedRayService,
            EnhancedRayConfig,
        )

        svc = EnhancedRayService(
            EnhancedRayConfig(
                model_id=model_id,
                prefill_tp=arguments.get("prefill_tp", 1),
                prefill_dp=arguments.get("prefill_dp", 4),
                decode_tp=arguments.get("decode_tp", 1),
                decode_dp=arguments.get("decode_dp", 4),
                kv_connector=arguments.get("kv_connector", "NixlConnector"),
            )
        )
        config = svc.generate_disaggregated_pd_deployment(model_id)
        pc = config["prefill_config"]
        dc = config["decode_config"]
        output_text = (
            f"⚡ **Disaggregated P/D Deployment — {model_id}**\n\n"
        )
        output_text += f"**Prefill:** TP={pc['tensor_parallel_size']}, DP={pc['data_parallel_size']} (compute-bound)\n"
        output_text += f"**Decode:** TP={dc['tensor_parallel_size']}, DP={dc['data_parallel_size']} (memory-bound)\n"
        output_text += (
            f"**KV Connector:** {config['kv_connector']['type']}\n\n"
        )
        output_text += f"**Config:**\n```json\n{json.dumps(config, indent=2, default=str)}\n```\n"
        if arguments.get("generate_script", True):
            script = svc.generate_disaggregated_pd_script(model_id)
            output_text += (
                f"\n**Executable Script:**\n```python\n{script}\n```\n"
            )
        output_text += "\n**suggest_action:** Deploy on a Ray cluster: save the script, then `ray_submit_job`."
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
            content=[
                TextContent(
                    type="text",
                    text=f"❌ Disagg P/D generation failed: {e}",
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_disagg_pd_deploy'] = _handle_ray_disagg_pd_deploy

async def _handle_ray_parallelism_strategy(arguments, cmd_args, tool_name, execute_terradev_command):
    model_id = arguments["model_id"]
    gpu_count = arguments.get("gpu_count", 8)
    gpu_mem = arguments.get("gpu_memory_gb", 80.0)
    try:
        from terradev_cli.ml_services.ray_enhanced import (
            EnhancedRayService,
            EnhancedRayConfig,
        )

        svc = EnhancedRayService(
            EnhancedRayConfig(model_id=model_id, gpu_count=gpu_count)
        )
        strategy = svc.compute_parallelism_strategy(gpu_count, gpu_mem)
        output_text = f"🧠 **Parallelism Strategy — {model_id}**\n\n"
        output_text += f"**Model:** {strategy['total_params_b']}B params ({strategy['active_params_b']}B active), {strategy['num_experts']} experts\n"
        output_text += f"**Weight:** {strategy['total_weight_gb']}GB total, {strategy['active_memory_gb']}GB active\n"
        output_text += f"**GPUs:** {strategy['gpu_count']}× {gpu_mem}GB\n\n"
        output_text += f"**Recommended:** TP={strategy['recommended_tp']}, DP={strategy['recommended_dp']}\n"
        output_text += f"**Expert Parallel:** {strategy['expert_parallel']} ({strategy['experts_per_rank']} experts/rank)\n"
        output_text += f"**EPLB:** {strategy['eplb_enabled']}\n\n"
        output_text += f"**Rationale:** {strategy['rationale']}\n\n"
        output_text += "**suggest_action:** Apply with `ray_wide_ep_deploy` or `moe_deploy`."
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
            content=[
                TextContent(
                    type="text", text=f"❌ Strategy computation failed: {e}"
                )
            ],
            isError=True,
        )
    return cmd_args

HANDLERS['ray_parallelism_strategy'] = _handle_ray_parallelism_strategy

async def _handle_wandb_list_projects(arguments, cmd_args, tool_name, execute_terradev_command):
    api_key = arguments["api_key"]
    entity = arguments.get("entity", "me")
    try:
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        ) as session:
            async with session.get(
                f"https://api.wandb.ai/v1/entities/{entity}/projects",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    projects = (
                        data.get("projects", data)
                        if isinstance(data, dict)
                        else data
                    )
                    output_text = f"📊 **W&B Projects — {entity}**\n\n"
                    if isinstance(projects, list):
                        for p in projects[:50]:
                            name = (
                                p.get("name", p)
                                if isinstance(p, dict)
                                else str(p)
                            )
                            output_text += f"  - **{name}**\n"
                    else:
                        output_text += f"```json\n{json.dumps(data, indent=2)[:2000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ W&B API returned {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['wandb_list_projects'] = _handle_wandb_list_projects

async def _handle_wandb_list_runs(arguments, cmd_args, tool_name, execute_terradev_command):
    api_key = arguments["api_key"]
    entity = arguments.get("entity", "me")
    project = arguments["project"]
    limit = arguments.get("limit", 50)
    try:
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        ) as session:
            async with session.get(
                f"https://api.wandb.ai/v1/entities/{entity}/projects/{project}/runs",
                params={"limit": limit},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    runs = (
                        data.get("runs", data)
                        if isinstance(data, dict)
                        else data
                    )
                    output_text = f"📋 **W&B Runs — {project}**\n\n"
                    if isinstance(runs, list):
                        for r in runs[:limit]:
                            name = (
                                r.get("name", r.get("id", "?"))
                                if isinstance(r, dict)
                                else str(r)
                            )
                            state = (
                                r.get("state", "?")
                                if isinstance(r, dict)
                                else ""
                            )
                            output_text += f"  - **{name}** ({state})\n"
                    else:
                        output_text += f"```json\n{json.dumps(data, indent=2)[:2000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['wandb_list_runs'] = _handle_wandb_list_runs

async def _handle_wandb_run_details(arguments, cmd_args, tool_name, execute_terradev_command):
    api_key = arguments["api_key"]
    run_id = arguments["run_id"]
    try:
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        ) as session:
            async with session.get(
                f"https://api.wandb.ai/v1/runs/{run_id}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output_text = f"🔍 **W&B Run Details — {run_id}**\n\n```json\n{json.dumps(data, indent=2)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['wandb_run_details'] = _handle_wandb_run_details

async def _handle_mlflow_list_experiments(arguments, cmd_args, tool_name, execute_terradev_command):
    uri = arguments["tracking_uri"]
    username = arguments.get("username")
    password = arguments.get("password")
    headers = {}
    if username and password:
        import base64 as b64

        headers["Authorization"] = (
            "Basic "
            + b64.b64encode(f"{username}:{password}".encode()).decode()
        )
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                f"{uri}/api/2.0/mlflow/experiments/search",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    exps = data.get("experiments", [])
                    output_text = f"🧪 **MLflow Experiments — {uri}**\n\n"
                    for e in exps:
                        output_text += f"  - **{e.get('name', '?')}** (ID: {e.get('experiment_id', '?')})\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ MLflow {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['mlflow_list_experiments'] = _handle_mlflow_list_experiments

async def _handle_mlflow_log_run(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.mlflow_service import (
            MLflowService,
            MLflowConfig,
        )

        config = MLflowConfig(
            tracking_uri=arguments["tracking_uri"],
            username=arguments.get("username"),
            password=arguments.get("password"),
        )
        svc = MLflowService(config)
        result = await svc.log_terradev_run(
            experiment_name=arguments["experiment_name"],
            run_name=arguments["run_name"],
            gpu_type=arguments.get("gpu_type", "unknown"),
            provider=arguments.get("provider", "unknown"),
            cost_per_hour=arguments.get("cost_per_hour", 0.0),
            duration_seconds=arguments.get("duration_seconds", 0.0),
            extra_metrics=arguments.get("metrics", {}),
        )
        output_text = "✅ **MLflow Run Logged**\n\n"
        output_text += f"**Experiment:** {arguments['experiment_name']}\n"
        output_text += f"**Run:** {arguments['run_name']}\n"
        output_text += f"**GPU:** {arguments.get('gpu_type', 'N/A')} ({arguments.get('provider', 'N/A')})\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```\n\n"
        output_text += "**suggest_action:** Register the model with `mlflow_register_model`."
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

HANDLERS['mlflow_log_run'] = _handle_mlflow_log_run

async def _handle_mlflow_register_model(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.mlflow_service import (
            MLflowService,
            MLflowConfig,
        )

        config = MLflowConfig(
            tracking_uri=arguments["tracking_uri"],
            username=arguments.get("username"),
            password=arguments.get("password"),
        )
        svc = MLflowService(config)
        result = await svc.register_terradev_model(
            model_name=arguments["model_name"],
            run_id=arguments["run_id"],
            model_uri=arguments.get(
                "model_uri", f"runs:/{arguments['run_id']}/model"
            ),
            tags=arguments.get("tags", {}),
        )
        output_text = (
            f"✅ **Model Registered: {arguments['model_name']}**\n\n"
        )
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```\n\n"
        output_text += "**suggest_action:** Deploy with `kserve_generate_yaml` or `infer_deploy`."
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

HANDLERS['mlflow_register_model'] = _handle_mlflow_register_model

async def _handle_dvc_status(arguments, cmd_args, tool_name, execute_terradev_command):
    repo = arguments["repo_path"]
    try:
        result = await asyncio.create_subprocess_exec(
            "dvc",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=30
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        output_text = (
            f"📦 **DVC Status — {repo}**\n\n{output or 'No changes.'}"
        )
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ DVC not installed. Run: `pip install dvc`",
                )
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['dvc_status'] = _handle_dvc_status

async def _handle_dvc_diff(arguments, cmd_args, tool_name, execute_terradev_command):
    repo = arguments["repo_path"]
    cmd = ["dvc", "diff"]
    if arguments.get("rev_a"):
        cmd.append(arguments["rev_a"])
    if arguments.get("rev_b"):
        cmd.append(arguments["rev_b"])
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=30
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        output_text = f"📦 **DVC Diff**\n\n{output or 'No differences.'}"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['dvc_diff'] = _handle_dvc_diff

async def _handle_dvc_stage_checkpoint(arguments, cmd_args, tool_name, execute_terradev_command):
    repo = arguments["repo_path"]
    ckpt = arguments["checkpoint_path"]
    msg = arguments.get("message", "Stage checkpoint via Terradev")
    remote = arguments.get("remote")
    try:
        from terradev_cli.ml_services.dvc_service import (
            DVCService,
            DVCConfig,
        )

        svc = DVCService(DVCConfig(repo_path=repo))
        result = await svc.stage_from_checkpoint(
            checkpoint_path=ckpt, commit_message=msg, remote=remote
        )
        output_text = "✅ **Checkpoint Staged**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```\n\n"
        output_text += "**suggest_action:** View changes with `dvc_diff` or push to remote with `dvc_push`."
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

HANDLERS['dvc_stage_checkpoint'] = _handle_dvc_stage_checkpoint

async def _handle_dvc_push(arguments, cmd_args, tool_name, execute_terradev_command):
    repo = arguments["repo_path"]
    cmd = ["dvc", "push"]
    if arguments.get("remote"):
        cmd.extend(["-r", arguments["remote"]])
    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=300
        )
        output = (
            stdout.decode() if result.returncode == 0 else stderr.decode()
        )
        output_text = f"📤 **DVC Push**\n\n{output}"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=result.returncode != 0,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['dvc_push'] = _handle_dvc_push

async def _handle_hf_list_models(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        params = {"limit": arguments.get("limit", 20)}
        if arguments.get("author"):
            params["author"] = arguments["author"]
        if arguments.get("search"):
            params["search"] = arguments["search"]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://huggingface.co/api/models",
                headers={"Authorization": f"Bearer {api_key}"},
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    output_text = f"🤗 **HuggingFace Models** ({len(models)} results)\n\n"
                    for m in models[: int(params["limit"])]:
                        downloads = m.get("downloads", 0)
                        likes = m.get("likes", 0)
                        pipeline = m.get("pipeline_tag", "N/A")
                        output_text += f"- **{m['modelId']}** — ⬇️ {downloads:,} | ❤️ {likes} | 🏷️ {pipeline}\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API error: {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_list_models'] = _handle_hf_list_models

async def _handle_hf_list_datasets(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        params = {"limit": arguments.get("limit", 20)}
        if arguments.get("author"):
            params["author"] = arguments["author"]
        if arguments.get("search"):
            params["search"] = arguments["search"]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://huggingface.co/api/datasets",
                headers={"Authorization": f"Bearer {api_key}"},
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    datasets = await resp.json()
                    output_text = f"🤗 **HuggingFace Datasets** ({len(datasets)} results)\n\n"
                    for d in datasets[: int(params["limit"])]:
                        downloads = d.get("downloads", 0)
                        output_text += (
                            f"- **{d['id']}** — ⬇️ {downloads:,}\n"
                        )
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API error: {resp.status}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_list_datasets'] = _handle_hf_list_datasets

async def _handle_hf_model_info(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        model_id = arguments["model_id"]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://huggingface.co/api/models/{model_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    info = await resp.json()
                    output_text = f"🤗 **Model: {model_id}**\n\n"
                    output_text += (
                        f"**Pipeline:** {info.get('pipeline_tag', 'N/A')}\n"
                    )
                    output_text += (
                        f"**Library:** {info.get('library_name', 'N/A')}\n"
                    )
                    output_text += (
                        f"**Downloads:** {info.get('downloads', 0):,}\n"
                    )
                    output_text += f"**Likes:** {info.get('likes', 0)}\n"
                    output_text += f"**License:** {info.get('cardData', {}).get('license', 'N/A') if isinstance(info.get('cardData'), dict) else 'N/A'}\n"
                    output_text += f"**Tags:** {', '.join(info.get('tags', [])[:15])}\n"
                    siblings = info.get("siblings", [])
                    total_size = sum(
                        s.get("size", 0)
                        for s in siblings
                        if isinstance(s, dict)
                    )
                    if total_size > 0:
                        output_text += (
                            f"**Total Size:** {total_size / 1e9:.2f} GB\n"
                        )
                    safetensors = info.get("safetensors", {})
                    if safetensors and isinstance(safetensors, dict):
                        params = safetensors.get("total", 0)
                        if params:
                            output_text += (
                                f"**Parameters:** {params / 1e9:.2f}B\n"
                            )
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                elif resp.status == 404:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Model not found: {model_id}",
                            )
                        ],
                        isError=True,
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_model_info'] = _handle_hf_model_info

async def _handle_hf_create_endpoint(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        payload = {
            "name": arguments["endpoint_name"],
            "model": {"repository": arguments["model_id"]},
            "compute": {
                "instanceType": arguments["instance_type"],
                "instanceSize": arguments.get("instance_size", "x1"),
                "scaling": {
                    "minReplicas": arguments.get("min_replicas", 0),
                    "maxReplicas": arguments.get("max_replicas", 1),
                },
            },
            "region": arguments.get("region", "us-east-1"),
            "type": "protected",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.endpoints.huggingface.cloud/v2/endpoint",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status in (200, 201, 202):
                    output_text = f"✅ **HF Endpoint Created: {arguments['endpoint_name']}**\n\n"
                    output_text += f"**Model:** {arguments['model_id']}\n"
                    output_text += (
                        f"**Instance:** {arguments['instance_type']}\n"
                    )
                    output_text += f"**Region:** {arguments.get('region', 'us-east-1')}\n"
                    output_text += f"**Status:** {body.get('status', {}).get('state', 'pending')}\n"
                    if body.get("status", {}).get("url"):
                        output_text += f"**URL:** {body['status']['url']}\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API {resp.status}: {json.dumps(body, default=str)[:800]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_create_endpoint'] = _handle_hf_create_endpoint

async def _handle_hf_list_endpoints(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.endpoints.huggingface.cloud/v2/endpoint",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    endpoints = await resp.json(content_type=None)
                    items = (
                        endpoints
                        if isinstance(endpoints, list)
                        else endpoints.get("items", [])
                    )
                    output_text = (
                        f"🤗 **HF Inference Endpoints** ({len(items)})\n\n"
                    )
                    for ep in items:
                        name = ep.get("name", "?")
                        state = ep.get("status", {}).get("state", "unknown")
                        url = ep.get("status", {}).get("url", "N/A")
                        model = ep.get("model", {}).get("repository", "?")
                        icon = (
                            "✅"
                            if state == "running"
                            else (
                                "⏳"
                                if state
                                in ("pending", "initializing", "updating")
                                else "🔴"
                            )
                        )
                        output_text += f"- {icon} **{name}** — {model} | {state} | {url}\n"
                    if not items:
                        output_text += "No endpoints found.\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_list_endpoints'] = _handle_hf_list_endpoints

async def _handle_hf_endpoint_info(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        ep_name = arguments["endpoint_name"]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.endpoints.huggingface.cloud/v2/endpoint/{ep_name}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    ep = await resp.json(content_type=None)
                    output_text = f"🤗 **Endpoint: {ep_name}**\n\n"
                    output_text += f"**Model:** {ep.get('model', {}).get('repository', '?')}\n"
                    output_text += f"**State:** {ep.get('status', {}).get('state', 'unknown')}\n"
                    output_text += f"**URL:** {ep.get('status', {}).get('url', 'N/A')}\n"
                    compute = ep.get("compute", {})
                    output_text += f"**Instance:** {compute.get('instanceType', '?')} ({compute.get('instanceSize', '?')})\n"
                    scaling = compute.get("scaling", {})
                    output_text += f"**Scaling:** {scaling.get('minReplicas', 0)} – {scaling.get('maxReplicas', 1)}\n"
                    output_text += f"**Region:** {ep.get('region', '?')}\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_endpoint_info'] = _handle_hf_endpoint_info

async def _handle_hf_delete_endpoint(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        ep_name = arguments["endpoint_name"]
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"https://api.endpoints.huggingface.cloud/v2/endpoint/{ep_name}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status in (200, 202, 204):
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"✅ **Endpoint deleted: {ep_name}**",
                            )
                        ]
                    )
                else:
                    body = await resp.text()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF API {resp.status}: {body[:500]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_delete_endpoint'] = _handle_hf_delete_endpoint

async def _handle_hf_endpoint_infer(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        api_key = arguments["api_key"]
        ep_name = arguments["endpoint_name"]
        payload = {"inputs": arguments["inputs"]}
        if arguments.get("parameters"):
            payload["parameters"] = arguments["parameters"]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.endpoints.huggingface.cloud/v2/endpoint/{ep_name}/inference",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status == 200:
                    output_text = f"🤗 **Inference Result — {ep_name}**\n\n"
                    output_text += f"```json\n{json.dumps(body, indent=2, default=str)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ HF Inference {resp.status}: {json.dumps(body, default=str)[:800]}",
                            )
                        ],
                        isError=True,
                    )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_endpoint_infer'] = _handle_hf_endpoint_infer

async def _handle_hf_smart_template(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.hf_smart_templates import HFSmartTemplates

        templates = HFSmartTemplates()
        model_id = arguments["model_id"]
        template_type = arguments.get("template_type", "auto")
        result = await templates.generate_template(
            model_id, template_type=template_type
        )
        output_text = f"🧠 **Smart Template — {model_id}**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="❌ Terradev CLI not found. Install: pip install terradev-cli",
                )
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['hf_smart_template'] = _handle_hf_smart_template

async def _handle_hf_hardware_recommend(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.hf_smart_templates import HFSmartTemplates

        templates = HFSmartTemplates()
        model_id = arguments["model_id"]
        budget = arguments.get("budget_constraint")
        result = await templates.recommend_hardware(
            model_id, budget_constraint=budget
        )
        output_text = f"🖥️ **Hardware Recommendation — {model_id}**\n\n"
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

HANDLERS['hf_hardware_recommend'] = _handle_hf_hardware_recommend

async def _handle_hf_hardware_compare(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.hf_smart_templates import HFSmartTemplates

        templates = HFSmartTemplates()
        model_id = arguments["model_id"]
        result = await templates.compare_hardware(model_id)
        output_text = f"📊 **Hardware Comparison — {model_id}**\n\n"
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

HANDLERS['hf_hardware_compare'] = _handle_hf_hardware_compare

async def _handle_langchain_create_workflow(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langchain_service import (
            LangChainService,
        )

        svc = LangChainService(api_key=arguments["api_key"])
        config = arguments["workflow_config"]
        langsmith_key = arguments.get("langsmith_api_key")
        result = await svc.create_workflow(
            config, langsmith_api_key=langsmith_key
        )
        output_text = "🔗 **LangChain Workflow Created**\n\n"
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

HANDLERS['langchain_create_workflow'] = _handle_langchain_create_workflow

async def _handle_langchain_create_sglang_pipeline(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langchain_service import (
            LangChainService,
        )

        svc = LangChainService(api_key=arguments["api_key"])
        config = arguments["pipeline_config"]
        result = await svc.create_sglang_pipeline(config)
        output_text = "🔗 **SGLang Pipeline Created**\n\n"
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

HANDLERS['langchain_create_sglang_pipeline'] = _handle_langchain_create_sglang_pipeline

async def _handle_langgraph_create_workflow(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langgraph_service import (
            LangGraphService,
        )

        svc = LangGraphService(api_key=arguments["api_key"])
        config = arguments["graph_config"]
        langsmith_key = arguments.get("langsmith_api_key")
        result = await svc.create_workflow(
            config, langsmith_api_key=langsmith_key
        )
        output_text = "🕸️ **LangGraph Workflow Created**\n\n"
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

HANDLERS['langgraph_create_workflow'] = _handle_langgraph_create_workflow

async def _handle_langgraph_orchestrator_worker(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langgraph_service import (
            LangGraphService,
        )

        svc = LangGraphService(api_key=arguments["api_key"])
        config = arguments["workflow_config"]
        result = await svc.create_orchestrator_worker(config)
        output_text = "🕸️ **Orchestrator-Worker Created**\n\n"
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

HANDLERS['langgraph_orchestrator_worker'] = _handle_langgraph_orchestrator_worker

async def _handle_langgraph_evaluation_workflow(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langgraph_service import (
            LangGraphService,
        )

        svc = LangGraphService(api_key=arguments["api_key"])
        config = arguments["evaluation_config"]
        result = await svc.create_evaluation_workflow(config)
        output_text = "🕸️ **Evaluation Workflow Created**\n\n"
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

HANDLERS['langgraph_evaluation_workflow'] = _handle_langgraph_evaluation_workflow

async def _handle_langgraph_workflow_status(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.langgraph_service import (
            LangGraphService,
        )

        svc = LangGraphService(api_key=arguments["api_key"])
        wf_id = arguments["workflow_id"]
        result = await svc.get_workflow_status(wf_id)
        output_text = f"🕸️ **Workflow Status — {wf_id}**\n\n"
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

HANDLERS['langgraph_workflow_status'] = _handle_langgraph_workflow_status

async def _handle_wandb_create_dashboard(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        config = arguments["dashboard_config"]
        result = await svc.create_dashboard(config)
        output_text = "📊 **W&B Dashboard Created**\n\n"
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

HANDLERS['wandb_create_dashboard'] = _handle_wandb_create_dashboard

async def _handle_wandb_create_terradev_dashboard(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        project = arguments.get("project", "terradev")
        # Parallel: create dashboard + alerts simultaneously
        dashboard_coro = svc.create_terradev_dashboard(project)
        alerts_coro = svc.create_terradev_alerts()
        dashboard_result, alerts_result = await asyncio.gather(
            dashboard_coro, alerts_coro, return_exceptions=True
        )
        output_text = f"📊 **Terradev Dashboard — {project}**\n\n"
        if not isinstance(dashboard_result, Exception):
            output_text += f"**Dashboard:** ✅ Created\n```json\n{json.dumps(dashboard_result, indent=2, default=str)[:2000]}\n```\n\n"
        else:
            output_text += f"**Dashboard:** ❌ {dashboard_result}\n\n"
        if not isinstance(alerts_result, Exception):
            output_text += f"**Alerts:** ✅ Configured\n```json\n{json.dumps(alerts_result, indent=2, default=str)[:1000]}\n```"
        else:
            output_text += f"**Alerts:** ❌ {alerts_result}"
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

HANDLERS['wandb_create_terradev_dashboard'] = _handle_wandb_create_terradev_dashboard

async def _handle_wandb_create_report(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        config = arguments["report_config"]
        result = await svc.create_report(config)
        output_text = "📝 **W&B Report Created**\n\n"
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

HANDLERS['wandb_create_report'] = _handle_wandb_create_report

async def _handle_wandb_create_terradev_report(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        metrics = arguments.get("metrics_data", {})
        result = await svc.create_terradev_report(metrics)
        output_text = "📝 **Terradev Report Generated**\n\n"
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

HANDLERS['wandb_create_terradev_report'] = _handle_wandb_create_terradev_report

async def _handle_wandb_setup_alerts(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        config = arguments["alert_config"]
        result = await svc.setup_alerts(config)
        output_text = "🔔 **W&B Alerts Configured**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
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

HANDLERS['wandb_setup_alerts'] = _handle_wandb_setup_alerts

async def _handle_wandb_create_terradev_alerts(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        result = await svc.create_terradev_alerts()
        output_text = "🔔 **Terradev Alerts Created**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
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

HANDLERS['wandb_create_terradev_alerts'] = _handle_wandb_create_terradev_alerts

async def _handle_wandb_dashboard_status(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.wandb_enhanced import WandBEnhanced

        svc = WandBEnhanced(
            api_key=arguments["api_key"], entity=arguments.get("entity")
        )
        result = await svc.dashboard_status()
        output_text = "📊 **W&B Monitoring Overview**\n\n"
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

HANDLERS['wandb_dashboard_status'] = _handle_wandb_dashboard_status

async def _handle_phoenix_projects(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['phoenix_projects'] = _handle_phoenix_projects

async def _handle_phoenix_spans(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("project"):
        cmd_args.extend(["--project", arguments["project"]])
    if arguments.get("filter"):
        cmd_args.extend(["--filter", arguments["filter"]])
    if arguments.get("limit"):
        cmd_args.extend(["--limit", str(arguments["limit"])])
    return cmd_args

HANDLERS['phoenix_spans'] = _handle_phoenix_spans

async def _handle_phoenix_trace(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--trace-id", arguments["trace_id"]])
    if arguments.get("project"):
        cmd_args.extend(["--project", arguments["project"]])
    return cmd_args

HANDLERS['phoenix_trace'] = _handle_phoenix_trace

async def _handle_phoenix_otel_env(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("project"):
        cmd_args.extend(["--project", arguments["project"]])
    return cmd_args

HANDLERS['phoenix_otel_env'] = _handle_phoenix_otel_env

async def _handle_phoenix_snippet(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("project"):
        cmd_args.extend(["--project", arguments["project"]])
    return cmd_args

HANDLERS['phoenix_snippet'] = _handle_phoenix_snippet

async def _handle_phoenix_k8s(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    return cmd_args

HANDLERS['phoenix_k8s'] = _handle_phoenix_k8s

async def _handle_guardrails_chat(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--message", arguments["message"]])
    if arguments.get("config_id"):
        cmd_args.extend(["--config-id", arguments["config_id"]])
    return cmd_args

HANDLERS['guardrails_chat'] = _handle_guardrails_chat

async def _handle_guardrails_generate_config(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("config_id"):
        cmd_args.extend(["--config-id", arguments["config_id"]])
    if arguments.get("output_dir"):
        cmd_args.extend(["--output-dir", arguments["output_dir"]])
    return cmd_args

HANDLERS['guardrails_generate_config'] = _handle_guardrails_generate_config

async def _handle_guardrails_k8s(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    return cmd_args

HANDLERS['guardrails_k8s'] = _handle_guardrails_k8s

async def _handle_qdrant_create_collection(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    if arguments.get("embedding_model"):
        cmd_args.extend(["--embedding-model", arguments["embedding_model"]])
    return cmd_args

HANDLERS['qdrant_create_collection'] = _handle_qdrant_create_collection

async def _handle_qdrant_info(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    return cmd_args

HANDLERS['qdrant_info'] = _handle_qdrant_info

async def _handle_qdrant_count(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("name"):
        cmd_args.extend(["--name", arguments["name"]])
    return cmd_args

HANDLERS['qdrant_count'] = _handle_qdrant_count

async def _handle_qdrant_k8s(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("namespace"):
        cmd_args.extend(["--namespace", arguments["namespace"]])
    return cmd_args

HANDLERS['qdrant_k8s'] = _handle_qdrant_k8s

async def _handle_deepeval_run(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("file"):
        cmd_args.extend(["--file", arguments["file"]])
    return cmd_args

HANDLERS['deepeval_run'] = _handle_deepeval_run

async def _handle_deepeval_metrics(arguments, cmd_args, tool_name, execute_terradev_command):
    return cmd_args

HANDLERS['deepeval_metrics'] = _handle_deepeval_metrics

async def _handle_deepeval_evaluate(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.extend(["--input", arguments["input"]])
    cmd_args.extend(["--actual-output", arguments["actual_output"]])
    cmd_args.extend(["--metric", arguments["metric"]])
    if arguments.get("expected_output"):
        cmd_args.extend(["--expected-output", arguments["expected_output"]])
    if arguments.get("context"):
        cmd_args.extend(["--context", arguments["context"]])
    if arguments.get("retrieval_context"):
        cmd_args.extend(["--retrieval-context", arguments["retrieval_context"]])
    if arguments.get("threshold"):
        cmd_args.extend(["--threshold", str(arguments["threshold"])])
    return cmd_args

HANDLERS['deepeval_evaluate'] = _handle_deepeval_evaluate

async def _handle_deepeval_init(arguments, cmd_args, tool_name, execute_terradev_command):
    if arguments.get("output"):
        cmd_args.extend(["--output", arguments["output"]])
    return cmd_args

HANDLERS['deepeval_init'] = _handle_deepeval_init