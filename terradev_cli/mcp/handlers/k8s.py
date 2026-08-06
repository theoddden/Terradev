"""MCP tool handlers for the k8s domain."""

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
generate_k8s_terraform_config = executor.generate_k8s_terraform_config

HANDLERS = {}


async def _handle_k8s_create(arguments, cmd_args, tool_name, execute_terradev_command):
    cluster_name = arguments["cluster_name"]
    gpu_type = arguments["gpu_type"]
    node_count = arguments.get("count", 1)
    multi_cloud = arguments.get("multi_cloud", False)
    prefer_spot = arguments.get("prefer_spot", True)
    use_terraform = arguments.get("use_terraform", True)

    if use_terraform:
        # Use persistent workspace so state survives for k8s_destroy
        ws_dir = _get_tf_workspace(f"k8s-{cluster_name}")
        try:
            # Generate K8s Terraform configuration
            k8s_config = generate_k8s_terraform_config(
                cluster_name, gpu_type, node_count, multi_cloud, prefer_spot
            )

            # Write configuration files
            main_tf_path = os.path.join(ws_dir, "main.tf")
            with open(main_tf_path, "w") as f:
                f.write(k8s_config)

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
                    "✅ Kubernetes cluster created via Terraform!\n\n"
                )
                output_text += f"**Cluster Name:** {cluster_name}\n"
                output_text += f"**GPU Type:** {gpu_type}\n"
                output_text += f"**Node Count:** {node_count}\n"
                output_text += f"**Multi-Cloud:** {multi_cloud}\n"
                output_text += f"**Spot Instances:** {prefer_spot}\n"
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
                        text=f"❌ K8s Terraform deployment failed: {str(e)}",
                    )
                ],
                isError=True,
            )
    else:
        # Fall back to regular terradev command
        cmd_args.extend([cluster_name])
        cmd_args.extend(["--gpu", gpu_type])
        if "count" in arguments:
            cmd_args.extend(["--count", str(arguments["count"])])
        if multi_cloud:
            cmd_args.append("--multi-cloud")
        if prefer_spot:
            cmd_args.append("--prefer-spot")
    return cmd_args

HANDLERS['k8s_create'] = _handle_k8s_create


async def _handle_k8s_info(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.append(arguments["cluster_name"])
    return cmd_args

HANDLERS['k8s_info'] = _handle_k8s_info


async def _handle_k8s_destroy(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args.append(arguments["cluster_name"])
    return cmd_args

HANDLERS['k8s_destroy'] = _handle_k8s_destroy


async def _handle_helm_generate(arguments, cmd_args, tool_name, execute_terradev_command):
    cmd_args = [
        "helm-generate",
        "--workload",
        arguments["workload"],
        "--image",
        arguments["image"],
    ]
    if arguments.get("gpu_type"):
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    if arguments.get("replicas"):
        cmd_args.extend(["--replicas", str(arguments["replicas"])])
    result = await execute_terradev_command(cmd_args)
    output = result["stdout"] if result["success"] else result["stderr"]
    output_text = "⎈ **Helm Chart Generated**\n\n"
    if result["success"]:
        output_text += output
        output_text += "\n\n**suggest_action:** Apply with `kubectl apply -f` or deploy to cluster with `k8s_create`."
    else:
        output_text += f"⚠️ {output}"
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)],
        isError=not result["success"],
    )

HANDLERS['helm_generate'] = _handle_helm_generate


async def _handle_kserve_generate_yaml(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.kserve_service import (
            KServeService,
            KServeConfig,
        )

        svc = KServeService(
            KServeConfig(namespace=arguments.get("namespace", "default"))
        )
        yaml_str = await svc.generate_inferenceservice_yaml(
            model_name=arguments["model_name"],
            model_uri=arguments["model_uri"],
            gpu_type=arguments["gpu_type"],
            gpu_count=arguments.get("gpu_count", 1),
            namespace=arguments.get("namespace", "default"),
            runtime=arguments.get("runtime", "vllm"),
            min_replicas=arguments.get("min_replicas", 1),
            max_replicas=arguments.get("max_replicas", 3),
        )
        output_text = f"☸️ **KServe InferenceService YAML — {arguments['model_name']}**\n\n"
        output_text += f"```yaml\n{yaml_str}\n```\n\n"
        output_text += "**suggest_action:** Apply with `kubectl apply -f <file>.yaml` or deploy to cluster with `k8s_create`."
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

HANDLERS['kserve_generate_yaml'] = _handle_kserve_generate_yaml


async def _handle_kserve_list(arguments, cmd_args, tool_name, execute_terradev_command):
    ns = arguments.get("namespace", "default")
    try:
        result = await asyncio.create_subprocess_exec(
            "kubectl",
            "get",
            "inferenceservices",
            "-n",
            ns,
            "-o",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=15
        )
        if result.returncode == 0:
            data = json.loads(stdout.decode())
            items = data.get("items", [])
            output_text = f"☸️ **KServe InferenceServices — {ns}**\n\n"
            if items:
                for item in items:
                    name = item.get("metadata", {}).get("name", "?")
                    ready = (
                        "✅"
                        if any(
                            c.get("status") == "True"
                            for c in item.get("status", {}).get(
                                "conditions", []
                            )
                        )
                        else "⏳"
                    )
                    url = item.get("status", {}).get("url", "N/A")
                    output_text += f"  - {ready} **{name}** → {url}\n"
            else:
                output_text += "No InferenceServices found."
            return CallToolResult(
                content=[TextContent(type="text", text=output_text)]
            )
        else:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ kubectl failed: {stderr.decode()}",
                    )
                ],
                isError=True,
            )
    except FileNotFoundError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ kubectl not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['kserve_list'] = _handle_kserve_list


async def _handle_kserve_status(arguments, cmd_args, tool_name, execute_terradev_command):
    name = arguments["name"]
    ns = arguments.get("namespace", "default")
    try:
        result = await asyncio.create_subprocess_exec(
            "kubectl",
            "get",
            "inferenceservice",
            name,
            "-n",
            ns,
            "-o",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=15
        )
        if result.returncode == 0:
            data = json.loads(stdout.decode())
            status = data.get("status", {})
            conditions = status.get("conditions", [])
            url = status.get("url", "N/A")
            output_text = f"☸️ **KServe Status — {name}**\n\n"
            output_text += f"**URL:** {url}\n\n**Conditions:**\n"
            for c in conditions:
                icon = "✅" if c.get("status") == "True" else "❌"
                output_text += f"  - {icon} **{c.get('type')}**: {c.get('message', '')}\n"
            return CallToolResult(
                content=[TextContent(type="text", text=output_text)]
            )
        else:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"❌ {stderr.decode()}")
                ],
                isError=True,
            )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['kserve_status'] = _handle_kserve_status


async def _handle_k8s_gpu_operator_install(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.kubernetes_enhanced import (
            EnhancedKubernetesService,
        )

        svc = EnhancedKubernetesService()
        cluster = arguments["cluster_name"]
        ns = arguments.get("namespace", "gpu-operator")
        driver_ver = arguments.get("driver_version")
        result = await svc.install_gpu_operator(
            cluster_name=cluster, namespace=ns, driver_version=driver_ver
        )
        output_text = f"🖥️ **GPU Operator Installed — {cluster}**\n\n"
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

HANDLERS['k8s_gpu_operator_install'] = _handle_k8s_gpu_operator_install


async def _handle_k8s_device_plugin(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.kubernetes_enhanced import (
            EnhancedKubernetesService,
        )

        svc = EnhancedKubernetesService()
        result = await svc.configure_device_plugin(
            cluster_name=arguments["cluster_name"],
            strategy=arguments.get("strategy", "none"),
            replicas=arguments.get("replicas", 2),
        )
        output_text = f"🔌 **Device Plugin Configured — {arguments['cluster_name']}**\n\n"
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

HANDLERS['k8s_device_plugin'] = _handle_k8s_device_plugin


async def _handle_k8s_mig_configure(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.kubernetes_enhanced import (
            EnhancedKubernetesService,
        )

        svc = EnhancedKubernetesService()
        result = await svc.configure_mig(
            cluster_name=arguments["cluster_name"],
            mig_profile=arguments["mig_profile"],
            gpu_indices=arguments.get("gpu_indices"),
        )
        output_text = (
            f"🔧 **MIG Configured — {arguments['cluster_name']}**\n\n"
        )
        output_text += f"**Profile:** {arguments['mig_profile']}\n\n"
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

HANDLERS['k8s_mig_configure'] = _handle_k8s_mig_configure


async def _handle_k8s_time_slicing(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.ml_services.kubernetes_enhanced import (
            EnhancedKubernetesService,
        )

        svc = EnhancedKubernetesService()
        result = await svc.configure_time_slicing(
            cluster_name=arguments["cluster_name"],
            replicas=arguments.get("replicas", 4),
            oversubscribe=arguments.get("oversubscribe", True),
        )
        output_text = f"⏱️ **Time-Slicing Configured — {arguments['cluster_name']}**\n\n"
        output_text += (
            f"**Replicas/GPU:** {arguments.get('replicas', 4)}\n\n"
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

HANDLERS['k8s_time_slicing'] = _handle_k8s_time_slicing


