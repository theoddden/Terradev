#!/usr/bin/env python3
"""
Terradev MCP Server v2.1.0 - Complete Agentic GPU Infrastructure for Claude Code

220 MCP tools: GPU provisioning, Kubernetes clusters, Karpenter auto-provisioning,
GitOps/ArgoCD automation, event-driven triggers, environment promotion, lineage tracking,
cross-provider migration, vLLM/SGLang/Ollama inference, Arize Phoenix trace observability,
NeMo Guardrails output safety, Qdrant vector DB, Ray cluster management, W&B/LangSmith/MLflow/DVC,
HuggingFace Hub, Datadog monitoring, data governance, cost optimization, FlashOptim training,
Langfuse LLM observability, Databricks MLOps, vLLM auto-optimization/analysis/benchmarking,
Enterprise SSO, Latitude.sh provider, and Terraform-powered parallel provisioning across 21+ cloud providers.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import time

# Import new feature tools from the sibling module (relative import).
# If the module is absent a WARNING is emitted so operators know which tools
# are missing, rather than silently dropping them.
try:
    from .new_feature_tools import ALL_NEW_TOOLS, COMMAND_MAP as NEW_COMMAND_MAP
except ImportError as _nft_err:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "new_feature_tools module not found (%s) — extended MCP tools will be unavailable.",
        _nft_err,
    )
    ALL_NEW_TOOLS = []
    NEW_COMMAND_MAP = {}
try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    from mcp.server import NotificationOptions
    from mcp.server.sse import TransportSecuritySettings
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        GetPromptRequest,
        GetPromptResult,
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListToolsRequest,
        ListToolsResult,
        ReadResourceRequest,
        ReadResourceResult,
        Resource,
        TextContent,
        TextResourceContents,
        Tool,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    InitializationOptions = None
    stdio_server = None
    SseServerTransport = None
    NotificationOptions = None
    TransportSecuritySettings = None
    CallToolRequest = None
    CallToolResult = None
    GetPromptRequest = None
    GetPromptResult = None
    ListPromptsRequest = None
    ListPromptsResult = None
    ListResourcesRequest = None
    ListResourcesResult = None
    ListToolsRequest = None
    ListToolsResult = None
    ReadResourceRequest = None
    ReadResourceResult = None
    Resource = None
    TextContent = None
    TextResourceContents = None
    Tool = None

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route
    import uvicorn
except ImportError:
    # SSE deps are optional — only needed for remote mode
    Starlette = None
    uvicorn = None

logger = logging.getLogger("terradev-mcp")


# Check if terradev CLI is available
def check_terradev_installation():
    try:
        cmd = _terradev_command() + ["--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Local GPU Discovery
async def discover_local_gpus() -> Dict[str, Any]:
    """Discover local GPU devices on the network and current machine.

    Returns a dict with:
    - local_devices: List of GPUs on current machine
    - total_vram: Total VRAM available locally
    - device_details: Detailed info per device
    """
    devices = []
    total_vram = 0

    try:
        # Try to import torch for CUDA detection
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "type": "cuda",
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
                devices.append(device_info)
                total_vram += device_info["vram_gb"]

        # Check for Apple Metal/MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Estimate unified memory (simplified - actual detection is complex)
            import platform

            if platform.system() == "Darwin":
                # Try to get system memory as proxy for unified memory
                try:
                    import psutil

                    total_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
                    device_info = {
                        "id": len(devices),
                        "type": "mps",
                        "name": "Apple Metal",
                        "vram_gb": total_mem_gb,  # Unified memory
                        "platform": platform.machine(),
                    }
                    devices.append(device_info)
                    total_vram += device_info["vram_gb"]
                except ImportError:
                    pass
    except ImportError:
        # torch not available, try nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            device_info = {
                                "id": int(parts[0]),
                                "type": "cuda",
                                "name": parts[1],
                                "vram_gb": round(float(parts[2]) / 1024, 2),
                            }
                            devices.append(device_info)
                            total_vram += device_info["vram_gb"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return {
        "local_devices": devices,
        "total_vram_gb": round(total_vram, 2),
        "device_count": len(devices),
        "has_local_gpu": len(devices) > 0,
    }


async def estimate_model_memory(model_name: str) -> float:
    """Estimate memory requirements for a model.

    Simple heuristic based on model size in name (e.g., '72B' -> 72 billion params).
    Returns estimated VRAM in GB.
    """
    import re

    # Extract parameter count from model name
    match = re.search(r"(\d+)B", model_name, re.IGNORECASE)
    if match:
        params_b = int(match.group(1))
        # Rough estimate: 2 bytes per param (fp16) + 20% overhead
        return params_b * 2 * 1.2

    # Default estimates for common models
    model_lower = model_name.lower()
    if "7b" in model_lower:
        return 16
    elif "13b" in model_lower:
        return 28
    elif "70b" in model_lower or "72b" in model_lower:
        return 150
    elif "405b" in model_lower:
        return 850

    # Unknown model, return conservative estimate
    return 20


# ── Datadog credential loader ────────────────────────────────────────────────
def _load_datadog_creds() -> Dict[str, str]:
    """Load Datadog credentials from ~/.terradev/credentials.json."""
    creds_path = os.path.join(os.path.expanduser("~"), ".terradev", "credentials.json")
    if os.path.exists(creds_path):
        with open(creds_path, "r") as f:
            all_creds = json.load(f)
        return {k: v for k, v in all_creds.items() if k.startswith("datadog_")}
    return {}


# ── Persistent Terraform workspaces ──────────────────────────────────────────
# Critical fix: Terraform state must survive beyond a single tool call.
# Previously used tempfile.TemporaryDirectory which destroyed terraform.tfstate
# immediately after apply, making it impossible to destroy/manage resources later.
TERRADEV_TF_STATE_DIR = os.path.join(os.path.expanduser("~"), ".terradev", "terraform")


def _get_tf_workspace(name: str) -> str:
    """Get or create a persistent Terraform workspace directory.

    State files (terraform.tfstate) are preserved across tool calls,
    enabling terraform destroy/plan on previously provisioned resources.
    """
    # Sanitize workspace name to prevent path traversal
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_.")
    if not safe_name:
        safe_name = "default"
    ws = os.path.join(TERRADEV_TF_STATE_DIR, safe_name)
    os.makedirs(ws, exist_ok=True)
    return ws


def _list_tf_workspaces() -> List[Dict[str, Any]]:
    """List all Terraform workspaces with their state status."""
    workspaces = []
    if os.path.isdir(TERRADEV_TF_STATE_DIR):
        for name in sorted(os.listdir(TERRADEV_TF_STATE_DIR)):
            ws_path = os.path.join(TERRADEV_TF_STATE_DIR, name)
            if os.path.isdir(ws_path):
                has_state = os.path.exists(os.path.join(ws_path, "terraform.tfstate"))
                workspaces.append(
                    {
                        "name": name,
                        "path": ws_path,
                        "has_state": has_state,
                    }
                )
    return workspaces


# ── Path validation ──────────────────────────────────────────────────────────
import re as _re

_SAFE_PATH_RE = _re.compile(r"^[a-zA-Z0-9_./@:~\-]+$")


def _validate_config_dir(config_dir: str) -> str:
    """Validate a user-provided config_dir to prevent path traversal.

    Rejects paths containing '..' or suspicious characters.
    Returns the resolved absolute path.
    """
    if ".." in config_dir:
        raise ValueError(
            f"Invalid config_dir: path traversal ('..') not allowed: {config_dir}"
        )
    resolved = os.path.realpath(os.path.expanduser(config_dir))
    if not os.path.isdir(resolved):
        raise ValueError(f"Invalid config_dir: directory does not exist: {resolved}")
    return resolved


# Execute terradev command safely with bug fixes
def _terradev_command() -> List[str]:
    """Return the command prefix to invoke the Terradev CLI.

    Prefers the installed `terradev` entry-point script. Falls back to
    `python -m terradev_cli` so the MCP server works in editable installs
    and during development without needing the script on PATH.
    """
    if shutil.which("terradev"):
        return ["terradev"]
    return [sys.executable, "-m", "terradev_cli"]


async def execute_terradev_command(args: List[str]) -> Dict[str, Any]:
    """Execute terradev CLI command with helpful error messages."""
    try:
        cmd = _terradev_command() + args

        # Apply bug fixes for known issues
        env = os.environ.copy()

        # Fix 3: Ensure proxy settings are respected
        env["TRUST_ENV"] = "true"

        # Fix 4: Ensure boto3 is available (will be handled by requirements)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await process.communicate()
        stderr_text = stderr.decode().strip()

        # Enhance error messages with helpful guidance
        if process.returncode != 0:
            stderr_text = enhance_error_message(stderr_text, args)

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr_text,
            "returncode": process.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "❌ terradev CLI not found.\n\n"
            + "📦 Install it with: pip install terradev-cli\n"
            + "📚 Docs: https://github.com/terradev-io/terradev-cli",
            "returncode": -1,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "stdout": "",
            "stderr": f"❌ Unexpected error: {str(e)}",
            "returncode": -1,
        }


_SHELL_CMD_ALLOWLIST: set = set()  # Populated by internal callers with hardcoded strings only


async def _UNSAFE_execute_shell_command(
    cmd: str, timeout: int = 120, _allow_caller: str = ""
) -> Dict[str, Any]:
    """INTERNAL USE ONLY — Execute a raw shell command using shell=True.

    SECURITY: This function is vulnerable to shell injection. It MUST NOT be called
    with any user-supplied or AI-agent-supplied input. Pass _allow_caller with the
    exact hardcoded string you are running to make the intent explicit at the call site.
    For all commands with dynamic inputs, use execute_safe_command() instead.
    """
    if not _allow_caller:
        raise RuntimeError(
            "_UNSAFE_execute_shell_command called without _allow_caller token — "
            "this indicates a new call site was added without security review. "
            "Use execute_safe_command() for any dynamic input."
        )
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr_bytes.decode().strip(),
            "returncode": process.returncode,
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


async def execute_safe_command(args: List[str], timeout: int = 120) -> Dict[str, Any]:
    """Execute a command safely using subprocess with list args (no shell=True).

    This is injection-safe. Use this for all commands with user/AI-provided inputs.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr_bytes.decode().strip(),
            "returncode": process.returncode,
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def enhance_error_message(stderr: str, args: List[str]) -> str:
    """Add helpful guidance to error messages."""
    # Check for common API key errors
    if "TERRADEV_RUNPOD_KEY" in stderr or "RunPod" in stderr:
        return (
            f"{stderr}\n\n"
            "💡 Looks like TERRADEV_RUNPOD_KEY isn't set.\n"
            "   Run: terradev setup runpod --quick\n"
            "   Or set: export TERRADEV_RUNPOD_KEY=your_key_here"
        )

    if "AWS" in stderr and "credentials" in stderr.lower():
        return (
            f"{stderr}\n\n"
            "💡 AWS credentials not configured.\n"
            "   Run: aws configure\n"
            "   Or: terradev setup aws --quick"
        )

    if "GOOGLE" in stderr or "GCP" in stderr:
        return (
            f"{stderr}\n\n"
            "💡 Google Cloud credentials not found.\n"
            "   Run: gcloud auth application-default login\n"
            "   Or: terradev setup gcp --quick"
        )

    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        # Extract module name
        import re

        match = re.search(r"No module named '([^']+)'", stderr)
        if match:
            module = match.group(1)
            return (
                f"{stderr}\n\n"
                f"💡 Missing Python package: {module}\n"
                f"   Run: pip install {module}"
            )

    if "permission denied" in stderr.lower():
        return (
            f"{stderr}\n\n"
            "💡 Permission denied. Try:\n"
            "   • Check file permissions\n"
            "   • Run with appropriate access rights\n"
            "   • Verify API key has required permissions"
        )

    # Return original error if no enhancement needed
    return stderr


# Execute generic Terraform command
async def execute_terraform_command(cmd: List[str], cwd: str) -> Dict[str, Any]:
    """Execute a Terraform command in the specified directory"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip(),
            "returncode": process.returncode,
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


# Execute Terraform for parallel provisioning
async def execute_terraform_parallel(
    gpu_type: str, count: int, providers: List[str] = None, max_price: float = None
) -> Dict[str, Any]:
    """Execute Terraform-based parallel provisioning for optimal efficiency"""

    # Use persistent workspace so terraform.tfstate survives for destroy/plan
    workspace_name = f"provision-{gpu_type}-x{count}"
    ws_dir = _get_tf_workspace(workspace_name)
    try:
        # Generate Terraform configuration for parallel provisioning
        terraform_config = generate_terraform_config(
            gpu_type, count, providers, max_price
        )

        # Write main.tf
        main_tf_path = os.path.join(ws_dir, "main.tf")
        with open(main_tf_path, "w") as f:
            f.write(terraform_config)

        # Write variables.tf
        vars_tf_path = os.path.join(ws_dir, "variables.tf")
        with open(vars_tf_path, "w") as f:
            f.write(generate_variables_file())

        # Initialize Terraform
        init_result = await asyncio.create_subprocess_exec(
            "terraform",
            "init",
            cwd=ws_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        init_stdout, init_stderr = await init_result.communicate()

        if init_result.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Terraform init failed: {init_stderr.decode()}",
                "returncode": init_result.returncode,
            }

        # Plan Terraform (dry run)
        plan_result = await asyncio.create_subprocess_exec(
            "terraform",
            "plan",
            "-out=tfplan",
            cwd=ws_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        plan_stdout, plan_stderr = await plan_result.communicate()

        if plan_result.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Terraform plan failed: {plan_stderr.decode()}",
                "returncode": plan_result.returncode,
            }

        # Apply Terraform
        apply_result = await asyncio.create_subprocess_exec(
            "terraform",
            "apply",
            "-auto-approve",
            "tfplan",
            cwd=ws_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        apply_stdout, apply_stderr = await apply_result.communicate()

        # Get outputs
        output_result = await asyncio.create_subprocess_exec(
            "terraform",
            "output",
            "-json",
            cwd=ws_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output_stdout, output_stderr = await output_result.communicate()

        outputs = {}
        if output_result.returncode == 0:
            try:
                outputs = json.loads(output_stdout.decode())
            except json.JSONDecodeError:
                pass

        return {
            "success": apply_result.returncode == 0,
            "stdout": apply_stdout.decode(),
            "stderr": apply_stderr.decode(),
            "returncode": apply_result.returncode,
            "terraform_outputs": outputs,
            "plan_output": plan_stdout.decode(),
            "workspace": workspace_name,
            "workspace_path": ws_dir,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Terraform execution failed: {str(e)}",
            "returncode": -1,
        }


def generate_k8s_terraform_config(
    cluster_name: str,
    gpu_type: str,
    node_count: int,
    multi_cloud: bool = False,
    prefer_spot: bool = True,
) -> str:
    """Generate Terraform configuration for Kubernetes clusters"""

    config = f"""
terraform {{
  required_providers {{
    terradev = {{
      source  = "theoddden/terradev"
      version = "~> 3.0"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }}
  }}
}}

variable "cluster_name" {{
  description = "Kubernetes cluster name"
  type        = string
  default     = "{cluster_name}"
}}

variable "gpu_type" {{
  description = "GPU type for nodes"
  type        = string
  default     = "{gpu_type}"
}}

variable "node_count" {{
  description = "Number of GPU nodes"
  type        = number
  default     = {node_count}
}}

variable "multi_cloud" {{
  description = "Use multi-cloud node pools"
  type        = bool
  default     = {str(multi_cloud).lower()}
}}

variable "prefer_spot" {{
  description = "Prefer spot instances"
  type        = bool
  default     = {str(prefer_spot).lower()}
}}

# Kubernetes cluster with GPU nodes
resource "terradev_kubernetes_cluster" "main" {{
  name        = var.cluster_name
  gpu_type    = var.gpu_type
  node_count  = var.node_count
  spot        = var.prefer_spot
  
  tags = {{
    Name        = var.cluster_name
    Provisioned = "terraform"
    GPU_Type    = var.gpu_type
    MultiCloud  = var.multi_cloud
  }}
}}
"""

    if multi_cloud:
        # Add multi-cloud node pools for enhanced resilience
        providers = [
            "runpod",
            "vastai",
            "lambda",
            "aws",
            "alibaba",
            "ovhcloud",
            "fluidstack",
            "hetzner",
            "siliconflow",
            "latitude",
        ]
        for i, provider in enumerate(providers[:node_count]):
            config += (
                "\n# Multi-cloud node pool - " + provider + "\n"
                'resource "terradev_node_pool" "pool_' + str(i) + '" {\n'
                "  cluster_name = terradev_kubernetes_cluster.main.name\n"
                '  provider     = "' + provider + '"\n'
                "  gpu_type     = var.gpu_type\n"
                "  node_count   = 1\n"
                "  spot         = var.prefer_spot\n"
                "\n"
                "  depends_on = [terradev_kubernetes_cluster.main]\n"
                "\n"
                "  tags = {\n"
                '    Name        = "${var.cluster_name}-pool-' + str(i) + '"\n'
                '    Provider    = "' + provider + '"\n'
                '    Provisioned = "terraform"\n'
                "  }\n"
                "}\n"
            )

    # Add outputs
    config += """
# Cluster outputs
output "cluster_name" {
  description = "Kubernetes cluster name"
  value       = terradev_kubernetes_cluster.main.name
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = terradev_kubernetes_cluster.main.endpoint
}

output "kubeconfig" {
  description = "Kubernetes configuration"
  value       = terradev_kubernetes_cluster.main.kubeconfig
  sensitive   = true
}

output "node_pools" {
  description = "Node pool information"
  value = {
"""

    if multi_cloud:
        for i in range(min(node_count, len(providers))):
            config += f"    pool_{i} = terradev_node_pool.pool_{i}[*].id,\n"

    config += """  }
}
"""

    return config


def generate_inference_terraform_config(
    model: str, gpu_type: str, endpoint_name: str = None
) -> str:
    """Generate Terraform configuration for inference deployments"""

    endpoint_name = (
        endpoint_name or f"inferx-{model.replace('/', '-').replace(':', '-')}"
    )

    config = (
        "terraform {\n"
        "  required_providers {\n"
        "    terradev = {\n"
        '      source  = "theoddden/terradev"\n'
        '      version = "~> 3.0"\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        'variable "model" {\n'
        '  description = "Model ID for deployment"\n'
        "  type        = string\n"
        '  default     = "' + model + '"\n'
        "}\n\n"
        'variable "gpu_type" {\n'
        '  description = "GPU type for inference"\n'
        "  type        = string\n"
        '  default     = "' + gpu_type + '"\n'
        "}\n\n"
        'variable "endpoint_name" {\n'
        '  description = "Inference endpoint name"\n'
        "  type        = string\n"
        '  default     = "' + endpoint_name + '"\n'
        "}\n\n"
    )

    config += """
# InferX serverless endpoint
resource "terradev_inference_endpoint" "main" {
  name        = var.endpoint_name
  model       = var.model
  gpu_type    = var.gpu_type

  tags = {
    Name        = var.endpoint_name
    Model       = var.model
    GPU_Type    = var.gpu_type
    Provisioned = "terraform"
  }
}

# HuggingFace Spaces deployment (optional)
resource "terradev_hf_space" "main" {
  count       = contains(["A10G", "L4", "T4"], var.gpu_type) ? 1 : 0
  name        = var.endpoint_name
  model_id    = var.model
  hardware    = var.gpu_type
  sdk         = "gradio"

  tags = {
    Name        = var.endpoint_name
    Model       = var.model
    Hardware    = var.gpu_type
    Provisioned = "terraform"
  }
}

# Outputs
output "endpoint_url" {
  description = "Inference endpoint URL"
  value       = terradev_inference_endpoint.main.url
}

output "endpoint_status" {
  description = "Endpoint deployment status"
  value       = terradev_inference_endpoint.main.status
}

output "hf_space_url" {
  description = "HuggingFace Space URL"
  value       = length(terradev_hf_space.main) > 0 ? terradev_hf_space.main[0].url : null
}
"""

    return config


def generate_terraform_config(
    gpu_type: str, count: int, providers: List[str] = None, max_price: float = None
) -> str:
    """Generate Terraform configuration for parallel GPU provisioning"""

    providers = providers or [
        "runpod",
        "vastai",
        "lambda",
        "aws",
        "alibaba",
        "ovhcloud",
        "fluidstack",
        "hetzner",
        "siliconflow",
    ]

    config = f"""
terraform {{
  required_providers {{
    terradev = {{
      source = "theoddden/terradev"
      version = "~> 3.0"
    }}
  }}
}}

variable "gpu_type" {{
  description = "GPU type to provision"
  type        = string
  default     = "{gpu_type}"
}}

variable "gpu_count" {{
  description = "Number of GPUs to provision"
  type        = number
  default     = {count}
}}

variable "max_price" {{
  description = "Maximum price per hour"
  type        = number
  default     = {max_price if max_price else "null"}
}}

# Parallel provisioning across multiple providers
"""

    # Add provider blocks for parallel provisioning
    for i, provider in enumerate(providers[:count]):  # Distribute across providers
        config += (
            '\nresource "terradev_instance" "gpu_' + str(i) + '" {\n'
            "  gpu_type    = var.gpu_type\n"
            '  provider    = "' + provider + '"\n'
            "  spot        = true\n"
            "  count       = 1\n"
            "\n"
            "  # Dynamic pricing and availability\n"
            '  dynamic "pricing" {\n'
            "    for_each = var.max_price != null ? [1] : []\n"
            "    content {\n"
            "      max_hourly = var.max_price\n"
            "    }\n"
            "  }\n"
            "\n"
            "  tags = {\n"
            '    Name        = "terradev-mcp-gpu-' + str(i) + '"\n'
            '    Provisioned = "terraform"\n'
            "    GPU_Type    = var.gpu_type\n"
            "  }\n"
            "}\n\n"
        )

    # Add outputs for instance information
    config += """
# Outputs for instance management
output "instance_ids" {
  description = "Provisioned instance IDs"
  value = [
"""

    for i in range(min(count, len(providers))):
        config += f"    terradev_instance.gpu_{i}[*].id,\n"

    config += """  ]
}

output "instance_ips" {
  description = "Instance IP addresses"
  value = [
"""

    for i in range(min(count, len(providers))):
        config += f"    terradev_instance.gpu_{i}[*].public_ip,\n"

    config += """  ]
}

output "provider_costs" {
  description = "Hourly costs by provider"
  value = {
"""

    for i, provider in enumerate(providers[:count]):
        config += f"    {provider} = terradev_instance.gpu_{i}[*].hourly_cost,\n"

    config += """  }
}
"""

    return config


def generate_variables_file() -> str:
    """Generate Terraform variables file"""
    return """
variable "gpu_type" {
  description = "GPU type to provision"
  type        = string
  
  validation {
    condition = contains([
      "H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"
    ], var.gpu_type)
    error_message = "GPU type must be one of: H100, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100."
  }
}

variable "gpu_count" {
  description = "Number of GPUs to provision"
  type        = number
  
  validation {
    condition     = var.gpu_count > 0 && var.gpu_count <= 32
    error_message = "GPU count must be between 1 and 32."
  }
}

variable "max_price" {
  description = "Maximum price per hour"
  type        = number
  default     = null
  
  validation {
    condition     = var.max_price == null || var.max_price > 0
    error_message = "Max price must be null or greater than 0."
  }
}
"""


# Create MCP server with enhanced scaling (only if MCP is available)
if MCP_AVAILABLE:
    server = Server("terradev-mcp")
else:
    # Dummy server for when MCP is not installed - allows decorators to be defined
    class _DummyServer:
        def list_tools(self):
            return lambda f: f
        def call_tool(self):
            return lambda f: f
        def list_resources(self):
            return lambda f: f
        def read_resource(self):
            return lambda f: f
        def list_prompts(self):
            return lambda f: f
        def get_prompt(self):
            return lambda f: f
    server = _DummyServer()

# Global state for lazy loading and scaling
_tools_loaded = False
_tool_registry: Dict[str, callable] = {}
_tool_schemas: Dict[str, Dict] = {}
_load_lock: Optional[asyncio.Lock] = None
_concurrent_requests = 0
_max_concurrent = 100
_request_semaphore: Optional[asyncio.Semaphore] = None


async def _ensure_tools_loaded():
    """Ensure tools are loaded (lazy loading)"""
    global _tools_loaded, _tool_registry, _tool_schemas, _load_lock

    # Lazy-create lock to avoid Python 3.9 event loop binding bug
    if _load_lock is None:
        _load_lock = asyncio.Lock()

    if _tools_loaded:
        return

    async with _load_lock:
        if _tools_loaded:
            return

        # Load tools on first request
        logger.info("Loading MCP tools (lazy loading)...")
        _build_all_tools()
        _tools_loaded = True
        logger.info(f"Loaded {len(_ALL_TOOLS)} tools successfully")


# Import Rust-based optimizer for tool compression and parallel dispatch
try:
    from terradev_mcp_optimizer import MCPOptimizer as RustMCPOptimizer

    optimizer = RustMCPOptimizer(
        enable_compression=True, strip_optional=True, enable_parallel=True
    )
    logger.info("Using Rust-based MCPOptimizer for 10-50x faster tool compression")
except ImportError:
    # Python fallback implementation
    optimizer = None
    logger.warning(
        "Rust MCPOptimizer not available, using Python fallback (install Rust and build for 10-50x speedup)"
    )

# ── Pre-compiled Tool Schemas (built once at module load) ────────────────────
_ALL_TOOLS = None

def _build_all_tools():
    global _ALL_TOOLS
    _ALL_TOOLS = []

    if MCP_AVAILABLE:
        _ALL_TOOLS = [
            Tool(
                name="quote_gpu",
                description="Get real-time GPU prices across 21+ cloud providers (incl. Alibaba, OVHcloud, FluidStack, Hetzner, SiliconFlow, Latitude.sh, Oracle, Crusoe, DigitalOcean)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "gpu_type": {
                        "type": "string",
                        "description": "GPU type (H100, H200, H800, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100, V100S, A6000)",
                        "enum": [
                            "H100",
                            "H200",
                            "H800",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "RTX3090",
                            "V100",
                            "V100S",
                            "A6000",
                        ],
                    },
                    "providers": {
                        "type": "string",
                        "description": "Optional comma-separated list of providers to filter",
                        "pattern": "^[a-z,-]*$",
                    },
                    "quick": {
                        "type": "boolean",
                        "description": "Quick-provision the cheapest option",
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        Tool(
            name="provision_gpu",
            description="Provision GPU instances using Terraform for optimal parallel efficiency",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to provision",
                        "enum": [
                            "H100",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "RTX3090",
                            "V100",
                        ],
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of GPUs to provision",
                        "minimum": 1,
                        "default": 1,
                    },
                    "providers": {
                        "type": "array",
                        "description": "Cloud providers for parallel distribution",
                        "items": {
                            "type": "string",
                            "enum": [
                                "runpod",
                                "vastai",
                                "lambda",
                                "aws",
                                "gcp",
                                "azure",
                                "coreweave",
                                "tensordock",
                                "oracle",
                                "crusoe",
                                "digitalocean",
                                "hyperstack",
                                "alibaba",
                                "ovhcloud",
                                "fluidstack",
                                "hetzner",
                                "siliconflow",
                                "latitude",
                                "huggingface",
                                "baseten",
                                "inferx",
                            ],
                        },
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per hour",
                        "minimum": 0,
                    },
                    "plan_only": {
                        "type": "boolean",
                        "description": "Generate Terraform plan without applying",
                        "default": False,
                    },
                    "state_file": {
                        "type": "string",
                        "description": "Terraform state file path (optional)",
                        "default": None,
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        Tool(
            name="terraform_plan",
            description="Generate Terraform execution plan for GPU provisioning",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration",
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)",
                    },
                    "destroy": {
                        "type": "boolean",
                        "description": "Generate destroy plan",
                        "default": False,
                    },
                },
                "required": ["config_dir"],
            },
        ),
        Tool(
            name="terraform_apply",
            description="Apply Terraform configuration for GPU provisioning",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration",
                    },
                    "plan_file": {
                        "type": "string",
                        "description": "Terraform plan file to apply (optional)",
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)",
                    },
                    "auto_approve": {
                        "type": "boolean",
                        "description": "Auto-approve the apply",
                        "default": True,
                    },
                },
                "required": ["config_dir"],
            },
        ),
        Tool(
            name="terraform_destroy",
            description="Destroy Terraform-managed GPU infrastructure",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration",
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)",
                    },
                    "auto_approve": {
                        "type": "boolean",
                        "description": "Auto-approve the destroy",
                        "default": True,
                    },
                },
                "required": ["config_dir"],
            },
        ),
        Tool(
            name="local_scan",
            description="Scan local machine and network for available GPU devices. Returns total VRAM pool for local-first provisioning.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="k8s_create",
            description="Create Kubernetes cluster with GPU nodes using Terraform for optimal multi-cloud deployment",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the cluster",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for nodes",
                        "enum": [
                            "H100",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "RTX3090",
                            "V100",
                        ],
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of GPU nodes",
                        "minimum": 1,
                        "default": 1,
                    },
                    "multi_cloud": {
                        "type": "boolean",
                        "description": "Use multi-cloud node pools for resilience",
                        "default": False,
                    },
                    "prefer_spot": {
                        "type": "boolean",
                        "description": "Prefer spot instances for cost savings",
                        "default": True,
                    },
                    "use_terraform": {
                        "type": "boolean",
                        "description": "Use Terraform for infrastructure as code",
                        "default": True,
                    },
                },
                "required": ["cluster_name", "gpu_type"],
            },
        ),
        Tool(
            name="k8s_list",
            description="List Kubernetes clusters",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="k8s_info",
            description="Get information about a specific cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {"type": "string", "description": "Name of the cluster"}
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="k8s_destroy",
            description="Destroy a Kubernetes cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the cluster to destroy",
                    }
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="inferx_deploy",
            description="Deploy model to InferX serverless platform using Terraform for infrastructure as code",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., meta-llama/Llama-2-7b-hf)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for inference",
                        "enum": [
                            "H100",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "RTX3090",
                            "V100",
                        ],
                    },
                    "endpoint_name": {
                        "type": "string",
                        "description": "Custom endpoint name (optional)",
                    },
                    "use_terraform": {
                        "type": "boolean",
                        "description": "Use Terraform for infrastructure as code",
                        "default": True,
                    },
                },
                "required": ["model", "gpu_type"],
            },
        ),
        Tool(
            name="inferx_status",
            description="Check InferX endpoint status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="inferx_list",
            description="List deployed InferX models",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="inferx_optimize",
            description="Get cost analysis for inference endpoints",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="hf_space_deploy",
            description="Deploy model to HuggingFace Spaces",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_name": {"type": "string", "description": "Name of the Space"},
                    "model_id": {"type": "string", "description": "HuggingFace model ID"},
                    "template": {
                        "type": "string",
                        "description": "Template type",
                        "enum": ["llm", "embedding", "image"],
                    },
                    "hardware": {"type": "string", "description": "Hardware specification"},
                    "sdk": {
                        "type": "string",
                        "description": "SDK to use (e.g., gradio, streamlit)",
                    },
                },
                "required": ["space_name", "model_id", "template"],
            },
        ),
        Tool(
            name="status",
            description="View all instances and costs with Terraform state optimization",
            inputSchema={
                "type": "object",
                "properties": {
                    "live": {
                        "type": "boolean",
                        "description": "Show live status vs cached Terraform state",
                        "default": False,
                    },
                    "use_terraform_state": {
                        "type": "boolean",
                        "description": "Use Terraform state for faster queries",
                        "default": True,
                    },
                    "state_file": {
                        "type": "string",
                        "description": "Path to Terraform state file (optional)",
                    },
                },
            },
        ),
        Tool(
            name="terraform_status",
            description="Fast status query using Terraform state",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration",
                    },
                    "show_outputs": {
                        "type": "boolean",
                        "description": "Show Terraform outputs",
                        "default": True,
                    },
                },
                "required": ["config_dir"],
            },
        ),
        Tool(
            name="manage_instance",
            description="Manage GPU instances (stop/start/terminate)",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_id": {"type": "string", "description": "Instance ID"},
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": ["stop", "start", "terminate"],
                    },
                },
                "required": ["instance_id", "action"],
            },
        ),
        Tool(
            name="analytics",
            description="Get cost analytics",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze",
                        "minimum": 1,
                        "default": 30,
                    }
                },
            },
        ),
        Tool(
            name="optimize",
            description="Find cheaper alternatives for running instances",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="setup_provider",
            description="Get setup instructions for a provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                        "enum": [
                            "runpod",
                            "aws",
                            "vastai",
                            "gcp",
                            "azure",
                            "lambda",
                            "coreweave",
                            "tensordock",
                            "oracle",
                            "crusoe",
                            "digitalocean",
                            "hyperstack",
                            "alibaba",
                            "ovhcloud",
                            "fluidstack",
                            "hetzner",
                            "siliconflow",
                            "latitude",
                            "huggingface",
                            "baseten",
                            "inferx",
                        ],
                    },
                    "quick": {
                        "type": "boolean",
                        "description": "Quick setup instructions",
                        "default": True,
                    },
                },
                "required": ["provider"],
            },
        ),
        Tool(
            name="configure_provider",
            description="Configure provider credentials",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to configure",
                        "enum": [
                            "runpod",
                            "aws",
                            "vastai",
                            "gcp",
                            "azure",
                            "lambda",
                            "coreweave",
                            "tensordock",
                            "oracle",
                            "crusoe",
                            "digitalocean",
                            "hyperstack",
                            "alibaba",
                            "ovhcloud",
                            "fluidstack",
                            "hetzner",
                            "siliconflow",
                            "latitude",
                            "huggingface",
                            "baseten",
                            "inferx",
                        ],
                    }
                },
                "required": ["provider"],
            },
        ),
        # ── v3.2.0 Tools: Semantic Routing, Disaggregated Inference, GPU Topology, Price Intelligence ──
        Tool(
            name="infer_route",
            description="Semantic-aware inference routing. Analyzes query content across 6 signal dimensions (modality, complexity, domain, language, safety, keywords), applies NUMA-aware endpoint scoring, and selects the optimal inference endpoint. Uses Terraform-style DAG parallel execution for signal extraction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Filter by model name (e.g., llama-3-70b, deepseek-coder-33b)",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Routing strategy",
                        "enum": ["latency", "cost", "score"],
                        "default": "latency",
                    },
                    "measure": {
                        "type": "boolean",
                        "description": "Run fresh latency measurements (HTTP TTFB / WebPageTest) before routing",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="infer_route_disagg",
            description="Disaggregated Prefill/Decode routing (DistServe architecture). Splits LLM inference into compute-bound prefill phase (routed to FLOPS-optimized GPUs like H100 SXM) and memory-bound decode phase (routed to bandwidth-optimized GPUs like MI300X). Tracks KV cache handoffs between endpoint pairs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name for disaggregated pair selection (e.g., llama-3-70b)",
                    },
                    "check_health": {
                        "type": "boolean",
                        "description": "Run health probes before selecting pairs",
                        "default": True,
                    },
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="infer_status",
            description="Show inference endpoint health, latency, failover status, and disaggregated phase assignments (PREFILL/DECODE/MIXED). Includes KV cache warm status per endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "check": {
                        "type": "boolean",
                        "description": "Run live health probes before showing status",
                        "default": False,
                    }
                },
            },
        ),
        Tool(
            name="infer_failover",
            description="Run health checks and auto-failover for inference endpoints. If a primary endpoint is unhealthy and has a backup configured, traffic automatically shifts to the backup provider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show what would happen without executing failover",
                        "default": False,
                    }
                },
            },
        ),
        Tool(
            name="gpu_topology",
            description="GPU NUMA topology report with intra-GPU XCD (Accelerated Compute Die) awareness. Models MI300X (8 XCDs, 192GB HBM3), MI300A (6 XCDs, 128GB), H200 (unified 141GB HBM3e), H100 (80GB). Reports PCIe locality (PIX/PXB/PHB/SYS), GPU-NIC pairing, SR-IOV VF status, and generates XCD-aware NCCL/AITER environment variables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_arch": {
                        "type": "string",
                        "description": "Filter by GPU architecture",
                        "enum": ["mi300x", "mi300a", "h200", "h100", "a100", "auto"],
                    },
                    "generate_env": {
                        "type": "boolean",
                        "description": "Generate XCD-aware NCCL/AITER/CK env vars for optimal attention kernel performance",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="price_intel",
            description="GPU price intelligence with quantitative analytics. Computes delta (rate of change), gamma (acceleration), and annualized realized volatility on GPU spot/on-demand prices across 21+ providers. Identifies cheapest time windows and provider arbitrage opportunities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to analyze",
                        "enum": [
                            "H100",
                            "H200",
                            "H800",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "RTX3090",
                            "V100",
                            "V100S",
                            "A6000",
                            "MI300X",
                        ],
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history to analyze",
                        "minimum": 1,
                        "default": 7,
                    },
                    "provider": {
                        "type": "string",
                        "description": "Filter to specific provider (optional)",
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        Tool(
            name="moe_deploy",
            description="Deploy Mixture-of-Experts models with production-ready cluster templates. Auto-applies vLLM cost optimizations (KV cache offloading for up to 9x throughput, MTP speculative decoding for up to 2.8x speed, sleep mode for 18-200x faster restarts). Supports GLM-5, Qwen 3.5, Mistral Large 3, DeepSeek V4, Llama 5. Configures NVLink topology, tensor parallelism, FP8 quantization, vLLM/SGLang backends, and GPU-aware HPA autoscaling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g., zai-org/GLM-5-FP8, Qwen/Qwen3.5-397B-A17B)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for MoE serving",
                        "enum": ["H100", "H200", "H800", "A100", "MI300X"],
                    },
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallelism size (must match NVLink domain)",
                        "enum": [1, 2, 4, 8],
                        "default": 8,
                    },
                    "backend": {
                        "type": "string",
                        "description": "Serving backend",
                        "enum": ["vllm", "sglang"],
                        "default": "vllm",
                    },
                    "quantization": {
                        "type": "string",
                        "description": "Quantization method",
                        "enum": ["fp8", "bf16", "awq", "gptq"],
                        "default": "fp8",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show deployment plan without executing",
                        "default": False,
                    },
                },
                "required": ["model_id", "gpu_type"],
            },
        ),
        Tool(
            name="gitops_init",
            description="Initialize GitOps repository with ArgoCD or Flux CD. Creates cluster manifests, app definitions, policy-as-code templates, and multi-environment structure. Supports GitHub, GitLab, Bitbucket, Azure DevOps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Git repository (e.g., my-org/infra)",
                    },
                    "tool": {
                        "type": "string",
                        "description": "GitOps tool to use",
                        "enum": ["argocd", "flux"],
                        "default": "argocd",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Git provider",
                        "enum": ["github", "gitlab", "bitbucket", "azure-devops"],
                        "default": "github",
                    },
                    "cluster": {"type": "string", "description": "Target cluster name"},
                },
                "required": ["repo"],
            },
        ),
        # ── v3.4.0 Tools: Training Pipeline, Checkpoints, Monitoring ──
        Tool(
            name="train",
            description="Launch distributed training on provisioned GPU nodes. Supports torchrun, deepspeed, accelerate, and megatron. Use from_provision='latest' to auto-resolve node IPs from your last provision command.",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Training script path (e.g., train.py)",
                    },
                    "framework": {
                        "type": "string",
                        "description": "Training framework",
                        "enum": ["torchrun", "deepspeed", "accelerate", "megatron"],
                        "default": "torchrun",
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision. Use 'latest' or a parallel_group_id.",
                    },
                    "nodes": {
                        "type": "array",
                        "description": "Manual node IPs",
                        "items": {"type": "string"},
                    },
                    "gpus_per_node": {
                        "type": "integer",
                        "description": "GPUs per node",
                        "default": 8,
                    },
                    "script_args": {
                        "type": "string",
                        "description": "Extra args for training script",
                    },
                },
                "required": ["script"],
            },
        ),
        Tool(
            name="train_status",
            description="List all training jobs and their state (created, running, completed, failed).",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Filter to a specific job ID",
                    }
                },
            },
        ),
        Tool(
            name="train_monitor",
            description="Real-time GPU monitoring for training jobs. Shows utilization, memory, temperature, power, and cost.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to monitor"},
                    "cost_rate": {
                        "type": "number",
                        "description": "$/GPU-hr for cost estimation",
                        "default": 2.0,
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="checkpoint_list",
            description="List all checkpoints for a training job.",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Job ID"}},
                "required": ["job_id"],
            },
        ),
        Tool(
            name="checkpoint_save",
            description="Manually trigger a checkpoint save for a running training job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "step": {"type": "integer", "description": "Step number"},
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="preflight",
            description="Pre-training validation: GPU availability, NCCL, RDMA, drivers across all nodes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "Node IPs to validate",
                        "items": {"type": "string"},
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision. 'latest' or a group ID.",
                    },
                },
            },
        ),
        Tool(
            name="price_discovery",
            description="Enhanced price discovery with capacity scoring, confidence intervals, and trend analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type",
                        "enum": [
                            "H100",
                            "H200",
                            "H800",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "V100S",
                            "A6000",
                            "MI300X",
                        ],
                    },
                    "region": {"type": "string", "description": "Filter by region"},
                    "hours": {
                        "type": "integer",
                        "description": "Hours of history",
                        "default": 24,
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        # ── v2.0.0 Tools: Complete Agentic Loop ─────────────────────────────
        # Training pipeline completion
        Tool(
            name="train_stop",
            description="Stop a running training job. Kills training processes on all nodes in parallel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to stop"}
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="train_resume",
            description="Resume a training job from its latest checkpoint. Rebuilds config with topology revalidation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to resume"},
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Specific checkpoint to resume from (default: latest)",
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="checkpoint_restore",
            description="Restore a specific checkpoint for a training job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "step": {
                        "type": "integer",
                        "description": "Checkpoint step to restore",
                    },
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Checkpoint ID to restore",
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="checkpoint_promote",
            description="Promote a checkpoint to a final model path for serving.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Checkpoint ID to promote",
                    },
                    "dest": {
                        "type": "string",
                        "description": "Destination path (e.g., /models/final)",
                    },
                },
                "required": ["job_id", "checkpoint_id", "dest"],
            },
        ),
        Tool(
            name="checkpoint_delete",
            description="Delete a checkpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Checkpoint ID to delete",
                    },
                },
                "required": ["job_id", "checkpoint_id"],
            },
        ),
        # Data staging
        Tool(
            name="stage",
            description="Compress, chunk, checksum, and position datasets near compute. Supports local paths, S3/GCS URIs, HTTP URLs, and HuggingFace dataset names. Returns staging plan with agent recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset path, S3 URI, GCS URI, HTTP URL, or HuggingFace name",
                    },
                    "target_regions": {
                        "type": "string",
                        "description": "Comma-separated target regions",
                    },
                    "compression": {
                        "type": "string",
                        "description": "Compression algorithm",
                        "enum": ["auto", "zstd", "gzip", "none"],
                        "default": "auto",
                    },
                    "plan_only": {
                        "type": "boolean",
                        "description": "Show staging plan without executing",
                        "default": False,
                    },
                },
                "required": ["dataset"],
            },
        ),
        # Inference deployment
        Tool(
            name="infer_deploy",
            description="Deploy an inference endpoint with auto-scaling, idle shutdown, and cost optimization. Returns estimated cost and requires_confirmation for expensive deployments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Model path or HuggingFace model ID",
                    },
                    "name": {"type": "string", "description": "Endpoint name"},
                    "provider": {
                        "type": "string",
                        "description": "Provider for deployment",
                        "enum": [
                            "runpod",
                            "vastai",
                            "lambda",
                            "aws",
                            "gcp",
                            "coreweave",
                            "alibaba",
                            "ovhcloud",
                            "fluidstack",
                            "hetzner",
                            "siliconflow",
                            "latitude",
                        ],
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type",
                        "enum": [
                            "H100",
                            "H200",
                            "H800",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "V100S",
                            "A6000",
                            "MI300X",
                        ],
                    },
                    "min_workers": {
                        "type": "integer",
                        "description": "Minimum workers",
                        "default": 0,
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Maximum workers",
                        "default": 3,
                    },
                    "idle_timeout": {
                        "type": "integer",
                        "description": "Idle timeout in seconds",
                        "default": 300,
                    },
                    "cost_optimize": {
                        "type": "boolean",
                        "description": "Enable cost optimization",
                        "default": True,
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show deployment plan without deploying",
                        "default": False,
                    },
                },
                "required": ["model_path", "name"],
            },
        ),
        # Manifest cache & drift detection
        Tool(
            name="up",
            description="CLI-native provisioning with manifest cache and drift detection. Use --fix-drift to detect and auto-fix drifted infrastructure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {
                        "type": "string",
                        "description": "Job name for manifest tracking",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type",
                        "default": "A100",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs",
                        "default": 1,
                    },
                    "hours": {
                        "type": "number",
                        "description": "Estimated runtime in hours",
                        "default": 1.0,
                    },
                    "budget": {"type": "number", "description": "Budget constraint ($/hr)"},
                    "region": {"type": "string", "description": "Preferred region"},
                    "ttl": {
                        "type": "string",
                        "description": "Time to live for nodes",
                        "default": "1h",
                    },
                    "fix_drift": {
                        "type": "boolean",
                        "description": "Detect and fix drift automatically",
                        "default": False,
                    },
                },
                "required": ["job"],
            },
        ),
        Tool(
            name="rollback",
            description="Explicit versioned rollback. Format: job@version (e.g., llama3@v3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_version": {
                        "type": "string",
                        "description": "Job@version string (e.g., llama3@v3)",
                    }
                },
                "required": ["job_version"],
            },
        ),
        Tool(
            name="manifests",
            description="List cached manifests and versions for jobs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {
                        "type": "string",
                        "description": "Show versions for specific job (optional)",
                    }
                },
            },
        ),
        # Smart deployment
        Tool(
            name="smart_deploy",
            description="AI-ranked deployment options with confidence/risk scoring. Returns recommendations with cost estimates and requires_confirmation for execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Container image"},
                    "workload": {
                        "type": "string",
                        "description": "Workload type",
                        "enum": ["training", "inference", "batch"],
                    },
                    "gpu_type": {"type": "string", "description": "GPU type"},
                    "budget": {"type": "number", "description": "Max $/hr budget"},
                    "option": {
                        "type": "integer",
                        "description": "Execute a specific recommended option by index",
                    },
                },
                "required": ["image", "workload"],
            },
        ),
        Tool(
            name="helm_generate",
            description="Generate Helm charts from workload specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workload": {
                        "type": "string",
                        "description": "Workload type",
                        "enum": ["training", "inference", "batch"],
                    },
                    "image": {"type": "string", "description": "Container image"},
                    "gpu_type": {"type": "string", "description": "GPU type"},
                    "replicas": {
                        "type": "integer",
                        "description": "Number of replicas",
                        "default": 1,
                    },
                },
                "required": ["workload", "image"],
            },
        ),
        # GitOps complete
        Tool(
            name="gitops_bootstrap",
            description="Bootstrap ArgoCD or Flux on the cluster.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "GitOps tool",
                        "enum": ["argocd", "flux"],
                    },
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "namespace": {
                        "type": "string",
                        "description": "Namespace",
                        "default": "gitops-system",
                    },
                },
                "required": ["tool", "cluster"],
            },
        ),
        Tool(
            name="gitops_sync",
            description="Sync cluster with Git repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "environment": {
                        "type": "string",
                        "description": "Environment to sync",
                        "default": "prod",
                    },
                    "tool": {
                        "type": "string",
                        "description": "GitOps tool",
                        "enum": ["argocd", "flux"],
                        "default": "argocd",
                    },
                },
                "required": ["cluster"],
            },
        ),
        Tool(
            name="gitops_validate",
            description="Validate GitOps configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "dry_run": {
                        "type": "boolean",
                        "description": "Dry run validation",
                        "default": True,
                    },
                },
            },
        ),
        # Orchestrator tools
        Tool(
            name="orchestrator_start",
            description="Start the model orchestrator for multi-model GPU sharing with eviction policies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_id": {"type": "integer", "description": "GPU ID", "default": 0},
                    "memory_gb": {
                        "type": "number",
                        "description": "Total GPU memory in GB",
                        "default": 80.0,
                    },
                    "policy": {
                        "type": "string",
                        "description": "Scaling policy",
                        "enum": ["billing_optimized", "latency_optimized", "hybrid"],
                        "default": "billing_optimized",
                    },
                },
            },
        ),
        Tool(
            name="orchestrator_register",
            description="Register a model with the orchestrator.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model identifier"},
                    "model_path": {
                        "type": "string",
                        "description": "Path to model weights",
                    },
                    "framework": {
                        "type": "string",
                        "description": "Framework",
                        "enum": ["pytorch", "vllm", "sglang"],
                        "default": "pytorch",
                    },
                },
                "required": ["model_id", "model_path"],
            },
        ),
        Tool(
            name="orchestrator_load",
            description="Load a model into GPU memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model to load"},
                    "force": {
                        "type": "boolean",
                        "description": "Force loading even if memory is full",
                        "default": False,
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="orchestrator_evict",
            description="Evict a model from GPU memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model to evict"}
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="orchestrator_status",
            description="Get orchestrator and model status including GPU memory utilization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Get details for specific model (optional)",
                    }
                },
            },
        ),
        Tool(
            name="orchestrator_infer",
            description="Test inference with a model via the orchestrator.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model to run inference on",
                    }
                },
                "required": ["model_id"],
            },
        ),
        # Warm pool tools
        Tool(
            name="warm_pool_start",
            description="Start the warm pool manager for intelligent model pre-warming. 5 strategies: traffic_based, time_based, priority_based, cost_optimized, latency_optimized.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Warm pool strategy",
                        "enum": [
                            "traffic_based",
                            "time_based",
                            "priority_based",
                            "cost_optimized",
                            "latency_optimized",
                        ],
                        "default": "traffic_based",
                    },
                    "max_warm": {
                        "type": "integer",
                        "description": "Max models to keep warm",
                        "default": 10,
                    },
                    "min_warm": {
                        "type": "integer",
                        "description": "Min models to keep warm",
                        "default": 3,
                    },
                },
            },
        ),
        Tool(
            name="warm_pool_status",
            description="Get warm pool status: hit rate, cold starts, memory saved, cost saved.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # Cost scaler tools
        Tool(
            name="cost_scaler_start",
            description="Start the budget-aware auto-scaling manager. 4 strategies: minimize_cost, balance_cost_latency, latency_critical, budget_constrained.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Cost strategy",
                        "enum": [
                            "minimize_cost",
                            "balance_cost_latency",
                            "latency_critical",
                            "budget_constrained",
                        ],
                        "default": "balance_cost_latency",
                    },
                    "budget": {
                        "type": "number",
                        "description": "Hourly budget in USD",
                        "default": 15.0,
                    },
                    "cost_per_gb": {
                        "type": "number",
                        "description": "Cost per GB per hour",
                        "default": 0.10,
                    },
                },
            },
        ),
        Tool(
            name="cost_scaler_status",
            description="Get cost scaler status with budget utilization, predictions, and optimization recommendations.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # InferX complete
        Tool(
            name="inferx_configure",
            description="Configure InferX serverless platform credentials.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "InferX API key"},
                    "endpoint": {
                        "type": "string",
                        "description": "InferX API endpoint",
                        "default": "https://api.inferx.net",
                    },
                    "region": {
                        "type": "string",
                        "description": "Region",
                        "default": "us-west-2",
                    },
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="inferx_delete",
            description="Delete an InferX model deployment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model deployment ID to delete",
                    }
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="inferx_usage",
            description="Get InferX account usage statistics: requests, cost, GPU hours, latency.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="inferx_quote",
            description="Get InferX pricing quotes for a GPU type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to quote",
                        "default": "A100",
                    },
                    "region": {"type": "string", "description": "Region for quote"},
                },
            },
        ),
        # HF Space deploy (already exists above)
        Tool(
            name="hf_space_status",
            description="Get HuggingFace Space deployment status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_name": {"type": "string", "description": "Space name to check"}
                },
                "required": ["space_name"],
            },
        ),
        # Workflow primitives
        Tool(
            name="run_workflow",
            description="Run a declarative YAML workflow that chains multiple Terradev commands (provision → preflight → train → monitor → checkpoint). Returns step-by-step execution status with cost estimates and confirmation gates for expensive operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Workflow YAML path or inline YAML string",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show execution plan without running",
                        "default": False,
                    },
                    "template": {
                        "type": "string",
                        "description": "Use built-in template",
                        "enum": [
                            "finetune-llama",
                            "inference-deploy",
                            "benchmark-gpu",
                            "cost-optimize",
                        ],
                    },
                },
            },
        ),
        # Active context — session-start awareness
        Tool(
            name="active_context",
            description="Get current Terradev state: running training jobs, active instances, spend-to-date, alerts. Call this on session start to resume context from previous sessions.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # ── v3.5.0: Multi-LoRA adapter management ────────────────────────
        Tool(
            name="lora_list",
            description="List LoRA adapters loaded on a running vLLM endpoint. Shows base models and hot-loaded fine-tuned adapters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"},
                },
                "required": ["endpoint"],
            },
        ),
        Tool(
            name="lora_add",
            description="Hot-load a LoRA adapter onto a running vLLM endpoint. The adapter becomes immediately available as a model name for inference requests. Uses vLLM's fused_moe_lora kernel for 454% higher output tokens/sec on MoE models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Adapter name (becomes the model name in API requests)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to adapter weights (local path or HuggingFace ID)",
                    },
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"},
                },
                "required": ["endpoint", "name", "path"],
            },
        ),
        Tool(
            name="lora_remove",
            description="Hot-unload a LoRA adapter from a running vLLM endpoint. Frees GPU memory for other adapters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "name": {"type": "string", "description": "Adapter name to unload"},
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"},
                },
                "required": ["endpoint", "name"],
            },
        ),
        # ── v4.0.0: ML Services — Ray ──────────────────────────────────────
        Tool(
            name="ray_status",
            description="Get Ray cluster status including node count, resources, memory, and running jobs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "detailed": {
                        "type": "boolean",
                        "description": "Include detailed memory and resource info",
                        "default": True,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="ray_start",
            description="Start a Ray cluster (head node or worker). For distributed ML training and inference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "head": {
                        "type": "boolean",
                        "description": "Start as head node (true) or worker (false)",
                        "default": True,
                    },
                    "port": {
                        "type": "integer",
                        "description": "Ray head port",
                        "default": 6379,
                    },
                    "num_gpus": {
                        "type": "integer",
                        "description": "Number of GPUs to expose",
                    },
                    "head_address": {
                        "type": "string",
                        "description": "Head node address for worker nodes (e.g. 10.0.0.1:6379)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="ray_stop",
            description="Stop the Ray cluster on the current node.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="ray_submit_job",
            description="Submit a job script to the Ray cluster for distributed execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Path to the Python script to submit",
                    },
                    "job_name": {"type": "string", "description": "Optional job name"},
                    "num_gpus": {
                        "type": "integer",
                        "description": "GPU resources to request",
                    },
                    "num_cpus": {
                        "type": "integer",
                        "description": "CPU resources to request",
                    },
                },
                "required": ["script"],
            },
        ),
        Tool(
            name="ray_list_jobs",
            description="List all running Ray jobs and tasks.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="ray_wide_ep_deploy",
            description="Generate a Ray Serve LLM Wide-EP (Expert Parallel) deployment for MoE models. Returns Python script and config for distributed MoE serving with EPLB and DeepEP.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g. zai-org/GLM-5-FP8, deepseek-ai/DeepSeek-V3)",
                    },
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallel size per EP rank",
                        "default": 1,
                    },
                    "dp_size": {
                        "type": "integer",
                        "description": "Data parallel / EP degree",
                        "default": 8,
                    },
                    "gpu_memory_utilization": {
                        "type": "number",
                        "description": "GPU memory fraction",
                        "default": 0.85,
                    },
                    "max_model_len": {
                        "type": "integer",
                        "description": "Max sequence length",
                        "default": 32768,
                    },
                    "generate_script": {
                        "type": "boolean",
                        "description": "Also generate executable Python script",
                        "default": True,
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="ray_disagg_pd_deploy",
            description="Generate a Ray Serve LLM disaggregated Prefill/Decode deployment. Splits inference into compute-bound prefill and memory-bound decode phases with KV cache transfer via NIXL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "HuggingFace model ID"},
                    "prefill_tp": {
                        "type": "integer",
                        "description": "Prefill tensor parallel size",
                        "default": 1,
                    },
                    "prefill_dp": {
                        "type": "integer",
                        "description": "Prefill data parallel size",
                        "default": 4,
                    },
                    "decode_tp": {
                        "type": "integer",
                        "description": "Decode tensor parallel size",
                        "default": 1,
                    },
                    "decode_dp": {
                        "type": "integer",
                        "description": "Decode data parallel size",
                        "default": 4,
                    },
                    "kv_connector": {
                        "type": "string",
                        "description": "KV transfer connector",
                        "enum": ["NixlConnector", "LMCacheConnector"],
                        "default": "NixlConnector",
                    },
                    "generate_script": {
                        "type": "boolean",
                        "description": "Also generate executable Python script",
                        "default": True,
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="ray_parallelism_strategy",
            description="Compute optimal TP/DP/EP parallelism strategy for a given MoE model and GPU count. Returns recommended configuration with rationale.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g. zai-org/GLM-5-FP8)",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of available GPUs",
                        "default": 8,
                    },
                    "gpu_memory_gb": {
                        "type": "number",
                        "description": "GPU memory per device in GB",
                        "default": 80.0,
                    },
                },
                "required": ["model_id"],
            },
        ),
        # ── v4.0.0: ML Services — vLLM Lifecycle ──────────────────────────
        Tool(
            name="vllm_start",
            description="Start a vLLM inference server on a remote instance via SSH/systemd. Supports Multi-LoRA, Sleep Mode, KV Offloading, Speculative Decoding.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_ip": {"type": "string", "description": "Target instance IP"},
                    "model": {"type": "string", "description": "HuggingFace model ID"},
                    "port": {
                        "type": "integer",
                        "description": "Server port",
                        "default": 8000,
                    },
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallel size",
                        "default": 1,
                    },
                    "gpu_memory_utilization": {
                        "type": "number",
                        "description": "GPU memory fraction",
                        "default": 0.9,
                    },
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                    "api_key": {"type": "string", "description": "vLLM API key to set"},
                },
                "required": ["instance_ip", "model"],
            },
        ),
        Tool(
            name="vllm_stop",
            description="Stop a vLLM server on a remote instance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_ip": {"type": "string", "description": "Target instance IP"},
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                },
                "required": ["instance_ip"],
            },
        ),
        Tool(
            name="vllm_inference",
            description="Test inference against a running vLLM endpoint (completions or chat).",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text (for completions mode)",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Chat messages array (for chat mode)",
                        "items": {"type": "object"},
                    },
                    "model": {"type": "string", "description": "Model name to use"},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens to generate",
                        "default": 100,
                    },
                    "api_key": {"type": "string", "description": "vLLM API key"},
                },
                "required": ["endpoint", "model"],
            },
        ),
        Tool(
            name="vllm_info",
            description="Get vLLM server info: loaded models, config, and health status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "api_key": {"type": "string", "description": "vLLM API key"},
                },
                "required": ["endpoint"],
            },
        ),
        Tool(
            name="vllm_sleep",
            description="Put a vLLM server to sleep. Level 1: offload to CPU (fast wake). Level 2: discard weights (minimal RAM).",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL"},
                    "level": {
                        "type": "integer",
                        "description": "Sleep level (1=CPU offload, 2=discard)",
                        "default": 1,
                        "enum": [1, 2],
                    },
                },
                "required": ["endpoint"],
            },
        ),
        Tool(
            name="vllm_wake",
            description="Wake a sleeping vLLM server. For Level 2 sleep, also reloads weights and resets prefix cache.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL"},
                    "sleep_level": {
                        "type": "integer",
                        "description": "The sleep level the server is at (affects wake procedure)",
                        "default": 1,
                    },
                },
                "required": ["endpoint"],
            },
        ),
        # ── v5.0.0: ML Services — SGLang Optimization Stack ──────────────────────────────────
        Tool(
            name="sglang",
            description="Complete SGLang optimization stack with workload-specific auto-tuning for 7 workload types: agentic chat, batch inference, low latency, MoE models, PD disaggregated, structured output, and RAG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "SGLang action to perform",
                        "enum": [
                            "optimize",
                            "detect",
                            "router",
                            "install",
                            "start",
                            "stop",
                            "inference",
                            "metrics",
                            "test",
                        ],
                    },
                    "model_path": {
                        "type": "string",
                        "description": "HuggingFace model ID or path",
                    },
                    "workload_type": {
                        "type": "string",
                        "description": "Workload type for optimization",
                        "enum": [
                            "agentic_chat",
                            "batch_inference",
                            "low_latency",
                            "moe_model",
                            "pd_disaggregated",
                            "structured_output",
                            "rag_workload",
                        ],
                    },
                    "user_description": {
                        "type": "string",
                        "description": "Natural language description of workload",
                    },
                    "host": {
                        "type": "string",
                        "description": "Server host",
                        "default": "0.0.0.0",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Server port",
                        "default": 8000,
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show optimization plan without launching",
                        "default": False,
                    },
                    "dp_size": {
                        "type": "integer",
                        "description": "Data parallel size for multi-replica",
                        "default": 8,
                    },
                    "instance_ip": {
                        "type": "string",
                        "description": "Remote instance IP for installation/deployment",
                    },
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user for remote operations",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                    "endpoint": {
                        "type": "string",
                        "description": "SGLang endpoint URL for inference/metrics",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text for inference testing",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Chat messages array",
                        "items": {"type": "object"},
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens to generate",
                        "default": 100,
                    },
                    "api_key": {"type": "string", "description": "API key for inference"},
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallel size",
                        "default": 1,
                    },
                    "enable_expert_parallel": {
                        "type": "boolean",
                        "description": "Enable MoE Expert Parallelism",
                        "default": False,
                    },
                },
                "required": ["action"],
            },
        ),
        # ── v4.0.0: ML Services — Legacy SGLang (deprecated) ──────────────────────────────────
        Tool(
            name="sglang_start",
            description="[DEPRECATED] Use sglang action='start' instead. Start an SGLang inference server on a remote instance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_ip": {"type": "string", "description": "Target instance IP"},
                    "model": {"type": "string", "description": "HuggingFace model ID"},
                    "port": {
                        "type": "integer",
                        "description": "Server port",
                        "default": 8000,
                    },
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallel size",
                        "default": 1,
                    },
                    "dp_size": {
                        "type": "integer",
                        "description": "Data parallel size",
                        "default": 8,
                    },
                    "enable_expert_parallel": {
                        "type": "boolean",
                        "description": "Enable MoE Expert Parallelism",
                        "default": False,
                    },
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                },
                "required": ["instance_ip", "model"],
            },
        ),
        Tool(
            name="sglang_stop",
            description="[DEPRECATED] Use sglang action='stop' instead. Stop an SGLang server on a remote instance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_ip": {"type": "string", "description": "Target instance IP"},
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                },
                "required": ["instance_ip"],
            },
        ),
        Tool(
            name="sglang_inference",
            description="[DEPRECATED] Use sglang action='inference' instead. Test inference against a running SGLang endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "SGLang endpoint URL (e.g. http://10.0.0.1:8000)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text (for completions mode)",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Chat messages array (for chat mode)",
                        "items": {"type": "object"},
                    },
                    "model": {"type": "string", "description": "Model name to use"},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens to generate",
                        "default": 100,
                    },
                    "api_key": {"type": "string", "description": "API key"},
                },
                "required": ["endpoint", "model"],
            },
        ),
        Tool(
            name="sglang_metrics",
            description="[DEPRECATED] Use sglang action='metrics' instead. Get SGLang server metrics from the Prometheus endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "SGLang endpoint URL (e.g. http://10.0.0.1:8000)",
                    }
                },
                "required": ["endpoint"],
            },
        ),
        # ── v4.0.0: ML Services — Ollama ──────────────────────────────────
        Tool(
            name="ollama_list",
            description="List models available on an Ollama server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Ollama endpoint URL",
                        "default": "http://localhost:11434",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="ollama_pull",
            description="Pull a model to an Ollama server on a remote instance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g. llama3.2, deepseek-r1, codellama)",
                    },
                    "instance_ip": {"type": "string", "description": "Target instance IP"},
                    "ssh_user": {
                        "type": "string",
                        "description": "SSH user",
                        "default": "root",
                    },
                    "ssh_key": {"type": "string", "description": "Path to SSH private key"},
                },
                "required": ["model", "instance_ip"],
            },
        ),
        Tool(
            name="ollama_generate",
            description="Generate text using an Ollama model (non-chat completions).",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Ollama endpoint URL",
                        "default": "http://localhost:11434",
                    },
                    "model": {"type": "string", "description": "Model name"},
                    "prompt": {"type": "string", "description": "Prompt text"},
                },
                "required": ["model", "prompt"],
            },
        ),
        Tool(
            name="ollama_chat",
            description="Chat with an Ollama model using the chat/completions API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Ollama endpoint URL",
                        "default": "http://localhost:11434",
                    },
                    "model": {"type": "string", "description": "Model name"},
                    "messages": {
                        "type": "array",
                        "description": "Chat messages [{role, content}]",
                        "items": {"type": "object"},
                    },
                },
                "required": ["model", "messages"],
            },
        ),
        Tool(
            name="ollama_model_info",
            description="Get detailed information about an Ollama model (parameters, template, license).",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Ollama endpoint URL",
                        "default": "http://localhost:11434",
                    },
                    "model": {"type": "string", "description": "Model name"},
                },
                "required": ["model"],
            },
        ),
        # ── v4.0.0: ML Services — Weights & Biases ───────────────────────
        Tool(
            name="wandb_list_projects",
            description="List all Weights & Biases projects for the configured entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {
                        "type": "string",
                        "description": "W&B entity (team/username)",
                    },
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="wandb_list_runs",
            description="List runs in a W&B project with status, metrics summary, and config.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                    "project": {"type": "string", "description": "W&B project name"},
                    "limit": {
                        "type": "integer",
                        "description": "Max runs to return",
                        "default": 50,
                    },
                },
                "required": ["api_key", "project"],
            },
        ),
        Tool(
            name="wandb_run_details",
            description="Get detailed info, metrics, and artifacts for a specific W&B run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "run_id": {"type": "string", "description": "W&B run ID"},
                },
                "required": ["api_key", "run_id"],
            },
        ),
        # ── v4.0.0: ML Services — LangSmith ──────────────────────────────
        Tool(
            name="langsmith_list_runs",
            description="List LangSmith runs with tracing data. Optionally correlate with Terradev GPU cost metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "LangSmith API key"},
                    "project": {"type": "string", "description": "LangSmith project name"},
                    "limit": {"type": "integer", "description": "Max runs", "default": 50},
                    "correlate_gpu": {
                        "type": "boolean",
                        "description": "Join with cost_tracking.db for cost-per-run",
                        "default": False,
                    },
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="langsmith_list_projects",
            description="List LangSmith projects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "LangSmith API key"}
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="langsmith_gpu_correlate",
            description="Correlate LangSmith runs with Terradev GPU provisioning data. Returns cost-per-run, GPU utilization, and provider breakdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "LangSmith API key"},
                    "project": {"type": "string", "description": "LangSmith project name"},
                    "days": {
                        "type": "integer",
                        "description": "Lookback period in days",
                        "default": 7,
                    },
                },
                "required": ["api_key"],
            },
        ),
        # ── v4.0.0: ML Services — MLflow ─────────────────────────────────
        Tool(
            name="mlflow_list_experiments",
            description="List MLflow experiments on the configured tracking server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tracking_uri": {
                        "type": "string",
                        "description": "MLflow tracking server URI",
                    },
                    "username": {"type": "string", "description": "Basic auth username"},
                    "password": {"type": "string", "description": "Basic auth password"},
                },
                "required": ["tracking_uri"],
            },
        ),
        Tool(
            name="mlflow_log_run",
            description="Log a Terradev training run to MLflow with auto-injected GPU type, provider, cost/hr, and duration as params.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tracking_uri": {
                        "type": "string",
                        "description": "MLflow tracking server URI",
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": "MLflow experiment name",
                    },
                    "run_name": {"type": "string", "description": "Run display name"},
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type used (e.g. H100)",
                    },
                    "provider": {"type": "string", "description": "Cloud provider"},
                    "cost_per_hour": {
                        "type": "number",
                        "description": "Cost per hour in USD",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Training duration in seconds",
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Additional metrics to log",
                    },
                    "username": {"type": "string", "description": "Basic auth username"},
                    "password": {"type": "string", "description": "Basic auth password"},
                },
                "required": ["tracking_uri", "experiment_name", "run_name"],
            },
        ),
        Tool(
            name="mlflow_register_model",
            description="Register a trained model in the MLflow model registry with Terradev provenance tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tracking_uri": {
                        "type": "string",
                        "description": "MLflow tracking server URI",
                    },
                    "model_name": {"type": "string", "description": "Model registry name"},
                    "run_id": {
                        "type": "string",
                        "description": "MLflow run ID that produced the model",
                    },
                    "model_uri": {
                        "type": "string",
                        "description": "Model artifact URI (e.g. runs:/<run_id>/model)",
                    },
                    "tags": {
                        "type": "object",
                        "description": "Additional tags to set on the model version",
                    },
                    "username": {"type": "string", "description": "Basic auth username"},
                    "password": {"type": "string", "description": "Basic auth password"},
                },
                "required": ["tracking_uri", "model_name", "run_id"],
            },
        ),
        # ── v4.0.0: ML Services — DVC ────────────────────────────────────
        Tool(
            name="dvc_status",
            description="Get DVC repository status: tracked files, remotes, and changes since last commit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the DVC repository",
                    }
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="dvc_diff",
            description="Show DVC diff between two revisions (e.g. training checkpoints). Shows added, modified, deleted files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the DVC repository",
                    },
                    "rev_a": {"type": "string", "description": "Base revision (git ref)"},
                    "rev_b": {
                        "type": "string",
                        "description": "Target revision (git ref, default: HEAD)",
                    },
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="dvc_stage_checkpoint",
            description="Atomic checkpoint staging: DVC add + push + git commit in one operation. Promotes a training checkpoint to versioned storage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the DVC repository",
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to the checkpoint file/directory to stage",
                    },
                    "message": {
                        "type": "string",
                        "description": "Git commit message",
                        "default": "Stage checkpoint via Terradev",
                    },
                    "remote": {
                        "type": "string",
                        "description": "DVC remote name to push to",
                    },
                },
                "required": ["repo_path", "checkpoint_path"],
            },
        ),
        Tool(
            name="dvc_push",
            description="Push DVC-tracked data to the configured remote storage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the DVC repository",
                    },
                    "remote": {
                        "type": "string",
                        "description": "DVC remote name (optional, uses default)",
                    },
                },
                "required": ["repo_path"],
            },
        ),
        # ── v4.0.0: ML Services — KServe ─────────────────────────────────
        Tool(
            name="kserve_generate_yaml",
            description="Generate a GPU-aware KServe InferenceService YAML manifest with NUMA pinning, resource limits derived from model size and VRAM, and topology hints.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name for the InferenceService",
                    },
                    "model_uri": {
                        "type": "string",
                        "description": "Model storage URI (e.g. s3://bucket/model or gs://bucket/model)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type (e.g. A100, H100)",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs",
                        "default": 1,
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace",
                        "default": "default",
                    },
                    "runtime": {
                        "type": "string",
                        "description": "Serving runtime",
                        "enum": ["vllm", "triton", "huggingface"],
                        "default": "vllm",
                    },
                    "min_replicas": {
                        "type": "integer",
                        "description": "Min replicas for autoscaling",
                        "default": 1,
                    },
                    "max_replicas": {
                        "type": "integer",
                        "description": "Max replicas for autoscaling",
                        "default": 3,
                    },
                },
                "required": ["model_name", "model_uri", "gpu_type"],
            },
        ),
        Tool(
            name="kserve_list",
            description="List KServe InferenceServices in a Kubernetes namespace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace",
                        "default": "default",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="kserve_status",
            description="Get detailed status of a KServe InferenceService including readiness, traffic split, and URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "InferenceService name"},
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace",
                        "default": "default",
                    },
                },
                "required": ["name"],
            },
        ),
        # ── v4.0.0: Egress Optimizer ─────────────────────────────────────
        Tool(
            name="egress_cheapest_route",
            description="Find the cheapest egress route between cloud providers/regions for model weights or dataset transfer. Supports multi-hop routing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_provider": {
                        "type": "string",
                        "description": "Source cloud provider (e.g. aws, gcp, azure)",
                    },
                    "source_region": {
                        "type": "string",
                        "description": "Source region (e.g. us-east-1)",
                    },
                    "dest_provider": {
                        "type": "string",
                        "description": "Destination cloud provider",
                    },
                    "dest_region": {"type": "string", "description": "Destination region"},
                    "size_gb": {"type": "number", "description": "Transfer size in GB"},
                },
                "required": [
                    "source_provider",
                    "source_region",
                    "dest_provider",
                    "dest_region",
                    "size_gb",
                ],
            },
        ),
        Tool(
            name="egress_optimize_staging",
            description="Optimize dataset or model staging across regions by finding the cheapest transfer plan. Integrates with the dataset stager for parallel uploads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_uri": {
                        "type": "string",
                        "description": "Source data URI (s3://, gs://, local path, or HF dataset ID)",
                    },
                    "target_regions": {
                        "type": "array",
                        "description": "Target regions as provider:region strings",
                        "items": {"type": "string"},
                    },
                    "size_gb": {
                        "type": "number",
                        "description": "Approximate data size in GB",
                    },
                },
                "required": ["source_uri", "target_regions", "size_gb"],
            },
        ),
        # ── v5.0.0: HuggingFace Hub Full Service ─────────────────────────
        Tool(
            name="hf_list_models",
            description="Search and browse HuggingFace Hub models. Filter by author, task, library. Returns model ID, downloads, likes, and tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author": {
                        "type": "string",
                        "description": "Filter by author/org (e.g. meta-llama, mistralai)",
                    },
                    "search": {
                        "type": "string",
                        "description": "Search query (e.g. 'code generation', 'llama')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 20,
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="hf_list_datasets",
            description="Search and browse HuggingFace Hub datasets. Filter by author and search query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Filter by author/org"},
                    "search": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 20,
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="hf_model_info",
            description="Get detailed model info: architecture, size, downloads, license, tags, pipeline_tag, and model card.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)",
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["model_id", "api_key"],
            },
        ),
        Tool(
            name="hf_create_endpoint",
            description="Create a HuggingFace Inference Endpoint (paid GPU endpoint). Supports custom GPU types, regions, and scaling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "HuggingFace model ID"},
                    "endpoint_name": {"type": "string", "description": "Endpoint name"},
                    "instance_type": {
                        "type": "string",
                        "description": "Instance type (e.g. nvidia-a100, nvidia-l4)",
                    },
                    "instance_size": {
                        "type": "string",
                        "description": "Instance size (e.g. x1, x2, x4)",
                    },
                    "region": {
                        "type": "string",
                        "description": "Region (e.g. us-east-1, eu-west-1)",
                        "default": "us-east-1",
                    },
                    "min_replicas": {
                        "type": "integer",
                        "description": "Min replicas (0 for scale-to-zero)",
                        "default": 0,
                    },
                    "max_replicas": {
                        "type": "integer",
                        "description": "Max replicas",
                        "default": 1,
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["model_id", "endpoint_name", "instance_type", "api_key"],
            },
        ),
        Tool(
            name="hf_list_endpoints",
            description="List all active HuggingFace Inference Endpoints with status, URL, and cost.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "HuggingFace API token"}
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="hf_endpoint_info",
            description="Get detailed info about a specific HuggingFace Inference Endpoint: status, URL, scaling config, cost.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_name": {"type": "string", "description": "Endpoint name"},
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["endpoint_name", "api_key"],
            },
        ),
        Tool(
            name="hf_delete_endpoint",
            description="Delete a HuggingFace Inference Endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_name": {
                        "type": "string",
                        "description": "Endpoint name to delete",
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["endpoint_name", "api_key"],
            },
        ),
        Tool(
            name="hf_endpoint_infer",
            description="Run inference on a HuggingFace Inference Endpoint. Supports text generation, embeddings, and custom inputs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_name": {"type": "string", "description": "Endpoint name"},
                    "inputs": {"type": "string", "description": "Input text or prompt"},
                    "parameters": {
                        "type": "object",
                        "description": "Generation parameters (max_new_tokens, temperature, etc.)",
                    },
                    "api_key": {"type": "string", "description": "HuggingFace API token"},
                },
                "required": ["endpoint_name", "inputs", "api_key"],
            },
        ),
        # ── v5.0.0: HF Smart Templates ───────────────────────────────────
        Tool(
            name="hf_smart_template",
            description="Auto-generate an optimized deployment template for any HuggingFace model. Analyzes model size, architecture, and quantization to select optimal hardware and generate ready-to-deploy configs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)",
                    },
                    "template_type": {
                        "type": "string",
                        "description": "Template type",
                        "enum": ["auto", "chat", "embedding", "vision", "audio"],
                        "default": "auto",
                    },
                    "space_name": {
                        "type": "string",
                        "description": "Optional HF Space name for deployment",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="hf_hardware_recommend",
            description="Get hardware recommendation with cost breakdown for any HuggingFace model. Returns optimal GPU type, estimated cost, and performance score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "HuggingFace model ID"},
                    "budget_constraint": {
                        "type": "number",
                        "description": "Max $/hr budget (optional)",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="hf_hardware_compare",
            description="Compare all hardware options for a HuggingFace model. Returns side-by-side cost, performance, and compatibility analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "HuggingFace model ID"}
                },
                "required": ["model_id"],
            },
        ),
        # ── v5.0.0: LangChain / LangGraph / LangSmith ────────────────────
        Tool(
            name="langchain_create_workflow",
            description="Create a LangChain workflow with LangSmith monitoring integration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_config": {
                        "type": "object",
                        "description": "Workflow configuration (name, steps, model, tools)",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "LangChain/LangSmith API key",
                    },
                    "langsmith_api_key": {
                        "type": "string",
                        "description": "LangSmith API key for tracing",
                    },
                },
                "required": ["workflow_config", "api_key"],
            },
        ),
        Tool(
            name="langchain_create_sglang_pipeline",
            description="Create an SGLang model-serving pipeline via LangChain. Connects LangChain agents to SGLang inference endpoints.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_config": {
                        "type": "object",
                        "description": "Pipeline config (model, endpoint, temperature, max_tokens)",
                    },
                    "api_key": {"type": "string", "description": "LangChain API key"},
                },
                "required": ["pipeline_config", "api_key"],
            },
        ),
        Tool(
            name="langsmith_create_project",
            description="Create a new LangSmith project for tracing and evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"},
                    "api_key": {"type": "string", "description": "LangSmith API key"},
                },
                "required": ["name", "api_key"],
            },
        ),
        Tool(
            name="langsmith_get_workspaces",
            description="List all LangSmith workspaces.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "LangSmith API key"}
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="langsmith_create_trace",
            description="Create a trace in LangSmith for observability. Tracks input/output, latency, token usage, and cost.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to attach trace to",
                    },
                    "trace_data": {
                        "type": "object",
                        "description": "Trace data (name, inputs, outputs, metadata)",
                    },
                    "api_key": {"type": "string", "description": "LangSmith API key"},
                },
                "required": ["run_id", "trace_data", "api_key"],
            },
        ),
        Tool(
            name="langgraph_create_workflow",
            description="Create a LangGraph stateful workflow with monitoring. Supports agent graphs, tool calling, and state persistence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_config": {
                        "type": "object",
                        "description": "Graph configuration (nodes, edges, state_schema)",
                    },
                    "api_key": {"type": "string", "description": "LangChain API key"},
                    "langsmith_api_key": {
                        "type": "string",
                        "description": "LangSmith API key for tracing",
                    },
                },
                "required": ["graph_config", "api_key"],
            },
        ),
        Tool(
            name="langgraph_orchestrator_worker",
            description="Create an orchestrator-worker pattern workflow in LangGraph. The orchestrator delegates tasks to specialized worker agents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_config": {
                        "type": "object",
                        "description": "Orchestrator-worker config (orchestrator_prompt, workers, routing_strategy)",
                    },
                    "api_key": {"type": "string", "description": "LangChain API key"},
                },
                "required": ["workflow_config", "api_key"],
            },
        ),
        Tool(
            name="langgraph_evaluation_workflow",
            description="Create an evaluator-optimizer workflow in LangGraph. Generates outputs, evaluates quality, and iteratively improves.",
            inputSchema={
                "type": "object",
                "properties": {
                    "evaluation_config": {
                        "type": "object",
                        "description": "Evaluation config (generator_prompt, evaluator_criteria, max_iterations)",
                    },
                    "api_key": {"type": "string", "description": "LangChain API key"},
                },
                "required": ["evaluation_config", "api_key"],
            },
        ),
        Tool(
            name="langgraph_workflow_status",
            description="Get the status and metrics of a LangGraph workflow execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to check",
                    },
                    "api_key": {"type": "string", "description": "LangChain API key"},
                },
                "required": ["workflow_id", "api_key"],
            },
        ),
        # ── v5.0.0: W&B Enhanced ─────────────────────────────────────────
        Tool(
            name="wandb_create_dashboard",
            description="Create a custom W&B dashboard with GPU metrics, training loss, and cost panels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dashboard_config": {
                        "type": "object",
                        "description": "Dashboard config (name, panels, metrics)",
                    },
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["dashboard_config", "api_key"],
            },
        ),
        Tool(
            name="wandb_create_terradev_dashboard",
            description="Auto-create a Terradev-specific W&B dashboard with GPU utilization, cost tracking, training metrics, and infrastructure panels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                    "project": {"type": "string", "description": "W&B project name"},
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="wandb_create_report",
            description="Create a W&B report with custom sections, charts, and narrative text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_config": {
                        "type": "object",
                        "description": "Report config (title, sections, metrics, description)",
                    },
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["report_config", "api_key"],
            },
        ),
        Tool(
            name="wandb_create_terradev_report",
            description="Auto-generate a Terradev infrastructure report: GPU costs, provider comparison, training efficiency, and recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metrics_data": {
                        "type": "object",
                        "description": "Metrics data to include (gpu_costs, training_runs, provider_stats)",
                    },
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="wandb_setup_alerts",
            description="Set up custom W&B alerts for GPU metrics: cost thresholds, utilization drops, training anomalies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_config": {
                        "type": "object",
                        "description": "Alert config (metric, threshold, condition, notification_channel)",
                    },
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["alert_config", "api_key"],
            },
        ),
        Tool(
            name="wandb_create_terradev_alerts",
            description="Auto-create standard Terradev alerts: GPU cost > budget, utilization < 50%, training loss spike, straggler detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["api_key"],
            },
        ),
        Tool(
            name="wandb_dashboard_status",
            description="Get comprehensive W&B monitoring overview: dashboards, reports, alerts, active runs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "W&B API key"},
                    "entity": {"type": "string", "description": "W&B entity"},
                },
                "required": ["api_key"],
            },
        ),
        # ── v5.0.0: Data Governance ───────────────────────────────────────
        Tool(
            name="governance_request_consent",
            description="Request user consent for data movement across cloud regions. GDPR/SOC2 compliant consent tracking with audit trail.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID requesting consent",
                    },
                    "consent_type": {
                        "type": "string",
                        "description": "Consent type",
                        "enum": [
                            "data_staging",
                            "cross_region",
                            "third_party",
                            "model_training",
                        ],
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Dataset being moved",
                    },
                    "source_location": {
                        "type": "string",
                        "description": "Source region/provider",
                    },
                    "target_location": {
                        "type": "string",
                        "description": "Target region/provider",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Purpose of data movement",
                    },
                },
                "required": ["user_id", "consent_type", "dataset_name", "purpose"],
            },
        ),
        Tool(
            name="governance_record_consent",
            description="Record a consent response (granted or denied) for a pending consent request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string", "description": "Consent request ID"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "granted": {
                        "type": "boolean",
                        "description": "Whether consent was granted",
                    },
                    "conditions": {
                        "type": "array",
                        "description": "Conditions attached to consent",
                        "items": {"type": "string"},
                    },
                },
                "required": ["request_id", "user_id", "granted"],
            },
        ),
        Tool(
            name="governance_evaluate_opa",
            description="Evaluate OPA (Open Policy Agent) policies for data access. Checks region restrictions, classification rules, and compliance requirements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID to evaluate"},
                    "dataset_name": {"type": "string", "description": "Dataset name"},
                    "action": {
                        "type": "string",
                        "description": "Action to evaluate",
                        "enum": ["read", "write", "move", "delete", "train"],
                    },
                    "target_location": {
                        "type": "string",
                        "description": "Target location for the action",
                    },
                },
                "required": ["user_id", "dataset_name", "action"],
            },
        ),
        Tool(
            name="governance_move_data",
            description="Move data with full governance audit trail. Requires prior consent and OPA policy approval. Tracks integrity, encryption, and compliance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "consent_request_id": {
                        "type": "string",
                        "description": "Approved consent request ID",
                    },
                    "dataset_name": {"type": "string", "description": "Dataset to move"},
                    "source_location": {"type": "string", "description": "Source location"},
                    "target_location": {"type": "string", "description": "Target location"},
                },
                "required": [
                    "user_id",
                    "consent_request_id",
                    "dataset_name",
                    "source_location",
                    "target_location",
                ],
            },
        ),
        Tool(
            name="governance_movement_history",
            description="Get data movement audit log. Filter by user, dataset, or time range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Filter by user ID"},
                    "dataset_name": {"type": "string", "description": "Filter by dataset"},
                    "limit": {
                        "type": "integer",
                        "description": "Max records",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="governance_compliance_report",
            description="Generate comprehensive compliance report: consent stats, policy evaluations, data movements, violations. For GDPR/SOC2/HIPAA audits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date (ISO format, e.g. 2025-01-01)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (ISO format, e.g. 2025-12-31)",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        ),
        # ── v5.0.0: Cost Optimizer Deep ───────────────────────────────────
        Tool(
            name="cost_analyze",
            description="Deep cost analysis of current GPU infrastructure: per-provider breakdown, utilization efficiency, waste identification, and optimization potential.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Lookback period in days",
                        "default": 30,
                    }
                },
            },
        ),
        Tool(
            name="cost_optimize_recommend",
            description="Generate actionable cost optimization recommendations: spot migration, GPU right-sizing, provider arbitrage, idle shutdown, and density packing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_savings": {
                        "type": "number",
                        "description": "Target savings percentage (e.g. 0.3 for 30%)",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Constraints (min_gpus, max_latency_ms, required_providers)",
                    },
                },
            },
        ),
        Tool(
            name="cost_simulate",
            description="Simulate cost optimization scenarios with ROI projections. Compare current vs optimized infrastructure costs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "object",
                        "description": "Scenario config (gpu_type, provider, count, spot, hours)",
                    },
                    "compare_with": {
                        "type": "object",
                        "description": "Current config to compare against",
                    },
                },
                "required": ["scenario"],
            },
        ),
        Tool(
            name="cost_budget_optimize",
            description="Find optimal GPU deployment under a strict budget constraint. Uses ML-based cost prediction and spot risk assessment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Total budget in USD"},
                    "gpu_type": {"type": "string", "description": "Required GPU type"},
                    "gpu_count": {
                        "type": "integer",
                        "description": "Required GPU count",
                        "default": 1,
                    },
                    "hours": {
                        "type": "number",
                        "description": "Required runtime in hours",
                        "default": 1.0,
                    },
                    "allow_spot": {
                        "type": "boolean",
                        "description": "Allow spot instances",
                        "default": True,
                    },
                },
                "required": ["budget"],
            },
        ),
        # ── v5.0.0: Price Intelligence Extended ──────────────────────────
        Tool(
            name="price_trends",
            description="Get GPU price trend analysis with delta (rate of change), gamma (acceleration), and annualized volatility. Identifies cheapest time windows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type",
                        "enum": [
                            "H100",
                            "H200",
                            "H800",
                            "A100",
                            "A10G",
                            "L40S",
                            "L4",
                            "T4",
                            "RTX4090",
                            "V100S",
                            "A6000",
                            "MI300X",
                        ],
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Hours of history",
                        "default": 24,
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        Tool(
            name="price_budget_optimize",
            description="Budget-first price optimization with ML-based cost prediction. Finds cheapest deployment plan under budget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Budget in USD"},
                    "gpu_type": {"type": "string", "description": "Required GPU type"},
                    "gpu_count": {
                        "type": "integer",
                        "description": "GPU count",
                        "default": 1,
                    },
                    "hours": {
                        "type": "number",
                        "description": "Runtime hours",
                        "default": 1.0,
                    },
                },
                "required": ["budget", "gpu_type"],
            },
        ),
        Tool(
            name="price_spot_risk",
            description="Spot instance risk assessment per provider. Returns interruption probability, mean time to interruption, and recommended mitigation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {"type": "string", "description": "GPU type"},
                    "provider": {
                        "type": "string",
                        "description": "Provider to assess (or 'all')",
                    },
                },
                "required": ["gpu_type"],
            },
        ),
        # ── v5.0.0: Training Orchestrator Extended ────────────────────────
        Tool(
            name="training_config_generate",
            description="Generate a complete training configuration from a declarative spec. Auto-detects framework, sets optimal parallelism, and configures distributed training.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Training job name"},
                    "framework": {
                        "type": "string",
                        "description": "Training framework",
                        "enum": ["torchrun", "deepspeed", "accelerate", "megatron"],
                        "default": "torchrun",
                    },
                    "script": {"type": "string", "description": "Training script path"},
                    "nodes": {
                        "type": "array",
                        "description": "Node IPs",
                        "items": {"type": "string"},
                    },
                    "gpus_per_node": {
                        "type": "integer",
                        "description": "GPUs per node",
                        "default": 8,
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve from provision ('latest' or group ID)",
                    },
                    "deepspeed_config": {
                        "type": "object",
                        "description": "DeepSpeed config overrides",
                    },
                    "script_args": {
                        "type": "string",
                        "description": "Extra script arguments",
                    },
                },
                "required": ["name", "script"],
            },
        ),
        Tool(
            name="training_launch_distributed",
            description="Full distributed training launch with framework auto-detection, topology validation, and monitoring. Combines preflight + train + monitor in one operation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Job name"},
                    "script": {"type": "string", "description": "Training script"},
                    "framework": {
                        "type": "string",
                        "description": "Framework",
                        "enum": ["torchrun", "deepspeed", "accelerate", "megatron"],
                        "default": "torchrun",
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision ('latest' or group ID)",
                    },
                    "nodes": {
                        "type": "array",
                        "description": "Manual node IPs",
                        "items": {"type": "string"},
                    },
                    "gpus_per_node": {
                        "type": "integer",
                        "description": "GPUs per node",
                        "default": 8,
                    },
                    "skip_preflight": {
                        "type": "boolean",
                        "description": "Skip preflight validation",
                        "default": False,
                    },
                },
                "required": ["name", "script"],
            },
        ),
        # ── v5.0.0: Training Monitor Extended ─────────────────────────────
        Tool(
            name="train_snapshot",
            description="Get complete training monitoring snapshot: GPU metrics (utilization, memory, temp, power), training metrics (loss, grad_norm, lr, throughput), straggler detection, and cost estimate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "cost_rate": {
                        "type": "number",
                        "description": "$/GPU-hr for cost estimation",
                        "default": 2.0,
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="train_detect_stragglers",
            description="Detect straggler nodes in distributed training. Identifies GPUs with significantly lower utilization that slow the whole job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "threshold": {
                        "type": "number",
                        "description": "Straggler threshold (0-1, default 0.7 = 70% of mean)",
                        "default": 0.7,
                    },
                },
                "required": ["job_id"],
            },
        ),
        # ── v5.0.0: Preflight Validator Extended ──────────────────────────
        Tool(
            name="preflight_report",
            description="Generate full preflight validation report with pass/warn/fail per check. Covers GPU drivers, CUDA, NCCL, RDMA, network, disk, and Docker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "Node IPs",
                        "items": {"type": "string"},
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision ('latest' or group ID)",
                    },
                    "checks": {
                        "type": "array",
                        "description": "Specific checks to run",
                        "items": {"type": "string"},
                    },
                },
            },
        ),
        Tool(
            name="preflight_gpu_check",
            description="GPU-specific preflight validation: NVIDIA drivers, CUDA version, GPU count, NCCL, NVLink topology, NCU stall-signature profiling, and adversarial config verification (V1-V3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "Node IPs",
                        "items": {"type": "string"},
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision",
                    },
                    "tensor_parallel_size": {
                        "type": "integer",
                        "description": "TP size for V1 adversarial check (vs GPU count)",
                    },
                    "model_precision": {
                        "type": "string",
                        "description": "Model precision (fp8, bf16, fp16) for V2 FP8 wall check",
                    },
                    "fp8_quant_scheme": {
                        "type": "string",
                        "description": "FP8 quant scheme (per_tensor, per_token) for Blackwell K-slab check",
                    },
                    "gpu_arch": {
                        "type": "string",
                        "description": "GPU architecture string (e.g. blackwell, hopper) for precision wall detection",
                    },
                    "max_batch_size": {
                        "type": "integer",
                        "description": "Max batch size for V3 launch-overhead dominance check",
                    },
                },
            },
        ),
        Tool(
            name="preflight_network_check",
            description="Network-specific preflight validation: RDMA availability, InfiniBand status, inter-node bandwidth, latency matrix, firewall rules.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "Node IPs",
                        "items": {"type": "string"},
                    },
                    "from_provision": {
                        "type": "string",
                        "description": "Resolve nodes from provision",
                    },
                },
            },
        ),
        # ── v5.0.0: Kubernetes Enhanced ───────────────────────────────────
        Tool(
            name="k8s_gpu_operator_install",
            description="Install NVIDIA GPU Operator on a Kubernetes cluster. Configures driver containers, device plugin, DCGM exporter, and GPU Feature Discovery.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Target cluster name",
                    },
                    "driver_version": {
                        "type": "string",
                        "description": "NVIDIA driver version (default: auto-detect)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Install namespace",
                        "default": "gpu-operator",
                    },
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="k8s_device_plugin",
            description="Configure Kubernetes GPU device plugin settings: time-slicing, MIG strategy, and resource naming.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Target cluster name",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Device plugin strategy",
                        "enum": ["none", "time-slicing", "mig-single", "mig-mixed"],
                        "default": "none",
                    },
                    "replicas": {
                        "type": "integer",
                        "description": "Time-slicing replicas per GPU",
                        "default": 2,
                    },
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="k8s_mig_configure",
            description="Configure Multi-Instance GPU (MIG) partitioning on A100/H100 GPUs. Splits a single GPU into isolated instances for multi-tenant workloads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Target cluster name",
                    },
                    "mig_profile": {
                        "type": "string",
                        "description": "MIG profile",
                        "enum": [
                            "1g.5gb",
                            "1g.10gb",
                            "2g.10gb",
                            "3g.20gb",
                            "4g.20gb",
                            "7g.40gb",
                            "7g.80gb",
                        ],
                    },
                    "gpu_indices": {
                        "type": "array",
                        "description": "GPU indices to configure",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["cluster_name", "mig_profile"],
            },
        ),
        Tool(
            name="k8s_time_slicing",
            description="Configure GPU time-slicing for Kubernetes. Allows multiple pods to share a single GPU with configurable oversubscription.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Target cluster name",
                    },
                    "replicas": {
                        "type": "integer",
                        "description": "Virtual GPUs per physical GPU",
                        "default": 4,
                    },
                    "oversubscribe": {
                        "type": "boolean",
                        "description": "Allow oversubscription",
                        "default": True,
                    },
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="k8s_monitoring_stack",
            description="Deploy Prometheus + Grafana GPU monitoring stack with DCGM dashboards, Karpenter metrics, and GPU utilization alerts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Target cluster name",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Monitoring namespace",
                        "default": "monitoring",
                    },
                    "grafana_password": {
                        "type": "string",
                        "description": "Grafana admin password",
                    },
                    "enable_alerting": {
                        "type": "boolean",
                        "description": "Enable GPU utilization alerts",
                        "default": True,
                    },
                },
                "required": ["cluster_name"],
            },
        ),
        # ── v5.1.0: Datadog Integration ──────────────────────────────────
        Tool(
            name="datadog_status",
            description="Check Datadog integration status: configured, site, API key presence.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="datadog_push_metrics",
            description="Push current GPU cost snapshot to Datadog: active instances, cost/hr, projected monthly, provider reliability, quote latency.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="datadog_send_event",
            description="Send a custom event to Datadog with title, text, and alert type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "text": {
                        "type": "string",
                        "description": "Event body (markdown supported)",
                    },
                    "alert_type": {
                        "type": "string",
                        "description": "Alert type",
                        "enum": ["info", "warning", "error", "success"],
                        "default": "info",
                    },
                },
                "required": ["title", "text"],
            },
        ),
        Tool(
            name="datadog_create_monitors",
            description="Create all Terradev GPU FinOps monitors in Datadog: budget alert, cost spike, idle GPU, spot volatility, provider degraded, egress anomaly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "description": "Single template name (or omit for all)",
                        "enum": [
                            "budget_alert",
                            "cost_spike",
                            "idle_gpu",
                            "spot_risk",
                            "provider_degraded",
                            "egress_anomaly",
                        ],
                    },
                },
            },
        ),
        Tool(
            name="datadog_list_monitors",
            description="List all Terradev-tagged monitors in Datadog with their current status.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="datadog_create_dashboard",
            description="Create the Terradev GPU FinOps dashboard in Datadog with 12 widgets: cost/hr, projected monthly, active GPUs, budget, provider reliability, volatility, latency, training util, egress, events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Custom dashboard title (default: Terradev GPU FinOps)",
                    },
                },
            },
        ),
        Tool(
            name="datadog_list_dashboards",
            description="List Terradev-related dashboards in Datadog.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="datadog_query",
            description="Query Datadog metrics using the metrics query language. Returns time series data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Datadog metric query (e.g. avg:terradev.gpu.cost_per_hour{*} by {provider})",
                    },
                    "from_seconds": {
                        "type": "integer",
                        "description": "Lookback window in seconds",
                        "default": 3600,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="datadog_terraform_export",
            description="Generate a complete Terraform module for the Datadog integration: provider.tf, monitors.tf, dashboard.tf, versions.tf. Run `terraform apply` to deploy all monitors and dashboards as IaC.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to write .tf files (default: ./datadog-terraform)",
                    },
                },
            },
        ),
        Tool(
            name="datadog_metric_catalog",
            description="List all Terradev metrics that can be pushed to Datadog with their types, units, and tags.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # ── v5.2.0: Arize Phoenix — LLM Trace Observability (ELv2) ──
        Tool(
            name="phoenix_test",
            description="Test connection to Arize Phoenix server. Returns collector endpoint and project count.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="phoenix_projects",
            description="List Phoenix projects (trace namespaces).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max projects to return",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="phoenix_spans",
            description="List recent spans for a Phoenix project. Supports SpanQuery DSL filters like \"span_kind == 'RETRIEVER'\" or \"status_code == 'ERROR'\".",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project ID or name"},
                    "filter": {
                        "type": "string",
                        "description": "SpanQuery DSL filter expression",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max spans to return",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="phoenix_trace",
            description="View full execution tree for a specific trace ID. Shows span hierarchy, latencies, and token counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "Trace ID to inspect"},
                    "project": {"type": "string", "description": "Project ID or name"},
                },
                "required": ["trace_id"],
            },
        ),
        Tool(
            name="phoenix_otel_env",
            description="Generate OpenTelemetry environment variables for instrumenting serving pods with Phoenix tracing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                },
            },
        ),
        Tool(
            name="phoenix_snippet",
            description="Generate Python instrumentation snippet for adding Phoenix tracing to LLM applications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                },
            },
        ),
        Tool(
            name="phoenix_k8s",
            description="Generate Kubernetes deployment manifest for self-hosted Arize Phoenix server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "K8s namespace",
                        "default": "observability",
                    },
                },
            },
        ),
        # ── v5.2.0: NeMo Guardrails — LLM Output Safety (Apache 2.0) ──
        Tool(
            name="guardrails_test",
            description="Test connection to NeMo Guardrails server.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="guardrails_chat",
            description="Send a message through NeMo Guardrails and return the safety-filtered response. Applies topical, jailbreak, PII, and factcheck rails.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send through guardrails",
                    },
                    "config_id": {
                        "type": "string",
                        "description": "Guardrails config_id to use",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="guardrails_generate_config",
            description="Generate default Colang 2.x guardrails configuration files (topical, jailbreak, PII, factcheck rails).",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_id": {"type": "string", "description": "Config ID name"},
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory",
                        "default": "./guardrails",
                    },
                },
            },
        ),
        Tool(
            name="guardrails_k8s",
            description="Generate Kubernetes deployment manifest for NeMo Guardrails server (standalone or sidecar mode).",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "K8s namespace",
                        "default": "guardrails",
                    },
                },
            },
        ),
        # ── v5.2.0: Qdrant — Vector Database for RAG (Apache 2.0) ──
        Tool(
            name="qdrant_test",
            description="Test connection to Qdrant vector database. Returns cluster info and collection count.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="qdrant_collections",
            description="List all Qdrant vector collections with their point counts and configurations.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="qdrant_create_collection",
            description="Create a Qdrant vector collection. Auto-configures vector dimensions from embedding model name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Collection name"},
                    "embedding_model": {
                        "type": "string",
                        "description": "Embedding model (auto-sets vector size, e.g. BAAI/bge-large-en-v1.5)",
                    },
                },
            },
        ),
        Tool(
            name="qdrant_info",
            description="Get detailed info and stats for a Qdrant collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Collection name"},
                },
            },
        ),
        Tool(
            name="qdrant_count",
            description="Count points (vectors) in a Qdrant collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Collection name"},
                },
            },
        ),
        Tool(
            name="qdrant_k8s",
            description="Generate Kubernetes StatefulSet manifest for self-hosted Qdrant vector database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "K8s namespace",
                        "default": "vector-db",
                    },
                },
            },
        ),
        # ── v5.3.0: New v4.0.11 Features - Karpenter, Triggers, Environments, Lineage, Migration ──
        # Note: ALL_NEW_TOOLS removed - mcp_new_features_tools module not available
        # ── v5.4.0: Langfuse LLM Observability ────────────────────────────────────
        Tool(
            name="langfuse_configure",
            description="Configure Langfuse credentials (public key, secret key, host URL).",
            inputSchema={
                "type": "object",
                "properties": {
                    "public_key": {
                        "type": "string",
                        "description": "Langfuse public key (pk-lf-...)",
                    },
                    "secret_key": {
                        "type": "string",
                        "description": "Langfuse secret key (sk-lf-...)",
                    },
                    "host": {
                        "type": "string",
                        "description": "Langfuse host URL",
                        "default": "https://cloud.langfuse.com",
                    },
                },
                "required": ["public_key", "secret_key"],
            },
        ),
        Tool(
            name="langfuse_test",
            description="Test Langfuse connectivity and list accessible projects.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="langfuse_traces",
            description="List recent LLM traces from Langfuse.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max traces to return",
                        "default": 20,
                    },
                    "name": {"type": "string", "description": "Filter by trace name"},
                },
            },
        ),
        Tool(
            name="langfuse_trace",
            description="Get a single Langfuse trace with all observations/spans.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "Trace ID"},
                },
                "required": ["trace_id"],
            },
        ),
        Tool(
            name="langfuse_scores",
            description="List evaluation scores from Langfuse, optionally filtered by trace or score name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "Filter by trace ID"},
                    "name": {"type": "string", "description": "Filter by score name"},
                    "limit": {
                        "type": "integer",
                        "description": "Max scores to return",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="langfuse_score",
            description="Create an evaluation score for a Langfuse trace (e.g. quality, accuracy, relevance).",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "Trace ID to score"},
                    "name": {
                        "type": "string",
                        "description": "Score name (e.g. quality, accuracy)",
                    },
                    "value": {"type": "number", "description": "Numeric score value"},
                    "observation_id": {
                        "type": "string",
                        "description": "Specific observation to score",
                    },
                    "comment": {"type": "string", "description": "Optional comment"},
                },
                "required": ["trace_id", "name", "value"],
            },
        ),
        Tool(
            name="langfuse_datasets",
            description="List Langfuse datasets for evaluation and fine-tuning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max datasets to return",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="langfuse_export_training_data",
            description="Export Langfuse traces as instruction/response pairs for LoRA fine-tuning. Filters by quality score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max pairs to export",
                        "default": 500,
                    },
                    "name": {"type": "string", "description": "Filter traces by name"},
                    "min_score": {
                        "type": "number",
                        "description": "Minimum quality score (0.0-1.0)",
                    },
                    "score_name": {
                        "type": "string",
                        "description": "Score name to filter on",
                        "default": "quality",
                    },
                },
            },
        ),
        Tool(
            name="langfuse_quality",
            description="Get aggregated quality metrics from Langfuse scores for drift detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "score_name": {
                        "type": "string",
                        "description": "Score name to aggregate",
                        "default": "quality",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Sample size",
                        "default": 200,
                    },
                },
            },
        ),
        Tool(
            name="langfuse_otel_env",
            description="Print OTEL environment variables for instrumenting LLM apps to send traces to Langfuse.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Langfuse project name",
                        "default": "default",
                    },
                },
            },
        ),
        Tool(
            name="langfuse_k8s",
            description="Generate Kubernetes deployment manifest for self-hosted Langfuse.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "K8s namespace",
                        "default": "observability",
                    },
                },
            },
        ),
        # ── v5.4.0: Databricks MLOps ──────────────────────────────────────────────
        Tool(
            name="databricks_configure",
            description="Configure Databricks credentials (workspace URL and personal access token).",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Databricks workspace URL (e.g. https://adb-xxx.azuredatabricks.net)",
                    },
                    "token": {
                        "type": "string",
                        "description": "Databricks personal access token (dapi...)",
                    },
                },
                "required": ["host", "token"],
            },
        ),
        Tool(
            name="databricks_test",
            description="Test Databricks connectivity and show cluster count.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="databricks_jobs",
            description="List Databricks jobs in the workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max jobs to return",
                        "default": 25,
                    },
                },
            },
        ),
        Tool(
            name="databricks_run",
            description="Trigger a Databricks job run by job ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Databricks job ID"},
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="databricks_run_status",
            description="Get status of a Databricks job run by run ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "Databricks run ID"},
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="databricks_clusters",
            description="List all Databricks clusters with state and node type.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="databricks_serving_endpoints",
            description="List Databricks model serving endpoints.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="databricks_deploy_model",
            description="Deploy a registered MLflow model to a Databricks serving endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_name": {
                        "type": "string",
                        "description": "Serving endpoint name",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Registered model name",
                    },
                    "model_version": {
                        "type": "string",
                        "description": "Model version",
                        "default": "1",
                    },
                    "workload_size": {
                        "type": "string",
                        "enum": ["Small", "Medium", "Large"],
                        "default": "Small",
                    },
                    "scale_to_zero": {"type": "boolean", "default": True},
                },
                "required": ["endpoint_name", "model_name"],
            },
        ),
        Tool(
            name="databricks_query",
            description="Query a Databricks model serving endpoint with a prompt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "Serving endpoint name"},
                    "prompt": {"type": "string", "description": "Prompt text"},
                },
                "required": ["endpoint", "prompt"],
            },
        ),
        Tool(
            name="databricks_mlflow_experiments",
            description="List MLflow experiments in Databricks workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max experiments to return",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="databricks_mlflow_models",
            description="List registered models in the Databricks MLflow Model Registry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max models to return",
                        "default": 50,
                    },
                },
            },
        ),
        # ── v5.4.0: vLLM Extended ─────────────────────────────────────────────────
        Tool(
            name="vllm_auto_optimize",
            description="Automatically optimize vLLM configuration by analyzing workload patterns. Selects optimal settings for the 6 critical knobs based on live endpoint metrics or sample request files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "endpoint": {
                        "type": "string",
                        "description": "Running vLLM endpoint to analyze (e.g. http://localhost:8000)",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs available",
                        "default": 1,
                    },
                    "output": {
                        "type": "string",
                        "enum": ["config", "args", "helm"],
                        "default": "config",
                    },
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="vllm_analyze",
            description="Analyze a running vLLM server's workload and return specific optimization recommendations with before/after comparisons.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "vLLM endpoint URL (e.g. http://localhost:8000)",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Analysis duration in seconds",
                        "default": 60,
                    },
                },
                "required": ["endpoint"],
            },
        ),
        Tool(
            name="vllm_benchmark",
            description="Benchmark a vLLM endpoint with concurrent requests. Returns throughput (req/s), success rate, and total latency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL"},
                    "concurrent": {
                        "type": "integer",
                        "description": "Number of concurrent requests",
                        "default": 1,
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Test prompt",
                        "default": "Explain quantum computing in simple terms.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "vLLM API key (if auth enabled)",
                    },
                },
                "required": ["endpoint"],
            },
        ),
    ]

    if ALL_NEW_TOOLS:
        _ALL_TOOLS.extend(ALL_NEW_TOOLS)

    return _ALL_TOOLS
# Pre-compress at module load — cached for all subsequent list_tools calls
_COMPRESSED_TOOLS = None

def _get_compressed_tools():
    global _COMPRESSED_TOOLS
    if _COMPRESSED_TOOLS is not None:
        return _COMPRESSED_TOOLS
    if _ALL_TOOLS is None:
        _build_all_tools()
    if optimizer and MCP_AVAILABLE:
        _COMPRESSED_TOOLS = optimizer.compress_tools(_ALL_TOOLS)
        return _COMPRESSED_TOOLS
    # Python fallback: strip optional fields
    def strip_optional_fields(tool):
        if isinstance(tool.inputSchema, dict):
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            if isinstance(props, dict) and isinstance(required, list):
                required_set = set(required)
                tool.inputSchema["properties"] = {
                    k: v for k, v in props.items() if k in required_set
                }
        return tool

    _COMPRESSED_TOOLS = [strip_optional_fields(tool) for tool in _ALL_TOOLS]
    return _COMPRESSED_TOOLS


@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available Terradev tools (lazy loading with compression)"""
    global _concurrent_requests, _request_semaphore

    # Lazy-create semaphore to avoid Python 3.9 event loop binding bug
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(_max_concurrent)

    # Adaptive concurrency control
    async with _request_semaphore:
        _concurrent_requests += 1

        try:
            # Ensure tools are loaded
            await _ensure_tools_loaded()

            # Return compressed tools
            return ListToolsResult(tools=_get_compressed_tools())
        finally:
            _concurrent_requests -= 1


@server.call_tool()
async def handle_call_tool(name_or_request, arguments=None, **kwargs) -> CallToolResult:
    """Handle tool calls with adaptive concurrency control.

    Supports both the v1.x MCP low-level decorator signature (name, arguments)
    and the older / test harness signature that passes a CallToolRequest.
    """
    global _concurrent_requests, _request_semaphore

    if isinstance(name_or_request, CallToolRequest):
        request = name_or_request
        tool_name = request.params.name
        arguments = request.params.arguments or {}
    else:
        tool_name = name_or_request
        arguments = arguments or {}

    # Lazy-create semaphore to avoid Python 3.9 event loop binding bug
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(_max_concurrent)

    # Adaptive concurrency control
    async with _request_semaphore:
        _concurrent_requests += 1

        try:
            # Ensure tools are loaded
            await _ensure_tools_loaded()

            # Expand compressed namespace tools back to original tool names
            if optimizer:
                original_tool_name, original_arguments = optimizer.expand_call(
                    tool_name, arguments
                )
                tool_name = original_tool_name
                arguments = original_arguments
            else:
                # Python fallback: handle namespace expansion
                if "." in tool_name:
                    parts = tool_name.split(".", 1)
                    tool_name = parts[1]

            # Map tool names to terradev commands
            command_map = {
                "quote_gpu": ["quote"],
                "provision_gpu": ["terraform"],  # Now uses Terraform by default
                "terraform_plan": ["terraform", "plan"],
                "terraform_apply": ["terraform", "apply"],
                "terraform_destroy": ["terraform", "destroy"],
                "terraform_status": ["terraform", "status"],  # Fast state queries
                "k8s_create": ["k8s", "create"],
                "k8s_list": ["k8s", "list"],
                "k8s_info": ["k8s", "info"],
                "k8s_destroy": ["k8s", "destroy"],
                "inferx_deploy": ["inferx", "deploy"],
                "inferx_status": ["inferx", "status"],
                "inferx_list": ["inferx", "list"],
                "inferx_optimize": ["inferx", "optimize"],
                "hf_space_deploy": ["hf-space"],
                "status": ["status"],
                "manage_instance": ["manage"],
                "analytics": ["analytics"],
                "optimize": ["optimize"],
                "setup_provider": ["setup"],
                "configure_provider": ["configure"],
                # v3.4.0 tools
                "train": ["train"],
                "train_status": ["train-status"],
                "train_monitor": ["monitor"],
                "checkpoint_list": ["checkpoint", "list"],
                "checkpoint_save": ["checkpoint", "save"],
                "preflight": ["preflight"],
                "price_discovery": ["price-discovery"],
                # v3.2.0 tools
                "infer_route": ["inference", "route"],
                "infer_route_disagg": ["inference", "route"],
                "infer_status": ["inference", "status"],
                "infer_failover": ["inference", "failover"],
                "gpu_topology": ["inference", "topology"],
                "price_intel": ["analytics"],
                "moe_deploy": ["provision"],
                "gitops_init": ["gitops", "init"],
                # v2.0.0 tools — complete agentic loop
                "train_stop": ["train", "stop"],
                "train_resume": ["train", "resume"],
                "checkpoint_restore": ["checkpoint", "restore"],
                "checkpoint_promote": ["checkpoint", "promote"],
                "checkpoint_delete": ["checkpoint", "delete"],
                "stage": ["stage"],
                "infer_deploy": ["infer-deploy"],
                "up": ["up"],
                "rollback": ["rollback"],
                "manifests": ["manifests"],
                "smart_deploy": ["smart-deploy"],
                "helm_generate": ["helm-generate"],
                "gitops_bootstrap": ["gitops", "bootstrap"],
                "gitops_sync": ["gitops", "sync"],
                "gitops_validate": ["gitops", "validate"],
                "orchestrator_start": ["orchestrator", "start"],
                "orchestrator_register": ["orchestrator", "register"],
                "orchestrator_load": ["orchestrator", "load"],
                "orchestrator_evict": ["orchestrator", "evict"],
                "orchestrator_status": ["orchestrator", "status"],
                "orchestrator_infer": ["orchestrator", "infer"],
                "warm_pool_start": ["warm-pool", "start"],
                "warm_pool_status": ["warm-pool", "status"],
                "cost_scaler_start": ["cost-scaler", "start"],
                "cost_scaler_status": ["cost-scaler", "status"],
                "inferx_configure": ["inferx", "configure"],
                "inferx_delete": ["inferx", "delete"],
                "inferx_usage": ["inferx", "usage"],
                "inferx_quote": ["inferx", "quote"],
                "hf_space_status": ["hf-space"],
                "run_workflow": ["workflow", "run"],
                "active_context": ["status"],  # Composite — handled specially
                # v3.5.0: Multi-LoRA
                "lora_list": ["lora", "list"],
                "lora_add": ["lora", "add"],
                "lora_remove": ["lora", "remove"],
                # v4.0.0: ML Services — handled by custom elif blocks below
                "ray_status": ["ml", "ray", "status"],
                "ray_start": ["ml", "ray", "start"],
                "ray_stop": ["ml", "ray", "stop"],
                "ray_submit_job": ["ml", "ray", "submit"],
                "ray_list_jobs": ["ml", "ray", "jobs"],
                "ray_wide_ep_deploy": ["ml", "ray", "wide-ep"],
                "ray_disagg_pd_deploy": ["ml", "ray", "disagg-pd"],
                "ray_parallelism_strategy": ["ml", "ray", "parallelism"],
                "vllm_start": ["ml", "vllm", "start"],
                "vllm_stop": ["ml", "vllm", "stop"],
                "vllm_inference": ["ml", "vllm", "inference"],
                "vllm_info": ["ml", "vllm", "info"],
                "vllm_sleep": ["ml", "vllm", "sleep"],
                "vllm_wake": ["ml", "vllm", "wake"],
                # v5.0.0: SGLang Optimization Stack
                "sglang": ["ml", "sglang"],
                # v4.0.0: Legacy SGLang (deprecated)
                "sglang_start": ["ml", "sglang", "start"],
                "sglang_stop": ["ml", "sglang", "stop"],
                "sglang_inference": ["ml", "sglang", "inference"],
                "sglang_metrics": ["ml", "sglang", "metrics"],
                "ollama_list": ["ml", "ollama", "list"],
                "ollama_pull": ["ml", "ollama", "pull"],
                "ollama_generate": ["ml", "ollama", "generate"],
                "ollama_chat": ["ml", "ollama", "chat"],
                "ollama_model_info": ["ml", "ollama", "info"],
                "wandb_list_projects": ["ml", "wandb", "projects"],
                "wandb_list_runs": ["ml", "wandb", "runs"],
                "wandb_run_details": ["ml", "wandb", "run-details"],
                "langsmith_list_runs": ["ml", "langsmith", "runs"],
                "langsmith_list_projects": ["ml", "langsmith", "projects"],
                "langsmith_gpu_correlate": ["ml", "langsmith", "gpu-correlate"],
                "mlflow_list_experiments": ["ml", "mlflow", "experiments"],
                "mlflow_log_run": ["ml", "mlflow", "log-run"],
                "mlflow_register_model": ["ml", "mlflow", "register"],
                "dvc_status": ["ml", "dvc", "status"],
                "dvc_diff": ["ml", "dvc", "diff"],
                "dvc_stage_checkpoint": ["ml", "dvc", "stage"],
                "dvc_push": ["ml", "dvc", "push"],
                "kserve_generate_yaml": ["ml", "kserve", "generate"],
                "kserve_list": ["ml", "kserve", "list"],
                "kserve_status": ["ml", "kserve", "status"],
                "egress_cheapest_route": ["egress", "route"],
                "egress_optimize_staging": ["egress", "optimize"],
                # v5.0.0: HuggingFace Hub — handled by custom elif blocks
                "hf_list_models": ["ml", "hf", "models"],
                "hf_list_datasets": ["ml", "hf", "datasets"],
                "hf_model_info": ["ml", "hf", "model-info"],
                "hf_create_endpoint": ["ml", "hf", "create-endpoint"],
                "hf_list_endpoints": ["ml", "hf", "list-endpoints"],
                "hf_endpoint_info": ["ml", "hf", "endpoint-info"],
                "hf_delete_endpoint": ["ml", "hf", "delete-endpoint"],
                "hf_endpoint_infer": ["ml", "hf", "endpoint-infer"],
                # v5.0.0: HF Smart Templates — handled by custom elif blocks
                "hf_smart_template": ["ml", "hf", "smart-template"],
                "hf_hardware_recommend": ["ml", "hf", "hardware-recommend"],
                "hf_hardware_compare": ["ml", "hf", "hardware-compare"],
                # v5.0.0: LangChain / LangGraph — handled by custom elif blocks
                "langchain_create_workflow": ["ml", "langchain", "workflow"],
                "langchain_create_sglang_pipeline": [
                    "ml",
                    "langchain",
                    "sglang-pipeline",
                ],
                "langsmith_create_project": ["ml", "langsmith", "create-project"],
                "langsmith_get_workspaces": ["ml", "langsmith", "workspaces"],
                "langsmith_create_trace": ["ml", "langsmith", "trace"],
                "langgraph_create_workflow": ["ml", "langgraph", "workflow"],
                "langgraph_orchestrator_worker": [
                    "ml",
                    "langgraph",
                    "orchestrator-worker",
                ],
                "langgraph_evaluation_workflow": ["ml", "langgraph", "evaluation"],
                "langgraph_workflow_status": ["ml", "langgraph", "status"],
                # v5.0.0: W&B Enhanced — handled by custom elif blocks
                "wandb_create_dashboard": ["ml", "wandb", "create-dashboard"],
                "wandb_create_terradev_dashboard": [
                    "ml",
                    "wandb",
                    "terradev-dashboard",
                ],
                "wandb_create_report": ["ml", "wandb", "create-report"],
                "wandb_create_terradev_report": ["ml", "wandb", "terradev-report"],
                "wandb_setup_alerts": ["ml", "wandb", "setup-alerts"],
                "wandb_create_terradev_alerts": ["ml", "wandb", "terradev-alerts"],
                "wandb_dashboard_status": ["ml", "wandb", "dashboard-status"],
                # v5.0.0: Data Governance — handled by custom elif blocks
                "governance_request_consent": ["governance", "consent-request"],
                "governance_record_consent": ["governance", "consent-record"],
                "governance_evaluate_opa": ["governance", "evaluate-opa"],
                "governance_move_data": ["governance", "move"],
                "governance_movement_history": ["governance", "history"],
                "governance_compliance_report": ["governance", "compliance-report"],
                # v5.0.0: Cost Optimizer Deep — handled by custom elif blocks
                "cost_analyze": ["cost", "analyze"],
                "cost_optimize_recommend": ["cost", "recommend"],
                "cost_simulate": ["cost", "simulate"],
                "cost_budget_optimize": ["cost", "budget-optimize"],
                # v5.0.0: Price Intelligence Extended — handled by custom elif blocks
                "price_trends": ["price", "trends"],
                "price_budget_optimize": ["price", "budget-optimize"],
                "price_spot_risk": ["price", "spot-risk"],
                # v5.0.0: Training Extended — handled by custom elif blocks
                "training_config_generate": ["train", "config-generate"],
                "training_launch_distributed": ["train", "launch-distributed"],
                "train_snapshot": ["train", "snapshot"],
                "train_detect_stragglers": ["train", "detect-stragglers"],
                # v5.0.0: Preflight Extended — handled by custom elif blocks
                "preflight_report": ["preflight", "report"],
                "preflight_gpu_check": ["preflight", "gpu-check"],
                "preflight_network_check": ["preflight", "network-check"],
                # v5.0.0: Kubernetes Enhanced — handled by custom elif blocks
                "k8s_gpu_operator_install": ["k8s", "gpu-operator"],
                "k8s_device_plugin": ["k8s", "device-plugin"],
                "k8s_mig_configure": ["k8s", "mig-configure"],
                "k8s_time_slicing": ["k8s", "time-slicing"],
                "k8s_monitoring_stack": ["k8s", "monitoring-stack"],
                # v5.1.0: Datadog Integration — handled by custom elif blocks
                "datadog_status": ["datadog", "status"],
                "datadog_push_metrics": ["datadog", "push"],
                "datadog_send_event": ["datadog", "event"],
                "datadog_create_monitors": ["datadog", "monitors-create"],
                "datadog_list_monitors": ["datadog", "monitors-list"],
                "datadog_create_dashboard": ["datadog", "dashboard-create"],
                "datadog_list_dashboards": ["datadog", "dashboard-list"],
                "datadog_query": ["datadog", "query"],
                "datadog_terraform_export": ["datadog", "terraform-export"],
                "datadog_metric_catalog": ["datadog", "metric-catalog"],
                # v5.2.0: Phoenix / Guardrails / Qdrant
                "phoenix_test": ["phoenix", "test"],
                "phoenix_projects": ["phoenix", "projects"],
                "phoenix_spans": ["phoenix", "spans"],
                "phoenix_trace": ["phoenix", "trace"],
                "phoenix_otel_env": ["phoenix", "otel-env"],
                "phoenix_snippet": ["phoenix", "snippet"],
                "phoenix_k8s": ["phoenix", "k8s"],
                "guardrails_test": ["guardrails", "test"],
                "guardrails_chat": ["guardrails", "chat"],
                "guardrails_generate_config": ["guardrails", "generate-config"],
                "guardrails_k8s": ["guardrails", "k8s"],
                "qdrant_test": ["qdrant", "test"],
                "qdrant_collections": ["qdrant", "collections"],
                "qdrant_create_collection": ["qdrant", "create-collection"],
                "qdrant_info": ["qdrant", "info"],
                "qdrant_count": ["qdrant", "count"],
                "qdrant_k8s": ["qdrant", "k8s"],
                # v5.3.0: New v4.0.11 Features - Karpenter, Triggers, Environments, Lineage, Migration
                **NEW_COMMAND_MAP,
                # v5.4.0: Langfuse LLM Observability
                "langfuse_configure": ["langfuse", "configure"],
                "langfuse_test": ["langfuse", "test"],
                "langfuse_traces": ["langfuse", "traces"],
                "langfuse_trace": ["langfuse", "trace"],
                "langfuse_scores": ["langfuse", "scores"],
                "langfuse_score": ["langfuse", "score"],
                "langfuse_datasets": ["langfuse", "datasets"],
                "langfuse_export_training_data": ["langfuse", "export-training-data"],
                "langfuse_quality": ["langfuse", "quality"],
                "langfuse_otel_env": ["langfuse", "otel-env"],
                "langfuse_k8s": ["langfuse", "k8s"],
                # v5.4.0: Databricks MLOps
                "databricks_configure": ["databricks", "configure"],
                "databricks_test": ["databricks", "test"],
                "databricks_jobs": ["databricks", "jobs"],
                "databricks_run": ["databricks", "run"],
                "databricks_run_status": ["databricks", "run-status"],
                "databricks_clusters": ["databricks", "clusters"],
                "databricks_serving_endpoints": ["databricks", "serving-endpoints"],
                "databricks_deploy_model": ["databricks", "deploy-model"],
                "databricks_query": ["databricks", "query"],
                "databricks_mlflow_experiments": [
                    "databricks",
                    "mlflow",
                    "experiments",
                ],
                "databricks_mlflow_models": ["databricks", "mlflow", "models"],
                # v5.4.0: vLLM Extended
                "vllm_auto_optimize": ["vllm", "auto-optimize"],
                "vllm_analyze": ["vllm", "analyze"],
                "vllm_benchmark": ["vllm", "benchmark"],
                # v4.0.12: New ML Services and Enhanced Features
                "phoenix": ["phoenix"],
                "guardrails": ["guardrails"],
                "qdrant": ["qdrant"],
                "latitude": ["latitude"],
                "flashoptim": ["flashoptim"],
                "vllm_enhanced": ["vllm-enhanced"],
                "sso": ["sso"],
                "cost_optimizer": ["cost-optimizer"],
                "governance": ["governance"],
                "local_scan": ["local-scan"],
            }

            if tool_name not in command_map:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Unknown tool: {tool_name}")
                    ],
                    isError=True,
                )

            # Build command arguments
            cmd_args = command_map[tool_name].copy()

            # Handle database tools via new_feature_tools handlers
            if tool_name in ["create_sqlite_connection", "create_postgresql_connection", "query_database", "upsert_database", "get_database_connection"]:
                if tool_name in NEW_COMMAND_MAP:
                    handler = NEW_COMMAND_MAP[tool_name]
                    result = await handler(arguments)
                    return CallToolResult(
                        content=[TextContent(type="text", text=result[0]["text"])]
                    )
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Database tool handler not found: {tool_name}")],
                        isError=True
                    )

            # Handle local_scan tool separately (doesn't use terradev CLI)
            if tool_name == "local_scan":
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

            if tool_name == "quote_gpu":
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

            if tool_name == "provision_gpu":
                # Use Terraform for all GPU provisioning (core IP)
                gpu_type = arguments["gpu_type"]
                count = arguments.get("count", 1)
                providers = arguments.get(
                    "providers", ["runpod", "vastai", "lambda", "aws"]
                )
                max_price = arguments.get("max_price")
                arguments.get("plan_only", False)

                result = await execute_terraform_parallel(
                    gpu_type, count, providers, max_price
                )

                if result["success"]:
                    output_text = "✅ GPU provisioning via Terraform successful!\n\n"
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

                    output_text += "\n**Terraform State:** Managed\n"
                    output_text += f"**Full Output:**\n{result['stdout']}"

                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Terraform provisioning failed: {result['stderr']}",
                            )
                        ],
                        isError=True,
                    )

            elif tool_name == "terraform_plan":
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
                                text=f"✅ Terraform plan generated:\n\n{result['stdout']}",
                            )
                        ]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Terraform plan failed: {result['stderr']}",
                            )
                        ],
                        isError=True,
                    )

            elif tool_name == "terraform_apply":
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
                                text=f"✅ Terraform apply successful:\n\n{result['stdout']}",
                            )
                        ]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Terraform apply failed: {result['stderr']}",
                            )
                        ],
                        isError=True,
                    )

            elif tool_name == "terraform_destroy":
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
                                text=f"✅ Terraform destroy successful:\n\n{result['stdout']}",
                            )
                        ]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Terraform destroy failed: {result['stderr']}",
                            )
                        ],
                        isError=True,
                    )

            elif tool_name == "k8s_create":
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

            elif tool_name == "k8s_info":
                cmd_args.append(arguments["cluster_name"])

            elif tool_name == "k8s_destroy":
                cmd_args.append(arguments["cluster_name"])

            elif tool_name == "inferx_deploy":
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

            elif tool_name == "hf_space_deploy":
                cmd_args.append(arguments["space_name"])
                cmd_args.extend(["--model-id", arguments["model_id"]])
                cmd_args.extend(["--template", arguments["template"]])
                if "hardware" in arguments:
                    cmd_args.extend(["--hardware", arguments["hardware"]])
                if "sdk" in arguments:
                    cmd_args.extend(["--sdk", arguments["sdk"]])

            elif tool_name == "terraform_status":
                config_dir = arguments["config_dir"]
                show_outputs = arguments.get("show_outputs", True)

                # Fast status query using Terraform state
                output_result = await execute_terraform_command(
                    ["terraform", "output", "-json"], config_dir
                )

                if output_result["success"] and show_outputs:
                    try:
                        outputs = json.loads(output_result["stdout"])
                        output_text = "✅ Terraform Status (from state):\n\n"

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
                            output_text += "**State File:** Terraform managed\n"

                        return CallToolResult(
                            content=[TextContent(type="text", text=output_text)]
                        )
                    except json.JSONDecodeError:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"✅ Terraform Status:\n\n{output_result['stdout']}",
                                )
                            ]
                        )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"❌ Terraform status query failed: {output_result['stderr']}",
                            )
                        ],
                        isError=True,
                    )

            elif tool_name == "status":
                if arguments.get("live"):
                    cmd_args.append("--live")

            elif tool_name == "manage_instance":
                cmd_args.extend(["-i", arguments["instance_id"]])
                cmd_args.extend(["-a", arguments["action"]])

            elif tool_name == "analytics":
                if "days" in arguments:
                    cmd_args.extend(["--days", str(arguments["days"])])

            elif tool_name == "setup_provider":
                cmd_args.append(arguments["provider"])
                if arguments.get("quick"):
                    cmd_args.append("--quick")

            elif tool_name == "configure_provider":
                cmd_args.extend(["--provider", arguments["provider"]])

            # ── v3.2.0 Handlers ──────────────────────────────────────────────────

            elif tool_name == "infer_route":
                # Semantic-aware inference routing via terradev inference route
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

            elif tool_name == "infer_route_disagg":
                # Disaggregated prefill/decode routing
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

            elif tool_name == "infer_status":
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

            elif tool_name == "infer_failover":
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

            elif tool_name == "gpu_topology":
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

            elif tool_name == "price_intel":
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

            elif tool_name == "moe_deploy":
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

            # ── v3.5.0 Handlers: Multi-LoRA ───────────────────────────────────

            elif tool_name == "lora_list":
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

            elif tool_name == "lora_add":
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

            elif tool_name == "lora_remove":
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

            # ── v3.4.0 Handlers ──────────────────────────────────────────────────

            elif tool_name == "train":
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

            elif tool_name == "train_status":
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

            elif tool_name == "train_monitor":
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

            elif tool_name == "checkpoint_list":
                cmd_args = ["checkpoint", "list", "--job", arguments["job_id"]]
                result = await execute_terradev_command(cmd_args)
                output = result["stdout"] if result["success"] else result["stderr"]
                output_text = "💾 **Checkpoints**\n\n" + output
                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)],
                    isError=not result["success"],
                )

            elif tool_name == "checkpoint_save":
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

            elif tool_name == "preflight":
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

            elif tool_name == "price_discovery":
                cmd_args = ["price-discovery", "--gpu-type", arguments["gpu_type"]]
                if "region" in arguments:
                    cmd_args.extend(["--region", arguments["region"]])
                if "hours" in arguments:
                    cmd_args.extend(["--hours", str(arguments["hours"])])
                result = await execute_terradev_command(cmd_args)
                output = result["stdout"] if result["success"] else result["stderr"]
                output_text = (
                    f"💰 **Price Discovery — {arguments['gpu_type']}**\n\n" + output
                )
                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)],
                    isError=not result["success"],
                )

            elif tool_name == "gitops_init":
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

            # ── v2.0.0 Handlers — Complete Agentic Loop ────────────────────────

            elif tool_name == "train_stop":
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

            elif tool_name == "train_resume":
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

            elif tool_name == "checkpoint_restore":
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

            elif tool_name == "checkpoint_promote":
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

            elif tool_name == "checkpoint_delete":
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

            elif tool_name == "stage":
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

            elif tool_name == "infer_deploy":
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

            elif tool_name == "up":
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

            elif tool_name == "rollback":
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

            elif tool_name == "manifests":
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

            elif tool_name == "smart_deploy":
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

            elif tool_name == "helm_generate":
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

            elif tool_name == "gitops_bootstrap":
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

            elif tool_name == "gitops_sync":
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

            elif tool_name == "gitops_validate":
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

            elif tool_name == "orchestrator_start":
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

            elif tool_name == "orchestrator_register":
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

            elif tool_name == "orchestrator_load":
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

            elif tool_name == "orchestrator_evict":
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

            elif tool_name == "orchestrator_status":
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

            elif tool_name == "orchestrator_infer":
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

            elif tool_name == "warm_pool_start":
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

            elif tool_name == "warm_pool_status":
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

            elif tool_name == "cost_scaler_start":
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

            elif tool_name == "cost_scaler_status":
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

            elif tool_name == "inferx_configure":
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

            elif tool_name == "inferx_delete":
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

            elif tool_name == "inferx_usage":
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

            elif tool_name == "inferx_quote":
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

            elif tool_name == "hf_space_status":
                cmd_args = ["hf-space", "--status", arguments["space_name"]]
                result = await execute_terradev_command(cmd_args)
                output = result["stdout"] if result["success"] else result["stderr"]
                output_text = (
                    f"🤗 **HF Space Status: {arguments['space_name']}**\n\n" + output
                )
                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)],
                    isError=not result["success"],
                )

            elif tool_name == "run_workflow":
                # Workflow primitives — runs a YAML pipeline or built-in template
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

            elif tool_name == "active_context":
                # Composite tool: gather state from multiple commands
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

            # ── v4.0.0 Handlers: ML Services ────────────────────────────────────

            elif tool_name == "ray_status":
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

            elif tool_name == "ray_start":
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

            elif tool_name == "ray_stop":
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

            elif tool_name == "ray_submit_job":
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

            elif tool_name == "ray_list_jobs":
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

            elif tool_name == "ray_wide_ep_deploy":
                # Use EnhancedRayService to generate Wide-EP config
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

            elif tool_name == "ray_disagg_pd_deploy":
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

            elif tool_name == "ray_parallelism_strategy":
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

            # ── vLLM Lifecycle Handlers ──────────────────────────────────────────

            elif tool_name == "vllm_start":
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

            elif tool_name == "vllm_stop":
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

            elif tool_name == "vllm_inference":
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

            elif tool_name == "vllm_info":
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

            elif tool_name == "vllm_sleep":
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

            elif tool_name == "vllm_wake":
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

            # ── v5.0.0: SGLang Optimization Stack Handlers ──────────────────────────────────────────────────

            elif tool_name == "sglang":
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

            # ── Legacy SGLang Handlers (deprecated) ──────────────────────────────────────────────────

            elif tool_name == "sglang_start":
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
                    output_text += "**suggest_action:** Test with `sglang_inference`. Check metrics with `sglang_metrics`."
                else:
                    output_text = f"❌ **Failed to start SGLang**\n\n{result['stderr']}"
                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)],
                    isError=not result["success"],
                )

            elif tool_name == "sglang_stop":
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

            elif tool_name == "sglang_inference":
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

            elif tool_name == "sglang_metrics":
                endpoint = arguments["endpoint"].rstrip("/")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{endpoint}/metrics",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status == 200:
                                raw = await resp.text()
                                output_text = f"📊 **SGLang Metrics — {endpoint}**\n\n```\n{raw[:3000]}\n```"
                                return CallToolResult(
                                    content=[TextContent(type="text", text=output_text)]
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
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            # ── Ollama Handlers ──────────────────────────────────────────────────

            elif tool_name == "ollama_list":
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

            elif tool_name == "ollama_pull":
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

            elif tool_name == "ollama_generate":
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

            elif tool_name == "ollama_chat":
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

            elif tool_name == "ollama_model_info":
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

            # ── W&B Handlers ─────────────────────────────────────────────────────

            elif tool_name == "wandb_list_projects":
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

            elif tool_name == "wandb_list_runs":
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

            elif tool_name == "wandb_run_details":
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

            # ── LangSmith Handlers ───────────────────────────────────────────────

            elif tool_name == "langsmith_list_projects":
                api_key = arguments["api_key"]
                try:
                    async with aiohttp.ClientSession(
                        headers={"x-api-key": api_key}
                    ) as session:
                        async with session.get(
                            "https://api.smith.langchain.com/api/v1/projects",
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                output_text = "🔗 **LangSmith Projects**\n\n"
                                for p in (
                                    data
                                    if isinstance(data, list)
                                    else data.get("projects", [])
                                )[:50]:
                                    name = (
                                        p.get("name", "?")
                                        if isinstance(p, dict)
                                        else str(p)
                                    )
                                    output_text += f"  - **{name}**\n"
                                return CallToolResult(
                                    content=[TextContent(type="text", text=output_text)]
                                )
                            else:
                                body = await resp.text()
                                return CallToolResult(
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=f"❌ LangSmith {resp.status}: {body[:500]}",
                                        )
                                    ],
                                    isError=True,
                                )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "langsmith_list_runs":
                api_key = arguments["api_key"]
                project = arguments.get("project", "default")
                limit = arguments.get("limit", 50)
                try:
                    async with aiohttp.ClientSession(
                        headers={"x-api-key": api_key}
                    ) as session:
                        async with session.get(
                            "https://api.smith.langchain.com/api/v1/runs",
                            params={"project_name": project, "limit": limit},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                runs = (
                                    data
                                    if isinstance(data, list)
                                    else data.get("runs", [])
                                )
                                output_text = f"📋 **LangSmith Runs — {project}**\n\n"
                                for r in runs[:limit]:
                                    name = (
                                        r.get("name", r.get("id", "?"))
                                        if isinstance(r, dict)
                                        else str(r)
                                    )
                                    output_text += f"  - **{name}**\n"
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

            elif tool_name == "langsmith_gpu_correlate":
                api_key = arguments["api_key"]
                project = arguments.get("project", "default")
                days = arguments.get("days", 7)
                try:
                    from terradev_cli.ml_services.langsmith_service import (
                        LangSmithService,
                        LangSmithConfig,
                    )

                    svc = LangSmithService(LangSmithConfig(api_key=api_key))
                    correlation = await svc.correlate_runs_with_gpu_metrics(
                        project_name=project, days=days
                    )
                    output_text = (
                        f"🔗💰 **GPU-Correlated LangSmith Runs — {project}**\n\n"
                    )
                    output_text += f"```json\n{json.dumps(correlation, indent=2, default=str)[:3000]}\n```\n\n"
                    output_text += "**suggest_action:** Use this data to identify cost-efficient GPU/provider combos for your LLM workloads."
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
                            TextContent(type="text", text=f"❌ Correlation failed: {e}")
                        ],
                        isError=True,
                    )

            # ── MLflow Handlers ──────────────────────────────────────────────────

            elif tool_name == "mlflow_list_experiments":
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

            elif tool_name == "mlflow_log_run":
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

            elif tool_name == "mlflow_register_model":
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

            # ── DVC Handlers ─────────────────────────────────────────────────────

            elif tool_name == "dvc_status":
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

            elif tool_name == "dvc_diff":
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

            elif tool_name == "dvc_stage_checkpoint":
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

            elif tool_name == "dvc_push":
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

            # ── KServe Handlers ──────────────────────────────────────────────────

            elif tool_name == "kserve_generate_yaml":
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

            elif tool_name == "kserve_list":
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

            elif tool_name == "kserve_status":
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

            # ── Egress Optimizer Handlers ────────────────────────────────────────

            elif tool_name == "egress_cheapest_route":
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

            elif tool_name == "egress_optimize_staging":
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

            # ── v5.0.0: HuggingFace Hub Handlers ────────────────────────────────

            elif tool_name == "hf_list_models":
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

            elif tool_name == "hf_list_datasets":
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

            elif tool_name == "hf_model_info":
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

            elif tool_name == "hf_create_endpoint":
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

            elif tool_name == "hf_list_endpoints":
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

            elif tool_name == "hf_endpoint_info":
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

            elif tool_name == "hf_delete_endpoint":
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

            elif tool_name == "hf_endpoint_infer":
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

            # ── v5.0.0: HF Smart Templates Handlers ──────────────────────────────

            elif tool_name == "hf_smart_template":
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

            elif tool_name == "hf_hardware_recommend":
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

            elif tool_name == "hf_hardware_compare":
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

            # ── v5.0.0: LangChain / LangGraph / LangSmith Handlers ───────────────

            elif tool_name == "langchain_create_workflow":
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

            elif tool_name == "langchain_create_sglang_pipeline":
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

            elif tool_name == "langsmith_create_project":
                try:
                    api_key = arguments["api_key"]
                    payload = {"name": arguments["name"]}
                    if arguments.get("description"):
                        payload["description"] = arguments["description"]
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.smith.langchain.com/api/v1/sessions",
                            headers={
                                "x-api-key": api_key,
                                "Content-Type": "application/json",
                            },
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            body = await resp.json(content_type=None)
                            if resp.status in (200, 201):
                                output_text = f"✅ **LangSmith Project Created: {arguments['name']}**\n\n"
                                output_text += f"**ID:** {body.get('id', 'N/A')}\n"
                                return CallToolResult(
                                    content=[TextContent(type="text", text=output_text)]
                                )
                            else:
                                return CallToolResult(
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=f"❌ LangSmith {resp.status}: {json.dumps(body, default=str)[:500]}",
                                        )
                                    ],
                                    isError=True,
                                )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "langsmith_get_workspaces":
                try:
                    api_key = arguments["api_key"]
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            "https://api.smith.langchain.com/api/v1/workspaces",
                            headers={"x-api-key": api_key},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if resp.status == 200:
                                workspaces = await resp.json(content_type=None)
                                items = (
                                    workspaces
                                    if isinstance(workspaces, list)
                                    else workspaces.get("workspaces", [workspaces])
                                )
                                output_text = (
                                    f"🔗 **LangSmith Workspaces** ({len(items)})\n\n"
                                )
                                for w in items:
                                    name = w.get(
                                        "display_name", w.get("name", w.get("id", "?"))
                                    )
                                    output_text += (
                                        f"- **{name}** (ID: {w.get('id', '?')})\n"
                                    )
                                return CallToolResult(
                                    content=[TextContent(type="text", text=output_text)]
                                )
                            else:
                                body = await resp.text()
                                return CallToolResult(
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=f"❌ LangSmith {resp.status}: {body[:500]}",
                                        )
                                    ],
                                    isError=True,
                                )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "langsmith_create_trace":
                try:
                    api_key = arguments["api_key"]
                    run_id = arguments["run_id"]
                    trace_data = arguments["trace_data"]
                    trace_data["id"] = run_id
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.smith.langchain.com/api/v1/runs",
                            headers={
                                "x-api-key": api_key,
                                "Content-Type": "application/json",
                            },
                            json=trace_data,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if resp.status in (200, 201, 202):
                                output_text = f"✅ **Trace Created: {run_id}**\n"
                                return CallToolResult(
                                    content=[TextContent(type="text", text=output_text)]
                                )
                            else:
                                body = await resp.text()
                                return CallToolResult(
                                    content=[
                                        TextContent(
                                            type="text",
                                            text=f"❌ LangSmith {resp.status}: {body[:500]}",
                                        )
                                    ],
                                    isError=True,
                                )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "langgraph_create_workflow":
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

            elif tool_name == "langgraph_orchestrator_worker":
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

            elif tool_name == "langgraph_evaluation_workflow":
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

            elif tool_name == "langgraph_workflow_status":
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

            # ── v5.0.0: W&B Enhanced Handlers ─────────────────────────────────────

            elif tool_name == "wandb_create_dashboard":
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

            elif tool_name == "wandb_create_terradev_dashboard":
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

            elif tool_name == "wandb_create_report":
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

            elif tool_name == "wandb_create_terradev_report":
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

            elif tool_name == "wandb_setup_alerts":
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

            elif tool_name == "wandb_create_terradev_alerts":
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

            elif tool_name == "wandb_dashboard_status":
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

            # ── v5.0.0: Data Governance Handlers ──────────────────────────────────

            elif tool_name == "governance_request_consent":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    result = await gov.request_consent(
                        user_id=arguments["user_id"],
                        consent_type=arguments["consent_type"],
                        dataset_name=arguments["dataset_name"],
                        purpose=arguments["purpose"],
                        source_location=arguments.get("source_location"),
                        target_location=arguments.get("target_location"),
                    )
                    output_text = "📋 **Consent Request Created**\n\n"
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

            elif tool_name == "governance_record_consent":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    result = await gov.record_consent(
                        request_id=arguments["request_id"],
                        user_id=arguments["user_id"],
                        granted=arguments["granted"],
                        conditions=arguments.get("conditions"),
                    )
                    icon = "✅" if arguments["granted"] else "❌"
                    output_text = f"{icon} **Consent {'Granted' if arguments['granted'] else 'Denied'}**\n\n"
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

            elif tool_name == "governance_evaluate_opa":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    result = await gov.evaluate_opa(
                        user_id=arguments["user_id"],
                        dataset_name=arguments["dataset_name"],
                        action=arguments["action"],
                        target_location=arguments.get("target_location"),
                    )
                    allowed = result.get(
                        "allowed", result.get("result", {}).get("allow", False)
                    )
                    icon = "✅" if allowed else "🚫"
                    output_text = f"{icon} **OPA Policy Evaluation**\n\n"
                    output_text += f"**Action:** {arguments['action']} on {arguments['dataset_name']}\n"
                    output_text += (
                        f"**Decision:** {'ALLOWED' if allowed else 'DENIED'}\n\n"
                    )
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

            elif tool_name == "governance_move_data":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    result = await gov.move_data(
                        user_id=arguments["user_id"],
                        consent_request_id=arguments["consent_request_id"],
                        dataset_name=arguments["dataset_name"],
                        source_location=arguments["source_location"],
                        target_location=arguments["target_location"],
                    )
                    output_text = f"📦 **Data Move — {arguments['dataset_name']}**\n\n"
                    output_text += f"**From:** {arguments['source_location']}\n"
                    output_text += f"**To:** {arguments['target_location']}\n\n"
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

            elif tool_name == "governance_movement_history":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    result = await gov.movement_history(
                        user_id=arguments.get("user_id"),
                        dataset_name=arguments.get("dataset_name"),
                        limit=arguments.get("limit", 50),
                    )
                    output_text = "📜 **Data Movement History**\n\n"
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

            elif tool_name == "governance_compliance_report":
                try:
                    from terradev_cli.core.data_governance import DataGovernance

                    gov = DataGovernance()
                    # Parallel: gather consent stats + movement history + policy evaluations
                    report = await gov.compliance_report(
                        start_date=arguments["start_date"],
                        end_date=arguments["end_date"],
                    )
                    output_text = "📋 **Compliance Report**\n\n"
                    output_text += f"**Period:** {arguments['start_date']} → {arguments['end_date']}\n\n"
                    output_text += f"```json\n{json.dumps(report, indent=2, default=str)[:5000]}\n```"
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

            # ── v5.0.0: Cost Optimizer Deep Handlers ──────────────────────────────

            elif tool_name == "cost_analyze":
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

            elif tool_name == "cost_optimize_recommend":
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

            elif tool_name == "cost_simulate":
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

            elif tool_name == "cost_budget_optimize":
                try:
                    from terradev_cli.core.cost_optimizer import CostOptimizer

                    cost_optimizer = CostOptimizer()
                    result = await cost_optimizer.budget_optimize(
                        budget=arguments["budget"],
                        gpu_type=arguments.get("gpu_type"),
                        gpu_count=arguments.get("gpu_count", 1),
                        hours=arguments.get("hours", 1.0),
                        allow_spot=arguments.get("allow_spot", True),
                    )
                    output_text = (
                        f"💰 **Budget Optimization — ${arguments['budget']}**\n\n"
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

            # ── v5.0.0: Price Intelligence Extended Handlers ──────────────────────

            elif tool_name == "price_trends":
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

            elif tool_name == "price_budget_optimize":
                try:
                    from terradev_cli.core.price_intelligence import PriceIntelligence

                    intel = PriceIntelligence()
                    result = await intel.budget_optimize(
                        budget=arguments["budget"],
                        gpu_type=arguments["gpu_type"],
                        gpu_count=arguments.get("gpu_count", 1),
                        hours=arguments.get("hours", 1.0),
                    )
                    output_text = "💰 **Price Budget Optimization**\n\n"
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

            elif tool_name == "price_spot_risk":
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

            # ── v5.0.0: Training Extended Handlers ────────────────────────────────

            elif tool_name == "training_config_generate":
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

            elif tool_name == "training_launch_distributed":
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

            elif tool_name == "train_snapshot":
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

            elif tool_name == "train_detect_stragglers":
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

            # ── v5.0.0: Preflight Extended Handlers ───────────────────────────────

            elif tool_name == "preflight_report":
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

            elif tool_name == "preflight_gpu_check":
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

            elif tool_name == "preflight_network_check":
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

            # ── v5.0.0: Kubernetes Enhanced Handlers ──────────────────────────────

            elif tool_name == "k8s_gpu_operator_install":
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

            elif tool_name == "k8s_device_plugin":
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

            elif tool_name == "k8s_mig_configure":
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

            elif tool_name == "k8s_time_slicing":
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

            elif tool_name == "k8s_monitoring_stack":
                try:
                    from terradev_cli.ml_services.kubernetes_enhanced import (
                        EnhancedKubernetesService,
                    )

                    svc = EnhancedKubernetesService()
                    result = await svc.install_monitoring_stack(
                        cluster_name=arguments["cluster_name"],
                        namespace=arguments.get("namespace", "monitoring"),
                        grafana_password=arguments.get("grafana_password"),
                        enable_alerting=arguments.get("enable_alerting", True),
                    )
                    output_text = f"📊 **Monitoring Stack Deployed — {arguments['cluster_name']}**\n\n"
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

            # ── v5.1.0: Datadog Integration Handlers ────────────────────────────

            elif tool_name == "datadog_status":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        get_status_summary,
                        METRIC_CATALOG,
                        MONITOR_TEMPLATES,
                    )

                    # Load creds from ~/.terradev/credentials.json
                    creds = _load_datadog_creds()
                    status = get_status_summary(creds)
                    status["available_metrics"] = len(METRIC_CATALOG)
                    status["available_monitors"] = list(MONITOR_TEMPLATES.keys())
                    output_text = "🐕 **Datadog Integration Status**\n\n"
                    output_text += f"```json\n{json.dumps(status, indent=2)}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_push_metrics":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        push_cost_snapshot,
                    )

                    creds = _load_datadog_creds()
                    result = await push_cost_snapshot(creds)
                    output_text = "📤 **Datadog Metrics Push**\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_send_event":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        send_event_async,
                    )

                    creds = _load_datadog_creds()
                    result = await send_event_async(
                        creds,
                        title=arguments["title"],
                        text=arguments["text"],
                        alert_type=arguments.get("alert_type", "info"),
                    )
                    output_text = "📨 **Datadog Event Sent**\n\n"
                    output_text += (
                        f"```json\n{json.dumps(result, indent=2, default=str)}\n```"
                    )
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_create_monitors":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        create_monitor,
                        create_all_monitors,
                    )

                    creds = _load_datadog_creds()
                    template = arguments.get("template")
                    if template:
                        result = await create_monitor(creds, template)
                    else:
                        result = await create_all_monitors(creds)
                    output_text = "🔔 **Datadog Monitors**\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_list_monitors":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        list_monitors,
                    )

                    creds = _load_datadog_creds()
                    result = await list_monitors(creds)
                    output_text = "📋 **Terradev Monitors in Datadog**\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_create_dashboard":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        create_dashboard,
                    )

                    creds = _load_datadog_creds()
                    result = await create_dashboard(
                        creds, custom_title=arguments.get("title")
                    )
                    output_text = "📊 **Datadog Dashboard Created**\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_list_dashboards":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        list_dashboards,
                    )

                    creds = _load_datadog_creds()
                    result = await list_dashboards(creds)
                    output_text = "📋 **Terradev Dashboards in Datadog**\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_query":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        query_metrics,
                    )

                    creds = _load_datadog_creds()
                    result = await query_metrics(
                        creds,
                        query=arguments["query"],
                        from_seconds=arguments.get("from_seconds", 3600),
                    )
                    output_text = f"🔍 **Datadog Query:** `{arguments['query']}`\n\n"
                    output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:5000]}\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_terraform_export":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        generate_full_terraform_module,
                    )

                    creds = _load_datadog_creds()
                    out_dir = arguments.get("output_dir", "./datadog-terraform")
                    files = generate_full_terraform_module(creds)
                    os.makedirs(out_dir, exist_ok=True)
                    written = []
                    for fname, content in files.items():
                        fpath = os.path.join(out_dir, fname)
                        with open(fpath, "w") as f:
                            f.write(content)
                        written.append(fpath)
                    output_text = f"🏗️ **Terraform Module Exported → `{out_dir}/`**\n\n"
                    output_text += "**Files:**\n" + "\n".join(
                        f"- `{w}`" for w in written
                    )
                    output_text += f"\n\n**Next steps:**\n```bash\ncd {out_dir}\nterraform init\nterraform plan\nterraform apply\n```"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            elif tool_name == "datadog_metric_catalog":
                try:
                    from terradev_cli.integrations.datadog_integration import (
                        METRIC_CATALOG,
                    )

                    output_text = "📖 **Terradev Metric Catalog for Datadog**\n\n"
                    output_text += f"**{len(METRIC_CATALOG)} metrics available:**\n\n"
                    for name, meta in METRIC_CATALOG.items():
                        tags = ", ".join(meta.get("tags", []))
                        output_text += f"- `{name}` ({meta['type']}, {meta['unit']}) — {meta['desc']}"
                        if tags:
                            output_text += f" [tags: {tags}]"
                        output_text += "\n"
                    return CallToolResult(
                        content=[TextContent(type="text", text=output_text)]
                    )
                except Exception as e:  # noqa: BLE001
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ {e}")], isError=True
                    )

            # ── v5.2.0: Phoenix / Guardrails / Qdrant argument builders ──────────

            elif tool_name == "phoenix_projects":
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "phoenix_spans":
                if arguments.get("project"):
                    cmd_args.extend(["--project", arguments["project"]])
                if arguments.get("filter"):
                    cmd_args.extend(["--filter", arguments["filter"]])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "phoenix_trace":
                cmd_args.extend(["--trace-id", arguments["trace_id"]])
                if arguments.get("project"):
                    cmd_args.extend(["--project", arguments["project"]])

            elif tool_name == "phoenix_otel_env":
                if arguments.get("project"):
                    cmd_args.extend(["--project", arguments["project"]])

            elif tool_name == "phoenix_snippet":
                if arguments.get("project"):
                    cmd_args.extend(["--project", arguments["project"]])

            elif tool_name == "phoenix_k8s":
                if arguments.get("namespace"):
                    cmd_args.extend(["--namespace", arguments["namespace"]])

            elif tool_name == "guardrails_chat":
                cmd_args.extend(["--message", arguments["message"]])
                if arguments.get("config_id"):
                    cmd_args.extend(["--config-id", arguments["config_id"]])

            elif tool_name == "guardrails_generate_config":
                if arguments.get("config_id"):
                    cmd_args.extend(["--config-id", arguments["config_id"]])
                if arguments.get("output_dir"):
                    cmd_args.extend(["--output-dir", arguments["output_dir"]])

            elif tool_name == "guardrails_k8s":
                if arguments.get("namespace"):
                    cmd_args.extend(["--namespace", arguments["namespace"]])

            elif tool_name == "qdrant_create_collection":
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])
                if arguments.get("embedding_model"):
                    cmd_args.extend(["--embedding-model", arguments["embedding_model"]])

            elif tool_name == "qdrant_info":
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])

            elif tool_name == "qdrant_count":
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])

            elif tool_name == "qdrant_k8s":
                if arguments.get("namespace"):
                    cmd_args.extend(["--namespace", arguments["namespace"]])

            # ── v5.3.0: New v4.0.11 Features - Argument Builders ────────────────────────

            # Karpenter tools
            elif tool_name == "karpenter_install":
                if arguments.get("version"):
                    cmd_args.extend(["--version", arguments["version"]])
                if arguments.get("cluster_name"):
                    cmd_args.extend(["--cluster", arguments["cluster_name"]])

            elif tool_name == "karpenter_status":
                if arguments.get("format"):
                    cmd_args.extend(["--format", arguments["format"]])

            elif tool_name == "karpenter_nodepools":
                if arguments.get("format"):
                    cmd_args.extend(["--format", arguments["format"]])

            elif tool_name == "karpenter_create_nodepool":
                if arguments.get("gpu_type"):
                    cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
                if arguments.get("cpu_limit"):
                    cmd_args.extend(["--cpu-limit", arguments["cpu_limit"]])
                if arguments.get("memory_limit"):
                    cmd_args.extend(["--memory-limit", arguments["memory_limit"]])

            elif tool_name == "karpenter_delete_nodepool":
                if arguments.get("name"):
                    cmd_args.extend([arguments["name"]])
                if arguments.get("yes"):
                    cmd_args.extend(["--yes"])

            elif tool_name == "karpenter_events":
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "karpenter_logs":
                if arguments.get("lines"):
                    cmd_args.extend(["--lines", str(arguments["lines"])])

            elif tool_name in ["karpenter_gpu_nodes", "karpenter_resources"]:
                if arguments.get("format"):
                    cmd_args.extend(["--format", arguments["format"]])

            # Triggers tools
            elif tool_name == "triggers_create":
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

            elif tool_name == "triggers_list":
                if arguments.get("type"):
                    cmd_args.extend(["--type", arguments["type"]])
                if arguments.get("status"):
                    cmd_args.extend(["--status", arguments["status"]])

            elif tool_name == "triggers_enable":
                if arguments.get("name"):
                    cmd_args.extend([arguments["name"]])
                if arguments.get("enabled") is not None:
                    cmd_args.append("--enable" if arguments["enabled"] else "--disable")

            elif tool_name == "triggers_execute":
                if arguments.get("name"):
                    cmd_args.extend([arguments["name"]])
                if arguments.get("dry_run"):
                    cmd_args.append("--dry-run")

            elif tool_name == "triggers_history":
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            # Environments tools
            elif tool_name == "environments_create":
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

            elif tool_name == "environments_promote":
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

            elif tool_name == "environments_list":
                if arguments.get("cluster"):
                    cmd_args.extend(["--cluster", arguments["cluster"]])

            elif tool_name == "environments_approve":
                if arguments.get("request_id"):
                    cmd_args.extend([arguments["request_id"]])
                if arguments.get("action"):
                    cmd_args.extend(["--action", arguments["action"]])
                if arguments.get("comment"):
                    cmd_args.extend(["--comment", arguments["comment"]])

            elif tool_name == "environments_history":
                if arguments.get("environment"):
                    cmd_args.extend(["--environment", arguments["environment"]])
                if arguments.get("artifact"):
                    cmd_args.extend(["--artifact", arguments["artifact"]])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            # Lineage tools
            elif tool_name == "lineage_trace":
                if arguments.get("artifact"):
                    cmd_args.extend([arguments["artifact"]])
                if arguments.get("depth"):
                    cmd_args.extend(["--depth", str(arguments["depth"])])
                if arguments.get("direction"):
                    cmd_args.extend(["--direction", arguments["direction"]])

            elif tool_name == "lineage_diff":
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

            elif tool_name == "lineage_export":
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

            elif tool_name == "lineage_checkpoint":
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

            elif tool_name == "lineage_graph":
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

            # Migration tools
            elif tool_name == "migrate_plan":
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

            elif tool_name == "migrate_execute":
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

            elif tool_name == "migrate_status":
                if arguments.get("plan_id"):
                    cmd_args.extend(["--plan", arguments["plan_id"]])
                if arguments.get("show_details") is not None:
                    cmd_args.extend(
                        ["--details" if arguments["show_details"] else "--summary"]
                    )

            elif tool_name == "migrate_rollback":
                if arguments.get("plan_id"):
                    cmd_args.extend(["--plan", arguments["plan_id"]])
                if arguments.get("force"):
                    cmd_args.append("--force")
                    if arguments.get("yes"):
                        cmd_args.append("--yes")

            elif tool_name == "migrate_compare":
                if arguments.get("plan_id"):
                    cmd_args.extend(["--plan", arguments["plan_id"]])
                if arguments.get("metrics"):
                    cmd_args.extend(["--metrics", ",".join(arguments["metrics"])])
                if arguments.get("duration"):
                    cmd_args.extend(["--duration", arguments["duration"]])

            # ── v5.4.0: Langfuse Argument Builders ───────────────────────────────

            elif tool_name == "langfuse_configure":
                cmd_args.extend(
                    [
                        "--public-key",
                        arguments["public_key"],
                        "--secret-key",
                        arguments["secret_key"],
                    ]
                )
                if arguments.get("host"):
                    cmd_args.extend(["--host", arguments["host"]])

            elif tool_name == "langfuse_traces":
                cmd_args.extend(["--format", "json"])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])

            elif tool_name == "langfuse_trace":
                cmd_args.append(arguments["trace_id"])
                cmd_args.extend(["--format", "json"])

            elif tool_name == "langfuse_scores":
                cmd_args.extend(["--format", "json"])
                if arguments.get("trace_id"):
                    cmd_args.extend(["--trace-id", arguments["trace_id"]])
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "langfuse_score":
                cmd_args.extend(
                    [
                        "--trace-id",
                        arguments["trace_id"],
                        "--name",
                        arguments["name"],
                        "--value",
                        str(arguments["value"]),
                    ]
                )
                if arguments.get("observation_id"):
                    cmd_args.extend(["--observation-id", arguments["observation_id"]])
                if arguments.get("comment"):
                    cmd_args.extend(["--comment", arguments["comment"]])

            elif tool_name == "langfuse_datasets":
                cmd_args.extend(["--format", "json"])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "langfuse_export_training_data":
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])
                if arguments.get("name"):
                    cmd_args.extend(["--name", arguments["name"]])
                if arguments.get("min_score") is not None:
                    cmd_args.extend(["--min-score", str(arguments["min_score"])])
                if arguments.get("score_name"):
                    cmd_args.extend(["--score-name", arguments["score_name"]])

            elif tool_name == "langfuse_quality":
                cmd_args.extend(["--format", "json"])
                if arguments.get("score_name"):
                    cmd_args.extend(["--score-name", arguments["score_name"]])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "langfuse_otel_env":
                if arguments.get("project"):
                    cmd_args.extend(["--project", arguments["project"]])

            elif tool_name == "langfuse_k8s":
                if arguments.get("namespace"):
                    cmd_args.extend(["--namespace", arguments["namespace"]])

            # ── v5.4.0: Databricks Argument Builders ─────────────────────────────

            elif tool_name == "databricks_configure":
                cmd_args.extend(
                    ["--host", arguments["host"], "--token", arguments["token"]]
                )

            elif tool_name == "databricks_jobs":
                cmd_args.extend(["--format", "json"])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            elif tool_name == "databricks_run":
                cmd_args.extend([str(arguments["job_id"]), "--format", "json"])

            elif tool_name == "databricks_run_status":
                cmd_args.extend([str(arguments["run_id"]), "--format", "json"])

            elif tool_name in ("databricks_clusters", "databricks_serving_endpoints"):
                cmd_args.extend(["--format", "json"])

            elif tool_name == "databricks_deploy_model":
                cmd_args.extend(
                    [
                        "--endpoint-name",
                        arguments["endpoint_name"],
                        "--model-name",
                        arguments["model_name"],
                        "--format",
                        "json",
                    ]
                )
                if arguments.get("model_version"):
                    cmd_args.extend(["--model-version", arguments["model_version"]])
                if arguments.get("workload_size"):
                    cmd_args.extend(["--workload-size", arguments["workload_size"]])
                if arguments.get("scale_to_zero") is False:
                    cmd_args.append("--no-scale-to-zero")

            elif tool_name == "databricks_query":
                cmd_args.extend(
                    [
                        "--endpoint",
                        arguments["endpoint"],
                        "--prompt",
                        arguments["prompt"],
                        "--format",
                        "json",
                    ]
                )

            elif tool_name in (
                "databricks_mlflow_experiments",
                "databricks_mlflow_models",
            ):
                cmd_args.extend(["--format", "json"])
                if arguments.get("limit"):
                    cmd_args.extend(["--limit", str(arguments["limit"])])

            # ── v5.4.0: vLLM Extended Argument Builders ──────────────────────────

            elif tool_name == "vllm_auto_optimize":
                cmd_args.extend(["--model", arguments["model"]])
                if arguments.get("endpoint"):
                    cmd_args.extend(["--endpoint", arguments["endpoint"]])
                if arguments.get("gpu_count"):
                    cmd_args.extend(["--gpu-count", str(arguments["gpu_count"])])
                if arguments.get("output"):
                    cmd_args.extend(["--output", arguments["output"]])

            elif tool_name == "vllm_analyze":
                cmd_args.extend(["--endpoint", arguments["endpoint"]])
                if arguments.get("duration"):
                    cmd_args.extend(["--duration", str(arguments["duration"])])

            elif tool_name == "vllm_benchmark":
                cmd_args.extend(["--endpoint", arguments["endpoint"]])
                if arguments.get("concurrent"):
                    cmd_args.extend(["--concurrent", str(arguments["concurrent"])])
                if arguments.get("prompt"):
                    cmd_args.extend(["--prompt", arguments["prompt"]])
                if arguments.get("api_key"):
                    cmd_args.extend(["--api-key", arguments["api_key"]])

            # Execute the command (generic fallback for simple tools)
            result = await execute_terradev_command(cmd_args)

            if result["success"]:
                return CallToolResult(
                    content=[TextContent(type="text", text=result["stdout"])]
                )
            else:
                error_msg = result["stderr"] or "Command failed"
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {error_msg}")],
                    isError=True,
                )
        except Exception as e:  # noqa: BLE001
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error: {e}")], isError=True
            )
        finally:
            _concurrent_requests -= 1


@server.list_resources()
async def handle_list_resources() -> ListResourcesResult:
    """List available MCP resources for session-start context and polling."""
    return ListResourcesResult(
        resources=[
            Resource(
                uri="terradev://active_context",
                name="Active Context",
                description="Current Terradev state: running jobs, active instances, spend-to-date, alerts. Read on session start.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://instances",
                name="Active Instances",
                description="Currently provisioned GPU instances across all providers.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://jobs",
                name="Training Jobs",
                description="All training jobs with status, progress, and ETA.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://spend",
                name="Spend Summary",
                description="Cost analytics and spend-to-date across all providers.",
                mimeType="application/json",
            ),
            Resource(
                uri="terradev://alerts",
                name="Alerts",
                description="Active alerts: straggler nodes, budget warnings, drift detected, failed health checks.",
                mimeType="application/json",
            ),
        ]
    )


@server.read_resource()
async def handle_read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read a Terradev resource."""
    uri = str(request.params.uri)

    if uri == "terradev://active_context":
        # Composite: jobs + instances + spend
        jobs = await execute_terradev_command(["train-status", "-f", "json"])
        instances = await execute_terradev_command(["status", "-f", "json"])
        spend = await execute_terradev_command(
            ["analytics", "--days", "7", "-f", "json"]
        )
        context = {
            "jobs": jobs["stdout"] if jobs["success"] else None,
            "instances": instances["stdout"] if instances["success"] else None,
            "spend_7d": spend["stdout"] if spend["success"] else None,
            "suggest_action": "Call active_context tool for formatted recommendations.",
        }
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps(context, indent=2),
                )
            ]
        )

    elif uri == "terradev://instances":
        result = await execute_terradev_command(["status", "-f", "json"])
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://jobs":
        result = await execute_terradev_command(["train-status", "-f", "json"])
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://spend":
        result = await execute_terradev_command(
            ["analytics", "--days", "30", "-f", "json"]
        )
        text = (
            result["stdout"]
            if result["success"]
            else json.dumps({"error": result["stderr"]})
        )
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    elif uri == "terradev://alerts":
        # Aggregate alerts from multiple sources
        alerts = []
        # Check for straggler nodes via monitor
        monitor = await execute_terradev_command(
            ["monitor", "--check-stragglers", "-f", "json"]
        )
        if monitor["success"] and monitor["stdout"].strip():
            alerts.append({"type": "straggler", "data": monitor["stdout"]})
        # Check cost budget
        cost = await execute_terradev_command(["cost-scaler-status", "-f", "json"])
        if cost["success"] and "exceed" in cost["stdout"].lower():
            alerts.append({"type": "budget_warning", "data": cost["stdout"]})
        # Check drift
        drift = await execute_terradev_command(
            ["manifests", "--check-drift", "-f", "json"]
        )
        if drift["success"] and "drift" in drift["stdout"].lower():
            alerts.append({"type": "drift_detected", "data": drift["stdout"]})
        text = json.dumps({"alerts": alerts, "count": len(alerts)}, indent=2)
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="application/json", text=text)
            ]
        )

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps({"error": f"Unknown resource: {uri}"}),
            )
        ]
    )


@server.list_prompts()
async def handle_list_prompts() -> ListPromptsResult:
    """List available prompts"""
    return ListPromptsResult(prompts=[])


@server.get_prompt()
async def handle_get_prompt(request: GetPromptRequest) -> GetPromptResult:
    """Get a prompt"""
    return GetPromptResult(description="", messages=[])


# ---------------------------------------------------------------------------
# OAuth 2.0 PKCE auth for Claude.ai Connectors
# ---------------------------------------------------------------------------

TERRADEV_MCP_BEARER_TOKEN = os.getenv("TERRADEV_MCP_BEARER_TOKEN", "")

# Comma-separated list of exact allowed redirect URIs for OAuth.
# Example: "https://claude.ai/oauth/callback,http://localhost:3000/callback"
# If unset, only localhost/127.0.0.1 URIs are permitted (safe local-dev default).
_ALLOWED_REDIRECT_URIS_RAW = os.getenv("TERRADEV_MCP_ALLOWED_REDIRECT_URIS", "")
_ALLOWED_REDIRECT_URIS: set = (
    {u.strip() for u in _ALLOWED_REDIRECT_URIS_RAW.split(",") if u.strip()}
    if _ALLOWED_REDIRECT_URIS_RAW
    else set()
)


def _is_redirect_uri_allowed(uri: str) -> bool:
    """Return True only if uri is on the explicit allowlist, or is a localhost
    URI when no explicit allowlist is configured."""
    if not uri:
        return False
    if _ALLOWED_REDIRECT_URIS:
        return uri in _ALLOWED_REDIRECT_URIS
    # No allowlist configured — permit localhost / 127.0.0.1 only
    from urllib.parse import urlparse
    try:
        parsed = urlparse(uri)
        return parsed.hostname in ("localhost", "127.0.0.1", "::1")
    except Exception:  # noqa: BLE001
        return False

# In-memory stores (single-instance server)
_auth_codes: Dict[str, Dict[str, Any]] = (
    {}
)  # code -> {client_id, code_challenge, redirect_uri, expires}
_access_tokens: Dict[str, Dict[str, Any]] = {}  # token -> {client_id, expires}


def _cleanup_expired():
    """Remove expired auth codes and tokens."""
    now = time.time()
    for store in (_auth_codes, _access_tokens):
        expired = [k for k, v in store.items() if v.get("expires", 0) < now]
        for k in expired:
            del store[k]


# ---------------------------------------------------------------------------
# OAuth endpoint handlers (added as Starlette routes)
# ---------------------------------------------------------------------------


async def oauth_authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 — OAuth Authorization Server Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": base + "/authorize",
            "token_endpoint": base + "/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
        }
    )


async def oauth_protected_resource(request: Request) -> JSONResponse:
    """RFC 9728 — OAuth Protected Resource Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": base,
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        }
    )


async def oauth_authorize(request: Request) -> Response:
    """OAuth 2.0 Authorization Endpoint — auto-approves if client_id matches our token."""
    from starlette.responses import RedirectResponse

    params = dict(request.query_params)
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "")
    state = params.get("state", "")

    logger.info(
        "OAuth authorize: client_id=%s... redirect=%s", client_id[:16], redirect_uri
    )

    # Validate redirect_uri against allowlist before doing anything else (open-redirect prevention)
    if not _is_redirect_uri_allowed(redirect_uri):
        logger.warning("OAuth authorize rejected: redirect_uri not in allowlist: %s", redirect_uri)
        return JSONResponse(
            {"error": "invalid_request", "error_description": "redirect_uri not allowed"},
            status_code=400,
        )

    # Validate client_id matches our configured token
    if TERRADEV_MCP_BEARER_TOKEN and client_id != TERRADEV_MCP_BEARER_TOKEN:
        logger.warning("OAuth authorize rejected: bad client_id")
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if code_challenge_method and code_challenge_method != "S256":
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Only S256 supported"},
            status_code=400,
        )

    # Generate authorization code
    _cleanup_expired()
    code = secrets.token_urlsafe(48)
    _auth_codes[code] = {
        "client_id": client_id,
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "expires": time.time() + 300,  # 5 min
    }

    # Redirect back to Claude.ai with the code
    sep = "&" if "?" in redirect_uri else "?"
    redirect = redirect_uri + sep + urlencode({"code": code, "state": state})
    logger.info("OAuth authorize: issuing code, redirecting to %s", redirect_uri)
    return RedirectResponse(url=redirect, status_code=302)


async def oauth_token(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Endpoint — exchanges auth code for access token (PKCE)."""
    if request.method == "GET":
        return JSONResponse({"error": "method_not_allowed"}, status_code=405)

    try:
        body = await request.form()
    except Exception:  # noqa: BLE001
        body = {}
    body = dict(body)

    grant_type = body.get("grant_type", "")
    code = body.get("code", "")
    code_verifier = body.get("code_verifier", "")
    client_id = body.get("client_id", "")
    body.get("redirect_uri", "")

    logger.info(
        "OAuth token: grant_type=%s client_id=%s...", grant_type, (client_id or "")[:16]
    )

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    _cleanup_expired()
    auth_data = _auth_codes.pop(code, None)
    if not auth_data:
        logger.warning("OAuth token: invalid or expired code")
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    # Verify PKCE code_challenge
    if auth_data.get("code_challenge") and code_verifier:
        digest = hashlib.sha256(code_verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if expected != auth_data["code_challenge"]:
            logger.warning("OAuth token: PKCE verification failed")
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "PKCE verification failed",
                },
                status_code=400,
            )

    # Issue access token
    access_token = secrets.token_urlsafe(48)
    _access_tokens[access_token] = {
        "client_id": auth_data["client_id"],
        "expires": time.time() + 86400,  # 24 hours
    }

    logger.info(
        "OAuth token: issued access token for client_id=%s...",
        auth_data["client_id"][:16],
    )
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 86400,
        }
    )


# ---------------------------------------------------------------------------
# ASGI auth middleware (validates Bearer tokens from OAuth flow)
# ---------------------------------------------------------------------------

# Paths that don't require auth
_PUBLIC_PATHS = frozenset(
    [
        "/health",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/sse",
        "/authorize",
        "/token",
    ]
)


class OAuthBearerMiddleware:
    """Pure ASGI middleware — validates OAuth Bearer tokens on protected routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "?")

        # Public routes pass through
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract auth header
        headers_raw = scope.get("headers", [])
        header_dict = {k.decode(): v.decode() for k, v in headers_raw}
        auth = header_dict.get("authorization", "")

        logger.info(
            "Request: %s %s host=%s auth=%s",
            method,
            path,
            header_dict.get("host", "-"),
            auth[:30] + "..." if auth else "-",
        )

        if not auth.startswith("Bearer "):
            logger.warning("Auth rejected for %s %s (no Bearer token)", method, path)
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        token = auth[7:]  # strip "Bearer "

        # Accept the raw configured token OR any valid OAuth-issued token
        _cleanup_expired()
        if token == TERRADEV_MCP_BEARER_TOKEN:
            await self.app(scope, receive, send)
            return

        if token in _access_tokens:
            await self.app(scope, receive, send)
            return

        logger.warning("Auth rejected for %s %s (invalid token)", method, path)
        response = JSONResponse({"error": "unauthorized"}, status_code=401)
        await response(scope, receive, send)


# ---------------------------------------------------------------------------
# SSE app factory
# ---------------------------------------------------------------------------


def create_sse_app() -> "Starlette":
    """Build the Starlette app that exposes the MCP server over SSE."""
    if Starlette is None:
        print(
            "Error: starlette/uvicorn not installed. "
            "Install with: pip install 'mcp[cli]' starlette uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "terradev-mcp.terradev.cloud",
            "localhost:8090",
            "127.0.0.1:8090",
        ],
        allowed_origins=[
            "https://claude.ai",
            "https://www.claude.ai",
            "https://terradev-mcp.terradev.cloud",
        ],
    )
    sse_transport = SseServerTransport("/messages", security_settings=security_settings)

    async def handle_messages(request: Request) -> None:
        await sse_transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "server": "terradev-mcp", "version": "2.0.1"}
        )

    # SSE handler wraps the MCP server
    class SseHandler:
        def __init__(self):
            self._server = server

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return

            # SSE endpoint only accepts GET requests
            if scope["method"] != "GET":
                from starlette.responses import Response

                response = Response(
                    "Method Not Allowed - SSE endpoint only accepts GET",
                    status_code=405,
                )
                await response(scope, receive, send)
                return

            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await self._server.run(
                    streams[0],
                    streams[1],
                    InitializationOptions(
                        server_name="terradev-mcp",
                        server_version="2.0.1",
                        capabilities=self._server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )

    sse_handler = SseHandler()

    inner_app = Starlette(
        debug=False,
        routes=[
            # OAuth 2.0 endpoints (public — handled before auth middleware)
            Route(
                "/.well-known/oauth-authorization-server",
                endpoint=oauth_authorization_server_metadata,
            ),
            Route(
                "/.well-known/oauth-protected-resource",
                endpoint=oauth_protected_resource,
            ),
            Route(
                "/.well-known/oauth-protected-resource/sse",
                endpoint=oauth_protected_resource,
            ),
            Route("/authorize", endpoint=oauth_authorize),
            Route("/token", endpoint=oauth_token, methods=["POST"]),
            # MCP endpoints
            Route("/health", endpoint=health),
            Route("/sse", endpoint=sse_handler),
            Route("/messages/{path:path}", endpoint=handle_messages, methods=["POST"]),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    app = OAuthBearerMiddleware(inner_app)
    return app


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_stdio():
    """Run in stdio mode (Claude Code / local)."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="terradev-mcp",
                    server_version="2.0.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def main():
    """Main entry point — supports both stdio and SSE transports."""
    parser = argparse.ArgumentParser(description="Terradev MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default, for Claude Code) or sse (remote, for Claude.ai Connectors)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8080, help="SSE port (default: 8080)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Check if terradev is installed
    if not check_terradev_installation():
        logger.warning(
            "terradev CLI not found. Tools will fail until installed: pip install terradev-cli"
        )

    if not os.getenv("TERRADEV_RUNPOD_KEY"):
        logger.warning(
            "TERRADEV_RUNPOD_KEY not set. Some functionality may be limited."
        )

    if args.transport == "stdio":
        asyncio.run(run_stdio())
    else:
        if not TERRADEV_MCP_BEARER_TOKEN:
            logger.warning(
                "TERRADEV_MCP_BEARER_TOKEN is not set — SSE endpoint is UNAUTHENTICATED. "
                "Set this env var in production."
            )
        app = create_sse_app()
        logger.info("Starting Terradev MCP SSE server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)


def run_server(transport: str = "stdio", host: Optional[str] = None, port: Optional[int] = None):
    """Start the Terradev MCP server (used by `terradev mcp serve`)."""
    if not MCP_AVAILABLE:
        print(
            "Error: mcp package not installed. Install with: pip install 'mcp[cli]'",
            file=sys.stderr,
        )
        sys.exit(1)

    argv = [sys.argv[0] if sys.argv else "terradev_mcp"]
    if transport != "stdio":
        argv.extend(["--transport", transport])
    if host is not None:
        argv.extend(["--host", host])
    if port is not None:
        argv.extend(["--port", str(port)])

    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv


def list_tools():
    """Print all registered Terradev MCP tools (used by `terradev mcp list-tools`)."""
    if not MCP_AVAILABLE:
        print(
            "Error: mcp package not installed. Install with: pip install 'mcp[cli]'",
            file=sys.stderr,
        )
        return

    _build_all_tools()
    print(f"Terradev MCP tools ({len(_ALL_TOOLS)}):\n")
    for tool in _ALL_TOOLS:
        print(f"  - {tool.name}: {tool.description}")


def install_config(client: str):
    """Install the MCP client configuration for the requested client."""
    import platform

    home = os.path.expanduser("~")
    configs = {
        "claude-desktop": (
            os.path.join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json")
            if platform.system() == "Darwin"
            else os.path.join(home, ".config", "claude", "claude_desktop_config.json")
        ),
        "cursor": os.path.join(home, ".cursor", "mcp.json"),
        "windsurf": os.path.join(home, ".config", "windsurf", "mcp_config.json"),
        "continue": os.path.join(home, ".continue", "mcp_config.json"),
        "cline": os.path.join(home, ".config", "cline", "mcp_config.json"),
    }

    if client not in configs:
        print(f"Error: unsupported MCP client '{client}'", file=sys.stderr)
        print(f"Supported clients: {', '.join(configs.keys())}", file=sys.stderr)
        sys.exit(1)

    path = configs[client]
    terradev_entry = {
        "command": sys.executable,
        "args": ["-m", "terradev_cli", "mcp", "serve"],
    }

    existing = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["terradev"] = terradev_entry

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Installed Terradev MCP config for {client} at {path}")


# ── Exposed command map used by tests and external callers ───────────────────
COMMAND_MAP: Dict[str, Any] = {}

try:
    _build_all_tools()
    if _ALL_TOOLS:
        COMMAND_MAP = {tool.name: None for tool in _ALL_TOOLS}
    if NEW_COMMAND_MAP:
        COMMAND_MAP.update(NEW_COMMAND_MAP)
except Exception:  # noqa: BLE001
    # Tool build may fail during partial imports; callers should still be able
    # to import COMMAND_MAP without raising.
    pass


if __name__ == "__main__":
    main()
