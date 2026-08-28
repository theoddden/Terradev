"""Terradev command execution helpers for the MCP server."""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import re as _re
import secrets
import shutil
import subprocess
import sys
import time
import platform
from typing import Any, Dict, List, Optional

logger = logging.getLogger("terradev-mcp")


def check_terradev_installation():
    try:
        cmd = _terradev_command() + ["--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

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

def _load_datadog_creds() -> Dict[str, str]:
    """Load Datadog credentials from ~/.terradev/credentials.json."""
    creds_path = os.path.join(os.path.expanduser("~"), ".terradev", "credentials.json")
    if os.path.exists(creds_path):
        with open(creds_path, "r") as f:
            all_creds = json.load(f)
        return {k: v for k, v in all_creds.items() if k.startswith("datadog_")}
    return {}

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

def _terradev_command() -> List[str]:
    """Return the command prefix to invoke the Terradev CLI.

    Prefers the installed `terradev` entry-point script. Falls back to
    `python -m terradev_cli` so the MCP server works in editable installs
    and during development without needing the script on PATH.
    """
    if shutil.which("terradev"):
        return ["terradev"]
    return [sys.executable, "-m", "terradev_cli"]

def build_cli_args(arguments: Dict[str, Any], cmd_args: List[str], positional: List[str]) -> List[str]:
    """Convert MCP tool arguments into a CLI argument list.

    - Positional argument names are appended in the order given.
    - Other arguments are appended as `--<key> <value>` (underscores become
      hyphens). Boolean True becomes a bare flag; False/None are ignored.
    """
    extra: List[str] = []
    for name in positional:
        if name in arguments:
            value = arguments[name]
            if isinstance(value, list):
                extra.extend(str(v) for v in value)
            else:
                extra.append(str(value))
    for key, value in arguments.items():
        if key in positional or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            extra.append(flag)
        elif isinstance(value, list):
            for item in value:
                extra.extend([flag, str(item)])
        else:
            extra.extend([flag, str(value)])
    return cmd_args + extra


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
            "aws",
            "gcp",
            "azure",
            "tensordock",
            "crusoe",
            "digitalocean",
            "hyperstack",
            "siliconflow",
            "latitude",
            "e2enetworks",
            "yottalabs"
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
        "aws",
        "gcp",
        "azure",
        "tensordock",
        "crusoe",
        "digitalocean",
        "hyperstack",
        "siliconflow",
        "latitude",
        "e2enetworks",
        "yottalabs"
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
