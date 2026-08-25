"""MCP tool schema definitions."""

from typing import Any, List

try:
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None

TOOLS = []

if Tool is not None:
    TOOLS = [

        Tool(name='provision_gpu', description='Provision GPU instances for optimal parallel efficiency', inputSchema={'type': 'object', 'properties': {'gpu_type': {'type': 'string', 'description': 'GPU type to provision', 'enum': ['H100', 'A100', 'A10G', 'L40S', 'L4', 'T4', 'RTX4090', 'RTX3090', 'V100']}, 'count': {'type': 'integer', 'description': 'Number of GPUs to provision', 'minimum': 1, 'default': 1}, 'providers': {'type': 'array', 'description': 'Cloud providers for parallel distribution', 'items': {'type': 'string', 'enum': ['runpod', 'vastai', 'aws', 'gcp', 'azure', 'tensordock', 'crusoe', 'digitalocean', 'hyperstack', 'ovhcloud', 'siliconflow', 'latitude', 'huggingface', 'baseten', 'inferx', 'yottalabs', 'e2enetworks']}}, 'max_price': {'type': 'number', 'description': 'Maximum price per hour', 'minimum': 0}, 'plan_only': {'type': 'boolean', 'description': 'Generate plan without applying', 'default': False}, 'state_file': {'type': 'string', 'description': 'State file path (optional)', 'default': None}}, 'required': ['gpu_type']}),
        Tool(name='terraform_plan', description='Generate execution plan for GPU provisioning', inputSchema={'type': 'object', 'properties': {'config_dir': {'type': 'string', 'description': 'Directory containing configuration'}, 'var_file': {'type': 'string', 'description': 'Variables file path (optional)'}, 'destroy': {'type': 'boolean', 'description': 'Generate destroy plan', 'default': False}}, 'required': ['config_dir']}),
        Tool(name='terraform_apply', description='Apply configuration for GPU provisioning', inputSchema={'type': 'object', 'properties': {'config_dir': {'type': 'string', 'description': 'Directory containing configuration'}, 'plan_file': {'type': 'string', 'description': 'Plan file to apply (optional)'}, 'var_file': {'type': 'string', 'description': 'Variables file path (optional)'}, 'auto_approve': {'type': 'boolean', 'description': 'Auto-approve the apply', 'default': True}}, 'required': ['config_dir']}),
        Tool(name='terraform_destroy', description='Destroy managed GPU infrastructure', inputSchema={'type': 'object', 'properties': {'config_dir': {'type': 'string', 'description': 'Directory containing configuration'}, 'var_file': {'type': 'string', 'description': 'Variables file path (optional)'}, 'auto_approve': {'type': 'boolean', 'description': 'Auto-approve the destroy', 'default': True}}, 'required': ['config_dir']}),
        Tool(name='terraform_status', description='Fast status query using state', inputSchema={'type': 'object', 'properties': {'config_dir': {'type': 'string', 'description': 'Directory containing configuration'}, 'show_outputs': {'type': 'boolean', 'description': 'Show outputs', 'default': True}}, 'required': ['config_dir']}),
        Tool(name='preflight_report', description='Generate full preflight validation report with pass/warn/fail per check. Covers GPU drivers, CUDA, NCCL, RDMA, network, disk, and Docker.', inputSchema={'type': 'object', 'properties': {'nodes': {'type': 'array', 'description': 'Node IPs', 'items': {'type': 'string'}}, 'from_provision': {'type': 'string', 'description': "Resolve nodes from provision ('latest' or group ID)"}, 'checks': {'type': 'array', 'description': 'Specific checks to run', 'items': {'type': 'string'}}}}),
        Tool(name='preflight_gpu_check', description='GPU-specific preflight validation: NVIDIA drivers, CUDA version, GPU count, NCCL, NVLink topology, NCU stall-signature profiling, and adversarial config verification (V1-V3).', inputSchema={'type': 'object', 'properties': {'nodes': {'type': 'array', 'description': 'Node IPs', 'items': {'type': 'string'}}, 'from_provision': {'type': 'string', 'description': 'Resolve nodes from provision'}, 'tensor_parallel_size': {'type': 'integer', 'description': 'TP size for V1 adversarial check (vs GPU count)'}, 'model_precision': {'type': 'string', 'description': 'Model precision (fp8, bf16, fp16) for V2 FP8 wall check'}, 'fp8_quant_scheme': {'type': 'string', 'description': 'FP8 quant scheme (per_tensor, per_token) for Blackwell K-slab check'}, 'gpu_arch': {'type': 'string', 'description': 'GPU architecture string (e.g. blackwell, hopper) for precision wall detection'}, 'max_batch_size': {'type': 'integer', 'description': 'Max batch size for V3 launch-overhead dominance check'}}}),
        Tool(name='preflight_network_check', description='Network-specific preflight validation: RDMA availability, InfiniBand status, inter-node bandwidth, latency matrix, firewall rules.', inputSchema={'type': 'object', 'properties': {'nodes': {'type': 'array', 'description': 'Node IPs', 'items': {'type': 'string'}}, 'from_provision': {'type': 'string', 'description': 'Resolve nodes from provision'}}}),
    ]
