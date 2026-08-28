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

        Tool(name='orchestrator_start', description='Start the model orchestrator for multi-model GPU sharing with eviction policies.', inputSchema={'type': 'object', 'properties': {'gpu_id': {'type': 'integer', 'description': 'GPU ID', 'default': 0}, 'memory_gb': {'type': 'number', 'description': 'Total GPU memory in GB', 'default': 80.0}, 'policy': {'type': 'string', 'description': 'Scaling policy', 'enum': ['billing_optimized', 'latency_optimized', 'hybrid'], 'default': 'billing_optimized'}}}),
        Tool(name='orchestrator_register', description='Register a model with the orchestrator.', inputSchema={'type': 'object', 'properties': {'model_id': {'type': 'string', 'description': 'Model identifier'}, 'model_path': {'type': 'string', 'description': 'Path to model weights'}, 'framework': {'type': 'string', 'description': 'Framework', 'enum': ['pytorch', 'vllm', 'sglang'], 'default': 'pytorch'}}, 'required': ['model_id', 'model_path']}),
        Tool(name='orchestrator_load', description='Load a model into GPU memory.', inputSchema={'type': 'object', 'properties': {'model_id': {'type': 'string', 'description': 'Model to load'}, 'force': {'type': 'boolean', 'description': 'Force loading even if memory is full', 'default': False}}, 'required': ['model_id']}),
        Tool(name='orchestrator_evict', description='Evict a model from GPU memory.', inputSchema={'type': 'object', 'properties': {'model_id': {'type': 'string', 'description': 'Model to evict'}}, 'required': ['model_id']}),
        Tool(name='orchestrator_status', description='Get orchestrator and model status including GPU memory utilization.', inputSchema={'type': 'object', 'properties': {'model_id': {'type': 'string', 'description': 'Get details for specific model (optional)'}}}),
        Tool(name='orchestrator_infer', description='Test inference with a model via the orchestrator.', inputSchema={'type': 'object', 'properties': {'model_id': {'type': 'string', 'description': 'Model to run inference on'}}, 'required': ['model_id']}),
        Tool(name='warm_pool_start', description='Start the warm pool manager for intelligent model pre-warming. 5 strategies: traffic_based, time_based, priority_based, cost_optimized, latency_optimized.', inputSchema={'type': 'object', 'properties': {'strategy': {'type': 'string', 'description': 'Warm pool strategy', 'enum': ['traffic_based', 'time_based', 'priority_based', 'cost_optimized', 'latency_optimized'], 'default': 'traffic_based'}, 'max_warm': {'type': 'integer', 'description': 'Max models to keep warm', 'default': 10}, 'min_warm': {'type': 'integer', 'description': 'Min models to keep warm', 'default': 3}}}),
        Tool(name='warm_pool_status', description='Get warm pool status: hit rate, cold starts, memory saved, cost saved.', inputSchema={'type': 'object', 'properties': {}}),
    ]
