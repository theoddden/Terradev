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
        Tool(name='ml_vllm_lora_link', description='Load the active registry version of an adapter onto a vLLM server.', inputSchema={'type': 'object', 'properties': {'endpoint': {'description': 'vLLM endpoint', 'type': 'string'}, 'name': {'description': 'Registered adapter name', 'type': 'string'}, 'api_key': {'description': 'vLLM API key', 'type': 'string'}}, 'required': ['endpoint', 'name']}),
        Tool(name='ml_vllm_lora_list', description='List LoRA adapters currently loaded on a vLLM server.', inputSchema={'type': 'object', 'properties': {'endpoint': {'description': 'vLLM endpoint', 'type': 'string'}, 'api_key': {'description': 'vLLM API key', 'type': 'string'}}, 'required': ['endpoint']}),
        Tool(name='ml_vllm_lora_load', description='Hot-load a LoRA adapter onto a running vLLM server.', inputSchema={'type': 'object', 'properties': {'endpoint': {'description': 'vLLM endpoint', 'type': 'string'}, 'name': {'description': 'Adapter name', 'type': 'string'}, 'path': {'description': 'Local path to adapter weights', 'type': 'string'}, 'api_key': {'description': 'vLLM API key', 'type': 'string'}, 'register': {'description': 'Register in LoRA registry before loading', 'type': 'boolean', 'default': False}, 'base_model': {'description': 'Base model name (required with --register)', 'type': 'string'}, 'rank': {'description': 'LoRA rank (default: 64)', 'type': 'integer', 'default': 64}}, 'required': ['endpoint', 'name', 'path']}),
        Tool(name='ml_vllm_lora_sync', description='Synchronize an adapter from the registry across multiple vLLM replicas.', inputSchema={'type': 'object', 'properties': {'name': {'description': 'Registered adapter name', 'type': 'string'}, 'replicas': {'description': 'Comma-separated host:port list', 'type': 'string'}}, 'required': ['name', 'replicas']}),
        Tool(name='ml_vllm_lora_unload', description='Hot-unload a LoRA adapter from a running vLLM server.', inputSchema={'type': 'object', 'properties': {'endpoint': {'description': 'vLLM endpoint', 'type': 'string'}, 'name': {'description': 'Adapter name to unload', 'type': 'string'}, 'api_key': {'description': 'vLLM API key', 'type': 'string'}}, 'required': ['endpoint', 'name']}),
    ]
