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
        Tool(name='train_unsloth_run', description='Run an Unsloth local model server.', inputSchema={'type': 'object', 'properties': {'model': {'description': 'Model to serve (e.g. unsloth/Llama-3.1-8B)', 'type': 'string'}, 'host': {'description': 'Server host', 'type': 'string', 'default': '127.0.0.1'}, 'port': {'description': 'Server port', 'type': 'integer', 'default': 8888}, 'enable_tools': {'description': 'Enable/disable tool use', 'type': 'boolean', 'default': False}, 'no_cloudflare': {'description': 'Do not use Cloudflare tunnel', 'type': 'boolean', 'default': False}, 'gguf_variant': {'description': 'Preferred GGUF quantization variant', 'type': 'string'}, 'context_length': {'description': 'Maximum context length', 'type': 'integer'}, 'no_load_in_4bit': {'description': 'Disable 4-bit loading', 'type': 'boolean', 'default': False}, 'tensor_parallel': {'description': 'Tensor parallel size', 'type': 'integer'}, 'pid_file': {'description': 'File to store the server PID', 'type': 'string', 'default': '.unsloth-run.pid'}}, 'required': ['model']}),
        Tool(name='train_unsloth_start', description="Start a coding agent backed by Unsloth's local model server.", inputSchema={'type': 'object', 'properties': {'agent': {'description': 'agent', 'type': 'string'}, 'model': {'description': 'Model to load and serve (e.g. unsloth/Llama-3.1-8B)', 'type': 'string'}, 'host': {'description': 'Server host', 'type': 'string', 'default': '127.0.0.1'}, 'port': {'description': 'Server port', 'type': 'integer', 'default': 8888}, 'enable_tools': {'description': 'Enable/disable tool use', 'type': 'boolean', 'default': True}, 'no_cloudflare': {'description': 'Do not use Cloudflare tunnel', 'type': 'boolean', 'default': False}, 'gguf_variant': {'description': 'Preferred GGUF quantization variant', 'type': 'string'}, 'context_length': {'description': 'Maximum context length', 'type': 'integer'}, 'no_load_in_4bit': {'description': 'Disable 4-bit loading', 'type': 'boolean', 'default': False}, 'tensor_parallel': {'description': 'Tensor parallel size', 'type': 'integer'}, 'project': {'description': 'Project directory', 'type': 'string', 'default': '.'}, 'background': {'description': 'Run in background instead of foreground', 'type': 'boolean', 'default': False}}, 'required': ['agent']}),
        Tool(name='train_unsloth_stop', description='Stop a running Unsloth server started with `unsloth run`.', inputSchema={'type': 'object', 'properties': {'pid_file': {'description': 'PID file written by unsloth run', 'type': 'string', 'default': '.unsloth-run.pid'}, 'signal': {'description': 'Signal to send', 'type': 'string', 'default': 'SIGTERM'}}}),
    ]
