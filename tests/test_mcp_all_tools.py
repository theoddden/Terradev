"""MCP tool dispatch smoke test — exercises every registered tool once.

This is a brute-force coverage driver for the giant `handle_call_tool` dispatch
function.  It does not validate the correctness of output text, only that the
server returns a `CallToolResult` for every tool without unhandled exceptions.
"""
import asyncio
import inspect
import json
import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")
from mcp.types import CallToolRequest


class _FakeResponse:
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status = status
        self._json = json_data if json_data is not None else {"data": [], "result": {}}
        self._text = text_data

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class _FakeAiohttpSession:
    """Minimal aiohttp.ClientSession replacement."""

    def __init__(self, *args, headers=None, **kwargs):
        self.headers = headers or {}
        self.closed = False

    def _make(self, **kwargs):
        return _FakeRequestContext(_FakeResponse())

    def request(self, method, url, **kwargs):
        return self._make()

    def get(self, url, **kwargs):
        return self._make()

    def post(self, url, **kwargs):
        return self._make()

    def put(self, url, **kwargs):
        return self._make()

    def delete(self, url, **kwargs):
        return self._make()

    def patch(self, url, **kwargs):
        return self._make()

    async def close(self):
        self.closed = True


# A kitchen-sink arguments dict that satisfies most explicit `arguments[...]`
# accesses inside `terradev_cli.mcp.server.handle_call_tool`.
GENERIC_ARGUMENTS: Dict[str, Any] = {
    "gpu_type": "H100",
    "count": 1,
    "providers": ["runpod", "vastai"],
    "max_price": 5.0,
    "plan_only": False,
    "local_first": True,
    "quick": True,
    "instance_id": "i-12345",
    "action": "stop",
    "live": True,
    "config_dir": "/tmp",
    "plan_file": "tfplan",
    "var_file": "vars.tfvars",
    "auto_approve": True,
    "cluster_name": "test-cluster",
    "node_count": 1,
    "release_name": "test-release",
    "chart": "test-chart",
    "values_file": "values.yaml",
    "repo": "test-repo",
    "branch": "main",
    "path": "/tmp",
    "command": "echo hello",
    "namespace": "default",
    "name": "test",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "model_name": "test-model",
    "model_path": "/tmp/model",
    "model_version": "1",
    "model_id": "model-123",
    "endpoint": "http://localhost:8000",
    "job_id": "job-123",
    "run_id": "run-456",
    "project": "test-project",
    "dataset": "test-dataset",
    "query": "SELECT 1",
    "sql": "SELECT 1",
    "collection": "test-collection",
    "trace_id": "trace-123",
    "vector": [0.1, 0.2, 0.3],
    "points": [],
    "limit": 10,
    "filter": "span_kind == 'RETRIEVER'",
    "image": "test-image",
    "tag": "latest",
    "workload_size": "small",
    "scale_to_zero": False,
    "output": "json",
    "duration": 60,
    "concurrent": 5,
    "prompt": "hello",
    "api_key": "test-api-key",
    "api_url": "http://localhost",
    "workspace": "test-workspace",
    "url": "http://localhost:6333",
    "host": "localhost",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "tracking_uri": "http://localhost:5000",
    "experiment_id": "1",
    "run_name": "test-run",
    "stage_name": "test-stage",
    "metric_key": "accuracy",
    "value": 0.95,
    "step": 1,
    "algorithm": "adam",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 1,
    "framework": "pytorch",
    "env": "dev",
    "region": "us-east-1",
    "provider": "aws",
    "budget": 10.0,
    "days": 7,
    "hours": 1,
    "start": "2024-01-01",
    "end": "2024-01-02",
    "instance_type": "t2.micro",
    "disk_size": 100,
    "network": "test-net",
    "subnet": "test-subnet",
    "security_group": "sg-123",
    "key_name": "test-key",
    "user": "test-user",
    "password": "test-password",
    "token": "test-token",
    "secret": "test-secret",
    "json": "{}",
    "yaml": "a: 1",
    "timeout": 10,
    "retries": 3,
    "interval": 5,
    "threshold": 0.5,
    "metric": "loss",
    "force": False,
    "dry_run": False,
    "all": False,
    "follow": False,
    "tail": 100,
    "since": "1h",
    "until": "now",
    "format": "json",
    "source": "src",
    "target": "tgt",
    "mapping": {},
    "schema": {},
    "fields": [],
    "headers": {},
    "body": "{}",
    "method": "GET",
    "verify": True,
    "ca_bundle": "/tmp/ca",
    "proxy": "http://proxy",
    "no_proxy": "localhost",
    "serving_config": {},
    "enable_expert_parallel": False,
    "enable_eplb": False,
    "enable_dbo": False,
    "trust_remote_code": False,
    "dashboard_enabled": False,
    "tracing_enabled": False,
    "metrics_enabled": False,
    "deployment_enabled": False,
    "observability_enabled": False,
    "lmcache_enabled": False,
    "disaggregation_enabled": False,
    "prefill_replicas": 1,
    "decode_replicas": 1,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 1,
    "block_size": 16,
    "enable_prefix_caching": True,
    "default_priority": 1,
    "workload_type": "interactive",
    "hardware_profile": "balanced",
    "attention_backend": "flashinfer",
    "speculative_algorithm": "none",
    "schedule_policy": "fcfs",
    "deep_ep_mode": "off",
    "quantization": "none",
    "dist_backend": "nccl",
    "type": "test",
    "port": 8000,
    "tp_size": 1,
    "dp_size": 1,
    "mem_fraction_static": 0.85,
}

_SENTINEL = object()


def _value_for_property(name: str, prop: Dict[str, Any]) -> Any:
    """Return a sensible default for one JSON Schema property."""
    enum = prop.get("enum")
    prop_type = prop.get("type")
    if isinstance(prop_type, list):
        prop_type = next((t for t in prop_type if t != "null"), "string")

    generic = GENERIC_ARGUMENTS.get(name, _SENTINEL)

    # Prefer generic when it matches the declared enum.
    if enum is not None and generic is not _SENTINEL:
        if generic in enum:
            return generic
        return enum[0]

    if generic is not _SENTINEL:
        if prop_type == "array":
            return generic if isinstance(generic, list) else [generic]
        if prop_type == "string":
            if isinstance(generic, str):
                return generic
            if isinstance(generic, (list, tuple)):
                return ",".join(str(v) for v in generic)
            return str(generic)
        if prop_type == "integer":
            if isinstance(generic, bool):
                return int(generic)
            try:
                return int(generic)
            except (ValueError, TypeError):
                return 1
        if prop_type == "number":
            try:
                return float(generic)
            except (ValueError, TypeError):
                return 1.0
        if prop_type == "boolean":
            return bool(generic)
        if prop_type == "object":
            return generic if isinstance(generic, dict) else {}
        return generic

    if enum is not None:
        return enum[0]

    if prop_type == "string":
        if name in ("endpoint", "api_url", "url"):
            return "http://localhost:8000"
        if name in ("host",):
            return "localhost"
        if name.endswith(("_dir", "_path")) or name in ("path", "config_dir", "plan_file", "var_file"):
            return "/tmp"
        if name in ("command",):
            return "echo hello"
        if name in ("query", "sql"):
            return "SELECT 1"
        return "test"

    if prop_type == "integer":
        return 1
    if prop_type == "number":
        return 1.0
    if prop_type == "boolean":
        return True
    if prop_type == "array":
        items = prop.get("items", {})
        if isinstance(items, dict) and "enum" in items:
            return [items["enum"][0]]
        return []
    if prop_type == "object":
        return _args_from_schema(prop)
    return None


def _args_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build an arguments dict that satisfies a tool's inputSchema."""
    args: Dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        args[name] = _value_for_property(name, prop)
    return args


def _mock_stdout():
    return json.dumps({
        "instances": [],
        "jobs": [],
        "data": [],
        "result": {},
        "metrics": [],
        "spend": 0.0,
    })


async def _call_tool(server, handle_call_tool, tool_name, args):
    request = CallToolRequest(
        method="tools/call",
        params={"name": tool_name, "arguments": args},
    )
    return await handle_call_tool(request)


@pytest.mark.asyncio
async def test_all_mcp_tools_dispatch_without_unhandled_error():
    from terradev_cli.mcp import server

    server._build_all_tools()
    tools = list(server._ALL_TOOLS)

    def build_args(tool):
        args = GENERIC_ARGUMENTS.copy()
        args.update(_args_from_schema(tool.inputSchema))
        return args

    exec_result = {
        "success": True,
        "stdout": _mock_stdout(),
        "stderr": "",
        "returncode": 0,
    }
    terraform_result = {
        "success": True,
        "stdout": "ok",
        "stderr": "",
        "returncode": 0,
    }
    terraform_parallel_result = {
        "success": True,
        "stdout": "ok",
        "stderr": "",
        "terraform_outputs": {
            "instance_ids": [],
            "instance_ips": [],
            "provider_costs": [],
        },
    }
    safe_cmd_result = {
        "success": True,
        "stdout": "ok",
        "stderr": "",
        "returncode": 0,
    }
    local_gpus = {
        "has_local_gpu": False,
        "device_count": 0,
        "total_vram_gb": 0,
        "local_devices": [],
    }

    with patch("terradev_cli.mcp.server.execute_terradev_command", new=AsyncMock(return_value=exec_result)):
        with patch("terradev_cli.mcp.server.execute_terraform_command", new=AsyncMock(return_value=terraform_result)):
            with patch("terradev_cli.mcp.server.execute_terraform_parallel", new=AsyncMock(return_value=terraform_parallel_result)):
                with patch("terradev_cli.mcp.server.execute_safe_command", new=AsyncMock(return_value=safe_cmd_result)):
                    with patch("terradev_cli.mcp.server.discover_local_gpus", new=AsyncMock(return_value=local_gpus)):
                        with patch("terradev_cli.mcp.server._validate_config_dir", return_value="/tmp"):
                            with patch("terradev_cli.mcp.server._load_datadog_creds", return_value={"api_key": "k", "app_key": "a"}):
                                with patch("terradev_cli.mcp.server._ensure_tools_loaded", new=AsyncMock()):
                                    with patch("terradev_cli.mcp.server._UNSAFE_execute_shell_command", new=AsyncMock(return_value=safe_cmd_result)):
                                        with patch("aiohttp.ClientSession", _FakeAiohttpSession):
                                            failures = []
                                            unknown_tools = []
                                            error_tools = []
                                            for tool in tools:
                                                tool_name = tool.name
                                                try:
                                                    result = await _call_tool(server, server.handle_call_tool, tool_name, build_args(tool))
                                                    assert result is not None
                                                    if getattr(result, "isError", False):
                                                        text = ""
                                                        if result.content:
                                                            text = getattr(result.content[0], "text", "") if result.content else ""
                                                        if text.startswith("Unknown tool:"):
                                                            unknown_tools.append(tool_name)
                                                        else:
                                                            error_tools.append(f"{tool_name}: {text}")
                                                except Exception as exc:  # noqa: BLE001
                                                    failures.append(f"{tool_name}: {exc}")
                                            assert not failures, "tools raised unhandled errors: " + "\n".join(failures[:20])
                                            assert not unknown_tools, f"tools missing from command_map (drift): {unknown_tools}"
                                            # Non-"Unknown tool" errors are expected when generic arguments do not
                                            # satisfy a specialized branch; surface them for visibility only.
                                            if error_tools:
                                                warnings.warn(f"{len(error_tools)} tools returned isError with schema-derived args (expected for branches needing specific values).")
