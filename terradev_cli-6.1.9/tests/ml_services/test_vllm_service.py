"""Unit tests for the vLLM ML service."""
import asyncio
from unittest.mock import MagicMock

import pytest

from terradev_cli.ml_services.vllm_service import (
    LoRAModule,
    VLLMConfig,
    VLLMService,
    WorkloadProfile,
)


@pytest.fixture
def svc():
    return VLLMService(VLLMConfig(model_name="meta-llama/Llama-2-7b-hf"))


@pytest.fixture
def lora_config():
    return VLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        enable_lora=True,
        lora_modules=[LoRAModule(name="adapter1", path="/path/to/adapter1")],
    )


def test_config_defaults():
    cfg = VLLMConfig(model_name="m")
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.gpu_memory_utilization == 0.95


def test_config_create_auto_optimized():
    workload = WorkloadProfile(
        avg_prompt_length=512,
        avg_response_length=256,
        requests_per_second=5.0,
        concurrent_users=10,
        gpu_count=1,
        model_size_gb=15,
    )
    cfg = VLLMConfig.create_auto_optimized("meta-llama/Llama-2-7b-hf", workload)
    assert cfg.model_name == "meta-llama/Llama-2-7b-hf"
    assert cfg.tensor_parallel_size == 1
    assert cfg.max_num_batched_tokens >= 4096


def test_config_reasoning_optimized():
    workload = WorkloadProfile(gpu_count=8, model_size_gb=80)
    cfg = VLLMConfig.create_auto_optimized("deepseek-ai/deepseek-r1", workload)
    assert cfg.model_name == "deepseek-ai/deepseek-r1"
    assert cfg.enable_chunked_prefill is True


def test_build_server_args(svc):
    args = svc._build_server_args()
    assert "vllm" in args
    assert "serve" in args
    assert svc.config.model_name in args
    assert "--host" in args
    assert "--port" in args
    assert "8000" in args


def test_build_server_args_with_lora(lora_config):
    svc = VLLMService(lora_config)
    args = svc._build_server_args()
    assert "--enable-lora" in args
    assert "--lora-modules" in args
    assert "adapter1=/path/to/adapter1" in args


def test_get_supported_models(svc):
    models = svc.get_supported_models()
    assert isinstance(models, list)
    assert "meta-llama/Llama-2-7b-hf" in models


def test_get_deployment_script(svc):
    script = svc.get_deployment_script(instance_ip="10.0.0.1")
    assert "vllm.service" in script
    assert svc.config.model_name in script
    assert "10.0.0.1" in script


def test_context_manager(fake_aiohttp):
    svc = VLLMService(VLLMConfig(model_name="m"))
    session = None

    async def run():
        nonlocal session
        async with svc as s:
            session = s.session
            assert session is not None
        assert session.closed

    asyncio.run(run())


@pytest.mark.asyncio
async def test_test_connection_success(svc, fake_aiohttp):
    fake_aiohttp.responses = [(200, {}, "")]
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["model"] == svc.config.model_name


@pytest.mark.asyncio
async def test_test_connection_failure(svc, fake_aiohttp):
    fake_aiohttp.responses = [(500, {}, "boom")]
    result = await svc.test_connection()
    assert result["status"] == "failed"
    assert "500" in result["error"]


@pytest.mark.asyncio
async def test_get_server_info(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"id": "m1"}]}, ""),
    ]
    result = await svc.get_server_info()
    assert result["status"] == "success"
    assert result["models"][0]["id"] == "m1"


@pytest.mark.asyncio
async def test_test_inference(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (
            200,
            {
                "choices": [{"text": "hello"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "",
        ),
    ]
    result = await svc.test_inference("hi")
    assert result["status"] == "success"
    assert result["response"] == "hello"


@pytest.mark.asyncio
async def test_test_chat_completion(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (
            200,
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {},
            },
            "",
        ),
    ]
    result = await svc.test_chat_completion([{"role": "user", "content": "hello"}])
    assert result["status"] == "success"
    assert result["response"] == "hi"


@pytest.mark.asyncio
async def test_lora_list(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (
            200,
            {
                "data": [
                    {"id": "base", "parent": None},
                    {"id": "adapter1", "parent": "base"},
                ]
            },
            "",
        ),
    ]
    result = await svc.lora_list()
    assert result["status"] == "success"
    assert len(result["lora_adapters"]) == 1


@pytest.mark.asyncio
async def test_lora_load(svc, fake_aiohttp):
    fake_aiohttp.responses = [(200, {}, "")]
    adapter = LoRAModule(name="adapter1", path="/path")
    result = await svc.lora_load(adapter)
    assert result["status"] == "loaded"
    assert result["adapter"] == "adapter1"


@pytest.mark.asyncio
async def test_lora_unload(svc, fake_aiohttp):
    fake_aiohttp.responses = [(200, {}, "")]
    result = await svc.lora_unload("adapter1")
    assert result["status"] == "unloaded"
    assert result["adapter"] == "adapter1"


def test_start_server_success(svc, monkeypatch):
    def _fake_run(args, **kwargs):
        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = asyncio.run(svc.start_server("10.0.0.1"))
    assert result["status"] == "started"
    assert result["model"] == svc.config.model_name


def test_start_server_bad_model_name(monkeypatch):
    bad_svc = VLLMService(VLLMConfig(model_name="model; rm -rf /"))
    result = asyncio.run(bad_svc.start_server("10.0.0.1"))
    assert result["status"] == "failed"
    assert "Unsafe model_name" in result["error"]


def test_stop_server_success(svc, monkeypatch):
    def _fake_run(args, **kwargs):
        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = asyncio.run(svc.stop_server("10.0.0.1"))
    assert result["status"] == "stopped"


def test_install_vllm_success(svc, monkeypatch):
    def _fake_run(args, **kwargs):
        class _Result:
            returncode = 0
            stdout = "installed"
            stderr = ""
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = asyncio.run(svc.install_vllm("10.0.0.1"))
    assert result["status"] == "installed"


def test_ssh_args(svc):
    args = svc._build_ssh_args("10.0.0.1", "root", "/key.pem")
    assert args[0] == "ssh"
    assert "-i" in args
    assert "/key.pem" in args
    assert "root@10.0.0.1" in args
