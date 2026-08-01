"""Unit tests for the SGLang ML service."""
import asyncio

import pytest

from terradev_cli.ml_services.sglang_service import (
    AttentionBackend,
    DeepEPMode,
    SchedulePolicy,
    SGLangConfig,
    SGLangOptimizer,
    SGLangService,
    SpeculativeAlgorithm,
    WorkloadType,
    create_sglang_service_from_credentials,
    get_sglang_setup_instructions,
)


@pytest.fixture
def svc():
    return SGLangService(
        SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.AGENTIC_CHAT,
        )
    )


def test_get_setup_instructions():
    instructions = get_sglang_setup_instructions()
    assert "SGLang" in instructions
    assert "pip install" in instructions


def test_create_service_from_credentials():
    svc = create_sglang_service_from_credentials(
        {
            "model_path": "model",
            "host": "0.0.0.0",
            "port": "8000",
            "api_key": "k",
            "tp_size": "2",
            "dp_size": "4",
        }
    )
    assert isinstance(svc, SGLangService)
    assert svc.config.model_path == "model"
    assert svc.config.tp == 2
    assert svc.config.dp_size == 4
    assert svc.config.api_key == "k"


def test_optimizer_detect_hardware():
    opt = SGLangOptimizer()
    hw = opt.detect_hardware()
    assert hw.gpu_type
    assert hw.memory_gb > 0


def test_optimizer_detect_model_type():
    opt = SGLangOptimizer()
    model_type, config = opt.detect_model_type("deepseek-ai/DeepSeek-V2")
    assert model_type == "deepseek"
    assert "workload_type" in config


def test_detect_workload_type(svc):
    wt = svc.detect_workload_type(
        "meta-llama/Llama-2-7b-hf", user_description="batch eval dataset"
    )
    assert wt == WorkloadType.BATCH_INFERENCE


def test_create_optimized_config(svc):
    config = svc.create_optimized_config(
        model_path="meta-llama/Llama-2-7b-hf",
        workload_type=WorkloadType.AGENTIC_CHAT,
    )
    assert config.workload_type == WorkloadType.AGENTIC_CHAT
    assert config.schedule_policy == SchedulePolicy.LPM
    assert config.attention_backend == AttentionBackend.FLASHINFER
    assert config.env_vars.get("SGLANG_CACHE_AWARE_ROUTING") == "1"


def test_create_optimized_config_moe(svc):
    config = svc.create_optimized_config(
        model_path="deepseek-ai/DeepSeek-V2",
    )
    assert config.tp == 8
    assert config.ep == 8
    assert config.enable_dp_attention is True


def test_generate_launch_command(svc):
    cmd = svc.generate_launch_command(svc.config)
    assert "python -m sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert svc.config.model_path in cmd


def test_generate_multi_replica_command(svc):
    cmd = svc.generate_multi_replica_command(svc.config, dp_size=2)
    assert "--dp-size" in cmd
    assert "2" in cmd


def test_validate_config(svc):
    warnings = svc.validate_config(svc.config)
    assert isinstance(warnings, list)


def test_get_optimization_summary(svc):
    summary = svc.get_optimization_summary(svc.config)
    assert summary["workload_type"] == WorkloadType.AGENTIC_CHAT.value
    assert summary["schedule_policy"] == SchedulePolicy.LPM.value
    assert "hardware_detected" in summary
    assert len(summary["optimizations_applied"]) > 0


def test_get_supported_models(svc):
    models = svc.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_get_deployment_script(svc):
    script = svc.get_deployment_script(instance_ip="10.0.0.1")
    assert "sglang.service" in script
    assert svc.config.model_path in script


def test_context_manager(fake_aiohttp):
    svc = SGLangService(
        SGLangConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            workload_type=WorkloadType.AGENTIC_CHAT,
        )
    )
    session = None

    async def run():
        nonlocal session
        async with svc as s:
            session = s.session
            assert session is not None
        assert session.closed

    asyncio.run(run())


@pytest.mark.asyncio
async def test_test_connection_success(svc, fake_aiohttp, monkeypatch):
    def _fake_run(*args, **kwargs):
        class _Result:
            returncode = 0
            stdout = "0.3.0"
            stderr = ""
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["sglang_version"] == "0.3.0"


@pytest.mark.asyncio
async def test_test_connection_failure(svc, fake_aiohttp, monkeypatch):
    def _fake_run(*args, **kwargs):
        class _Result:
            returncode = 1
            stdout = ""
            stderr = "not installed"
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = await svc.test_connection()
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_get_server_info(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"data": [{"id": "m1"}]}, ""),
    ]
    result = await svc.get_server_info()
    assert result["status"] == "success"
    assert result["models"][0]["id"] == "m1"


@pytest.mark.asyncio
async def test_get_server_metrics(svc, fake_aiohttp):
    metrics_text = "prompt_tokens 10\ncompletion_tokens 20\n"
    fake_aiohttp.responses = [
        (200, {}, metrics_text),
    ]
    result = await svc.get_server_metrics()
    assert result["status"] == "success"
    assert result["metrics"]["prompt_tokens"] == 10.0
    assert result["metrics"]["completion_tokens"] == 20.0


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
    result = await svc.test_inference("hi", max_tokens=10)
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
    result = await svc.test_chat_completion(
        [{"role": "user", "content": "hello"}]
    )
    assert result["status"] == "success"
    assert result["response"] == "hi"


def test_ssh_args(svc):
    args = svc._build_ssh_args("10.0.0.1", "root", "/key.pem")
    assert args[0] == "ssh"
    assert "-i" in args
    assert "/key.pem" in args
    assert "root@10.0.0.1" in args
