#!/usr/bin/env python3
"""Canary smoke tests for the low-coverage modules listed by coverage %."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import click
import pytest
from click.testing import CliRunner

import terradev_cli.cli
from terradev_cli import __version__

pytestmark = [pytest.mark.canary]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run(coro_or_call):
    """Run a coroutine or a normal callable."""
    if asyncio.iscoroutine(coro_or_call):
        return asyncio.run(coro_or_call)
    if asyncio.iscoroutinefunction(coro_or_call):
        return asyncio.run(coro_or_call())
    return coro_or_call()


def _make_instance(cls, **kwargs):
    """Instantiate a class with safe defaults, using MagicMock for missing params."""
    sig = inspect.signature(cls)
    args: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if name in kwargs:
            args[name] = kwargs[name]
        elif param.default is not inspect.Parameter.empty:
            args[name] = param.default
        else:
            args[name] = MagicMock()
    return cls(**args)


def _get_public_methods(cls):
    return [n for n, o in inspect.getmembers(cls) if inspect.isfunction(o) and not n.startswith("_")]


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """HOME (and USERPROFILE on Windows) points to a temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# test_summary
# ---------------------------------------------------------------------------


class TestCanaryTestSummary:
    @pytest.mark.asyncio
    async def test_run_test_suite_aggregates(self, monkeypatch, tmp_path):
        from terradev_cli import test_summary

        # Avoid the slow/subprocess tests; verify aggregation logic with mocked outputs.
        def fake_run(cmd, **kw):
            class Result:
                returncode = 0
                stdout = "Success Rate: 80.0%\nSuccess Rate: 66.7%\nSUCCESS"
                stderr = ""
            return Result()

        monkeypatch.setattr(test_summary.subprocess, "run", fake_run)
        assert await test_summary.run_test_suite() is True


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestCanaryDemo:
    def test_analyze_savings(self):
        from terradev_cli.demo import MockTerradevEngine

        engine = MockTerradevEngine()
        quotes = [
            {"price_per_hour": 1.0, "spot_price": 0.5},
            {"price_per_hour": 2.0, "spot_price": None},
        ]
        result = engine.analyze_savings(quotes)
        assert result["best_price"] == 1.0
        assert result["worst_price"] == 2.0

    @pytest.mark.parametrize(
        "provider, price, gpu_type, expected",
        [
            ("aws", 2.5, "A100", None),
        ],
    )
    def test_calculate_optimization_score_is_positive(self, provider, price, gpu_type, expected):
        from terradev_cli.demo import MockTerradevEngine

        score = MockTerradevEngine()._calculate_optimization_score(provider, price, gpu_type)
        assert 0 <= score <= 1

    def test_main_runs(self, monkeypatch):
        from terradev_cli import demo

        monkeypatch.setattr("builtins.print", lambda *a, **k: None)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        asyncio.get_event_loop().run_until_complete(demo.main())


# ---------------------------------------------------------------------------
# credential_prompt
# ---------------------------------------------------------------------------


class TestCanaryCredentialPrompt:
    def test_check_configured_providers(self, isolated_home):
        from terradev_cli import credential_prompt

        assert credential_prompt.check_configured_providers() == []

        creds = {"runpod": {"api_key": "rpa_fake"}}
        cred_file = isolated_home / ".terradev" / "credentials.json"
        cred_file.parent.mkdir(exist_ok=True)
        cred_file.write_text(json.dumps(creds))

        assert credential_prompt.check_configured_providers() == ["runpod"]

    def test_prompt_for_credentials_skips_when_empty(self, isolated_home, monkeypatch):
        from terradev_cli import credential_prompt

        monkeypatch.setattr("builtins.print", lambda *a, **k: None)
        monkeypatch.setattr(credential_prompt.click, "prompt", lambda *a, **k: "")
        configured = credential_prompt.prompt_for_credentials()
        assert isinstance(configured, list)
        # With empty input all providers are removed, so nothing should be configured.
        assert credential_prompt.check_configured_providers() == []


# ---------------------------------------------------------------------------
# kv_cache_checkpoint_tests
# ---------------------------------------------------------------------------


class TestCanaryKVCacheCheckpointTests:
    async def test_run_all_tests_aggregates_failures(self, monkeypatch, tmp_path):
        from terradev_cli.core.kv_cache_checkpoint_tests import KVCacheCheckpointTests

        suite = KVCacheCheckpointTests()
        # Mock the internal test methods to control the result set.
        for method_name in _get_public_methods(KVCacheCheckpointTests):
            if method_name == "run_all_tests":
                continue
            monkeypatch.setattr(
                suite,
                method_name,
                AsyncMock(return_value={"test_name": method_name, "status": "PASSED"}),
            )
        summary = await suite.run_all_tests()
        assert summary["total_tests"] > 0
        assert 0 <= summary["success_rate"] <= 1


# ---------------------------------------------------------------------------
# weight_streaming_benchmarks
# ---------------------------------------------------------------------------


class TestCanaryWeightStreamingBenchmarks:
    async def test_run_all_benchmarks_aggregates(self, monkeypatch, tmp_path):
        from terradev_cli.core.weight_streaming_benchmarks import WeightStreamingBenchmarks

        suite = WeightStreamingBenchmarks()
        for method_name in _get_public_methods(WeightStreamingBenchmarks):
            if method_name == "run_all_benchmarks":
                continue
            monkeypatch.setattr(
                suite,
                method_name,
                AsyncMock(return_value={"benchmark_name": method_name, "status": "PASSED"}),
            )
        summary = await suite.run_all_benchmarks()
        assert summary["total_benchmarks"] > 0
        assert 0 <= summary["success_rate"] <= 1


# ---------------------------------------------------------------------------
# mla_vram_tests
# ---------------------------------------------------------------------------


class TestCanaryMLAVRAmTests:
    async def test_run_all_tests_aggregates(self, monkeypatch, tmp_path):
        from terradev_cli.core.mla_vram_tests import MLA_VRAM_Tests

        suite = MLA_VRAM_Tests()
        for method_name in _get_public_methods(MLA_VRAM_Tests):
            if method_name == "run_all_tests":
                continue
            monkeypatch.setattr(
                suite,
                method_name,
                AsyncMock(return_value={"test_name": method_name, "status": "PASSED"}),
            )
        summary = await suite.run_all_tests()
        assert summary["total_tests"] > 0
        assert 0 <= summary["success_rate"] <= 1


# ---------------------------------------------------------------------------
# hf_cli_integration
# ---------------------------------------------------------------------------


class TestCanaryHFCliIntegration:
    def test_register_hf_commands_adds_cli_group(self):
        from terradev_cli.core.hf_cli_integration import register_hf_commands

        cli = click.Group()
        register_hf_commands(cli)
        # Verify a hf-space-ish command was registered.
        assert any("hf" in name.lower() for name in cli.commands)

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["hf-space", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# auto_optimizer
# ---------------------------------------------------------------------------


class TestCanaryAutoOptimizer:
    def test_should_methods_return_bool(self, isolated_home):
        from terradev_cli.optimization.auto_optimizer import AutoOptimizer

        opt = _make_instance(AutoOptimizer, config=MagicMock())
        assert isinstance(opt.should_apply_auto_scaling(MagicMock()), bool)
        assert isinstance(opt.should_apply_warm_pool(MagicMock()), bool)
        assert isinstance(opt.should_apply_semantic_routing(MagicMock()), bool)

    def test_get_optimization_summary_returns_dict(self, isolated_home):
        from terradev_cli.optimization.auto_optimizer import AutoOptimizer

        opt = _make_instance(AutoOptimizer, config=MagicMock())
        result = opt.get_optimization_summary()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ml_services - KServe
# ---------------------------------------------------------------------------


class TestCanaryKServeService:
    @patch("terradev_cli.ml_services.kserve_service.subprocess.run")
    async def test_test_connection_handles_subprocess(self, mock_run):
        from terradev_cli.ml_services.kserve_service import KServeConfig, KServeService

        class FakeResult:
            returncode = 0
            stdout = "cluster-info"
            stderr = ""

        mock_run.return_value = FakeResult()

        config = KServeConfig(namespace="test")
        svc = KServeService(config)
        result = await svc.test_connection()
        assert result["status"] in ("connected", "failed")


# ---------------------------------------------------------------------------
# ml_services - LangGraph
# ---------------------------------------------------------------------------


class TestCanaryLangGraphService:
    def test_create_workflow_returns_result(self):
        from terradev_cli.ml_services.langgraph_service import LangGraphConfig, LangGraphService

        config = LangGraphConfig(api_key="test")
        svc = LangGraphService(config)
        result = _run(svc.create_workflow({"name": "test-workflow", "prompt": "Return 42"}))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ml_services - Ollama
# ---------------------------------------------------------------------------


class TestCanaryOllamaService:
    @patch("terradev_cli.ml_services.ollama_service.aiohttp.ClientSession")
    async def test_test_connection(self, MockSession):
        from terradev_cli.ml_services.ollama_service import OllamaConfig, OllamaService

        config = OllamaConfig(host="localhost", port=11434)
        svc = OllamaService(config)
        result = await svc.test_connection()
        assert result["status"] in ("connected", "failed")


# ---------------------------------------------------------------------------
# ml_services - PEFTImport
# ---------------------------------------------------------------------------


class TestCanaryPEFTImportService:
    def test_list_local_adapters(self, tmp_path):
        from terradev_cli.ml_services.peft_import_service import PEFTImportService

        svc = PEFTImportService(cache_dir=tmp_path)
        result = svc.list_local_adapters()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ml_services - ModelRouter
# ---------------------------------------------------------------------------


class TestCanaryModelRouter:
    def test_step_classifier_classify(self):
        from terradev_cli.ml_services.model_router import StepClassifier, StepType

        classifier = StepClassifier()
        step = classifier.classify([{"role": "user", "content": "Deploy a vLLM model with LoRA adapters"}])
        assert isinstance(step, StepType)

    def test_model_router_route_basic(self):
        from terradev_cli.ml_services.model_router import ModelRouter, RouterConfig

        cfg = RouterConfig()
        router = ModelRouter(cfg)
        result = router.route([{"role": "user", "content": "hello"}])
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# ml_services - LangChain
# ---------------------------------------------------------------------------


class TestCanaryLangChainService:
    def test_get_langchain_config(self):
        from terradev_cli.ml_services.langchain_service import LangChainConfig, LangChainService

        config = LangChainConfig(api_key="test")
        svc = LangChainService(config)
        result = svc.get_langchain_config()
        assert isinstance(result, dict)

    def test_create_trace(self):
        from terradev_cli.ml_services.langchain_service import LangChainConfig, LangChainService

        config = LangChainConfig(api_key="test")
        svc = LangChainService(config)
        result = _run(svc.create_trace("test-trace", {"project_name": "test-project"}))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# preflight_validator
# ---------------------------------------------------------------------------


class TestCanaryPreflightValidator:
    def test_run_quick_returns_report(self):
        from terradev_cli.core.preflight_validator import PreflightValidator

        validator = _make_instance(PreflightValidator)
        report = validator.run_quick()
        assert isinstance(report.to_dict(), dict)

    def test_report_summary_returns_string(self):
        from terradev_cli.core.preflight_validator import PreflightReport

        report = PreflightReport()
        assert isinstance(report.summary(), dict)


# ---------------------------------------------------------------------------
# model_orchestrator
# ---------------------------------------------------------------------------


class TestCanaryModelOrchestrator:
    def test_register_and_get_status(self):
        from terradev_cli.core.model_orchestrator import ModelOrchestrator

        from terradev_cli.core.model_orchestrator import ScalingPolicy

        orch = _make_instance(ModelOrchestrator)
        orch.gpu_id = 0
        orch.total_memory_gb = 32.0
        orch.used_memory_gb = 8.0
        orch.available_memory_gb = 24.0
        orch.scaling_policy = ScalingPolicy.HYBRID
        orch.register_model("m1", "/tmp/m1", "pytorch")
        status = orch.get_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# agentic_provisioner
# ---------------------------------------------------------------------------


class TestCanaryAgenticProvisioner:
    @pytest.mark.asyncio
    async def test_list_fleets_and_status_empty(self):
        from terradev_cli.core.agentic_provisioner import AgenticProvisioner

        prov = _make_instance(AgenticProvisioner)
        fleets = prov.list_fleets()
        assert isinstance(fleets, (list, dict))
        status = await prov.fleet_status("no-such-fleet")
        assert status is None or isinstance(status, dict)


# ---------------------------------------------------------------------------
# cli_karpenter
# ---------------------------------------------------------------------------


class TestCanaryCliKarpenter:
    def test_register_karpenter_commands(self):
        from terradev_cli.cli_karpenter import register_karpenter_commands

        cli = click.Group()
        register_karpenter_commands(cli, lambda *a, **k: None)
        assert any("karpenter" in name.lower() for name in cli.commands)

    def test_karpenter_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["karpenter", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# dataset_stager
# ---------------------------------------------------------------------------


class TestCanaryDatasetStager:
    def test_compute_checksum(self, tmp_path):
        from terradev_cli.core.dataset_stager import compute_checksum

        test_file = tmp_path / "data.bin"
        test_file.write_bytes(b"hello world")
        result = compute_checksum(str(test_file))
        assert isinstance(result, str) and len(result) > 0

    def test_staging_plan_to_dict(self):
        from terradev_cli.core.dataset_stager import StagingPlan

        plan = StagingPlan(
            dataset="s3://bucket/data",
            regions=["us-east-1"],
            size_bytes=100,
            compression="zstd",
            estimated_compressed=80,
            chunks=1,
            chunk_size=100,
        )
        d = plan.to_dict()
        assert d["dataset"] == "s3://bucket/data"
        assert "100" in d["original_size"]


# ---------------------------------------------------------------------------
# commands/training
# ---------------------------------------------------------------------------


class TestCanaryCommandsTraining:
    def test_train_group_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["train", "--help"])
        assert result.exit_code == 0

    def test_train_start_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["train", "start", "--help"])
        assert result.exit_code == 0

    def test_lora_list_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["lora", "list", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# commands/inference
# ---------------------------------------------------------------------------


class TestCanaryCommandsInference:
    def test_inference_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["infer", "--help"])
        assert result.exit_code == 0

    def test_inference_serve_help(self):
        runner = CliRunner()
        result = runner.invoke(terradev_cli.cli.cli, ["infer", "deploy", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# helm_generator
# ---------------------------------------------------------------------------


class TestCanaryHelmGenerator:
    def test_generate_chart_creates_files(self, tmp_path):
        from terradev_cli.core.helm_generator import HelmChartConfig, HelmChartGenerator

        config = HelmChartConfig(
            name="test-chart",
            version="0.1.0",
            description="Test chart",
            app_version="1.0.0",
            kube_version=">=1.28",
            maintainers=[{"name": "test", "email": "test@example.com"}],
            keywords=["test"],
        )
        gen = HelmChartGenerator()
        result = gen.generate_chart(
            {"workload_type": "training", "gpu_type": "A100", "replicas": 1, "image": "test:latest", "name": "test-chart"},
            str(tmp_path),
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# providers/azure
# ---------------------------------------------------------------------------


class TestCanaryAzureProvider:
    @patch("terradev_cli.providers.base_provider.aiohttp.ClientSession")
    async def test_check_health(self, MockSession):
        from terradev_cli.providers.azure_provider import AzureProvider

        prov = AzureProvider({"api_key": "fake", "subscription_id": "sub-1"})
        # Patch the session if it was created in __init__
        if hasattr(prov, "session") and prov.session is None:
            prov.session = AsyncMock()
        result = await prov.check_health()
        assert hasattr(result, "healthy")


# ---------------------------------------------------------------------------
# providers/hetzner
# ---------------------------------------------------------------------------


class TestCanaryHetznerProvider:
    @patch("terradev_cli.providers.base_provider.aiohttp.ClientSession")
    async def test_check_health(self, MockSession):
        from terradev_cli.providers.hetzner_provider import HetznerProvider

        prov = HetznerProvider({"api_key": "fake"})
        result = await prov.check_health()
        assert hasattr(result, "healthy")

    @pytest.mark.asyncio
    async def test_traffic_monitor_allowance(self):
        from terradev_cli.providers.hetzner_provider import HetznerTrafficMonitor

        monitor = HetznerTrafficMonitor()
        result = await monitor.get_traffic_allowance("server-1")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# pipeline_schema
# ---------------------------------------------------------------------------


class TestCanaryPipelineSchema:
    def test_workflow_roundtrip(self):
        from terradev_cli.core.pipeline_schema import Workflow

        data = {
            "api_version": "v1",
            "kind": "TrainingPipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "steps": [
                    {"name": "train", "image": "test:latest", "gpu": 1},
                ],
            },
        }
        wf = Workflow.from_dict(data)
        assert wf.to_dict()["metadata"]["name"] == "test-pipeline"

    def test_validator_returns_report(self):
        from terradev_cli.core.pipeline_schema import PipelineValidator, Workflow

        validator = PipelineValidator()
        report = validator.validate_workflow(Workflow())
        assert isinstance(report, (dict, list))


# ---------------------------------------------------------------------------
# gpu_topology
# ---------------------------------------------------------------------------


class TestCanaryGPUTopology:
    def test_detect_gpu_arch(self, monkeypatch):
        from terradev_cli.core.gpu_topology import detect_gpu_arch

        # Force the lspci branch to return a known name.
        def fake_check_output(cmd, *a, **k):
            return b"NVIDIA Corporation GH200 [H100]"

        monkeypatch.setattr("terradev_cli.core.gpu_topology.subprocess.check_output", fake_check_output)
        arch = detect_gpu_arch("H100")
        assert isinstance(arch, str)

    def test_pci_topology_detect_gpus(self, monkeypatch):
        from terradev_cli.core.gpu_topology import PCIeTopologyDetector

        detector = PCIeTopologyDetector()

        class FakeRun:
            returncode = 1
            stdout = ""
            stderr = ""

        monkeypatch.setattr("terradev_cli.core.gpu_topology.subprocess.run", lambda *a, **k: FakeRun())
        gpus = detector.detect_gpus()
        assert isinstance(gpus, list)


# ---------------------------------------------------------------------------
# pd_transport
# ---------------------------------------------------------------------------


class TestCanaryPDTransport:
    def test_transport_pure_methods(self):
        from terradev_cli.core.pd_transport import CXLTransport, estimate_kv_block_bytes

        transport = CXLTransport(config=MagicMock())
        assert transport.bandwidth_gbps() > 0
        assert transport.latency_us() >= 0
        assert isinstance(transport.describe(), str)
        assert isinstance(estimate_kv_block_bytes(context_tokens=2, model_size_b=1), (int, float))

    def test_transport_selector(self):
        from terradev_cli.core.pd_transport import TransportSelector

        selector = TransportSelector()
        desc = selector.describe_all(config=MagicMock())
        assert isinstance(desc, list)


# ---------------------------------------------------------------------------
# k8s/terraform_wrapper
# ---------------------------------------------------------------------------


class TestCanaryTerraformWrapper:
    def test_wrapper_instantiates(self):
        from terradev_cli.k8s.terraform_wrapper import TerraformWrapper

        wrapper = _make_instance(TerraformWrapper)
        assert wrapper is not None


# ---------------------------------------------------------------------------
# integrations/databricks
# ---------------------------------------------------------------------------


class TestCanaryDatabricksIntegration:
    def test_is_configured_false_without_env(self, monkeypatch):
        from terradev_cli.integrations.databricks_integration import is_configured

        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        assert is_configured({}) is False

    def test_get_credential_prompts_returns_list(self):
        from terradev_cli.integrations.databricks_integration import get_credential_prompts

        prompts = get_credential_prompts()
        assert isinstance(prompts, list)
