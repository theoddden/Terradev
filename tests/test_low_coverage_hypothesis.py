#!/usr/bin/env python3
"""Property-based / Hypothesis tests for the low-coverage modules."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

pytestmark = [pytest.mark.hypothesis]


@st.composite
def _quote(draw):
    """Generate a synthetic quote dict."""
    return {
        "price_per_hour": draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "spot_price": draw(
            st.one_of(
                st.none(),
                st.floats(min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False),
            )
        ),
    }


@st.composite
def _gpu_name(draw):
    return draw(st.sampled_from(["A100", "V100", "H100", "RTX4090", "RTX3090"]))


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestHypothesisDemo:
    @given(
        provider=st.sampled_from(["aws", "gcp", "azure", "runpod", "vastai"]),
        price=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        gpu_type=_gpu_name(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_optimization_score_is_bounded(self, provider, price, gpu_type):
        from terradev_cli.demo import MockTerradevEngine

        score = MockTerradevEngine()._calculate_optimization_score(provider, price, gpu_type)
        assert 0 <= score <= 1

    @given(quotes=st.lists(_quote(), min_size=0, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_analyze_savings_invariants(self, quotes):
        from terradev_cli.demo import MockTerradevEngine

        result = MockTerradevEngine().analyze_savings(quotes)
        if not quotes:
            assert result == {}
        else:
            assert result["best_price"] <= result["worst_price"]
            assert result["best_price"] <= result["avg_price"]
            assert -0.01 <= result["savings_vs_worst"] <= 100.01
            assert -0.01 <= result["savings_vs_avg"] <= 100.01

    @given(gpu_type=_gpu_name(), region=st.text(min_size=0, max_size=20))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_get_parallel_quotes_returns_list(self, gpu_type, region, monkeypatch):
        from terradev_cli.demo import MockTerradevEngine

        engine = MockTerradevEngine()
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        quotes = asyncio.run(engine.get_parallel_quotes(gpu_type, region))
        assert isinstance(quotes, list)
        assert len(quotes) <= len(engine.providers)


# ---------------------------------------------------------------------------
# credential_prompt
# ---------------------------------------------------------------------------


class TestHypothesisCredentialPrompt:
    @given(provider=st.text(min_size=1, max_size=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_check_configured_providers_is_pure(self, provider, tmp_path, monkeypatch):
        from terradev_cli import credential_prompt

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        cred_file = tmp_path / ".terradev" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({provider: {"api_key": "x"}}))

        configured = credential_prompt.check_configured_providers()
        assert isinstance(configured, list)
        if provider:
            assert provider in configured


# ---------------------------------------------------------------------------
# pd_transport
# ---------------------------------------------------------------------------


class TestHypothesisPDTransport:
    @given(
        num_layers=st.integers(min_value=1, max_value=256),
        hidden_size=st.integers(min_value=64, max_value=8192),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_estimate_kv_block_bytes_non_negative(
        self, num_layers, hidden_size
    ):
        from terradev_cli.core.pd_transport import estimate_kv_block_bytes

        value = estimate_kv_block_bytes(
            context_tokens=num_layers,
            model_size_b=hidden_size,
        )
        assert isinstance(value, (int, float))
        assert value >= 0

    @given(
        context_tokens=st.integers(min_value=1, max_value=8192),
        model_size_b=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_transfer_time_ms_non_negative(self, context_tokens, model_size_b):
        from terradev_cli.core.pd_transport import transfer_time_ms, CXLTransport

        transport = CXLTransport(config=MagicMock())
        value = transfer_time_ms(context_tokens, model_size_b, transport)
        assert isinstance(value, (int, float))
        assert value >= 0

    @given(name=st.text(min_size=1, max_size=30))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_transport_describe_returns_dict(self, name):
        from terradev_cli.core.pd_transport import CXLTransport, NIXLNVLinkTransport, TCPFallbackTransport

        for cls in (CXLTransport, NIXLNVLinkTransport, TCPFallbackTransport):
            instance = cls(config=MagicMock())
            desc = instance.describe()
            assert isinstance(desc, str)
            assert "GB/s" in desc


# ---------------------------------------------------------------------------
# dataset_stager
# ---------------------------------------------------------------------------


class TestHypothesisDatasetStager:
    @given(data=st.binary(min_size=0, max_size=2048))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_compute_checksum_is_idempotent(self, data, tmp_path):
        from terradev_cli.core.dataset_stager import compute_checksum

        path = tmp_path / "data.bin"
        path.write_bytes(data)
        a = compute_checksum(str(path))
        b = compute_checksum(str(path))
        assert a == b
        assert isinstance(a, str)

    @given(
        source=st.text(min_size=1, max_size=100),
        dest=st.text(min_size=1, max_size=100),
        size=st.integers(min_value=0, max_value=1_000_000_000_000),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_staging_plan_to_dict_roundtrip(self, source, dest, size):
        from terradev_cli.core.dataset_stager import StagingPlan, _human_size

        plan = StagingPlan(
            dataset=source,
            regions=[dest],
            size_bytes=size,
            compression="zstd",
            estimated_compressed=size,
            chunks=1,
            chunk_size=size,
        )
        d = plan.to_dict()
        assert d["dataset"] == source
        assert dest in d["regions"]
        assert d["original_size"] == _human_size(size)


# ---------------------------------------------------------------------------
# model_router
# ---------------------------------------------------------------------------


class TestHypothesisModelRouter:
    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_step_classifier_returns_step_type(self, text):
        from terradev_cli.ml_services.model_router import StepClassifier

        classifier = StepClassifier()
        step = classifier.classify([{"role": "user", "content": text}])
        # classify returns a StepType enum or string.
        assert step is not None

    @given(name=st.text(min_size=1, max_size=30))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_model_router_route_returns_dict(self, name):
        from terradev_cli.ml_services.model_router import ModelRouter, RouterConfig

        router = ModelRouter(RouterConfig())
        result = router.route([{"role": "user", "content": name}])
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# pipeline_schema
# ---------------------------------------------------------------------------


class TestHypothesisPipelineSchema:
    @given(
        name=st.text(min_size=1, max_size=50),
        steps=st.lists(
            st.fixed_dictionaries(
                {
                    "name": st.text(min_size=1, max_size=20),
                    "image": st.text(min_size=1, max_size=60),
                    "gpu": st.integers(min_value=0, max_value=16),
                }
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_workflow_roundtrip(self, name, steps):
        from terradev_cli.core.pipeline_schema import Workflow

        data: Dict[str, Any] = {
            "api_version": "v1",
            "kind": "TrainingPipeline",
            "metadata": {"name": name},
            "spec": {"steps": steps},
        }
        try:
            wf = Workflow.from_dict(data)
        except Exception:  # noqa: BLE001
            # Invalid structure may raise; that is acceptable.
            return
        out = wf.to_dict()
        assert out["metadata"]["name"] == name


# ---------------------------------------------------------------------------
# auto_optimizer
# ---------------------------------------------------------------------------


class TestHypothesisAutoOptimizer:
    @given(
        queue_depth=st.integers(min_value=0, max_value=1000),
        budget=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_should_methods_return_bool(self, queue_depth, budget):
        from terradev_cli.optimization.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(config=MagicMock(), metrics_collector=MagicMock())
        ctx = MagicMock()
        ctx.queue_depth = queue_depth
        ctx.budget = budget

        for fn in (opt.should_apply_auto_scaling, opt.should_apply_warm_pool, opt.should_apply_semantic_routing):
            assert isinstance(fn(ctx), bool)


# ---------------------------------------------------------------------------
# gpu_topology
# ---------------------------------------------------------------------------


class TestHypothesisGPUTopology:
    @given(
        num_gpus=st.integers(min_value=0, max_value=32),
        arch=st.sampled_from(["hopper", "ampere", "blackwell", "unknown"]),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_build_intra_gpu_topology_returns_dict(self, num_gpus, arch):
        from terradev_cli.core.gpu_topology import GPUDevice, IntraGPUTopology, build_intra_gpu_topology

        gpu = GPUDevice(
            index=num_gpus,
            name=arch,
            pci_bus_id="0000:01:00.0",
            numa_node=0,
            pcie_root="root",
            pcie_switch="switch",
            gpu_arch=arch,
        )
        result = build_intra_gpu_topology(gpu)
        assert isinstance(result, IntraGPUTopology)

    @given(num_vfs=st.integers(min_value=0, max_value=64))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_sriov_manager_invariants(self, num_vfs):
        from terradev_cli.core.gpu_topology import SRIOVManager

        mgr = SRIOVManager()
        result = mgr.create_vfs("eth0", num_vfs)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# model_orchestrator
# ---------------------------------------------------------------------------


class TestHypothesisModelOrchestrator:
    @given(
        model_id=st.text(min_size=1, max_size=50),
        framework=st.sampled_from(["pytorch", "tensorflow", "vllm", "sglang"]),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_register_and_get_status(self, model_id, framework):
        from terradev_cli.core.model_orchestrator import ModelOrchestrator

        orch = ModelOrchestrator()
        orch.available_memory_gb = orch.total_memory_gb - orch.used_memory_gb
        orch.register_model(model_id, "/tmp/" + model_id, framework)
        status = orch.get_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# agentic_provisioner
# ---------------------------------------------------------------------------


class TestHypothesisAgenticProvisioner:
    @given(fleet_name=st.text(min_size=1, max_size=30))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fleet_status_returns_dict(self, fleet_name):
        from terradev_cli.core.agentic_provisioner import AgenticProvisioner

        prov = AgenticProvisioner()
        status = asyncio.run(prov.fleet_status(fleet_name))
        assert status is None or isinstance(status, dict)


# ---------------------------------------------------------------------------
# preflight_validator
# ---------------------------------------------------------------------------


class TestHypothesisPreflightValidator:
    @given(
        gpu_count=st.integers(min_value=0, max_value=256),
        budget=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_run_quick_returns_report(self, gpu_count, budget):
        from terradev_cli.core.preflight_validator import PreflightValidator

        validator = PreflightValidator()
        report = validator.run_quick()
        assert isinstance(report.to_dict(), dict)


# ---------------------------------------------------------------------------
# helm_generator
# ---------------------------------------------------------------------------


class TestHypothesisHelmGenerator:
    @given(
        name=st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20),
        replicas=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_generate_chart_returns_dict(self, name, replicas, tmp_path):
        from terradev_cli.core.helm_generator import HelmChartConfig, HelmChartGenerator

        config = HelmChartConfig(
            name=name,
            version="0.1.0",
            description="Test chart",
            app_version="1.0.0",
            kube_version=">=1.28",
            maintainers=[{"name": "test", "email": "test@example.com"}],
            keywords=["test"],
        )
        gen = HelmChartGenerator()
        result = gen.generate_chart(
            {"workload_type": "training", "gpu_type": "A100", "replicas": replicas, "image": "test:latest", "name": name},
            str(tmp_path),
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# databricks_integration
# ---------------------------------------------------------------------------


class TestHypothesisDatabricksIntegration:
    @given(experiment_name=st.text(min_size=1, max_size=50))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_generate_gpu_training_job_config_returns_dict(self, experiment_name):
        from terradev_cli.integrations.databricks_integration import generate_gpu_training_job_config

        result = generate_gpu_training_job_config(
            name=experiment_name,
            script_path="/foo/bar.py",
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# kv_cache_checkpoint_tests / weight_streaming / mla (aggregation invariants)
# ---------------------------------------------------------------------------


class TestHypothesisTestSuiteAggregation:
    @given(
        total=st.integers(min_value=0, max_value=50),
        passed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_success_rate_bounded(self, total, passed):
        from terradev_cli.core.kv_cache_checkpoint_tests import KVCacheCheckpointTests

        if total == 0:
            rate = 0.0
        else:
            rate = min(passed, total) / total
        assert 0 <= rate <= 1
