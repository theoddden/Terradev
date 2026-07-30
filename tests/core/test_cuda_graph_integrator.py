"""Tests for terradev_cli.core.cuda_graph_integrator.

CUDA Graph integrator recommends graph capture for model-endpoint pairs.
"""

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from terradev_cli.core.cuda_graph_integrator import (
    CUDAGraphIntegrator,
    CUDAGraphRecommendation,
    get_cuda_graph_integrator,
)


@dataclass
class FakeNUMAScore:
    pcie_locality: str
    metadata: Dict[str, Any]


class FakeNUMAScorer:
    def score_endpoint(self, endpoint_id, gpu_index=None, model_type=None):
        return FakeNUMAScore(
            pcie_locality="PIX",
            metadata={
                "cuda_graph_score": 0.92,
                "cuda_graph_recommended": True,
                "graph_optimization_potential": "high",
            },
        )


class FakeWarmPool:
    def __init__(self):
        self.model_priorities = {}
        self.metrics = type("Metrics", (), {"cuda_graph_optimized_models": 0})()


@pytest.fixture
def integrator():
    return CUDAGraphIntegrator(FakeNUMAScorer(), FakeWarmPool())


def test_analyze_model_endpoint(integrator):
    """analyze_model_endpoint returns a recommendation and caches it."""
    rec = integrator.analyze_model_endpoint("llama-7b", "ep-1", gpu_index=0)
    assert isinstance(rec, CUDAGraphRecommendation)
    assert rec.model_id == "llama-7b"
    assert rec.use_cuda_graphs is True
    assert rec.optimization_score == 0.92
    assert rec.priority_boost >= 0

    # Caching
    assert (
        integrator._recommendations_cache[f"{rec.model_id}:{rec.endpoint_id}"] == rec
    )


def test_detect_model_type(integrator):
    """Model types are inferred from the model identifier."""
    assert integrator._detect_model_type("mixtral-8x7b-moe") == "moe"
    assert integrator._detect_model_type("llama-3-70b") == "transformer"
    assert integrator._detect_model_type("resnet-50") == "cnn"
    assert integrator._detect_model_type("foo") == "unknown"


def test_get_expected_speedup(integrator):
    """Speedup ranges map to optimization potential."""
    assert "1.5-3x" in integrator._get_expected_speedup("high")
    assert "2-5x" in integrator._get_expected_speedup("optimal")
    assert "<1.2x" in integrator._get_expected_speedup("unknown")


def test_get_memory_requirements(integrator):
    """Memory guidance scales with graph score."""
    assert "4-8GB" in integrator._get_memory_requirements(0.9)
    assert "1-2GB" in integrator._get_memory_requirements(0.3)


def test_update_warm_pool_priority(integrator):
    """High-score recommendations boost warm pool priority."""
    rec = integrator.analyze_model_endpoint("llama-7b", "ep-1", gpu_index=0)
    assert integrator.warm_pool.model_priorities["llama-7b"] > 0
    assert integrator.warm_pool.metrics.cuda_graph_optimized_models == 1


def test_get_optimization_summary(integrator):
    """Summary aggregates cached recommendations."""
    integrator.analyze_model_endpoint("llama-7b", "ep-1")
    summary = integrator.get_optimization_summary()
    assert summary["total_models"] == 1
    assert summary["cuda_graph_compatible"] == 1
    assert summary["high_potential"] == 1


def test_should_prefer_endpoint(integrator):
    """Endpoint selection prefers higher graph scores and NUMA alignment."""
    integrator.analyze_model_endpoint("llama-7b", "ep-1", gpu_index=0)

    # Score a worse endpoint
    class WorseScorer:
        def score_endpoint(self, *a, **k):
            return FakeNUMAScore(
                pcie_locality="PHB",
                metadata={
                    "cuda_graph_score": 0.5,
                    "cuda_graph_recommended": False,
                    "graph_optimization_potential": "low",
                },
            )

    integrator.numa_scorer = WorseScorer()
    integrator.analyze_model_endpoint("llama-7b", "ep-2", gpu_index=1)

    preferred = integrator.should_prefer_endpoint("llama-7b", "ep-1", "ep-2")
    assert preferred == "ep-1"


def test_get_recommendations_for_model(integrator):
    """Recommendations can be filtered by model."""
    integrator.analyze_model_endpoint("llama-7b", "ep-1")
    integrator.analyze_model_endpoint("llama-7b", "ep-2")
    recs = integrator.get_recommendations_for_model("llama-7b")
    assert len(recs) == 2


def test_get_cuda_graph_integrator_singleton():
    """get_cuda_graph_integrator returns a singleton."""
    scorer = FakeNUMAScorer()
    warm = FakeWarmPool()
    i1 = get_cuda_graph_integrator(scorer, warm)
    i2 = get_cuda_graph_integrator(scorer, warm)
    assert i1 is i2
