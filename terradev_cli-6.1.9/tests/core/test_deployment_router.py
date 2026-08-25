"""Tests for terradev_cli.core.deployment_router.

The deployment router is the client-facing recommendation engine: given a GPU
need, region, and budget it must return scored, ranked deployment options.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.core.deployment_router import (
    DeploymentOption,
    DeploymentRequirements,
    DeploymentType,
    SmartDeploymentRouter,
    RequirementsAnalyzer,
    DirectDeploymentStrategy,
    KubernetesDeploymentStrategy,
    HybridDeploymentStrategy,
)


class FakePriceEngine:
    """In-memory price engine that never hits the network."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_realtime_prices(self, gpu_type, region=None):
        from terradev_cli.core.price_discovery import PriceInfo
        from datetime import datetime
        return [
            PriceInfo(
                provider="runpod",
                gpu_type=gpu_type,
                price=2.0,
                instance_type="A100",
                region=region or "us-east-1",
                capacity="high",
                confidence=0.95,
                last_updated=datetime.now(),
                spot=False,
            ),
            PriceInfo(
                provider="aws",
                gpu_type=gpu_type,
                price=3.5,
                instance_type="p4d.24xlarge",
                region=region or "us-east-1",
                capacity="medium",
                confidence=0.85,
                last_updated=datetime.now(),
                spot=True,
            ),
        ]


@pytest.fixture
def router():
    r = SmartDeploymentRouter()
    r.price_engine = FakePriceEngine()
    return r


@pytest.mark.asyncio
async def test_recommend_deployments_returns_scored_options(router):
    """The router returns ranked options across direct, k8s, and hybrid."""
    request = {
        "gpu_type": "A100",
        "gpu_count": 8,
        "estimated_hours": 2.0,
        "budget": 100.0,
    }
    options = await router.recommend_deployments(request)
    assert len(options) > 0
    assert all(hasattr(o, "score") for o in options)
    assert options[0].score >= options[-1].score


@pytest.mark.asyncio
async def test_recommend_deployments_respects_budget(router):
    """Options that exceed budget get a zero cost score, pushing them down."""
    request = {
        "gpu_type": "A100",
        "gpu_count": 8,
        "estimated_hours": 10.0,
        "budget": 50.0,
    }
    options = await router.recommend_deployments(request)
    req = RequirementsAnalyzer().analyze(request)
    # All options cost more than $50 (8 GPUs * 10 hours * $2/hr = $160+),
    # so the cost component must be zero even if the total score isn't.
    assert all(o.estimated_total_cost > req.budget for o in options)
    assert all(router._calculate_cost_score(o, req) == 0.0 for o in options)


@pytest.mark.asyncio
async def test_recommend_deployments_prefers_direct_for_small_workloads(router):
    """Small workloads should prefer direct deployment over k8s/hybrid."""
    request = {
        "gpu_type": "A100",
        "gpu_count": 1,
        "estimated_hours": 0.5,
    }
    options = await router.recommend_deployments(request)
    assert options[0].type == DeploymentType.DIRECT


def test_requirements_analyzer_defaults():
    """Missing keys should be filled with sensible defaults."""
    analyzer = RequirementsAnalyzer()
    req = analyzer.analyze({})
    assert req.gpu_type == "A100"
    assert req.gpu_count == 1
    assert req.memory_gb == 16
    assert req.budget is None


def test_deployment_requirements_dataclass():
    """DeploymentRequirements can be instantiated with custom values."""
    req = DeploymentRequirements(
        gpu_type="H100", gpu_count=4, budget=50.0, region="eu-west-1"
    )
    assert req.gpu_type == "H100"
    assert req.gpu_count == 4
    assert req.budget == 50.0


@pytest.mark.asyncio
async def test_direct_strategy_generates_options():
    """Direct strategy creates one option per price info."""
    from terradev_cli.core.price_discovery import PriceInfo
    from datetime import datetime

    price = PriceInfo(
        provider="runpod",
        gpu_type="A100",
        price=2.0,
        instance_type="A100",
        region="us-east-1",
        capacity="high",
        confidence=0.95,
        last_updated=datetime.now(),
        spot=False,
    )
    strategy = DirectDeploymentStrategy()
    options = await strategy.generate_options(
        DeploymentRequirements(gpu_type="A100", gpu_count=2, estimated_hours=1.0),
        [price],
    )
    assert len(options) == 1
    assert options[0].type == DeploymentType.DIRECT
    assert options[0].estimated_total_cost == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_kubernetes_strategy_skips_without_kubeconfig():
    """K8s strategy returns empty options if kubeconfig is absent."""
    strategy = KubernetesDeploymentStrategy()
    with patch.dict("os.environ", {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            options = await strategy.generate_options(
                DeploymentRequirements(gpu_type="A100"), []
            )
    assert options == []


@pytest.mark.asyncio
async def test_hybrid_strategy_only_complex_workloads():
    """Hybrid only generates options for large, long workloads."""
    strategy = HybridDeploymentStrategy()
    small = await strategy.generate_options(
        DeploymentRequirements(gpu_type="A100", gpu_count=1, estimated_hours=0.5), []
    )
    assert small == []

    from terradev_cli.core.price_discovery import PriceInfo
    from datetime import datetime
    price = PriceInfo(
        provider="aws",
        gpu_type="A100",
        price=3.0,
        instance_type="p4d.24xlarge",
        region="us-east-1",
        capacity="high",
        confidence=0.9,
        last_updated=datetime.now(),
        spot=False,
    )
    large = await strategy.generate_options(
        DeploymentRequirements(gpu_type="A100", gpu_count=8, estimated_hours=4.0),
        [price],
    )
    assert len(large) == 1
    assert large[0].type == DeploymentType.HYBRID
    assert large[0].metadata["hybrid_overhead"] == 1.2


@pytest.mark.asyncio
async def test_execute_delegates_to_strategy(router):
    """execute_deployment dispatches to the correct strategy."""
    option = MagicMock(spec=DeploymentOption)
    option.type = DeploymentType.DIRECT
    option.provider = "runpod"
    option.instance_type = "A100"
    option.setup_time_minutes = 5

    result = await router.execute_deployment(option, DeploymentRequirements(gpu_type="A100"))
    assert result["status"] == "deploying"
    assert "direct-" in result["deployment_id"]
    assert result["provider"] == "runpod"


def test_score_components_are_within_range(router):
    """Each scoring factor returns a value between 0 and 1."""
    option = DeploymentOption(
        type=DeploymentType.DIRECT,
        provider="runpod",
        instance_type="A100",
        price_per_hour=2.0,
        estimated_total_cost=4.0,
        confidence=0.95,
        risk_score=0.1,
        setup_time_minutes=5,
        pros=[],
        cons=[],
        metadata={},
    )
    req = DeploymentRequirements(gpu_type="A100", gpu_count=1, budget=100.0)

    assert 0.0 <= router._calculate_cost_score(option, req) <= 1.0
    assert 0.0 <= router._calculate_performance_score(option, req) <= 1.0
    assert 0.0 <= router._calculate_convenience_score(option, req) <= 1.0
    assert 0.0 <= router._calculate_reliability_score(option) <= 1.0
    assert 0.0 <= router._calculate_speed_score(option, req) <= 1.0
