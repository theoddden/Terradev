"""Tests for terradev_cli.core.parallel_provisioner.

ParallelProvisioner coordinates multi-cloud provisioning and builds cheapest-spread
allocation plans.
"""

import pytest

from terradev_cli.core.parallel_provisioner import (
    ParallelProvisioner,
    ProvisionResult,
)


class FakeProvider:
    """A stub async provider that returns a successful provision."""

    def __init__(self, raise_error=None):
        self.raise_error = raise_error

    async def provision_instance(self, gpu_type, region, spot):
        if self.raise_error:
            raise self.raise_error
        return {"instance_id": f"id-{gpu_type}", "price_per_hour": 0.5}


class FakeFactory:
    """A stub factory returning FakeProvider instances."""

    def __init__(self, provider_error=None):
        self.provider_error = provider_error

    def create_provider(self, provider_name, credentials):
        return FakeProvider(raise_error=self.provider_error)


def test_provision_result_to_dict():
    """ProvisionResult serializes to a dictionary."""
    r = ProvisionResult(
        provider="runpod",
        region="us-east-1",
        instance_id="i-1",
        gpu_type="A100",
        price_hr=1.5,
        spot=False,
        status="active",
        error=None,
        elapsed_ms=100.0,
    )
    d = r.to_dict()
    assert d["provider"] == "runpod"
    assert d["status"] == "active"
    assert d["elapsed_ms"] == 100.0


@pytest.mark.asyncio
async def test_provision_parallel_success(monkeypatch):
    """provision_parallel runs providers concurrently and returns group results."""
    pp = ParallelProvisioner()
    monkeypatch.setattr(pp, "factory", FakeFactory())

    allocations = [
        {
            "provider": "runpod",
            "credentials": {"api_key": "k"},
            "gpu_type": "A100",
            "region": "us-east-1",
            "spot": False,
        },
        {
            "provider": "vastai",
            "credentials": {"api_key": "k"},
            "gpu_type": "A100",
            "region": "us-east-1",
            "spot": True,
        },
    ]

    group_id, results = await pp.provision_parallel(allocations)
    assert group_id.startswith("pg_")
    assert len(results) == 2
    assert all(r.status == "active" for r in results)
    assert all(r.error is None for r in results)


@pytest.mark.asyncio
async def test_provision_parallel_failure(monkeypatch):
    """Provision failures are captured with error text."""
    pp = ParallelProvisioner()
    monkeypatch.setattr(pp, "factory", FakeFactory(provider_error=RuntimeError("no capacity")))

    group_id, results = await pp.provision_parallel(
        [{"provider": "runpod", "gpu_type": "A100"}]
    )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "no capacity" in results[0].error


def test_build_cheapest_spread():
    """build_cheapest_spread respects count and per-provider caps."""
    pp = ParallelProvisioner()
    quotes = [
        {"provider": "RunPod", "gpu_type": "A100", "region": "us-east-1", "price": 0.5, "availability": "spot"},
        {"provider": "RunPod", "gpu_type": "A100", "region": "us-west-2", "price": 0.6, "availability": "on_demand"},
        {"provider": "Vast.ai", "gpu_type": "A100", "region": "us-east-1", "price": 0.4, "availability": "on_demand"},
        {"provider": "Vast.ai", "gpu_type": "A100", "region": "eu-west-1", "price": 0.45, "availability": "spot"},
    ]

    allocations = pp.build_cheapest_spread(quotes, count=3, max_price=0.55)
    assert len(allocations) == 3
    providers = [a["provider"] for a in allocations]
    # Vast.ai lowercases to "vast.ai" (dot preserved), not "vastai"
    assert providers.count("vast.ai") == 2
    assert providers.count("runpod") == 1
    assert allocations[0]["provider"] == "vast.ai"


def test_build_cheapest_spread_empty():
    """Empty quotes or quotes above max price return an empty plan."""
    pp = ParallelProvisioner()
    assert pp.build_cheapest_spread([], count=1) == []
    assert pp.build_cheapest_spread(
        [{"provider": "RunPod", "price": 1.0}], count=1, max_price=0.5
    ) == []
