"""Tests for terradev_cli.core.lora_cost_attribution.

Multi-tenant cost tracking is a business moat: clients need per-adapter and
per-tenant billing. These tests cover the cost math and aggregation paths.
"""

import pytest

from terradev_cli.core.lora_cost_attribution import (
    CostAttributionService,
    CostConfig,
)


@pytest.fixture
def service(tmp_path):
    return CostAttributionService(
        config=CostConfig(),
        config_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_record_inference_cost_tracks_adapter_and_tenant(service):
    """An inference request updates adapter, tenant, and event history."""
    cost = await service.record_inference_cost(
        adapter_name="medical-lora",
        tenant_id="hospital-a",
        replica_id="r1",
        gpu_seconds=3600,
        tokens=5000,
        instance_type="a100",
    )

    assert cost > 0
    record = await service.get_adapter_cost("medical-lora")
    assert record is not None
    assert record.requests_served == 1
    assert record.gpu_hours == 1.0
    assert record.tokens_processed == 5000

    tenant = await service.get_tenant_cost("hospital-a")
    assert tenant is not None
    assert tenant.requests_served == 1
    assert tenant.adapters == {"medical-lora"}


@pytest.mark.asyncio
async def test_record_inference_cost_disabled(service):
    """When tracking is disabled, recording returns zero and does not add state."""
    service.config.enable_tracking = False
    cost = await service.record_inference_cost(
        adapter_name="x", tenant_id=None, replica_id="r1", gpu_seconds=3600, tokens=1000
    )
    assert cost == 0.0
    assert len(service.cost_events) == 0


@pytest.mark.asyncio
async def test_record_storage_cost_updates_adapter(service):
    """Storage cost is attributed to the adapter and tenant."""
    cost = await service.record_storage_cost(
        adapter_name="legal-lora", tenant_id="firm-b", storage_gb=10.0
    )
    assert cost == pytest.approx(1.0)  # $0.10 per GB per month

    record = await service.get_adapter_cost("legal-lora")
    assert record.storage_gb == 10.0
    assert record.total_cost_usd == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_get_cost_summary_and_breakdown(service):
    """Summary and breakdown aggregate events correctly."""
    await service.record_inference_cost(
        "adapter-a", "tenant-1", "r1", 3600, 1000, "a100"
    )
    await service.record_inference_cost(
        "adapter-a", "tenant-1", "r2", 7200, 2000, "a100"
    )
    await service.record_inference_cost(
        "adapter-b", "tenant-2", "r1", 1800, 500, "a10g"
    )

    summary = await service.get_cost_summary(days=30)
    assert summary["total_requests"] == 3
    assert summary["total_cost_usd"] > 0
    assert len(summary["top_adapters"]) == 2
    assert summary["top_adapters"][0]["name"] == "adapter-a"

    breakdown = await service.get_cost_breakdown("adapter-a", days=30)
    assert breakdown["total_requests"] == 2
    assert breakdown["total_cost_usd"] > 0
    assert len(breakdown["cost_by_replica"]) == 2


@pytest.mark.asyncio
async def test_get_warm_pool_recommendations(service):
    """Recommendations flag high-cost adapters and budget overruns."""
    # Large spenders
    await service.record_inference_cost(
        "big-adapter", "tenant-x", "r1", 360_000, 1_000_000, "h100"
    )

    recs = await service.get_warm_pool_recommendations()
    assert any(r["type"] == "keep_warm" for r in recs)

    budget_recs = await service.get_warm_pool_recommendations(budget_limit_usd=1.0)
    assert any(r["type"] == "budget_alert" for r in budget_recs)


@pytest.mark.asyncio
async def test_persistence_roundtrip(tmp_path):
    """Cost data is saved and reloaded from disk."""
    service1 = CostAttributionService(CostConfig(), config_dir=tmp_path)
    await service1.record_inference_cost("persist", "tenant", "r1", 3600, 1000, "a100")

    service2 = CostAttributionService(CostConfig(), config_dir=tmp_path)
    record = await service2.get_adapter_cost("persist")
    assert record is not None
    assert record.requests_served == 1
