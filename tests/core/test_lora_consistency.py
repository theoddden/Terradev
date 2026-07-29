"""Tests for terradev_cli.core.lora_consistency.

LoRA cross-replica consistency is the difference between a 1-node demo and a
production multi-replica deployment. These tests cover broadcast, discovery,
and sync behavior.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from terradev_cli.core.lora_consistency import LoRAConsistencyManager, ReplicaInfo
from terradev_cli.ml_services.lora_registry import AdapterRegistry
from terradev_cli.ml_services.vllm_service import LoRAModule


@pytest.fixture
def registry(tmp_path):
    return AdapterRegistry(db_path=str(tmp_path / "registry.db"))


@pytest.fixture
def manager(registry):
    return LoRAConsistencyManager(
        registry=registry,
        replicas=[
            {"replica_id": "r1", "host": "10.0.0.1", "port": 8000},
            {"replica_id": "r2", "host": "10.0.0.2", "port": 8000},
        ],
    )


@pytest.mark.asyncio
async def test_static_replica_discovery(manager):
    """Static replica config is parsed into ReplicaInfo objects."""
    replicas = [
        {"replica_id": "r3", "host": "10.0.0.3", "port": 8001},
    ]
    result = await manager.discover_replicas_static(replicas)
    assert len(result) == 3
    assert "r3" in manager.replicas
    assert manager.replicas["r3"].port == 8001


def test_healthy_replicas_after_marking(manager):
    """Healthy replicas are tracked and can be queried."""
    manager.replicas["r1"].is_healthy = False
    healthy = manager.get_healthy_replicas()
    assert len(healthy) == 1
    assert healthy[0].replica_id == "r2"


@pytest.mark.asyncio
async def test_broadcast_load_to_replicas_requires_quorum(manager, registry):
    """Loading needs a majority of replicas to succeed."""
    version = registry.register_adapter(
        adapter_name="test",
        base_model="base",
        path="/tmp/test",
        rank=8,
        performance_metrics={"quality": 0.9},
    )
    adapter = LoRAModule(name="test", path="/tmp/test")

    # Two replicas: one succeeds, one fails -> no quorum
    manager._load_on_replica = AsyncMock(side_effect=[
        {"status": "loaded"},
        {"status": "error", "error": "timeout"},
    ])

    result = await manager.broadcast_load_to_replicas(adapter, version.version_id)
    assert result["status"] == "failed"
    assert "Quorum" in result["error"]


@pytest.mark.asyncio
async def test_broadcast_load_succeeds_with_quorum(manager, registry):
    """When all replicas succeed, broadcast load returns success."""
    version = registry.register_adapter(
        adapter_name="test",
        base_model="base",
        path="/tmp/test",
        rank=8,
    )
    adapter = LoRAModule(name="test", path="/tmp/test")

    manager._load_on_replica = AsyncMock(return_value={"status": "loaded"})

    result = await manager.broadcast_load_to_replicas(adapter, version.version_id)
    assert result["status"] == "success"
    assert result["successful"] == 2


@pytest.mark.asyncio
async def test_broadcast_unload_from_replicas(manager, registry):
    """Unload broadcast tracks successes and failures."""
    manager._unload_from_replica = AsyncMock(side_effect=[
        {"status": "unloaded"},
        {"status": "error", "error": "not loaded"},
    ])

    result = await manager.broadcast_unload_from_replicas("test")
    assert result["successful"] == 1
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_verify_consistency_no_replicas(manager, registry):
    """Consistency is trivially true if there are no expected replicas."""
    manager.replicas.clear()
    version = registry.register_adapter("test", "base", "/tmp/test", 8)
    result = await manager.verify_consistency("test", version.version_id)
    assert result["is_consistent"] is True


@pytest.mark.asyncio
async def test_verify_consistency_detects_missing_and_mismatched(manager, registry):
    """Consistency detects replicas missing an adapter or running the wrong version."""
    v1 = registry.register_adapter("test", "base", "/tmp/v1", 8)
    v2 = registry.register_adapter("test", "base", "/tmp/v2", 8)
    registry.mark_version_active("test", v1.version_id)

    # r1 has v1, r2 has v2, r3 missing entirely
    manager.replicas["r3"] = ReplicaInfo(
        replica_id="r3", host="10.0.0.3", port=8000,
        last_heartbeat=datetime.now(),
    )
    registry.record_replica_load("r1", "test", v1.version_id)
    registry.record_replica_load("r2", "test", v2.version_id)

    result = await manager.verify_consistency("test", v1.version_id)
    assert result["is_consistent"] is False
    assert "r3" in result["missing_replicas"]
    assert len(result["version_mismatches"]) == 1
    assert result["version_mismatches"][0]["replica_id"] == "r2"
