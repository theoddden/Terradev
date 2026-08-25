"""Tests for terradev_cli.core.lora_versioning.

LoRA rollback and drift detection are client-critical: a bad adapter push
should be reversible in seconds, and drift should trigger the right action
(rollback, retrain, or monitor).
"""

from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from terradev_cli.core.lora_versioning import DriftDetectionResult, LoRAVersioningManager
from terradev_cli.ml_services.lora_registry import AdapterRegistry, AdapterStatus


def _version(
    registry: AdapterRegistry,
    version_id: str,
    status: AdapterStatus = AdapterStatus.REGISTERED,
    quality: float = 0.0,
    created_at: Optional[datetime] = None,
):
    """Register and return an adapter version with the requested id."""
    v = registry.register_adapter(
        adapter_name="test-adapter",
        base_model="base",
        path=f"/tmp/{version_id}",
        rank=8,
        performance_metrics={"quality": quality} if quality else {},
        version_id=version_id,
        status=status,
    )
    return v


@pytest.fixture
def registry(tmp_path):
    return AdapterRegistry(db_path=tmp_path / "registry.db")


@pytest.fixture
def manager(registry):
    return LoRAVersioningManager(registry=registry)


@pytest.mark.asyncio
async def test_rollback_to_previous_stable_version(registry, manager):
    v1 = _version(registry, "v1", quality=0.85)
    v2 = _version(registry, "v2", quality=0.90)
    v3 = _version(registry, "v3", AdapterStatus.ACTIVE, quality=0.90)
    registry.mark_version_active("test-adapter", "v3")

    result = await manager.rollback_adapter("test-adapter")

    assert result.success is True
    assert result.from_version_id == "v3"
    assert result.to_version_id == "v2"
    assert registry.get_active_version("test-adapter").version_id == "v2"


@pytest.mark.asyncio
async def test_rollback_to_specific_target_version(registry, manager):
    v1 = _version(registry, "v1", quality=0.85)
    v2 = _version(registry, "v2", quality=0.90)
    registry.mark_version_active("test-adapter", "v2")

    result = await manager.rollback_adapter("test-adapter", target_version_id="v1")

    assert result.success is True
    assert result.to_version_id == "v1"
    assert registry.get_active_version("test-adapter").version_id == "v1"


@pytest.mark.asyncio
async def test_rollback_fails_with_no_versions(manager):
    result = await manager.rollback_adapter("unknown-adapter")
    assert result.success is False
    assert "No versions" in result.error


@pytest.mark.asyncio
async def test_rollback_fails_with_missing_target(registry, manager):
    _version(registry, "v1")
    result = await manager.rollback_adapter("test-adapter", target_version_id="missing")
    assert result.success is False
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_drift_without_drift_service_returns_monitor(registry, manager):
    _version(registry, "v1", AdapterStatus.ACTIVE, quality=0.85)
    registry.mark_version_active("test-adapter", "v1")

    result = await manager.detect_drift("test-adapter")
    assert isinstance(result, DriftDetectionResult)
    assert result.has_drift is False
    assert result.recommended_action == "monitor"


@pytest.mark.asyncio
async def test_drift_detected_with_drift_service(registry):
    _version(registry, "v1", AdapterStatus.ACTIVE, quality=0.85)
    registry.mark_version_active("test-adapter", "v1")

    drift_service = AsyncMock()
    drift_service.detect_adapter_drift = AsyncMock(
        return_value={"has_drift": True, "current_score": 0.50}
    )

    manager = LoRAVersioningManager(registry=registry, drift_service=drift_service)
    result = await manager.detect_drift("test-adapter", drift_threshold=0.1)

    assert result.has_drift is True
    assert result.recommended_action == "rollback"
    assert result.drift_magnitude > 0


@pytest.mark.asyncio
async def test_auto_rollback_on_severe_drift(registry):
    v1 = _version(registry, "v1", quality=0.85)
    v2 = _version(registry, "v2", quality=0.90)
    v3 = _version(registry, "v3", AdapterStatus.ACTIVE, quality=0.90)
    registry.mark_version_active("test-adapter", "v3")

    drift_service = AsyncMock()
    drift_service.detect_adapter_drift = AsyncMock(
        return_value={"has_drift": True, "current_score": 0.50}
    )

    manager = LoRAVersioningManager(registry=registry, drift_service=drift_service)
    result = await manager.auto_rollback_on_drift("test-adapter", drift_threshold=0.1)

    assert result["status"] == "rolled_back"
    assert result["action_taken"] == "rollback"
    assert result["rollback_result"].success is True


def test_version_history_includes_active_flag(registry, manager):
    _version(registry, "v1", quality=0.85)
    registry.mark_version_active("test-adapter", "v1")

    history = manager.get_version_history("test-adapter")
    assert len(history) == 1
    assert history[0]["version_id"] == "v1"
    assert history[0]["is_active"] is True


def test_compare_versions_returns_delta(registry, manager):
    v1 = _version(registry, "v1", quality=0.80)
    v1.performance_metrics = {"quality": 0.80, "latency": 10.0}
    v2 = _version(registry, "v2", quality=0.90)
    v2.performance_metrics = {"quality": 0.90, "latency": 8.0}

    registry.update_performance_metrics(v1.version_id, v1.performance_metrics)
    registry.update_performance_metrics(v2.version_id, v2.performance_metrics)

    comparison = manager.compare_versions("test-adapter", v1.version_id, v2.version_id)
    assert comparison["performance_delta"]["quality"] == pytest.approx(0.10)
    assert comparison["performance_delta"]["latency"] == pytest.approx(-2.0)


def test_compare_versions_fails_when_missing(registry, manager):
    result = manager.compare_versions("test-adapter", "missing-a", "missing-b")
    assert "error" in result
