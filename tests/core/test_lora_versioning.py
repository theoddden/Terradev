"""Tests for terradev_cli.core.lora_versioning.

LoRA rollback and drift detection are client-critical: a bad adapter push
should be reversible in seconds, and drift should trigger the right action
(rollback, retrain, or monitor).
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from terradev_cli.core.lora_versioning import DriftDetectionResult, LoRAVersioningManager
from terradev_cli.ml_services.lora_registry import AdapterRegistry, AdapterStatus, AdapterVersion


def _version(
    version_id: str,
    status: AdapterStatus = AdapterStatus.REGISTERED,
    quality: float = 0.0,
    created_at: datetime | None = None,
) -> AdapterVersion:
    return AdapterVersion(
        version_id=version_id,
        adapter_name="test-adapter",
        base_model="base",
        path=f"/tmp/{version_id}",
        rank=8,
        created_at=created_at or datetime.now(),
        performance_metrics={"quality": quality} if quality else {},
        status=status,
    )


@pytest.fixture
def registry(tmp_path):
    """Create a real in-memory registry."""
    reg = AdapterRegistry(db_path=str(tmp_path / "registry.db"))
    return reg


@pytest.fixture
def manager(registry):
    return LoRAVersioningManager(registry=registry)


@pytest.mark.asyncio
async def test_rollback_to_previous_stable_version(registry, manager):
    """When no target is specified, rollback picks the most recent stable version."""
    v1 = _version("v1", quality=0.85)
    v2 = _version("v2", quality=0.90)
    v3 = _version("v3", status=AdapterStatus.ACTIVE, quality=0.60)

    for v in [v1, v2, v3]:
        registry.register_adapter("test-adapter", v)
    registry.mark_version_active("test-adapter", "v3")

    result = await manager.rollback_adapter("test-adapter")

    assert result.success is True
    assert result.from_version_id == "v3"
    assert result.to_version_id == "v2"
    assert registry.get_active_version("test-adapter").version_id == "v2"


@pytest.mark.asyncio
async def test_rollback_to_specific_target_version(registry, manager):
    """Rollback can target an explicit version id."""
    v1 = _version("v1", quality=0.85)
    v2 = _version("v2", quality=0.90)

    registry.register_adapter("test-adapter", v1)
    registry.register_adapter("test-adapter", v2)
    registry.mark_version_active("test-adapter", "v2")

    result = await manager.rollback_adapter("test-adapter", target_version_id="v1")

    assert result.success is True
    assert result.to_version_id == "v1"
    assert registry.get_active_version("test-adapter").version_id == "v1"


@pytest.mark.asyncio
async def test_rollback_fails_with_no_versions(manager):
    """Rolling back an unknown adapter is a clean failure."""
    result = await manager.rollback_adapter("unknown-adapter")
    assert result.success is False
    assert "No versions" in result.error


@pytest.mark.asyncio
async def test_rollback_fails_with_missing_target(registry, manager):
    """Rolling back to a non-existent version returns a clear error."""
    v1 = _version("v1")
    registry.register_adapter("test-adapter", v1)
    result = await manager.rollback_adapter("test-adapter", target_version_id="missing")
    assert result.success is False
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_drift_without_drift_service_returns_monitor(manager):
    """If no drift service is attached, the recommendation is always monitor."""
    v1 = _version("v1", quality=0.85)
    registry.register_adapter("test-adapter", v1)
    registry.mark_version_active("test-adapter", "v1")

    result = await manager.detect_drift("test-adapter")
    assert isinstance(result, DriftDetectionResult)
    assert result.has_drift is False
    assert result.recommended_action == "monitor"


@pytest.mark.asyncio
async def test_drift_detected_with_drift_service(registry):
    """Drift service with a score drop produces rollback/retrain recommendations."""
    v1 = _version("v1", quality=0.85)
    registry.register_adapter("test-adapter", v1)
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
    """Severe drift triggers the full rollback path."""
    v1 = _version("v1", quality=0.85)
    v2 = _version("v2", quality=0.90)
    v3 = _version("v3", status=AdapterStatus.ACTIVE, quality=0.60)

    for v in [v1, v2, v3]:
        registry.register_adapter("test-adapter", v)
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
    """Version history exposes status and active flags."""
    v1 = _version("v1", quality=0.85)
    registry.register_adapter("test-adapter", v1)
    registry.mark_version_active("test-adapter", "v1")

    history = manager.get_version_history("test-adapter")
    assert len(history) == 1
    assert history[0]["version_id"] == "v1"
    assert history[0]["is_active"] is True


def test_compare_versions_returns_delta(registry, manager):
    """Comparing two versions calculates per-metric deltas."""
    v1 = _version("v1", quality=0.80)
    v1.performance_metrics = {"quality": 0.80, "latency": 10.0}
    v2 = _version("v2", quality=0.90)
    v2.performance_metrics = {"quality": 0.90, "latency": 8.0}

    registry.register_adapter("test-adapter", v1)
    registry.register_adapter("test-adapter", v2)

    comparison = manager.compare_versions("test-adapter", "v1", "v2")
    assert comparison["performance_delta"]["quality"] == pytest.approx(0.10)
    assert comparison["performance_delta"]["latency"] == pytest.approx(-2.0)


def test_compare_versions_fails_when_missing(registry, manager):
    """Comparing non-existent versions returns an error."""
    result = manager.compare_versions("test-adapter", "missing-a", "missing-b")
    assert "error" in result
