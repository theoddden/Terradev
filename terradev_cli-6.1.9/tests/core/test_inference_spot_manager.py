"""Tests for terradev_cli.core.inference_spot_manager.

Inference-on-spot manager snapshots and restores inference state for spot
preemption handling.
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from terradev_cli.core.inference_spot_manager import (
    InferenceSpotConfig,
    InferenceSpotManager,
    InferenceSpotState,
)


class FakeKVManager:
    def __init__(self, *args, **kwargs):
        self.saved = []
        self.restored = []

    async def save_checkpoint(self, **kwargs):
        self.saved.append(kwargs)
        return type("Checkpoint", (), {"checkpoint_id": "kv-1"})()

    async def load_checkpoint(self, checkpoint_id):
        self.restored.append(checkpoint_id)
        return type("Checkpoint", (), {"checkpoint_id": "kv-1"})()

    async def restore_checkpoint(self, checkpoint_id, target_path):
        self.restored.append((checkpoint_id, target_path))


class FakeModelManager:
    def __init__(self, *args, **kwargs):
        pass

    async def save_checkpoint(self, **kwargs):
        return "model-1"

    async def restore_checkpoint(self, checkpoint_id, target_path):
        pass


@pytest.fixture
def manager(tmp_path, monkeypatch):
    config = InferenceSpotConfig(
        enable_spot_checkpointing=True,
        checkpoint_dir=str(tmp_path),
        spot_termination_check_interval_seconds=1,
    )
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.KVCacheCheckpointManager",
        FakeKVManager,
    )
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.CheckpointManager",
        FakeModelManager,
    )
    return InferenceSpotManager(config)


def test_config_defaults():
    """InferenceSpotConfig has sensible defaults."""
    config = InferenceSpotConfig()
    assert config.enable_spot_checkpointing is True
    assert config.storage_backend == "s3"


def test_manager_initializes_checkpoints_dir(tmp_path, monkeypatch):
    """InferenceSpotManager creates its checkpoint directory."""
    config = InferenceSpotConfig(
        checkpoint_dir=str(tmp_path / "spot"),
        enable_spot_checkpointing=True,
    )
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.KVCacheCheckpointManager",
        FakeKVManager,
    )
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.CheckpointManager",
        FakeModelManager,
    )
    mgr = InferenceSpotManager(config)
    assert mgr.checkpoint_dir.is_dir()


@pytest.mark.asyncio
async def test_start_and_stop_spot_monitoring(manager, monkeypatch):
    """Spot monitoring can be started and stopped."""
    monkeypatch.setattr(manager, "_check_spot_termination", AsyncMock(return_value=False))

    await manager.start_spot_monitoring("ep-1", "llama-7b")
    assert manager.active_state is not None
    assert manager.active_state.endpoint_name == "ep-1"

    await manager.stop_spot_monitoring()
    assert manager._spot_monitor_task.cancelled() or manager._spot_monitor_task.done()


@pytest.mark.asyncio
async def test_handle_spot_termination(manager, monkeypatch):
    """Spot termination handler snapshots state and marks saved."""
    await manager.start_spot_monitoring("ep-1", "llama-7b")
    manager.active_state.state = "active"

    monkeypatch.setattr(manager, "_trigger_vllm_sleep_mode", AsyncMock())
    monkeypatch.setattr(
        manager,
        "_snapshot_kv_cache",
        AsyncMock(return_value=type("CP", (), {"checkpoint_id": "kv-1"})()),
    )
    monkeypatch.setattr(manager, "_snapshot_model_state", AsyncMock(return_value="model-1"))
    monkeypatch.setattr(manager, "_capture_in_flight_requests", AsyncMock(return_value=[{"id": "r1"}]))
    monkeypatch.setattr(manager, "_trigger_reprovision", AsyncMock())

    await manager._handle_spot_termination()
    assert manager.active_state.state == "saved"
    assert manager.active_state.kv_cache_checkpoint_id == "kv-1"
    assert manager.active_state.model_checkpoint_id == "model-1"


@pytest.mark.asyncio
async def test_check_spot_termination_subprocess(tmp_path, monkeypatch):
    """Spot termination check runs cloud metadata curls."""
    import subprocess

    class FakeResult:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.KVCacheCheckpointManager",
        FakeKVManager,
    )
    monkeypatch.setattr(
        "terradev_cli.core.inference_spot_manager.CheckpointManager",
        FakeModelManager,
    )

    config = InferenceSpotConfig(
        checkpoint_dir=str(tmp_path),
        nvme_path=str(tmp_path),
    )
    manager = InferenceSpotManager(config)
    result = await manager._check_spot_termination()
    assert result is False


@pytest.mark.asyncio
async def test_restore_checkpoint(manager, monkeypatch):
    """restore_checkpoint restores KV and model state."""
    await manager.start_spot_monitoring("ep-1", "llama-7b")

    monkeypatch.setattr(manager, "_restore_kv_cache", AsyncMock())
    monkeypatch.setattr(manager, "_restore_model_state", AsyncMock())
    monkeypatch.setattr(manager, "_replay_in_flight_requests", AsyncMock())

    assert await manager.restore_checkpoint("ckpt-1") is True


@pytest.mark.asyncio
async def test_get_instance_id_fallback(manager):
    """Instance ID falls back to 'unknown' when not on cloud metadata."""
    assert manager._get_instance_id() == "unknown"


def test_inference_spot_state():
    """InferenceSpotState dataclass holds checkpoint metadata."""
    state = InferenceSpotState(
        checkpoint_id="ckpt-1",
        model_id="llama-7b",
        endpoint_name="ep-1",
        provider="runpod",
        region="us-east-1",
        instance_id="i-1",
        created_at=None,
        expires_at=None,
    )
    assert state.state == "active"
