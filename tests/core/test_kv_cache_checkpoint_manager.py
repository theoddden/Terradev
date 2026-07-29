"""Tests for terradev_cli.core.kv_cache_checkpoint_manager.

These cover the P0 KV cache checkpointing feature end-to-end: create, restore,
spot-termination save, and local storage round-trip.  They guard the client
promise that a preempted spot instance can serialize state and resume.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from terradev_cli.core.kv_cache_checkpoint_manager import (
    CheckpointConfig,
    CheckpointState,
    KVCacheCheckpointManager,
)


def _config(tmp_path, **overrides):
    return CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        nvme_path=str(tmp_path / "nvme"),
        storage_backend="local",
        compression_enabled=True,
        **overrides,
    )


@pytest.mark.asyncio
async def test_create_and_restore_local_checkpoint(tmp_path):
    """Create a checkpoint, restore it, and verify data + state."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"layer": 3, "kv_cache": [[1.0, 2.0], [3.0, 4.0]]}
    checkpoint_id = await manager.create_checkpoint(
        model_id="test-model",
        request_id="req-1",
        kv_cache_data=data,
        context_length=1024,
        batch_size=1,
        num_layers=4,
        num_heads=8,
        head_dim=64,
    )

    assert checkpoint_id is not None
    assert checkpoint_id in manager.checkpoints
    cp = manager.checkpoints[checkpoint_id]
    assert cp.state == CheckpointState.SAVED

    restored = await manager.restore_checkpoint(checkpoint_id, "req-2")
    assert restored == data
    assert cp.state == CheckpointState.LOADED
    assert manager.active_checkpoints["req-2"] == cp

    await manager.cleanup()


@pytest.mark.asyncio
async def test_restore_unknown_checkpoint_returns_none(tmp_path):
    """Restoring a non-existent checkpoint is safe and returns None."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()
    assert await manager.restore_checkpoint("missing-id", "req-1") is None
    await manager.cleanup()


@pytest.mark.asyncio
async def test_restore_expired_checkpoint_returns_none(tmp_path):
    """Expired checkpoints are rejected."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"tokens": [1, 2, 3]}
    checkpoint_id = await manager.create_checkpoint(
        model_id="m",
        request_id="r",
        kv_cache_data=data,
        context_length=8,
        batch_size=1,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )

    cp = manager.checkpoints[checkpoint_id]
    cp.expires_at = datetime.now() - timedelta(hours=1)

    assert await manager.restore_checkpoint(checkpoint_id, "r2") is None
    assert cp.state == CheckpointState.EXPIRED
    await manager.cleanup()


@pytest.mark.asyncio
async def test_disabled_checkpointing_returns_none(tmp_path):
    """When checkpointing is disabled, create returns None and no data is saved."""
    config = _config(tmp_path, enable_checkpointing=False)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    assert (
        await manager.create_checkpoint(
            model_id="m",
            request_id="r",
            kv_cache_data={"x": 1},
            context_length=8,
            batch_size=1,
            num_layers=2,
            num_heads=2,
            head_dim=32,
        )
        is None
    )
    assert len(manager.checkpoints) == 0
    await manager.cleanup()


@pytest.mark.asyncio
async def test_checksum_mismatch_fails_restore(tmp_path):
    """Corrupting the saved file after creation causes restore to fail."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"value": 42}
    checkpoint_id = await manager.create_checkpoint(
        model_id="m",
        request_id="r",
        kv_cache_data=data,
        context_length=8,
        batch_size=1,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )

    cp = manager.checkpoints[checkpoint_id]
    # Corrupt the on-disk bytes
    Path(cp.storage_path).write_bytes(b"not-the-real-data")

    assert await manager.restore_checkpoint(checkpoint_id, "r2") is None
    assert cp.state == CheckpointState.FAILED
    await manager.cleanup()


@pytest.mark.asyncio
async def test_spot_termination_saves_active_checkpoints(tmp_path):
    """Spot-termination notice triggers saving of active checkpoints."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"tokens": list(range(10))}
    checkpoint_id = await manager.create_checkpoint(
        model_id="m",
        request_id="r",
        kv_cache_data=data,
        context_length=8,
        batch_size=1,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )

    cp = manager.checkpoints[checkpoint_id]
    cp.state = CheckpointState.ACTIVE

    success = await manager.handle_spot_termination("i-123", "aws", "us-east-1")
    assert success is True
    assert cp.provider == "aws"
    assert cp.region == "us-east-1"
    assert cp.instance_id == "i-123"

    await manager.cleanup()


@pytest.mark.asyncio
async def test_restore_on_new_instance_roundtrip(tmp_path):
    """A new instance can discover and restore previously saved checkpoints."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"layer": 7}
    checkpoint_id = await manager.create_checkpoint(
        model_id="m",
        request_id="r",
        kv_cache_data=data,
        context_length=8,
        batch_size=1,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )

    result = await manager.restore_on_new_instance("i-999", "aws", "us-east-1")
    assert result["successful_restores"] == 1
    assert result["total_checkpoints"] == 1
    assert checkpoint_id in result["restore_results"]
    assert result["restore_results"][checkpoint_id]["success"] is True
    assert result["restore_results"][checkpoint_id]["kv_cache_data"] == data

    await manager.cleanup()


@pytest.mark.asyncio
async def test_get_status_reflects_state(tmp_path):
    """Status returns the expected structure after operations."""
    config = _config(tmp_path)
    manager = KVCacheCheckpointManager(config)
    await manager.initialize()

    data = {"x": 1}
    await manager.create_checkpoint(
        model_id="m",
        request_id="r",
        kv_cache_data=data,
        context_length=8,
        batch_size=1,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )

    status = manager.get_status()
    assert status["active_checkpoints"] == 1
    assert status["total_checkpoints"] == 1
    assert status["storage_backend"] == "local"
    assert status["metrics"]["total_created"] == 1

    await manager.cleanup()
