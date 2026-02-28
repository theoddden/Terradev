#!/usr/bin/env python3
"""
Tests for CheckpointManager — manifest-based atomic checkpointing.

Covers:
  1. Local storage backend (put/get/exists/delete/list_prefix)
  2. Save checkpoint with manifest + parallel shard writes
  3. Restore checkpoint (latest or by step)
  4. List / promote / delete
  5. Retention policy enforcement
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.core.checkpoint_manager import (
    CheckpointManager,
    LocalStorage,
    CheckpointManifest,
)
from terradev_cli.core.job_state_manager import JobStateManager


@pytest.fixture
def tmp_env(tmp_path):
    """Create temp dirs for checkpoint storage and job DB."""
    ckpt_dir = str(tmp_path / "checkpoints")
    db_path = str(tmp_path / "jobs.db")
    state_mgr = JobStateManager(db_path=db_path)
    yield ckpt_dir, state_mgr
    state_mgr.close()


# ── LocalStorage Backend ─────────────────────────────────────────────────────


class TestLocalStorage:
    def test_put_and_get(self, tmp_path):
        store = LocalStorage()
        src = str(tmp_path / "source.bin")
        dst = str(tmp_path / "dest.bin")
        Path(src).write_bytes(b"hello checkpoint")
        assert store.put(src, dst) is True
        assert Path(dst).read_bytes() == b"hello checkpoint"

    def test_get_reverse(self, tmp_path):
        store = LocalStorage()
        src = str(tmp_path / "remote.bin")
        dst = str(tmp_path / "local.bin")
        Path(src).write_bytes(b"data")
        assert store.get(src, dst) is True
        assert Path(dst).read_bytes() == b"data"

    def test_exists(self, tmp_path):
        store = LocalStorage()
        f = str(tmp_path / "exists.bin")
        assert store.exists(f) is False
        Path(f).write_bytes(b"x")
        assert store.exists(f) is True

    def test_delete_file(self, tmp_path):
        store = LocalStorage()
        f = str(tmp_path / "to_delete.bin")
        Path(f).write_bytes(b"x")
        assert store.delete(f) is True
        assert not Path(f).exists()

    def test_delete_directory(self, tmp_path):
        store = LocalStorage()
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "file.bin").write_bytes(b"x")
        assert store.delete(str(d)) is True
        assert not d.exists()

    def test_list_prefix(self, tmp_path):
        store = LocalStorage()
        d = tmp_path / "listing"
        d.mkdir()
        (d / "a.bin").write_bytes(b"x")
        (d / "b.bin").write_bytes(b"y")
        result = store.list_prefix(str(d))
        assert len(result) == 2


# ── CheckpointManager Save/Restore ──────────────────────────────────────────


class TestCheckpointSaveRestore:
    def _make_shards(self, tmp_path, n=4):
        """Create dummy shard files."""
        shards = {}
        for i in range(n):
            shard = tmp_path / f"shard_{i}.pt"
            shard.write_bytes(os.urandom(128))
            shards[i] = str(shard)
        return shards

    def test_save_creates_manifest(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(
            base_dir=ckpt_dir,
            state_manager=state_mgr,
        )
        job = state_mgr.create_job("j", "torchrun", {}, ["n1"])
        shards = self._make_shards(tmp_path)

        manifest = cm.save(
            job_id=job.id,
            step=100,
            shard_paths=shards,
        )
        assert isinstance(manifest, CheckpointManifest)
        assert manifest.step == 100
        assert manifest.job_id == job.id
        assert manifest.shard_count == 4

    def test_restore_latest(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(base_dir=ckpt_dir, state_manager=state_mgr)
        job = state_mgr.create_job("j", "torchrun", {}, [])

        cm.save(job.id, 100, self._make_shards(tmp_path, 2))
        cm.save(job.id, 200, self._make_shards(tmp_path, 2))

        restored = cm.restore(job.id)
        assert restored.step == 200

    def test_restore_specific_step(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(base_dir=ckpt_dir, state_manager=state_mgr)
        job = state_mgr.create_job("j", "torchrun", {}, [])

        cm.save(job.id, 100, self._make_shards(tmp_path, 2))
        cm.save(job.id, 200, self._make_shards(tmp_path, 2))

        restored = cm.restore(job.id, step=100)
        assert restored.step == 100

    def test_list_checkpoints(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(base_dir=ckpt_dir, state_manager=state_mgr)
        job = state_mgr.create_job("j", "torchrun", {}, [])

        cm.save(job.id, 100, self._make_shards(tmp_path, 1))
        cm.save(job.id, 200, self._make_shards(tmp_path, 1))
        cm.save(job.id, 300, self._make_shards(tmp_path, 1))

        ckpts = cm.list(job.id)
        assert len(ckpts) >= 3

    def test_delete_checkpoint(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(base_dir=ckpt_dir, state_manager=state_mgr)
        job = state_mgr.create_job("j", "torchrun", {}, [])

        manifest = cm.save(job.id, 100, self._make_shards(tmp_path, 1))
        cm.delete(job.id, manifest.checkpoint_id)
        # Should not raise; verify list no longer includes it
        ckpts = cm.list(job.id)
        ids = [c.get("checkpoint_id", c.get("id", "")) for c in ckpts]
        assert manifest.checkpoint_id not in ids


# ── Retention Policy ──────────────────────────────────────────────────────────


class TestRetentionPolicy:
    def test_retention_keeps_last_n(self, tmp_env, tmp_path):
        ckpt_dir, state_mgr = tmp_env
        cm = CheckpointManager(
            base_dir=ckpt_dir,
            state_manager=state_mgr,
            retention=2,
        )
        job = state_mgr.create_job("j", "torchrun", {}, [])

        for step in [100, 200, 300, 400]:
            shard = tmp_path / f"shard_{step}.pt"
            shard.write_bytes(os.urandom(64))
            cm.save(job.id, step, {0: str(shard)})

        # With retention=2, oldest checkpoints should be cleaned up
        ckpts = cm.list(job.id)
        steps = [c.get("step", 0) for c in ckpts]
        # At minimum, the two most recent should be present
        assert 400 in steps
        assert 300 in steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
