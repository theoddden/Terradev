"""Tests for terradev_cli.core.weight_streaming_manager.

These guard the P0 weight-streaming cold-start feature.  The goal is fast,
verified cold starts for large models without needing real object stores or
GPU runtimes.
"""

from unittest.mock import AsyncMock

import pytest

from terradev_cli.core.weight_streaming_manager import (
    LayerChunk,
    StreamingConfig,
    StreamingState,
    WeightStreamingManager,
)


def _config(tmp_path, **overrides):
    return StreamingConfig(
        model_id="test-model",
        model_path="http://example.com/model",
        framework="custom",
        total_layers=4,
        chunk_size_layers=2,
        parallel_downloads=2,
        parallel_computes=2,
        min_chunks_for_compute=1,
        storage_backend="local",
        timeout_seconds=5,
        **overrides,
    )


@pytest.mark.asyncio
async def test_initialize_generates_layer_chunks(tmp_path):
    """Initialize breaks the model into the expected layer chunks."""
    config = _config(tmp_path)
    manager = WeightStreamingManager(config)

    assert await manager.initialize() is True
    assert manager.state == StreamingState.DOWNLOADING
    assert len(manager.chunks) == 2

    chunk = manager.chunks[0]
    assert isinstance(chunk, LayerChunk)
    assert chunk.layer_start == 0
    assert chunk.layer_end == 2
    assert chunk.downloaded is False
    assert chunk.loaded is False


@pytest.mark.asyncio
async def test_start_streaming_with_mocked_workers(tmp_path):
    """Full streaming run completes when downloads and loads are mocked."""
    config = _config(tmp_path)
    manager = WeightStreamingManager(config)
    assert await manager.initialize() is True

    # Bypass real HTTP / GPU calls
    manager._download_chunk = AsyncMock(return_value=True)
    manager._load_chunk = AsyncMock(return_value=True)

    callback = AsyncMock()
    completed = await manager.start_streaming(first_token_callback=callback)

    assert completed is True
    assert manager.state == StreamingState.COMPLETED
    assert all(chunk.downloaded for chunk in manager.chunks)
    assert all(chunk.loaded for chunk in manager.chunks)
    assert manager.metrics.first_token_time is not None
    assert callback.called


@pytest.mark.asyncio
async def test_start_streaming_fails_from_bad_state(tmp_path):
    """Starting streaming before initialize returns False."""
    config = _config(tmp_path)
    manager = WeightStreamingManager(config)
    assert await manager.start_streaming() is False


@pytest.mark.asyncio
async def test_get_status_and_metrics_reflect_progress(tmp_path):
    """Status and metrics expose download/load progress."""
    config = _config(tmp_path)
    manager = WeightStreamingManager(config)
    await manager.initialize()

    manager._download_chunk = AsyncMock(return_value=True)
    manager._load_chunk = AsyncMock(return_value=True)
    await manager.start_streaming()

    status = manager.get_status()
    assert status["state"] == "completed"
    assert status["downloaded_chunks"] == 2
    assert status["loaded_chunks"] == 2
    assert status["progress"]["download"] == 100.0
    assert status["progress"]["load"] == 100.0

    metrics = manager.get_metrics()
    assert metrics.total_bytes_downloaded > 0
    assert metrics.total_layers_loaded == 4
    assert metrics.time_to_first_token_ms is not None
    assert metrics.total_time_ms is not None


def test_generate_checksum_is_stable():
    """Checksums are deterministic for a given chunk id."""
    config = _config()
    manager = WeightStreamingManager(config)
    c1 = manager._generate_checksum("chunk_0_1")
    c2 = manager._generate_checksum("chunk_0_1")
    assert c1 == c2
    assert c1 != manager._generate_checksum("chunk_2_3")


def test_get_next_download_and_compute_chunks():
    """Queue selection returns untaken chunks and skips already active ones."""
    config = _config(
        total_layers=4, chunk_size_layers=2, parallel_downloads=1, parallel_computes=1
    )
    manager = WeightStreamingManager(config)

    # Manually create two chunks
    from pathlib import Path

    chunk_a = LayerChunk(
        chunk_id="a",
        layer_start=0,
        layer_end=2,
        size_bytes=1,
        download_url="",
        local_path=Path("/tmp/a"),
        checksum="",
    )
    chunk_b = LayerChunk(
        chunk_id="b",
        layer_start=2,
        layer_end=4,
        size_bytes=1,
        download_url="",
        local_path=Path("/tmp/b"),
        checksum="",
    )
    manager.chunks = [chunk_a, chunk_b]

    assert manager._get_next_download_chunk() == chunk_a
    assert manager._get_next_download_chunk() == chunk_b
    # When all chunks are active, return None
    assert manager._get_next_download_chunk() is None

    # Only downloaded chunks become available for compute
    chunk_a.downloaded = True
    assert manager._get_next_compute_chunk() == chunk_a
    assert manager._get_next_compute_chunk() is None
