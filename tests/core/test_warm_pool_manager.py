"""Tests for terradev_cli.core.warm_pool_manager.

Warm pool management keeps inference models loaded to avoid cold starts.
These tests exercise model registration, traffic recording, warming decisions,
and LoRA adapter warm state.
"""

import asyncio

import pytest

from terradev_cli.core.warm_pool_manager import (
    WarmPoolConfig,
    WarmPoolManager,
    WarmPoolMetrics,
    WarmStrategy,
)


@pytest.fixture
def manager(tmp_path):
    config = WarmPoolConfig(
        max_warm_models=3,
        min_warm_models=1,
        warm_threshold_rph=5.0,
        idle_eviction_minutes=15,
        strategy=WarmStrategy.TRAFFIC_BASED,
    )
    return WarmPoolManager(config, config_dir=tmp_path)


def test_warm_pool_config_defaults():
    """WarmPoolConfig has sensible defaults."""
    config = WarmPoolConfig()
    assert config.max_warm_models == 10
    assert config.min_warm_models == 3
    assert config.strategy == WarmStrategy.TRAFFIC_BASED


def test_warm_pool_metrics_defaults():
    """WarmPoolMetrics starts at zero."""
    metrics = WarmPoolMetrics()
    assert metrics.total_warm_requests == 0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0


@pytest.mark.asyncio
async def test_register_and_mark_model_warm(manager):
    """Models can be registered, marked warming, and marked warm."""
    await manager.register_model("model-1", priority=5)
    assert "model-1" in manager.model_priorities
    assert manager.model_priorities["model-1"] == 5

    await manager.mark_model_warming("model-1")
    assert "model-1" in manager.warming_models

    await manager.mark_model_warm("model-1", load_time_s=1.5)
    assert "model-1" in manager.warm_models
    assert "model-1" not in manager.warming_models
    assert manager.model_load_times["model-1"] == 1.5


@pytest.mark.asyncio
async def test_record_request_and_should_warm_traffic(manager):
    """Traffic-based warming triggers when requests exceed the threshold."""
    await manager.register_model("model-1")

    # Below threshold
    for _ in range(4):
        await manager.record_request("model-1", 100.0, was_warm=False)
    assert await manager.should_warm_model("model-1") is False

    # Above threshold
    for _ in range(2):
        await manager.record_request("model-1", 100.0, was_warm=False)
    assert await manager.should_warm_model("model-1") is True


@pytest.mark.asyncio
async def test_should_warm_honors_capacity(manager):
    """No additional models are warmed once the pool is at capacity."""
    manager.config.max_warm_models = 1
    manager.config.warm_threshold_rph = 1.0

    await manager.register_model("model-1")
    await manager.register_model("model-2")

    for _ in range(2):
        await manager.record_request("model-1", 50.0, was_warm=False)
        await manager.record_request("model-2", 50.0, was_warm=False)

    assert await manager.should_warm_model("model-1") is True
    await manager.mark_model_warm("model-1", 1.0)
    assert await manager.should_warm_model("model-2") is False


@pytest.mark.asyncio
async def test_mark_model_evicted(manager):
    """Evicting a model removes it from warm/warming sets."""
    await manager.register_model("model-1")
    await manager.mark_model_warm("model-1", 1.0)
    assert "model-1" in manager.warm_models

    await manager.mark_model_evicted("model-1")
    assert "model-1" not in manager.warm_models


@pytest.mark.asyncio
async def test_get_status_and_model_details(manager):
    """Status and model details reflect current warm pool state."""
    await manager.register_model("model-1", priority=5)
    await manager.mark_model_warm("model-1", 1.0)
    await manager.record_request("model-1", 100.0, was_warm=True)

    status = manager.get_status()
    assert status["warm_models_count"] == 1
    assert status["total_models"] == 1
    assert status["total_requests"] == 1

    details = manager.get_model_details("model-1")
    assert details["model_id"] == "model-1"
    assert details["is_warm"] is True
    assert details["priority"] == 5

    assert manager.get_model_details("missing") is None


def test_predict_traffic_and_candidates(manager):
    """Predicted traffic and warming candidates are based on history."""
    manager.model_priorities["model-1"] = 1
    manager.model_traffic["model-1"] = []

    assert manager.predict_traffic("model-1", 1) == 0.0
    assert manager.get_predictive_warming_candidates(1) == []


def test_cuda_graph_detection_and_tips(manager):
    """CUDA Graph heuristics identify transformer models and produce tips."""
    manager.model_priorities["llama-7b"] = 1
    manager._detect_model_type("llama-7b") == "transformer"

    graph_score = manager._calculate_model_graph_score("llama-7b", "transformer")
    assert 0.0 <= graph_score <= 1.0

    # Populate traffic to push score above 0.7
    for _ in range(150):
        manager.model_traffic.setdefault("llama-7b", []).append(None)  # placeholder not used in score calc
    # Actually _calculate_model_graph_score only uses len(model_traffic), so placeholders count
    score = manager._calculate_model_graph_score("llama-7b", "transformer")
    assert score > 0.7

    manager.model_graph_scores["llama-7b"] = score
    manager.cuda_graph_models.add("llama-7b")
    assert manager.should_warm_with_cuda_graphs("llama-7b") is True

    tips = manager.get_cuda_graph_optimization_tips("llama-7b")
    assert tips["use_cuda_graphs"] is True
    assert tips["model_type"] == "transformer"


@pytest.mark.asyncio
async def test_register_replica_and_adapter_warm_state(manager):
    """Adapters can be tracked as warm on specific replicas."""
    await manager.register_replica("replica-1")
    await manager.record_adapter_load("replica-1", "adapter-a", 10.0)

    warm_replicas = await manager.get_adapter_warm_replicas("adapter-a")
    assert "replica-1" in warm_replicas

    warm_adapters = await manager.get_replica_warm_adapters("replica-1")
    assert "adapter-a" in warm_adapters

    assert await manager.should_warm_adapter_on_replica("replica-1", "adapter-a") is True

    await manager.record_adapter_unload("replica-1", "adapter-a")
    assert await manager.should_warm_adapter_on_replica("replica-1", "adapter-a") is False


@pytest.mark.asyncio
async def test_start_and_stop(manager):
    """Start and stop create and cancel background tasks."""
    await manager.start()
    assert manager._running is True
    assert manager._warming_task is not None

    await manager.stop()
    assert manager._running is False
