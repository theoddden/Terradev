"""
Smoke tests for small terradev_cli core/entry modules that were at zero coverage.
These tests exercise the Python fallback branches without external dependencies.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terradev_cli.core.manifest_cache import Manifest, ManifestCache, ManifestNode
from terradev_cli.core.monitoring import MetricsCollector


# ── Entry points ──────────────────────────────────────────────────────────────

def test_main_entry_falls_back_to_cli():
    """__main__.main falls back to .cli.cli when optimization modules are missing."""
    import terradev_cli.__main__ as main_mod

    fake_cli = MagicMock()
    with patch("terradev_cli.cli.cli", fake_cli):
        main_mod.main()
    fake_cli.assert_called_once()


def test_entry_point_main():
    """entry_point.main invokes the cli group and handles KeyboardInterrupt."""
    import terradev_cli.entry_point as ep_mod

    fake_cli = MagicMock()
    with patch("terradev_cli.cli.cli", fake_cli):
        ep_mod.main()
    fake_cli.assert_called_once()

    # KeyboardInterrupt path
    fake_cli = MagicMock(side_effect=KeyboardInterrupt)
    with patch("terradev_cli.cli.cli", fake_cli):
        with pytest.raises(SystemExit) as exc:
            ep_mod.main()
        assert exc.value.code == 0


@pytest.mark.asyncio
async def test_command_executor_executes_and_returns_dict():
    """command_executor falls back to asyncio subprocess."""
    from terradev_cli.core.command_executor import execute_command, execute_parallel

    result = await execute_command("echo", ["hello"])
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["returncode"] == 0

    results = await execute_parallel([("echo", ["a"], None), ("echo", ["b"], None)])
    assert len(results) == 2
    assert all(r["success"] for r in results)


# ── Config validator ──────────────────────────────────────────────────────────

def test_config_validator_python_fallback():
    """ConfigValidator uses the Python fallback when Rust is unavailable."""
    from terradev_cli.core.config_validator import ConfigValidator

    schema = json.dumps({
        "required": ["name"],
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    })

    validator = ConfigValidator(schema)

    valid = validator.validate(json.dumps({"name": "x", "age": 30}))
    assert valid["is_valid"] is True

    missing = validator.validate(json.dumps({"age": 30}))
    assert missing["is_valid"] is False
    assert any("Missing" in e for e in missing["errors"])

    wrong_type = validator.validate(json.dumps({"name": 123, "age": "x"}))
    assert wrong_type["is_valid"] is False


# ── Distributed lock ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_distributed_lock_manager_lifecycle():
    """DistributedLockManager Python fallback handles acquire/release/renew."""
    from terradev_cli.core.distributed_lock import DistributedLockManager

    manager = DistributedLockManager()

    lease = await manager.acquire("k", "h1")
    assert lease is not None

    second = await manager.acquire("k", "h2")
    assert second is None

    assert await manager.renew("k", "h1", lease) is True
    assert await manager.release("k", "h1", lease) is True
    assert await manager.release("k", "h1", lease) is False


# ── Cache manager ───────────────────────────────────────────────────────────────

def test_cache_manager_python_fallback_and_eviction():
    """CacheManager Python fallback puts/gets/updates access count and evicts."""
    from terradev_cli.core.cache_manager import CacheManager

    cache = CacheManager(max_capacity=2, policy="tinylfu")
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.access_count("a") == 1
    cache.get("a")
    assert cache.access_count("a") == 2

    # Trigger eviction for the third key
    cache.put("c", 3)
    assert cache.get("c") == 3


# ── Quota manager ───────────────────────────────────────────────────────────────

def test_quota_manager_python_fallback():
    """QuotaManager Python fallback tracks limits/consumption."""
    from terradev_cli.core.quota_manager import QuotaManager

    qm = QuotaManager()
    qm.set_quota("gpu", 10)
    assert qm.check_quota("gpu", 5) is True
    qm.consume_quota("gpu", 5)
    assert qm.check_quota("gpu", 6) is False
    qm.release_quota("gpu", 2)
    assert qm.get_quota("gpu")["remaining"] == 7
    assert "gpu" in qm.list_quotas()


# ── Manifest cache ──────────────────────────────────────────────────────────────

def test_manifest_cache_store_load_list_delete(tmp_path):
    """ManifestCache stores/loads/lists/deletes manifests and hashes datasets."""
    cache_dir = tmp_path / "manifests"
    cache = ManifestCache(str(cache_dir))

    node = ManifestNode(
        provider="aws",
        pod_id="p1",
        instance_id="i-1",
        gpus=1,
        gpu_type="A100",
        region="us-east-1",
        status="active",
        created_at="2024-01-01",
        ttl="1h",
    )
    manifest = Manifest(
        job="test-job",
        version="v1",
        nodes=[node],
        dataset_hash="sha256:abc",
        ttl="1h",
        created_at="2024-01-01",
        metadata={},
    )

    path = cache.store_manifest(manifest)
    assert Path(path).exists()

    loaded = cache.load_manifest("test-job")
    assert loaded is not None
    assert loaded.job == "test-job"

    versions = cache.list_versions("test-job")
    assert "v1" in versions

    assert cache.delete_manifest("test-job", "v1") is True

    # Dataset hash for file
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello")
    h = cache.compute_dataset_hash(str(data_file))
    assert h.startswith("sha256:")

    # Dataset hash for directory
    sub = tmp_path / "data_dir"
    sub.mkdir()
    (sub / "f.txt").write_text("x")
    h2 = cache.compute_dataset_hash(str(sub))
    assert h2.startswith("sha256:")


# ── Parallel provisioner ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_provisioner_build_and_provision():
    """ParallelProvisioner builds allocations and provisions with mocked providers."""
    from terradev_cli.core.parallel_provisioner import ParallelProvisioner

    pp = ParallelProvisioner()

    quotes = [
        {"provider": "RunPod", "price": 1.0, "gpu_type": "A100", "region": "us-east-1"},
        {"provider": "Lambda", "price": 0.9, "gpu_type": "A100", "region": "us-west-2"},
        {"provider": "RunPod", "price": 1.1, "gpu_type": "A100", "region": "us-east-1"},
    ]
    allocations = pp.build_cheapest_spread(quotes, 2, max_price=1.05)
    assert len(allocations) == 2

    # Provision with mocked factory
    fake_provider = MagicMock()
    fake_provider.provision_instance = AsyncMock(
        return_value={"instance_id": "id-1", "price_per_hour": 0.9}
    )
    fake_factory = MagicMock()
    fake_factory.create_provider.return_value = fake_provider

    pp.factory = fake_factory
    group_id, results = await pp.provision_parallel(allocations[:1])
    assert group_id.startswith("pg_")
    assert len(results) == 1
    assert results[0].status == "active"


# ── Optimization config ───────────────────────────────────────────────────────

def test_optimization_config_manager(tmp_path):
    """OptimizationConfigManager loads/saves/updates configuration."""
    from terradev_cli.core import optimization_config as oc

    # Reset singleton
    oc._config_manager = None

    config_path = tmp_path / "optimization.json"
    manager = oc.OptimizationConfigManager(str(config_path))
    config = manager.get_config()
    assert config.auto_optimize is True
    assert config.cuco_config.enabled is True

    manager.update_config({"auto_optimize": False, "cuco_config": {"enabled": False}})
    assert config_path.exists()

    # Reload
    manager2 = oc.OptimizationConfigManager(str(config_path))
    assert manager2.get_config().auto_optimize is False
    assert manager2.get_config().cuco_config.enabled is False

    # Global helpers (seed with a fresh temp-path manager to avoid /etc)
    oc._config_manager = None
    global_path = tmp_path / "optimization_global.json"
    manager_for_global = oc.OptimizationConfigManager(str(global_path))
    oc._config_manager = manager_for_global
    cfg = oc.get_optimization_config()
    assert cfg.auto_optimize is True
    oc.update_optimization_config({"cost_threshold": 2.0})
    assert oc.get_optimization_config().cost_threshold == 2.0


def test_optimization_config_helpers():
    """OptimizationConfigManager returns P95/workload lookup dicts."""
    from terradev_cli.core.optimization_config import OptimizationConfigManager

    manager = OptimizationConfigManager("/nonexistent/path.json")
    assert "flash_attention" in manager.get_p95_boundaries()
    assert "moe" in manager.get_workload_requirements()


# ── GPU discovery ───────────────────────────────────────────────────────────────

def test_gpu_discovery_wrapper_python_fallback():
    """GPUDiscoveryWrapper returns an empty fallback state without NVML."""
    from terradev_cli.core.gpu_discovery import GPUDiscoveryWrapper

    wrapper = GPUDiscoveryWrapper(cache_ttl_secs=5)
    state = wrapper.discover_gpus()
    assert state["total_count"] == 0
    assert state["gpus"] == []
    assert wrapper.get_gpu_by_index(0) is None


# ── Rust telemetry backend ─────────────────────────────────────────────────────

def test_rust_telemetry_backend_raises_without_rust():
    """RustTelemetryBackend requires the Rust telemetry extension."""
    from terradev_cli.core.rust_telemetry import RustTelemetryBackend, USE_RUST_TELEMETRY

    if not USE_RUST_TELEMETRY:
        with pytest.raises(ImportError):
            RustTelemetryBackend()


# ── Monitoring/metrics collector ────────────────────────────────────────────────

def test_metrics_collector_records_and_resets():
    """MetricsCollector records, increments, returns and resets metrics."""
    collector = MetricsCollector(config={"env": "test"})
    collector.record("latency", 12.5, tags={"region": "us-east-1"})
    collector.increment("requests")
    metrics = collector.get_metrics()
    assert len(metrics) == 2
    collector.reset()
    assert collector.get_metrics() == []
