"""Tests for terradev_cli.core.optimization_config.

OptimizationConfig controls auto-optimization and CUCo workload settings.
"""

from terradev_cli.core.optimization_config import (
    CUCoConfig,
    OptimizationConfig,
    OptimizationConfigManager,
    get_optimization_config,
    update_optimization_config,
)


def test_cu_config_defaults():
    """CUCoConfig has sensible defaults."""
    cuco = CUCoConfig()
    assert cuco.enabled is True
    assert cuco.min_gpu_count == 2
    assert cuco.min_communication_intensity == 0.3


def test_optimization_config_defaults():
    """OptimizationConfig creates a default CUCoConfig if none provided."""
    config = OptimizationConfig()
    assert config.auto_optimize is True
    assert config.optimization_interval == 300
    assert config.cuco_config is not None
    assert config.cuco_config.enabled is True


def test_config_manager_default(tmp_path):
    """An empty config file path yields the default optimization config."""
    manager = OptimizationConfigManager(config_path=str(tmp_path / "optimization.json"))
    config = manager.get_config()
    assert config.auto_optimize is True
    assert config.cuco_config.min_gpu_count == 2


def test_config_manager_save_and_load(tmp_path):
    """Configs can be saved and reloaded."""
    manager = OptimizationConfigManager(config_path=str(tmp_path / "optimization.json"))
    manager.update_config(
        {"auto_optimize": False, "cuco_config": {"min_gpu_count": 8}}
    )

    reloaded = OptimizationConfigManager(config_path=str(tmp_path / "optimization.json"))
    assert reloaded.get_config().auto_optimize is False
    assert reloaded.get_config().cuco_config.min_gpu_count == 8


def test_config_manager_workload_requirements():
    """Workload requirement dictionaries are returned."""
    manager = OptimizationConfigManager()
    reqs = manager.get_workload_requirements()
    assert "moe" in reqs
    assert "attention" in reqs
    assert "llm_training" in reqs


def test_config_manager_p95_boundaries():
    """P95 boundary dictionaries are returned."""
    manager = OptimizationConfigManager()
    boundaries = manager.get_p95_boundaries()
    assert "flash_attention" in boundaries
    assert "gemm_allgather" in boundaries


def test_global_get_and_update(tmp_path, monkeypatch):
    """Global helpers read and modify the shared config."""
    from terradev_cli.core import optimization_config as oc

    manager = OptimizationConfigManager(config_path=str(tmp_path / "opt.json"))
    monkeypatch.setattr(oc, "_config_manager", manager)

    config = get_optimization_config()
    assert config.auto_optimize is True

    update_optimization_config({"auto_optimize": False})
    updated = get_optimization_config()
    assert updated.auto_optimize is False
