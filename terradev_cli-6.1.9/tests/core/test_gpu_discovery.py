"""Tests for terradev_cli.core.gpu_discovery.

GPU discovery must gracefully fall back to an empty state when NVML/Rust is
unavailable (e.g., in CI or on Mac).
"""

from terradev_cli.core.gpu_discovery import GPUDiscoveryWrapper


def test_wrapper_can_be_instantiated():
    """The wrapper accepts a TTL and initializes Python fallback."""
    wrapper = GPUDiscoveryWrapper(cache_ttl_secs=10)
    assert wrapper._cache_ttl == 10


def test_discover_gpus_returns_safe_fallback():
    """Without NVML, discover_gpus returns an empty list and count 0."""
    wrapper = GPUDiscoveryWrapper()
    result = wrapper.discover_gpus()
    assert isinstance(result, dict)
    assert result.get("total_count") == 0
    assert result.get("gpus") == []


def test_get_gpu_by_index_returns_none_without_gpus():
    """Lookup by index returns None when no GPUs are present."""
    wrapper = GPUDiscoveryWrapper()
    assert wrapper.get_gpu_by_index(0) is None
