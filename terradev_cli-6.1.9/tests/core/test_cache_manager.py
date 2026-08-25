"""Tests for terradev_cli.core.cache_manager.

CacheManager provides a simple key-value cache with LRU-style eviction and
optional Rust backend integration.
"""

import pytest

from terradev_cli.core.cache_manager import CacheManager


@pytest.fixture
def cache():
    return CacheManager(max_capacity=3, policy="lru")


def test_put_and_get(cache):
    """Values can be stored and retrieved."""
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"


def test_get_missing_returns_none(cache):
    """Missing keys return None."""
    assert cache.get("missing") is None


def test_access_count(cache):
    """Access count increments on get."""
    cache.put("key1", "value1")
    assert cache.access_count("key1") == 0
    cache.get("key1")
    cache.get("key1")
    assert cache.access_count("key1") == 2


def test_eviction_on_capacity(cache):
    """The least-accessed key is evicted when capacity is exceeded."""
    for i in range(3):
        cache.put(f"key{i}", f"value{i}")

    # Access key0 once, key1 twice, key2 not at all
    cache.get("key0")
    cache.get("key1")
    cache.get("key1")

    cache.put("key3", "value3")  # should evict key2 (lowest access count)
    assert cache.get("key2") is None
    assert cache.get("key0") == "value0"
    assert cache.get("key1") == "value1"
    assert cache.get("key3") == "value3"


def test_put_updates_existing(cache):
    """Putting the same key updates its value."""
    cache.put("key1", "old")
    cache.put("key1", "new")
    assert cache.get("key1") == "new"


@pytest.mark.parametrize("value", [{"a": 1}, [1, 2, 3], 42, 3.14])
def test_put_various_types(cache, value):
    """Non-string values are stored and retrieved intact."""
    cache.put("key", value)
    assert cache.get("key") == value
