"""Tests for terradev_cli.core.async_config.

The async config manager is used throughout the CLI for non-blocking config
I/O, caching, and atomic writes. These tests protect that path.
"""

from datetime import datetime

import pytest

from terradev_cli.core.async_config import AsyncConfigManager


@pytest.fixture
def manager(tmp_path):
    return AsyncConfigManager(config_dir=tmp_path)


@pytest.mark.asyncio
async def test_load_missing_config_returns_empty_dict(manager):
    """Loading a non-existent config file returns an empty dict, not an error."""
    data = await manager.load_json("missing.json")
    assert data == {}


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(manager):
    """Data saved can be loaded back."""
    await manager.save_json("settings.json", {"theme": "dark", "gpus": 2})
    loaded = await manager.load_json("settings.json")
    assert loaded == {"theme": "dark", "gpus": 2}


@pytest.mark.asyncio
async def test_cache_returns_same_object_on_second_load(manager):
    """Second load uses the in-memory cache."""
    await manager.save_json("settings.json", {"value": 1})
    first = await manager.load_json("settings.json")
    # Mutate local copy; cached version should be isolated because we got .copy()
    first["value"] = 999
    second = await manager.load_json("settings.json")
    assert second["value"] == 1


@pytest.mark.asyncio
async def test_update_json_merges(manager):
    """update_json with merge=True adds keys while keeping existing ones."""
    await manager.save_json("settings.json", {"a": 1, "b": 2})
    ok = await manager.update_json("settings.json", {"b": 3, "c": 4})
    assert ok is True

    data = await manager.load_json("settings.json", use_cache=False)
    assert data == {"a": 1, "b": 3, "c": 4}


@pytest.mark.asyncio
async def test_update_json_replace(manager):
    """update_json with merge=False replaces the entire file."""
    await manager.save_json("settings.json", {"a": 1, "b": 2})
    ok = await manager.update_json("settings.json", {"only": "this"}, merge=False)
    assert ok is True

    data = await manager.load_json("settings.json", use_cache=False)
    assert data == {"only": "this"}


@pytest.mark.asyncio
async def test_delete_and_get_key(manager):
    """Keys can be deleted, get returns defaults, and set writes a single key."""
    await manager.save_json("settings.json", {"a": 1, "b": 2})

    assert await manager.get_key("settings.json", "a") == 1
    assert await manager.get_key("settings.json", "missing", default="x") == "x"

    assert await manager.delete_key("settings.json", "a") is True
    assert await manager.get_key("settings.json", "a") is None

    assert await manager.set_key("settings.json", "new", 42) is True
    assert await manager.get_key("settings.json", "new") == 42


@pytest.mark.asyncio
async def test_invalid_json_returns_empty_dict(manager):
    """A corrupted config file is treated as an empty dict."""
    bad_file = manager.config_dir / "corrupt.json"
    bad_file.write_text("{not json")

    data = await manager.load_json("corrupt.json")
    assert data == {}


def test_cache_stats_and_invalidation(manager):
    """Cache stats track entries, and invalidation clears them."""
    # Manually populate cache
    manager._cache["a"] = {"x": 1}
    manager._cache_timestamps["a"] = datetime.now()
    stats = manager.get_cache_stats()
    assert stats["cached_files"] == 1

    manager.invalidate_cache("a")
    assert manager.get_cache_stats()["cached_files"] == 0

    manager._cache["a"] = {"x": 1}
    manager._cache_timestamps["a"] = datetime.now()
    manager.invalidate_cache()
    assert manager.get_cache_stats()["cached_files"] == 0
