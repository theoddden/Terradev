"""Tests for terradev_cli.core.session_manager.

Session management is critical for HTTP provider APIs: aiohttp sessions
must be pooled, reused, and cleaned up correctly.
"""

import asyncio

import pytest

from terradev_cli.core.session_manager import (
    SessionConfig,
    SessionManager,
    cleanup_global_sessions,
    get_session_manager,
)


@pytest.fixture
async def manager():
    m = SessionManager(config=SessionConfig())
    yield m
    await m.close_all()


@pytest.mark.asyncio
async def test_get_session_creates_and_reuses(manager):
    """get_session creates a session and reuses it for the same provider."""
    s1 = await manager.get_session("aws")
    s2 = await manager.get_session("aws")
    assert s1 is s2
    assert "aws" in manager._sessions


@pytest.mark.asyncio
async def test_get_session_different_providers(manager):
    """Different providers get different sessions."""
    s1 = await manager.get_session("aws")
    s2 = await manager.get_session("gcp")
    assert s1 is not s2
    assert len(manager._sessions) == 2


@pytest.mark.asyncio
async def test_is_session_expired(manager):
    """A session is considered expired after the idle timeout."""
    manager._session_timeout = 0
    await manager.get_session("aws")
    await asyncio.sleep(0.01)
    assert manager._is_session_expired("aws") is True

    # Fetching a new session should replace the expired one
    s2 = await manager.get_session("aws")
    assert s2 is not None


@pytest.mark.asyncio
async def test_cleanup_expired_sessions(manager):
    """cleanup_expired_sessions closes expired sessions."""
    manager._session_timeout = 0
    await manager.get_session("aws")
    await asyncio.sleep(0.01)
    await manager.cleanup_expired_sessions()
    assert "aws" not in manager._sessions


@pytest.mark.asyncio
async def test_close_all(manager):
    """close_all cancels cleanup and closes active sessions."""
    await manager.get_session("aws")
    await manager.get_session("gcp")
    await manager.close_all()
    assert manager._sessions == {}
    assert manager._session_last_used == {}


@pytest.mark.asyncio
async def test_get_stats(manager):
    """get_stats returns the current session count and limits."""
    await manager.get_session("aws")
    stats = manager.get_stats()
    assert stats["active_sessions"] == 1
    assert "aws" in stats["providers"]
    assert stats["connection_limit"] == manager.config.limit
    assert stats["per_host_limit"] == manager.config.limit_per_host


@pytest.mark.asyncio
async def test_global_session_manager_singleton():
    """get_session_manager returns a stable global singleton."""
    s1 = get_session_manager()
    s2 = get_session_manager()
    assert s1 is s2

    await cleanup_global_sessions()
