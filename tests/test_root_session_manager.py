"""Tests for terradev_cli.core.session_manager."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ── SessionConfig ───────────────────────────────────────────────────────────

class TestSessionConfig:
    def test_default_values(self):
        from terradev_cli.core.session_manager import SessionConfig
        import aiohttp

        cfg = SessionConfig()
        assert cfg.limit == 100
        assert cfg.limit_per_host == 10
        assert cfg.enable_cleanup_closed is True
        assert isinstance(cfg.timeout, aiohttp.ClientTimeout)

    def test_custom_values(self):
        from terradev_cli.core.session_manager import SessionConfig
        import aiohttp

        cfg = SessionConfig(limit=50, limit_per_host=5)
        assert cfg.limit == 50
        assert cfg.limit_per_host == 5


# ── SessionManager ──────────────────────────────────────────────────────────

class TestSessionManager:
    @pytest.fixture
    def manager(self):
        from terradev_cli.core.session_manager import SessionManager
        return SessionManager()

    def test_init_default_config(self, manager):
        assert manager._sessions == {}
        assert manager._session_last_used == {}
        assert manager._cleanup_task is None

    def test_init_custom_config(self):
        from terradev_cli.core.session_manager import SessionManager, SessionConfig
        cfg = SessionConfig(limit=50)
        mgr = SessionManager(config=cfg)
        assert mgr.config.limit == 50

    def test_is_session_expired_missing_provider(self, manager):
        assert manager._is_session_expired("unknown") is True

    def test_is_session_expired_fresh(self, manager):
        manager._session_last_used["runpod"] = datetime.now()
        assert manager._is_session_expired("runpod") is False

    def test_is_session_expired_old(self, manager):
        manager._session_last_used["runpod"] = datetime.now() - timedelta(seconds=99999)
        assert manager._is_session_expired("runpod") is True

    def test_get_stats_empty(self, manager):
        stats = manager.get_stats()
        assert stats["active_sessions"] == 0
        assert stats["providers"] == []
        assert stats["connection_limit"] == 100
        assert stats["per_host_limit"] == 10

    async def test_get_session_creates_session(self, manager):
        mock_session = MagicMock()
        mock_session.closed = False

        with patch.object(manager, "_create_session", new=AsyncMock()) as mock_create:
            async def side_effect(provider):
                manager._sessions[provider] = mock_session
                manager._session_last_used[provider] = datetime.now()
            mock_create.side_effect = side_effect

            session = await manager.get_session("runpod")
            assert session is mock_session
            mock_create.assert_called_once_with("runpod")

    async def test_get_session_reuses_existing(self, manager):
        mock_session = MagicMock()
        manager._sessions["runpod"] = mock_session
        manager._session_last_used["runpod"] = datetime.now()

        with patch.object(manager, "_create_session", new=AsyncMock()) as mock_create:
            session = await manager.get_session("runpod")
            assert session is mock_session
            mock_create.assert_not_called()

    async def test_get_session_recreates_expired(self, manager):
        old_session = MagicMock()
        old_session.close = AsyncMock()
        manager._sessions["runpod"] = old_session
        manager._session_last_used["runpod"] = datetime.now() - timedelta(seconds=99999)

        new_session = MagicMock()

        async def create_side(provider):
            manager._sessions[provider] = new_session
            manager._session_last_used[provider] = datetime.now()

        with patch.object(manager, "_create_session", new=AsyncMock(side_effect=create_side)):
            session = await manager.get_session("runpod")
            assert session is new_session

    async def test_cleanup_expired_sessions(self, manager):
        old_session = MagicMock()
        old_session.close = AsyncMock()
        manager._sessions["runpod"] = old_session
        manager._session_last_used["runpod"] = datetime.now() - timedelta(seconds=99999)

        fresh_session = MagicMock()
        fresh_session.close = AsyncMock()
        manager._sessions["vastai"] = fresh_session
        manager._session_last_used["vastai"] = datetime.now()

        await manager.cleanup_expired_sessions()

        assert "runpod" not in manager._sessions
        assert "vastai" in manager._sessions
        old_session.close.assert_awaited_once()

    async def test_close_all_closes_sessions(self, manager):
        s1 = MagicMock()
        s1.close = AsyncMock()
        s2 = MagicMock()
        s2.close = AsyncMock()
        manager._sessions["a"] = s1
        manager._sessions["b"] = s2

        await manager.close_all()

        s1.close.assert_awaited_once()
        s2.close.assert_awaited_once()
        assert manager._sessions == {}
        assert manager._session_last_used == {}

    async def test_close_all_cancels_cleanup_task(self, manager):
        started = asyncio.Event()
        cancelled = False

        async def fake_bg():
            nonlocal cancelled
            try:
                started.set()
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                cancelled = True
                raise

        manager._cleanup_task = asyncio.create_task(fake_bg())
        await started.wait()  # ensure task is running before we cancel
        await manager.close_all()
        assert cancelled

    async def test_start_background_cleanup_creates_task(self, manager):
        await manager.start_background_cleanup()
        assert manager._cleanup_task is not None
        manager._cleanup_task.cancel()
        try:
            await manager._cleanup_task
        except asyncio.CancelledError:
            pass

    async def test_start_background_cleanup_does_not_duplicate(self, manager):
        await manager.start_background_cleanup()
        task1 = manager._cleanup_task
        await manager.start_background_cleanup()
        task2 = manager._cleanup_task
        assert task1 is task2
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    def test_get_stats_with_sessions(self, manager):
        manager._sessions["runpod"] = MagicMock()
        manager._sessions["vastai"] = MagicMock()
        stats = manager.get_stats()
        assert stats["active_sessions"] == 2
        assert "runpod" in stats["providers"]
        assert "vastai" in stats["providers"]


# ── Global helpers ──────────────────────────────────────────────────────────

class TestGlobalSessionManager:
    def test_get_session_manager_returns_instance(self):
        import terradev_cli.core.session_manager as mod
        mod._global_session_manager = None  # reset

        from terradev_cli.core.session_manager import get_session_manager, SessionManager
        mgr = get_session_manager()
        assert isinstance(mgr, SessionManager)

    def test_get_session_manager_singleton(self):
        import terradev_cli.core.session_manager as mod
        mod._global_session_manager = None  # reset

        from terradev_cli.core.session_manager import get_session_manager
        a = get_session_manager()
        b = get_session_manager()
        assert a is b

    async def test_cleanup_global_sessions(self):
        import terradev_cli.core.session_manager as mod

        fake_mgr = MagicMock()
        fake_mgr.close_all = AsyncMock()
        mod._global_session_manager = fake_mgr

        from terradev_cli.core.session_manager import cleanup_global_sessions
        await cleanup_global_sessions()

        fake_mgr.close_all.assert_awaited_once()
        assert mod._global_session_manager is None

    async def test_cleanup_global_sessions_noop_when_none(self):
        import terradev_cli.core.session_manager as mod
        mod._global_session_manager = None

        from terradev_cli.core.session_manager import cleanup_global_sessions
        await cleanup_global_sessions()  # should not raise
