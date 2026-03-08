#!/usr/bin/env python3
"""
Session Manager - Connection pooling for provider APIs
Reduces connection overhead and improves scalability
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for HTTP sessions"""
    timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=30)
    limit: int = 100  # Connection pool size
    limit_per_host: int = 10  # Per-host connection limit
    enable_cleanup_closed: bool = True


class SessionManager:
    """
    Manages aiohttp sessions with connection pooling and automatic cleanup.
    
    Features:
    - Connection pooling per provider
    - Automatic session cleanup
    - Adaptive timeout management
    - Connection reuse across requests
    """
    
    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._session_last_used: Dict[str, datetime] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
        self._session_timeout = 1800  # 30 minutes
        self._lock = asyncio.Lock()
        
    async def get_session(self, provider: str) -> aiohttp.ClientSession:
        """Get or create a session for the specified provider"""
        async with self._lock:
            if provider not in self._sessions or self._is_session_expired(provider):
                await self._create_session(provider)
            
            self._session_last_used[provider] = datetime.now()
            return self._sessions[provider]
    
    async def _create_session(self, provider: str):
        """Create a new session for the provider"""
        # Close existing session if present
        if provider in self._sessions:
            await self._sessions[provider].close()
        
        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.limit,
            limit_per_host=self.config.limit_per_host,
            enable_cleanup_closed=self.config.enable_cleanup_closed,
            force_close=False,  # Keep connections alive
            keepalive_timeout=30,  # Keep connections open for 30s
            use_dns_cache=True,
        )
        
        # Create session with optimized settings
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.config.timeout,
            headers={
                'User-Agent': 'Terradev-CLI/3.7.2',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
            }
        )
        
        self._sessions[provider] = session
        self._session_last_used[provider] = datetime.now()
        logger.debug(f"Created new session for provider: {provider}")
    
    def _is_session_expired(self, provider: str) -> bool:
        """Check if a session has expired and should be recreated"""
        if provider not in self._session_last_used:
            return True
        
        last_used = self._session_last_used[provider]
        return datetime.now() - last_used > timedelta(seconds=self._session_timeout)
    
    @asynccontextmanager
    async def request(self, provider: str, method: str, url: str, **kwargs):
        """Context manager for making requests with automatic session management"""
        session = await self.get_session(provider)
        try:
            async with session.request(method, url, **kwargs) as response:
                yield response
        except Exception as e:
            logger.error(f"Request failed for {provider}: {e}")
            raise
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions to free resources"""
        async with self._lock:
            expired_providers = [
                provider for provider in self._sessions.keys()
                if self._is_session_expired(provider)
            ]
            
            for provider in expired_providers:
                await self._sessions[provider].close()
                del self._sessions[provider]
                del self._session_last_used[provider]
                logger.debug(f"Cleaned up expired session for provider: {provider}")
    
    async def start_background_cleanup(self):
        """Start background task for periodic session cleanup"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
    
    async def _background_cleanup(self):
        """Background task that periodically cleans up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def close_all(self):
        """Close all sessions and cleanup resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()
            self._session_last_used.clear()
    
    def get_stats(self) -> Dict[str, any]:
        """Get session manager statistics"""
        return {
            'active_sessions': len(self._sessions),
            'providers': list(self._sessions.keys()),
            'last_cleanup': datetime.now() - self._session_last_used.get('cleanup', datetime.now()),
            'connection_limit': self.config.limit,
            'per_host_limit': self.config.limit_per_host,
        }


# Global session manager instance
_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager


async def cleanup_global_sessions():
    """Cleanup the global session manager"""
    global _global_session_manager
    if _global_session_manager:
        await _global_session_manager.close_all()
        _global_session_manager = None
