#!/usr/bin/env python3
"""
Distributed Lock Manager - Multi-node coordination with TTL-based leases

Rust implementation provides:
- Lock-free quota tracking
- TTL-based lease management
- Automatic expiration cleanup
- Renewal support
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

USE_RUST_LOCK = False


class DistributedLockManager:
    """Distributed lock manager with Rust backend or Python fallback"""

    def __init__(self):
        if USE_RUST_LOCK:
            self._rust_lock = PyDistributedLock()
        else:
            self._locks: Dict[str, Tuple[str, datetime]] = {}
            self._lock = threading.Lock()

    async def acquire(
        self, key: str, holder: str, ttl_seconds: int = 3600
    ) -> Optional[str]:
        """Acquire a distributed lock with TTL"""
        if USE_RUST_LOCK:
            grant = await self._rust_lock.acquire(key, holder, ttl_seconds)
            return grant.lease_id
        else:
            # Python fallback with in-memory dict
            with self._lock:
                if key in self._locks:
                    _stored_lease, expiry = self._locks[key]
                    if datetime.now() < expiry:
                        return None
                lease_id = str(uuid.uuid4())
                self._locks[key] = (
                    lease_id,
                    datetime.now() + timedelta(seconds=ttl_seconds),
                )
            return lease_id

    async def release(self, key: str, holder: str, lease_id: str) -> bool:
        """Release a distributed lock"""
        if USE_RUST_LOCK:
            return await self._rust_lock.release(key, holder, lease_id)
        else:
            # Python fallback
            with self._lock:
                if key in self._locks:
                    stored_lease, _ = self._locks[key]
                    if stored_lease == lease_id:
                        del self._locks[key]
                        return True
            return False

    async def renew(
        self, key: str, holder: str, lease_id: str, ttl_seconds: int = 3600
    ) -> bool:
        """Renew a lock's TTL"""
        if USE_RUST_LOCK:
            return await self._rust_lock.renew(key, holder, lease_id, ttl_seconds)
        else:
            # Python fallback
            with self._lock:
                if key in self._locks:
                    stored_lease, _ = self._locks[key]
                    if stored_lease == lease_id:
                        self._locks[key] = (
                            lease_id,
                            datetime.now() + timedelta(seconds=ttl_seconds),
                        )
                        return True
            return False
