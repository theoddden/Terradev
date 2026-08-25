"""Tests for terradev_cli.core.distributed_lock.

The distributed lock manager provides TTL-based coordination. These tests
cover the Python fallback in-memory implementation.
"""

import asyncio

import pytest

from terradev_cli.core.distributed_lock import DistributedLockManager


@pytest.fixture
def lock_mgr():
    return DistributedLockManager()


@pytest.mark.asyncio
async def test_acquire_and_release(lock_mgr):
    """A lock can be acquired and then released by its lease."""
    lease = await lock_mgr.acquire("resource-1", "holder-a", ttl_seconds=10)
    assert lease is not None
    assert isinstance(lease, str)

    assert await lock_mgr.release("resource-1", "holder-a", lease) is True
    assert await lock_mgr.release("resource-1", "holder-a", lease) is False


@pytest.mark.asyncio
async def test_acquire_rejects_when_locked(lock_mgr):
    """A second acquire on a held lock returns None."""
    lease = await lock_mgr.acquire("resource-1", "holder-a", ttl_seconds=10)
    assert await lock_mgr.acquire("resource-1", "holder-b", ttl_seconds=10) is None

    # After release, another holder can acquire
    await lock_mgr.release("resource-1", "holder-a", lease)
    lease2 = await lock_mgr.acquire("resource-1", "holder-b", ttl_seconds=10)
    assert lease2 is not None


@pytest.mark.asyncio
async def test_acquire_after_expiry(lock_mgr):
    """A lock with a short TTL can be re-acquired after expiration."""
    lease = await lock_mgr.acquire("resource-1", "holder-a", ttl_seconds=0)
    await asyncio.sleep(0.01)

    lease2 = await lock_mgr.acquire("resource-1", "holder-b", ttl_seconds=10)
    assert lease2 is not None
    assert lease2 != lease


@pytest.mark.asyncio
async def test_renew_extends_lease(lock_mgr):
    """Renewing a lock updates its expiry time."""
    lease = await lock_mgr.acquire("resource-1", "holder-a", ttl_seconds=1)
    assert await lock_mgr.renew("resource-1", "holder-a", lease, ttl_seconds=10) is True

    # After original TTL, lock should still be held because it was renewed
    await asyncio.sleep(0.5)
    assert await lock_mgr.acquire("resource-1", "holder-b", ttl_seconds=10) is None


@pytest.mark.asyncio
async def test_renew_invalid_lease_fails(lock_mgr):
    """Renewing with an unknown lease returns False."""
    assert await lock_mgr.renew("resource-1", "holder-a", "bad-lease", 10) is False
