"""Tests for terradev_cli.core.quota_manager.

Quota enforcement prevents cost overruns and noisy-neighbor problems.
These tests cover the Python fallback in-memory quota manager.
"""

from terradev_cli.core.quota_manager import QuotaManager


def test_set_and_get_quota():
    """Quotas can be set, queried, and listed."""
    qm = QuotaManager()
    qm.set_quota("gpus", 10)

    quota = qm.get_quota("gpus")
    assert quota is not None
    assert quota["limit"] == 10
    assert quota["used"] == 0
    assert quota["remaining"] == 10

    assert qm.list_quotas() == ["gpus"]


def test_check_consume_and_release_quota():
    """Quota can be checked, consumed, and released."""
    qm = QuotaManager()
    qm.set_quota("gpus", 5)

    assert qm.check_quota("gpus", 3) is True
    qm.consume_quota("gpus", 3)
    assert qm.get_quota("gpus")["used"] == 3

    assert qm.check_quota("gpus", 3) is False
    qm.consume_quota("gpus", 10)  # over-consumption allowed by current impl
    assert qm.get_quota("gpus")["used"] == 13

    qm.release_quota("gpus", 5)
    assert qm.get_quota("gpus")["used"] == 8

    qm.release_quota("gpus", 100)
    assert qm.get_quota("gpus")["used"] == 0


def test_unconfigured_quota_is_unlimited():
    """A resource with no quota set always passes checks."""
    qm = QuotaManager()
    assert qm.check_quota("unknown", 1000) is True
    assert qm.get_quota("unknown") is None
