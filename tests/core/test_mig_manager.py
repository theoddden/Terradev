"""Tests for terradev_cli.core.mig_manager.

MIGManager determines whether Multi-Instance GPU (MIG) can be enabled for a
given provider, GPU, and instance type.
"""

from unittest.mock import AsyncMock

import pytest

from terradev_cli.core.mig_manager import MIGManager


class FakeExecutor:
    """Fake remote executor for MIG enablement tests."""

    def __init__(self, exit_code=0, stdout="", stderr=""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    async def execute_command(self, instance_id, command, async_exec=False):
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@pytest.fixture
def manager():
    return MIGManager("aws")


@pytest.mark.asyncio
async def test_check_mig_support_known_gpu(manager):
    """MIG is supported for known GPUs on compatible instance types."""
    result = await manager.check_mig_support("A100", "p4d.24xlarge")
    assert result["supported"] is True
    assert result["provider"] == "aws"


@pytest.mark.asyncio
async def test_check_mig_support_unknown_gpu(manager):
    """Unknown GPUs are reported as unsupported."""
    result = await manager.check_mig_support("RTX3090", "p4d.24xlarge")
    assert result["supported"] is False
    assert "does not support MIG" in result["reason"]


@pytest.mark.asyncio
async def test_check_mig_support_incompatible_instance(manager):
    """Incompatible instance types are reported."""
    result = await manager.check_mig_support("A100", "t2.micro")
    assert result["supported"] is False
    assert result["action_required"]
    assert "p4d.24xlarge" in result["action_required"]


@pytest.mark.asyncio
async def test_enable_mig_success(manager, monkeypatch):
    """MIG enable command succeeds and status is checked."""
    executor = FakeExecutor(exit_code=0, stdout="Enabled")
    monkeypatch.setattr(
        manager, "_check_mig_status", AsyncMock(return_value={"enabled": True})
    )

    result = await manager.enable_mig("A100", "i-123", executor)
    assert result["success"] is True
    assert result["mig_status"]["enabled"] is True


@pytest.mark.asyncio
async def test_enable_mig_command_failure(manager):
    """MIG enable command failure returns a structured error."""
    executor = FakeExecutor(exit_code=1, stderr="driver error")
    result = await manager.enable_mig("A100", "i-123", executor)
    assert result["success"] is False
    assert "driver error" in result["reason"]


@pytest.mark.asyncio
async def test_create_mig_instances(manager):
    """MIG instances can be created with valid profiles."""
    executor = FakeExecutor(exit_code=0, stdout="")
    profiles = ["1g.5gb", "2g.10gb"]
    result = await manager.create_mig_instances("A100", "i-123", profiles, executor)
    assert result["success"] is True
    assert result["total_instances"] == 2


@pytest.mark.asyncio
async def test_create_mig_instances_invalid_profile(manager):
    """Invalid MIG profile returns an error."""
    executor = FakeExecutor(exit_code=0)
    result = await manager.create_mig_instances(
        "A100", "i-123", ["invalid_profile"], executor
    )
    assert result["success"] is False
    assert "Invalid MIG profile" in result["reason"]


@pytest.mark.asyncio
async def test_get_mig_cost_analysis(manager):
    """Cost analysis includes profiles and utilization metrics."""
    result = await manager.get_mig_cost_analysis("A100", "p4d.24xlarge", 32.77)
    assert result["mig_supported"] is True
    assert result["full_gpu_cost_per_hour"] == 32.77
    assert "mig_options" in result
