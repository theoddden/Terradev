"""Tests for terradev_cli.core.command_executor.

Command execution wraps asyncio subprocess calls. These tests exercise the
Python fallback with safe shell commands.
"""

import pytest

from terradev_cli.core.command_executor import execute_command, execute_parallel


@pytest.mark.asyncio
async def test_execute_command_success():
    """execute_command runs a command and returns stdout/returncode."""
    result = await execute_command("echo", ["hello"])
    assert result["success"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_execute_command_with_cwd(tmp_path):
    """execute_command respects the cwd argument."""
    (tmp_path / "file.txt").write_text("data")
    result = await execute_command("ls", ["-1"], cwd=str(tmp_path))
    assert result["success"] is True
    assert "file.txt" in result["stdout"]


@pytest.mark.asyncio
async def test_execute_command_failure():
    """execute_command reports non-zero return codes."""
    result = await execute_command("ls", ["/nonexistent_directory_for_test"])
    assert result["success"] is False
    assert result["returncode"] != 0


@pytest.mark.asyncio
async def test_execute_parallel(tmp_path):
    """execute_parallel runs multiple commands concurrently."""
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")

    commands = [
        ("basename", [str(tmp_path / "a.txt")], None),
        ("basename", [str(tmp_path / "b.txt")], None),
    ]
    results = await execute_parallel(commands)

    assert len(results) == 2
    assert all(r["success"] for r in results)
    assert "a.txt" in results[0]["stdout"]
    assert "b.txt" in results[1]["stdout"]
