#!/usr/bin/env python3
"""Tests for ``terradev agent sandbox``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from terradev_cli.commands.agent_infra.sandbox import (
    LocalRuntime,
    RuntimeFactory,
    SandboxRunner,
    sandbox,
)
from terradev_cli.commands.agent_infra.core import (
    NetworkPolicy,
    ResourceLimits,
    SandboxConfig,
    SandboxTimeoutError,
)


@pytest.fixture(autouse=True)
def _allow_local_runtime(monkeypatch):
    """Enable the local test runtime for every test in this file."""
    monkeypatch.setenv("TERRADEV_AGENT_SANDBOX_LOCAL", "1")


def test_sandbox_config_defaults():
    cfg = SandboxConfig(payload="echo hi")
    assert cfg.runtime == "auto"
    assert cfg.network.mode == "none"
    assert cfg.resources.read_only is True
    assert cfg.isolation == "none"


def test_resource_limits_parsing():
    limits = ResourceLimits(memory="512m", vcpus=2, pids=64)
    assert limits.vcpus == 2
    assert limits.pids == 64
    assert limits.memory == "512m"


def test_network_policy_validation():
    with pytest.raises(ValueError):
        NetworkPolicy(mode="invalid")


def test_local_runtime_runs_python():
    async def _main():
        runtime = LocalRuntime()
        assert await runtime.is_available()

        cfg = SandboxConfig(
            runtime="local",
            payload="python3 -c \"print('hello')\"",
            dev_mode=True,
            resources=ResourceLimits(timeout_seconds=10),
        )
        command = ["python3", "-c", "print('hello')"]
        result = await runtime.run(cfg, command=command)
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.runtime == "local"

    asyncio.run(_main())


def test_local_runtime_sync():
    """`LocalRuntime.run` is async; exercise it through the runner."""
    cfg = SandboxConfig(
        runtime="local",
        payload="echo ok",
        dev_mode=True,
        resources=ResourceLimits(timeout_seconds=10),
    )
    runner = SandboxRunner(cfg)
    result = asyncio.run(runner.run(["echo", "ok"]))
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_sandbox_runner_auto_selects_local():
    cfg = SandboxConfig(runtime="auto", payload="echo test", dev_mode=True)
    runner = SandboxRunner(cfg)
    result = asyncio.run(runner.run(["echo", "test"]))
    assert result.exit_code == 0
    assert "test" in result.stdout


def test_sandbox_timeout_raises():
    cfg = SandboxConfig(
        runtime="local",
        payload="python3 -c 'import time; time.sleep(5)'",
        dev_mode=True,
        resources=ResourceLimits(timeout_seconds=0.1),
    )
    runner = SandboxRunner(cfg)
    with pytest.raises(SandboxTimeoutError):
        asyncio.run(runner.run(["python3", "-c", "import time; time.sleep(5)"]))


def test_sandbox_runtimes_command(runner: CliRunner):
    result = runner.invoke(sandbox, ["runtimes"])
    assert result.exit_code == 0
    assert "local" in result.output


def test_sandbox_run_help(runner: CliRunner):
    result = runner.invoke(sandbox, ["run", "--help"])
    assert result.exit_code == 0
    assert "--runtime" in result.output
    assert "--network" in result.output


def test_sandbox_run_json_output(runner: CliRunner):
    result = runner.invoke(
        sandbox,
        [
            "run",
            "--runtime",
            "local",
            "--dev",
            "--format",
            "json",
            "--timeout",
            "10",
            "echo",
            "hello",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["exit_code"] == 0
    assert data["runtime"] == "local"
    assert "hello" in data["stdout"]


def test_runtime_factory_priority():
    cls = RuntimeFactory.get("local")
    assert cls is LocalRuntime
    names = RuntimeFactory.list_runtimes()
    assert "local" in names
    assert "firecracker" in names
