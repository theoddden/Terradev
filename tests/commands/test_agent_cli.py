#!/usr/bin/env python3
"""End-to-end CLI wiring tests for the terradev agent subcommands."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture(autouse=True)
def _skip_onboarding_and_allow_local(monkeypatch):
    monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
    monkeypatch.setenv("TERRADEV_AGENT_SANDBOX_LOCAL", "1")


def test_agent_root_help(runner: CliRunner):
    result = runner.invoke(cli, ["agent", "--help"])
    assert result.exit_code == 0, result.output
    assert "sandbox" in result.output
    assert "mesh" in result.output
    assert "mcp" in result.output


def test_agent_sandbox_run_cli(runner: CliRunner):
    result = runner.invoke(cli, [
        "agent", "sandbox", "run",
        "--runtime", "local",
        "--dev",
        "--format", "json",
        "--timeout", "10",
        "echo", "hello",
    ])
    assert result.exit_code == 0, repr(result.output)
    # The sandbox command prints its result as a single JSON object.
    data = json.loads(result.output.strip())
    assert data["exit_code"] == 0
    assert data["runtime"] == "local"
    assert "hello" in data["stdout"]


def test_agent_mesh_peers_cli(runner: CliRunner):
    result = runner.invoke(cli, ["agent", "mesh", "peers", "--help"])
    assert result.exit_code == 0, result.output
    assert "--transport" in result.output


def test_agent_mcp_registry_cli(runner: CliRunner, tmp_path):
    path = tmp_path / "mcp_registry.json"
    result = runner.invoke(cli, [
        "agent", "mcp", "registry", "add",
        "--config", str(path),
        "--name", "test",
        "--command", "python3",
        "--args", "-m,http.server",
    ])
    assert result.exit_code == 0, result.output
    assert "test" in result.output
