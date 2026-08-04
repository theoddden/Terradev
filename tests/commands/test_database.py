"""Tests for the universal database subcommand."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_database_help(runner):
    r = runner.invoke(cli, ["database", "--help"])
    assert r.exit_code == 0
    assert "crud" in r.output
    assert "search" in r.output


def test_database_up_crud_down(runner, tmp_path, monkeypatch):
    """End-to-end database up/crud/down cycle with the sqlite stub."""
    monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
    monkeypatch.setenv("TERRADEV_OUTPUT", "human")

    # up
    r = runner.invoke(cli, ["database", "up", "--adapter", "sqlite", "--config", json.dumps({"path": str(tmp_path / "db.sqlite")})])
    assert r.exit_code == 0

    # crud
    r = runner.invoke(
        cli,
        [
            "database",
            "crud",
            "--adapter",
            "sqlite",
            "--config",
            json.dumps({"path": str(tmp_path / "db.sqlite")}),
            "--operation",
            "select",
            "--table",
            "test",
            "--filters",
            json.dumps({"id": 1}),
        ],
    )
    assert r.exit_code == 0
    assert "operation" in r.output

    # search against redis stub
    r = runner.invoke(
        cli,
        [
            "database",
            "search",
            "--adapter",
            "redis",
            "--config",
            json.dumps({"url": "redis://localhost:6379"}),
            "--table",
            "docs",
            "--vector",
            json.dumps([0.1, 0.2, 0.3]),
            "--top-k",
            "3",
        ],
    )
    assert r.exit_code == 0
