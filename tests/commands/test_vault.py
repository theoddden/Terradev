"""Tests for the terradev vault command."""

import json
import os

import pytest
from click.testing import CliRunner

from terradev_cli.commands import cli


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Provide a CliRunner with an isolated home dir and human output."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TERRADEV_SKIP_ONBOARDING", "1")
    monkeypatch.setenv("TERRADEV_OUTPUT", "human")
    return CliRunner()


def test_vault_help(runner):
    r = runner.invoke(cli, ["vault", "--help"])
    assert r.exit_code == 0
    assert "set" in r.output
    assert "sync" in r.output
    assert "run" in r.output


def test_vault_set_get_list_remove(runner):
    r = runner.invoke(cli, ["vault", "set", "runpod", "api_key", "--value", "rpa_secret"])
    assert r.exit_code == 0

    r = runner.invoke(cli, ["vault", "list"])
    assert r.exit_code == 0
    assert "runpod" in r.output
    assert "api_key" in r.output

    r = runner.invoke(cli, ["vault", "get", "runpod", "api_key"])
    assert r.exit_code == 0
    assert "***" in r.output
    assert "rpa_secret" not in r.output

    r = runner.invoke(cli, ["vault", "get", "runpod", "api_key", "--raw"])
    assert r.exit_code == 0
    assert "rpa_secret" in r.output

    r = runner.invoke(cli, ["vault", "remove", "runpod"])
    assert r.exit_code == 0

    r = runner.invoke(cli, ["vault", "list"])
    assert r.exit_code == 0
    assert "runpod" not in r.output


def test_vault_set_from_env(runner, monkeypatch):
    monkeypatch.setenv("MY_RUNPOD_KEY", "rpa_from_env")
    r = runner.invoke(
        cli, ["vault", "set", "runpod", "api_key", "--from-env", "MY_RUNPOD_KEY"]
    )
    assert r.exit_code == 0

    r = runner.invoke(cli, ["vault", "get", "runpod", "api_key", "--raw"])
    assert r.exit_code == 0
    assert "rpa_from_env" in r.output


def test_vault_sync(runner, monkeypatch):
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "rpa_sync")
    monkeypatch.setenv("TERRADEV_AWS_SECRET_KEY", "aws_sync")

    r = runner.invoke(cli, ["vault", "sync"])
    assert r.exit_code == 0
    assert "2 secret" in r.output

    r = runner.invoke(cli, ["vault", "list"])
    assert r.exit_code == 0
    assert "runpod" in r.output
    assert "aws" in r.output


def test_vault_verify(runner, monkeypatch):
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "rpa_secret")
    r = runner.invoke(cli, ["vault", "sync"])
    assert r.exit_code == 0

    r = runner.invoke(cli, ["vault", "verify"])
    assert r.exit_code == 0
    assert "runpod" in r.output


def test_vault_env(runner, monkeypatch):
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "rpa_secret")
    r = runner.invoke(cli, ["vault", "env", "runpod", "--raw"])
    assert r.exit_code == 0
    assert "TERRADEV_RUNPOD_API_KEY" in r.output
    assert "rpa_secret" in r.output
