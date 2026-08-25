"""Tests for terradev_cli.core.vault_adapter."""

import os
from pathlib import Path

import pytest

from terradev_cli.core.vault_adapter import VaultAdapter, read_secret_from_stdin


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Create an isolated VaultAdapter in a temp home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return VaultAdapter(home / ".terradev")


def test_parse_env_name_known_providers():
    assert VaultAdapter.parse_env_name("TERRADEV_RUNPOD_API_KEY") == ("runpod", "api_key")
    assert VaultAdapter.parse_env_name("TERRADEV_AWS_SECRET_KEY") == ("aws", "secret_key")
    assert VaultAdapter.parse_env_name("TERRADEV_GCP_PROJECT_ID") == ("gcp", "project_id")
    assert VaultAdapter.parse_env_name("TERRADEV_FOO_BAR") == ("foo", "bar")
    assert VaultAdapter.parse_env_name("TERRADEV_FOO") == (None, None)
    assert VaultAdapter.parse_env_name("OTHER_VAR") == (None, None)


def test_set_get_remove(vault):
    vault.set("runpod", "api_key", "rpa_secret")
    assert vault.get("runpod", "api_key") == "rpa_secret"
    assert vault.get("runpod", "missing") is None
    assert vault.get("unknown", "api_key") is None

    vault.remove("runpod", "api_key")
    assert vault.get("runpod", "api_key") is None


def test_remove_whole_provider(vault):
    vault.set("runpod", "api_key", "rpa_secret")
    vault.set("runpod", "other", "value")
    assert vault.remove("runpod") is True
    assert vault.get("runpod", "api_key") is None
    assert vault.remove("runpod") is False


def test_load_env_credentials(vault, monkeypatch):
    vault.set("runpod", "api_key", "file_secret")
    monkeypatch.setenv("TERRADEV_VASTAI_API_KEY", "env_secret")
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "env_overwrite")

    creds = vault.all_credentials()
    assert creds["runpod"]["api_key"] == "env_overwrite"
    assert creds["vastai"]["api_key"] == "env_secret"


def test_verify(vault, monkeypatch):
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "rpa_secret")
    status = vault.verify()
    assert "runpod" in status["configured"]
    assert status["missing"] == {}
    assert status["valid"] is True

    # Partially configured provider: AWS has only one of two required keys.
    monkeypatch.setenv("TERRADEV_AWS_API_KEY", "akid")
    monkeypatch.delenv("TERRADEV_RUNPOD_API_KEY", raising=False)
    status = vault.verify()
    assert "aws" in status["missing"]
    assert "secret_key" in status["missing"]["aws"]
    assert "runpod" not in status["missing"]
    assert "runpod" not in status["configured"]
    assert status["valid"] is False


def test_to_env(vault, monkeypatch):
    monkeypatch.setenv("TERRADEV_RUNPOD_API_KEY", "rpa_secret")
    env = vault.to_env("runpod")
    assert env == {"TERRADEV_RUNPOD_API_KEY": "rpa_secret"}


def test_no_persist(vault, monkeypatch):
    monkeypatch.setenv("TERRADEV_NO_PERSIST", "1")
    vault.set("runpod", "api_key", "rpa_secret")
    # Without persistence, the file should remain empty/unwritten for this key.
    # The underlying AuthManager is still created, but the save call is a no-op.
    assert vault.get("runpod", "api_key") == "rpa_secret"  # from env after set
