"""Tests for terradev_cli.core.auth.

AuthManager stores provider credentials encrypted at rest and provides
provider-specific HTTP headers.
"""

import json
from pathlib import Path

import pytest

from terradev_cli.core.auth import AuthManager


@pytest.fixture
def auth(tmp_path):
    return AuthManager.load(str(tmp_path / "auth.json"))


def test_load_creates_new_auth_file(auth, tmp_path):
    """Loading a non-existent auth path creates a new encrypted store."""
    assert Path(tmp_path / "auth.json").exists()
    assert Path(tmp_path / ".keyfile").exists()
    assert auth.encryption_key is not None
    assert auth.fernet is not None


def test_set_and_get_credentials(auth):
    """Credentials can be set and retrieved."""
    auth.set_credentials("aws", "AKIA...", "secret...")
    cred = auth.get_credentials("aws")
    assert cred["api_key"] == "AKIA..."
    assert cred["secret_key"] == "secret..."


def test_has_credentials_and_list_providers(auth):
    """has_credentials and list_providers reflect stored providers."""
    assert auth.has_credentials() is False
    auth.set_credentials("aws", "key")
    assert auth.has_credentials() is True
    assert auth.list_providers() == ["aws"]


def test_validate_credentials(auth):
    """validate_credentials checks that an API key exists."""
    assert auth.validate_credentials("aws") is False
    auth.set_credentials("aws", "")
    assert auth.validate_credentials("aws") is False
    auth.set_credentials("aws", "key")
    assert auth.validate_credentials("aws") is True


def test_get_provider_auth_headers(auth):
    """Provider-specific headers are generated from credentials."""
    auth.set_credentials("gcp", "token123")
    assert auth.get_provider_auth_headers("gcp") == {
        "Authorization": "Bearer token123"
    }

    auth.set_credentials("aws", "key")
    assert auth.get_provider_auth_headers("aws") == {}

    assert auth.get_provider_auth_headers("missing") == {}


def test_remove_credentials(auth):
    """Credentials can be removed."""
    auth.set_credentials("aws", "key")
    assert auth.remove_credentials("aws") is True
    assert auth.remove_credentials("aws") is False
    assert auth.get_credentials("aws") is None


def test_rotate_api_key(auth):
    """API keys can be rotated with a previous-key backup."""
    auth.set_credentials("aws", "old-key")
    assert auth.rotate_api_key("aws", "new-key") is True
    assert auth.get_credentials("aws")["api_key"] == "new-key"
    assert auth.get_credentials("aws")["previous_key"] == "old-key"

    assert auth.rotate_api_key("missing", "x") is False


def test_save_and_load_roundtrip(tmp_path):
    """Credentials persist and reload through the encrypted file."""
    auth_file = tmp_path / "auth.json"
    auth = AuthManager.load(str(auth_file))
    auth.set_credentials("runpod", "token")
    auth.save(str(auth_file))

    reloaded = AuthManager.load(str(auth_file))
    assert reloaded.get_credentials("runpod")["api_key"] == "token"


def test_backup_and_restore_credentials(auth, tmp_path):
    """Credentials can be backed up and restored."""
    auth.set_credentials("aws", "key", "secret")
    backup = tmp_path / "backup.json"

    assert auth.backup_credentials(str(backup)) is True
    auth.clear_all_credentials()
    assert auth.get_credentials("aws") is None

    assert auth.restore_credentials(str(backup)) is True
    assert auth.get_credentials("aws")["api_key"] == "key"
    assert auth.get_credentials("aws")["secret_key"] == "secret"


def test_credential_summary(auth):
    """get_credential_summary provides a safe overview."""
    auth.set_credentials("aws", "key", "secret")
    summary = auth.get_credential_summary()
    assert summary["aws"]["has_api_key"] is True
    assert summary["aws"]["has_secret_key"] is True
