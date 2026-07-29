"""Tests for terradev_cli.core.credential_vault.

Credential handling must not leak secrets and must degrade gracefully when
the Rust backend is unavailable.
"""

import pytest

from terradev_cli.core.credential_vault import CredentialVault


@pytest.fixture
def vault():
    return CredentialVault()


def test_store_and_retrieve(vault):
    """Credentials round-trip through the vault."""
    vault.store("aws_key", b"secret-access-key", provider="aws")
    assert vault.retrieve("aws_key") == b"secret-access-key"


def test_retrieve_missing_returns_none(vault):
    """Missing credentials return None, not an exception."""
    assert vault.retrieve("does-not-exist") is None


def test_list_and_delete(vault):
    """List returns stored names and delete removes them."""
    vault.store("a", b"1")
    vault.store("b", b"2")
    names = vault.list()
    assert "a" in names
    assert "b" in names

    vault.delete("a")
    assert "a" not in vault.list()
    assert vault.retrieve("a") is None


def test_delete_missing_is_silent(vault):
    """Deleting a non-existent credential does not raise."""
    vault.delete("missing")  # should not raise


def test_overwrite_existing(vault):
    """Storing the same name overwrites the previous value."""
    vault.store("token", b"old")
    vault.store("token", b"new")
    assert vault.retrieve("token") == b"new"


def test_get_metadata_without_rust(vault):
    """Python fallback returns None for metadata."""
    vault.store("x", b"v", provider="p")
    assert vault.get_metadata("x") is None
