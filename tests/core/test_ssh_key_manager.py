"""Tests for terradev_cli.core.ssh_key_manager.

SSH key management generates Ed25519 keypairs, encrypts private keys at rest,
and provides temp decrypted keys for provisioning workflows.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from terradev_cli.core.ssh_key_manager import (
    delete_provision_keys,
    generate_provision_keypair,
    get_provision_ssh_key_path,
    get_public_key,
)


@pytest.fixture
def isolated_ssh_dir(monkeypatch, tmp_path):
    """Redirect the SSH key directory to a temp path."""
    with patch("terradev_cli.core.ssh_key_manager._SSH_DIR", tmp_path):
        with patch("terradev_cli.core.ssh_key_manager._KEYFILE_PATH", tmp_path / ".keyfile"):
            yield tmp_path


def test_generate_provision_keypair(isolated_ssh_dir):
    """A keypair is generated, encrypted, and the public key is readable."""
    priv_path, pub_key = generate_provision_keypair("group-1")
    assert Path(priv_path).exists()
    assert pub_key.startswith("ssh-ed25519")

    stored_pub = get_public_key("group-1")
    assert stored_pub == pub_key


def test_get_provision_ssh_key_path(isolated_ssh_dir):
    """get_provision_ssh_key_path returns the encrypted private key path."""
    generate_provision_keypair("group-1")
    path = get_provision_ssh_key_path("group-1")
    assert path is not None
    assert path.endswith("group-1.key")

    assert get_provision_ssh_key_path("missing") is None


def test_delete_provision_keys(isolated_ssh_dir):
    """delete_provision_keys removes both private and public keys."""
    generate_provision_keypair("group-1")
    assert get_provision_ssh_key_path("group-1") is not None

    assert delete_provision_keys("group-1") is True
    assert get_provision_ssh_key_path("group-1") is None
    assert get_public_key("group-1") is None

    assert delete_provision_keys("missing") is False
