"""Tests for the SSH key manager — keygen, encrypt/decrypt, cleanup."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_ssh_dir(tmp_path, monkeypatch):
    """Redirect SSH key storage to a temp directory for every test."""
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir()
    terradev_dir = tmp_path / ".terradev"
    terradev_dir.mkdir()
    keyfile = terradev_dir / ".keyfile"

    import terradev_cli.core.ssh_key_manager as skm
    monkeypatch.setattr(skm, "_SSH_DIR", ssh_dir)
    monkeypatch.setattr(skm, "_TERRADEV_DIR", terradev_dir)
    monkeypatch.setattr(skm, "_KEYFILE_PATH", keyfile)
    # Clear any cached temp key list from prior tests
    skm._temp_key_files.clear()
    yield ssh_dir


class TestGenerateProvisionKeypair:
    def test_creates_encrypted_key_and_pubkey(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import generate_provision_keypair

        priv_path, pub_ssh = generate_provision_keypair("pg_test_001")

        assert os.path.exists(priv_path)
        assert pub_ssh.startswith("ssh-ed25519 ")

        pub_path = isolated_ssh_dir / "pg_test_001.pub"
        assert pub_path.exists()
        assert pub_path.read_text().strip() == pub_ssh

        # Private key file should be encrypted (not plain PEM)
        raw = Path(priv_path).read_bytes()
        assert b"OPENSSH PRIVATE KEY" not in raw  # Fernet-encrypted, not plaintext

    def test_permissions_are_strict(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import generate_provision_keypair

        priv_path, _ = generate_provision_keypair("pg_perms_test")
        stat = os.stat(priv_path)
        assert oct(stat.st_mode)[-3:] == "600"

    def test_two_groups_get_different_keys(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import generate_provision_keypair

        _, pub1 = generate_provision_keypair("pg_a")
        _, pub2 = generate_provision_keypair("pg_b")
        assert pub1 != pub2


class TestDecryptPrivateKey:
    def test_decrypt_returns_temp_path_with_valid_pem(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import (
            generate_provision_keypair,
            decrypt_private_key,
        )

        generate_provision_keypair("pg_decrypt_test")
        tmp_path = decrypt_private_key("pg_decrypt_test")

        assert tmp_path is not None
        assert os.path.exists(tmp_path)
        content = Path(tmp_path).read_text()
        assert "OPENSSH PRIVATE KEY" in content

        # Strict permissions
        stat = os.stat(tmp_path)
        assert oct(stat.st_mode)[-3:] == "600"

    def test_decrypt_nonexistent_returns_none(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import decrypt_private_key

        assert decrypt_private_key("pg_does_not_exist") is None

    def test_cleanup_removes_temp_keys(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import (
            generate_provision_keypair,
            decrypt_private_key,
            _cleanup_temp_keys,
            _temp_key_files,
        )

        generate_provision_keypair("pg_cleanup_test")
        tmp_path = decrypt_private_key("pg_cleanup_test")
        assert os.path.exists(tmp_path)
        assert len(_temp_key_files) == 1

        _cleanup_temp_keys()
        assert not os.path.exists(tmp_path)
        assert len(_temp_key_files) == 0


class TestGetPublicKey:
    def test_returns_pubkey_string(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import (
            generate_provision_keypair,
            get_public_key,
        )

        _, pub_ssh = generate_provision_keypair("pg_pub_test")
        result = get_public_key("pg_pub_test")
        assert result == pub_ssh

    def test_returns_none_for_missing(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import get_public_key

        assert get_public_key("pg_nope") is None


class TestDeleteProvisionKeys:
    def test_deletes_both_files(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import (
            generate_provision_keypair,
            delete_provision_keys,
        )

        generate_provision_keypair("pg_del_test")
        assert (isolated_ssh_dir / "pg_del_test.key").exists()
        assert (isolated_ssh_dir / "pg_del_test.pub").exists()

        deleted = delete_provision_keys("pg_del_test")
        assert deleted is True
        assert not (isolated_ssh_dir / "pg_del_test.key").exists()
        assert not (isolated_ssh_dir / "pg_del_test.pub").exists()

    def test_returns_false_for_missing(self, isolated_ssh_dir):
        from terradev_cli.core.ssh_key_manager import delete_provision_keys

        assert delete_provision_keys("pg_nonexistent") is False


class TestFernetKeyReuse:
    def test_shared_keyfile_created_once(self, isolated_ssh_dir, tmp_path):
        from terradev_cli.core.ssh_key_manager import generate_provision_keypair
        import terradev_cli.core.ssh_key_manager as skm

        generate_provision_keypair("pg_fernet_a")
        keyfile_content_a = skm._KEYFILE_PATH.read_bytes()

        generate_provision_keypair("pg_fernet_b")
        keyfile_content_b = skm._KEYFILE_PATH.read_bytes()

        # Same Fernet key reused across provision groups
        assert keyfile_content_a == keyfile_content_b
