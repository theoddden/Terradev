"""
Terradev SSH Key Manager
Generates per-provision Ed25519 keypairs, encrypts private keys at rest
using the same Fernet key as AuthManager, and provides temp decrypted
keys for train/preflight/monitor commands.
"""

import atexit
import os
import signal
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TERRADEV_DIR = Path.home() / ".terradev"
_SSH_DIR = _TERRADEV_DIR / "ssh"
_KEYFILE_PATH = _TERRADEV_DIR / ".keyfile"


# ---------------------------------------------------------------------------
# Fernet helpers (reuses the same keyfile as auth.py)
# ---------------------------------------------------------------------------

def _load_fernet() -> Fernet:
    """Load or create the shared Fernet key used by AuthManager."""
    _SSH_DIR.mkdir(parents=True, exist_ok=True)
    if _KEYFILE_PATH.exists():
        raw = _KEYFILE_PATH.read_bytes().strip()
        return Fernet(raw)
    # First time — generate
    key = Fernet.generate_key()
    _KEYFILE_PATH.write_bytes(key)
    os.chmod(_KEYFILE_PATH, 0o600)
    return Fernet(key)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_provision_keypair(parallel_group_id: str) -> Tuple[str, str]:
    """Generate an Ed25519 keypair for a provision group.

    Returns (private_key_path, public_key_string).
    The private key is Fernet-encrypted at rest.
    """
    _SSH_DIR.mkdir(parents=True, exist_ok=True)

    priv_key = Ed25519PrivateKey.generate()

    # Serialize private key (PEM, no password — encryption is via Fernet)
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Serialize public key (OpenSSH format)
    pub_ssh = priv_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()

    # Encrypt private key with Fernet
    fernet = _load_fernet()
    encrypted_priv = fernet.encrypt(priv_pem)

    # Write encrypted private key
    priv_path = _SSH_DIR / f"{parallel_group_id}.key"
    priv_path.write_bytes(encrypted_priv)
    os.chmod(priv_path, 0o600)

    # Write public key (not sensitive)
    pub_path = _SSH_DIR / f"{parallel_group_id}.pub"
    pub_path.write_text(pub_ssh + "\n")
    os.chmod(pub_path, 0o644)

    return str(priv_path), pub_ssh


def get_public_key(parallel_group_id: str) -> Optional[str]:
    """Read the public key for a provision group, or None if missing."""
    pub_path = _SSH_DIR / f"{parallel_group_id}.pub"
    if pub_path.exists():
        return pub_path.read_text().strip()
    return None


# ---------------------------------------------------------------------------
# Decryption for use
# ---------------------------------------------------------------------------

# Track temp files so we can clean them up
_temp_key_files: list = []


def _cleanup_temp_keys():
    """Remove all decrypted temp key files."""
    for path in _temp_key_files:
        try:
            os.unlink(path)
        except OSError:
            pass
    _temp_key_files.clear()


# Register cleanup on exit and common signals
atexit.register(_cleanup_temp_keys)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        _prev = signal.getsignal(_sig)

        def _handler(signum, frame, _prev=_prev):
            _cleanup_temp_keys()
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(_sig, _handler)
    except (OSError, ValueError):
        pass  # Can't set signal handlers in some contexts (threads, etc.)


def decrypt_private_key(parallel_group_id: str) -> Optional[str]:
    """Decrypt the private key to a temp file and return its path.

    The temp file is 0600, deleted on process exit or SIGINT/SIGTERM.
    Returns None if the encrypted key doesn't exist.
    """
    enc_path = _SSH_DIR / f"{parallel_group_id}.key"
    if not enc_path.exists():
        return None

    fernet = _load_fernet()
    priv_pem = fernet.decrypt(enc_path.read_bytes())

    # Write to a temp file with strict permissions
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"terradev_ssh_{parallel_group_id[:16]}_",
        suffix=".key",
    )
    try:
        os.write(fd, priv_pem)
    finally:
        os.close(fd)
    os.chmod(tmp_path, 0o600)

    _temp_key_files.append(tmp_path)
    return tmp_path


def get_provision_ssh_key_path(parallel_group_id: str) -> Optional[str]:
    """Return the encrypted private key path for a provision group, or None."""
    path = _SSH_DIR / f"{parallel_group_id}.key"
    return str(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def delete_provision_keys(parallel_group_id: str) -> bool:
    """Delete both keys for a provision group (e.g. after terminate)."""
    deleted = False
    for suffix in (".key", ".pub"):
        p = _SSH_DIR / f"{parallel_group_id}{suffix}"
        if p.exists():
            p.unlink()
            deleted = True
    return deleted
