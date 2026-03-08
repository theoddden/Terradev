#!/usr/bin/env python3
"""
Terradev Authentication Management
Handles secure credential storage and management
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
import base64
import hashlib
import secrets


class AuthManager:
    """Manages authentication credentials securely"""

    def __init__(self):
        self.credentials: Dict[str, Dict[str, str]] = {}
        self.encryption_key: Optional[bytes] = None
        self.fernet: Optional[Fernet] = None

    @classmethod
    def load(cls, auth_file: str) -> "AuthManager":
        """Load authentication from file"""
        auth_manager = cls()
        auth_file_path = Path(auth_file)
        key_file_path = auth_file_path.parent / ".keyfile"

        if not auth_file_path.exists():
            auth_manager._create_new_auth_file(auth_file_path, key_file_path)
            return auth_manager

        try:
            # Load encryption key from SEPARATE file
            auth_manager._load_key(key_file_path)

            with open(auth_file_path, "r") as f:
                data = json.load(f)

            # Migrate: if the old file still embeds the key, pull it out
            legacy_key = data.get("encryption_key")
            if legacy_key and not auth_manager.fernet:
                auth_manager.encryption_key = base64.b64decode(legacy_key.encode())
                auth_manager.fernet = Fernet(auth_manager.encryption_key)
                # Persist key to separate file and re-save creds without it
                auth_manager._save_key(key_file_path)
                auth_manager.save(str(auth_file_path))

            # Load encrypted credentials
            encrypted_credentials = data.get("credentials", {})
            for provider, cred_data in encrypted_credentials.items():
                if auth_manager.fernet:
                    decrypted = {}
                    for key, value in cred_data.items():
                        try:
                            decrypted[key] = auth_manager.fernet.decrypt(
                                value.encode()
                            ).decode()
                        except Exception:
                            decrypted[key] = value
                    auth_manager.credentials[provider] = decrypted
                else:
                    auth_manager.credentials[provider] = cred_data

        except Exception as e:
            print(f"Error loading auth file: {e}")
            auth_manager._create_new_auth_file(auth_file_path, key_file_path)

        return auth_manager

    # ── Key management (separate from credential store) ──────────

    def _load_key(self, key_file_path: Path) -> None:
        """Load encryption key from its own file."""
        if key_file_path.exists():
            raw = key_file_path.read_bytes().strip()
            self.encryption_key = raw
            self.fernet = Fernet(self.encryption_key)

    def _save_key(self, key_file_path: Path) -> None:
        """Persist encryption key to its own file with strict permissions."""
        key_file_path.parent.mkdir(parents=True, exist_ok=True)
        key_file_path.write_bytes(self.encryption_key)
        os.chmod(key_file_path, 0o600)

    def _create_new_auth_file(self, auth_file_path: Path, key_file_path: Path) -> None:
        """Create new authentication file with a separate key file."""
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self._save_key(key_file_path)
        self.save(str(auth_file_path))

    def save(self, auth_file: str) -> None:
        """Save authentication to file"""
        auth_file_path = Path(auth_file)
        auth_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Encrypt credentials — refuse to write plaintext
        if not self.fernet:
            raise RuntimeError(
                "No encryption key available — cannot save credentials without encryption. "
                "Run `terradev configure` to re-initialize."
            )
        encrypted_credentials = {}
        for provider, cred_data in self.credentials.items():
            encrypted = {}
            for key, value in cred_data.items():
                encrypted[key] = self.fernet.encrypt(value.encode()).decode()
            encrypted_credentials[provider] = encrypted

        # Prepare data — encryption key is stored separately, NOT here
        data = {
            "credentials": encrypted_credentials,
            "version": "2.0",
        }

        with open(auth_file_path, "w") as f:
            json.dump(data, f, indent=2)

        # Set secure permissions
        os.chmod(auth_file_path, 0o600)

    def set_credentials(
        self, provider: str, api_key: str, secret_key: Optional[str] = None
    ) -> None:
        """Set credentials for a provider"""
        self.credentials[provider] = {
            "api_key": api_key,
            "secret_key": secret_key or "",
            "updated_at": str(os.times()[4]),  # Current time
        }

    def get_credentials(self, provider: str) -> Optional[Dict[str, str]]:
        """Get credentials for a provider"""
        return self.credentials.get(provider)

    def has_credentials(self) -> bool:
        """Check if any credentials are configured"""
        return len(self.credentials) > 0

    def remove_credentials(self, provider: str) -> bool:
        """Remove credentials for a provider"""
        if provider in self.credentials:
            del self.credentials[provider]
            return True
        return False

    def list_providers(self) -> list:
        """List all configured providers"""
        return list(self.credentials.keys())

    def validate_credentials(self, provider: str) -> bool:
        """Validate that credentials are present for provider"""
        cred = self.get_credentials(provider)
        if not cred:
            return False

        # Check that API key is present
        if not cred.get("api_key"):
            return False

        return True

    def get_provider_auth_headers(self, provider: str) -> Dict[str, str]:
        """Get authentication headers for provider"""
        cred = self.get_credentials(provider)
        if not cred:
            return {}

        headers = {}

        # Provider-specific header formats
        if provider == "aws":
            # AWS uses signature v4 - handled by boto3
            pass
        elif provider == "gcp":
            # Google Cloud uses OAuth2 token
            headers["Authorization"] = f"Bearer {cred['api_key']}"
        elif provider == "azure":
            # Azure uses Bearer token
            headers["Authorization"] = f"Bearer {cred['api_key']}"
        elif provider in ["runpod", "vastai", "lambda_labs", "coreweave", "tensordock"]:
            # Most GPU cloud providers use API key in header
            headers["Authorization"] = f"Bearer {cred['api_key']}"

        return headers

    def rotate_api_key(self, provider: str, new_api_key: str) -> bool:
        """Rotate API key for provider"""
        if provider not in self.credentials:
            return False

        # Store old key for backup
        old_key = self.credentials[provider]["api_key"]

        # Update with new key
        self.credentials[provider]["api_key"] = new_api_key
        self.credentials[provider]["previous_key"] = old_key
        self.credentials[provider]["rotated_at"] = str(os.times()[4])

        return True

    def backup_credentials(self, backup_file: str) -> bool:
        """Create encrypted backup of credentials"""
        try:
            backup_path = Path(backup_file)
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Encrypt each credential value before writing
            encrypted_credentials = {}
            if self.fernet:
                for provider, cred_data in self.credentials.items():
                    encrypted = {}
                    for key, value in cred_data.items():
                        encrypted[key] = self.fernet.encrypt(value.encode()).decode()
                    encrypted_credentials[provider] = encrypted
            else:
                raise RuntimeError(
                    "No encryption key available — cannot create backup without encryption"
                )

            backup_data = {
                "credentials": encrypted_credentials,
                "version": "2.0",
                "backup_date": str(os.times()[4]),
            }

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2)

            # Set secure permissions
            os.chmod(backup_path, 0o600)

            return True

        except Exception as e:
            print(f"Error creating backup: {e}")
            return False

    def restore_credentials(self, backup_file: str) -> bool:
        """Restore credentials from encrypted backup"""
        try:
            backup_path = Path(backup_file)

            if not backup_path.exists():
                print(f"Backup file not found: {backup_file}")
                return False

            if not self.fernet:
                raise RuntimeError(
                    "No encryption key available — cannot restore backup without encryption"
                )

            with open(backup_path, "r") as f:
                backup_data = json.load(f)

            # Decrypt each credential value before restoring
            encrypted_credentials = backup_data.get("credentials", {})
            decrypted_credentials = {}
            for provider, cred_data in encrypted_credentials.items():
                decrypted = {}
                for key, value in cred_data.items():
                    try:
                        decrypted[key] = self.fernet.decrypt(value.encode()).decode()
                    except Exception:
                        decrypted[key] = value
                decrypted_credentials[provider] = decrypted

            self.credentials = decrypted_credentials
            return True

        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False

    def clear_all_credentials(self) -> None:
        """Clear all credentials"""
        self.credentials.clear()

    def get_credential_summary(self) -> Dict[str, str]:
        """Get summary of configured credentials"""
        summary = {}
        for provider in self.credentials:
            cred = self.credentials[provider]
            summary[provider] = {
                "has_api_key": bool(cred.get("api_key")),
                "has_secret_key": bool(cred.get("secret_key")),
                "last_updated": cred.get("updated_at", "Unknown"),
            }
        return summary
