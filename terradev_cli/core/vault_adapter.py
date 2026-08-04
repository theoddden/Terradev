#!/usr/bin/env python3
"""Vault adapter: bridge Terradev CLI credentials with environment variables.

The vault uses the existing AuthManager-encrypted ``~/.terradev/credentials.json``
for at-rest storage and the CredentialVault (Rust-backed when available) for
in-memory secret handling.  In CI/CD pipelines, secrets can be passed as
``TERRADEV_<PROVIDER>_<KEY>`` environment variables and imported with a single
``terradev vault sync``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from terradev_cli.core.auth import AuthManager
from terradev_cli.core.credential_vault import CredentialVault


# Provider credential schemas used for env parsing and ``vault verify``.
# These must stay in sync with ``terradev_cli.commands._api.validate_credentials``.
PROVIDER_SCHEMAS: Dict[str, List[str]] = {
    "runpod": ["api_key"],
    "vastai": ["api_key"],
    "aws": ["api_key", "secret_key"],
    "gcp": ["project_id", "credentials_file"],
    "azure": ["subscription_id", "tenant_id", "client_id", "client_secret"],
    "lambda_labs": ["api_key"],
    "coreweave": ["api_key"],
    "tensordock": ["api_key", "api_token"],
    "huggingface": ["api_key", "namespace"],
    "baseten": ["api_key"],
    "oracle": ["api_key", "tenancy_ocid", "compartment_ocid", "region"],
    "crusoe": ["access_key", "secret_key", "project_id"],
}

# Convenience aliases for keys that are named differently in env vars.
KEY_ALIASES: Dict[str, str] = {
    "access_key_id": "api_key",  # AWS
    "secret_access_key": "secret_key",  # AWS
    "service_account_key": "credentials_file",  # GCP
    "subscription": "subscription_id",  # Azure
    "tenant": "tenant_id",  # Azure
    "client": "client_id",  # Azure
    "secret": "client_secret",  # Azure
    "lambda_labs_api_key": "api_key",
    "region_name": "region",  # Oracle
}


class VaultAdapter:
    """Read, write, and merge credentials from encrypted file and env vars."""

    ENV_PREFIX = "TERRADEV_"
    _no_persist = False
    EXCLUDED = {
        "TERRADEV_SKIP_ONBOARDING",
        "TERRADEV_OUTPUT",
        "TERRADEV_NO_PERSIST",
        "TERRADEV_NODE_HEARTBEAT",
        "TERRADEV_NODE_HEARTBEAT_INTERVAL",
    }

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or (Path.home() / ".terradev")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_file = self.config_dir / "credentials.json"
        self.key_file = self.config_dir / ".keyfile"
        self._no_persist = os.environ.get("TERRADEV_NO_PERSIST", "").lower() in (
            "1",
            "true",
            "yes",
        )

    # ── AuthManager helpers ─────────────────────────────────────────────────

    def _load_auth(self) -> AuthManager:
        return AuthManager.load(str(self.credentials_file))

    def _save_auth(self, auth: AuthManager) -> None:
        if self._no_persist:
            return
        auth.save(str(self.credentials_file))

    # ── Credential loading / merging ────────────────────────────────────────

    def load_credentials(self) -> Dict[str, Dict[str, str]]:
        """Return a deep copy of stored credentials (file-backed)."""
        auth = self._load_auth()
        return self._deep_copy(auth.credentials)

    def save_credentials(self, credentials: Dict[str, Dict[str, str]]) -> None:
        """Persist credentials to encrypted file (unless NO_PERSIST)."""
        auth = self._load_auth()
        auth.credentials = self._deep_copy(credentials)
        self._save_auth(auth)

    def load_env_credentials(
        self,
        base: Optional[Dict] = None,
        known_only: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """Merge ``TERRADEV_*`` environment variables into credentials.

        When ``known_only`` is True, only env vars whose provider is in
        ``PROVIDER_SCHEMAS`` (or already present in ``base``) are merged.
        This prevents CI-only tokens such as ``TERRADEV_GITHUB_TOKEN``
        from being treated as cloud provider credentials.
        """
        credentials = self._deep_copy(base or {})
        for env_name, value in os.environ.items():
            if not value or not env_name.startswith(self.ENV_PREFIX):
                continue
            if env_name in self.EXCLUDED:
                continue
            provider, key = self.parse_env_name(env_name)
            if not provider or not key:
                continue
            if known_only and provider not in PROVIDER_SCHEMAS and provider not in credentials:
                continue
            credentials.setdefault(provider, {})[key] = value
        return credentials

    def all_credentials(self) -> Dict[str, Dict[str, str]]:
        """Stored credentials merged with environment variables."""
        return self.load_env_credentials(self.load_credentials())

    # ── Single-key operations ───────────────────────────────────────────────

    def set(self, provider: str, key: str, value: str) -> None:
        """Store a single credential and persist the vault."""
        provider = provider.lower()
        key = self._canonical_key(key)
        credentials = self.load_credentials()
        credentials.setdefault(provider, {})[key] = value
        self.save_credentials(credentials)

    def get(self, provider: str, key: str) -> Optional[str]:
        """Return the raw value for a provider/key (file or env)."""
        provider = provider.lower()
        key = self._canonical_key(key)
        creds = self.all_credentials()
        return creds.get(provider, {}).get(key)

    def remove(self, provider: str, key: Optional[str] = None) -> bool:
        """Delete a whole provider or a single key and persist."""
        provider = provider.lower()
        credentials = self.load_credentials()
        if key is None:
            if provider in credentials:
                del credentials[provider]
                self.save_credentials(credentials)
                return True
            return False

        key = self._canonical_key(key)
        provider_creds = credentials.get(provider)
        if provider_creds and key in provider_creds:
            del provider_creds[key]
            if not provider_creds:
                del credentials[provider]
            self.save_credentials(credentials)
            return True
        return False

    # ── Verification ────────────────────────────────────────────────────────

    def verify(self) -> Dict[str, Any]:
        """Check which providers are fully configured and which keys are missing."""
        creds = self.all_credentials()
        configured: List[str] = []
        missing: Dict[str, List[str]] = {}

        for provider, required_keys in PROVIDER_SCHEMAS.items():
            provider_creds = creds.get(provider, {})
            if not provider_creds:
                continue

            missing_keys = [
                k for k in required_keys if not provider_creds.get(k, "").strip()
            ]
            if missing_keys:
                missing[provider] = missing_keys
            else:
                configured.append(provider)

        # Providers present in env/file but not in schema: list as configured
        # if they have at least one non-empty value.
        for provider, provider_creds in creds.items():
            if provider not in PROVIDER_SCHEMAS and any(
                v.strip() for v in provider_creds.values()
            ):
                configured.append(provider)

        return {
            "configured": sorted(set(configured)),
            "missing": missing,
            "valid": not missing,
        }

    # ── Environment export helpers ──────────────────────────────────────────

    def to_env(self, provider: str) -> Dict[str, str]:
        """Convert provider credentials to env-var style exports."""
        provider = provider.lower()
        creds = self.all_credentials()
        provider_creds = creds.get(provider, {})
        return {
            f"{self.ENV_PREFIX}{provider.upper()}_{self._env_key(k)}": v
            for k, v in provider_creds.items()
        }

    def build_run_env(self, provider: Optional[str] = None) -> Dict[str, str]:
        """Build an environment for ``vault run`` / ``vault exec``."""
        env = dict(os.environ)
        if provider:
            env.update(self.to_env(provider))
        else:
            for prov in self.all_credentials():
                env.update(self.to_env(prov))
        return env

    def in_memory_vault(self) -> CredentialVault:
        """Return a CredentialVault pre-populated with all current secrets."""
        vault = CredentialVault()
        for provider, provider_creds in self.all_credentials().items():
            for key, value in provider_creds.items():
                vault.store(f"{provider}/{key}", value.encode(), provider)
        return vault

    # ── Environment naming helpers ──────────────────────────────────────────

    @classmethod
    def parse_env_name(cls, env_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse ``TERRADEV_PROVIDER_KEY`` into (provider, key).

        Known multi-word providers (e.g. ``lambda_labs``) are matched first.
        """
        if not env_name.startswith(cls.ENV_PREFIX):
            return None, None
        rest = env_name[len(cls.ENV_PREFIX) :]

        # Sort providers by length descending so ``lambda_labs`` beats ``lambda``.
        for provider in sorted(PROVIDER_SCHEMAS, key=len, reverse=True):
            prov_prefix = provider.upper() + "_"
            if rest.startswith(prov_prefix):
                key = rest[len(prov_prefix) :].lower()
                key = cls._canonical_key(key)
                return provider, key

        # Fallback: split on the first underscore.
        if "_" in rest:
            provider, key = rest.split("_", 1)
            return provider.lower(), cls._canonical_key(key)

        return None, None

    @staticmethod
    def _canonical_key(key: str) -> str:
        key = key.lower().strip()
        return KEY_ALIASES.get(key, key)

    @staticmethod
    def _env_key(key: str) -> str:
        # Some stored keys already contain underscores; preserve them for env form.
        return key.upper()

    @staticmethod
    def _deep_copy(credentials: Dict) -> Dict[str, Dict[str, str]]:
        return {
            provider: {k: str(v) for k, v in provider_creds.items()}
            for provider, provider_creds in (credentials or {}).items()
        }


def read_secret_from_stdin() -> str:
    """Read a secret from stdin, trimming whitespace."""
    if sys.stdin.isatty():
        return sys.stdin.readline().strip()
    return (sys.stdin.buffer.read().decode("utf-8")).strip()
