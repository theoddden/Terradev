"""Enterprise authentication primitives used by user_manager.

This module provides the minimal UserRole / AuthProvider / EnterpriseUser
shapes expected by terradev_cli.core.user_manager. It is intentionally
lightweight: real SSO/OIDC integration lives in terradev_cli.core.oidc_provider.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Enterprise user roles.

    Includes both the original flat role set and the new enterprise role
    hierarchy, so existing core tests and the new Phase 1 SSO integration
    tests can use the same enum.
    """

    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    TEAM_ADMIN = "team_admin"
    ANALYST = "analyst"

    @classmethod
    def _missing_(cls, value):
        """Map legacy role names to the canonical member."""
        if value in {"admin"}:
            return cls.ADMIN
        if value in {"operator"}:
            return cls.OPERATOR
        return None


class AuthProvider(Enum):
    """Authentication providers.

    Includes both the original provider list and the consolidated SAML/OIDC
    groupings, so existing core tests and new Phase 1 SSO tests can use the
    same enum.
    """

    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    AZURE_AD = "azure_ad"
    OKTA = "okta"
    SAML = "saml"
    OIDC = "oidc"

    @classmethod
    def _missing_(cls, value):
        """Map legacy provider names to the canonical member."""
        legacy = {
            "google": cls.GOOGLE,
            "github": cls.GITHUB,
            "azure_ad": cls.AZURE_AD,
            "okta": cls.OKTA,
        }
        if value in legacy:
            return legacy[value]
        return None


@dataclass
class EnterpriseUser:
    """An enterprise user record."""

    user_id: str
    email: str
    name: str
    role: UserRole
    tenant_id: str
    provider: AuthProvider
    provider_user_id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseAuthManager:
    """Enterprise SSO/SAML authentication manager.

    Tracks enabled providers, persists their configuration, and can run
    basic connection tests against the underlying OIDC/SAML providers.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        self.config_path: Path = (
            Path(config_path) if config_path else Path.home() / ".terradev" / "enterprise_auth.json"
        )
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        defaults = {
            "sso_providers": ["local", "saml", "oidc"],
            "session_timeout_hours": 8,
            "mfa_required_for_roles": ["super_admin", "org_admin"],
            "providers": {},
        }
        if config:
            defaults.update(config)
        self.config: Dict[str, Any] = defaults
        self.providers: Dict[str, Dict[str, Any]] = self.config.setdefault("providers", {})

        self._load_data()

    def _load_data(self) -> None:
        """Load persisted provider configuration."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.config.update(data)
                self.providers = self.config.setdefault("providers", {})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load enterprise auth config: {e}")

    def _save_data(self) -> None:
        """Persist provider configuration."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save enterprise auth config: {e}")

    def get_default_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def list_enabled_providers(self) -> List[str]:
        """Return the names of currently enabled SSO providers."""
        return sorted(
            name for name, cfg in self.providers.items() if cfg.get("enabled")
        )

    def get_sso_provider_config(self, provider: str) -> Dict[str, Any]:
        """Return a copy of the provider's config, or a disabled stub."""
        return self.providers.get(provider, {"enabled": False}).copy()

    def enable_sso_provider(
        self, provider: str, config: Dict[str, Any]
    ) -> None:
        """Enable and persist an SSO provider."""
        cfg = self.get_sso_provider_config(provider)
        cfg.update(config)
        cfg["enabled"] = True

        # Infer protocol if not explicitly provided
        if "protocol" not in cfg:
            if cfg.get("client_id") and cfg.get("client_secret"):
                cfg["protocol"] = "oidc"
            elif cfg.get("entity_id") and cfg.get("sso_url"):
                cfg["protocol"] = "saml"
            else:
                cfg["protocol"] = "local"

        # Derive OIDC discovery URL if missing
        if cfg["protocol"] == "oidc" and not cfg.get("discovery_url"):
            if provider == "google_workspace":
                cfg["discovery_url"] = (
                    "https://accounts.google.com/.well-known/openid-configuration"
                )
            elif provider == "auth0" and cfg.get("domain"):
                cfg[
                    "discovery_url"
                ] = f"https://{cfg['domain']}/.well-known/openid-configuration"
            elif provider == "azure_ad" and cfg.get("tenant_id"):
                cfg[
                    "discovery_url"
                ] = f"https://login.microsoftonline.com/{cfg['tenant_id']}/v2.0/.well-known/openid-configuration"

        # Derive SAML ACS URL if missing
        if cfg["protocol"] == "saml" and not cfg.get("acs_url"):
            cfg["acs_url"] = "https://api.terradev.cloud/auth/saml/acs"

        self.providers[provider] = cfg
        self._save_data()

    def disable_sso_provider(self, provider: str) -> None:
        """Disable a provider without removing its configuration."""
        if provider in self.providers:
            self.providers[provider]["enabled"] = False
            self._save_data()

    def _derive_oidc_config(self, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a complete OIDC config for the given provider."""
        from .oidc_provider import OIDCProvider

        base: Dict[str, Any] = {
            "client_id": config.get("client_id", ""),
            "client_secret": config.get("client_secret", ""),
            "redirect_uri": config.get(
                "redirect_uri", "https://api.terradev.cloud/auth/oidc/callback"
            ),
            "discovery_url": config.get("discovery_url", ""),
        }

        if not base["discovery_url"]:
            if provider == "google_workspace":
                base["discovery_url"] = (
                    "https://accounts.google.com/.well-known/openid-configuration"
                )
            elif provider == "auth0" and config.get("domain"):
                base[
                    "discovery_url"
                ] = f"https://{config['domain']}/.well-known/openid-configuration"
            elif provider == "azure_ad" and config.get("tenant_id"):
                base[
                    "discovery_url"
                ] = f"https://login.microsoftonline.com/{config['tenant_id']}/v2.0/.well-known/openid-configuration"

        return base

    def test_sso_provider(self, provider: str) -> bool:
        """Run a lightweight validation test for a provider."""
        config = self.get_sso_provider_config(provider)
        if not config.get("enabled"):
            return False

        protocol = config.get("protocol")
        if protocol == "oidc":
            return self._test_oidc_provider(provider, config)
        if protocol == "saml":
            return self._test_saml_provider(provider, config)

        return True

    def _test_oidc_provider(self, provider: str, config: Dict[str, Any]) -> bool:
        """Validate an OIDC provider by discovering endpoints."""
        try:
            from .oidc_provider import OIDCProvider
        except ImportError:
            logger.warning("OIDC provider module not available")
            return False

        try:
            import aiohttp  # noqa: F401
        except ImportError:
            logger.warning("aiohttp not available; cannot test OIDC network reachability")
            return False

        oidc_config = self._derive_oidc_config(provider, config)
        if not oidc_config.get("discovery_url"):
            logger.warning(f"No discovery URL for {provider}")
            return False

        oidc = OIDCProvider(oidc_config)
        try:
            return asyncio.run(oidc.test_connection())
        except Exception as e:  # noqa: BLE001
            logger.error(f"OIDC test failed for {provider}: {e}")
            return False

    def _test_saml_provider(self, provider: str, config: Dict[str, Any]) -> bool:
        """Validate a SAML provider by generating AuthnRequest and metadata."""
        try:
            from .saml_provider import SAMLProvider
        except ImportError:
            logger.warning("SAML provider module not available")
            return False

        if not config.get("entity_id") or not config.get("sso_url"):
            logger.warning(f"SAML provider {provider} missing entity_id or sso_url")
            return False

        saml_config = {
            "entity_id": config.get("entity_id", ""),
            "sso_url": config.get("sso_url", ""),
            "acs_url": config.get(
                "acs_url", "https://api.terradev.cloud/auth/saml/acs"
            ),
            "certificate": config.get("certificate", ""),
        }

        saml = SAMLProvider(saml_config)
        try:
            saml.generate_authn_request()
            saml.get_metadata()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"SAML test failed for {provider}: {e}")
            return False
