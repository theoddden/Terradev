"""Enterprise authentication primitives used by user_manager.

This module provides the minimal UserRole / AuthProvider / EnterpriseUser
shapes expected by terradev_cli.core.user_manager. It is intentionally
lightweight: real SSO/OIDC integration lives in terradev_cli.core.oidc_provider.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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

    Lightweight manager that tracks enabled providers and session policy.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {
            "sso_providers": ["local", "saml", "oidc"],
            "session_timeout_hours": 8,
            "mfa_required_for_roles": ["super_admin", "org_admin"],
        }

    def get_default_config(self) -> Dict[str, Any]:
        return self.config.copy()
