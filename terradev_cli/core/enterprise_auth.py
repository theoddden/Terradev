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
    """Enterprise user roles."""

    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    VIEWER = "viewer"


class AuthProvider(Enum):
    """Authentication providers."""

    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    AZURE_AD = "azure_ad"
    OKTA = "okta"


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
