"""Tests for terradev_cli.core.enterprise_auth.

Enterprise authentication primitives define user roles, providers, and the
EnterpriseUser record used by the user manager.
"""

from datetime import datetime

from terradev_cli.core.enterprise_auth import (
    AuthProvider,
    EnterpriseUser,
    UserRole,
)


def test_user_role_enum():
    """UserRole exposes the expected role values."""
    assert UserRole.ADMIN.value == "admin"
    assert UserRole.DEVELOPER.value == "developer"
    assert UserRole.OPERATOR.value == "operator"
    assert UserRole.VIEWER.value == "viewer"


def test_auth_provider_enum():
    """AuthProvider exposes the expected provider values."""
    assert AuthProvider.LOCAL.value == "local"
    assert AuthProvider.GOOGLE.value == "google"
    assert AuthProvider.GITHUB.value == "github"
    assert AuthProvider.AZURE_AD.value == "azure_ad"
    assert AuthProvider.OKTA.value == "okta"


def test_enterprise_user_defaults():
    """EnterpriseUser provides sensible defaults for optional fields."""
    user = EnterpriseUser(
        user_id="u-1",
        email="dev@example.com",
        name="Dev User",
        role=UserRole.DEVELOPER,
        tenant_id="t-1",
        provider=AuthProvider.LOCAL,
        provider_user_id="local-1",
        created_at=datetime.now(),
    )
    assert user.mfa_enabled is False
    assert user.permissions == []
    assert user.metadata == {}
    assert user.last_login is None


def test_enterprise_user_custom_metadata():
    """EnterpriseUser accepts metadata and permissions."""
    user = EnterpriseUser(
        user_id="u-2",
        email="admin@example.com",
        name="Admin User",
        role=UserRole.ADMIN,
        tenant_id="t-1",
        provider=AuthProvider.OKTA,
        provider_user_id="okta-2",
        created_at=datetime.now(),
        last_login=datetime.now(),
        mfa_enabled=True,
        mfa_secret="secret",
        permissions=["provision", "delete"],
        metadata={"team": "platform"},
    )
    assert user.mfa_enabled is True
    assert user.permissions == ["provision", "delete"]
    assert user.metadata["team"] == "platform"
