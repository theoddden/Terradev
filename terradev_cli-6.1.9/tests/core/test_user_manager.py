"""Tests for terradev_cli.core.user_manager.

User management is the enterprise identity layer. These tests cover tenants,
users, invites, and bulk import.
"""

import pytest

from terradev_cli.core.enterprise_auth import AuthProvider, UserRole
from terradev_cli.core.user_manager import UserManager


@pytest.fixture
def manager(tmp_path):
    return UserManager(config_path=str(tmp_path / "users.json"))


def test_create_and_get_tenant(manager):
    """Tenants can be created and retrieved."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    assert tenant.name == "acme"
    assert tenant.domain == "acme.example.com"
    assert manager.get_tenant(tenant.tenant_id) == tenant
    assert tenant in manager.list_tenants()


def test_update_tenant(manager):
    """Tenant fields can be updated and persisted."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    assert manager.update_tenant(tenant.tenant_id, max_users=500) is True
    assert manager.get_tenant(tenant.tenant_id).max_users == 500
    assert manager.update_tenant("missing", max_users=1) is False


def test_create_and_get_user(manager):
    """Users can be created, retrieved, and listed by tenant."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    user = manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "role": "developer",
            "tenant_id": tenant.tenant_id,
            "provider": "local",
        }
    )
    assert user.email == "alice@example.com"
    assert user.role == UserRole.DEVELOPER
    assert user.provider == AuthProvider.LOCAL

    assert manager.get_user(user.user_id) == user
    assert manager.get_user_by_email("ALICE@EXAMPLE.COM") == user
    assert manager.get_users_by_tenant(tenant.tenant_id) == [user]


def test_tenant_user_limit(manager):
    """Creating users beyond the tenant max raises ValueError."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    tenant.max_users = 1

    manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "tenant_id": tenant.tenant_id,
        }
    )

    with pytest.raises(ValueError, match="maximum user limit"):
        manager.create_user(
            {
                "email": "bob@example.com",
                "name": "Bob",
                "tenant_id": tenant.tenant_id,
            }
        )


def test_update_and_delete_user(manager):
    """Users can be updated and deleted."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    user = manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "tenant_id": tenant.tenant_id,
        }
    )

    assert manager.update_user(user.user_id, name="Alice Smith") is True
    assert manager.get_user(user.user_id).name == "Alice Smith"

    assert manager.update_user(user.user_id, role="admin") is True
    assert manager.get_user(user.user_id).role == UserRole.ADMIN

    assert manager.delete_user(user.user_id) is True
    assert manager.get_user(user.user_id) is None
    assert manager.delete_user(user.user_id) is False


def test_invite_user_and_accept(manager):
    """Invitations can be created, looked up, and accepted."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    user = manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "tenant_id": tenant.tenant_id,
        }
    )

    invite = manager.invite_user(
        tenant_id=tenant.tenant_id,
        email="bob@example.com",
        role=UserRole.DEVELOPER,
        invited_by=user.user_id,
    )
    assert invite.email == "bob@example.com"
    assert not invite.accepted

    assert manager.get_invite(invite.invite_id) == invite
    assert manager.get_invite_by_email("bob@example.com") == invite

    new_user = manager.create_user(
        {
            "email": "bob@example.com",
            "name": "Bob",
            "tenant_id": "",
        }
    )
    assert manager.accept_invite(invite.invite_id, new_user.user_id) is True
    assert manager.get_user(new_user.user_id).tenant_id == tenant.tenant_id
    assert manager.get_user(new_user.user_id).role == UserRole.DEVELOPER

    # Second accept fails
    assert manager.accept_invite(invite.invite_id, new_user.user_id) is False


def test_invite_existing_user_fails(manager):
    """Inviting an existing user email raises ValueError."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "tenant_id": tenant.tenant_id,
        }
    )

    with pytest.raises(ValueError, match="already exists"):
        manager.invite_user(
            tenant_id=tenant.tenant_id,
            email="alice@example.com",
            role=UserRole.DEVELOPER,
            invited_by="admin",
        )


def test_cleanup_expired_invites(manager):
    """Expired invitations can be cleaned up."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    invite = manager.invite_user(
        tenant_id=tenant.tenant_id,
        email="bob@example.com",
        role=UserRole.DEVELOPER,
        invited_by="admin",
        expires_days=-1,
    )
    assert manager.get_invite(invite.invite_id) is not None
    assert manager.cleanup_expired_invites() == 1
    assert manager.get_invite(invite.invite_id) is None


def test_get_user_statistics(manager):
    """Statistics aggregate users by role and provider."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    manager.create_user(
        {
            "email": "alice@example.com",
            "name": "Alice",
            "tenant_id": tenant.tenant_id,
            "role": "admin",
            "provider": "google",
        }
    )
    manager.create_user(
        {
            "email": "bob@example.com",
            "name": "Bob",
            "tenant_id": tenant.tenant_id,
            "role": "developer",
            "provider": "local",
        }
    )

    stats = manager.get_user_statistics(tenant.tenant_id)
    assert stats["total_users"] == 2
    assert stats["by_role"]["admin"] == 1
    assert stats["by_role"]["developer"] == 1
    assert stats["by_provider"]["google"] == 1
    assert stats["by_provider"]["local"] == 1


def test_bulk_import_users(manager):
    """Bulk import creates users up to the tenant limit."""
    tenant = manager.create_tenant("acme", "acme.example.com")
    tenant.max_users = 3

    results = manager.bulk_import_users(
        [
            {"email": "a@example.com", "name": "A", "role": "developer"},
            {"email": "b@example.com", "name": "B", "role": "developer"},
            {"email": "a@example.com", "name": "A", "role": "developer"},
        ],
        tenant.tenant_id,
    )
    assert results["imported"] == 2
    assert results["skipped"] == 1

    # Exceed tenant limit
    results = manager.bulk_import_users(
        [
            {"email": "c@example.com", "name": "C", "role": "developer"},
        ],
        tenant.tenant_id,
    )
    assert results["imported"] == 1
    assert len(manager.get_users_by_tenant(tenant.tenant_id)) == 3
