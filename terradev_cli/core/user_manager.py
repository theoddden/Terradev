#!/usr/bin/env python3
"""
Terradev User Management
Handles enterprise user management and role assignments
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .enterprise_auth import EnterpriseUser, UserRole, AuthProvider

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Enterprise tenant/organization"""
    tenant_id: str
    name: str
    domain: str
    created_at: datetime
    settings: Dict[str, Any]
    max_users: int = 100
    active: bool = True


@dataclass
class UserInvite:
    """User invitation for onboarding"""
    invite_id: str
    tenant_id: str
    email: str
    role: UserRole
    invited_by: str
    created_at: datetime
    expires_at: datetime
    accepted: bool = False
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None


class UserManager:
    """Enterprise user management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or (Path.home() / ".terradev" / "user_management.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, EnterpriseUser] = {}
        self.invites: Dict[str, UserInvite] = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load user management data"""
        data_file = self.config_path
        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)
                
                # Load tenants
                for tenant_data in data.get('tenants', []):
                    tenant = Tenant(
                        tenant_id=tenant_data['tenant_id'],
                        name=tenant_data['name'],
                        domain=tenant_data['domain'],
                        created_at=datetime.fromisoformat(tenant_data['created_at']),
                        settings=tenant_data['settings'],
                        max_users=tenant_data.get('max_users', 100),
                        active=tenant_data.get('active', True)
                    )
                    self.tenants[tenant.tenant_id] = tenant
                
                # Load users
                for user_data in data.get('users', []):
                    user = EnterpriseUser(
                        user_id=user_data['user_id'],
                        email=user_data['email'],
                        name=user_data['name'],
                        role=UserRole(user_data['role']),
                        tenant_id=user_data['tenant_id'],
                        provider=AuthProvider(user_data['provider']),
                        provider_user_id=user_data['provider_user_id'],
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                        mfa_enabled=user_data.get('mfa_enabled', False),
                        mfa_secret=user_data.get('mfa_secret'),
                        permissions=user_data.get('permissions', []),
                        metadata=user_data.get('metadata', {})
                    )
                    self.users[user.user_id] = user
                
                # Load invites
                for invite_data in data.get('invites', []):
                    invite = UserInvite(
                        invite_id=invite_data['invite_id'],
                        tenant_id=invite_data['tenant_id'],
                        email=invite_data['email'],
                        role=UserRole(invite_data['role']),
                        invited_by=invite_data['invited_by'],
                        created_at=datetime.fromisoformat(invite_data['created_at']),
                        expires_at=datetime.fromisoformat(invite_data['expires_at']),
                        accepted=invite_data.get('accepted', False),
                        accepted_by=invite_data.get('accepted_by'),
                        accepted_at=datetime.fromisoformat(invite_data['accepted_at']) if invite_data.get('accepted_at') else None
                    )
                    self.invites[invite.invite_id] = invite
                    
            except Exception as e:
                logger.error(f"Failed to load user management data: {e}")
    
    def _save_data(self):
        """Save user management data"""
        data = {
            'tenants': [asdict(tenant) for tenant in self.tenants.values()],
            'users': [self._serialize_user(user) for user in self.users.values()],
            'invites': [asdict(invite) for invite in self.invites.values()]
        }
        
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _serialize_user(self, user: EnterpriseUser) -> Dict[str, Any]:
        """Serialize user for JSON storage"""
        return {
            'user_id': user.user_id,
            'email': user.email,
            'name': user.name,
            'role': user.role.value,
            'tenant_id': user.tenant_id,
            'provider': user.provider.value,
            'provider_user_id': user.provider_user_id,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'mfa_enabled': user.mfa_enabled,
            'mfa_secret': user.mfa_secret,
            'permissions': user.permissions,
            'metadata': user.metadata
        }
    
    def create_tenant(self, name: str, domain: str, settings: Optional[Dict[str, Any]] = None) -> Tenant:
        """Create a new tenant"""
        tenant_id = str(uuid.uuid4())
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            domain=domain,
            created_at=datetime.utcnow(),
            settings=settings or {},
            max_users=100,
            active=True
        )
        
        self.tenants[tenant_id] = tenant
        self._save_data()
        
        logger.info(f"Created tenant: {name} ({tenant_id})")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants"""
        return list(self.tenants.values())
    
    def update_tenant(self, tenant_id: str, **kwargs) -> bool:
        """Update tenant settings"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        self._save_data()
        return True
    
    def create_user(self, user_data: Dict[str, Any]) -> EnterpriseUser:
        """Create a new user"""
        user = EnterpriseUser(
            user_id=str(uuid.uuid4()),
            email=user_data['email'],
            name=user_data['name'],
            role=UserRole(user_data.get('role', 'developer')),
            tenant_id=user_data['tenant_id'],
            provider=AuthProvider(user_data.get('provider', 'local')),
            provider_user_id=user_data.get('provider_user_id', user_data['email']),
            created_at=datetime.utcnow(),
            permissions=user_data.get('permissions', []),
            metadata=user_data.get('metadata', {})
        )
        
        # Check tenant user limit
        tenant = self.tenants.get(user.tenant_id)
        if tenant and len(self.get_users_by_tenant(user.tenant_id)) >= tenant.max_users:
            raise ValueError(f"Tenant {tenant.name} has reached maximum user limit")
        
        self.users[user.user_id] = user
        self._save_data()
        
        logger.info(f"Created user: {user.email} ({user.user_id})")
        return user
    
    def get_user(self, user_id: str) -> Optional[EnterpriseUser]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[EnterpriseUser]:
        """Get user by email"""
        for user in self.users.values():
            if user.email.lower() == email.lower():
                return user
        return None
    
    def get_users_by_tenant(self, tenant_id: str) -> List[EnterpriseUser]:
        """Get all users in a tenant"""
        return [user for user in self.users.values() if user.tenant_id == tenant_id]
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user information"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                if key == 'role' and isinstance(value, str):
                    setattr(user, key, UserRole(value))
                else:
                    setattr(user, key, value)
        
        self._save_data()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        del self.users[user_id]
        self._save_data()
        
        logger.info(f"Deleted user: {user.email} ({user_id})")
        return True
    
    def invite_user(self, tenant_id: str, email: str, role: UserRole, invited_by: str, 
                    expires_days: int = 7) -> UserInvite:
        """Invite a user to join a tenant"""
        # Check if user already exists
        existing_user = self.get_user_by_email(email)
        if existing_user:
            raise ValueError(f"User {email} already exists")
        
        # Check tenant exists
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Check tenant user limit
        current_users = len(self.get_users_by_tenant(tenant_id))
        if current_users >= tenant.max_users:
            raise ValueError(f"Tenant {tenant.name} has reached maximum user limit")
        
        # Create invite
        invite_id = str(uuid.uuid4())
        invite = UserInvite(
            invite_id=invite_id,
            tenant_id=tenant_id,
            email=email,
            role=role,
            invited_by=invited_by,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_days)
        )
        
        self.invites[invite_id] = invite
        self._save_data()
        
        logger.info(f"Created invite for {email} to tenant {tenant.name}")
        return invite
    
    def get_invite(self, invite_id: str) -> Optional[UserInvite]:
        """Get invite by ID"""
        return self.invites.get(invite_id)
    
    def get_invite_by_email(self, email: str) -> Optional[UserInvite]:
        """Get pending invite by email"""
        for invite in self.invites.values():
            if (invite.email.lower() == email.lower() and 
                not invite.accepted and 
                datetime.utcnow() < invite.expires_at):
                return invite
        return None
    
    def accept_invite(self, invite_id: str, user_id: str) -> bool:
        """Accept a user invitation"""
        invite = self.invites.get(invite_id)
        if not invite:
            return False
        
        if invite.accepted:
            return False
        
        if datetime.utcnow() > invite.expires_at:
            return False
        
        # Update invite
        invite.accepted = True
        invite.accepted_by = user_id
        invite.accepted_at = datetime.utcnow()
        
        # Update user's tenant and role
        user = self.users.get(user_id)
        if user:
            user.tenant_id = invite.tenant_id
            user.role = invite.role
        
        self._save_data()
        
        logger.info(f"Invite {invite_id} accepted by user {user_id}")
        return True
    
    def cleanup_expired_invites(self) -> int:
        """Clean up expired invitations"""
        expired_count = 0
        now = datetime.utcnow()
        
        expired_invites = [
            invite_id for invite_id, invite in self.invites.items()
            if not invite.accepted and now > invite.expires_at
        ]
        
        for invite_id in expired_invites:
            del self.invites[invite_id]
            expired_count += 1
        
        if expired_count > 0:
            self._save_data()
            logger.info(f"Cleaned up {expired_count} expired invites")
        
        return expired_count
    
    def get_user_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user statistics"""
        users = self.get_users_by_tenant(tenant_id) if tenant_id else list(self.users.values())
        
        stats = {
            'total_users': len(users),
            'by_role': {},
            'by_provider': {},
            'mfa_enabled': 0,
            'active_today': 0,
            'active_this_week': 0,
            'active_this_month': 0
        }
        
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        for user in users:
            # By role
            role_name = user.role.value
            stats['by_role'][role_name] = stats['by_role'].get(role_name, 0) + 1
            
            # By provider
            provider_name = user.provider.value
            stats['by_provider'][provider_name] = stats['by_provider'].get(provider_name, 0) + 1
            
            # MFA
            if user.mfa_enabled:
                stats['mfa_enabled'] += 1
            
            # Activity
            if user.last_login:
                if user.last_login >= today:
                    stats['active_today'] += 1
                if user.last_login >= week_ago:
                    stats['active_this_week'] += 1
                if user.last_login >= month_ago:
                    stats['active_this_month'] += 1
        
        return stats
    
    def bulk_import_users(self, users_data: List[Dict[str, Any]], tenant_id: str) -> Dict[str, Any]:
        """Bulk import users for a tenant"""
        results = {
            'imported': 0,
            'skipped': 0,
            'errors': []
        }
        
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            results['errors'].append(f"Tenant {tenant_id} not found")
            return results
        
        current_users = len(self.get_users_by_tenant(tenant_id))
        available_slots = tenant.max_users - current_users
        
        for user_data in users_data:
            if available_slots <= 0:
                results['errors'].append("Tenant user limit reached")
                break
            
            try:
                # Check if user already exists
                if self.get_user_by_email(user_data['email']):
                    results['skipped'] += 1
                    continue
                
                # Create user
                user_data['tenant_id'] = tenant_id
                self.create_user(user_data)
                results['imported'] += 1
                available_slots -= 1
                
            except Exception as e:
                results['errors'].append(f"Failed to import {user_data.get('email', 'unknown')}: {e}")
        
        self._save_data()
        return results
