#!/usr/bin/env python3
"""
Terradev Enterprise Authentication Manager
Handles SSO integration with SAML 2.0 and OpenID Connect providers
"""

import json
import time
import uuid
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    import redis
except ImportError:
    redis = None

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError:
    hashes = PBKDF2HMAC = default_backend = None

logger = logging.getLogger(__name__)


class AuthProvider(Enum):
    """Supported authentication providers"""
    SAML = "saml"
    OIDC = "oidc"
    LOCAL = "local"


class UserRole(Enum):
    """Enterprise user roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"      # Full system access
    ORG_ADMIN = "org_admin"          # Organization management
    TEAM_ADMIN = "team_admin"        # Team management
    DEVELOPER = "developer"          # Resource provisioning
    ANALYST = "analyst"             # Read-only analytics
    VIEWER = "viewer"               # Basic read access


@dataclass
class EnterpriseUser:
    """Enterprise user profile with SSO integration"""
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


@dataclass
class AuthSession:
    """User session with enterprise features"""
    session_id: str
    user_id: str
    tenant_id: str
    created_at: datetime
    expires_at: datetime
    provider: AuthProvider
    mfa_verified: bool = False
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SessionStore:
    """Session storage backend with Redis support"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed, using memory storage: {e}")
        
        # Fallback to memory storage
        self._memory_store = {}
    
    async def store_session(self, session: AuthSession) -> bool:
        """Store session data"""
        session_data = {
            'user_id': session.user_id,
            'tenant_id': session.tenant_id,
            'created_at': session.created_at.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'provider': session.provider.value,
            'mfa_verified': session.mfa_verified,
            'ip_address': session.ip_address,
            'user_agent': session.user_agent
        }
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session.session_id}",
                    int((session.expires_at - datetime.utcnow()).total_seconds()),
                    json.dumps(session_data)
                )
                return True
            except Exception as e:
                logger.error(f"Redis session storage failed: {e}")
        
        # Fallback to memory
        self._memory_store[session.session_id] = session_data
        return True
    
    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Retrieve session data"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"session:{session_id}")
                if data:
                    session_data = json.loads(data)
                    return self._deserialize_session(session_id, session_data)
            except Exception as e:
                logger.error(f"Redis session retrieval failed: {e}")
        
        # Fallback to memory
        data = self._memory_store.get(session_id)
        if data:
            return self._deserialize_session(session_id, data)
        
        return None
    
    def _deserialize_session(self, session_id: str, data: Dict[str, Any]) -> AuthSession:
        """Deserialize session data"""
        return AuthSession(
            session_id=session_id,
            user_id=data['user_id'],
            tenant_id=data['tenant_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            provider=AuthProvider(data['provider']),
            mfa_verified=data.get('mfa_verified', False),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent')
        )
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        if self.redis_client:
            try:
                await self.redis_client.delete(f"session:{session_id}")
                return True
            except Exception as e:
                logger.error(f"Redis session revocation failed: {e}")
        
        # Fallback to memory
        self._memory_store.pop(session_id, None)
        return True


class EnterpriseAuthManager:
    """Enterprise authentication manager with SSO support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or (Path.home() / ".terradev" / "enterprise_auth.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize session store
        self.session_store = SessionStore()
        
        # Load configuration
        self.config = self._load_config()
        
        # User storage (in production, use database)
        self.users: Dict[str, EnterpriseUser] = {}
        self._load_users()
        
        # MFA settings
        self.mfa_required_roles = {UserRole.ORG_ADMIN, UserRole.SUPER_ADMIN}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enterprise authentication configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load enterprise auth config: {e}")
        
        # Default configuration
        default_config = {
            "sso_providers": {
                "azure_ad": {
                    "type": "saml",
                    "enabled": False,
                    "entity_id": "",
                    "sso_url": "",
                    "certificate": ""
                },
                "okta": {
                    "type": "saml", 
                    "enabled": False,
                    "entity_id": "",
                    "sso_url": "",
                    "certificate": ""
                },
                "google_workspace": {
                    "type": "oidc",
                    "enabled": False,
                    "client_id": "",
                    "client_secret": "",
                    "discovery_url": "https://accounts.google.com/.well-known/openid-configuration"
                },
                "auth0": {
                    "type": "oidc",
                    "enabled": False,
                    "client_id": "",
                    "client_secret": "",
                    "domain": ""
                }
            },
            "session_timeout_hours": 8,
            "mfa_required": True,
            "redis_url": None
        }
        
        # Save default config
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_users(self):
        """Load enterprise users from storage"""
        users_file = self.config_path.parent / "enterprise_users.json"
        if users_file.exists():
            try:
                with open(users_file, "r") as f:
                    users_data = json.load(f)
                    for user_data in users_data:
                        user = self._deserialize_user(user_data)
                        self.users[user.user_id] = user
            except Exception as e:
                logger.error(f"Failed to load enterprise users: {e}")
    
    def _save_users(self):
        """Save enterprise users to storage"""
        users_file = self.config_path.parent / "enterprise_users.json"
        users_data = []
        for user in self.users.values():
            users_data.append(self._serialize_user(user))
        
        with open(users_file, "w") as f:
            json.dump(users_data, f, indent=2)
    
    def _serialize_user(self, user: EnterpriseUser) -> Dict[str, Any]:
        """Serialize user for storage"""
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
    
    def _deserialize_user(self, data: Dict[str, Any]) -> EnterpriseUser:
        """Deserialize user from storage"""
        return EnterpriseUser(
            user_id=data['user_id'],
            email=data['email'],
            name=data['name'],
            role=UserRole(data['role']),
            tenant_id=data['tenant_id'],
            provider=AuthProvider(data['provider']),
            provider_user_id=data['provider_user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            mfa_enabled=data.get('mfa_enabled', False),
            mfa_secret=data.get('mfa_secret'),
            permissions=data.get('permissions', []),
            metadata=data.get('metadata', {})
        )
    
    async def create_user_session(self, user: EnterpriseUser, 
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None) -> AuthSession:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=self.config['session_timeout_hours'])
        
        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            provider=user.provider,
            mfa_verified=False,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        await self.session_store.store_session(session)
        
        # Update user last login
        user.last_login = datetime.utcnow()
        self._save_users()
        
        return session
    
    async def validate_session(self, session_id: str) -> Optional[EnterpriseUser]:
        """Validate a session and return the associated user"""
        session = await self.session_store.get_session(session_id)
        
        if not session:
            return None
        
        # Check if session is expired
        if datetime.utcnow() > session.expires_at:
            await self.session_store.revoke_session(session_id)
            return None
        
        # Get user
        user = self.users.get(session.user_id)
        if not user:
            await self.session_store.revoke_session(session_id)
            return None
        
        return user
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a user session"""
        return await self.session_store.revoke_session(session_id)
    
    def get_user_permissions(self, user: EnterpriseUser) -> List[str]:
        """Get user permissions based on role and custom permissions"""
        # Role-based permissions
        role_permissions = {
            UserRole.SUPER_ADMIN: [
                'system:admin', 'users:manage', 'tenants:manage', 
                'providers:manage', 'billing:manage', 'audit:view'
            ],
            UserRole.ORG_ADMIN: [
                'org:admin', 'teams:manage', 'users:invite', 'billing:view',
                'resources:manage', 'audit:view'
            ],
            UserRole.TEAM_ADMIN: [
                'team:admin', 'members:manage', 'resources:provision',
                'costs:view', 'audit:team'
            ],
            UserRole.DEVELOPER: [
                'resources:provision', 'resources:manage', 'costs:view',
                'datasets:access', 'models:deploy'
            ],
            UserRole.ANALYST: [
                'analytics:view', 'reports:view', 'costs:view', 'logs:view'
            ],
            UserRole.VIEWER: [
                'resources:view', 'status:view', 'basic:read'
            ]
        }
        
        permissions = role_permissions.get(user.role, [])
        
        # Add custom permissions
        permissions.extend(user.permissions)
        
        # Remove duplicates
        return list(set(permissions))
    
    def has_permission(self, user: EnterpriseUser, permission: str) -> bool:
        """Check if user has a specific permission"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def is_mfa_required(self, user: EnterpriseUser) -> bool:
        """Check if MFA is required for this user"""
        return (self.config.get('mfa_required', False) and 
                user.role in self.mfa_required_roles) or user.mfa_enabled
    
    def get_sso_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get SSO provider configuration"""
        return self.config.get('sso_providers', {}).get(provider_name)
    
    def enable_sso_provider(self, provider_name: str, config: Dict[str, Any]):
        """Enable and configure an SSO provider"""
        if provider_name not in self.config['sso_providers']:
            raise ValueError(f"Unknown SSO provider: {provider_name}")
        
        self.config['sso_providers'][provider_name].update(config)
        self.config['sso_providers'][provider_name]['enabled'] = True
        
        # Save configuration
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def list_enabled_providers(self) -> List[str]:
        """List enabled SSO providers"""
        enabled = []
        for name, config in self.config.get('sso_providers', {}).items():
            if config.get('enabled', False):
                enabled.append(name)
        return enabled
    
    async def create_or_update_user(self, user_data: Dict[str, Any]) -> EnterpriseUser:
        """Create or update a user from SSO provider data"""
        provider_user_id = user_data['provider_user_id']
        provider = AuthProvider(user_data['provider'])
        
        # Check if user exists
        existing_user = None
        for user in self.users.values():
            if (user.provider == provider and 
                user.provider_user_id == provider_user_id):
                existing_user = user
                break
        
        if existing_user:
            # Update existing user
            existing_user.email = user_data.get('email', existing_user.email)
            existing_user.name = user_data.get('name', existing_user.name)
            existing_user.metadata.update(user_data.get('metadata', {}))
            user = existing_user
        else:
            # Create new user
            user = EnterpriseUser(
                user_id=str(uuid.uuid4()),
                email=user_data['email'],
                name=user_data['name'],
                role=UserRole(user_data.get('role', 'developer')),
                tenant_id=user_data['tenant_id'],
                provider=provider,
                provider_user_id=provider_user_id,
                created_at=datetime.utcnow(),
                permissions=user_data.get('permissions', []),
                metadata=user_data.get('metadata', {})
            )
            self.users[user.user_id] = user
        
        self._save_users()
        return user
    
    def get_users_by_tenant(self, tenant_id: str) -> List[EnterpriseUser]:
        """Get all users in a tenant"""
        return [user for user in self.users.values() if user.tenant_id == tenant_id]
    
    def update_user_role(self, user_id: str, new_role: UserRole) -> bool:
        """Update user role"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.role = new_role
        self._save_users()
        return True
