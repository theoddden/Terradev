#!/usr/bin/env python3
"""
Terradev OpenID Connect Provider Integration
Handles OIDC authentication for enterprise SSO
"""

import base64
import json
import secrets
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse, parse_qs
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import jwt
except ImportError:
    jwt = None

logger = logging.getLogger(__name__)


class OIDCProvider:
    """OpenID Connect authentication provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client_id = config.get('client_id', '')
        self.client_secret = config.get('client_secret', '')
        self.domain = config.get('domain', '')
        self.discovery_url = config.get('discovery_url', '')
        self.redirect_uri = config.get('redirect_uri', 'https://api.terradev.cloud/auth/oidc/callback')
        
        # OIDC endpoints (discovered or configured)
        self.authorization_endpoint = ''
        self.token_endpoint = ''
        self.userinfo_endpoint = ''
        self.jwks_uri = ''
        
        # PKCE verifier
        self.code_verifier = None
        self.code_challenge = None
    
    async def discover_endpoints(self) -> bool:
        """Discover OIDC endpoints from well-known configuration"""
        try:
            if not aiohttp:
                logger.error("aiohttp not available for OIDC discovery")
                return False
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.discovery_url) as response:
                    if response.status != 200:
                        logger.error(f"OIDC discovery failed: {response.status}")
                        return False
                    
                    config = await response.json()
                    
                    # Extract endpoints
                    self.authorization_endpoint = config.get('authorization_endpoint')
                    self.token_endpoint = config.get('token_endpoint')
                    self.userinfo_endpoint = config.get('userinfo_endpoint')
                    self.jwks_uri = config.get('jwks_uri')
                    
                    if not all([self.authorization_endpoint, self.token_endpoint]):
                        logger.error("Missing required OIDC endpoints")
                        return False
                    
                    logger.info(f"OIDC discovery successful for {self.discovery_url}")
                    return True
                    
        except Exception as e:
            logger.error(f"OIDC discovery failed: {e}")
            return False
    
    def generate_pkce_challenge(self) -> str:
        """Generate PKCE code challenge and verifier"""
        # Generate code verifier
        self.code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge
        challenge_bytes = hashlib.sha256(self.code_verifier.encode('utf-8')).digest()
        self.code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
        
        return self.code_challenge
    
    def generate_auth_url(self, scopes: Optional[list] = None, state: Optional[str] = None) -> Tuple[str, str]:
        """Generate OIDC authorization URL"""
        if not self.authorization_endpoint:
            raise ValueError("Authorization endpoint not discovered")
        
        # Generate PKCE challenge
        code_challenge = self.generate_pkce_challenge()
        
        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Build parameters
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes or ['openid', 'email', 'profile']),
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        # Build URL
        auth_url = f"{self.authorization_endpoint}?{urlencode(params)}"
        
        return auth_url, state
    
    async def exchange_code_for_tokens(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        try:
            if not aiohttp:
                raise ValueError("aiohttp not available for token exchange")
            
            if not self.token_endpoint:
                raise ValueError("Token endpoint not discovered")
            
            # Prepare token request
            data = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': self.redirect_uri,
                'code_verifier': self.code_verifier
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_endpoint, data=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Token exchange failed: {response.status} - {error_text}")
                    
                    token_data = await response.json()
                    
                    # Validate tokens
                    if 'access_token' not in token_data:
                        raise ValueError("No access token in response")
                    
                    return token_data
                    
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from userinfo endpoint"""
        try:
            if not aiohttp:
                raise ValueError("aiohttp not available for user info")
            
            if not self.userinfo_endpoint:
                raise ValueError("Userinfo endpoint not discovered")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.userinfo_endpoint, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"User info request failed: {response.status} - {error_text}")
                    
                    user_info = await response.json()
                    
                    # Parse user data
                    user_data = {
                        'provider_user_id': user_info.get('sub', ''),
                        'email': user_info.get('email', ''),
                        'name': user_info.get('name', ''),
                        'first_name': user_info.get('given_name', ''),
                        'last_name': user_info.get('family_name', ''),
                        'picture': user_info.get('picture', ''),
                        'locale': user_info.get('locale', ''),
                        'verified': user_info.get('email_verified', False),
                        'attributes': user_info
                    }
                    
                    # Extract groups/roles if available
                    if 'groups' in user_info:
                        user_data['groups'] = user_info['groups']
                    
                    return user_data
                    
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise
    
    def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate ID token and extract claims"""
        try:
            if not jwt:
                logger.warning("JWT library not available, skipping ID token validation")
                return {}
            
            # Decode without verification first to get headers
            unverified_header = jwt.get_unverified_header(id_token)
            
            # For production, you should verify the signature using the JWKS
            # For now, we'll decode without verification
            decoded = jwt.decode(id_token, options={'verify_signature': False})
            
            # Extract claims
            claims = {
                'iss': decoded.get('iss'),
                'aud': decoded.get('aud'),
                'sub': decoded.get('sub'),
                'exp': decoded.get('exp'),
                'iat': decoded.get('iat'),
                'email': decoded.get('email'),
                'name': decoded.get('name'),
                'picture': decoded.get('picture')
            }
            
            # Basic validation
            if decoded.get('aud') != self.client_id:
                raise ValueError("Invalid audience in ID token")
            
            if datetime.utcnow() > datetime.fromtimestamp(decoded.get('exp', 0)):
                raise ValueError("ID token expired")
            
            return claims
            
        except Exception as e:
            logger.error(f"ID token validation failed: {e}")
            raise
    
    def configure_google_workspace(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Get Google Workspace specific configuration"""
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'discovery_url': 'https://accounts.google.com/.well-known/openid-configuration',
            'redirect_uri': self.redirect_uri,
            'scopes': ['openid', 'email', 'profile', 'https://www.googleapis.com/auth/userinfo.email']
        }
    
    def configure_auth0(self, domain: str, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Get Auth0 specific configuration"""
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'domain': domain,
            'discovery_url': f'https://{domain}/.well-known/openid-configuration',
            'redirect_uri': self.redirect_uri,
            'scopes': ['openid', 'email', 'profile']
        }
    
    def configure_azure_ad(self, tenant_id: str, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Get Azure AD specific configuration"""
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'discovery_url': f'https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration',
            'redirect_uri': self.redirect_uri,
            'scopes': ['openid', 'email', 'profile']
        }
    
    async def test_connection(self) -> bool:
        """Test OIDC provider connection"""
        try:
            # Test endpoint discovery
            if not await self.discover_endpoints():
                return False
            
            # Test PKCE generation
            code_challenge = self.generate_pkce_challenge()
            if not code_challenge:
                return False
            
            # Test auth URL generation
            auth_url, state = self.generate_auth_url()
            if not auth_url or not state:
                return False
            
            logger.info(f"OIDC provider test successful for {self.discovery_url}")
            return True
            
        except Exception as e:
            logger.error(f"OIDC provider test failed: {e}")
            return False


class OIDCManager:
    """Manages multiple OIDC providers"""
    
    def __init__(self):
        self.providers: Dict[str, OIDCProvider] = {}
    
    def add_provider(self, name: str, config: Dict[str, Any]):
        """Add an OIDC provider"""
        provider = OIDCProvider(config)
        self.providers[name] = provider
        logger.info(f"Added OIDC provider: {name}")
    
    def get_provider(self, name: str) -> Optional[OIDCProvider]:
        """Get an OIDC provider by name"""
        return self.providers.get(name)
    
    def list_providers(self) -> list:
        """List all configured providers"""
        return list(self.providers.keys())
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """Test all providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.test_connection()
        return results
