#!/usr/bin/env python3
"""
Test script for Phase 1 SSO Integration
Verifies that enterprise authentication components work correctly
"""

import sys
import os
from pathlib import Path

# Add terradev_cli to path
sys.path.insert(0, str(Path(__file__).parent / "terradev_cli"))

def test_imports():
    """Test that all enterprise auth components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from core.enterprise_auth import EnterpriseAuthManager, UserRole, AuthProvider
        print("✅ enterprise_auth imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enterprise_auth: {e}")
        return False
    
    try:
        from core.saml_provider import SAMLProvider, SAMLManager
        print("✅ saml_provider imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import saml_provider: {e}")
        return False
    
    try:
        from core.oidc_provider import OIDCProvider, OIDCManager
        print("✅ oidc_provider imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import oidc_provider: {e}")
        return False
    
    try:
        from core.user_manager import UserManager, Tenant, UserInvite
        print("✅ user_manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import user_manager: {e}")
        return False
    
    return True

def test_enterprise_auth():
    """Test enterprise auth manager basic functionality"""
    print("\n🧪 Testing EnterpriseAuthManager...")
    
    try:
        from core.enterprise_auth import EnterpriseAuthManager, UserRole, AuthProvider
        
        # Create manager
        manager = EnterpriseAuthManager()
        print("✅ EnterpriseAuthManager created successfully")
        
        # Test role enum
        roles = list(UserRole)
        expected_roles = ['super_admin', 'org_admin', 'team_admin', 'developer', 'analyst', 'viewer']
        actual_roles = [role.value for role in roles]
        
        if set(actual_roles) == set(expected_roles):
            print("✅ UserRole enum correct")
        else:
            print(f"❌ UserRole enum mismatch: {actual_roles}")
            return False
        
        # Test provider enum
        providers = list(AuthProvider)
        expected_providers = ['saml', 'oidc', 'local']
        actual_providers = [provider.value for provider in providers]
        
        if set(actual_providers) == set(expected_providers):
            print("✅ AuthProvider enum correct")
        else:
            print(f"❌ AuthProvider enum mismatch: {actual_providers}")
            return False
        
        # Test configuration
        config = manager.config
        if 'sso_providers' in config and 'session_timeout_hours' in config:
            print("✅ Default configuration loaded")
        else:
            print("❌ Default configuration missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ EnterpriseAuthManager test failed: {e}")
        return False

def test_saml_provider():
    """Test SAML provider basic functionality"""
    print("\n🧪 Testing SAMLProvider...")
    
    try:
        from core.saml_provider import SAMLProvider
        
        # Create provider with test config
        config = {
            'entity_id': 'test-entity',
            'sso_url': 'https://test-sso.com/saml',
            'certificate': 'test-cert',
            'acs_url': 'https://api.terradev.cloud/auth/saml/acs'
        }
        
        provider = SAMLProvider(config)
        print("✅ SAMLProvider created successfully")
        
        # Test auth request generation
        auth_request, relay_state = provider.generate_authn_request()
        if auth_request and relay_state:
            print("✅ SAML auth request generated")
        else:
            print("❌ SAML auth request generation failed")
            return False
        
        # Test metadata generation
        metadata = provider.get_metadata()
        if metadata and 'EntityDescriptor' in metadata:
            print("✅ SAML metadata generated")
        else:
            print("❌ SAML metadata generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ SAMLProvider test failed: {e}")
        return False

def test_oidc_provider():
    """Test OIDC provider basic functionality"""
    print("\n🧪 Testing OIDCProvider...")
    
    try:
        from core.oidc_provider import OIDCProvider
        
        # Create provider with test config
        config = {
            'client_id': 'test-client-id',
            'client_secret': 'test-client-secret',
            'discovery_url': 'https://accounts.google.com/.well-known/openid-configuration',
            'redirect_uri': 'https://api.terradev.cloud/auth/oidc/callback'
        }
        
        provider = OIDCProvider(config)
        print("✅ OIDCProvider created successfully")
        
        # Test PKCE challenge generation
        code_challenge = provider.generate_pkce_challenge()
        if code_challenge and provider.code_verifier:
            print("✅ PKCE challenge generated")
        else:
            print("❌ PKCE challenge generation failed")
            return False
        
        # Test auth URL generation (skip discovery test)
        # Manually set endpoints to avoid network call
        provider.authorization_endpoint = 'https://accounts.google.com/o/oauth2/v2/auth'
        provider.token_endpoint = 'https://oauth2.googleapis.com/token'
        
        auth_url, state = provider.generate_auth_url()
        if auth_url and state and 'code_challenge' in auth_url:
            print("✅ OIDC auth URL generated")
        else:
            print("❌ OIDC auth URL generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OIDCProvider test failed: {e}")
        return False

def test_user_manager():
    """Test user manager basic functionality"""
    print("\n🧪 Testing UserManager...")
    
    try:
        from core.user_manager import UserManager, UserRole, AuthProvider
        
        # Create manager
        manager = UserManager()
        print("✅ UserManager created successfully")
        
        # Test tenant creation
        tenant = manager.create_tenant("Test Company", "testcompany.com")
        if tenant and tenant.tenant_id and tenant.name == "Test Company":
            print("✅ Tenant created successfully")
        else:
            print("❌ Tenant creation failed")
            return False
        
        # Test user creation
        user_data = {
            'email': 'test@testcompany.com',
            'name': 'Test User',
            'role': 'developer',
            'tenant_id': tenant.tenant_id,
            'provider': 'local'
        }
        
        user = manager.create_user(user_data)
        if user and user.user_id and user.email == 'test@testcompany.com':
            print("✅ User created successfully")
        else:
            print("❌ User creation failed")
            return False
        
        # Test user retrieval
        retrieved_user = manager.get_user(user.user_id)
        if retrieved_user and retrieved_user.email == user.email:
            print("✅ User retrieval successful")
        else:
            print("❌ User retrieval failed")
            return False
        
        # Test tenant users
        tenant_users = manager.get_users_by_tenant(tenant.tenant_id)
        if len(tenant_users) == 1 and tenant_users[0].email == user.email:
            print("✅ Tenant users retrieval successful")
        else:
            print("❌ Tenant users retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ UserManager test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("\n🧪 Testing CLI integration...")
    
    try:
        from cli import TerradevAPI
        
        # Create API instance
        api = TerradevAPI()
        print("✅ TerradevAPI created successfully")
        
        # Test enterprise tier detection
        is_enterprise = api._is_enterprise_tier()
        print(f"✅ Enterprise tier detection: {is_enterprise}")
        
        # Test enterprise auth lazy loading
        if api._is_enterprise_tier():
            if api.enterprise_auth is not None:
                print("✅ Enterprise auth loaded for enterprise tier")
            else:
                print("⚠️  Enterprise auth not loaded (dependencies missing)")
        else:
            print("✅ Enterprise auth not loaded for non-enterprise tier")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def main():
    """Run all Phase 1 tests"""
    print("🚀 Starting Phase 1 SSO Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_enterprise_auth,
        test_saml_provider,
        test_oidc_provider,
        test_user_manager,
        test_cli_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Phase 1 SSO tests passed!")
        print("\n📋 Next Steps:")
        print("1. Install enterprise dependencies: pip install terradev-cli[enterprise]")
        print("2. Test SSO commands: terradev sso status")
        print("3. Configure a provider: terradev sso configure --provider google_workspace")
        return True
    else:
        print("❌ Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
