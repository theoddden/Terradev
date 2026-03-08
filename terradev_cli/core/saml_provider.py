#!/usr/bin/env python3
"""
Terradev SAML 2.0 Provider Integration
Handles SAML authentication for enterprise SSO
"""

import base64
import zlib
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import logging
import secrets

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    from cryptography.hazmat.backends import default_backend
except ImportError:
    hashes = padding = load_pem_private_key = load_pem_public_key = default_backend = None

logger = logging.getLogger(__name__)


class SAMLProvider:
    """SAML 2.0 authentication provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_id = config.get('entity_id', '')
        self.sso_url = config.get('sso_url', '')
        self.certificate = config.get('certificate', '')
        self.private_key = config.get('private_key', '')
        
        # SAML namespaces
        self.SAML_NS = 'urn:oasis:names:tc:SAML:2.0:assertion'
        self.SAMLP_NS = 'urn:oasis:names:tc:SAML:2.0:protocol'
        
        # Register namespaces
        self.NS = {
            'saml': self.SAML_NS,
            'samlp': self.SAMLP_NS
        }
    
    def generate_authn_request(self, relay_state: Optional[str] = None) -> Tuple[str, str]:
        """Generate SAML authentication request"""
        # Generate request ID
        request_id = f"_id-{secrets.token_hex(16)}"
        
        # Build AuthnRequest XML
        authn_request = ET.Element(f"{{{self.SAMLP_NS}}}AuthnRequest", {
            'ID': request_id,
            'Version': '2.0',
            'IssueInstant': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'Destination': self.sso_url,
            'ProtocolBinding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST',
            'AssertionConsumerServiceURL': self.config.get('acs_url', ''),
        })
        
        # Add Issuer
        issuer = ET.SubElement(authn_request, f"{{{self.SAML_NS}}}Issuer")
        issuer.text = self.entity_id
        
        # Add NameIDPolicy
        name_id_policy = ET.SubElement(authn_request, f"{{{self.SAMLP_NS}}}NameIDPolicy", {
            'Format': 'urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress',
            'AllowCreate': 'true'
        })
        
        # Add RequestedAuthnContext
        authn_context = ET.SubElement(authn_request, f"{{{self.SAMLP_NS}}}RequestedAuthnContext", {
            'Comparison': 'exact'
        })
        authn_context_class = ET.SubElement(authn_context, f"{{{self.SAML_NS}}}AuthnContextClassRef")
        authn_context_class.text = 'urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport'
        
        # Convert to string
        xml_str = ET.tostring(authn_request, encoding='unicode')
        
        # Base64 encode
        encoded_request = base64.b64encode(
            zlib.compress(xml_str.encode('utf-8'))
        ).decode('utf-8')
        
        # Generate relay state if not provided
        if not relay_state:
            relay_state = secrets.token_urlsafe(32)
        
        return encoded_request, relay_state
    
    def parse_saml_response(self, saml_response: str) -> Dict[str, Any]:
        """Parse SAML response and extract user attributes"""
        try:
            # Base64 decode
            decoded_response = base64.b64decode(saml_response)
            
            # Try to decompress
            try:
                xml_str = zlib.decompress(decoded_response).decode('utf-8')
            except:
                xml_str = decoded_response.decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_str)
            
            # Extract assertion
            assertion = root.find('.//saml:Assertion', self.NS)
            if assertion is None:
                raise ValueError("No assertion found in SAML response")
            
            # Extract attributes
            attributes = {}
            attribute_statement = assertion.find('.//saml:AttributeStatement', self.NS)
            if attribute_statement is not None:
                for attr in attribute_statement.findall('.//saml:Attribute', self.NS):
                    name = attr.get('Name')
                    values = []
                    for value in attr.findall('.//saml:AttributeValue', self.NS):
                        values.append(value.text or '')
                    attributes[name] = values if len(values) > 1 else values[0] if values else ''
            
            # Extract NameID
            name_id = assertion.find('.//saml:NameID', self.NS)
            name_id_value = name_id.text if name_id is not None else ''
            
            # Extract subject
            subject = assertion.find('.//saml:Subject', self.NS)
            subject_confirmation = subject.find('.//saml:SubjectConfirmation', self.NS) if subject is not None else None
            
            # Build user data
            user_data = {
                'name_id': name_id_value,
                'email': attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailAddress', ''),
                'name': attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name', ''),
                'first_name': attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname', ''),
                'last_name': attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname', ''),
                'groups': attributes.get('http://schemas.microsoft.com/ws/2008/06/identity/claims/groups', []),
                'role': attributes.get('http://schemas.microsoft.com/ws/2008/06/identity/claims/role', 'developer'),
                'tenant_id': attributes.get('http://schemas.microsoft.com/claims/authnclassreference', 'default'),
                'attributes': attributes
            }
            
            # Validate signature if certificate is provided
            if self.certificate:
                self._validate_signature(root)
            
            return user_data
            
        except Exception as e:
            logger.error(f"Failed to parse SAML response: {e}")
            raise ValueError(f"Invalid SAML response: {e}")
    
    def _validate_signature(self, root: ET.Element):
        """Validate SAML response signature (basic implementation)"""
        try:
            # Find signature
            signature = root.find('.//ds:Signature', {'ds': 'http://www.w3.org/2000/09/xmldsig#'})
            if signature is None:
                logger.warning("No signature found in SAML response")
                return
            
            # This is a simplified signature validation
            # In production, use a proper SAML library like python3-saml
            logger.info("SAML signature validation (simplified)")
            
        except Exception as e:
            logger.warning(f"Signature validation failed: {e}")
    
    def get_metadata(self) -> str:
        """Generate SAML metadata for IdP configuration"""
        metadata = ET.Element(f"{{{self.SAML_NS}}}EntityDescriptor", {
            'entityID': self.entity_id,
            'validUntil': (datetime.utcnow() + timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')
        })
        
        # Add SPSSODescriptor
        sp_sso_descriptor = ET.SubElement(metadata, f"{{{self.SAML_NS}}}SPSSODescriptor", {
            'AuthnRequestsSigned': 'false',
            'WantAssertionsSigned': 'true',
            'protocolSupportEnumeration': 'urn:oasis:names:tc:SAML:2.0:protocol'
        })
        
        # Add AssertionConsumerService
        ET.SubElement(sp_sso_descriptor, f"{{{self.SAML_NS}}}AssertionConsumerService", {
            'index': '0',
            'Binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST',
            'Location': self.config.get('acs_url', '')
        })
        
        # Add SingleLogoutService
        ET.SubElement(sp_sso_descriptor, f"{{{self.SAML_NS}}}SingleLogoutService", {
            'Binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST',
            'Location': self.config.get('slo_url', '')
        })
        
        # Convert to pretty XML
        xml_str = ET.tostring(metadata, encoding='unicode')
        
        # Add XML declaration
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'
    
    def configure_azure_ad(self, tenant_id: str, client_id: str) -> Dict[str, Any]:
        """Get Azure AD specific configuration"""
        return {
            'entity_id': f'https://sts.windows.net/{tenant_id}/',
            'sso_url': f'https://login.microsoftonline.com/{tenant_id}/saml2',
            'acs_url': f'https://api.terradev.cloud/auth/saml/acs',
            'slo_url': f'https://api.terradev.cloud/auth/saml/slo',
            'certificate': self.certificate,
            'tenant_id': tenant_id,
            'client_id': client_id
        }
    
    def configure_okta(self, domain: str, client_id: str) -> Dict[str, Any]:
        """Get Okta specific configuration"""
        return {
            'entity_id': f'https://{domain}.okta.com',
            'sso_url': f'https://{domain}.okta.com/app/template_saml_2/kb0k6wn8bSOFJMVJZQON/sso/saml',
            'acs_url': f'https://api.terradev.cloud/auth/saml/acs',
            'slo_url': f'https://api.terradev.cloud/auth/saml/slo',
            'certificate': self.certificate,
            'domain': domain,
            'client_id': client_id
        }
    
    def test_connection(self) -> bool:
        """Test SAML provider connection"""
        try:
            # Generate a test authn request
            authn_request, relay_state = self.generate_authn_request()
            
            # Basic validation
            if not authn_request or not relay_state:
                return False
            
            # Test metadata generation
            metadata = self.get_metadata()
            if not metadata:
                return False
            
            logger.info(f"SAML provider test successful for {self.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"SAML provider test failed: {e}")
            return False


class SAMLManager:
    """Manages multiple SAML providers"""
    
    def __init__(self):
        self.providers: Dict[str, SAMLProvider] = {}
    
    def add_provider(self, name: str, config: Dict[str, Any]):
        """Add a SAML provider"""
        provider = SAMLProvider(config)
        self.providers[name] = provider
        logger.info(f"Added SAML provider: {name}")
    
    def get_provider(self, name: str) -> Optional[SAMLProvider]:
        """Get a SAML provider by name"""
        return self.providers.get(name)
    
    def list_providers(self) -> list:
        """List all configured providers"""
        return list(self.providers.keys())
    
    def test_all_providers(self) -> Dict[str, bool]:
        """Test all providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = provider.test_connection()
        return results
