"""SAML authentication provider for enterprise SSO.

This is a lightweight, dependency-free implementation used for the Phase 1 SSO
integration.  It does not require ``pysaml2`` / ``onelogin`` libraries; all
SAML artifacts are generated as simple XML strings for bootstrapping tests.
"""

import base64
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SAMLAuthRequest:
    """SAML authentication request context."""

    saml_request: str
    relay_state: str


class SAMLProvider:
    """Lightweight SAML 2.0 service provider."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_id = config.get("entity_id", "terradev-sp")
        self.sso_url = config.get("sso_url", "")
        self.acs_url = config.get("acs_url", "")
        self.certificate = config.get("certificate", "")

    def generate_authn_request(self) -> tuple:
        """Generate a SAML AuthnRequest and a relay state token."""
        request_id = f"_{uuid.uuid4().hex}"
        relay_state = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode()

        authn_request = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
            ' ID="{request_id}" Version="2.0" IssueInstant="2025-01-01T00:00:00Z"'
            ' Destination="{sso_url}" ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"'
            ' AssertionConsumerServiceURL="{acs_url}">'
            '<saml:Issuer xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">{entity_id}</saml:Issuer>'
            '<samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"'
            ' AllowCreate="true"/>'
            '</samlp:AuthnRequest>'
        ).format(
            request_id=request_id,
            sso_url=self._escape(self.sso_url),
            acs_url=self._escape(self.acs_url),
            entity_id=self._escape(self.entity_id),
        )

        # deflate + base64 encode to mimic SAMLRequest query parameter
        compressed = zlib.compress(authn_request.encode("utf-8"))[2:-4]
        saml_request = base64.b64encode(compressed).decode()

        return saml_request, relay_state

    def get_metadata(self) -> str:
        """Generate minimal SAML SP metadata XML."""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata"'
            ' entityID="{entity_id}">'
            '<SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true"'
            ' protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
            '<AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"'
            ' Location="{acs_url}" index="0" isDefault="true"/>'
            '</SPSSODescriptor>'
            '</EntityDescriptor>'
        ).format(
            entity_id=self._escape(self.entity_id),
            acs_url=self._escape(self.acs_url),
        )

    def _escape(self, value: str) -> str:
        """Minimal XML escaping."""
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


class SAMLManager:
    """High-level manager for SAML provider lifecycle."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers: Dict[str, SAMLProvider] = {}

    def add_provider(self, name: str, provider: SAMLProvider) -> None:
        self.providers[name] = provider

    def get_provider(self, name: str) -> Optional[SAMLProvider]:
        return self.providers.get(name)
