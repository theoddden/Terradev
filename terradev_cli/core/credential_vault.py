#!/usr/bin/env python3
"""
Credential Vault - Secure credential storage with zeroization guarantees

Rust implementation provides:
- RAII guarantees
- Automatic memory zeroization on drop
- Type-safe secret access
- Prevents credential leaks
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Rust credential vault integration
try:
    from terradev_credential_vault import PyCredentialVault
    USE_RUST_VAULT = True
    logger.info("Using Rust credential vault for memory-safe secret handling")
except ImportError:
    USE_RUST_VAULT = False
    logger.info("Rust credential vault not available, using Python fallback")


class CredentialVault:
    """Credential vault with Rust backend or Python fallback"""
    
    def __init__(self):
        if USE_RUST_VAULT:
            self._rust_vault = PyCredentialVault()
        else:
            self._credentials: Dict[str, bytes] = {}
    
    def store(self, name: str, value: bytes, provider: str = ""):
        """Store a credential"""
        if USE_RUST_VAULT:
            self._rust_vault.store(name, value, provider)
        else:
            self._credentials[name] = value
    
    def retrieve(self, name: str) -> Optional[bytes]:
        """Retrieve a credential"""
        if USE_RUST_VAULT:
            return self._rust_vault.retrieve(name)
        else:
            return self._credentials.get(name)
    
    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get credential metadata"""
        if USE_RUST_VAULT:
            metadata = self._rust_vault.get_metadata(name)
            if metadata:
                return {
                    "created_at": metadata.created_at,
                    "provider": metadata.provider,
                }
            return None
        else:
            return None
    
    def list(self) -> list:
        """List all credential names"""
        if USE_RUST_VAULT:
            return self._rust_vault.list()
        else:
            return list(self._credentials.keys())
    
    def delete(self, name: str):
        """Delete a credential"""
        if USE_RUST_VAULT:
            self._rust_vault.delete(name)
        else:
            self._credentials.pop(name, None)
