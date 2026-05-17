#!/usr/bin/env python3
"""
Quota Manager - Lock-free resource quota enforcement

Rust implementation provides:
- Deterministic quota tracking
- No GC pauses
- Leak-proof resource limits
- Prevents cost overruns
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Rust quota manager integration
try:
    from terradev_quota_manager import PyQuotaManager
    USE_RUST_QUOTA = True
    logger.info("Using Rust quota manager for lock-free enforcement")
except ImportError:
    USE_RUST_QUOTA = False
    logger.info("Rust quota manager not available, using Python fallback")


class QuotaManager:
    """Quota manager with Rust backend or Python fallback"""
    
    def __init__(self):
        if USE_RUST_QUOTA:
            self._rust_manager = PyQuotaManager()
        else:
            self._quotas: Dict[str, Dict] = {}
    
    def set_quota(self, resource: str, limit: int):
        """Set a quota for a resource"""
        if USE_RUST_QUOTA:
            self._rust_manager.set_quota(resource, limit)
        else:
            self._quotas[resource] = {"limit": limit, "used": 0}
    
    def check_quota(self, resource: str, amount: int) -> bool:
        """Check if quota is available"""
        if USE_RUST_QUOTA:
            return self._rust_manager.check_quota(resource, amount)
        else:
            quota = self._quotas.get(resource)
            if not quota:
                return True
            return quota["used"] + amount <= quota["limit"]
    
    def consume_quota(self, resource: str, amount: int):
        """Consume quota"""
        if USE_RUST_QUOTA:
            self._rust_manager.consume_quota(resource, amount)
        else:
            if resource in self._quotas:
                self._quotas[resource]["used"] += amount
    
    def release_quota(self, resource: str, amount: int):
        """Release quota"""
        if USE_RUST_QUOTA:
            self._rust_manager.release_quota(resource, amount)
        else:
            if resource in self._quotas:
                self._quotas[resource]["used"] = max(0, self._quotas[resource]["used"] - amount)
    
    def get_quota(self, resource: str) -> Optional[Dict]:
        """Get quota status"""
        if USE_RUST_QUOTA:
            quota = self._rust_manager.get_quota(resource)
            if quota:
                return {
                    "limit": quota.limit,
                    "used": quota.used,
                    "remaining": quota.remaining,
                }
            return None
        else:
            quota = self._quotas.get(resource)
            if quota:
                return {
                    "limit": quota["limit"],
                    "used": quota["used"],
                    "remaining": quota["limit"] - quota["used"],
                }
            return None
    
    def list_quotas(self) -> list:
        """List all quotas"""
        if USE_RUST_QUOTA:
            return self._rust_manager.list_quotas()
        else:
            return list(self._quotas.keys())
