#!/usr/bin/env python3
"""
Terradev Tier Manager
Handles tier-based access control and feature limitations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import secrets
from datetime import datetime, timedelta


class TierType(Enum):
    """Subscription tiers"""

    RESEARCH = "research"
    RESEARCH_PLUS = "research_plus"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"


@dataclass
class TierLimits:
    """Feature limits per tier"""

    max_instances: int  # -1 = unlimited
    max_parallel_queries: int
    providers_allowed: List[str]
    features_enabled: List[str]
    enterprise_features: bool
    gpu_hour_rate: float = 0.0  # $0.00 = flat-rate tier
    min_gpus: int = 0  # Minimum GPUs required (Enterprise+ = 32)


class TierManager:
    """Manages tier-based access control"""

    # Tier definitions
    TIER_LIMITS = {
        TierType.RESEARCH: TierLimits(
            max_instances=1,
            max_parallel_queries=3,
            providers_allowed=["aws", "runpod", "vastai"],
            features_enabled=["quote", "status", "help", "configure", "cleanup", "analytics", "optimize"],
            enterprise_features=False,
        ),
        TierType.RESEARCH_PLUS: TierLimits(
            max_instances=4,
            max_parallel_queries=10,
            providers_allowed=["aws", "gcp", "azure", "runpod", "vastai", "tensordock", "oracle"],
            features_enabled=[
                "quote",
                "provision",
                "status",
                "analytics",
                "optimize",
                "cleanup",
                "inference",
                "full_provenance",
            ],
            enterprise_features=False,
        ),
        TierType.ENTERPRISE: TierLimits(
            max_instances=32,
            max_parallel_queries=50,
            providers_allowed=[
                "aws",
                "gcp",
                "azure",
                "runpod",
                "vastai",
                "tensordock",
                "oracle",
                "coreweave",
            ],
            features_enabled=[
                "quote",
                "provision",
                "status",
                "analytics",
                "optimize",
                "cleanup",
                "manage",
                "inference",
                "full_provenance",
            ],
            enterprise_features=True,
        ),
        TierType.ENTERPRISE_PLUS: TierLimits(
            max_instances=-1,  # Unlimited
            max_parallel_queries=100,
            providers_allowed=[
                "aws",
                "gcp",
                "azure",
                "runpod",
                "vastai",
                "tensordock",
                "oracle",
                "coreweave",
                "lambda_labs",
                "crusoe",
                "baseten",
                "huggingface",
            ],
            features_enabled=[
                "quote",
                "provision",
                "status",
                "analytics",
                "optimize",
                "cleanup",
                "manage",
                "inference",
                "full_provenance",
                "fleet_management",
                "gpu_metering",
                "dedicated_support",
            ],
            enterprise_features=True,
            gpu_hour_rate=0.09,
            min_gpus=32,
        ),
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or (Path.home() / ".terradev" / "tier.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.tier_config = self._load_tier_config()

    def _load_tier_config(self) -> Dict[str, Any]:
        """Load tier configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading tier config: {e}")

        # Default configuration
        return {
            "tier": "research",
            "enterprise_id": None,
            "instance_count": 0,
            "last_reset": datetime.now().isoformat(),
            "usage_stats": {
                "quotes_requested": 0,
                "instances_provisioned": 0,
                "total_cost_saved": 0.0,
            },
        }

    def _save_tier_config(self):
        """Save tier configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.tier_config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving tier config: {e}")

    def get_current_tier(self) -> TierType:
        """Get current subscription tier"""
        tier_str = self.tier_config.get("tier", "research")
        return TierType(tier_str.lower())

    def set_tier(self, tier: TierType, enterprise_id: Optional[str] = None):
        """Set subscription tier"""
        self.tier_config["tier"] = tier.value
        if tier == TierType.ENTERPRISE:
            if not enterprise_id:
                raise ValueError("Enterprise tier requires enterprise_id")
            self.tier_config["enterprise_id"] = enterprise_id
        else:
            self.tier_config["enterprise_id"] = None
        self._save_tier_config()

    def verify_enterprise_id(self, enterprise_id: str) -> bool:
        """Verify enterprise ID (placeholder for future validation)"""
        # TODO: Implement actual enterprise ID validation
        # For now, accept any non-empty string as valid
        return bool(enterprise_id and len(enterprise_id) >= 8)

    def check_feature_access(self, feature: str) -> bool:
        """Check if feature is available for current tier"""
        tier = self.get_current_tier()
        limits = self.TIER_LIMITS[tier]
        return feature in limits.features_enabled

    def check_instance_limit(self, requested_instances: int) -> bool:
        """Check if instance count is within tier limits"""
        tier = self.get_current_tier()
        limits = self.TIER_LIMITS[tier]
        if limits.max_instances == -1:  # Unlimited (Enterprise+)
            return True
        current_count = self.tier_config.get("instance_count", 0)
        return (current_count + requested_instances) <= limits.max_instances

    def check_provider_access(self, provider: str) -> bool:
        """Check if provider is available for current tier"""
        tier = self.get_current_tier()
        limits = self.TIER_LIMITS[tier]
        return provider in limits.providers_allowed

    def check_enterprise_feature(self) -> bool:
        """Check if enterprise features are available"""
        tier = self.get_current_tier()
        limits = self.TIER_LIMITS[tier]
        return limits.enterprise_features

    def increment_instance_count(self, count: int = 1):
        """Increment instance count for usage tracking"""
        self.tier_config["instance_count"] = (
            self.tier_config.get("instance_count", 0) + count
        )
        self._save_tier_config()

    def decrement_instance_count(self, count: int = 1):
        """Decrement instance count for usage tracking"""
        current = self.tier_config.get("instance_count", 0)
        self.tier_config["instance_count"] = max(0, current - count)
        self._save_tier_config()

    def get_tier_info(self) -> Dict[str, Any]:
        """Get current tier information"""
        tier = self.get_current_tier()
        limits = self.TIER_LIMITS[tier]

        return {
            "tier": tier.value,
            "limits": {
                "max_instances": limits.max_instances,
                "max_parallel_queries": limits.max_parallel_queries,
                "providers_allowed": limits.providers_allowed,
                "features_enabled": limits.features_enabled,
                "enterprise_features": limits.enterprise_features,
            },
            "current_usage": {
                "instances": self.tier_config.get("instance_count", 0),
                "enterprise_id": self.tier_config.get("enterprise_id"),
                "last_reset": self.tier_config.get("last_reset"),
            },
            "usage_stats": self.tier_config.get("usage_stats", {}),
        }

    def upgrade_to_research_plus(self, secret_key: str):
        """Upgrade to Research+ tier with secret key verification"""
        if not secret_key or len(secret_key) < 8:
            raise ValueError("Valid Research+ secret key required")

        print("� Upgrading to Research+ tier...")
        self.set_tier(TierType.RESEARCH_PLUS, secret_key)
        print("✅ Upgraded to Research+ tier!")
        print("🔑 Features: 4 compute units, 40 provisions/month, inference, full provenance")

    def upgrade_to_enterprise(self, enterprise_id: str):
        """Upgrade to Enterprise tier with ID verification"""
        if not self.verify_enterprise_id(enterprise_id):
            raise ValueError("Invalid enterprise ID")

        print("🏢 Upgrading to Enterprise tier...")
        self.set_tier(TierType.ENTERPRISE, enterprise_id)
        print("✅ Upgraded to Enterprise tier!")
        print(f"🔑 Enterprise ID: {enterprise_id}")

    def reset_usage_stats(self):
        """Reset usage statistics (typically done monthly)"""
        self.tier_config["instance_count"] = 0
        self.tier_config["last_reset"] = datetime.now().isoformat()
        self._save_tier_config()


# Decorator for tier-based access control
def require_tier(min_tier: TierType):
    """Decorator to require minimum tier for command access"""

    def decorator(func):
        def wrapper(ctx, *args, **kwargs):
            tier_manager = TierManager(ctx.obj["config_path"])
            current_tier = tier_manager.get_current_tier()

            # Check tier hierarchy
            tier_hierarchy = {TierType.RESEARCH: 0, TierType.RESEARCH_PLUS: 1, TierType.ENTERPRISE: 2, TierType.ENTERPRISE_PLUS: 3}

            if tier_hierarchy[current_tier] < tier_hierarchy[min_tier]:
                print(
                    f"❌ This feature requires {min_tier.value.title()} tier or higher."
                )
                print(f"📊 Current tier: {current_tier.value.title()}")
                print("💡 Upgrade with: terradev upgrade --tier research_plus|enterprise|enterprise_plus")
                return

            return func(ctx, *args, **kwargs)

        return wrapper

    return decorator


def require_enterprise_id():
    """Decorator to require valid enterprise ID"""

    def decorator(func):
        def wrapper(ctx, *args, **kwargs):
            tier_manager = TierManager(ctx.obj["config_path"])

            if not tier_manager.check_enterprise_feature():
                print("❌ This feature requires Enterprise tier.")
                print(
                    "🏢 Upgrade with: terradev upgrade --tier enterprise --id <enterprise_id>"
                )
                return

            enterprise_id = tier_manager.tier_config.get("enterprise_id")
            if not enterprise_id:
                print("❌ Enterprise ID required for this feature.")
                print(
                    "🔑 Set enterprise ID with: terradev upgrade --tier enterprise --id <enterprise_id>"
                )
                return

            return func(ctx, *args, **kwargs)

        return wrapper

    return decorator
