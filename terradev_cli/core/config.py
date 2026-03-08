#!/usr/bin/env python3
"""
Terradev Configuration Management
Handles configuration settings and provider preferences
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ProviderType(Enum):
    """Supported cloud providers"""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    RUNPOD = "runpod"
    VASTAI = "vastai"
    LAMBDA_LABS = "lambda_labs"
    COREWEAVE = "coreweave"
    TENSORDOCK = "tensordock"
    HUGGINGFACE = "huggingface"
    BASETEN = "baseten"
    ORACLE = "oracle"


@dataclass
class ProviderConfig:
    """Configuration for a cloud provider"""

    name: str
    enabled: bool
    default_region: str
    api_endpoint: Optional[str]
    reliability_score: float
    priority: int
    metadata: Dict[str, Any]


@dataclass
class TerradevConfig:
    """Main Terradev configuration"""

    default_providers: List[str]
    parallel_queries: int
    max_price_threshold: float
    preferred_regions: List[str]
    providers: Dict[str, ProviderConfig]
    optimization_settings: Dict[str, Any]
    analytics_settings: Dict[str, Any]

    @classmethod
    def load(cls, config_path: str) -> "TerradevConfig":
        """Load configuration from file"""
        config_file = Path(config_path)

        if not config_file.exists():
            # Create default configuration
            config = cls._create_default()
            config.save(config_path)
            return config

        try:
            with open(config_file, "r") as f:
                data = json.load(f)

            # Convert provider configs
            providers = {}
            for name, provider_data in data.get("providers", {}).items():
                providers[name] = ProviderConfig(**provider_data)

            return cls(
                default_providers=data.get(
                    "default_providers", ["aws", "gcp", "azure"]
                ),
                parallel_queries=data.get("parallel_queries", 6),
                max_price_threshold=data.get("max_price_threshold", 10.0),
                preferred_regions=data.get(
                    "preferred_regions", ["us-east-1", "us-west-2", "eu-west-1"]
                ),
                providers=providers,
                optimization_settings=data.get("optimization_settings", {}),
                analytics_settings=data.get("analytics_settings", {}),
            )

        except Exception as e:
            print(f"Error loading config: {e}")
            return cls._create_default()

    def save(self, config_path: str) -> None:
        """Save configuration to file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            "default_providers": self.default_providers,
            "parallel_queries": self.parallel_queries,
            "max_price_threshold": self.max_price_threshold,
            "preferred_regions": self.preferred_regions,
            "providers": {
                name: asdict(config) for name, config in self.providers.items()
            },
            "optimization_settings": self.optimization_settings,
            "analytics_settings": self.analytics_settings,
        }

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

        # Set secure permissions
        os.chmod(config_file, 0o600)

    @classmethod
    def _create_default(cls) -> "TerradevConfig":
        """Create default configuration"""
        providers = {
            "aws": ProviderConfig(
                name="aws",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://ec2.amazonaws.com",
                reliability_score=0.95,
                priority=1,
                metadata={
                    "instance_types": ["p3.2xlarge", "p3.8xlarge", "p4d.24xlarge"],
                    "gpu_types": ["V100", "A100", "H100"],
                },
            ),
            "gcp": ProviderConfig(
                name="gcp",
                enabled=True,
                default_region="us-central1",
                api_endpoint="https://compute.googleapis.com",
                reliability_score=0.93,
                priority=2,
                metadata={
                    "instance_types": ["n1-standard-4", "n1-standard-8"],
                    "gpu_types": ["V100", "A100", "T4"],
                },
            ),
            "azure": ProviderConfig(
                name="azure",
                enabled=True,
                default_region="eastus",
                api_endpoint="https://management.azure.com",
                reliability_score=0.92,
                priority=3,
                metadata={
                    "instance_types": ["Standard_NC6", "Standard_NC12"],
                    "gpu_types": ["V100", "A100"],
                },
            ),
            "runpod": ProviderConfig(
                name="runpod",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://api.runpod.io",
                reliability_score=0.85,
                priority=4,
                metadata={
                    "instance_types": ["gpu+1xrtx3090", "gpu+2xrtx3090"],
                    "gpu_types": ["RTX3090", "RTX4090", "A100"],
                },
            ),
            "vastai": ProviderConfig(
                name="vastai",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://console.vast.ai/api/v0",
                reliability_score=0.80,
                priority=5,
                metadata={
                    "instance_types": ["A100-SXM4-40GB", "RTX4090"],
                    "gpu_types": ["A100", "RTX4090", "RTX3090"],
                },
            ),
            "lambda_labs": ProviderConfig(
                name="lambda_labs",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://api.lambdalabs.com",
                reliability_score=0.82,
                priority=6,
                metadata={
                    "instance_types": ["gpu_1x_a10", "gpu_8x_a100_40gb"],
                    "gpu_types": ["A10", "A100", "RTX6000"],
                },
            ),
            "coreweave": ProviderConfig(
                name="coreweave",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://api.coreweave.com",
                reliability_score=0.88,
                priority=7,
                metadata={
                    "instance_types": ["a100-40gb", "rtx4090"],
                    "gpu_types": ["A100", "RTX4090", "RTX3090"],
                },
            ),
            "tensordock": ProviderConfig(
                name="tensordock",
                enabled=True,
                default_region="us-east-1",
                api_endpoint="https://api.tensordock.com",
                reliability_score=0.78,
                priority=8,
                metadata={
                    "instance_types": ["rtx4090", "a100"],
                    "gpu_types": ["RTX4090", "A100", "RTX3090"],
                },
            ),
        }

        return cls(
            default_providers=["aws", "gcp", "azure", "runpod"],
            parallel_queries=6,
            max_price_threshold=10.0,
            preferred_regions=["us-east-1", "us-west-2", "eu-west-1"],
            providers=providers,
            optimization_settings={
                "price_weight": 0.4,
                "latency_weight": 0.2,
                "reliability_weight": 0.3,
                "availability_weight": 0.1,
            },
            analytics_settings={
                "retention_days": 30,
                "enable_cost_tracking": True,
                "enable_usage_tracking": True,
            },
        )

    def add_provider(self, provider_name: str, default_region: str) -> None:
        """Add a new provider configuration"""
        if provider_name not in self.providers:
            self.providers[provider_name] = ProviderConfig(
                name=provider_name,
                enabled=True,
                default_region=default_region,
                api_endpoint=None,
                reliability_score=0.8,
                priority=len(self.providers) + 1,
                metadata={},
            )

    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers"""
        return [name for name, config in self.providers.items() if config.enabled]

    def get_provider_reliability(self, provider_name: str) -> float:
        """Get reliability score for provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].reliability_score
        return 0.5  # Default reliability

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "default_providers": self.default_providers,
            "parallel_queries": self.parallel_queries,
            "max_price_threshold": self.max_price_threshold,
            "preferred_regions": self.preferred_regions,
            "providers": {
                name: asdict(config) for name, config in self.providers.items()
            },
            "optimization_settings": self.optimization_settings,
            "analytics_settings": self.analytics_settings,
        }
