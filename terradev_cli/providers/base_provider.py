#!/usr/bin/env python3
"""
Base Provider - Abstract base class for cloud providers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import aiohttp
import json


class BaseProvider(ABC):
    """Abstract base class for cloud providers"""

    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @abstractmethod
    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes for GPU type"""
        pass

    @abstractmethod
    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        """Provision an instance"""
        pass

    @abstractmethod
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status"""
        pass

    @abstractmethod
    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        """Stop an instance"""
        pass

    @abstractmethod
    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        """Start an instance"""
        pass

    @abstractmethod
    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """Terminate an instance"""
        pass

    @abstractmethod
    async def list_instances(self) -> List[Dict[str, Any]]:
        """List all instances"""
        pass

    @abstractmethod
    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        """Execute command on instance"""
        pass

    # Shared rate limiter instance across all providers
    _rate_limiter = None

    @classmethod
    def _get_rate_limiter(cls):
        """Lazy-init a shared RateLimiter (returns None if deps missing)"""
        if cls._rate_limiter is None:
            try:
                from terradev_cli.core.rate_limiter import RateLimiter
                cls._rate_limiter = RateLimiter()
            except Exception:
                cls._rate_limiter = False  # sentinel: don't retry
        return cls._rate_limiter if cls._rate_limiter is not False else None

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with authentication and rate limiting"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Acquire rate-limit permit for this provider (best-effort)
        rl = self._get_rate_limiter()
        if rl:
            try:
                await rl.acquire(self.name)
            except Exception:
                pass  # proceed even if rate limiter fails

        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())

        async with self.session.request(
            method, url, headers=headers, **kwargs
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")

            return await response.json()

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        pass

    def _calculate_latency(self, region: str) -> float:
        """Calculate estimated latency to region"""
        # Simplified latency calculation based on region
        latency_map = {
            "us-east-1": 10.0,
            "us-west-2": 25.0,
            "eu-west-1": 75.0,
            "asia-east-1": 150.0,
            "us-central1": 20.0,
            "europe-west1": 80.0,
        }
        return latency_map.get(region, 50.0)

    def _get_gpu_specs(self, gpu_type: str) -> Dict[str, Any]:
        """Get GPU specifications"""
        gpu_specs = {
            "A100": {
                "memory_gb": 40,
                "compute_capability": "8.0",
                "tflops": 19.5,
                "bandwidth_gb_s": 1555,
            },
            "V100": {
                "memory_gb": 32,
                "compute_capability": "7.0",
                "tflops": 15.7,
                "bandwidth_gb_s": 900,
            },
            "RTX4090": {
                "memory_gb": 24,
                "compute_capability": "8.9",
                "tflops": 82.6,
                "bandwidth_gb_s": 1008,
            },
            "RTX3090": {
                "memory_gb": 24,
                "compute_capability": "8.6",
                "tflops": 35.6,
                "bandwidth_gb_s": 936,
            },
            "H100": {
                "memory_gb": 80,
                "compute_capability": "9.0",
                "tflops": 67.0,
                "bandwidth_gb_s": 3350,
            },
        }
        return gpu_specs.get(gpu_type, {})

    def _estimate_price(self, instance_type: str, gpu_type: str, region: str) -> float:
        """Estimate price for instance"""
        # Simplified pricing model
        base_prices = {
            "A100": 2.5,
            "V100": 2.0,
            "RTX4090": 1.5,
            "RTX3090": 1.2,
            "H100": 4.0,
        }

        base_price = base_prices.get(gpu_type, 1.0)

        # Region multiplier
        region_multipliers = {
            "us-east-1": 1.0,
            "us-west-2": 1.1,
            "eu-west-1": 1.2,
            "asia-east-1": 1.3,
        }

        region_multiplier = region_multipliers.get(region, 1.0)

        return base_price * region_multiplier
