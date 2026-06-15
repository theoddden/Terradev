#!/usr/bin/env python3
"""
Base Provider - Abstract base class for cloud providers

Provides both new typed APIs (get_quotes, provision, get_instance) and
backwards-compatible Dict-based shims for existing provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import asdict
import asyncio
import aiohttp
import logging
import time

logger = logging.getLogger(__name__)

# Rust connection pool integration - optional dependency
# If not available, falls back to pure Python aiohttp connection pooling
USE_RUST_POOL = False

# Import new typed contracts
from .types import (
    GPUDescriptor,
    GPUVendor,
    InstanceStatus,
    Quote,
    QuoteRequest,
    ProvisionRequest,
    ProvisionResult,
    InstanceInfo,
    ProviderEvent,
    HealthStatus,
)
from .gpu_catalog import normalize


class BaseProvider(ABC):
    """Abstract base class for cloud providers"""

    # Shared TCP connector for connection pooling across all providers
    _shared_connector: Optional[aiohttp.TCPConnector] = None
    _connector_lock: Optional[asyncio.Lock] = None

    @classmethod
    async def _get_shared_connector(cls) -> aiohttp.TCPConnector:
        """Get or create shared TCP connector with connection pooling"""
        # Lazy-create lock to avoid Python 3.9 event loop binding bug
        if cls._connector_lock is None:
            cls._connector_lock = asyncio.Lock()

        if cls._shared_connector is None or cls._shared_connector.closed:
            async with cls._connector_lock:
                if cls._shared_connector is None or cls._shared_connector.closed:
                    cls._shared_connector = aiohttp.TCPConnector(
                        limit=100,  # max concurrent connections
                        limit_per_host=20,  # max connections per host
                        ttl_dns_cache=300,  # DNS cache TTL
                        use_dns_cache=True,
                        enable_cleanup_closed=True,
                    )
        return cls._shared_connector

    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self.session: Optional[aiohttp.ClientSession] = None
        self._owns_session: bool = False  # True when we lazily created the session

    async def __aenter__(self):
        """Async context manager entry"""
        connector = await self._get_shared_connector()
        self.session = aiohttp.ClientSession(connector=connector)
        self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.aclose()

    async def aclose(self):
        """Close the underlying aiohttp session if we own it.

        Call this explicitly when not using async-with context manager:
            provider = RunPodProvider(creds)
            try:
                result = await provider.get_instance_quotes(...)
            finally:
                await provider.aclose()
        """
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        self._owns_session = False

    # Abstract methods for old Dict-based API - providers must implement these
    @abstractmethod
    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get instance quotes for GPU type (old Dict-based API)"""
        pass

    @abstractmethod
    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str, ssh_public_key: str = ""
    ) -> Dict[str, Any]:
        """Provision an instance (old Dict-based API)"""
        pass

    @abstractmethod
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status (old Dict-based API)"""
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

    # ── New Typed APIs (to be implemented by providers) ─────────────────────
    # These are optional for now - providers can implement them when migrating
    # from the old Dict-based API above.

    async def get_quotes(self, request: QuoteRequest) -> List[Quote]:
        """
        Get instance quotes for GPU type (new typed API).

        Providers should implement this method when migrating from the old
        Dict-based API. Default implementation returns empty list.
        """
        return []

    async def provision(self, request: ProvisionRequest) -> ProvisionResult:
        """
        Provision an instance (new typed API).

        Providers should implement this method when migrating from the old
        Dict-based API. Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement provision(). "
            "Please use provision_instance() for now."
        )

    async def get_instance(self, instance_id: str) -> InstanceInfo:
        """
        Get instance status (new typed API).

        Providers should implement this method when migrating from the old
        Dict-based API. Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_instance(). "
            "Please use get_instance_status() for now."
        )

    # ── Optional Health & Event Methods (default implementations) ──────────

    async def check_health(self) -> HealthStatus:
        """
        Default health check: lightweight API call.

        Providers can override with custom health endpoints.
        """
        try:
            start = time.time()
            await self.list_instances()
            latency_ms = (time.time() - start) * 1000
            return HealthStatus(
                healthy=True,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                reason=str(e),
                timestamp=time.time(),
            )

    async def subscribe_events(
        self,
        instance_ids: List[str],
        callback: Callable[[ProviderEvent], Awaitable[None]],
        poll_interval_s: int = 30,
    ) -> asyncio.Task:
        """
        Default polling-based event detection.

        Providers with native webhooks (RunPod, AWS CloudWatch) can override
        this with streaming implementations.

        Inspired by HarmonAIze macro-level pub/sub abstraction.
        """
        async def _poll_loop():
            last_states: Dict[str, InstanceStatus] = {}

            while True:
                for iid in instance_ids:
                    try:
                        info = await self.get_instance(iid)
                        last_status = last_states.get(iid)

                        # Detect state changes
                        if last_status != info.status:
                            if info.status == InstanceStatus.PREEMPTED:
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="preempted",
                                        payload={"status": info.status.value},
                                        timestamp=time.time(),
                                    )
                                )
                            elif info.status == InstanceStatus.RUNNING and last_status in (
                                InstanceStatus.PENDING,
                                InstanceStatus.STARTING,
                            ):
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="recovered",
                                        payload={"status": info.status.value},
                                        timestamp=time.time(),
                                    )
                                )
                            elif info.status == InstanceStatus.FAILED:
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="health_degraded",
                                        payload={"status": info.status.value},
                                        timestamp=time.time(),
                                    )
                                )

                        last_states[iid] = info.status
                    except Exception as e:
                        # Log but don't break the loop
                        logger.debug(f"poll_loop error for {iid}: {e}")

                await asyncio.sleep(poll_interval_s)

        return asyncio.create_task(_poll_loop())

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
        if not self.session or self.session.closed:
            connector = await self._get_shared_connector()
            self.session = aiohttp.ClientSession(connector=connector)
            self._owns_session = True

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
                "tflops": 1979.0,
                "bandwidth_gb_s": 3350,
            },
            "H200": {
                "memory_gb": 141,
                "compute_capability": "9.0",
                "tflops": 1979.0,
                "bandwidth_gb_s": 4800,
            },
            "MI300X": {
                "memory_gb": 192,
                "compute_capability": "9.4",
                "tflops": 1307.4,
                "bandwidth_gb_s": 5300,
            },
            "A100-80GB": {
                "memory_gb": 80,
                "compute_capability": "8.0",
                "tflops": 312.0,
                "bandwidth_gb_s": 2000,
            },
            "B200": {
                "memory_gb": 192,
                "compute_capability": "10.0",
                "tflops": 4500.0,
                "bandwidth_gb_s": 8000,
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
