#!/usr/bin/env python3
"""
Base Provider - Abstract base class for cloud providers

Defines the Dict-based provider interface used by all concrete provider
implementations. The event subscription system uses InstanceStatus for
polling-based state-change detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Awaitable
import asyncio
import aiohttp
import logging
import time

logger = logging.getLogger(__name__)

# Rust connection pool integration - optional dependency
# If not available, falls back to pure Python aiohttp connection pooling
USE_RUST_POOL = False

from .types import (
    InstanceStatus,
    ProviderEvent,
    HealthStatus,
)
from .gpu_catalog import normalize as _catalog_normalize


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

    # ── Optional Health & Event Methods (default implementations) ──────────

    # Override this in providers that expose a dedicated health/ping endpoint.
    _health_endpoint: Optional[str] = None

    async def check_health(self) -> HealthStatus:
        """
        Default health check: lightweight HEAD request against the provider
        base URL using the provider's own auth headers.

        Intentionally avoids calling list_instances() — that may deserialise
        hundreds of records just to confirm the provider is reachable.

        Providers with a dedicated health endpoint should set _health_endpoint
        or override this method.
        """
        if not self.session or self.session.closed:
            connector = await self._get_shared_connector()
            self.session = aiohttp.ClientSession(connector=connector)
            self._owns_session = True

        try:
            start = time.time()
            headers = self._get_auth_headers()
            target = self._health_endpoint
            if target:
                async with self.session.head(
                    target, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    latency_ms = (time.time() - start) * 1000
                    return HealthStatus(
                        healthy=resp.status < 500,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                    )
            else:
                # No dedicated endpoint — fall back to list_instances() but
                # bound to a 5-second timeout so we don't block for long.
                async with asyncio.timeout(5):
                    await self.list_instances()
                latency_ms = (time.time() - start) * 1000
                return HealthStatus(
                    healthy=True,
                    latency_ms=latency_ms,
                    timestamp=time.time(),
                )
        except Exception as e:  # noqa: BLE001
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
        _STATUS_MAP = {
            "running": InstanceStatus.RUNNING,
            "pending": InstanceStatus.PENDING,
            "starting": InstanceStatus.STARTING,
            "stopped": InstanceStatus.STOPPED,
            "failed": InstanceStatus.FAILED,
            "terminated": InstanceStatus.TERMINATED,
            "preempted": InstanceStatus.PREEMPTED,
        }

        async def _poll_loop():
            last_states: Dict[str, InstanceStatus] = {}

            while True:
                for iid in instance_ids:
                    try:
                        raw = await self.get_instance_status(iid)
                        raw_status = raw.get("status", "").lower()
                        status = _STATUS_MAP.get(raw_status, InstanceStatus.UNKNOWN)
                        last_status = last_states.get(iid)

                        # Detect state changes
                        if last_status != status:
                            if status == InstanceStatus.PREEMPTED:
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="preempted",
                                        payload={"status": status.value},
                                        timestamp=time.time(),
                                    )
                                )
                            elif status == InstanceStatus.RUNNING and last_status in (
                                InstanceStatus.PENDING,
                                InstanceStatus.STARTING,
                            ):
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="recovered",
                                        payload={"status": status.value},
                                        timestamp=time.time(),
                                    )
                                )
                            elif status == InstanceStatus.FAILED:
                                await callback(
                                    ProviderEvent(
                                        provider=self.name,
                                        instance_id=iid,
                                        event_type="health_degraded",
                                        payload={"status": status.value},
                                        timestamp=time.time(),
                                    )
                                )

                        last_states[iid] = status
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"poll_loop error for {iid}: {e}")

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
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
        """
        Return GPU specifications from the canonical gpu_catalog.

        This is the single source of truth — do not maintain a separate
        hardcoded dict here.  Returns an empty dict if the GPU is unknown.
        """
        descriptor = _catalog_normalize(gpu_type)
        if descriptor is None:
            return {}
        return {
            "memory_gb": descriptor.vram_gb,
            "compute_capability": descriptor.compute_capability or "unknown",
            "tflops_bf16": descriptor.tflops_bf16,
            "tflops_fp16": descriptor.tflops_fp16,
            "tflops_fp32": descriptor.tflops_fp32,
            "bandwidth_gb_s": descriptor.bandwidth_gb_s,
            "nvlink": descriptor.nvlink,
            "vendor": descriptor.vendor.value,
        }

    def _estimate_price(self, instance_type: str, gpu_type: str, region: str) -> Optional[float]:
        """
        Intentionally returns None — there is no reliable static price table.

        Callers should use live provider quotes.  This method exists only for
        backward compatibility with subclasses that call super()._estimate_price().
        If you need a price fallback, implement it in the concrete provider using
        its own API documentation.
        """
        logger.warning(
            "_estimate_price() called for gpu_type=%s region=%s — returning None. "
            "Use live provider quotes instead of static fallback prices.",
            gpu_type, region,
        )
        return None
