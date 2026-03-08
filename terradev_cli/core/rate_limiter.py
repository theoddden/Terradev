#!/usr/bin/env python3
"""
Terradev Rate Limiter and Throttling System
Handles API rate limiting, retries, and provider-specific throttling
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
try:
    from asyncio_throttle import Throttler
except ImportError:
    Throttler = None
try:
    import backoff
except ImportError:
    backoff = None
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class ProviderRateLimit:
    """Rate limit configuration for a provider"""

    requests_per_second: float = 10.0
    requests_per_minute: int = 600
    burst_limit: int = 20
    retry_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: float = 30.0
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


@dataclass
class RateLimitMetrics:
    """Rate limiting metrics"""

    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    current_rate: float = 0.0


class RateLimiter:
    """Advanced rate limiter with provider-specific configurations and adaptive scaling"""

    def __init__(self):
        self.provider_limits: Dict[str, ProviderRateLimit] = {}
        self.provider_throttlers: Dict[str, Throttler] = {}
        self.provider_metrics: Dict[str, RateLimitMetrics] = {}
        self.global_throttler: Optional[Throttler] = None
        self._global_semaphore: Optional[asyncio.Semaphore] = None
        self._adaptive_limits: Dict[str, float] = {}
        self._last_adjustment: Dict[str, datetime] = {}

        # Initialize default provider limits
        self._initialize_default_limits()

        # Global rate limiting with adaptive scaling
        if Throttler is not None:
            self.global_throttler = Throttler(
                rate_limit=50, period=60
            )  # 50 requests per minute globally
        
        # Adaptive global semaphore for enterprise loads
        self._global_semaphore = asyncio.Semaphore(100)  # Base limit
        
        # Background task for adaptive scaling
        self._scaling_task: Optional[asyncio.Task] = None

    def _initialize_default_limits(self):
        """Initialize default rate limits for common providers"""
        default_limits = {
            "aws": ProviderRateLimit(
                requests_per_second=20.0,
                requests_per_minute=1000,
                burst_limit=50,
                retry_attempts=3,
                backoff_factor=1.5,
                timeout=30.0,
            ),
            "gcp": ProviderRateLimit(
                requests_per_second=15.0,
                requests_per_minute=900,
                burst_limit=30,
                retry_attempts=3,
                backoff_factor=2.0,
                timeout=25.0,
            ),
            "azure": ProviderRateLimit(
                requests_per_second=10.0,
                requests_per_minute=600,
                burst_limit=25,
                retry_attempts=3,
                backoff_factor=2.0,
                timeout=35.0,
            ),
            "runpod": ProviderRateLimit(
                requests_per_second=5.0,
                requests_per_minute=300,
                burst_limit=15,
                retry_attempts=5,
                backoff_factor=1.5,
                timeout=20.0,
            ),
            "vastai": ProviderRateLimit(
                requests_per_second=3.0,
                requests_per_minute=180,
                burst_limit=10,
                retry_attempts=4,
                backoff_factor=2.0,
                timeout=25.0,
            ),
            "lambda_labs": ProviderRateLimit(
                requests_per_second=4.0,
                requests_per_minute=240,
                burst_limit=12,
                retry_attempts=3,
                backoff_factor=1.8,
                timeout=30.0,
            ),
            "coreweave": ProviderRateLimit(
                requests_per_second=8.0,
                requests_per_minute=480,
                burst_limit=20,
                retry_attempts=3,
                backoff_factor=1.5,
                timeout=25.0,
            ),
            "tensordock": ProviderRateLimit(
                requests_per_second=2.0,
                requests_per_minute=120,
                burst_limit=8,
                retry_attempts=5,
                backoff_factor=2.5,
                timeout=20.0,
            ),
        }

        for provider, limits in default_limits.items():
            self.set_provider_limit(provider, limits)

    def set_provider_limit(self, provider: str, limit: ProviderRateLimit):
        """Set rate limit for a specific provider"""
        self.provider_limits[provider] = limit

        # Create throttler for the provider
        if Throttler is not None:
            self.provider_throttlers[provider] = Throttler(
                rate_limit=limit.requests_per_second, period=1.0
            )

        # Initialize metrics
        self.provider_metrics[provider] = RateLimitMetrics()

        logger.info(
            f"Set rate limit for {provider}: {limit.requests_per_second} req/s, {limit.requests_per_minute} req/min"
        )

    def get_provider_limit(self, provider: str) -> Optional[ProviderRateLimit]:
        """Get rate limit configuration for a provider"""
        return self.provider_limits.get(provider)

    async def acquire(self, provider: str) -> bool:
        """Acquire rate limit permit for a provider"""
        if provider not in self.provider_throttlers and provider not in self.provider_limits:
            logger.warning(f"No rate limit configured for provider: {provider}")
            return True

        try:
            # Check global rate limit first
            if self.global_throttler:
                await self.global_throttler.acquire()

            # Then check provider-specific rate limit
            if provider in self.provider_throttlers:
                await self.provider_throttlers[provider].acquire()

            # Update metrics
            metrics = self.provider_metrics[provider]
            metrics.total_requests += 1
            metrics.last_request_time = datetime.now()

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Rate limit timeout for provider: {provider}")
            metrics.rate_limited_requests += 1
            return False

    async def execute_with_rate_limit(
        self, provider: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute a function with rate limiting and retries"""
        limit = self.get_provider_limit(provider)
        if not limit:
            # No rate limiting configured
            return await func(*args, **kwargs)

        metrics = self.provider_metrics[provider]
        start_time = time.time()

        @backoff.on_exception(
            backoff.expo,
            (aiohttp.ClientError, asyncio.TimeoutError),
            max_tries=limit.retry_attempts,
            base=limit.backoff_factor,
            max_value=60.0,
        )
        async def _execute_with_retry():
            # Acquire rate limit permit
            if not await self.acquire(provider):
                raise aiohttp.ClientError(f"Rate limit exceeded for {provider}")

            # Execute the function with timeout
            try:
                async with asyncio.timeout(limit.timeout):
                    result = await func(*args, **kwargs)

                    # Update success metrics
                    metrics.successful_requests += 1
                    response_time = time.time() - start_time
                    metrics.average_response_time = (
                        metrics.average_response_time
                        * (metrics.successful_requests - 1)
                        + response_time
                    ) / metrics.successful_requests

                    return result

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {provider} after {limit.timeout}s")
                raise
            except Exception as e:
                metrics.failed_requests += 1
                raise

        try:
            return await _execute_with_retry()
        except Exception as e:
            logger.error(f"Failed to execute function for {provider}: {e}")
            raise

    def get_provider_metrics(self, provider: str) -> Optional[RateLimitMetrics]:
        """Get rate limiting metrics for a provider"""
        return self.provider_metrics.get(provider)

    def get_all_metrics(self) -> Dict[str, RateLimitMetrics]:
        """Get all rate limiting metrics"""
        return self.provider_metrics.copy()

    def reset_metrics(self, provider: Optional[str] = None):
        """Reset metrics for a provider or all providers"""
        if provider:
            if provider in self.provider_metrics:
                self.provider_metrics[provider] = RateLimitMetrics()
        else:
            for p in self.provider_metrics:
                self.provider_metrics[p] = RateLimitMetrics()

    def calculate_current_rate(self, provider: str) -> float:
        """Calculate current request rate for a provider"""
        metrics = self.provider_metrics.get(provider)
        if not metrics or not metrics.last_request_time:
            return 0.0

        # Calculate requests per second over the last minute
        time_diff = (datetime.now() - metrics.last_request_time).total_seconds()
        if time_diff > 60:
            return 0.0

        # Simple calculation - in production, use sliding window
        return metrics.total_requests / max(time_diff, 1.0)

    def is_rate_limited(self, provider: str) -> bool:
        """Check if a provider is currently rate limited"""
        current_rate = self.calculate_current_rate(provider)
        limit = self.get_provider_limit(provider)

        if not limit:
            return False

        return current_rate > limit.requests_per_second

    def get_adaptive_delay(self, provider: str) -> float:
        """Get adaptive delay based on current rate limiting status"""
        limit = self.get_provider_limit(provider)
        if not limit:
            return 0.0

        current_rate = self.calculate_current_rate(provider)
        rate_ratio = current_rate / limit.requests_per_second

        if rate_ratio < 0.5:
            return 0.0  # No delay needed
        elif rate_ratio < 0.8:
            return 0.1  # Small delay
        elif rate_ratio < 0.95:
            return 0.5  # Moderate delay
        else:
            return 1.0  # Significant delay

    async def wait_if_needed(self, provider: str):
        """Wait if rate limiting is needed for a provider"""
        delay = self.get_adaptive_delay(provider)
        if delay > 0:
            await asyncio.sleep(delay)

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting status report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "global_rate_limit": {"requests_per_minute": 50, "current_rate": 0.0},
            "providers": {},
        }

        for provider, metrics in self.provider_metrics.items():
            limit = self.get_provider_limit(provider)
            current_rate = self.calculate_current_rate(provider)

            provider_report = {
                "configured_limit": {
                    "requests_per_second": limit.requests_per_second if limit else 0,
                    "requests_per_minute": limit.requests_per_minute if limit else 0,
                    "burst_limit": limit.burst_limit if limit else 0,
                },
                "current_metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "rate_limited_requests": metrics.rate_limited_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.successful_requests
                    / max(metrics.total_requests, 1),
                    "average_response_time": metrics.average_response_time,
                    "current_rate": current_rate,
                    "is_rate_limited": self.is_rate_limited(provider),
                },
                "last_request": (
                    metrics.last_request_time.isoformat()
                    if metrics.last_request_time
                    else None
                ),
            }

            report["providers"][provider] = provider_report

        return report


class RateLimitedSession:
    """Rate-limited aiohttp session wrapper"""

    def __init__(self, rate_limiter: RateLimiter, provider: str):
        self.rate_limiter = rate_limiter
        self.provider = provider
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited HTTP request"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        async def _make_request():
            return await self.session.request(method, url, **kwargs)

        return await self.rate_limiter.execute_with_rate_limit(
            self.provider, _make_request
        )

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited GET request"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited POST request"""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited PUT request"""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited DELETE request"""
        return await self.request("DELETE", url, **kwargs)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def create_rate_limited_session(provider: str) -> RateLimitedSession:
    """Create a rate-limited session for a provider"""
    return RateLimitedSession(get_rate_limiter(), provider)


# Context manager for rate limiting
@asynccontextmanager
async def rate_limit_context(provider: str):
    """Context manager for rate limiting operations"""
    rate_limiter = get_rate_limiter()
    await rate_limiter.wait_if_needed(provider)
    yield rate_limiter
