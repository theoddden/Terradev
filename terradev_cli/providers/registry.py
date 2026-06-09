#!/usr/bin/env python3
"""
Provider Registry - Circuit breaker and health tracking for cloud providers.

Wraps ProviderFactory with circuit-breaker logic, health metrics, and
spot preemption rate tracking. Inspired by SkyServe's SpotHedge
per-zone preemptiveness scoring.

Benefits:
- Circuit breaker: skip failing providers to avoid wasted API calls
- Spot preemption tracking: rank providers by reliability for spot workloads
- Health metrics: latency tracking, success/failure rates
- Provider ranking: sort by health → spot score → latency
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional
from collections import defaultdict

from .types import ProviderHealth, HealthStatus
from .provider_factory import ProviderFactory
from .provider_profiles import get_profile

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Wraps ProviderFactory with circuit breaker and health tracking.

    Tracks per-provider health metrics and spot preemption rates.
    Used for intelligent provider selection and circuit breaker decisions.
    """

    # Circuit breaker thresholds
    FAILURE_THRESHOLD = 3  # open circuit after N consecutive failures
    RECOVERY_WINDOW_S = 120  # try again after N seconds (2 min)

    # Spot preemption scoring (SkyServe SpotHedge-inspired)
    PREEMPTION_DECAY_HALF_LIFE_S = 3600  # 1 hour half-life for preemption history
    PREEMPTION_WEIGHT = 0.7  # weight for preemption rate in scoring
    LATENCY_WEIGHT = 0.3  # weight for latency in scoring

    def __init__(self, factory: Optional[ProviderFactory] = None):
        self.factory = factory or ProviderFactory()
        self._health: Dict[str, ProviderHealth] = defaultdict(
            lambda: ProviderHealth(provider="")
        )
        self._spot_preemptions: Dict[str, List[float]] = defaultdict(list)  # provider → timestamps
        self._lock: Optional[asyncio.Lock] = None

    def is_healthy(self, provider: str) -> bool:
        """
        Check if provider is healthy (circuit breaker not open).

        Circuit breaker opens after FAILURE_THRESHOLD consecutive failures
        and closes after RECOVERY_WINDOW_S of recovery time.
        """
        health = self._health[provider]
        health.provider = provider  # ensure name is set

        # If circuit is open, check if recovery window has passed
        if health.consecutive_failures >= self.FAILURE_THRESHOLD:
            if time.time() - health.last_failure_ts < self.RECOVERY_WINDOW_S:
                return False  # circuit still open
            else:
                # Recovery window passed, reset failures
                health.consecutive_failures = 0
                logger.info(f"Circuit breaker recovered for provider: {provider}")

        return True

    async def record_success(self, provider: str, latency_ms: float):
        """
        Record a successful operation for a provider.

        Resets consecutive failure counter and updates latency metrics.
        """
        # Lazy-create lock to avoid Python 3.9 event loop binding bug
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            health = self._health[provider]
            health.provider = provider
            health.consecutive_failures = 0
            health.last_success_ts = time.time()
            health.total_provisions += 1

            # Exponential moving average for latency
            if health.avg_latency_ms == 0:
                health.avg_latency_ms = latency_ms
            else:
                health.avg_latency_ms = 0.9 * health.avg_latency_ms + 0.1 * latency_ms

    async def record_failure(self, provider: str, error: str = ""):
        """
        Record a failed operation for a provider.

        Increments consecutive failure counter. Opens circuit breaker
        after FAILURE_THRESHOLD consecutive failures.
        """
        # Lazy-create lock to avoid Python 3.9 event loop binding bug
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            health = self._health[provider]
            health.provider = provider
            health.consecutive_failures += 1
            health.last_failure_ts = time.time()
            health.total_failures += 1

            if health.consecutive_failures >= self.FAILURE_THRESHOLD:
                logger.warning(
                    f"Circuit breaker opened for provider: {provider} "
                    f"(failures: {health.consecutive_failures}, error: {error})"
                )

    async def record_preemption(self, provider: str, region: str = ""):
        """
        Record a spot instance preemption for a provider.

        Used to calculate spot preemption rate for provider ranking.
        """
        # Lazy-create lock to avoid Python 3.9 event loop binding bug
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            now = time.time()
            self._spot_preemptions[provider].append(now)

            # Clean old preemptions (older than 24 hours)
            cutoff = now - 86400
            self._spot_preemptions[provider] = [
                ts for ts in self._spot_preemptions[provider] if ts > cutoff
            ]

    def get_spot_score(self, provider: str, region: str = "") -> float:
        """
        Calculate spot preemption risk score for a provider.

        Lower score = safer for spot instances.
        Combines preemption rate with recency (more recent = higher risk).

        Equivalent to SkyServe's P(preemption) estimate per zone.

        Returns:
            Score between 0.0 (safest) and 1.0 (riskiest)
        """
        preemptions = self._spot_preemptions.get(provider, [])
        if not preemptions:
            return 0.0  # no history, assume safe

        now = time.time()
        # Weight preemptions by recency (exponential decay)
        weighted_count = 0.0
        for ts in preemptions:
            age = now - ts
            decay = 0.5 ** (age / self.PREEMPTION_DECAY_HALF_LIFE_S)
            weighted_count += decay

        # Normalize to 0-1 range (heuristic: 10 preemptions/day = 1.0 risk)
        return min(weighted_count / 10.0, 1.0)

    def get_health(self, provider: str) -> ProviderHealth:
        """Get current health metrics for a provider."""
        health = self._health[provider]
        health.provider = provider
        return health

    def get_all_health(self) -> Dict[str, ProviderHealth]:
        """Get health metrics for all tracked providers."""
        return dict(self._health)

    def ranked_providers(
        self,
        gpu_canonical: str,
        spot: bool = False,
        max_providers: int = 10,
    ) -> List[str]:
        """
        Return providers ranked by suitability for the given GPU and workload.

        Ranking criteria (in order):
        1. Health (circuit breaker status)
        2. Provider profile quirks (egress costs, fallback routing, etc.)
        3. Spot preemption score (if spot=True)
        4. Average latency

        Args:
            gpu_canonical: Canonical GPU name (e.g., "H100-80GB")
            spot: If True, prioritize providers with low preemption rates
            max_providers: Maximum number of providers to return

        Returns:
            List of provider names sorted by suitability
        """
        # Get all providers that support this GPU (from factory)
        # TODO: add capability pre-filtering once providers implement capabilities()
        all_providers = self.factory.get_supported_providers()

        # Filter by health (circuit breaker)
        healthy_providers = [p for p in all_providers if self.is_healthy(p)]

        # Score each provider
        scored = []
        for provider in healthy_providers:
            health = self._health[provider]
            health.provider = provider
            
            # Get provider profile for quirk-aware scoring
            profile = get_profile(provider)

            # Base score from health (success rate)
            if health.total_provisions > 0:
                success_rate = 1.0 - (health.total_failures / health.total_provisions)
            else:
                success_rate = 1.0  # no history, assume healthy

            # Egress cost penalty (lower egress = better for data-heavy workloads)
            egress_penalty = profile.egress_cost * 0.1  # scale factor

            # Spot preemption penalty
            if spot:
                spot_score = self.get_spot_score(provider)
                spot_penalty = spot_score * self.PREEMPTION_WEIGHT
            else:
                spot_penalty = 0.0

            # Latency penalty (normalize to 0-1, assume 500ms = 1.0)
            latency_penalty = min(health.avg_latency_ms / 500.0, 1.0) * self.LATENCY_WEIGHT

            # Combined score (higher = better)
            combined_score = success_rate - egress_penalty - spot_penalty - latency_penalty

            scored.append((provider, combined_score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top N provider names
        return [p for p, _ in scored[:max_providers]]

    def reset_health(self, provider: str):
        """Reset health metrics for a provider (manual recovery)."""
        self._health[provider] = ProviderHealth(provider=provider)
        self._spot_preemptions[provider] = []
        logger.info(f"Reset health metrics for provider: {provider}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get overall registry statistics.

        Useful for monitoring and debugging.
        """
        total_providers = len(self._health)
        healthy_providers = sum(1 for p in self._health.values() if self.is_healthy(p.provider))
        total_provisions = sum(h.total_provisions for h in self._health.values())
        total_failures = sum(h.total_failures for h in self._health.values())

        return {
            "total_providers": total_providers,
            "healthy_providers": healthy_providers,
            "total_provisions": total_provisions,
            "total_failures": total_failures,
            "overall_success_rate": (
                1.0 - (total_failures / total_provisions) if total_provisions > 0 else 1.0
            ),
        }
