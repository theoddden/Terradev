#!/usr/bin/env python3
"""
Parallel Provisioning Engine — The Differentiator.

Deploys instances across multiple clouds simultaneously using asyncio.
Supports strategies: cheapest-spread, redundant, latency-optimized.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class ProvisionResult:
    """Result of a single provision attempt."""
    __slots__ = ("provider", "region", "instance_id", "gpu_type", "price_hr",
                 "spot", "status", "error", "elapsed_ms")

    def __init__(self, provider: str, region: str, instance_id: str,
                 gpu_type: str, price_hr: float, spot: bool,
                 status: str, error: Optional[str], elapsed_ms: float):
        self.provider = provider
        self.region = region
        self.instance_id = instance_id
        self.gpu_type = gpu_type
        self.price_hr = price_hr
        self.spot = spot
        self.status = status
        self.error = error
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "region": self.region,
            "instance_id": self.instance_id,
            "gpu_type": self.gpu_type,
            "price_hr": self.price_hr,
            "spot": self.spot,
            "status": self.status,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


class ParallelProvisioner:
    """
    Provisions GPU instances across multiple clouds in parallel.

    Usage:
        pp = ParallelProvisioner()
        results = asyncio.run(pp.provision_parallel(allocations))
    """

    def __init__(self):
        from terradev_cli.providers.provider_factory import ProviderFactory
        self.factory = ProviderFactory()

    async def _provision_one(
        self,
        provider_name: str,
        credentials: Dict[str, str],
        gpu_type: str,
        region: str,
        spot: bool,
    ) -> ProvisionResult:
        """Provision a single instance on one provider."""
        t0 = time.monotonic()
        try:
            provider = self.factory.create_provider(provider_name, credentials)
            result = await provider.provision_instance(
                gpu_type=gpu_type,
                region=region,
                spot=spot,
            )
            elapsed = (time.monotonic() - t0) * 1000
            instance_id = result.get("instance_id", f"{provider_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}")
            return ProvisionResult(
                provider=provider_name,
                region=region,
                instance_id=instance_id,
                gpu_type=gpu_type,
                price_hr=result.get("price_per_hour", 0),
                spot=spot,
                status="active",
                error=None,
                elapsed_ms=round(elapsed, 1),
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return ProvisionResult(
                provider=provider_name,
                region=region,
                instance_id="",
                gpu_type=gpu_type,
                price_hr=0,
                spot=spot,
                status="failed",
                error=str(e),
                elapsed_ms=round(elapsed, 1),
            )

    async def provision_parallel(
        self,
        allocations: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> Tuple[str, List[ProvisionResult]]:
        """
        Provision across multiple clouds simultaneously.

        allocations: list of dicts, each with:
            provider, credentials, gpu_type, region, spot (bool)
        max_concurrency: auto-calculated if None (adaptive scaling)

        Returns (parallel_group_id, list of ProvisionResult).
        """
        # Dynamic concurrency scaling - adaptive based on allocation size
        if max_concurrency is None:
            max_concurrency = min(50, max(6, len(allocations) * 2))
        
        group_id = f"pg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        sem = asyncio.Semaphore(max_concurrency)

        async def _guarded(alloc: Dict[str, Any]) -> ProvisionResult:
            async with sem:
                return await self._provision_one(
                    provider_name=alloc["provider"],
                    credentials=alloc.get("credentials", {}),
                    gpu_type=alloc["gpu_type"],
                    region=alloc.get("region", "us-east-1"),
                    spot=alloc.get("spot", False),
                )

        results = await asyncio.gather(*[_guarded(a) for a in allocations])
        return group_id, list(results)

    def build_cheapest_spread(
        self,
        quotes: List[Dict[str, Any]],
        count: int,
        max_price: Optional[float] = None,
        credentials_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Given sorted quotes, build an allocation plan that spreads instances
        across the cheapest providers (no more than ceil(count/2) on any one provider
        for resilience).

        Returns a list of allocation dicts ready for provision_parallel().
        """
        if max_price:
            quotes = [q for q in quotes if q.get("price", 999) <= max_price]

        if not quotes:
            return []

        # Sort by price
        quotes_sorted = sorted(quotes, key=lambda q: q.get("price", 999))

        allocations = []
        provider_counts: Dict[str, int] = {}
        max_per_provider = max((count + 1) // 2, 1)

        creds_map = credentials_map or {}

        for q in quotes_sorted:
            if len(allocations) >= count:
                break
            prov = q.get("provider", "").lower().replace(" ", "_")
            if provider_counts.get(prov, 0) >= max_per_provider:
                continue
            provider_counts[prov] = provider_counts.get(prov, 0) + 1
            allocations.append({
                "provider": prov,
                "credentials": creds_map.get(prov, {}),
                "gpu_type": q.get("gpu_type", "A100"),
                "region": q.get("region", "us-east-1"),
                "spot": q.get("availability") == "spot",
                "price_hr": q.get("price", 0),
            })

        # If we still need more, relax the per-provider cap
        if len(allocations) < count:
            for q in quotes_sorted:
                if len(allocations) >= count:
                    break
                prov = q.get("provider", "").lower().replace(" ", "_")
                allocations.append({
                    "provider": prov,
                    "credentials": creds_map.get(prov, {}),
                    "gpu_type": q.get("gpu_type", "A100"),
                    "region": q.get("region", "us-east-1"),
                    "spot": q.get("availability") == "spot",
                    "price_hr": q.get("price", 0),
                })

        return allocations[:count]
