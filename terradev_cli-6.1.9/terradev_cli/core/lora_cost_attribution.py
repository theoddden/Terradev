#!/usr/bin/env python3
"""
LoRA Cost Attribution Service - Track GPU and compute costs per adapter/tenant

Provides:
- Per-adapter cost tracking (GPU time, compute, storage)
- Per-tenant cost aggregation
- Cost-aware warm pool optimization
- Billing and chargeback support
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class CostUnit(str, Enum):
    """Units for cost tracking"""
    GPU_HOURS = "gpu_hours"
    TOKENS = "tokens"
    REQUESTS = "requests"
    STORAGE_GB = "storage_gb"


@dataclass
class CostConfig:
    """Configuration for cost attribution"""

    # GPU cost per hour (by instance type)
    gpu_cost_per_hour: Dict[str, float] = field(
        default_factory=lambda: {
            "a10g": 1.50,  # AWS g5.xlarge
            "a100": 3.50,  # AWS p4d.24xlarge (per GPU)
            "h100": 4.50,  # AWS p5.48xlarge (per GPU)
            "l4": 0.80,  # AWS g6.xlarge
        }
    )
    # Storage cost per GB per month
    storage_cost_per_gb_month: float = 0.10
    # Token cost per 1k tokens (for API calls)
    token_cost_per_1k: float = 0.001
    # Enable cost tracking
    enable_tracking: bool = True
    # Cost attribution window (days)
    attribution_window_days: int = 30


@dataclass
class AdapterCostRecord:
    """Cost record for a single adapter"""

    adapter_name: str
    tenant_id: Optional[str] = None
    gpu_hours: float = 0.0
    tokens_processed: int = 0
    requests_served: int = 0
    storage_gb: float = 0.0
    total_cost_usd: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TenantCostRecord:
    """Aggregated cost record for a tenant"""

    tenant_id: str
    adapters: Set[str] = field(default_factory=set)
    gpu_hours: float = 0.0
    tokens_processed: int = 0
    requests_served: int = 0
    storage_gb: float = 0.0
    total_cost_usd: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CostEvent:
    """A single cost event (request, load, etc.)"""

    adapter_name: str
    tenant_id: Optional[str]
    replica_id: str
    event_type: str  # "inference", "load", "unload", "storage"
    gpu_seconds: float = 0.0
    tokens: int = 0
    storage_gb: float = 0.0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CostAttributionService:
    """
    Service for tracking and attributing costs to LoRA adapters and tenants.

    Provides:
    - Real-time cost tracking per adapter
    - Tenant-level cost aggregation
    - Cost-aware warm pool recommendations
    - Billing and chargeback support
    """

    def __init__(self, config: CostConfig, config_dir: Optional[Path] = None):
        self.config = config
        self.config_dir = config_dir or Path.home() / ".terradev"

        # Cost state
        self.adapter_costs: Dict[str, AdapterCostRecord] = {}
        self.tenant_costs: Dict[str, TenantCostRecord] = {}
        self.cost_events: List[CostEvent] = []

        # Metrics file
        self.cost_db_file = self.config_dir / "lora_cost_attribution.json"

        # Load historical data
        self._load_cost_data()

    async def record_inference_cost(
        self,
        adapter_name: str,
        tenant_id: Optional[str],
        replica_id: str,
        gpu_seconds: float,
        tokens: int,
        instance_type: str = "a10g",
    ) -> float:
        """
        Record cost for an inference request.

        Args:
            adapter_name: LoRA adapter name
            tenant_id: Tenant ID (if multi-tenant)
            replica_id: Replica that served the request
            gpu_seconds: GPU time used
            tokens: Number of tokens processed
            instance_type: GPU instance type for cost calculation

        Returns:
            Cost in USD for this request
        """
        if not self.config.enable_tracking:
            return 0.0

        # Calculate cost
        gpu_cost_per_hour = self.config.gpu_cost_per_hour.get(instance_type, 1.50)
        gpu_cost = (gpu_seconds / 3600) * gpu_cost_per_hour
        token_cost = (tokens / 1000) * self.config.token_cost_per_1k
        total_cost = gpu_cost + token_cost

        # Record event
        event = CostEvent(
            adapter_name=adapter_name,
            tenant_id=tenant_id,
            replica_id=replica_id,
            event_type="inference",
            gpu_seconds=gpu_seconds,
            tokens=tokens,
            cost_usd=total_cost,
        )
        self.cost_events.append(event)

        # Update adapter cost
        if adapter_name not in self.adapter_costs:
            self.adapter_costs[adapter_name] = AdapterCostRecord(
                adapter_name=adapter_name, tenant_id=tenant_id
            )

        record = self.adapter_costs[adapter_name]
        record.gpu_hours += gpu_seconds / 3600
        record.tokens_processed += tokens
        record.requests_served += 1
        record.total_cost_usd += total_cost
        record.last_updated = datetime.now()

        # Update tenant cost
        if tenant_id:
            if tenant_id not in self.tenant_costs:
                self.tenant_costs[tenant_id] = TenantCostRecord(tenant_id=tenant_id)
            tenant_record = self.tenant_costs[tenant_id]
            tenant_record.adapters.add(adapter_name)
            tenant_record.gpu_hours += gpu_seconds / 3600
            tenant_record.tokens_processed += tokens
            tenant_record.requests_served += 1
            tenant_record.total_cost_usd += total_cost
            tenant_record.last_updated = datetime.now()

        # Clean old events
        self._clean_old_events()

        # Save to disk
        self._save_cost_data()

        logger.debug(
            f"Recorded inference cost: {adapter_name} ${total_cost:.4f} "
            f"(gpu: ${gpu_cost:.4f}, tokens: ${token_cost:.4f})"
        )

        return total_cost

    async def record_storage_cost(
        self,
        adapter_name: str,
        tenant_id: Optional[str],
        storage_gb: float,
    ) -> float:
        """
        Record storage cost for an adapter.

        Args:
            adapter_name: LoRA adapter name
            tenant_id: Tenant ID (if multi-tenant)
            storage_gb: Storage size in GB

        Returns:
            Monthly storage cost in USD
        """
        if not self.config.enable_tracking:
            return 0.0

        # Calculate monthly storage cost
        monthly_cost = storage_gb * self.config.storage_cost_per_gb_month

        # Update adapter cost
        if adapter_name not in self.adapter_costs:
            self.adapter_costs[adapter_name] = AdapterCostRecord(
                adapter_name=adapter_name, tenant_id=tenant_id
            )

        record = self.adapter_costs[adapter_name]
        record.storage_gb = storage_gb
        record.total_cost_usd += monthly_cost
        record.last_updated = datetime.now()

        # Update tenant cost
        if tenant_id:
            if tenant_id not in self.tenant_costs:
                self.tenant_costs[tenant_id] = TenantCostRecord(tenant_id=tenant_id)
            tenant_record = self.tenant_costs[tenant_id]
            tenant_record.adapters.add(adapter_name)
            tenant_record.storage_gb += storage_gb
            tenant_record.total_cost_usd += monthly_cost
            tenant_record.last_updated = datetime.now()

        self._save_cost_data()

        logger.debug(f"Recorded storage cost: {adapter_name} ${monthly_cost:.4f}/month ({storage_gb}GB)")

        return monthly_cost

    async def get_adapter_cost(self, adapter_name: str) -> Optional[AdapterCostRecord]:
        """Get cost record for a specific adapter"""
        return self.adapter_costs.get(adapter_name)

    async def get_tenant_cost(self, tenant_id: str) -> Optional[TenantCostRecord]:
        """Get cost record for a specific tenant"""
        return self.tenant_costs.get(tenant_id)

    async def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cost summary for the specified time window.

        Args:
            days: Number of days to include in summary

        Returns:
            Summary dict with total costs, top adapters, top tenants
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Filter events within window
        recent_events = [
            e for e in self.cost_events if e.timestamp > cutoff
        ]

        # Calculate totals
        total_gpu_hours = sum(e.gpu_seconds / 3600 for e in recent_events)
        total_tokens = sum(e.tokens for e in recent_events)
        total_requests = len(recent_events)
        total_cost = sum(e.cost_usd for e in recent_events)

        # Top adapters by cost
        adapter_costs: Dict[str, float] = {}
        for event in recent_events:
            adapter_costs[event.adapter_name] = (
                adapter_costs.get(event.adapter_name, 0.0) + event.cost_usd
            )
        top_adapters = sorted(adapter_costs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top tenants by cost
        tenant_costs: Dict[str, float] = {}
        for event in recent_events:
            if event.tenant_id:
                tenant_costs[event.tenant_id] = (
                    tenant_costs.get(event.tenant_id, 0.0) + event.cost_usd
                )
        top_tenants = sorted(tenant_costs.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "window_days": days,
            "total_gpu_hours": round(total_gpu_hours, 2),
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "total_cost_usd": round(total_cost, 2),
            "top_adapters": [
                {"name": name, "cost_usd": round(cost, 2)}
                for name, cost in top_adapters
            ],
            "top_tenants": [
                {"tenant_id": tid, "cost_usd": round(cost, 2)}
                for tid, cost in top_tenants
            ],
        }

    async def get_cost_breakdown(
        self, adapter_name: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown for an adapter.

        Args:
            adapter_name: Adapter name
            days: Number of days to include

        Returns:
            Detailed breakdown by cost type
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Filter events for this adapter
        adapter_events = [
            e for e in self.cost_events
            if e.adapter_name == adapter_name and e.timestamp > cutoff
        ]

        # Calculate breakdown
        gpu_cost = sum(
            (e.gpu_seconds / 3600) * self.config.gpu_cost_per_hour.get("a10g", 1.50)
            for e in adapter_events
        )
        token_cost = sum(
            (e.tokens / 1000) * self.config.token_cost_per_1k
            for e in adapter_events
        )
        total_cost = sum(e.cost_usd for e in adapter_events)

        # Per-replica breakdown
        replica_costs: Dict[str, float] = {}
        for event in adapter_events:
            replica_costs[event.replica_id] = (
                replica_costs.get(event.replica_id, 0.0) + event.cost_usd
            )

        return {
            "adapter_name": adapter_name,
            "window_days": days,
            "total_requests": len(adapter_events),
            "gpu_cost_usd": round(gpu_cost, 2),
            "token_cost_usd": round(token_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "cost_by_replica": [
                {"replica_id": rid, "cost_usd": round(cost, 2)}
                for rid, cost in sorted(replica_costs.items(), key=lambda x: x[1], reverse=True)
            ],
        }

    async def get_warm_pool_recommendations(
        self, budget_limit_usd: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get cost-aware warm pool recommendations.

        Args:
            budget_limit_usd: Optional budget limit for warm pool

        Returns:
            List of recommendations for warm pool optimization
        """
        recommendations = []

        # Get recent cost data
        summary = await self.get_cost_summary(days=7)

        # High-cost adapters that should be kept warm
        for adapter in summary["top_adapters"]:
            if adapter["cost_usd"] > 10.0:  # High cost threshold
                recommendations.append({
                    "type": "keep_warm",
                    "adapter_name": adapter["name"],
                    "reason": f"High cost adapter (${adapter['cost_usd']}/week)",
                    "priority": "high",
                })

        # Low-cost adapters that can be evicted
        for adapter in summary["top_adapters"]:
            if adapter["cost_usd"] < 1.0:  # Low cost threshold
                recommendations.append({
                    "type": "consider_eviction",
                    "adapter_name": adapter["name"],
                    "reason": f"Low cost adapter (${adapter['cost_usd']}/week)",
                    "priority": "low",
                })

        # Budget-aware recommendations
        if budget_limit_usd:
            current_weekly_cost = summary["total_cost_usd"] / 30 * 7
            if current_weekly_cost > budget_limit_usd:
                recommendations.append({
                    "type": "budget_alert",
                    "reason": f"Weekly cost ${current_weekly_cost:.2f} exceeds budget ${budget_limit_usd:.2f}",
                    "priority": "critical",
                    "suggested_action": "Evict low-priority adapters",
                })

        return recommendations

    def _clean_old_events(self):
        """Remove cost events older than attribution window"""
        cutoff = datetime.now() - timedelta(days=self.config.attribution_window_days)
        self.cost_events = [e for e in self.cost_events if e.timestamp > cutoff]

    def _load_cost_data(self):
        """Load cost data from disk"""
        try:
            if self.cost_db_file.exists():
                with open(self.cost_db_file) as f:
                    data = json.load(f)

                    # Load adapter costs
                    for adapter_data in data.get("adapter_costs", []):
                        record = AdapterCostRecord(**adapter_data)
                        record.last_updated = datetime.fromisoformat(adapter_data["last_updated"])
                        self.adapter_costs[record.adapter_name] = record

                    # Load tenant costs
                    for tenant_data in data.get("tenant_costs", []):
                        record = TenantCostRecord(**tenant_data)
                        record.adapters = set(tenant_data["adapters"])
                        record.last_updated = datetime.fromisoformat(tenant_data["last_updated"])
                        self.tenant_costs[record.tenant_id] = record

                    # Load cost events
                    for event_data in data.get("cost_events", []):
                        event = CostEvent(**event_data)
                        event.timestamp = datetime.fromisoformat(event_data["timestamp"])
                        self.cost_events.append(event)

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to load cost data: {e}")

    def _save_cost_data(self):
        """Save cost data to disk"""
        try:
            data = {
                "adapter_costs": [
                    {
                        **record.__dict__,
                        "last_updated": record.last_updated.isoformat(),
                    }
                    for record in self.adapter_costs.values()
                ],
                "tenant_costs": [
                    {
                        **record.__dict__,
                        "adapters": list(record.adapters),
                        "last_updated": record.last_updated.isoformat(),
                    }
                    for record in self.tenant_costs.values()
                ],
                "cost_events": [
                    {
                        **event.__dict__,
                        "timestamp": event.timestamp.isoformat(),
                    }
                    for event in self.cost_events[-10000:]  # Keep last 10k events
                ],
            }
            with open(self.cost_db_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to save cost data: {e}")
