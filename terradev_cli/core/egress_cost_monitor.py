#!/usr/bin/env python3
"""
Egress Cost Monitor - Proactive egress cost warnings and optimization

CRITICAL FIXES v4.0.0:
- Cross-cloud egress cost warnings
- Multi-hop routing recommendations
- Real-time cost tracking
- Budget alert integration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EgressCostLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EgressCostAlert:
    level: EgressCostLevel
    message: str
    estimated_cost: float
    data_size_gb: float
    recommendation: str
    alternative_providers: List[str]


class EgressCostMonitor:
    """Monitor and warn about egress costs across cloud providers"""

    # Egress costs per GB (USD) - major cloud providers
    EGRESS_COSTS = {
        # AWS
        "aws": {
            "us-east-1": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "us-west-2": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "eu-west-1": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "ap-southeast-1": {"aws": 0.14, "gcp": 0.19, "azure": 0.17, "lambda_labs": 0.14},
        },
        # GCP
        "gcp": {
            "us-central1": {"aws": 0.12, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.12},
            "us-west1": {"aws": 0.12, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.12},
            "europe-west1": {"aws": 0.12, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.12},
            "asia-east1": {"aws": 0.19, "gcp": 0.19, "azure": 0.17, "lambda_labs": 0.19},
        },
        # Azure
        "azure": {
            "eastus": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "westus2": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "westeurope": {"aws": 0.09, "gcp": 0.12, "azure": 0.09, "lambda_labs": 0.09},
            "southeastasia": {"aws": 0.14, "gcp": 0.19, "azure": 0.17, "lambda_labs": 0.14},
        },
        # GPU-first providers (typically zero egress)
        "lambda_labs": {
            "us-east-1": {"aws": 0.0, "gcp": 0.0, "azure": 0.0, "lambda_labs": 0.0},
            "us-west-2": {"aws": 0.0, "gcp": 0.0, "azure": 0.0, "lambda_labs": 0.0},
        },
        "runpod": {
            "us-east": {"aws": 0.0, "gcp": 0.0, "azure": 0.0, "runpod": 0.0},
        },
        "vastai": {
            "us-west": {"aws": 0.01, "gcp": 0.01, "azure": 0.01, "vastai": 0.01},
        },
    }

    # Cost thresholds for alerts (USD)
    COST_THRESHOLDS = {
        EgressCostLevel.LOW: 1.0,      # $1
        EgressCostLevel.MEDIUM: 10.0,   # $10
        EgressCostLevel.HIGH: 50.0,     # $50
        EgressCostLevel.CRITICAL: 100.0, # $100
    }

    def __init__(self, budget_limit: Optional[float] = None):
        self.budget_limit = budget_limit
        self.cost_history: List[Dict[str, Any]] = []

    async def analyze_egress_cost(
        self,
        src_provider: str,
        src_region: str,
        dst_provider: str,
        dst_region: str,
        data_size_gb: float,
        operation_type: str = "transfer"
    ) -> Dict[str, Any]:
        """Analyze egress cost and generate alerts"""
        
        # Get base egress cost
        base_cost = self._get_egress_cost(src_provider, src_region, dst_provider)
        estimated_cost = base_cost * data_size_gb

        # Determine alert level
        alert_level = self._determine_alert_level(estimated_cost)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            src_provider, src_region, dst_provider, dst_region, data_size_gb, estimated_cost
        )

        # Find alternative routes
        alternatives = await self._find_alternative_routes(
            src_provider, src_region, dst_provider, dst_region, data_size_gb
        )

        # Create alert if needed
        alert = None
        if alert_level != EgressCostLevel.LOW:
            alert = EgressCostAlert(
                level=alert_level,
                message=self._generate_alert_message(alert_level, estimated_cost, data_size_gb),
                estimated_cost=estimated_cost,
                data_size_gb=data_size_gb,
                recommendation=recommendations[0] if recommendations else "No specific recommendations",
                alternative_providers=[alt["provider"] for alt in alternatives[:3]],
            )

        # Store in history
        self._store_cost_record({
            "timestamp": datetime.now(),
            "src_provider": src_provider,
            "src_region": src_region,
            "dst_provider": dst_provider,
            "dst_region": dst_region,
            "data_size_gb": data_size_gb,
            "estimated_cost": estimated_cost,
            "alert_level": alert_level.value,
            "operation_type": operation_type,
        })

        return {
            "estimated_cost": estimated_cost,
            "cost_per_gb": base_cost,
            "alert_level": alert_level.value,
            "alert": alert.__dict__ if alert else None,
            "recommendations": recommendations,
            "alternative_routes": alternatives,
            "budget_remaining": self.budget_limit - self._get_monthly_spend() if self.budget_limit else None,
            "budget_exceeded": self.budget_limit and (self._get_monthly_spend() + estimated_cost) > self.budget_limit,
        }

    def _get_egress_cost(self, src_provider: str, src_region: str, dst_provider: str) -> float:
        """Get egress cost per GB"""
        provider_costs = self.EGRESS_COSTS.get(src_provider.lower(), {})
        region_costs = provider_costs.get(src_region, {})
        return region_costs.get(dst_provider.lower(), 0.09)  # Default to $0.09/GB

    def _determine_alert_level(self, estimated_cost: float) -> EgressCostLevel:
        """Determine alert level based on cost"""
        if estimated_cost >= self.COST_THRESHOLDS[EgressCostLevel.CRITICAL]:
            return EgressCostLevel.CRITICAL
        elif estimated_cost >= self.COST_THRESHOLDS[EgressCostLevel.HIGH]:
            return EgressCostLevel.HIGH
        elif estimated_cost >= self.COST_THRESHOLDS[EgressCostLevel.MEDIUM]:
            return EgressCostLevel.MEDIUM
        else:
            return EgressCostLevel.LOW

    def _generate_alert_message(self, level: EgressCostLevel, cost: float, size_gb: float) -> str:
        """Generate alert message"""
        messages = {
            EgressCostLevel.MEDIUM: f"⚠️  Medium egress cost: ${cost:.2f} for {size_gb:.1f}GB transfer",
            EgressCostLevel.HIGH: f"🚨 High egress cost: ${cost:.2f} for {size_gb:.1f}GB transfer",
            EgressCostLevel.CRITICAL: f"💀 CRITICAL egress cost: ${cost:.2f} for {size_gb:.1f}GB transfer",
        }
        return messages.get(level, f"Egress cost: ${cost:.2f} for {size_gb:.1f}GB")

    async def _generate_recommendations(
        self,
        src_provider: str,
        src_region: str,
        dst_provider: str,
        dst_region: str,
        data_size_gb: float,
        estimated_cost: float
    ) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []

        # Check if both providers are in same cloud
        if src_provider == dst_provider:
            recommendations.append("Same-cloud transfer - use internal networking to minimize costs")

        # Check for zero-egress alternatives
        zero_egress_providers = ["lambda_labs", "runpod"]
        if src_provider in zero_egress_providers or dst_provider in zero_egress_providers:
            recommendations.append("Consider using zero-egress providers for cost savings")

        # Large data transfer recommendations
        if data_size_gb > 100:
            recommendations.append("For large transfers (>100GB), consider using physical storage shipment")
            recommendations.append("Use compression to reduce data size before transfer")

        # High cost recommendations
        if estimated_cost > 50:
            recommendations.append("High egress cost detected - consider multi-hop routing")
            recommendations.append("Evaluate if data can be processed closer to source")

        # Regional optimization
        if src_provider != dst_provider:
            recommendations.append("Cross-cloud transfer detected - check for regional cost variations")

        return recommendations

    async def _find_alternative_routes(
        self,
        src_provider: str,
        src_region: str,
        dst_provider: str,
        dst_region: str,
        data_size_gb: float
    ) -> List[Dict[str, Any]]:
        """Find alternative routing options to minimize egress costs"""
        alternatives = []

        # Zero-egress provider alternatives
        zero_egress_providers = ["lambda_labs", "runpod"]
        
        for provider in zero_egress_providers:
            if provider != src_provider and provider != dst_provider:
                # Calculate cost via this provider
                cost_to_provider = self._get_egress_cost(src_provider, src_region, provider) * data_size_gb
                cost_from_provider = self._get_egress_cost(provider, "us-east-1", dst_provider) * data_size_gb
                total_cost = cost_to_provider + cost_from_provider

                alternatives.append({
                    "provider": provider,
                    "route": f"{src_provider} → {provider} → {dst_provider}",
                    "total_cost": total_cost,
                    "savings": max(0, (self._get_egress_cost(src_provider, src_region, dst_provider) * data_size_gb) - total_cost),
                    "hops": 2,
                    "zero_egress": True,
                })

        # Same-region alternatives
        if src_provider != dst_provider:
            # Check if there are same-region instances
            same_region_cost = self._get_egress_cost(src_provider, src_region, dst_provider) * data_size_gb
            if same_region_cost > 0:
                alternatives.append({
                    "provider": f"{src_provider}-{src_region}",
                    "route": f"Stay within {src_provider} {src_region}",
                    "total_cost": same_region_cost * 0.5,  # Assume 50% savings for same-region
                    "savings": same_region_cost * 0.5,
                    "hops": 0,
                    "zero_egress": False,
                })

        # Sort by savings
        alternatives.sort(key=lambda x: x["savings"], reverse=True)
        return alternatives[:5]  # Return top 5 alternatives

    def _store_cost_record(self, record: Dict[str, Any]):
        """Store cost record in history"""
        self.cost_history.append(record)
        
        # Keep only last 1000 records
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]

    def _get_monthly_spend(self) -> float:
        """Calculate total spend for current month"""
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        monthly_costs = [
            record["estimated_cost"] 
            for record in self.cost_history 
            if record["timestamp"] >= current_month
        ]
        
        return sum(monthly_costs)

    async def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get cost summary for the specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_costs = [
            record for record in self.cost_history 
            if record["timestamp"] >= cutoff_date
        ]

        if not recent_costs:
            return {
                "period_days": days,
                "total_cost": 0.0,
                "total_data_gb": 0.0,
                "transfers": 0,
                "avg_cost_per_transfer": 0.0,
                "top_routes": [],
                "cost_trend": "stable",
            }

        total_cost = sum(record["estimated_cost"] for record in recent_costs)
        total_data = sum(record["data_size_gb"] for record in recent_costs)
        
        # Group by route
        route_costs = {}
        for record in recent_costs:
            route = f"{record['src_provider']}→{record['dst_provider']}"
            route_costs[route] = route_costs.get(route, 0) + record["estimated_cost"]

        # Sort routes by cost
        top_routes = sorted(route_costs.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate trend (simple comparison of first half vs second half)
        mid_point = len(recent_costs) // 2
        first_half_cost = sum(record["estimated_cost"] for record in recent_costs[:mid_point])
        second_half_cost = sum(record["estimated_cost"] for record in recent_costs[mid_point:])
        
        if second_half_cost > first_half_cost * 1.2:
            trend = "increasing"
        elif second_half_cost < first_half_cost * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "period_days": days,
            "total_cost": round(total_cost, 2),
            "total_data_gb": round(total_data, 2),
            "transfers": len(recent_costs),
            "avg_cost_per_transfer": round(total_cost / len(recent_costs), 2),
            "top_routes": [{"route": route, "cost": round(cost, 2)} for route, cost in top_routes],
            "cost_trend": trend,
            "budget_remaining": self.budget_limit - total_cost if self.budget_limit else None,
            "budget_utilization": round((total_cost / self.budget_limit) * 100, 1) if self.budget_limit else None,
        }

    async def check_budget_alerts(self) -> List[EgressCostAlert]:
        """Check for budget-related alerts"""
        alerts = []

        if not self.budget_limit:
            return alerts

        current_spend = self._get_monthly_spend()
        utilization = (current_spend / self.budget_limit) * 100

        # Budget utilization alerts
        if utilization >= 90:
            alerts.append(EgressCostAlert(
                level=EgressCostLevel.CRITICAL,
                message=f"💀 Budget critical: {utilization:.1f}% of monthly budget used",
                estimated_cost=current_spend,
                data_size_gb=0,
                recommendation="Immediately review all data transfers and consider alternatives",
                alternative_providers=["lambda_labs", "runpod"],
            ))
        elif utilization >= 75:
            alerts.append(EgressCostAlert(
                level=EgressCostLevel.HIGH,
                message=f"🚨 Budget warning: {utilization:.1f}% of monthly budget used",
                estimated_cost=current_spend,
                data_size_gb=0,
                recommendation="Monitor remaining budget carefully and optimize transfers",
                alternative_providers=["lambda_labs", "runpod"],
            ))
        elif utilization >= 50:
            alerts.append(EgressCostAlert(
                level=EgressCostLevel.MEDIUM,
                message=f"⚠️ Budget notice: {utilization:.1f}% of monthly budget used",
                estimated_cost=current_spend,
                data_size_gb=0,
                recommendation="Continue monitoring egress costs",
                alternative_providers=[],
            ))

        return alerts
