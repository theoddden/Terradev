#!/usr/bin/env python3
"""
Public IP Billing Tracker - Monitor and track public IP billing across providers

CRITICAL FIXES v4.0.0:
- Public IP cost tracking for CoreWeave and other providers
- Idle IP detection and optimization recommendations
- Billing alerts for unused public IPs
- Cost optimization for IP address management
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IPStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    ASSIGNED = "assigned"
    UNUSED = "unused"


@dataclass
class PublicIPRecord:
    provider: str
    instance_id: str
    ip_address: str
    region: str
    status: IPStatus
    hourly_cost: float
    last_active: datetime
    created_at: datetime
    metadata: Dict[str, Any]


class PublicIPBillingTracker:
    """Track public IP billing across cloud providers"""

    # Public IP costs per hour by provider
    IP_COSTS_PER_HOUR = {
        "aws": 0.0045,  # ~$3.24/month
        "gcp": 0.0025,  # ~$1.80/month
        "azure": 0.0036, # ~$2.59/month
        "coreweave": 0.005, # Separate charge, ~$3.60/month
        "digitalocean": 0.005, # ~$3.60/month
        "vultr": 0.005, # ~$3.60/month
        "linode": 0.005, # ~$3.60/month
        "hetzner": 0.0, # Included in server price
        "runpod": 0.0, # Included
        "lambda_labs": 0.0, # Included
    }

    def __init__(self):
        self.ip_records: List[PublicIPRecord] = []
        self.cost_history: List[Dict[str, Any]] = []

    async def track_public_ip(
        self,
        provider: str,
        instance_id: str,
        ip_address: str,
        region: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a public IP assignment"""
        
        hourly_cost = self.IP_COSTS_PER_HOUR.get(provider.lower(), 0.005)
        
        record = PublicIPRecord(
            provider=provider.lower(),
            instance_id=instance_id,
            ip_address=ip_address,
            region=region,
            status=IPStatus.ACTIVE,
            hourly_cost=hourly_cost,
            last_active=datetime.now(),
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        self.ip_records.append(record)
        
        return {
            "ip_address": ip_address,
            "provider": provider,
            "instance_id": instance_id,
            "region": region,
            "hourly_cost": hourly_cost,
            "monthly_cost_estimate": hourly_cost * 730,  # 30.42 days * 24 hours
            "billing_separate": provider in ["coreweave", "digitalocean", "vultr", "linode"],
            "included_in_instance_cost": provider in ["hetzner", "runpod", "lambda_labs"],
            "status": record.status.value,
            "recommendations": self._get_ip_recommendations(record),
        }

    async def update_ip_status(
        self,
        ip_address: str,
        status: IPStatus,
        last_active: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Update IP status and activity"""
        
        for record in self.ip_records:
            if record.ip_address == ip_address:
                record.status = status
                if last_active:
                    record.last_active = last_active
                
                return {
                    "ip_address": ip_address,
                    "previous_status": record.status.value,
                    "new_status": status.value,
                    "last_active": record.last_active.isoformat(),
                    "idle_hours": (datetime.now() - record.last_active).total_seconds() / 3600,
                    "recommendations": self._get_ip_recommendations(record),
                }
        
        return {"error": f"IP address {ip_address} not found"}

    async def analyze_ip_costs(self, days: int = 30) -> Dict[str, Any]:
        """Analyze public IP costs over specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            record for record in self.ip_records 
            if record.created_at >= cutoff_date
        ]

        if not recent_records:
            return {
                "period_days": days,
                "total_cost": 0.0,
                "total_ips": 0,
                "active_ips": 0,
                "idle_ips": 0,
                "unused_ips": 0,
                "cost_by_provider": {},
                "idle_cost_estimate": 0.0,
                "optimization_savings": 0.0,
                "recommendations": [],
            }

        total_cost = 0.0
        cost_by_provider = {}
        status_counts = {status.value: 0 for status in IPStatus}
        idle_cost_estimate = 0.0

        for record in recent_records:
            # Calculate cost for the period
            hours_active = min(
                (datetime.now() - record.created_at).total_seconds() / 3600,
                days * 24
            )
            period_cost = record.hourly_cost * hours_active
            total_cost += period_cost

            # Provider breakdown
            provider = record.provider
            if provider not in cost_by_provider:
                cost_by_provider[provider] = 0.0
            cost_by_provider[provider] += period_cost

            # Status counts
            status_counts[record.status.value] += 1

            # Idle cost estimation
            if record.status == IPStatus.IDLE:
                idle_hours = (datetime.now() - record.last_active).total_seconds() / 3600
                idle_cost_estimate += record.hourly_cost * idle_hours

        # Calculate optimization potential
        optimization_savings = self._calculate_optimization_savings(recent_records, days)

        return {
            "period_days": days,
            "total_cost": round(total_cost, 2),
            "total_ips": len(recent_records),
            "active_ips": status_counts[IPStatus.ACTIVE.value],
            "idle_ips": status_counts[IPStatus.IDLE.value],
            "unused_ips": status_counts[IPStatus.UNUSED.value],
            "cost_by_provider": {
                provider: round(cost, 2) for provider, cost in cost_by_provider.items()
            },
            "idle_cost_estimate": round(idle_cost_estimate, 2),
            "optimization_savings": round(optimization_savings, 2),
            "recommendations": await self._generate_cost_recommendations(recent_records),
        }

    async def detect_idle_ips(self, idle_threshold_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect idle public IPs that can be released"""
        
        idle_ips = []
        current_time = datetime.now()

        for record in self.ip_records:
            if record.status == IPStatus.ACTIVE:
                idle_hours = (current_time - record.last_active).total_seconds() / 3600
                
                if idle_hours >= idle_threshold_hours:
                    monthly_savings = record.hourly_cost * 730  # Monthly savings if released
                    
                    idle_ips.append({
                        "ip_address": record.ip_address,
                        "provider": record.provider,
                        "instance_id": record.instance_id,
                        "region": record.region,
                        "idle_hours": round(idle_hours, 1),
                        "hourly_cost": record.hourly_cost,
                        "monthly_savings": round(monthly_savings, 2),
                        "last_active": record.last_active.isoformat(),
                        "recommendation": self._get_idle_ip_recommendation(record, idle_hours),
                        "action_required": "Release or reassign IP",
                    })

        # Sort by potential savings
        idle_ips.sort(key=lambda x: x["monthly_savings"], reverse=True)
        return idle_ips

    async def get_billing_alerts(self, budget_limit: Optional[float] = None) -> List[Dict[str, Any]]:
        """Generate billing alerts for public IP usage"""
        
        alerts = []
        current_cost_analysis = await self.analyze_ip_costs(30)  # Last 30 days
        
        # Budget alerts
        if budget_limit and current_cost_analysis["total_cost"] > budget_limit:
            alerts.append({
                "type": "budget_exceeded",
                "severity": "critical",
                "message": f"Public IP billing ${current_cost_analysis['total_cost']:.2f} exceeds budget ${budget_limit:.2f}",
                "current_cost": current_cost_analysis["total_cost"],
                "budget_limit": budget_limit,
                "overage": current_cost_analysis["total_cost"] - budget_limit,
                "recommendation": "Review and release unused public IPs",
            })

        # High idle cost alerts
        if current_cost_analysis["idle_cost_estimate"] > 10.0:
            alerts.append({
                "type": "high_idle_cost",
                "severity": "high",
                "message": f"High idle IP cost estimate: ${current_cost_analysis['idle_cost_estimate']:.2f}/month",
                "idle_cost_estimate": current_cost_analysis["idle_cost_estimate"],
                "recommendation": "Release idle IPs to save costs",
            })

        # Provider-specific alerts
        for provider, cost in current_cost_analysis["cost_by_provider"].items():
            if provider in ["coreweave", "digitalocean", "vultr", "linode"] and cost > 20.0:
                alerts.append({
                    "type": "provider_separate_billing",
                    "severity": "medium",
                    "message": f"High separate billing cost for {provider}: ${cost:.2f}",
                    "provider": provider,
                    "cost": cost,
                    "recommendation": f"Review {provider} IP usage - billed separately from instances",
                })

        return alerts

    def _get_ip_recommendations(self, record: PublicIPRecord) -> List[str]:
        """Get recommendations for IP management"""
        recommendations = []

        if record.provider in ["coreweave", "digitalocean", "vultr", "linode"]:
            recommendations.append("Public IP billed separately - monitor usage carefully")
        
        if record.provider in ["aws", "gcp", "azure"]:
            recommendations.append("Consider using Elastic IP reservations for long-term assignments")
        
        if record.provider in ["hetzner", "runpod", "lambda_labs"]:
            recommendations.append("Public IP included in instance cost - no additional charges")

        return recommendations

    def _get_idle_ip_recommendation(self, record: PublicIPRecord, idle_hours: float) -> str:
        """Get specific recommendation for idle IP"""
        
        if idle_hours > 168:  # 1 week
            return f"IP idle for {idle_hours:.0f} hours - release immediately to save ${record.hourly_cost * 730:.2f}/month"
        elif idle_hours > 72:  # 3 days
            return f"IP idle for {idle_hours:.0f} hours - consider releasing if not needed soon"
        else:
            return f"IP idle for {idle_hours:.0f} hours - monitor for continued inactivity"

    def _calculate_optimization_savings(self, records: List[PublicIPRecord], days: int) -> float:
        """Calculate potential savings from IP optimization"""
        
        savings = 0.0
        current_time = datetime.now()
        
        for record in records:
            if record.status == IPStatus.IDLE:
                idle_hours = (current_time - record.last_active).total_seconds() / 3600
                # Calculate savings if IP was released when it became idle
                savings += record.hourly_cost * min(idle_hours, days * 24)
            elif record.status == IPStatus.UNUSED:
                # Full savings for unused IPs
                total_hours = (current_time - record.created_at).total_seconds() / 3600
                savings += record.hourly_cost * min(total_hours, days * 24)

        return savings

    async def _generate_cost_recommendations(self, records: List[PublicIPRecord]) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        # Count by status
        idle_count = sum(1 for r in records if r.status == IPStatus.IDLE)
        unused_count = sum(1 for r in records if r.status == IPStatus.UNUSED)
        
        if idle_count > 0:
            recommendations.append(f"Release {idle_count} idle IPs to save on monthly costs")
        
        if unused_count > 0:
            recommendations.append(f"Clean up {unused_count} unused IPs")
        
        # Provider-specific recommendations
        coreweave_count = sum(1 for r in records if r.provider == "coreweave")
        if coreweave_count > 5:
            recommendations.append(f"Review {coreweave_count} CoreWeave IPs - billed separately")
        
        return recommendations

    async def get_provider_summary(self, provider: str) -> Dict[str, Any]:
        """Get IP billing summary for specific provider"""
        
        provider_records = [r for r in self.ip_records if r.provider == provider.lower()]
        
        if not provider_records:
            return {
                "provider": provider,
                "total_ips": 0,
                "total_cost": 0.0,
                "hourly_cost_total": 0.0,
                "monthly_cost_estimate": 0.0,
                "status_breakdown": {},
                "billing_model": "unknown",
            }

        hourly_cost_total = sum(r.hourly_cost for r in provider_records)
        monthly_cost_estimate = hourly_cost_total * 730
        
        status_breakdown = {}
        for status in IPStatus:
            status_breakdown[status.value] = sum(1 for r in provider_records if r.status == status)

        # Determine billing model
        if provider.lower() in ["hetzner", "runpod", "lambda_labs"]:
            billing_model = "included_in_instance_cost"
        elif provider.lower() in ["coreweave", "digitalocean", "vultr", "linode"]:
            billing_model = "separate_charge"
        else:
            billing_model = "standard_cloud_pricing"

        return {
            "provider": provider,
            "total_ips": len(provider_records),
            "total_cost": round(sum(r.hourly_cost * ((datetime.now() - r.created_at).total_seconds() / 3600) for r in provider_records), 2),
            "hourly_cost_total": round(hourly_cost_total, 4),
            "monthly_cost_estimate": round(monthly_cost_estimate, 2),
            "status_breakdown": status_breakdown,
            "billing_model": billing_model,
            "cost_per_ip": round(hourly_cost_total / len(provider_records), 4) if provider_records else 0,
            "recommendations": self._get_provider_recommendations(provider.lower()),
        }

    def _get_provider_recommendations(self, provider: str) -> List[str]:
        """Get provider-specific recommendations"""
        
        recommendations = []
        
        if provider == "coreweave":
            recommendations.extend([
                "CoreWeave bills public IPs separately - monitor carefully",
                "Consider using internal networking when possible",
                "Release unused IPs immediately to avoid charges",
            ])
        elif provider == "aws":
            recommendations.extend([
                "Use Elastic IP reservations for long-term assignments",
                "Consider using NAT Gateways for cost optimization",
                "Release unused Elastic IPs to avoid charges",
            ])
        elif provider == "gcp":
            recommendations.extend([
                "Use static IPs for long-term assignments",
                "Consider using Cloud NAT for cost optimization",
                "Release unused static IPs to avoid charges",
            ])
        elif provider == "azure":
            recommendations.extend([
                "Use reserved public IPs for long-term assignments",
                "Consider using Azure NAT for cost optimization",
                "Release unused public IPs to avoid charges",
            ])
        
        return recommendations
