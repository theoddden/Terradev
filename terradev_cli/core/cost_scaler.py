#!/usr/bin/env python3
"""
Cost-Aware Scaling Manager - Billing-optimized scaling policies for bursty workloads

Addresses Reddit post pain points:
1. "Billing alignment matters more than peak latency" - cost-first scaling
2. Users care about not paying for idle VRAM - intelligent cost optimization
3. Deterministic scaling under memory pressure - predictable cost behavior
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CostStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"              # Always minimize cost
    BALANCE_COST_LATENCY = "balance_cost_latency"  # Balance cost vs performance
    LATENCY_CRITICAL = "latency_critical"        # Prioritize latency over cost
    BUDGET_CONSTRAINED = "budget_constrained"    # Strict budget limits


@dataclass
class CostConfig:
    """Configuration for cost-aware scaling"""
    hourly_budget_usd: float = 10.0              # Maximum hourly budget
    cost_per_gb_hour_usd: float = 0.10           # GPU memory cost per GB/hour
    cold_start_cost_penalty_usd: float = 0.05     # Cost penalty for cold starts
    peak_hour_multiplier: float = 1.5            # Cost multiplier during peak hours
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16, 17, 18])
    strategy: CostStrategy = CostStrategy.BALANCE_COST_LATENCY
    enable_cost_prediction: bool = True           # Predict future costs
    cost_threshold_for_warming: float = 0.5     # USD/hour threshold for warming


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    total_cost_usd: float = 0.0
    current_hourly_cost_usd: float = 0.0
    memory_cost_usd: float = 0.0
    cold_start_penalty_usd: float = 0.0
    cost_savings_usd: float = 0.0
    budget_utilization_percent: float = 0.0
    hourly_costs: List[float] = field(default_factory=list)  # Last 24 hours


class CostScaler:
    """
    Cost-aware scaling manager for multi-model inference.
    
    Solves the "billing alignment matters more than peak latency" problem by:
    1. Prioritizing cost optimization over latency when appropriate
    2. Predictive cost management based on usage patterns
    3. Budget-aware scaling decisions
    4. Transparent cost reporting and optimization
    """
    
    def __init__(self, config: CostConfig, config_dir: Optional[Path] = None):
        self.config = config
        self.config_dir = config_dir or Path.home() / '.terradev'
        
        # Cost tracking
        self.model_memory_usage: Dict[str, float] = {}  # GB per model
        self.model_load_costs: Dict[str, float] = {}     # Cost per load
        self.hourly_cost_history: List[Tuple[datetime, float]] = []
        self.current_memory_usage_gb: float = 0.0
        
        # Metrics
        self.metrics = CostMetrics()
        self.metrics_file = self.config_dir / 'cost_metrics.json'
        self.cost_history_file = self.config_dir / 'cost_history.json'
        
        # Background tasks
        self._cost_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load historical data
        self._load_metrics()
        self._load_cost_history()
    
    async def start(self):
        """Start cost monitoring background tasks"""
        self._running = True
        self._cost_monitor_task = asyncio.create_task(self._cost_monitor())
        logger.info("Cost Scaler started")
    
    async def stop(self):
        """Stop cost monitoring"""
        self._running = False
        if self._cost_monitor_task:
            self._cost_monitor_task.cancel()
        logger.info("Cost Scaler stopped")
    
    # ── Cost Calculation ──
    
    def calculate_memory_cost(self, memory_gb: float, hours: float = 1.0) -> float:
        """Calculate cost for GPU memory usage"""
        base_cost = memory_gb * self.config.cost_per_gb_hour_usd * hours
        
        # Apply peak hour multiplier
        if self._is_peak_hour():
            base_cost *= self.config.peak_hour_multiplier
        
        return base_cost
    
    def calculate_cold_start_penalty(self, model_id: str) -> float:
        """Calculate cold start penalty cost"""
        return self.config.cold_start_cost_penalty_usd
    
    def get_current_hourly_cost(self) -> float:
        """Get current hourly cost based on active models"""
        memory_cost = self.calculate_memory_cost(self.current_memory_usage_gb)
        
        # Add cold start penalties for recent loads
        recent_penalties = sum(
            cost for model_id, cost in self.model_load_costs.items()
            if self._was_recently_loaded(model_id)
        )
        
        total_cost = memory_cost + recent_penalties
        
        # Update metrics
        self.metrics.current_hourly_cost_usd = total_cost
        self.metrics.memory_cost_usd = memory_cost
        self.metrics.cold_start_penalty_usd = recent_penalties
        self.metrics.budget_utilization_percent = (
            total_cost / self.config.hourly_budget_usd
        ) * 100
        
        return total_cost
    
    def _is_peak_hour(self) -> bool:
        """Check if current time is during peak hours"""
        return datetime.now().hour in self.config.peak_hours
    
    def _was_recently_loaded(self, model_id: str, minutes: int = 5) -> bool:
        """Check if a model was recently loaded (for penalty calculation)"""
        # This would integrate with actual load timestamps
        # For now, assume recent if in model_load_costs
        return model_id in self.model_load_costs
    
    # ── Scaling Decisions ──
    
    def should_load_model(self, model_id: str, estimated_memory_gb: float) -> Tuple[bool, str]:
        """
        Determine if a model should be loaded based on cost considerations.
        
        Returns:
            (should_load, reason)
        """
        current_cost = self.get_current_hourly_cost()
        additional_cost = self.calculate_memory_cost(estimated_memory_gb)
        new_total_cost = current_cost + additional_cost
        
        # Check budget constraints
        if new_total_cost > self.config.hourly_budget_usd:
            if self.config.strategy == CostStrategy.BUDGET_CONSTRAINED:
                return False, f"Would exceed budget: ${new_total_cost:.2f} > ${self.config.hourly_budget_usd:.2f}/hour"
        
        # Strategy-specific decisions
        if self.config.strategy == CostStrategy.MINIMIZE_COST:
            return self._should_load_minimize_cost(model_id, estimated_memory_gb, new_total_cost)
        elif self.config.strategy == CostStrategy.BALANCE_COST_LATENCY:
            return self._should_load_balance(model_id, estimated_memory_gb, new_total_cost)
        elif self.config.strategy == CostStrategy.LATENCY_CRITICAL:
            return self._should_load_latency_critical(model_id, estimated_memory_gb, new_total_cost)
        elif self.config.strategy == CostStrategy.BUDGET_CONSTRAINED:
            return self._should_load_budget_constrained(model_id, estimated_memory_gb, new_total_cost)
        
        return True, "Default: load allowed"
    
    def _should_load_minimize_cost(self, model_id: str, memory_gb: float, new_cost: float) -> Tuple[bool, str]:
        """Minimize cost strategy: only load if absolutely necessary"""
        # Only load if under very low cost threshold
        if new_cost > self.config.cost_threshold_for_warming:
            return False, f"Cost too high: ${new_cost:.2f} > ${self.config.cost_threshold_for_warming:.2f}"
        
        # Check if model has recent traffic (justification for cost)
        if not self._has_recent_traffic(model_id):
            return False, "No recent traffic to justify cost"
        
        return True, "Cost within threshold and has traffic"
    
    def _should_load_balance(self, model_id: str, memory_gb: float, new_cost: float) -> Tuple[bool, str]:
        """Balance cost and latency strategy"""
        # Allow higher cost during peak hours
        cost_multiplier = 1.5 if self._is_peak_hour() else 1.0
        adjusted_threshold = self.config.cost_threshold_for_warming * cost_multiplier
        
        if new_cost > adjusted_threshold:
            return False, f"Cost exceeds adjusted threshold: ${new_cost:.2f} > ${adjusted_threshold:.2f}"
        
        # Consider traffic patterns
        traffic_score = self._get_traffic_score(model_id)
        if traffic_score < 0.3:  # Low traffic
            return False, f"Low traffic score: {traffic_score:.2f}"
        
        return True, "Cost and latency balanced"
    
    def _should_load_latency_critical(self, model_id: str, memory_gb: float, new_cost: float) -> Tuple[bool, str]:
        """Latency-critical strategy: prioritize performance"""
        # Only block if severely over budget
        if new_cost > self.config.hourly_budget_usd * 2:
            return False, f"Severely over budget: ${new_cost:.2f} > ${self.config.hourly_budget_usd * 2:.2f}"
        
        return True, "Latency critical: prioritize performance"
    
    def _should_load_budget_constrained(self, model_id: str, memory_gb: float, new_cost: float) -> Tuple[bool, str]:
        """Budget-constrained strategy: strict budget limits"""
        if new_cost > self.config.hourly_budget_usd:
            return False, f"Exceeds budget: ${new_cost:.2f} > ${self.config.hourly_budget_usd:.2f}"
        
        # Only load if under 80% budget utilization
        if self.metrics.budget_utilization_percent > 80:
            return False, f"Budget utilization too high: {self.metrics.budget_utilization_percent:.1f}%"
        
        return True, "Within budget constraints"
    
    def _has_recent_traffic(self, model_id: str, hours: int = 1) -> bool:
        """Check if model has recent traffic"""
        # This would integrate with traffic data from warm pool manager
        # For now, assume no recent traffic
        return False
    
    def _get_traffic_score(self, model_id: str) -> float:
        """Get traffic score for a model (0.0 to 1.0)"""
        # This would integrate with traffic data
        # For now, return a default score
        return 0.5
    
    # ── Model Cost Tracking ──
    
    def register_model_load(self, model_id: str, memory_gb: float, load_time_s: float):
        """Register a model load for cost tracking"""
        # Track memory usage
        self.model_memory_usage[model_id] = memory_gb
        self.current_memory_usage_gb += memory_gb
        
        # Calculate and track load cost
        load_cost = self.calculate_cold_start_penalty(model_id)
        self.model_load_costs[model_id] = load_cost
        
        logger.info(f"Model {model_id} loaded: {memory_gb:.1f}GB, ${load_cost:.3f} penalty")
    
    def register_model_eviction(self, model_id: str):
        """Register a model eviction for cost tracking"""
        if model_id in self.model_memory_usage:
            memory_gb = self.model_memory_usage[model_id]
            self.current_memory_usage_gb -= memory_gb
            del self.model_memory_usage[model_id]
        
        # Remove load cost after some time
        if model_id in self.model_load_costs:
            del self.model_load_costs[model_id]
        
        logger.info(f"Model {model_id} evicted, freed memory")
    
    def calculate_cost_savings(self, baseline_cost_per_hour: float) -> float:
        """Calculate cost savings compared to baseline"""
        current_cost = self.get_current_hourly_cost()
        savings = max(0, baseline_cost_per_hour - current_cost)
        
        self.metrics.cost_savings_usd = savings
        return savings
    
    # ── Cost Prediction ──
    
    def predict_hourly_cost(self, hours_ahead: int = 1) -> float:
        """Predict cost for future hours based on patterns"""
        if not self.config.enable_cost_prediction:
            return self.get_current_hourly_cost()
        
        # Get historical costs for same hour
        target_hour = (datetime.now() + timedelta(hours=hours_ahead)).hour
        historical_costs = []
        
        for timestamp, cost in self.hourly_cost_history:
            if timestamp.hour == target_hour:
                historical_costs.append(cost)
        
        if not historical_costs:
            # No historical data, use current cost with peak adjustment
            current_cost = self.get_current_hourly_cost()
            return current_cost * (1.5 if target_hour in self.config.peak_hours else 1.0)
        
        # Return average historical cost
        return sum(historical_costs) / len(historical_costs)
    
    def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for cost optimization"""
        recommendations = []
        current_cost = self.get_current_hourly_cost()
        
        # Budget optimization
        if self.metrics.budget_utilization_percent > 90:
            recommendations.append({
                'type': 'budget',
                'priority': 'high',
                'message': f"Budget utilization at {self.metrics.budget_utilization_percent:.1f}%",
                'action': 'Consider evicting low-priority models',
                'potential_savings': f"${current_cost * 0.3:.2f}/hour"
            })
        
        # Memory optimization
        if self.current_memory_usage_gb > 60:  # Assuming 80GB total
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'message': f"High memory usage: {self.current_memory_usage_gb:.1f}GB",
                'action': 'Enable aggressive eviction policy',
                'potential_savings': f"${self.calculate_memory_cost(10):.2f}/hour"
            })
        
        # Peak hour optimization
        if self._is_peak_hour() and current_cost > self.config.hourly_budget_usd * 0.8:
            recommendations.append({
                'type': 'peak_hour',
                'priority': 'medium',
                'message': 'High cost during peak hours',
                'action': 'Switch to cost-optimized scaling during peaks',
                'potential_savings': f"${current_cost * 0.2:.2f}/hour"
            })
        
        return recommendations
    
    # ── Background Tasks ──
    
    async def _cost_monitor(self):
        """Background task to monitor and record costs"""
        while self._running:
            try:
                await self._record_hourly_cost()
                await asyncio.sleep(300)  # Record every 5 minutes
            except Exception as e:
                logger.error(f"Cost monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _record_hourly_cost(self):
        """Record current hourly cost"""
        now = datetime.now()
        current_cost = self.get_current_hourly_cost()
        
        # Record in history
        self.hourly_cost_history.append((now, current_cost))
        
        # Keep only last 24 hours
        cutoff = now - timedelta(hours=24)
        self.hourly_cost_history = [
            (timestamp, cost) for timestamp, cost in self.hourly_cost_history
            if timestamp > cutoff
        ]
        
        # Update hourly costs list for metrics
        self.metrics.hourly_costs = [cost for _, cost in self.hourly_cost_history[-24:]]
        
        # Update total cost (cumulative)
        self.metrics.total_cost_usd += current_cost * (5/60)  # 5 minutes worth
        
        # Save data
        self._save_metrics()
        self._save_cost_history()
        
        logger.debug(f"Cost recorded: ${current_cost:.3f}/hour, "
                    f"budget: {self.metrics.budget_utilization_percent:.1f}%")
    
    # ── Metrics and Reporting ──
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive cost status"""
        return {
            'current_hourly_cost_usd': self.metrics.current_hourly_cost_usd,
            'budget_utilization_percent': self.metrics.budget_utilization_percent,
            'memory_cost_usd': self.metrics.memory_cost_usd,
            'cold_start_penalty_usd': self.metrics.cold_start_penalty_usd,
            'total_cost_usd': self.metrics.total_cost_usd,
            'cost_savings_usd': self.metrics.cost_savings_usd,
            'current_memory_usage_gb': self.current_memory_usage_gb,
            'active_models': len(self.model_memory_usage),
            'strategy': self.config.strategy.value,
            'is_peak_hour': self._is_peak_hour(),
            'predicted_cost_1h': self.predict_hourly_cost(1),
            'predicted_cost_2h': self.predict_hourly_cost(2),
        }
    
    def get_model_cost_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cost details for a specific model"""
        if model_id not in self.model_memory_usage:
            return None
        
        memory_gb = self.model_memory_usage[model_id]
        hourly_cost = self.calculate_memory_cost(memory_gb)
        
        return {
            'model_id': model_id,
            'memory_gb': memory_gb,
            'hourly_cost_usd': hourly_cost,
            'cold_start_penalty_usd': self.model_load_costs.get(model_id, 0.0),
            'total_cost_today': hourly_cost * 24,  # Rough estimate
            'cost_rank': self._get_model_cost_rank(model_id),
        }
    
    def _get_model_cost_rank(self, model_id: str) -> int:
        """Get cost rank of model (1 = most expensive)"""
        model_costs = [
            (mid, self.calculate_memory_cost(mem_gb))
            for mid, mem_gb in self.model_memory_usage.items()
        ]
        model_costs.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (mid, _) in enumerate(model_costs, 1):
            if mid == model_id:
                return rank
        
        return len(model_costs) + 1
    
    def _load_metrics(self):
        """Load metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = CostMetrics(**data)
            except Exception as e:
                logger.warning(f"Failed to load cost metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'total_cost_usd': self.metrics.total_cost_usd,
            'current_hourly_cost_usd': self.metrics.current_hourly_cost_usd,
            'memory_cost_usd': self.metrics.memory_cost_usd,
            'cold_start_penalty_usd': self.metrics.cold_start_penalty_usd,
            'cost_savings_usd': self.metrics.cost_savings_usd,
            'budget_utilization_percent': self.metrics.budget_utilization_percent,
            'hourly_costs': self.metrics.hourly_costs,
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_cost_history(self):
        """Load cost history from disk"""
        if self.cost_history_file.exists():
            try:
                with open(self.cost_history_file, 'r') as f:
                    data = json.load(f)
                    self.hourly_cost_history = [
                        (datetime.fromisoformat(ts), cost)
                        for ts, cost in data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load cost history: {e}")
                self.hourly_cost_history = []
    
    def _save_cost_history(self):
        """Save cost history to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = [
            (timestamp.isoformat(), cost)
            for timestamp, cost in self.hourly_cost_history
        ]
        
        with open(self.cost_history_file, 'w') as f:
            json.dump(data, f, indent=2)
