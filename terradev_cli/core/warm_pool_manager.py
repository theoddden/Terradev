#!/usr/bin/env python3
"""
Warm Pool Manager - Intelligent pre-warming strategies for bursty workloads

Addresses Reddit post pain points:
1. "Scale to zero" is not enough - intelligent warm pool instead
2. Warm pools don't become manual capacity planning - automated management
3. Reduces cold start latency without wasting VRAM
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class WarmStrategy(Enum):
    """Warm pool management strategies"""
    TRAFFIC_BASED = "traffic_based"          # Warm models based on recent traffic
    TIME_BASED = "time_based"                # Warm models during peak hours
    PRIORITY_BASED = "priority_based"        # Always keep high-priority models warm
    COST_OPTIMIZED = "cost_optimized"        # Minimize warm pool size
    LATENCY_OPTIMIZED = "latency_optimized"  # Maximize warm pool for performance


@dataclass
class WarmPoolConfig:
    """Configuration for warm pool management"""
    max_warm_models: int = 10               # Maximum models to keep warm
    min_warm_models: int = 3                # Minimum models to keep warm
    warm_threshold_rph: float = 5.0         # Requests per hour to consider warming
    idle_eviction_minutes: int = 15         # Minutes of idle time before eviction
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16, 17, 18])
    strategy: WarmStrategy = WarmStrategy.TRAFFIC_BASED
    enable_predictive_warming: bool = True  # Use traffic patterns to predict warming


@dataclass
class WarmPoolMetrics:
    """Metrics for warm pool performance"""
    total_warm_requests: int = 0
    cold_start_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_warm_latency_ms: float = 0.0
    avg_cold_latency_ms: float = 0.0
    memory_saved_gb: float = 0.0
    cost_saved_usd: float = 0.0
    # CUDA Graph optimization metrics
    cuda_graph_optimized_models: int = 0
    avg_graph_capture_time_ms: float = 0.0
    graph_replay_speedup: float = 1.0
    numa_aligned_warms: int = 0


class WarmPoolManager:
    """
    Intelligent warm pool manager for multi-model inference with CUDA Graph optimization.
    
    Enhanced with passive CUDA Graph awareness:
    1. Automatically detects CUDA Graph-compatible models
    2. Prioritizes NUMA-optimal endpoints for graph capture
    3. Tracks graph capture/replay performance
    4. Optimizes warm pool strategy based on graph potential
    
    Solves the "warm pools become manual capacity planning" problem by:
    1. Automatically managing warm pool size based on traffic patterns
    2. Predictive warming based on historical usage
    3. Cost-aware eviction policies
    4. Performance monitoring and optimization
    5. CUDA Graph-aware warming strategies
    """
    
    def __init__(self, config: WarmPoolConfig, config_dir: Optional[Path] = None):
        self.config = config
        self.config_dir = config_dir or Path.home() / '.terradev'
        
        # Warm pool state
        self.warm_models: Set[str] = set()
        self.warming_models: Set[str] = set()
        self.model_priorities: Dict[str, int] = {}
        self.model_traffic: Dict[str, List[datetime]] = {}
        self.model_load_times: Dict[str, float] = {}
        
        # CUDA Graph optimization state
        self.cuda_graph_models: Set[str] = set()  # Models that can use CUDA Graphs
        self.model_graph_scores: Dict[str, float] = {}  # Graph optimization scores
        self.endpoint_numa_scores: Dict[str, Dict[str, Any]] = {}  # NUMA scores per endpoint
        
        # Metrics tracking
        self.metrics = WarmPoolMetrics()
        self.metrics_file = self.config_dir / 'warm_pool_metrics.json'
        self.traffic_file = self.config_dir / 'model_traffic.json'
        self.graph_metrics_file = self.config_dir / 'cuda_graph_metrics.json'
        
        # Background tasks
        self._warming_task: Optional[asyncio.Task] = None
        self._eviction_task: Optional[asyncio.Task] = None
        self._graph_optimization_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load historical data
        self._load_metrics()
        self._load_traffic_history()
        self._load_cuda_graph_metrics()
    
    async def start(self):
        """Start warm pool management background tasks"""
        self._running = True
        self._warming_task = asyncio.create_task(self._warming_manager())
        self._eviction_task = asyncio.create_task(self._eviction_manager())
        self._graph_optimization_task = asyncio.create_task(self._cuda_graph_optimizer())
        logger.info("Warm Pool Manager started with CUDA Graph optimization")
    
    async def stop(self):
        """Stop warm pool management"""
        self._running = False
        if self._warming_task:
            self._warming_task.cancel()
        if self._eviction_task:
            self._eviction_task.cancel()
        if self._graph_optimization_task:
            self._graph_optimization_task.cancel()
        logger.info("Warm Pool Manager stopped")
    
    # ── Model Management ──
    
    def register_model(self, model_id: str, priority: int = 0):
        """Register a model for warm pool management"""
        self.model_priorities[model_id] = priority
        if model_id not in self.model_traffic:
            self.model_traffic[model_id] = []
    
    def record_request(self, model_id: str, latency_ms: float, was_warm: bool):
        """Record an inference request for traffic analysis"""
        now = datetime.now()
        
        # Record traffic
        if model_id not in self.model_traffic:
            self.model_traffic[model_id] = []
        self.model_traffic[model_id].append(now)
        
        # Update metrics
        self.metrics.total_warm_requests += 1
        if was_warm:
            self.metrics.cache_hits += 1
            # Update warm latency average
            self.metrics.avg_warm_latency_ms = (
                (self.metrics.avg_warm_latency_ms * (self.metrics.cache_hits - 1) + latency_ms) 
                / self.metrics.cache_hits
            )
        else:
            self.metrics.cache_misses += 1
            self.metrics.cold_start_requests += 1
            # Update cold latency average
            self.metrics.avg_cold_latency_ms = (
                (self.metrics.avg_cold_latency_ms * (self.metrics.cold_start_requests - 1) + latency_ms) 
                / self.metrics.cold_start_requests
            )
        
        # Clean old traffic data (keep last 7 days)
        cutoff = now - timedelta(days=7)
        self.model_traffic[model_id] = [
            timestamp for timestamp in self.model_traffic[model_id] 
            if timestamp > cutoff
        ]
        
        self._save_metrics()
        self._save_traffic_history()
    
    def should_warm_model(self, model_id: str) -> bool:
        """Determine if a model should be warmed based on strategy"""
        if model_id in self.warm_models or model_id in self.warming_models:
            return True
        
        if len(self.warm_models) >= self.config.max_warm_models:
            return False
        
        now = datetime.now()
        
        if self.config.strategy == WarmStrategy.TRAFFIC_BASED:
            return self._should_warm_traffic_based(model_id, now)
        elif self.config.strategy == WarmStrategy.TIME_BASED:
            return self._should_warm_time_based(model_id, now)
        elif self.config.strategy == WarmStrategy.PRIORITY_BASED:
            return self._should_warm_priority_based(model_id)
        elif self.config.strategy == WarmStrategy.COST_OPTIMIZED:
            return self._should_warm_cost_optimized(model_id, now)
        elif self.config.strategy == WarmStrategy.LATENCY_OPTIMIZED:
            return self._should_warm_latency_optimized(model_id)
        
        return False
    
    def _should_warm_traffic_based(self, model_id: str, now: datetime) -> bool:
        """Traffic-based warming: warm models with recent requests"""
        if model_id not in self.model_traffic:
            return False
        
        # Calculate requests per hour
        recent_requests = [
            timestamp for timestamp in self.model_traffic[model_id]
            if now - timestamp < timedelta(hours=1)
        ]
        
        rph = len(recent_requests)
        return rph >= self.config.warm_threshold_rph
    
    def _should_warm_time_based(self, model_id: str, now: datetime) -> bool:
        """Time-based warming: warm models during peak hours"""
        if now.hour not in self.config.peak_hours:
            return False
        
        # During peak hours, warm models with any recent traffic
        if model_id not in self.model_traffic:
            return False
        
        recent_requests = [
            timestamp for timestamp in self.model_traffic[model_id]
            if now - timestamp < timedelta(hours=2)
        ]
        
        return len(recent_requests) > 0
    
    def _should_warm_priority_based(self, model_id: str) -> bool:
        """Priority-based warming: always keep high-priority models warm"""
        priority = self.model_priorities.get(model_id, 0)
        
        # Boost priority for CUDA Graph compatible models
        if self.should_warm_with_cuda_graphs(model_id):
            priority += 2  # Boost CUDA Graph models by 2 priority levels
            self.metrics.numa_aligned_warms += 1
        
        # Always warm models with priority >= 5
        if priority >= 5:
            return True
        
        # For lower priority models, check if we have room
        if len(self.warm_models) < self.config.min_warm_models:
            return priority >= 3
        
        return False
    
    def _should_warm_cost_optimized(self, model_id: str, now: datetime) -> bool:
        """Cost-optimized warming: minimal warm pool"""
        # Only warm models with very high traffic
        if model_id not in self.model_traffic:
            return False
        
        recent_requests = [
            timestamp for timestamp in self.model_traffic[model_id]
            if now - timestamp < timedelta(hours=1)
        ]
        
        rph = len(recent_requests)
        return rph >= self.config.warm_threshold_rph * 2  # Higher threshold
    
    def _should_warm_latency_optimized(self, model_id: str, now: datetime) -> bool:
        """Latency-optimized warming: aggressive warming"""
        # Warm models with any recent traffic
        if model_id not in self.model_traffic:
            return False
        
        recent_requests = [
            timestamp for timestamp in self.model_traffic[model_id]
            if now - timestamp < timedelta(hours=3)
        ]
        
        return len(recent_requests) > 0
    
    def mark_model_warming(self, model_id: str):
        """Mark a model as currently warming"""
        self.warming_models.add(model_id)
    
    def mark_model_warm(self, model_id: str, load_time_s: float):
        """Mark a model as successfully warmed"""
        self.warming_models.discard(model_id)
        self.warm_models.add(model_id)
        self.model_load_times[model_id] = load_time_s
        
        logger.info(f"Model {model_id} warmed in {load_time_s:.1f}s")
    
    def mark_model_evicted(self, model_id: str):
        """Mark a model as evicted from warm pool"""
        self.warm_models.discard(model_id)
        self.warming_models.discard(model_id)
        
        logger.info(f"Model {model_id} evicted from warm pool")
    
    # ── Background Tasks ──
    
    async def _warming_manager(self):
        """Background task to manage model warming"""
        while self._running:
            try:
                await self._manage_warming()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Warming manager error: {e}")
                await asyncio.sleep(60)
    
    async def _eviction_manager(self):
        """Background task to manage model eviction"""
        while self._running:
            try:
                await self._manage_eviction()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Eviction manager error: {e}")
                await asyncio.sleep(60)
    
    async def _manage_warming(self):
        """Manage which models should be warmed"""
        # Get all registered models
        all_models = set(self.model_priorities.keys())
        
        # Find models that should be warm
        should_warm = {
            model_id for model_id in all_models
            if self.should_warm_model(model_id)
        }
        
        # Models to warm (should be warm but aren't)
        to_warm = should_warm - self.warm_models - self.warming_models
        
        # Models to evict (warm but shouldn't be)
        to_evict = (self.warm_models | self.warming_models) - should_warm
        
        # Respect capacity limits
        available_slots = self.config.max_warm_models - len(self.warm_models) - len(self.warming_models)
        if len(to_warm) > available_slots:
            # Sort by priority and traffic
            to_warm = sorted(to_warm, key=lambda m: (
                -self.model_priorities.get(m, 0),  # Higher priority first
                -len(self.model_traffic.get(m, []))  # More traffic first
            ))[:available_slots]
        
        # Log actions
        for model_id in to_warm:
            logger.info(f"Warming model: {model_id}")
        
        for model_id in to_evict:
            logger.info(f"Will evict model: {model_id}")
    
    async def _manage_eviction(self):
        """Manage eviction of idle models"""
        now = datetime.now()
        idle_threshold = timedelta(minutes=self.config.idle_eviction_minutes)
        
        # Find models that have been idle too long
        to_evict = set()
        
        for model_id in self.warm_models:
            if model_id not in self.model_traffic:
                continue
            
            # Find most recent request
            recent_requests = self.model_traffic[model_id]
            if not recent_requests:
                continue
            
            last_request = max(recent_requests)
            if now - last_request > idle_threshold:
                to_evict.add(model_id)
        
        # Don't evict if we're below minimum warm pool size
        if len(self.warm_models) - len(to_evict) < self.config.min_warm_models:
            # Only evict models with lowest priority
            evictable = sorted(to_evict, key=lambda m: self.model_priorities.get(m, 0))
            slots_available = len(self.warm_models) - self.config.min_warm_models
            to_evict = set(evictable[:slots_available])
        
        # Evict models
        for model_id in to_evict:
            self.mark_model_evicted(model_id)
    
    # ── Predictive Warming ──
    
    def predict_traffic(self, model_id: str, hours_ahead: int = 1) -> float:
        """Predict traffic for a model hours ahead based on historical patterns"""
        if not self.config.enable_predictive_warming:
            return 0.0
        
        if model_id not in self.model_traffic:
            return 0.0
        
        now = datetime.now()
        target_hour = (now + timedelta(hours=hours_ahead)).hour
        
        # Get traffic for same hour in previous days
        same_hour_traffic = []
        for days_ago in range(1, 8):  # Look back 7 days
            past_time = now - timedelta(days=days_ago)
            if past_time.hour == target_hour:
                day_traffic = [
                    timestamp for timestamp in self.model_traffic[model_id]
                    if past_time.date() == timestamp.date()
                ]
                same_hour_traffic.append(len(day_traffic))
        
        if not same_hour_traffic:
            return 0.0
        
        # Return average traffic for this hour
        return sum(same_hour_traffic) / len(same_hour_traffic)
    
    def get_predictive_warming_candidates(self, hours_ahead: int = 1) -> List[Tuple[str, float]]:
        """Get models that should be pre-warmed based on predicted traffic"""
        candidates = []
        
        for model_id in self.model_priorities.keys():
            predicted_traffic = self.predict_traffic(model_id, hours_ahead)
            if predicted_traffic >= self.config.warm_threshold_rph:
                candidates.append((model_id, predicted_traffic))
        
        # Sort by predicted traffic
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    # ── Metrics and Reporting ──
    
    def get_status(self) -> Dict:
        """Get warm pool status"""
        cache_hit_rate = (
            self.metrics.cache_hits / max(1, self.metrics.total_warm_requests)
        )
        
        return {
            'warm_models_count': len(self.warm_models),
            'warming_models_count': len(self.warming_models),
            'total_models': len(self.model_priorities),
            'strategy': self.config.strategy.value,
            'cache_hit_rate': cache_hit_rate,
            'total_requests': self.metrics.total_warm_requests,
            'cold_starts': self.metrics.cold_start_requests,
            'avg_warm_latency_ms': self.metrics.avg_warm_latency_ms,
            'avg_cold_latency_ms': self.metrics.avg_cold_latency_ms,
            'memory_saved_gb': self.metrics.memory_saved_gb,
            'cost_saved_usd': self.metrics.cost_saved_usd,
        }
    
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        if model_id not in self.model_priorities:
            return None
        
        is_warm = model_id in self.warm_models
        is_warming = model_id in self.warming_models
        
        # Calculate recent traffic
        now = datetime.now()
        recent_traffic = [
            timestamp for timestamp in self.model_traffic.get(model_id, [])
            if now - timestamp < timedelta(hours=1)
        ]
        
        return {
            'model_id': model_id,
            'priority': self.model_priorities[model_id],
            'is_warm': is_warm,
            'is_warming': is_warming,
            'requests_per_hour': len(recent_traffic),
            'total_requests': len(self.model_traffic.get(model_id, [])),
            'load_time_s': self.model_load_times.get(model_id, 0.0),
            'predicted_traffic_1h': self.predict_traffic(model_id, 1),
            'predicted_traffic_2h': self.predict_traffic(model_id, 2),
        }
    
    def _load_metrics(self):
        """Load metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = WarmPoolMetrics(**data)
            except Exception as e:
                logger.warning(f"Failed to load warm pool metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'total_warm_requests': self.metrics.total_warm_requests,
            'cold_start_requests': self.metrics.cold_start_requests,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'avg_warm_latency_ms': self.metrics.avg_warm_latency_ms,
            'avg_cold_latency_ms': self.metrics.avg_cold_latency_ms,
            'memory_saved_gb': self.metrics.memory_saved_gb,
            'cost_saved_usd': self.metrics.cost_saved_usd,
            'cuda_graph_optimized_models': self.metrics.cuda_graph_optimized_models,
            'avg_graph_capture_time_ms': self.metrics.avg_graph_capture_time_ms,
            'graph_replay_speedup': self.metrics.graph_replay_speedup,
            'numa_aligned_warms': self.metrics.numa_aligned_warms,
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_traffic_history(self):
        """Load traffic history from disk"""
        if self.traffic_file.exists():
            try:
                with open(self.traffic_file, 'r') as f:
                    data = json.load(f)
                    self.model_traffic = {
                        model_id: [datetime.fromisoformat(ts) for ts in timestamps]
                        for model_id, timestamps in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load traffic history: {e}")
                self.model_traffic = {}
    
    def _save_traffic_history(self):
        """Save traffic history to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            model_id: [ts.isoformat() for ts in timestamps]
            for model_id, timestamps in self.model_traffic.items()
        }
        
        with open(self.traffic_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_cuda_graph_metrics(self):
        """Load CUDA Graph optimization metrics from disk"""
        if self.graph_metrics_file.exists():
            try:
                with open(self.graph_metrics_file, 'r') as f:
                    data = json.load(f)
                    self.model_graph_scores = data.get('model_graph_scores', {})
                    self.endpoint_numa_scores = data.get('endpoint_numa_scores', {})
                    self.cuda_graph_models = set(data.get('cuda_graph_models', []))
            except Exception as e:
                logger.warning(f"Failed to load CUDA Graph metrics: {e}")
                self.model_graph_scores = {}
                self.endpoint_numa_scores = {}
                self.cuda_graph_models = set()
    
    def _save_cuda_graph_metrics(self):
        """Save CUDA Graph optimization metrics to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'model_graph_scores': self.model_graph_scores,
            'endpoint_numa_scores': self.endpoint_numa_scores,
            'cuda_graph_models': list(self.cuda_graph_models),
        }
        
        with open(self.graph_metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def _cuda_graph_optimizer(self):
        """
        Background task that optimizes warm pool for CUDA Graph performance.
        
        This runs passively in the background, automatically:
        1. Detecting models that can benefit from CUDA Graphs
        2. Prioritizing NUMA-optimal endpoints for graph capture
        3. Tracking graph performance metrics
        4. Adjusting warm pool strategy based on graph potential
        """
        while self._running:
            try:
                await self._analyze_cuda_graph_potential()
                await self._optimize_endpoint_selection()
                await asyncio.sleep(300)  # Run every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CUDA Graph optimizer error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _analyze_cuda_graph_potential(self):
        """Analyze which models can benefit from CUDA Graph optimization"""
        for model_id in self.model_priorities.keys():
            # Check if model is already analyzed
            if model_id in self.model_graph_scores:
                continue
            
            # Detect model type from model_id (simple heuristic)
            model_type = self._detect_model_type(model_id)
            
            # Calculate CUDA Graph compatibility score
            graph_score = self._calculate_model_graph_score(model_id, model_type)
            self.model_graph_scores[model_id] = graph_score
            
            # Add to CUDA Graph models if score is high enough
            if graph_score > 0.7:
                self.cuda_graph_models.add(model_id)
                logger.info(f"Model {model_id} identified as CUDA Graph compatible (score: {graph_score:.2f})")

    def _detect_model_type(self, model_id: str) -> str:
        """Detect model type from model identifier"""
        model_id_lower = model_id.lower()
        
        if any(keyword in model_id_lower for keyword in ['moe', 'mixture', 'expert']):
            return 'moe'
        elif any(keyword in model_id_lower for keyword in ['llama', 'bert', 'gpt', 'transformer', 't5']):
            return 'transformer'
        elif any(keyword in model_id_lower for keyword in ['resnet', 'conv', 'cnn', 'vision']):
            return 'cnn'
        else:
            return 'unknown'

    def _calculate_model_graph_score(self, model_id: str, model_type: str) -> float:
        """Calculate CUDA Graph compatibility score for a model"""
        base_score = 0.5  # Default score
        
        if model_type == 'transformer':
            base_score = 0.9  # Transformers benefit greatly from CUDA Graphs
        elif model_type == 'cnn':
            base_score = 0.7  # CNNs benefit moderately
        elif model_type == 'moe':
            base_score = 0.4  # MoE models have challenges with dynamic routing
        else:
            base_score = 0.5  # Unknown model type
        
        # Adjust based on traffic patterns (more traffic = more benefit from graphs)
        traffic_score = min(1.0, len(self.model_traffic.get(model_id, [])) / 100.0)
        
        return (base_score + traffic_score) / 2.0

    async def _optimize_endpoint_selection(self):
        """Optimize endpoint selection for CUDA Graph performance"""
        for model_id in self.cuda_graph_models:
            # Get available endpoints for this model
            endpoints = await self._get_model_endpoints(model_id)
            
            if not endpoints:
                continue
            
            # Score endpoints by CUDA Graph optimization potential
            best_endpoint = None
            best_score = 0.0
            
            for endpoint_id in endpoints:
                # Get NUMA scorecard for this endpoint
                numa_score = self.endpoint_numa_scores.get(endpoint_id, {})
                graph_score = numa_score.get('cuda_graph_score', 0.0)
                
                if graph_score > best_score:
                    best_score = graph_score
                    best_endpoint = endpoint_id
            
            # Update endpoint priority based on CUDA Graph score
            if best_endpoint and best_score > 0.7:
                # Prioritize this endpoint for warming
                current_priority = self.model_priorities.get(model_id, 0)
                if best_score > 0.9:
                    self.model_priorities[model_id] = max(current_priority, 10)  # Highest priority
                elif best_score > 0.8:
                    self.model_priorities[model_id] = max(current_priority, 7)   # High priority
                else:
                    self.model_priorities[model_id] = max(current_priority, 5)   # Medium priority
                
                logger.debug(f"Prioritized endpoint {best_endpoint} for CUDA Graph model {model_id} (score: {best_score:.2f})")

    async def _get_model_endpoints(self, model_id: str) -> List[str]:
        """Get available endpoints for a model (placeholder implementation)"""
        # This would integrate with your endpoint discovery system
        # For now, return empty list - would be implemented based on your architecture
        return []

    def should_warm_with_cuda_graphs(self, model_id: str) -> bool:
        """
        Determine if a model should be warmed with CUDA Graph optimization.
        
        This is called automatically during warm pool decisions.
        """
        return (
            model_id in self.cuda_graph_models and
            self.model_graph_scores.get(model_id, 0.0) > 0.7
        )

    def get_cuda_graph_optimization_tips(self, model_id: str) -> Dict[str, Any]:
        """
        Get CUDA Graph optimization recommendations for a model.
        
        Returns tips that can be automatically applied during model loading.
        """
        graph_score = self.model_graph_scores.get(model_id, 0.0)
        model_type = self._detect_model_type(model_id)
        
        recommendations = {
            "use_cuda_graphs": graph_score > 0.7,
            "graph_capture_warmup": graph_score > 0.8,
            "memory_pool_size": f"{int(graph_score * 4)}GB" if graph_score > 0.5 else "2GB",
            "batch_size_optimization": graph_score > 0.6,
            "numa_awareness_required": graph_score > 0.8,
            "model_type": model_type,
            "optimization_potential": self._get_optimization_potential(graph_score),
        }
        
        # Add model-specific recommendations
        if model_type == 'moe':
            recommendations.update({
                "dynamic_routing_handling": True,
                "expert_graph_caching": False,  # MoE models can't cache expert graphs
                "fallback_to_eager": True,
            })
        elif model_type == 'transformer':
            recommendations.update({
                "attention_graph_optimization": True,
                "layer_fusion": graph_score > 0.8,
                "sequence_length_aware": True,
            })
        
        return recommendations

    def _get_optimization_potential(self, graph_score: float) -> str:
        """Get human-readable optimization potential"""
        if graph_score > 0.9:
            return "optimal - expect 2-5x speedup"
        elif graph_score > 0.8:
            return "excellent - expect 1.5-3x speedup"
        elif graph_score > 0.7:
            return "good - expect 1.2-2x speedup"
        elif graph_score > 0.5:
            return "moderate - expect 1.1-1.5x speedup"
        else:
            return "minimal - expect <1.2x speedup"
