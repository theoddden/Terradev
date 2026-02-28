#!/usr/bin/env python3
"""
Model Orchestrator - Multi-model scheduling and memory management for bursty workloads

Solves the Reddit post problems:
1. Intelligent warm pool management (not just scale-to-zero)
2. Memory-aware model scheduling and eviction
3. Billing-optimized scaling policies
4. Deterministic restore/eviction under memory pressure
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
from pathlib import Path

from .warm_pool_manager import WarmPoolManager, WarmPoolConfig, WarmStrategy
from .cost_scaler import CostScaler, CostConfig, CostStrategy

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model state in the orchestrator"""
    COLD = "cold"           # Not loaded, no memory usage
    WARMING = "warming"     # Currently loading/warming up
    WARM = "warm"           # Loaded in memory, ready to serve
    SERVING = "serving"     # Actively serving requests
    EVICTING = "evicting"   # Being evicted from memory
    ERROR = "error"         # Failed to load/serve


class ScalingPolicy(Enum):
    """Scaling strategies for bursty workloads"""
    BILLING_OPTIMIZED = "billing_optimized"  # Minimize idle VRAM costs
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize cold starts
    HYBRID = "hybrid"                         # Balance cost vs latency


@dataclass
class MoEProfile:
    """MoE-specific model characteristics for memory and parallelism planning"""
    is_moe: bool = False
    total_params_b: float = 0.0       # Total parameters (e.g., 744B for GLM-5)
    active_params_b: float = 0.0      # Active parameters per token (e.g., 40B)
    num_experts: int = 0              # Total expert count
    experts_per_token: int = 0        # Top-K routing
    total_weight_gb: float = 0.0      # Full model weight footprint
    active_memory_gb: float = 0.0     # Memory during inference (active experts + KV cache)
    shared_weight_gb: float = 0.0     # Non-expert weights (always replicated across ranks)
    tp_size: int = 1                  # Tensor parallelism degree per EP rank
    dp_size: int = 1                  # Data parallelism / EP degree
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    enable_dbo: bool = False


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_id: str
    load_time_s: float = 0.0          # Time to load from cold to warm
    warmup_time_s: float = 0.0        # Time to warm up CUDA kernels
    memory_gb: float = 0.0            # VRAM usage when loaded (per-GPU for EP models)
    last_request: Optional[datetime] = None
    requests_per_hour: float = 0.0    # Recent request rate
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0


@dataclass
class ModelInstance:
    """A model instance managed by the orchestrator"""
    model_id: str
    model_path: str
    framework: str  # pytorch, tensorflow, vllm, sglang
    gpu_id: int
    state: ModelState = ModelState.COLD
    metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(model_id=""))
    moe_profile: MoEProfile = field(default_factory=MoEProfile)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    load_start_time: Optional[datetime] = None
    warm_until: Optional[datetime] = None  # When to evict (billing optimization)
    priority: int = 0  # Higher = less likely to evict
    tags: Set[str] = field(default_factory=set)


class ModelOrchestrator:
    """
    Multi-model orchestrator for bursty inference workloads.
    
    Key features:
    1. Memory-aware scheduling with deterministic eviction
    2. Intelligent warm pool management
    3. Billing-optimized scaling policies
    4. Fast restore times for cached models
    """
    
    # Configuration thresholds
    DEFAULT_MEMORY_THRESHOLD_GB = 60.0  # Leave 20GB on 80GB H100
    WARM_POOL_TARGET_SIZE = 10  # Keep top N models warm
    COLD_START_TIMEOUT_S = 300  # 5 minutes max cold start time
    IDLE_EVICTION_MINUTES = 15  # Evict models idle for 15 minutes
    
    def __init__(self, gpu_id: int = 0, total_memory_gb: float = 80.0,
                 scaling_policy: ScalingPolicy = ScalingPolicy.BILLING_OPTIMIZED,
                 config_dir: Optional[Path] = None):
        self.gpu_id = gpu_id
        self.total_memory_gb = total_memory_gb
        self.available_memory_gb = total_memory_gb
        self.scaling_policy = scaling_policy
        self.config_dir = config_dir or Path.home() / '.terradev'
        
        # Model management
        self.models: Dict[str, ModelInstance] = {}
        self.loading_queue: List[str] = []  # Models waiting to load
        self.serving_queue: List[str] = []   # Models actively serving
        
        # Memory tracking
        self.used_memory_gb = 0.0
        self.memory_threshold_gb = total_memory_gb * 0.8  # Use 80% max
        
        # Initialize warm pool manager
        warm_config = WarmPoolConfig(
            max_warm_models=15 if scaling_policy == ScalingPolicy.LATENCY_OPTIMIZED else 10,
            min_warm_models=5 if scaling_policy == ScalingPolicy.LATENCY_OPTIMIZED else 3,
            strategy=WarmStrategy.TRAFFIC_BASED,
            enable_predictive_warming=True
        )
        self.warm_pool_manager = WarmPoolManager(warm_config, config_dir)
        
        # Initialize cost scaler
        cost_config = CostConfig(
            hourly_budget_usd=15.0,
            strategy=CostStrategy.BALANCE_COST_LATENCY,
            enable_cost_prediction=True
        )
        self.cost_scaler = CostScaler(cost_config, config_dir)
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._eviction_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.metrics_file = self.config_dir / 'model_metrics.json'
        self.metrics: Dict[str, ModelMetrics] = {}
        self._load_metrics()
    
    async def start(self):
        """Start the orchestrator background tasks"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_memory_usage())
        self._eviction_task = asyncio.create_task(self._eviction_manager())
        
        # Start warm pool manager and cost scaler
        await self.warm_pool_manager.start()
        await self.cost_scaler.start()
        
        logger.info(f"Model Orchestrator started on GPU {self.gpu_id}")
    
    async def stop(self):
        """Stop the orchestrator and cleanup"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._eviction_task:
            self._eviction_task.cancel()
        
        # Stop warm pool manager and cost scaler
        await self.warm_pool_manager.stop()
        await self.cost_scaler.stop()
        
        # Evict all models
        for model_id in list(self.models.keys()):
            await self.evict_model(model_id)
        
        logger.info("Model Orchestrator stopped")
    
    # ── Model Registration ──
    
    def register_model(self, model_id: str, model_path: str, framework: str,
                      priority: int = 0, tags: Optional[Set[str]] = None,
                      moe_profile: Optional[MoEProfile] = None) -> ModelInstance:
        """Register a new model for orchestration.

        For MoE models, pass an ``MoEProfile`` to enable EP-aware memory
        estimation and parallelism planning.
        """
        instance = ModelInstance(
            model_id=model_id,
            model_path=model_path,
            framework=framework,
            gpu_id=self.gpu_id,
            priority=priority,
            tags=tags or set(),
            moe_profile=moe_profile or MoEProfile(),
        )
        
        # Load historical metrics if available
        if model_id in self.metrics:
            instance.metrics = self.metrics[model_id]
        
        self.models[model_id] = instance
        
        # Register with warm pool manager and cost scaler
        self.warm_pool_manager.register_model(model_id, priority)
        
        self._save_metrics()
        return instance
    
    # ── Model Loading and Warming ──
    
    async def load_model(self, model_id: str, force: bool = False) -> bool:
        """Load a model into GPU memory with intelligent scheduling"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not registered")
            return False
        
        instance = self.models[model_id]
        
        if instance.state == ModelState.WARM:
            logger.info(f"Model {model_id} already warm")
            return True
        
        if instance.state == ModelState.WARMING:
            logger.info(f"Model {model_id} already warming")
            return True
        
        # Check cost considerations
        estimated_memory = instance.metrics.memory_gb or self._estimate_model_memory(instance)
        should_load, cost_reason = self.cost_scaler.should_load_model(model_id, estimated_memory)
        
        if not should_load and not force:
            logger.info(f"Cost optimization prevented loading {model_id}: {cost_reason}")
            return False
        
        # Check warm pool manager recommendation
        if not force and not self.warm_pool_manager.should_warm_model(model_id):
            logger.info(f"Warm pool manager recommends not warming {model_id}")
            return False
        
        # Check memory availability
        if not force and not self._can_load_model(instance):
            # Try to evict lower priority models
            await self._make_room_for_model(instance)
        
        # Start loading process
        instance.state = ModelState.WARMING
        instance.load_start_time = datetime.now()
        
        # Mark as warming in warm pool manager
        self.warm_pool_manager.mark_model_warming(model_id)
        
        try:
            success = await self._actually_load_model(instance)
            if success:
                await self._warmup_model(instance)
                instance.state = ModelState.WARM
                instance.last_accessed = datetime.now()
                self.used_memory_gb += instance.metrics.memory_gb
                
                # Register with cost scaler
                self.cost_scaler.register_model_load(model_id, instance.metrics.memory_gb, instance.metrics.load_time_s)
                
                # Mark as warm in warm pool manager
                self.warm_pool_manager.mark_model_warm(model_id, instance.metrics.load_time_s)
                
                logger.info(f"Model {model_id} loaded and warmed ({instance.metrics.memory_gb:.1f}GB)")
                return True
            else:
                instance.state = ModelState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            instance.state = ModelState.ERROR
            return False
    
    async def _actually_load_model(self, instance: ModelInstance) -> bool:
        """Actually load the model framework-specific"""
        start_time = time.time()
        
        try:
            if instance.framework == "pytorch":
                success = await self._load_pytorch_model(instance)
            elif instance.framework == "vllm":
                success = await self._load_vllm_model(instance)
            elif instance.framework == "sglang":
                success = await self._load_sglang_model(instance)
            else:
                logger.error(f"Unsupported framework: {instance.framework}")
                return False
            
            if success:
                instance.metrics.load_time_s = time.time() - start_time
                # Estimate memory usage (could be more precise with framework APIs)
                instance.metrics.memory_gb = await self._measure_model_memory(instance)
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading model {instance.model_id}: {e}")
            return False
    
    async def _load_pytorch_model(self, instance: ModelInstance) -> bool:
        """Load a PyTorch model"""
        # This would integrate with PyTorch model loading
        # For now, simulate the process
        await asyncio.sleep(2)  # Simulate loading time
        return True
    
    async def _load_vllm_model(self, instance: ModelInstance) -> bool:
        """Load a VLLM model for efficient inference"""
        # VLLM integration for LLM serving
        await asyncio.sleep(3)  # Simulate VLLM loading
        return True
    
    async def _load_sglang_model(self, instance: ModelInstance) -> bool:
        """Load an SGLang model"""
        # SGLang integration for structured generation
        await asyncio.sleep(2.5)  # Simulate SGLang loading
        return True
    
    async def _warmup_model(self, instance: ModelInstance):
        """Warm up CUDA kernels and framework"""
        start_time = time.time()
        
        try:
            # Framework-specific warmup
            if instance.framework == "pytorch":
                await self._warmup_pytorch(instance)
            elif instance.framework == "vllm":
                await self._warmup_vllm(instance)
            elif instance.framework == "sglang":
                await self._warmup_sglang(instance)
            
            instance.metrics.warmup_time_s = time.time() - start_time
            logger.info(f"Model {instance.model_id} warmed up in {instance.metrics.warmup_time_s:.1f}s")
            
        except Exception as e:
            logger.warning(f"Warmup failed for {instance.model_id}: {e}")
    
    async def _warmup_pytorch(self, instance: ModelInstance):
        """Warm up PyTorch CUDA kernels"""
        # Dummy forward pass to warm up CUDA
        await asyncio.sleep(1)  # Simulate warmup
    
    async def _warmup_vllm(self, instance: ModelInstance):
        """Warm up VLLM inference engine"""
        await asyncio.sleep(0.5)  # VLLM typically warms up fast
    
    async def _warmup_sglang(self, instance: ModelInstance):
        """Warm up SGLang runtime"""
        await asyncio.sleep(0.8)  # SGLang warmup
    
    # ── Memory Management ──
    
    def _can_load_model(self, instance: ModelInstance) -> bool:
        """Check if we have enough memory to load a model"""
        estimated_memory = instance.metrics.memory_gb or self._estimate_model_memory(instance)
        return (self.used_memory_gb + estimated_memory) <= self.memory_threshold_gb
    
    def _estimate_model_memory(self, instance: ModelInstance) -> float:
        """Estimate per-GPU memory usage, MoE-aware.

        For MoE models with Expert Parallelism (EP), memory per GPU is:
          weight_per_gpu = (expert_weights / dp_size) + shared_weights
          active_memory  = weight_per_gpu + kv_cache_per_gpu

        This is fundamentally different from dense models where weight_per_gpu
        = total_weights / tp_size.  The distinction matters because MoE total
        weight >> active memory (e.g., 744B total but only 40B active).
        """
        moe = instance.moe_profile

        if moe.is_moe and moe.total_weight_gb > 0:
            dp = max(moe.dp_size, 1)
            tp = max(moe.tp_size, 1)

            if moe.enable_expert_parallel:
                # EP: experts distributed, shared weights replicated
                expert_weight_gb = moe.total_weight_gb - moe.shared_weight_gb
                weight_per_gpu = (expert_weight_gb / dp) + (moe.shared_weight_gb / tp)
            else:
                # Pure TP: all weights sharded equally
                weight_per_gpu = moe.total_weight_gb / (tp * dp)

            # KV cache overhead estimate: ~15% of GPU mem for 32k context
            kv_cache_gb = self.total_memory_gb * 0.15
            per_gpu = weight_per_gpu + kv_cache_gb
            logger.debug(
                f"MoE memory estimate for {instance.model_id}: "
                f"weight/gpu={weight_per_gpu:.1f}GB, kv={kv_cache_gb:.1f}GB, "
                f"total/gpu={per_gpu:.1f}GB (EP={moe.enable_expert_parallel}, "
                f"TP={tp}, DP={dp})"
            )
            return per_gpu

        # Dense model fallback
        if instance.framework == "vllm":
            return 15.0
        elif instance.framework == "sglang":
            return 12.0
        else:
            return 10.0
    
    async def _make_room_for_model(self, target_instance: ModelInstance):
        """Evict models to make room for a new model"""
        candidates = [
            (model_id, instance) for model_id, instance in self.models.items()
            if instance.state == ModelState.WARM and instance.priority < target_instance.priority
        ]
        
        # Sort by priority (lowest first) and last accessed time
        candidates.sort(key=lambda x: (x[1].priority, x[1].last_accessed))
        
        freed_memory = 0.0
        needed_memory = self._estimate_model_memory(target_instance)
        
        for model_id, instance in candidates:
            if freed_memory >= needed_memory:
                break
            
            await self.evict_model(model_id)
            freed_memory += instance.metrics.memory_gb
            logger.info(f"Evicted {model_id} to make room for {target_instance.model_id}")
    
    async def evict_model(self, model_id: str) -> bool:
        """Evict a model from GPU memory"""
        if model_id not in self.models:
            return False
        
        instance = self.models[model_id]
        
        if instance.state != ModelState.WARM:
            return True  # Already not in memory
        
        instance.state = ModelState.EVICTING
        
        try:
            # Framework-specific eviction
            if instance.framework == "pytorch":
                await self._evict_pytorch_model(instance)
            elif instance.framework == "vllm":
                await self._evict_vllm_model(instance)
            elif instance.framework == "sglang":
                await self._evict_sglang_model(instance)
            
            instance.state = ModelState.COLD
            self.used_memory_gb -= instance.metrics.memory_gb
            instance.metrics.memory_gb = 0.0
            
            # Notify warm pool manager and cost scaler
            self.warm_pool_manager.mark_model_evicted(model_id)
            self.cost_scaler.register_model_eviction(model_id)
            
            logger.info(f"Evicted model {model_id} ({self.used_memory_gb:.1f}GB used)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to evict model {model_id}: {e}")
            instance.state = ModelState.ERROR
            return False
    
    async def _evict_pytorch_model(self, instance: ModelInstance):
        """Evict PyTorch model from GPU"""
        # PyTorch model cleanup
        await asyncio.sleep(0.5)  # Simulate cleanup time
    
    async def _evict_vllm_model(self, instance: ModelInstance):
        """Evict VLLM model"""
        await asyncio.sleep(0.3)  # VLLM cleanup
    
    async def _evict_sglang_model(self, instance: ModelInstance):
        """Evict SGLang model"""
        await asyncio.sleep(0.4)  # SGLang cleanup
    
    # ── Background Tasks ──
    
    async def _monitor_memory_usage(self):
        """Monitor GPU memory usage in background"""
        while self._running:
            try:
                # Get actual GPU memory usage (would use nvidia-ml-py in production)
                actual_used = await self._get_gpu_memory_usage()
                if actual_used:
                    self.used_memory_gb = actual_used
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get actual GPU memory usage"""
        # In production, use nvidia-ml-py or similar
        # For now, return our tracked usage
        return self.used_memory_gb
    
    async def _eviction_manager(self):
        """Background task to manage model eviction based on policy"""
        while self._running:
            try:
                await self._apply_scaling_policy()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Eviction manager error: {e}")
                await asyncio.sleep(60)
    
    async def _apply_scaling_policy(self):
        """Apply the configured scaling policy"""
        if self.scaling_policy == ScalingPolicy.BILLING_OPTIMIZED:
            await self._billing_optimized_eviction()
        elif self.scaling_policy == ScalingPolicy.LATENCY_OPTIMIZED:
            await self._latency_optimized_warming()
        elif self.scaling_policy == ScalingPolicy.HYBRID:
            await self._hybrid_policy()
    
    async def _billing_optimized_eviction(self):
        """Evict idle models to minimize costs"""
        now = datetime.now()
        idle_threshold = timedelta(minutes=self.IDLE_EVICTION_MINUTES)
        
        for model_id, instance in self.models.items():
            if (instance.state == ModelState.WARM and 
                now - instance.last_accessed > idle_threshold):
                
                logger.info(f"Billing optimization: evicting idle model {model_id}")
                await self.evict_model(model_id)
    
    async def _latency_optimized_warming(self):
        """Keep popular models warm to minimize latency"""
        # Sort models by recent request frequency
        warm_models = [
            (model_id, instance) for model_id, instance in self.models.items()
            if instance.state == ModelState.WARM
        ]
        
        warm_models.sort(key=lambda x: x[1].metrics.requests_per_hour, reverse=True)
        
        # Keep top N models warm
        if len(warm_models) > self.WARM_POOL_TARGET_SIZE:
            for model_id, _ in warm_models[self.WARM_POOL_TARGET_SIZE:]:
                logger.info(f"Latency optimization: evicting less popular model {model_id}")
                await self.evict_model(model_id)
    
    async def _hybrid_policy(self):
        """Balance cost and latency"""
        # Keep models with high request rate or recent access warm
        now = datetime.now()
        recent_threshold = timedelta(minutes=5)
        high_traffic_threshold = 10.0  # requests per hour
        
        for model_id, instance in self.models.items():
            if instance.state != ModelState.WARM:
                continue
            
            should_evict = (
                instance.metrics.requests_per_hour < high_traffic_threshold and
                now - instance.last_accessed > recent_threshold
            )
            
            if should_evict:
                logger.info(f"Hybrid policy: evicting model {model_id}")
                await self.evict_model(model_id)
    
    # ── Request Handling ──
    
    async def handle_request(self, model_id: str) -> Tuple[bool, float]:
        """Handle an inference request for a model"""
        if model_id not in self.models:
            return False, 0.0
        
        instance = self.models[model_id]
        start_time = time.time()
        was_warm = instance.state == ModelState.WARM
        
        # Ensure model is loaded
        if instance.state != ModelState.WARM:
            load_success = await self.load_model(model_id)
            if not load_success:
                return False, 0.0
        
        # Update metrics
        instance.last_accessed = datetime.now()
        instance.metrics.requests_per_hour = self._calculate_request_rate(instance)
        
        # Handle the actual inference (framework-specific)
        try:
            if instance.framework == "pytorch":
                result = await self._infer_pytorch(instance)
            elif instance.framework == "vllm":
                result = await self._infer_vllm(instance)
            elif instance.framework == "sglang":
                result = await self._infer_sglang(instance)
            else:
                return False, 0.0
            
            latency_ms = (time.time() - start_time) * 1000
            instance.metrics.avg_latency_ms = latency_ms
            
            # Record request in warm pool manager and cost scaler
            self.warm_pool_manager.record_request(model_id, latency_ms, was_warm)
            
            return True, latency_ms
            
        except Exception as e:
            logger.error(f"Inference error for {model_id}: {e}")
            instance.metrics.error_rate = min(instance.metrics.error_rate + 0.1, 1.0)
            return False, 0.0
    
    async def _infer_pytorch(self, instance: ModelInstance) -> bool:
        """Run PyTorch inference"""
        await asyncio.sleep(0.1)  # Simulate inference time
        return True
    
    async def _infer_vllm(self, instance: ModelInstance) -> bool:
        """Run VLLM inference"""
        await asyncio.sleep(0.05)  # VLLM is optimized for speed
        return True
    
    async def _infer_sglang(self, instance: ModelInstance) -> bool:
        """Run SGLang inference"""
        await asyncio.sleep(0.08)  # SGLang inference
        return True
    
    def _calculate_request_rate(self, instance: ModelInstance) -> float:
        """Calculate requests per hour for a model"""
        # Simple exponential moving average
        if not instance.last_request:
            instance.metrics.requests_per_hour = 1.0
        else:
            # More sophisticated calculation would use actual request history
            instance.metrics.requests_per_hour *= 0.9  # Decay factor
        return instance.metrics.requests_per_hour
    
    async def _measure_model_memory(self, instance: ModelInstance) -> float:
        """Measure actual memory usage of a loaded model"""
        # In production, use framework-specific APIs
        # For now, return our estimate
        return self._estimate_model_memory(instance)
    
    # ── Metrics and Persistence ──
    
    def _load_metrics(self):
        """Load historical metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = {
                        model_id: ModelMetrics(model_id=model_id, **metrics)
                        for model_id, metrics in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
                self.metrics = {}
        else:
            self.metrics = {}
    
    def _save_metrics(self):
        """Save metrics to disk"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            model_id: {
                'load_time_s': metrics.load_time_s,
                'warmup_time_s': metrics.warmup_time_s,
                'memory_gb': metrics.memory_gb,
                'requests_per_hour': metrics.requests_per_hour,
                'avg_latency_ms': metrics.avg_latency_ms,
                'error_rate': metrics.error_rate,
            }
            for model_id, metrics in self.metrics.items()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    # ── Status and Reporting ──
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        models_by_state = {}
        for state in ModelState:
            models_by_state[state.value] = [
                model_id for model_id, instance in self.models.items()
                if instance.state == state
            ]
        
        warm_models = [
            instance for instance in self.models.values()
            if instance.state == ModelState.WARM
        ]
        
        return {
            'gpu_id': self.gpu_id,
            'total_memory_gb': self.total_memory_gb,
            'used_memory_gb': self.used_memory_gb,
            'available_memory_gb': self.available_memory_gb,
            'scaling_policy': self.scaling_policy.value,
            'total_models': len(self.models),
            'models_by_state': models_by_state,
            'warm_models_count': len(warm_models),
            'warm_models_memory_gb': sum(m.metrics.memory_gb for m in warm_models),
            'memory_utilization_percent': (self.used_memory_gb / self.total_memory_gb) * 100,
        }
    
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        if model_id not in self.models:
            return None
        
        instance = self.models[model_id]
        return {
            'model_id': instance.model_id,
            'model_path': instance.model_path,
            'framework': instance.framework,
            'state': instance.state.value,
            'priority': instance.priority,
            'tags': list(instance.tags),
            'metrics': {
                'load_time_s': instance.metrics.load_time_s,
                'warmup_time_s': instance.metrics.warmup_time_s,
                'memory_gb': instance.metrics.memory_gb,
                'requests_per_hour': instance.metrics.requests_per_hour,
                'avg_latency_ms': instance.metrics.avg_latency_ms,
                'error_rate': instance.metrics.error_rate,
            },
            'created_at': instance.created_at.isoformat(),
            'last_accessed': instance.last_accessed.isoformat(),
        }
