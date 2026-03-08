#!/usr/bin/env python3
"""
MLA-Aware VRAM Estimator - Accurate memory estimation for DeepSeek V3/R1 and Kimi K2

CRITICAL FIXES v4.1.0:
- Multi-head Latent Attention (MLA) detection and KV cache optimization
- 5-13x KV cache reduction for MLA architectures
- Prevents over-provisioning of GPUs for DeepSeek/Kimi models
- Architecture-aware memory calculation for standard vs MLA models

Standard transformers use Multi-Head Attention:
KV cache per token = 2 × num_heads × head_dim × num_layers × bytes_per_param
For 70B model at BF16: ~5-8GB for 4K context

MLA architectures (DeepSeek V3/R1, Kimi K2) compress KV cache:
KV cache is 5-13x smaller through low-rank latent space compression
DeepSeek-V3 at 32K context ≈ Llama-3 70B at 4K context KV cache
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Attention architecture types"""
    STANDARD_MHA = "standard_mha"          # Multi-Head Attention (standard)
    MULTI_HEAD_LATENT = "mla"            # Multi-head Latent Attention (DeepSeek/Kimi)
    FLASH_ATTENTION = "flash_attention"    # Flash Attention optimized
    SPARSE_ATTENTION = "sparse_attention" # Sparse attention patterns


@dataclass
class ModelArchitecture:
    """Model architecture specification for VRAM estimation"""
    model_id: str
    total_params_b: float                 # Total parameters in billions
    num_layers: int                       # Number of transformer layers
    num_heads: int                        # Number of attention heads
    head_dim: int                         # Dimension per attention head
    hidden_size: int                      # Hidden dimension
    intermediate_size: int                 # FFN intermediate size
    attention_type: AttentionType         # Attention architecture
    bytes_per_param: float = 2.0          # BF16=2, FP32=4, FP8=1
    mla_compression_ratio: float = 1.0     # KV cache compression ratio for MLA
    context_window: int = 4096            # Default context window
    max_context: int = 32768              # Maximum supported context
    is_moe: bool = False                   # Mixture of Experts
    num_experts: int = 0                  # Total experts for MoE
    experts_per_token: int = 0            # Active experts per token


@dataclass
class VRAMBreakdown:
    """Detailed VRAM usage breakdown"""
    model_weights_gb: float               # Base model weights
    kv_cache_gb: float                    # KV cache for current context
    activation_cache_gb: float            # Activation cache
    overhead_gb: float                    # Framework overhead
    total_gb: float                       # Total VRAM usage
    per_gpu_gb: float                     # Per-GPU usage (for multi-GPU)
    gpu_count: int                        # Recommended GPU count
    architecture: ModelArchitecture      # Architecture info
    context_tokens: int                  # Current context length
    batch_size: int = 1                   # Batch size


class MLA_VRAMEstimator:
    """MLA-aware VRAM estimator for accurate memory planning"""
    
    # Model registry with MLA architecture detection
    MODEL_REGISTRY = {
        # DeepSeek models with MLA
        "deepseek-v3": ModelArchitecture(
            model_id="deepseek-v3",
            total_params_b=671.0,
            num_layers=61,
            num_heads=64,
            head_dim=128,
            hidden_size=8192,
            intermediate_size=22016,
            attention_type=AttentionType.MULTI_HEAD_LATENT,
            mla_compression_ratio=0.08,  # ~12.5x KV cache reduction
            context_window=4096,
            max_context=32768,
        ),
        "deepseek-v3-lite": ModelArchitecture(
            model_id="deepseek-v3-lite",
            total_params_b=7.0,
            num_layers=28,
            num_heads=32,
            head_dim=128,
            hidden_size=4096,
            intermediate_size=11008,
            attention_type=AttentionType.MULTI_HEAD_LATENT,
            mla_compression_ratio=0.12,  # ~8.3x KV cache reduction
            context_window=4096,
            max_context=16384,
        ),
        "deepseek-r1": ModelArchitecture(
            model_id="deepseek-r1",
            total_params_b=671.0,
            num_layers=61,
            num_heads=64,
            head_dim=128,
            hidden_size=8192,
            intermediate_size=22016,
            attention_type=AttentionType.MULTI_HEAD_LATENT,
            mla_compression_ratio=0.08,
            context_window=4096,
            max_context=32768,
        ),
        "deepseek-coder-v2": ModelArchitecture(
            model_id="deepseek-coder-v2",
            total_params_b=236.0,
            num_layers=61,
            num_heads=48,
            head_dim=128,
            hidden_size=6144,
            intermediate_size=16384,
            attention_type=AttentionType.MULTI_HEAD_LATENT,
            mla_compression_ratio=0.10,
            context_window=4096,
            max_context=16384,
        ),
        
        # Kimi models with MLA
        "kimi-k2": ModelArchitecture(
            model_id="kimi-k2",
            total_params_b=120.0,
            num_layers=48,
            num_heads=40,
            head_dim=128,
            hidden_size=5120,
            intermediate_size=13824,
            attention_type=AttentionType.MULTI_HEAD_LATENT,
            mla_compression_ratio=0.10,
            context_window=4096,
            max_context=32768,
        ),
        
        # Standard transformer models (for comparison)
        "llama-3-70b": ModelArchitecture(
            model_id="llama-3-70b",
            total_params_b=70.0,
            num_layers=80,
            num_heads=64,
            head_dim=128,
            hidden_size=8192,
            intermediate_size=28672,
            attention_type=AttentionType.STANDARD_MHA,
            context_window=8192,
            max_context=8192,
        ),
        "mistral-7b": ModelArchitecture(
            model_id="mistral-7b",
            total_params_b=7.0,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            hidden_size=4096,
            intermediate_size=14336,
            attention_type=AttentionType.STANDARD_MHA,
            context_window=4096,
            max_context=4096,
        ),
        "qwen-72b": ModelArchitecture(
            model_id="qwen-72b",
            total_params_b=72.0,
            num_layers=80,
            num_heads=64,
            head_dim=128,
            hidden_size=8192,
            intermediate_size=29568,
            attention_type=AttentionType.STANDARD_MHA,
            context_window=8192,
            max_context=8192,
        ),
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def estimate_vram(
        self,
        model_id: str,
        context_tokens: int = 4096,
        batch_size: int = 1,
        target_gpu_vram_gb: float = 80.0,
        precision: str = "bf16"
    ) -> VRAMBreakdown:
        """Estimate VRAM usage with MLA-aware calculations"""
        
        # Get model architecture
        arch = self._get_model_architecture(model_id)
        if not arch:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Set precision
        precision_map = {"fp32": 4.0, "bf16": 2.0, "fp8": 1.0}
        arch.bytes_per_param = precision_map.get(precision.lower(), 2.0)
        
        # Calculate model weights
        model_weights_gb = self._calculate_model_weights(arch)
        
        # Calculate KV cache (MLA-aware)
        kv_cache_gb = self._calculate_kv_cache(arch, context_tokens, batch_size)
        
        # Calculate activation cache
        activation_cache_gb = self._calculate_activation_cache(arch, context_tokens, batch_size)
        
        # Framework overhead
        overhead_gb = self._calculate_overhead(arch)
        
        # Total VRAM
        total_gb = model_weights_gb + kv_cache_gb + activation_cache_gb + overhead_gb
        
        # Calculate GPU requirements
        gpu_count = max(1, int(total_gb / target_gpu_vram_gb))
        per_gpu_gb = total_gb / gpu_count
        
        return VRAMBreakdown(
            model_weights_gb=model_weights_gb,
            kv_cache_gb=kv_cache_gb,
            activation_cache_gb=activation_cache_gb,
            overhead_gb=overhead_gb,
            total_gb=total_gb,
            per_gpu_gb=per_gpu_gb,
            gpu_count=gpu_count,
            architecture=arch,
            context_tokens=context_tokens,
            batch_size=batch_size,
        )
    
    def _get_model_architecture(self, model_id: str) -> Optional[ModelArchitecture]:
        """Get model architecture from registry"""
        # Direct lookup
        if model_id in self.MODEL_REGISTRY:
            return self.MODEL_REGISTRY[model_id]
        
        # Fuzzy matching for common patterns
        model_lower = model_id.lower()
        
        # DeepSeek patterns
        if "deepseek" in model_lower:
            if "v3" in model_lower:
                if "lite" in model_lower:
                    return self.MODEL_REGISTRY["deepseek-v3-lite"]
                return self.MODEL_REGISTRY["deepseek-v3"]
            elif "r1" in model_lower:
                return self.MODEL_REGISTRY["deepseek-r1"]
            elif "coder" in model_lower and "v2" in model_lower:
                return self.MODEL_REGISTRY["deepseek-coder-v2"]
        
        # Kimi patterns
        elif "kimi" in model_lower and "k2" in model_lower:
            return self.MODEL_REGISTRY["kimi-k2"]
        
        # Standard transformer patterns
        elif "llama" in model_lower and "70b" in model_lower:
            return self.MODEL_REGISTRY["llama-3-70b"]
        elif "mistral" in model_lower and "7b" in model_lower:
            return self.MODEL_REGISTRY["mistral-7b"]
        elif "qwen" in model_lower and "72b" in model_lower:
            return self.MODEL_REGISTRY["qwen-72b"]
        
        return None
    
    def _calculate_model_weights(self, arch: ModelArchitecture) -> float:
        """Calculate model weights memory usage"""
        if arch.is_moe and arch.num_experts > 0:
            # MoE models: shared weights + expert weights
            shared_ratio = 0.3  # ~30% of weights are shared (embeddings, layernorm, etc.)
            shared_weights_gb = arch.total_params_b * shared_ratio * arch.bytes_per_param
            expert_weights_gb = arch.total_params_b * (1 - shared_ratio) * arch.bytes_per_param
            
            # For EP, only load active experts per GPU
            if arch.experts_per_token > 0:
                active_expert_ratio = arch.experts_per_token / arch.num_experts
                expert_weights_gb *= active_expert_ratio
            
            return shared_weights_gb + expert_weights_gb
        else:
            # Standard dense models
            return arch.total_params_b * arch.bytes_per_param
    
    def _calculate_kv_cache(self, arch: ModelArchitecture, context_tokens: int, batch_size: int) -> float:
        """Calculate KV cache with MLA compression"""
        
        if arch.attention_type == AttentionType.MULTI_HEAD_LATENT:
            # MLA compression: KV cache is compressed into latent space
            # Standard formula: 2 × num_heads × head_dim × num_layers × context_tokens × batch_size × bytes_per_param
            standard_kv_gb = (2 * arch.num_heads * arch.head_dim * arch.num_layers * 
                           context_tokens * batch_size * arch.bytes_per_param) / (1024**3)
            
            # Apply MLA compression ratio
            compressed_kv_gb = standard_kv_gb * arch.mla_compression_ratio
            
            self.logger.debug(f"MLA KV cache: {standard_kv_gb:.2f}GB → {compressed_kv_gb:.2f}GB "
                            f"(compression: {arch.mla_compression_ratio:.2f}x)")
            
            return compressed_kv_gb
        
        else:
            # Standard MHA KV cache
            return (2 * arch.num_heads * arch.head_dim * arch.num_layers * 
                   context_tokens * batch_size * arch.bytes_per_param) / (1024**3)
    
    def _calculate_activation_cache(self, arch: ModelArchitecture, context_tokens: int, batch_size: int) -> float:
        """Calculate activation cache memory"""
        # Rough estimation: ~10% of model weights for standard context
        # Scales with context size for very long contexts
        base_activation_gb = arch.total_params_b * arch.bytes_per_param * 0.1
        
        # Scale factor for long contexts (>16K)
        if context_tokens > 16384:
            scale_factor = min(2.0, context_tokens / 16384)
            base_activation_gb *= scale_factor
        
        return base_activation_gb
    
    def _calculate_overhead(self, arch: ModelArchitecture) -> float:
        """Calculate framework overhead"""
        # Base overhead for CUDA kernels, framework, etc.
        base_overhead = 2.0  # 2GB base overhead
        
        # Additional overhead for MLA (latent space management)
        if arch.attention_type == AttentionType.MULTI_HEAD_LATENT:
            base_overhead += 0.5  # Extra 500MB for MLA operations
        
        # MoE overhead
        if arch.is_moe:
            base_overhead += 1.0  # Extra 1GB for routing logic
        
        return base_overhead
    
    def compare_standard_vs_mla(
        self,
        model_id: str,
        context_tokens: int = 4096,
        batch_size: int = 1,
        target_gpu_vram_gb: float = 80.0
    ) -> Dict[str, Any]:
        """Compare standard vs MLA memory usage for educational purposes"""
        
        arch = self._get_model_architecture(model_id)
        if not arch:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Calculate with actual architecture
        actual_estimate = self.estimate_vram(model_id, context_tokens, batch_size, target_gpu_vram_gb)
        
        # Calculate what it would be with standard MHA
        standard_arch = ModelArchitecture(
            model_id=arch.model_id,
            total_params_b=arch.total_params_b,
            num_layers=arch.num_layers,
            num_heads=arch.num_heads,
            head_dim=arch.head_dim,
            hidden_size=arch.hidden_size,
            intermediate_size=arch.intermediate_size,
            attention_type=AttentionType.STANDARD_MHA,
            bytes_per_param=arch.bytes_per_param,
            context_window=arch.context_window,
            max_context=arch.max_context,
            is_moe=arch.is_moe,
            num_experts=arch.num_experts,
            experts_per_token=arch.experts_per_token,
        )
        
        # Temporarily replace architecture for calculation
        original_arch = arch
        self.MODEL_REGISTRY[model_id] = standard_arch
        
        try:
            standard_estimate = self.estimate_vram(model_id, context_tokens, batch_size, target_gpu_vram_gb)
        finally:
            # Restore original architecture
            self.MODEL_REGISTRY[model_id] = original_arch
        
        # Calculate savings
        kv_cache_savings_gb = standard_estimate.kv_cache_gb - actual_estimate.kv_cache_gb
        total_savings_gb = standard_estimate.total_gb - actual_estimate.total_gb
        gpu_savings = standard_estimate.gpu_count - actual_estimate.gpu_count
        
        return {
            "model_id": model_id,
            "context_tokens": context_tokens,
            "batch_size": batch_size,
            "actual_architecture": arch.attention_type.value,
            "standard_mha_estimate": {
                "total_gb": round(standard_estimate.total_gb, 2),
                "kv_cache_gb": round(standard_estimate.kv_cache_gb, 2),
                "gpu_count": standard_estimate.gpu_count,
                "per_gpu_gb": round(standard_estimate.per_gpu_gb, 2),
            },
            "mla_estimate": {
                "total_gb": round(actual_estimate.total_gb, 2),
                "kv_cache_gb": round(actual_estimate.kv_cache_gb, 2),
                "gpu_count": actual_estimate.gpu_count,
                "per_gpu_gb": round(actual_estimate.per_gpu_gb, 2),
            },
            "savings": {
                "kv_cache_savings_gb": round(kv_cache_savings_gb, 2),
                "kv_cache_compression_ratio": round(standard_estimate.kv_cache_gb / actual_estimate.kv_cache_gb, 2),
                "total_savings_gb": round(total_savings_gb, 2),
                "gpu_savings": gpu_savings,
                "cost_savings_percent": round((total_savings_gb / standard_estimate.total_gb) * 100, 1),
            },
            "recommendation": self._get_mla_recommendation(actual_estimate, standard_estimate),
        }
    
    def _get_mla_recommendation(self, mla_estimate: VRAMBreakdown, standard_estimate: VRAMBreakdown) -> str:
        """Get recommendation based on MLA vs standard comparison"""
        
        if mla_estimate.gpu_count < standard_estimate.gpu_count:
            return (f"MLA reduces GPU requirements from {standard_estimate.gpu_count} to "
                   f"{mla_estimate.gpu_count} GPUs - significant cost savings!")
        elif mla_estimate.total_gb < standard_estimate.total_gb * 0.7:
            return f"MLA provides {round((1 - mla_estimate.total_gb/standard_estimate.total_gb) * 100, 1)}% memory reduction"
        else:
            return "MLA provides modest memory benefits"
    
    def register_model(self, arch: ModelArchitecture) -> None:
        """Register a new model architecture"""
        self.MODEL_REGISTRY[arch.model_id] = arch
        self.logger.info(f"Registered model architecture: {arch.model_id}")
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model IDs"""
        return list(self.MODEL_REGISTRY.keys())
    
    def is_mla_model(self, model_id: str) -> bool:
        """Check if a model uses MLA architecture"""
        arch = self._get_model_architecture(model_id)
        return arch.attention_type == AttentionType.MULTI_HEAD_LATENT if arch else False
