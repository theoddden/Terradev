#!/usr/bin/env python3
"""
Terradev SGLang Service - Complete Production Optimization Stack

Implements workload-specific auto-optimization for SGLang serving with:
- Agentic/Multi-turn chat (LPM, RadixAttention, cache-aware routing)
- High-throughput batch inference (FCFS, CUDA graphs, FP8)
- Low-latency/real-time (EAGLE3, Spec V2, capped concurrency)
- MoE models (DeepEP, TBO/SBO, EPLB, redundant experts)
- PD disaggregated serving (prefill/decode separation)
- Hardware-specific configs (H100/H200, H20, GB200, AMD)
- Structured output/RAG/JSON decoding (xGrammar, FSM)

Auto-applies optimizations based on model architecture, workload type, and hardware detection.
"""

import os
import json
import asyncio
import aiohttp
import subprocess
import tempfile
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """SGLang workload types with specific optimization profiles"""
    AGENTIC_CHAT = "agentic_chat"
    BATCH_INFERENCE = "batch_inference"
    LOW_LATENCY = "low_latency"
    MOE_MODEL = "moe_model"
    PD_DISAGGREGATED = "pd_disaggregated"
    STRUCTURED_OUTPUT = "structured_output"
    RAG_WORKLOAD = "rag_workload"


class SchedulePolicy(Enum):
    """SGLang schedule policies"""
    LPM = "lpm"  # Longest Prefix Match
    FCFS = "fcfs"  # First Come First Served


class AttentionBackend(Enum):
    """SGLang attention backends"""
    FLASHINFER = "flashinfer"
    FA3 = "fa3"
    TRITON = "triton"


class SpeculativeAlgorithm(Enum):
    """Speculative decoding algorithms"""
    EAGLE = "EAGLE"
    MEDUSA = "MEDUSA"
    NONE = None


class DeepEPMode(Enum):
    """DeepEP modes for MoE models"""
    AUTO = "auto"
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"


@dataclass
class SGLangConfig:
    """Complete SGLang configuration with workload-specific optimizations"""
    # Core model settings
    model_path: str
    workload_type: WorkloadType
    schedule_policy: SchedulePolicy = SchedulePolicy.LPM
    attention_backend: AttentionBackend = AttentionBackend.FLASHINFER
    
    # Memory and batching
    mem_fraction_static: float = 0.82
    chunked_prefill_size: int = 16384
    max_running_requests: int = 256
    disable_radix_cache: bool = False
    
    # CUDA graph optimization
    cuda_graph_bs: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    # Speculative decoding
    speculative_algorithm: Optional[SpeculativeAlgorithm] = None
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4
    enable_spec_v2: bool = False
    
    # MoE-specific settings
    tp: int = 1
    ep: int = 1
    dp_size: int = 1
    enable_dp_attention: bool = False
    moe_a2a_backend: str = "deepep"
    moe_runner_backend: str = "deep_gemm"
    deepep_mode: DeepEPMode = DeepEPMode.AUTO
    enable_eplb: bool = False
    ep_num_redundant_experts: int = 0
    enable_two_batch_overlap: bool = False
    enable_single_batch_overlap: bool = False
    
    # PD disaggregation
    disaggregation_mode: Optional[str] = None  # "prefill" or "decode"
    disaggregation_ib_device: Optional[str] = None
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    
    # Hardware optimization
    quantization: Optional[str] = None  # "fp8", "int8", etc.
    kv_cache_dtype: Optional[str] = None  # "fp8_e4m3", etc.
    enable_torch_compile: bool = False
    
    # Structured output
    enable_xgrammar: bool = True
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # Advanced settings
    page_size: int = 1
    enable_dp_lm_head: bool = False
    moe_dense_tp_size: int = 1
    
    # Legacy compatibility
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True
    dashboard_enabled: bool = False
    tracing_enabled: bool = False
    metrics_enabled: bool = False
    deployment_enabled: bool = False
    observability_enabled: bool = False
    
    # Auto-detection fields
    model_name: str = ""
    serving_config: Optional[Dict[str, Any]] = None
    enable_expert_parallel: bool = False
    enable_dbo: bool = False


@dataclass
class HardwareProfile:
    """Hardware-specific optimization profiles"""
    gpu_type: str
    memory_gb: int
    bandwidth_gbps: float
    architecture: str
    num_gpus_per_node: int = 8
    
    # Optimization defaults
    default_mem_fraction: float = 0.82
    default_attention_backend: AttentionBackend = AttentionBackend.FLASHINFER
    supports_fp8: bool = True
    supports_fa3: bool = False
    special_optimizations: List[str] = field(default_factory=list)


class SGLangOptimizer:
    """SGLang workload-specific optimization engine"""
    
    def __init__(self):
        self.hardware_profiles = self._init_hardware_profiles()
        self.model_patterns = self._init_model_patterns()
    
    def _init_hardware_profiles(self) -> Dict[str, HardwareProfile]:
        """Initialize hardware-specific optimization profiles"""
        return {
            "h100": HardwareProfile(
                gpu_type="H100",
                memory_gb=80,
                bandwidth_gbps=2000,
                architecture="hopper",
                default_mem_fraction=0.82,
                supports_fp8=True,
                supports_fa3=False,
                special_optimizations=["flashinfer", "fp8_kv_cache"]
            ),
            "h200": HardwareProfile(
                gpu_type="H200",
                memory_gb=141,
                bandwidth_gbps=4800,
                architecture="hopper",
                default_mem_fraction=0.85,
                supports_fp8=True,
                supports_fa3=False,
                special_optimizations=["flashinfer", "fp8_kv_cache", "high_memory"]
            ),
            "h20": HardwareProfile(
                gpu_type="H20",
                memory_gb=96,
                bandwidth_gbps=900,
                architecture="hopper",
                default_mem_fraction=0.80,
                supports_fp8=True,
                supports_fa3=True,
                special_optimizations=["fa3", "swapab", "moe_optimization"]
            ),
            "gb200": HardwareProfile(
                gpu_type="GB200",
                memory_gb=192,
                bandwidth_gbps=1800,
                architecture="blackwell",
                default_mem_fraction=0.85,
                supports_fp8=True,
                supports_fa3=True,
                special_optimizations=["rack_scale", "nvlink", "moe_dense_tp"]
            ),
            "amd_mi300x": HardwareProfile(
                gpu_type="MI300X",
                memory_gb=192,
                bandwidth_gbps=3500,
                architecture="cdna",
                default_mem_fraction=0.80,
                supports_fp8=False,
                supports_fa3=False,
                special_optimizations=["triton", "rocm_eplb"]
            ),
        }
    
    def _init_model_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model architecture detection patterns"""
        return {
            # MoE models
            "deepseek": {
                "type": "moe",
                "experts": 64,  # DeepSeek V3
                "default_tp": 8,
                "default_ep": 8,
                "workload_type": WorkloadType.MOE_MODEL
            },
            "kimi": {
                "type": "moe",
                "experts": 32,  # Kimi K2
                "default_tp": 4,
                "default_ep": 8,
                "workload_type": WorkloadType.MOE_MODEL
            },
            "qwen_moe": {
                "type": "moe",
                "experts": 16,
                "default_tp": 4,
                "default_ep": 4,
                "workload_type": WorkloadType.MOE_MODEL
            },
            # Dense models
            "llama": {
                "type": "dense",
                "workload_type": WorkloadType.AGENTIC_CHAT
            },
            "mistral": {
                "type": "dense",
                "workload_type": WorkloadType.AGENTIC_CHAT
            },
            "qwen": {
                "type": "dense",
                "workload_type": WorkloadType.AGENTIC_CHAT
            },
            "glm": {
                "type": "dense",
                "workload_type": WorkloadType.AGENTIC_CHAT
            }
        }
    
    def detect_hardware(self) -> HardwareProfile:
        """Detect current hardware profile"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                first_gpu = lines[0].split(',')
                gpu_name = first_gpu[0].strip().lower()
                memory_mb = int(first_gpu[1].strip())
                
                # Determine GPU type
                for hw_key, profile in self.hardware_profiles.items():
                    if hw_key.lower() in gpu_name:
                        # Update memory if detected
                        detected_memory_gb = memory_mb // 1024
                        if detected_memory_gb != profile.memory_gb:
                            profile.memory_gb = detected_memory_gb
                        return profile
                
                # Default to H100 profile if unknown
                logger.warning(f"Unknown GPU {gpu_name}, defaulting to H100 profile")
                return self.hardware_profiles["h100"]
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
        
        # Fallback to H100
        return self.hardware_profiles["h100"]
    
    def detect_model_type(self, model_path: str) -> Tuple[str, Dict[str, Any]]:
        """Detect model architecture from model path"""
        model_path_lower = model_path.lower()
        
        for pattern, config in self.model_patterns.items():
            if pattern in model_path_lower:
                return pattern, config
        
        # Default to dense agentic model
        return "unknown_dense", {"type": "dense", "workload_type": WorkloadType.AGENTIC_CHAT}


class SGLangService:
    """Complete SGLang service management with auto-optimization"""

    def __init__(self, config: Optional[SGLangConfig] = None):
        self.config = config
        self.optimizer = SGLangOptimizer()
        self.active_configs: Dict[str, SGLangConfig] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        if config:
            self.base_url = f"http://{config.host}:{config.port}/v1"
        else:
            self.base_url = ""

    async def __aenter__(self):
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def detect_workload_type(self, model_path: str, user_description: Optional[str] = None) -> WorkloadType:
        """Auto-detect workload type from model and user description"""
        model_type, model_config = self.optimizer.detect_model_type(model_path)
        
        # User description takes precedence
        if user_description:
            desc_lower = user_description.lower()
            if any(term in desc_lower for term in ["agentic", "multi-turn", "chat", "conversation"]):
                return WorkloadType.AGENTIC_CHAT
            elif any(term in desc_lower for term in ["batch", "eval", "dataset", "processing"]):
                return WorkloadType.BATCH_INFERENCE
            elif any(term in desc_lower for term in ["latency", "real-time", "ttft", "tpot"]):
                return WorkloadType.LOW_LATENCY
            elif any(term in desc_lower for term in ["structured", "json", "grammar"]):
                return WorkloadType.STRUCTURED_OUTPUT
            elif any(term in desc_lower for term in ["rag", "retrieval", "document"]):
                return WorkloadType.RAG_WORKLOAD
            elif any(term in desc_lower for term in ["disaggregated", "prefill", "decode", "pd"]):
                return WorkloadType.PD_DISAGGREGATED
        
        # Fall back to model-based detection
        return model_config.get("workload_type", WorkloadType.AGENTIC_CHAT)
    
    def create_optimized_config(self, 
                              model_path: str,
                              workload_type: Optional[WorkloadType] = None,
                              hardware_profile: Optional[HardwareProfile] = None,
                              user_description: Optional[str] = None,
                              **kwargs) -> SGLangConfig:
        """Create optimized SGLang configuration"""
        
        # Detect hardware if not provided
        if hardware_profile is None:
            hardware_profile = self.optimizer.detect_hardware()
        
        # Detect workload type if not provided
        if workload_type is None:
            workload_type = self.detect_workload_type(model_path, user_description)
        
        # Create base config
        config = SGLangConfig(
            model_path=model_path,
            workload_type=workload_type,
            attention_backend=hardware_profile.default_attention_backend,
            mem_fraction_static=hardware_profile.default_mem_fraction,
            **kwargs
        )
        
        # Apply hardware-specific defaults
        if hardware_profile.supports_fp8:
            config.kv_cache_dtype = "fp8_e4m3"
        
        # Apply workload-specific optimizations
        config = self.optimize_for_workload(config, hardware_profile)
        
        # Store for reference
        config_id = f"{model_path}_{workload_type.value}"
        self.active_configs[config_id] = config
        
        return config
    
    def optimize_for_workload(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Apply workload-specific optimizations"""
        
        if config.workload_type == WorkloadType.AGENTIC_CHAT:
            return self._optimize_agentic_chat(config, hardware)
        elif config.workload_type == WorkloadType.BATCH_INFERENCE:
            return self._optimize_batch_inference(config, hardware)
        elif config.workload_type == WorkloadType.LOW_LATENCY:
            return self._optimize_low_latency(config, hardware)
        elif config.workload_type == WorkloadType.MOE_MODEL:
            return self._optimize_moe_model(config, hardware)
        elif config.workload_type == WorkloadType.PD_DISAGGREGATED:
            return self._optimize_pd_disaggregated(config, hardware)
        elif config.workload_type == WorkloadType.STRUCTURED_OUTPUT:
            return self._optimize_structured_output(config, hardware)
        elif config.workload_type == WorkloadType.RAG_WORKLOAD:
            return self._optimize_rag_workload(config, hardware)
        
        return config
    
    def _optimize_agentic_chat(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for agentic/multi-turn chat workloads"""
        # LPM is critical for prefix sharing in agentic workloads
        config.schedule_policy = SchedulePolicy.LPM
        config.mem_fraction_static = 0.82
        config.chunked_prefill_size = 8192
        config.max_running_requests = 256
        config.disable_radix_cache = False  # RadixAttention is key advantage
        
        # Hardware-specific tuning
        config.attention_backend = hardware.default_attention_backend
        if hardware.supports_fp8:
            config.kv_cache_dtype = "fp8_e4m3"
        
        # Cache-aware routing environment
        config.env_vars["SGLANG_CACHE_AWARE_ROUTING"] = "1"
        
        return config
    
    def _optimize_batch_inference(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for high-throughput batch inference"""
        # FCFS for stateless batch - no prefix sharing
        config.schedule_policy = SchedulePolicy.FCFS
        config.mem_fraction_static = 0.85
        config.chunked_prefill_size = 16384
        config.max_running_requests = 512
        config.disable_radix_cache = True  # Stateless batch = radix overhead
        
        # Explicit CUDA graphs for known batch sizes
        config.cuda_graph_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        # FP8 quantization for throughput
        if hardware.supports_fp8:
            config.quantization = "fp8"
            config.kv_cache_dtype = "fp8_e4m3"
        
        # Torch compile for small models
        config.enable_torch_compile = True
        
        return config
    
    def _optimize_low_latency(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for low-latency/real-time inference"""
        # LPM for shared prefixes, but tuned for latency
        config.schedule_policy = SchedulePolicy.LPM
        config.mem_fraction_static = 0.75  # Lower = less contention
        config.chunked_prefill_size = 4096  # Smaller chunks = lower TTFT variance
        config.max_running_requests = 64  # Cap concurrency to protect tail latency
        
        # Small-to-medium CUDA graphs only
        config.cuda_graph_bs = [1, 2, 4, 8, 16, 32]
        
        # EAGLE3 speculative decoding for decode-heavy workloads
        config.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        config.speculative_num_steps = 3
        config.speculative_eagle_topk = 1  # Chain decoding = lower latency
        config.speculative_num_draft_tokens = 4
        config.enable_spec_v2 = True  # Overlap scheduler
        
        # Spec V2 environment variable
        config.env_vars["SGLANG_ENABLE_SPEC_V2"] = "1"
        
        return config
    
    def _optimize_moe_model(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for MoE models (DeepSeek V3, Kimi K2, Qwen MoE)"""
        # Detect specific MoE model
        model_type, model_config = self.optimizer.detect_model_type(config.model_path)
        
        # Set tensor and expert parallelism
        if model_type == "deepseek":
            config.tp = 8
            config.ep = 8
            config.ep_num_redundant_experts = 32  # Pre-replicate top experts
        elif model_type == "kimi":
            config.tp = 4
            config.ep = 8
            config.ep_num_redundant_experts = 16
        else:  # Qwen MoE and others
            config.tp = 4
            config.ep = 4
            config.ep_num_redundant_experts = 8
        
        config.enable_dp_attention = True
        config.moe_a2a_backend = "deepep"
        config.moe_runner_backend = "deep_gemm"
        config.deepep_mode = DeepEPMode.AUTO  # Auto-switch prefill/decode mode
        config.enable_eplb = True
        config.enable_two_batch_overlap = True  # Up to 2x throughput
        config.enable_single_batch_overlap = True  # Complements TBO
        
        # Hardware-specific MoE tuning
        if hardware.gpu_type == "H20":
            # H20 needs MoE→QKV→FP8 stacking
            config.moe_runner_backend = "swapab"
            if hardware.supports_fp8:
                config.quantization = "fp8"
                config.kv_cache_dtype = "fp8_e4m3"
            config.attention_backend = AttentionBackend.FA3
        elif hardware.gpu_type == "GB200":
            # GB200 NVL72 rack-scale
            config.moe_dense_tp_size = 1  # Critical for NVL72
            config.enable_dp_lm_head = True
            config.chunked_prefill_size = 524288
        
        return config
    
    def _optimize_pd_disaggregated(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for PD disaggregated serving"""
        # Set disaggregation mode based on config
        if config.disaggregation_mode == "prefill":
            # Prefill node config
            config.mem_fraction_static = 0.85
            config.chunked_prefill_size = 524288  # 512K
            config.max_running_requests = 8192
            config.disable_radix_cache = True  # Prefill nodes are stateless
            config.deepep_mode = DeepEPMode.NORMAL  # Prefill = always normal
            config.enable_two_batch_overlap = True
            config.enable_eplb = True
            config.ep_num_redundant_experts = 32
            config.page_size = 1
            
            # Prefill-specific environment
            config.env_vars["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "4"
            config.env_vars["SGLANG_TBO_DEBUG"] = "1"
            
        elif config.disaggregation_mode == "decode":
            # Decode node config
            config.mem_fraction_static = 0.82
            config.max_running_requests = 4096
            config.deepep_mode = DeepEPMode.LOW_LATENCY  # Decode = always low_latency
            config.enable_two_batch_overlap = True
            config.enable_eplb = True
        
        # Common PD settings
        config.enable_dp_attention = True
        config.enable_dp_lm_head = True
        config.enable_deepep_moe = True
        
        return config
    
    def _optimize_structured_output(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for structured output/JSON decoding"""
        # LPM for shared system prompts in structured workloads
        config.schedule_policy = SchedulePolicy.LPM
        config.mem_fraction_static = 0.80
        config.chunked_prefill_size = 8192
        
        # xGrammar backend for 10x faster structured output
        config.enable_xgrammar = True
        
        # FSM optimization is automatic with xGrammar
        config.env_vars["SGLANG_XGRAMMAR_ENABLED"] = "1"
        
        return config
    
    def _optimize_rag_workload(self, config: SGLangConfig, hardware: HardwareProfile) -> SGLangConfig:
        """Optimize for RAG workloads"""
        # RAG has high prefix reuse (document embeddings, shared context)
        config.schedule_policy = SchedulePolicy.LPM
        config.mem_fraction_static = 0.80
        config.chunked_prefill_size = 8192
        config.disable_radix_cache = False  # High prefix reuse = radix beneficial
        
        # RadixAttention will handle document corpus reuse
        config.env_vars["SGLANG_RADIX_ATTENTION"] = "1"
        
        return config
    
    def generate_launch_command(self, config: SGLangConfig) -> str:
        """Generate complete SGLang launch command with all optimizations"""
        cmd_parts = ["python", "-m", "sglang.launch_server"]
        
        # Core model settings
        cmd_parts.extend(["--model-path", config.model_path])
        cmd_parts.extend(["--schedule-policy", config.schedule_policy.value])
        cmd_parts.extend(["--mem-fraction-static", str(config.mem_fraction_static)])
        cmd_parts.extend(["--chunked-prefill-size", str(config.chunked_prefill_size)])
        cmd_parts.extend(["--max-running-requests", str(config.max_running_requests)])
        cmd_parts.extend(["--attention-backend", config.attention_backend.value])
        
        # CUDA graphs
        if config.cuda_graph_bs:
            bs_str = " ".join(str(bs) for bs in config.cuda_graph_bs)
            cmd_parts.extend(["--cuda-graph-bs"] + config.cuda_graph_bs)
        
        # Radix cache
        if config.disable_radix_cache:
            cmd_parts.append("--disable-radix-cache")
        
        # Speculative decoding
        if config.speculative_algorithm:
            cmd_parts.extend(["--speculative-algorithm", config.speculative_algorithm.value])
            cmd_parts.extend(["--speculative-num-steps", str(config.speculative_num_steps)])
            cmd_parts.extend(["--speculative-eagle-topk", str(config.speculative_eagle_topk)])
            cmd_parts.extend(["--speculative-num-draft-tokens", str(config.speculative_num_draft_tokens)])
        
        # MoE settings
        if config.tp > 1:
            cmd_parts.extend(["--tp", str(config.tp)])
        if config.ep > 1:
            cmd_parts.extend(["--ep", str(config.ep)])
        if config.dp_size > 1:
            cmd_parts.extend(["--dp-size", str(config.dp_size)])
        if config.enable_dp_attention:
            cmd_parts.append("--enable-dp-attention")
        if config.moe_a2a_backend:
            cmd_parts.extend(["--moe-a2a-backend", config.moe_a2a_backend])
        if config.moe_runner_backend:
            cmd_parts.extend(["--moe-runner-backend", config.moe_runner_backend])
        if config.deepep_mode:
            cmd_parts.extend(["--deepep-mode", config.deepep_mode.value])
        if config.enable_eplb:
            cmd_parts.append("--enable-eplb")
        if config.ep_num_redundant_experts > 0:
            cmd_parts.extend(["--ep-num-redundant-experts", str(config.ep_num_redundant_experts)])
        if config.enable_two_batch_overlap:
            cmd_parts.append("--enable-two-batch-overlap")
        if config.enable_single_batch_overlap:
            cmd_parts.append("--enable-single-batch-overlap")
        
        # PD disaggregation
        if config.disaggregation_mode:
            cmd_parts.extend(["--disaggregation-mode", config.disaggregation_mode])
        if config.disaggregation_ib_device:
            cmd_parts.extend(["--disaggregation-ib-device", config.disaggregation_ib_device])
        if config.dist_init_addr:
            cmd_parts.extend(["--dist-init-addr", config.dist_init_addr])
        if config.nnodes > 1:
            cmd_parts.extend(["--nnodes", str(config.nnodes)])
        
        # Quantization
        if config.quantization:
            cmd_parts.extend(["--quantization", config.quantization])
        if config.kv_cache_dtype:
            cmd_parts.extend(["--kv-cache-dtype", config.kv_cache_dtype])
        
        # Additional optimizations
        if config.enable_torch_compile:
            cmd_parts.append("--enable-torch-compile")
        if config.page_size != 1:
            cmd_parts.extend(["--page-size", str(config.page_size)])
        if config.enable_dp_lm_head:
            cmd_parts.append("--enable-dp-lm-head")
        if config.moe_dense_tp_size != 1:
            cmd_parts.extend(["--moe-dense-tp-size", str(config.moe_dense_tp_size)])
        
        # Build command string
        command = " ".join(cmd_parts)
        
        # Add environment variables
        if config.env_vars:
            env_str = " ".join([f"{k}={v}" for k, v in config.env_vars.items()])
            command = f"{env_str} {command}"
        
        return command
    
    def generate_multi_replica_command(self, config: SGLangConfig, dp_size: int) -> str:
        """Generate cache-aware router command for multi-replica deployments"""
        cmd_parts = ["python", "-m", "sglang_router.launch_server"]
        cmd_parts.extend(["--model-path", config.model_path])
        cmd_parts.extend(["--dp-size", str(dp_size)])
        cmd_parts.extend(["--schedule-policy", config.schedule_policy.value])
        
        return " ".join(cmd_parts)
    
    def validate_config(self, config: SGLangConfig) -> List[str]:
        """Validate SGLang configuration and return warnings/errors"""
        warnings = []
        
        # Hardware compatibility checks
        hardware = self.optimizer.detect_hardware()
        
        if config.quantization == "fp8" and not hardware.supports_fp8:
            warnings.append(f"FP8 quantization not supported on {hardware.gpu_type}")
        
        if config.attention_backend == AttentionBackend.FA3 and not hardware.supports_fa3:
            warnings.append(f"FA3 attention backend not supported on {hardware.gpu_type}")
        
        # Workload-specific checks
        if config.workload_type == WorkloadType.LOW_LATENCY and config.max_running_requests > 128:
            warnings.append("High max-running-requests may impact latency SLA")
        
        if config.workload_type == WorkloadType.MOE_MODEL and config.ep_num_redundant_experts > 32:
            warnings.append("High redundant expert count may exceed memory limits")
        
        # MoE-specific checks
        if config.enable_two_batch_overlap and not config.enable_dp_attention:
            warnings.append("TBO recommended with DP attention enabled")
        
        if config.deepep_mode == DeepEPMode.AUTO and config.disaggregation_mode:
            warnings.append("Consider using specific DeepEP mode for disaggregated serving")
        
        return warnings
    
    def get_optimization_summary(self, config: SGLangConfig) -> Dict[str, Any]:
        """Get human-readable optimization summary"""
        summary = {
            "workload_type": config.workload_type.value,
            "schedule_policy": config.schedule_policy.value,
            "attention_backend": config.attention_backend.value,
            "optimizations_applied": [],
            "performance_expectations": {},
            "hardware_tuned": False
        }
        
        # Detect hardware
        hardware = self.optimizer.detect_hardware()
        summary["hardware_detected"] = f"{hardware.gpu_type} ({hardware.memory_gb}GB)"
        
        # Workload-specific optimizations
        if config.workload_type == WorkloadType.AGENTIC_CHAT:
            summary["optimizations_applied"].extend([
                "LPM schedule policy for prefix sharing",
                "RadixAttention enabled for cache hits",
                "Cache-aware routing environment"
            ])
            summary["performance_expectations"] = {
                "cache_hit_rate": "75-90%",
                "gpu_utilization": "95-98%",
                "throughput_gain": "1.9x with multi-replica"
            }
        
        elif config.workload_type == WorkloadType.BATCH_INFERENCE:
            summary["optimizations_applied"].extend([
                "FCFS schedule for stateless processing",
                "Radix cache disabled for batch efficiency",
                "Explicit CUDA graph batch sizes",
                "FP8 quantization for throughput"
            ])
            summary["performance_expectations"] = {
                "tokens_per_second": "Maximum",
                "memory_efficiency": "High",
                "batch_optimization": "Pre-compiled graphs"
            }
        
        elif config.workload_type == WorkloadType.LOW_LATENCY:
            summary["optimizations_applied"].extend([
                "EAGLE3 speculative decoding",
                "Spec V2 overlap scheduler",
                "Capped concurrency for tail latency",
                "Small-batch CUDA graphs"
            ])
            summary["performance_expectations"] = {
                "ttft_improvement": "30-50%",
                "tpot_improvement": "20-40%",
                "latency_variance": "Low"
            }
        
        elif config.workload_type == WorkloadType.MOE_MODEL:
            summary["optimizations_applied"].extend([
                f"TP={config.tp}, EP={config.ep} configuration",
                "DeepEP auto mode switching",
                "Two-batch overlap enabled",
                f"{config.ep_num_redundant_experts} redundant experts",
                "EPLB load balancing"
            ])
            summary["performance_expectations"] = {
                "throughput_gain": "Up to 2x with TBO",
                "expert_utilization": "High",
                "communication_optimized": "True"
            }
        
        # Hardware-specific optimizations
        if hardware.gpu_type == "H20":
            summary["optimizations_applied"].append("H20 MoE→QKV→FP8 stacking")
            summary["hardware_tuned"] = True
        elif hardware.gpu_type == "GB200":
            summary["optimizations_applied"].append("GB200 rack-scale optimization")
            summary["hardware_tuned"] = True
        
        return summary

    # ── Connection ──

    async def test_connection(self) -> Dict[str, Any]:
        """Test SGLang availability (local install or running server)"""
        try:
            result = subprocess.run(
                ["python3", "-c", "import sglang; print(sglang.__version__)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return {
                    "status": "connected",
                    "sglang_version": version,
                    "model": self.config.model_name,
                    "endpoint": self.base_url,
                    "tp_size": self.config.tp_size,
                    "dp_size": self.config.dp_size,
                    "expert_parallel": self.config.enable_expert_parallel,
                }
            else:
                return {
                    "status": "failed",
                    "error": "SGLang not installed. Run: pip install sglang[all]",
                }
        except FileNotFoundError:
            return {"status": "failed", "error": "python3 not found"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ── SSH helpers ──

    def _build_ssh_command(self, ip: str, user: str, key: Optional[str], script: str) -> str:
        """Build SSH command for remote execution"""
        ssh_base = f"ssh -i {key} {user}@{ip}" if key else f"ssh {user}@{ip}"
        return f'{ssh_base} "{script}"'

    # ── Remote Installation ──

    async def install_on_instance(
        self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install SGLang on a remote instance via SSH"""
        try:
            install_script = """
#!/bin/bash
set -e
pip install --upgrade pip
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
python3 -c "import sglang; print('SGLang', sglang.__version__, 'installed')"
"""
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, install_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                return {
                    "status": "installed",
                    "instance_ip": instance_ip,
                    "provider": "sglang",
                    "output": result.stdout,
                }
            else:
                return {"status": "failed", "error": f"Installation failed: {result.stderr}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to install SGLang: {e}"}

    # ── Server Lifecycle ──

    async def start_server(
        self,
        instance_ip: str,
        ssh_user: str = "root",
        ssh_key: Optional[str] = None,
        additional_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Start SGLang server on a remote instance via systemd"""
        try:
            server_cmd = [
                "python3", "-m", "sglang.launch_server",
                "--model-path", self.config.model_name,
                "--host", self.config.host,
                "--port", str(self.config.port),
                "--tp-size", str(self.config.tp_size),
                "--dp-size", str(self.config.dp_size),
                "--mem-fraction-static", str(self.config.mem_fraction_static),
            ]

            if self.config.trust_remote_code:
                server_cmd.append("--trust-remote-code")
            if self.config.max_model_len:
                server_cmd.extend(["--max-total-tokens", str(self.config.max_model_len)])

            # MoE Expert Parallelism flags
            if self.config.enable_expert_parallel:
                server_cmd.append("--enable-expert-parallel")
            if self.config.enable_eplb:
                server_cmd.append("--enable-eplb")
            if self.config.enable_dbo:
                server_cmd.append("--enable-dbo")

            if additional_args:
                server_cmd.extend(additional_args)

            service_content = f"""[Unit]
Description=SGLang Server for {self.config.model_name}
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart={" ".join(server_cmd)}
Restart=always
RestartSec=10
Environment=PYTHONPATH=/root
Environment=VLLM_USE_DEEP_GEMM=1
Environment=VLLM_ALL2ALL_BACKEND=deepep_low_latency

[Install]
WantedBy=multi-user.target
"""

            setup_script = f"""#!/bin/bash
set -e
echo '{service_content}' > /etc/systemd/system/sglang.service
systemctl daemon-reload
systemctl enable sglang
systemctl start sglang
sleep 10
systemctl status sglang
"""

            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, setup_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {
                    "status": "started",
                    "instance_ip": instance_ip,
                    "provider": "sglang",
                    "model": self.config.model_name,
                    "endpoint": f"http://{instance_ip}:{self.config.port}/v1",
                    "tp_size": self.config.tp_size,
                    "dp_size": self.config.dp_size,
                    "expert_parallel": self.config.enable_expert_parallel,
                    "output": result.stdout,
                }
            else:
                return {"status": "failed", "error": f"Failed to start server: {result.stderr}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to start SGLang server: {e}"}

    async def stop_server(
        self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop SGLang server on a remote instance"""
        try:
            stop_script = """#!/bin/bash
systemctl stop sglang
systemctl disable sglang
rm -f /etc/systemd/system/sglang.service
systemctl daemon-reload
"""
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, stop_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {
                    "status": "stopped",
                    "instance_ip": instance_ip,
                    "provider": "sglang",
                    "output": result.stdout,
                }
            else:
                return {"status": "failed", "error": f"Failed to stop server: {result.stderr}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to stop SGLang server: {e}"}

    # ── Inference API (OpenAI-compatible) ──

    async def test_inference(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Test SGLang inference via OpenAI-compatible completions endpoint"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"{self.base_url}/completions"
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            }
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            async with self.session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "sglang",
                        "model": self.config.model_name,
                        "prompt": prompt,
                        "response": result["choices"][0]["text"],
                        "usage": result.get("usage", {}),
                        "endpoint": self.base_url,
                    }
                else:
                    error_text = await response.text()
                    return {"status": "failed", "error": f"Inference failed: {response.status} - {error_text}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to test inference: {e}"}

    async def test_chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Test SGLang chat completion via OpenAI-compatible endpoint"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            }
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            async with self.session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "sglang",
                        "model": self.config.model_name,
                        "messages": messages,
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "endpoint": self.base_url,
                    }
                else:
                    error_text = await response.text()
                    return {"status": "failed", "error": f"Chat failed: {response.status} - {error_text}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to test chat completion: {e}"}

    # ── Server Info & Metrics ──

    async def get_server_info(self) -> Dict[str, Any]:
        """Get SGLang server info (models endpoint)"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"{self.base_url}/models"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "sglang",
                        "models": result.get("data", []),
                        "endpoint": self.base_url,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    error_text = await response.text()
                    return {"status": "failed", "error": f"Server info failed: {response.status} - {error_text}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to get server info: {e}"}

    async def get_server_metrics(self) -> Dict[str, Any]:
        """Get SGLang server metrics from the /metrics Prometheus endpoint"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            metrics_url = f"http://{self.config.host}:{self.config.port}/metrics"
            async with self.session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    raw = await response.text()
                    # Parse key metrics from Prometheus text format
                    metrics = {}
                    for line in raw.split("\n"):
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            try:
                                metrics[parts[0]] = float(parts[1])
                            except ValueError:
                                pass
                    return {
                        "status": "success",
                        "provider": "sglang",
                        "endpoint": metrics_url,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {"status": "failed", "error": f"Metrics endpoint returned {response.status}"}
        except Exception as e:
            return {"status": "failed", "error": f"Failed to get metrics: {e}"}

    # ── Deployment Script Generation ──

    def get_deployment_script(
        self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None,
    ) -> str:
        """Generate deployment script for SGLang with MoE EP support"""
        ep_flags = ""
        if self.config.enable_expert_parallel:
            ep_flags += " \\\n    --enable-expert-parallel"
        if self.config.enable_eplb:
            ep_flags += " \\\n    --enable-eplb"
        if self.config.enable_dbo:
            ep_flags += " \\\n    --enable-dbo"

        return f"""#!/bin/bash
# SGLang Deployment Script for Terradev
# Target: {instance_ip}

echo "Deploying SGLang for {self.config.model_name}..."

# Install SGLang
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Create systemd service
cat > /etc/systemd/system/sglang.service << 'EOF'
[Unit]
Description=SGLang Server for {self.config.model_name}
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=python3 -m sglang.launch_server \\
    --model-path {self.config.model_name} \\
    --host {self.config.host} \\
    --port {self.config.port} \\
    --tp-size {self.config.tp_size} \\
    --dp-size {self.config.dp_size} \\
    --mem-fraction-static {self.config.mem_fraction_static} \\
    --trust-remote-code{ep_flags}
Restart=always
RestartSec=10
Environment=VLLM_USE_DEEP_GEMM=1
Environment=VLLM_ALL2ALL_BACKEND=deepep_low_latency

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable sglang
systemctl start sglang

echo "SGLang server started on http://{instance_ip}:{self.config.port}/v1"
echo "Test with: curl http://{instance_ip}:{self.config.port}/v1/models"
"""

    def get_supported_models(self) -> List[str]:
        """Get list of supported MoE models"""
        return [
            "zai-org/GLM-5-FP8",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen3.5-397B-A17B",
            "mistralai/Mistral-Large-3-MoE",
            "meta-llama/Llama-4-405B-MoE",
            "mistralai/Mixtral-8x7B-v0.1",
            "mistralai/Mixtral-8x22B-v0.1",
            "meta-llama/Llama-2-7b-hf",
            "Qwen/Qwen-7B",
        ]


def create_sglang_service_from_credentials(credentials: Dict[str, str]) -> SGLangService:
    """Create SGLangService from credential dictionary"""
    config = SGLangConfig(
        model_name=credentials.get("model_name", credentials.get("model_path", "")),
        host=credentials.get("host", "0.0.0.0"),
        port=int(credentials.get("port", "8000")),
        api_key=credentials.get("api_key", ""),
        tp_size=int(credentials.get("tp_size", "1")),
        dp_size=int(credentials.get("dp_size", "8")),
        mem_fraction_static=float(credentials.get("mem_fraction_static", "0.85")),
        max_model_len=int(credentials["max_model_len"]) if credentials.get("max_model_len") else None,
        trust_remote_code=credentials.get("trust_remote_code", "true").lower() == "true",
        enable_expert_parallel=credentials.get("enable_expert_parallel", "false").lower() == "true",
        enable_eplb=credentials.get("enable_eplb", "false").lower() == "true",
        enable_dbo=credentials.get("enable_dbo", "false").lower() == "true",
        model_path=credentials.get("model_path"),
        serving_config=credentials.get("serving_config", {}),
        dashboard_enabled=credentials.get("dashboard_enabled", "false").lower() == "true",
        tracing_enabled=credentials.get("tracing_enabled", "false").lower() == "true",
        metrics_enabled=credentials.get("metrics_enabled", "false").lower() == "true",
        deployment_enabled=credentials.get("deployment_enabled", "false").lower() == "true",
        observability_enabled=credentials.get("observability_enabled", "false").lower() == "true",
    )
    return SGLangService(config)


def get_sglang_setup_instructions() -> str:
    """Get setup instructions for SGLang with MoE EP support"""
    return """
SGLang Setup Instructions (with MoE Expert Parallelism):

1. Install SGLang with FlashInfer:
   pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

2. Configure Terradev with SGLang:
   terradev configure --provider sglang \\
     --model-name zai-org/GLM-5-FP8 \\
     --tp-size 1 --dp-size 8 \\
     --enable-expert-parallel true \\
     --enable-eplb true \\
     --enable-dbo true

3. Deploy to a provisioned instance:
   terradev provision -g H100 -n 8
   terradev ml sglang --start --instance-ip <IP>

4. Test inference:
   terradev ml sglang --test-inference --prompt "Hello, world!"

Required Credentials:
  - model_name: HuggingFace model ID or local path (required)
  - api_key: API key for authentication (optional)
  - tp_size: Tensor parallelism per EP rank (default: 1)
  - dp_size: Data parallelism / EP degree (default: 8)
  - enable_expert_parallel: Enable MoE EP (default: false)
  - enable_eplb: Expert load balancing (default: false)
  - enable_dbo: Dual-batch overlap (default: false)

Serving Endpoints (OpenAI-compatible):
  - Completions: http://<IP>:8000/v1/completions
  - Chat:        http://<IP>:8000/v1/chat/completions
  - Models:      http://<IP>:8000/v1/models
  - Health:      http://<IP>:8000/health
  - Metrics:     http://<IP>:8000/metrics

MoE Expert Parallelism:
  With EP enabled (TP=1, DP=8), experts are distributed across 8 ranks.
  Each rank holds ~32 experts (for 256-expert models like GLM-5).
  EPLB rebalances experts at runtime based on actual token routing.
  DBO overlaps compute with all-to-all communication for throughput.
"""
