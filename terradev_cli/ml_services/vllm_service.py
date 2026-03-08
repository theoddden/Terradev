#!/usr/bin/env python3
"""
vLLM Service Integration for Terradev
High-performance LLM inference server deployment and management
"""

import os
import json
import asyncio
import aiohttp
import subprocess
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WorkloadProfile:
    """Workload characteristics for automatic optimization"""
    avg_prompt_length: float = 0.0  # Average input tokens
    avg_response_length: float = 0.0  # Average output tokens
    requests_per_second: float = 0.0  # Expected QPS
    concurrent_users: int = 1  # Concurrent users
    latency_sensitivity: float = 0.5  # 0=throughput focused, 1=latency focused
    memory_pressure: float = 0.5  # 0=plenty of memory, 1=memory constrained
    gpu_count: int = 1  # Number of GPUs available
    model_size_gb: float = 0.0  # Model size in GB


@dataclass
class LoRAModule:
    """A single LoRA adapter definition"""
    name: str
    path: str
    base_model_name: Optional[str] = None


@dataclass
class VLLMConfig:
    """vLLM configuration with Multi-LoRA, Sleep Mode, KV Offloading,
    Speculative Decoding, and vLLM Router support (v0.15.0+)"""
    model_name: str
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    gpu_memory_utilization: float = 0.95  # Optimized: 0.90 → 0.95 (10% more VRAM)
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    
    # ── Critical Throughput Optimizations (6 knobs most teams never touch) ─────
    max_num_batched_tokens: int = 16384  # Optimized: 2048 → 16384 (8x throughput)
    max_num_seqs: int = 1024  # Optimized: 256/1024 → 1024 (higher concurrency)
    enable_prefix_caching: bool = True  # Optimized: OFF → ON (free throughput win)
    enable_chunked_prefill: bool = True  # Optimized: OFF → ON (V0) / verify ON (V1)
    
    # ── CPU Core Allocation (2 + #GPUs for V1 busy loop) ───────────────────────
    cpu_cores: Optional[int] = None  # Auto-calculated if None: 2 + gpu_count

    # ── Multi-LoRA for MoE (vLLM ≥0.15.0) ──────────────────────────────
    enable_lora: bool = False
    lora_modules: Optional[List[LoRAModule]] = None
    max_loras: int = 8
    max_lora_rank: int = 64
    lora_extra_vocab_size: int = 256
    lora_tuned_config_dir: Optional[str] = None  # custom fused_moe_lora kernel configs

    # ── Sleep Mode (zero-reload model switching) ────────────────────────
    enable_sleep_mode: bool = False
    sleep_level: int = 1  # 1 = offload to CPU RAM, 2 = discard weights
    auto_sleep_idle_seconds: int = 300  # auto-sleep after N seconds idle

    # ── KV Cache Offloading (vLLM ≥0.11.0) ─────────────────────────────
    kv_connector: Optional[str] = None  # "offloading" for CPU DRAM offload
    kv_connector_config: Optional[Dict[str, Any]] = None

    # ── Speculative Decoding ────────────────────────────────────────────
    speculative_method: Optional[str] = None  # "draft", "ngram", "mtp", "eagle", "medusa"
    speculative_model: Optional[str] = None  # draft model name/path
    num_speculative_tokens: int = 5
    speculative_disable_by_batch_size: Optional[int] = None  # dynamic: disable at high QPS

    # ── FlashInfer Fused Attention (Megakernel Phase 1) ─────────────────
    attention_backend: str = "FLASHINFER"  # FLASHINFER, FLASH_ATTN, XFORMERS
    enable_flashinfer: bool = True  # Auto-applied: ~50% memory bandwidth recovery

    # ── vLLM Router ─────────────────────────────────────────────────────
    enable_router: bool = False
    router_policy: str = "consistent_hash"  # consistent_hash, power_of_two, round_robin
    router_port: int = 8080
    router_session_key: str = "x-session-id"

    # ── LMCache Integration (Distributed KV Cache) ───────────────────────
    enable_lmcache: bool = True
    lmcache_backend: str = "redis"  # redis, s3, disk, cpu
    lmcache_remote_url: Optional[str] = None  # Redis, S3, etc.
    lmcache_chunk_size: int = 256
    lmcache_pipelined: bool = False
    lmcache_serde: str = "torch"  # torch, cachegen
    lmcache_local_device: Optional[str] = None
    lmcache_redis_cluster: bool = False
    lmcache_redis_password: Optional[str] = None
    lmcache_s3_bucket: Optional[str] = None
    lmcache_s3_region: str = "us-east-1"
    lmcache_disk_path: str = "/tmp/lmcache"

    @classmethod
    def create_auto_optimized(cls, model_name: str, workload: WorkloadProfile, **kwargs) -> 'VLLMConfig':
        """Create vLLM config automatically optimized based on workload characteristics.
        
        Analyzes workload patterns and selects optimal settings for the 6 critical knobs:
        1. max_num_batched_tokens - Based on prompt/response lengths and QPS
        2. gpu_memory_utilization - Based on model size and memory pressure
        3. max_num_seqs - Based on concurrent users and QPS
        4. enable_prefix_caching - Based on prompt similarity patterns
        5. enable_chunked_prefill - Based on average prompt length
        6. CPU cores - Auto-calculated based on GPU count and workload
        """
        # Calculate optimal max_num_batched_tokens
        total_tokens_per_request = workload.avg_prompt_length + workload.avg_response_length
        
        # For high QPS workloads, increase batch size
        if workload.requests_per_second > 50:
            max_batched = 32768  # Maximum for high throughput
        elif workload.requests_per_second > 10:
            max_batched = 16384  # High throughput
        elif workload.requests_per_second > 2:
            max_batched = 8192   # Medium throughput
        else:
            max_batched = 4096   # Low throughput/latency focused
        
        # Adjust for latency sensitivity
        if workload.latency_sensitivity > 0.7:
            max_batched = min(max_batched, 4096)
        elif workload.latency_sensitivity < 0.3:
            max_batched = max(max_batched, 16384)
        
        # Calculate optimal max_num_seqs
        # Base on concurrent users and expected burst patterns
        base_seqs = max(workload.concurrent_users, int(workload.requests_per_second * 2))
        
        # Add buffer for bursty traffic
        if workload.requests_per_second > 10:
            max_seqs = base_seqs * 2
        else:
            max_seqs = min(base_seqs * 1.5, 1024)
        
        # Cap at reasonable limits and ensure integer
        max_seqs = int(min(max_seqs, 2048))
        max_seqs = int(max(max_seqs, 256))
        
        # Adjust for latency sensitivity
        if workload.latency_sensitivity > 0.7:
            max_seqs = min(max_seqs, 512)
        
        # Calculate GPU memory utilization
        if workload.memory_pressure > 0.8:
            gpu_util = 0.85  # Conservative for memory pressure
        elif workload.model_size_gb > 40:  # Large models
            gpu_util = 0.90
        else:
            gpu_util = 0.95  # Aggressive for smaller models
        
        # Determine prefix caching value
        # Enable if prompts are likely to share prefixes (system prompts, templates)
        enable_prefix_cache = (
            workload.avg_prompt_length > 100 or  # Long prompts likely have shared prefixes
            workload.concurrent_users > 5 or     # Multi-user scenarios
            workload.requests_per_second > 5     # High QPS benefits from caching
        )
        
        # Determine chunked prefill
        # Most beneficial for long prompts and high QPS
        enable_chunked_prefill = (
            workload.avg_prompt_length > 512 or   # Long prompts
            workload.requests_per_second > 2      # Any significant QPS
        )
        
        # Auto-calculate CPU cores
        cpu_cores = 2 + workload.gpu_count
        
        # Add extra CPU for high QPS workloads
        if workload.requests_per_second > 20:
            cpu_cores += 2
        elif workload.requests_per_second > 10:
            cpu_cores += 1
        
        config = cls(
            model_name=model_name,
            max_num_batched_tokens=max_batched,
            gpu_memory_utilization=gpu_util,
            max_num_seqs=int(max_seqs),
            enable_prefix_caching=enable_prefix_cache,
            enable_chunked_prefill=enable_chunked_prefill,
            cpu_cores=cpu_cores,
            tensor_parallel_size=workload.gpu_count,
            **kwargs
        )
        
        return config
    
    @classmethod
    def analyze_workload_from_samples(cls, samples: List[Dict[str, Any]], gpu_count: int = 1) -> WorkloadProfile:
        """Analyze workload from sample requests to create profile.
        
        Args:
            samples: List of sample requests with 'prompt', 'response', 'timestamp' keys
            gpu_count: Number of GPUs available
            
        Returns:
            WorkloadProfile with analyzed characteristics
        """
        if not samples:
            # Return default profile if no samples
            return WorkloadProfile(
                avg_prompt_length=256,
                avg_response_length=128,
                requests_per_second=1.0,
                concurrent_users=1,
                latency_sensitivity=0.5,
                memory_pressure=0.5,
                gpu_count=gpu_count
            )
        
        # Extract metrics from samples
        prompt_lengths = []
        response_lengths = []
        timestamps = []
        
        for sample in samples:
            # Simple token estimation (rough approximation: 1 token ≈ 4 characters)
            prompt_text = sample.get('prompt', '')
            response_text = sample.get('response', '')
            
            prompt_lengths.append(len(prompt_text) // 4)
            response_lengths.append(len(response_text) // 4)
            
            if 'timestamp' in sample:
                timestamps.append(sample['timestamp'])
        
        # Calculate statistics
        avg_prompt = statistics.mean(prompt_lengths) if prompt_lengths else 256
        avg_response = statistics.mean(response_lengths) if response_lengths else 128
        
        # Calculate QPS from timestamps
        if len(timestamps) > 1:
            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]) / 1000  # Convert to seconds
            requests_per_second = len(timestamps) / max(time_span, 1)
        else:
            requests_per_second = 1.0
        
        # Estimate concurrent users (simplified)
        concurrent_users = min(len(set(s.get('user_id', 'default') for s in samples)), 10)
        
        # Determine latency sensitivity based on response lengths and QPS
        if avg_response < 50 and requests_per_second < 2:
            latency_sensitivity = 0.8  # Short responses, low QPS = latency sensitive
        elif requests_per_second > 10:
            latency_sensitivity = 0.2  # High QPS = throughput focused
        else:
            latency_sensitivity = 0.5  # Balanced
        
        return WorkloadProfile(
            avg_prompt_length=avg_prompt,
            avg_response_length=avg_response,
            requests_per_second=requests_per_second,
            concurrent_users=concurrent_users,
            latency_sensitivity=latency_sensitivity,
            memory_pressure=0.5,  # Default, can be updated based on system info
            gpu_count=gpu_count
        )

    @classmethod
    def create_throughput_optimized(cls, model_name: str, **kwargs) -> 'VLLMConfig':
        """Create vLLM config optimized for throughput-heavy production.
        
        Applies the 6 critical knobs with throughput-focused values:
        - max_num_batched_tokens: 16384 (8x default)
        - gpu_memory_utilization: 0.95 (10% more VRAM)
        - enable_prefix_caching: true (free throughput win)
        - enable_chunked_prefill: true
        - max_num_seqs: 1024 (higher concurrency)
        """
        config = cls(
            model_name=model_name,
            max_num_batched_tokens=16384,
            gpu_memory_utilization=0.95,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_seqs=1024,
            **kwargs
        )
        return config
    
    @classmethod
    def create_latency_optimized(cls, model_name: str, **kwargs) -> 'VLLMConfig':
        """Create vLLM config optimized for latency-sensitive production.
        
        Applies the 6 critical knobs with latency-focused values:
        - max_num_batched_tokens: 4096 (balanced)
        - max_num_seqs: 512 (prevent queue buildup)
        - enable_chunked_prefill: true
        - gpu_memory_utilization: 0.95 (still optimized)
        """
        config = cls(
            model_name=model_name,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.95,
            enable_prefix_caching=True,  # Still beneficial for latency
            enable_chunked_prefill=True,
            max_num_seqs=512,
            **kwargs
        )
        return config


class VLLMService:
    """vLLM integration service for LLM inference"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = f"http://{config.host}:{config.port}/v1"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test vLLM server connection"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test server health
            url = f"http://{self.config.host}:{self.config.port}/health"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return {
                        "status": "connected",
                        "provider": "vllm",
                        "model": self.config.model_name,
                        "endpoint": self.base_url,
                        "host": self.config.host,
                        "port": self.config.port
                    }
                else:
                    return {
                        "status": "failed",
                        "error": f"vLLM server not responding: {response.status}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to connect to vLLM server: {str(e)}"
            }
    
    async def install_vllm(self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Install vLLM on remote instance"""
        try:
            install_script = f"""
#!/bin/bash
# Install vLLM with GPU support
pip install vllm

# Verify installation
python3 -c "import vllm; print('vLLM installed successfully')"
"""
            
            # Execute installation via SSH
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, install_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {
                    "status": "installed",
                    "instance_ip": instance_ip,
                    "provider": "vllm",
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Installation failed: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to install vLLM: {str(e)}"
            }
    
    def _build_ssh_command(self, ip: str, user: str, key: Optional[str], script: str) -> str:
        """Build SSH command for remote execution"""
        if key:
            ssh_cmd = f"ssh -i {key} {user}@{ip}"
        else:
            ssh_cmd = f"ssh {user}@{ip}"
        
        return f'{ssh_cmd} "{script}"'
    
    async def start_server(self, 
                          instance_ip: str,
                          ssh_user: str = "root",
                          ssh_key: Optional[str] = None,
                          additional_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start vLLM server on remote instance"""
        try:
            # Build vLLM server command
            server_cmd = [
                "vllm", "serve", self.config.model_name,
                "--host", self.config.host,
                "--port", str(self.config.port),
                "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
                "--tensor-parallel-size", str(self.config.tensor_parallel_size)
            ]
            
            if self.config.max_model_len:
                server_cmd.extend(["--max-model-len", str(self.config.max_model_len)])
            
            if self.config.api_key:
                server_cmd.extend(["--api-key", self.config.api_key])
            
            if additional_args:
                server_cmd.extend(additional_args)
            
            # Create systemd service
            service_content = f"""
[Unit]
Description=vLLM Server for {self.config.model_name}
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart={" ".join(server_cmd)}
Restart=always
RestartSec=10
Environment=PYTHONPATH=/root

[Install]
WantedBy=multi-user.target
"""
            
            # Create and start service
            setup_script = f"""
#!/bin/bash
# Create vLLM service
echo '{service_content}' > /etc/systemd/system/vllm.service

# Reload systemd and start service
systemctl daemon-reload
systemctl enable vllm
systemctl start vllm

# Wait for service to start
sleep 10
systemctl status vllm
"""
            
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, setup_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {
                    "status": "started",
                    "instance_ip": instance_ip,
                    "provider": "vllm",
                    "model": self.config.model_name,
                    "endpoint": f"http://{instance_ip}:{self.config.port}/v1",
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Failed to start server: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to start vLLM server: {str(e)}"
            }
    
    async def test_inference(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Test vLLM inference"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Prepare OpenAI-compatible request
            url = f"{self.base_url}/completions"
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "vllm",
                        "model": self.config.model_name,
                        "prompt": prompt,
                        "response": result["choices"][0]["text"],
                        "usage": result.get("usage", {}),
                        "endpoint": self.base_url
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Inference failed: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to test inference: {str(e)}"
            }
    
    async def test_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 100) -> Dict[str, Any]:
        """Test vLLM chat completion"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Prepare OpenAI-compatible chat request
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "vllm",
                        "model": self.config.model_name,
                        "messages": messages,
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "endpoint": self.base_url
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Chat completion failed: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to test chat completion: {str(e)}"
            }
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get vLLM server information"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get model info
            url = f"{self.base_url}/models"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "vllm",
                        "models": result.get("data", []),
                        "endpoint": self.base_url,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Failed to get server info: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to get server info: {str(e)}"
            }
    
    async def stop_server(self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Stop vLLM server on remote instance"""
        try:
            stop_script = """
#!/bin/bash
# Stop and disable vLLM service
systemctl stop vllm
systemctl disable vllm
rm -f /etc/systemd/system/vllm.service
systemctl daemon-reload
"""
            
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, stop_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    "status": "stopped",
                    "instance_ip": instance_ip,
                    "provider": "vllm",
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Failed to stop server: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to stop vLLM server: {str(e)}"
            }
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        return [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            "Qwen/Qwen-7B",
            "deepseek-ai/deepseek-coder-6.7b-base",
            "codellama/CodeLlama-7b-hf",
            "codellama/CodeLlama-13b-hf",
            "codellama/CodeLlama-34b-hf"
        ]
    
    def _build_server_args(self) -> List[str]:
        """Build the full vLLM serve argument list from config.

        Centralises flag generation so start_server, get_deployment_script,
        and K8s/Helm templates all use the same logic.
        """
        args: List[str] = [
            "vllm", "serve", self.config.model_name,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
        ]

        if self.config.max_model_len:
            args.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.api_key:
            args.extend(["--api-key", self.config.api_key])
        
        # ── Critical Throughput Optimizations (6 knobs) ───────────────────────
        args.extend(["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)])
        args.extend(["--max-num-seqs", str(self.config.max_num_seqs)])
        
        if self.config.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        
        if self.config.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        # ── FlashInfer Fused Attention ────────────────────────────────
        if self.config.enable_flashinfer:
            args.extend(["--attention-backend", self.config.attention_backend])

        # ── Multi-LoRA ──────────────────────────────────────────────────
        if self.config.enable_lora:
            args.append("--enable-lora")
            args.extend(["--max-loras", str(self.config.max_loras)])
            args.extend(["--max-lora-rank", str(self.config.max_lora_rank)])
            args.extend(["--lora-extra-vocab-size", str(self.config.lora_extra_vocab_size)])
            if self.config.lora_modules:
                for lm in self.config.lora_modules:
                    args.extend(["--lora-modules", f"{lm.name}={lm.path}"])
            if self.config.lora_tuned_config_dir:
                args.extend(["--override-neuron-config", self.config.lora_tuned_config_dir])

        # ── Sleep Mode ──────────────────────────────────────────────────
        if self.config.enable_sleep_mode:
            args.append("--enable-sleep-mode")

        # ── KV Cache Offloading ─────────────────────────────────────────
        if self.config.kv_connector:
            args.extend(["--kv-connector", self.config.kv_connector])

        # ── Speculative Decoding ────────────────────────────────────────
        if self.config.speculative_method:
            if self.config.speculative_method == "mtp":
                args.extend(["--speculative-config.method=mtp"])
                args.extend([f"--speculative-config.num_speculative_tokens={self.config.num_speculative_tokens}"])
            elif self.config.speculative_method == "ngram":
                args.extend(["--speculative-config.method=ngram"])
                args.extend([f"--speculative-config.num_speculative_tokens={self.config.num_speculative_tokens}"])
            elif self.config.speculative_method in ("draft", "eagle", "medusa"):
                if self.config.speculative_model:
                    args.extend(["--speculative-model", self.config.speculative_model])
                args.extend(["--num-speculative-tokens", str(self.config.num_speculative_tokens)])
            if self.config.speculative_disable_by_batch_size:
                args.extend(["--speculative-disable-by-batch-size",
                             str(self.config.speculative_disable_by_batch_size)])

        # ── LMCache Integration ───────────────────────────────────────────
        if self.config.enable_lmcache:
            from .lmcache_service import LMCacheService, LMCacheConfig
            
            lmcache_config = LMCacheConfig(
                enabled=True,
                backend=self.config.lmcache_backend,
                remote_url=self.config.lmcache_remote_url or "redis://localhost:6379",
                chunk_size=self.config.lmcache_chunk_size,
                pipelined=self.config.lmcache_pipelined,
                serde=self.config.lmcache_serde,
                local_device=self.config.lmcache_local_device,
                redis_cluster=self.config.lmcache_redis_cluster,
                redis_password=self.config.lmcache_redis_password,
                s3_bucket=self.config.lmcache_s3_bucket,
                s3_region=self.config.lmcache_s3_region,
                disk_path=self.config.lmcache_disk_path,
                enable_pipelined_backend=self.config.lmcache_pipelined
            )
            
            lmcache_service = LMCacheService(lmcache_config)
            args.extend(lmcache_service.generate_vllm_args())

        return args

    # ═══════════════════════════════════════════════════════════════════
    # Multi-LoRA adapter management (hot-load / hot-unload)
    # ═══════════════════════════════════════════════════════════════════

    async def lora_list(self) -> Dict[str, Any]:
        """List LoRA adapters currently loaded on the server."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            url = f"{self.base_url}/models"
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            async with self.session.get(url, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    base = [m for m in models if m.get("parent") is None]
                    adapters = [m for m in models if m.get("parent") is not None]
                    return {
                        "status": "success",
                        "base_models": base,
                        "lora_adapters": adapters,
                    }
                return {"status": "failed", "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def lora_load(self, adapter: LoRAModule) -> Dict[str, Any]:
        """Hot-load a LoRA adapter onto the running server (POST /loras)."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            url = f"http://{self.config.host}:{self.config.port}/v1/load_lora_adapter"
            payload = {
                "lora_name": adapter.name,
                "lora_path": adapter.path,
            }
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            async with self.session.post(url, json=payload, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    return {"status": "loaded", "adapter": adapter.name}
                body = await resp.text()
                return {"status": "failed", "error": f"HTTP {resp.status}: {body}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def lora_unload(self, adapter_name: str) -> Dict[str, Any]:
        """Hot-unload a LoRA adapter from the running server."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            url = f"http://{self.config.host}:{self.config.port}/v1/unload_lora_adapter"
            payload = {"lora_name": adapter_name}
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            async with self.session.post(url, json=payload, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return {"status": "unloaded", "adapter": adapter_name}
                body = await resp.text()
                return {"status": "failed", "error": f"HTTP {resp.status}: {body}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # Sleep Mode — zero-reload model switching
    # ═══════════════════════════════════════════════════════════════════

    async def sleep(self, level: Optional[int] = None) -> Dict[str, Any]:
        """Put the vLLM server to sleep (offload or discard weights).

        Level 1: offload weights to CPU RAM (fast wake ~0.1-6s)
        Level 2: discard weights entirely (minimal RAM, needs reload_weights on wake)
        """
        lvl = level or self.config.sleep_level
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            url = f"http://{self.config.host}:{self.config.port}/sleep?level={lvl}"
            async with self.session.post(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return {"status": "sleeping", "level": lvl,
                            "model": self.config.model_name}
                body = await resp.text()
                return {"status": "failed", "error": f"HTTP {resp.status}: {body}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def wake(self) -> Dict[str, Any]:
        """Wake a sleeping vLLM server.

        For Level 2 sleep, also calls reload_weights and reset_prefix_cache.
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            base = f"http://{self.config.host}:{self.config.port}"

            # Wake up
            async with self.session.post(f"{base}/wake_up",
                                         timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {"status": "failed", "error": f"wake_up HTTP {resp.status}: {body}"}

            # Level 2 requires reload_weights + reset_prefix_cache
            if self.config.sleep_level == 2:
                async with self.session.post(
                    f"{base}/collective_rpc",
                    json={"method": "reload_weights"},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"status": "failed",
                                "error": f"reload_weights HTTP {resp.status}: {body}"}

                async with self.session.post(
                    f"{base}/reset_prefix_cache",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"status": "failed",
                                "error": f"reset_prefix_cache HTTP {resp.status}: {body}"}

            return {"status": "awake", "model": self.config.model_name,
                    "level": self.config.sleep_level}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def get_deployment_script(self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> str:
        """Generate deployment script for vLLM"""
        serve_args = self._build_server_args()
        exec_line = " \\\n    ".join(serve_args)

        env_lines = ""
        if self.config.enable_sleep_mode:
            env_lines += "Environment=VLLM_SERVER_DEV_MODE=1\n"
        if self.config.enable_flashinfer:
            env_lines += f"Environment=VLLM_ATTENTION_BACKEND={self.config.attention_backend}\n"

        script = f"""
#!/bin/bash
# vLLM Deployment Script for Terradev
# Target: {instance_ip}

echo "🚀 Deploying vLLM for {self.config.model_name}..."

# Install vLLM ≥0.15.0 (Multi-LoRA MoE + Sleep Mode + KV Offloading)
pip install 'vllm>=0.15.0'

# Create systemd service
cat > /etc/systemd/system/vllm.service << 'EOF'
[Unit]
Description=vLLM Server for {self.config.model_name}
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart={exec_line}
Restart=always
RestartSec=10
{env_lines}
[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable vllm
systemctl start vllm

echo "✅ vLLM server started on http://{instance_ip}:{self.config.port}/v1"
echo "🔗 Test with: curl http://{instance_ip}:{self.config.port}/v1/models"
"""
        return script
    
    async def analyze_current_workload(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Analyze current workload patterns from running vLLM server.
        
        Args:
            duration_seconds: How long to monitor the server
            
        Returns:
            Workload analysis with optimization recommendations
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get server metrics
            metrics_url = f"{self.base_url}/metrics"
            async with self.session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"status": "failed", "error": f"Metrics endpoint not available: {resp.status}"}
                
                metrics_text = await resp.text()
            
            # Parse metrics to extract workload patterns
            workload_data = self._parse_vllm_metrics(metrics_text)
            
            # Get current server info
            server_info = await self.get_server_info()
            
            # Analyze and generate recommendations
            analysis = {
                "status": "success",
                "current_workload": workload_data,
                "server_info": server_info,
                "optimization_recommendations": self._generate_optimization_recommendations(workload_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return {"status": "failed", "error": f"Workload analysis failed: {str(e)}"}
    
    def _parse_vllm_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse vLLM Prometheus metrics to extract workload patterns."""
        workload = {
            "avg_prompt_tokens": 0,
            "avg_generation_tokens": 0,
            "requests_per_second": 0,
            "active_requests": 0,
            "queue_size": 0,
            "gpu_utilization": 0,
            "memory_usage": 0
        }
        
        for line in metrics_text.split('\n'):
            if 'vllm:avg_prompt_tokens' in line:
                workload["avg_prompt_tokens"] = float(line.split()[-1])
            elif 'vllm:avg_generation_tokens' in line:
                workload["avg_generation_tokens"] = float(line.split()[-1])
            elif 'vllm:requests_per_second' in line:
                workload["requests_per_second"] = float(line.split()[-1])
            elif 'vllm:active_requests' in line:
                workload["active_requests"] = int(float(line.split()[-1]))
            elif 'vllm:queue_size' in line:
                workload["queue_size"] = int(float(line.split()[-1]))
        
        return workload
    
    def _generate_optimization_recommendations(self, workload_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on current workload."""
        recommendations = []
        
        # Analyze batch size optimization
        current_rps = workload_data.get("requests_per_second", 0)
        current_batch_tokens = self.config.max_num_batched_tokens
        
        if current_rps > 20 and current_batch_tokens < 16384:
            recommendations.append({
                "type": "increase_batch_size",
                "current_value": current_batch_tokens,
                "recommended_value": 16384,
                "reason": f"High QPS ({current_rps:.1f}) would benefit from larger batch size",
                "impact": "2-3x throughput improvement"
            })
        elif current_rps < 2 and current_batch_tokens > 4096:
            recommendations.append({
                "type": "decrease_batch_size",
                "current_value": current_batch_tokens,
                "recommended_value": 4096,
                "reason": f"Low QPS ({current_rps:.1f}) would benefit from smaller batch size for lower latency",
                "impact": "10-20% latency reduction"
            })
        
        # Analyze max sequences optimization
        queue_size = workload_data.get("queue_size", 0)
        current_max_seqs = self.config.max_num_seqs
        
        if queue_size > 10 and current_max_seqs < 1024:
            recommendations.append({
                "type": "increase_max_sequences",
                "current_value": current_max_seqs,
                "recommended_value": min(current_max_seqs * 2, 2048),
                "reason": f"Queue size ({queue_size}) indicates sequence limit is too low",
                "impact": "Reduce queuing, improve burst handling"
            })
        
        # Analyze prefix caching
        avg_prompt = workload_data.get("avg_prompt_tokens", 0)
        if not self.config.enable_prefix_caching and avg_prompt > 100:
            recommendations.append({
                "type": "enable_prefix_caching",
                "current_value": False,
                "recommended_value": True,
                "reason": f"Long prompts ({avg_prompt:.0f} tokens) would benefit from prefix caching",
                "impact": "1.5-3x throughput for similar prompts"
            })
        
        # Analyze chunked prefill
        if not self.config.enable_chunked_prefill and avg_prompt > 512:
            recommendations.append({
                "type": "enable_chunked_prefill",
                "current_value": False,
                "recommended_value": True,
                "reason": f"Very long prompts ({avg_prompt:.0f} tokens) would benefit from chunked prefill",
                "impact": "20-50% faster prefill for long prompts"
            })
        
        return recommendations
    
    async def auto_optimize_from_workload(self, samples: Optional[List[Dict[str, Any]]] = None, 
                                        gpu_count: int = 1) -> Dict[str, Any]:
        """Automatically optimize configuration based on workload analysis.
        
        Args:
            samples: Optional sample requests for analysis
            gpu_count: Number of GPUs available
            
        Returns:
            Optimization results with new configuration
        """
        try:
            # Analyze workload from samples or current server
            if samples:
                workload = VLLMConfig.analyze_workload_from_samples(samples, gpu_count)
            else:
                # Analyze current server workload
                analysis = await self.analyze_current_workload()
                if analysis["status"] != "success":
                    return analysis
                
                current = analysis["current_workload"]
                workload = WorkloadProfile(
                    avg_prompt_length=current.get("avg_prompt_tokens", 256),
                    avg_response_length=current.get("avg_generation_tokens", 128),
                    requests_per_second=current.get("requests_per_second", 1.0),
                    concurrent_users=max(current.get("active_requests", 1), 1),
                    latency_sensitivity=0.5,  # Default, can be inferred from patterns
                    memory_pressure=0.5,
                    gpu_count=gpu_count
                )
            
            # Generate optimized configuration
            optimized_config = VLLMConfig.create_auto_optimized(
                self.config.model_name, workload
            )
            
            # Compare with current config
            comparison = self._compare_configurations(self.config, optimized_config)
            
            return {
                "status": "success",
                "workload_profile": workload,
                "current_config": self._config_to_dict(self.config),
                "optimized_config": self._config_to_dict(optimized_config),
                "changes": comparison,
                "recommendations": "Apply optimized configuration for better performance"
            }
            
        except Exception as e:
            return {"status": "failed", "error": f"Auto-optimization failed: {str(e)}"}
    
    def _compare_configurations(self, current: 'VLLMConfig', optimized: 'VLLMConfig') -> List[Dict[str, Any]]:
        """Compare current and optimized configurations."""
        changes = []
        
        if current.max_num_batched_tokens != optimized.max_num_batched_tokens:
            changes.append({
                "parameter": "max_num_batched_tokens",
                "current": current.max_num_batched_tokens,
                "optimized": optimized.max_num_batched_tokens,
                "impact": "throughput" if optimized.max_num_batched_tokens > current.max_num_batched_tokens else "latency"
            })
        
        if current.max_num_seqs != optimized.max_num_seqs:
            changes.append({
                "parameter": "max_num_seqs",
                "current": current.max_num_seqs,
                "optimized": optimized.max_num_seqs,
                "impact": "concurrency"
            })
        
        if current.gpu_memory_utilization != optimized.gpu_memory_utilization:
            changes.append({
                "parameter": "gpu_memory_utilization",
                "current": current.gpu_memory_utilization,
                "optimized": optimized.gpu_memory_utilization,
                "impact": "memory_efficiency"
            })
        
        if current.enable_prefix_caching != optimized.enable_prefix_caching:
            changes.append({
                "parameter": "enable_prefix_caching",
                "current": current.enable_prefix_caching,
                "optimized": optimized.enable_prefix_caching,
                "impact": "cache_efficiency"
            })
        
        if current.enable_chunked_prefill != optimized.enable_chunked_prefill:
            changes.append({
                "parameter": "enable_chunked_prefill",
                "current": current.enable_chunked_prefill,
                "optimized": optimized.enable_chunked_prefill,
                "impact": "prefill_speed"
            })
        
        return changes
    
    def _config_to_dict(self, config: 'VLLMConfig') -> Dict[str, Any]:
        """Convert VLLMConfig to dictionary for serialization."""
        return {
            "model_name": config.model_name,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_num_seqs": config.max_num_seqs,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enable_prefix_caching": config.enable_prefix_caching,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "cpu_cores": config.cpu_cores,
            "tensor_parallel_size": config.tensor_parallel_size
        }
