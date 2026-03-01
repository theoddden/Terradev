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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


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
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1

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
