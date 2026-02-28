#!/usr/bin/env python3
"""
SGLang Service Integration for Terradev
Real lifecycle management: SSH/systemd server start/stop, OpenAI-compatible
API calls, MoE Expert Parallelism support, metrics collection.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import subprocess


@dataclass
class SGLangConfig:
    """SGLang configuration"""
    model_name: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    tp_size: int = 1
    dp_size: int = 8
    mem_fraction_static: float = 0.85
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True
    # MoE Expert Parallelism
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    enable_dbo: bool = False
    # Legacy compat fields
    model_path: Optional[str] = None
    serving_config: Optional[Dict[str, Any]] = None
    dashboard_enabled: bool = False
    tracing_enabled: bool = False
    metrics_enabled: bool = False
    deployment_enabled: bool = False
    observability_enabled: bool = False


class SGLangService:
    """SGLang integration service with real server lifecycle management"""

    def __init__(self, config: SGLangConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = f"http://{config.host}:{config.port}/v1"

    async def __aenter__(self):
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

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
