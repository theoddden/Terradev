#!/usr/bin/env python3
"""
KServe Service Integration for Terradev
Manages KServe InferenceService deployments on Kubernetes.

Terradev-specific features:
  - generate_inferenceservice_yaml() — produces a complete KServe manifest with
    GPU topology hints (NUMA pinning, resource limits calculated from MoEProfile)
  - watch_rollout() — async generator that streams rollout events instead of
    one-shot status polling
"""

import logging
import math
import os
import json
import asyncio
import subprocess
import aiohttp
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KServeConfig:
    """KServe configuration"""
    kubernetes_config: Optional[str] = None  # Path to kubeconfig
    namespace: str = "default"
    auth_token: Optional[str] = None
    cluster_endpoint: Optional[str] = None


class KServeService:
    """KServe integration service for model deployment and management"""
    
    def __init__(self, config: KServeConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test KServe connection and get cluster info"""
        try:
            # Try to use kubectl command first
            import subprocess
            
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "status": "connected",
                    "method": "kubectl",
                    "cluster_info": result.stdout,
                    "namespace": self.config.namespace
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "kubectl not found. Please install kubectl."
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def list_inference_services(self) -> List[Dict[str, Any]]:
        """List all InferenceServices in the namespace"""
        try:
            import subprocess
            
            result = subprocess.run([
                "kubectl", "get", "inferenceservices",
                "-n", self.config.namespace,
                "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                services = []
                
                for item in data.get("items", []):
                    service = {
                        "name": item["metadata"]["name"],
                        "namespace": item["metadata"]["namespace"],
                        "created": item["metadata"]["creationTimestamp"],
                        "predictor": item.get("spec", {}).get("predictor", {}),
                        "status": item.get("status", {})
                    }
                    services.append(service)
                
                return services
            else:
                raise Exception(f"kubectl command failed: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to list InferenceServices: {e}")
    
    async def create_inference_service(self, 
                                      name: str,
                                      model_uri: str,
                                      framework: str = "tensorflow",
                                      runtime_version: str = "latest",
                                      min_replicas: int = 1,
                                      max_replicas: int = 3) -> Dict[str, Any]:
        """Create a new InferenceService"""
        
        # Generate InferenceService YAML
        service_spec = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": name,
                "namespace": self.config.namespace
            },
            "spec": {
                "predictor": {
                    "framework": framework,
                    "runtimeVersion": runtime_version,
                    "model": {
                        "modelFormat": {
                            "name": framework
                        },
                        "storageUri": model_uri
                    },
                    "minReplicas": min_replicas,
                    "maxReplicas": max_replicas
                }
            }
        }
        
        try:
            # Write spec to temporary file and apply
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                json.dump(service_spec, f, indent=2)
                temp_file = f.name
            
            try:
                result = subprocess.run([
                    "kubectl", "apply", "-f", temp_file,
                    "-n", self.config.namespace
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    return {
                        "status": "created",
                        "name": name,
                        "namespace": self.config.namespace,
                        "output": result.stdout
                    }
                else:
                    raise Exception(f"Failed to create InferenceService: {result.stderr}")
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            raise Exception(f"Failed to create InferenceService {name}: {e}")
    
    async def delete_inference_service(self, name: str) -> Dict[str, Any]:
        """Delete an InferenceService"""
        try:
            import subprocess
            
            result = subprocess.run([
                "kubectl", "delete", "inferenceservice", name,
                "-n", self.config.namespace
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    "status": "deleted",
                    "name": name,
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to delete InferenceService: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to delete InferenceService {name}: {e}")
    
    async def get_service_url(self, name: str) -> Optional[str]:
        """Get the prediction URL for an InferenceService"""
        try:
            import subprocess
            
            # Get the service URL
            result = subprocess.run([
                "kubectl", "get", "inferenceservice", name,
                "-n", self.config.namespace,
                "-o", "jsonpath='{.status.url}'"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                url = result.stdout.strip().strip("'\"")
                return url
            else:
                # Try to construct URL from service name
                cluster_info = subprocess.run([
                    "kubectl", "config", "view", "--minify", "-o", "jsonpath='{.clusters[0].cluster.server}'"
                ], capture_output=True, text=True, timeout=10)
                
                if cluster_info.returncode == 0:
                    base_url = cluster_info.stdout.strip().strip("'\"")
                    return f"{base_url}/serving/{self.config.namespace}/v1/models/{name}:predict"
                
            return None
            
        except Exception as e:
            raise Exception(f"Failed to get service URL for {name}: {e}")
    
    async def predict(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction request to the InferenceService"""
        try:
            url = await self.get_service_url(name)
            if not url:
                raise Exception(f"Could not get URL for service {name}")
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Make prediction request
            async with self.session.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Prediction failed: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to make prediction to {name}: {e}")

    # ── Terradev-specific: GPU-aware manifest generation ─────────────

    def generate_inferenceservice_yaml(
        self,
        name: str,
        model_uri: str,
        framework: str = "pytorch",
        *,
        gpu_type: Optional[str] = None,
        gpu_count: int = 1,
        vram_gb: Optional[float] = None,
        model_size_b: Optional[float] = None,
        numa_pinning: bool = False,
        max_replicas: int = 3,
        min_replicas: int = 1,
        target_utilization: int = 80,
        runtime_version: str = "latest",
        extra_env: Optional[Dict[str, str]] = None,
        extra_annotations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete KServe InferenceService manifest with Terradev GPU topology hints.

        Calculates resource limits from model size / VRAM, adds NUMA pinning
        annotations, GPU resource requests, and Terradev-specific labels for
        cost tracking and routing integration.

        Returns the manifest as a dict (YAML-serializable).
        """
        # Auto-calculate VRAM if model size is given (fp16: 2 bytes/param + 20% overhead)
        if vram_gb is None and model_size_b:
            vram_gb = math.ceil(model_size_b * 2 * 1.2)

        # Auto-calculate GPU count from VRAM requirement
        if vram_gb and gpu_count == 1:
            # Common GPU VRAM sizes
            gpu_vram_map = {
                "A100": 80, "A100-80G": 80, "A100-40G": 40,
                "H100": 80, "H200": 141,
                "A10G": 24, "L4": 24, "T4": 16,
                "L40S": 48, "A6000": 48, "RTX4090": 24,
            }
            single_gpu_vram = gpu_vram_map.get(gpu_type or "", 80)
            if vram_gb > single_gpu_vram:
                gpu_count = math.ceil(vram_gb / single_gpu_vram)

        # Calculate memory request (VRAM + ~4GB system overhead per GPU)
        mem_request_gi = (vram_gb or 16) + (4 * gpu_count)

        # Build annotations
        annotations: Dict[str, str] = {
            "terradev.io/managed": "true",
        }
        if gpu_type:
            annotations["terradev.io/gpu-type"] = gpu_type
        if numa_pinning:
            annotations["terradev.io/numa-pinning"] = "true"
            # Topology Manager hint for single-numa-node policy
            annotations["topologymanager.kubernetes.io/policy"] = "single-numa-node"
        # Autoscaling
        annotations["autoscaling.knative.dev/target"] = str(target_utilization)
        if extra_annotations:
            annotations.update(extra_annotations)

        # Build env vars
        env_list = []
        if numa_pinning:
            env_list.append({"name": "CUDA_VISIBLE_DEVICES", "value": ",".join(str(i) for i in range(gpu_count))})
            env_list.append({"name": "NCCL_TOPOLOGY", "value": "NUMA"})
        if extra_env:
            for k, v in extra_env.items():
                env_list.append({"name": k, "value": v})

        # Build container resources
        resources = {
            "limits": {
                "nvidia.com/gpu": str(gpu_count),
                "memory": f"{mem_request_gi}Gi",
                "cpu": str(gpu_count * 4),  # 4 CPUs per GPU is a reasonable default
            },
            "requests": {
                "nvidia.com/gpu": str(gpu_count),
                "memory": f"{max(mem_request_gi // 2, 8)}Gi",
                "cpu": str(max(gpu_count * 2, 2)),
            },
        }

        # Node selector for GPU type
        node_selector: Dict[str, str] = {}
        if gpu_type:
            node_selector["nvidia.com/gpu.product"] = gpu_type

        # Build the manifest
        manifest: Dict[str, Any] = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": name,
                "namespace": self.config.namespace,
                "annotations": annotations,
                "labels": {
                    "terradev.io/managed": "true",
                    "terradev.io/gpu-type": gpu_type or "auto",
                    "terradev.io/gpu-count": str(gpu_count),
                },
            },
            "spec": {
                "predictor": {
                    "minReplicas": min_replicas,
                    "maxReplicas": max_replicas,
                    framework: {
                        "runtimeVersion": runtime_version,
                        "storageUri": model_uri,
                        "resources": resources,
                    },
                },
            },
        }

        # Add node selector if we have GPU type
        if node_selector:
            manifest["spec"]["predictor"][framework]["nodeSelector"] = node_selector

        # Add env if any
        if env_list:
            manifest["spec"]["predictor"][framework]["env"] = env_list

        # Add tolerations for GPU nodes
        manifest["spec"]["predictor"]["tolerations"] = [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            },
        ]

        return {
            "manifest": manifest,
            "summary": {
                "name": name,
                "namespace": self.config.namespace,
                "framework": framework,
                "model_uri": model_uri,
                "gpu_type": gpu_type or "auto",
                "gpu_count": gpu_count,
                "vram_gb": vram_gb,
                "numa_pinning": numa_pinning,
                "replicas": f"{min_replicas}-{max_replicas}",
                "memory_limit": f"{mem_request_gi}Gi",
            },
        }

    async def watch_rollout(
        self,
        name: str,
        *,
        timeout_seconds: int = 300,
        poll_interval: float = 5.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async generator that streams rollout events for an InferenceService.

        Instead of one-shot status, yields status dicts every poll_interval
        seconds until the rollout is complete, failed, or times out.

        Usage:
            async for event in service.watch_rollout("my-model"):
                print(event["phase"], event["ready_replicas"])
        """
        import time
        deadline = time.monotonic() + timeout_seconds
        prev_phase = None

        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    [
                        "kubectl", "get", "inferenceservice", name,
                        "-n", self.config.namespace,
                        "-o", "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

                if result.returncode != 0:
                    yield {
                        "phase": "error",
                        "name": name,
                        "message": result.stderr.strip(),
                        "elapsed_seconds": round(timeout_seconds - (deadline - time.monotonic()), 1),
                    }
                    await asyncio.sleep(poll_interval)
                    continue

                data = json.loads(result.stdout)
                status = data.get("status", {})
                conditions = status.get("conditions", [])

                # Determine phase
                phase = "unknown"
                message = ""
                for cond in conditions:
                    if cond.get("type") == "Ready":
                        if cond.get("status") == "True":
                            phase = "ready"
                        elif cond.get("status") == "False":
                            phase = "progressing"
                        message = cond.get("message", "")
                        break

                # Check for failure conditions
                for cond in conditions:
                    if cond.get("type") == "IngressReady" and cond.get("status") == "False":
                        if "revision" in cond.get("message", "").lower() and "failed" in cond.get("message", "").lower():
                            phase = "failed"
                            message = cond.get("message", "")

                ready_replicas = status.get("readyReplicas", 0)
                url = status.get("url", "")
                elapsed = round(timeout_seconds - (deadline - time.monotonic()), 1)

                event = {
                    "phase": phase,
                    "name": name,
                    "namespace": self.config.namespace,
                    "ready_replicas": ready_replicas,
                    "url": url,
                    "message": message,
                    "elapsed_seconds": elapsed,
                    "conditions": [
                        {"type": c.get("type"), "status": c.get("status"), "message": c.get("message", "")}
                        for c in conditions
                    ],
                }

                # Only yield if phase changed or on first poll
                if phase != prev_phase or prev_phase is None:
                    yield event
                    prev_phase = phase

                # Terminal states
                if phase == "ready":
                    yield {**event, "message": "Rollout complete"}
                    return
                if phase == "failed":
                    yield {**event, "message": f"Rollout failed: {message}"}
                    return

            except Exception as e:
                yield {
                    "phase": "error",
                    "name": name,
                    "message": str(e),
                    "elapsed_seconds": round(timeout_seconds - (deadline - time.monotonic()), 1),
                }

            await asyncio.sleep(poll_interval)

        # Timeout
        yield {
            "phase": "timeout",
            "name": name,
            "message": f"Rollout did not complete within {timeout_seconds}s",
            "elapsed_seconds": timeout_seconds,
        }


def create_kserve_service_from_credentials(credentials: Dict[str, str]) -> KServeService:
    """Create KServeService from credential dictionary"""
    config = KServeConfig(
        kubernetes_config=credentials.get("kubeconfig_path"),
        namespace=credentials.get("namespace", "default"),
        auth_token=credentials.get("auth_token"),
        cluster_endpoint=credentials.get("cluster_endpoint")
    )
    
    return KServeService(config)


def get_kserve_setup_instructions() -> str:
    """Get setup instructions for KServe"""
    return """
🚀 KServe Setup Instructions:

1. Install KServe on your Kubernetes cluster:
   kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml

2. Install kubectl if not already installed:
   # macOS
   brew install kubectl
   
   # Linux
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   
   # Windows (using Chocolatey)
   choco install kubernetes-cli

3. Configure kubectl to connect to your cluster:
   # For local clusters (minikube, kind, etc.)
   kubectl config use-context your-cluster-name
   
   # For cloud providers, use their CLI tools to configure access

4. Test the connection:
   kubectl cluster-info

5. Configure Terradev with your KServe credentials:
   terradev configure --provider kserve --kubeconfig-path ~/.kube/config --namespace default

📋 Required Credentials:
- kubeconfig_path: Path to Kubernetes config file (optional, uses default)
- namespace: Kubernetes namespace (default: "default")
- auth_token: Authentication token (optional, uses kubeconfig)
- cluster_endpoint: Cluster API endpoint (optional, uses kubeconfig)

💡 Usage Examples:
# List all InferenceServices
terradev kserve list

# Create a new InferenceService
terradev kserve create --name my-model --model-uri s3://my-bucket/model --framework tensorflow

# Make predictions
terradev kserve predict --name my-model --data '{"instances": [[1.0, 2.0, 3.0]]}'
"""
