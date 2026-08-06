#!/usr/bin/env python3
"""
Enhanced Kubernetes Service with Karpenter integration
"""

import os
import json
import logging
import aiohttp
import subprocess
from typing import Dict, List, Any, Optional

from .kubernetes_service import KubernetesConfig  # single canonical definition

logger = logging.getLogger(__name__)


class EnhancedKubernetesService:
    """Enhanced Kubernetes service with deep monitoring integration"""

    def __init__(self, config: Optional[KubernetesConfig] = None):
        self.config = config or KubernetesConfig()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            status = {
                "kubernetes": await self._get_cluster_status(),
                "karpenter": await self._get_karpenter_status(env),
            }

            return status

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def _get_cluster_status(self) -> Dict[str, Any]:
        """Get detailed cluster status"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=15,
                env=os.environ.copy(),
            )

            if result.returncode == 0:
                nodes_data = json.loads(result.stdout)

                status = {
                    "total_nodes": len(nodes_data.get("items", [])),
                    "ready_nodes": len(
                        [
                            n
                            for n in nodes_data.get("items", [])
                            if n.get("status", {})
                            .get("conditions", [{}])[-1]
                            .get("type")
                            == "Ready"
                        ]
                    ),
                    "gpu_nodes": len(
                        [
                            n
                            for n in nodes_data.get("items", [])
                            if "nvidia.com/gpu"
                            in n.get("status", {}).get("capacity", {})
                        ]
                    ),
                    "node_pools": self._get_node_pools_summary(nodes_data),
                }

                return status
            else:
                raise Exception(f"Failed to get cluster status: {result.stderr}")

        except Exception as e:  # noqa: BLE001
            raise Exception(f"Failed to get cluster status: {e}")

    def _get_node_pools_summary(self, nodes_data: Dict) -> Dict[str, Any]:
        """Get node pools summary"""
        pools = {}

        for node in nodes_data.get("items", []):
            labels = node.get("metadata", {}).get("labels", {})
            pool_name = labels.get("karpenter.sh/nodepool", "default")

            if pool_name not in pools:
                pools[pool_name] = {"count": 0, "instance_types": set(), "gpu_count": 0}

            pools[pool_name]["count"] += 1

            instance_type = labels.get("node.kubernetes.io/instance-type", "unknown")
            pools[pool_name]["instance_types"].add(instance_type)

            gpu_capacity = (
                node.get("status", {}).get("capacity", {}).get("nvidia.com/gpu", "0")
            )
            if gpu_capacity and gpu_capacity != "0":
                pools[pool_name]["gpu_count"] += int(gpu_capacity)

        # Convert sets to lists for JSON serialization
        for pool_name in pools:
            pools[pool_name]["instance_types"] = list(
                pools[pool_name]["instance_types"]
            )

        return pools

    async def _get_karpenter_status(self, env: Dict[str, str]) -> Dict[str, Any]:
        """Check Karpenter status"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", "-n", "karpenter", "-l", "app=karpenter"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "karpenter" in line and "Running" in line:
                        return {"status": "healthy", "details": line.strip()}

            return {"status": "unhealthy", "error": "Karpenter not running"}

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def install_gpu_operator(
        self,
        cluster_name: str = "",
        namespace: str = "gpu-operator",
        driver_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install NVIDIA GPU Operator on the cluster."""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # Add NVIDIA Helm repo
            subprocess.run(
                [
                    "helm",
                    "repo",
                    "add",
                    "nvidia",
                    "https://helm.ngc.nvidia.com/nvidia",
                    "--force-update",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            subprocess.run(
                ["helm", "repo", "update"],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )

            helm_cmd = [
                "helm",
                "upgrade",
                "--install",
                "gpu-operator",
                "nvidia/gpu-operator",
                "--namespace",
                namespace,
                "--create-namespace",
                "--set",
                "driver.enabled=true",
                "--set",
                "toolkit.enabled=true",
                "--set",
                "devicePlugin.enabled=true",
                "--set",
                "dcgmExporter.enabled=true",
                "--set",
                "gfd.enabled=true",
                "--wait",
                "--timeout=10m",
            ]
            if driver_version:
                helm_cmd.extend(["--set", f"driver.version={driver_version}"])

            result = subprocess.run(
                helm_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            if result.returncode != 0:
                raise Exception(f"GPU Operator install failed: {result.stderr}")

            return {
                "status": "installed",
                "cluster": cluster_name or self.config.cluster_name or "current",
                "namespace": namespace,
                "driver_version": driver_version or "auto-detect",
                "components": [
                    "driver",
                    "toolkit",
                    "device-plugin",
                    "dcgm-exporter",
                    "gfd",
                ],
            }
        except FileNotFoundError:
            return {"status": "failed", "error": "helm not found — install Helm first"}
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def configure_device_plugin(
        self,
        cluster_name: str = "",
        strategy: str = "none",
        replicas: int = 2,
    ) -> Dict[str, Any]:
        """Configure NVIDIA device plugin (MIG strategy + time-slicing)."""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            config_yaml = json.dumps(
                {
                    "version": "v1",
                    "flags": {"migStrategy": strategy},
                    "sharing": {
                        "timeSlicing": {
                            "renameByDefault": False,
                            "resources": [
                                {"name": "nvidia.com/gpu", "replicas": replicas}
                            ],
                        }
                    },
                }
            )

            # Apply as ConfigMap
            cm_manifest = (
                "apiVersion: v1\nkind: ConfigMap\nmetadata:\n"
                "  name: nvidia-device-plugin\n  namespace: gpu-operator\n"
                "data:\n  config.json: |\n"
            )
            for line in config_yaml.splitlines():
                cm_manifest += f"    {line}\n"

            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=cm_manifest,
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                raise Exception(f"ConfigMap apply failed: {result.stderr}")

            return {
                "status": "configured",
                "cluster": cluster_name or self.config.cluster_name or "current",
                "strategy": strategy,
                "replicas": replicas,
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def configure_mig(
        self,
        cluster_name: str = "",
        mig_profile: str = "all-1g.10gb",
        gpu_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Configure Multi-Instance GPU (MIG) partitioning on A100/H100."""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # MIG profiles: 1g.10gb, 2g.20gb, 3g.40gb, 4g.40gb, 7g.80gb (A100)

            # Label nodes with MIG config
            label_cmd = [
                "kubectl",
                "label",
                "nodes",
                "--all",
                f"nvidia.com/mig.config={mig_profile}",
                "--overwrite",
            ]
            if gpu_indices:
                label_cmd = [
                    "kubectl",
                    "label",
                    "nodes",
                    f"--selector=nvidia.com/gpu.count>={max(gpu_indices) + 1}",
                    f"nvidia.com/mig.config={mig_profile}",
                    "--overwrite",
                ]

            result = subprocess.run(
                label_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                raise Exception(f"MIG label failed: {result.stderr}")

            return {
                "status": "configured",
                "cluster": cluster_name or self.config.cluster_name or "current",
                "mig_profile": mig_profile,
                "gpu_indices": gpu_indices or "all",
                "note": "GPU Operator will apply MIG config on next device-plugin restart",
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def configure_time_slicing(
        self,
        cluster_name: str = "",
        replicas: int = 4,
        oversubscribe: bool = True,
    ) -> Dict[str, Any]:
        """Configure GPU time-slicing for multi-tenant pod sharing."""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            ts_config = {
                "version": "v1",
                "sharing": {
                    "timeSlicing": {
                        "renameByDefault": False,
                        "failRequestsGreaterThanOne": not oversubscribe,
                        "resources": [
                            {"name": "nvidia.com/gpu", "replicas": replicas},
                        ],
                    }
                },
            }

            cm_manifest = (
                "apiVersion: v1\nkind: ConfigMap\nmetadata:\n"
                "  name: time-slicing-config\n  namespace: gpu-operator\n"
                "data:\n  config.json: |\n"
            )
            for line in json.dumps(ts_config, indent=2).splitlines():
                cm_manifest += f"    {line}\n"

            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=cm_manifest,
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                raise Exception(f"Time-slicing ConfigMap apply failed: {result.stderr}")

            # Patch ClusterPolicy to use this ConfigMap
            patch_cmd = [
                "kubectl",
                "patch",
                "clusterpolicy/cluster-policy",
                "-n",
                "gpu-operator",
                "--type=merge",
                "-p",
                json.dumps(
                    {
                        "spec": {
                            "devicePlugin": {
                                "config": {
                                    "name": "time-slicing-config",
                                    "default": "config.json",
                                }
                            }
                        }
                    }
                ),
            ]
            patch_result = subprocess.run(
                patch_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            return {
                "status": "configured",
                "cluster": cluster_name or self.config.cluster_name or "current",
                "replicas_per_gpu": replicas,
                "oversubscribe": oversubscribe,
                "cluster_policy_patched": patch_result.returncode == 0,
                "note": f"Each physical GPU now appears as {replicas} virtual GPUs",
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def get_cluster_resources(self) -> Dict[str, Any]:
        """Get cluster resource information"""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # Get nodes with resource info
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )

            resources = {"total_cpu": 0, "total_memory": 0, "total_gpu": 0, "nodes": []}

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            node_name = parts[0]
                            cpu_cores = parts[1].replace("m", "")
                            memory = parts[2].replace("Mi", "")

                            try:
                                cpu_int = (
                                    int(cpu_cores) / 1000
                                    if "m" in parts[1]
                                    else int(cpu_cores)
                                )
                                mem_gb = int(memory) / 1024

                                resources["nodes"].append(
                                    {
                                        "name": node_name,
                                        "cpu_cores": cpu_int,
                                        "memory_gb": mem_gb,
                                    }
                                )

                                resources["total_cpu"] += cpu_int
                                resources["total_memory"] += mem_gb
                            except ValueError:
                                continue

            # Get GPU resources
            gpu_result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "nodes",
                    "-o",
                    'jsonpath=\'{range .items[*]}{{.metadata.name}}{{" "}}{{.status.capacity.nvidia.com/gpu}}{{"\\n"}}{end}\'',
                ],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )

            if gpu_result.returncode == 0:
                for line in gpu_result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                gpu_count = int(parts[1])
                                resources["total_gpu"] += gpu_count
                            except ValueError:
                                continue

            return resources

        except Exception as e:  # noqa: BLE001
            raise Exception(f"Failed to get cluster resources: {e}")

    def get_enhanced_config(self) -> Dict[str, str]:
        """Get enhanced Kubernetes configuration for environment variables"""
        config: Dict[str, str] = {}
        if self.config.kubeconfig_path:
            config["KUBECONFIG"] = self.config.kubeconfig_path
        if self.config.cluster_name:
            config["KUBERNETES_CLUSTER_NAME"] = self.config.cluster_name
        if self.config.namespace:
            config["KUBERNETES_NAMESPACE"] = self.config.namespace
        if self.config.aws_region:
            config["AWS_DEFAULT_REGION"] = self.config.aws_region


        return config

    # ---------------------------------------------------------------------------
    # DRA (Dynamic Resource Allocation) - K8s 1.32+ GA
    # ---------------------------------------------------------------------------

    async def install_dra_driver(self) -> Dict[str, Any]:
        """Install NVIDIA DRA driver for K8s 1.32+.

        DRA replaces device-plugin for GPU allocation.
        Falls back to device plugin if DRA not supported.
        """
        if not self.config.dra_enabled:
            return {
                "status": "skipped",
                "message": "DRA not enabled in configuration, using device plugin fallback",
            }

        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # Create ResourceClass for GPU
            resource_class = f"""apiVersion: resource.k8s.io/v1beta1
kind: ResourceClass
metadata:
  name: {self.config.dra_driver_name}
driverName: {self.config.dra_driver_name}
parameters:
  - name: "count"
    value: "1"
  - name: "type"
    value: "gpu"
suitableNodeCount: 1
"""

            # Apply ResourceClass
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=resource_class,
                text=True,
                timeout=30,
                env=env,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to apply ResourceClass: {result.stderr}")

            return {
                "status": "installed",
                "driver": self.config.dra_driver_name,
                "message": "DRA ResourceClass installed successfully",
            }

        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"DRA installation failed, falling back to device plugin: {e}"
            )
            # Fall back to device plugin
            return await self.install_device_plugin()

    async def install_device_plugin(self) -> Dict[str, Any]:
        """Install NVIDIA device plugin (fallback for pre-DRA K8s)."""
        if not self.config.device_plugin_enabled:
            return {"status": "skipped", "message": "Device plugin not enabled"}

        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # Install NVIDIA device plugin via Helm
            result = subprocess.run(
                [
                    "helm",
                    "repo",
                    "add",
                    "nvidia",
                    "https://nvidia.github.io/gpu-operator",
                    "--force-update",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to add NVIDIA Helm repo: {result.stderr}")

            result = subprocess.run(
                [
                    "helm",
                    "install",
                    "nvidia-device-plugin",
                    "nvidia/gpu-device-plugin",
                    "--namespace",
                    "kube-system",
                    "--create-namespace",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to install device plugin: {result.stderr}")

            return {
                "status": "installed",
                "message": "NVIDIA device plugin installed (DRA fallback)",
            }

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def configure_dra_mig(self) -> Dict[str, Any]:
        """Configure MIG (Multi-Instance GPU) via DRA."""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path

            # MIG ResourceClass for DRA
            mig_resource_class = """apiVersion: resource.k8s.io/v1beta1
kind: ResourceClass
metadata:
  name: nvidia.com/mig
driverName: nvidia.com/mig
parameters:
  - name: "mig-profile"
    value: "1g.5gb"
suitableNodeCount: 1
"""

            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=mig_resource_class,
                text=True,
                timeout=30,
                env=env,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to configure MIG: {result.stderr}")

            return {"status": "configured", "message": "MIG configured via DRA"}

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}


def create_enhanced_kubernetes_service_from_credentials(
    credentials: Dict[str, str]
) -> EnhancedKubernetesService:
    """Create enhanced KubernetesService from credential dictionary"""
    config = KubernetesConfig(
        kubeconfig_path=credentials.get("kubeconfig_path"),
        cluster_name=credentials.get("cluster_name"),
        namespace=credentials.get("namespace", "default"),
        karpenter_enabled=credentials.get("karpenter_enabled", "false").lower()
        == "true",
        karpenter_version=credentials.get("karpenter_version", "v1.10.0"),
        aws_region=credentials.get("aws_region", "us-east-1"),
        aws_account_id=credentials.get("aws_account_id"),
    )

    return EnhancedKubernetesService(config)


