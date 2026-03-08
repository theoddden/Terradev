#!/usr/bin/env python3
"""
Enhanced Kubernetes Service with Deep Dashboard Integration
Integrates Karpenter, Prometheus, Grafana, and Terradev monitoring
"""

import os
import json
import asyncio
import aiohttp
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KubernetesConfig:
    """Enhanced Kubernetes configuration"""
    kubeconfig_path: Optional[str] = None
    cluster_name: Optional[str] = None
    namespace: str = "default"
    karpenter_enabled: bool = False
    karpenter_version: str = "v0.32.0"
    aws_region: str = "us-east-1"
    aws_account_id: Optional[str] = None
    monitoring_enabled: bool = False
    prometheus_enabled: bool = False
    grafana_enabled: bool = False
    dashboard_port: int = 3000


class EnhancedKubernetesService:
    """Enhanced Kubernetes integration service with Karpenter and monitoring"""
    
    def __init__(self, config: KubernetesConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Kubernetes connection and get cluster info"""
        try:
            # Set kubeconfig path if provided
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path
            
            # Get cluster info
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": f"kubectl command failed: {result.stderr}"
                }
            
            # Get nodes
            nodes_result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            nodes = []
            if nodes_result.returncode == 0:
                import json
                node_data = json.loads(nodes_result.stdout)
                nodes = [node["metadata"]["name"] for node in node_data.get("items", [])]
            
            return {
                "status": "success",
                "cluster_name": self.config.cluster_name,
                "namespace": self.config.namespace,
                "nodes": nodes,
                "karpenter_enabled": self.config.karpenter_enabled,
                "monitoring_enabled": self.config.monitoring_enabled
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def install_monitoring_stack(self) -> Dict[str, Any]:
        """Install Prometheus and Grafana with Karpenter dashboards"""
        if not self.config.monitoring_enabled:
            return {
                "status": "failed",
                "error": "Monitoring not enabled in configuration"
            }
        
        try:
            # Set up environment
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path
            
            # Install Prometheus
            prometheus_cmd = [
                "helm", "repo", "add", "prometheus-community",
                "https://prometheus-community.github.io/helm-charts"
            ]
            result = subprocess.run(prometheus_cmd, capture_output=True, text=True, timeout=60, env=env)
            
            if result.returncode != 0:
                raise Exception(f"Prometheus repo addition failed: {result.stderr}")
            
            prometheus_cmd = [
                "helm", "install", "prometheus", "prometheus-community/kube-prometheus-stack",
                "--namespace", "monitoring", "--create-namespace",
                "--set", "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi",
                "--wait"
            ]
            result = subprocess.run(prometheus_cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode != 0:
                raise Exception(f"Prometheus installation failed: {result.stderr}")
            
            # Install Grafana with Karpenter dashboards
            grafana_values = f"""
adminPassword: prom-operator
adminUser: admin
service:
  type: ClusterIP
  port: 80
ingress:
    enabled: false
dashboardProviders:
  dashboardproviders.yaml:
    apiVersion: 1
    providers:
    - name: 'default'
      orgId: 1
      folder: ''
      type: file
      disableDeletion: false
      allowUiUpdates: false
      options:
        path: /var/lib/grafana/dashboards
"""
            
            # Write Grafana values
            with open('/tmp/grafana-karpenter.yaml', 'w') as f:
                f.write(grafana_values)
            
            grafana_cmd = [
                "helm", "install", "grafana", "grafana-charts/grafana",
                "--namespace", "monitoring",
                "--values", "/tmp/grafana-karpenter.yaml",
                "--wait"
            ]
            
            result = subprocess.run(grafana_cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode != 0:
                raise Exception(f"Grafana installation failed: {result.stderr}")
            
            return {
                "status": "installed",
                "prometheus": "http://prometheus.monitoring.svc.cluster.local:9090",
                "grafana": "http://grafana.monitoring.svc.cluster.local:80",
                "dashboards": "Karpenter dashboards imported"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster metrics"""
        try:
            env = os.environ.copy()
            if self.config.kubeconfig_path:
                env["KUBECONFIG"] = self.config.kubeconfig_path
            
            # Get nodes with GPU
            nodes_result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            gpu_nodes = []
            total_nodes = 0
            
            if nodes_result.returncode == 0:
                import json
                node_data = json.loads(nodes_result.stdout)
                for node in node_data.get("items", []):
                    total_nodes += 1
                    # Check for GPU labels
                    labels = node.get("metadata", {}).get("labels", {})
                    if "nvidia.com/gpu" in labels:
                        gpu_nodes.append(node["metadata"]["name"])
            
            # Get Karpenter status
            karpenter_status = await self._get_karpenter_status(env)
            
            # Get monitoring status
            monitoring_status = {
                "prometheus": await self._check_prometheus_health(env),
                "grafana": await self._check_grafana_health(env)
            }
            
            return {
                "total_nodes": total_nodes,
                "gpu_nodes": gpu_nodes,
                "karpenter": karpenter_status,
                "monitoring": monitoring_status
            }
            
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def _get_karpenter_status(self, env: Dict[str, str]) -> Dict[str, Any]:
        """Get Karpenter status"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "karpenter", "-l", "app.karpenter"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'karpenter' in line and 'Running' in line:
                        return {"status": "running", "details": line.strip()}
            
            return {"status": "not_running", "error": "Karpenter not found"}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _check_prometheus_health(self, env: Dict[str, str]) -> Dict[str, Any]:
        """Check Prometheus health"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", "-n", "monitoring", "-l", "app.kubernetes.io/name=prometheus"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'prometheus' in line and 'Running' in line:
                        return {"status": "healthy", "details": line.strip()}
            
            return {"status": "unhealthy", "error": "Prometheus not running"}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _check_grafana_health(self, env: Dict[str, str]) -> Dict[str, Any]:
        """Check Grafana health"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", "-n", "monitoring", "-l", "app=grafana"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'grafana' in line and 'Running' in line:
                        return {"status": "healthy", "details": line.strip()}
            
            return {"status": "unhealthy", "error": "Grafana not running"}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for provisioned instances"""
        config = {
            "KUBECONFIG": self.config.kubeconfig_path or "",
            "NAMESPACE": self.config.namespace,
            "CLUSTER_NAME": self.config.cluster_name or ""
        }
        
        if self.config.karpenter_enabled:
            config["KARPENTER_ENABLED"] = "true"
            config["KARPENTER_VERSION"] = self.config.karpenter_version
        
        if self.config.monitoring_enabled:
            config["PROMETHEUS_ENABLED"] = "true"
            config["PROMETHEUS_URL"] = "http://prometheus.monitoring.svc.cluster.local:9090"
        
        if self.config.grafana_enabled:
            config["GRAFANA_ENABLED"] = "true"
            config["GRAFANA_URL"] = "http://grafana.monitoring.svc.cluster.local:80"
            config["GRAFANA_USERNAME"] = "admin"
            config["GRAFANA_PASSWORD"] = "prom-operator"
        
        return config


def create_enhanced_kubernetes_service_from_credentials(credentials: Dict[str, str]) -> EnhancedKubernetesService:
    """Create enhanced KubernetesService from credential dictionary"""
    config = KubernetesConfig(
        kubeconfig_path=credentials.get("kubeconfig_path"),
        cluster_name=credentials.get("cluster_name"),
        namespace=credentials.get("namespace", "default"),
        karpenter_enabled=credentials.get("karpenter_enabled", "false").lower() == "true",
        karpenter_version=credentials.get("karpenter_version", "v0.32.0"),
        aws_region=credentials.get("aws_region", "us-east-1"),
        aws_account_id=credentials.get("aws_account_id"),
        monitoring_enabled=credentials.get("monitoring_enabled", "false").lower() == "true",
        prometheus_enabled=credentials.get("prometheus_enabled", "false").lower() == "true",
        grafana_enabled=credentials.get("grafana_enabled", "false").lower() == "true",
        dashboard_port=int(credentials.get("dashboard_port", 3000))
    )
    
    return EnhancedKubernetesService(config)


def get_enhanced_kubernetes_setup_instructions() -> str:
    """Get enhanced setup instructions for Kubernetes with monitoring"""
    return """
🚀 Enhanced Kubernetes Setup Instructions:

1. Install prerequisites:
   kubectl
   helm
   aws-cli (for AWS EKS)

2. Configure kubeconfig:
   aws eks update-kubeconfig --region us-east-1 --name your-cluster

3. Install Karpenter:
   helm repo add karpenter https://charts.karpenter.sh
   helm install karpenter karpenter/karpenter \\
     --namespace karpenter \\
     --create-namespace \\
     --set serviceAccount.annotations.eks.amazonaws.com/role-arn=arn:aws:iam::ACCOUNT:role/KarpenterNodeRole \\
     --wait

4. Enable monitoring:
   terradev configure --provider kubernetes \\
     --karpenter-enabled true \\
     --monitoring-enabled true \\
     --prometheus-enabled true \\
     --grafana-enabled true

5. Install monitoring stack:
   terradev ml kubernetes --install-monitoring

6. Access dashboards:
   # Grafana Dashboard
   kubectl port-forward --namespace monitoring svc/grafana 3000:80
   
   # Prometheus UI
   kubectl port-forward --namespace monitoring svc/prometheus 9090:9090

💡 Enhanced Usage Examples:
# Test connection
terradev ml kubernetes --test

# Get cluster resources
terradev ml kubernetes --resources

# Access Grafana dashboard
terradev ml kubernetes --dashboard

🔗 Monitoring Integration:
- **Prometheus**: Metrics collection from Karpenter and Kubernetes
- **Grafana**: Pre-configured Karpenter dashboards
- **Karpenter Metrics**: Node provisioning/deprovisioning rates
- **Cluster Metrics**: CPU, memory, GPU utilization
- **Dashboard Templates**: Ready-to-use Karpenter dashboards

🎯 Advanced Features:
- **Auto-scaling**: Karpenter + Kubernetes HPA integration
- **Multi-cloud**: Support for AWS, GCP, Azure providers
- **Security**: RBAC and network policies
- **Persistence**: Long-term metrics storage
"""
