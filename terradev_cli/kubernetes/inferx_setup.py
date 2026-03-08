#!/usr/bin/env python3
"""
InferX Kubernetes Setup Script
Deploy InferX serverless inference platform on Kubernetes
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import click
import yaml

from kubernetes import client, config
from kubernetes.client.rest import ApiException


class InferXK8sSetup:
    """InferX Kubernetes setup and management"""
    
    def __init__(self, kubeconfig: Optional[str] = None):
        """Initialize Kubernetes client"""
        try:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                config.load_kube_config()
        except:
            config.load_incluster_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.rbac_v1 = client.RbacAuthorizationV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.custom_api = client.CustomObjectsApi()
        
    async def deploy_inferx_platform(
        self, 
        namespace: str = "inferx",
        gpu_nodes: int = 2,
        gpu_type: str = "A100",
        snapshot_enabled: bool = True,
        gpu_slicing: bool = True,
        multi_tenant: bool = True
    ) -> Dict[str, Any]:
        """Deploy InferX platform on Kubernetes"""
        
        print(f"🚀 Deploying InferX platform on Kubernetes...")
        print(f"📦 Namespace: {namespace}")
        print(f"🎮 GPU Nodes: {gpu_nodes}")
        print(f"🎮 GPU Type: {gpu_type}")
        print(f"📸 Snapshot: {'Enabled' if snapshot_enabled else 'Disabled'}")
        print(f"🔪 GPU Slicing: {'Enabled' if gpu_slicing else 'Disabled'}")
        print(f"🏢 Multi-tenant: {'Enabled' if multi_tenant else 'Disabled'}")
        
        results = {
            "namespace": namespace,
            "deployments": [],
            "services": [],
            "storage_classes": [],
            "errors": []
        }
        
        try:
            # 1. Create namespace
            await self._create_namespace(namespace)
            results["namespace"] = namespace
            
            # 2. Deploy infrastructure
            infra_manifests = self._load_manifests("inferx-infrastructure.yaml")
            for manifest in infra_manifests:
                await self._apply_manifest(manifest, namespace)
            
            # 3. Deploy platform services
            platform_manifests = self._load_manifests("inferx-platform.yaml")
            for manifest in platform_manifests:
                await self._apply_manifest(manifest, namespace)
            
            # 4. Deploy model CRDs and examples
            model_manifests = self._load_manifests("inferx-models.yaml")
            for manifest in model_manifests:
                await self._apply_manifest(manifest, namespace)
            
            # 5. Wait for deployments to be ready
            await self._wait_for_deployments(namespace)
            
            # 6. Get service endpoints
            endpoints = await self._get_service_endpoints(namespace)
            results["endpoints"] = endpoints
            
            print(f"✅ InferX platform deployed successfully!")
            print(f"🌐 Gateway: {endpoints.get('inferx-gateway', 'N/A')}")
            print(f"📊 Dashboard: {endpoints.get('inferx-dashboard', 'N/A')}")
            
            return results
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            results["errors"].append(str(e))
            return results
    
    async def _create_namespace(self, namespace: str):
        """Create Kubernetes namespace"""
        try:
            ns = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=namespace,
                    labels={
                        "name": namespace,
                        "platform": "inferx-serverless"
                    }
                )
            )
            self.v1.create_namespace(body=ns)
            print(f"✅ Created namespace: {namespace}")
        except ApiException as e:
            if e.status == 409:
                print(f"ℹ️  Namespace {namespace} already exists")
            else:
                raise
    
    def _load_manifests(self, filename: str) -> List[Dict[str, Any]]:
        """Load Kubernetes manifests from file"""
        current_dir = Path(__file__).parent
        manifest_file = current_dir / filename
        
        with open(manifest_file, 'r') as f:
            documents = yaml.safe_load_all(f.read())
        
        manifests = []
        for doc in documents:
            if doc is not None:
                manifests.append(doc)
        
        return manifests
    
    async def _apply_manifest(self, manifest: Dict[str, Any], namespace: str):
        """Apply Kubernetes manifest"""
        api_version = manifest.get("apiVersion")
        kind = manifest.get("kind")
        name = manifest.get("metadata", {}).get("name")
        
        try:
            if api_version == "v1":
                if kind == "Namespace":
                    self.v1.create_namespace(body=manifest)
                elif kind == "Service":
                    self.v1.create_namespaced_service(namespace=namespace, body=manifest)
                elif kind == "Secret":
                    self.v1.create_namespaced_secret(namespace=namespace, body=manifest)
                elif kind == "ConfigMap":
                    self.v1.create_namespaced_config_map(namespace=namespace, body=manifest)
                elif kind == "PersistentVolumeClaim":
                    self.v1.create_namespaced_persistent_volume_claim(
                        namespace=namespace, body=manifest
                    )
                elif kind == "ResourceQuota":
                    self.v1.create_namespaced_resource_quota(namespace=namespace, body=manifest)
                elif kind == "LimitRange":
                    self.v1.create_namespaced_limit_range(namespace=namespace, body=manifest)
                elif kind == "ServiceAccount":
                    self.v1.create_namespaced_service_account(namespace=namespace, body=manifest)
                    
            elif api_version == "apps/v1":
                if kind == "Deployment":
                    self.apps_v1.create_namespaced_deployment(namespace=namespace, body=manifest)
                elif kind == "StatefulSet":
                    self.apps_v1.create_namespaced_stateful_set(namespace=namespace, body=manifest)
                elif kind == "DaemonSet":
                    self.apps_v1.create_namespaced_daemon_set(namespace=namespace, body=manifest)
                    
            elif api_version == "rbac.authorization.k8s.io/v1":
                if kind == "ClusterRole":
                    self.rbac_v1.create_cluster_role(body=manifest)
                elif kind == "ClusterRoleBinding":
                    self.rbac_v1.create_cluster_role_binding(body=manifest)
                    
            elif api_version == "networking.k8s.io/v1":
                if kind == "NetworkPolicy":
                    self.networking_v1.create_namespaced_network_policy(
                        namespace=namespace, body=manifest
                    )
                elif kind == "Ingress":
                    self.networking_v1.create_namespaced_ingress(
                        namespace=namespace, body=manifest
                    )
                    
            elif api_version == "storage.k8s.io/v1":
                storage_api = client.StorageV1Api()
                if kind == "StorageClass":
                    storage_api.create_storage_class(body=manifest)
                    
            elif api_version == "policy/v1":
                policy_api = client.PolicyV1Api()
                if kind == "PodDisruptionBudget":
                    policy_api.create_namespaced_pod_disruption_budget(
                        namespace=namespace, body=manifest
                    )
                    
            elif api_version == "apiextensions.k8s.io/v1":
                api_extensions = client.ApiextensionsV1Api()
                if kind == "CustomResourceDefinition":
                    api_extensions.create_custom_resource_definition(body=manifest)
                    
            elif api_version == "keda.sh/v1alpha1":
                # KEDA scaling (if available)
                pass
                
            elif api_version == "karpenter.sh/v1beta1":
                # Karpenter node pools (if available)
                pass
                
            elif api_version == "karpenter.k8s.aws/v1beta1":
                # Karpenter AWS node classes (if available)
                pass
                
            print(f"✅ Created {kind}/{name}")
            
        except ApiException as e:
            if e.status == 409:
                print(f"ℹ️  {kind}/{name} already exists")
            else:
                print(f"❌ Failed to create {kind}/{name}: {e}")
                raise
    
    async def _wait_for_deployments(self, namespace: str, timeout: int = 300):
        """Wait for all deployments to be ready"""
        print(f"⏳ Waiting for deployments to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)
                all_ready = True
                
                for deploy in deployments.items:
                    if deploy.status.ready_replicas != deploy.spec.replicas:
                        all_ready = False
                        break
                
                if all_ready:
                    print(f"✅ All deployments are ready")
                    return
                
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"⚠️  Error checking deployment status: {e}")
                await asyncio.sleep(10)
        
        raise Exception(f"Timeout waiting for deployments to be ready")
    
    async def _get_service_endpoints(self, namespace: str) -> Dict[str, str]:
        """Get service endpoints"""
        endpoints = {}
        
        try:
            services = self.v1.list_namespaced_service(namespace=namespace)
            
            for service in services.items:
                name = service.metadata.name
                port = service.spec.ports[0].port if service.spec.ports else 80
                
                if service.spec.type == "LoadBalancer":
                    # Wait for LoadBalancer IP
                    lb_ip = await self._get_loadbalancer_ip(name, namespace)
                    if lb_ip:
                        endpoints[name] = f"http://{lb_ip}:{port}"
                elif service.spec.type == "ClusterIP":
                    endpoints[name] = f"{name}.{namespace}.svc.cluster.local:{port}"
                else:
                    endpoints[name] = f"{name}:{port}"
        
        except Exception as e:
            print(f"⚠️  Error getting service endpoints: {e}")
        
        return endpoints
    
    async def _get_loadbalancer_ip(self, service_name: str, namespace: str) -> Optional[str]:
        """Get LoadBalancer IP address"""
        for i in range(30):  # Wait up to 5 minutes
            try:
                service = self.v1.read_namespaced_service(service_name, namespace)
                if service.status.load_balancer and service.status.load_balancer.ingress:
                    ingress = service.status.load_balancer.ingress[0]
                    return ingress.ip or ingress.hostname
            except:
                pass
            
            await asyncio.sleep(10)
        
        return None
    
    async def get_cluster_status(self, namespace: str = "inferx") -> Dict[str, Any]:
        """Get InferX cluster status"""
        try:
            status = {
                "namespace": namespace,
                "deployments": {},
                "pods": {},
                "services": {},
                "gpu_nodes": 0,
                "models_deployed": 0,
                "errors": []
            }
            
            # Get deployments
            deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)
            for deploy in deployments.items:
                status["deployments"][deploy.metadata.name] = {
                    "ready_replicas": deploy.status.ready_replicas or 0,
                    "desired_replicas": deploy.spec.replicas,
                    "status": "Ready" if deploy.status.ready_replicas == deploy.spec.replicas else "Not Ready"
                }
            
            # Get pods
            pods = self.v1.list_namespaced_pod(namespace=namespace)
            for pod in pods.items:
                pod_name = pod.metadata.name
                status["pods"][pod_name] = {
                    "phase": pod.status.phase,
                    "node": pod.spec.node_name,
                    "ready": any([c.ready for c in (pod.status.container_statuses or [])])
                }
            
            # Get services
            services = self.v1.list_namespaced_service(namespace=namespace)
            for service in services.items:
                status["services"][service.metadata.name] = {
                    "type": service.spec.type,
                    "cluster_ip": service.spec.cluster_ip,
                    "external_ip": self._get_service_external_ip(service)
                }
            
            # Get GPU nodes
            nodes = self.v1.list_node()
            for node in nodes.items:
                if self._is_gpu_node(node):
                    status["gpu_nodes"] += 1
            
            # Get model functions
            try:
                model_functions = self.custom_api.list_namespaced_custom_object(
                    group="inferx.io",
                    version="v1",
                    namespace=namespace,
                    plural="modelfunctions"
                )
                status["models_deployed"] = len(model_functions.get("items", []))
            except:
                pass
            
            return status
            
        except Exception as e:
            return {"error": str(e)}
    
    def _is_gpu_node(self, node) -> bool:
        """Check if node has GPU resources"""
        if not node.status.allocatable:
            return False
        
        return "nvidia.com/gpu" in node.status.allocatable
    
    def _get_service_external_ip(self, service) -> Optional[str]:
        """Get service external IP"""
        if service.status.load_balancer and service.status.load_balancer.ingress:
            ingress = service.status.load_balancer.ingress[0]
            return ingress.ip or ingress.hostname
        return None
    
    async def scale_platform(
        self, 
        namespace: str = "inferx",
        replicas: Optional[int] = None,
        gpu_nodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Scale InferX platform"""
        
        results = {"scaled": [], "errors": []}
        
        try:
            if replicas:
                # Scale platform deployment
                body = {"spec": {"replicas": replicas}}
                self.apps_v1.patch_namespaced_deployment_scale(
                    name="inferx-platform",
                    namespace=namespace,
                    body=body
                )
                results["scaled"].append(f"Platform scaled to {replicas} replicas")
            
            if gpu_nodes:
                # Scale GPU nodes (requires Karpenter or similar)
                print(f"📝 GPU node scaling requires Karpenter or cluster autoscaler")
                results["scaled"].append(f"GPU node scaling requested: {gpu_nodes}")
            
            return results
            
        except Exception as e:
            results["errors"].append(str(e))
            return results
    
    async def delete_platform(self, namespace: str = "inferx") -> Dict[str, Any]:
        """Delete InferX platform"""
        
        results = {"deleted": [], "errors": []}
        
        try:
            # Delete all resources in namespace
            self.v1.delete_namespace(name=namespace)
            results["deleted"].append(f"Namespace {namespace} deleted")
            
            return results
            
        except Exception as e:
            results["errors"].append(str(e))
            return results


# CLI Commands
@click.group()
def inferx_k8s():
    """InferX Kubernetes management commands"""
    pass


@inferx_k8s.command()
@click.option('--namespace', default='inferx', help='Kubernetes namespace')
@click.option('--gpu-nodes', default=2, help='Number of GPU nodes')
@click.option('--gpu-type', default='A100', help='GPU type')
@click.option('--snapshot/--no-snapshot', default=True, help='Enable snapshot technology')
@click.option('--gpu-slicing/--no-gpu-slicing', default=True, help='Enable GPU slicing')
@click.option('--multi-tenant/--no-multi-tenant', default=True, help='Enable multi-tenant isolation')
@click.option('--kubeconfig', help='Kubeconfig file path')
def deploy(namespace, gpu_nodes, gpu_type, snapshot, gpu_slicing, multi_tenant, kubeconfig):
    """Deploy InferX platform on Kubernetes"""
    
    setup = InferXK8sSetup(kubeconfig)
    
    try:
        results = asyncio.run(setup.deploy_inferx_platform(
            namespace=namespace,
            gpu_nodes=gpu_nodes,
            gpu_type=gpu_type,
            snapshot_enabled=snapshot,
            gpu_slicing=gpu_slicing,
            multi_tenant=multi_tenant
        ))
        
        print(f"\n🎉 Deployment Summary:")
        print(f"📦 Namespace: {results['namespace']}")
        print(f"🌐 Endpoints: {len(results.get('endpoints', {}))}")
        
        if results.get('errors'):
            print(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   - {error}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")


@inferx_k8s.command()
@click.option('--namespace', default='inferx', help='Kubernetes namespace')
@click.option('--kubeconfig', help='Kubeconfig file path')
def status(namespace, kubeconfig):
    """Get InferX cluster status"""
    
    setup = InferXK8sSetup(kubeconfig)
    
    try:
        status = asyncio.run(setup.get_cluster_status(namespace))
        
        print(f"📊 InferX Cluster Status")
        print(f"=" * 40)
        print(f"📦 Namespace: {status.get('namespace', 'Unknown')}")
        print(f"🎮 GPU Nodes: {status.get('gpu_nodes', 0)}")
        print(f"📦 Models Deployed: {status.get('models_deployed', 0)}")
        print()
        
        print(f"🚀 Deployments:")
        for name, info in status.get('deployments', {}).items():
            print(f"   {name}: {info['ready_replicas']}/{info['desired_replicas']} ({info['status']})")
        
        print(f"🌐 Services:")
        for name, info in status.get('services', {}).items():
            external_ip = f" ({info['external_ip']})" if info['external_ip'] else ""
            print(f"   {name}: {info['type']}{external_ip}")
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")


@inferx_k8s.command()
@click.option('--namespace', default='inferx', help='Kubernetes namespace')
@click.option('--replicas', type=int, help='Number of platform replicas')
@click.option('--gpu-nodes', type=int, help='Number of GPU nodes')
@click.option('--kubeconfig', help='Kubeconfig file path')
def scale(namespace, replicas, gpu_nodes, kubeconfig):
    """Scale InferX platform"""
    
    setup = InferXK8sSetup(kubeconfig)
    
    try:
        results = asyncio.run(setup.scale_platform(
            namespace=namespace,
            replicas=replicas,
            gpu_nodes=gpu_nodes
        ))
        
        print(f"📊 Scaling Results:")
        for item in results.get('scaled', []):
            print(f"   ✅ {item}")
        
        for error in results.get('errors', []):
            print(f"   ❌ {error}")
        
    except Exception as e:
        print(f"❌ Scaling failed: {e}")


@inferx_k8s.command()
@click.option('--namespace', default='inferx', help='Kubernetes namespace')
@click.option('--kubeconfig', help='Kubeconfig file path')
@click.confirmation_option(prompt='Are you sure you want to delete the InferX platform?')
def delete(namespace, kubeconfig):
    """Delete InferX platform"""
    
    setup = InferXK8sSetup(kubeconfig)
    
    try:
        results = asyncio.run(setup.delete_platform(namespace))
        
        print(f"📊 Deletion Results:")
        for item in results.get('deleted', []):
            print(f"   ✅ {item}")
        
        for error in results.get('errors', []):
            print(f"   ❌ {error}")
        
    except Exception as e:
        print(f"❌ Deletion failed: {e}")


if __name__ == '__main__':
    inferx_k8s()
