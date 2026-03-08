#!/usr/bin/env python3
"""
Ray Service Integration for Terradev
Manages Ray clusters, jobs, and distributed computing
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
class RayConfig:
    """Ray configuration"""
    dashboard_uri: Optional[str] = None
    cluster_name: Optional[str] = None
    auth_token: Optional[str] = None
    head_node_ip: Optional[str] = None
    head_node_port: int = 6379
    namespace: str = "default"


class RayService:
    """Ray integration service for distributed computing"""
    
    def __init__(self, config: RayConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        if self.config.auth_token:
            headers = {"Authorization": f"Bearer {self.config.auth_token}"}
            self.session = aiohttp.ClientSession(headers=headers)
        else:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Ray connection and get cluster status"""
        try:
            # Check if Ray is installed
            result = subprocess.run(
                ["ray", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": "Ray not installed. Run: pip install ray[default]"
                }
            
            # Extract Ray version
            ray_version = result.stdout.strip() if result.stdout else "unknown"
            
            # Try to get cluster status
            status_result = subprocess.run(
                ["ray", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if status_result.returncode == 0:
                return {
                    "status": "connected",
                    "ray_version": ray_version,
                    "cluster_name": self.config.cluster_name or "local",
                    "dashboard_uri": self.config.dashboard_uri,
                    "output": status_result.stdout
                }
            else:
                return {
                    "status": "not_connected",
                    "ray_version": ray_version,
                    "error": "Ray cluster not running. Start with 'ray start --head' or check connection.",
                    "suggestion": "Run: ray start --head"
                }
                
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "Ray not installed. Run: pip install ray[default]"
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed", 
                "error": "Ray command timed out. Check if Ray is properly installed."
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get detailed cluster status"""
        try:
            result = subprocess.run(
                ["ray", "status", "--details"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # Parse status output
                status_lines = result.stdout.strip().split('\n')
                status_info = {
                    "status": "running",
                    "details": status_lines
                }
                
                # Try to get node information
                try:
                    nodes_result = subprocess.run(
                        ["ray", "status", "--memory"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if nodes_result.returncode == 0:
                        status_info["memory_info"] = nodes_result.stdout.strip()
                except:
                    pass
                
                return status_info
            else:
                raise Exception(f"Failed to get cluster status: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to get cluster status: {e}")
    
    async def list_nodes(self) -> List[Dict[str, Any]]:
        """List all nodes in the cluster"""
        try:
            result = subprocess.run(
                ["ray", "status", "--memory"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # Parse node information
                nodes = []
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if 'node:' in line.lower() or 'node' in line.lower():
                        node_info = {
                            "info": line.strip(),
                            "type": "worker" if "worker" in line.lower() else "head"
                        }
                        nodes.append(node_info)
                
                return nodes
            else:
                return []
                
        except Exception as e:
            raise Exception(f"Failed to list nodes: {e}")
    
    async def submit_job(self, job_script: str, job_name: Optional[str] = None, resources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit a Ray job"""
        try:
            cmd = ["ray", "submit", job_script]
            
            if job_name:
                cmd.extend(["--name", job_name])
            
            if resources:
                for key, value in resources.items():
                    cmd.extend(["--resources", f"{key}={value}"])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "status": "submitted",
                    "job_name": job_name,
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to submit job: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to submit job: {e}")
    
    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List all running jobs"""
        try:
            result = subprocess.run(
                ["ray", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                jobs = []
                # Parse job information from status output
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if 'job' in line.lower() or 'task' in line.lower():
                        job_info = {
                            "info": line.strip(),
                            "status": "running"
                        }
                        jobs.append(job_info)
                
                return jobs
            else:
                return []
                
        except Exception as e:
            raise Exception(f"Failed to list jobs: {e}")
    
    async def get_dashboard_url(self) -> Optional[str]:
        """Get Ray dashboard URL"""
        try:
            if self.config.dashboard_uri:
                return self.config.dashboard_uri
            
            # Try to get dashboard URL from Ray status
            result = subprocess.run(
                ["ray", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Look for dashboard URL in output
                for line in result.stdout.strip().split('\n'):
                    if 'dashboard' in line.lower() and 'http' in line:
                        # Extract URL from line
                        import re
                        url_match = re.search(r'https?://[^\s]+', line)
                        if url_match:
                            return url_match.group(0)
            
            return None
            
        except Exception as e:
            raise Exception(f"Failed to get dashboard URL: {e}")
    
    async def start_cluster(self, head_node: bool = True, workers: int = 0, port: int = 6379) -> Dict[str, Any]:
        """Start a Ray cluster"""
        try:
            if head_node:
                # Start head node
                cmd = ["ray", "start", "--head", "--port", str(port)]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return {
                        "status": "head_started",
                        "port": port,
                        "output": result.stdout
                    }
                else:
                    raise Exception(f"Failed to start head node: {result.stderr}")
            
            elif workers > 0:
                # Start worker nodes
                cmd = ["ray", "start", "--address", f"{self.config.head_node_ip}:{port}"]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return {
                        "status": "worker_started",
                        "head_address": f"{self.config.head_node_ip}:{port}",
                        "output": result.stdout
                    }
                else:
                    raise Exception(f"Failed to start worker: {result.stderr}")
            
            else:
                raise Exception("Must specify either head_node=True or workers > 0")
                
        except Exception as e:
            raise Exception(f"Failed to start cluster: {e}")
    
    async def stop_cluster(self) -> Dict[str, Any]:
        """Stop Ray cluster"""
        try:
            result = subprocess.run(
                ["ray", "stop"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "status": "stopped",
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to stop cluster: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to stop cluster: {e}")
    
    async def get_cluster_resources(self) -> Dict[str, Any]:
        """Get cluster resource information"""
        try:
            result = subprocess.run(
                ["ray", "status", "--memory"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # Parse resource information
                resources = {
                    "memory_info": result.stdout.strip(),
                    "status": "available"
                }
                
                # Try to get more detailed resource info
                try:
                    detailed_result = subprocess.run(
                        ["ray", "status", "--details"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if detailed_result.returncode == 0:
                        resources["details"] = detailed_result.stdout.strip()
                except:
                    pass
                
                return resources
            else:
                raise Exception(f"Failed to get cluster resources: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to get cluster resources: {e}")
    
    def get_ray_config(self) -> Dict[str, str]:
        """Get Ray configuration for Python scripts"""
        config = {}
        
        if self.config.dashboard_uri:
            config["RAY_DASHBOARD_URI"] = self.config.dashboard_uri
            
        if self.config.cluster_name:
            config["RAY_CLUSTER_NAME"] = self.config.cluster_name
            
        if self.config.auth_token:
            config["RAY_AUTH_TOKEN"] = self.config.auth_token
            
        if self.config.head_node_ip:
            config["RAY_HEAD_NODE_IP"] = self.config.head_node_ip
            
        if self.config.head_node_port:
            config["RAY_HEAD_NODE_PORT"] = str(self.config.head_node_port)
            
        return config
    
    async def get_job_submission_command(self, script_path: str, job_name: Optional[str] = None, resources: Optional[Dict[str, Any]] = None) -> str:
        """Generate Ray job submission command"""
        cmd = ["ray", "submit", script_path]
        
        if job_name:
            cmd.extend(["--name", job_name])
        
        if resources:
            for key, value in resources.items():
                cmd.extend(["--resources", f"{key}={value}"])
        
        return " ".join(cmd)


def create_ray_service_from_credentials(credentials: Dict[str, str]) -> RayService:
    """Create RayService from credential dictionary"""
    config = RayConfig(
        dashboard_uri=credentials.get("dashboard_uri"),
        cluster_name=credentials.get("cluster_name"),
        auth_token=credentials.get("auth_token"),
        head_node_ip=credentials.get("head_node_ip"),
        head_node_port=int(credentials.get("head_node_port", 6379)),
        namespace=credentials.get("namespace", "default")
    )
    
    return RayService(config)


def get_ray_setup_instructions() -> str:
    """Get setup instructions for Ray"""
    return """
🚀 Ray Setup Instructions:

1. Install Ray:
   # Basic installation
   pip install ray[default]
   
   # For distributed training and ML workloads
   pip install ray[default,train]
   
   # For full ML ecosystem
   pip install ray[default,train,tune]

2. Start a Ray cluster:

   Local cluster (single machine):
   ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

   Multi-node cluster:
   # Head node
   ray start --head --port=6379 --dashboard-host=0.0.0.0
   
   # Worker nodes
   ray start --address=HEAD_NODE_IP:6379

3. Start Ray with dashboard:
   ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

4. Configure Terradev with your Ray credentials:
   terradev configure --provider ray --dashboard-uri http://localhost:8265 --head-node-ip localhost

📋 Required Credentials:
- dashboard_uri: Ray dashboard URI (optional)
- cluster_name: Cluster name (optional)
- auth_token: Authentication token (optional)
- head_node_ip: Head node IP address (optional)
- head_node_port: Head node port (default: 6379)
- namespace: Kubernetes namespace (default: "default")

💡 Usage Examples:
# Test Ray connection
terradev ml ray --test

# Get cluster status
terradev ml ray --status

# List nodes
terradev ml ray --list-nodes

# Start a cluster
terradev ml ray --start --head --port 6379

# Submit a job
terradev ml ray --submit --job-script my_script.py --job-name my-job

# List running jobs
terradev ml ray --list-jobs

# Get dashboard URL
terradev ml ray --dashboard

# Stop cluster
terradev ml ray --stop

# Get cluster resources
terradev ml ray --resources

🔗 Integration with Terradev:
Ray can be used alongside Terradev's provisioning:
- Provision GPU instances with Terradev
- Start Ray clusters on provisioned instances
- Scale distributed training with Ray
- Monitor jobs through Ray dashboard

🐳 Docker Ray Cluster:
# Head node
docker run -p 6379:6379 -p 8265:8265 rayproject/ray:latest ray start --head --dashboard-host=0.0.0.0

# Worker node
docker run rayproject/ray:latest ray start --address=HEAD_NODE_IP:6379

☸️ Kubernetes Ray:
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: ray-cluster
spec:
  rayVersion: '2.53.0'
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      port: '6379'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.53.0
          ports:
          - containerPort: 6379
          - containerPort: 8265

📝 Example Ray Script:
import ray

@ray.remote
def compute_task(data):
    # Your distributed computation
    return sum(data)

# Connect to Ray cluster
ray.init(address="ray://head-node:6379")

# Submit task
futures = [compute_task.remote([1, 2, 3, 4, 5]) for _ in range(10)]
results = ray.get(futures)

print(f"Results: {results}")

🎯 Terradev + Ray Workflow:
# 1. Provision GPUs
terradev provision -g A100 -n 4

# 2. Start Ray cluster
terradev ml ray --start

# 3. Submit distributed training
ray submit train.py --resources '{"GPU": 4}'

# 4. Monitor progress
terradev ml ray --dashboard
"""
