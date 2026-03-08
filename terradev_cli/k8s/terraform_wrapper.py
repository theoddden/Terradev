#!/usr/bin/env python3
"""
Kubernetes Terraform Wrapper for Terradev
Handles multi-cloud GPU cluster provisioning and management
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add terradev_cli to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.telemetry import MandatoryTelemetryClient

class TerraformWrapper:
    def __init__(self):
        self.config = Config()
        self.telemetry = MandatoryTelemetryClient()
        self.terraform_dir = Path(__file__).parent.parent / "terraform"
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _run_terraform_command(self, command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run Terraform command with proper error handling"""
        work_dir = cwd or self.terraform_dir
        
        self.logger.info(f"Running Terraform command: {' '.join(command)} in {work_dir}")
        
        try:
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Terraform command failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            
            self.logger.info(f"Terraform command completed successfully")
            return result
            
        except subprocess.TimeoutExpired:
            self.logger.error("Terraform command timed out")
            raise
        except Exception as e:
            self.logger.error(f"Error running Terraform command: {e}")
            raise
    
    def _load_credentials(self) -> Dict:
        """Load cloud provider credentials"""
        credentials_file = Path.home() / ".terradev" / "credentials.json"
        
        if not credentials_file.exists():
            raise FileNotFoundError(f"Credentials file not found: {credentials_file}")
        
        with open(credentials_file, 'r') as f:
            return json.load(f)
    
    def _create_terraform_vars(self, cluster_config: Dict) -> Dict:
        """Create Terraform variables from cluster configuration.

        Topology optimization is applied automatically:
          - Kubelet Topology Manager set to 'restricted' with prefer-closest-numa-nodes
          - GPUDirect RDMA enabled via nvidia_peermem kernel module in userData
          - SR-IOV enabled when multi-node GPU clusters are requested
          - NCCL environment variables injected for optimal GPU-NIC pairing
        """
        credentials = self._load_credentials()

        node_count = cluster_config["node_count"]
        gpu_type = cluster_config["gpu_type"]

        # Topology-aware defaults — users never need to set these
        topology_config = {
            "topology_manager_policy": "restricted",
            "topology_manager_scope": "container",
            "prefer_closest_numa": True,
            "cpu_manager_policy": "static",
            "enable_gpudirect_rdma": True,
            "enable_sriov": node_count > 1,
            "nccl_ib_disable": "0",
            "nccl_net_gdr_read": "1",
            "nccl_net_gdr_level": "PIX",
        }

        vars = {
            "cluster_name": cluster_config["name"],
            "gpu_type": gpu_type,
            "total_nodes": node_count,
            "max_price_per_hour": cluster_config.get("max_price", 4.00),
            "prefer_spot": cluster_config.get("prefer_spot", True),
            "control_plane_type": cluster_config.get("control_plane", "eks"),
            "aws_region": cluster_config.get("aws_region", "us-west-2"),
            "gcp_region": cluster_config.get("gcp_region", "us-central1"),
            "gcp_project": cluster_config.get("gcp_project", ""),
            "tailscale_authkey": credentials.get("tailscale", {}).get("auth_key", ""),
            "kubernetes_version": cluster_config.get("k8s_version", "1.28"),
            "enable_monitoring": cluster_config.get("monitoring", True),
            "enable_logging": cluster_config.get("logging", True),
            # Topology optimization (auto-applied)
            "topology_manager_policy": topology_config["topology_manager_policy"],
            "topology_manager_scope": topology_config["topology_manager_scope"],
            "prefer_closest_numa": topology_config["prefer_closest_numa"],
            "cpu_manager_policy": topology_config["cpu_manager_policy"],
            "enable_gpudirect_rdma": topology_config["enable_gpudirect_rdma"],
            "enable_sriov": topology_config["enable_sriov"],
            "nccl_ib_disable": topology_config["nccl_ib_disable"],
            "nccl_net_gdr_read": topology_config["nccl_net_gdr_read"],
            "nccl_net_gdr_level": topology_config["nccl_net_gdr_level"],
            "tags": {
                "Project": "Terradev",
                "Cluster": cluster_config["name"],
                "GPUType": gpu_type,
                "Environment": cluster_config.get("environment", "production"),
                "TopologyOptimized": "true",
            }
        }
        
        return vars
    
    def _write_terraform_vars(self, vars: Dict) -> Path:
        """Write Terraform variables to file"""
        vars_file = self.terraform_dir / "terraform.tfvars"
        
        with open(vars_file, 'w') as f:
            for key, value in vars.items():
                if isinstance(value, bool):
                    f.write(f'{key} = {"true" if value else "false"}\n')
                elif isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                elif isinstance(value, (int, float)):
                    f.write(f'{key} = {value}\n')
                elif isinstance(value, dict):
                    f.write(f'{key} = {json.dumps(value)}\n')
                else:
                    f.write(f'{key} = {value}\n')
        
        self.logger.info(f"Terraform variables written to {vars_file}")
        return vars_file
    
    def init_terraform(self) -> bool:
        """Initialize Terraform"""
        try:
            self._run_terraform_command(["terraform", "init"])
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Terraform: {e}")
            return False
    
    def create_cluster(self, cluster_config: Dict) -> bool:
        """Create Kubernetes cluster"""
        try:
            # Log telemetry
            self.telemetry.log_action('k8s_cluster_create', {
                'cluster_name': cluster_config['name'],
                'gpu_type': cluster_config['gpu_type'],
                'node_count': cluster_config['node_count'],
                'multi_cloud': cluster_config.get('multi_cloud', False),
                'max_price': cluster_config.get('max_price', 4.00),
                'prefer_spot': cluster_config.get('prefer_spot', True)
            })
            
            # Write Terraform variables
            vars = self._create_terraform_vars(cluster_config)
            self._write_terraform_vars(vars)
            
            # Initialize Terraform
            if not self.init_terraform():
                return False
            
            # Plan and apply
            self.logger.info("Planning Terraform deployment...")
            self._run_terraform_command(["terraform", "plan", "-out=cluster.plan"])
            
            self.logger.info("Applying Terraform configuration...")
            result = self._run_terraform_command(["terraform", "apply", "cluster.plan", "-auto-approve"])
            
            # Parse outputs
            outputs = self._parse_terraform_outputs(result.stdout)
            
            # Save cluster info
            self._save_cluster_info(cluster_config["name"], outputs)
            
            self.logger.info(f"Cluster {cluster_config['name']} created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster: {e}")
            return False
    
    def destroy_cluster(self, cluster_name: str) -> bool:
        """Destroy Kubernetes cluster"""
        try:
            # Log telemetry
            self.telemetry.log_action('k8s_cluster_destroy', {
                'cluster_name': cluster_name
            })
            
            # Load cluster config
            cluster_config = self._load_cluster_config(cluster_name)
            vars = self._create_terraform_vars(cluster_config)
            self._write_terraform_vars(vars)
            
            # Destroy
            self.logger.info(f"Destroying cluster {cluster_name}...")
            self._run_terraform_command(["terraform", "destroy", "-auto-approve"])
            
            # Remove cluster info
            self._remove_cluster_info(cluster_name)
            
            self.logger.info(f"Cluster {cluster_name} destroyed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy cluster: {e}")
            return False
    
    def get_cluster_info(self, cluster_name: str) -> Optional[Dict]:
        """Get cluster information"""
        try:
            cluster_file = Path.home() / ".terradev" / "clusters" / f"{cluster_name}.json"
            
            if not cluster_file.exists():
                return None
            
            with open(cluster_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to get cluster info: {e}")
            return None
    
    def list_clusters(self) -> List[Dict]:
        """List all clusters"""
        try:
            clusters_dir = Path.home() / ".terradev" / "clusters"
            
            if not clusters_dir.exists():
                return []
            
            clusters = []
            for cluster_file in clusters_dir.glob("*.json"):
                try:
                    with open(cluster_file, 'r') as f:
                        cluster_info = json.load(f)
                        clusters.append(cluster_info)
                except Exception as e:
                    self.logger.warning(f"Failed to read cluster file {cluster_file}: {e}")
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to list clusters: {e}")
            return []
    
    def _parse_terraform_outputs(self, output: str) -> Dict:
        """Parse Terraform outputs"""
        outputs = {}
        
        try:
            # Run terraform output to get structured data
            result = self._run_terraform_command(["terraform", "output", "-json"])
            outputs = json.loads(result.stdout)
        except Exception as e:
            self.logger.warning(f"Failed to parse Terraform outputs: {e}")
        
        return outputs
    
    def _save_cluster_info(self, cluster_name: str, outputs: Dict) -> None:
        """Save cluster information"""
        try:
            clusters_dir = Path.home() / ".terradev" / "clusters"
            clusters_dir.mkdir(parents=True, exist_ok=True)
            
            cluster_file = clusters_dir / f"{cluster_name}.json"
            
            cluster_info = {
                "name": cluster_name,
                "created_at": subprocess.check_output(["date", "-I"]).decode().strip(),
                "outputs": outputs,
                "status": "active"
            }
            
            with open(cluster_file, 'w') as f:
                json.dump(cluster_info, f, indent=2)
            
            self.logger.info(f"Cluster info saved to {cluster_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cluster info: {e}")
    
    def _load_cluster_config(self, cluster_name: str) -> Dict:
        """Load cluster configuration"""
        cluster_info = self.get_cluster_info(cluster_name)
        
        if not cluster_info:
            raise FileNotFoundError(f"Cluster {cluster_name} not found")
        
        # Reconstruct basic config from saved info
        outputs = cluster_info.get("outputs", {})
        
        return {
            "name": cluster_name,
            "gpu_type": outputs.get("gpu_summary", {}).get("gpu_type", "H100"),
            "node_count": outputs.get("total_nodes", 1),
            "max_price": outputs.get("gpu_summary", {}).get("max_price", 4.00),
            "prefer_spot": True
        }
    
    def _remove_cluster_info(self, cluster_name: str) -> None:
        """Remove cluster information"""
        try:
            cluster_file = Path.home() / ".terradev" / "clusters" / f"{cluster_name}.json"
            
            if cluster_file.exists():
                cluster_file.unlink()
                self.logger.info(f"Cluster info removed for {cluster_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to remove cluster info: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Terradev Kubernetes Terraform Wrapper")
    parser.add_argument("command", choices=["create", "destroy", "list", "info"], help="Command to execute")
    parser.add_argument("--cluster", help="Cluster name")
    parser.add_argument("--gpu", help="GPU type (H100, A100, L40)")
    parser.add_argument("--count", type=int, help="Number of nodes")
    parser.add_argument("--max-price", type=float, help="Maximum price per hour")
    parser.add_argument("--multi-cloud", action="store_true", help="Use multi-cloud provisioning")
    parser.add_argument("--prefer-spot", action="store_true", help="Prefer spot instances")
    parser.add_argument("--aws-region", help="AWS region")
    parser.add_argument("--gcp-region", help="GCP region")
    parser.add_argument("--control-plane", choices=["eks", "gke", "self-hosted"], help="Control plane type")
    
    args = parser.parse_args()
    
    wrapper = TerraformWrapper()
    
    try:
        if args.command == "create":
            if not args.cluster or not args.gpu or not args.count:
                print("Error: --cluster, --gpu, and --count are required for create command")
                sys.exit(1)
            
            cluster_config = {
                "name": args.cluster,
                "gpu_type": args.gpu,
                "node_count": args.count,
                "max_price": args.max_price or 4.00,
                "multi_cloud": args.multi_cloud,
                "prefer_spot": args.prefer_spot,
                "aws_region": args.aws_region,
                "gcp_region": args.gcp_region,
                "control_plane": args.control_plane
            }
            
            success = wrapper.create_cluster(cluster_config)
            sys.exit(0 if success else 1)
            
        elif args.command == "destroy":
            if not args.cluster:
                print("Error: --cluster is required for destroy command")
                sys.exit(1)
            
            success = wrapper.destroy_cluster(args.cluster)
            sys.exit(0 if success else 1)
            
        elif args.command == "list":
            clusters = wrapper.list_clusters()
            
            if not clusters:
                print("No clusters found")
            else:
                print("Available clusters:")
                for cluster in clusters:
                    print(f"  - {cluster['name']} ({cluster.get('status', 'unknown')})")
            
        elif args.command == "info":
            if not args.cluster:
                print("Error: --cluster is required for info command")
                sys.exit(1)
            
            info = wrapper.get_cluster_info(args.cluster)
            
            if not info:
                print(f"Cluster {args.cluster} not found")
                sys.exit(1)
            
            print(f"Cluster: {info['name']}")
            print(f"Status: {info.get('status', 'unknown')}")
            print(f"Created: {info.get('created_at', 'unknown')}")
            
            outputs = info.get('outputs', {})
            if outputs:
                print("\nOutputs:")
                for key, value in outputs.items():
                    print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
