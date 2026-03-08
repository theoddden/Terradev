#!/usr/bin/env python3
"""
Terradev CLI - Terraform-based Parallel GPU Provisioning
Leverages Terraform for IaC with parallel optimization and provenance tracking
"""

import argparse
import asyncio
import json
import os
import sys
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import subprocess
import tempfile
import shutil

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "parallel_orchestration"))

# Import our parallel engine
try:
    from parallel_provisioning_engine import ParallelProvisioningEngine, ProviderRequest, InstanceType, Provider
    print("✅ Successfully imported parallel provisioning engine")
except ImportError as e:
    print(f"❌ Error importing parallel_provisioning_engine: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}...")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProviderCredentials:
    """Provider credentials configuration"""
    provider: str
    credentials: Dict[str, str]
    enabled: bool = True
    last_validated: Optional[datetime] = None

@dataclass
class ProvenanceRecord:
    """Provenance tracking record"""
    operation_id: str
    operation_type: str
    timestamp: datetime
    user_id: str
    team_id: str
    project_id: str
    providers_queried: List[str]
    selected_provider: str
    configuration: Dict[str, Any]
    terraform_code: str
    cost_estimate: float
    security_hash: str
    compliance_flags: List[str]

class TerradevCLI:
    """Main CLI application"""
    
    def __init__(self):
        self.engine = ParallelProvisioningEngine()
        self.credentials = {}
        self.provenance_records = []
        self.config_dir = Path.home() / ".terradev"
        self.credentials_file = self.config_dir / "credentials.json"
        self.provenance_file = self.config_dir / "provenance.json"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load existing configuration
        self.load_credentials()
        self.load_provenance()
    
    def load_credentials(self):
        """Load provider credentials from file"""
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    data = json.load(f)
                    self.credentials = {
                        provider: ProviderCredentials(
                            provider=provider,
                            credentials=creds['credentials'],
                            enabled=creds.get('enabled', True),
                            last_validated=datetime.fromisoformat(creds['last_validated']) if creds.get('last_validated') else None
                        )
                        for provider, creds in data.items()
                    }
                logger.info(f"Loaded credentials for {len(self.credentials)} providers")
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self.credentials = {}
    
    def save_credentials(self):
        """Save provider credentials to file"""
        try:
            data = {
                provider: {
                    'credentials': creds.credentials,
                    'enabled': creds.enabled,
                    'last_validated': creds.last_validated.isoformat() if creds.last_validated else None
                }
                for provider, creds in self.credentials.items()
            }
            
            with open(self.credentials_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Set secure permissions
            os.chmod(self.credentials_file, 0o600)
            logger.info("Credentials saved successfully")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def load_provenance(self):
        """Load provenance records from file"""
        if self.provenance_file.exists():
            try:
                with open(self.provenance_file, 'r') as f:
                    data = json.load(f)
                    self.provenance_records = [
                        ProvenanceRecord(
                            operation_id=record['operation_id'],
                            operation_type=record['operation_type'],
                            timestamp=datetime.fromisoformat(record['timestamp']),
                            user_id=record['user_id'],
                            team_id=record['team_id'],
                            project_id=record['project_id'],
                            providers_queried=record['providers_queried'],
                            selected_provider=record['selected_provider'],
                            configuration=record['configuration'],
                            terraform_code=record['terraform_code'],
                            cost_estimate=record['cost_estimate'],
                            security_hash=record['security_hash'],
                            compliance_flags=record['compliance_flags']
                        )
                        for record in data
                    ]
                logger.info(f"Loaded {len(self.provenance_records)} provenance records")
            except Exception as e:
                logger.error(f"Failed to load provenance: {e}")
                self.provenance_records = []
    
    def save_provenance(self):
        """Save provenance records to file"""
        try:
            data = [
                {
                    'operation_id': record.operation_id,
                    'operation_type': record.operation_type,
                    'timestamp': record.timestamp.isoformat(),
                    'user_id': record.user_id,
                    'team_id': record.team_id,
                    'project_id': record.project_id,
                    'providers_queried': record.providers_queried,
                    'selected_provider': record.selected_provider,
                    'configuration': record.configuration,
                    'terraform_code': record.terraform_code,
                    'cost_estimate': record.cost_estimate,
                    'security_hash': record.security_hash,
                    'compliance_flags': record.compliance_flags
                }
                for record in self.provenance_records
            ]
            
            with open(self.provenance_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Provenance records saved successfully")
        except Exception as e:
            logger.error(f"Failed to save provenance: {e}")
    
    def generate_security_hash(self, configuration: Dict[str, Any]) -> str:
        """Generate security hash for configuration"""
        config_str = json.dumps(configuration, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    async def configure_provider(self, provider: str, interactive: bool = False):
        """Configure provider credentials"""
        print(f"🔧 Configuring {provider} provider...")
        
        if interactive:
            # Interactive credential input
            credentials = {}
            
            if provider == "aws":
                credentials["access_key_id"] = input("AWS Access Key ID: ")
                credentials["secret_access_key"] = input("AWS Secret Access Key: ")
                credentials["region"] = input("Default Region (us-east-1): ") or "us-east-1"
            
            elif provider == "gcp":
                credentials["project_id"] = input("GCP Project ID: ")
                credentials["credentials_file"] = input("Path to service account key file: ")
            
            elif provider == "azure":
                credentials["subscription_id"] = input("Azure Subscription ID: ")
                credentials["tenant_id"] = input("Azure Tenant ID: ")
                credentials["client_id"] = input("Azure Client ID: ")
                credentials["client_secret"] = input("Azure Client Secret: ")
            
            elif provider == "vast_ai":
                credentials["api_key"] = input("Vast.AI API Key: ")
            
            elif provider == "runpod":
                credentials["api_key"] = input("RunPod API Key: ")
            
            elif provider == "lambda_labs":
                credentials["api_key"] = input("Lambda Labs API Key: ")
            
            elif provider == "coreweave":
                credentials["api_key"] = input("CoreWeave API Key: ")
            
            elif provider == "tensor_dock":
                credentials["api_key"] = input("TensorDock API Key: ")
            
            else:
                print(f"❌ Unknown provider: {provider}")
                return
            
            # Validate credentials
            if await self.validate_credentials(provider, credentials):
                self.credentials[provider] = ProviderCredentials(
                    provider=provider,
                    credentials=credentials,
                    enabled=True,
                    last_validated=datetime.now()
                )
                self.save_credentials()
                print(f"✅ {provider} credentials configured successfully")
            else:
                print(f"❌ {provider} credentials validation failed")
        
        else:
            print(f"📝 Use interactive mode with --interactive flag")
            print(f"   terradev configure {provider} --interactive")
    
    async def validate_credentials(self, provider: str, credentials: Dict[str, str]) -> bool:
        """Validate provider credentials"""
        try:
            # Mock validation - in production, this would test actual API calls
            if provider == "aws":
                # Test AWS credentials
                return len(credentials.get("access_key_id", "")) > 0 and len(credentials.get("secret_access_key", "")) > 0
            
            elif provider == "gcp":
                # Test GCP credentials
                return len(credentials.get("project_id", "")) > 0 and os.path.exists(credentials.get("credentials_file", ""))
            
            elif provider == "azure":
                # Test Azure credentials
                return all(len(credentials.get(key, "")) > 0 for key in ["subscription_id", "tenant_id", "client_id", "client_secret"])
            
            else:
                # Test API key-based providers
                return len(credentials.get("api_key", "")) > 0
        
        except Exception as e:
            logger.error(f"Credential validation failed for {provider}: {e}")
            return False
    
    def list_providers(self):
        """List configured providers"""
        print("🔑 Configured Providers:")
        print("-" * 50)
        
        if not self.credentials:
            print("No providers configured. Use 'terradev configure <provider>'")
            return
        
        for provider, creds in self.credentials.items():
            status = "✅ Enabled" if creds.enabled else "❌ Disabled"
            last_validated = creds.last_validated.strftime("%Y-%m-%d %H:%M") if creds.last_validated else "Never"
            print(f"{provider:<15} {status:<12} {last_validated}")
    
    async def optimize_and_generate(self, gpu_type: str, duration_hours: int, 
                                  user_id: str, team_id: str, project_id: str,
                                  output_dir: str = None):
        """Main optimization and Terraform generation"""
        print(f"🚀 Starting parallel optimization for {gpu_type}...")
        
        # Create optimization request
        request = ProviderRequest(
            provider=None,  # Query all providers
            instance_type=InstanceType.SPOT,
            gpu_type=gpu_type,
            region="us-west-2",
            min_memory_gb=64,
            min_storage_gb=100,
            max_price_per_hour=10.0,
            duration_hours=duration_hours
        )
        
        # Run parallel optimization
        optimization_result = await self.engine.parallel_optimize_and_deploy(request)
        
        if not optimization_result.winner:
            print("❌ No suitable instances found")
            return
        
        # Generate Terraform code
        terraform_code = self.generate_terraform_code(optimization_result)
        
        # Create provenance record
        operation_id = hashlib.sha256(f"{user_id}{team_id}{project_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        provenance = ProvenanceRecord(
            operation_id=operation_id,
            operation_type="optimize_and_deploy",
            timestamp=datetime.now(),
            user_id=user_id,
            team_id=team_id,
            project_id=project_id,
            providers_queried=[r.provider.value for r in optimization_result.all_responses],
            selected_provider=optimization_result.winner.provider.value,
            configuration={
                "gpu_type": gpu_type,
                "duration_hours": duration_hours,
                "instance_type": optimization_result.winner.instance_type.value,
                "region": optimization_result.winner.region,
                "price_per_hour": optimization_result.winner.price_per_hour
            },
            terraform_code=terraform_code,
            cost_estimate=optimization_result.winner.price_per_hour * duration_hours,
            security_hash=self.generate_security_hash(optimization_result.winner.__dict__),
            compliance_flags=["gdpr", "soc2", "iso27001"]  # Would be dynamic
        )
        
        # Save provenance
        self.provenance_records.append(provenance)
        self.save_provenance()
        
        # Output results
        print(f"\n🏆 Optimization Results:")
        print(f"   Operation ID: {operation_id}")
        print(f"   Winner: {optimization_result.winner.provider.value}")
        print(f"   GPU: {optimization_result.winner.gpu_type}")
        print(f"   Price: ${optimization_result.winner.price_per_hour:.2f}/hr")
        print(f"   Total Cost: ${optimization_result.winner.price_per_hour * duration_hours:.2f}")
        print(f"   Speedup: {optimization_result.parallel_speedup:.1f}x faster")
        print(f"   Savings: {optimization_result.cost_savings_percent:.1f}%")
        
        # Save Terraform code
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            tf_file = output_path / f"terradev_{operation_id}.tf"
            with open(tf_file, 'w') as f:
                f.write(terraform_code)
            
            print(f"\n📄 Terraform code saved to: {tf_file}")
            
            # Generate variables file
            vars_file = output_path / f"terradev_{operation_id}.tfvars"
            vars_content = self.generate_tfvars(optimization_result)
            with open(vars_file, 'w') as f:
                f.write(vars_content)
            
            print(f"📄 Variables file saved to: {vars_file}")
        
        return optimization_result, provenance
    
    def generate_terraform_code(self, optimization_result) -> str:
        """Generate Terraform code for the optimized provider"""
        provider = optimization_result.winner.provider
        config = optimization_result.winner
        
        if provider == Provider.AWS:
            return self.generate_aws_terraform(config)
        elif provider == Provider.GCP:
            return self.generate_gcp_terraform(config)
        elif provider == Provider.AZURE:
            return self.generate_azure_terraform(config)
        elif provider == Provider.VAST_AI:
            return self.generate_vast_terraform(config)
        else:
            return f"# Terraform code for {provider.value}\n# Not yet implemented"
    
    def generate_aws_terraform(self, config) -> str:
        """Generate AWS Terraform code"""
        return f'''# AWS GPU Instance - Generated by Terradev
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{config.region}"
}}

resource "aws_security_group" "gpu_sg" {{
  name_prefix = "terradev-gpu-"
  description = "Security group for GPU instance"
  
  ingress {{
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

resource "aws_instance" "gpu_instance" {{
  ami           = "ami-0c02fb55956c7d3165"  # Deep Learning AMI
  instance_type = "p3.2xlarge"  # {config.gpu_type}
  
  security_groups = [aws_security_group.gpu_sg.name]
  
  tags = {{
    Name        = "terradev-gpu-{config.gpu_type.lower()}"
    Environment = "production"
    Project     = "terradev-optimized"
    GeneratedAt = "{datetime.now().isoformat()}"
  }}
  
  # Spot instance configuration
  instance_market_options {{
    market_type = "spot"
    spot_options {{
      instance_interruption_behavior = "stop"
      spot_instance_type = "persistent"
    }}
  }}
}}

output "instance_id" {{
  value = aws_instance.gpu_instance.id
}}

output "public_ip" {{
  value = aws_instance.gpu_instance.public_ip
}}

output "instance_type" {{
  value = aws_instance.gpu_instance.instance_type
}}
'''
    
    def generate_gcp_terraform(self, config) -> str:
        """Generate GCP Terraform code"""
        return f'''# GCP GPU Instance - Generated by Terradev
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = var.gcp_project_id
  region   = "{config.region}"
}}

variable "gcp_project_id" {{
  description = "GCP Project ID"
  type        = string
}}

resource "google_compute_instance" "gpu_instance" {{
  name         = "terradev-gpu-{config.gpu_type.lower()}"
  machine_type = "n1-standard-4"
  zone         = "{config.region}-a"
  
  boot_disk {{
    initialize_params {{
      image = "debian-cloud/debian-10"
      size  = 100
    }}
  }}
  
  network_interface {{
    network = "default"
    access_config {{
      # Ephemeral IP
    }}
  }}
  
  guest_accelerator {{
    type  = "nvidia-tesla-{config.gpu_type.lower()}"
    count = 1
  }}
  
  tags = {{
    environment = "production"
    project     = "terradev-optimized"
    generated_at = "{datetime.now().isoformat()}"
  }}
  
  # Spot instance configuration
  scheduling {{
    preemptible = true
  }}
}}

output "instance_id" {{
  value = google_compute_instance.gpu_instance.id
}}

output "public_ip" {{
  value = google_compute_instance.gpu_instance.network_interface[0].access_config[0].nat_ip
}}
'''
    
    def generate_vast_terraform(self, config) -> str:
        """Generate Vast.ai Terraform code (using external provider)"""
        return f'''# Vast.ai GPU Instance - Generated by Terradev
terraform {{
  required_providers {{
    vastai = {{
      source  = "vastai/vastai"
      version = "~> 1.0"
    }}
  }}
}}

provider "vastai" {{
  api_key = var.vastai_api_key
}}

variable "vastai_api_key" {{
  description = "Vast.ai API Key"
  type        = string
  sensitive   = true
}}

resource "vastai_instance" "gpu_instance" {{
  name        = "terradev-gpu-{config.gpu_type.lower()}"
  image       = "pytorch/pytorch:latest"
  disk_size   = 100
  gpu_type    = "{config.gpu_type}"
  gpu_count   = 1
  
  tags = {{
    environment = "production"
    project     = "terradev-optimized"
    generated_at = "{datetime.now().isoformat()}"
  }}
}}

output "instance_id" {{
  value = vastai_instance.gpu_instance.id
}}

output "ssh_host" {{
  value = vastai_instance.gpu_instance.ssh_host
}}
'''
    
    def generate_tfvars(self, optimization_result) -> str:
        """Generate Terraform variables file"""
        return f'''# Terraform Variables - Generated by Terradev
# Generated: {datetime.now().isoformat()}
# Provider: {optimization_result.winner.provider.value}
# GPU: {optimization_result.winner.gpu_type}
# Cost: ${optimization_result.winner.price_per_hour:.2f}/hr

# Provider-specific variables
'''
    
    def show_provenance(self, operation_id: str = None):
        """Show provenance records"""
        print("📋 Provenance Records:")
        print("-" * 80)
        
        records = self.provenance_records
        if operation_id:
            records = [r for r in records if r.operation_id == operation_id]
        
        if not records:
            print("No provenance records found")
            return
        
        for record in records:
            print(f"Operation ID: {record.operation_id}")
            print(f"Type: {record.operation_type}")
            print(f"Timestamp: {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"User: {record.user_id}")
            print(f"Team: {record.team_id}")
            print(f"Project: {record.project_id}")
            print(f"Providers Queried: {', '.join(record.providers_queried)}")
            print(f"Selected Provider: {record.selected_provider}")
            print(f"Cost Estimate: ${record.cost_estimate:.2f}")
            print(f"Security Hash: {record.security_hash}")
            print(f"Compliance: {', '.join(record.compliance_flags)}")
            print("-" * 80)
    
    def audit_provenance(self, user_id: str = None, team_id: str = None, project_id: str = None):
        """Audit provenance records"""
        print("🔍 Provenance Audit:")
        print("-" * 50)
        
        records = self.provenance_records
        
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if team_id:
            records = [r for r in records if r.team_id == team_id]
        if project_id:
            records = [r for r in records if r.project_id == project_id]
        
        if not records:
            print("No matching records found")
            return
        
        # Calculate statistics
        total_operations = len(records)
        total_cost = sum(r.cost_estimate for r in records)
        providers_used = set()
        compliance_issues = []
        
        for record in records:
            providers_used.update(record.providers_queried)
            if not record.compliance_flags:
                compliance_issues.append(record.operation_id)
        
        print(f"Total Operations: {total_operations}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Providers Used: {', '.join(sorted(providers_used))}")
        print(f"Compliance Issues: {len(compliance_issues)}")
        
        if compliance_issues:
            print(f"Issues: {', '.join(compliance_issues)}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Terradev CLI - Parallel GPU Provisioning")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Configure command
    configure_parser = subparsers.add_parser('configure', help='Configure provider credentials')
    configure_parser.add_argument('provider', help='Provider name')
    configure_parser.add_argument('--interactive', action='store_true', help='Interactive configuration')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List configured providers')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize and generate Terraform')
    optimize_parser.add_argument('--gpu-type', required=True, help='GPU type (A100, H100, A10G, etc.)')
    optimize_parser.add_argument('--duration', type=int, required=True, help='Duration in hours')
    optimize_parser.add_argument('--user-id', required=True, help='User ID')
    optimize_parser.add_argument('--team-id', required=True, help='Team ID')
    optimize_parser.add_argument('--project-id', required=True, help='Project ID')
    optimize_parser.add_argument('--output', help='Output directory')
    
    # Provenance command
    provenance_parser = subparsers.add_parser('provenance', help='Show provenance records')
    provenance_parser.add_argument('--operation-id', help='Specific operation ID')
    
    # Audit command
    audit_parser = subparsers.add_parser('audit', help='Audit provenance records')
    audit_parser.add_argument('--user-id', help='Filter by user ID')
    audit_parser.add_argument('--team-id', help='Filter by team ID')
    audit_parser.add_argument('--project-id', help='Filter by project ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TerradevCLI()
    
    if args.command == 'configure':
        asyncio.run(cli.configure_provider(args.provider, args.interactive))
    
    elif args.command == 'list':
        cli.list_providers()
    
    elif args.command == 'optimize':
        asyncio.run(cli.optimize_and_generate(
            args.gpu_type, args.duration, args.user_id, args.team_id, args.project_id, args.output
        ))
    
    elif args.command == 'provenance':
        cli.show_provenance(args.operation_id)
    
    elif args.command == 'audit':
        cli.audit_provenance(args.user_id, args.team_id, args.project_id)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
