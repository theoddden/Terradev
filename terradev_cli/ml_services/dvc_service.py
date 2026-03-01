#!/usr/bin/env python3
"""
DVC (Data Version Control) Service Integration for Terradev
Manages DVC repositories, data versioning, and remote storage
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DVCConfig:
    """DVC configuration"""
    repo_path: str
    remote_storage: Optional[str] = None
    remote_type: Optional[str] = None  # s3, gs, azure, ssh, etc.
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    gcp_credentials_path: Optional[str] = None
    azure_connection_string: Optional[str] = None


class DVCService:
    """DVC integration service for data versioning and storage"""
    
    def __init__(self, config: DVCConfig):
        self.config = config
        self.repo_path = Path(config.repo_path)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def test_connection(self) -> Dict[str, Any]:
        """Test DVC installation and repository status"""
        try:
            # Check if DVC is installed
            result = subprocess.run(
                ["dvc", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": "DVC not installed. Run: pip install dvc"
                }
            
            # Check if we're in a DVC repository
            result = subprocess.run(
                ["dvc", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.repo_path),
            )
            
            if result.returncode == 0:
                return {
                    "status": "connected",
                    "dvc_version": result.stdout.split()[2],
                    "repo_path": str(self.repo_path),
                    "remote_storage": self.config.remote_storage
                }
            else:
                return {
                    "status": "not_initialized",
                    "error": "Not a DVC repository. Run 'dvc init' first."
                }
                
        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "DVC not installed. Run: pip install dvc"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def init_repo(self, force: bool = False) -> Dict[str, Any]:
        """Initialize a DVC repository"""
        try:
            cmd = ["dvc", "init"]
            if force:
                cmd.append("--force")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(self.repo_path))
            
            if result.returncode == 0:
                return {
                    "status": "initialized",
                    "repo_path": str(self.repo_path),
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to initialize DVC repo: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to initialize DVC repository: {e}")
    
    async def add_remote(self, name: str, url: str, remote_type: str = None) -> Dict[str, Any]:
        """Add a remote storage location"""
        try:
            cmd = ["dvc", "remote", "add", "-d", name, url]
            if remote_type:
                cmd.extend(["--type", remote_type])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(self.repo_path))
            
            if result.returncode == 0:
                return {
                    "status": "added",
                    "name": name,
                    "url": url,
                    "type": remote_type,
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to add remote: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to add remote {name}: {e}")
    
    async def list_remotes(self) -> List[Dict[str, Any]]:
        """List all configured remotes"""
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.repo_path),
            )
            
            if result.returncode == 0:
                remotes = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            remotes.append({
                                "name": parts[0],
                                "url": parts[1]
                            })
                return remotes
            else:
                return []
                
        except Exception as e:
            raise Exception(f"Failed to list remotes: {e}")
    
    async def add_data(self, data_path: str) -> Dict[str, Any]:
        """Add data file/directory to DVC tracking"""
        try:
            result = subprocess.run(
                ["dvc", "add", data_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.repo_path),
            )
            
            if result.returncode == 0:
                return {
                    "status": "added",
                    "data_path": data_path,
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to add data {data_path}: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to add data {data_path}: {e}")
    
    async def push_data(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Push data to remote storage"""
        try:
            cmd = ["dvc", "push"]
            if targets:
                cmd.extend(targets)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(self.repo_path))
            
            if result.returncode == 0:
                return {
                    "status": "pushed",
                    "targets": targets or "all",
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to push data: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to push data: {e}")
    
    async def pull_data(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Pull data from remote storage"""
        try:
            cmd = ["dvc", "pull"]
            if targets:
                cmd.extend(targets)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(self.repo_path))
            
            if result.returncode == 0:
                return {
                    "status": "pulled",
                    "targets": targets or "all",
                    "output": result.stdout
                }
            else:
                raise Exception(f"Failed to pull data: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to pull data: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get DVC repository status"""
        try:
            result = subprocess.run(
                ["dvc", "status"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_path),
            )
            
            if result.returncode == 0:
                # Parse status output
                status_lines = result.stdout.strip().split('\n')
                status_info = {
                    "status": "ok",
                    "details": status_lines
                }
                
                # Check for changes
                if "different" in result.stdout or "new" in result.stdout:
                    status_info["has_changes"] = True
                else:
                    status_info["has_changes"] = False
                
                return status_info
            else:
                raise Exception(f"Failed to get status: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to get status: {e}")
    
    async def list_tracked_files(self) -> List[Dict[str, Any]]:
        """List all DVC tracked files"""
        try:
            # Read .dvc files to get tracked data
            dvc_files = list(self.repo_path.glob("**/*.dvc"))
            tracked_files = []
            
            for dvc_file in dvc_files:
                try:
                    with open(dvc_file, 'r') as f:
                        dvc_content = json.load(f)
                    
                    tracked_files.append({
                        "dvc_file": str(dvc_file.relative_to(self.repo_path)),
                        "outs": dvc_content.get("outs", []),
                        "md5": dvc_content.get("md5", "")
                    })
                except Exception:
                    continue
            
            return tracked_files
            
        except Exception as e:
            raise Exception(f"Failed to list tracked files: {e}")
    
    async def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up DVC cache"""
        try:
            result = subprocess.run(
                ["dvc", "cache", "dir"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.repo_path),
            )
            
            if result.returncode == 0:
                cache_dir = result.stdout.strip()
                
                # Run cleanup
                result = subprocess.run(
                    ["dvc", "gc"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self.repo_path),
                )
                
                if result.returncode == 0:
                    return {
                        "status": "cleaned",
                        "cache_dir": cache_dir,
                        "output": result.stdout
                    }
                else:
                    raise Exception(f"Failed to cleanup cache: {result.stderr}")
            else:
                raise Exception(f"Failed to get cache directory: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to cleanup cache: {e}")
    
    def get_environment_config(self) -> Dict[str, str]:
        """Get environment variables for DVC"""
        config = {}
        
        if self.config.aws_access_key_id:
            config["AWS_ACCESS_KEY_ID"] = self.config.aws_access_key_id
            
        if self.config.aws_secret_access_key:
            config["AWS_SECRET_ACCESS_KEY"] = self.config.aws_secret_access_key
            
        if self.config.gcp_credentials_path:
            config["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.gcp_credentials_path
            
        if self.config.azure_connection_string:
            config["AZURE_STORAGE_CONNECTION_STRING"] = self.config.azure_connection_string
            
        return config


def create_dvc_service_from_credentials(credentials: Dict[str, str]) -> DVCService:
    """Create DVCService from credential dictionary"""
    config = DVCConfig(
        repo_path=credentials.get("repo_path", "."),
        remote_storage=credentials.get("remote_storage"),
        remote_type=credentials.get("remote_type"),
        aws_access_key_id=credentials.get("aws_access_key_id"),
        aws_secret_access_key=credentials.get("aws_secret_access_key"),
        gcp_credentials_path=credentials.get("gcp_credentials_path"),
        azure_connection_string=credentials.get("azure_connection_string")
    )
    
    return DVCService(config)


def get_dvc_setup_instructions() -> str:
    """Get setup instructions for DVC"""
    return """
🚀 DVC (Data Version Control) Setup Instructions:

1. Install DVC:
   pip install dvc[all]  # Includes all remote storage support

2. Initialize a DVC repository:
   cd your-project
   dvc init

3. Configure remote storage (choose one):

   Amazon S3:
   dvc remote add -d storage s3://your-bucket/dvc-store
   export AWS_ACCESS_KEY_ID=your-key
   export AWS_SECRET_ACCESS_KEY=your-secret

   Google Cloud Storage:
   dvc remote add -d storage gs://your-bucket/dvc-store
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

   Azure Blob Storage:
   dvc remote add -d storage azure://your-container/dvc-store
   export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"

   SSH:
   dvc remote add -d storage ssh://user@host/path/to/storage

4. Add data to DVC tracking:
   dvc add data/dataset.csv

5. Push data to remote:
   dvc push

6. Configure Terradev with your DVC credentials:
   terradev configure --provider dvc --repo-path /path/to/project --remote-storage s3://your-bucket/dvc-store

📋 Required Credentials:
- repo_path: Path to DVC repository (default: ".")
- remote_storage: Remote storage URL (optional)
- remote_type: Storage type (s3, gs, azure, ssh, etc.)
- aws_access_key_id: AWS access key (for S3)
- aws_secret_access_key: AWS secret key (for S3)
- gcp_credentials_path: Path to GCP credentials file (for GCS)
- azure_connection_string: Azure storage connection string (for Azure)

💡 Usage Examples:
# Test DVC connection
terradev dvc test

# Initialize repository
terradev dvc init

# Add remote storage
terradev dvc add-remote --name storage --url s3://my-bucket/dvc-store --type s3

# Add data to tracking
terradev dvc add --data data/dataset.csv

# Push to remote
terradev dvc push

# Pull from remote
terradev dvc pull

# Get repository status
terradev dvc status

# List tracked files
terradev dvc list-files

# Cleanup cache
terradev dvc cleanup

🔗 Integration with Terradev:
DVC can be used alongside Terradev's dataset staging:
- Stage datasets with DVC versioning
- Push to cloud storage for multi-cloud access
- Track data lineage with DVC + Terradev provisioning
"""
