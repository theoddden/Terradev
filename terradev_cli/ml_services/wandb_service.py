#!/usr/bin/env python3
"""
Weights & Biases Service Integration for Terradev
Enhanced W&B integration with project management, runs tracking, and artifact handling
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WAndBConfig:
    """W&B configuration"""
    api_key: str
    entity: Optional[str] = None
    project: Optional[str] = None
    base_url: Optional[str] = None
    team: Optional[str] = None


class WAndBService:
    """W&B integration service for experiment tracking and collaboration"""
    
    def __init__(self, config: WAndBConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_base = config.base_url or "https://api.wandb.ai"
        
    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test W&B connection and get user info"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Test API access
            url = f"{self.api_base}/v1/user"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    user_data = await response.json()
                    return {
                        "status": "connected",
                        "entity": self.config.entity or user_data.get("username"),
                        "project": self.config.project or "terradev",
                        "base_url": self.api_base,
                        "user": user_data
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"API request failed: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all W&B projects"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            entity = self.config.entity or "me"
            url = f"{self.api_base}/v1/entities/{entity}/projects"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("projects", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list projects: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list projects: {e}")
    
    async def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new W&B project"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            entity = self.config.entity or "me"
            url = f"{self.api_base}/v1/entities/{entity}/projects"
            payload = {
                "name": name,
                "description": description
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create project: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to create project {name}: {e}")
    
    async def list_runs(self, project_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs in a project"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            entity = self.config.entity or "me"
            project = project_name or self.config.project or "terradev"
            url = f"{self.api_base}/v1/entities/{entity}/projects/{project}/runs"
            params = {"limit": limit}
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("runs", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list runs: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list runs: {e}")
    
    async def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/v1/runs/{run_id}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get run details: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get run details for {run_id}: {e}")
    
    async def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific run"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/v1/runs/{run_id}/history?stream=true"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get run metrics: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get run metrics for {run_id}: {e}")
    
    async def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """Get artifacts for a specific run"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/v1/artifacts?run={run_id}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("artifacts", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get run artifacts: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get run artifacts for {run_id}: {e}")
    
    def get_wandb_config(self) -> Dict[str, str]:
        """Get W&B configuration for environment variables"""
        config = {
            "WANDB_API_KEY": self.config.api_key
        }
        
        if self.config.entity:
            config["WANDB_ENTITY"] = self.config.entity
            
        if self.config.project:
            config["WANDB_PROJECT"] = self.config.project
        else:
            config["WANDB_PROJECT"] = "terradev"
            
        if self.config.base_url:
            config["WANDB_BASE_URL"] = self.config.base_url
            
        if self.config.team:
            config["WANDB_TEAM"] = self.config.team
            
        return config
    
    def build_run_config(
        self,
        gpu_type: str,
        provider: str,
        price_per_hour: float,
        region: str,
        instance_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a W&B run config dict with Terradev GPU cost metadata"""
        config = {
            "terradev_gpu_type": gpu_type,
            "terradev_provider": provider,
            "terradev_price_per_hour": price_per_hour,
            "terradev_region": region,
            "terradev_instance_id": instance_id,
            "terradev_created_at": datetime.now().isoformat(),
        }
        if extra:
            config.update(extra)
        return config
    
    def generate_setup_script(self) -> str:
        """Generate a shell snippet that sets up W&B on a remote instance"""
        config = self.get_wandb_config()
        script_lines = ["# W&B Setup Script (generated by Terradev)"]
        
        for key, value in config.items():
            script_lines.append(f"export {key}='{value}'")
        
        script_lines.extend([
            "",
            "# Test W&B connection",
            "python -c \"import wandb; wandb.login(); print('W&B configured successfully')\"",
            "",
            "# Example usage in training script:",
            "# import wandb",
            "# wandb.init(project='terradev')",
            "# wandb.log({'accuracy': 0.95, 'loss': 0.05})"
        ])
        
        return "\n".join(script_lines)
    
    async def export_runs_data(self, project_name: Optional[str] = None, format: str = "json") -> str:
        """Export runs data"""
        try:
            runs = await self.list_runs(project_name, limit=1000)
            
            if format.lower() == "json":
                return json.dumps(runs, indent=2)
            elif format.lower() == "csv":
                import csv
                import io
                
                if not runs:
                    return ""
                
                output = io.StringIO()
                fieldnames = ["id", "name", "state", "created_at", "config", "summary"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for run in runs:
                    writer.writerow({
                        "id": run.get("id", ""),
                        "name": run.get("name", ""),
                        "state": run.get("state", ""),
                        "created_at": run.get("createdAt", ""),
                        "config": json.dumps(run.get("config", {})),
                        "summary": json.dumps(run.get("summary", {}))
                    })
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise Exception(f"Failed to export runs data: {e}")


def create_wandb_service_from_credentials(credentials: Dict[str, str]) -> WAndBService:
    """Create WAndBService from credential dictionary"""
    config = WAndBConfig(
        api_key=credentials["api_key"],
        entity=credentials.get("entity"),
        project=credentials.get("project"),
        base_url=credentials.get("base_url"),
        team=credentials.get("team")
    )
    
    return WAndBService(config)


def get_wandb_setup_instructions() -> str:
    """Get setup instructions for W&B"""
    return """
🚀 Weights & Biases Setup Instructions:

1. Create a W&B account:
   - Go to https://wandb.ai
   - Sign up for a free account

2. Create an API key:
   - Navigate to https://wandb.ai/settings
   - Click "Create API Key"
   - Copy the API key

3. Find your entity (team/username):
   - In W&B UI, go to your profile
   - Your entity is shown in the URL (e.g., wandb.ai/your-entity)
   - Or use the API: curl -H "Authorization: Bearer YOUR_API_KEY" https://api.wandb.ai/v1/user

4. Configure Terradev with your W&B credentials:
   terradev configure --provider wandb --api-key YOUR_API_KEY --entity your-entity

📋 Required Credentials:
- api_key: W&B API key (required)
- entity: W&B entity (team/username, optional)
- project: Default project name (optional, default: "terradev")
- base_url: W&B server URL (optional, for self-hosted)
- team: W&B team name (optional)

💡 Usage Examples:
# Test connection
terradev ml wandb --test

# List projects
terradev ml wandb --list-projects

# Create a new project
terradev ml wandb --create-project my-project --description "My ML project"

# List runs
terradev ml wandb --list-runs --project my-project

# Get run details
terradev ml wandb --run-details RUN_ID

# Export runs data
terradev ml wandb --export --project my-project --format json > runs.json

🔗 Environment Variables for Training:
Add these to your ML training scripts:
export WANDB_API_KEY="your-api-key"
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="your-project"

Then in your Python code:
import wandb
wandb.init(project="your-project")
wandb.log({"accuracy": 0.95, "loss": 0.05})

🎯 Terradev Integration:
W&B automatically tracks Terradev provisioning metadata:
- GPU type and provider
- Cost per hour
- Region information
- Instance ID
- Creation timestamp

📊 Advanced Features:
# Custom run configuration with Terradev metadata
config = wandb_service.build_run_config(
    gpu_type="A100",
    provider="aws", 
    price_per_hour=4.50,
    region="us-east-1",
    instance_id="i-1234567890"
)

# Use in training script
wandb.init(config=config)

🐳 Docker Integration:
# In your Dockerfile
ENV WANDB_API_KEY="your-key"
ENV WANDB_PROJECT="terradev"

# Or use environment file
echo "WANDB_API_KEY=your-key" > .env.wandb
docker run --env-file .env.wandb your-image

☁️ Self-hosted W&B:
# For self-hosted W&B Server
terradev configure --provider wandb --api-key YOUR_KEY --base_url https://your-wandb-server.com

📝 Example Training Script:
import wandb
import terradev_cli.ml_services.wandb_service as wandb_service

# Initialize W&B with Terradev metadata
wandb.init(project="terradev-training")

# Log metrics
wandb.log({
    "epoch": 1,
    "loss": 0.5,
    "accuracy": 0.85,
    "learning_rate": 0.001
})

# Log artifacts
wandb.save("model.pth")
wandb.log_artifact("dataset.csv", type="dataset")

print("Training completed! Check W&B dashboard for results.")
"""
