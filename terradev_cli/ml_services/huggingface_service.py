#!/usr/bin/env python3
"""
Hugging Face Service Integration for Terradev
Enhanced Hugging Face integration with model management, datasets, and inference
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration"""
    api_key: str
    namespace: Optional[str] = None
    organization: Optional[str] = None
    endpoint_url: Optional[str] = None


class HuggingFaceService:
    """Hugging Face integration service for models, datasets, and inference"""
    
    def __init__(self, config: HuggingFaceConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_base = "https://api.endpoints.huggingface.cloud/v2"
        self.hub_api = "https://huggingface.co/api"
        
    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Hugging Face connection and get user info"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Test Hub API access
            url = f"{self.hub_api}/whoami"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    user_data = await response.json()
                    return {
                        "status": "connected",
                        "namespace": self.config.namespace or user_data.get("name"),
                        "organization": self.config.organization,
                        "endpoint_url": self.config.endpoint_url,
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
    
    async def list_models(self, author: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List Hugging Face models"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            params = {"limit": limit}
            if author:
                params["author"] = author
            elif self.config.organization:
                params["author"] = self.config.organization
            
            url = f"{self.hub_api}/models"
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list models: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list models: {e}")
    
    async def list_datasets(self, author: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List Hugging Face datasets"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            params = {"limit": limit}
            if author:
                params["author"] = author
            elif self.config.organization:
                params["author"] = self.config.organization
            
            url = f"{self.hub_api}/datasets"
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list datasets: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list datasets: {e}")
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.hub_api}/models/{model_id}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get model info: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get model info for {model_id}: {e}")
    
    async def create_inference_endpoint(
        self,
        model_id: str,
        endpoint_name: str,
        instance_type: str = "gpu-medium-a10g",
        scaling_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Hugging Face inference endpoint"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            namespace = self.config.namespace or self.config.organization
            if not namespace:
                raise Exception("Namespace or organization must be configured")
            
            # Default scaling config
            scaling = scaling_config or {"minReplica": 1, "maxReplica": 1}
            
            payload = {
                "name": endpoint_name,
                "type": "protected",
                "compute": {
                    "accelerator": "gpu",
                    "instanceType": instance_type,
                    "instanceSize": "x1",
                    "scaling": scaling
                },
                "model": {
                    "framework": "pytorch",
                    "image": {"huggingface": {}},
                    "repository": model_id,
                },
                "provider": {
                    "region": "us-east-1",
                    "vendor": "aws"
                }
            }
            
            url = f"{self.api_base}/endpoint/{namespace}"
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create endpoint: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to create inference endpoint: {e}")
    
    async def list_inference_endpoints(self) -> List[Dict[str, Any]]:
        """List all inference endpoints"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            namespace = self.config.namespace or self.config.organization
            if not namespace:
                raise Exception("Namespace or organization must be configured")
            
            url = f"{self.api_base}/endpoint/{namespace}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list endpoints: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list inference endpoints: {e}")
    
    async def get_inference_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get information about a specific inference endpoint"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            namespace = self.config.namespace or self.config.organization
            if not namespace:
                raise Exception("Namespace or organization must be configured")
            
            url = f"{self.api_base}/endpoint/{namespace}/{endpoint_name}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get endpoint: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get inference endpoint {endpoint_name}: {e}")
    
    async def delete_inference_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Delete an inference endpoint"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            namespace = self.config.namespace or self.config.organization
            if not namespace:
                raise Exception("Namespace or organization must be configured")
            
            url = f"{self.api_base}/endpoint/{namespace}/{endpoint_name}"
            async with self.session.delete(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to delete endpoint: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to delete inference endpoint {endpoint_name}: {e}")
    
    async def run_inference(self, endpoint_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on an endpoint"""
        try:
            endpoint_info = await self.get_inference_endpoint(endpoint_name)
            endpoint_url = endpoint_info.get("url")
            
            if not endpoint_url:
                raise Exception(f"Endpoint {endpoint_name} does not have a URL")
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {"inputs": inputs}
            async with self.session.post(
                endpoint_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Inference failed: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to run inference on {endpoint_name}: {e}")
    
    def get_huggingface_config(self) -> Dict[str, str]:
        """Get Hugging Face configuration for environment variables"""
        config = {
            "HF_TOKEN": self.config.api_key
        }
        
        if self.config.namespace:
            config["HF_NAMESPACE"] = self.config.namespace
            
        if self.config.organization:
            config["HF_ORGANIZATION"] = self.config.organization
            
        if self.config.endpoint_url:
            config["HF_ENDPOINT_URL"] = self.config.endpoint_url
            
        return config
    
    def generate_setup_script(self) -> str:
        """Generate a shell snippet that sets up Hugging Face on a remote instance"""
        config = self.get_huggingface_config()
        script_lines = ["# Hugging Face Setup Script (generated by Terradev)"]
        
        for key, value in config.items():
            script_lines.append(f"export {key}='{value}'")
        
        script_lines.extend([
            "",
            "# Install Hugging Face libraries",
            "pip install transformers datasets accelerate",
            "",
            "# Test Hugging Face connection",
            "python -c \"from huggingface_hub import HfApi; api = HfApi(); print('Hugging Face configured successfully')\"",
            "",
            "# Example usage in training script:",
            "# from transformers import AutoTokenizer, AutoModelForCausalLM",
            "# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')",
            "# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')"
        ])
        
        return "\n".join(script_lines)


def create_huggingface_service_from_credentials(credentials: Dict[str, str]) -> HuggingFaceService:
    """Create HuggingFaceService from credential dictionary"""
    config = HuggingFaceConfig(
        api_key=credentials["api_key"],
        namespace=credentials.get("namespace"),
        organization=credentials.get("organization"),
        endpoint_url=credentials.get("endpoint_url")
    )
    
    return HuggingFaceService(config)


def get_huggingface_setup_instructions() -> str:
    """Get setup instructions for Hugging Face"""
    return """
🚀 Hugging Face Setup Instructions:

1. Create a Hugging Face account:
   - Go to https://huggingface.co
   - Sign up for a free account

2. Create an access token:
   - Navigate to https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose token type (read, write, or fine-grained)
   - Copy the token

3. Find your namespace/organization:
   - In Hugging Face UI, go to your profile
   - Your username is your namespace
   - For organizations, check your organization page

4. Configure Terradev with your Hugging Face credentials:
   terradev configure --provider huggingface --api-key YOUR_TOKEN --namespace your-username

📋 Required Credentials:
- api_key: Hugging Face access token (required)
- namespace: Hugging Face namespace/username (optional)
- organization: Hugging Face organization (optional)
- endpoint_url: Custom endpoint URL (optional)

💡 Usage Examples:
# Test connection
terradev ml huggingface --test

# List models
terradev ml huggingface --list-models

# List your models
terradev ml huggingface --list-models --author your-username

# List datasets
terradev ml huggingface --list-datasets

# Create inference endpoint
terradev ml huggingface --create-endpoint --model meta-llama/Llama-2-7b-hf --name my-endpoint

# List inference endpoints
terradev ml huggingface --list-endpoints

# Run inference
terradev ml huggingface --inference --endpoint my-endpoint --inputs '{"text": "Hello world"}'

🔗 Environment Variables for Training:
Add these to your ML training scripts:
export HF_TOKEN="your-token"
export HF_NAMESPACE="your-username"

Then in your Python code:
from huggingface_hub import HfApi
api = HfApi(token="your-token")

🎯 Terradev Integration:
Hugging Face integrates seamlessly with Terradev:
- Provision GPUs with Terradev
- Deploy models with Hugging Face Endpoints
- Use datasets from Hugging Face Hub
- Track experiments with W&B integration

📊 Model Management:
# Download models
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# Upload models
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="model.pth",
    path_in_repo="model.pth",
    repo_id="your-username/your-model",
    repo_type="model"
)

🚀 Inference Endpoints:
# Create GPU-powered inference endpoint
terradev ml huggingface --create-endpoint \\
    --model meta-llama/Llama-2-7b-hf \\
    --name llama-2-7b \\
    --instance-type gpu-medium-a10g \\
    --min-replica 1 \\
    --max-replica 3

# Run inference
terradev ml huggingface --inference \\
    --endpoint llama-2-7b \\
    --inputs '{"inputs": "What is machine learning?"}'

📚 Dataset Integration:
# Load datasets
from datasets import load_dataset
dataset = load_dataset("imdb")

# Use with Terradev staging
terradev stage -d ./dataset --target-regions us-east-1,eu-west-1

🐳 Docker Integration:
# In your Dockerfile
ENV HF_TOKEN="your-token"
ENV HF_HOME="/root/.cache/huggingface"

# Install libraries
RUN pip install transformers datasets accelerate

☁️ Enterprise Features:
# Private models and datasets
# Enterprise inference endpoints
# Custom model hosting
# Team collaboration tools
# Advanced security features

📝 Example Training Script:
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Upload checkpoint
    if epoch % 5 == 0:
        api.upload_file(
            path_or_fileobj=f"checkpoint_{epoch}.pth",
            path_in_repo=f"checkpoints/checkpoint_{epoch}.pth",
            repo_id="your-username/your-model"
        )

print("Training completed! Model uploaded to Hugging Face Hub.")
"""
