#!/usr/bin/env python3
"""
Ollama Service Integration for Terradev
Local LLM deployment and management
"""

import os
import json
import asyncio
import aiohttp
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    host: str = "localhost"
    port: int = 11434
    model_name: Optional[str] = None
    base_url: Optional[str] = None


class OllamaService:
    """Ollama integration service for local LLM deployment"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.base_url or f"http://{config.host}:{config.port}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Ollama server connection"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test server health
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return {
                        "status": "connected",
                        "provider": "ollama",
                        "endpoint": self.base_url,
                        "host": self.config.host,
                        "port": self.config.port
                    }
                else:
                    return {
                        "status": "failed",
                        "error": f"Ollama server not responding: {response.status}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to connect to Ollama server: {str(e)}"
            }
    
    async def install_ollama(self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Install Ollama on remote instance"""
        try:
            install_script = """
#!/bin/bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
systemctl enable ollama
systemctl start ollama

# Wait for service to start
sleep 5

# Verify installation
ollama --version
"""
            
            # Execute installation via SSH
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, install_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {
                    "status": "installed",
                    "instance_ip": instance_ip,
                    "provider": "ollama",
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Installation failed: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to install Ollama: {str(e)}"
            }
    
    def _build_ssh_command(self, ip: str, user: str, key: Optional[str], script: str) -> str:
        """Build SSH command for remote execution"""
        if key:
            ssh_cmd = f"ssh -i {key} {user}@{ip}"
        else:
            ssh_cmd = f"ssh {user}@{ip}"
        
        return f'{ssh_cmd} "{script}"'
    
    async def pull_model(self, model_name: str, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Pull a model on remote instance"""
        try:
            pull_script = f"""
#!/bin/bash
# Pull model: {model_name}
ollama pull {model_name}

# Verify model is available
ollama list | grep {model_name}
"""
            
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, pull_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=600)  # 10 minutes for large models
            
            if result.returncode == 0:
                return {
                    "status": "pulled",
                    "instance_ip": instance_ip,
                    "provider": "ollama",
                    "model": model_name,
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Failed to pull model: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to pull model: {str(e)}"
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model in data.get("models", []):
                        models.append({
                            "name": model["name"],
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at", ""),
                            "digest": model.get("digest", "")
                        })
                    
                    return {
                        "status": "success",
                        "provider": "ollama",
                        "models": models,
                        "endpoint": self.base_url
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Failed to list models: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to list models: {str(e)}"
            }
    
    async def generate_text(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Generate text using Ollama"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    if stream:
                        # Handle streaming response
                        result_text = ""
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode().strip())
                                    if "response" in data:
                                        result_text += data["response"]
                                except json.JSONDecodeError:
                                    continue
                        
                        return {
                            "status": "success",
                            "provider": "ollama",
                            "model": model,
                            "prompt": prompt,
                            "response": result_text,
                            "stream": True
                        }
                    else:
                        # Handle non-streaming response
                        result = await response.json()
                        return {
                            "status": "success",
                            "provider": "ollama",
                            "model": model,
                            "prompt": prompt,
                            "response": result.get("response", ""),
                            "done": result.get("done", False),
                            "usage": result.get("usage", {})
                        }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Generation failed: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to generate text: {str(e)}"
            }
    
    async def chat_completion(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Chat completion using Ollama"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    if stream:
                        # Handle streaming response
                        result_text = ""
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode().strip())
                                    if "message" in data and "content" in data["message"]:
                                        result_text += data["message"]["content"]
                                except json.JSONDecodeError:
                                    continue
                        
                        return {
                            "status": "success",
                            "provider": "ollama",
                            "model": model,
                            "messages": messages,
                            "response": result_text,
                            "stream": True
                        }
                    else:
                        # Handle non-streaming response
                        result = await response.json()
                        return {
                            "status": "success",
                            "provider": "ollama",
                            "model": model,
                            "messages": messages,
                            "response": result.get("message", {}).get("content", ""),
                            "done": result.get("done", False),
                            "usage": result.get("usage", {})
                        }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Chat completion failed: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to complete chat: {str(e)}"
            }
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/api/show"
            payload = {"name": model}
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "provider": "ollama",
                        "model": model,
                        "info": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Failed to get model info: {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to get model info: {str(e)}"
            }
    
    async def delete_model(self, model: str, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Delete a model on remote instance"""
        try:
            delete_script = f"""
#!/bin/bash
# Delete model: {model}
ollama rm {model}

# Verify deletion
ollama list
"""
            
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, delete_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {
                    "status": "deleted",
                    "instance_ip": instance_ip,
                    "provider": "ollama",
                    "model": model,
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Failed to delete model: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to delete model: {str(e)}"
            }
    
    def get_popular_models(self) -> List[Dict[str, Any]]:
        """Get list of popular Ollama models"""
        return [
            {
                "name": "llama3.2",
                "description": "Meta's Llama 3.2 - Latest and most capable",
                "size": "4.7GB",
                "tags": ["latest", "general", "chat"]
            },
            {
                "name": "deepseek-r1",
                "description": "DeepSeek R1 - Advanced reasoning model",
                "size": "4.7GB",
                "tags": ["reasoning", "advanced", "chat"]
            },
            {
                "name": "qwen2.5",
                "description": "Alibaba's Qwen 2.5 - Multilingual model",
                "size": "4.7GB",
                "tags": ["multilingual", "general", "chat"]
            },
            {
                "name": "gemma2",
                "description": "Google's Gemma 2 - Efficient and capable",
                "size": "4.7GB",
                "tags": ["efficient", "general", "chat"]
            },
            {
                "name": "codellama",
                "description": "CodeLlama - Specialized for code generation",
                "size": "3.8GB",
                "tags": ["code", "programming", "technical"]
            },
            {
                "name": "mistral",
                "description": "Mistral 7B - High performance, efficient",
                "size": "4.1GB",
                "tags": ["efficient", "general", "chat"]
            }
        ]
    
    def get_deployment_script(self, instance_ip: str, model: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> str:
        """Generate deployment script for Ollama"""
        script = f"""
#!/bin/bash
# Ollama Deployment Script for Terradev
# Target: {instance_ip}
# Model: {model}

echo "🚀 Deploying Ollama with {model}..."

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
systemctl enable ollama
systemctl start ollama

# Wait for service to start
sleep 5

# Pull model
echo "📥 Pulling {model}..."
ollama pull {model}

# Verify deployment
echo "✅ Ollama deployed successfully!"
echo "🔗 API endpoint: http://{instance_ip}:11434/api"
echo "🧪 Test with: curl http://{instance_ip}:11434/api/generate -d '{{\"model\": \"{model}\", \"prompt\": \"Hello!\"}}'"
echo "📋 Available models:"
ollama list
"""
        return script
    
    async def get_server_status(self, instance_ip: str, ssh_user: str = "root", ssh_key: Optional[str] = None) -> Dict[str, Any]:
        """Get Ollama server status on remote instance"""
        try:
            status_script = """
#!/bin/bash
# Check Ollama service status
systemctl is-active ollama
systemctl status ollama --no-pager

# Check if port is listening
netstat -tlnp | grep :11434 || ss -tlnp | grep :11434

# Check Ollama version
ollama --version 2>/dev/null || echo "Ollama not installed"
"""
            
            ssh_cmd = self._build_ssh_command(instance_ip, ssh_user, ssh_key, status_script)
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "instance_ip": instance_ip,
                    "provider": "ollama",
                    "output": result.stdout
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Failed to get status: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to get server status: {str(e)}"
            }
