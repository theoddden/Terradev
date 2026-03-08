#!/usr/bin/env python3
"""
Hyperstack Provider Integration for Terradev
GPU VM provisioning and management via NexGen Cloud
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HyperstackConfig:
    """Hyperstack configuration"""
    api_key: str
    environment: str = "default-CANADA-1"  # Canada, USA, or NORWAY
    ssh_key_name: Optional[str] = None


class HyperstackProvider:
    """Hyperstack GPU provider for Terradev"""
    
    def __init__(self, config: HyperstackConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_base = "https://infrahub-api.nexgencloud.com/v1"
        
    async def __aenter__(self):
        headers = {
            "apiKey": self.config.api_key,
            "Content-Type": "application/json"
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Hyperstack API connection"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Test by getting environments
            url = f"{self.api_base}/environments"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    env_data = await response.json()
                    return {
                        "status": "connected",
                        "provider": "hyperstack",
                        "environment": self.config.environment,
                        "environments": env_data
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
    
    async def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Get available GPU instance types"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Get flavors (instance types)
            url = f"{self.api_base}/flavors"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    gpu_flavors = []
                    
                    for flavor in data.get("flavors", []):
                        # Filter for GPU flavors
                        if any(gpu_type in flavor.get("name", "").lower() 
                               for gpu_type in ["h100", "h200", "a100", "l40"]):
                            gpu_flavors.append({
                                "id": flavor["id"],
                                "name": flavor["name"],
                                "vcpus": flavor.get("vcpus", 0),
                                "memory_mb": flavor.get("memory_mb", 0),
                                "disk_gb": flavor.get("disk_gb", 0),
                                "price_per_hour": flavor.get("price_per_hour", 0),
                                "gpu_count": self._extract_gpu_count(flavor["name"]),
                                "gpu_type": self._extract_gpu_type(flavor["name"])
                            })
                    
                    return gpu_flavors
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get GPU flavors: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get GPU instances: {e}")
    
    def _extract_gpu_count(self, flavor_name: str) -> int:
        """Extract GPU count from flavor name"""
        if "x1" in flavor_name:
            return 1
        elif "x8" in flavor_name:
            return 8
        return 1
    
    def _extract_gpu_type(self, flavor_name: str) -> str:
        """Extract GPU type from flavor name"""
        if "h100" in flavor_name.lower():
            return "H100"
        elif "h200" in flavor_name.lower():
            return "H200"
        elif "a100" in flavor_name.lower():
            return "A100"
        elif "l40" in flavor_name.lower():
            return "L40"
        return "Unknown"
    
    async def get_pricing(self, gpu_type: Optional[str] = None) -> Dict[str, Any]:
        """Get GPU instance pricing"""
        try:
            gpu_instances = await self.get_gpu_instances()
            
            pricing_data = {}
            for instance in gpu_instances:
                if gpu_type and gpu_type.lower() not in instance["gpu_type"].lower():
                    continue
                
                pricing_data[instance["name"]] = {
                    "hourly_price": instance["price_per_hour"],
                    "memory_gb": instance["memory_mb"] // 1024,
                    "vcpus": instance["vcpus"],
                    "disk_gb": instance["disk_gb"],
                    "gpu_count": instance["gpu_count"],
                    "gpu_type": instance["gpu_type"]
                }
            
            return {
                "provider": "hyperstack",
                "gpu_type": gpu_type,
                "pricing": pricing_data,
                "currency": "USD",
                "environment": self.config.environment,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to get pricing: {e}")
    
    async def get_stock_details(self) -> Dict[str, Any]:
        """Get GPU stock availability"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/stock"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    stock_data = await response.json()
                    return {
                        "provider": "hyperstack",
                        "stock": stock_data,
                        "environment": self.config.environment,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get stock details: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get stock details: {e}")
    
    async def provision_instance(self, 
                               name: str,
                               flavor_name: str,
                               environment: Optional[str] = None,
                               ssh_key_name: Optional[str] = None,
                               assign_floating_ip: bool = True) -> Dict[str, Any]:
        """Provision a GPU VM"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Use provided environment or default
            target_environment = environment or self.config.environment
            target_ssh_key = ssh_key_name or self.config.ssh_key_name
            
            # Create VM payload
            payload = {
                "name": name,
                "environment_name": target_environment,
                "key_name": target_ssh_key,
                "image_name": "Ubuntu Server 22.04 LTS (Jammy Jellyfish)",
                "flavor_name": flavor_name,
                "count": 1,
                "assign_floating_ip": assign_floating_ip,
                "security_rules": [
                    {
                        "direction": "ingress",
                        "protocol": "tcp",
                        "port_range_min": 22,
                        "port_range_max": 22,
                        "remote_ip_prefix": "0.0.0.0/0"
                    }
                ]
            }
            
            url = f"{self.api_base}/virtual-machines"
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 202:  # Accepted for provisioning
                    vm_data = await response.json()
                    
                    return {
                        "status": "provisioning",
                        "instance_id": str(vm_data.get("id", "")),
                        "name": name,
                        "provider": "hyperstack",
                        "flavor_name": flavor_name,
                        "environment": target_environment,
                        "ssh_key_name": target_ssh_key,
                        "assign_floating_ip": assign_floating_ip,
                        "created_at": datetime.now().isoformat(),
                        "response": vm_data
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to provision instance: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to provision instance: {e}")
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/virtual-machines/{instance_id}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    vm_data = await response.json()
                    
                    return {
                        "status": "active",
                        "instance_id": instance_id,
                        "name": vm_data.get("name", ""),
                        "provider": "hyperstack",
                        "flavor_name": vm_data.get("flavor_name", ""),
                        "environment": vm_data.get("environment_name", ""),
                        "status": vm_data.get("status", ""),
                        "created_at": vm_data.get("created_at", ""),
                        "public_ip": vm_data.get("floating_ip"),
                        "private_ip": vm_data.get("private_ip"),
                        "vcpus": vm_data.get("vcpus", 0),
                        "memory_mb": vm_data.get("memory_mb", 0),
                        "disk_gb": vm_data.get("disk_gb", 0),
                        "tags": vm_data.get("tags", [])
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get instance status: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get instance status: {e}")
    
    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """Terminate a GPU VM"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/virtual-machines/{instance_id}"
            async with self.session.delete(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status in [202, 204]:  # Accepted or No Content
                    return {
                        "status": "terminated",
                        "instance_id": instance_id,
                        "provider": "hyperstack",
                        "terminated_at": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to terminate instance: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to terminate instance: {e}")
    
    async def get_environments(self) -> List[Dict[str, Any]]:
        """Get available environments"""
        try:
            if not self.session:
                headers = {
                    "apiKey": self.config.api_key,
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/environments"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    environments = []
                    
                    for env in data.get("environments", []):
                        environments.append({
                            "name": env.get("name", ""),
                            "display_name": env.get("display_name", ""),
                            "region": env.get("region", ""),
                            "available": env.get("available", False)
                        })
                    
                    return environments
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get environments: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get environments: {e}")
