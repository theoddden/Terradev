#!/usr/bin/env python3
"""
DigitalOcean Provider Integration for Terradev
GPU Droplet provisioning and management
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DigitalOceanConfig:
    """DigitalOcean configuration"""
    api_token: str
    region: str = "tor1"  # Toronto has GPUs
    ssh_key_ids: Optional[List[int]] = None


class DigitalOceanProvider:
    """DigitalOcean GPU provider for Terradev"""
    
    def __init__(self, config: DigitalOceanConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_base = "https://api.digitalocean.com/v2"
        
    async def __aenter__(self):
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test DigitalOcean API connection"""
        try:
            if not self.session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Test account info
            url = f"{self.api_base}/account"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    account_data = await response.json()
                    return {
                        "status": "connected",
                        "provider": "digitalocean",
                        "account": account_data.get("account", {}),
                        "region": self.config.region
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
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Get sizes with GPU filter
            url = f"{self.api_base}/sizes"
            params = {"filter": "gpu"}
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    gpu_sizes = []
                    
                    for size in data.get("sizes", []):
                        if "gpu" in size.get("slug", "").lower():
                            gpu_sizes.append({
                                "slug": size["slug"],
                                "memory": size["memory"],
                                "vcpus": size["vcpus"],
                                "disk": size["disk"],
                                "price_monthly": size.get("price_monthly", 0),
                                "price_hourly": size.get("price_hourly", 0),
                                "regions": size.get("regions", []),
                                "description": size.get("description", "")
                            })
                    
                    return gpu_sizes
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get GPU sizes: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get GPU instances: {e}")
    
    async def get_pricing(self, gpu_type: Optional[str] = None) -> Dict[str, Any]:
        """Get GPU instance pricing"""
        try:
            gpu_instances = await self.get_gpu_instances()
            
            pricing_data = {}
            for instance in gpu_instances:
                if gpu_type and gpu_type not in instance["slug"]:
                    continue
                
                pricing_data[instance["slug"]] = {
                    "hourly_price": instance["price_hourly"],
                    "monthly_price": instance["price_monthly"],
                    "memory_gb": instance["memory"],
                    "vcpus": instance["vcpus"],
                    "disk_gb": instance["disk"],
                    "regions": instance["regions"],
                    "description": instance["description"]
                }
            
            return {
                "provider": "digitalocean",
                "gpu_type": gpu_type,
                "pricing": pricing_data,
                "currency": "USD",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to get pricing: {e}")
    
    async def provision_instance(self, 
                               name: str,
                               gpu_type: str,
                               region: Optional[str] = None,
                               ssh_key_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Provision a GPU Droplet"""
        try:
            if not self.session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Use provided region or default
            target_region = region or self.config.region
            target_ssh_keys = ssh_key_ids or self.config.ssh_key_ids or []
            
            # Create GPU Droplet
            url = f"{self.api_base}/droplets"
            payload = {
                "name": name,
                "region": target_region,
                "size": gpu_type,
                "image": "ubuntu-22-04-x64-gpu",  # AI/ML-ready image
                "ssh_keys": target_ssh_keys,
                "backups": False,
                "monitoring": True,
                "tags": ["terradev", "gpu", "ml"]
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 202:  # Accepted for provisioning
                    droplet_data = await response.json()
                    droplet_id = droplet_data["droplet"]["id"]
                    
                    return {
                        "status": "provisioning",
                        "instance_id": str(droplet_id),
                        "name": name,
                        "provider": "digitalocean",
                        "gpu_type": gpu_type,
                        "region": target_region,
                        "image": "ubuntu-22-04-x64-gpu",
                        "ssh_keys": target_ssh_keys,
                        "created_at": datetime.now().isoformat(),
                        "action_id": droplet_data.get("links", {}).get("actions", [{}])[0].get("id")
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
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/droplets/{instance_id}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    droplet_data = await response.json()
                    droplet = droplet_data["droplet"]
                    
                    # Extract network info
                    public_ip = None
                    private_ip = None
                    
                    for network in droplet.get("networks", {}).get("v4", []):
                        if network["type"] == "public":
                            public_ip = network["ip_address"]
                        elif network["type"] == "private":
                            private_ip = network["ip_address"]
                    
                    return {
                        "status": "active",
                        "instance_id": instance_id,
                        "name": droplet["name"],
                        "provider": "digitalocean",
                        "gpu_type": droplet["size"]["slug"],
                        "region": droplet["region"]["slug"],
                        "status": droplet["status"],
                        "created_at": droplet["created_at"],
                        "public_ip": public_ip,
                        "private_ip": private_ip,
                        "memory": droplet["memory"],
                        "vcpus": droplet["vcpus"],
                        "disk": droplet["disk"],
                        "features": droplet.get("features", [])
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get instance status: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get instance status: {e}")
    
    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """Terminate a GPU Droplet"""
        try:
            if not self.session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/droplets/{instance_id}"
            async with self.session.delete(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 204:  # No Content - Success
                    return {
                        "status": "terminated",
                        "instance_id": instance_id,
                        "provider": "digitalocean",
                        "terminated_at": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to terminate instance: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to terminate instance: {e}")
    
    async def get_regions(self) -> List[Dict[str, Any]]:
        """Get available regions"""
        try:
            if not self.session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json"
                }
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.api_base}/regions"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    regions = []
                    
                    for region in data.get("regions", []):
                        # Filter for regions with GPU support
                        if region.get("slug") in ["tor1", "nyc3", "sfo3"]:
                            regions.append({
                                "slug": region["slug"],
                                "name": region["name"],
                                "available": region.get("available", False),
                                "features": region.get("features", [])
                            })
                    
                    return regions
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get regions: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get regions: {e}")
