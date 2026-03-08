#!/usr/bin/env python3
"""
Terradev LMCache Service Integration
Distributed KV cache layer for vLLM with Redis/S3 backends

Features:
  1. KV cache sharing across multiple vLLM instances
  2. CPU/Disk/Redis/S3 storage backends
  3. Zero-copy GPU-to-CPU offloading
  4. Disaggregated prefill/decode support
  5. 3-10x TTFT reduction for repeated prefixes

References:
  - LMCache: https://github.com/LMCache/LMCache
  - vLLM Integration: https://docs.lmcache.ai/getting_started/quickstart/
"""

import asyncio
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LMCacheConfig:
    """LMCache configuration for Terradev vLLM deployments"""
    enabled: bool = False
    backend: str = "redis"  # redis, s3, disk, cpu
    remote_url: str = "redis://localhost:6379"
    chunk_size: int = 256
    pipelined: bool = False
    serde: str = "torch"  # torch, cachegen
    local_device: Optional[str] = None  # cuda, cpu, file://path
    
    # Redis-specific
    redis_cluster: bool = False
    redis_sentinel: bool = False
    redis_password: Optional[str] = None
    
    # S3-specific
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_endpoint: Optional[str] = None
    
    # Disk-specific
    disk_path: str = "/tmp/lmcache"
    
    # Performance tuning
    enable_pipelined_backend: bool = False
    enable_debug: bool = False


class LMCacheService:
    """LMCache integration service for Terradev vLLM deployments"""
    
    def __init__(self, config: LMCacheConfig):
        self.config = config
        
    def generate_vllm_args(self) -> List[str]:
        """Generate vLLM arguments for LMCache integration"""
        args = []
        if not self.config.enabled:
            return args
            
        # Core LMCache flags
        args.extend([
            "--lmcache-enable",
            f"--lmcache-backend={self.config.backend}",
            f"--lmcache-chunk-size={self.config.chunk_size}",
            f"--lmcache-remote-serde={self.config.serde}",
        ])
        
        # Backend-specific configuration
        if self.config.backend == "redis":
            if self.config.redis_cluster:
                args.extend([
                    "--lmcache-redis-cluster",
                    f"--lmcache-redis-url={self.config.remote_url}",
                ])
            else:
                args.extend([
                    f"--lmcache-redis-url={self.config.remote_url}",
                ])
            if self.config.redis_password:
                args.append(f"--lmcache-redis-password={self.config.redis_password}")
                
        elif self.config.backend == "s3":
            args.extend([
                f"--lmcache-s3-bucket={self.config.s3_bucket}",
                f"--lmcache-s3-region={self.config.s3_region}",
            ])
            if self.config.s3_endpoint:
                args.append(f"--lmcache-s3-endpoint={self.config.s3_endpoint}")
                
        elif self.config.backend == "disk":
            args.extend([
                f"--lmcache-disk-path={self.config.disk_path}",
            ])
            
        elif self.config.backend == "cpu":
            # CPU backend doesn't need additional URL
            pass
            
        # Performance flags
        if self.config.enable_pipelined_backend:
            args.append("--lmcache-pipelined-backend")
            
        if self.config.enable_debug:
            args.append("--lmcache-debug")
            
        return args
    
    def generate_redis_deployment(self, 
                                 namespace: str = "lmcache",
                                 cluster_size: int = 3,
                                 memory_gb: int = 16) -> Dict[str, Any]:
        """Generate Redis cluster deployment manifest for LMCache"""
        
        redis_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "lmcache-redis",
                "namespace": namespace,
                "labels": {
                    "app": "lmcache-redis",
                    "component": "kv-cache"
                }
            },
            "spec": {
                "ports": [{
                    "port": 6379,
                    "targetPort": 6379,
                    "name": "redis"
                }],
                "selector": {
                    "app": "lmcache-redis"
                }
            }
        }
        
        redis_statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "lmcache-redis",
                "namespace": namespace,
                "labels": {
                    "app": "lmcache-redis",
                    "component": "kv-cache"
                }
            },
            "spec": {
                "serviceName": "lmcache-redis",
                "replicas": cluster_size,
                "selector": {
                    "matchLabels": {
                        "app": "lmcache-redis"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "lmcache-redis",
                            "component": "kv-cache"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "redis",
                            "image": "redis:7-alpine",
                            "ports": [{"containerPort": 6379}],
                            "resources": {
                                "requests": {
                                    "cpu": "1",
                                    "memory": f"{memory_gb}Gi"
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": f"{memory_gb + 4}Gi"
                                }
                            },
                            "volumeMounts": [{
                                "name": "redis-data",
                                "mountPath": "/data"
                            }],
                            "command": [
                                "redis-server",
                                "--appendonly", "yes",
                                "--maxmemory", f"{memory_gb - 2}gb",
                                "--maxmemory-policy", "allkeys-lru"
                            ]
                        }],
                        "volumes": [{
                            "name": "redis-data"
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {
                        "name": "redis-data"
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {
                            "requests": {
                                "storage": f"{memory_gb * 2}Gi"
                            }
                        }
                    }
                }]
            }
        }
        
        return {
            "service": redis_service,
            "statefulset": redis_statefulset
        }
    
    def generate_s3_bucket_config(self, 
                                 bucket_name: str,
                                 region: str = "us-east-1") -> Dict[str, Any]:
        """Generate S3 bucket configuration for LMCache"""
        return {
            "bucket": bucket_name,
            "region": region,
            "versioning": True,
            "lifecycle_rules": [
                {
                    "id": "lmcache-expiry",
                    "status": "Enabled",
                    "expiration": {"days": 7},
                    "filter": {"prefix": "lmcache/"}
                }
            ],
            "cors": [
                {
                    "allowed_headers": ["*"],
                    "allowed_methods": ["GET", "PUT", "DELETE"],
                    "allowed_origins": ["*"],
                    "max_age_seconds": 3600
                }
            ]
        }
    
    def generate_config_yaml(self) -> str:
        """Generate LMCache configuration YAML"""
        config_dict = {
            "chunk_size": self.config.chunk_size,
            "local_device": self.config.local_device or "cuda",
            "remote_url": self.config.remote_url,
            "remote_serde": self.config.serde,
            "pipelined_backend": self.config.enable_pipelined_backend,
            "debug": self.config.enable_debug
        }
        
        return yaml.dump(config_dict, default_flow_style=False)
    
    async def test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection for LMCache backend"""
        try:
            import redis
            
            # Parse Redis URL
            if self.config.remote_url.startswith("redis://"):
                host_port = self.config.remote_url[8:].split(":")
                host = host_port[0]
                port = int(host_port[1]) if len(host_port) > 1 else 6379
                
                r = redis.Redis(host=host, port=port, 
                              password=self.config.redis_password,
                              decode_responses=True,
                              socket_connect_timeout=5)
                
                # Test connection
                r.ping()
                
                # Test basic operations
                test_key = "lmcache:test"
                r.set(test_key, "test_value")
                value = r.get(test_key)
                r.delete(test_key)
                
                return {
                    "status": "connected",
                    "backend": "redis",
                    "host": host,
                    "port": port,
                    "test_passed": value == "test_value"
                }
            else:
                return {
                    "status": "failed",
                    "error": "Invalid Redis URL format"
                }
                
        except ImportError:
            return {
                "status": "failed",
                "error": "redis package not installed"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def estimate_cache_savings(self,
                                   model_name: str,
                                   context_length: int = 8192,
                                   requests_per_hour: int = 1000,
                                   prefix_reuse_rate: float = 0.3) -> Dict[str, Any]:
        """Estimate potential savings from LMCache KV cache sharing"""
        
        # Rough estimates based on LMCache benchmarks
        # These are conservative estimates for planning purposes
        
        # KV cache size estimation (rough)
        # Assume 2KB per token for KV cache (varies by model)
        kv_cache_size_per_token = 2048  # bytes
        total_kv_cache_size = context_length * kv_cache_size_per_token / (1024**3)  # GB
        
        # Compute savings from prefix reuse
        reused_tokens_per_request = int(context_length * prefix_reuse_rate)
        compute_savings_per_request = reused_tokens_per_request / context_length
        
        # TTFT improvement (3-10x according to LMCache)
        ttft_improvement_factor = 5.0  # Conservative estimate
        
        # Memory savings (KV cache offloading)
        memory_offload_ratio = 0.4  # 40% of KV cache can be offloaded
        
        # Calculate monthly savings
        monthly_requests = requests_per_hour * 24 * 30
        compute_savings_monthly = monthly_requests * compute_savings_per_request
        
        return {
            "model": model_name,
            "context_length": context_length,
            "total_kv_cache_gb": round(total_kv_cache_size, 2),
            "prefix_reuse_rate": prefix_reuse_rate,
            "requests_per_hour": requests_per_hour,
            "monthly_requests": monthly_requests,
            "compute_savings": {
                "requests_saved_per_hour": round(requests_per_hour * compute_savings_per_request),
                "monthly_compute_savings_percent": round(compute_savings_per_request * 100, 1),
                "ttft_improvement_factor": ttft_improvement_factor
            },
            "memory_savings": {
                "kv_cache_offloaded_gb": round(total_kv_cache_size * memory_offload_ratio, 2),
                "memory_savings_percent": round(memory_offload_ratio * 100, 1)
            },
            "estimated_monthly_cost_reduction": f"{round(compute_savings_monthly * 0.5, 0)}%"  # Rough estimate
        }
    
    def get_deployment_recommendations(self, 
                                      gpu_count: int,
                                      model_size_gb: float,
                                      expected_qps: float) -> Dict[str, Any]:
        """Get deployment recommendations for LMCache configuration"""
        
        recommendations = {
            "backend": "redis",
            "cluster_size": 3,
            "memory_per_node_gb": 16,
            "chunk_size": 256,
            "enable_pipelined": True
        }
        
        # Adjust based on scale
        if gpu_count >= 32:
            recommendations["cluster_size"] = 6
            recommendations["memory_per_node_gb"] = 32
            recommendations["backend"] = "redis"
        elif gpu_count >= 8:
            recommendations["cluster_size"] = 3
            recommendations["memory_per_node_gb"] = 16
        else:
            recommendations["cluster_size"] = 1
            recommendations["backend"] = "disk"  # Single instance can use disk backend
            
        # Adjust based on model size
        if model_size_gb > 100:
            recommendations["chunk_size"] = 512  # Larger chunks for big models
        elif model_size_gb > 50:
            recommendations["chunk_size"] = 256
        else:
            recommendations["chunk_size"] = 128
            
        # Adjust based on QPS
        if expected_qps > 100:
            recommendations["enable_pipelined"] = True
            recommendations["memory_per_node_gb"] += 8
            
        return recommendations


# ── Utility Functions ───────────────────────────────────────────────────────

def create_lmcache_config_from_dict(config_dict: Dict[str, Any]) -> LMCacheConfig:
    """Create LMCacheConfig from dictionary"""
    return LMCacheConfig(
        enabled=config_dict.get("enabled", False),
        backend=config_dict.get("backend", "redis"),
        remote_url=config_dict.get("remote_url", "redis://localhost:6379"),
        chunk_size=config_dict.get("chunk_size", 256),
        pipelined=config_dict.get("pipelined", False),
        serde=config_dict.get("serde", "torch"),
        local_device=config_dict.get("local_device"),
        redis_cluster=config_dict.get("redis_cluster", False),
        redis_sentinel=config_dict.get("redis_sentinel", False),
        redis_password=config_dict.get("redis_password"),
        s3_bucket=config_dict.get("s3_bucket"),
        s3_region=config_dict.get("s3_region", "us-east-1"),
        s3_endpoint=config_dict.get("s3_endpoint"),
        disk_path=config_dict.get("disk_path", "/tmp/lmcache"),
        enable_pipelined_backend=config_dict.get("enable_pipelined_backend", False),
        enable_debug=config_dict.get("enable_debug", False)
    )


def validate_lmcache_config(config: LMCacheConfig) -> List[str]:
    """Validate LMCache configuration and return list of issues"""
    issues = []
    
    if not config.enabled:
        return issues  # No validation needed if disabled
        
    if config.backend == "redis":
        if not config.remote_url or not config.remote_url.startswith("redis://"):
            issues.append("Redis backend requires valid redis:// URL")
        if config.redis_cluster and config.redis_sentinel:
            issues.append("Cannot enable both Redis cluster and Sentinel modes")
            
    elif config.backend == "s3":
        if not config.s3_bucket:
            issues.append("S3 backend requires bucket name")
            
    elif config.backend == "disk":
        if not config.disk_path:
            issues.append("Disk backend requires path")
            
    elif config.backend == "cpu":
        # CPU backend doesn't need additional validation
        pass
    else:
        issues.append(f"Unknown backend: {config.backend}")
        
    if config.chunk_size <= 0 or config.chunk_size > 4096:
        issues.append("Chunk size must be between 1 and 4096")
        
    return issues
