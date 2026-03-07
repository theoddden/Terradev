"""
Terradev Optimization Configuration

Central configuration for all optimization features including CUCo integration.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class CUCoConfig:
    """Configuration for CUCo optimization"""
    enabled: bool = True
    min_gpu_count: int = 2
    min_communication_intensity: float = 0.3
    min_performance_gain: float = 1.2
    max_cost_increase: float = 0.5
    auto_apply: bool = True
    monitoring_enabled: bool = True
    p95_strict_mode: bool = False
    
@dataclass 
class OptimizationConfig:
    """Main optimization configuration"""
    auto_optimize: bool = True
    optimization_interval: int = 300  # 5 minutes
    performance_threshold: float = 0.8
    cost_threshold: float = 1.5
    enable_cuco: bool = True
    enable_warm_pool: bool = True
    enable_semantic_routing: bool = True
    enable_auto_scaling: bool = True
    
    # CUCo specific config
    cuco_config: CUCoConfig = None
    
    def __post_init__(self):
        if self.cuco_config is None:
            self.cuco_config = CUCoConfig()

class OptimizationConfigManager:
    """Manager for optimization configuration"""
    
    def __init__(self, config_path: str = "/etc/terradev/optimization.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> OptimizationConfig:
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                print(f"Error loading optimization config: {e}")
                return OptimizationConfig()
        else:
            return OptimizationConfig()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> OptimizationConfig:
        """Convert dict to OptimizationConfig"""
        cuco_data = data.get("cuco_config", {})
        cuco_config = CUCoConfig(**cuco_data)
        
        config_data = {k: v for k, v in data.items() if k != "cuco_config"}
        config_data["cuco_config"] = cuco_config
        
        return OptimizationConfig(**config_data)
    
    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(self.config)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Convert OptimizationConfig to dict"""
        return {
            "auto_optimize": config.auto_optimize,
            "optimization_interval": config.optimization_interval,
            "performance_threshold": config.performance_threshold,
            "cost_threshold": config.cost_threshold,
            "enable_cuco": config.enable_cuco,
            "enable_warm_pool": config.enable_warm_pool,
            "enable_semantic_routing": config.enable_semantic_routing,
            "enable_auto_scaling": config.enable_auto_scaling,
            "cuco_config": {
                "enabled": config.cuco_config.enabled,
                "min_gpu_count": config.cuco_config.min_gpu_count,
                "min_communication_intensity": config.cuco_config.min_communication_intensity,
                "min_performance_gain": config.cuco_config.min_performance_gain,
                "max_cost_increase": config.cuco_config.max_cost_increase,
                "auto_apply": config.cuco_config.auto_apply,
                "monitoring_enabled": config.cuco_config.monitoring_enabled,
                "p95_strict_mode": config.cuco_config.p95_strict_mode
            }
        }
    
    def get_config(self) -> OptimizationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key == "cuco_config" and isinstance(value, dict):
                for cuco_key, cuco_value in value.items():
                    if hasattr(self.config.cuco_config, cuco_key):
                        setattr(self.config.cuco_config, cuco_key, cuco_value)
        
        self.save_config()
    
    def get_p95_boundaries(self) -> Dict[str, Dict[str, float]]:
        """Get P95 boundaries for different workload types"""
        return {
            "flash_attention": {
                "fusion_efficiency": 0.87,
                "overlap_ratio": 0.78,
                "speedup": 1.13,
                "memory_util": 0.82,
                "compute_util": 0.91,
                "network_util": 0.72
            },
            "moe_dispatch": {
                "fusion_efficiency": 0.84,
                "overlap_ratio": 0.76,
                "speedup": 1.18,
                "memory_util": 0.79,
                "compute_util": 0.89,
                "network_util": 0.71
            },
            "kv_cache_transfer": {
                "fusion_efficiency": 0.83,
                "overlap_ratio": 0.74,
                "speedup": 1.09,
                "memory_util": 0.81,
                "compute_util": 0.88,
                "network_util": 0.69
            },
            "gemm_allgather": {
                "fusion_efficiency": 0.86,
                "overlap_ratio": 0.77,
                "speedup": 1.26,
                "memory_util": 0.83,
                "compute_util": 0.92,
                "network_util": 0.73
            }
        }
    
    def get_workload_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get requirements for different workload types"""
        return {
            "moe": {
                "min_gpu_count": 2,
                "min_communication_intensity": 0.4,
                "preferred_topology": "infiniband",
                "memory_requirement": "high",
                "network_requirement": "high"
            },
            "attention": {
                "min_gpu_count": 2,
                "min_communication_intensity": 0.3,
                "preferred_topology": "nvlink",
                "memory_requirement": "high",
                "network_requirement": "medium"
            },
            "llm_training": {
                "min_gpu_count": 4,
                "min_communication_intensity": 0.5,
                "preferred_topology": "infiniband",
                "memory_requirement": "very_high",
                "network_requirement": "high"
            },
            "distributed_inference": {
                "min_gpu_count": 2,
                "min_communication_intensity": 0.2,
                "preferred_topology": "nvlink",
                "memory_requirement": "medium",
                "network_requirement": "low"
            }
        }

# Global config manager instance
_config_manager = None

def get_optimization_config() -> OptimizationConfig:
    """Get global optimization configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = OptimizationConfigManager()
    return _config_manager.get_config()

def update_optimization_config(updates: Dict[str, Any]):
    """Update global optimization configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = OptimizationConfigManager()
    _config_manager.update_config(updates)
