#!/usr/bin/env python3
"""
Smart Deployment Router
Automatically selects optimal deployment strategy based on requirements and constraints
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class DeploymentType(Enum):
    DIRECT = "direct"
    KUBERNETES = "kubernetes"
    HYBRID = "hybrid"

@dataclass
class DeploymentRequirements:
    gpu_type: str
    gpu_count: int = 1
    memory_gb: int = 16
    storage_gb: int = 100
    estimated_hours: float = 1.0
    workload_type: str = "training"
    budget: Optional[float] = None
    region: Optional[str] = None
    ports: List[int] = None
    environment_vars: Dict[str, str] = None

@dataclass
class DeploymentOption:
    type: DeploymentType
    provider: str
    instance_type: str
    price_per_hour: float
    estimated_total_cost: float
    confidence: float
    risk_score: float
    setup_time_minutes: int
    pros: List[str]
    cons: List[str]
    metadata: Dict[str, Any]

class SmartDeploymentRouter:
    """Intelligent deployment strategy selector"""
    
    def __init__(self):
        self.price_engine = None
        self.requirements_analyzer = RequirementsAnalyzer()
        self.deployment_strategies = {
            DeploymentType.DIRECT: DirectDeploymentStrategy(),
            DeploymentType.KUBERNETES: KubernetesDeploymentStrategy(),
            DeploymentType.HYBRID: HybridDeploymentStrategy()
        }
    
    async def recommend_deployments(self, user_request: Dict[str, Any]) -> List[DeploymentOption]:
        """Recommend optimal deployment strategies"""
        # Analyze requirements
        requirements = self.requirements_analyzer.analyze(user_request)
        
        # Generate all possible deployment options
        all_options = await self._generate_all_options(requirements)
        
        # Score and rank options
        scored_options = []
        for option in all_options:
            score = await self._score_option(option, requirements)
            option.score = score
            scored_options.append(option)
        
        # Sort by score (highest first)
        scored_options.sort(key=lambda x: x.score, reverse=True)
        
        return scored_options[:5]  # Return top 5 options
    
    async def _generate_all_options(self, requirements: DeploymentRequirements) -> List[DeploymentOption]:
        """Generate all possible deployment options"""
        options = []
        
        # Import here to avoid circular imports
        from .price_discovery import PriceDiscoveryEngine
        
        # Initialize price engine if needed
        if self.price_engine is None:
            self.price_engine = PriceDiscoveryEngine()
        
        # Get price information
        async with self.price_engine as engine:
            price_infos = await engine.get_realtime_prices(requirements.gpu_type, requirements.region)
        
        # Generate options for each strategy
        for strategy_type, strategy in self.deployment_strategies.items():
            strategy_options = await strategy.generate_options(requirements, price_infos)
            options.extend(strategy_options)
        
        return options
    
    async def _score_option(self, option: DeploymentOption, requirements: DeploymentRequirements) -> float:
        """Score deployment option on multiple factors"""
        score = 0.0
        
        # Cost factor (30%)
        cost_score = self._calculate_cost_score(option, requirements)
        score += cost_score * 0.3
        
        # Performance factor (25%)
        perf_score = self._calculate_performance_score(option, requirements)
        score += perf_score * 0.25
        
        # Convenience factor (20%)
        conv_score = self._calculate_convenience_score(option, requirements)
        score += conv_score * 0.2
        
        # Reliability factor (15%)
        rel_score = self._calculate_reliability_score(option)
        score += rel_score * 0.15
        
        # Speed factor (10%)
        speed_score = self._calculate_speed_score(option, requirements)
        score += speed_score * 0.1
        
        return score
    
    def _calculate_cost_score(self, option: DeploymentOption, requirements: DeploymentRequirements) -> float:
        """Calculate cost score (lower cost = higher score)"""
        if requirements.budget and option.estimated_total_cost > requirements.budget:
            return 0.0  # Over budget = zero score
        
        # Normalize cost (assuming max reasonable cost is $100/hr)
        max_reasonable_cost = 100.0
        normalized_cost = min(option.price_per_hour / max_reasonable_cost, 1.0)
        
        return 1.0 - normalized_cost
    
    def _calculate_performance_score(self, option: DeploymentOption, requirements: DeploymentRequirements) -> float:
        """Calculate performance score"""
        # Base score on instance type performance
        performance_tiers = {
            'p4d.24xlarge': 1.0,  # Best
            'p3.8xlarge': 0.9,
            'p3.2xlarge': 0.8,
            'g5.12xlarge': 0.85,
            'g5.xlarge': 0.75,
            'a2-highgpu-1g': 0.8,
            'Standard_NC96ads_A100_v4': 0.85,
            'A100': 0.9,  # Generic A100
            'V100': 0.7,
            'RTX 3090': 0.6,
            'RTX 4090': 0.75
        }
        
        base_score = performance_tiers.get(option.instance_type, 0.5)
        
        # Adjust for GPU count
        gpu_factor = min(requirements.gpu_count / 4.0, 1.0)  # 4 GPUs = optimal
        
        return base_score * gpu_factor
    
    def _calculate_convenience_score(self, option: DeploymentOption, requirements: DeploymentRequirements) -> float:
        """Calculate convenience score"""
        convenience_factors = {
            DeploymentType.DIRECT: 0.9,  # Easiest
            DeploymentType.KUBERNETES: 0.6,  # Medium complexity
            DeploymentType.HYBRID: 0.4  # Most complex
        }
        
        base_score = convenience_factors.get(option.type, 0.5)
        
        # Adjust for provider ease of use
        provider_ease = {
            'runpod': 1.0,  # Easiest
            'lambda': 0.9,
            'vastai': 0.8,
            'tensordock': 0.8,
            'coreweave': 0.7,
            'aws': 0.6,
            'gcp': 0.6,
            'azure': 0.5
        }
        
        provider_factor = provider_ease.get(option.provider, 0.5)
        
        return base_score * provider_factor
    
    def _calculate_reliability_score(self, option: DeploymentOption) -> float:
        """Calculate reliability score"""
        # Base reliability on confidence and risk
        base_reliability = option.confidence * (1.0 - option.risk_score)
        
        # Adjust for deployment type
        type_reliability = {
            DeploymentType.DIRECT: 0.9,  # Simple = reliable
            DeploymentType.KUBERNETES: 0.85,  # K8s is reliable
            DeploymentType.HYBRID: 0.7  # Complex = less reliable
        }
        
        type_factor = type_reliability.get(option.type, 0.5)
        
        return base_reliability * type_factor
    
    def _calculate_speed_score(self, option: DeploymentOption, requirements: DeploymentRequirements) -> float:
        """Calculate speed score (how fast to get running)"""
        # Normalize setup time (assuming max reasonable time is 30 minutes)
        max_reasonable_time = 30
        normalized_time = min(option.setup_time_minutes / max_reasonable_time, 1.0)
        
        return 1.0 - normalized_time
    
    async def execute_deployment(self, chosen_option: DeploymentOption, requirements: DeploymentRequirements) -> Dict[str, Any]:
        """Execute the chosen deployment"""
        strategy = self.deployment_strategies[chosen_option.type]
        return await strategy.deploy(chosen_option, requirements)

class RequirementsAnalyzer:
    """Analyze user requirements and normalize them"""
    
    def analyze(self, user_request: Dict[str, Any]) -> DeploymentRequirements:
        """Analyze and normalize user requirements"""
        return DeploymentRequirements(
            gpu_type=user_request.get('gpu_type', 'A100'),
            gpu_count=user_request.get('gpu_count', 1),
            memory_gb=user_request.get('memory_gb', 16),
            storage_gb=user_request.get('storage_gb', 100),
            estimated_hours=user_request.get('estimated_hours', 1.0),
            workload_type=user_request.get('workload_type', 'training'),
            budget=user_request.get('budget'),
            region=user_request.get('region'),
            ports=user_request.get('ports', []),
            environment_vars=user_request.get('environment_vars', {})
        )

class DirectDeploymentStrategy:
    """Strategy for direct instance deployment"""
    
    async def generate_options(self, requirements: DeploymentRequirements, price_infos: List) -> List[DeploymentOption]:
        """Generate direct deployment options"""
        options = []
        
        for price_info in price_infos:
            # Calculate total cost
            total_cost = price_info.price * requirements.gpu_count * requirements.estimated_hours
            
            option = DeploymentOption(
                type=DeploymentType.DIRECT,
                provider=price_info.provider,
                instance_type=price_info.instance_type,
                price_per_hour=price_info.price,
                estimated_total_cost=total_cost,
                confidence=price_info.confidence,
                risk_score=0.1 if not price_info.spot else 0.2,
                setup_time_minutes=5,  # Fast setup for direct instances
                pros=[
                    "Fast deployment",
                    "Simple management",
                    "Direct control",
                    "Cost predictable"
                ],
                cons=[
                    "Manual scaling",
                    "No orchestration",
                    "Limited to single instance"
                ],
                metadata={
                    'spot': price_info.spot,
                    'capacity': price_info.capacity,
                    'region': price_info.region
                }
            )
            options.append(option)
        
        return options
    
    async def deploy(self, option: DeploymentOption, requirements: DeploymentRequirements) -> Dict[str, Any]:
        """Deploy direct instance"""
        # This would integrate with existing provision functionality
        return {
            'status': 'deploying',
            'deployment_id': f"direct-{option.provider}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'estimated_ready_time': f"{option.setup_time_minutes} minutes",
            'provider': option.provider,
            'instance_type': option.instance_type
        }

class KubernetesDeploymentStrategy:
    """Strategy for Kubernetes deployment"""
    
    async def generate_options(self, requirements: DeploymentRequirements, price_infos: List) -> List[DeploymentOption]:
        """Generate Kubernetes deployment options"""
        options = []
        
        # Only generate K8s options if user has K8s configured
        if not self._has_kubernetes_configured():
            return options
        
        for price_info in price_infos:
            # K8s adds overhead but provides orchestration
            k8s_overhead = 1.1  # 10% overhead for K8s
            total_cost = price_info.price * requirements.gpu_count * requirements.estimated_hours * k8s_overhead
            
            option = DeploymentOption(
                type=DeploymentType.KUBERNETES,
                provider=price_info.provider,
                instance_type=price_info.instance_type,
                price_per_hour=price_info.price * k8s_overhead,
                estimated_total_cost=total_cost,
                confidence=price_info.confidence * 0.9,  # Slightly lower confidence for K8s
                risk_score=0.05 if not price_info.spot else 0.15,
                setup_time_minutes=15,  # Longer setup for K8s
                pros=[
                    "Auto-scaling",
                    "Load balancing",
                    "Service discovery",
                    "Rolling updates",
                    "Easy management"
                ],
                cons=[
                    "Complex setup",
                    "Higher overhead",
                    "Steeper learning curve"
                ],
                metadata={
                    'spot': price_info.spot,
                    'capacity': price_info.capacity,
                    'region': price_info.region,
                    'k8s_overhead': k8s_overhead
                }
            )
            options.append(option)
        
        return options
    
    def _has_kubernetes_configured(self) -> bool:
        """Check if user has Kubernetes configured"""
        # This would check for kubeconfig, cluster access, etc.
        import os
        from pathlib import Path
        
        kubeconfig = os.environ.get('KUBECONFIG', Path.home() / '.kube' / 'config')
        return kubeconfig.exists()
    
    async def deploy(self, option: DeploymentOption, requirements: DeploymentRequirements) -> Dict[str, Any]:
        """Deploy Kubernetes workload"""
        # This would integrate with existing k8s functionality
        return {
            'status': 'deploying',
            'deployment_id': f"k8s-{option.provider}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'estimated_ready_time': f"{option.setup_time_minutes} minutes",
            'provider': option.provider,
            'instance_type': option.instance_type,
            'workload_type': requirements.workload_type
        }

class HybridDeploymentStrategy:
    """Strategy for hybrid deployments"""
    
    async def generate_options(self, requirements: DeploymentRequirements, price_infos: List) -> List[DeploymentOption]:
        """Generate hybrid deployment options"""
        options = []
        
        # Hybrid only makes sense for complex workloads
        if requirements.gpu_count < 4 or requirements.estimated_hours < 2:
            return options
        
        for price_info in price_infos:
            # Hybrid has highest overhead
            hybrid_overhead = 1.2  # 20% overhead
            total_cost = price_info.price * requirements.gpu_count * requirements.estimated_hours * hybrid_overhead
            
            option = DeploymentOption(
                type=DeploymentType.HYBRID,
                provider=price_info.provider,
                instance_type=price_info.instance_type,
                price_per_hour=price_info.price * hybrid_overhead,
                estimated_total_cost=total_cost,
                confidence=price_info.confidence * 0.8,  # Lower confidence for hybrid
                risk_score=0.08 if not price_info.spot else 0.18,
                setup_time_minutes=25,  # Longest setup time
                pros=[
                    "Maximum flexibility",
                    "Best of both worlds",
                    "Advanced orchestration",
                    "Complex workloads supported"
                ],
                cons=[
                    "Most complex",
                    "Highest overhead",
                    "Requires expertise",
                    "Harder to debug"
                ],
                metadata={
                    'spot': price_info.spot,
                    'capacity': price_info.capacity,
                    'region': price_info.region,
                    'hybrid_overhead': hybrid_overhead
                }
            )
            options.append(option)
        
        return options
    
    async def deploy(self, option: DeploymentOption, requirements: DeploymentRequirements) -> Dict[str, Any]:
        """Deploy hybrid workload"""
        return {
            'status': 'deploying',
            'deployment_id': f"hybrid-{option.provider}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'estimated_ready_time': f"{option.setup_time_minutes} minutes",
            'provider': option.provider,
            'instance_type': option.instance_type,
            'complexity': 'high'
        }
