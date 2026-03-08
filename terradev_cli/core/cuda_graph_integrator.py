#!/usr/bin/env python3
"""
CUDA Graph Integration Layer - Passive optimization without user intervention.

This module bridges the semantic router's NUMA analysis with the warm pool manager
to provide automatic CUDA Graph optimization recommendations.

No CLI commands required - everything happens passively in the background.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .semantic_router import NUMAEndpointScorer, NUMAScorecard
from .warm_pool_manager import WarmPoolManager

logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphRecommendation:
    """CUDA Graph optimization recommendation for a model-endpoint pair"""
    model_id: str
    endpoint_id: str
    use_cuda_graphs: bool
    optimization_score: float
    numa_alignment: str
    expected_speedup: str
    memory_requirements: str
    model_type: str
    priority_boost: int


class CUDAGraphIntegrator:
    """
    Passive CUDA Graph optimization integrator.
    
    Automatically:
    1. Analyzes model-endpoint compatibility with CUDA Graphs
    2. Provides optimization recommendations
    3. Updates warm pool priorities based on graph potential
    4. Tracks optimization effectiveness
    
    No user intervention required - everything runs in the background.
    """
    
    def __init__(self, numa_scorer: NUMAEndpointScorer, warm_pool: WarmPoolManager):
        self.numa_scorer = numa_scorer
        self.warm_pool = warm_pool
        self._recommendations_cache: Dict[str, CUDAGraphRecommendation] = {}
        
    def analyze_model_endpoint(self, model_id: str, endpoint_id: str, 
                             gpu_index: Optional[int] = None) -> CUDAGraphRecommendation:
        """
        Analyze CUDA Graph compatibility for a model-endpoint pair.
        
        Called automatically during endpoint scoring and warm pool decisions.
        """
        # Detect model type
        model_type = self._detect_model_type(model_id)
        
        # Get NUMA scorecard with CUDA Graph analysis
        numa_scorecard = self.numa_scorer.score_endpoint(
            endpoint_id=endpoint_id,
            gpu_index=gpu_index,
            model_type=model_type
        )
        
        # Extract CUDA Graph optimization data
        graph_score = numa_scorecard.metadata.get('cuda_graph_score', 0.0)
        use_cuda_graphs = numa_scorecard.metadata.get('cuda_graph_recommended', False)
        numa_optimal = numa_scorecard.metadata.get('numa_optimal_for_graphs', False)
        graph_potential = numa_scorecard.metadata.get('graph_optimization_potential', 'low')
        
        # Calculate priority boost based on optimization potential
        priority_boost = self._calculate_priority_boost(graph_score, model_type)
        
        # Generate optimization recommendations
        recommendation = CUDAGraphRecommendation(
            model_id=model_id,
            endpoint_id=endpoint_id,
            use_cuda_graphs=use_cuda_graphs,
            optimization_score=graph_score,
            numa_alignment=numa_scorecard.pcie_locality,
            expected_speedup=self._get_expected_speedup(graph_potential),
            memory_requirements=self._get_memory_requirements(graph_score),
            model_type=model_type,
            priority_boost=priority_boost
        )
        
        # Cache recommendation
        cache_key = f"{model_id}:{endpoint_id}"
        self._recommendations_cache[cache_key] = recommendation
        
        # Update warm pool if this is a high-priority optimization
        if use_cuda_graphs and graph_score > 0.8:
            self._update_warm_pool_priority(recommendation)
        
        return recommendation
    
    def _detect_model_type(self, model_id: str) -> str:
        """Detect model type from model identifier"""
        model_id_lower = model_id.lower()
        
        if any(keyword in model_id_lower for keyword in ['moe', 'mixture', 'expert']):
            return 'moe'
        elif any(keyword in model_id_lower for keyword in ['llama', 'bert', 'gpt', 'transformer', 't5']):
            return 'transformer'
        elif any(keyword in model_id_lower for keyword in ['resnet', 'conv', 'cnn', 'vision']):
            return 'cnn'
        else:
            return 'unknown'
    
    def _calculate_priority_boost(self, graph_score: float, model_type: str) -> int:
        """Calculate priority boost for warm pool based on CUDA Graph potential"""
        base_boost = 0
        
        if graph_score > 0.9:
            base_boost = 3  # Highest priority
        elif graph_score > 0.8:
            base_boost = 2  # High priority
        elif graph_score > 0.7:
            base_boost = 1  # Medium priority
        
        # Adjust for model type
        if model_type == 'transformer':
            base_boost += 1  # Transformers benefit most
        elif model_type == 'moe':
            base_boost -= 1  # MoE models have challenges
        
        return max(0, base_boost)
    
    def _get_expected_speedup(self, graph_potential: str) -> str:
        """Get expected speedup based on optimization potential"""
        speedup_map = {
            'optimal': '2-5x',
            'high': '1.5-3x',
            'medium': '1.2-2x',
            'low': '1.1-1.5x',
        }
        return speedup_map.get(graph_potential, '<1.2x')
    
    def _get_memory_requirements(self, graph_score: float) -> str:
        """Get memory requirements for CUDA Graph optimization"""
        if graph_score > 0.8:
            return '4-8GB'
        elif graph_score > 0.6:
            return '2-4GB'
        else:
            return '1-2GB'
    
    def _update_warm_pool_priority(self, recommendation: CUDAGraphRecommendation):
        """Update warm pool priority based on CUDA Graph recommendation"""
        current_priority = self.warm_pool.model_priorities.get(recommendation.model_id, 0)
        new_priority = current_priority + recommendation.priority_boost
        
        # Update warm pool priority
        self.warm_pool.model_priorities[recommendation.model_id] = new_priority
        
        # Update CUDA Graph metrics
        self.warm_pool.metrics.cuda_graph_optimized_models += 1
        
        logger.info(f"Updated priority for CUDA Graph model {recommendation.model_id}: {current_priority} -> {new_priority}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of CUDA Graph optimization opportunities.
        
        This provides insights into the optimization landscape without requiring user action.
        """
        total_recommendations = len(self._recommendations_cache)
        if total_recommendations == 0:
            return {
                'total_models': 0,
                'cuda_graph_compatible': 0,
                'high_potential': 0,
                'numa_optimal': 0,
                'average_score': 0.0,
            }
        
        # Analyze recommendations
        compatible = sum(1 for r in self._recommendations_cache.values() if r.use_cuda_graphs)
        high_potential = sum(1 for r in self._recommendations_cache.values() if r.optimization_score > 0.8)
        numa_optimal = sum(1 for r in self._recommendations_cache.values() if r.numa_alignment in ('PIX', 'PXB'))
        avg_score = sum(r.optimization_score for r in self._recommendations_cache.values()) / total_recommendations
        
        # Model type breakdown
        model_types = {}
        for rec in self._recommendations_cache.values():
            model_types[rec.model_type] = model_types.get(rec.model_type, 0) + 1
        
        return {
            'total_models': total_recommendations,
            'cuda_graph_compatible': compatible,
            'high_potential': high_potential,
            'numa_optimal': numa_optimal,
            'average_score': avg_score,
            'model_types': model_types,
            'optimization_potential': f"{(high_potential / total_recommendations * 100):.1f}%" if total_recommendations > 0 else "0%",
        }
    
    def get_recommendations_for_model(self, model_id: str) -> List[CUDAGraphRecommendation]:
        """Get all CUDA Graph recommendations for a specific model"""
        recommendations = []
        
        for key, rec in self._recommendations_cache.items():
            if rec.model_id == model_id:
                recommendations.append(rec)
        
        # Sort by optimization score (best first)
        recommendations.sort(key=lambda r: r.optimization_score, reverse=True)
        return recommendations
    
    def should_prefer_endpoint(self, model_id: str, endpoint1_id: str, endpoint2_id: str,
                             gpu1_index: Optional[int] = None, gpu2_index: Optional[int] = None) -> Optional[str]:
        """
        Determine which endpoint is preferred for CUDA Graph optimization.
        
        Called automatically during routing decisions.
        """
        # Get recommendations for both endpoints
        rec1_key = f"{model_id}:{endpoint1_id}"
        rec2_key = f"{model_id}:{endpoint2_id}"
        
        rec1 = self._recommendations_cache.get(rec1_key)
        rec2 = self._recommendations_cache.get(rec2_key)
        
        # If we don't have recommendations, analyze on the fly
        if rec1 is None:
            rec1 = self.analyze_model_endpoint(model_id, endpoint1_id, gpu1_index)
        if rec2 is None:
            rec2 = self.analyze_model_endpoint(model_id, endpoint2_id, gpu2_index)
        
        # Compare optimization scores
        if rec1.optimization_score > rec2.optimization_score + 0.1:  # 10% tolerance
            return endpoint1_id
        elif rec2.optimization_score > rec1.optimization_score + 0.1:
            return endpoint2_id
        
        # If scores are similar, prefer NUMA-optimal endpoint
        if rec1.numa_alignment in ('PIX', 'PXB') and rec2.numa_alignment not in ('PIX', 'PXB'):
            return endpoint1_id
        elif rec2.numa_alignment in ('PIX', 'PXB') and rec1.numa_alignment not in ('PIX', 'PXB'):
            return endpoint2_id
        
        # No clear preference
        return None


# Singleton instance for easy access
_integrator_instance: Optional[CUDAGraphIntegrator] = None


def get_cuda_graph_integrator(numa_scorer: NUMAEndpointScorer, 
                            warm_pool: WarmPoolManager) -> CUDAGraphIntegrator:
    """Get or create the singleton CUDA Graph integrator instance"""
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = CUDAGraphIntegrator(numa_scorer, warm_pool)
    return _integrator_instance


def analyze_endpoint_for_graphs(model_id: str, endpoint_id: str, 
                                gpu_index: Optional[int] = None) -> CUDAGraphRecommendation:
    """
    Convenience function to analyze an endpoint for CUDA Graph optimization.
    
    This is the main entry point for automatic CUDA Graph analysis.
    """
    from .semantic_router import get_default_numa_scorer
    from .warm_pool_manager import get_default_warm_pool
    
    numa_scorer = get_default_numa_scorer()
    warm_pool = get_default_warm_pool()
    integrator = get_cuda_graph_integrator(numa_scorer, warm_pool)
    
    return integrator.analyze_model_endpoint(model_id, endpoint_id, gpu_index)
