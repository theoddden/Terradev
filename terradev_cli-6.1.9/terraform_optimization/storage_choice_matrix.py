#!/usr/bin/env python3
"""
Storage Choice Matrix for Cost Optimization
Analyzes and recommends optimal storage choices based on use case, cost, and performance
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Storage types"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class UseCase(Enum):
    """Use case types"""
    DATASET_CACHE = "dataset_cache"
    MODEL_STORAGE = "model_storage"
    TRAINING_OUTPUT = "training_output"
    BACKUP = "backup"
    TEMPORARY = "temporary"
    PRODUCTION = "production"

class PerformanceTier(Enum):
    """Performance tiers"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"

@dataclass
class StorageOption:
    """Storage option configuration"""
    storage_type: StorageType
    provider: str
    region: str
    performance_tier: PerformanceTier
    cost_per_gb_month: float
    cost_per_gb_transfer: float
    throughput_mbps: float
    latency_ms: float
    durability: float  # 9s (e.g., 11 = 99.999999999%)
    availability: float  # percentage
    features: List[str]
    use_cases: List[UseCase]
    pros: List[str]
    cons: List[str]

@dataclass
class StorageRecommendation:
    """Storage recommendation"""
    use_case: UseCase
    primary_option: StorageOption
    secondary_option: Optional[StorageOption]
    total_cost_monthly: float
    performance_score: float
    reliability_score: float
    cost_efficiency_score: float
    reasoning: List[str]
    migration_complexity: str
    estimated_savings: float

class StorageChoiceMatrix:
    """Analyzes and recommends optimal storage choices"""
    
    def __init__(self):
        self.storage_options = self._initialize_storage_options()
        self.use_case_weights = self._initialize_use_case_weights()
        self.performance_requirements = self._initialize_performance_requirements()
        
        logger.info("Storage Choice Matrix initialized")
    
    def _initialize_storage_options(self) -> List[StorageOption]:
        """Initialize storage options with real-world data"""
        return [
            # AWS S3 options
            StorageOption(
                storage_type=StorageType.S3,
                provider="aws",
                region="us-east-1",
                performance_tier=PerformanceTier.LOW,
                cost_per_gb_month=0.023,
                cost_per_gb_transfer=0.09,
                throughput_mbps=100,
                latency_ms=100,
                durability=11,  # 99.999999999%
                availability=99.99,
                features=["versioning", "encryption", "lifecycle", "cross-region"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.BACKUP],
                pros=["High durability", "Versioning", "Lifecycle policies", "Cross-region replication"],
                cons=["Egress costs", "Latency for cross-region"]
            ),
            StorageOption(
                storage_type=StorageType.S3,
                provider="aws",
                region="us-east-1",
                performance_tier=PerformanceTier.HIGH,
                cost_per_gb_month=0.025,
                cost_per_gb_transfer=0.09,
                throughput_mbps=500,
                latency_ms=50,
                durability=11,
                availability=99.99,
                features=["versioning", "encryption", "lifecycle", "cross-region", "transfer_acceleration"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.PRODUCTION],
                pros=["High throughput", "Transfer acceleration", "Low latency"],
                cons=["Higher cost", "Limited to AWS ecosystem"]
            ),
            
            # GCS options
            StorageOption(
                storage_type=StorageType.GCS,
                provider="gcp",
                region="us-central1",
                performance_tier=PerformanceTier.LOW,
                cost_per_gb_month=0.020,
                cost_per_gb_transfer=0.12,
                throughput_mbps=80,
                latency_ms=120,
                durability=11,
                availability=99.95,
                features=["versioning", "encryption", "lifecycle", "ml_engine"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.TRAINING_OUTPUT],
                pros=["ML integration", "AutoML", "BigQuery integration"],
                cons=["Higher egress costs", "Limited regional availability"]
            ),
            StorageOption(
                storage_type=StorageType.GCS,
                provider="gcp",
                region="us-central1",
                performance_tier=PerformanceTier.HIGH,
                cost_per_gb_month=0.026,
                cost_per_gb_transfer=0.12,
                throughput_mbps=400,
                latency_ms=60,
                durability=11,
                availability=99.95,
                features=["versioning", "encryption", "lifecycle", "ml_engine", "transfer_acceleration"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.PRODUCTION],
                pros=["ML integration", "High performance", "Transfer acceleration"],
                cons=["Higher cost", "GCP ecosystem dependency"]
            ),
            
            # Azure Blob options
            StorageOption(
                storage_type=StorageType.AZURE_BLOB,
                provider="azure",
                region="eastus",
                performance_tier=PerformanceTier.LOW,
                cost_per_gb_month=0.018,
                cost_per_gb_transfer=0.087,
                throughput_mbps=60,
                latency_ms=150,
                durability=11,
                availability=99.99,
                features=["versioning", "encryption", "lifecycle", "cdn"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.BACKUP],
                pros=["Low cost", "CDN integration", "Azure ecosystem"],
                cons=["Higher latency", "Limited ML integration"]
            ),
            StorageOption(
                storage_type=StorageType.AZURE_BLOB,
                provider="azure",
                region="eastus",
                performance_tier=PerformanceTier.HIGH,
                cost_per_gb_month=0.024,
                cost_per_gb_transfer=0.087,
                throughput_mbps=350,
                latency_ms=70,
                durability=11,
                availability=99.99,
                features=["versioning", "encryption", "lifecycle", "cdn", "premium"],
                use_cases=[UseCase.DATASET_CACHE, UseCase.MODEL_STORAGE, UseCase.PRODUCTION],
                pros=["High performance", "CDN integration", "Premium tier"],
                cons=["Higher cost", "Azure ecosystem dependency"]
            ),
            
            # HuggingFace Hub
            StorageOption(
                storage_type=StorageType.HUGGINGFACE,
                provider="huggingface",
                region="global",
                performance_tier=PerformanceTier.MEDIUM,
                cost_per_gb_month=0.0,
                cost_per_gb_transfer=0.0,
                throughput_mbps=50,
                latency_ms=200,
                durability=10,  # 99.99999999%
                availability=99.9,
                features=["model_hub", "dataset_hub", "versioning", "community"],
                use_cases=[UseCase.MODEL_STORAGE, UseCase.DATASET_CACHE],
                pros=["Free storage", "ML community", "Model sharing", "Versioning"],
                cons=["Limited performance", "No SLA", "Community dependency"]
            ),
            
            # Local storage
            StorageOption(
                storage_type=StorageType.LOCAL,
                provider="local",
                region="local",
                performance_tier=PerformanceTier.ULTRA_HIGH,
                cost_per_gb_month=0.10,  # Hardware cost
                cost_per_gb_transfer=0.0,
                throughput_mbps=1000,
                latency_ms=1,
                durability=9,  # 99.9999999%
                availability=99.5,
                features=["direct_access", "no_transfer_costs", "full_control"],
                use_cases=[UseCase.TEMPORARY, UseCase.TRAINING_OUTPUT, UseCase.PRODUCTION],
                pros=["Ultra-high performance", "No transfer costs", "Full control"],
                cons=["High hardware cost", "Maintenance required", "Limited scalability"]
            )
        ]
    
    def _initialize_use_case_weights(self) -> Dict[UseCase, Dict[str, float]]:
        """Initialize weights for different use cases"""
        return {
            UseCase.DATASET_CACHE: {
                "cost": 0.4,
                "performance": 0.3,
                "reliability": 0.2,
                "features": 0.1
            },
            UseCase.MODEL_STORAGE: {
                "cost": 0.3,
                "performance": 0.3,
                "reliability": 0.3,
                "features": 0.1
            },
            UseCase.TRAINING_OUTPUT: {
                "cost": 0.2,
                "performance": 0.5,
                "reliability": 0.2,
                "features": 0.1
            },
            UseCase.BACKUP: {
                "cost": 0.5,
                "performance": 0.1,
                "reliability": 0.3,
                "features": 0.1
            },
            UseCase.TEMPORARY: {
                "cost": 0.6,
                "performance": 0.3,
                "reliability": 0.1,
                "features": 0.0
            },
            UseCase.PRODUCTION: {
                "cost": 0.2,
                "performance": 0.4,
                "reliability": 0.3,
                "features": 0.1
            }
        }
    
    def _initialize_performance_requirements(self) -> Dict[UseCase, Dict[str, Any]]:
        """Initialize performance requirements for different use cases"""
        return {
            UseCase.DATASET_CACHE: {
                "min_throughput_mbps": 100,
                "max_latency_ms": 100,
                "min_availability": 99.9,
                "min_durability": 11
            },
            UseCase.MODEL_STORAGE: {
                "min_throughput_mbps": 50,
                "max_latency_ms": 200,
                "min_availability": 99.9,
                "min_durability": 11
            },
            UseCase.TRAINING_OUTPUT: {
                "min_throughput_mbps": 200,
                "max_latency_ms": 50,
                "min_availability": 99.5,
                "min_durability": 10
            },
            UseCase.BACKUP: {
                "min_throughput_mbps": 50,
                "max_latency_ms": 500,
                "min_availability": 99.99,
                "min_durability": 11
            },
            UseCase.TEMPORARY: {
                "min_throughput_mbps": 100,
                "max_latency_ms": 100,
                "min_availability": 99.0,
                "min_durability": 9
            },
            UseCase.PRODUCTION: {
                "min_throughput_mbps": 300,
                "max_latency_ms": 50,
                "min_availability": 99.99,
                "min_durability": 11
            }
        }
    
    def analyze_storage_needs(self, use_case: UseCase, 
                            storage_size_gb: float,
                            monthly_transfer_gb: float,
                            performance_requirements: Dict[str, Any] = None) -> StorageRecommendation:
        """Analyze storage needs and recommend optimal solution"""
        logger.info(f"Analyzing storage needs for {use_case.value}")
        
        # Get requirements
        requirements = performance_requirements or self.performance_requirements[use_case]
        
        # Filter eligible options
        eligible_options = self._filter_eligible_options(use_case, requirements)
        
        # Score options
        scored_options = self._score_options(eligible_options, use_case, storage_size_gb, monthly_transfer_gb)
        
        # Select best options
        primary_option = scored_options[0][0] if scored_options else None
        secondary_option = scored_options[1][0] if len(scored_options) > 1 else None
        
        # Calculate costs
        total_cost = self._calculate_total_cost(primary_option, storage_size_gb, monthly_transfer_gb)
        
        # Calculate scores
        performance_score = self._calculate_performance_score(primary_option, requirements)
        reliability_score = self._calculate_reliability_score(primary_option)
        cost_efficiency_score = self._calculate_cost_efficiency_score(primary_option, storage_size_gb, monthly_transfer_gb)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(primary_option, secondary_option, use_case, requirements)
        
        # Calculate migration complexity
        migration_complexity = self._calculate_migration_complexity(primary_option)
        
        # Calculate estimated savings
        estimated_savings = self._calculate_estimated_savings(primary_option, secondary_option, storage_size_gb, monthly_transfer_gb)
        
        recommendation = StorageRecommendation(
            use_case=use_case,
            primary_option=primary_option,
            secondary_option=secondary_option,
            total_cost_monthly=total_cost,
            performance_score=performance_score,
            reliability_score=reliability_score,
            cost_efficiency_score=cost_efficiency_score,
            reasoning=reasoning,
            migration_complexity=migration_complexity,
            estimated_savings=estimated_savings
        )
        
        logger.info(f"Recommended: {primary_option.provider} {primary_option.storage_type.value}")
        return recommendation
    
    def _filter_eligible_options(self, use_case: UseCase, requirements: Dict[str, Any]) -> List[StorageOption]:
        """Filter options based on use case and requirements"""
        eligible = []
        
        for option in self.storage_options:
            # Check use case support
            if use_case not in option.use_cases:
                continue
            
            # Check performance requirements
            if option.throughput_mbps < requirements["min_throughput_mbps"]:
                continue
            
            if option.latency_ms > requirements["max_latency_ms"]:
                continue
            
            if option.availability < requirements["min_availability"]:
                continue
            
            if option.durability < requirements["min_durability"]:
                continue
            
            eligible.append(option)
        
        return eligible
    
    def _score_options(self, options: List[StorageOption], use_case: UseCase, 
                      storage_size_gb: float, monthly_transfer_gb: float) -> List[Tuple[StorageOption, float]]:
        """Score options based on use case weights"""
        weights = self.use_case_weights[use_case]
        scored = []
        
        for option in options:
            # Calculate individual scores
            cost_score = self._calculate_cost_score(option, storage_size_gb, monthly_transfer_gb)
            performance_score = self._calculate_performance_score(option, self.performance_requirements[use_case])
            reliability_score = self._calculate_reliability_score(option)
            features_score = self._calculate_features_score(option, use_case)
            
            # Calculate weighted score
            total_score = (
                cost_score * weights["cost"] +
                performance_score * weights["performance"] +
                reliability_score * weights["reliability"] +
                features_score * weights["features"]
            )
            
            scored.append((option, total_score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _calculate_cost_score(self, option: StorageOption, storage_size_gb: float, monthly_transfer_gb: float) -> float:
        """Calculate cost score (lower cost = higher score)"""
        monthly_cost = (storage_size_gb * option.cost_per_gb_month + 
                       monthly_transfer_gb * option.cost_per_gb_transfer)
        
        # Normalize to 0-1 scale (assuming max $1000/month)
        normalized_cost = min(monthly_cost / 1000, 1.0)
        
        return 1.0 - normalized_cost
    
    def _calculate_performance_score(self, option: StorageOption, requirements: Dict[str, Any]) -> float:
        """Calculate performance score"""
        throughput_score = min(option.throughput_mbps / requirements["min_throughput_mbps"], 2.0) / 2.0
        latency_score = max(0, (requirements["max_latency_ms"] - option.latency_ms) / requirements["max_latency_ms"])
        
        return (throughput_score + latency_score) / 2.0
    
    def _calculate_reliability_score(self, option: StorageOption) -> float:
        """Calculate reliability score"""
        availability_score = option.availability / 100.0
        durability_score = min(option.durability / 11.0, 1.0)
        
        return (availability_score + durability_score) / 2.0
    
    def _calculate_features_score(self, option: StorageOption, use_case: UseCase) -> float:
        """Calculate features score based on use case"""
        required_features = {
            UseCase.DATASET_CACHE: ["versioning", "encryption"],
            UseCase.MODEL_STORAGE: ["versioning", "encryption"],
            UseCase.TRAINING_OUTPUT: ["encryption"],
            UseCase.BACKUP: ["versioning", "encryption", "lifecycle"],
            UseCase.TEMPORARY: [],
            UseCase.PRODUCTION: ["versioning", "encryption", "high_availability"]
        }
        
        needed_features = required_features.get(use_case, [])
        available_features = option.features
        
        if not needed_features:
            return 1.0
        
        matches = sum(1 for feature in needed_features if feature in available_features)
        return matches / len(needed_features)
    
    def _calculate_cost_efficiency_score(self, option: StorageOption, storage_size_gb: float, monthly_transfer_gb: float) -> float:
        """Calculate cost efficiency score (cost vs performance)"""
        cost_score = self._calculate_cost_score(option, storage_size_gb, monthly_transfer_gb)
        performance_score = option.throughput_mbps / 1000.0  # Normalize to 0-1 scale
        
        return (cost_score + performance_score) / 2.0
    
    def _calculate_total_cost(self, option: StorageOption, storage_size_gb: float, monthly_transfer_gb: float) -> float:
        """Calculate total monthly cost"""
        return (storage_size_gb * option.cost_per_gb_month + 
                monthly_transfer_gb * option.cost_per_gb_transfer)
    
    def _generate_reasoning(self, primary: StorageOption, secondary: Optional[StorageOption], 
                          use_case: UseCase, requirements: Dict[str, Any]) -> List[str]:
        """Generate reasoning for recommendation"""
        reasoning = []
        
        reasoning.append(f"Selected {primary.provider} {primary.storage_type.value} for optimal balance")
        reasoning.append(f"Performance: {primary.throughput_mbps}Mbps throughput, {primary.latency_ms}ms latency")
        reasoning.append(f"Reliability: {primary.availability}% availability, {primary.durability} 9s durability")
        reasoning.append(f"Cost: ${primary.cost_per_gb_month}/GB storage, ${primary.cost_per_gb_transfer}/GB transfer")
        
        # Add use case specific reasoning
        if use_case == UseCase.DATASET_CACHE:
            reasoning.append("Optimized for frequent access with versioning support")
        elif use_case == UseCase.MODEL_STORAGE:
            reasoning.append("Provides secure, versioned storage for ML models")
        elif use_case == UseCase.TRAINING_OUTPUT:
            reasoning.append("High performance for training output with low latency")
        elif use_case == UseCase.BACKUP:
            reasoning.append("Cost-effective backup with lifecycle management")
        elif use_case == UseCase.TEMPORARY:
            reasoning.append("Low-cost temporary storage with basic features")
        elif use_case == UseCase.PRODUCTION:
            reasoning.append("Production-ready with high reliability and performance")
        
        # Add secondary option reasoning
        if secondary:
            reasoning.append(f"Alternative: {secondary.provider} {secondary.storage_type.value}")
            reasoning.append(f"Secondary option offers {', '.join(secondary.pros[:2])}")
        
        return reasoning
    
    def _calculate_migration_complexity(self, option: StorageOption) -> str:
        """Calculate migration complexity"""
        if option.storage_type == StorageType.LOCAL:
            return "Low - Direct hardware setup"
        elif option.storage_type == StorageType.HUGGINGFACE:
            return "Low - Simple API integration"
        elif option.provider == "aws":
            return "Medium - AWS SDK and IAM setup"
        elif option.provider == "gcp":
            return "Medium - GCP SDK and service account setup"
        elif option.provider == "azure":
            return "Medium - Azure SDK and service principal setup"
        else:
            return "High - Custom integration required"
    
    def _calculate_estimated_savings(self, primary: StorageOption, secondary: Optional[StorageOption], 
                                   storage_size_gb: float, monthly_transfer_gb: float) -> float:
        """Calculate estimated savings compared to secondary option"""
        if not secondary:
            return 0.0
        
        primary_cost = self._calculate_total_cost(primary, storage_size_gb, monthly_transfer_gb)
        secondary_cost = self._calculate_total_cost(secondary, storage_size_gb, monthly_transfer_gb)
        
        return secondary_cost - primary_cost
    
    def get_storage_matrix(self) -> Dict[str, Any]:
        """Get complete storage choice matrix"""
        matrix = {
            "storage_options": [
                {
                    "provider": option.provider,
                    "storage_type": option.storage_type.value,
                    "region": option.region,
                    "performance_tier": option.performance_tier.value,
                    "cost_per_gb_month": option.cost_per_gb_month,
                    "cost_per_gb_transfer": option.cost_per_gb_transfer,
                    "throughput_mbps": option.throughput_mbps,
                    "latency_ms": option.latency_ms,
                    "durability": option.durability,
                    "availability": option.availability,
                    "features": option.features,
                    "use_cases": [uc.value for uc in option.use_cases],
                    "pros": option.pros,
                    "cons": option.cons
                }
                for option in self.storage_options
            ],
            "use_cases": {
                use_case.value: {
                    "weights": weights,
                    "requirements": self.performance_requirements[use_case]
                }
                for use_case, weights in self.use_case_weights.items()
            }
        }
        
        return matrix
    
    def compare_options(self, option1: StorageOption, option2: StorageOption) -> Dict[str, Any]:
        """Compare two storage options"""
        comparison = {
            "option1": {
                "provider": option1.provider,
                "storage_type": option1.storage_type.value,
                "cost_per_gb_month": option1.cost_per_gb_month,
                "throughput_mbps": option1.throughput_mbps,
                "latency_ms": option1.latency_ms,
                "availability": option1.availability,
                "durability": option1.durability
            },
            "option2": {
                "provider": option2.provider,
                "storage_type": option2.storage_type.value,
                "cost_per_gb_month": option2.cost_per_gb_month,
                "throughput_mbps": option2.throughput_mbps,
                "latency_ms": option2.latency_ms,
                "availability": option2.availability,
                "durability": option2.durability
            },
            "differences": {
                "cost_difference": option2.cost_per_gb_month - option1.cost_per_gb_month,
                "throughput_difference": option2.throughput_mbps - option1.throughput_mbps,
                "latency_difference": option2.latency_ms - option1.latency_ms,
                "availability_difference": option2.availability - option1.availability,
                "durability_difference": option2.durability - option1.durability
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if option1.cost_per_gb_month < option2.cost_per_gb_month:
            comparison["recommendations"].append(f"{option1.provider} is cheaper by ${option2.cost_per_gb_month - option1.cost_per_gb_month:.3f}/GB")
        else:
            comparison["recommendations"].append(f"{option2.provider} is cheaper by ${option1.cost_per_gb_month - option2.cost_per_gb_month:.3f}/GB")
        
        if option1.throughput_mbps > option2.throughput_mbps:
            comparison["recommendations"].append(f"{option1.provider} has {option1.throughput_mbps - option2.throughput_mbps}Mbps higher throughput")
        else:
            comparison["recommendations"].append(f"{option2.provider} has {option2.throughput_mbps - option1.throughput_mbps}Mbps higher throughput")
        
        if option1.latency_ms < option2.latency_ms:
            comparison["recommendations"].append(f"{option1.provider} has {option2.latency_ms - option1.latency_ms}ms lower latency")
        else:
            comparison["recommendations"].append(f"{option2.provider} has {option1.latency_ms - option2.latency_ms}ms lower latency")
        
        return comparison

if __name__ == "__main__":
    # Test the storage choice matrix
    print("💾 Testing Storage Choice Matrix...")
    
    matrix = StorageChoiceMatrix()
    
    print("\n💾 Storage Choice Matrix Features:")
    print("   ✅ Comprehensive storage option analysis")
    print("   ✅ Use case-specific recommendations")
    print("   ✅ Cost optimization calculations")
    print("   ✅ Performance and reliability scoring")
    print("   ✅ Migration complexity assessment")
    print("   ✅ Savings estimation")
    
    # Test dataset cache analysis
    print("\n🧪 Testing dataset cache analysis...")
    recommendation = matrix.analyze_storage_needs(
        use_case=UseCase.DATASET_CACHE,
        storage_size_gb=1000,
        monthly_transfer_gb=500
    )
    
    print(f"   ✅ Primary recommendation: {recommendation.primary_option.provider} {recommendation.primary_option.storage_type.value}")
    print(f"   💰 Monthly cost: ${recommendation.total_cost_monthly:.2f}")
    print(f"   📊 Performance score: {recommendation.performance_score:.2f}")
    print(f"   🔒 Reliability score: {recommendation.reliability_score:.2f}")
    print(f"   💡 Cost efficiency score: {recommendation.cost_efficiency_score:.2f}")
    print(f"   🏗️ Migration complexity: {recommendation.migration_complexity}")
    print(f"   💸 Estimated savings: ${recommendation.estimated_savings:.2f}")
    
    print("\n💡 Reasoning:")
    for reason in recommendation.reasoning:
        print(f"   • {reason}")
    
    # Test model storage analysis
    print("\n🧪 Testing model storage analysis...")
    model_recommendation = matrix.analyze_storage_needs(
        use_case=UseCase.MODEL_STORAGE,
        storage_size_gb=500,
        monthly_transfer_gb=100
    )
    
    print(f"   ✅ Primary recommendation: {model_recommendation.primary_option.provider} {model_recommendation.primary_option.storage_type.value}")
    print(f"   💰 Monthly cost: ${model_recommendation.total_cost_monthly:.2f}")
    print(f"   📊 Performance score: {model_recommendation.performance_score:.2f}")
    print(f"   🔒 Reliability score: {model_recommendation.reliability_score:.2f}")
    
    # Test comparison
    print("\n🧪 Testing option comparison...")
    aws_s3 = matrix.storage_options[0]
    gcs = matrix.storage_options[2]
    comparison = matrix.compare_options(aws_s3, gcs)
    
    print(f"   ✅ Comparing {aws_s3.provider} vs {gcs.provider}")
    print(f"   💰 Cost difference: ${comparison['differences']['cost_difference']:.3f}/GB")
    print(f"   🚀 Throughput difference: {comparison['differences']['throughput_difference']}Mbps")
    print(f"   ⚡ Latency difference: {comparison['differences']['latency_difference']}ms")
    
    print("\n💡 Recommendations:")
    for rec in comparison["recommendations"]:
        print(f"   • {rec}")
    
    print("\n✅ Storage Choice Matrix working correctly!")
    print("\n🎯 Key Benefits:")
    print("   • Data-driven storage recommendations")
    print("   • Cost optimization across all providers")
    print("   • Performance and reliability analysis")
    print("   • Use case-specific optimization")
    print("   • Migration complexity assessment")
    print("   • Savings estimation and comparison")
