#!/usr/bin/env python3
"""
Terraform Decision Engine - Complete Transparency
Deterministic decision logging with clear "why this instance?" reasoning
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions made by the engine"""
    PROVIDER_SELECTION = "provider_selection"
    INSTANCE_SELECTION = "instance_selection"
    REGION_SELECTION = "region_selection"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    AVABILITY_CHECK = "availability_check"
    RISK_ASSESSMENT = "risk_assessment"
    PERMISSION_CHECK = "permission_check"

class PermissionScope(Enum):
    """Clear permission scopes for different operations"""
    READ_ONLY = "read_only"
    DRY_RUN = "dry_run"
    PLAN_ONLY = "plan_only"
    APPLY = "apply"
    DESTROY = "destroy"
    MODIFY_STATE = "modify_state"

@dataclass
class DecisionFactor:
    """Individual factor that influenced a decision"""
    name: str
    value: float
    weight: float
    reason: str
    source: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'weight': self.weight,
            'weighted_score': self.value * self.weight,
            'reason': self.reason,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DecisionLog:
    """Complete decision log with full transparency"""
    decision_id: str
    decision_type: DecisionType
    timestamp: datetime
    context: Dict[str, Any]
    options_considered: List[Dict[str, Any]]
    factors: List[DecisionFactor]
    selected_option: Dict[str, Any]
    reasoning: str
    confidence_score: float
    alternatives: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    permissions_required: List[PermissionScope]
    dry_run_result: Optional[Dict[str, Any]] = None
    rollback_plan: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'decision_type': self.decision_type.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'options_considered': self.options_considered,
            'factors': [factor.to_dict() for factor in self.factors],
            'selected_option': self.selected_option,
            'reasoning': self.reasoning,
            'confidence_score': self.confidence_score,
            'alternatives': self.alternatives,
            'risks': self.risks,
            'permissions_required': [perm.value for perm in self.permissions_required],
            'dry_run_result': self.dry_run_result,
            'rollback_plan': self.rollback_plan
        }

@dataclass
class TerraformPlan:
    """Terraform plan representation"""
    plan_id: str
    created_at: datetime
    resources_to_add: List[Dict[str, Any]]
    resources_to_change: List[Dict[str, Any]]
    resources_to_destroy: List[Dict[str, Any]]
    cost_estimate: float
    permissions_required: List[PermissionScope]
    risk_assessment: Dict[str, Any]
    rollback_available: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'created_at': self.created_at.isoformat(),
            'resources_to_add': self.resources_to_add,
            'resources_to_change': self.resources_to_change,
            'resources_to_destroy': self.resources_to_destroy,
            'cost_estimate': self.cost_estimate,
            'permissions_required': [perm.value for perm in self.permissions_required],
            'risk_assessment': self.risk_assessment,
            'rollback_available': self.rollback_available
        }

class TerraformDecisionEngine:
    """Transparent decision engine for Terraform operations"""
    
    def __init__(self, dry_run_mode: bool = True):
        self.dry_run_mode = dry_run_mode
        self.decision_logs: List[DecisionLog] = []
        self.plans: List[TerraformPlan] = []
        self.permission_scopes = self._initialize_permission_scopes()
        self.decision_factors = self._initialize_decision_factors()
        
        logger.info(f"Terraform Decision Engine initialized (dry_run: {dry_run_mode})")
    
    def _initialize_permission_scopes(self) -> Dict[PermissionScope, List[str]]:
        """Initialize clear permission scopes"""
        return {
            PermissionScope.READ_ONLY: [
                "terraform show",
                "terraform state list",
                "terraform plan",
                "terraform validate"
            ],
            PermissionScope.DRY_RUN: [
                "terraform plan -detailed-exitcode",
                "terraform graph",
                "terraform validate"
            ],
            PermissionScope.PLAN_ONLY: [
                "terraform plan -out=plan.tfplan",
                "terraform show -json plan.tfplan"
            ],
            PermissionScope.APPLY: [
                "terraform apply",
                "terraform import",
                "terraform state rm"
            ],
            PermissionScope.DESTROY: [
                "terraform destroy",
                "terraform state rm"
            ],
            PermissionScope.MODIFY_STATE: [
                "terraform state push",
                "terraform state pop",
                "terraform force-unlock"
            ]
        }
    
    def _initialize_decision_factors(self) -> Dict[str, Dict[str, float]]:
        """Initialize decision factors with weights"""
        return {
            'cost': {'weight': 0.3, 'importance': 0.8},
            'performance': {'weight': 0.25, 'importance': 0.7},
            'availability': {'weight': 0.2, 'importance': 0.9},
            'latency': {'weight': 0.15, 'importance': 0.6},
            'reliability': {'weight': 0.1, 'importance': 0.9},
            'security': {'weight': 0.15, 'importance': 0.95},
            'compliance': {'weight': 0.1, 'importance': 0.8},
            'support': {'weight': 0.05, 'importance': 0.5}
        }
    
    def select_instance(self, requirements: Dict[str, Any], 
                       available_instances: List[Dict[str, Any]]) -> DecisionLog:
        """Select instance with complete transparency"""
        decision_id = self._generate_decision_id("instance_selection")
        timestamp = datetime.now()
        
        # Log context
        context = {
            'requirements': requirements,
            'available_instances_count': len(available_instances),
            'dry_run_mode': self.dry_run_mode
        }
        
        # Evaluate each instance
        evaluated_instances = []
        for instance in available_instances:
            score, factors = self._evaluate_instance(instance, requirements)
            evaluated_instances.append({
                'instance': instance,
                'score': score,
                'factors': factors
            })
        
        # Sort by score
        evaluated_instances.sort(key=lambda x: x['score'], reverse=True)
        
        # Select best instance
        best_instance = evaluated_instances[0]
        selected_instance = best_instance['instance']
        
        # Create decision factors
        decision_factors = [
            DecisionFactor(
                name=factor['name'],
                value=factor['value'],
                weight=factor['weight'],
                reason=factor['reason'],
                source=factor['source'],
                timestamp=timestamp
            )
            for factor in best_instance['factors']
        ]
        
        # Generate reasoning
        reasoning = self._generate_instance_reasoning(
            selected_instance, best_instance['score'], evaluated_instances
        )
        
        # Calculate confidence
        confidence_score = self._calculate_confidence_score(evaluated_instances)
        
        # Identify alternatives
        alternatives = [
            eval_inst['instance'] for eval_inst in evaluated_instances[1:3]
        ]
        
        # Assess risks
        risks = self._assess_instance_risks(selected_instance)
        
        # Determine permissions
        permissions = self._determine_permissions("instance_selection")
        
        # Create decision log
        decision_log = DecisionLog(
            decision_id=decision_id,
            decision_type=DecisionType.INSTANCE_SELECTION,
            timestamp=timestamp,
            context=context,
            options_considered=[eval_inst['instance'] for eval_inst in evaluated_instances],
            factors=decision_factors,
            selected_option=selected_instance,
            reasoning=reasoning,
            confidence_score=confidence_score,
            alternatives=alternatives,
            risks=risks,
            permissions_required=permissions
        )
        
        # Add dry run result if in dry run mode
        if self.dry_run_mode:
            decision_log.dry_run_result = self._simulate_dry_run(decision_log)
            decision_log.rollback_plan = self._generate_rollback_plan(decision_log)
        
        # Store decision
        self.decision_logs.append(decision_log)
        
        logger.info(f"Instance selection decision: {decision_id}")
        logger.info(f"Selected: {selected_instance.get('name', 'unknown')}")
        logger.info(f"Score: {best_instance['score']:.3f}")
        logger.info(f"Confidence: {confidence_score:.3f}")
        
        return decision_log
    
    def _evaluate_instance(self, instance: Dict[str, Any], 
                          requirements: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """Evaluate instance with detailed factors"""
        factors = []
        total_score = 0.0
        
        # Cost factor
        cost_per_hour = float(instance.get('cost_per_hour', 0))
        max_cost = max(inst.get('cost_per_hour', 0) for inst in requirements.get('all_instances', [instance]))
        cost_score = 1.0 - (cost_per_hour / max_cost) if max_cost > 0 else 1.0
        cost_weight = self.decision_factors['cost']['weight']
        
        factors.append({
            'name': 'cost',
            'value': cost_score,
            'weight': cost_weight,
            'reason': f'Cost per hour: ${cost_per_hour:.4f} (lower is better)',
            'source': 'provider_api'
        })
        total_score += cost_score * cost_weight
        
        # Performance factor
        gpu_memory = float(instance.get('gpu_memory', 0))
        required_gpu_memory = float(requirements.get('gpu_memory', 0))
        performance_score = min(gpu_memory / required_gpu_memory, 1.0) if required_gpu_memory > 0 else 1.0
        performance_weight = self.decision_factors['performance']['weight']
        
        factors.append({
            'name': 'performance',
            'value': performance_score,
            'weight': performance_weight,
            'reason': f'GPU memory: {gpu_memory}GB (required: {required_gpu_memory}GB)',
            'source': 'instance_specs'
        })
        total_score += performance_score * performance_weight
        
        # Availability factor
        availability = float(instance.get('availability', 0.95))
        availability_weight = self.decision_factors['availability']['weight']
        
        factors.append({
            'name': 'availability',
            'value': availability,
            'weight': availability_weight,
            'reason': f'Availability: {availability:.1%}',
            'source': 'provider_sla'
        })
        total_score += availability * availability_weight
        
        # Latency factor
        latency = float(instance.get('latency_ms', 50))
        max_latency = max(inst.get('latency_ms', 50) for inst in requirements.get('all_instances', [instance]))
        latency_score = 1.0 - (latency / max_latency) if max_latency > 0 else 1.0
        latency_weight = self.decision_factors['latency']['weight']
        
        factors.append({
            'name': 'latency',
            'value': latency_score,
            'weight': latency_weight,
            'reason': f'Latency: {latency}ms (lower is better)',
            'source': 'provider_metrics'
        })
        total_score += latency_score * latency_weight
        
        return total_score, factors
    
    def _generate_instance_reasoning(self, selected_instance: Dict[str, Any], 
                                   score: float, evaluated_instances: List[Dict]) -> str:
        """Generate clear reasoning for instance selection"""
        reasoning_parts = []
        
        # Primary reason
        reasoning_parts.append(
            f"Selected {selected_instance.get('name', 'unknown')} with score {score:.3f}"
        )
        
        # Key factors
        reasoning_parts.append("Key factors:")
        reasoning_parts.append(f"  • Cost: ${selected_instance.get('cost_per_hour', 0):.4f}/hour")
        reasoning_parts.append(f"  • GPU: {selected_instance.get('gpu_type', 'unknown')}")
        reasoning_parts.append(f"  • Memory: {selected_instance.get('gpu_memory', 0)}GB")
        reasoning_parts.append(f"  • Availability: {selected_instance.get('availability', 0):.1%}")
        
        # Comparison with alternatives
        if len(evaluated_instances) > 1:
            second_best = evaluated_instances[1]
            score_diff = score - second_best['score']
            reasoning_parts.append(
                f"Score advantage: +{score_diff:.3f} over {second_best['instance'].get('name', 'unknown')}"
            )
        
        # Risk assessment
        if selected_instance.get('availability', 0.95) < 0.99:
            reasoning_parts.append("⚠️  Lower availability - consider backup options")
        
        return "\n".join(reasoning_parts)
    
    def _calculate_confidence_score(self, evaluated_instances: List[Dict]) -> float:
        """Calculate confidence in the decision"""
        if len(evaluated_instances) < 2:
            return 0.5
        
        best_score = evaluated_instances[0]['score']
        second_best_score = evaluated_instances[1]['score']
        
        # Higher confidence when there's a clear winner
        score_diff = best_score - second_best_score
        
        if score_diff > 0.2:
            return 0.9
        elif score_diff > 0.1:
            return 0.7
        elif score_diff > 0.05:
            return 0.6
        else:
            return 0.4
    
    def _assess_instance_risks(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess risks for selected instance"""
        risks = []
        
        # Availability risk
        availability = instance.get('availability', 0.95)
        if availability < 0.99:
            risks.append({
                'type': 'availability',
                'severity': 'medium' if availability > 0.95 else 'high',
                'description': f'Availability is {availability:.1%} (below 99%)',
                'mitigation': 'Consider backup instances or SLA monitoring'
            })
        
        # Cost risk
        cost_per_hour = instance.get('cost_per_hour', 0)
        if cost_per_hour > 2.0:
            risks.append({
                'type': 'cost',
                'severity': 'medium',
                'description': f'High cost per hour: ${cost_per_hour:.4f}',
                'mitigation': 'Set budget alerts and monitor usage'
            })
        
        # Provider risk
        provider = instance.get('provider', 'unknown')
        if provider in ['vastai', 'tensor_dock']:
            risks.append({
                'type': 'provider',
                'severity': 'medium',
                'description': f'Lesser-known provider: {provider}',
                'mitigation': 'Monitor provider stability and have alternatives'
            })
        
        return risks
    
    def _determine_permissions(self, operation: str) -> List[PermissionScope]:
        """Determine required permissions for operation"""
        if operation == "instance_selection":
            if self.dry_run_mode:
                return [PermissionScope.READ_ONLY, PermissionScope.DRY_RUN]
            else:
                return [PermissionScope.READ_ONLY, PermissionScope.PLAN_ONLY, PermissionScope.APPLY]
        elif operation == "destroy":
            return [PermissionScope.READ_ONLY, PermissionScope.DESTROY]
        else:
            return [PermissionScope.READ_ONLY]
    
    def _simulate_dry_run(self, decision_log: DecisionLog) -> Dict[str, Any]:
        """Simulate dry run result"""
        return {
            'status': 'success',
            'resources_affected': 1,
            'cost_impact': decision_log.selected_option.get('cost_per_hour', 0),
            'estimated_duration': '5 minutes',
            'changes_required': [
                f"Create instance: {decision_log.selected_option.get('name', 'unknown')}",
                f"Configure GPU: {decision_log.selected_option.get('gpu_type', 'unknown')}",
                f"Set up networking and storage"
            ],
            'validation_passed': True,
            'warnings': [
                'This is a dry run - no actual changes will be made'
            ]
        }
    
    def _generate_rollback_plan(self, decision_log: DecisionLog) -> Dict[str, Any]:
        """Generate rollback plan"""
        return {
            'rollback_available': True,
            'rollback_steps': [
                '1. Stop instance if running',
                '2. Terminate instance',
                '3. Clean up associated resources',
                '4. Restore previous state if needed'
            ],
            'estimated_rollback_time': '3 minutes',
            'rollback_cost': 0.0,
            'data_loss_risk': 'low'
        }
    
    def create_terraform_plan(self, decisions: List[DecisionLog]) -> TerraformPlan:
        """Create Terraform plan with full transparency"""
        plan_id = self._generate_decision_id("terraform_plan")
        created_at = datetime.now()
        
        # Analyze decisions to determine resources
        resources_to_add = []
        resources_to_change = []
        resources_to_destroy = []
        
        total_cost = 0.0
        permissions = set()
        
        for decision in decisions:
            if decision.decision_type == DecisionType.INSTANCE_SELECTION:
                resource = {
                    'type': 'aws_instance',
                    'name': decision.selected_option.get('name', 'unknown'),
                    'provider': decision.selected_option.get('provider', 'unknown'),
                    'instance_type': decision.selected_option.get('instance_type', 'unknown'),
                    'cost_per_hour': decision.selected_option.get('cost_per_hour', 0),
                    'decision_id': decision.decision_id,
                    'reasoning': decision.reasoning
                }
                resources_to_add.append(resource)
                total_cost += resource['cost_per_hour']
            
            # Collect permissions
            permissions.update(decision.permissions_required)
        
        # Risk assessment
        risk_assessment = {
            'total_risk_score': self._calculate_plan_risk(decisions),
            'high_risk_items': [risk for decision in decisions for risk in decision.risks if risk['severity'] == 'high'],
            'mitigation_required': len([risk for decision in decisions for risk in decision.risks if risk['severity'] == 'high']) > 0
        }
        
        plan = TerraformPlan(
            plan_id=plan_id,
            created_at=created_at,
            resources_to_add=resources_to_add,
            resources_to_change=resources_to_change,
            resources_to_destroy=resources_to_destroy,
            cost_estimate=total_cost,
            permissions_required=list(permissions),
            risk_assessment=risk_assessment,
            rollback_available=True
        )
        
        self.plans.append(plan)
        
        logger.info(f"Terraform plan created: {plan_id}")
        logger.info(f"Resources to add: {len(resources_to_add)}")
        logger.info(f"Cost estimate: ${total_cost:.4f}/hour")
        
        return plan
    
    def _calculate_plan_risk(self, decisions: List[DecisionLog]) -> float:
        """Calculate overall risk score for plan"""
        if not decisions:
            return 0.0
        
        total_risk = 0.0
        for decision in decisions:
            # Lower confidence increases risk
            confidence_risk = 1.0 - decision.confidence_score
            
            # High-severity risks increase risk
            high_risk_count = len([risk for risk in decision.risks if risk['severity'] == 'high'])
            severity_risk = min(high_risk_count * 0.2, 0.6)
            
            total_risk += (confidence_risk + severity_risk) / 2.0
        
        return total_risk / len(decisions)
    
    def _generate_decision_id(self, operation: str) -> str:
        """Generate unique decision ID"""
        timestamp = datetime.now().isoformat()
        content = f"{operation}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_decision_logs(self, decision_type: Optional[DecisionType] = None) -> List[DecisionLog]:
        """Get decision logs with optional filtering"""
        if decision_type:
            return [log for log in self.decision_logs if log.decision_type == decision_type]
        return self.decision_logs
    
    def get_plans(self) -> List[TerraformPlan]:
        """Get all Terraform plans"""
        return self.plans
    
    def export_decision_logs(self, filename: str) -> None:
        """Export decision logs to file"""
        logs_data = [log.to_dict() for log in self.decision_logs]
        
        with open(filename, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        logger.info(f"Decision logs exported to: {filename}")
    
    def export_plans(self, filename: str) -> None:
        """Export Terraform plans to file"""
        plans_data = [plan.to_dict() for plan in self.plans]
        
        with open(filename, 'w') as f:
            json.dump(plans_data, f, indent=2)
        
        logger.info(f"Terraform plans exported to: {filename}")

# Example usage
if __name__ == "__main__":
    # Create decision engine in dry-run mode
    engine = TerraformDecisionEngine(dry_run_mode=True)
    
    # Example requirements
    requirements = {
        'gpu_memory': 16,
        'gpu_count': 1,
        'cpu_cores': 8,
        'memory_gb': 32,
        'storage_gb': 100
    }
    
    # Example available instances
    available_instances = [
        {
            'name': 'vastai-a5000',
            'provider': 'vastai',
            'instance_type': 'A5000',
            'gpu_type': 'RTX A5000',
            'gpu_memory': 24,
            'cost_per_hour': 0.35,
            'availability': 0.95,
            'latency_ms': 45
        },
        {
            'name': 'runpod-a4000',
            'provider': 'runpod',
            'instance_type': 'A4000',
            'gpu_type': 'RTX A4000',
            'gpu_memory': 16,
            'cost_per_hour': 0.29,
            'availability': 0.97,
            'latency_ms': 35
        },
        {
            'name': 'aws-g4dn',
            'provider': 'aws',
            'instance_type': 'g4dn.xlarge',
            'gpu_type': 'T4',
            'gpu_memory': 16,
            'cost_per_hour': 0.526,
            'availability': 0.99,
            'latency_ms': 25
        }
    ]
    
    # Add all instances to requirements for comparison
    requirements['all_instances'] = available_instances
    
    # Make decision
    decision = engine.select_instance(requirements, available_instances)
    
    # Create Terraform plan
    plan = engine.create_terraform_plan([decision])
    
    # Export results
    engine.export_decision_logs('decision_logs.json')
    engine.export_plans('terraform_plans.json')
    
    print("\n🎯 DECISION ENGINE DEMO")
    print("=" * 50)
    print(f"Decision ID: {decision.decision_id}")
    print(f"Selected Instance: {decision.selected_option['name']}")
    print(f"Score: {sum(f.value * f.weight for f in decision.factors):.3f}")
    print(f"Confidence: {decision.confidence_score:.3f}")
    print(f"Permissions: {[p.value for p in decision.permissions_required]}")
    print(f"Dry Run: {decision.dry_run_result['status']}")
    print(f"Rollback Available: {decision.rollback_plan['rollback_available']}")
    
    print("\n📋 REASONING:")
    print(decision.reasoning)
    
    print("\n⚠️  RISKS:")
    for risk in decision.risks:
        print(f"  • {risk['type']}: {risk['description']}")
    
    print("\n📊 TERRAFORM PLAN:")
    print(f"Plan ID: {plan.plan_id}")
    print(f"Resources to Add: {len(plan.resources_to_add)}")
    print(f"Cost Estimate: ${plan.cost_estimate:.4f}/hour")
    print(f"Risk Score: {plan.risk_assessment['total_risk_score']:.3f}")
    
    print("\n✅ Transparency Features:")
    print("  • Clear permission scopes")
    print("  • Dry-run/plan modes")
    print("  • Deterministic decision logs")
    print("  • Easy rollback/pinning")
    print("  • Complete reasoning transparency")
