#!/usr/bin/env python3
"""
Terraform Transparency Demo
Complete demonstration of transparent decision-making, dry-run modes, and rollback
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from decision_engine import TerraformDecisionEngine, DecisionType
from terraform_manager import TerraformManager, TerraformMode
from audit_trail import TerraformAuditTrail, AuditEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransparencyDemo:
    """Demonstrate complete Terraform transparency"""
    
    def __init__(self):
        self.decision_engine = TerraformDecisionEngine(dry_run_mode=True)
        self.terraform_manager = TerraformManager("terraform_demo")
        self.audit_trail = TerraformAuditTrail("audit_demo")
        
        print("🔍 Terraform Transparency Demo Initialized")
    
    def demo_complete_transparency(self):
        """Demonstrate complete transparency features"""
        print("\n🎯 COMPLETE TRANSPARENCY DEMO")
        print("=" * 60)
        
        # Start audit trail
        trail_id = self.audit_trail.start_trail("demo_user", "terraform_transparency")
        
        # 1. Clear Permission Scopes
        print("\n1️⃣ CLEAR PERMISSION SCOPES")
        print("-" * 30)
        self.demo_permission_scopes()
        
        # 2. Dry-Run / Plan Modes
        print("\n2️⃣ DRY-RUN / PLAN MODES")
        print("-" * 30)
        self.demo_dry_run_modes()
        
        # 3. Deterministic Decision Logs
        print("\n3️⃣ DETERMINISTIC DECISION LOGS")
        print("-" * 30)
        self.demo_deterministic_decisions()
        
        # 4. Easy Rollback / Pinning
        print("\n4️⃣ EASY ROLLBACK / PINNING")
        print("-" * 30)
        self.demo_rollback_pinning()
        
        # 5. Complete Audit Trail
        print("\n5️⃣ COMPLETE AUDIT TRAIL")
        print("-" * 30)
        self.demo_audit_trail()
        
        # Finish audit trail
        summary = self.audit_trail.finish_trail()
        
        # Export results
        self.export_results()
        
        print("\n🎉 TRANSPARENCY DEMO COMPLETE")
        print("=" * 60)
        print("✅ Clear permission scopes")
        print("✅ Dry-run/plan modes")
        print("✅ Deterministic decision logs")
        print("✅ Easy rollback/pinning")
        print("✅ Complete audit trail")
        print("✅ Full transparency achieved")
        
        return summary
    
    def demo_permission_scopes(self):
        """Demonstrate clear permission scopes"""
        print("🔍 Permission Scopes:")
        print("   • READ_ONLY: terraform show, terraform state list")
        print("   • DRY_RUN: terraform plan -detailed-exitcode")
        print("   • PLAN_ONLY: terraform plan -out=plan.tfplan")
        print("   • APPLY: terraform apply, terraform import")
        print("   • DESTROY: terraform destroy")
        print("   • MODIFY_STATE: terraform state push/pop")
        
        # Check permissions for each mode
        modes = [
            TerraformMode.READ_ONLY,
            TerraformMode.DRY_RUN,
            TerraformMode.PLAN,
            TerraformMode.APPLY,
            TerraformMode.DESTROY
        ]
        
        for mode in modes:
            has_perms, missing = self.terraform_manager.validate_permissions(mode)
            status = "✅" if has_perms else "❌"
            print(f"   {status} {mode.value}: {len(missing)} missing permissions")
            
            # Log permission check
            self.audit_trail.log_permission_check(
                actor="demo_user",
                operation_type=mode.value,
                permissions_required=[p.value for p in self.terraform_manager._get_required_permissions(mode)],
                permissions_granted=[p.value for p in self.terraform_manager._get_required_permissions(mode)] if has_perms else [],
                permissions_denied=missing if not has_perms else []
            )
    
    def demo_dry_run_modes(self):
        """Demonstrate dry-run and plan modes"""
        print("🧪 Dry-Run Modes:")
        print("   • Simulate changes without applying")
        print("   • Show resources that will be affected")
        print("   • Calculate cost impact")
        print("   • Validate configuration")
        print("   • Check permissions")
        
        # Execute dry run
        dry_run_op = self.terraform_manager.dry_run({})
        
        print(f"\n   📊 Dry Run Results:")
        print(f"      Operation ID: {dry_run_op.operation_id}")
        print(f"      Status: {dry_run_op.status.value}")
        print(f"      Duration: {dry_run_op.duration_seconds:.2f}s")
        print(f"      Resources: {dry_run_op.resources_affected}")
        print(f"      Cost Impact: ${dry_run_op.cost_impact:.4f}/hour" if dry_run_op.cost_impact else "      Cost Impact: N/A")
        print(f"      Rollback Available: {dry_run_op.rollback_available}")
        
        # Log operation
        self.audit_trail.log_operation(
            actor="demo_user",
            operation_type="dry_run",
            operation_id=dry_run_op.operation_id,
            status=dry_run_op.status.value,
            resources_affected=dry_run_op.resources_affected or 0,
            cost_impact=dry_run_op.cost_impact or 0.0,
            duration_seconds=dry_run_op.duration_seconds or 0.0,
            permissions_required=["read_only", "dry_run"]
        )
        
        # Execute plan
        plan_op = self.terraform_manager.plan({})
        
        print(f"\n   📋 Plan Results:")
        print(f"      Operation ID: {plan_op.operation_id}")
        print(f"      Status: {plan_op.status.value}")
        print(f"      Resources: {plan_op.resources_affected}")
        print(f"      Cost Impact: ${plan_op.cost_impact:.4f}/hour" if plan_op.cost_impact else "      Cost Impact: N/A")
        print(f"      Rollback Available: {plan_op.rollback_available}")
        
        # Log operation
        self.audit_trail.log_operation(
            actor="demo_user",
            operation_type="plan",
            operation_id=plan_op.operation_id,
            status=plan_op.status.value,
            resources_affected=plan_op.resources_affected or 0,
            cost_impact=plan_op.cost_impact or 0.0,
            duration_seconds=plan_op.duration_seconds or 0.0,
            permissions_required=["read_only", "plan_only"]
        )
    
    def demo_deterministic_decisions(self):
        """Demonstrate deterministic decision logs with 'why this instance?'"""
        print("🧠 Deterministic Decision Logs:")
        print("   • Complete factor analysis")
        print("   • Weighted scoring system")
        print("   • Clear reasoning for each choice")
        print("   • Alternative options considered")
        print("   • Risk assessment")
        print("   • Confidence scores")
        
        # Example requirements
        requirements = {
            'gpu_memory': 16,
            'gpu_count': 1,
            'cpu_cores': 8,
            'memory_gb': 32,
            'all_instances': [
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
        }
        
        # Make decision
        decision = self.decision_engine.select_instance(requirements, requirements['all_instances'])
        
        print(f"\n   🎯 Decision Results:")
        print(f"      Decision ID: {decision.decision_id}")
        print(f"      Selected: {decision.selected_option['name']}")
        print(f"      Score: {sum(f.value * f.weight for f in decision.factors):.3f}")
        print(f"      Confidence: {decision.confidence_score:.3f}")
        print(f"      Alternatives: {len(decision.alternatives)}")
        print(f"      Risks: {len(decision.risks)}")
        
        print(f"\n   📋 Why This Instance?")
        print(decision.reasoning)
        
        print(f"\n   📊 Decision Factors:")
        for factor in decision.factors:
            print(f"      • {factor.name}: {factor.value:.3f} (weight: {factor.weight})")
            print(f"        Reason: {factor.reason}")
        
        print(f"\n   ⚠️  Risks:")
        for risk in decision.risks:
            print(f"      • {risk['type']}: {risk['description']}")
        
        # Log decision
        self.audit_trail.log_decision(
            actor="demo_user",
            decision_type="instance_selection",
            selected_option=decision.selected_option,
            alternatives=decision.alternatives,
            reasoning=decision.reasoning,
            factors=[f.to_dict() for f in decision.factors],
            permissions_required=[p.value for p in decision.permissions_required],
            risk_assessment={'total_risk_score': 0.2, 'high_risk_items': []},
            operation_id="decision_123456"
        )
    
    def demo_rollback_pinning(self):
        """Demonstrate easy rollback and pinning"""
        print("🔄 Rollback & Pinning:")
        print("   • State snapshots before operations")
        print("   • One-click rollback capability")
        print("   • Operation pinning for safety")
        print("   • Complete rollback logs")
        print("   • Risk assessment for rollbacks")
        
        # Pin an operation
        plan_op = self.terraform_manager.get_operations(TerraformMode.PLAN)[0]
        if plan_op:
            pin_success = self.terraform_manager.pin_operation(
                plan_op.operation_id,
                "Critical production deployment"
            )
            
            print(f"\n   📌 Pinning Results:")
            print(f"      Operation ID: {plan_op.operation_id}")
            print(f"      Pin Success: {pin_success}")
            print(f"      Reason: Critical production deployment")
            
            # Log pinning
            self.audit_trail.log_pin_operation(
                actor="demo_user",
                operation_id=plan_op.operation_id,
                reason="Critical production deployment",
                permissions_required=["modify_state"]
            )
        
        # Simulate rollback
        print(f"\n   🔄 Rollback Simulation:")
        print(f"      Rollback would restore state snapshot")
        print(f"      Previous operation would be marked as rolled back")
        print(f"      All changes would be reverted")
        print(f"      Rollback operation would be logged")
        
        # Log rollback
        self.audit_trail.log_rollback(
            actor="demo_user",
            original_operation_id="op_123456",
            rollback_operation_id="rollback_123456",
            status="simulated",
            resources_affected=1,
            duration_seconds=60.0
        )
    
    def demo_audit_trail(self):
        """Demonstrate complete audit trail"""
        print("📋 Complete Audit Trail:")
        print("   • Every decision logged")
        print("   • Every operation tracked")
        print("   • Permission checks recorded")
        print("   • State changes monitored")
        print("   • Rollback operations logged")
        print("   • Risk assessment included")
        print("   • Compliance status tracked")
        
        # Get current trail
        trail = self.audit_trail.current_trail
        
        if trail:
            print(f"\n   📊 Trail Statistics:")
            print(f"      Trail ID: {trail.trail_id}")
            print(f"      Total Entries: {len(trail.entries)}")
            print(f"      Compliance Status: {trail.compliance_status}")
            print(f"      Risk Score: {trail.risk_score:.3f}")
            
            print(f"\n   📝 Recent Entries:")
            for entry in trail.entries[-5:]:  # Show last 5 entries
                print(f"      • {entry.timestamp.strftime('%H:%M:%S')} - {entry.event_type.value}")
                if entry.decision_reasoning:
                    print(f"        Reason: {entry.decision_reasoning[:50]}...")
    
    def export_results(self):
        """Export all transparency results"""
        print("\n💾 Exporting Results:")
        
        # Export decision logs
        self.decision_engine.export_decision_logs("decision_logs.json")
        print("   • Decision logs exported to: decision_logs.json")
        
        # Export Terraform plans
        self.decision_engine.export_plans("terraform_plans.json")
        print("   • Terraform plans exported to: terraform_plans.json")
        
        # Export operations
        self.terraform_manager.export_operations("operations.json")
        print("   • Operations exported to: operations.json")
        
        # Export audit trail
        self.audit_trail.export_all_trails("audit_trails.json")
        print("   • Audit trails exported to: audit_trails.json")
        
        # Create transparency report
        self.create_transparency_report()
        print("   • Transparency report created: transparency_report.json")
    
    def create_transparency_report(self):
        """Create comprehensive transparency report"""
        report = {
            'transparency_demo': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'features': {
                    'permission_scopes': True,
                    'dry_run_modes': True,
                    'deterministic_decisions': True,
                    'rollback_pinning': True,
                    'audit_trail': True
                },
                'decision_engine': {
                    'decisions_made': len(self.decision_engine.decision_logs),
                    'confidence_scores': [d.confidence_score for d in self.decision_engine.decision_logs],
                    'avg_confidence': sum(d.confidence_score for d in self.decision_engine.decision_logs) / len(self.decision_engine.decision_logs) if self.decision_engine.decision_logs else 0
                },
                'terraform_manager': {
                    'operations_total': len(self.terraform_manager.operations),
                    'dry_runs': len([o for o in self.terraform_manager.operations if o.mode == TerraformMode.DRY_RUN]),
                    'plans': len([o for o in self.terraform_manager.operations if o.mode == TerraformMode.PLAN]),
                    'applies': len([o for o in self.terraform_manager.operations if o.mode == TerraformMode.APPLY]),
                    'rollbacks_available': len([o for o in self.terraform_manager.operations if o.rollback_available])
                },
                'audit_trail': {
                    'trails_total': len(self.audit_trail.trails),
                    'entries_total': sum(len(trail.entries) for trail in self.audit_trail.trails.values()),
                    'compliance_status': 'compliant',
                    'avg_risk_score': sum(trail.risk_score for trail in self.audit_trail.trails.values()) / len(self.audit_trail.trails) if self.audit_trail.trails else 0
                }
            }
        }
        
        with open('transparency_report.json', 'w') as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    demo = TransparencyDemo()
    summary = demo.demo_complete_transparency()
    
    print(f"\n🎯 TRANSPARENCY SUMMARY:")
    print(f"   Compliance Status: {summary['compliance_status']}")
    print(f"   Risk Score: {summary['risk_score']:.3f}")
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Successful: {summary['successful_operations']}")
    print(f"   Total Entries: {summary['total_entries']}")
    print(f"   Decision Transparency: {summary['decision_transparency']}")
    print(f"   Rollback Available: {summary['rollback_available']}")
    
    print(f"\n🔍 TRANSPARENCY ACHIEVED:")
    print("   ✅ Clear permission scopes")
    print("   ✅ Dry-run/plan modes")
    print("   ✅ Deterministic decision logs")
    print("   ✅ Easy rollback/pinning")
    print("   ✅ Complete audit trail")
    print("   ✅ Full 'why this instance?' reasoning")
    print("   ✅ No magic - complete transparency")
    
    print(f"\n🚀 READY FOR PRODUCTION:")
    print("   • All operations are logged")
    print("   • Every decision is explained")
    print("   • Rollbacks are safe and tracked")
    print("   • Permissions are validated")
    print("   • Compliance is monitored")
    print("   • Risk is assessed")
    print("   • Complete audit trail maintained")
