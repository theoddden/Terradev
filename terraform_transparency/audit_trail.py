#!/usr/bin/env python3
"""
Terraform Audit Trail - Complete Transparency
Deterministic decision logs with full "why this instance?" reasoning
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditEvent(Enum):
    """Types of audit events"""
    DECISION_MADE = "decision_made"
    OPERATION_STARTED = "operation_started"
    OPERATION_COMPLETED = "operation_completed"
    STATE_CHANGED = "state_changed"
    PERMISSION_CHECKED = "permission_checked"
    ROLLBACK_EXECUTED = "rollback_executed"
    PINNED_OPERATION = "pinned_operation"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class AuditEntry:
    """Single audit entry"""
    entry_id: str
    timestamp: datetime
    event_type: AuditEvent
    actor: str
    operation_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Dict[str, Any]
    decision_reasoning: Optional[str]
    permissions_required: List[str]
    risk_assessment: Optional[Dict[str, Any]]
    rollback_available: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'operation_id': self.operation_id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'decision_reasoning': self.decision_reasoning,
            'permissions_required': self.permissions_required,
            'risk_assessment': self.risk_assessment,
            'rollback_available': self.rollback_available,
            'metadata': self.metadata
        }

@dataclass
class AuditTrail:
    """Complete audit trail"""
    trail_id: str
    created_at: datetime
    entries: List[AuditEntry]
    summary: Dict[str, Any]
    compliance_status: str
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trail_id': self.trail_id,
            'created_at': self.created_at.isoformat(),
            'entries': [entry.to_dict() for entry in self.entries],
            'summary': self.summary,
            'compliance_status': self.compliance_status,
            'risk_score': self.risk_score
        }

class TerraformAuditTrail:
    """Complete audit trail for Terraform operations"""
    
    def __init__(self, storage_path: str = "audit_trail"):
        self.storage_path = storage_path
        self.trails: Dict[str, AuditTrail] = {}
        self.current_trail: Optional[AuditTrail] = None
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"Audit trail initialized in {storage_path}")
    
    def start_trail(self, actor: str, operation_type: str) -> str:
        """Start a new audit trail"""
        trail_id = self._generate_trail_id()
        
        trail = AuditTrail(
            trail_id=trail_id,
            created_at=datetime.now(),
            entries=[],
            summary={
                'actor': actor,
                'operation_type': operation_type,
                'started_at': datetime.now().isoformat(),
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'rollback_operations': 0
            },
            compliance_status="in_progress",
            risk_score=0.0
        )
        
        self.trails[trail_id] = trail
        self.current_trail = trail
        
        # Add initial entry
        self.add_entry(
            event_type=AuditEvent.OPERATION_STARTED,
            actor=actor,
            details={
                'operation_type': operation_type,
                'trail_started': True
            },
            metadata={'trail_id': trail_id}
        )
        
        logger.info(f"Audit trail started: {trail_id}")
        return trail_id
    
    def add_entry(self, event_type: AuditEvent, actor: str, 
                   operation_id: Optional[str] = None,
                   resource_type: Optional[str] = None,
                   resource_id: Optional[str] = None,
                   details: Optional[Dict[str, Any]] = None,
                   decision_reasoning: Optional[str] = None,
                   permissions_required: Optional[List[str]] = None,
                   risk_assessment: Optional[Dict[str, Any]] = None,
                   rollback_available: bool = False,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add entry to current audit trail"""
        if not self.current_trail:
            raise ValueError("No active audit trail. Start a trail first.")
        
        entry_id = self._generate_entry_id()
        
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            operation_id=operation_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            decision_reasoning=decision_reasoning,
            permissions_required=permissions_required or [],
            risk_assessment=risk_assessment,
            rollback_available=rollback_available,
            metadata=metadata or {}
        )
        
        self.current_trail.entries.append(entry)
        
        # Update summary
        self._update_summary(event_type)
        
        # Update risk score
        self._update_risk_score()
        
        # Update compliance status
        self._update_compliance_status()
        
        logger.info(f"Audit entry added: {entry_id} - {event_type.value}")
        return entry_id
    
    def log_decision(self, actor: str, decision_type: str, 
                     selected_option: Dict[str, Any],
                     alternatives: List[Dict[str, Any]],
                     reasoning: str,
                     factors: List[Dict[str, Any]],
                     permissions_required: List[str],
                     risk_assessment: Dict[str, Any],
                     operation_id: Optional[str] = None) -> str:
        """Log decision with complete transparency"""
        details = {
            'decision_type': decision_type,
            'selected_option': selected_option,
            'alternatives': alternatives,
            'factors': factors,
            'alternatives_count': len(alternatives)
        }
        
        return self.add_entry(
            event_type=AuditEvent.DECISION_MADE,
            actor=actor,
            operation_id=operation_id,
            resource_type=selected_option.get('type', 'unknown'),
            resource_id=selected_option.get('id', 'unknown'),
            details=details,
            decision_reasoning=reasoning,
            permissions_required=permissions_required,
            risk_assessment=risk_assessment,
            rollback_available=True,
            metadata={
                'decision_transparency': 'complete',
                'why_this_instance': reasoning,
                'factors_considered': len(factors)
            }
        )
    
    def log_operation(self, actor: str, operation_type: str,
                      operation_id: str, status: str,
                      resources_affected: int,
                      cost_impact: float,
                      duration_seconds: float,
                      permissions_required: List[str],
                      error_message: Optional[str] = None) -> str:
        """Log operation with full details"""
        details = {
            'operation_type': operation_type,
            'status': status,
            'resources_affected': resources_affected,
            'cost_impact': cost_impact,
            'duration_seconds': duration_seconds,
            'error_message': error_message
        }
        
        event_type = AuditEvent.OPERATION_COMPLETED
        if status == 'failed':
            event_type = AuditEvent.ERROR_OCCURRED
        
        return self.add_entry(
            event_type=event_type,
            actor=actor,
            operation_id=operation_id,
            details=details,
            permissions_required=permissions_required,
            rollback_available=status == 'success',
            metadata={
                'operation_transparency': 'complete',
                'cost_transparency': cost_impact,
                'duration_transparency': duration_seconds
            }
        )
    
    def log_rollback(self, actor: str, original_operation_id: str,
                     rollback_operation_id: str, status: str,
                     resources_affected: int,
                     duration_seconds: float) -> str:
        """Log rollback operation"""
        details = {
            'original_operation_id': original_operation_id,
            'rollback_operation_id': rollback_operation_id,
            'status': status,
            'resources_affected': resources_affected,
            'duration_seconds': duration_seconds
        }
        
        return self.add_entry(
            event_type=AuditEvent.ROLLBACK_EXECUTED,
            actor=actor,
            operation_id=rollback_operation_id,
            details=details,
            permissions_required=['modify_state'],
            rollback_available=False,
            metadata={
                'rollback_transparency': 'complete',
                'original_operation': original_operation_id
            }
        )
    
    def log_permission_check(self, actor: str, operation_type: str,
                           permissions_required: List[str],
                           permissions_granted: List[str],
                           permissions_denied: List[str]) -> str:
        """Log permission check"""
        details = {
            'operation_type': operation_type,
            'permissions_required': permissions_required,
            'permissions_granted': permissions_granted,
            'permissions_denied': permissions_denied,
            'check_passed': len(permissions_denied) == 0
        }
        
        return self.add_entry(
            event_type=AuditEvent.PERMISSION_CHECKED,
            actor=actor,
            details=details,
            permissions_required=permissions_required,
            rollback_available=False,
            metadata={
                'permission_transparency': 'complete',
                'access_control': 'checked'
            }
        )
    
    def log_state_change(self, actor: str, operation_id: str,
                        state_before: Dict[str, Any],
                        state_after: Dict[str, Any],
                        resources_changed: List[str]) -> str:
        """Log state change"""
        details = {
            'operation_id': operation_id,
            'state_before': state_before,
            'state_after': state_after,
            'resources_changed': resources_changed,
            'changes_count': len(resources_changed)
        }
        
        return self.add_entry(
            event_type=AuditEvent.STATE_CHANGED,
            actor=actor,
            operation_id=operation_id,
            details=details,
            permissions_required=['modify_state'],
            rollback_available=True,
            metadata={
                'state_transparency': 'complete',
                'change_tracking': 'enabled'
            }
        )
    
    def log_pin_operation(self, actor: str, operation_id: str,
                         reason: str, permissions_required: List[str]) -> str:
        """Log operation pinning"""
        details = {
            'operation_id': operation_id,
            'pin_reason': reason,
            'pinned_at': datetime.now().isoformat()
        }
        
        return self.add_entry(
            event_type=AuditEvent.PINNED_OPERATION,
            actor=actor,
            operation_id=operation_id,
            details=details,
            permissions_required=permissions_required,
            rollback_available=False,
            metadata={
                'pin_transparency': 'complete',
                'safety_measure': 'enabled'
            }
        )
    
    def finish_trail(self) -> Dict[str, Any]:
        """Finish current audit trail and return summary"""
        if not self.current_trail:
            raise ValueError("No active audit trail to finish.")
        
        trail = self.current_trail
        
        # Add completion entry
        self.add_entry(
            event_type=AuditEvent.OPERATION_COMPLETED,
            actor="system",
            details={
                'trail_completed': True,
                'final_status': 'completed'
            },
            metadata={'trail_completion': True}
        )
        
        # Generate final summary
        summary = self._generate_final_summary(trail)
        
        # Save trail to file
        self._save_trail(trail)
        
        logger.info(f"Audit trail completed: {trail.trail_id}")
        logger.info(f"Final status: {trail.compliance_status}")
        logger.info(f"Risk score: {trail.risk_score:.3f}")
        
        self.current_trail = None
        
        return summary
    
    def get_trail(self, trail_id: str) -> Optional[AuditTrail]:
        """Get audit trail by ID"""
        return self.trails.get(trail_id)
    
    def get_all_trails(self) -> Dict[str, AuditTrail]:
        """Get all audit trails"""
        return self.trails
    
    def export_trail(self, trail_id: str, filename: str) -> None:
        """Export audit trail to file"""
        trail = self.get_trail(trail_id)
        if not trail:
            raise ValueError(f"Trail {trail_id} not found")
        
        with open(filename, 'w') as f:
            json.dump(trail.to_dict(), f, indent=2)
        
        logger.info(f"Audit trail exported to: {filename}")
    
    def export_all_trails(self, filename: str) -> None:
        """Export all audit trails to file"""
        trails_data = {
            trail_id: trail.to_dict() for trail_id, trail in self.trails.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(trails_data, f, indent=2)
        
        logger.info(f"All audit trails exported to: {filename}")
    
    def _update_summary(self, event_type: AuditEvent) -> None:
        """Update trail summary based on event type"""
        if not self.current_trail:
            return
        
        summary = self.current_trail.summary
        
        if event_type == AuditEvent.OPERATION_COMPLETED:
            summary['total_operations'] += 1
            # Would check actual status from details
            summary['successful_operations'] += 1
        elif event_type == AuditEvent.ERROR_OCCURRED:
            summary['total_operations'] += 1
            summary['failed_operations'] += 1
        elif event_type == AuditEvent.ROLLBACK_EXECUTED:
            summary['rollback_operations'] += 1
    
    def _update_risk_score(self) -> None:
        """Update risk score based on entries"""
        if not self.current_trail:
            return
        
        risk_factors = []
        
        for entry in self.current_trail.entries:
            # High-risk events
            if entry.event_type == AuditEvent.ERROR_OCCURRED:
                risk_factors.append(0.3)
            elif entry.event_type == AuditEvent.ROLLBACK_EXECUTED:
                risk_factors.append(0.2)
            elif entry.event_type == AuditEvent.PINNED_OPERATION:
                risk_factors.append(0.1)
            
            # Risk assessment from entry
            if entry.risk_assessment:
                risk_score = entry.risk_assessment.get('total_risk_score', 0.0)
                risk_factors.append(risk_score)
            
            # Denied permissions
            if entry.permissions_required and entry.details.get('permissions_denied'):
                risk_factors.append(0.4)
        
        # Calculate average risk score
        if risk_factors:
            self.current_trail.risk_score = sum(risk_factors) / len(risk_factors)
        else:
            self.current_trail.risk_score = 0.0
    
    def _update_compliance_status(self) -> None:
        """Update compliance status based on entries"""
        if not self.current_trail:
            return
        
        risk_score = self.current_trail.risk_score
        failed_ops = self.current_trail.summary.get('failed_operations', 0)
        total_ops = self.current_trail.summary.get('total_operations', 1)
        
        if risk_score > 0.7 or failed_ops > 0:
            self.current_trail.compliance_status = "non_compliant"
        elif risk_score > 0.3:
            self.current_trail.compliance_status = "warning"
        else:
            self.current_trail.compliance_status = "compliant"
    
    def _generate_final_summary(self, trail: AuditTrail) -> Dict[str, Any]:
        """Generate final summary for audit trail"""
        summary = trail.summary.copy()
        summary.update({
            'trail_id': trail.trail_id,
            'created_at': trail.created_at.isoformat(),
            'completed_at': datetime.now().isoformat(),
            'total_entries': len(trail.entries),
            'compliance_status': trail.compliance_status,
            'risk_score': trail.risk_score,
            'decision_transparency': 'complete',
            'rollback_available': any(entry.rollback_available for entry in trail.entries)
        })
        
        return summary
    
    def _save_trail(self, trail: AuditTrail) -> None:
        """Save audit trail to file"""
        filename = os.path.join(self.storage_path, f"trail_{trail.trail_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(trail.to_dict(), f, indent=2)
    
    def _generate_trail_id(self) -> str:
        """Generate unique trail ID"""
        timestamp = datetime.now().isoformat()
        content = f"trail_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().isoformat()
        content = f"entry_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

# Example usage
if __name__ == "__main__":
    # Create audit trail
    audit = TerraformAuditTrail("audit_demo")
    
    print("🔍 AUDIT TRAIL DEMO")
    print("=" * 50)
    
    # Start trail
    trail_id = audit.start_trail("admin", "terraform_apply")
    print(f"Started trail: {trail_id}")
    
    # Log permission check
    audit.log_permission_check(
        actor="admin",
        operation_type="apply",
        permissions_required=["read_only", "plan_only", "apply"],
        permissions_granted=["read_only", "plan_only", "apply"],
        permissions_denied=[]
    )
    
    # Log decision
    audit.log_decision(
        actor="admin",
        decision_type="instance_selection",
        selected_option={
            'name': 'vastai-a5000',
            'type': 'gpu_instance',
            'id': 'i-1234567890abcdef0'
        },
        alternatives=[
            {'name': 'runpod-a4000', 'type': 'gpu_instance'},
            {'name': 'aws-g4dn', 'type': 'gpu_instance'}
        ],
        reasoning="Selected vastai-a5000 due to best cost-performance ratio",
        factors=[
            {'name': 'cost', 'value': 0.8, 'weight': 0.3, 'reason': 'Best cost per hour'},
            {'name': 'performance', 'value': 0.9, 'weight': 0.25, 'reason': 'High GPU memory'},
            {'name': 'availability', 'value': 0.95, 'weight': 0.2, 'reason': 'Good availability'}
        ],
        permissions_required=["read_only", "plan_only", "apply"],
        risk_assessment={'total_risk_score': 0.2, 'high_risk_items': []},
        operation_id="op_1234567890"
    )
    
    # Log operation
    audit.log_operation(
        actor="admin",
        operation_type="apply",
        operation_id="op_1234567890",
        status="success",
        resources_affected=1,
        cost_impact=0.35,
        duration_seconds=120.5,
        permissions_required=["read_only", "plan_only", "apply"]
    )
    
    # Log state change
    audit.log_state_change(
        actor="admin",
        operation_id="op_1234567890",
        state_before={'resources': 5},
        state_after={'resources': 6},
        resources_changed=['aws_instance.vastai_a5000']
    )
    
    # Log pin operation
    audit.log_pin_operation(
        actor="admin",
        operation_id="op_1234567890",
        reason="Critical production deployment",
        permissions_required=["modify_state"]
    )
    
    # Finish trail
    summary = audit.finish_trail()
    
    print(f"\n📊 Trail Summary:")
    print(f"  Trail ID: {summary['trail_id']}")
    print(f"  Status: {summary['compliance_status']}")
    print(f"  Risk Score: {summary['risk_score']:.3f}")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Successful: {summary['successful_operations']}")
    print(f"  Failed: {summary['failed_operations']}")
    print(f"  Rollbacks: {summary['rollback_operations']}")
    print(f"  Total Entries: {summary['total_entries']}")
    
    # Export trail
    audit.export_trail(trail_id, f"audit_trail_{trail_id}.json")
    
    print("\n✅ Audit Trail Features:")
    print("  • Complete decision transparency")
    print("  • Deterministic 'why this instance?' reasoning")
    print("  • Permission tracking")
    print("  • State change auditing")
    print("  • Rollback tracking")
    print("  • Risk assessment")
    print("  • Compliance monitoring")
    print("  • Complete audit trail")
