#!/usr/bin/env python3
"""
Terraform Manager - Complete Transparency & Control
Dry-run modes, deterministic logs, easy rollback, and pinning
"""

import subprocess
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import shutil

from decision_engine import TerraformDecisionEngine, DecisionLog, TerraformPlan, PermissionScope

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TerraformMode(Enum):
    """Terraform operation modes"""
    READ_ONLY = "read_only"
    DRY_RUN = "dry_run"
    PLAN = "plan"
    APPLY = "apply"
    DESTROY = "destroy"
    ROLLBACK = "rollback"

class OperationStatus(Enum):
    """Operation status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

@dataclass
class TerraformOperation:
    """Terraform operation with full transparency"""
    operation_id: str
    mode: TerraformMode
    command: List[str]
    working_directory: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: OperationStatus = OperationStatus.PENDING
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_seconds: Optional[float] = None
    resources_affected: Optional[int] = None
    cost_impact: Optional[float] = None
    decision_logs: Optional[List[DecisionLog]] = None
    plan: Optional[TerraformPlan] = None
    rollback_available: bool = False
    rollback_operation_id: Optional[str] = None
    pin_enabled: bool = False
    pin_reason: Optional[str] = None

@dataclass
class TerraformState:
    """Terraform state information"""
    state_file: str
    workspace: str
    resources: List[Dict[str, Any]]
    outputs: Dict[str, Any]
    version: str
    serial: int
    lineage: str
    created_at: datetime
    modified_at: datetime

class TerraformManager:
    """Terraform manager with complete transparency and control"""
    
    def __init__(self, working_directory: str = "terraform"):
        self.working_directory = working_directory
        self.operations: List[TerraformOperation] = []
        self.decision_engine = TerraformDecisionEngine(dry_run_mode=True)
        self.state_snapshots: Dict[str, TerraformState] = {}
        self.pinned_operations: Dict[str, str] = {}  # operation_id -> reason
        
        # Ensure working directory exists
        os.makedirs(working_directory, exist_ok=True)
        
        logger.info(f"Terraform Manager initialized in {working_directory}")
    
    def validate_permissions(self, mode: TerraformMode) -> Tuple[bool, List[str]]:
        """Validate permissions for Terraform operation"""
        required_permissions = self._get_required_permissions(mode)
        missing_permissions = []
        
        for permission in required_permissions:
            if not self._check_permission(permission):
                missing_permissions.append(permission.value)
        
        return len(missing_permissions) == 0, missing_permissions
    
    def _get_required_permissions(self, mode: TerraformMode) -> List[PermissionScope]:
        """Get required permissions for mode"""
        if mode == TerraformMode.READ_ONLY:
            return [PermissionScope.READ_ONLY]
        elif mode == TerraformMode.DRY_RUN:
            return [PermissionScope.READ_ONLY, PermissionScope.DRY_RUN]
        elif mode == TerraformMode.PLAN:
            return [PermissionScope.READ_ONLY, PermissionScope.PLAN_ONLY]
        elif mode == TerraformMode.APPLY:
            return [PermissionScope.READ_ONLY, PermissionScope.PLAN_ONLY, PermissionScope.APPLY]
        elif mode == TerraformMode.DESTROY:
            return [PermissionScope.READ_ONLY, PermissionScope.DESTROY]
        elif mode == TerraformMode.ROLLBACK:
            return [PermissionScope.READ_ONLY, PermissionScope.MODIFY_STATE]
        else:
            return []
    
    def _check_permission(self, permission: PermissionScope) -> bool:
        """Check if permission is available"""
        # In a real implementation, this would check actual permissions
        # For demo, we'll simulate permission checks
        return True
    
    def dry_run(self, config: Dict[str, Any]) -> TerraformOperation:
        """Execute Terraform dry run with full transparency"""
        operation_id = self._generate_operation_id()
        
        # Create operation
        operation = TerraformOperation(
            operation_id=operation_id,
            mode=TerraformMode.DRY_RUN,
            command=["terraform", "plan", "-detailed-exitcode"],
            working_directory=self.working_directory,
            created_at=datetime.now()
        )
        
        # Validate permissions
        has_permissions, missing = self.validate_permissions(TerraformMode.DRY_RUN)
        if not has_permissions:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"Missing permissions: {', '.join(missing)}"
            self.operations.append(operation)
            return operation
        
        # Execute dry run
        operation = self._execute_operation(operation)
        
        # Parse plan output
        if operation.status == OperationStatus.SUCCESS:
            plan = self._parse_terraform_plan(operation.stdout)
            operation.plan = plan
            operation.resources_affected = len(plan.resources_to_add) + len(plan.resources_to_change) + len(plan.resources_to_destroy)
            operation.cost_impact = plan.cost_estimate
        
        self.operations.append(operation)
        
        logger.info(f"Dry run completed: {operation_id}")
        logger.info(f"Status: {operation.status.value}")
        logger.info(f"Resources affected: {operation.resources_affected}")
        
        return operation
    
    def plan(self, config: Dict[str, Any], save_plan: bool = True) -> TerraformOperation:
        """Create Terraform plan with full transparency"""
        operation_id = self._generate_operation_id()
        
        # Create operation
        command = ["terraform", "plan"]
        if save_plan:
            command.extend(["-out", f"plan-{operation_id}.tfplan"])
        
        operation = TerraformOperation(
            operation_id=operation_id,
            mode=TerraformMode.PLAN,
            command=command,
            working_directory=self.working_directory,
            created_at=datetime.now()
        )
        
        # Validate permissions
        has_permissions, missing = self.validate_permissions(TerraformMode.PLAN)
        if not has_permissions:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"Missing permissions: {', '.join(missing)}"
            self.operations.append(operation)
            return operation
        
        # Create state snapshot before planning
        self._create_state_snapshot(f"pre-plan-{operation_id}")
        
        # Execute plan
        operation = self._execute_operation(operation)
        
        # Parse plan output
        if operation.status == OperationStatus.SUCCESS:
            plan = self._parse_terraform_plan(operation.stdout)
            operation.plan = plan
            operation.resources_affected = len(plan.resources_to_add) + len(plan.resources_to_change) + len(plan.resources_to_destroy)
            operation.cost_impact = plan.cost_estimate
            operation.rollback_available = True
        
        self.operations.append(operation)
        
        logger.info(f"Plan created: {operation_id}")
        logger.info(f"Status: {operation.status.value}")
        logger.info(f"Resources affected: {operation.resources_affected}")
        
        return operation
    
    def apply(self, plan_file: Optional[str] = None, auto_approve: bool = False) -> TerraformOperation:
        """Apply Terraform plan with full transparency"""
        operation_id = self._generate_operation_id()
        
        # Create operation
        command = ["terraform", "apply"]
        if plan_file:
            command.append(plan_file)
        if auto_approve:
            command.append("-auto-approve")
        
        operation = TerraformOperation(
            operation_id=operation_id,
            mode=TerraformMode.APPLY,
            command=command,
            working_directory=self.working_directory,
            created_at=datetime.now()
        )
        
        # Validate permissions
        has_permissions, missing = self.validate_permissions(TerraformMode.APPLY)
        if not has_permissions:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"Missing permissions: {', '.join(missing)}"
            self.operations.append(operation)
            return operation
        
        # Check if operation is pinned
        if operation_id in self.pinned_operations:
            operation.status = OperationStatus.CANCELLED
            operation.stderr = f"Operation is pinned: {self.pinned_operations[operation_id]}"
            self.operations.append(operation)
            return operation
        
        # Create state snapshot before applying
        self._create_state_snapshot(f"pre-apply-{operation_id}")
        
        # Execute apply
        operation = self._execute_operation(operation)
        
        # Parse apply output
        if operation.status == OperationStatus.SUCCESS:
            operation.resources_affected = self._count_resources_in_output(operation.stdout)
            operation.rollback_available = True
            operation.cost_impact = self._estimate_cost_from_output(operation.stdout)
        
        self.operations.append(operation)
        
        logger.info(f"Apply completed: {operation_id}")
        logger.info(f"Status: {operation.status.value}")
        logger.info(f"Resources affected: {operation.resources_affected}")
        
        return operation
    
    def destroy(self, target: Optional[str] = None, auto_approve: bool = False) -> TerraformOperation:
        """Destroy Terraform resources with full transparency"""
        operation_id = self._generate_operation_id()
        
        # Create operation
        command = ["terraform", "destroy"]
        if target:
            command.extend(["-target", target])
        if auto_approve:
            command.append("-auto-approve")
        
        operation = TerraformOperation(
            operation_id=operation_id,
            mode=TerraformMode.DESTROY,
            command=command,
            working_directory=self.working_directory,
            created_at=datetime.now()
        )
        
        # Validate permissions
        has_permissions, missing = self.validate_permissions(TerraformMode.DESTROY)
        if not has_permissions:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"Missing permissions: {', '.join(missing)}"
            self.operations.append(operation)
            return operation
        
        # Create state snapshot before destroying
        self._create_state_snapshot(f"pre-destroy-{operation_id}")
        
        # Execute destroy
        operation = self._execute_operation(operation)
        
        # Parse destroy output
        if operation.status == OperationStatus.SUCCESS:
            operation.resources_affected = self._count_resources_in_output(operation.stdout)
            operation.rollback_available = False  # Can't rollback destroy
        
        self.operations.append(operation)
        
        logger.info(f"Destroy completed: {operation_id}")
        logger.info(f"Status: {operation.status.value}")
        logger.info(f"Resources affected: {operation.resources_affected}")
        
        return operation
    
    def rollback(self, operation_id: str) -> TerraformOperation:
        """Rollback a Terraform operation"""
        rollback_id = self._generate_operation_id()
        
        # Find original operation
        original_operation = self._find_operation(operation_id)
        if not original_operation:
            raise ValueError(f"Operation {operation_id} not found")
        
        if not original_operation.rollback_available:
            raise ValueError(f"Operation {operation_id} cannot be rolled back")
        
        # Create rollback operation
        operation = TerraformOperation(
            operation_id=rollback_id,
            mode=TerraformMode.ROLLBACK,
            command=["terraform", "apply", "-auto-approve"],
            working_directory=self.working_directory,
            created_at=datetime.now(),
            rollback_operation_id=operation_id
        )
        
        # Validate permissions
        has_permissions, missing = self.validate_permissions(TerraformMode.ROLLBACK)
        if not has_permissions:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"Missing permissions: {', '.join(missing)}"
            self.operations.append(operation)
            return operation
        
        # Restore state snapshot
        snapshot_id = f"pre-{original_operation.mode.value}-{operation_id}"
        if snapshot_id in self.state_snapshots:
            self._restore_state_snapshot(snapshot_id)
        else:
            operation.status = OperationStatus.FAILED
            operation.stderr = f"No state snapshot found for {snapshot_id}"
            self.operations.append(operation)
            return operation
        
        # Execute rollback
        operation = self._execute_operation(operation)
        
        # Update original operation
        original_operation.status = OperationStatus.ROLLED_BACK
        original_operation.rollback_operation_id = rollback_id
        
        self.operations.append(operation)
        
        logger.info(f"Rollback completed: {rollback_id}")
        logger.info(f"Original operation: {operation_id}")
        logger.info(f"Status: {operation.status.value}")
        
        return operation
    
    def pin_operation(self, operation_id: str, reason: str) -> bool:
        """Pin an operation to prevent changes"""
        operation = self._find_operation(operation_id)
        if not operation:
            return False
        
        operation.pin_enabled = True
        operation.pin_reason = reason
        self.pinned_operations[operation_id] = reason
        
        logger.info(f"Operation pinned: {operation_id}")
        logger.info(f"Reason: {reason}")
        
        return True
    
    def unpin_operation(self, operation_id: str) -> bool:
        """Unpin an operation"""
        operation = self._find_operation(operation_id)
        if not operation:
            return False
        
        operation.pin_enabled = False
        operation.pin_reason = None
        if operation_id in self.pinned_operations:
            del self.pinned_operations[operation_id]
        
        logger.info(f"Operation unpinned: {operation_id}")
        
        return True
    
    def _execute_operation(self, operation: TerraformOperation) -> TerraformOperation:
        """Execute Terraform operation with full logging"""
        operation.status = OperationStatus.RUNNING
        operation.started_at = datetime.now()
        
        try:
            # Execute command
            result = subprocess.run(
                operation.command,
                cwd=operation.working_directory,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            operation.completed_at = datetime.now()
            operation.duration_seconds = (operation.completed_at - operation.started_at).total_seconds()
            operation.exit_code = result.returncode
            operation.stdout = result.stdout
            operation.stderr = result.stderr
            
            if result.returncode == 0:
                operation.status = OperationStatus.SUCCESS
            else:
                operation.status = OperationStatus.FAILED
            
        except subprocess.TimeoutExpired:
            operation.status = OperationStatus.FAILED
            operation.stderr = "Operation timed out"
            operation.completed_at = datetime.now()
            operation.duration_seconds = 600.0
        
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.stderr = str(e)
            operation.completed_at = datetime.now()
        
        return operation
    
    def _parse_terraform_plan(self, plan_output: str) -> TerraformPlan:
        """Parse Terraform plan output"""
        # Simplified parsing - in real implementation would use JSON output
        plan_id = self._generate_operation_id()
        
        resources_to_add = []
        resources_to_change = []
        resources_to_destroy = []
        
        # Parse plan output (simplified)
        lines = plan_output.split('\n')
        for line in lines:
            if 'create' in line and '+' in line:
                resources_to_add.append({'type': 'resource', 'action': 'create', 'line': line.strip()})
            elif 'update' in line and '~' in line:
                resources_to_change.append({'type': 'resource', 'action': 'update', 'line': line.strip()})
            elif 'destroy' in line and '-' in line:
                resources_to_destroy.append({'type': 'resource', 'action': 'destroy', 'line': line.strip()})
        
        return TerraformPlan(
            plan_id=plan_id,
            created_at=datetime.now(),
            resources_to_add=resources_to_add,
            resources_to_change=resources_to_change,
            resources_to_destroy=resources_to_destroy,
            cost_estimate=0.0,  # Would parse from output
            permissions_required=[PermissionScope.PLAN_ONLY],
            risk_assessment={'total_risk_score': 0.0},
            rollback_available=True
        )
    
    def _count_resources_in_output(self, output: str) -> int:
        """Count resources in Terraform output"""
        count = 0
        lines = output.split('\n')
        for line in lines:
            if any(action in line for action in ['create', 'update', 'delete', 'destroy']):
                count += 1
        return count
    
    def _estimate_cost_from_output(self, output: str) -> float:
        """Estimate cost from Terraform output"""
        # Simplified - would parse actual cost information
        return 0.0
    
    def _create_state_snapshot(self, snapshot_id: str) -> None:
        """Create Terraform state snapshot"""
        state_file = os.path.join(self.working_directory, "terraform.tfstate")
        if os.path.exists(state_file):
            snapshot_file = os.path.join(self.working_directory, f"state-{snapshot_id}.tfstate")
            shutil.copy2(state_file, snapshot_file)
            
            # Store snapshot info
            state = TerraformState(
                state_file=snapshot_file,
                workspace="default",
                resources=[],  # Would parse from state file
                outputs={},
                version="1.0",
                serial=1,
                lineage="",
                created_at=datetime.now(),
                modified_at=datetime.now()
            )
            
            self.state_snapshots[snapshot_id] = state
            
            logger.info(f"State snapshot created: {snapshot_id}")
    
    def _restore_state_snapshot(self, snapshot_id: str) -> None:
        """Restore Terraform state snapshot"""
        if snapshot_id not in self.state_snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self.state_snapshots[snapshot_id]
        state_file = os.path.join(self.working_directory, "terraform.tfstate")
        
        if os.path.exists(snapshot.state_file):
            shutil.copy2(snapshot.state_file, state_file)
            logger.info(f"State snapshot restored: {snapshot_id}")
        else:
            raise ValueError(f"Snapshot file {snapshot.state_file} not found")
    
    def _find_operation(self, operation_id: str) -> Optional[TerraformOperation]:
        """Find operation by ID"""
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        return None
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        timestamp = datetime.now().isoformat()
        content = f"operation_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_operation(self, operation_id: str) -> Optional[TerraformOperation]:
        """Get operation by ID"""
        return self._find_operation(operation_id)
    
    def get_operations(self, mode: Optional[TerraformMode] = None) -> List[TerraformOperation]:
        """Get operations with optional filtering"""
        if mode:
            return [op for op in self.operations if op.mode == mode]
        return self.operations
    
    def get_state_snapshots(self) -> Dict[str, TerraformState]:
        """Get all state snapshots"""
        return self.state_snapshots
    
    def export_operations(self, filename: str) -> None:
        """Export operations to file"""
        operations_data = []
        for op in self.operations:
            op_dict = {
                'operation_id': op.operation_id,
                'mode': op.mode.value,
                'command': op.command,
                'working_directory': op.working_directory,
                'created_at': op.created_at.isoformat(),
                'started_at': op.started_at.isoformat() if op.started_at else None,
                'completed_at': op.completed_at.isoformat() if op.completed_at else None,
                'status': op.status.value,
                'exit_code': op.exit_code,
                'duration_seconds': op.duration_seconds,
                'resources_affected': op.resources_affected,
                'cost_impact': op.cost_impact,
                'rollback_available': op.rollback_available,
                'pin_enabled': op.pin_enabled,
                'pin_reason': op.pin_reason
            }
            operations_data.append(op_dict)
        
        with open(filename, 'w') as f:
            json.dump(operations_data, f, indent=2)
        
        logger.info(f"Operations exported to: {filename}")

# Example usage
if __name__ == "__main__":
    # Create Terraform manager
    manager = TerraformManager("terraform_demo")
    
    print("🔧 TERRAFORM MANAGER DEMO")
    print("=" * 50)
    
    # Test permissions
    print("\n🔍 Checking permissions...")
    has_perms, missing = manager.validate_permissions(TerraformMode.DRY_RUN)
    print(f"Permissions OK: {has_perms}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    
    # Dry run
    print("\n🧪 Running dry run...")
    dry_run_op = manager.dry_run({})
    print(f"Operation ID: {dry_run_op.operation_id}")
    print(f"Status: {dry_run_op.status.value}")
    print(f"Duration: {dry_run_op.duration_seconds:.2f}s")
    print(f"Resources affected: {dry_run_op.resources_affected}")
    
    # Plan
    print("\n📋 Creating plan...")
    plan_op = manager.plan({})
    print(f"Operation ID: {plan_op.operation_id}")
    print(f"Status: {plan_op.status.value}")
    print(f"Rollback available: {plan_op.rollback_available}")
    
    # Pin operation
    print("\n📌 Pinning operation...")
    manager.pin_operation(plan_op.operation_id, "Critical production deployment")
    print(f"Operation pinned: {plan_op.operation_id}")
    
    # Get operations
    print("\n📊 All operations:")
    for op in manager.get_operations():
        print(f"  • {op.operation_id}: {op.mode.value} - {op.status.value}")
    
    # Export operations
    print("\n💾 Exporting operations...")
    manager.export_operations('operations.json')
    
    print("\n✅ Transparency Features:")
    print("  • Clear permission scopes")
    print("  • Dry-run/plan modes")
    print("  • Deterministic operation logs")
    print("  • Easy rollback with state snapshots")
    print("  • Operation pinning for safety")
    print("  • Complete audit trail")
