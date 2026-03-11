#!/usr/bin/env python3
"""
Auto Lineage System - Automatic artifact tracking on every pipeline execution
No manual tagging required - lineage is automatically captured
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime
from pathlib import Path
import hashlib
import logging

from .event_system import lineage_service, event_bus, EventType, ArtifactType, Environment

class LineageRecordType(Enum):
    """Types of lineage records"""
    EXECUTION = "execution"
    DEPLOYMENT = "deployment"
    PROMOTION = "promotion"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"

@dataclass
class LineageRecord:
    """Complete lineage record for a pipeline execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: LineageRecordType = LineageRecordType.EXECUTION
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_id: str = ""
    environment: Environment = Environment.DEV
    duration_seconds: float = 0.0
    status: str = "running"  # running, completed, failed
    
    # Input artifacts
    datasets: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    configs: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    
    # Output artifacts
    output_models: List[str] = field(default_factory=list)
    output_checkpoints: List[str] = field(default_factory=list)
    output_metrics: List[str] = field(default_factory=list)
    output_evaluations: List[str] = field(default_factory=list)
    
    # Execution context
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    git_commit: Optional[str] = None
    code_hash: Optional[str] = None
    
    # Resource usage
    gpu_hours: float = 0.0
    compute_cost: float = 0.0
    storage_gb: float = 0.0
    
    # Metadata
    triggered_by: str = ""
    trigger_event_id: Optional[str] = None
    parent_execution_id: Optional[str] = None
    child_execution_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_id": self.pipeline_id,
            "environment": self.environment.value,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "datasets": self.datasets,
            "models": self.models,
            "configs": self.configs,
            "checkpoints": self.checkpoints,
            "output_models": self.output_models,
            "output_checkpoints": self.output_checkpoints,
            "output_metrics": self.output_metrics,
            "output_evaluations": self.output_evaluations,
            "hyperparameters": self.hyperparameters,
            "environment_variables": self.environment_variables,
            "git_commit": self.git_commit,
            "code_hash": self.code_hash,
            "gpu_hours": self.gpu_hours,
            "compute_cost": self.compute_cost,
            "storage_gb": self.storage_gb,
            "triggered_by": self.triggered_by,
            "trigger_event_id": self.trigger_event_id,
            "parent_execution_id": self.parent_execution_id,
            "child_execution_ids": self.child_execution_ids
        }

class AutoLineageTracker:
    """Automatic lineage tracking for all pipeline executions"""
    
    def __init__(self):
        self.active_executions: Dict[str, LineageRecord] = {}
        self.completed_executions: List[LineageRecord] = []
        self.logger = logging.getLogger(__name__)
        
        # Subscribe to events for automatic tracking
        event_bus.subscribe(EventType.TRAINING_COMPLETED, self._on_training_completed)
        event_bus.subscribe(EventType.DEPLOYMENT_COMPLETED, self._on_deployment_completed)
        event_bus.subscribe(EventType.PROMOTION_REQUESTED, self._on_promotion_requested)
        event_bus.subscribe(EventType.CHECKPOINT_CREATED, self._on_checkpoint_created)
        event_bus.subscribe(EventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
    
    def start_execution(self, pipeline_id: str, environment: Environment = Environment.DEV,
                       triggered_by: str = "manual", trigger_event_id: Optional[str] = None,
                       parent_execution_id: Optional[str] = None) -> LineageRecord:
        """Start tracking a new pipeline execution"""
        record = LineageRecord(
            pipeline_id=pipeline_id,
            environment=environment,
            triggered_by=triggered_by,
            trigger_event_id=trigger_event_id,
            parent_execution_id=parent_execution_id
        )
        
        self.active_executions[record.id] = record
        self.logger.info(f"Started lineage tracking for execution: {record.id}")
        
        return record
    
    def add_input_artifact(self, execution_id: str, artifact_type: ArtifactType, 
                          artifact_id: str):
        """Add input artifact to execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            
            if artifact_type == ArtifactType.DATASET:
                execution.datasets.append(artifact_id)
            elif artifact_type == ArtifactType.MODEL:
                execution.models.append(artifact_id)
            elif artifact_type == ArtifactType.CONFIG:
                execution.configs.append(artifact_id)
            elif artifact_type == ArtifactType.CHECKPOINT:
                execution.checkpoints.append(artifact_id)
    
    def add_output_artifact(self, execution_id: str, artifact_type: ArtifactType,
                           artifact_id: str):
        """Add output artifact to execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            
            if artifact_type == ArtifactType.MODEL:
                execution.output_models.append(artifact_id)
            elif artifact_type == ArtifactType.CHECKPOINT:
                execution.output_checkpoints.append(artifact_id)
            elif artifact_type == ArtifactType.METRICS:
                execution.output_metrics.append(artifact_id)
            elif artifact_type == ArtifactType.EVALUATION:
                execution.output_evaluations.append(artifact_id)
    
    def set_hyperparameters(self, execution_id: str, hyperparams: Dict[str, Any]):
        """Set hyperparameters for execution"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].hyperparameters = hyperparams
    
    def set_environment_variables(self, execution_id: str, env_vars: Dict[str, str]):
        """Set environment variables for execution"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].environment_variables = env_vars
    
    def set_git_context(self, execution_id: str, commit: Optional[str] = None,
                       code_hash: Optional[str] = None):
        """Set git context for execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.git_commit = commit
            execution.code_hash = code_hash
    
    def set_resource_usage(self, execution_id: str, gpu_hours: float = 0.0,
                          compute_cost: float = 0.0, storage_gb: float = 0.0):
        """Set resource usage for execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.gpu_hours = gpu_hours
            execution.compute_cost = compute_cost
            execution.storage_gb = storage_gb
    
    def complete_execution(self, execution_id: str, status: str = "completed"):
        """Complete execution and move to history"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = status
            execution.duration_seconds = (datetime.now() - execution.timestamp).total_seconds()
            
            self.completed_executions.append(execution)
            del self.active_executions[execution_id]
            
            self.logger.info(f"Completed lineage tracking for execution: {execution_id} ({status})")
    
    def _on_training_completed(self, event):
        """Handle training completed event"""
        if event.data.get("execution_id"):
            self.complete_execution(event.data["execution_id"], "completed")
    
    def _on_deployment_completed(self, event):
        """Handle deployment completed event"""
        if event.data.get("execution_id"):
            self.complete_execution(event.data["execution_id"], "completed")
    
    def _on_promotion_requested(self, event):
        """Handle promotion requested event"""
        # Create lineage record for promotion
        record = LineageRecord(
            type=LineageRecordType.PROMOTION,
            pipeline_id=f"promotion-{event.data.get('artifact_id', 'unknown')}",
            environment=Environment(event.data.get("to_env", "staging")),
            triggered_by=event.data.get("requested_by", "system")
        )
        
        self.completed_executions.append(record)
    
    def _on_checkpoint_created(self, event):
        """Handle checkpoint created event"""
        if event.data.get("execution_id"):
            self.add_output_artifact(
                event.data["execution_id"],
                ArtifactType.CHECKPOINT,
                event.data.get("checkpoint_id", "")
            )
    
    def _on_evaluation_completed(self, event):
        """Handle evaluation completed event"""
        if event.data.get("execution_id"):
            self.add_output_artifact(
                event.data["execution_id"],
                ArtifactType.EVALUATION,
                event.data.get("evaluation_id", "")
            )
    
    def get_lineage_for_model(self, model_name: str, environment: Optional[Environment] = None) -> List[LineageRecord]:
        """Get all lineage records for a specific model"""
        records = []
        
        for record in self.completed_executions:
            # Check if this record involves the model
            model_artifacts = record.models + record.output_models
            
            for model_id in model_artifacts:
                if model_id in lineage_service.artifacts:
                    artifact = lineage_service.artifacts[model_id]
                    if artifact.name == model_name:
                        if environment is None or artifact.environment == environment:
                            records.append(record)
                            break
        
        return sorted(records, key=lambda x: x.timestamp, reverse=True)
    
    def diff_executions(self, execution_id_1: str, execution_id_2: str) -> Dict[str, Any]:
        """Compare two executions and return differences"""
        exec_1 = None
        exec_2 = None
        
        for record in self.completed_executions:
            if record.id == execution_id_1:
                exec_1 = record
            elif record.id == execution_id_2:
                exec_2 = record
        
        if not exec_1 or not exec_2:
            return {"error": "One or both executions not found"}
        
        diff = {
            "execution_1": {
                "id": exec_1.id,
                "timestamp": exec_1.timestamp.isoformat(),
                "environment": exec_1.environment.value
            },
            "execution_2": {
                "id": exec_2.id,
                "timestamp": exec_2.timestamp.isoformat(),
                "environment": exec_2.environment.value
            },
            "differences": {}
        }
        
        # Compare hyperparameters
        hp_diff = self._compare_dicts(exec_1.hyperparameters, exec_2.hyperparameters)
        if hp_diff:
            diff["differences"]["hyperparameters"] = hp_diff
        
        # Compare environment variables
        env_diff = self._compare_dicts(exec_1.environment_variables, exec_2.environment_variables)
        if env_diff:
            diff["differences"]["environment_variables"] = env_diff
        
        # Compare input artifacts
        input_diff = self._compare_lists(exec_1.datasets, exec_2.datasets, "datasets")
        input_diff.update(self._compare_lists(exec_1.models, exec_2.models, "models"))
        input_diff.update(self._compare_lists(exec_1.configs, exec_2.configs, "configs"))
        if input_diff:
            diff["differences"]["inputs"] = input_diff
        
        # Compare resource usage
        resource_diff = {}
        if abs(exec_1.gpu_hours - exec_2.gpu_hours) > 0.01:
            resource_diff["gpu_hours"] = {"exec1": exec_1.gpu_hours, "exec2": exec_2.gpu_hours}
        if abs(exec_1.compute_cost - exec_2.compute_cost) > 0.01:
            resource_diff["compute_cost"] = {"exec1": exec_1.compute_cost, "exec2": exec_2.compute_cost}
        if resource_diff:
            diff["differences"]["resources"] = resource_diff
        
        return diff
    
    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """Compare two dictionaries and return differences"""
        diff = {}
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 != val2:
                diff[key] = {"exec1": val1, "exec2": val2}
        
        return diff
    
    def _compare_lists(self, list1: List[str], list2: List[str], name: str) -> Dict[str, Any]:
        """Compare two lists and return differences"""
        set1 = set(list1)
        set2 = set(list2)
        
        diff = {}
        
        added = set2 - set1
        removed = set1 - set2
        
        if added:
            diff[f"{name}_added"] = list(added)
        if removed:
            diff[f"{name}_removed"] = list(removed)
        
        return diff
    
    def trace_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Trace complete lineage from a checkpoint backwards"""
        # Find the execution that created this checkpoint
        creating_execution = None
        for record in self.completed_executions:
            if checkpoint_id in record.output_checkpoints:
                creating_execution = record
                break
        
        if not creating_execution:
            return {"error": "Checkpoint not found in lineage records"}
        
        trace = {
            "checkpoint_id": checkpoint_id,
            "created_by": {
                "execution_id": creating_execution.id,
                "timestamp": creating_execution.timestamp.isoformat(),
                "pipeline_id": creating_execution.pipeline_id,
                "environment": creating_execution.environment.value
            },
            "inputs": {},
            "ancestors": []
        }
        
        # Get input artifacts
        for dataset_id in creating_execution.datasets:
            if dataset_id in lineage_service.artifacts:
                artifact = lineage_service.artifacts[dataset_id]
                trace["inputs"]["datasets"] = trace["inputs"].get("datasets", [])
                trace["inputs"]["datasets"].append({
                    "id": artifact.id,
                    "name": artifact.name,
                    "uri": artifact.uri,
                    "hash": artifact.hash
                })
        
        for model_id in creating_execution.models:
            if model_id in lineage_service.artifacts:
                artifact = lineage_service.artifacts[model_id]
                trace["inputs"]["models"] = trace["inputs"].get("models", [])
                trace["inputs"]["models"].append({
                    "id": artifact.id,
                    "name": artifact.name,
                    "uri": artifact.uri,
                    "hash": artifact.hash
                })
        
        # Trace back through parent executions
        current_execution = creating_execution
        visited = set()
        
        while current_execution and current_execution.id not in visited:
            visited.add(current_execution.id)
            
            if current_execution.parent_execution_id:
                parent_id = current_execution.parent_execution_id
                for record in self.completed_executions:
                    if record.id == parent_id:
                        trace["ancestors"].append({
                            "execution_id": record.id,
                            "timestamp": record.timestamp.isoformat(),
                            "pipeline_id": record.pipeline_id,
                            "environment": record.environment.value,
                            "status": record.status
                        })
                        current_execution = record
                        break
            else:
                break
        
        return trace
    
    def export_lineage(self, format: str = "json", model_name: Optional[str] = None,
                      environment: Optional[Environment] = None) -> str:
        """Export lineage data for compliance reports"""
        if model_name:
            records = self.get_lineage_for_model(model_name, environment)
        else:
            records = self.completed_executions
            if environment:
                records = [r for r in records if r.environment == environment]
        
        if format == "json":
            return json.dumps([record.to_dict() for record in records], indent=2)
        elif format == "csv":
            # Simple CSV export
            lines = []
            lines.append("id,timestamp,pipeline_id,environment,status,duration,gpu_hours,cost")
            
            for record in records:
                lines.append(f"{record.id},{record.timestamp.isoformat()},{record.pipeline_id},"
                           f"{record.environment.value},{record.status},{record.duration_seconds},"
                           f"{record.gpu_hours},{record.compute_cost}")
            
            return "\n".join(lines)
        else:
            return json.dumps([record.to_dict() for record in records], indent=2)

# Global instance
auto_lineage = AutoLineageTracker()
