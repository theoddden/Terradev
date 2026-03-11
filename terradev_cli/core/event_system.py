#!/usr/bin/env python3
"""
Terradev Event System - Triggers, Environment Promotion, and Artifact Lineage
Production-grade automation infrastructure for ML workflows
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import logging

class EventType(Enum):
    """Event types for triggers"""
    DATASET_LANDED = "dataset_landed"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    SCHEDULE_TRIGGERED = "schedule_triggered"
    CHECKPOINT_CREATED = "checkpoint_created"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    DEPLOYMENT_COMPLETED = "deployment_completed"
    PROMOTION_REQUESTED = "promotion_requested"
    ARTIFACT_REGISTERED = "artifact_registered"
    THRESHOLD_BREACHED = "threshold_breached"

class TriggerType(Enum):
    """Trigger types"""
    EVENT_BASED = "event_based"
    SCHEDULE = "schedule"
    CONDITION = "condition"
    MANUAL = "manual"

class Environment(Enum):
    """Deployment environments"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TESTING = "testing"

class ArtifactType(Enum):
    """Artifact types for lineage"""
    DATASET = "dataset"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    CONFIG = "config"
    DEPLOYMENT = "deployment"
    EVALUATION = "evaluation"

@dataclass
class Event:
    """Event representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.DATASET_LANDED
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata
        }

@dataclass
class Trigger:
    """Trigger definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: TriggerType = TriggerType.EVENT_BASED
    event_type: Optional[EventType] = None
    condition: Optional[str] = None
    schedule: Optional[str] = None
    target_pipeline: str = ""
    target_environment: Environment = Environment.DEV
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Artifact:
    """Artifact for lineage tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ArtifactType = ArtifactType.DATASET
    name: str = ""
    version: str = "v1"
    uri: str = ""
    hash: str = ""
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    environment: Environment = Environment.DEV
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)

@dataclass
class Promotion:
    """Environment promotion request"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_id: str = ""
    from_env: Environment = Environment.DEV
    to_env: Environment = Environment.STAGING
    status: str = "pending"  # pending, approved, rejected, completed
    requested_at: datetime = field(default_factory=datetime.now)
    requested_by: str = ""
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pipeline_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """Event bus for publishing and subscribing to events"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event: Event):
        """Publish event to all subscribers"""
        self.event_history.append(event)
        self.logger.info(f"Published event: {event.type.value} from {event.source}")
        
        if event.type in self.subscribers:
            for callback in self.subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
    
    def get_events(self, event_type: Optional[EventType] = None, 
                   since: Optional[datetime] = None, limit: int = 100) -> List[Event]:
        """Get events with optional filtering"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]

class TriggerManager:
    """Manage triggers and event-driven automation"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.triggers: Dict[str, Trigger] = {}
        self.logger = logging.getLogger(__name__)
        
        # Subscribe to all event types
        for event_type in EventType:
            self.event_bus.subscribe(event_type, self._handle_event)
    
    def create_trigger(self, name: str, trigger_type: TriggerType, 
                      target_pipeline: str, **kwargs) -> Trigger:
        """Create a new trigger"""
        trigger = Trigger(
            name=name,
            type=trigger_type,
            target_pipeline=target_pipeline,
            **kwargs
        )
        
        self.triggers[trigger.id] = trigger
        self.logger.info(f"Created trigger: {name} ({trigger_type.value})")
        
        return trigger
    
    def _handle_event(self, event: Event):
        """Handle incoming events and check triggers"""
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            should_trigger = False
            
            if trigger.type == TriggerType.EVENT_BASED and trigger.event_type == event.type:
                should_trigger = True
            elif trigger.type == TriggerType.CONDITION and trigger.condition:
                should_trigger = self._evaluate_condition(trigger.condition, event)
            
            if should_trigger:
                self._execute_trigger(trigger, event)
    
    def _evaluate_condition(self, condition: str, event: Event) -> bool:
        """Evaluate trigger condition (simplified)"""
        # In production, this would use a proper expression parser
        if "drift" in condition.lower() and event.type == EventType.MODEL_DRIFT_DETECTED:
            drift_score = event.data.get("drift_score", 0)
            threshold = 0.1  # Default threshold
            
            if ">" in condition:
                threshold = float(condition.split(">")[1].strip())
            
            return drift_score > threshold
        
        return False
    
    def _execute_trigger(self, trigger: Trigger, event: Event):
        """Execute trigger (launch pipeline)"""
        trigger.last_triggered = datetime.now()
        trigger.trigger_count += 1
        
        self.logger.info(f"Trigger '{trigger.name}' fired by event {event.type.value}")
        
        # In production, this would actually launch the pipeline
        # For now, we just log and could integrate with existing job system
        
        # Publish trigger execution event
        trigger_event = Event(
            type=EventType.SCHEDULE_TRIGGERED,
            source="trigger_manager",
            data={
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
                "target_pipeline": trigger.target_pipeline,
                "environment": trigger.target_environment.value,
                "original_event": event.to_dict()
            }
        )
        self.event_bus.publish(trigger_event)

class LineageService:
    """Track artifact lineage and relationships"""
    
    def __init__(self):
        self.artifacts: Dict[str, Artifact] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_artifact(self, artifact_type: ArtifactType, name: str, uri: str,
                         environment: Environment = Environment.DEV, **kwargs) -> Artifact:
        """Register a new artifact"""
        artifact = Artifact(
            type=artifact_type,
            name=name,
            uri=uri,
            environment=environment,
            **kwargs
        )
        
        self.artifacts[artifact.id] = artifact
        self.logger.info(f"Registered artifact: {artifact_type.value} {name} in {environment.value}")
        
        return artifact
    
    def add_relationship(self, parent_id: str, child_id: str):
        """Add parent-child relationship between artifacts"""
        if parent_id in self.artifacts and child_id in self.artifacts:
            self.artifacts[parent_id].child_ids.append(child_id)
            self.artifacts[child_id].parent_ids.append(parent_id)
    
    def get_lineage(self, artifact_id: str, direction: str = "both") -> Dict[str, List[Artifact]]:
        """Get lineage graph for artifact"""
        if artifact_id not in self.artifacts:
            return {}
        
        result = {"parents": [], "children": []}
        
        if direction in ["up", "both"]:
            # Get all parents recursively
            visited = set()
            queue = [artifact_id]
            
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                
                if current_id in self.artifacts:
                    for parent_id in self.artifacts[current_id].parent_ids:
                        if parent_id in self.artifacts and parent_id not in visited:
                            result["parents"].append(self.artifacts[parent_id])
                            queue.append(parent_id)
        
        if direction in ["down", "both"]:
            # Get all children recursively
            visited = set()
            queue = [artifact_id]
            
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                
                if current_id in self.artifacts:
                    for child_id in self.artifacts[current_id].child_ids:
                        if child_id in self.artifacts and child_id not in visited:
                            result["children"].append(self.artifacts[child_id])
                            queue.append(child_id)
        
        return result
    
    def get_production_artifacts(self, artifact_type: Optional[ArtifactType] = None) -> List[Artifact]:
        """Get all artifacts in production environment"""
        artifacts = [a for a in self.artifacts.values() if a.environment == Environment.PROD]
        
        if artifact_type:
            artifacts = [a for a in artifacts if a.type == artifact_type]
        
        return sorted(artifacts, key=lambda x: x.created_at, reverse=True)

class EnvironmentManager:
    """Manage environment promotion and lifecycle"""
    
    def __init__(self, lineage_service: LineageService, event_bus: EventBus):
        self.lineage_service = lineage_service
        self.event_bus = event_bus
        self.promotions: Dict[str, Promotion] = {}
        self.logger = logging.getLogger(__name__)
    
    def request_promotion(self, artifact_id: str, from_env: Environment, 
                         to_env: Environment, requested_by: str) -> Promotion:
        """Request environment promotion"""
        promotion = Promotion(
            artifact_id=artifact_id,
            from_env=from_env,
            to_env=to_env,
            requested_by=requested_by
        )
        
        self.promotions[promotion.id] = promotion
        self.logger.info(f"Promotion requested: {from_env.value} -> {to_env.value} for artifact {artifact_id}")
        
        # Publish promotion requested event
        event = Event(
            type=EventType.PROMOTION_REQUESTED,
            source="environment_manager",
            data={
                "promotion_id": promotion.id,
                "artifact_id": artifact_id,
                "from_env": from_env.value,
                "to_env": to_env.value,
                "requested_by": requested_by
            }
        )
        self.event_bus.publish(event)
        
        return promotion
    
    def approve_promotion(self, promotion_id: str, approved_by: str) -> bool:
        """Approve and execute promotion"""
        if promotion_id not in self.promotions:
            return False
        
        promotion = self.promotions[promotion_id]
        promotion.status = "approved"
        promotion.approved_by = approved_by
        promotion.approved_at = datetime.now()
        
        # Execute promotion (copy artifact to new environment)
        if promotion.artifact_id in self.lineage_service.artifacts:
            artifact = self.lineage_service.artifacts[promotion.artifact_id]
            
            # Create new artifact in target environment
            new_artifact = self.lineage_service.register_artifact(
                artifact_type=artifact.type,
                name=artifact.name,
                uri=artifact.uri,  # In production, this would copy to new location
                environment=promotion.to_env,
                hash=artifact.hash,
                size_bytes=artifact.size_bytes,
                created_by=approved_by,
                metadata={
                    **artifact.metadata,
                    "promoted_from": promotion.from_env.value,
                    "promotion_id": promotion_id
                }
            )
            
            # Link lineage
            self.lineage_service.add_relationship(artifact.id, new_artifact.id)
            
            promotion.status = "completed"
            promotion.completed_at = datetime.now()
            
            self.logger.info(f"Promotion completed: {promotion.from_env.value} -> {promotion.to_env.value}")
            
            return True
        
        return False
    
    def get_promotion_history(self, artifact_id: Optional[str] = None) -> List[Promotion]:
        """Get promotion history"""
        promotions = list(self.promotions.values())
        
        if artifact_id:
            promotions = [p for p in promotions if p.artifact_id == artifact_id]
        
        return sorted(promotions, key=lambda x: x.requested_at, reverse=True)

# Global instances
event_bus = EventBus()
trigger_manager = TriggerManager(event_bus)
lineage_service = LineageService()
environment_manager = EnvironmentManager(lineage_service, event_bus)
