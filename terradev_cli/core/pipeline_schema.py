#!/usr/bin/env python3
"""
Terradev Pipeline Schema - Argo-compatible with Terradev extensions
Supports import/export/validation of YAML pipelines
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

class WorkflowType(Enum):
    """Supported workflow types"""
    TRAINING = "training"
    INFERENCE = "inference" 
    MIGRATION = "migration"
    EVALUATION = "evaluation"
    CUSTOM = "custom"

class Provider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    RUNPOD = "runpod"
    VASTAI = "vastai"
    CRUSOE = "crusoe"
    LAMBDA_LABS = "lambda-labs"
    COREWEAVE = "coreweave"
    TENSORDOCK = "tensordock"
    HETZNER = "hetzner"
    FLUIDSTACK = "fluidstack"
    ALIBABA = "alibaba"
    OVHCLOUD = "ovhcloud"
    SILICONFLOW = "siliconflow"

class GPUType(Enum):
    """Supported GPU types"""
    A100 = "A100"
    H100 = "H100"
    V100 = "V100"
    RTX4090 = "RTX4090"
    RTX3090 = "RTX3090"
    A6000 = "A6000"
    A5000 = "A5000"
    L40S = "L40S"
    L40 = "L40"
    T4 = "T4"

@dataclass
class TerradevAnnotations:
    """Terradev-specific annotations for Argo workflows"""
    provider: Optional[Provider] = None
    gpu_type: Optional[GPUType] = None
    gpu_count: Optional[int] = None
    tier: Optional[str] = None  # research, research+, enterprise
    cost_optimization: Optional[str] = None  # conservative, moderate, aggressive
    migration_enabled: Optional[bool] = None
    monitoring: Optional[str] = None  # prometheus, wandb, none
    dry_run: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to annotation dictionary format"""
        annotations = {}
        if self.provider:
            annotations["terradev.io/provider"] = self.provider.value
        if self.gpu_type:
            annotations["terradev.io/gpu-type"] = self.gpu_type.value
        if self.gpu_count:
            annotations["terradev.io/gpu-count"] = str(self.gpu_count)
        if self.tier:
            annotations["terradev.io/tier"] = self.tier
        if self.cost_optimization:
            annotations["terradev.io/cost-optimization"] = self.cost_optimization
        if self.migration_enabled is not None:
            annotations["terradev.io/migration-enabled"] = str(self.migration_enabled).lower()
        if self.monitoring:
            annotations["terradev.io/monitoring"] = self.monitoring
        if self.dry_run is not None:
            annotations["terradev.io/dry-run"] = str(self.dry_run).lower()
        return annotations
    
    @classmethod
    def from_dict(cls, annotations: Dict[str, str]) -> 'TerradevAnnotations':
        """Create from annotation dictionary"""
        terradev = cls()
        
        if "terradev.io/provider" in annotations:
            try:
                terradev.provider = Provider(annotations["terradev.io/provider"])
            except ValueError:
                pass
                
        if "terradev.io/gpu-type" in annotations:
            try:
                terradev.gpu_type = GPUType(annotations["terradev.io/gpu-type"])
            except ValueError:
                pass
                
        if "terradev.io/gpu-count" in annotations:
            try:
                terradev.gpu_count = int(annotations["terradev.io/gpu-count"])
            except ValueError:
                pass
                
        if "terradev.io/tier" in annotations:
            terradev.tier = annotations["terradev.io/tier"]
            
        if "terradev.io/cost-optimization" in annotations:
            terradev.cost_optimization = annotations["terradev.io/cost-optimization"]
            
        if "terradev.io/migration-enabled" in annotations:
            terradev.migration_enabled = annotations["terradev.io/migration-enabled"].lower() == "true"
            
        if "terradev.io/monitoring" in annotations:
            terradev.monitoring = annotations["terradev.io/monitoring"]
            
        if "terradev.io/dry-run" in annotations:
            terradev.dry_run = annotations["terradev.io/dry-run"].lower() == "true"
            
        return terradev

@dataclass
class WorkflowMetadata:
    """Workflow metadata"""
    name: str
    namespace: Optional[str] = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    generate_name: Optional[str] = None
    
    @property
    def terradev_annotations(self) -> TerradevAnnotations:
        """Extract Terradev-specific annotations"""
        return TerradevAnnotations.from_dict(self.annotations)

@dataclass
class Container:
    """Container definition"""
    image: str
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    env: List[Dict[str, str]] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    volume_mounts: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class Template:
    """Workflow template definition"""
    name: str
    container: Optional[Container] = None
    steps: Optional[List[List[Dict[str, Any]]]] = None
    dag: Optional[Dict[str, Any]] = None
    script: Optional[Dict[str, Any]] = None
    resource: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    metadata: Optional[WorkflowMetadata] = None
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class WorkflowSpec:
    """Workflow specification"""
    entrypoint: str
    templates: List[Template]
    arguments: Optional[Dict[str, Any]] = None
    service_account_name: Optional[str] = None
    security_context: Optional[Dict[str, Any]] = None
    pod_spec_patch: Optional[str] = None
    parallelism: Optional[int] = None
    node_selector: Dict[str, str] = field(default_factory=dict)
    affinity: Optional[Dict[str, Any]] = None
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    image_pull_secrets: List[Dict[str, str]] = field(default_factory=list)
    active_deadline_seconds: Optional[int] = None
    ttl_strategy: Optional[Dict[str, Any]] = None
    artifact_repository: Optional[Dict[str, Any]] = None

@dataclass
class Workflow:
    """Complete Argo-compatible workflow with Terradev extensions"""
    api_version: str = "argoproj.io/v1alpha1"
    kind: str = "Workflow"
    metadata: Optional[WorkflowMetadata] = None
    spec: Optional[WorkflowSpec] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        result = {
            "apiVersion": self.api_version,
            "kind": self.kind
        }
        
        if self.metadata:
            metadata_dict = {
                "name": self.metadata.name
            }
            if self.metadata.namespace:
                metadata_dict["namespace"] = self.metadata.namespace
            if self.metadata.labels:
                metadata_dict["labels"] = self.metadata.labels
            if self.metadata.annotations:
                metadata_dict["annotations"] = self.metadata.annotations
            if self.metadata.generate_name:
                metadata_dict["generateName"] = self.metadata.generate_name
            result["metadata"] = metadata_dict
            
        if self.spec:
            spec_dict = {
                "entrypoint": self.spec.entrypoint,
                "templates": []
            }
            
            # Convert templates
            for template in self.spec.templates:
                template_dict = {"name": template.name}
                
                if template.container:
                    container_dict = {
                        "image": template.container.image
                    }
                    if template.container.command:
                        container_dict["command"] = template.container.command
                    if template.container.args:
                        container_dict["args"] = template.container.args
                    if template.container.env:
                        container_dict["env"] = template.container.env
                    if template.container.resources:
                        container_dict["resources"] = template.container.resources
                    if template.container.volume_mounts:
                        container_dict["volumeMounts"] = template.container.volume_mounts
                    template_dict["container"] = container_dict
                    
                if template.steps:
                    template_dict["steps"] = template.steps
                if template.dag:
                    template_dict["dag"] = template.dag
                if template.script:
                    template_dict["script"] = template.script
                if template.resource:
                    template_dict["resource"] = template.resource
                if template.inputs:
                    template_dict["inputs"] = template.inputs
                if template.outputs:
                    template_dict["outputs"] = template.outputs
                if template.metadata:
                    template_dict["metadata"] = {
                        "name": template.metadata.name
                    }
                    if template.metadata.labels:
                        template_dict["metadata"]["labels"] = template.metadata.labels
                    if template.metadata.annotations:
                        template_dict["metadata"]["annotations"] = template.metadata.annotations
                if template.annotations:
                    template_dict["annotations"] = template.annotations
                    
                spec_dict["templates"].append(template_dict)
            
            # Add other spec fields
            if self.spec.arguments:
                spec_dict["arguments"] = self.spec.arguments
            if self.spec.service_account_name:
                spec_dict["serviceAccountName"] = self.spec.service_account_name
            if self.spec.security_context:
                spec_dict["securityContext"] = self.spec.security_context
            if self.spec.pod_spec_patch:
                spec_dict["podSpecPatch"] = self.spec.pod_spec_patch
            if self.spec.parallelism:
                spec_dict["parallelism"] = self.spec.parallelism
            if self.spec.node_selector:
                spec_dict["nodeSelector"] = self.spec.node_selector
            if self.spec.affinity:
                spec_dict["affinity"] = self.spec.affinity
            if self.spec.tolerations:
                spec_dict["tolerations"] = self.spec.tolerations
            if self.spec.image_pull_secrets:
                spec_dict["imagePullSecrets"] = self.spec.image_pull_secrets
            if self.spec.active_deadline_seconds:
                spec_dict["activeDeadlineSeconds"] = self.spec.active_deadline_seconds
            if self.spec.ttl_strategy:
                spec_dict["ttlStrategy"] = self.spec.ttl_strategy
            if self.spec.artifact_repository:
                spec_dict["artifactRepository"] = self.spec.artifact_repository
                
            result["spec"] = spec_dict
            
        return result
    
    def to_yaml(self) -> str:
        """Convert to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create from dictionary"""
        workflow = cls()
        
        if "apiVersion" in data:
            workflow.api_version = data["apiVersion"]
        if "kind" in data:
            workflow.kind = data["kind"]
            
        if "metadata" in data:
            meta_data = data["metadata"]
            workflow.metadata = WorkflowMetadata(
                name=meta_data.get("name", ""),
                namespace=meta_data.get("namespace"),
                labels=meta_data.get("labels", {}),
                annotations=meta_data.get("annotations", {}),
                generate_name=meta_data.get("generateName")
            )
            
        if "spec" in data:
            spec_data = data["spec"]
            workflow.spec = WorkflowSpec(
                entrypoint=spec_data.get("entrypoint", ""),
                templates=[],
                arguments=spec_data.get("arguments"),
                service_account_name=spec_data.get("serviceAccountName"),
                security_context=spec_data.get("securityContext"),
                pod_spec_patch=spec_data.get("podSpecPatch"),
                parallelism=spec_data.get("parallelism"),
                node_selector=spec_data.get("nodeSelector", {}),
                affinity=spec_data.get("affinity"),
                tolerations=spec_data.get("tolerations", []),
                image_pull_secrets=spec_data.get("imagePullSecrets", []),
                active_deadline_seconds=spec_data.get("activeDeadlineSeconds"),
                ttl_strategy=spec_data.get("ttlStrategy"),
                artifact_repository=spec_data.get("artifactRepository")
            )
            
            # Parse templates
            for template_data in spec_data.get("templates", []):
                template = Template(name=template_data.get("name", ""))
                
                if "container" in template_data:
                    container_data = template_data["container"]
                    template.container = Container(
                        image=container_data.get("image", ""),
                        command=container_data.get("command", []),
                        args=container_data.get("args", []),
                        env=container_data.get("env", []),
                        resources=container_data.get("resources", {}),
                        volume_mounts=container_data.get("volumeMounts", [])
                    )
                    
                if "steps" in template_data:
                    template.steps = template_data["steps"]
                if "dag" in template_data:
                    template.dag = template_data["dag"]
                if "script" in template_data:
                    template.script = template_data["script"]
                if "resource" in template_data:
                    template.resource = template_data["resource"]
                if "inputs" in template_data:
                    template.inputs = template_data["inputs"]
                if "outputs" in template_data:
                    template.outputs = template_data["outputs"]
                if "metadata" in template_data:
                    meta_data = template_data["metadata"]
                    template.metadata = WorkflowMetadata(
                        name=meta_data.get("name", ""),
                        namespace=meta_data.get("namespace"),
                        labels=meta_data.get("labels", {}),
                        annotations=meta_data.get("annotations", {}),
                        generate_name=meta_data.get("generateName")
                    )
                if "annotations" in template_data:
                    template.annotations = template_data["annotations"]
                    
                workflow.spec.templates.append(template)
                
        return workflow
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Workflow':
        """Create from YAML string"""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Workflow':
        """Create from YAML file"""
        with open(file_path, 'r') as f:
            return cls.from_yaml(f.read())

class PipelineValidator:
    """Validate Terradev pipelines"""
    
    @staticmethod
    def validate_workflow(workflow: Workflow) -> List[str]:
        """Validate workflow and return list of errors"""
        errors = []
        
        # Check required fields
        if not workflow.metadata or not workflow.metadata.name:
            errors.append("Workflow metadata.name is required")
            
        if not workflow.spec or not workflow.spec.entrypoint:
            errors.append("Workflow spec.entrypoint is required")
            
        if not workflow.spec or not workflow.spec.templates:
            errors.append("Workflow spec.templates is required")
            
        # Check that entrypoint template exists
        if workflow.spec and workflow.spec.entrypoint and workflow.spec.templates:
            template_names = [t.name for t in workflow.spec.templates]
            if workflow.spec.entrypoint not in template_names:
                errors.append(f"Entrypoint template '{workflow.spec.entrypoint}' not found in templates")
                
        # Validate Terradev annotations
        if workflow.metadata:
            terradev_ann = workflow.metadata.terradev_annotations
            
            if terradev_ann.gpu_count and (terradev_ann.gpu_count < 1 or terradev_ann.gpu_count > 32):
                errors.append("GPU count must be between 1 and 32")
                
            if terradev_ann.tier and terradev_ann.tier not in ["research", "research+", "enterprise"]:
                errors.append("Tier must be one of: research, research+, enterprise")
                
            if terradev_ann.cost_optimization and terradev_ann.cost_optimization not in ["conservative", "moderate", "aggressive"]:
                errors.append("Cost optimization must be one of: conservative, moderate, aggressive")
                
            if terradev_ann.monitoring and terradev_ann.monitoring not in ["prometheus", "wandb", "none"]:
                errors.append("Monitoring must be one of: prometheus, wandb, none")
                
        return errors
    
    @staticmethod
    def validate_yaml_file(file_path: str) -> tuple[bool, List[str]]:
        """Validate YAML file and return (is_valid, errors)"""
        try:
            workflow = Workflow.from_file(file_path)
            errors = PipelineValidator.validate_workflow(workflow)
            return len(errors) == 0, errors
        except Exception as e:
            return False, [f"Failed to parse YAML: {str(e)}"]
