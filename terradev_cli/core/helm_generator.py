#!/usr/bin/env python3
"""
Helm Chart Generator
Generate production-grade Helm charts from Terradev workloads for Kubernetes deployment.

Supports workload types: training, inference, cost-optimized, high-performance,
                         moe-inference, rag, vllm-optimized
Supports stack integrations: qdrant, phoenix, guardrails
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# GPU product labels used by the NVIDIA GPU Operator / device plugin.
GPU_NODE_LABELS: Dict[str, str] = {
    'A100': 'NVIDIA-A100-SXM4-80GB',
    'A100-40G': 'NVIDIA-A100-SXM4-40GB',
    'A100-80G': 'NVIDIA-A100-SXM4-80GB',
    'H100': 'NVIDIA-H100-80GB-HBM3',
    'H100_SXM': 'NVIDIA-H100-80GB-HBM3',
    'H200': 'NVIDIA-H200-141GB-HBM3e',
    'V100': 'Tesla-V100-SXM2-16GB',
    'L4': 'NVIDIA-L4',
    'L40S': 'NVIDIA-L40S',
    'RTX 3090': 'NVIDIA-GeForce-RTX-3090',
    'RTX 4090': 'NVIDIA-GeForce-RTX-4090',
    'T4': 'Tesla-T4',
}

# GPU VRAM in GB — both variants where applicable
GPU_MEMORY_GB: Dict[str, int] = {
    'A100': 80, 'A100-40G': 40, 'A100-80G': 80,
    'H100': 80, 'H100_SXM': 80, 'H200': 141,
    'V100': 16, 'L4': 24, 'L40S': 48,
    'RTX 3090': 24, 'RTX 4090': 24, 'T4': 16,
}


@dataclass
class HelmChartConfig:
    name: str
    version: str
    description: str
    app_version: str
    kube_version: str
    maintainers: List[Dict[str, str]]
    keywords: List[str]
    dependencies: List[Dict[str, Any]] = field(default_factory=list)


class HelmChartGenerator:
    """Generate production-grade Helm charts from Terradev workloads"""

    WORKLOAD_TYPES = [
        'training', 'inference', 'cost-optimized', 'high-performance',
        'moe-inference', 'rag', 'vllm-optimized',
    ]
    STACK_COMPONENTS = ['qdrant', 'phoenix', 'guardrails']

    def __init__(self):
        self.chart_templates = {
            'training': self._get_training_template(),
            'inference': self._get_inference_template(),
            'cost-optimized': self._get_cost_optimized_template(),
            'high-performance': self._get_high_performance_template(),
            'moe-inference': self._get_moe_inference_template(),
            'rag': self._get_rag_template(),
            'vllm-optimized': self._get_vllm_optimized_template(),
        }

    def generate_chart(self, workload_config: Dict[str, Any], output_dir: str) -> str:
        """Generate complete Helm chart from Terradev workload"""
        chart_name = workload_config.get('name', f"terradev-{workload_config['workload_type']}")
        chart_path = Path(output_dir) / chart_name

        chart_path.mkdir(parents=True, exist_ok=True)
        (chart_path / "templates").mkdir(exist_ok=True)
        (chart_path / "charts").mkdir(exist_ok=True)

        chart_config = self._generate_chart_config(workload_config, chart_name)
        self._write_chart_yaml(chart_path, chart_config)

        values = self._generate_values(workload_config)
        self._write_values_yaml(chart_path, values)

        templates = self._generate_templates(workload_config)
        self._write_templates(chart_path, templates)

        readme = self._generate_readme(workload_config, chart_name)
        self._write_readme(chart_path, readme)

        return str(chart_path)

    def _generate_chart_config(self, workload: Dict[str, Any], chart_name: str) -> HelmChartConfig:
        """Generate Chart.yaml configuration"""
        deps: List[Dict[str, Any]] = []
        for stack in workload.get('stack', []):
            if stack == 'qdrant':
                deps.append({'name': 'qdrant', 'version': '0.9.x',
                             'repository': 'https://qdrant.github.io/qdrant-helm',
                             'condition': 'qdrant.enabled'})
        return HelmChartConfig(
            name=chart_name, version="1.0.0",
            description=f"Terradev {workload['workload_type'].title()} workload for {workload.get('gpu_type', 'GPU')}",
            app_version="1.0.0", kube_version=">=1.24.0-0",
            maintainers=[{"name": "Terradev", "email": "support@terradev.dev", "url": "https://terradev.dev"}],
            keywords=["gpu", "machine-learning", "kubernetes", workload['workload_type'], workload.get('gpu_type', 'gpu')],
            dependencies=deps,
        )

    def _generate_values(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate values.yaml"""
        wtype = workload['workload_type']
        base_values = self.chart_templates.get(wtype, self.chart_templates['inference'])

        values = {
            **base_values,
            'image': {'repository': workload['image'], 'tag': workload.get('tag', 'latest'), 'pullPolicy': 'IfNotPresent'},
            'gpu': {
                'type': workload['gpu_type'], 'count': workload.get('gpu_count', 1),
                'memory': workload.get('memory_gb', 16), 'storage': workload.get('storage_gb', 100),
                'nodeLabel': GPU_NODE_LABELS.get(workload['gpu_type'], workload['gpu_type']),
            },
            'resources': self._calculate_resources(workload),
            'budget': {'maxHourlyRate': workload.get('budget'), 'enforce': workload.get('budget') is not None},
            'terradev': {'provider': workload.get('provider', 'auto'), 'region': workload.get('region', 'us-east-1'), 'spot': workload.get('spot', True)},
            'securityContext': {'runAsNonRoot': True, 'runAsUser': 1000, 'runAsGroup': 1000, 'fsGroup': 1000, 'capabilities': {'drop': ['ALL']}},
            'podSecurityContext': {'fsGroup': 1000, 'seccompProfile': {'type': 'RuntimeDefault'}},
            'serviceAccount': {'create': True, 'name': '', 'annotations': {}},
            'metrics': {'enabled': True, 'serviceMonitor': {'enabled': True, 'interval': '15s', 'path': '/metrics'}},
        }

        # Health probes, autoscaling, PDB for non-job workloads
        if wtype not in ('training', 'cost-optimized'):
            values['probes'] = {
                'startup': {'enabled': True, 'path': '/health', 'initialDelaySeconds': 30, 'periodSeconds': 10, 'failureThreshold': 30},
                'liveness': {'enabled': True, 'path': '/health', 'periodSeconds': 15, 'failureThreshold': 3},
                'readiness': {'enabled': True, 'path': '/health', 'periodSeconds': 5, 'failureThreshold': 2},
            }
            values['autoscaling'] = {'enabled': False, 'minReplicas': 1, 'maxReplicas': 4, 'targetGPUUtilization': 80, 'targetCPUUtilizationPercentage': 80}
            values['podDisruptionBudget'] = {'enabled': True, 'minAvailable': 1}

        if workload.get('environment_vars'):
            values['env'] = workload['environment_vars']

        # Ports for all non-job workloads — default 8000 if none specified
        ports = workload.get('ports', [])
        if wtype not in ('training', 'cost-optimized'):
            if not ports:
                ports = [8000]
            values['service'] = {'type': 'LoadBalancer', 'ports': [{'port': p, 'targetPort': p} for p in ports], 'annotations': {}}

        # Stack integrations (qdrant, phoenix, guardrails)
        for stack in workload.get('stack', []):
            stack_values = self._get_stack_values(stack, workload)
            if stack_values:
                values.update(stack_values)

        return values

    def _calculate_resources(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource requirements"""
        gpu_count = workload.get('gpu_count', 1)
        memory_gb = workload.get('memory_gb', 16)

        # Estimate CPU based on GPU count (rule of thumb: 4-8 CPU cores per GPU)
        cpu_cores = gpu_count * 6

        # GPU VRAM + system memory overhead
        gpu_memory = GPU_MEMORY_GB.get(workload['gpu_type'], 16) * gpu_count
        total_memory = max(memory_gb, gpu_memory + 8)  # Add 8GB for system

        return {
            'requests': {
                'cpu': str(cpu_cores),
                'memory': f"{total_memory}Gi",
                'nvidia.com/gpu': str(gpu_count),
            },
            'limits': {
                'cpu': str(cpu_cores * 2),
                'memory': f"{total_memory * 2}Gi",
                'nvidia.com/gpu': str(gpu_count),
            },
        }

    # ── Stack integrations ─────────────────────────────────────────────

    def _get_stack_values(self, stack: str, workload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get Helm values for a stack component by importing its service."""
        if stack == 'qdrant':
            try:
                from ml_services.qdrant_service import QdrantService, QdrantConfig
                return QdrantService(QdrantConfig()).generate_helm_values()
            except ImportError:
                return {'qdrant': {'enabled': True, 'image': 'qdrant/qdrant:latest', 'replicas': 1,
                                   'ports': {'rest': 6333, 'grpc': 6334}, 'persistence': {'enabled': True, 'size': '100Gi'},
                                   'resources': {'requests': {'cpu': '500m', 'memory': '2Gi'}, 'limits': {'cpu': '4', 'memory': '8Gi'}}}}
        elif stack == 'phoenix':
            try:
                from ml_services.phoenix_service import PhoenixService, PhoenixConfig
                return PhoenixService(PhoenixConfig()).generate_helm_values()
            except ImportError:
                return {'phoenix': {'enabled': True, 'image': 'arizephoenix/phoenix:latest', 'port': 6006,
                                    'persistence': {'enabled': True, 'size': '50Gi'},
                                    'resources': {'requests': {'cpu': '500m', 'memory': '1Gi'}, 'limits': {'cpu': '2', 'memory': '4Gi'}}}}
        elif stack == 'guardrails':
            try:
                from ml_services.guardrails_service import GuardrailsService, GuardrailsConfig
                return GuardrailsService(GuardrailsConfig()).generate_helm_values()
            except ImportError:
                return {'guardrails': {'enabled': True, 'image': 'nvcr.io/nvidia/nemo-guardrails:latest', 'port': 8090,
                                       'deploymentMode': 'standalone',
                                       'resources': {'requests': {'cpu': '500m', 'memory': '1Gi'}, 'limits': {'cpu': '2', 'memory': '4Gi'}}}}
        return None
    
    # ── template generation ────────────────────────────────────────────

    def _generate_templates(self, workload: Dict[str, Any]) -> Dict[str, str]:
        """Generate Kubernetes templates"""
        templates = {}
        wtype = workload['workload_type']

        if wtype in ('training', 'cost-optimized'):
            templates['job.yaml'] = self._generate_job_template(workload)
        else:
            templates['deployment.yaml'] = self._generate_deployment_template(workload)
            templates['service.yaml'] = self._generate_service_template(workload)

        templates['configmap.yaml'] = self._generate_configmap_template(workload)

        if workload.get('storage_gb', 0) > 0:
            templates['pvc.yaml'] = self._generate_pvc_template(workload)

        templates['serviceaccount.yaml'] = self._generate_serviceaccount_template()
        if wtype not in ('training', 'cost-optimized'):
            templates['hpa.yaml'] = self._generate_hpa_template()
            templates['pdb.yaml'] = self._generate_pdb_template()

        return templates

    def _generate_job_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Job template"""
        return """apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "terradev.fullname" . }}-{{ .Release.Revision }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
spec:
  backoffLimit: {{ .Values.backoffLimit | default 3 }}
  {{- if .Values.ttlSecondsAfterFinished }}
  ttlSecondsAfterFinished: {{ .Values.ttlSecondsAfterFinished }}
  {{- end }}
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        {{- include "terradev.selectorLabels" . | nindent 8 }}
    spec:
      restartPolicy: {{ .Values.restartPolicy | default "Never" }}
      serviceAccountName: {{ include "terradev.serviceAccountName" . }}
      {{- with .Values.podSecurityContext }}
      securityContext:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        {{- if .Values.command }}
        command:
          {{- toYaml .Values.command | nindent 10 }}
        {{- end }}
        {{- if .Values.env }}
        envFrom:
        - configMapRef:
            name: {{ include "terradev.fullname" . }}-config
        {{- end }}
        {{- with .Values.securityContext }}
        securityContext:
          {{- toYaml . | nindent 10 }}
        {{- end }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- if .Values.gpu.storage }}
        volumeMounts:
        - name: storage
          mountPath: /data
        {{- end }}
      {{- if .Values.gpu.storage }}
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: {{ include "terradev.fullname" . }}-storage
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}"""

    def _generate_deployment_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Deployment template"""
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "terradev.fullname" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount | default 1 }}
  {{- end }}
  {{- with .Values.strategy }}
  strategy:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "terradev.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
      labels:
        {{- include "terradev.selectorLabels" . | nindent 8 }}
    spec:
      serviceAccountName: {{ include "terradev.serviceAccountName" . }}
      {{- with .Values.podSecurityContext }}
      securityContext:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        {{- if .Values.command }}
        command:
          {{- toYaml .Values.command | nindent 10 }}
        {{- end }}
        {{- if .Values.env }}
        envFrom:
        - configMapRef:
            name: {{ include "terradev.fullname" . }}-config
        {{- end }}
        {{- with .Values.securityContext }}
        securityContext:
          {{- toYaml . | nindent 10 }}
        {{- end }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- if .Values.service }}
        ports:
        {{- range .Values.service.ports }}
        - containerPort: {{ .targetPort }}
          protocol: TCP
        {{- end }}
        {{- end }}
        {{- if and .Values.probes .Values.probes.startup .Values.probes.startup.enabled }}
        startupProbe:
          httpGet:
            path: {{ .Values.probes.startup.path }}
            port: {{ (index .Values.service.ports 0).targetPort }}
          initialDelaySeconds: {{ .Values.probes.startup.initialDelaySeconds }}
          periodSeconds: {{ .Values.probes.startup.periodSeconds }}
          failureThreshold: {{ .Values.probes.startup.failureThreshold }}
        {{- end }}
        {{- if and .Values.probes .Values.probes.liveness .Values.probes.liveness.enabled }}
        livenessProbe:
          httpGet:
            path: {{ .Values.probes.liveness.path }}
            port: {{ (index .Values.service.ports 0).targetPort }}
          periodSeconds: {{ .Values.probes.liveness.periodSeconds }}
          failureThreshold: {{ .Values.probes.liveness.failureThreshold }}
        {{- end }}
        {{- if and .Values.probes .Values.probes.readiness .Values.probes.readiness.enabled }}
        readinessProbe:
          httpGet:
            path: {{ .Values.probes.readiness.path }}
            port: {{ (index .Values.service.ports 0).targetPort }}
          periodSeconds: {{ .Values.probes.readiness.periodSeconds }}
          failureThreshold: {{ .Values.probes.readiness.failureThreshold }}
        {{- end }}
        {{- if .Values.gpu.storage }}
        volumeMounts:
        - name: storage
          mountPath: /data
        {{- end }}
      {{- if .Values.gpu.storage }}
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: {{ include "terradev.fullname" . }}-storage
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      terminationGracePeriodSeconds: {{ .Values.terminationGracePeriodSeconds | default 30 }}"""

    def _generate_service_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Service template"""
        return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "terradev.fullname" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
  {{- with .Values.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.service.type | default "ClusterIP" }}
  ports:
  {{- range .Values.service.ports }}
    - port: {{ .port }}
      targetPort: {{ .targetPort }}
      protocol: TCP
      name: port-{{ .port }}
  {{- end }}
  selector:
    {{- include "terradev.selectorLabels" . | nindent 4 }}"""

    def _generate_configmap_template(self, workload: Dict[str, Any]) -> str:
        """Generate ConfigMap template — fully data-driven from .Values.env"""
        return """{{- if .Values.env }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "terradev.fullname" . }}-config
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
data:
  {{- range $key, $value := .Values.env }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
{{- end }}"""

    def _generate_pvc_template(self, workload: Dict[str, Any]) -> str:
        """Generate PersistentVolumeClaim template — pure raw string, no f-string"""
        return """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "terradev.fullname" . }}-storage
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {{ .Values.gpu.storage }}Gi
  {{- if .Values.persistence }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
  {{- else }}
  storageClassName: gp3
  {{- end }}"""

    def _generate_serviceaccount_template(self) -> str:
        """Generate ServiceAccount template"""
        return """{{- if .Values.serviceAccount.create }}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "terradev.serviceAccountName" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}"""

    def _generate_hpa_template(self) -> str:
        """Generate HorizontalPodAutoscaler template"""
        return """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "terradev.fullname" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "terradev.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
{{- end }}"""

    def _generate_pdb_template(self) -> str:
        """Generate PodDisruptionBudget template"""
        return """{{- if .Values.podDisruptionBudget.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "terradev.fullname" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4 }}
spec:
  minAvailable: {{ .Values.podDisruptionBudget.minAvailable }}
  selector:
    matchLabels:
      {{- include "terradev.selectorLabels" . | nindent 6 }}
{{- end }}"""
    
    # ── workload type templates ────────────────────────────────────────

    def _get_training_template(self) -> Dict[str, Any]:
        """Get training workload template"""
        return {
            'workloadType': 'Job',
            'restartPolicy': 'Never',
            'backoffLimit': 3,
            'ttlSecondsAfterFinished': 300,
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
            ],
        }

    def _get_inference_template(self) -> Dict[str, Any]:
        """Get inference workload template"""
        return {
            'workloadType': 'Deployment',
            'replicaCount': 1,
            'strategy': {'type': 'Recreate'},
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
            ],
        }

    def _get_cost_optimized_template(self) -> Dict[str, Any]:
        """Get cost-optimized workload template"""
        return {
            'workloadType': 'Job',
            'restartPolicy': 'Never',
            'backoffLimit': 2,
            'ttlSecondsAfterFinished': 60,
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
                {'key': 'spot', 'operator': 'Exists', 'effect': 'NoSchedule'},
            ],
        }

    def _get_high_performance_template(self) -> Dict[str, Any]:
        """Get high-performance workload template"""
        return {
            'workloadType': 'Deployment',
            'replicaCount': 1,
            'strategy': {'type': 'Recreate'},
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
                {'key': 'dedicated', 'operator': 'Equal', 'value': 'gpu-inference', 'effect': 'NoSchedule'},
            ],
            'affinity': {
                'podAntiAffinity': {
                    'preferredDuringSchedulingIgnoredDuringExecution': [{
                        'weight': 100,
                        'podAffinityTerm': {
                            'labelSelector': {'matchExpressions': [
                                {'key': 'app.kubernetes.io/name', 'operator': 'In',
                                 'values': ['{{ include "terradev.name" . }}']}
                            ]},
                            'topologyKey': 'kubernetes.io/hostname',
                        },
                    }]
                }
            },
        }

    # ── MoE / RAG / vLLM workload types (Strategic #9) ────────────────

    def _get_moe_inference_template(self) -> Dict[str, Any]:
        """Get MoE inference workload template — mirrors clusters/moe-template"""
        return {
            'workloadType': 'Deployment',
            'replicaCount': 1,
            'strategy': {'type': 'Recreate'},
            'serving': {
                'backend': 'vllm', 'port': 8000, 'host': '0.0.0.0',
                'expertParallel': {
                    'enabled': True, 'dataParallelSize': 8,
                    'enableEplb': True, 'enableDbo': True,
                    'all2allBackend': 'deepep_low_latency',
                },
                'vllm': {
                    'tensorParallelSize': 8, 'gpuMemoryUtilization': 0.95,
                    'maxModelLen': 32768, 'dtype': 'auto', 'trustRemoteCode': True,
                    'maxNumBatchedTokens': 16384, 'maxNumSeqs': 1024,
                    'enablePrefixCaching': True, 'enableChunkedPrefill': True,
                },
                'flashinfer': {'enabled': True, 'backend': 'FLASHINFER'},
                'sleepMode': {'enabled': True, 'level': 1},
                'kvOffloading': {'enabled': True, 'connector': 'offloading'},
                'speculative': {'enabled': True, 'method': 'mtp', 'numTokens': 1},
                'lmcache': {
                    'enabled': True, 'backend': 'redis',
                    'redisUrl': 'redis://lmcache-redis:6379',
                    'chunkSize': 256, 'pipelined': True,
                },
                'lora': {'enabled': False, 'maxLoras': 8, 'maxLoraRank': 64},
                'router': {'enabled': False, 'replicas': 2, 'port': 8080, 'policy': 'consistent_hash'},
            },
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
                {'key': 'dedicated', 'operator': 'Equal', 'value': 'gpu-inference', 'effect': 'NoSchedule'},
            ],
            'persistence': {
                'modelCache': {'enabled': True, 'size': '500Gi', 'storageClass': 'nvme-ssd', 'mountPath': '/models'},
                'sharedMemory': {'enabled': True, 'size': '32Gi', 'mountPath': '/dev/shm'},
            },
            'env': {
                'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                'NCCL_P2P_DISABLE': '0', 'NCCL_IB_DISABLE': '0', 'NCCL_SOCKET_IFNAME': 'eth0',
                'VLLM_ATTENTION_BACKEND': 'FLASHINFER', 'VLLM_USE_DEEP_GEMM': '1', 'VLLM_SERVER_DEV_MODE': '1',
            },
        }

    def _get_rag_template(self) -> Dict[str, Any]:
        """Get RAG infrastructure workload template — mirrors clusters/rag-template"""
        return {
            'workloadType': 'Deployment',
            'replicaCount': 1,
            'strategy': {'type': 'Recreate'},
            'serving': {
                'backend': 'vllm', 'port': 8000,
                'vllm': {'tensorParallelSize': 1, 'gpuMemoryUtilization': 0.9, 'maxModelLen': 32768},
                'flashInfer': {'enabled': True},
                'sleepMode': {'enabled': True},
                'kvOffloading': {'enabled': True, 'connector': 'offloading'},
                'speculative': {'enabled': True, 'method': 'mtp', 'numTokens': 5},
                'lmcache': {'enabled': True, 'backend': 'redis', 'remoteUrl': 'redis://redis-svc:6379'},
            },
            'embedding': {
                'model': 'BAAI/bge-large-en-v1.5', 'replicas': 1, 'port': 8001, 'backend': 'fastembed',
                'resources': {'requests': {'cpu': '2', 'memory': '4Gi'}, 'limits': {'cpu': '4', 'memory': '8Gi'}},
            },
            'redis': {'enabled': True, 'image': 'redis:7-alpine', 'port': 6379, 'persistence': {'enabled': True, 'size': '10Gi'}},
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
            ],
        }

    def _get_vllm_optimized_template(self) -> Dict[str, Any]:
        """Get vLLM-optimized inference template with all 6 critical knobs"""
        return {
            'workloadType': 'Deployment',
            'replicaCount': 1,
            'strategy': {'type': 'Recreate'},
            'serving': {
                'backend': 'vllm', 'port': 8000,
                'vllm': {
                    'tensorParallelSize': 1, 'gpuMemoryUtilization': 0.92,
                    'maxModelLen': 8192, 'dtype': 'auto', 'trustRemoteCode': True,
                    'maxNumBatchedTokens': 8192, 'maxNumSeqs': 256,
                    'enablePrefixCaching': True, 'enableChunkedPrefill': True,
                },
                'flashinfer': {'enabled': True, 'backend': 'FLASHINFER'},
                'sleepMode': {'enabled': True, 'level': 1},
                'kvOffloading': {'enabled': True, 'connector': 'offloading'},
                'speculative': {'enabled': False},
            },
            'nodeSelector': {'nvidia.com/gpu.product': '{{ .Values.gpu.nodeLabel }}'},
            'tolerations': [
                {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
            ],
            'env': {
                'VLLM_ATTENTION_BACKEND': 'FLASHINFER', 'VLLM_USE_DEEP_GEMM': '1',
            },
        }

    # ── file writers ───────────────────────────────────────────────────

    def _write_chart_yaml(self, chart_path: Path, config: HelmChartConfig):
        """Write Chart.yaml"""
        chart_data = {
            'apiVersion': 'v2',
            'name': config.name,
            'description': config.description,
            'type': 'application',
            'version': config.version,
            'appVersion': config.app_version,
            'kubeVersion': config.kube_version,
            'maintainers': config.maintainers,
            'keywords': config.keywords,
        }
        if config.dependencies:
            chart_data['dependencies'] = config.dependencies

        with open(chart_path / 'Chart.yaml', 'w') as f:
            yaml.dump(chart_data, f, default_flow_style=False)

    def _write_values_yaml(self, chart_path: Path, values: Dict[str, Any]):
        """Write values.yaml"""
        with open(chart_path / 'values.yaml', 'w') as f:
            yaml.dump(values, f, default_flow_style=False)

    def _write_templates(self, chart_path: Path, templates: Dict[str, str]):
        """Write template files"""
        for filename, content in templates.items():
            with open(chart_path / 'templates' / filename, 'w') as f:
                f.write(content)
        self._write_helper_templates(chart_path)

    def _write_helper_templates(self, chart_path: Path):
        """Write helper templates"""
        helpers = {
            '_helpers.tpl': self._get_helpers_tpl(),
            'NOTES.txt': self._get_notes_txt(),
        }
        for filename, content in helpers.items():
            with open(chart_path / 'templates' / filename, 'w') as f:
                f.write(content)

    @staticmethod
    def _get_helpers_tpl() -> str:
        return """{{- /*
Expand the name of the chart.
*/}}
{{- define "terradev.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- /*
Create a default fully qualified app name.
*/}}
{{- define "terradev.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- /*
Create chart name and version as used by the chart label.
*/}}
{{- define "terradev.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- /*
Common labels
*/}}
{{- define "terradev.labels" -}}
helm.sh/chart: {{ include "terradev.chart" . }}
{{ include "terradev.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- /*
Selector labels
*/}}
{{- define "terradev.selectorLabels" -}}
app.kubernetes.io/name: {{ include "terradev.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- /*
Create the name of the service account to use
*/}}
{{- define "terradev.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "terradev.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
"""

    @staticmethod
    def _get_notes_txt() -> str:
        return """Terradev GPU Workload deployed!

GPU: {{ .Values.gpu.type }} x{{ .Values.gpu.count }}
Node label: {{ .Values.gpu.nodeLabel }}

{{- if .Values.budget.enforce }}
Budget: ${{ .Values.budget.maxHourlyRate }}/hr (enforced)
{{- end }}

{{- if eq .Values.workloadType "Job" }}
Check job status:
  kubectl get jobs -l app.kubernetes.io/name={{ include "terradev.fullname" . }}
View logs:
  kubectl logs job/{{ include "terradev.fullname" . }}
{{- else }}
Check deployment:
  kubectl get deploy {{ include "terradev.fullname" . }}
View logs:
  kubectl logs deploy/{{ include "terradev.fullname" . }}
{{- if .Values.service }}
Service:
  kubectl get svc {{ include "terradev.fullname" . }}
{{- end }}
{{- end }}
"""
    
    def _write_readme(self, chart_path: Path, readme: str):
        """Write README.md"""
        with open(chart_path / 'README.md', 'w') as f:
            f.write(readme)

    def _generate_readme(self, workload: Dict[str, Any], chart_name: str) -> str:
        """Generate README content"""
        gpu_label = GPU_NODE_LABELS.get(workload['gpu_type'], workload['gpu_type'])
        stacks = workload.get('stack', [])
        stack_section = ""
        if stacks:
            stack_section = "\n### Stack Integrations\n\n" + "\n".join(f"- `{s}`" for s in stacks) + "\n"

        return f"""# {chart_name}

Terradev Helm chart for **{workload['workload_type']}** workloads on {workload['gpu_type']} GPUs.

## Quick Start

```bash
terradev helm-generate \\
  --workload {workload['workload_type']} \\
  --gpu-type {workload['gpu_type']} \\
  --image {workload['image']} \\
  --gpu-count {workload.get('gpu_count', 1)} \\
  --output {chart_name}

cd {chart_name}
helm install my-{workload['workload_type']} . --namespace terradev-workloads
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image | `{workload['image']}` |
| `gpu.type` | GPU type | `{workload['gpu_type']}` |
| `gpu.count` | Number of GPUs | `{workload.get('gpu_count', 1)}` |
| `gpu.nodeLabel` | K8s node label | `{gpu_label}` |
| `gpu.storage` | Storage in GB | `{workload.get('storage_gb', 100)}` |
| `budget.maxHourlyRate` | Max $/hr | `{workload.get('budget')}` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `podDisruptionBudget.enabled` | Enable PDB | `true` |
| `serviceAccount.create` | Create SA | `true` |

## Workload Types

| Type | K8s Kind | Use Case |
|------|----------|----------|
| `training` | Job | Model training, batch processing |
| `inference` | Deployment + Service | Model serving, real-time inference |
| `cost-optimized` | Job | Budget-constrained, spot instances |
| `high-performance` | Deployment | Multi-GPU, anti-affinity |
| `moe-inference` | Deployment | MoE expert parallel, vLLM optimized |
| `rag` | Deployment | RAG stack (vLLM + Qdrant + Embedding) |
| `vllm-optimized` | Deployment | vLLM with FlashInfer, KV offloading |
{stack_section}
## Production Features

- **Health probes**: startup + liveness + readiness
- **Security context**: runAsNonRoot, seccomp, drop ALL capabilities
- **ServiceAccount + RBAC**: auto-created
- **HPA**: configurable autoscaling (disabled by default)
- **PDB**: minAvailable=1
- **Config checksum**: auto-restart on ConfigMap change
- **Metrics**: ServiceMonitor for Prometheus

## Monitoring

```bash
kubectl get nodes -l nvidia.com/gpu.product={gpu_label}
kubectl logs deploy/my-{workload['workload_type']}
kubectl get events --field-selector reason=FailedScheduling
```

## More Information

- [Terradev Documentation](https://terradev.dev/docs)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
"""
