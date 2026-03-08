#!/usr/bin/env python3
"""
Helm Chart Generator
Generate Helm charts from Terradev workloads for Kubernetes deployment
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HelmChartConfig:
    name: str
    version: str
    description: str
    app_version: str
    kube_version: str
    maintainers: List[Dict[str, str]]
    keywords: List[str]

class HelmChartGenerator:
    """Generate Helm charts from Terradev workloads"""
    
    def __init__(self):
        self.chart_templates = {
            'training': self._get_training_template(),
            'inference': self._get_inference_template(),
            'cost-optimized': self._get_cost_optimized_template(),
            'high-performance': self._get_high_performance_template()
        }
    
    def generate_chart(self, workload_config: Dict[str, Any], output_dir: str) -> str:
        """Generate complete Helm chart from Terradev workload"""
        chart_name = workload_config.get('name', f"terradev-{workload_config['workload_type']}")
        chart_path = Path(output_dir) / chart_name
        
        # Create chart directory structure
        chart_path.mkdir(parents=True, exist_ok=True)
        (chart_path / "templates").mkdir(exist_ok=True)
        (chart_path / "charts").mkdir(exist_ok=True)
        
        # Generate Chart.yaml
        chart_config = self._generate_chart_config(workload_config, chart_name)
        self._write_chart_yaml(chart_path, chart_config)
        
        # Generate values.yaml
        values = self._generate_values(workload_config)
        self._write_values_yaml(chart_path, values)
        
        # Generate templates
        templates = self._generate_templates(workload_config)
        self._write_templates(chart_path, templates)
        
        # Generate README
        readme = self._generate_readme(workload_config, chart_name)
        self._write_readme(chart_path, readme)
        
        return str(chart_path)
    
    def _generate_chart_config(self, workload: Dict[str, Any], chart_name: str) -> HelmChartConfig:
        """Generate Chart.yaml configuration"""
        return HelmChartConfig(
            name=chart_name,
            version="1.0.0",
            description=f"Terradev {workload['workload_type'].title()} workload for {workload['gpu_type']}",
            app_version="1.0.0",
            kube_version=">=1.20.0-0",
            maintainers=[
                {
                    "name": "Terradev",
                    "email": "support@terradev.dev",
                    "url": "https://terradev.dev"
                }
            ],
            keywords=["gpu", "machine-learning", "kubernetes", workload['workload_type'], workload['gpu_type']]
        )
    
    def _generate_values(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate values.yaml"""
        base_values = self.chart_templates[workload['workload_type']]
        
        # Override with workload-specific values
        values = {
            **base_values,
            'image': {
                'repository': workload['image'],
                'tag': workload.get('tag', 'latest'),
                'pullPolicy': 'IfNotPresent'
            },
            'gpu': {
                'type': workload['gpu_type'],
                'count': workload.get('gpu_count', 1),
                'memory': workload.get('memory_gb', 16),
                'storage': workload.get('storage_gb', 100)
            },
            'resources': self._calculate_resources(workload),
            'budget': {
                'maxHourlyRate': workload.get('budget'),
                'enforce': workload.get('budget') is not None
            },
            'terradev': {
                'provider': workload.get('provider', 'auto'),
                'region': workload.get('region', 'us-east-1'),
                'spot': workload.get('spot', True)
            }
        }
        
        # Add environment variables
        if workload.get('environment_vars'):
            values['env'] = workload['environment_vars']
        
        # Add ports for inference workloads
        if workload['workload_type'] == 'inference' and workload.get('ports'):
            values['service'] = {
                'type': 'LoadBalancer',
                'ports': [{'port': port, 'targetPort': port} for port in workload['ports']]
            }
        
        return values
    
    def _calculate_resources(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource requirements"""
        gpu_count = workload.get('gpu_count', 1)
        memory_gb = workload.get('memory_gb', 16)
        
        # Estimate CPU based on GPU count (rule of thumb: 4-8 CPU cores per GPU)
        cpu_cores = gpu_count * 6
        
        # Add memory for GPU (GPU memory + system memory)
        gpu_memory_map = {
            'A100': 40,  # 40GB per A100
            'H100': 80,  # 80GB per H100
            'V100': 16,  # 16GB per V100
            'RTX 3090': 24,  # 24GB per RTX 3090
            'RTX 4090': 24,  # 24GB per RTX 4090
        }
        
        gpu_memory = gpu_memory_map.get(workload['gpu_type'], 16) * gpu_count
        total_memory = max(memory_gb, gpu_memory + 8)  # Add 8GB for system
        
        return {
            'requests': {
                'cpu': f"{cpu_cores}m",
                'memory': f"{total_memory}Gi",
                'nvidia.com/gpu': str(gpu_count)
            },
            'limits': {
                'cpu': f"{cpu_cores * 2}m",
                'memory': f"{total_memory * 2}Gi",
                'nvidia.com/gpu': str(gpu_count)
            }
        }
    
    def _generate_templates(self, workload: Dict[str, Any]) -> Dict[str, str]:
        """Generate Kubernetes templates"""
        templates = {}
        
        if workload['workload_type'] in ['training', 'cost-optimized']:
            # Generate Job template
            templates['job.yaml'] = self._generate_job_template(workload)
        else:
            # Generate Deployment template
            templates['deployment.yaml'] = self._generate_deployment_template(workload)
            
            # Generate Service template for inference
            if workload.get('ports'):
                templates['service.yaml'] = self._generate_service_template(workload)
        
        # Generate ConfigMap for environment variables
        if workload.get('environment_vars'):
            templates['configmap.yaml'] = self._generate_configmap_template(workload)
        
        # Generate PVC for storage
        if workload.get('storage_gb', 0) > 0:
            templates['pvc.yaml'] = self._generate_pvc_template(workload)
        
        return templates
    
    def _generate_job_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Job template"""
        return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: "{{{{ include "terradev.fullname" . }}}}-{{{{ .Release.Revision }}}}"
  labels:
    {{{{- include "terradev.labels" . | nindent 4}}}}
spec:
  backoffLimit: 3
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        {{{{- include "terradev.selectorLabels" . | nindent 8}}}}
    spec:
      restartPolicy: Never
      containers:
      - name: "{{{{ .Chart.Name }}}}"
        image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag }}}}"
        imagePullPolicy: {{{{{ .Values.image.pullPolicy | quote }}}}
        command: {self._format_command(workload.get('command', []))}
        {{{{- if .Values.env }}}}
        envFrom:
        - configMapRef:
            name: "{{{{ include "terradev.fullname" . }}}-config"
        {{{{- end }}}}
        resources:
          {{{{- toYaml .Values.resources | nindent 10}}}}
        {{{{- if .Values.storage }}}}
        volumeMounts:
        - name: storage
          mountPath: /data
        {{{{- end }}}}
      {{{{- if .Values.storage }}}}
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: "{{{{ include "terradev.fullname" . }}}-storage"
      {{{{- end}}}}
      nodeSelector:
        {{{{- include "terradev.nodeSelector" . | nindent 8}}}}
      tolerations:
        {{{{- include "terradev.tolerations" . | nindent 8}}}}
      affinity:
        {{{{- include "terradev.affinity" . | nindent 8}}}}"""
    
    def _generate_deployment_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Deployment template"""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ include "terradev.fullname" . }}}
  labels:
    {{- include "terradev.labels" . | nindent 4}}
spec:
  replicas: 1
  selector:
    matchLabels:
      {{- include "terradev.selectorLabels" . | nindent 6}}
  template:
    metadata:
      labels:
        {{- include "terradev.selectorLabels" . | nindent 8}}
    spec:
      containers:
      - name: {{{{ .Chart.Name }}}
        image: "{{{{ .Values.image.repository }}}:{{{{ .Values.image.tag }}}}"
        imagePullPolicy: {{{{ .Values.image.pullPolicy | quote }}}
        command: {self._format_command(workload.get('command', []))}
        {{- if .Values.env }}
        envFrom:
        - configMapRef:
            name: {{{{ include "terradev.fullname" . }}}-config
        {{- end }}
        resources:
          {{- toYaml .Values.resources | nindent 10}}
        {{- if .Values.service.ports }}
        ports:
        {{- range .Values.service.ports }}
        - containerPort: {{{{ .targetPort }}}
          protocol: TCP
        {{- end }}
        {{- end }}
        {{- if .Values.storage }}
        volumeMounts:
        - name: storage
          mountPath: /data
        {{- end }}
      {{- if .Values.storage }}
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: {{{{ include "terradev.fullname" . }}}-storage
      {{- end}}
      nodeSelector:
        {{- include "terradev.nodeSelector" . | nindent 8}}
      tolerations:
        {{- include "terradev.tolerations" . | nindent 8}}
      affinity:
        {{- include "terradev.affinity" . | nindent 8}}"""
    
    def _generate_service_template(self, workload: Dict[str, Any]) -> str:
        """Generate Kubernetes Service template"""
        return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "terradev.fullname" . }}
  labels:
    {{- include "terradev.labels" . | nindent 4}}
spec:
  type: {{ .Values.service.type }}
  ports:
  {{- range .Values.service.ports }}
    - port: {{ .port }}
      targetPort: {{ .targetPort }}
      protocol: TCP
      name: port-{{ .port }}
  {{- end }}
  selector:
    {{- include "terradev.selectorLabels" . | nindent 4}}"""
    
    def _generate_configmap_template(self, workload: Dict[str, Any]) -> str:
        """Generate ConfigMap template"""
        env_vars = workload.get('environment_vars', {})
        return """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "terradev.fullname" . }}-config
  labels:
    {{- include "terradev.labels" . | nindent 4}}
data:
""" + '\n'.join([f'  {k}: "{v}"' for k, v in env_vars.items()])
    
    def _generate_pvc_template(self, workload: Dict[str, Any]) -> str:
        """Generate PersistentVolumeClaim template"""
        storage_gb = workload.get('storage_gb', 100)
        return f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{{{ include "terradev.fullname" . }}}-storage
  labels:
    {{- include "terradev.labels" . | nindent 4}}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {storage_gb}Gi
  storageClassName: gp3"""
    
    def _format_command(self, command: List[str]) -> str:
        """Format command for YAML"""
        if not command:
            return "[]"
        return str(command).replace("'", '"')
    
    def _get_training_template(self) -> Dict[str, Any]:
        """Get training workload template"""
        return {
            'workloadType': 'Job',
            'restartPolicy': 'Never',
            'backoffLimit': 3,
            'ttlSecondsAfterFinished': 300,
            'nodeSelector': {
                'accelerator': 'nvidia-tesla-a100'
            },
            'tolerations': [
                {
                    'key': 'nvidia.com/gpu',
                    'operator': 'Exists',
                    'effect': 'NoSchedule'
                }
            ]
        }
    
    def _get_inference_template(self) -> Dict[str, Any]:
        """Get inference workload template"""
        return {
            'workloadType': 'Deployment',
            'replicas': 1,
            'service': {
                'type': 'LoadBalancer'
            },
            'nodeSelector': {
                'accelerator': 'nvidia-tesla-a100'
            },
            'tolerations': [
                {
                    'key': 'nvidia.com/gpu',
                    'operator': 'Exists',
                    'effect': 'NoSchedule'
                }
            ]
        }
    
    def _get_cost_optimized_template(self) -> Dict[str, Any]:
        """Get cost-optimized workload template"""
        return {
            'workloadType': 'Job',
            'restartPolicy': 'Never',
            'backoffLimit': 2,
            'ttlSecondsAfterFinished': 60,
            'nodeSelector': {
                'accelerator': 'nvidia-tesla-a100',
                'instance-type': 'g4dn'
            },
            'tolerations': [
                {
                    'key': 'nvidia.com/gpu',
                    'operator': 'Exists',
                    'effect': 'NoSchedule'
                },
                {
                    'key': 'spot',
                    'operator': 'Exists',
                    'effect': 'NoSchedule'
                }
            ]
        }
    
    def _get_high_performance_template(self) -> Dict[str, Any]:
        """Get high-performance workload template"""
        return {
            'workloadType': 'Deployment',
            'replicas': 1,
            'nodeSelector': {
                'accelerator': 'nvidia-tesla-a100',
                'instance-type': 'p4d'
            },
            'tolerations': [
                {
                    'key': 'nvidia.com/gpu',
                    'operator': 'Exists',
                    'effect': 'NoSchedule'
                }
            ],
            'affinity': {
                'nodeAffinity': {
                    'requiredDuringSchedulingIgnoredDuringExecution': {
                        'nodeSelectorTerms': [
                            {
                                'matchExpressions': [
                                    {
                                        'key': 'topology.kubernetes.io/zone',
                                        'operator': 'In',
                                        'values': ['us-east-1a', 'us-east-1b', 'us-east-1c']
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        }
    
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
            'keywords': config.keywords
        }
        
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
        
        # Generate helper templates
        self._write_helper_templates(chart_path)
    
    def _write_helper_templates(self, chart_path: Path):
        """Write helper templates"""
        helpers = {
            '_helpers.tpl': """{{- /*
Generate basic labels for Terradev workloads
*/}}
{{- define "terradev.labels" -}}
helm.sh/chart: {{ include "terradev.chart" . }}
{{ include "terradev.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end }}

{{- /*
Selector labels
*/}}
{{- define "terradev.selectorLabels" -}}
app.kubernetes.io/name: {{ include "terradev.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- /*
Create the name of the chart
*/}}
{{- define "terradev.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- /*
Create chart name and version as used by the chart label
*/}}
{{- define "terradev.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- /*
Common labels
*/}}
{{- define "terradev.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- if contains $name .Release.Name }}
{{- $name }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- /*
Node selector for GPU workloads
*/}}
{{- define "terradev.nodeSelector" -}}
accelerator: nvidia-tesla-{{ .Values.gpu.type | lower }}
{{- if eq .Values.terradev.spot true }}
instance-type: {{ .Values.terradev.provider }}-spot
{{- end }}
{{- end }}

{{- /*
Tolerations for GPU workloads
*/}}
{{- define "terradev.tolerations" -}}
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
{{- if eq .Values.terradev.spot true }}
- key: spot
  operator: Exists
  effect: NoSchedule
{{- end }}
{{- end }}

{{- /*
Affinity for GPU workloads
*/}}
{{- define "terradev.affinity" -}}
nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      - key: accelerator
        operator: In
        values:
        - nvidia-tesla-{{ .Values.gpu.type | lower }}
{{- if eq .Values.terradev.spot true }}
      - key: instance-type
        operator: In
        values:
        - spot
{{- end }}
{{- end }}
""",
            'NOTES.txt': """Terradev GPU Workload

This chart deploys a {{ .Values.gpu.type }} GPU workload on Kubernetes.

{{- if .Values.budget.enforce }}
Budget enforcement is enabled with a maximum rate of ${{ .Values.budget.maxHourlyRate }}/hour.
{{- end }}

## GPU Resources
- Type: {{ .Values.gpu.type }}
- Count: {{ .Values.gpu.count }}
- Memory: {{ .Values.resources.requests.memory }}

## Accessing the Workload

{{- if eq .Values.workloadType "Job" }}
Check job status:
  kubectl get jobs -l app.kubernetes.io/name={{ include "terradev.fullname" . }}

View job logs:
  kubectl logs job/{{ include "terradev.fullname" . }}
{{- else }}
Check deployment status:
  kubectl get deployments -l app.kubernetes.io/name={{ include "terradev.fullname" . }}

View pod logs:
  kubectl logs deployment/{{ include "terradev.fullname" . }}

{{- if .Values.service }}
Access the service:
  kubectl get service {{ include "terradev.fullname" . }}
{{- end }}
{{- end }}
"""
        }
        
        for filename, content in helpers.items():
            with open(chart_path / 'templates' / filename, 'w') as f:
                f.write(content)
    
    def _write_readme(self, chart_path: Path, readme: str):
        """Write README.md"""
        with open(chart_path / 'README.md', 'w') as f:
            f.write(readme)
    
    def _generate_readme(self, workload: Dict[str, Any], chart_name: str) -> str:
        """Generate README content"""
        return f"""# {chart_name}

Terradev Helm chart for {workload['workload_type'].title()} workloads using {workload['gpu_type']} GPUs.

## Description

This chart deploys a GPU-accelerated workload on Kubernetes with automatic node provisioning via Karpenter.

## Prerequisites

- Kubernetes 1.20+
- Karpenter installed and configured
- NVIDIA GPU operator installed
- Sufficient GPU node capacity or Karpenter configured for GPU instances

## Installation

### Add the Terradev Helm repository

```bash
helm repo add terradev https://charts.terradev.dev
helm repo update
```

### Install the chart

```bash
helm install my-{workload['workload_type']} terradev/{chart_name} \\
  --set image.repository={workload['image']} \\
  --set gpu.type={workload['gpu_type']} \\
  --set gpu.count={workload.get('gpu_count', 1)} \\
  --set budget.maxHourlyRate={workload.get('budget')} \\
  --namespace terradev-workloads
```

## Configuration

The following table lists the configurable parameters of the {chart_name} chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `{workload['image']}` |
| `image.tag` | Container image tag | `latest` |
| `gpu.type` | GPU type to use | `{workload['gpu_type']}` |
| `gpu.count` | Number of GPUs | `{workload.get('gpu_count', 1)}` |
| `gpu.memory` | GPU memory in GB | `{workload.get('memory_gb', 16)}` |
| `gpu.storage` | Storage size in GB | `{workload.get('storage_gb', 100)}` |
| `budget.maxHourlyRate` | Maximum hourly rate in USD | `{workload.get('budget')}` |
| `budget.enforce` | Whether to enforce budget limits | `true` |
| `terradev.provider` | Cloud provider | `auto` |
| `terradev.region` | Cloud region | `us-east-1` |
| `terradev.spot` | Use spot instances | `true` |

## GPU Workload Types

### Training
- **Type**: Kubernetes Job
- **Use Case**: Model training, batch processing
- **Features**: Automatic cleanup, restart on failure

### Inference
- **Type**: Kubernetes Deployment + Service
- **Use Case**: Model serving, real-time inference
- **Features**: Load balancing, external access

### Cost-Optimized
- **Type**: Kubernetes Job
- **Use Case**: Budget-constrained workloads
- **Features**: Spot instances, aggressive cleanup

### High-Performance
- **Type**: Kubernetes Deployment
- **Use Case**: Large-scale training, HPC workloads
- **Features**: Multi-AZ, high-performance networking

## Monitoring and Logs

### Check workload status

```bash
# For training jobs
kubectl get jobs -l app.kubernetes.io/name=my-{workload['workload_type']}

# For inference deployments
kubectl get deployments -l app.kubernetes.io/name=my-{workload['workload_type']}
```

### View logs

```bash
# View pod logs
kubectl logs -l app.kubernetes.io/name=my-{workload['workload_type']}

# Follow logs
kubectl logs -f -l app.kubernetes.io/name=my-{workload['workload_type']}
```

### Monitor GPU utilization

```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-{workload['gpu_type'].lower()}

# View GPU metrics
kubectl describe node <gpu-node-name>
```

## Cost Management

The chart includes built-in cost management features:

- **Budget Enforcement**: Automatically stops workloads exceeding budget
- **Spot Optimization**: Prefers spot instances for cost savings
- **Resource Efficiency**: Right-sized GPU allocations

### Budget Alerts

Set up budget alerts to monitor spending:

```bash
# Check current spending
kubectl get events --field-selector reason=BudgetExceeded
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   kubectl get nodes -l accelerator=nvidia-tesla-{workload['gpu_type'].lower()}
   kubectl describe node <gpu-node>
   ```

2. **Workload Pending**
   ```bash
   kubectl get events --field-selector reason=FailedScheduling
   ```

3. **Budget Exceeded**
   ```bash
   kubectl get events --field-selector reason=BudgetExceeded
   ```

### Getting Help

- [Terradev Documentation](https://terradev.dev/docs)
- [Karpenter Documentation](https://karpenter.sh/)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)

## Contributing

This chart is maintained by the Terradev team. For contributions and issues, please visit our [GitHub repository](https://github.com/terradev/helm-charts).

## License

This chart is licensed under the Apache License 2.0.
"""
