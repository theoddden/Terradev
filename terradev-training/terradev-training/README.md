# terradev-training

Terradev Helm chart for **training** workloads on A100 GPUs.

## Quick Start

```bash
terradev helm-generate \
  --workload training \
  --gpu-type A100 \
  --image test \
  --gpu-count 1 \
  --output terradev-training

cd terradev-training
helm install my-training . --namespace terradev-workloads
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image | `test` |
| `gpu.type` | GPU type | `A100` |
| `gpu.count` | Number of GPUs | `1` |
| `gpu.nodeLabel` | K8s node label | `NVIDIA-A100-SXM4-80GB` |
| `gpu.storage` | Storage in GB | `100` |
| `budget.maxHourlyRate` | Max $/hr | `None` |
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
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB
kubectl logs deploy/my-training
kubectl get events --field-selector reason=FailedScheduling
```

## More Information

- [Terradev Documentation](https://terradev.dev/docs)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
