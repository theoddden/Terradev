# GLM-5 Cluster Configuration for Terradev

Production-optimized cluster structure for deploying [GLM-5](https://github.com/theoddden/GLM-5-x-Terradev) (744B params, 40B active MoE) on Terradev infrastructure.

## Model Specs

| Property | Value |
|----------|-------|
| Parameters | 744B total, 40B active (MoE) |
| Precision | FP8 (recommended), BF16 (full) |
| VRAM Required | ~744GB FP8, ~1.5TB BF16 |
| Tensor Parallelism | 8 (minimum) |
| GPU Requirement | 8× H100 SXM 80GB (minimum) |
| Serving Backends | vLLM, SGLang |
| Context Window | Long-context via DeepSeek Sparse Attention |

## Quick Start

```bash
# One-command deploy with Terradev CLI
terradev glm5-deploy --backend vllm --provider runpod --dry-run

# Or use the task YAML directly
terradev provision --task clusters/glm-5/task.yaml

# Kubernetes deployment
terradev k8s create --config clusters/glm-5/k8s/
```

## Directory Structure

```
clusters/glm-5/
├── README.md                     # This file
├── task.yaml                     # Terradev CLI task definition
├── terraform/
│   ├── main.tf                   # GPU node provisioning (multi-cloud)
│   ├── variables.tf              # Configurable parameters
│   └── outputs.tf                # Provisioned endpoints
├── k8s/
│   ├── namespace.yaml            # glm-5-inference namespace
│   ├── vllm-deployment.yaml      # vLLM serving deployment
│   ├── sglang-deployment.yaml    # SGLang serving deployment
│   ├── service.yaml              # LoadBalancer + health checks
│   ├── hpa.yaml                  # GPU-aware autoscaling
│   ├── pdb.yaml                  # Pod disruption budget
│   └── model-cache-pvc.yaml      # Shared model weights storage
└── helm/
    └── values-glm5.yaml          # Helm overrides for GLM-5
```

## Supported Providers (by cost)

| Provider | GPU | Est. Cost/hr | TP Support | Availability |
|----------|-----|-------------|------------|--------------|
| RunPod | 8× H100 SXM | ~$23.60 | ✅ NVLink | High |
| Vast.ai | 8× H100 SXM | ~$21.00 | ✅ NVLink | Medium |
| Lambda Labs | 8× H100 SXM | ~$27.60 | ✅ NVLink | Medium |
| AWS (p5.48xlarge) | 8× H100 SXM | ~$98.32 | ✅ NVLink | High |
| CoreWeave | 8× H100 SXM | ~$35.28 | ✅ NVLink | High |

## Backends

### vLLM (recommended for production)
- MTP speculative decoding (1 speculative token)
- Tool calling via `glm47` parser
- Reasoning via `glm45` parser
- Auto tool choice enabled

### SGLang (recommended for throughput)
- EAGLE speculative decoding (3 steps, 4 draft tokens)
- Higher throughput for batch workloads
- Hopper and Blackwell Docker images available
