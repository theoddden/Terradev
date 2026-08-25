# MoE Model Cluster Template for Terradev

A **generalized, parameterized cluster template** for deploying any Mixture-of-Experts (MoE) large language model on Terradev infrastructure. Designed for the current wave of MoE models: GLM-5, Qwen 3.5, Mistral Large 3, DeepSeek V4, Llama 5, and beyond.

## Why MoE-Optimized?

Every major model release in 2026 uses Mixture-of-Experts architecture. MoE models have unique infrastructure requirements:

- **High aggregate VRAM** — total params are large (400B–750B+), but active params are small (17B–41B)
- **NVLink mandatory** — tensor parallelism across 4–8 GPUs requires high-bandwidth interconnect
- **Large shared memory** — NCCL collective ops need 16–32GB `/dev/shm`
- **Fast model loading** — NVMe storage for 200–800GB weight files
- **FP8 quantization** — halves VRAM vs BF16, supported on H100/H200

## Supported Models (tested configs)

| Model | Total Params | Active Params | TP Size | FP8 VRAM | GPU Requirement |
|-------|-------------|---------------|---------|----------|-----------------|
| GLM-5 | 744B | 40B | 8 | ~744GB | 8× H100 SXM 80GB |
| Qwen 3.5 | 397B | 17B | 4–8 | ~397GB | 4–8× H100 SXM 80GB |
| Mistral Large 3 | 675B | 41B | 8 | ~675GB | 8× H100 SXM 80GB |
| DeepSeek V3/V4 | 671B | 37B | 8 | ~671GB | 8× H100 SXM 80GB |
| Llama 5 (est.) | ~600B+ | ~40B | 8 | ~600GB | 8× H100 SXM 80GB |
| Qwen 3.5-122B | 122B | 10B | 2–4 | ~122GB | 2–4× H100 SXM 80GB |
| Qwen 3.5-35B | 35B | 3B | 1 | ~35GB | 1× H100/A100 80GB |

## Quick Start

```bash
# Deploy any MoE model with one command
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=zai-org/GLM-5-FP8 \
  --set tp_size=8 \
  --set gpu_count=8

# Or deploy Qwen 3.5 flagship
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-397B-A17B \
  --set tp_size=8 \
  --set gpu_count=8

# Smaller model on fewer GPUs
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-122B-A10B \
  --set tp_size=4 \
  --set gpu_count=4

# Kubernetes
kubectl apply -f clusters/moe-template/k8s/

# Helm
helm upgrade --install moe-inference ./helm/terradev \
  -f clusters/moe-template/helm/values-moe.yaml \
  --set model.id=zai-org/GLM-5-FP8

# Terraform
cd clusters/moe-template/terraform && terraform apply \
  -var="model_id=zai-org/GLM-5-FP8" \
  -var="gpu_count=8"
```

## Directory Structure

```
clusters/moe-template/
├── README.md                       # This file
├── task.yaml                       # Terradev CLI task (parameterized)
├── terraform/
│   ├── main.tf                     # Multi-cloud GPU provisioning
│   ├── variables.tf                # All configurable knobs
│   └── outputs.tf                  # Endpoint + cost outputs
├── k8s/
│   ├── namespace.yaml              # moe-inference namespace
│   ├── deployment.yaml             # Parameterized vLLM/SGLang deployment
│   ├── service.yaml                # LoadBalancer + health checks
│   ├── hpa.yaml                    # GPU-aware autoscaling
│   ├── pdb.yaml                    # Pod disruption budget
│   └── model-cache-pvc.yaml        # NVMe model weight storage
└── helm/
    └── values-moe.yaml             # Helm overrides (parameterized)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | *(required)* | HuggingFace model ID |
| `model_name` | `moe-model` | Served model name |
| `gpu_type` | `H100_SXM` | GPU type (H100_SXM, H200_SXM, A100_SXM_80GB) |
| `gpu_count` | `8` | GPUs per node (must match TP size) |
| `tp_size` | `8` | Tensor parallel degree |
| `backend` | `vllm` | Serving backend (vllm, sglang) |
| `gpu_memory_util` | `0.85` | GPU memory utilization fraction |
| `precision` | `fp8` | Model precision (fp8, bf16, fp16) |
| `disk_size_gb` | `500` | Disk for model weights + KV cache |
| `num_replicas` | `1` | Serving replicas (each = full GPU node) |
| `max_model_len` | `32768` | Maximum sequence length |

## Auto-Applied Cost Optimizations

Every deployment from this template automatically enables vLLM optimizations that reduce your inference costs — **no configuration needed**:

| Optimization | Impact | How |
|---|---|---|
| KV Cache Offloading | Up to 9x throughput | `--kv-connector=offloading` — spills KV cache to CPU DRAM |
| MTP Speculative Decoding | Up to 2.8x generation speed | `--speculative-config.method=mtp` — batch-verified draft tokens |
| Sleep Mode | 18-200x faster than cold restart | `--enable-sleep-mode` — idle models hibernate to CPU RAM |
| Expert Load Balancing | Eliminates GPU hotspots | `--enable-eplb` — runtime expert redistribution |
| DeepEP + DeepGEMM | Lower per-token latency | Optimized MoE all-to-all and GEMM kernels |

These are enabled by default in `k8s/deployment.yaml` and `helm/values-moe.yaml`. To disable any, comment out the relevant flag.

## Multi-LoRA Serving

Serve **N fine-tuned LoRA adapters on one base MoE model** using a single GPU set. Uses vLLM's `fused_moe_lora` kernel (454% higher output tokens/sec, 87% lower TTFT).

```bash
# Hot-load adapters onto a running endpoint
terradev lora add -e http://<endpoint>:8000 -n customer-a -p /adapters/customer-a
terradev lora add -e http://<endpoint>:8000 -n customer-b -p /adapters/customer-b

# Use adapter name as the model in API requests
curl http://<endpoint>:8000/v1/chat/completions \
  -d '{"model": "customer-a", "messages": [{"role": "user", "content": "Hello"}]}'

# Manage adapters
terradev lora list   -e http://<endpoint>:8000
terradev lora remove -e http://<endpoint>:8000 -n customer-b
```

To enable Multi-LoRA in the K8s template, uncomment the `--enable-lora` block in `deployment.yaml` and add your adapter volume mount.

## Provider Cost Estimates (8× H100 SXM)

| Provider | Est. $/hr | Availability |
|----------|-----------|--------------|
| Vast.ai | ~$21.00 | Medium |
| RunPod | ~$23.60 | High |
| Lambda | ~$27.60 | Medium |
| CoreWeave | ~$35.28 | High |
| AWS p5.48xl | ~$98.32 | High |
