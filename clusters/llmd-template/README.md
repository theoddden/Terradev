# llm-d Cluster Template

Terradev cluster template for **llm-d** (CNCF Sandbox, March 2026) - Kubernetes-native distributed LLM inference with KServe, Gateway API Inference Extension, and vLLM.

## Why llm-d?

llm-d solves the disaggregated prefill/decode problem natively in Kubernetes:
- **KServe LLMInferenceService** - Standard CRD for model serving
- **Gateway API Inference Extension** - HTTP routing and load balancing
- **LeaderWorkerSet** - Multi-pod orchestration for distributed models
- **NIXL-aware KV transfer** - Zero-copy GPU-to-GPU over RDMA
- **NUMA-aware scheduling** - Integrates with Terradev's topology layer

## Usage

```bash
# Deploy with vLLM backend
terradev provision --task clusters/llmd-template/task.yaml \
  --set model_id=meta-llama/Llama-2-7b-hf \
  --set tp_size=2 \
  --set gpu_count=2

# Deploy with NVIDIA Dynamo orchestration
terradev provision --task clusters/llmd-template/task.yaml \
  --set model_id=meta-llama/Llama-2-7b-hf \
  --set backend=dynamo \
  --set tp_size=2
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    KServe Control Plane                  │
│  (LLMInferenceService + LeaderWorkerSet + Gateway API)  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Terradev NUMA-Optimized Nodes               │
│  (PIX/PXB/PHB topology, GPUDirect RDMA, SR-IOV)         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              vLLM / SGLang / TRT-LLM Engines            │
│  (Disaggregated prefill/decode, NIXL KV transfer)        │
└─────────────────────────────────────────────────────────┘
```

## Comparison to MoE Template

| Feature | moe-template | llmd-template |
|---------|--------------|---------------|
| Orchestration | Manual shell scripts | KServe CRDs |
| Scaling | Manual | KEDA + Gateway API |
| Routing | Custom | Gateway API Inference Extension |
| Multi-node | Ray | LeaderWorkerSet |
| NUMA integration | Yes | Yes (inherited) |

## Backend Options

- **vllm** (default) - High-throughput inference engine
- **sglang** - Workload-specific optimizations
- **dynamo** - NVIDIA Dynamo orchestration layer
- **tensorrt_llm** - NVIDIA TensorRT-LLM engine
