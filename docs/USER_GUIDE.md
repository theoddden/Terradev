# Terradev CLI User Guide — v5.1.5

## Table of Contents

1. [Installation](#installation)
2. [BYOAPI Configuration](#byoapi-configuration)
3. [GPU Pricing & Provisioning](#gpu-pricing--provisioning)
4. [Instance Management](#instance-management)
5. [Distributed Training](#distributed-training)
6. [vLLM Inference Optimization](#vllm-inference-optimization)
7. [MoE Templates & Disaggregated Inference](#moe-templates--disaggregated-inference)
8. [Kubernetes GPU Clusters](#kubernetes-gpu-clusters)
9. [RAG Stack (Qdrant + Phoenix + Guardrails)](#rag-stack)
10. [LoRA Adapters](#lora-adapters)
11. [Cost Analytics](#cost-analytics)
12. [MCP Server (Claude Code)](#mcp-server-claude-code)
13. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Minimal install (core CLI only)
pip install terradev-cli

# Full install with all cloud SDKs and ML integrations
pip install terradev-cli[all]

# Verify installation
terradev --help
```

---

## BYOAPI Configuration

API keys are stored locally at `~/.terradev/credentials.json` and **never sent to Terradev servers**.

```bash
# Interactive setup wizard for any provider
terradev configure --provider runpod
terradev configure --provider vastai
terradev configure --provider aws
terradev configure --provider gcp
terradev configure --provider lambda_labs

# Quick setup with provider-specific instructions
terradev setup runpod --quick
terradev setup vastai --quick
```

**Supported providers (21+):** RunPod, Vast.ai, Lambda Labs, CoreWeave, AWS, GCP, Azure, Hyperstack, TensorDock, Latitude.sh (bare metal + VM), FluidStack, Alibaba Cloud, OVHcloud, Hetzner, SiliconFlow, Paperspace, Crusoe, Lepton, Voltage Park, Genesis Cloud, Nebius.

---

## GPU Pricing & Provisioning

### Real-Time Price Quotes

```bash
# Quote across all configured providers
terradev quote -g H100
terradev quote -g A100
terradev quote -g L40S
terradev quote -g RTX4090

# Include local GPUs in pool
terradev quote -g RTX4090 --include-local

# Filter by provider or region
terradev quote -g H100 --provider runpod
terradev quote -g H100 --region us-east-1
```

Output is sorted cheapest-first with price/hr, provider, region, spot availability, and score.

### Provision with NUMA Topology Optimization

```bash
# Provision with auto-topology (NUMA, RDMA, SR-IOV applied automatically)
terradev provision -g H100 -n 4 --parallel 6

# Preview without provisioning
terradev provision -g A100 -n 2 --dry-run

# Set price ceiling
terradev provision -g A100 --max-price 2.50

# Force NUMA alignment + RDMA
terradev provision -g H100 -n 4 --ensure-numa-alignment --enable-rdma

# Provision via task file (MoE clusters, RAG clusters, etc.)
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3-30B-A3B

# Prefer local GPUs, overflow to cloud
terradev provision -g RTX4090 --prefer-local
```

**What's auto-applied on every provision:**
- NUMA alignment — GPU and NIC forced to the same NUMA node
- GPUDirect RDMA — `nvidia_peermem` loaded, zero-copy GPU-to-GPU
- CPU pinning — static CPU manager policy
- SR-IOV — virtual functions per GPU for isolated RDMA paths
- NCCL tuning — InfiniBand enabled, `GDR_LEVEL=PIX`, `GDR_READ=1`

### Provision + Deploy + Run (One Command)

```bash
# Provision, pull image, run command
terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"

# Keep an inference server alive
terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
```

### Local GPU Discovery

```bash
# Scan local machine
terradev local scan
terradev local scan --detailed

# Scan remote machine via SSH
terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa

# Register GPU into compute pool
terradev local register --name "workstation-4090"

# View full pool (local + cloud)
terradev local pool
```

---

## Instance Management

```bash
# View all running instances and live cost
terradev status --live
terradev status --format json

# Manage lifecycle
terradev manage -i <instance-id> -a stop
terradev manage -i <instance-id> -a start
terradev manage -i <instance-id> -a terminate

# Run a command on a provisioned instance
terradev execute -i <instance-id> -c "nvidia-smi"
terradev execute -i <instance-id> -c "python train.py" --async-exec

# Clean up unused resources
terradev cleanup --dry-run
terradev cleanup --force --older-than 24
```

---

## Distributed Training

### Pre-Flight Validation

Always run preflight before launching training on new nodes:

```bash
# Full validation: GPUs, NCCL, RDMA, drivers
terradev preflight

# With detailed diagnostics
terradev preflight --detailed

# Network-only test
terradev preflight --network-test

# Check FlashOptim compatibility
terradev preflight --flashoptim-check
```

### Launch Training

```bash
# Launch on last provisioned nodes (auto-resolves IPs)
terradev train --script train.py --from-provision latest

# Supports torchrun, DeepSpeed, Accelerate, Megatron
terradev train --script train.py --from-provision latest \
  --script-args "--batch-size 32 --gradient-checkpointing"

# With FlashOptim explicitly on/off (auto-detected by default)
terradev train --script train.py --flashoptim auto   # default
terradev train --script train.py --flashoptim off
terradev train --script train.py --flashoptim on \
  --flashoptim-optimizer adamw --flashoptim-master-weight-bits 8

# Local GPU with cloud overflow
terradev train --script train.py --pool workstation-4090 --overflow-to-cloud
```

**FlashOptim** (Databricks) is auto-injected when `bf16`/`fp16` is detected and total VRAM ≥ 40GB. No flags needed — you just get faster training.

### Monitor and Checkpoint

```bash
# Live GPU utilization and cost
terradev monitor --job <job-id>
terradev monitor --job <job-id> --memory-usage
terradev monitor --job <job-id> --bottleneck-analysis

# Training status
terradev train-status
terradev train-status --job <job-id> | grep flashoptim

# Checkpoint management
terradev checkpoint list --job <job-id>
terradev checkpoint list --job <job-id> --verify
terradev checkpoint save --job <job-id> --force
terradev checkpoint validate --checkpoint <path>
terradev checkpoint repair --checkpoint <path>
```

### Dataset Staging

```bash
# Stage local dataset to cloud regions
terradev stage -d ./my-dataset --target-regions us-east-1 \
  --parallel-streams 64 --compression zstd

# Stage HuggingFace dataset
terradev stage --hf-dataset openai/webtext \
  --target-regions us-east-1 --preprocess "shuffle,cache"

# Check staging status
terradev stage --status --dataset-id <id>
terradev stage --list-cached --region us-east-1
```

---

## vLLM Inference Optimization

### Auto-Tune the 6 Critical Knobs

| Knob | Default | Optimized | Impact |
|------|---------|-----------|--------|
| `max-num-batched-tokens` | 2048 | 16384 | 8x throughput |
| `gpu-memory-utilization` | 0.90 | 0.95 | 5% more VRAM |
| `max-num-seqs` | 256 | 512–2048 | Prevent queuing |
| `enable-prefix-caching` | OFF | ON | Free throughput |
| `enable-chunked-prefill` | OFF | ON | Better prefill |
| CPU cores | 2+GPUs | Optimized | Prevent starvation |

```bash
# Auto-tune from workload profile
terradev vllm auto-optimize -s workload.json -m meta-llama/Llama-3-8B -g 4

# Analyze a running server
terradev vllm analyze -e http://localhost:8000

# Benchmark
terradev vllm benchmark -e http://localhost:8000 -c 10

# Start vLLM directly
terradev ml vllm --start \
  --instance-ip <ip> \
  --model meta-llama/Llama-3-8B \
  --tp-size 2 \
  --enable-lora \
  --enable-kv-offloading \
  --enable-sleep-mode \
  --port 8000
```

---

## MoE Templates & Disaggregated Inference

### Deploy MoE Model (Auto-Optimized)

```bash
# Large MoE model — all optimizations auto-applied
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3-235B-A22B

# Smaller MoE
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3-30B-A3B --set tp_size=4 --set gpu_count=4
```

Auto-applied (zero config):
- **KV cache offloading** — spills to CPU DRAM, up to 9x throughput
- **MTP speculative decoding** — up to 2.8x faster generation
- **Sleep mode** — idle models hibernate to CPU RAM, 18–200x faster than cold restart
- **Expert load balancing** — rebalances routing at runtime
- **LMCache** — distributes KV cache across instances via Redis

### Disaggregated Prefill/Decode

```bash
# Deploy with P/D separation (requires RDMA from Step 4 topology)
terradev ml ray --deploy-pd \
  --model zai-org/GLM-5-FP8 \
  --prefill-tp 8 --decode-tp 1 --decode-dp 24
```

Sticky routing is automatic: once a prefill GPU hands off a KV cache, future requests with the same prefix route to the same decode GPU, avoiding redundant NIXL transfers.

### InferX Serverless Burst

```bash
terradev inferx deploy \
  --endpoint burst-api \
  --model-id meta-llama/Llama-3-8B \
  --cold-start-threshold 100 \
  --burst-capacity 10 \
  --failover-strategy active-passive

terradev inferx status --endpoint burst-api --detailed
terradev inferx failover --endpoint burst-api --test-load 5000
terradev inferx list
terradev inferx usage
```

---

## Kubernetes GPU Clusters

```bash
# Create topology-optimized cluster
terradev k8s create my-cluster --gpu H100 --count 8 --prefer-spot

# Cluster management
terradev k8s list
terradev k8s info my-cluster
terradev k8s destroy my-cluster

# GPU operator and device plugins
terradev k8s gpu-operator-install --cluster my-cluster
terradev k8s device-plugin --cluster my-cluster

# Multi-Instance GPU (MIG) configuration
terradev k8s mig-configure --cluster my-cluster --profile 3g.40gb

# GPU time-slicing
terradev k8s time-slicing --cluster my-cluster --replicas 4

# Deploy full monitoring stack (Prometheus + Grafana + DCGM)
terradev k8s monitoring-stack --cluster my-cluster
```

---

## RAG Stack

### Qdrant Vector DB

```bash
terradev qdrant test --url http://localhost:6333
terradev qdrant collections
terradev qdrant create-collection --name my-docs --model BAAI/bge-large-en-v1.5
terradev qdrant info --collection my-docs
terradev qdrant count --collection my-docs
terradev qdrant k8s --cluster my-cluster
```

### Arize Phoenix (LLM Trace Observability)

```bash
terradev phoenix test --url http://localhost:6006
terradev phoenix projects
terradev phoenix spans --project-id <id> --limit 50
terradev phoenix trace --trace-id <id>
terradev phoenix otel-env
terradev phoenix snippet --framework langchain
terradev phoenix k8s --cluster my-cluster
```

### NeMo Guardrails (Output Safety)

```bash
terradev guardrails test --url http://localhost:8000
terradev guardrails chat --config-id my-config --message "Hello"
terradev guardrails generate-config --type topical --output ./guardrails/
terradev guardrails k8s --cluster my-cluster
```

### Deploy Full RAG Template

```bash
terradev provision --task clusters/rag-template/task.yaml \
  --set model_id=meta-llama/Llama-3-8B \
  --set embedding_model=BAAI/bge-large-en-v1.5 \
  --set enable_phoenix=true \
  --set enable_guardrails=true
```

---

## LoRA Adapters

```bash
# Load adapters onto a running vLLM server
terradev lora add -e http://<ip>:8000 -n my-adapter -p ./adapters/my-adapter
terradev lora add -e http://<ip>:8000 -n customer-a -p ./adapters/customer-a

# List loaded adapters
terradev lora list -e http://<ip>:8000

# Remove adapter
terradev lora remove -e http://<ip>:8000 -n my-adapter

# Check per-adapter metrics
terradev lora metrics -e http://<ip>:8000
```

---

## Cost Analytics

```bash
# Spend over last N days
terradev analytics --days 30

# Find cheaper alternatives for running instances
terradev optimize

# Real-time cost analysis
terradev cost analyze --instance-id <id>
terradev cost simulate --gpu H100 --hours 720
terradev cost budget-optimize --budget 500 --gpu H100

# Price trends
terradev price trends --gpu H100 --days 30
terradev price spot-risk --gpu H100 --provider runpod
```

---

## MCP Server (Claude Code)

Terradev ships a Rust-based MCP server with **168 tools** for Claude Code and any MCP-compatible agent.

```bash
# Install the MCP package
npm install -g terradev-mcp

# Add to Claude Code settings (~/.config/claude/claude_desktop_config.json)
# {
#   "mcpServers": {
#     "terradev": {
#       "command": "terradev-mcp",
#       "args": []
#     }
#   }
# }
```

Claude can then use all 168 tools: provision GPUs, launch training jobs, query prices, manage checkpoints, deploy MoE clusters, configure Qdrant, trace with Phoenix, set up guardrails, manage K8s clusters, and more — all through natural language.

---

## Troubleshooting

### NCCL / Network Issues

```bash
# Check inter-node connectivity
terradev preflight --detailed
terradev preflight --network-test

# Re-provision with explicit RDMA
terradev provision -g H100 -n 4 --ensure-rdma --enable-gpudirect
```

### GPU Memory / OOM

```bash
# Check memory across nodes
terradev monitor --job <job-id> --memory-usage
terradev execute -i <id> -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"

# Fix: gradient checkpointing
terradev train --script train.py --script-args "--batch-size 16 --gradient-checkpointing"
```

### FlashOptim Compatibility

```bash
# Check if FlashOptim is being applied
terradev train-status --job <job-id> | grep flashoptim
terradev preflight --flashoptim-check

# Disable
terradev train --script train.py --flashoptim off
```

### Dataset Staging Failures

```bash
terradev stage --status --dataset-id <id>
# Re-stage with higher parallelism
terradev stage -d ./dataset --target-regions us-east-1 --parallel-streams 64
```

### Checkpoint Recovery

```bash
terradev checkpoint list --job <job-id> --verify
terradev checkpoint repair --checkpoint <path>
terradev checkpoint save --job <job-id> --force
```

### Slow Training

```bash
# Bottleneck analysis
terradev monitor --job <job-id> --bottleneck-analysis

# Common fixes
# 1. Mixed precision
terradev train --script train.py --script-args "--mixed-precision --bf16"
# 2. More GPU parallelism
terradev provision -g H100 -n 8 --parallel 12
```

---

## Support

- **GitHub Issues**: [github.com/theoddden/Terradev/issues](https://github.com/theoddden/Terradev/issues)
- **GitHub Discussions**: [github.com/theoddden/Terradev/discussions](https://github.com/theoddden/Terradev/discussions)
- **Full Command Reference**: [COMPLETE_COMMAND_REFERENCE.md](../terradev_cli/COMPLETE_COMMAND_REFERENCE.md)
