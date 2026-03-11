# Terradev CLI v4.0.11

**Cross-Cloud Compute Optimization Platform with Migration & Evaluation**

![Terradev Demo](https://raw.githubusercontent.com/theoddden/Terradev/main/demo/terradev-demo.gif)

Terradev is a cross-cloud compute-provisioning CLI that compresses + stages datasets, provisions optimal instances + nodes, and deploys **3-5x faster** than sequential provisioning.

## What's New in v4.0.11

**🎯 Production-Grade Automation: Triggers, Environments & Lineage**

The three critical missing pillars that transform Terradev from a CLI tool into an enterprise-grade ML platform:

### 🔄 Event-Driven Triggers (`terradev triggers`)
- **Zero-touch automation**: Dataset lands → auto-train, Model drifts → auto-retrain  
- **Schedule-based**: Cron jobs for weekly evaluations and maintenance
- **Condition-based**: Drift scores, performance thresholds, cost limits
- **19-Provider Support**: Works across all cloud providers
- **Manual override**: Full control when needed

### 🏗️ Environment Promotion (`terradev environments`) 
- **Dev → Staging → Prod**: Proper lifecycle management
- **Approval workflow**: Request → Approve → Execute with audit trail
- **Environment isolation**: Separate artifacts and configurations
- **Promotion history**: Complete audit trail for compliance
- **Automatic lineage**: Links artifacts across environments

### 🔍 Auto Lineage System (`terradev lineage`)
- **Zero manual tagging**: Automatic artifact tracking on every execution
- **Complete provenance**: Data → Model → Deployment chain
- **Execution diffing**: Compare any two pipeline runs
- **Compliance export**: JSON/CSV for auditors and regulators
- **Checkpoint tracing**: Work backwards from any artifact

### 💰 Intelligent Spot/On-Demand Selection
- **Smart auto-selection**: Training → on-demand, Inference → spot
- **Cost transparency**: Real-time savings calculations (60-80%)
- **Manual override**: `--spot` and `--on-demand` flags
- **Safety features**: Automatic state checkpointing and recovery

### 🔍 Model & Endpoint Evaluation (`terradev eval`)
- **Model Evaluation**: `terradev eval --model model.pth --dataset test.json`
- **Endpoint Testing**: `terradev eval --endpoint http://localhost:8000 --metrics latency`
- **Baseline Comparison**: Automatic improvement/regression detection
- **A/B Model Testing**: Side-by-side comparison with winner determination
- **Multiple Metrics**: Accuracy, perplexity, latency, throughput, cost

### 🎯 Complete ML Lifecycle
- **Train → Eval → Deploy**: Full workflow now supported
- **Risk Assessment**: Confidence scoring and migration warnings
- **Cost Optimization**: Multi-hop data transfer routing
- **Production Planning**: Detailed downtime and cost estimates

## Previous Features (v4.0.1)

**Critical Cloud Provider Feature Gaps Implementation**

Revolutionary new features for production ML workloads with spot instance resilience:

### 🚀 MLA-Aware VRAM Estimation
- **DeepSeek V3/R1 & Kimi K2 Support**: 12.5x KV cache compression (2440GB → 195GB)
- **57.3% Cost Savings**: Prevents over-provisioning with accurate MLA calculations
- **GPU Count Optimization**: Intelligent recommendations for large models
- **Model Registry**: Comprehensive MLA architecture flags

### ⚡ Weight Streaming
- **3.6x Faster Cold Starts**: 30-45 minutes → under 3 minutes
- **Parallel Download/Compute**: Async streaming of model layer chunks
- **vLLM/SGLang Integration**: Seamless framework compatibility
- **Multi-Storage Support**: HTTP, S3, GCS, VAST Data backends

### 💾 Preemptible KV Cache Checkpointing
- **<2 Minute Recovery**: Spot interruption handling with state preservation
- **NVMe Serialization**: ~8 second checkpoint creation
- **Cloud Storage Backup**: S3/GCS/VAST Data persistence
- **Zero Data Loss**: Complete failure → brief pause with resume

### 🎯 Real-World Impact
- **60-80% Cost Savings**: Reliable spot instance usage for stateful workloads
- **99.4% Uptime**: Enterprise-grade reliability with cloud-native economics
- **Multi-Cloud Flexibility**: 19 cloud provider integrations
- **Production Ready**: Comprehensive test coverage and validation

### 🎯 Auto-Apply Decision Tree
```bash
# Auto-optimize any model for workload type
terradev sglang optimize deepseek-ai/DeepSeek-V3

# Detect workload from description
terradev sglang detect meta-llama/Llama-2-7b-hf --user-description "Real-time API"

# Multi-replica cache-aware routing
terradev sglang router meta-llama/Llama-2-7b-hf --dp-size 8
```

### 📊 Performance Gains
- **Agentic Chat**: 1.9x throughput with multi-replica, 95-98% GPU utilization
- **Batch Inference**: Maximum tokens/second with pre-compiled CUDA graphs
- **Low Latency**: 30-50% TTFT improvement, 20-40% TPOT improvement
- **MoE Models**: Up to 2x throughput with Two-Batch Overlap
- **Cache-Aware Routing**: 3.8x higher cache hit rate

### 🔧 Hardware Optimization
- **H100/H200**: FlashInfer + FP8 KV cache optimization
- **H20**: FA3 + MoE→QKV→FP8 stacking + swapAB runner
- **GB200 NVL72**: Rack-scale TP + NUMA-aware placement
- **AMD MI300X**: Triton backend + ROCm EPLB tuning

## What's New in v3.7.3

Performance and scalability improvements for enterprise deployments.

### CUDA Graph Optimization with NUMA Awareness

Revolutionary passive CUDA Graph optimization that automatically analyzes and optimizes GPU topology for maximum graph performance:

```bash
# Automatic CUDA Graph optimization - no configuration needed
terradev provision -g H100 -n 4

# NUMA-aware endpoint selection happens automatically
# CUDA Graph compatibility is detected passively
# Warm pool prioritizes graph-compatible models
```

#### Performance Gains:
- 2-5x speedup for CUDA Graph workloads with optimal NUMA topology
- 30-50% bandwidth penalty eliminated through automatic GPU/NIC alignment
- Zero configuration - everything runs passively in the background
- Model-aware optimization - different strategies for transformers vs MoE models

#### NUMA Topology Intelligence
- **PIX (Same PCIe Switch)**: Optimal for CUDA Graphs (1.0 score)
- **PXB (Same Root Complex)**: Very good (0.8 score)
- **PHB (Same NUMA Node)**: Good (0.6 score)
- **SYS (Cross-Socket)**: Poor for graphs (0.3 score)

#### Model-Specific Optimization
- **Transformers**: Highest priority (0.9 base score) - benefit most from graphs
- **CNNs**: Moderate priority (0.7 base score) - benefit moderately
- **MoE Models**: Lower priority (0.4 base score) - dynamic routing challenges
- **Auto-detection**: Model types identified automatically from model IDs

#### Background Optimization
- **Passive Analysis**: Runs automatically every 5 minutes
- **Warm Pool Enhancement**: CUDA Graph models get higher priority
- **Endpoint Selection**: Routes to NUMA-optimal endpoints automatically
- **Performance Tracking**: Monitors graph capture time and replay speedup

## Complete Tutorial

### Step 1: Install Terradev
```bash
pip install terradev-cli
```

For all cloud provider SDKs and ML integrations:
```bash
pip install terradev-cli[all]
```

Verify and list commands:
```bash
terradev --help
```

### Step 2: Configure Your First Cloud Provider
Terradev supports 19 GPU cloud providers. Start with one, RunPod is the fastest to set up:

```bash
terradev setup runpod --quick
```

This shows you where to get your API key. Then configure it:

```bash
terradev configure --provider runpod
```

Paste your API key when prompted. It's stored locally at ~/.terradev/credentials.json, never sent to a Terradev server. Add more providers later:

```bash
terradev configure --provider vastai
terradev configure --provider lambda_labs
terradev configure --provider aws
```

The more providers you configure, the better your price coverage.

### Step 3: Get Real-Time GPU Prices
Check pricing across every provider you've configured:

```bash
terradev quote -g A100
```

Output is a table sorted cheapest-first: price/hour, provider, region, spot vs. on-demand. Try different GPUs:

```bash
terradev quote -g H100
terradev quote -g L40S
terradev quote -g RTX4090
```

### Step 4: Provision
Most clouds hand you GPUs with suboptimal topology by default. Your GPU and NIC end up on different NUMA nodes, RDMA is disabled, and the kubelet Topology Manager is set to none. That's a 30-50% bandwidth penalty on every distributed operation and you'll never see it in nvidia-smi.

When you provision through Terradev, topology optimization is automatic:

```bash
terradev provision -g H100 -n 4 --parallel 6
```

What happens behind the scenes:
- **NUMA alignment** — GPU and NIC forced to the same NUMA node
- **GPUDirect RDMA** — nvidia_peermem loaded, zero-copy GPU-to-GPU transfers
- **CPU pinning** — static CPU manager policy, no core migration
- **SR-IOV** — virtual functions created per GPU for isolated RDMA paths
- **NCCL tuning** — InfiniBand enabled, GDR_LEVEL=PIX, GDR_READ=1

You don't configure any of this. It's applied automatically.

To preview the plan without launching:
```bash
terradev provision -g A100 -n 2 --dry-run
```

To set a price ceiling:
```bash
terradev provision -g A100 --max-price 2.50
```

### Step 5: Run a Workload

**Option A** — Run a command on your provisioned instance:
```bash
terradev execute -i <instance-id> -c "nvidia-smi"
terradev execute -i <instance-id> -c "python train.py"
```

**Option B** — One command that provisions, deploys a container, and runs:
```bash
terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"
```

**Option C** — Keep an inference server alive:
```bash
terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
```

### Step 6: Manage Your Instances
```bash
# See all running instances and current cost
terradev status --live

# Stop (keeps allocation)
terradev manage -i <instance-id> -a stop

# Restart
terradev manage -i <instance-id> -a start

# Terminate and release
terradev manage -i <instance-id> -a terminate
```

### Step 7: Track Costs and Find Savings
```bash
# View spend over the last 30 days
terradev analytics --days 30

# Find cheaper alternatives for running instances
terradev optimize
```

### Step 8: Distributed Training Pipeline
Now that your nodes have correct topology, distributed training actually runs at full bandwidth:

```bash
# Validate GPUs, NCCL, RDMA, and drivers before launching
terradev preflight

# Launch training on the nodes you just provisioned
terradev train --script train.py --from-provision latest

# Watch GPU utilization and cost in real time
terradev monitor --job my-job

# Check status
terradev train-status

# 6. List checkpoints when done
terradev checkpoint list --job my-job
```

The `--from-provision latest` flag auto-resolves IPs from your last provision command. Supports torchrun, DeepSpeed, Accelerate, and Megatron.

### Step 9: Optimize vLLM Inference (The 6 Knobs)
If you're serving a model with vLLM, there are 6 settings most teams leave at defaults — each one costs throughput:

| Knob | Default | Optimized | Impact |
|------|---------|-----------|--------|
| max-num-batched-tokens | 2048 | 16384 | 8x throughput |
| gpu-memory-utilization | 0.90 | 0.95 | 5% more VRAM |
| max-num-seqs | 256/1024 | 512-2048 | Prevent queuing |
| enable-prefix-caching | OFF | ON | Free throughput win |
| enable-chunked-prefill | OFF | ON | Better prefill |
| CPU Cores | 2 + #GPUs | Optimized | Prevent starvation |

Auto-tune all six from your workload profile:
```bash
terradev vllm auto-optimize -s workload.json -m meta-llama/Llama-2-7b-hf -g 4
```

Or analyze a running server:
```bash
terradev vllm analyze -e http://localhost:8000
```

Benchmark:
```bash
terradev vllm benchmark -e http://localhost:8000 -c 10
```

### Step 10: Deploy a MoE Model with Auto-Applied Optimizations
For large Mixture-of-Experts models (GLM-5, Qwen 3.5, DeepSeek V4), Terradev's MoE templates include every optimization auto-applied — KV cache offloading, speculative decoding, sleep mode, expert load balancing:

```bash
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-397B-A17B
```

Or a smaller model:
```bash
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-122B-A10B --set tp_size=4 --set gpu_count=4
```

What's auto-applied (no flags needed):
- **KV cache offloading** — spills to CPU DRAM, up to 9x throughput
- **MTP speculative decoding** — up to 2.8x faster generation
- **Sleep mode** — idle models hibernate to CPU RAM, 18-200x faster than cold restart
- **Expert load balancing** — rebalances routing at runtime
- **LMCache** — distributes KV cache across instances via Redis

### Step 11: Disaggregated Prefill/Decode (Advanced)
This separates inference into two GPU pools optimized for each phase:

- **Prefill (compute-bound)** — processes input prompt, wants high FLOPS
- **Decode (memory-bound)** — generates tokens, wants high HBM bandwidth

The KV cache transfers between them via NIXL — zero-copy GPU-to-GPU over RDMA. This is why getting the NUMA topology right in Step 4 matters: NIXL only runs at full speed when the GPU and NIC share a PCIe switch.

```bash
terradev ml ray --deploy-pd \
  --model zai-org/GLM-5-FP8 \
  --prefill-tp 8 --decode-tp 1 --decode-dp 24
```

Terradev's inference router automatically uses sticky routing. Once a prefill GPU hands off a KV cache to a decode GPU, future requests with the same prefix go to that same decode GPU, avoiding redundant transfers.

### Step 12: Create a Kubernetes GPU Cluster
For production, create a topology-optimized K8s cluster:

```bash
terradev k8s create my-cluster --gpu H100 --count 8 --prefer-spot
```

This auto-configures Karpenter NodePools with NUMA-aligned kubelet Topology Manager, GPUDirect RDMA, and PCIe locality enforcement.

```bash
# List clusters
terradev k8s list

# Get cluster info
terradev k8s info my-cluster

# Tear down
terradev k8s destroy my-cluster
```

## Why This Order Matters
Each step builds on the one before it:

- **Step 4**: NUMA / RDMA / SR-IOV topology ← foundation
- **Step 8**: Distributed training at full BW ← depends on topology
- **Step 9**: vLLM knob tuning ← depends on correct memory layout
- **Step 10**: KV cache offloading + sleep mode ← depends on CPU bus not saturated
- **Step 11**: Disaggregated P/D ← depends on RDMA for KV transfer

If the provisioning layer is wrong, every optimization above it underperforms. A disaggregated P/D setup with a cross-NUMA KV transfer is slower than a monolithic setup with correct topology.

Terradev handles the foundation automatically so the rest of the stack works the way it's supposed to.

## Complete Workflow Examples

### Example 1: LLM Inference Service
```bash
#!/bin/bash

# Complete LLM deployment workflow

# 1. Find cheapest GPU
terradev quote -g A100 --quick
# 2. Provision with auto-optimization
terradev provision -g A100 -n 2 --parallel 4
# 3. Deploy optimized vLLM
terradev ml vllm --start --instance-ip $(terradev status --json | jq -r '.[0].ip') --model meta-llama/Llama-2-7b-hf --tp-size 2
# 4. Set up monitoring
terradev monitor --endpoint llama-api --live
# 5. Add customer adapter
terradev lora add -e http://$(terradev status --json | jq -r '.[0].ip'):8000 -n customer-a -p ./adapters/customer-a
```

### Example 2: MoE Model Production Deployment
```bash
#!/bin/bash

# GLM-5 production deployment

# 1. Deploy MoE cluster
terradev provision --task clusters/moe-template/task.yaml --set model_id=zai-org/GLM-5-FP8 --set tp_size=8
# 2. Deploy monitoring
terradev k8s monitoring-stack --cluster glm-5-cluster
# 3. Set up warm pool for bursty traffic
terradev ml warm-pool --configure --strategy traffic_based --max-warm-models 5 --endpoint glm-5-api
# 4. Test failover
terradev inferx failover --endpoint glm-5-api --test-load 5000
```

### Example 3: InferX + LoRA Hybrid Deployment (Production Multi-Tenant)
```bash
#!/bin/bash

# Production deployment with cold start failover and multi-tenant LoRA adapters

echo "🚀 Deploying InferX + LoRA Hybrid Inference Service"

# 1. Deploy baseline reserved GPUs for steady traffic
echo "📍 Step 1: Provision reserved baseline capacity"
terradev provision -g H100 -n 2 --parallel 4 \
  --tag baseline-llm \
  --max-price 2.50

BASELINE_IP=$(terradev status --json | jq -r '.[] | select(.tags[] | contains("baseline-llm")) | .ip' | head -1)

# 2. Deploy optimized vLLM with LoRA support on baseline
echo "📍 Step 2: Deploy vLLM with LoRA adapter support"
terradev ml vllm --start \
  --instance-ip $BASELINE_IP \
  --model meta-llama/Llama-2-7b-hf \
  --tp-size 2 \
  --enable-lora \
  --enable-kv-offloading \
  --enable-sleep-mode \
  --port 8000

# 3. Load customer-specific LoRA adapters
echo "📍 Step 3: Load multi-tenant LoRA adapters"
terradev lora add -e http://$BASELINE_IP:8000 \
  -n customer-enterprise-a \
  -p ./adapters/customer-enterprise-a

terradev lora add -e http://$BASELINE_IP:8000 \
  -n customer-startup-b \
  -p ./adapters/customer-startup-b

terradev lora add -e http://$BASELINE_IP:8000 \
  -n customer-internal \
  -p ./adapters/customer-internal

# 4. Configure InferX for cold start and burst handling
echo "📍 Step 4: Configure InferX for serverless burst capacity"
terradev inferx deploy \
  --endpoint burst-llm-api \
  --model-id meta-llama/Llama-2-7b-hf \
  --baseline-endpoint http://$BASELINE_IP:8000 \
  --cold-start-threshold 100 \
  --burst-capacity 10 \
  --failover-strategy active-passive

# 5. Set up intelligent routing with semantic awareness
echo "📍 Step 5: Configure semantic routing for multi-tenant requests"
cat > routing-config.yaml << EOF
rules:
  - name: "enterprise_customers"
    condition: "header:x-customer-id == 'enterprise-a'"
    route_to: "baseline"
    lora_adapter: "customer-enterprise-a"
    strategy: "latency"

  - name: "startup_customers" 
    condition: "header:x-customer-id == 'startup-b'"
    route_to: "baseline"
    lora_adapter: "customer-startup-b"
    strategy: "cost"

  - name: "internal_workloads"
    condition: "header:x-api-key starts_with 'internal_'"
    route_to: "baseline"
    lora_adapter: "customer-internal"
    strategy: "throughput"

  - name: "burst_traffic"
    condition: "request_rate > 50"
    route_to: "inferx"
    strategy: "auto-scale"

  - name: "fallback"
    condition: "default"
    route_to: "baseline"
    lora_adapter: "customer-internal"
    strategy: "round-robin"
EOF

terradev semantic-router --deploy --config routing-config.yaml

# 6. Configure warm pool for frequently used adapters
echo "📍 Step 6: Configure warm pool for LoRA adapters"
terradev ml warm-pool --configure \
  --strategy adapter_based \
  --max-warm-models 5 \
  --warm-adapters customer-enterprise-a,customer-internal \
  --idle-eviction-minutes 10 \
  --enable-predictive-warming

# 7. Set up comprehensive monitoring and alerting
echo "📍 Step 7: Deploy monitoring stack"
terradev k8s monitoring-stack --cluster production

# Configure W&B for ML observability
terradev ml wandb --setup-alerts \
  --endpoint http://$BASELINE_IP:8000 \
  --metric-thresholds "latency_p95<2000,throughput>100,gpu_utilization>80" \
  --alert-channels slack,email

# Configure InferX-specific monitoring
terradev inferx status --endpoint burst-llm-api --detailed
terradev inferx failover --endpoint burst-llm-api --test-load 1000

# 8. Test the complete setup
echo "📍 Step 8: Testing complete deployment"
echo "Testing baseline endpoint with LoRA..."
curl -X POST http://$BASELINE_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-customer-id: enterprise-a" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Hello from enterprise customer!"}],
    "max_tokens": 100
  }'

echo "Testing InferX burst endpoint..."
curl -X POST https://inferx.terradev.cloud/burst-llm-api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INFERX_API_KEY" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf", 
    "messages": [{"role": "user", "content": "Hello from burst traffic!"}],
    "max_tokens": 100
  }'

echo "📍 Step 9: Deployment summary"
echo "✅ Baseline endpoint: http://$BASELINE_IP:8000"
echo "✅ InferX endpoint: https://inferx.terradev.cloud/burst-llm-api"
echo "✅ LoRA adapters loaded: $(terradev lora list -e http://$BASELINE_IP:8000 --count)"
echo "✅ Semantic routing: Active"
echo "✅ Warm pool: Configured for top adapters"
echo "✅ Monitoring: W&B + Prometheus + Grafana"

# 10. Set up automated LoRA updates
echo "📍 Step 10: Configure automated LoRA adapter updates"
cat > lora-update-config.yaml << EOF
adapters:
  - name: "customer-enterprise-a"
    path: "./adapters/customer-enterprise-a"
    update_strategy: "rolling"
    health_check: true
    rollback_on_failure: true
    
  - name: "customer-startup-b"
    path: "./adapters/customer-startup-b" 
    update_strategy: "blue_green"
    health_check: true
    rollback_on_failure: true

monitoring:
  update_frequency: "hourly"
  health_check_timeout: "30s"
  rollback_threshold: "error_rate > 0.05"
EOF

terradev lora auto-update --config lora-update-config.yaml

echo "🎉 InferX + LoRA Hybrid Deployment Complete!"
echo ""
echo "📊 Next Steps:"
echo "1. Monitor performance: terradev monitor --endpoint hybrid-llm --live"
echo "2. Check LoRA performance: terradev lora metrics --endpoint http://$BASELINE_IP:8000"
echo "3. Test failover: terradev inferx failover --endpoint burst-llm-api --test-load 5000"
echo "4. Update adapters: terradev lora update -n customer-enterprise-a -p ./new-adapters/"

## Quick Reference
```bash
# Set up cloud provider credentials
terradev configure

# Real-time GPU pricing across up to 19 clouds
terradev quote -g H100 

# Provision with auto topology optimization
terradev provision -g H100 -n 4

# Provision + deploy + run in one command
terradev run --gpu A100 --image ...

# View running instances and costs
terradev status --live

# Launch training on provisioned nodes
terradev train --from-provision latest

# Auto-tune 6 critical vLLM knobs
terradev vllm auto-optimize

# Topology-optimized Kubernetes cluster
terradev k8s create

# Cost analytics
terradev analytics --days 30

# Find cheaper alternatives
terradev optimize
```

## Features

- **19 Cloud Providers**: RunPod, VastAI, Lambda Labs, AWS, GCP, Azure, Oracle, and more
- **Automatic Topology Optimization**: NUMA alignment, RDMA, CPU pinning
- **vLLM Auto-Optimization**: 6 critical knobs tuned automatically
- **MoE Model Support**: KV cache offloading, speculative decoding, sleep mode
- **Distributed Training**: torchrun, DeepSpeed, Accelerate, Megatron support
- **Kubernetes Integration**: Topology-optimized GPU clusters
- **Cost Analytics**: Real-time cost tracking and optimization recommendations
- **GitOps Automation**: Production-ready workflows with ArgoCD/Flux
- **CUDA Graph Optimization**: Passive NUMA-aware graph performance optimization

## Installation

```bash
# Basic installation
pip install terradev-cli

# With all cloud provider SDKs
pip install terradev-cli[all]

# Individual provider support
pip install terradev-cli[aws]      # AWS
pip install terradev-cli[gcp]      # Google Cloud
pip install terradev-cli[azure]    # Azure
pip install terradev-cli[hf]       # HuggingFace Spaces
```

## Configuration

Your API keys are stored locally at ~/.terradev/credentials.json and never sent to Terradev servers.

```bash
# Configure multiple providers
terradev configure --provider runpod
terradev configure --provider vastai
terradev configure --provider aws
terradev configure --provider gcp
```

## Performance

- **2-8x throughput improvements** with vLLM optimization
- **30-50% bandwidth penalty eliminated** with NUMA topology
- **2-5x CUDA Graph speedup** with optimal topology
- **Up to 90% cost savings** with automatic provider switching

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

BUSL 1.1 License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Full User Guide](USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/theoddden/Terradev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theoddden/Terradev/discussions)
- **Community**: [Discord Server](https://discord.gg/terradev)
