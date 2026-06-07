# Terradev CLI v5.2.1

**An imperative command-line-interface for AI workload orchestration.**

![Terradev Demo](https://raw.githubusercontent.com/theoddden/Terradev/main/demo/terradev-demo.gif)

**License: Apache 2.0** - Free and open source for commercial and personal use.

pypi.org/project/terradev-cli/

Terradev is a cross-cloud compute-provisioning CLI that compresses + stages datasets, provisions optimal instances + nodes, and deploys **3-5x faster** than sequential provisioning.

**NOTES ON 5.2.1**

Added two new BYOAPI providers: **Yotta Labs (Shakti Cloud)** and **E2E Networks** — India's leading GPU clouds. Yotta Labs uses a pod-based compute model (similar to RunPod), and E2E Networks is a traditional VM-style hyperscaler that is NSE-listed and MeitY empanelled. Both are BYOAPI: your key, stored locally, never touches a Terradev server.

```bash
terradev configure --provider yottalabs
terradev configure --provider e2enetworks
```

**NOTES ON 5.0.0**

We removed the paywall, open-sourced Terradev, and added Rust accelerators for safe and snappy delivery...

With the Rust DAG orchestrator, the execution graph enforces correct sequencing and idempotency at the runtime level. You or the agent can issue commands freely... the orchestrator ensures they're safe to execute.

218 tools not including subcommand/flags require heavy context. The Rust MCP orchestrator processes tool calls with minimal overhead: deserializing, routing, executing, and responding faster than pure-Python-based MCP servers by an order of magnitude. For an agent running a complex provisioning workflow across 23 cloud providers, that compounds across every tool call in the chain.

## BYOAPI Configuration

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
- **<2 minute spot recovery** with KV cache checkpointing
- **up to 3.6x faster cold starts** with weight streaming
- **Up to 50% cost savings** with MLA-aware VRAM estimation
  
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
Terradev supports 23 GPU cloud providers. Start with one, RunPod is the fastest to set up:

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

echo " Deploying InferX + LoRA Hybrid Inference Service"

# 1. Deploy baseline reserved GPUs for steady traffic
echo " Step 1: Provision reserved baseline capacity"
terradev provision -g H100 -n 2 --parallel 4 \
  --tag baseline-llm \
  --max-price 2.50

BASELINE_IP=$(terradev status --json | jq -r '.[] | select(.tags[] | contains("baseline-llm")) | .ip' | head -1)

# 2. Deploy optimized vLLM with LoRA support on baseline
echo " Step 2: Deploy vLLM with LoRA adapter support"
terradev ml vllm --start \
  --instance-ip $BASELINE_IP \
  --model meta-llama/Llama-2-7b-hf \
  --tp-size 2 \
  --enable-lora \
  --enable-kv-offloading \
  --enable-sleep-mode \
  --port 8000

# 3. Load customer-specific LoRA adapters
echo " Step 3: Load multi-tenant LoRA adapters"
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
echo " Step 4: Configure InferX for serverless burst capacity"
terradev inferx deploy \
  --endpoint burst-llm-api \
  --model-id meta-llama/Llama-2-7b-hf \
  --baseline-endpoint http://$BASELINE_IP:8000 \
  --cold-start-threshold 100 \
  --burst-capacity 10 \
  --failover-strategy active-passive

# 5. Set up intelligent routing with semantic awareness
echo " Step 5: Configure semantic routing for multi-tenant requests"
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
echo " Step 6: Configure warm pool for LoRA adapters"
terradev ml warm-pool --configure \
  --strategy adapter_based \
  --max-warm-models 5 \
  --warm-adapters customer-enterprise-a,customer-internal \
  --idle-eviction-minutes 10 \
  --enable-predictive-warming

# 7. Set up comprehensive monitoring and alerting
echo " Step 7: Deploy monitoring stack"
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
echo " Step 8: Testing complete deployment"
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

echo " Step 9: Deployment summary"
echo " Baseline endpoint: http://$BASELINE_IP:8000"
echo " InferX endpoint: https://inferx.terradev.cloud/burst-llm-api"
echo " LoRA adapters loaded: $(terradev lora list -e http://$BASELINE_IP:8000 --count)"
echo " Semantic routing: Active"
echo " Warm pool: Configured for top adapters"
echo " Monitoring: W&B + Prometheus + Grafana"

# 10. Set up automated LoRA updates
echo " Step 10: Configure automated LoRA adapter updates"
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

echo " InferX + LoRA Hybrid Deployment Complete!"
echo ""
echo " Next Steps:"
echo "1. Monitor performance: terradev monitor --endpoint hybrid-llm --live"
echo "2. Check LoRA performance: terradev lora metrics --endpoint http://$BASELINE_IP:8000"
echo "3. Test failover: terradev inferx failover --endpoint burst-llm-api --test-load 5000"
echo "4. Update adapters: terradev lora update -n customer-enterprise-a -p ./new-adapters/"
```

## Bare Metal GPU Access with IPMI Management (Latitude.sh)

Most GPU clouds give you a virtual machine. You get a slice of hardware, shared kernel paths, and a hypervisor layer between your workload and the GPU. For most ML workloads this is fine. For compliance-sensitive deployments — HIPAA, FedRAMP, financial services, defense contractors — it isn't. Virtualization introduces attestation gaps that auditors reject.

Latitude.sh is the only provider in Terradev's fleet that offers both **bare metal** and **virtual machine** GPU instances from the same API. Bare metal gives you the physical server — dedicated hardware, no virtualization overhead, and IPMI out-of-band management.

```bash
# See both bare metal and VM options side by side
terradev quote -g H100 --provider latitude

# Provision dedicated bare metal with IPMI access
terradev provision --provider latitude --gpu H100 --instance-type bare-metal

# Provision a VM (faster spin-up, slightly lower cost)
terradev provision --provider latitude --gpu H100 --instance-type vm
```

### IPMI: Why It Matters for Enterprise

IPMI (Intelligent Platform Management Interface) gives you out-of-band server management independent of the OS and GPU stack. If a training job deadlocks the kernel, you don't wait for a cloud provider ticket — you power-cycle via IPMI directly. Security teams can verify hardware attestation. Compliance frameworks that require dedicated hardware and physical access controls are satisfied.

```bash
# Instance status includes IPMI access endpoint when bare metal
terradev status --live --provider latitude

# Output includes:
# ipmi_access: true
# ipmi_endpoint: 10.x.x.x
# isolation: bare_metal
```
---

## Local GPU Discovery and Hybrid Compute Pools

Every other GPU orchestration platform assumes you're renting compute. Terradev doesn't.

If you have a GPU in your local machine — a gaming rig, a workstation, a university compute node you have SSH access to — Terradev can discover it, register it into your compute pool, and incorporate it into provisioning decisions alongside cloud providers.

```bash
# Scan local machine for GPUs
terradev local scan

# Output:
# Found 1 local GPU:
#   [0] NVIDIA RTX 4090  24GB  Driver 545.29  Util: 3%  Temp: 42C
#
# Register in pool? [y/N]:

# Register local GPU into your pool
terradev local register --name "workstation-4090"

# Scan a remote machine you have SSH access to
terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa

# View your full compute pool (local + cloud)
terradev local pool
```

Pool output:
```
COMPUTE POOL (4 resources)
workstation-4090    RTX 4090    24GB    local        $0.00/hr   Free
runpod-h100-001     H100        80GB    runpod       $2.49/hr   Running
vastai-a100-002     A100        40GB    vastai        $1.82/hr   Running
lambda-a10g-003     A10G        24GB    lambda_labs  $0.60/hr   Idle
```

### Why This Matters

The `terradev quote` command normally shows cloud provider pricing. With local GPUs registered, the pool includes your own hardware — priced at $0/hr. For a university researcher with a 3090 workstation running overnight jobs, or a startup with a rack of 4090s before they've moved to cloud: Terradev routes workloads to the cheapest available compute, and $0/hr always wins.

```bash
# Get quotes including local pool
terradev quote -g RTX4090 --include-local

# Output:
# GPU       PROVIDER                 $/HR    AVAILABLE
# RTX 4090  local (workstation-4090) $0.00   Yes
# RTX 4090  vastai                   $0.34   Yes
# RTX 4090  runpod                   $0.39   Yes

# Provision to cheapest available — prefers local automatically
terradev provision -g RTX4090 --prefer-local

# Launch training on local GPU with cloud overflow
terradev train --script train.py --pool workstation-4090 --overflow-to-cloud
```

### How Discovery Works

The local scanner uses direct NVML bindings (Rust backend, 5–10x faster than `nvidia-smi` parsing) to introspect every GPU on the target machine:

- GPU model, memory, PCIe bus ID, NUMA affinity
- Current utilization, memory usage, temperature, clock speeds
- Driver version, CUDA version, compute capability
- Multi-GPU topology (NVLink, PCIe switch topology)

Falls back to `nvidia-smi` parsing automatically if the Rust NVML extension isn't available.

```bash
# Detailed hardware report for local GPU
terradev local scan --detailed

# Output includes:
# GPU 0: RTX 4090  VRAM 24GB  PCIe x16  NUMA node 0
#   NVLink: none (single GPU)
#   Compute: 8.9  Driver: 545.29  CUDA: 12.3
#   P-state: P0  Temp: 42C  Power: 45W / 450W TDP
```
---

## Quick Reference
```bash
# Set up cloud provider credentials
terradev configure

# Real-time GPU pricing across 21+ clouds
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

### Troubleshooting Training Workflows

**NCCL Connectivity Problems**
```bash
# Symptoms: Training hangs, NCCL errors, slow communication

# Diagnosis: Check inter-node connectivity
terradev preflight --detailed
terradev execute -i <node-id> -c "nccl_test -b 8G -e 8G -s 1073741824"

# Fix: Re-provision with proper NUMA alignment
terradev provision -g H100 -n 4 --parallel 6 --ensure-numa-alignment
```

**GPU Memory Issues**
```bash
# Symptoms: OOM errors, CUDA out of memory

# Diagnosis: Check memory usage across nodes
terradev monitor --job <job-id> --memory-usage
terradev execute -i <node-id> -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"

# Fix: Reduce batch size or enable gradient checkpointing
terradev train --script train.py --from-provision latest --script-args "--batch-size 16 --gradient-checkpointing"
```

**Dataset Staging Failures**
```bash
# Symptoms: Slow data loading, transfer timeouts

# Diagnosis: Check dataset cache status
terradev stage --status --dataset-id <dataset-id>
terradev stage --list-cached --region us-east-1

# Fix: Re-stage with higher parallelism or compression
terradev stage -d ./my-dataset --target-regions us-east-1 --parallel-streams 64 --compression zstd
```

**FlashOptim Compatibility Issues**
```bash
# Symptoms: FlashOptim fails to apply, training crashes

# Diagnosis: Check FlashOptim compatibility
terradev train-status --job <job-id> | grep flashoptim
terradev preflight --flashoptim-check

# Fix: Disable FlashOptim or adjust configuration
terradev train --script train.py --flashoptim off --from-provision latest
# or with manual configuration
terradev train --script train.py --flashoptim on --flashoptim-optimizer adamw --flashoptim-master-weight-bits 8
```

**Checkpoint Recovery Issues**
```bash
# Symptoms: Can't resume from checkpoint, corrupted checkpoints

# Diagnosis: Verify checkpoint integrity
terradev checkpoint list --job <job-id> --verify
terradev checkpoint validate --checkpoint <checkpoint-path>

# Fix: Create new checkpoint or repair existing
terradev checkpoint save --job <job-id> --force
terradev checkpoint repair --checkpoint <checkpoint-path>
```

**Performance Optimization**

**Slow Training Speed**
```bash
# Diagnose bottlenecks
terradev monitor --job <job-id> --bottleneck-analysis
terradev execute -i <node-id> -c "nvtop --interval 1"

# Common fixes
# 1. Enable mixed precision training
terradev train --script train.py --script-args "--mixed-precision --fp16"

# 2. Optimize data loading
terradev stage --hf-dataset <dataset> --target-regions us-east-1 --preprocess "shuffle,cache"

# 3. Increase parallelism
terradev provision -g H100 -n 8 --parallel 12
```

**Network Bottlenecks**
```bash
# Check network performance between nodes
terradev preflight --network-test
terradev execute -i <node-id> -c "ibstat -v"

# Fixes for RDMA/InfiniBand issues
terradev provision -g H100 -n 4 --ensure-rdma --enable-gpudirect
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Apache 2.0.

## Support

- **Documentation**: [Full User Guide](USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/theoddden/Terradev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theoddden/Terradev/discussions)
