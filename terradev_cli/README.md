# Terradev CLI v6.1.0

**Cross-cloud GPU infrastructure CLI for training, inference, and AI workload orchestration.**

![Terradev Demo](https://raw.githubusercontent.com/theoddden/Terradev/main/demo/terradev-demo.gif)

**License: Apache 2.0** - Free and open source for commercial and personal use.

https://terradev.cloud/

Terradev is a cross-cloud compute control plane for AI workloads, not just a provisioning wrapper. 

Combines quoting, provisioning, topology optimization, training orchestration, inference tuning, and cost analytics in one CLI, with a Rust-accelerated idempotent runtime underneath.

Continued focus on lower cost, faster provisioning, and topology-aware execution with local credential storage.

Model agnostic. Dataset agnostic. GPU agnostic. Provider agnostic. The only thing Terradev is not agnostic about is correctness: it enforces topology, idempotency, and sequencing.

**NOTES ON 5.7.5**

- Expanded Python 3.9 support with a compatibility shim for the MCP SDK, plus bundled `EnterpriseAuthManager` and `SAMLProvider` scaffolding for enterprise SSO.
- Hardened the test suite: provider conformance, Kubernetes CPU parsing, and major-provider error paths are now fully exercised.
- Lowered `requires-python` to `>=3.9` and gated the `mcp` dependency on Python `>=3.10`.

**NOTES ON 5.4.1**

Fixed **credential persistence scope bug** in `terradev configure`:

- **Bug**: `api` variable was only instantiated inside the `else` block (interactive mode), but `api.save_credentials()` was called unconditionally at line 1530. This caused `UnboundLocalError` when using `--provider` flag (single-provider mode), and would have overwritten credentials with empty data in interactive mode.

- **Fix**:
  - Moved `api = TerradevAPI()` to line 952 (before `if provider:` branching) to ensure it's always in scope
  - Removed duplicate `api` instantiation from the `else` block
  - Moved `api.save_credentials()` inside the `else` block (line 1531) - it only runs in interactive mode since single-provider mode saves directly to file at line 1186-1187

- **Impact**: Both `terradev configure --provider <name>` and `terradev configure` (interactive) now correctly persist credentials to `~/.terradev/credentials.json`.

**NOTES ON 5.4.0**

Fixed **credential persistence bug** in `terradev configure` interactive mode:

- **Bug**: Interactive mode (`terradev configure` without `--provider` flag) crashed with `NameError: name 'api' is not defined` at line 1515, preventing credentials from being saved to disk. This caused `terradev quote` to return empty results for all providers.

- **Fix**: 
  - Added `api = TerradevAPI()` initialization in interactive mode
  - Added `api.load_credentials()` after `prompt_for_credentials()` to reload credentials from disk
  - Fixed incorrectly nested Kubernetes configuration prompts (were inside "No providers configured" else block)
  - Restored `api.save_credentials()` to save all credentials (provider + ML integrations + Kubernetes settings)

- **Impact**: Users can now successfully configure providers in interactive mode and retrieve real-time pricing via `terradev quote`.

**NOTES ON 5.3.9**

Added **LoRAX (LoRA eXchange) integration** and **HuggingFace PEFT import** for production-grade multi-LoRA inference serving:

- **LoRAX Service** (`ml_services/lorax_service.py`): Async HTTP client for Predibase LoRAX multi-LoRA inference server that serves thousands of fine-tuned models on a single GPU with dynamic adapter loading, heterogeneous continuous batching, and adapter exchange scheduling.

- **LoRAX CLI commands** (`terradev lora lorax`):
  ```bash
  terradev lora lorax deploy -m mistralai/Mistral-7B-Instruct-v0.1 --docker
  terradev lora lorax test --host localhost --port 8080
  terradev lora lorax list-adapters
  terradev lora lorax load-adapter -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
  terradev lora lorax unload-adapter -a my-adapter
  terradev lora lorax generate -p "What is 2+2?" -a my-adapter
  terradev lora lorax sync-registry
  ```

- **PEFT Import Service** (`ml_services/peft_import_service.py`): Download, validate, and prepare LoRA adapters from HuggingFace using the PEFT library with auto-detection of rank, alpha, and target modules.

- **PEFT CLI commands** (`terradev lora peft`):
  ```bash
  terradev lora peft import -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
  terradev lora peft import -a username/adapter --local-name my-adapter --register --base-model mistralai/Mistral-7B-Instruct-v0.1
  terradev lora peft list
  terradev lora peft validate -p ~/.terradev/peft_adapters/username--adapter
  terradev lora peft delete -a username/adapter
  ```

- **LoRAX Helm Template** (`clusters/lorax-template/helm/`): Production-ready Kubernetes manifests with GPU resource limits, storage configuration.

- **Registry Integration**: One-step import from HuggingFace and automatic registration in Terradev LoRA registry with version tracking, cross-replica sync, and cost attribution.

**NOTES ON 5.3.3**

Added **provider registration and profiling system** for intelligent quirk-aware routing across 23 cloud providers, and registration for custom providers from .yaml import:

- **ProviderProfile schema** (`providers/types.py`): Encodes provider-specific behaviors including API style (REST/GraphQL/JSON:API), authentication type (Bearer/Basic/HMAC/X-Api-Key), rate limits, spot instance support, egress costs, fallback routing, capacity checks, container image pinning, and spot interruption handling.

- **Built-in profiles** (`providers/provider_profiles.py`): Pre-configured profiles for all 23 providers (RunPod, Vast.ai, Lambda Labs, AWS, GCP, Azure, Oracle, Crusoe, CoreWeave, DigitalOcean, Yotta Labs, E2E Networks, FluidStack, Alibaba, OVHcloud, Hetzner, SiliconFlow, TensorDock, Baseten, HuggingFace, Hyperstack, InferX, Latitude).

- **Dynamic registration**: Users can register custom provider profiles programmatically or load from YAML/JSON files for internal clusters or proprietary cloud providers.

- **Profile-aware routing** (`providers/registry.py`): `ProviderRegistry.ranked_providers()` now incorporates provider profiles into scoring, using egress costs, fallback routing preferences, and spot preemption rates for intelligent provider selection.

- **CLI commands** (`terradev providers`): New command group for managing custom provider profiles:
  ```bash
  terradev providers load-profiles ~/.terradev/custom_providers.yaml
  terradev providers list-profiles
  terradev providers show-profile runpod
  terradev providers remove-profile my_custom_provider
  terradev providers export-example -o ~/.terradev/custom_providers.yaml
  ```

**NOTES ON 5.2.1**

Added two new BYOAPI providers: **Yotta Labs (Shakti Cloud)** and **E2E Networks** — India's leading GPU clouds. Yotta Labs uses a pod-based compute model (similar to RunPod), and E2E Networks is a traditional VM-style hyperscaler that is NSE-listed and MeitY empanelled. Both are BYOAPI: your key, stored locally, never touches a Terradev server.

```bash
terradev configure --provider yottalabs
terradev configure --provider e2enetworks
```

**NOTES ON 5.0.0**

We removed the paywall, open-sourced Terradev, and added Rust accelerators for safe and snappy delivery...

With the Rust DAG orchestrator, the execution graph enforces correct sequencing and idempotency at the runtime level. You or the agent can issue commands freely... the orchestrator ensures they're safe to execute.

217 tools not including subcommand/flags require heavy context. The Rust MCP orchestrator processes tool calls with minimal overhead: deserializing, routing, executing, and responding faster than pure-Python-based MCP servers by an order of magnitude. For an agent running a complex provisioning workflow across 23 cloud providers, that compounds across every tool call in the chain.

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

## Inference Provider Gateway

`terradev gateway` is the API gateway for OpenAI/Anthropic-compatible inference. It can also manage inference-only providers directly through provider subcommands.

### Supported inference providers

- `huggingface` - HuggingFace Inference Endpoints
- `baseten` - Baseten model deployments
- `siliconflow` - SiliconFlow model API
- `inferx` - InferX serverless inference

### Typical workflow

```bash
# 1. Configure the provider
terradev gateway huggingface configure --api-key $HF_TOKEN --namespace hf-user
terradev gateway baseten configure --api-key $BASETEN_API_KEY
terradev gateway siliconflow configure --api-key $SILICONFLOW_KEY
terradev gateway inferx configure --api-key $INFERX_KEY

# 2. Deploy a model
terradev gateway huggingface deploy --model meta-llama/Llama-3.1-8B-Instruct --gpu-type A100
terradev gateway siliconflow deploy --model meta-llama/Llama-3.1-8B-Instruct --gpu-type A100
terradev gateway inferx deploy --model meta-llama/Llama-3.1-8B-Instruct

# 3. List and check status
terradev gateway huggingface list
terradev gateway huggingface status <endpoint-id>

# 4. Chat
terradev gateway huggingface chat --model <endpoint-id> --prompt "Hello"

# 5. Delete
terradev gateway huggingface delete <endpoint-id>
```

### `terradev infer` with inference-only providers

`infer deploy` compares quotes from all configured providers (including the inference-only ones). `infer endpoint` deploys the model and registers it with the inference router.

```bash
terradev infer deploy -m meta-llama/Llama-3.1-8B-Instruct
terradev infer endpoint meta-llama/Llama-3.1-8B-Instruct -n my-ep --dry-run
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Apache 2.0.

## Support

- **Issues**: [GitHub Issues](https://github.com/theoddden/Terradev/issues)
