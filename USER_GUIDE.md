# Terradev User Guide

## What is Terradev?

Terradev is a command line tool that finds the cheapest GPU across 19 cloud providers and provisions it for you in one command. It queries prices in real time, runs those queries in parallel, and picks the lowest cost option automatically. You keep your own API keys on your own machine. Terradev never proxies, stores, or transmits your credentials through a third party.

The problem it solves is straightforward. GPU prices for the same hardware vary by 3x or more across providers. Checking each dashboard manually takes time, and you almost always overpay. Terradev eliminates that.

## Installing

```bash
pip install terradev-cli
```

If you want HuggingFace Spaces deployment support:

```bash
pip install terradev-cli[hf]
```

If you want every provider SDK and ML service included:

```bash
pip install terradev-cli[all]
```

Requires Python 3.9 or later.

## First Run

The first time you run any Terradev command, it walks you through an interactive onboarding flow. It will ask you to paste API keys for each cloud provider you want to use. You can start with just one (RunPod is the easiest) and add more later.

If you want to skip onboarding and come back to it:

```bash
terradev configure --provider runpod
```

You can rerun onboarding at any time:

```bash
terradev onboarding --force
```

## Setting Up Providers

Terradev supports 19 cloud providers. You do not need all of them. Configure whichever ones you have accounts with.

**GPU Cloud Providers:**
- RunPod
- Vast.ai
- AWS
- Google Cloud (GCP)
- Azure
- Lambda Labs
- CoreWeave
- TensorDock
- Oracle Cloud
- Crusoe Cloud
- DigitalOcean
- HyperStack
- Alibaba Cloud
- OVHcloud
- FluidStack
- Hetzner
- SiliconFlow
- Baseten
- HuggingFace

To configure any provider:

```bash
terradev configure --provider runpod
terradev configure --provider aws
terradev configure --provider vastai
```

Each provider will prompt you for its specific credentials (API key, access key, project ID, etc). These are saved locally at `~/.terradev/credentials.json`. They never leave your machine.

To see quick setup instructions for a provider before you configure it:

```bash
terradev setup runpod --quick
terradev setup aws --quick
```

## Getting GPU Prices

The `quote` command checks real time pricing across every provider you have configured:

```bash
terradev quote -g A100
terradev quote -g H100
terradev quote -g RTX4090
```

Available GPU types include: H100, H200, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100.

The output shows price per hour, provider, region, and whether it is spot or on demand pricing, sorted from cheapest to most expensive.

To immediately provision the cheapest result:

```bash
terradev quote -g A100 --quick
```

## Provisioning GPUs

To provision the cheapest available instance:

```bash
terradev provision -g A100
```

To provision multiple GPUs in parallel across clouds:

```bash
terradev provision -g H100 -n 4 --parallel 6
```

To see the plan without actually launching anything:

```bash
terradev provision -g A100 -n 2 --dry-run
```

To set a price ceiling:

```bash
terradev provision -g A100 --max-price 2.50
```

## Managing Instances

Once instances are running, you can manage them:

```bash
# See all running instances and total cost
terradev status --live

# Stop an instance (keeps it allocated)
terradev manage -i <instance-id> -a stop

# Restart a stopped instance
terradev manage -i <instance-id> -a start

# Terminate and release the instance
terradev manage -i <instance-id> -a terminate
```

## Running Commands on Instances

Execute commands directly on your provisioned instances:

```bash
terradev execute -i <instance-id> -c "python train.py"
terradev execute -i <instance-id> -c "nvidia-smi"
```

## One Command Workloads

The `run` command combines provisioning, container deployment, and execution into a single step:

```bash
# Provision a GPU, pull a container, and run a training script
terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"

# Keep an inference server running
terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
```

## Dataset Staging

Stage datasets near your compute to avoid slow transfers during training:

```bash
terradev stage -d ./my-dataset --target-regions us-east-1,eu-west-1
```

This compresses, chunks, and uploads your data to storage near the target regions so your instances can pull it quickly.

## Cost Analytics and Optimization

View your spending over time:

```bash
terradev analytics --days 30
```

Find cheaper alternatives for instances that are currently running:

```bash
terradev optimize
```

## Training Pipeline

Terradev includes a full distributed training pipeline. After provisioning, you can launch training across your GPU nodes:

```bash
# Provision nodes
terradev provision -g H100 -n 4 --parallel 6

# Launch training using nodes from the last provision
terradev train --script train.py --from-provision latest

# Check the status of all training jobs
terradev train-status

# Watch GPU utilization, memory, and cost in real time
terradev monitor --job my-job

# List checkpoints
terradev checkpoint list --job my-job

# Manually save a checkpoint
terradev checkpoint save --job my-job
```

The training orchestrator supports `torchrun`, `deepspeed`, `accelerate`, and `megatron` as backends. It handles multi node coordination automatically.

Before launching a training job, you can validate that your nodes are ready:

```bash
terradev preflight
```

This checks GPU availability, NCCL, RDMA, and driver versions across all nodes.

## Inference and Model Serving

### InferX Serverless Platform

Deploy models for serving with cold starts under 2 seconds:

```bash
# Deploy a model
terradev inferx deploy --endpoint my-llama-api --model-id meta-llama/Llama-2-7b-hf --hardware a10g

# Check endpoint health
terradev inferx status --endpoint my-llama-api

# Run failover tests
terradev inferx failover --endpoint my-llama-api --test-load 1000

# Get cost analysis
terradev inferx cost-analysis --days 30 --endpoint my-llama-api
```

### HuggingFace Spaces

Deploy any HuggingFace model to Spaces with one command:

```bash
# LLM template
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Embedding model template
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 --template embedding

# Image model template
terradev hf-space my-image --model-id runwayml/stable-diffusion-v1-5 --template image

# Custom hardware and SDK
terradev hf-space my-model --model-id microsoft/DialoGPT-medium --hardware a10g-large --sdk gradio --private
```

Requires `pip install terradev-cli[hf]` and the `HF_TOKEN` environment variable to be set.

### LoRA Adapter Management

If you are running a vLLM endpoint, you can hot load LoRA adapters onto it without restarting the server:

```bash
# Load an adapter
terradev lora add -e http://<endpoint>:8000 -n customer-a -p /adapters/customer-a

# List loaded adapters
terradev lora list -e http://<endpoint>:8000

# Remove an adapter
terradev lora remove -e http://<endpoint>:8000 -n customer-b
```

Each adapter becomes a model name in the OpenAI compatible API. Clients just specify the adapter name in their requests:

```bash
curl http://<endpoint>:8000/v1/chat/completions \
  -d '{"model": "customer-a", "messages": [{"role": "user", "content": "Hello"}]}'
```

## MoE Model Deployment

Terradev includes production ready cluster templates optimized for Mixture of Experts models. These are the dominant architecture for every major 2026 release (GLM 5, Qwen 3.5, Mistral Large 3, DeepSeek V4, Llama 5).

### Quick Deploy

```bash
# Deploy GLM 5
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=zai-org/GLM-5-FP8 --set tp_size=8

# Deploy Qwen 3.5
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-397B-A17B

# Smaller model on fewer GPUs
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-122B-A10B --set tp_size=4 --set gpu_count=4
```

### What Gets Auto Applied

Every MoE deployment includes these optimizations by default, with no configuration needed:

- **KV Cache Offloading** spills KV cache to CPU DRAM so the GPU never has to recompute prefills. Up to 9x throughput improvement.
- **MTP Speculative Decoding** uses a small draft model to predict tokens that the full model verifies in batch. Up to 2.8x faster generation.
- **Sleep Mode** hibernates idle models to CPU RAM instead of holding GPU memory. Waking up is 18 to 200x faster than a cold restart.
- **Expert Load Balancing** rebalances MoE expert routing at runtime based on actual traffic. Eliminates GPU hotspots.
- **DeepEP and DeepGEMM** are optimized all to all and GEMM kernels for MoE expert computation. Lowers per token latency.

### Kubernetes Deployment

You can also deploy MoE templates directly to Kubernetes:

```bash
kubectl apply -f clusters/moe-template/k8s/
```

Or with Helm:

```bash
helm upgrade --install moe-inference ./helm/terradev \
  -f clusters/moe-template/helm/values-moe.yaml \
  --set model.id=zai-org/GLM-5-FP8
```

## Kubernetes GPU Clusters

Create production Kubernetes clusters with GPU nodes:

```bash
# Create a multi cloud cluster
terradev k8s create my-cluster --gpu H100 --count 8 --multi-cloud --prefer-spot

# List your clusters
terradev k8s list

# Get cluster details
terradev k8s info my-cluster

# Tear down a cluster
terradev k8s destroy my-cluster
```

### Topology Optimization

When you create a cluster, Terradev automatically configures GPU topology behind the scenes:

- NUMA alignment via kubelet Topology Manager in restricted mode
- CPU pinning with static CPU manager policy
- GPUDirect RDMA with nvidia_peermem
- SR IOV with VF per GPU pairing for multi node clusters
- NCCL tuning (IB enabled, GDR_LEVEL=PIX, GDR_READ=1)
- PCIe locality enforcement so GPU NIC pairs share the same NUMA node
- Karpenter NodePool configuration for topology aware GPU scheduling

Without this, Kubernetes randomly assigns GPUs and NICs across NUMA nodes and PCIe switches. A bad pairing can cut RDMA bandwidth by 30 to 50%. Terradev prevents this automatically.

## Ray Serve and Expert Parallelism

For serving 600B+ MoE models at scale, Terradev supports Wide Expert Parallelism, disaggregated Prefill/Decode, and NIXL KV cache transfer:

```bash
# Wide EP across 32 GPUs
terradev ml ray --deploy-wide-ep \
  --model zai-org/GLM-5-FP8 \
  --tp-size 1 --dp-size 32

# Disaggregated Prefill/Decode with NIXL
terradev ml ray --deploy-pd \
  --model zai-org/GLM-5-FP8 \
  --prefill-tp 8 --decode-tp 1 --decode-dp 24

# SGLang with Expert Parallelism
terradev ml sglang --start --instance-ip <IP> \
  --model zai-org/GLM-5-FP8 \
  --tp-size 1 --dp-size 8 \
  --enable-expert-parallel --enable-eplb --enable-dbo
```

Expert Parallelism distributes MoE experts across GPUs, which lets you serve a 744B parameter model on 8 GPUs where pure tensor parallelism would run out of memory.

## GitOps

Terradev can manage GitOps workflows with ArgoCD or Flux CD:

```bash
# Initialize a GitOps repository
terradev gitops init --provider github --repo my-org/infra --tool argocd --cluster production

# Bootstrap the tool onto your cluster
terradev gitops bootstrap --tool argocd --cluster production

# Sync your cluster with the Git repository
terradev gitops sync --cluster production --environment prod

# Validate before applying
terradev gitops validate --dry-run --cluster production
```

Supports GitHub, GitLab, Bitbucket, and Azure DevOps. Includes automated repository structure, policy templates (Gatekeeper/Kyverno), multi environment support, and resource quota management.

## Manifest Cache and Rollback

Track and roll back deployments:

```bash
# Provision with manifest tracking
terradev up --job my-training --gpu-type A100 --gpu-count 4

# Detect and fix drift
terradev up --job my-training --fix-drift

# Roll back to a previous version
terradev rollback my-training@v2

# List all manifests for a job
terradev manifests --job my-training
```

## Observability Integrations

Terradev connects to your existing ML tools. All data flows directly from your instances to your services.

### Weights and Biases

```bash
terradev configure --provider wandb --api-key YOUR_KEY --dashboard-enabled true
terradev ml wandb --test
```

Auto injects `WANDB_*` environment variables into provisioned containers.

### Prometheus and Grafana

```bash
terradev configure --provider prometheus --api-key PUSHGATEWAY_URL
terradev integrations --export-grafana
```

Pushes provision and termination metrics to your Pushgateway. Exports a ready to import Grafana dashboard.

### MLflow

```bash
terradev configure --provider mlflow
terradev ml mlflow --list-experiments
```

### LangSmith

```bash
terradev configure --provider langsmith
terradev ml langchain --create-workflow my-workflow
```

### DVC

```bash
terradev configure --provider dvc
```

All integrations follow the same BYOAPI model. Your keys stay on your machine.

## MCP Server (Claude Code / Windsurf)

You can use Terradev from AI coding assistants that support MCP:

```bash
npm install -g terradev-mcp
```

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "terradev": {
      "command": "terradev-mcp"
    }
  }
}
```

This gives your AI assistant access to all Terradev functionality: quoting, provisioning, cluster management, inference deployment, and cost analytics.

## Jupyter and Notebooks

```bash
pip install terradev-jupyter
```

Then in a notebook:

```python
%load_ext terradev_jupyter

%terradev quote -g A100
%terradev provision -g H100 --dry-run
```

## GitHub Actions

```yaml
- uses: theodden/terradev-action@v1
  with:
    gpu-type: A100
    max-price: "1.50"
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

## Pricing Tiers

Terradev itself has four tiers:

**Research (Free)**
10 provisions per month. 1 concurrent instance. 1 user seat. Access to all 19 providers. Cost tracking, dataset staging, and basic egress optimization included.

**Research+ ($49.99/month)**
100 provisions per month. 8 concurrent instances. 1 user seat. Full egress optimization. Inference endpoints.

**Enterprise ($299.99/month)**
Unlimited provisions. 32 concurrent instances. 5 user seats. All providers with priority support. SLA guarantee. Full provenance and audit trail.

**Enterprise+ ($0.09 per GPU hour)**
Unlimited everything. Minimum commitment of 32 GPUs. Metered billing through Stripe. Fleet management dashboard. Dedicated support.

To upgrade:

```bash
terradev upgrade --tier research_plus
```

This opens a Stripe checkout page. After payment:

```bash
terradev upgrade --activate --email you@example.com
```

## Credential Security

Your API keys are stored at `~/.terradev/credentials.json` on your local machine. Every API call goes directly from your machine to the cloud provider. There is no middleman, no shared account, no markup on provider pricing.

This means:

- No third party holds your cloud keys
- If you stop using Terradev, your cloud accounts are untouched
- Compatible with SOC2, HIPAA, and security policies that prohibit sharing credentials with SaaS vendors
- Every provision is logged locally with provider, cost, and timestamp

## Command Reference

| Command | What it does |
|---------|-------------|
| `terradev configure` | Set up API credentials for a provider |
| `terradev setup` | Show setup instructions for a provider |
| `terradev quote` | Get real time GPU prices across all clouds |
| `terradev provision` | Provision instances with multi cloud arbitrage |
| `terradev manage` | Stop, start, or terminate instances |
| `terradev status` | View all instances and cost summary |
| `terradev execute` | Run commands on provisioned instances |
| `terradev run` | Provision + deploy container + execute in one command |
| `terradev stage` | Compress and stage datasets near compute |
| `terradev analytics` | Cost analytics with daily spend trends |
| `terradev optimize` | Find cheaper alternatives for running instances |
| `terradev train` | Launch distributed training |
| `terradev train-status` | Check training job status |
| `terradev monitor` | Real time GPU metrics |
| `terradev checkpoint` | List or save training checkpoints |
| `terradev preflight` | Validate nodes before training |
| `terradev hf-space` | Deploy models to HuggingFace Spaces |
| `terradev inferx` | InferX serverless inference platform |
| `terradev lora` | Manage LoRA adapters on running endpoints |
| `terradev k8s` | Create and manage Kubernetes GPU clusters |
| `terradev ml` | ML service integrations (Ray, vLLM, SGLang, W&B) |
| `terradev gitops` | GitOps repository and deployment management |
| `terradev up` | Manifest cache and drift detection |
| `terradev rollback` | Roll back to a previous deployment |
| `terradev manifests` | List cached deployment manifests |
| `terradev integrations` | Show integration status |
| `terradev price-discovery` | Enhanced price analytics with confidence scoring |
| `terradev upgrade` | Manage your subscription tier |
| `terradev onboarding` | Rerun the first time setup flow |

## Getting Help

- Documentation: https://github.com/theoddden/Terradev
- PyPI: https://pypi.org/project/terradev-cli/
- Support: support@terradev.com
