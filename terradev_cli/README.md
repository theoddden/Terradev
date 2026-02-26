# Terradev CLI v3.1.8

BYOAPI, cross-cloud GPU provisioning and cost optimization platform with GitOps automation.

**GitHub Repository**: https://github.com/theoddden/terradev

## Why Terradev?

Developers overpay by only accessing single-cloud workflows or using sequential provisioning with inefficient egress + rate-limiting.

Terradev is a cross-cloud compute-provisioning CLI that compresses + stages datasets, provisions optimal instances + nodes, and deploys 3-5x faster than sequential provisioning.

## GitOps Automation

Production-ready GitOps workflows based on real-world Kubernetes experience.

*Terradev automatically configures topology with NUMA alignment when creating K8s clusters. GPU-NIC pairing is optimized at provisioning time, and no manual kubelet configuration is required.*

## HuggingFace Spaces Integration

Deploy any HuggingFace model to Spaces with one command:

```bash
# Install HF Spaces support
pip install terradev-cli[hf]

# Set your HF token
export HF_TOKEN=your_huggingface_token

# Deploy Llama 2 with one click
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Deploy custom model with GPU
terradev hf-space my-model --model-id microsoft/DialoGPT-medium \
  --hardware a10g-large --sdk gradio

# Result:
# Space URL: https://huggingface.co/spaces/username/my-llama
```

### HF Spaces Features
- **One-Click Deployment**: No manual configuration required
- **Template-Based**: LLM, embedding, and image model templates
- **Multi-Hardware**: CPU-basic to A100-large GPU tiers
- **Auto-Generated Apps**: Gradio, Streamlit, and Docker support
- **Revenue Streams**: Hardware upgrades, private spaces, template licensing

### Available Templates
```bash
# LLM Template (A10G GPU)
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Embedding Template (CPU-upgrade)
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 --template embedding

# Image Model Template (T4 GPU)
terradev hf-space my-image --model-id runwayml/stable-diffusion-v1-5 --template image
```

## MoE Cluster Templates (NEW in v3.1.8)

Deploy any Mixture-of-Experts model with production-ready infrastructure — one command for GLM-5, Qwen 3.5, Mistral Large 3, DeepSeek V4, Llama 5.

```bash
# Deploy any MoE model
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=zai-org/GLM-5-FP8 --set tp_size=8

# Smaller MoE on fewer GPUs
terradev provision --task clusters/moe-template/task.yaml \
  --set model_id=Qwen/Qwen3.5-122B-A10B --set tp_size=4 --set gpu_count=4
```

- **Parameterized**: model_id, tp_size, gpu_count, backend, precision — all configurable
- **Terraform + K8s + Helm**: Full stack included
- **NVLink + NUMA**: Topology-aware GPU placement enforced
- **FP8 + vLLM/SGLang**: Both backends, speculative decoding ready

## Installation

```bash
pip install terradev-cli
```

With HF Spaces support:
```bash
pip install terradev-cli[hf]        # HuggingFace Spaces deployment
pip install terradev-cli[all]        # All cloud providers + ML services + HF Spaces
```

## Quick Start

```bash
# 1. Get setup instructions for any provider
terradev setup runpod --quick
terradev setup aws --quick

# 2. Configure your cloud credentials (BYOAPI — you own your keys)
terradev configure --provider runpod
terradev configure --provider aws
terradev configure --provider vastai

# 3. Deploy to HuggingFace Spaces
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 --template embedding
terradev hf-space my-image --model-id runwayml/stable-diffusion-v1-5 --template image

# 4. Get enhanced quotes with conversion prompts
terradev quote -g A100
terradev quote -g A100 --quick  # Quick provision best quote

# 5. Provision the cheapest instance (real API call)
terradev provision -g A100

# 6. Configure ML services
terradev configure --provider wandb --dashboard-enabled true
terradev configure --provider langchain --tracing-enabled true

# 7. Use ML services
terradev ml wandb --test
terradev ml langchain --create-workflow my-workflow

# 8. View analytics
python user_analytics.py

# 9. Provision 4x H100s in parallel across multiple clouds
terradev provision -g H100 -n 4 --parallel 6

# 10. Dry-run to see the allocation plan without launching
terradev provision -g A100 -n 2 --dry-run

# 11. Manage running instances
terradev status --live
terradev manage -i <instance-id> -a stop
terradev manage -i <instance-id> -a start
terradev manage -i <instance-id> -a terminate

# 12. Execute commands on provisioned instances
terradev execute -i <instance-id> -c "python train.py"

# 13. Stage datasets near compute (compress + chunk + upload)
terradev stage -d ./my-dataset --target-regions us-east-1,eu-west-1

# 14. View cost analytics from the tracking database
terradev analytics --days 30

# 15. Find cheaper alternatives for running instances
terradev optimize

# 16. One-command Docker workload (provision + deploy + run)
terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"

# 17. Keep an inference server alive
terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
```

## BYOAuth — Bring Your Own Authentication

Terradev never touches, stores, or proxies your cloud credentials through a third party. Your API keys stay on your machine in `~/.terradev/credentials.json` — encrypted at rest, never transmitted.

**How it works:**
1. You run `terradev configure --provider <name>` and enter your API key
2. Credentials are stored locally in your home directory — never sent to Terradev servers
3. Every API call goes directly from your machine to the cloud provider
4. No middleman account, no shared credentials, no markup on provider pricing

**Why this matters:**
- **Zero trust exposure** — No third party holds your AWS/GCP/Azure keys
- **No vendor lock-in** — If you stop using Terradev, your cloud accounts are untouched
- **Enterprise-ready** — Compliant with SOC2, HIPAA, and internal security policies that prohibit sharing credentials with SaaS vendors
- **Full audit trail** — Every provision is logged locally with provider, cost, and timestamp

## CLI Commands

| Command | Description |
|---------|-------------|
| `terradev configure` | Set up API credentials for any provider |
| `terradev quote` | Get real-time GPU pricing across all clouds |
| `terradev provision` | Provision instances with parallel multi-cloud arbitrage |
| `terradev manage` | Stop, start, terminate, or check instance status |
| `terradev status` | View all instances and cost summary |
| `terradev execute` | Run commands on provisioned instances |
| `terradev stage` | Compress, chunk, and stage datasets near compute |
| `terradev analytics` | Cost analytics with daily spend trends |
| `terradev optimize` | Find cheaper alternatives for running instances |
| `terradev run` | Provision + deploy Docker container + execute in one command |
| `terradev hf-space` | **NEW:** One-click HuggingFace Spaces deployment |
| `terradev up` | **NEW:** Manifest cache + drift detection |
| `terradev rollback` | **NEW:** Versioned rollback to any deployment |
| `terradev manifests` | **NEW:** List cached deployment manifests |
| `terradev integrations` | Show status of W&B, Prometheus, and infra hooks |

### HF Spaces Commands (NEW!)
```bash
# Deploy Llama 2 to HF Spaces
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Deploy with custom hardware
terradev hf-space my-model --model-id microsoft/DialoGPT-medium \
  --hardware a10g-large --sdk gradio --private

# Deploy embedding model
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --template embedding --env BATCH_SIZE=64
```

### Manifest Cache Commands (NEW!)
```bash
# Provision with manifest cache
terradev up --job my-training --gpu-type A100 --gpu-count 4

# Fix drift automatically
terradev up --job my-training --fix-drift

# Rollback to previous version
terradev rollback my-training@v2

# List all cached manifests
terradev manifests --job my-training
```

## Observability & ML Integrations

Terradev facilitates connections to your existing tools via BYOAPI — your keys stay local, all data flows directly from your instances to your services.

| Integration | What Terradev Does | Setup |
|-------------|-------------------|-------|
| **Weights & Biases** | Auto-injects WANDB_* env vars into provisioned containers | `terradev configure --provider wandb --api-key YOUR_KEY` |
| **Prometheus** | Pushes provision/terminate metrics to your Pushgateway | `terradev configure --provider prometheus --api-key PUSHGATEWAY_URL` |
| **Grafana** | Exports a ready-to-import dashboard JSON | `terradev integrations --export-grafana` |

> Prices queried in real-time from all 10+ providers. Actual savings vary by availability.

## Pricing Tiers

| Feature | Research (Free) | Research+ ($49.99/mo) | Enterprise ($299.99/mo) |
|----------|------------------|------------------------|------------------------|
| Max concurrent instances | 1 | 8 | 32 |
| Provisions/month | 10 | 100 | Unlimited |
| Providers | All 11 | All 11 | All 11 + priority |
| Cost tracking | Yes | Yes | Yes |
| Dataset staging | Yes | Yes | Yes |
| Egress optimization | Basic | Full | Full + custom routes |

## Integrations

### Jupyter / Colab / VS Code Notebooks
```bash
pip install terradev-jupyter
%load_ext terradev_jupyter

%terradev quote -g A100
%terradev provision -g H100 --dry-run
%terradev run --gpu A100 --image pytorch/pytorch:latest --dry-run
```

### GitHub Actions
```yaml
- uses: theodden/terradev-action@v1
  with:
    gpu-type: A100
    max-price: "1.50"
  env:
    TERRADEV_RUNPOD_KEY: ${{ secrets.RUNPOD_API_KEY }}
```

### Docker (One-Command Workloads)
```bash
terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"
terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
```

## Requirements

- Python >= 3.9
- Cloud provider API keys (configured via `terradev configure`)

## License

Business Source License 1.1 (BUSL-1.1) - see LICENSE file for details
