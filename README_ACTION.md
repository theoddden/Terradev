# Terradev CLI - GitHub Action

Use Terradev CLI in your GitHub Actions workflows to optimize GPU costs across multiple cloud providers.

## 🚀 Features

- **Multi-cloud GPU pricing comparison** - Get real-time pricing from AWS, GCP, Azure, RunPod, Lambda Labs, and more
- **Cost optimization** - Find the optimal provider for your GPU workloads
- **Automated provisioning** - Deploy GPU instances with optimal pricing
- **HuggingFace Spaces deployment** - Deploy models to HuggingFace Spaces
- **Real-time cost calculation** - Including hidden fees (egress, network, storage)

## 📦 Usage

### Basic GPU Pricing Quote

```yaml
name: GPU Cost Analysis
on: [push]

jobs:
  gpu-pricing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Get GPU Pricing
        uses: theoddden/terradev@v2.9.2
        with:
          command: 'quote'
          gpu_type: 'A100'
      
      - name: Get H100 Pricing
        uses: theoddden/terradev@v2.9.2
        with:
          command: 'quote'
          gpu_type: 'H100'
```

### Provision GPU Instance

```yaml
name: Deploy GPU Workload
on: [push]

jobs:
  deploy-gpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Provision GPU Instance
        uses: theoddden/terradev@v2.9.2
        with:
          command: 'provision'
          gpu_type: 'A100'
          duration_hours: '4'
          cloud_provider: 'auto'
          region: 'auto'
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          GCP_SERVICE_ACCOUNT_KEY: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
```

### Deploy to HuggingFace Spaces

```yaml
name: Deploy to HuggingFace Spaces
on: [push]

jobs:
  deploy-hf:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to HuggingFace Spaces
        uses: theoddden/terradev@v2.9.2
        with:
          command: 'hf-space'
          huggingface_model: 'meta-llama/Llama-2-7b-hf'
        env:
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
```

## ⚙️ Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `command` | Command to run (`quote`, `provision`, `status`, `hf-space`) | Yes | `quote` |
| `gpu_type` | GPU type (`A100`, `H100`, `V100`, `RTX4090`, etc.) | Yes | `A100` |
| `duration_hours` | Duration for provision in hours | No | `1` |
| `cloud_provider` | Preferred cloud provider (`aws`, `gcp`, `azure`, `runpod`, `lambda`, etc.) | No | `auto` |
| `region` | Preferred region for deployment | No | `auto` |
| `huggingface_model` | HuggingFace model ID for Spaces deployment | No | `` |

## 📤 Outputs

| Output | Description |
|--------|-------------|
| `pricing_data` | GPU pricing data across providers |
| `optimal_provider` | Recommended optimal provider |
| `estimated_cost` | Estimated cost for the workload |
| `deployment_url` | URL for deployed workload (if applicable) |

## 🔐 Environment Variables

For provisioning and deployment, you'll need to configure cloud provider credentials:

```yaml
env:
  # AWS Credentials
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  
  # GCP Credentials
  GCP_SERVICE_ACCOUNT_KEY: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
  
  # Azure Credentials
  AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
  AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
  AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
  
  # HuggingFace Token
  HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
```

## 🎯 Use Cases

### 1. Cost Optimization in CI/CD
```yaml
- name: Optimize GPU Costs
  uses: theoddden/terradev@v2.9.2
  with:
    command: 'quote'
    gpu_type: 'A100'
```

### 2. Automated ML Training
```yaml
- name: Deploy Training Job
  uses: theoddden/terradev@v2.9.2
  with:
    command: 'provision'
    gpu_type: 'A100'
    duration_hours: '8'
```

### 3. Model Deployment
```yaml
- name: Deploy Model to Spaces
  uses: theoddden/terradev@v2.9.2
  with:
    command: 'hf-space'
    huggingface_model: 'meta-llama/Llama-2-7b-hf'
```

## 📊 Example Output

```
🚀 Terradev CLI - GitHub Action
Command: quote
GPU Type: A100
Duration: 1 hours
Provider: auto
Region: auto

✅ Command executed successfully
STDOUT:
=== A100 GPU Pricing Comparison ===
AWS (us-east-1): $2.40/hr
GCP (us-central1): $2.20/hr
Azure (eastus): $2.50/hr
RunPod: $1.80/hr ⭐ OPTIMAL
Lambda Labs: $1.90/hr

💰 Potential Savings: $0.70/hr (29% savings)
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the Business Source License 1.1 (BUSL-1.1) - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub Repository**: https://github.com/theoddden/terradev
- **PyPI Package**: https://pypi.org/project/terradev-cli/
- **Docker Hub**: https://hub.docker.com/r/theoddden/terradev
- **Documentation**: https://github.com/theoddden/terradev/blob/main/README.md

## 🆘 Support

- 📧 Email: team@terradev.com
- 💬 Discord: [Join our Discord](https://discord.gg/terradev)
- 🐛 Issues: [GitHub Issues](https://github.com/theoddden/terradev/issues)
