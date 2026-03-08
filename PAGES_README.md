# Terradev CLI - GitHub Pages

This is the GitHub Pages site for Terradev CLI.

## Quick Links

- **GitHub Repository:** https://github.com/theoddden/terradev
- **PyPI Package:** https://pypi.org/project/terradev-cli/2.9.2/
- **Documentation:** https://github.com/theoddden/terradev/blob/main/README.md

## About Terradev CLI

Terradev CLI is a cross-cloud GPU provisioning and cost optimization platform that helps developers save 30% on end-to-end compute provisioning costs with real-time cloud arbitrage.

### Key Features

- 💰 **Cost Optimization** - Save 30% on GPU compute costs
- ☁️ **Multi-Cloud** - AWS, GCP, Azure, RunPod & more
- 🤖 **ML Ready** - HuggingFace Spaces integration
- 🚀 **Fast Deployment** - 3-5x faster than sequential provisioning

## Installation

```bash
pip install terradev-cli==2.9.2
```

## Usage

```bash
# Get GPU pricing across providers
terradev quote -g A100

# Provision optimal GPU instance
terradev provision -g A100

# Deploy to HuggingFace Spaces
terradev hf-space my-model --model-id meta-llama/Llama-2-7b-hf
```

## License

Business Source License 1.1 (BUSL-1.1) - Free for evaluation, testing, and internal business use.

📄 **License Details:** https://github.com/theoddden/terradev?tab=License-1-ov-file
