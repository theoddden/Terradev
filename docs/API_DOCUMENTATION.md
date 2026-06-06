# Terradev API Documentation — v5.1.5

## Overview

Terradev is a **BYOAPI CLI** — there is no hosted REST API. All operations run locally using your own cloud provider credentials. This document describes the three integration interfaces:

1. **CLI** — Command-line interface for interactive use
2. **MCP Server** — JSON-RPC 2.0 protocol for AI agents (Claude Code, etc.)
3. **Python SDK** — Programmatic access via `terradev_cli` package

---

## CLI Interface

### Installation

```bash
pip install terradev-cli
```

### Usage

```bash
# Configure provider credentials
terradev configure --provider runpod
terradev configure --provider aws

# Get GPU prices
terradev quote --gpu-type H100

# Provision instances
terradev provision --gpu-type H100 --count 2 --spot

# Check status
terradev status
```

See [COMPLETE_COMMAND_REFERENCE.md](../terradev_cli/COMPLETE_COMMAND_REFERENCE.md) for full command reference.

---

## MCP Server (JSON-RPC 2.0)

### Installation

```bash
npm install -g terradev-mcp
```

### Configuration (Claude Code)

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "terradev": {
      "command": "terradev-mcp",
      "args": []
    }
  }
}
```

### Available Tools

The MCP server exposes 218 tools across these categories:

- **GPU Provisioning**: quote_gpu, provision_gpu, setup_provider, configure_provider
- **Training**: train_launch, train_status, train_stop, train_resume, train_snapshot
- **Inference**: infer_route, infer_deploy, infer_scale, vllm_analyze, vllm_benchmark
- **Kubernetes**: k8s_create, k8s_destroy, k8s_gpu_operator_install, k8s_mig_configure
- **RAG Stack**: qdrant (test, collections, create-collection), phoenix (projects, spans, trace), guardrails (chat, generate-config)
- **Observability**: langfuse (traces, scores, datasets), databricks (jobs, clusters, serving-endpoints)
- **Cost Optimization**: cost_analyze, cost_optimize_recommend, cost_simulate, cost_budget_optimize
- **Data Governance**: governance_request_consent, governance_evaluate_opa, governance_compliance_report

### Tool Example

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "quote_gpu",
    "arguments": {
      "gpu_type": "H100",
      "quick": true
    }
  }
}
```

---

## Python SDK

### Installation

```bash
pip install terradev-cli
```

### Usage

```python
from terradev_cli.providers.provider_factory import ProviderFactory
from terradev_cli.core.credentials import CredentialManager

# Load credentials
cred_mgr = CredentialManager()
credentials = cred_mgr.load_credentials()

# Create provider factory
factory = ProviderFactory()

# Get prices from a provider
provider = factory.create_provider("runpod", credentials["runpod"])
prices = provider.get_prices(gpu_type="H100")

print(prices)
```

### Programmatic Provisioning

```python
from terradev_cli.cli import cli
from click.testing import CliRunner

runner = CliRunner()

# Run CLI commands programmatically
result = runner.invoke(cli, ['quote', '--gpu-type', 'H100'])
print(result.output)

result = runner.invoke(cli, ['provision', '--gpu-type', 'H100', '--count', '2', '--spot'])
print(result.output)
```

---

## Supported Providers

21+ cloud providers are supported, including:

- **Major Clouds**: AWS, GCP, Azure, Alibaba
- **GPU Specialists**: RunPod, Vast.ai, Lambda Labs, CoreWeave, TensorDock, Crusoe, FluidStack, Hetzner, SiliconFlow, Hyperstack, Latitude.sh
- **ML Platforms**: HuggingFace, BaseTen, InferX
- **Others**: OVHcloud, Oracle, DigitalOcean

See [providers/](../terradev_cli/providers/) for implementation details.

---

## Authentication

All authentication is BYO (Bring Your Own API). Credentials are stored locally at:

```
~/.terradev/credentials.json
```

This file is never transmitted to Terradev servers. Configure each provider:

```bash
terradev configure --provider runpod
terradev configure --provider aws
terradev configure --provider gcp
```

---

## License

Apache 2.0 — Free and open source for commercial and personal use.

---

## Support

- **Documentation**: https://github.com/theoddden/Terradev
- **Issues**: https://github.com/theoddden/Terradev/issues
- **Changelog**: https://github.com/theoddden/Terradev/blob/main/terradev_cli/CHANGELOG.md
