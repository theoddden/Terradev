# Terradev Datadog Module

Terraform module that deploys GPU FinOps monitoring to Datadog:

- **6 monitors** — budget, cost spike, idle GPU, spot volatility, provider degraded, egress anomaly
- **1 dashboard** — 12-widget GPU cost intelligence dashboard
- **Fully parameterized** — thresholds, notification channels, tags

## Usage

```hcl
module "datadog" {
  source = "./modules/datadog"

  datadog_api_key = var.datadog_api_key
  datadog_app_key = var.datadog_app_key
  datadog_site    = "datadoghq.com"

  notification_channel      = "@slack-gpu-alerts"
  budget_threshold_critical  = 80
  idle_gpu_threshold         = 10
  spot_volatility_threshold  = 100
}
```

## From Terradev CLI

```bash
# Export Terraform module from stored credentials
terradev datadog terraform-export --output-dir ./datadog-terraform

# Or via MCP (Claude Code)
# → datadog_terraform_export tool
```

## Inputs

| Variable | Description | Default |
|----------|-------------|---------|
| `datadog_api_key` | Datadog API Key | (required) |
| `datadog_app_key` | Datadog Application Key | (required) |
| `datadog_site` | Datadog site | `datadoghq.com` |
| `notification_channel` | Alert target | `@slack-terradev-alerts` |
| `budget_threshold_critical` | Budget alert % | `80` |
| `cost_spike_threshold` | Cost spike % over 4h | `50` |
| `idle_gpu_threshold` | GPU util % for idle | `10` |
| `spot_volatility_threshold` | Spot vol % | `100` |
| `provider_reliability_threshold` | Reliability score | `70` |
| `create_dashboard` | Create dashboard | `true` |
| `create_monitors` | Create monitors | `true` |
| `extra_tags` | Additional tags | `[]` |

## Outputs

| Output | Description |
|--------|-------------|
| `monitor_ids` | Map of monitor name → Datadog ID |
| `dashboard_id` | Dashboard ID |
| `dashboard_url` | Dashboard URL |
