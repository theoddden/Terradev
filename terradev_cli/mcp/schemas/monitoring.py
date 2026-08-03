"""MCP tool schema definitions."""

from typing import Any, List

try:
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None

TOOLS = []

if Tool is not None:
    TOOLS = [

        Tool(name='datadog_status', description='Check Datadog integration status: configured, site, API key presence.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='datadog_push_metrics', description='Push current GPU cost snapshot to Datadog: active instances, cost/hr, projected monthly, provider reliability, quote latency.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='datadog_send_event', description='Send a custom event to Datadog with title, text, and alert type.', inputSchema={'type': 'object', 'properties': {'title': {'type': 'string', 'description': 'Event title'}, 'text': {'type': 'string', 'description': 'Event body (markdown supported)'}, 'alert_type': {'type': 'string', 'description': 'Alert type', 'enum': ['info', 'warning', 'error', 'success'], 'default': 'info'}}, 'required': ['title', 'text']}),
        Tool(name='datadog_create_monitors', description='Create all Terradev GPU FinOps monitors in Datadog: budget alert, cost spike, idle GPU, spot volatility, provider degraded, egress anomaly.', inputSchema={'type': 'object', 'properties': {'template': {'type': 'string', 'description': 'Single template name (or omit for all)', 'enum': ['budget_alert', 'cost_spike', 'idle_gpu', 'spot_risk', 'provider_degraded', 'egress_anomaly']}}}),
        Tool(name='datadog_list_monitors', description='List all Terradev-tagged monitors in Datadog with their current status.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='datadog_create_dashboard', description='Create the Terradev GPU FinOps dashboard in Datadog with 12 widgets: cost/hr, projected monthly, active GPUs, budget, provider reliability, volatility, latency, training util, egress, events.', inputSchema={'type': 'object', 'properties': {'title': {'type': 'string', 'description': 'Custom dashboard title (default: Terradev GPU FinOps)'}}}),
        Tool(name='datadog_list_dashboards', description='List Terradev-related dashboards in Datadog.', inputSchema={'type': 'object', 'properties': {}}),
        Tool(name='datadog_query', description='Query Datadog metrics using the metrics query language. Returns time series data.', inputSchema={'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Datadog metric query (e.g. avg:terradev.gpu.cost_per_hour{*} by {provider})'}, 'from_seconds': {'type': 'integer', 'description': 'Lookback window in seconds', 'default': 3600}}, 'required': ['query']}),
        Tool(name='datadog_terraform_export', description='Generate a complete Terraform module for the Datadog integration: provider.tf, monitors.tf, dashboard.tf, versions.tf. Run `terraform apply` to deploy all monitors and dashboards as IaC.', inputSchema={'type': 'object', 'properties': {'output_dir': {'type': 'string', 'description': 'Directory to write .tf files (default: ./datadog-terraform)'}}}),
        Tool(name='datadog_metric_catalog', description='List all Terradev metrics that can be pushed to Datadog with their types, units, and tags.', inputSchema={'type': 'object', 'properties': {}}),
    ]
