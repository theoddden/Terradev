#!/usr/bin/env python3
"""
Prometheus Integration — BYOAPI facilitation layer

Stores the user's Pushgateway URL and optional auth credentials locally,
and provides helpers to:
  - Push Terradev metrics (provisions, cost, utilization) to their Pushgateway
  - Generate scrape configs for their Prometheus instance
  - Build metric payloads without requiring the prometheus_client SDK at import time

Terradev pushes lightweight metrics on provision/terminate events.
The user's existing Prometheus + Grafana stack handles storage and visualization.
"""

from typing import Dict, Any, Optional, List
import time
import json


# ── Credential keys the user provides via `terradev configure` ────────────
REQUIRED_CREDENTIALS = {
    "pushgateway_url": "Prometheus Pushgateway URL (e.g. http://pushgateway:9091)",
}

OPTIONAL_CREDENTIALS = {
    "username": "Basic auth username (if Pushgateway is auth-protected)",
    "password": "Basic auth password",
}


def get_credential_prompts() -> List[Dict[str, str]]:
    """Return the list of credential prompts for interactive configure."""
    return [
        {"key": "prometheus_pushgateway_url", "prompt": "Pushgateway URL (e.g. http://pushgateway:9091)", "required": True, "hide": False},
        {"key": "prometheus_username", "prompt": "Pushgateway Username (optional)", "required": False, "hide": False},
        {"key": "prometheus_password", "prompt": "Pushgateway Password (optional)", "required": False, "hide": True},
    ]


# ── Metric definitions ───────────────────────────────────────────────────

METRICS = {
    "terradev_provisions_total": {
        "type": "counter",
        "help": "Total GPU instances provisioned via Terradev",
        "labels": ["provider", "gpu_type", "region"],
    },
    "terradev_gpu_cost_per_hour": {
        "type": "gauge",
        "help": "Current GPU cost in USD per hour",
        "labels": ["provider", "gpu_type", "region", "instance_id"],
    },
    "terradev_total_cost_usd": {
        "type": "gauge",
        "help": "Accumulated total cost in USD for an instance",
        "labels": ["provider", "instance_id"],
    },
    "terradev_provision_duration_seconds": {
        "type": "gauge",
        "help": "Instance uptime in seconds",
        "labels": ["instance_id"],
    },
    "terradev_active_instances": {
        "type": "gauge",
        "help": "Number of currently active instances",
        "labels": ["provider"],
    },
    "terradev_quote_price": {
        "type": "gauge",
        "help": "Latest quoted price from a provider",
        "labels": ["provider", "gpu_type", "region"],
    },
}


def build_metric_payload(
    metric_name: str,
    value: float,
    labels: Dict[str, str],
) -> str:
    """
    Build a single metric line in Prometheus text exposition format.

    This can be pushed to a Pushgateway via HTTP POST without needing
    the prometheus_client library installed.
    """
    meta = METRICS.get(metric_name, {})
    lines = []

    if meta:
        lines.append(f"# HELP {metric_name} {meta.get('help', '')}")
        lines.append(f"# TYPE {metric_name} {meta.get('type', 'gauge')}")

    label_str = ""
    if labels:
        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        label_str = "{" + ",".join(pairs) + "}"

    lines.append(f"{metric_name}{label_str} {value}")
    return "\n".join(lines) + "\n"


def build_provision_metrics(
    provider: str,
    gpu_type: str,
    region: str,
    instance_id: str,
    price_per_hour: float,
) -> str:
    """Build the full metric payload for a provision event."""
    parts = []

    parts.append(build_metric_payload(
        "terradev_provisions_total",
        1,
        {"provider": provider, "gpu_type": gpu_type, "region": region},
    ))

    parts.append(build_metric_payload(
        "terradev_gpu_cost_per_hour",
        price_per_hour,
        {"provider": provider, "gpu_type": gpu_type, "region": region, "instance_id": instance_id},
    ))

    return "\n".join(parts)


def build_terminate_metrics(
    provider: str,
    instance_id: str,
    total_cost: float,
    duration_seconds: float,
) -> str:
    """Build the full metric payload for a terminate event."""
    parts = []

    parts.append(build_metric_payload(
        "terradev_total_cost_usd",
        total_cost,
        {"provider": provider, "instance_id": instance_id},
    ))

    parts.append(build_metric_payload(
        "terradev_provision_duration_seconds",
        duration_seconds,
        {"instance_id": instance_id},
    ))

    return "\n".join(parts)


def get_push_url(credentials: Dict[str, str], job: str = "terradev") -> str:
    """
    Build the Pushgateway push URL.

    Format: {pushgateway_url}/metrics/job/{job}
    """
    base = credentials.get("prometheus_pushgateway_url", "").rstrip("/")
    return f"{base}/metrics/job/{job}"


def get_auth_headers(credentials: Dict[str, str]) -> Dict[str, str]:
    """
    Build HTTP auth headers for Pushgateway requests.

    Returns Basic auth header if username/password are configured.
    """
    import base64

    username = credentials.get("prometheus_username", "")
    password = credentials.get("prometheus_password", "")

    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    return {}


def push_metrics(
    credentials: Dict[str, str],
    payload: str,
    job: str = "terradev",
) -> Dict[str, Any]:
    """
    Push metrics to the Pushgateway via HTTP POST.

    Uses urllib so there's zero dependency on prometheus_client.
    Returns a status dict with success/error info.
    """
    import urllib.request
    import urllib.error

    url = get_push_url(credentials, job)
    headers = get_auth_headers(credentials)
    headers["Content-Type"] = "text/plain; version=0.0.4"

    try:
        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"success": True, "status_code": resp.status}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_scrape_config(job_name: str = "terradev") -> str:
    """
    Generate a Prometheus scrape config snippet for the user's prometheus.yml.

    This targets the Pushgateway so Prometheus picks up Terradev metrics.
    """
    config = {
        "job_name": job_name,
        "honor_labels": True,
        "static_configs": [
            {"targets": ["pushgateway:9091"]}
        ],
    }

    lines = [
        f"  - job_name: '{job_name}'",
        "    honor_labels: true",
        "    static_configs:",
        "      - targets: ['pushgateway:9091']",
    ]
    return "\n".join(lines)


def generate_grafana_dashboard_json(
    title: str = "Terradev GPU Cost Dashboard",
) -> Dict[str, Any]:
    """
    Generate a minimal Grafana dashboard JSON that visualizes Terradev metrics.

    Users can import this into their Grafana instance.
    """
    return {
        "dashboard": {
            "title": title,
            "tags": ["terradev", "gpu", "cost"],
            "panels": [
                {
                    "title": "GPU Cost per Hour by Provider",
                    "type": "timeseries",
                    "targets": [{"expr": "terradev_gpu_cost_per_hour", "legendFormat": "{{provider}} — {{gpu_type}}"}],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                },
                {
                    "title": "Total Provisions",
                    "type": "stat",
                    "targets": [{"expr": "sum(terradev_provisions_total)", "legendFormat": "Total"}],
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                },
                {
                    "title": "Active Instances by Provider",
                    "type": "bargauge",
                    "targets": [{"expr": "terradev_active_instances", "legendFormat": "{{provider}}"}],
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                },
                {
                    "title": "Accumulated Cost per Instance",
                    "type": "table",
                    "targets": [{"expr": "terradev_total_cost_usd", "legendFormat": "{{instance_id}}"}],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                },
                {
                    "title": "Quote Prices Over Time",
                    "type": "timeseries",
                    "targets": [{"expr": "terradev_quote_price", "legendFormat": "{{provider}} — {{gpu_type}}"}],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                },
            ],
            "time": {"from": "now-24h", "to": "now"},
            "refresh": "30s",
        }
    }


def is_configured(credentials: Dict[str, str]) -> bool:
    """Check if Prometheus credentials are stored."""
    return bool(credentials.get("prometheus_pushgateway_url", ""))


def get_status_summary(credentials: Dict[str, str]) -> Dict[str, Any]:
    """Return a summary of Prometheus integration status."""
    configured = is_configured(credentials)
    return {
        "integration": "prometheus",
        "name": "Prometheus",
        "configured": configured,
        "pushgateway_url": credentials.get("prometheus_pushgateway_url", "(not set)"),
        "auth_enabled": bool(credentials.get("prometheus_username", "")),
    }
