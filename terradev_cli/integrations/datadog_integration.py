#!/usr/bin/env python3
"""
Datadog Integration — Deep FinOps observability layer for Terradev

Pushes GPU cost metrics, provision events, price intelligence, and
infrastructure health to Datadog via the v2 API.  Provides helpers to:

  - Submit custom metrics (gauge, count, distribution) via /api/v2/series
  - Create / update monitors for budget, utilization, and spot risk
  - Create GPU cost dashboards programmatically
  - Send events on provision / terminate / scale actions
  - Query existing metrics for anomaly detection
  - Generate Terraform variable files for the datadog provider

Zero SDK dependency — all calls go through urllib/aiohttp so the user
only needs a DD_API_KEY + DD_APP_KEY.  BYOAPI pattern: keys stay local
in ~/.terradev/credentials.json, never leave the machine.
"""

from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime, timedelta


# ── Credential prompts ─────────────────────────────────────────────────

REQUIRED_CREDENTIALS = {
    "api_key": "Datadog API Key (from Organization Settings → API Keys)",
    "app_key": "Datadog Application Key (from Organization Settings → Application Keys)",
}

OPTIONAL_CREDENTIALS = {
    "site": "Datadog site (datadoghq.com | datadoghq.eu | us3.datadoghq.com | us5.datadoghq.com | ap1.datadoghq.com)",
}


def get_credential_prompts() -> List[Dict[str, str]]:
    return [
        {"key": "datadog_api_key", "prompt": "Datadog API Key", "required": True, "hide": True},
        {"key": "datadog_app_key", "prompt": "Datadog Application Key", "required": True, "hide": True},
        {"key": "datadog_site", "prompt": "Datadog site (default: datadoghq.com)", "required": False, "hide": False},
    ]


def _get_site(creds: Dict[str, str]) -> str:
    return creds.get("datadog_site", "datadoghq.com").strip() or "datadoghq.com"

def _base_url(creds: Dict[str, str]) -> str:
    return f"https://api.{_get_site(creds)}"

def _auth_headers(creds: Dict[str, str]) -> Dict[str, str]:
    return {
        "DD-API-KEY": creds.get("datadog_api_key", ""),
        "DD-APPLICATION-KEY": creds.get("datadog_app_key", ""),
        "Content-Type": "application/json",
    }

def is_configured(creds: Dict[str, str]) -> bool:
    return bool(creds.get("datadog_api_key")) and bool(creds.get("datadog_app_key"))

def get_status_summary(creds: Dict[str, str]) -> Dict[str, Any]:
    return {
        "integration": "datadog", "name": "Datadog",
        "configured": is_configured(creds), "site": _get_site(creds),
        "api_key_set": bool(creds.get("datadog_api_key", "")),
        "app_key_set": bool(creds.get("datadog_app_key", "")),
    }


# ═══════════════════════════════════════════════════════════════════════
# METRIC CATALOG
# ═══════════════════════════════════════════════════════════════════════

PREFIX = "terradev"

METRIC_CATALOG = {
    f"{PREFIX}.gpu.cost_per_hour":      {"type": "gauge",  "unit": "dollar",      "desc": "Current GPU cost $/hr",               "tags": ["provider","gpu_type","region","instance_id","spot"]},
    f"{PREFIX}.gpu.total_cost":         {"type": "gauge",  "unit": "dollar",      "desc": "Accumulated total cost",              "tags": ["provider","instance_id"]},
    f"{PREFIX}.gpu.monthly_projected":  {"type": "gauge",  "unit": "dollar",      "desc": "Projected monthly spend",             "tags": ["provider"]},
    f"{PREFIX}.gpu.budget_utilization": {"type": "gauge",  "unit": "percent",     "desc": "Budget consumed %",                   "tags": []},
    f"{PREFIX}.provisions.total":       {"type": "count",  "unit": "instance",    "desc": "Total provisions",                    "tags": ["provider","gpu_type","region"]},
    f"{PREFIX}.provisions.active":      {"type": "gauge",  "unit": "instance",    "desc": "Active instances",                    "tags": ["provider"]},
    f"{PREFIX}.provisions.duration_s":  {"type": "gauge",  "unit": "second",      "desc": "Instance uptime",                     "tags": ["instance_id"]},
    f"{PREFIX}.price.quote":            {"type": "gauge",  "unit": "dollar",      "desc": "Latest quoted price",                 "tags": ["provider","gpu_type","region","spot"]},
    f"{PREFIX}.price.delta_1h":         {"type": "gauge",  "unit": "dollar",      "desc": "1h price change",                     "tags": ["provider","gpu_type"]},
    f"{PREFIX}.price.volatility":       {"type": "gauge",  "unit": "percent",     "desc": "Annualized volatility",               "tags": ["provider","gpu_type"]},
    f"{PREFIX}.price.stability_score":  {"type": "gauge",  "unit": "score",       "desc": "Stability score 0-100",               "tags": ["provider","gpu_type"]},
    f"{PREFIX}.provider.reliability":   {"type": "gauge",  "unit": "score",       "desc": "Reliability score 0-100",             "tags": ["provider"]},
    f"{PREFIX}.provider.latency_ms":    {"type": "gauge",  "unit": "millisecond", "desc": "Quote API latency",                   "tags": ["provider"]},
    f"{PREFIX}.provider.availability":  {"type": "gauge",  "unit": "percent",     "desc": "GPU availability rate 24h",           "tags": ["provider","gpu_type"]},
    f"{PREFIX}.egress.bytes":           {"type": "count",  "unit": "byte",        "desc": "Cross-cloud transfer",                "tags": ["src_provider","dst_provider"]},
    f"{PREFIX}.egress.cost":            {"type": "gauge",  "unit": "dollar",      "desc": "Egress cost",                         "tags": ["src_provider","dst_provider"]},
    f"{PREFIX}.training.gpu_util":      {"type": "gauge",  "unit": "percent",     "desc": "Training GPU utilization",            "tags": ["job_id","gpu_type"]},
    f"{PREFIX}.training.cost_per_epoch":{"type": "gauge",  "unit": "dollar",      "desc": "Cost per epoch",                      "tags": ["job_id"]},
}


# ═══════════════════════════════════════════════════════════════════════
# METRIC SUBMISSION  (v2 API — /api/v2/series)
# ═══════════════════════════════════════════════════════════════════════

_DD_TYPE = {"gauge": 3, "count": 1, "rate": 2}

def _build_series(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    ts = int(time.time())
    series = []
    for m in metrics:
        tags = [f"{k}:{v}" for k, v in (m.get("tags") or {}).items()]
        series.append({
            "metric": m["metric"], "type": _DD_TYPE.get(m.get("type", "gauge"), 3),
            "points": [{"timestamp": m.get("timestamp", ts), "value": m["value"]}],
            "tags": tags,
        })
    return {"series": series}


def submit_metrics_sync(creds: Dict[str, str], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    import urllib.request, urllib.error
    url = f"{_base_url(creds)}/api/v2/series"
    payload = json.dumps(_build_series(metrics)).encode()
    try:
        req = urllib.request.Request(url, data=payload, headers=_auth_headers(creds), method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"success": True, "status_code": resp.status, "metrics_sent": len(metrics)}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def submit_metrics_async(creds: Dict[str, str], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return submit_metrics_sync(creds, metrics)
    url = f"{_base_url(creds)}/api/v2/series"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=_build_series(metrics), headers=_auth_headers(creds),
                              timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status < 300:
                    return {"success": True, "status_code": r.status, "metrics_sent": len(metrics)}
                return {"success": False, "error": f"HTTP {r.status}: {(await r.text())[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# EVENT SUBMISSION  (v1 API — /api/v1/events)
# ═══════════════════════════════════════════════════════════════════════

def _build_event(title: str, text: str, alert_type: str = "info", tags: Optional[Dict[str, str]] = None) -> Dict:
    tl = [f"{k}:{v}" for k, v in (tags or {}).items()]
    tl.append("source:terradev")
    return {"title": title, "text": text, "alert_type": alert_type, "tags": tl, "source_type_name": "terradev"}


def send_event_sync(creds: Dict[str, str], title: str, text: str,
                     alert_type: str = "info", tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import urllib.request, urllib.error
    url = f"{_base_url(creds)}/api/v1/events"
    payload = json.dumps(_build_event(title, text, alert_type, tags)).encode()
    try:
        req = urllib.request.Request(url, data=payload, headers=_auth_headers(creds), method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"success": True, "status_code": resp.status}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_event_async(creds: Dict[str, str], title: str, text: str,
                            alert_type: str = "info", tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return send_event_sync(creds, title, text, alert_type, tags)
    url = f"{_base_url(creds)}/api/v1/events"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=_build_event(title, text, alert_type, tags),
                              headers=_auth_headers(creds), timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status < 300:
                    return {"success": True, "status_code": r.status}
                return {"success": False, "error": f"HTTP {r.status}: {(await r.text())[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# MONITOR MANAGEMENT  (v1 API — /api/v1/monitor)
# ═══════════════════════════════════════════════════════════════════════

MONITOR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "budget_alert": {
        "name": "[Terradev] GPU Budget Alert", "type": "metric alert",
        "query": "avg(last_1h):avg:terradev.gpu.budget_utilization{*} > 80",
        "message": "GPU budget >80%. Current: {{value}}%\n\n- Switch to spot\n- Downsize idle GPUs\n- Shut idle instances\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 80, "warning": 60}, "notify_no_data": False, "renotify_interval": 60},
        "tags": ["terradev", "finops", "budget"], "priority": 2,
    },
    "cost_spike": {
        "name": "[Terradev] GPU Cost Spike", "type": "metric alert",
        "query": "pct_change(avg(last_1h),last_4h):avg:terradev.gpu.cost_per_hour{*} > 50",
        "message": "GPU cost spiked >50% vs 4h baseline.\n\n- Unintended expensive provision?\n- Spot fallback?\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 50, "warning": 25}, "notify_no_data": False, "renotify_interval": 120},
        "tags": ["terradev", "finops", "cost-spike"], "priority": 2,
    },
    "idle_gpu": {
        "name": "[Terradev] Idle GPU Detection", "type": "metric alert",
        "query": "avg(last_30m):avg:terradev.training.gpu_util{*} by {instance_id} < 10",
        "message": "GPU {{instance_id.name}} <10% util for 30m. Terminate or downsize.\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 10, "warning": 25}, "notify_no_data": False, "renotify_interval": 60},
        "tags": ["terradev", "finops", "idle"], "priority": 3,
    },
    "spot_risk": {
        "name": "[Terradev] Spot Volatility", "type": "metric alert",
        "query": "avg(last_15m):avg:terradev.price.volatility{spot:true} by {provider,gpu_type} > 100",
        "message": "High spot volatility {{provider.name}} {{gpu_type.name}}. Vol: {{value}}%\n\n- Checkpoint auto-save\n- On-demand fallback\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 100, "warning": 60}, "notify_no_data": False, "renotify_interval": 30},
        "tags": ["terradev", "finops", "spot"], "priority": 1,
    },
    "provider_degraded": {
        "name": "[Terradev] Provider Degraded", "type": "metric alert",
        "query": "avg(last_1h):avg:terradev.provider.reliability{*} by {provider} < 70",
        "message": "Provider {{provider.name}} reliability <70. Score: {{value}}\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 70, "warning": 85}, "notify_no_data": False, "renotify_interval": 120},
        "tags": ["terradev", "finops", "provider"], "priority": 3,
    },
    "egress_anomaly": {
        "name": "[Terradev] Egress Anomaly", "type": "metric alert",
        "query": "avg(last_1h):anomalies(avg:terradev.egress.cost{*}, 'agile', 3) >= 1",
        "message": "Anomalous egress cost.\n\n- Cross-cloud transfers?\n- Missing compression?\n\n@slack-terradev-alerts",
        "options": {"thresholds": {"critical": 1}, "notify_no_data": False},
        "tags": ["terradev", "finops", "egress"], "priority": 3,
    },
}


async def create_monitor(creds: Dict[str, str], template_name: str,
                          overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if template_name not in MONITOR_TEMPLATES:
        return {"success": False, "error": f"Unknown template: {template_name}. Available: {list(MONITOR_TEMPLATES.keys())}"}
    monitor = {**MONITOR_TEMPLATES[template_name]}
    if overrides:
        monitor.update(overrides)
    try:
        import aiohttp
    except ImportError:
        return _create_monitor_sync(creds, monitor)
    url = f"{_base_url(creds)}/api/v1/monitor"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=monitor, headers=_auth_headers(creds),
                              timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
                if r.status < 300:
                    return {"success": True, "monitor_id": body.get("id"), "name": monitor["name"]}
                return {"success": False, "error": f"HTTP {r.status}: {json.dumps(body)[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _create_monitor_sync(creds: Dict[str, str], monitor: Dict) -> Dict[str, Any]:
    import urllib.request, urllib.error
    url = f"{_base_url(creds)}/api/v1/monitor"
    try:
        req = urllib.request.Request(url, data=json.dumps(monitor).encode(),
                                     headers=_auth_headers(creds), method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            return {"success": True, "monitor_id": body.get("id"), "name": monitor["name"]}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def create_all_monitors(creds: Dict[str, str]) -> Dict[str, Any]:
    results = {}
    for name in MONITOR_TEMPLATES:
        results[name] = await create_monitor(creds, name)
    ok = sum(1 for r in results.values() if r.get("success"))
    return {"monitors_created": ok, "monitors_failed": len(results) - ok, "details": results}


async def list_monitors(creds: Dict[str, str]) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}
    url = f"{_base_url(creds)}/api/v1/monitor?monitor_tags=terradev"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=_auth_headers(creds),
                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
                if r.status < 300:
                    mons = [{"id": m["id"], "name": m["name"], "status": m.get("overall_state", "unknown"),
                             "query": m.get("query", ""), "tags": m.get("tags", [])} for m in body]
                    return {"success": True, "count": len(mons), "monitors": mons}
                return {"success": False, "error": f"HTTP {r.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def delete_monitor(creds: Dict[str, str], monitor_id: int) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}
    url = f"{_base_url(creds)}/api/v1/monitor/{monitor_id}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.delete(url, headers=_auth_headers(creds),
                                timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status < 300:
                    return {"success": True, "deleted": monitor_id}
                return {"success": False, "error": f"HTTP {r.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER  (v1 API — /api/v1/dashboard)
# ═══════════════════════════════════════════════════════════════════════

def _gpu_cost_dashboard() -> Dict[str, Any]:
    """Full Terradev GPU FinOps dashboard definition."""
    def _qv(title, query, unit="", cond=None):
        w = {"type": "query_value", "title": title,
             "requests": [{"queries": [{"data_source": "metrics", "name": "a", "query": query}],
                           "formulas": [{"formula": "a"}], "response_format": "scalar"}]}
        if unit:
            w["custom_unit"] = unit
        if cond:
            w["conditional_formats"] = cond
        return w

    def _ts(title, query, display="line"):
        return {"type": "timeseries", "title": title,
                "requests": [{"queries": [{"data_source": "metrics", "name": "a", "query": query}],
                              "formulas": [{"formula": "a"}], "response_format": "timeseries",
                              "display_type": display}]}

    budget_cond = [
        {"comparator": ">", "value": 80, "palette": "white_on_red"},
        {"comparator": ">", "value": 60, "palette": "white_on_yellow"},
        {"comparator": "<=", "value": 60, "palette": "white_on_green"},
    ]

    return {
        "title": "Terradev GPU FinOps",
        "description": "Multi-cloud GPU cost intelligence — provisioned by Terradev",
        "layout_type": "ordered",
        "tags": ["terradev", "finops", "gpu"],
        "widgets": [
            {"definition": _qv("Hourly Spend", "sum:terradev.gpu.cost_per_hour{*}", "$/hr"),
             "layout": {"x": 0, "y": 0, "width": 3, "height": 2}},
            {"definition": _qv("Monthly Projected", "avg:terradev.gpu.monthly_projected{*}", "$"),
             "layout": {"x": 3, "y": 0, "width": 3, "height": 2}},
            {"definition": _qv("Active GPUs", "sum:terradev.provisions.active{*}"),
             "layout": {"x": 6, "y": 0, "width": 3, "height": 2}},
            {"definition": _qv("Budget Used", "avg:terradev.gpu.budget_utilization{*}", "%", budget_cond),
             "layout": {"x": 9, "y": 0, "width": 3, "height": 2}},
            {"definition": _ts("GPU Cost/hr by Provider", "avg:terradev.gpu.cost_per_hour{*} by {provider}", "bars"),
             "layout": {"x": 0, "y": 2, "width": 6, "height": 3}},
            {"definition": _ts("Quote Prices by GPU", "avg:terradev.price.quote{*} by {gpu_type,provider}"),
             "layout": {"x": 6, "y": 2, "width": 6, "height": 3}},
            {"definition": {"type": "toplist", "title": "Provider Reliability",
                "requests": [{"queries": [{"data_source": "metrics", "name": "a",
                    "query": "avg:terradev.provider.reliability{*} by {provider}"}],
                    "formulas": [{"formula": "a", "limit": {"count": 20, "order": "desc"}}],
                    "response_format": "scalar"}]},
             "layout": {"x": 0, "y": 5, "width": 4, "height": 3}},
            {"definition": _ts("Price Volatility", "avg:terradev.price.volatility{*} by {provider,gpu_type}"),
             "layout": {"x": 4, "y": 5, "width": 4, "height": 3}},
            {"definition": _ts("Quote API Latency", "avg:terradev.provider.latency_ms{*} by {provider}"),
             "layout": {"x": 8, "y": 5, "width": 4, "height": 3}},
            {"definition": _ts("Training GPU Util", "avg:terradev.training.gpu_util{*} by {job_id}"),
             "layout": {"x": 0, "y": 8, "width": 6, "height": 3}},
            {"definition": _ts("Egress Cost", "sum:terradev.egress.cost{*} by {src_provider,dst_provider}.as_count()", "bars"),
             "layout": {"x": 6, "y": 8, "width": 6, "height": 3}},
            {"definition": {"type": "event_stream", "title": "Terradev Events",
                "query": "source:terradev", "tags_execution": "and", "event_size": "l"},
             "layout": {"x": 0, "y": 11, "width": 12, "height": 3}},
        ],
    }


async def create_dashboard(creds: Dict[str, str], custom_title: Optional[str] = None) -> Dict[str, Any]:
    dashboard = _gpu_cost_dashboard()
    if custom_title:
        dashboard["title"] = custom_title
    try:
        import aiohttp
    except ImportError:
        return _create_dashboard_sync(creds, dashboard)
    url = f"{_base_url(creds)}/api/v1/dashboard"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=dashboard, headers=_auth_headers(creds),
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
                body = await r.json(content_type=None)
                if r.status < 300:
                    dash_url = body.get("url", "")
                    if dash_url and not dash_url.startswith("http"):
                        dash_url = f"https://app.{_get_site(creds)}{dash_url}"
                    return {"success": True, "dashboard_id": body.get("id"),
                            "title": dashboard["title"], "url": dash_url,
                            "widgets": len(dashboard["widgets"])}
                return {"success": False, "error": f"HTTP {r.status}: {json.dumps(body)[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _create_dashboard_sync(creds: Dict[str, str], dashboard: Dict) -> Dict[str, Any]:
    import urllib.request, urllib.error
    url = f"{_base_url(creds)}/api/v1/dashboard"
    try:
        req = urllib.request.Request(url, data=json.dumps(dashboard).encode(),
                                     headers=_auth_headers(creds), method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
            return {"success": True, "dashboard_id": body.get("id"),
                    "title": dashboard["title"], "widgets": len(dashboard["widgets"])}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_dashboards(creds: Dict[str, str]) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}
    url = f"{_base_url(creds)}/api/v1/dashboard"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=_auth_headers(creds),
                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
                if r.status < 300:
                    all_d = body.get("dashboards", [])
                    td = [d for d in all_d if "terradev" in d.get("title", "").lower()]
                    return {"success": True, "count": len(td),
                            "dashboards": [{"id": d["id"], "title": d.get("title", ""),
                                            "url": d.get("url", "")} for d in td]}
                return {"success": False, "error": f"HTTP {r.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# METRIC QUERY  (v1 API — /api/v1/query)
# ═══════════════════════════════════════════════════════════════════════

async def query_metrics(creds: Dict[str, str], query: str,
                         from_seconds: int = 3600) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}
    now = int(time.time())
    url = f"{_base_url(creds)}/api/v1/query?from={now - from_seconds}&to={now}&query={query}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=_auth_headers(creds),
                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                body = await r.json(content_type=None)
                if r.status < 300:
                    return {"success": True, "series_count": len(body.get("series", [])), "data": body}
                return {"success": False, "error": f"HTTP {r.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# HIGH-LEVEL HOOKS  (called from provision/terminate flows)
# ═══════════════════════════════════════════════════════════════════════

async def on_provision(creds: Dict[str, str], provider: str, gpu_type: str,
                        region: str, instance_id: str, price_hr: float,
                        spot: bool = False) -> Dict[str, Any]:
    """Push metrics + event on GPU provision."""
    if not is_configured(creds):
        return {"skipped": True, "reason": "Datadog not configured"}
    metrics = [
        {"metric": f"{PREFIX}.gpu.cost_per_hour", "value": price_hr, "type": "gauge",
         "tags": {"provider": provider, "gpu_type": gpu_type, "region": region,
                  "instance_id": instance_id, "spot": str(spot).lower()}},
        {"metric": f"{PREFIX}.provisions.total", "value": 1, "type": "count",
         "tags": {"provider": provider, "gpu_type": gpu_type, "region": region}},
    ]
    m_result = await submit_metrics_async(creds, metrics)
    e_result = await send_event_async(
        creds, title=f"GPU Provisioned: {gpu_type} on {provider}",
        text=f"Instance `{instance_id}` in {region}. ${price_hr:.4f}/hr | Spot: {spot}",
        alert_type="info",
        tags={"provider": provider, "gpu_type": gpu_type, "region": region},
    )
    return {"metrics": m_result, "event": e_result}


async def on_terminate(creds: Dict[str, str], provider: str, instance_id: str,
                        total_cost: float, duration_seconds: float) -> Dict[str, Any]:
    """Push metrics + event on GPU terminate."""
    if not is_configured(creds):
        return {"skipped": True, "reason": "Datadog not configured"}
    hours = duration_seconds / 3600
    metrics = [
        {"metric": f"{PREFIX}.gpu.total_cost", "value": total_cost, "type": "gauge",
         "tags": {"provider": provider, "instance_id": instance_id}},
        {"metric": f"{PREFIX}.provisions.duration_s", "value": duration_seconds, "type": "gauge",
         "tags": {"instance_id": instance_id}},
    ]
    m_result = await submit_metrics_async(creds, metrics)
    e_result = await send_event_async(
        creds, title=f"GPU Terminated: {instance_id}",
        text=f"Provider: {provider}\nDuration: {hours:.1f}h\nTotal: ${total_cost:.2f}",
        alert_type="success",
        tags={"provider": provider, "instance_id": instance_id},
    )
    return {"metrics": m_result, "event": e_result}


async def push_cost_snapshot(creds: Dict[str, str]) -> Dict[str, Any]:
    """Pull current state from cost_tracker + price_intelligence and push to DD.

    Main periodic sync — call from `terradev datadog push` or a cron.
    """
    if not is_configured(creds):
        return {"skipped": True, "reason": "Datadog not configured"}

    metrics: List[Dict[str, Any]] = []

    # Cost tracker data
    try:
        from terradev_cli.core.cost_tracker import get_spend_summary, get_active_instances
        summary = get_spend_summary(days=30)
        active = get_active_instances()

        if summary["days"] > 0 and summary["total_provision_cost"] > 0:
            daily_avg = summary["total_provision_cost"] / summary["days"]
            metrics.append({"metric": f"{PREFIX}.gpu.monthly_projected",
                            "value": round(daily_avg * 30, 2), "type": "gauge", "tags": {}})

        from collections import Counter
        prov_counts: Dict[str, int] = Counter()
        for inst in active:
            prov_counts[inst.get("provider", "unknown")] += 1
        for prov, count in prov_counts.items():
            metrics.append({"metric": f"{PREFIX}.provisions.active",
                            "value": count, "type": "gauge", "tags": {"provider": prov}})
        for inst in active:
            metrics.append({"metric": f"{PREFIX}.gpu.cost_per_hour",
                            "value": inst.get("price_hr", 0), "type": "gauge",
                            "tags": {"provider": inst.get("provider", ""),
                                     "gpu_type": inst.get("gpu_type", ""),
                                     "region": inst.get("region", ""),
                                     "instance_id": inst.get("instance_id", ""),
                                     "spot": str(inst.get("spot", False)).lower()}})
    except Exception:
        pass

    # Price intelligence data
    try:
        from terradev_cli.core.price_intelligence import get_provider_ranking
        ranking = get_provider_ranking()
        for entry in ranking:
            prov = entry.get("provider", "")
            metrics.append({"metric": f"{PREFIX}.provider.reliability",
                            "value": entry.get("overall_score", 0), "type": "gauge",
                            "tags": {"provider": prov}})
            if entry.get("avg_quote_latency_ms"):
                metrics.append({"metric": f"{PREFIX}.provider.latency_ms",
                                "value": entry["avg_quote_latency_ms"], "type": "gauge",
                                "tags": {"provider": prov}})
    except Exception:
        pass

    if not metrics:
        return {"success": True, "metrics_sent": 0, "note": "No data to push"}
    return await submit_metrics_async(creds, metrics)


# ═══════════════════════════════════════════════════════════════════════
# TERRAFORM EXPORT — generate .tf files for Datadog provider
# ═══════════════════════════════════════════════════════════════════════

def generate_terraform_tfvars(creds: Dict[str, str]) -> str:
    return (
        f'# Auto-generated by Terradev — do not commit to version control\n'
        f'datadog_api_key = "{creds.get("datadog_api_key", "")}"\n'
        f'datadog_app_key = "{creds.get("datadog_app_key", "")}"\n'
        f'datadog_site    = "{_get_site(creds)}"\n'
    )


def generate_terraform_provider_block() -> str:
    return '''variable "datadog_api_key" {
  type      = string
  sensitive = true
}
variable "datadog_app_key" {
  type      = string
  sensitive = true
}
variable "datadog_site" {
  type    = string
  default = "datadoghq.com"
}

provider "datadog" {
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
  api_url = "https://api.${var.datadog_site}"
}
'''


def generate_terraform_monitors() -> str:
    blocks = []
    for key, tmpl in MONITOR_TEMPLATES.items():
        safe_key = key.replace("-", "_")
        thresholds = tmpl.get("options", {}).get("thresholds", {})
        crit = thresholds.get("critical", 0)
        warn = thresholds.get("warning", 0)
        tags_str = ", ".join(f'"{t}"' for t in tmpl.get("tags", []))
        msg = tmpl["message"].replace('"', '\\"').replace("\n", "\\n")
        blocks.append(f'''resource "datadog_monitor" "terradev_{safe_key}" {{
  name    = "{tmpl['name']}"
  type    = "{tmpl['type']}"
  query   = "{tmpl['query']}"
  message = "{msg}"

  monitor_thresholds {{
    critical = {crit}
    warning  = {warn}
  }}

  notify_no_data = false
  tags           = [{tags_str}]
  priority       = {tmpl.get("priority", 3)}
}}''')
    return "\n\n".join(blocks)


def generate_terraform_dashboard() -> str:
    dash = _gpu_cost_dashboard()
    widgets_json = json.dumps(dash["widgets"], indent=2)
    return f'''resource "datadog_dashboard_json" "terradev_gpu_finops" {{
  dashboard = jsonencode({{
    title       = "{dash['title']}"
    description = "{dash['description']}"
    layout_type = "ordered"
    tags        = {json.dumps(dash['tags'])}
    widgets     = jsondecode(<<-EOT
{widgets_json}
EOT
    )
  }})
}}'''


def generate_full_terraform_module(creds: Dict[str, str]) -> Dict[str, str]:
    """Generate a complete Terraform module for Datadog integration.

    Returns a dict of filename → content.
    """
    return {
        "provider.tf": generate_terraform_provider_block(),
        "monitors.tf": generate_terraform_monitors(),
        "dashboard.tf": generate_terraform_dashboard(),
        "terraform.tfvars": generate_terraform_tfvars(creds),
        "versions.tf": '''terraform {
  required_providers {
    datadog = {
      source  = "DataDog/datadog"
      version = "~> 3.0"
    }
  }
}
''',
    }
