#!/usr/bin/env python3
"""
Helicone Integration — LLM Gateway & Observability for Terradev

Helicone is an open-source LLM proxy that sits between your app and LLM
providers, logging every request with latency, cost, tokens, and custom
properties.  Provides caching, rate limiting, retries, and fallbacks.

License: Apache 2.0 (self-hosted), Proprietary (Cloud)

API notes (verified from docs):
  - Gateway: Replace provider base URL with https://gateway.helicone.ai
  - Gateway auth: Helicone-Auth: Bearer <key> (NOT Authorization: Bearer)
  - API (reading): https://api.helicone.ai — Authorization: Bearer <key>
  - Key types: sk- (write/gateway), pk- (read/API queries)
  - EU variant: eu- prefix on keys, eu.helicone.ai domains
  - Request query: POST /v1/request/query with rich filter DSL
  - Properties: Helicone-Property-* headers for custom metadata
  - Features: Helicone-Cache-Enabled, Helicone-Retry-Enabled headers

BYOAPI: Keys stay local in ~/.terradev/credentials.json
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ── Credential management ─────────────────────────────────────────────

REQUIRED_CREDENTIALS = {
    "api_key": "Helicone API Key (sk-helicone-... for write, pk-helicone-... for read)",
}

OPTIONAL_CREDENTIALS = {
    "eu": "Use EU region (true/false, default: false)",
}


def get_credential_prompts() -> List[Dict[str, str]]:
    return [
        {"key": "helicone_api_key", "prompt": "Helicone API Key", "required": True, "hide": True},
        {"key": "helicone_eu", "prompt": "EU region? (true/false, default: false)", "required": False, "hide": False},
    ]


def _is_eu(creds: Dict[str, str]) -> bool:
    return creds.get("helicone_eu", "false").lower() in ("true", "1", "yes")


def _gateway_url(creds: Dict[str, str]) -> str:
    return "https://eu.gateway.helicone.ai" if _is_eu(creds) else "https://gateway.helicone.ai"


def _api_url(creds: Dict[str, str]) -> str:
    return "https://eu.api.helicone.ai" if _is_eu(creds) else "https://api.helicone.ai"


def _api_auth_headers(creds: Dict[str, str]) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {creds.get('helicone_api_key', '')}",
        "Content-Type": "application/json",
    }


def _gateway_auth_headers(creds: Dict[str, str]) -> Dict[str, str]:
    """Gateway uses Helicone-Auth header, NOT Authorization."""
    return {
        "Helicone-Auth": f"Bearer {creds.get('helicone_api_key', '')}",
    }


def is_configured(creds: Dict[str, str]) -> bool:
    return bool(creds.get("helicone_api_key"))


def get_status_summary(creds: Dict[str, str]) -> Dict[str, Any]:
    return {
        "integration": "helicone", "name": "Helicone",
        "configured": is_configured(creds),
        "region": "EU" if _is_eu(creds) else "US",
        "gateway_url": _gateway_url(creds),
        "api_url": _api_url(creds),
    }


# ═══════════════════════════════════════════════════════════════════════
# REQUEST LOG QUERIES (POST /v1/request/query)
# ═══════════════════════════════════════════════════════════════════════

def _build_request_query(
    *,
    limit: int = 20,
    offset: int = 0,
    model: Optional[str] = None,
    status_gte: Optional[int] = None,
    status_lte: Optional[int] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    user_id: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    include_inputs: bool = True,
    properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the filter body for POST /v1/request/query."""
    body: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "sort": {sort_by: sort_order},
        "includeInputs": include_inputs,
        "filter": {},
    }

    rmt_filter: Dict[str, Any] = {}
    req_filter: Dict[str, Any] = {}

    if model:
        rmt_filter["model"] = {"equals": model}
    if status_gte is not None:
        rmt_filter["status"] = {"gte": status_gte}
    if status_lte is not None:
        rmt_filter.setdefault("status", {})["lte"] = status_lte
    if created_after:
        rmt_filter["request_created_at"] = {"gte": created_after}
    if created_before:
        rmt_filter.setdefault("request_created_at", {})["lte"] = created_before
    if user_id:
        rmt_filter["user_id"] = {"equals": user_id}

    if properties:
        rmt_filter["properties"] = {k: {"equals": v} for k, v in properties.items()}

    if rmt_filter:
        body["filter"]["request_response_rmt"] = rmt_filter
    if req_filter:
        body["filter"]["request"] = req_filter

    return body


async def query_requests(creds: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """Query request logs from Helicone API."""
    try:
        import aiohttp
    except ImportError:
        return _query_requests_sync(creds, **kwargs)

    url = f"{_api_url(creds)}/v1/request/query"
    body = _build_request_query(**kwargs)

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body, headers=_api_auth_headers(creds),
                              timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    return {"success": True, "data": await r.json()}
                return {"success": False, "error": f"HTTP {r.status}: {(await r.text())[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _query_requests_sync(creds: Dict[str, str], **kwargs) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    url = f"{_api_url(creds)}/v1/request/query"
    body = _build_request_query(**kwargs)
    payload = json.dumps(body).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers=_api_auth_headers(creds), method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"success": True, "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# COST AGGREGATION
# ═══════════════════════════════════════════════════════════════════════

async def get_cost_summary(creds: Dict[str, str], hours: int = 24) -> Dict[str, Any]:
    """Aggregate cost data from recent requests."""
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    result = await query_requests(creds, limit=1000, created_after=since, include_inputs=False)

    if not result.get("success"):
        return result

    requests_data = result.get("data", [])
    if isinstance(requests_data, dict):
        requests_data = requests_data.get("data", [])

    total_cost = 0.0
    total_tokens = 0
    total_requests = 0
    by_model: Dict[str, Dict[str, Any]] = {}

    for req in requests_data:
        total_requests += 1
        cost = req.get("cost_usd") or req.get("response_cost") or 0
        tokens = req.get("total_tokens") or 0
        model = req.get("model") or req.get("request_model") or "unknown"

        total_cost += float(cost)
        total_tokens += int(tokens)

        if model not in by_model:
            by_model[model] = {"cost": 0.0, "tokens": 0, "requests": 0}
        by_model[model]["cost"] += float(cost)
        by_model[model]["tokens"] += int(tokens)
        by_model[model]["requests"] += 1

    return {
        "success": True,
        "period_hours": hours,
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "total_requests": total_requests,
        "by_model": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_model.items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# FEEDBACK / SCORING
# ═══════════════════════════════════════════════════════════════════════

async def submit_feedback(creds: Dict[str, str], request_id: str, rating: bool) -> Dict[str, Any]:
    """Submit feedback (thumbs up/down) for a Helicone request."""
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}

    url = f"{_api_url(creds)}/v1/request/{request_id}/feedback"
    body = {"rating": rating}

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body, headers=_api_auth_headers(creds),
                              timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status < 300:
                    return {"success": True, "request_id": request_id, "rating": rating}
                return {"success": False, "error": f"HTTP {r.status}: {(await r.text())[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# PROPERTY UPDATES
# ═══════════════════════════════════════════════════════════════════════

async def update_request_property(
    creds: Dict[str, str], request_id: str, key: str, value: str,
) -> Dict[str, Any]:
    """Update a custom property on an existing request."""
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp required"}

    url = f"{_api_url(creds)}/v1/request/{request_id}/property"
    body = {"key": key, "value": value}

    try:
        async with aiohttp.ClientSession() as s:
            async with s.put(url, json=body, headers=_api_auth_headers(creds),
                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status < 300:
                    return {"success": True}
                return {"success": False, "error": f"HTTP {r.status}: {(await r.text())[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# GATEWAY CONFIGURATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def generate_gateway_config(
    creds: Dict[str, str],
    *,
    provider_base_url: str = "https://api.openai.com",
    cache_enabled: bool = False,
    retry_enabled: bool = False,
    retry_count: int = 3,
    rate_limit: Optional[str] = None,
    custom_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate configuration for routing LLM requests through Helicone gateway.

    Returns a dict with headers and base_url to use instead of the provider's.
    """
    gateway = _gateway_url(creds)
    headers = dict(_gateway_auth_headers(creds))

    # Helicone routes to the target provider via the URL path
    # e.g., gateway.helicone.ai/v1/... proxies to api.openai.com/v1/...
    headers["Helicone-Target-Url"] = provider_base_url

    if cache_enabled:
        headers["Helicone-Cache-Enabled"] = "true"

    if retry_enabled:
        headers["Helicone-Retry-Enabled"] = "true"
        headers["helicone-retry-num"] = str(retry_count)

    if rate_limit:
        headers["Helicone-RateLimit-Policy"] = rate_limit

    if custom_properties:
        for k, v in custom_properties.items():
            headers[f"Helicone-Property-{k}"] = v

    return {
        "base_url": f"{gateway}/v1",
        "headers": headers,
        "original_provider": provider_base_url,
        "features": {
            "cache": cache_enabled,
            "retry": retry_enabled,
            "rate_limit": rate_limit,
        },
    }


def generate_vllm_gateway_env(
    creds: Dict[str, str],
    vllm_endpoint: str = "http://localhost:8000",
) -> Dict[str, str]:
    """Generate env vars for routing vLLM inference through Helicone.

    For use with OpenAI-compatible clients pointing at vLLM.
    """
    gateway = _gateway_url(creds)
    api_key = creds.get("helicone_api_key", "")
    return {
        "OPENAI_API_BASE": f"{gateway}/v1",
        "HELICONE_AUTH": f"Bearer {api_key}",
        "HELICONE_TARGET_URL": vllm_endpoint,
        "HELICONE_CACHE_ENABLED": "false",
        "HELICONE_PROPERTY_SOURCE": "terradev",
    }


def generate_gateway_snippet(creds: Dict[str, str], provider: str = "openai") -> str:
    """Generate a Python code snippet for using Helicone gateway."""
    gateway = _gateway_url(creds)
    api_key = creds.get("helicone_api_key", "")

    if provider == "openai":
        return (
            f'from openai import OpenAI\n\n'
            f'client = OpenAI(\n'
            f'    api_key="your-openai-key",\n'
            f'    base_url="{gateway}/v1",\n'
            f'    default_headers={{\n'
            f'        "Helicone-Auth": "Bearer {api_key[:8]}...",\n'
            f'    }}\n'
            f')\n\n'
            f'response = client.chat.completions.create(\n'
            f'    model="gpt-4o",\n'
            f'    messages=[{{"role": "user", "content": "Hello"}}]\n'
            f')\n'
        )
    elif provider == "vllm":
        return (
            f'from openai import OpenAI\n\n'
            f'# Route vLLM through Helicone for logging\n'
            f'client = OpenAI(\n'
            f'    api_key="not-needed",\n'
            f'    base_url="{gateway}/v1",\n'
            f'    default_headers={{\n'
            f'        "Helicone-Auth": "Bearer {api_key[:8]}...",\n'
            f'        "Helicone-Target-Url": "http://your-vllm:8000",\n'
            f'        "Helicone-Property-Source": "terradev",\n'
            f'    }}\n'
            f')\n\n'
            f'response = client.chat.completions.create(\n'
            f'    model="your-model",\n'
            f'    messages=[{{"role": "user", "content": "Hello"}}]\n'
            f')\n'
        )
    return f"# Unsupported provider: {provider}"


def get_helicone_setup_instructions() -> str:
    return """
Helicone Setup Instructions:

1. Sign up at https://www.helicone.ai
   Or self-host: docker compose up -d (from helicone repo)

2. Get your API keys from the Helicone dashboard:
   - sk-helicone-... (write key, for gateway proxy)
   - pk-helicone-... (read key, for API queries)

3. Configure Terradev:
   terradev helicone configure --api-key sk-helicone-...

4. Route LLM requests through Helicone (zero code change):
   # Just change your base URL:
   # Before: https://api.openai.com/v1
   # After:  https://gateway.helicone.ai/v1
   # Add header: Helicone-Auth: Bearer sk-helicone-...

Required Credentials:
- api_key: Helicone API Key (sk-helicone-... or pk-helicone-...)
- eu: Whether to use EU region (optional, default: false)

Usage:
terradev helicone test                        # Test connection
terradev helicone requests [--model gpt-4o]   # Query request logs
terradev helicone costs [--hours 24]          # Aggregate costs
terradev helicone gateway-config              # Generate gateway config
terradev helicone snippet [--provider openai] # Code snippet
"""


async def test_connection(creds: Dict[str, str]) -> Dict[str, Any]:
    """Test Helicone API connectivity by querying 1 request."""
    result = await query_requests(creds, limit=1)
    if result.get("success"):
        return {
            "status": "connected",
            "api_url": _api_url(creds),
            "gateway_url": _gateway_url(creds),
            "region": "EU" if _is_eu(creds) else "US",
        }
    return {"status": "failed", "error": result.get("error", "Unknown error")}
