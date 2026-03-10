#!/usr/bin/env python3
"""
Databricks Integration — MLOps, Jobs, Clusters & Model Serving for Terradev

Unified data + AI platform: GPU clusters, job orchestration, model serving,
MLflow (managed), Unity Catalog, and Feature Store.

API notes (verified from docs):
  - Auth: Authorization: Bearer <PAT> (Personal Access Token)
  - Alt auth: OAuth M2M (service principals), Azure AD tokens
  - Base URL: https://<workspace-id>.cloud.databricks.com
  - Jobs API 2.1: /api/2.1/jobs/* — create, run-now, list, runs/get
  - Clusters API 2.0: /api/2.0/clusters/* — create, get, start, delete
  - Model Serving 2.0: /api/2.0/serving-endpoints/* — CRUD + invocations
  - MLflow 2.0: /api/2.0/mlflow/* — same MLflow REST API, PAT auth
  - All responses are JSON

BYOAPI: PAT stored locally in ~/.terradev/credentials.json
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ── Credential management ─────────────────────────────────────────────

REQUIRED_CREDENTIALS = {
    "host": "Databricks workspace URL (e.g. https://dbc-abc123.cloud.databricks.com)",
    "token": "Personal Access Token (dapi...)",
}


def get_credential_prompts() -> List[Dict[str, str]]:
    return [
        {"key": "databricks_host", "prompt": "Databricks workspace URL", "required": True, "hide": False},
        {"key": "databricks_token", "prompt": "Databricks PAT (dapi...)", "required": True, "hide": True},
    ]


def _base_url(creds: Dict[str, str]) -> str:
    host = creds.get("databricks_host", "").rstrip("/")
    if not host.startswith("http"):
        host = f"https://{host}"
    return host


def _auth_headers(creds: Dict[str, str]) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {creds.get('databricks_token', '')}",
        "Content-Type": "application/json",
    }


def is_configured(creds: Dict[str, str]) -> bool:
    return bool(creds.get("databricks_host")) and bool(creds.get("databricks_token"))


def get_status_summary(creds: Dict[str, str]) -> Dict[str, Any]:
    return {
        "integration": "databricks", "name": "Databricks",
        "configured": is_configured(creds),
        "host": _base_url(creds),
        "token_set": bool(creds.get("databricks_token", "")),
    }


# ═══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HTTP (sync + async, zero SDK dependency)
# ═══════════════════════════════════════════════════════════════════════

def _request_sync(creds: Dict[str, str], method: str, path: str,
                  body: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
    import urllib.request
    import urllib.error
    import urllib.parse

    url = f"{_base_url(creds)}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    data = json.dumps(body).encode() if body else None

    try:
        req = urllib.request.Request(url, data=data, headers=_auth_headers(creds), method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"success": True, "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _request_async(creds: Dict[str, str], method: str, path: str,
                         body: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return _request_sync(creds, method, path, body, params)

    url = f"{_base_url(creds)}{path}"

    try:
        async with aiohttp.ClientSession() as s:
            kwargs: Dict[str, Any] = {
                "headers": _auth_headers(creds),
                "timeout": aiohttp.ClientTimeout(total=30),
            }
            if body:
                kwargs["json"] = body
            if params:
                kwargs["params"] = params

            async with s.request(method, url, **kwargs) as r:
                if r.status < 300:
                    data = await r.json(content_type=None)
                    return {"success": True, "data": data}
                text = await r.text()
                return {"success": False, "error": f"HTTP {r.status}: {text[:500]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# CONNECTION TEST
# ═══════════════════════════════════════════════════════════════════════

async def test_connection(creds: Dict[str, str]) -> Dict[str, Any]:
    """Test Databricks connectivity by listing clusters."""
    result = await _request_async(creds, "GET", "/api/2.0/clusters/list")
    if result.get("success"):
        clusters = result.get("data", {}).get("clusters", [])
        return {
            "status": "connected",
            "host": _base_url(creds),
            "clusters": len(clusters),
        }
    return {"status": "failed", "error": result.get("error", "Unknown error")}


# ═══════════════════════════════════════════════════════════════════════
# JOBS API (2.1)
# ═══════════════════════════════════════════════════════════════════════

async def list_jobs(creds: Dict[str, str], limit: int = 25, offset: int = 0) -> Dict[str, Any]:
    """List all jobs in the workspace."""
    return await _request_async(creds, "GET", "/api/2.1/jobs/list",
                                params={"limit": limit, "offset": offset})


async def get_job(creds: Dict[str, str], job_id: int) -> Dict[str, Any]:
    """Get job details."""
    return await _request_async(creds, "GET", "/api/2.1/jobs/get", params={"job_id": job_id})


async def create_job(creds: Dict[str, str], job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new job.

    job_config should match Databricks Jobs API 2.1 CreateJob schema.
    Minimum: {"name": "...", "tasks": [{"task_key": "...", ...}]}
    """
    return await _request_async(creds, "POST", "/api/2.1/jobs/create", body=job_config)


async def run_job(creds: Dict[str, str], job_id: int,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Trigger a job run (run-now)."""
    body: Dict[str, Any] = {"job_id": job_id}
    if params:
        body.update(params)
    return await _request_async(creds, "POST", "/api/2.1/jobs/run-now", body=body)


async def submit_run(creds: Dict[str, str], run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a one-time run (no saved job).

    Useful for ad-hoc training or preprocessing tasks.
    """
    return await _request_async(creds, "POST", "/api/2.1/jobs/runs/submit", body=run_config)


async def get_run(creds: Dict[str, str], run_id: int) -> Dict[str, Any]:
    """Get run status and details."""
    return await _request_async(creds, "GET", "/api/2.1/jobs/runs/get", params={"run_id": run_id})


async def list_runs(creds: Dict[str, str], job_id: Optional[int] = None,
                    limit: int = 25, offset: int = 0) -> Dict[str, Any]:
    """List job runs, optionally filtered by job_id."""
    p: Dict[str, Any] = {"limit": limit, "offset": offset}
    if job_id is not None:
        p["job_id"] = job_id
    return await _request_async(creds, "GET", "/api/2.1/jobs/runs/list", params=p)


async def cancel_run(creds: Dict[str, str], run_id: int) -> Dict[str, Any]:
    """Cancel an active run."""
    return await _request_async(creds, "POST", "/api/2.1/jobs/runs/cancel", body={"run_id": run_id})


# ═══════════════════════════════════════════════════════════════════════
# CLUSTERS API (2.0)
# ═══════════════════════════════════════════════════════════════════════

async def list_clusters(creds: Dict[str, str]) -> Dict[str, Any]:
    """List all clusters."""
    return await _request_async(creds, "GET", "/api/2.0/clusters/list")


async def get_cluster(creds: Dict[str, str], cluster_id: str) -> Dict[str, Any]:
    """Get cluster details."""
    return await _request_async(creds, "GET", "/api/2.0/clusters/get",
                                params={"cluster_id": cluster_id})


async def create_cluster(creds: Dict[str, str], cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new cluster.

    cluster_config should match Databricks Clusters API 2.0 schema.
    For GPU clusters, set node_type_id to a GPU instance (e.g. 'p4d.24xlarge').
    """
    return await _request_async(creds, "POST", "/api/2.0/clusters/create", body=cluster_config)


async def start_cluster(creds: Dict[str, str], cluster_id: str) -> Dict[str, Any]:
    """Start a terminated cluster."""
    return await _request_async(creds, "POST", "/api/2.0/clusters/start",
                                body={"cluster_id": cluster_id})


async def terminate_cluster(creds: Dict[str, str], cluster_id: str) -> Dict[str, Any]:
    """Terminate a running cluster."""
    return await _request_async(creds, "POST", "/api/2.0/clusters/delete",
                                body={"cluster_id": cluster_id})


# ═══════════════════════════════════════════════════════════════════════
# MODEL SERVING (2.0)
# ═══════════════════════════════════════════════════════════════════════

async def list_serving_endpoints(creds: Dict[str, str]) -> Dict[str, Any]:
    """List all model serving endpoints."""
    return await _request_async(creds, "GET", "/api/2.0/serving-endpoints")


async def get_serving_endpoint(creds: Dict[str, str], name: str) -> Dict[str, Any]:
    """Get serving endpoint details."""
    return await _request_async(creds, "GET", f"/api/2.0/serving-endpoints/{name}")


async def create_serving_endpoint(
    creds: Dict[str, str],
    name: str,
    model_name: str,
    model_version: str,
    *,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> Dict[str, Any]:
    """Create a model serving endpoint.

    workload_size: Small, Medium, Large
    """
    body = {
        "name": name,
        "config": {
            "served_models": [{
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
            }],
        },
    }
    return await _request_async(creds, "POST", "/api/2.0/serving-endpoints", body=body)


async def update_serving_endpoint(
    creds: Dict[str, str],
    name: str,
    model_name: str,
    model_version: str,
    *,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> Dict[str, Any]:
    """Update a serving endpoint's config (e.g. new model version)."""
    body = {
        "served_models": [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero,
        }],
    }
    return await _request_async(creds, "PUT", f"/api/2.0/serving-endpoints/{name}/config", body=body)


async def delete_serving_endpoint(creds: Dict[str, str], name: str) -> Dict[str, Any]:
    """Delete a serving endpoint."""
    return await _request_async(creds, "DELETE", f"/api/2.0/serving-endpoints/{name}")


async def query_serving_endpoint(
    creds: Dict[str, str],
    name: str,
    inputs: Any,
) -> Dict[str, Any]:
    """Query a model serving endpoint (inference).

    Supports both custom models and foundation models (OpenAI-compatible).
    """
    body: Dict[str, Any]
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict) and "role" in inputs[0]:
        # OpenAI-compatible chat format
        body = {"messages": inputs}
    elif isinstance(inputs, dict):
        body = inputs
    else:
        body = {"inputs": inputs}

    return await _request_async(creds, "POST",
                                f"/serving-endpoints/{name}/invocations", body=body)


# ═══════════════════════════════════════════════════════════════════════
# MLFLOW (Databricks-hosted, same REST API, PAT auth)
# ═══════════════════════════════════════════════════════════════════════

async def mlflow_list_experiments(creds: Dict[str, str], max_results: int = 50) -> Dict[str, Any]:
    """List MLflow experiments on Databricks."""
    return await _request_async(creds, "GET", "/api/2.0/mlflow/experiments/list",
                                params={"max_results": max_results})


async def mlflow_get_experiment(creds: Dict[str, str], experiment_id: str) -> Dict[str, Any]:
    """Get MLflow experiment details."""
    return await _request_async(creds, "GET", "/api/2.0/mlflow/experiments/get",
                                params={"experiment_id": experiment_id})


async def mlflow_search_runs(
    creds: Dict[str, str],
    experiment_ids: List[str],
    filter_string: str = "",
    max_results: int = 50,
    order_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search MLflow runs."""
    body: Dict[str, Any] = {
        "experiment_ids": experiment_ids,
        "max_results": max_results,
    }
    if filter_string:
        body["filter"] = filter_string
    if order_by:
        body["order_by"] = order_by
    return await _request_async(creds, "POST", "/api/2.0/mlflow/runs/search", body=body)


async def mlflow_list_registered_models(creds: Dict[str, str], max_results: int = 50) -> Dict[str, Any]:
    """List registered models in Model Registry."""
    return await _request_async(creds, "GET", "/api/2.0/mlflow/registered-models/list",
                                params={"max_results": max_results})


async def mlflow_get_model_versions(creds: Dict[str, str], name: str) -> Dict[str, Any]:
    """Get all versions of a registered model."""
    return await _request_async(creds, "POST", "/api/2.0/mlflow/registered-models/get-latest-versions",
                                body={"name": name})


async def mlflow_create_registered_model(
    creds: Dict[str, str], name: str,
    tags: Optional[List[Dict[str, str]]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Register a new model in Model Registry."""
    body: Dict[str, Any] = {"name": name}
    if tags:
        body["tags"] = tags
    if description:
        body["description"] = description
    return await _request_async(creds, "POST", "/api/2.0/mlflow/registered-models/create", body=body)


# ═══════════════════════════════════════════════════════════════════════
# TERRADEV-SPECIFIC HELPERS
# ═══════════════════════════════════════════════════════════════════════

def generate_gpu_training_job_config(
    *,
    name: str = "terradev-training",
    script_path: str = "/Workspace/training/train.py",
    cluster_node_type: str = "p4d.24xlarge",
    num_workers: int = 1,
    spark_version: str = "15.4.x-gpu-ml-scala2.12",
    python_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a Databricks job config for GPU training.

    Returns a dict ready to pass to create_job().
    """
    return {
        "name": name,
        "tasks": [{
            "task_key": "train",
            "spark_python_task": {
                "python_file": script_path,
                "parameters": python_params or [],
            },
            "new_cluster": {
                "spark_version": spark_version,
                "node_type_id": cluster_node_type,
                "num_workers": num_workers,
                "spark_conf": {
                    "spark.databricks.gpu.enabled": "true",
                },
                "spark_env_vars": {
                    "NCCL_DEBUG": "WARN",
                },
            },
        }],
        "timeout_seconds": 86400,
        "max_concurrent_runs": 1,
    }


def generate_model_deploy_config(
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str = "1",
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> Dict[str, Any]:
    """Generate a serving endpoint config for model deployment."""
    return {
        "name": endpoint_name,
        "config": {
            "served_models": [{
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
            }],
        },
    }


def get_databricks_setup_instructions() -> str:
    return """
Databricks Setup Instructions:

1. Get your workspace URL from the Databricks console:
   https://dbc-abc123.cloud.databricks.com

2. Generate a Personal Access Token (PAT):
   Settings -> Developer -> Access Tokens -> Generate New Token

3. Configure Terradev:
   terradev databricks configure \\
     --host https://dbc-abc123.cloud.databricks.com \\
     --token dapi...

Required Credentials:
- host: Databricks workspace URL
- token: Personal Access Token (PAT)

Usage:
terradev databricks test                           # Test connection
terradev databricks jobs                           # List jobs
terradev databricks run <job-id>                   # Trigger job run
terradev databricks run-status <run-id>            # Get run status
terradev databricks clusters                       # List clusters
terradev databricks serving-endpoints              # List model endpoints
terradev databricks deploy-model --name <model>    # Deploy model
terradev databricks query --endpoint <name>        # Query endpoint
terradev databricks mlflow experiments             # List MLflow experiments
terradev databricks mlflow models                  # List registered models
"""
