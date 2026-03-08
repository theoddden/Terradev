#!/usr/bin/env python3
"""
MLflow Service Integration for Terradev
Manages MLflow experiments, runs, and model registry.

Terradev-specific features:
  - _request() with exponential backoff retry (3 attempts, jitter)
  - _ensure_session() — deduplicated session with optional BasicAuth
  - log_terradev_run() — auto-logs GPU type, provider, cost, duration as MLflow params
  - register_terradev_model() — tags models with provenance (checkpoint, training job)
"""

import logging
import os
import json
import asyncio
import random
import sqlite3
import aiohttp
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# cost_tracking.db for Terradev-specific logging
_COST_DB = Path.home() / ".terradev" / "cost_tracking.db"

# Retry defaults
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 10.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str
    username: Optional[str] = None
    password: Optional[str] = None
    experiment_name: Optional[str] = None
    registry_uri: Optional[str] = None


class MLflowService:
    """MLflow integration service for experiment tracking and model registry"""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    # ── Session & request helpers ────────────────────────────────────

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create the aiohttp session once, with auth if configured."""
        if self.session is None or self.session.closed:
            kwargs: Dict[str, Any] = {}
            if self.config.username and self.config.password:
                kwargs["auth"] = aiohttp.BasicAuth(self.config.username, self.config.password)
            self.session = aiohttp.ClientSession(**kwargs)
        return self.session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict] = None,
        json_body: Optional[Any] = None,
        timeout: float = 30,
        retries: int = _MAX_RETRIES,
    ) -> Any:
        """HTTP request with exponential backoff and jitter."""
        session = self._ensure_session()
        url = f"{self.config.tracking_uri}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                async with session.request(
                    method, url,
                    params=params,
                    json=json_body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                        wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                        logger.warning(
                            "MLflow %s %s → %d, retrying in %.1fs (attempt %d/%d)",
                            method.upper(), path, resp.status, wait, attempt + 1, retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"MLflow API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    logger.warning(
                        "MLflow %s %s network error: %s, retrying in %.1fs",
                        method.upper(), path, e, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── Core API methods ─────────────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        """Test MLflow connection and get server info"""
        try:
            data = await self._request("GET", "/api/2.0/mlflow/experiments/list", timeout=10)
            return {
                "status": "connected",
                "tracking_uri": self.config.tracking_uri,
                "experiments_count": len(data.get("experiments", [])),
                "registry_uri": self.config.registry_uri
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        data = await self._request("GET", "/api/2.0/mlflow/experiments/list")
        return data.get("experiments", [])
    
    async def create_experiment(self, name: str, artifact_location: Optional[str] = None) -> Dict[str, Any]:
        """Create a new experiment"""
        payload: Dict[str, Any] = {"name": name}
        if artifact_location:
            payload["artifact_location"] = artifact_location
        return await self._request("POST", "/api/2.0/mlflow/experiments/create", json_body=payload)
    
    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details"""
        return await self._request("GET", "/api/2.0/mlflow/experiments/get", params={"experiment_id": experiment_id})
    
    async def list_runs(self, experiment_ids: Optional[List[str]] = None, max_results: int = 1000) -> List[Dict[str, Any]]:
        """List runs in experiments"""
        payload: Dict[str, Any] = {"max_results": max_results}
        if experiment_ids:
            eid_list = ', '.join(f'"{eid}"' for eid in experiment_ids)
            payload["filter"] = f"experiment_id IN ({eid_list})"
        data = await self._request("POST", "/api/2.0/mlflow/runs/search", json_body=payload)
        return data.get("runs", [])
    
    async def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run details"""
        return await self._request("GET", "/api/2.0/mlflow/runs/get", params={"run_id": run_id})
    
    async def log_run(self, run_id: str, metrics: Dict[str, float], params: Dict[str, Any], tags: Dict[str, str]) -> Dict[str, Any]:
        """Log metrics, parameters, and tags to a run"""
        # Log metrics
        if metrics:
            await self._request("POST", "/api/2.0/mlflow/runs/log-metrics", json_body={
                "run_id": run_id,
                "metrics": [{"key": k, "value": v, "timestamp": int(datetime.now().timestamp() * 1000)} for k, v in metrics.items()],
            })
        # Log parameters
        if params:
            await self._request("POST", "/api/2.0/mlflow/runs/log-parameters", json_body={
                "run_id": run_id,
                "parameters": [{"key": k, "value": str(v)} for k, v in params.items()],
            })
        # Log tags
        if tags:
            await self._request("POST", "/api/2.0/mlflow/runs/set-tags", json_body={
                "run_id": run_id,
                "tags": tags,
            })
        return {"status": "logged", "run_id": run_id}
    
    async def list_registered_models(self) -> List[Dict[str, Any]]:
        """List registered models"""
        data = await self._request("GET", "/api/2.0/mlflow/registered-models/list")
        return data.get("registered_models", [])
    
    async def create_model_version(self, name: str, source: str, run_id: Optional[str] = None, description: str = "") -> Dict[str, Any]:
        """Create a new model version"""
        return await self._request("POST", "/api/2.0/mlflow/model-versions/create", json_body={
            "name": name,
            "source": source,
            "run_id": run_id,
            "description": description,
        })
    
    def get_tracking_config(self) -> Dict[str, str]:
        """Get environment variables for MLflow tracking"""
        config = {
            "MLFLOW_TRACKING_URI": self.config.tracking_uri
        }
        
        if self.config.username:
            config["MLFLOW_TRACKING_USERNAME"] = self.config.username
            
        if self.config.password:
            config["MLFLOW_TRACKING_PASSWORD"] = self.config.password
            
        if self.config.registry_uri:
            config["MLFLOW_REGISTRY_URI"] = self.config.registry_uri
            
        return config
    
    # ── Terradev-specific: auto-logging & provenance ────────────────

    async def log_terradev_run(
        self,
        experiment_name: str,
        gpu_type: str,
        provider: str,
        region: str,
        price_hr: float,
        duration_hrs: float,
        *,
        instance_id: Optional[str] = None,
        spot: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create an MLflow run with full Terradev GPU provenance auto-logged.

        Logs GPU type, provider, region, cost/hr, total cost, duration, spot
        as MLflow params so every training run is traceable to the
        infrastructure that produced it.  Also queries cost_tracking.db
        for cumulative spend if available.
        """
        # Ensure experiment exists
        experiments = await self.list_experiments()
        exp_id = None
        for exp in experiments:
            if exp.get("name") == experiment_name:
                exp_id = exp.get("experiment_id")
                break
        if not exp_id:
            result = await self.create_experiment(experiment_name)
            exp_id = result.get("experiment_id")

        # Create run
        run_result = await self._request("POST", "/api/2.0/mlflow/runs/create", json_body={
            "experiment_id": exp_id,
            "start_time": int(datetime.now().timestamp() * 1000),
            "tags": [
                {"key": "mlflow.source.name", "value": "terradev"},
                {"key": "terradev.provider", "value": provider},
                {"key": "terradev.gpu_type", "value": gpu_type},
                {"key": "terradev.managed", "value": "true"},
            ],
        })
        run_id = run_result.get("run", {}).get("info", {}).get("run_id")
        if not run_id:
            raise Exception(f"Failed to create MLflow run: {run_result}")

        # Build Terradev params
        total_cost = round(price_hr * duration_hrs, 4)
        td_params: Dict[str, Any] = {
            "terradev.gpu_type": gpu_type,
            "terradev.provider": provider,
            "terradev.region": region,
            "terradev.price_hr": price_hr,
            "terradev.duration_hrs": round(duration_hrs, 4),
            "terradev.total_cost": total_cost,
            "terradev.spot": str(spot),
        }
        if instance_id:
            td_params["terradev.instance_id"] = instance_id

        # Add cumulative spend from cost_tracking.db
        cumulative = self._get_cumulative_spend(provider)
        if cumulative is not None:
            td_params["terradev.cumulative_spend"] = round(cumulative, 2)

        if extra_params:
            td_params.update(extra_params)

        td_metrics: Dict[str, float] = {"terradev.run_cost": total_cost}
        if extra_metrics:
            td_metrics.update(extra_metrics)

        td_tags: Dict[str, str] = {
            "terradev.provider": provider,
            "terradev.gpu_type": gpu_type,
        }
        if extra_tags:
            td_tags.update(extra_tags)

        await self.log_run(run_id, td_metrics, td_params, td_tags)

        return {
            "status": "logged",
            "run_id": run_id,
            "experiment_id": exp_id,
            "total_cost": total_cost,
            "params_logged": list(td_params.keys()),
        }

    async def register_terradev_model(
        self,
        model_name: str,
        source: str,
        *,
        run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        training_job_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
        provider: Optional[str] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a model version with Terradev provenance tags.

        Tags the model version with which checkpoint produced it,
        which training job, GPU type, and provider — full lineage
        from infrastructure to model artifact.
        """
        # Build rich description
        prov_lines = [description] if description else []
        if checkpoint_id:
            prov_lines.append(f"Checkpoint: {checkpoint_id}")
        if training_job_id:
            prov_lines.append(f"Training Job: {training_job_id}")
        if gpu_type:
            prov_lines.append(f"GPU: {gpu_type}")
        if provider:
            prov_lines.append(f"Provider: {provider}")
        full_desc = " | ".join(prov_lines) if prov_lines else ""

        # Ensure registered model exists
        models = await self.list_registered_models()
        model_exists = any(m.get("name") == model_name for m in models)
        if not model_exists:
            await self._request("POST", "/api/2.0/mlflow/registered-models/create", json_body={
                "name": model_name,
                "description": f"Terradev-managed model: {full_desc}",
                "tags": [
                    {"key": "terradev.managed", "value": "true"},
                ],
            })

        # Create model version
        version_result = await self.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            description=full_desc,
        )

        # Tag the version with provenance
        version = version_result.get("model_version", {}).get("version")
        if version:
            tag_pairs = [("terradev.managed", "true")]
            if checkpoint_id:
                tag_pairs.append(("terradev.checkpoint_id", checkpoint_id))
            if training_job_id:
                tag_pairs.append(("terradev.training_job_id", training_job_id))
            if gpu_type:
                tag_pairs.append(("terradev.gpu_type", gpu_type))
            if provider:
                tag_pairs.append(("terradev.provider", provider))
            for key, value in tag_pairs:
                try:
                    await self._request("POST", "/api/2.0/mlflow/model-versions/set-tag", json_body={
                        "name": model_name,
                        "version": version,
                        "key": key,
                        "value": value,
                    })
                except Exception as e:
                    logger.warning("Failed to set model version tag %s: %s", key, e)

        return {
            "status": "registered",
            "model_name": model_name,
            "version": version,
            "provenance": {
                "checkpoint_id": checkpoint_id,
                "training_job_id": training_job_id,
                "gpu_type": gpu_type,
                "provider": provider,
                "run_id": run_id,
            },
        }

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_cumulative_spend(provider: Optional[str] = None) -> Optional[float]:
        """Query cost_tracking.db for cumulative spend."""
        if not _COST_DB.exists():
            return None
        try:
            conn = sqlite3.connect(str(_COST_DB))
            if provider:
                row = conn.execute(
                    "SELECT SUM(price_hr * CAST((julianday(COALESCE(end_ts, datetime('now'))) - julianday(ts)) * 24 AS REAL)) "
                    "FROM provisions WHERE provider = ?",
                    (provider,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT SUM(price_hr * CAST((julianday(COALESCE(end_ts, datetime('now'))) - julianday(ts)) * 24 AS REAL)) "
                    "FROM provisions"
                ).fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            logger.debug("Failed to read cumulative spend: %s", e)
            return None

    # ── Existing export method ───────────────────────────────────────

    async def export_experiment_data(self, experiment_id: str, format: str = "json") -> str:
        """Export experiment data"""
        try:
            runs = await self.list_runs([experiment_id])
            
            if format.lower() == "json":
                return json.dumps(runs, indent=2)
            elif format.lower() == "csv":
                import csv
                import io
                
                if not runs:
                    return ""
                
                output = io.StringIO()
                fieldnames = ["run_id", "experiment_id", "status", "start_time", "end_time", "artifact_uri"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for run in runs:
                    info = run.get("info", {})
                    writer.writerow({
                        "run_id": info.get("run_id", ""),
                        "experiment_id": info.get("experiment_id", ""),
                        "status": info.get("status", ""),
                        "start_time": info.get("start_time", ""),
                        "end_time": info.get("end_time", ""),
                        "artifact_uri": info.get("artifact_uri", "")
                    })
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise Exception(f"Failed to export experiment data: {e}")


def create_mlflow_service_from_credentials(credentials: Dict[str, str]) -> MLflowService:
    """Create MLflowService from credential dictionary"""
    config = MLflowConfig(
        tracking_uri=credentials["tracking_uri"],
        username=credentials.get("username"),
        password=credentials.get("password"),
        experiment_name=credentials.get("experiment_name"),
        registry_uri=credentials.get("registry_uri")
    )
    
    return MLflowService(config)


def get_mlflow_setup_instructions() -> str:
    """Get setup instructions for MLflow"""
    return """
🚀 MLflow Setup Instructions:

1. Install MLflow:
   pip install mlflow

2. Start MLflow tracking server:
   # Basic server
   mlflow server
   
   # With authentication
   mlflow server --app-name basic-auth
   
   # With custom host and port
   mlflow server --host 0.0.0.0 --port 5000

3. Set up authentication (optional but recommended):
   export MLFLOW_FLASK_SECRET_KEY="your-secret-key"
   export MLFLOW_TRACKING_USERNAME="admin"
   export MLFLOW_TRACKING_PASSWORD="your-password"

4. Configure Terradev with your MLflow credentials:
   terradev configure --provider mlflow --tracking-uri http://localhost:5000 --username admin --password your-password

📋 Required Credentials:
- tracking_uri: MLflow tracking server URI (required)
- username: Username for basic auth (optional)
- password: Password for basic auth (optional)
- experiment_name: Default experiment name (optional)
- registry_uri: Model registry URI (optional)

💡 Usage Examples:
# Test connection
terradev mlflow test

# List experiments
terradev mlflow list-experiments

# Create a new experiment
terradev mlflow create-experiment --name my-experiment

# List runs in an experiment
terradev mlflow list-runs --experiment-id your-experiment-id

# Log metrics to a run
terradev mlflow log-run --run-id your-run-id --metrics '{"accuracy": 0.95, "loss": 0.05}'

# List registered models
terradev mlflow list-models

# Create a model version
terradev mlflow create-model --name my-model --source s3://my-bucket/model

# Export experiment data
terradev mlflow export --experiment-id your-experiment-id --format json > experiment.json

🔗 Environment Variables for Tracking:
Add these to your ML training scripts:
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="your-password"

Then in your Python code:
import mlflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)

🐳 Docker MLflow Server:
docker run -p 5000:5000 --rm -e MLFLOW_FLASK_SECRET_KEY=your-key python:3.9-slim bash -c "
pip install mlflow && mlflow server --host 0.0.0.0
"

☸️ Kubernetes MLflow:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow-server
        image: python:3.9-slim
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_FLASK_SECRET_KEY
          value: "your-secret-key"
        command:
        - bash
        - -c
        - |
          pip install mlflow &&
          mlflow server --host 0.0.0.0 --port 5000
"""
