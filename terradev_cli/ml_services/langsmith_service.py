#!/usr/bin/env python3
"""
LangSmith Service Integration for Terradev
Manages LangSmith tracing and evaluation workflows.

Terradev-specific features:
  - _request() with exponential backoff retry (3 attempts, jitter)
  - inject_terradev_metadata() — auto-inject GPU provision metadata into run metadata
  - correlate_runs_with_gpu_metrics() — join LangSmith runs with cost_tracking.db
    to compute cost-per-run, GPU utilization per run, and provider breakdown
"""

import logging
import os
import json
import asyncio
import random
import sqlite3
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# cost_tracking.db for GPU-correlated tracing
_COST_DB = Path.home() / ".terradev" / "cost_tracking.db"

# Retry defaults
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5     # seconds
_BACKOFF_MAX = 10.0     # cap
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class LangSmithConfig:
    """LangSmith configuration"""
    api_key: str
    endpoint: str = "https://api.smith.langchain.com"
    workspace_id: Optional[str] = None
    project_name: Optional[str] = None


class LangSmithService:
    """LangSmith integration service for tracing and evaluation"""
    
    def __init__(self, config: LangSmithConfig):
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
        """Lazily create the aiohttp session once."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
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
        """HTTP request with exponential backoff and jitter.

        Retries on 429 / 5xx.  Returns parsed JSON on success,
        raises on non-retryable errors.
        """
        session = self._ensure_session()
        url = f"{self.config.endpoint}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                async with session.request(
                    method, url,
                    params=params,
                    json=json_body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
                    if resp.status in _RETRYABLE_STATUSES and attempt < retries - 1:
                        wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                        logger.warning(
                            "LangSmith %s %s → %d, retrying in %.1fs (attempt %d/%d)",
                            method.upper(), path, resp.status, wait, attempt + 1, retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    error_text = await resp.text()
                    raise Exception(f"LangSmith API {resp.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), _BACKOFF_MAX)
                    logger.warning(
                        "LangSmith %s %s network error: %s, retrying in %.1fs",
                        method.upper(), path, e, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc or Exception("Request failed after retries")

    # ── Core API methods ─────────────────────────────────────────────

    async def test_connection(self) -> Dict[str, Any]:
        """Test LangSmith connection and get workspace info"""
        try:
            data = await self._request("GET", "/v1/sessions", timeout=10)
            return {
                "status": "connected",
                "workspace_id": self.config.workspace_id,
                "endpoint": self.config.endpoint,
                "sessions_count": len(data.get("sessions", []))
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the workspace"""
        data = await self._request("GET", "/v1/projects")
        return data.get("projects", [])
    
    async def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new LangSmith project"""
        return await self._request("POST", "/v1/projects", json_body={
            "name": name,
            "description": description,
        })
    
    async def list_runs(self, project_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs in a project or workspace"""
        params: Dict[str, Any] = {"limit": limit}
        if project_name:
            params["project_name"] = project_name
        elif self.config.project_name:
            params["project_name"] = self.config.project_name
        data = await self._request("GET", "/v1/runs", params=params)
        return data.get("runs", [])
    
    async def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run"""
        return await self._request("GET", f"/v1/runs/{run_id}")
    
    async def create_dataset(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new dataset for evaluation"""
        return await self._request("POST", "/v1/datasets", json_body={
            "name": name,
            "description": description,
        })
    
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in the workspace"""
        data = await self._request("GET", "/v1/datasets")
        return data.get("datasets", [])
    
    def get_tracing_config(self) -> Dict[str, str]:
        """Get environment variables for LangSmith tracing"""
        config = {
            "LANGSMITH_API_KEY": self.config.api_key,
            "LANGSMITH_ENDPOINT": self.config.endpoint
        }
        
        if self.config.workspace_id:
            config["LANGSMITH_WORKSPACE_ID"] = self.config.workspace_id
            
        if self.config.project_name:
            config["LANGSMITH_PROJECT"] = self.config.project_name
            
        return config
    
    # ── Terradev-specific: GPU-correlated tracing ──────────────────────

    async def inject_terradev_metadata(
        self,
        run_id: str,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Auto-inject Terradev provision metadata into a LangSmith run.

        Pulls the active (or specified) provision from cost_tracking.db and
        patches the run's metadata with GPU type, provider, cost/hr, region,
        and instance ID — so every LangSmith trace is correlated with the
        infrastructure that produced it.
        """
        metadata = self._get_provision_metadata(instance_id)
        if not metadata:
            return {"status": "skipped", "reason": "No active Terradev provisions found"}

        # Patch the run's extra/metadata via the LangSmith update-run endpoint
        patch_body = {
            "extra": {
                "terradev": metadata,
            },
            "tags": [
                f"gpu:{metadata.get('gpu_type', 'unknown')}",
                f"provider:{metadata.get('provider', 'unknown')}",
                "terradev-managed",
            ],
        }
        result = await self._request("PATCH", f"/v1/runs/{run_id}", json_body=patch_body)
        return {
            "status": "injected",
            "run_id": run_id,
            "terradev_metadata": metadata,
            "response": result,
        }

    async def correlate_runs_with_gpu_metrics(
        self,
        project_name: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Join LangSmith runs with Terradev's cost_tracking.db.

        For each run, computes:
          - cost_per_run: estimated GPU cost during that run's wall-clock time
          - gpu_type, provider, region from the provision that was active
          - total_runs, total_cost, cost breakdown by provider

        Returns a summary dict suitable for dashboards or CLI output.
        """
        runs = await self.list_runs(project_name=project_name, limit=limit)
        provisions = self._get_all_provisions()

        correlated = []
        total_cost = 0.0
        provider_costs: Dict[str, float] = {}

        for run in runs:
            start = run.get("start_time") or run.get("start_dt")
            end = run.get("end_time") or run.get("end_dt")
            if not start or not end:
                correlated.append({**run, "terradev_cost": None, "terradev_provision": None})
                continue

            # Find the provision that was active during this run
            matched_prov = None
            for prov in provisions:
                prov_start = prov.get("ts", "")
                prov_end = prov.get("end_ts") or "9999-12-31"
                if prov_start <= start and prov_end >= end:
                    matched_prov = prov
                    break

            if matched_prov:
                # Estimate cost: price_hr * (run duration in hours)
                try:
                    from datetime import datetime as _dt
                    t0 = _dt.fromisoformat(start.replace("Z", "+00:00"))
                    t1 = _dt.fromisoformat(end.replace("Z", "+00:00"))
                    hours = (t1 - t0).total_seconds() / 3600
                    run_cost = round(matched_prov.get("price_hr", 0) * hours, 6)
                except Exception:
                    hours = 0
                    run_cost = 0.0

                total_cost += run_cost
                prov_name = matched_prov.get("provider", "unknown")
                provider_costs[prov_name] = provider_costs.get(prov_name, 0) + run_cost

                correlated.append({
                    "run_id": run.get("id"),
                    "run_name": run.get("name"),
                    "start_time": start,
                    "end_time": end,
                    "duration_hrs": round(hours, 4),
                    "terradev_cost": run_cost,
                    "terradev_provision": {
                        "instance_id": matched_prov.get("instance_id"),
                        "provider": prov_name,
                        "gpu_type": matched_prov.get("gpu_type"),
                        "region": matched_prov.get("region"),
                        "price_hr": matched_prov.get("price_hr"),
                    },
                })
            else:
                correlated.append({
                    "run_id": run.get("id"),
                    "run_name": run.get("name"),
                    "start_time": start,
                    "end_time": end,
                    "terradev_cost": None,
                    "terradev_provision": None,
                })

        return {
            "total_runs": len(correlated),
            "correlated_runs": sum(1 for r in correlated if r.get("terradev_cost") is not None),
            "total_gpu_cost": round(total_cost, 4),
            "cost_by_provider": provider_costs,
            "runs": correlated,
        }

    # ── Private helpers for cost_tracking.db queries ─────────────────

    @staticmethod
    def _get_provision_metadata(instance_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Query cost_tracking.db for a provision's metadata."""
        if not _COST_DB.exists():
            return None
        try:
            conn = sqlite3.connect(str(_COST_DB))
            conn.row_factory = sqlite3.Row
            if instance_id:
                row = conn.execute(
                    "SELECT * FROM provisions WHERE instance_id = ? ORDER BY ts DESC LIMIT 1",
                    (instance_id,),
                ).fetchone()
            else:
                # Most recent active provision
                row = conn.execute(
                    "SELECT * FROM provisions WHERE status IN ('provisioning', 'running') "
                    "ORDER BY ts DESC LIMIT 1"
                ).fetchone()
            conn.close()
            if row:
                return {
                    "instance_id": row["instance_id"],
                    "provider": row["provider"],
                    "gpu_type": row["gpu_type"],
                    "region": row["region"],
                    "price_hr": row["price_hr"],
                    "spot": bool(row["spot"]),
                    "status": row["status"],
                }
        except Exception as e:
            logger.debug("Failed to read provision metadata: %s", e)
        return None

    @staticmethod
    def _get_all_provisions() -> List[Dict[str, Any]]:
        """Get all provisions from cost_tracking.db for correlation."""
        if not _COST_DB.exists():
            return []
        try:
            conn = sqlite3.connect(str(_COST_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM provisions ORDER BY ts DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("Failed to read provisions: %s", e)
            return []

    # ── Existing export method ───────────────────────────────────────

    async def export_runs(self, project_name: Optional[str] = None, format: str = "json") -> str:
        """Export runs data"""
        try:
            runs = await self.list_runs(project_name, limit=1000)
            
            if format.lower() == "json":
                return json.dumps(runs, indent=2)
            elif format.lower() == "csv":
                import csv
                import io
                
                if not runs:
                    return ""
                
                output = io.StringIO()
                fieldnames = ["id", "name", "start_time", "end_time", "status", "error"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for run in runs:
                    writer.writerow({
                        "id": run.get("id", ""),
                        "name": run.get("name", ""),
                        "start_time": run.get("start_time", ""),
                        "end_time": run.get("end_time", ""),
                        "status": run.get("status", ""),
                        "error": run.get("error", "")
                    })
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise Exception(f"Failed to export runs: {e}")


def create_langsmith_service_from_credentials(credentials: Dict[str, str]) -> LangSmithService:
    """Create LangSmithService from credential dictionary"""
    config = LangSmithConfig(
        api_key=credentials["api_key"],
        endpoint=credentials.get("endpoint", "https://api.smith.langchain.com"),
        workspace_id=credentials.get("workspace_id"),
        project_name=credentials.get("project_name")
    )
    
    return LangSmithService(config)


def get_langsmith_setup_instructions() -> str:
    """Get setup instructions for LangSmith"""
    return """
🚀 LangSmith Setup Instructions:

1. Create a LangSmith account:
   - Go to https://smith.langchain.com
   - Sign up for a free account

2. Create an API key:
   - Navigate to Settings → API Keys
   - Click "Create API Key"
   - Choose scope (organization or workspace-scoped)
   - Set expiration (recommended: 90 days)
   - Copy the API key

3. Find your Workspace ID:
   - In LangSmith UI, go to Settings
   - Your Workspace ID is shown in the overview
   - Or use the API: curl -H "Authorization: Bearer YOUR_API_KEY" https://api.smith.langchain.com/v1/workspaces

4. Configure Terradev with your LangSmith credentials:
   terradev configure --provider langsmith --api-key YOUR_API_KEY --workspace-id YOUR_WORKSPACE_ID

📋 Required Credentials:
- api_key: LangSmith API key (required)
- endpoint: API endpoint (default: "https://api.smith.langchain.com")
- workspace_id: Workspace ID (optional but recommended)
- project_name: Default project name (optional)

💡 Usage Examples:
# Test connection
terradev langsmith test

# List projects
terradev langsmith list-projects

# Create a new project
terradev langsmith create-project --name my-project --description "My ML project"

# List recent runs
terradev langsmith list-runs --project my-project --limit 50

# Export runs data
terradev langsmith export --project my-project --format json > runs.json

# Get tracing environment variables
terradev langsmith tracing-config

🔗 Environment Variables for Tracing:
Add these to your ML training scripts:
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_WORKSPACE_ID="your-workspace-id"
export LANGSMITH_PROJECT="your-project-name"

Then in your Python code:
from langchain.callbacks import LangSmithTracer
tracer = LangSmithTracer()
"""
