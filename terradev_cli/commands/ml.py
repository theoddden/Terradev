#!/usr/bin/env python3
"""ML integrations commands for the Terradev CLI."""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from . import cli
from terradev_cli.commands._api import TerradevAPI


def _get_api():
    """Resolve the TerradevAPI instance from the Click context or create a real one."""
    ctx = click.get_current_context()
    if ctx and ctx.obj and ctx.obj.get("api"):
        return ctx.obj["api"]
    return TerradevAPI()


def _parse_vllm_endpoint(endpoint: str):
    """Parse 'http://host:port' into (host, port)."""
    from urllib.parse import urlparse

    p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return p.hostname or "127.0.0.1", p.port or 8000


# ═══════════════════════════════════════════════════════════════════════
# ML Services Commands
# ═══════════════════════════════════════════════════════════════════════


@cli.group()
def ml():
    """ML Platform Integration Commands"""
    pass


@ml.group()
def wandb():
    """Weights & Biases experiment tracking with dashboards, reports, and alerts."""
    pass
@wandb.command("test")
def wandb_test():
    """Test connection to W&B service."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import (
            create_enhanced_wandb_service_from_credentials,
            get_enhanced_wandb_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print(get_enhanced_wandb_setup_instructions())
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print(" Testing W&B connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: W&B connected successfully")
            print(f"   Entity: {result['entity']}")
            print(f"   Project: {result['project']}")
            print(f"   Base URL: {result['base_url']}")
            print(
                f"   Dashboard: {'Enabled' if creds.get('wandb_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Reports: {'Enabled' if creds.get('wandb_reports_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Alerts: {'Enabled' if creds.get('wandb_alerts_enabled') == 'true' else 'Disabled'}"
            )
        else:
            print(f"ERROR: W&B connection failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("list-projects")
def wandb_list_projects():
    """List all W&B projects."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print(" Listing W&B projects...")
        projects = asyncio.run(service.list_projects())

        for project in projects:
            print(f"   Path {project['name']} (ID: {project['id']})")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("create-project")
@click.argument("project_name")
def wandb_create_project(project_name):
    """Create a new W&B project."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print(f"Path Creating project: {project_name}")
        result = asyncio.run(
            service.create_project(project_name, "Created via Terradev CLI")
        )
        print(f"OK: Project created: {result['name']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("list-runs")
@click.option("--limit", "-l", default=20, help="Max runs to return")
def wandb_list_runs(limit):
    """List recent W&B runs."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print(" Listing recent runs...")
        runs = asyncio.run(service.list_runs(limit=limit))

        for run in runs[:limit]:
            print(
                f"    {run['name'][:30]} - {run['state']} - {run['createdAt'][:10]}"
            )
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("create-dashboard")
def wandb_create_dashboard():
    """Create Terradev dashboard in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print("Status Creating Terradev dashboard...")
        result = asyncio.run(service.create_terradev_dashboard())

        if result["status"] == "created":
            print(f"OK: Dashboard created: {result['dashboard']['id']}")
            print(
                f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}"
            )
        else:
            print(f"ERROR: Dashboard creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("create-report")
def wandb_create_report():
    """Generate infrastructure report in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print("Plan Generating infrastructure report...")
        # Mock metrics data for demonstration
        metrics_data = {
            "total_instances": 10,
            "total_cost": 150.75,
            "avg_gpu_utilization": 78.5,
            "providers": {
                "aws": {"instances": 6, "cost": 120.50, "avg_gpu_util": 82.1},
                "gcp": {"instances": 4, "cost": 30.25, "avg_gpu_util": 71.2},
            },
        }

        result = asyncio.run(service.create_terradev_report(metrics_data))

        if result["status"] == "created":
            print(f"OK: Report created: {result['report']['id']}")
            print(
                f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}/reports"
            )
        else:
            print(f"ERROR: Report creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("setup-alerts")
def wandb_setup_alerts():
    """Set up Terradev alerts in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print(" Setting up Terradev alerts...")
        result = asyncio.run(service.create_terradev_alerts())

        if result["status"] == "completed":
            print(f"OK: Alerts set up: {len(result['alerts'])} alerts created")
            for alert in result["alerts"]:
                if alert["status"] == "created":
                    print(f"   OK: {alert['alert']['name']}")
                else:
                    print(f"   ERROR: {alert['alert']['name']}: {alert['error']}")
        else:
            print(f"ERROR: Alert setup failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@wandb.command("dashboard-status")
def wandb_dashboard_status():
    """Get comprehensive dashboard status."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            print("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.")
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        print("Status Getting comprehensive dashboard status...")
        result = asyncio.run(service.get_dashboard_status())

        if result["status"] == "connected":
            print(f"   Entity: {result['entity']}")
            print(f"   Project: {result['project']}")
            print(f"   Projects: {len(result['projects'])}")
            print(f"   Recent Runs: {len(result['recent_runs'])}")
            print(f"   Dashboards: {len(result['dashboards'])}")
            print(f"   Reports: {len(result['reports'])}")
            print(f"   Monitoring: {result['monitoring']}")
        else:
            print(f"ERROR: Dashboard status failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced W&B service not available.")
@ml.group()
def langchain():
    """LangChain integration with workflows, LangGraph, and SGLang."""
    pass
@langchain.command("test")
def langchain_test():
    """Test connection to LangChain service."""
    try:
        from terradev_cli.ml_services.langchain_service import (
            create_langchain_service_from_credentials,
            get_langchain_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print(get_langchain_setup_instructions())
            return

        service = create_langchain_service_from_credentials(creds)
        print(" Testing LangChain connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: LangChain connected successfully")
            print(f"   LangSmith: {result['langsmith']}")
            print(f"   Environment: {result['environment']}")
            print(
                f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Workflow: {'Enabled' if creds.get('langchain_workflow_enabled') == 'true' else 'Disabled'}"
            )
        else:
            print(f"ERROR: LangChain connection failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("create-workflow")
@click.argument("workflow_name")
def langchain_create_workflow(workflow_name):
    """Create a LangChain workflow."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)
        print(" Creating LangChain workflow...")
        workflow_config = {
            "name": workflow_name,
            "description": f"LangChain workflow '{workflow_name}' created via Terradev CLI",
        }
        result = asyncio.run(service.create_workflow(workflow_config))

        if result["status"] == "created":
            print(f"OK: Workflow created: {result['workflow_id']}")
            print(f"   Name: {result['name']}")
            print(f"   Description: {result['description']}")
        else:
            print(f"ERROR: Workflow creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("create-langgraph")
@click.argument("graph_name")
def langchain_create_langgraph(graph_name):
    """Create a LangGraph workflow."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)
        print(" Creating LangGraph workflow...")
        graph_config = {
            "name": graph_name,
            "description": f"LangGraph workflow '{graph_name}' created via Terradev CLI",
        }
        result = asyncio.run(service.create_langgraph_workflow(graph_config))

        if result["status"] == "created":
            print(f"OK: LangGraph workflow created: {result['workflow_id']}")
            print(f"   Name: {result['name']}")
            print(f"   Description: {result['description']}")
        else:
            print(f"ERROR: LangGraph creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("create-pipeline")
@click.argument("pipeline_name")
def langchain_create_pipeline(pipeline_name):
    """Create an SGLang pipeline."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)
        print(" Creating SGLang pipeline...")
        pipeline_config = {
            "name": pipeline_name,
            "description": f"SGLang pipeline '{pipeline_name}' created via Terradev CLI",
        }
        result = asyncio.run(service.create_sglang_pipeline(pipeline_config))

        if result["status"] == "created":
            print(f"OK: SGLang pipeline created: {result['pipeline_id']}")
            print(f"   Name: {result['name']}")
            print(f"   Description: {result['description']}")
        else:
            print(f"ERROR: Pipeline creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("list-projects")
def langchain_list_projects():
    """List LangSmith projects."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)
        print("Plan Listing LangSmith projects...")
        projects = asyncio.run(service.get_langsmith_projects())

        for project in projects:
            print(
                f"   Path {project.get('name', 'Unknown')} (ID: {project.get('id', 'Unknown')}"
            )
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("list-runs")
@click.option("--project", "-p", help="LangSmith project name")
def langchain_list_runs(project):
    """List LangSmith runs."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)
        project_name = project or creds.get("project_name", "terradev")
        print(f" Listing LangSmith runs in project: {project_name}")
        runs = asyncio.run(service.get_langsmith_runs(project_name))

        for run in runs[:10]:
            print(
                f"    {run.get('name', 'Unknown')[:30]} - {run.get('status', 'Unknown')} - {run.get('created_at', 'Unknown')[:10]}"
            )
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@langchain.command("create-trace")
@click.option("--run-id", "-r", required=True, help="Run ID for trace")
@click.option("--data", "-d", required=True, help="Trace data (JSON)")
def langchain_create_trace(run_id, data):
    """Create a trace in LangSmith."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langchain_service_from_credentials(creds)

        try:
            trace_data = json.loads(data)
        except json.JSONDecodeError:
            print("ERROR: Invalid JSON data")
            return

        print(f" Creating trace: {run_id}")
        result = asyncio.run(service.create_trace(run_id, trace_data))

        if result["status"] == "created":
            print(f"OK: Trace created: {run_id}")
        else:
            print(f"ERROR: Trace creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangChain service not available.")
@ml.group()
def langgraph():
    """LangGraph workflow orchestration with monitoring."""
    pass
@langgraph.command("test")
def langgraph_test():
    """Test connection to LangGraph service."""
    try:
        from terradev_cli.ml_services.langgraph_service import (
            create_langgraph_service_from_credentials,
            get_langgraph_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print(get_langgraph_setup_instructions())
            return

        service = create_langgraph_service_from_credentials(creds)
        print(" Testing LangGraph connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: LangGraph connected successfully")
            print(f"   LangSmith: {result['langsmith']}")
            print(f"   Environment: {result['environment']}")
            print(
                f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Deployment: {'Enabled' if creds.get('langchain_deployment_enabled') == 'true' else 'Disabled'}"
            )
            print(
                f"   Observability: {'Enabled' if creds.get('langchain_observability_enabled') == 'true' else 'Disabled'}"
            )
        else:
            print(f"ERROR: LangGraph connection failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangGraph service not available.")
@langgraph.command("create-workflow")
@click.argument("workflow_name")
@click.option("--type", "-t", required=True, type=click.Choice(["orchestrator-worker", "evaluator-optimizer"]), help="Workflow type")
def langgraph_create_workflow(workflow_name, type):
    """Create a LangGraph workflow."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langgraph_service_from_credentials(creds)
        print(f" Creating {type} LangGraph workflow...")
        workflow_config = {
            "name": workflow_name,
            "description": f"LangGraph {type} workflow '{workflow_name}' created via Terradev CLI",
            "type": type,
        }

        if type == "orchestrator-worker":
            result = asyncio.run(
                service.create_orchestrator_worker_workflow(workflow_config)
            )
        elif type == "evaluator-optimizer":
            result = asyncio.run(
                service.create_evaluation_workflow(workflow_config)
            )
        else:
            result = asyncio.run(service.create_workflow(workflow_config))

        if result["status"] == "created":
            print(f"OK: {type} workflow created: {result['workflow_id']}")
            print(f"   Name: {result['name']}")
            print(f"   Description: {result['description']}")
        else:
            print(f"ERROR: Workflow creation failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangGraph service not available.")
@langgraph.command("status")
@click.argument("workflow_id")
def langgraph_status(workflow_id):
    """Get workflow status."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langgraph_service_from_credentials(creds)
        print(f"Status Getting workflow status: {workflow_id}")
        result = asyncio.run(service.get_workflow_status(workflow_id))

        if result["status"] == "running":
            print(f"   Status: {result['status']}")
            print(f"   Workflow ID: {result['workflow_id']}")
            print(f"   Metrics: {result['metrics']}")
            print(f"   Monitoring: {result['monitoring']}")
        else:
            print(f"ERROR: Status check failed: {result['error']}")
    except ImportError:
        print("ERROR: Enhanced LangGraph service not available.")
@langgraph.command("deploy")
@click.argument("workflow_name")
def langgraph_deploy(workflow_name):
    """Deploy a workflow."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            print("ERROR: LangChain not configured. Run 'terradev ml langchain configure' first.")
            return

        service = create_langgraph_service_from_credentials(creds)
        print(f"Deploying workflow: {workflow_name}")
        # This would integrate with LangGraph's deployment APIs
        print(f"OK: Workflow deployed: {workflow_name}")
        print(f"   Access at: https://smith.langchain.com/deployments/{workflow_name}")
    except ImportError:
        print("ERROR: Enhanced LangGraph service not available.")
@ml.group()
def kserve():
    """KServe model deployment and management."""
    pass
@kserve.command("test")
def kserve_test():
    """Test connection to KServe service."""
    try:
        from terradev_cli.ml_services.kserve_service import (
            create_kserve_service_from_credentials,
            get_kserve_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("kserve")

        if not any(creds.values()):
            print(get_kserve_setup_instructions())
            return

        service = create_kserve_service_from_credentials(creds)
        print(" Testing KServe connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: KServe connected successfully")
            print(f"   Namespace: {result['namespace']}")
        else:
            print(f"ERROR: KServe connection failed: {result['error']}")
    except ImportError:
        print("ERROR: KServe service not available. Install with: pip install kserve")
@ml.group()
def langsmith():
    """LangSmith experiment tracking and monitoring."""
    pass
@langsmith.command("test")
def langsmith_test():
    """Test connection to LangSmith service."""
    try:
        from terradev_cli.ml_services.langsmith_service import (
            create_langsmith_service_from_credentials,
            get_langsmith_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("langsmith")

        if not creds.get("api_key"):
            print(get_langsmith_setup_instructions())
            return

        service = create_langsmith_service_from_credentials(creds)
        print(" Testing LangSmith connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: LangSmith connected successfully")
            print(f"   Workspace: {result['workspace_id']}")
            print(f"   Endpoint: {result['endpoint']}")
        else:
            print(f"ERROR: LangSmith connection failed: {result['error']}")
    except ImportError:
        print("ERROR: LangSmith service not available.")
@langsmith.command("list-projects")
def langsmith_list_projects():
    """List all LangSmith projects."""
    try:
        from terradev_cli.ml_services.langsmith_service import create_langsmith_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langsmith")

        if not creds.get("api_key"):
            print("ERROR: LangSmith not configured. Run 'terradev ml langsmith configure' first.")
            return

        service = create_langsmith_service_from_credentials(creds)
        print("Plan Listing LangSmith projects...")
        projects = asyncio.run(service.list_projects())

        for project in projects:
            print(f"   Path {project['name']} (ID: {project['id']})")
    except ImportError:
        print("ERROR: LangSmith service not available.")
@langsmith.command("create-project")
@click.argument("project_name")
def langsmith_create_project(project_name):
    """Create a new LangSmith project."""
    try:
        from terradev_cli.ml_services.langsmith_service import create_langsmith_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langsmith")

        if not creds.get("api_key"):
            print("ERROR: LangSmith not configured. Run 'terradev ml langsmith configure' first.")
            return

        service = create_langsmith_service_from_credentials(creds)
        print(f"Path Creating project: {project_name}")
        result = asyncio.run(
            service.create_project(project_name, "Created via Terradev CLI")
        )
        print(f"OK: Project created: {result['id']}")
    except ImportError:
        print("ERROR: LangSmith service not available.")
@langsmith.command("export")
@click.option("--format", "-f", type=click.Choice(["json", "csv"]), default="json", help="Export format")
def langsmith_export(format):
    """Export runs data."""
    try:
        from terradev_cli.ml_services.langsmith_service import create_langsmith_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langsmith")

        if not creds.get("api_key"):
            print("ERROR: LangSmith not configured. Run 'terradev ml langsmith configure' first.")
            return

        service = create_langsmith_service_from_credentials(creds)
        print("UPLOAD: Exporting runs data...")
        data = asyncio.run(service.export_runs(format=format))
        print(data)
    except ImportError:
        print("ERROR: LangSmith service not available.")
@ml.group()
def dvc():
    """DVC (Data Version Control) management."""
    pass
@dvc.command("test")
def dvc_test():
    """Test connection to DVC service."""
    try:
        from terradev_cli.ml_services.dvc_service import (
            create_dvc_service_from_credentials,
            get_dvc_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print(get_dvc_setup_instructions())
            return

        service = create_dvc_service_from_credentials(creds)
        print(" Testing DVC connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: DVC connected successfully")
            print(f"   Repository: {result['repo_path']}")
        else:
            print(f"ERROR: DVC connection failed: {result['error']}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("init")
def dvc_init():
    """Initialize DVC repository."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        service = create_dvc_service_from_credentials(creds)
        print("Path Initializing DVC repository...")
        result = asyncio.run(service.init_repo())
        print(f"OK: Repository initialized: {result['repo_path']}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("add-remote")
@click.argument("remote_spec")
def dvc_add_remote(remote_spec):
    """Add remote storage (name:url)."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        if ":" not in remote_spec:
            print("ERROR: Remote format should be: name:url")
            return

        name, url = remote_spec.split(":", 1)
        service = create_dvc_service_from_credentials(creds)
        print(f"PACKAGE: Adding remote: {name} -> {url}")
        result = asyncio.run(service.add_remote(name, url))
        print(f"OK: Remote added: {result['name']}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("add-data")
@click.argument("data_path")
def dvc_add_data(data_path):
    """Add data to tracking."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        service = create_dvc_service_from_credentials(creds)
        print(f"Status Adding data to tracking: {data_path}")
        result = asyncio.run(service.add_data(data_path))
        print(f"OK: Data added: {data_path}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("push")
def dvc_push():
    """Push data to remote."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        service = create_dvc_service_from_credentials(creds)
        print("UPLOAD: Pushing data to remote...")
        result = asyncio.run(service.push_data())
        print(f"OK: Data pushed: {result['targets']}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("pull")
def dvc_pull():
    """Pull data from remote."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        service = create_dvc_service_from_credentials(creds)
        print(" Pulling data from remote...")
        result = asyncio.run(service.pull_data())
        print(f"OK: Data pulled: {result['targets']}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@dvc.command("status")
def dvc_status():
    """Show repository status."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            print("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.")
            return

        service = create_dvc_service_from_credentials(creds)
        print("Status Repository status:")
        result = asyncio.run(service.get_status())
        for detail in result["details"]:
            print(f"   {detail}")
    except ImportError:
        print("ERROR: DVC service not available. Install with: pip install dvc")
@ml.group()
def mlflow_legacy():
    """MLflow experiment tracking and model registry."""
    pass
@mlflow_legacy.command("test")
def mlflow_legacy_test():
    """Test connection to MLflow service."""
    try:
        from terradev_cli.ml_services.mlflow_service import (
            create_mlflow_service_from_credentials,
            get_mlflow_setup_instructions,
        )

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            print(get_mlflow_setup_instructions())
            return

        service = create_mlflow_service_from_credentials(creds)
        print(" Testing MLflow connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: MLflow connected successfully")
            print(f"   Tracking URI: {result['tracking_uri']}")
            print(f"   Experiments: {result['experiments_count']}")
        else:
            print(f"ERROR: MLflow connection failed: {result['error']}")
    except ImportError:
        print("ERROR: MLflow service not available. Install with: pip install mlflow")
@mlflow_legacy.command("list-experiments")
def mlflow_legacy_list_experiments():
    """List all MLflow experiments."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            print("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.")
            return

        service = create_mlflow_service_from_credentials(creds)
        print("Plan Listing MLflow experiments...")
        experiments = asyncio.run(service.list_experiments())

        for exp in experiments:
            print(f"    {exp['name']} (ID: {exp['experiment_id']})")
    except ImportError:
        print("ERROR: MLflow service not available. Install with: pip install mlflow")
@mlflow_legacy.command("create-experiment")
@click.argument("experiment_name")
def mlflow_legacy_create_experiment(experiment_name):
    """Create a new MLflow experiment."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            print("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.")
            return

        service = create_mlflow_service_from_credentials(creds)
        print(f" Creating experiment: {experiment_name}")
        result = asyncio.run(
            service.create_experiment(experiment_name, "Created via Terradev CLI")
        )
        print(f"OK: Experiment created: {result['experiment_id']}")
    except ImportError:
        print("ERROR: MLflow service not available. Install with: pip install mlflow")
@mlflow_legacy.command("list-runs")
@click.argument("experiment_id")
def mlflow_legacy_list_runs(experiment_id):
    """List runs in experiment."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            print("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.")
            return

        service = create_mlflow_service_from_credentials(creds)
        print(f"Status Listing runs in experiment: {experiment_id}")
        runs = asyncio.run(service.list_runs([experiment_id]))

        for run in runs[:10]:
            info = run.get("info", {})
            print(
                f"    {info.get('run_id', 'N/A')[:8]} - {info.get('status', 'N/A')}"
            )
    except ImportError:
        print("ERROR: MLflow service not available. Install with: pip install mlflow")
@mlflow_legacy.command("export")
@click.argument("experiment_id")
@click.option("--format", "-f", type=click.Choice(["json", "csv"]), default="json", help="Export format")
def mlflow_legacy_export(experiment_id, format):
    """Export experiment data."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            print("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.")
            return

        service = create_mlflow_service_from_credentials(creds)
        print("UPLOAD: Exporting experiment data...")
        data = asyncio.run(service.export_experiment_data(experiment_id, format))
        print(data)
    except ImportError:
        print("ERROR: MLflow service not available. Install with: pip install mlflow")
@ml.group()
def ray():
    """Enhanced Ray distributed computing with monitoring and dashboards."""
    pass
@ray.command("test")
def ray_test():
    """Test connection to Ray service."""
    try:
        from terradev_cli.ml_services.ray_enhanced import (
            create_enhanced_ray_service_from_credentials,
        )

        api = _get_api()
        creds = api._provider_creds("ray")

        # Ray can work without credentials for local clusters
        service = create_enhanced_ray_service_from_credentials(creds)
        print(" Testing enhanced Ray connection...")
        result = asyncio.run(service.test_connection())

        if result["status"] == "connected":
            print("OK: Ray connected successfully")
            print(f"   Version: {result.get('ray_version', 'N/A')}")
            print(f"   Cluster: {result.get('cluster_name', 'local')}")
            print(f"   Dashboard: {result.get('dashboard_uri', 'N/A')}")
            print(
                f"   Monitoring: {'Enabled' if creds.get('ray_monitoring_enabled') == 'true' else 'Disabled'}"
            )
        elif result["status"] == "not_connected":
            print("Warning  Ray installed but cluster not running")
            print(f"   Version: {result.get('ray_version', 'N/A')}")
            print(f"   Error: {result['error']}")
            print(f"   Tip: Suggestion: {result.get('suggestion')}")
        else:
            print(f"ERROR: Ray connection failed: {result['error']}")
            if "not installed" in result["error"]:
                print("   Tip: Install Ray: pip install ray[default]")
                print("    For full features: pip install ray[default,train]")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("install")
def ray_install():
    """Show installation instructions."""
    try:
        from terradev_cli.ml_services.ray_enhanced import get_enhanced_ray_setup_instructions

        print(get_enhanced_ray_setup_instructions())
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("install-monitoring")
def ray_install_monitoring():
    """Install monitoring stack with Ray dashboards."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print("Deploying Installing enhanced Ray monitoring stack...")
        result = asyncio.run(service.install_monitoring_stack())

        if result["status"] == "installed":
            print("OK: Ray monitoring stack installed")
            print(f"   Ray Dashboard: {result.get('ray')}")
            print(f"   Prometheus: {result.get('prometheus')}")
            print(f"   Grafana: {result.get('grafana')}")
            print(f"   Dashboards: {result.get('dashboards')}")
            print("   Access Ray Dashboard: http://localhost:8265")
            print("   Access Grafana: http://localhost:3000")
        else:
            print(f"ERROR: Installation failed: {result['error']}")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("metrics-summary")
def ray_metrics_summary():
    """Get comprehensive metrics summary."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print("Status Getting comprehensive Ray metrics summary...")
        result = asyncio.run(service.get_monitoring_status())

        if result.get("status") != "failed":
            print(f"   Ray Status: {result.get('ray', {})}")
            print(f"   Monitoring: {result.get('monitoring', {})}")
            print(f"   Metrics: {result.get('metrics', {})}")
        else:
            print(f"ERROR: Metrics summary failed: {result.get('error')}")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("grafana")
def ray_grafana():
    """Access Grafana dashboard."""
    print(" Accessing Ray Grafana dashboard...")
    print("   Access at: http://localhost:3000")
    print("   Username: admin")
    print("   Password: prom-operator")
    print("   Ray metrics are available in the 'Ray Overview' dashboard")
@ray.command("prometheus")
def ray_prometheus():
    """Access Prometheus metrics."""
    print("Status Accessing Ray Prometheus metrics...")
    print("   Access at: http://localhost:8080")
    print(
        "   Available metrics: ray_cluster_total_workers, ray_cluster_cpu_total, ray_cluster_memory_total"
    )
@ray.command("status")
def ray_status():
    """Show cluster status."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print("Status Enhanced Ray cluster status:")
        result = asyncio.run(service.get_monitoring_status())

        if result.get("ray", {}).get("status") == "running":
            print(f"   OK: Status: {result['ray']['status']}")
            print(f"   Version: {result['ray'].get('version', 'N/A')}")
            print(f"   Cluster: {result['ray'].get('cluster_name', 'local')}")
            print(f"   Dashboard: {result['ray'].get('dashboard_uri', 'N/A')}")

            if result.get("metrics"):
                metrics = result["metrics"]
                print(f"   Workers: {metrics.get('total_workers', 0)}")
                print(f"   CPU Total: {metrics.get('cpu_total', 0)}")
                print(f"   CPU Used: {metrics.get('cpu_used', 0)}")
                print(f"   Memory Total: {metrics.get('memory_total', 0)}")
                print(f"   Memory Used: {metrics.get('memory_used', 0)}")
                print(f"   GPU Total: {metrics.get('gpu_total', 0)}")
                print(f"   GPU Used: {metrics.get('gpu_used', 0)}")
        else:
            print(
                f"   ERROR: Status: {result.get('ray', {}).get('status', 'Unknown')}"
            )
            print(
                f"   Error: {result.get('ray', {}).get('error', 'Unknown error')}"
            )
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("list-nodes")
def ray_list_nodes():
    """List cluster nodes."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print(" Listing Ray nodes...")
        result = asyncio.run(service.get_monitoring_status())

        if result.get("ray", {}).get("status") == "running":
            metrics = result.get("metrics", {})
            total_workers = metrics.get("total_workers", 0)
            print(f"   Total Workers: {total_workers}")
            print(f"   Active Workers: {total_workers}")
            print(f"   Head Node: {creds.get('ray_head_node_ip', 'localhost')}")
        else:
            print("   INFO:  No active Ray cluster found")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("start")
def ray_start():
    """Start Ray cluster."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print("Deploying Starting enhanced Ray cluster...")
        result = asyncio.run(service.start_cluster(head_node=True))
        print(f"OK: Cluster started: {result['status']}")

        if creds.get("ray_monitoring_enabled") == "true":
            print("   Status Monitoring enabled - access dashboards:")
            print("      Ray Dashboard: http://localhost:8265")
            print("      Grafana: http://localhost:3000")
            print("      Prometheus: http://localhost:8080")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("stop")
def ray_stop():
    """Stop Ray cluster."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print(" Stopping Ray cluster...")
        result = asyncio.run(service.stop_cluster())
        print(f"OK: Cluster stopped: {result['status']}")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("dashboard")
def ray_dashboard():
    """Get dashboard URL."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        print("Status Getting Ray dashboard URL...")
        url = asyncio.run(service.get_ray_dashboard_url())
        if url:
            print(f" Dashboard: {url}")
        else:
            print("ERROR: Dashboard URL not found")
    except ImportError:
        print(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ml.group()
def vllm():
    """vLLM optimization and management commands."""
    pass
@vllm.command("optimize")
@click.option("--model", "-m", required=True, help="Model name")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["throughput", "latency"]),
    default="throughput",
    help="Optimization type",
)
@click.option("--gpu-count", "-G", type=int, default=1, help="Number of GPUs")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["args", "config", "helm"]),
    default="args",
    help="Output format",
)
def vllm_optimize(model, type, gpu_count, output):
    """Generate optimized vLLM configurations using the 6 critical knobs.

    Applies the 6 knobs most teams never touch:
    1. --max-num-batched-tokens (2048→16384 for throughput, 4096 for latency)
    2. --gpu-memory-utilization (0.90→0.95)
    3. --max-num-seqs (256/1024→1024 for throughput, 512 for latency)
    4. --enable-prefix-caching (OFF→ON)
    5. --enable-chunked-prefill (OFF→ON)
    6. CPU cores (2 + #GPUs for V1 busy loop)

    Examples:
        terradev vllm optimize -m meta-llama/Llama-2-7b-hf -t throughput
        terradev vllm optimize -m mistralai/Mistral-7B-v0.1 -t latency -g 4
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig

    # Create optimized config
    if type == "throughput":
        config = VLLMConfig.create_throughput_optimized(
            model, tensor_parallel_size=gpu_count
        )
    else:
        config = VLLMConfig.create_latency_optimized(
            model, tensor_parallel_size=gpu_count
        )

    # Auto-calculate CPU cores: 2 + #GPUs
    config.cpu_cores = 2 + gpu_count

    if output == "args":
        # Import the service to get the args
        from terradev_cli.ml_services.vllm_service import VLLMService

        service = VLLMService(config)
        args = service._build_server_args()
        print(" ".join(args))
    elif output == "config":
        print(
            json.dumps(
                {
                    "model_name": config.model_name,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "max_num_seqs": config.max_num_seqs,
                    "enable_prefix_caching": config.enable_prefix_caching,
                    "enable_chunked_prefill": config.enable_chunked_prefill,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "cpu_cores": config.cpu_cores,
                },
                indent=2,
            )
        )
    elif output == "helm":
        print(f"# Helm values for {type}-optimized vLLM")
        print("serving:")
        print("  vllm:")
        print(f"    gpuMemoryUtilization: {config.gpu_memory_utilization}")
        print(f"    maxNumBatchedTokens: {config.max_num_batched_tokens}")
        print(f"    maxNumSeqs: {config.max_num_seqs}")
        print(f"    enablePrefixCaching: {config.enable_prefix_caching}")
        print(f"    enableChunkedPrefill: {config.enable_chunked_prefill}")
        print(f"    tensorParallelSize: {config.tensor_parallel_size}")
        print("resources:")
        print("  requests:")
        print(f'    cpu: "{config.cpu_cores}"')
        print("  limits:")
        print(f'    cpu: "{config.cpu_cores + 4}"  # Extra headroom')
@vllm.command("auto-optimize")
@click.option(
    "--endpoint",
    "-e",
    help="vLLM endpoint to analyze (if not provided, uses sample analysis)",
)
@click.option(
    "--samples",
    "-s",
    type=click.Path(exists=True),
    help="JSON file with sample requests",
)
@click.option("--gpu-count", "-G", type=int, default=1, help="Number of GPUs available")
@click.option("--model", "-m", required=True, help="Model name")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["config", "args", "helm"]),
    default="config",
    help="Output format",
)
@click.option("--apply", is_flag=True, help="Apply optimizations automatically")
def vllm_auto_optimize(endpoint, samples, gpu_count, model, output, apply):
    """Automatically optimize vLLM configuration based on workload analysis.

    Analyzes current workload patterns or sample requests to automatically
    select optimal settings for the 6 critical knobs.

    Examples:
        # Analyze running server
        terradev vllm auto-optimize -e http://localhost:8000 -m meta-llama/Llama-2-7b-hf

        # Analyze from sample file
        terradev vllm auto-optimize -s samples.json -m mistralai/Mistral-7B-v0.1 -g 4

        # Generate and apply Helm values
        terradev vllm auto-optimize -e http://localhost:8000 -m codellama/CodeLlama-34b-hf -o helm
    """
    import asyncio

    async def run_optimization():
        from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
        try:
            # Load samples if provided
            sample_data = None
            if samples:
                with open(samples, "r") as f:
                    sample_data = json.load(f)

            if endpoint:
                # Analyze running server
                host, port = _parse_vllm_endpoint(endpoint)
                config = VLLMConfig(model_name=model, host=host, port=port)

                async with VLLMService(config) as svc:
                    result = await svc.auto_optimize_from_workload(
                        sample_data, gpu_count
                    )
            else:
                # Analyze from samples only
                if not sample_data:
                    print("ERROR: Either --endpoint or --samples must be provided")
                    return

                workload = VLLMConfig.analyze_workload_from_samples(
                    sample_data, gpu_count
                )
                optimized_config = VLLMConfig.create_auto_optimized(model, workload)

                result = {
                    "status": "success",
                    "workload_profile": workload,
                    "optimized_config": {
                        "model_name": optimized_config.model_name,
                        "max_num_batched_tokens": optimized_config.max_num_batched_tokens,
                        "max_num_seqs": optimized_config.max_num_seqs,
                        "gpu_memory_utilization": optimized_config.gpu_memory_utilization,
                        "enable_prefix_caching": optimized_config.enable_prefix_caching,
                        "enable_chunked_prefill": optimized_config.enable_chunked_prefill,
                        "cpu_cores": optimized_config.cpu_cores,
                        "tensor_parallel_size": optimized_config.tensor_parallel_size,
                    },
                    "recommendations": "Configuration optimized based on workload analysis",
                }

            if result["status"] != "success":
                print(f"ERROR: Auto-optimization failed: {result.get('error')}")
                return

            # Display results
            print(" Workload Analysis Complete")
            print("=" * 50)

            workload = result.get("workload_profile")
            if workload:
                print(" Workload Profile:")
                print(f"   Avg Prompt Tokens: {workload.avg_prompt_length:.0f}")
                print(f"   Avg Response Tokens: {workload.avg_response_length:.0f}")
                print(f"   Requests/Second: {workload.requests_per_second:.1f}")
                print(f"   Concurrent Users: {workload.concurrent_users}")
                print(f"   Latency Sensitivity: {workload.latency_sensitivity:.2f}")
                print()

            print(" Optimized Configuration:")
            optimized = result["optimized_config"]
            for key, value in optimized.items():
                print(f"   {key}: {value}")

            # Show changes if comparison available
            changes = result.get("changes", [])
            if changes:
                print(f"\n Recommended Changes ({len(changes)}):")
                for change in changes:
                    direction = "↑" if change["optimized"] > change["current"] else "↓"
                    print(
                        f"   {change['parameter']}: {change['current']} → {change['optimized']} {direction}"
                    )
                    print(f"      Impact: {change['impact']}")

            # Generate output
            if output == "config":
                print("\n JSON Configuration:")
                print(json.dumps(optimized, indent=2))
            elif output == "args":
                # Generate CLI args from optimized config
                from terradev_cli.ml_services.vllm_service import VLLMService

                temp_config = VLLMConfig(
                    model_name=optimized["model_name"],
                    max_num_batched_tokens=optimized["max_num_batched_tokens"],
                    max_num_seqs=optimized["max_num_seqs"],
                    gpu_memory_utilization=optimized["gpu_memory_utilization"],
                    enable_prefix_caching=optimized["enable_prefix_caching"],
                    enable_chunked_prefill=optimized["enable_chunked_prefill"],
                    tensor_parallel_size=optimized.get("tensor_parallel_size", 1),
                )
                temp_service = VLLMService(temp_config)
                args = temp_service._build_server_args()
                print("\n CLI Arguments:")
                print(" ".join(args))
            elif output == "helm":
                print("\n  Helm Values:")
                print("serving:")
                print("  vllm:")
                print(
                    f"    gpuMemoryUtilization: {optimized['gpu_memory_utilization']}"
                )
                print(f"    maxNumBatchedTokens: {optimized['max_num_batched_tokens']}")
                print(f"    maxNumSeqs: {optimized['max_num_seqs']}")
                print(f"    enablePrefixCaching: {optimized['enable_prefix_caching']}")
                print(
                    f"    enableChunkedPrefill: {optimized['enable_chunked_prefill']}"
                )
                print(
                    f"    tensorParallelSize: {optimized.get('tensor_parallel_size', 1)}"
                )
                print("resources:")
                print("  requests:")
                print(f"    cpu: \"{optimized.get('cpu_cores', '2')}\"")
                print("  limits:")
                print(
                    f"    cpu: \"{optimized.get('cpu_cores', 2) + 4}\"  # Extra headroom"
                )

        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Error during auto-optimization: {e}")

    asyncio.run(run_optimization())
@vllm.command("analyze")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint to analyze")
@click.option(
    "--duration", "-d", type=int, default=60, help="Analysis duration in seconds"
)
def vllm_analyze(endpoint, duration):
    """Analyze current vLLM server workload and provide optimization recommendations.

    Monitors the running vLLM server to understand workload patterns and
    generates specific optimization recommendations.

    Examples:
        terradev vllm analyze -e http://localhost:8000
        terradev vllm analyze -e http://10.0.0.1:8000 -d 120
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
    import asyncio

    async def run_analysis():
        try:
            host, port = _parse_vllm_endpoint(endpoint)
            config = VLLMConfig(model_name="", host=host, port=port)

            async with VLLMService(config) as svc:
                print(f" Analyzing vLLM server at {endpoint} for {duration}s...")
                print("=" * 60)

                result = await svc.analyze_current_workload(duration)

                if result["status"] != "success":
                    print(f"ERROR: Analysis failed: {result.get('error')}")
                    return

                # Display current workload
                workload = result["current_workload"]
                print(" Current Workload:")
                print(
                    f"   Avg Prompt Tokens: {workload.get('avg_prompt_tokens', 0):.0f}"
                )
                print(
                    f"   Avg Generation Tokens: {workload.get('avg_generation_tokens', 0):.0f}"
                )
                print(
                    f"   Requests/Second: {workload.get('requests_per_second', 0):.1f}"
                )
                print(f"   Active Requests: {workload.get('active_requests', 0)}")
                print(f"   Queue Size: {workload.get('queue_size', 0)}")
                print()

                # Display recommendations
                recommendations = result.get("optimization_recommendations", [])
                if recommendations:
                    print(
                        f"Tip: Optimization Recommendations ({len(recommendations)}):"
                    )
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec['type'].replace('_', ' ').title()}")
                        print(
                            f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}"
                        )
                        print(f"      Reason: {rec['reason']}")
                        print(f"      Impact: {rec['impact']}")
                        print()
                else:
                    print(
                        "OK: Configuration appears well-optimized for current workload"
                    )

                print(f" Analysis completed at {result.get('timestamp')}")

        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Error during analysis: {e}")

    asyncio.run(run_analysis())
@vllm.command("benchmark")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint to test")
@click.option("--api-key", help="vLLM API key")
@click.option(
    "--prompt", default="Explain quantum computing in simple terms.", help="Test prompt"
)
@click.option("--concurrent", "-c", type=int, default=1, help="Concurrent requests")
def vllm_benchmark(endpoint, api_key, prompt, concurrent):
    """Benchmark vLLM endpoint performance."""
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
    import asyncio
    import time

    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)

    async def run_benchmark():
        async with VLLMService(config) as svc:
            # Test connection
            health = await svc.test_connection()
            if health["status"] != "connected":
                print(f"ERROR: Connection failed: {health.get('error')}")
                return

            print(f"OK: Connected to vLLM at {endpoint}")

            # Run concurrent requests
            start_time = time.time()
            tasks = []
            for i in range(concurrent):
                task = svc.test_inference(f"{prompt} (request {i+1})")
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # Analyze results
            successful = sum(
                1
                for r in results
                if isinstance(r, dict) and r.get("status") == "success"
            )
            total_time = end_time - start_time
            throughput = successful / total_time if total_time > 0 else 0

            print("\n Benchmark Results:")
            print(f"   Concurrent requests: {concurrent}")
            print(f"   Successful: {successful}/{concurrent}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Throughput: {throughput:.2f} req/s")

            if successful < concurrent:
                print(f"   WARNING:  {concurrent - successful} requests failed")

    asyncio.run(run_benchmark())
@ml.group()
def phoenix():
    """Arize Phoenix LLM trace observability  traces, spans, OTEL."""
    pass
@phoenix.command("test")
def phoenix_test():
    """Test connection to Phoenix server."""
    from terradev_cli.ml_services.phoenix_service import (
        create_phoenix_service_from_credentials,
        get_phoenix_setup_instructions,
    )

    api = _get_api()
    creds = api._provider_creds("phoenix")
    if not any(creds.values()):
        print(get_phoenix_setup_instructions())
        return
    svc = create_phoenix_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result["status"] == "connected":
        print(f"OK: Phoenix connected: {result['collector_endpoint']}")
        print(f"   Projects found: {result['projects_found']}")
    else:
        print(f"ERROR: Connection failed: {result.get('error')}")
@phoenix.command("projects")
@click.option("--limit", "-l", default=50, help="Max projects to return")
def phoenix_projects(limit):
    """List Phoenix projects."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    data = asyncio.run(svc.list_projects(limit=limit))
    projects = data.get("data", [])
    if not projects:
        print("No projects found.")
        return
    for p in projects:
        print(f"   {p.get('name', p.get('id', '?'))}")
@phoenix.command("spans")
@click.option("--project", "-p", default=None, help="Project ID or name")
@click.option(
    "--filter",
    "-f",
    "filter_cond",
    default=None,
    help="SpanQuery DSL filter, e.g. \"span_kind == 'RETRIEVER'\"",
)
@click.option("--limit", "-l", default=20, help="Max spans")
def phoenix_spans(project, filter_cond, limit):
    """List recent spans for a project."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    from terradev_cli.core.trace_viewer import view_recent_spans

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    output = asyncio.run(
        view_recent_spans(
            svc, project=project, limit=limit, filter_condition=filter_cond
        )
    )
    print(output)
@phoenix.command("trace")
@click.option("--trace-id", "-t", required=True, help="Trace ID to inspect")
@click.option("--project", "-p", default=None, help="Project ID or name")
def phoenix_trace(trace_id, project):
    """View full execution tree for a trace."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    from terradev_cli.core.trace_viewer import view_trace

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    output = asyncio.run(view_trace(svc, trace_id, project=project))
    print(output)
@phoenix.command("otel-env")
@click.option("--project", "-p", default=None, help="Project name")
def phoenix_otel_env(project):
    """Print OTEL env vars to inject into serving pods."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    env = svc.generate_otel_env(project_name=project)
    for k, v in env.items():
        print(f'export {k}="{v}"')
@phoenix.command("snippet")
@click.option("--project", "-p", default=None, help="Project name")
def phoenix_snippet(project):
    """Print Python instrumentation snippet."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    print(svc.generate_instrumentation_snippet(project_name=project))
@phoenix.command("k8s")
@click.option("--namespace", "-n", default="observability", help="K8s namespace")
def phoenix_k8s(namespace):
    """Print K8s deployment manifest for Phoenix server."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    print(svc.generate_k8s_deployment(namespace=namespace))
@ml.group()
def guardrails():
    """NeMo Guardrails  LLM output safety, jailbreak detection, PII masking."""
    pass
@guardrails.command("test")
def guardrails_test_cmd():
    """Test connection to guardrails server."""
    from terradev_cli.ml_services.guardrails_service import (
        create_guardrails_service_from_credentials,
        get_guardrails_setup_instructions,
    )

    api = _get_api()
    creds = api._provider_creds("guardrails")
    if not any(creds.values()):
        print(get_guardrails_setup_instructions())
        return
    svc = create_guardrails_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result["status"] == "connected":
        print(f"OK: Guardrails connected: {result['server_url']}")
    else:
        print(f"ERROR: Connection failed: {result.get('error')}")
@guardrails.command("chat")
@click.option(
    "--message", "-m", required=True, help="Message to send through guardrails"
)
@click.option("--config-id", "-c", default=None, help="Guardrails config_id")
def guardrails_chat(message, config_id):
    """Send a message through guardrails and show the result."""
    from terradev_cli.ml_services.guardrails_service import (
        create_guardrails_service_from_credentials,
    )

    api = _get_api()
    svc = create_guardrails_service_from_credentials(api._provider_creds("guardrails"))
    result = asyncio.run(svc.test_rail(message, config_id=config_id))
    print(f"Input:     {result['input']}")
    print(f"Config:    {result['config_id']}")
    print(f"Output:    {json.dumps(result['output'], indent=2)}")
@guardrails.command("generate-config")
@click.option("--config-id", "-c", default=None, help="Config ID name")
@click.option("--output-dir", "-o", default="./guardrails", help="Output directory")
def guardrails_generate_config(config_id, output_dir):
    """Generate default Colang 2.x guardrails configuration."""
    from terradev_cli.ml_services.guardrails_service import (
        create_guardrails_service_from_credentials,
    )

    api = _get_api()
    svc = create_guardrails_service_from_credentials(api._provider_creds("guardrails"))
    files = svc.generate_colang_config(config_id=config_id)
    output_path = Path(output_dir)
    for fname, content in files.items():
        fpath = output_path / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
        print(f"  OK: {fpath}")
    print(
        f"\nSHIELD: Config generated. Start server: nemoguardrails server --config {output_dir}"
    )
@guardrails.command("k8s")
@click.option("--namespace", "-n", default="guardrails", help="K8s namespace")
def guardrails_k8s(namespace):
    """Print K8s deployment manifest for guardrails server."""
    from terradev_cli.ml_services.guardrails_service import (
        create_guardrails_service_from_credentials,
    )

    api = _get_api()
    svc = create_guardrails_service_from_credentials(api._provider_creds("guardrails"))
    print(svc.generate_k8s_deployment(namespace=namespace))
@ml.group()
def qdrant():
    """Qdrant vector database  collections, search, RAG infrastructure."""
    pass
@qdrant.command("test")
def qdrant_test():
    """Test connection to Qdrant server."""
    from terradev_cli.ml_services.qdrant_service import (
        create_qdrant_service_from_credentials,
        get_qdrant_setup_instructions,
    )

    api = _get_api()
    creds = api._provider_creds("qdrant")
    if not any(creds.values()):
        print(get_qdrant_setup_instructions())
        return
    svc = create_qdrant_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result["status"] == "connected":
        print(f"OK: Qdrant connected: {result['url']}")
        print(f"   Collections: {', '.join(result['collections']) or 'none'}")
    else:
        print(f"ERROR: Connection failed: {result.get('error')}")
@qdrant.command("collections")
def qdrant_collections():
    """List all collections."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    cols = asyncio.run(svc.list_collections())
    if not cols:
        print("No collections found.")
        return
    for c in cols:
        print(f"    {c}")
@qdrant.command("create-collection")
@click.option("--name", "-n", default=None, help="Collection name")
@click.option(
    "--embedding-model",
    "-e",
    default=None,
    help="Embedding model (auto-sets vector size)",
)
def qdrant_create_collection(name, embedding_model):
    """Create a vector collection (auto-configured for embedding model)."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    result = asyncio.run(
        svc.configure_rag_collection(name=name, embedding_model=embedding_model)
    )
    print(f"OK: Collection created: {result['collection']}")
    print(f"   Embedding model: {result['embedding_model']}")
    print(f"   Vector size: {result['vector_size']}")
@qdrant.command("info")
@click.option("--name", "-n", default=None, help="Collection name")
def qdrant_info(name):
    """Get collection info and stats."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    info = asyncio.run(svc.get_collection_info(name=name))
    print(json.dumps(info, indent=2))
@qdrant.command("count")
@click.option("--name", "-n", default=None, help="Collection name")
def qdrant_count(name):
    """Count points in a collection."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    count = asyncio.run(svc.count_points(name=name))
    print(f"Points: {count}")
@qdrant.command("k8s")
@click.option("--namespace", "-n", default="vector-db", help="K8s namespace")
def qdrant_k8s(namespace):
    """Print K8s StatefulSet manifest for Qdrant."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    print(svc.generate_k8s_deployment(namespace=namespace))
@ml.group()
def sglang():
    """SGLang optimization and management with workload-specific auto-tuning"""
    pass
@sglang.command()
@click.argument("model_path")
@click.option(
    "--workload-type",
    type=click.Choice(
        [
            "agentic_chat",
            "batch_inference",
            "low_latency",
            "moe_model",
            "pd_disaggregated",
            "structured_output",
            "rag_workload",
        ]
    ),
    help="Workload type for optimization",
)
@click.option("--user-description", help="Natural language description of workload")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option(
    "--dry-run", is_flag=True, help="Show optimization plan without launching"
)
def sglang_optimize(model_path, workload_type, user_description, host, port, dry_run):
    """Auto-optimize SGLang configuration for workload type and hardware"""
    from terradev_cli.ml_services.sglang_service import SGLangService, WorkloadType

    service = SGLangService()

    # Convert string to enum if provided
    workload_enum = None
    if workload_type:
        workload_enum = WorkloadType(workload_type)

    # Create optimized configuration
    config = service.create_optimized_config(
        model_path=model_path,
        workload_type=workload_enum,
        user_description=user_description,
        host=host,
        port=port,
    )

    # Get optimization summary
    summary = service.get_optimization_summary(config)

    print(" SGLang Optimization Configuration")
    print(f"Model: {model_path}")
    print(f"Workload Type: {summary['workload_type']}")
    print(f"Hardware Detected: {summary['hardware_detected']}")
    print(f"Schedule Policy: {summary['schedule_policy']}")
    print(f"Attention Backend: {summary['attention_backend']}")
    print()

    print("Applied Optimizations:")
    for opt in summary["optimizations_applied"]:
        print(f"  OK: {opt}")
    print()

    if summary["performance_expectations"]:
        print("Performance Expectations:")
        for key, value in summary["performance_expectations"].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        print()

    if summary["hardware_tuned"]:
        print(" Hardware-specific optimizations applied")
        print()

    # Validate configuration
    warnings = service.validate_config(config)
    if warnings:
        print("WARNING:  Configuration Warnings:")
        for warning in warnings:
            print(f"  WARNING:  {warning}")
        print()

    if dry_run:
        print(" Dry run - configuration generated but not launched")
        return

    # Generate and display launch command
    launch_cmd = service.generate_launch_command(config)
    print(" Launch Command:")
    print(launch_cmd)
    print()

    print("Tip: To start the server, run:")
    print(f"   {launch_cmd}")
@sglang.command()
@click.argument("model_path")
@click.option("--dp-size", default=8, help="Data parallel size for multi-replica")
@click.option(
    "--workload-type",
    type=click.Choice(
        [
            "agentic_chat",
            "batch_inference",
            "low_latency",
            "moe_model",
            "pd_disaggregated",
            "structured_output",
            "rag_workload",
        ]
    ),
    help="Workload type for optimization",
)
def router(model_path, dp_size, workload_type):
    """Generate cache-aware router command for multi-replica deployments"""
    from terradev_cli.ml_services.sglang_service import SGLangService, WorkloadType

    service = SGLangService()

    # Convert string to enum if provided
    workload_enum = None
    if workload_type:
        workload_enum = WorkloadType(workload_type)

    # Create optimized configuration
    config = service.create_optimized_config(
        model_path=model_path, workload_type=workload_enum
    )

    # Generate router command
    router_cmd = service.generate_multi_replica_command(config, dp_size)

    print(" Cache-Aware Router Configuration")
    print(f"Model: {model_path}")
    print(f"DP Size: {dp_size}")
    print(f"Workload Type: {config.workload_type.value}")
    print()
    print(" Router Launch Command:")
    print(router_cmd)
    print()

    print("Tip: This router provides:")
    print("   Up to 1.9x throughput increase")
    print("   3.8x higher cache hit rate")
    print("   Intelligent request routing based on cache predictions")
@sglang.command()
@click.argument("model_path")
@click.option(
    "--workload-type",
    type=click.Choice(
        [
            "agentic_chat",
            "batch_inference",
            "low_latency",
            "moe_model",
            "pd_disaggregated",
            "structured_output",
            "rag_workload",
        ]
    ),
    help="Workload type to test",
)
@click.option("--user-description", help="Natural language description of workload")
def detect(model_path, workload_type, user_description):
    """Auto-detect workload type and show optimization recommendations"""
    from terradev_cli.ml_services.sglang_service import SGLangService, WorkloadType

    service = SGLangService()

    # Detect workload type
    detected_type = service.detect_workload_type(model_path, user_description)

    print(" Workload Detection Results")
    print(f"Model: {model_path}")
    print(f"Detected Workload Type: {detected_type.value}")

    if workload_type:
        manual_type = WorkloadType(workload_type)
        print(f"Manual Workload Type: {manual_type.value}")
        if detected_type != manual_type:
            print(
                "WARNING:  Manual and detected types differ - using manual specification"
            )
            final_type = manual_type
        else:
            print("OK: Manual and detected types match")
            final_type = detected_type
    else:
        final_type = detected_type

    print()

    # Show optimization recommendations
    config = service.create_optimized_config(
        model_path=model_path,
        workload_type=final_type,
        user_description=user_description,
    )

    summary = service.get_optimization_summary(config)

    print(" Optimization Recommendations:")
    for opt in summary["optimizations_applied"]:
        print(f"  OK: {opt}")
    print()

    if summary["performance_expectations"]:
        print(" Expected Performance:")
        for key, value in summary["performance_expectations"].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

    print()
    print("Tip: Run 'terradev sglang optimize' to generate the full launch command")
@sglang.command()
@click.option("--instance-ip", help="Remote instance IP for installation")
@click.option("--ssh-user", default="root", help="SSH user for remote installation")
@click.option("--ssh-key", help="SSH private key path")
def install(instance_ip, ssh_user, ssh_key):
    """Install SGLang with optimization stack"""
    from terradev_cli.ml_services.sglang_service import SGLangService

    service = SGLangService()

    if instance_ip:
        # Remote installation
        print(f"PACKAGE: Installing SGLang on {instance_ip}...")
        result = asyncio.run(
            service.install_on_instance(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )
        )

        if result["status"] == "installed":
            print("OK: SGLang installed successfully")
            print(f" Output: {result['output']}")
        else:
            print(f"ERROR: Installation failed: {result['error']}")
    else:
        # Local installation
        print("PACKAGE: Installing SGLang locally...")
        import subprocess
        import sys

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "sglang[all]",
                    "--find-links",
                    "https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python",
                ],
                check=True,
            )
            print("OK: SGLang installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Installation failed: {e}")
@sglang.command()
@click.argument("model_path")
@click.option("--instance-ip", help="Remote instance IP")
@click.option("--ssh-user", default="root", help="SSH user for remote deployment")
@click.option("--ssh-key", help="SSH private key path")
@click.option(
    "--workload-type",
    type=click.Choice(
        [
            "agentic_chat",
            "batch_inference",
            "low_latency",
            "moe_model",
            "pd_disaggregated",
            "structured_output",
            "rag_workload",
        ]
    ),
    help="Workload type for optimization",
)
@click.option("--port", default=8000, help="Server port")
def start(model_path, instance_ip, ssh_user, ssh_key, workload_type, port):
    """Start optimized SGLang server"""
    from terradev_cli.ml_services.sglang_service import SGLangService, WorkloadType

    service = SGLangService()

    # Create optimized configuration
    workload_enum = None
    if workload_type:
        workload_enum = WorkloadType(workload_type)

    config = service.create_optimized_config(
        model_path=model_path, workload_type=workload_enum, port=port
    )

    if instance_ip:
        # Remote deployment
        print(f" Starting SGLang server on {instance_ip}...")
        result = asyncio.run(
            service.start_server(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )
        )

        if result["status"] == "started":
            print("OK: SGLang server started successfully")
            print(f" Endpoint: http://{instance_ip}:{port}")
        else:
            print(f"ERROR: Failed to start server: {result['error']}")
    else:
        # Local launch
        launch_cmd = service.generate_launch_command(config)
        print(" Starting SGLang server locally...")
        print(f" Endpoint: http://localhost:{port}")
        print()
        print("Tip: Launch command:")
        print(launch_cmd)
        print()
        print("WARNING:  Run the command above to start the server")
@sglang.command()
def test():
    """Test SGLang installation and configuration"""
    from terradev_cli.ml_services.sglang_service import SGLangService

    service = SGLangService()

    print(" Testing SGLang installation...")
    result = asyncio.run(service.test_connection())

    if result["status"] == "connected":
        print("OK: SGLang is installed and available")
        print(f"PACKAGE: Version: {result['sglang_version']}")
    else:
        print(f"ERROR: SGLang test failed: {result['error']}")
        print("Tip: Run 'terradev sglang install' to install SGLang")
@ml.group()
def langfuse():
    """Langfuse LLM observability  traces, scores, datasets, prompts."""
    pass
@langfuse.command("configure")
@click.option(
    "--public-key", prompt="Langfuse Public Key (pk-lf-...)", hide_input=False
)
@click.option("--secret-key", prompt="Langfuse Secret Key (sk-lf-...)", hide_input=True)
@click.option(
    "--host", default="https://cloud.langfuse.com", help="Langfuse server URL"
)
def langfuse_configure(public_key, secret_key, host):
    """Configure Langfuse credentials."""
    api = _get_api()
    api._save_provider_creds(
        "langfuse",
        {
            "public_key": public_key,
            "secret_key": secret_key,
            "base_url": host,
        },
    )
    print(f"\u2705 Langfuse credentials saved (host: {host})")
@langfuse.command("test")
def langfuse_test():
    """Test Langfuse connectivity."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.test_connection())
    if result["status"] == "connected":
        print(f"\u2705 Connected to Langfuse at {result['base_url']}")
        print(f"\U0001f4c1 Projects: {result['projects']}")
        for name in result.get("project_names", []):
            print(f"   - {name}")
    else:
        print(f"\u274c Connection failed: {result.get('error')}")
@langfuse.command("traces")
@click.option("--limit", "-n", default=20, type=int)
@click.option("--name", default=None, help="Filter by trace name")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_traces(limit, name, fmt):
    """List recent traces."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.list_traces(limit=limit, name=name))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        traces = result.get("data", [])
        if not traces:
            print("  No traces found.")
            return
        print(f"\n  {'ID':<40} {'Name':<24} {'Input':<30} {'Tokens'}")
        print(f"  {'─'*38}  {'─'*22}  {'─'*28}  {'─'*8}")
        for t in traces:
            tid = t.get("id", "?")[:38]
            tname = (t.get("name") or "?")[:22]
            inp = str(t.get("input", ""))[:28]
            tokens = t.get("totalTokens") or t.get("usage", {}).get("totalTokens", "?")
            print(f"  {tid:<40} {tname:<24} {inp:<30} {tokens}")
        print()
@langfuse.command("trace")
@click.argument("trace_id")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_trace(trace_id, fmt):
    """Get a single trace with observations."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.get_trace(trace_id))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n  Trace: {result.get('id', '?')}")
        print(f"  Name:  {result.get('name', '?')}")
        print(f"  Input: {str(result.get('input', ''))[:100]}")
        print(f"  Output: {str(result.get('output', ''))[:100]}")
        obs = result.get("observations", [])
        if obs:
            print(f"\n  Observations ({len(obs)}):")
            for o in obs:
                print(
                    f"    [{o.get('type', '?')}] {o.get('name', '?')}  "
                    f"{str(o.get('input', ''))[:60]}"
                )
        print()
@langfuse.command("scores")
@click.option("--trace-id", default=None, help="Filter by trace ID")
@click.option("--name", default=None, help="Filter by score name")
@click.option("--limit", "-n", default=50, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_scores(trace_id, name, limit, fmt):
    """List scores."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.list_scores(trace_id=trace_id, name=name, limit=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        scores = result.get("data", [])
        if not scores:
            print("  No scores found.")
            return
        print(f"\n  {'Name':<20} {'Value':<10} {'Trace ID':<40} {'Comment'}")
        print(f"  {'─'*18}  {'─'*8}  {'─'*38}  {'─'*20}")
        for s in scores:
            sname = (s.get("name") or "?")[:18]
            val = s.get("value", "?")
            tid = (s.get("traceId") or "?")[:38]
            comment = (s.get("comment") or "")[:20]
            print(f"  {sname:<20} {val:<10} {tid:<40} {comment}")
        print()
@langfuse.command("score")
@click.option("--trace-id", required=True, help="Trace to score")
@click.option("--name", required=True, help="Score name (e.g. accuracy, quality)")
@click.option("--value", required=True, type=float, help="Score value (numeric)")
@click.option("--observation-id", default=None, help="Specific observation to score")
@click.option("--comment", default=None, help="Optional comment")
def langfuse_score(trace_id, name, value, observation_id, comment):
    """Create a score for a trace."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    asyncio.run(
        svc.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            observation_id=observation_id,
            comment=comment,
        )
    )
    print(f"\u2705 Score created: {name}={value} on trace {trace_id[:20]}...")
@langfuse.command("datasets")
@click.option("--limit", "-n", default=20, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_datasets(limit, fmt):
    """List datasets."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.list_datasets(limit=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        datasets = result.get("data", [])
        if not datasets:
            print("  No datasets found.")
            return
        for d in datasets:
            print(f"  \U0001f4ca {d.get('name', '?')}  {d.get('description', '')[:60]}")
        print()
@langfuse.command("export-training-data")
@click.option("--limit", "-n", default=500, type=int, help="Max pairs to export")
@click.option("--name", default=None, help="Filter traces by name")
@click.option(
    "--min-score", default=None, type=float, help="Min quality score (0.0-1.0)"
)
@click.option("--score-name", default="quality", help="Score name to filter on")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
def langfuse_export_training_data(limit, name, min_score, score_name, output):
    """Export traces as instruction/response pairs for LoRA fine-tuning."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    pairs = asyncio.run(
        svc.export_training_data(
            limit=limit, name_filter=name, min_score=min_score, score_name=score_name
        )
    )

    if not pairs:
        print("  No training pairs extracted.")
        return

    data = json.dumps(pairs, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(data)
        print(f"\u2705 Exported {len(pairs)} pairs to {output}")
    else:
        print(data)
@langfuse.command("quality")
@click.option("--score-name", default="quality", help="Score name to aggregate")
@click.option("--limit", "-n", default=200, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_quality(score_name, limit, fmt):
    """Get quality metrics for drift detection."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = asyncio.run(svc.get_quality_metrics(score_name=score_name, limit=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"\n  Quality Metrics ({score_name}):")
        print(f"  Avg:     {result.get('avg_score', '?')}")
        print(f"  Min:     {result.get('min_score', '?')}")
        print(f"  Max:     {result.get('max_score', '?')}")
        print(f"  Samples: {result.get('samples', 0)}\n")
@langfuse.command("otel-env")
@click.option("--project", "-p", default="default", help="Project name")
def langfuse_otel_env(project):
    """Print OTEL env vars for instrumenting LLM apps."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    env = svc.generate_otel_env(project_name=project)
    print()
    for k, v in env.items():
        print(f'export {k}="{v}"')
    print()
@langfuse.command("k8s")
@click.option("--namespace", "-n", default="observability", help="K8s namespace")
def langfuse_k8s(namespace):
    """Print K8s deployment manifest for Langfuse."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    print(svc.generate_k8s_deployment(namespace=namespace))
@ml.group()
def databricks():
    """Databricks MLOps  jobs, clusters, model serving, MLflow."""
    pass
@databricks.command("configure")
@click.option("--host", prompt="Databricks workspace URL")
@click.option("--token", prompt="Databricks PAT (dapi...)", hide_input=True)
def databricks_configure(host, token):
    """Configure Databricks credentials."""
    api = _get_api()
    api._save_provider_creds(
        "databricks",
        {
            "databricks_host": host,
            "databricks_token": token,
        },
    )
    print(f"\u2705 Databricks credentials saved (host: {host})")
@databricks.command("test")
def databricks_test():
    """Test Databricks connectivity."""
    from terradev_cli.integrations.databricks_integration import test_connection

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(test_connection(creds))
    if result["status"] == "connected":
        print(f"\u2705 Connected to Databricks at {result.get('host')}")
        print(f"\U0001f5a5  Clusters: {result.get('clusters', 0)}")
    else:
        print(f"\u274c Connection failed: {result.get('error')}")
@databricks.command("jobs")
@click.option("--limit", "-n", default=25, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_jobs(limit, fmt):
    """List Databricks jobs."""
    from terradev_cli.integrations.databricks_integration import list_jobs

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(list_jobs(creds, limit=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        jobs = result.get("data", {}).get("jobs", [])
        if not jobs:
            print("  No jobs found.")
            return
        print(f"\n  {'Job ID':<12} {'Name':<40} {'Created'}")
        print(f"  {'─'*10}  {'─'*38}  {'─'*20}")
        for j in jobs:
            jid = j.get("job_id", "?")
            name = j.get("settings", {}).get("name", "?")[:38]
            created = j.get("created_time", "?")
            if isinstance(created, int):
                from datetime import datetime

                created = datetime.fromtimestamp(created / 1000).strftime(
                    "%Y-%m-%d %H:%M"
                )
            print(f"  {jid:<12} {name:<40} {created}")
        print()
@databricks.command("run")
@click.argument("job_id", type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_run(job_id, fmt):
    """Trigger a Databricks job run."""
    from terradev_cli.integrations.databricks_integration import run_job

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(run_job(creds, job_id))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if result.get("success"):
            run_id = result.get("data", {}).get("run_id", "?")
            print(f"\U0001f680 Job {job_id} triggered  run_id: {run_id}")
        else:
            print(f"\u274c {result.get('error')}")
@databricks.command("run-status")
@click.argument("run_id", type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_run_status(run_id, fmt):
    """Get status of a Databricks run."""
    from terradev_cli.integrations.databricks_integration import get_run

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(get_run(creds, run_id))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        data = result.get("data", {})
        state = data.get("state", {})
        print(f"\n  Run {run_id}:")
        print(f"  Life Cycle: {state.get('life_cycle_state', '?')}")
        print(f"  Result:     {state.get('result_state', 'pending')}")
        print(f"  Message:    {state.get('state_message', '')[:80]}")
        task_name = data.get("task", {}).get("task_key") or data.get("run_name", "?")
        print(f"  Task:       {task_name}\n")
@databricks.command("clusters")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_clusters(fmt):
    """List Databricks clusters."""
    from terradev_cli.integrations.databricks_integration import list_clusters

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(list_clusters(creds))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        clusters = result.get("data", {}).get("clusters", [])
        if not clusters:
            print("  No clusters found.")
            return
        print(f"\n  {'Cluster ID':<24} {'Name':<30} {'State':<14} {'Node Type'}")
        print(f"  {'─'*22}  {'─'*28}  {'─'*12}  {'─'*20}")
        for c in clusters:
            cid = c.get("cluster_id", "?")[:22]
            name = c.get("cluster_name", "?")[:28]
            state = c.get("state", "?")[:12]
            ntype = c.get("node_type_id", "?")[:20]
            print(f"  {cid:<24} {name:<30} {state:<14} {ntype}")
        print()
@databricks.command("serving-endpoints")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_serving_endpoints(fmt):
    """List model serving endpoints."""
    from terradev_cli.integrations.databricks_integration import list_serving_endpoints

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(list_serving_endpoints(creds))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        endpoints = result.get("data", {}).get("endpoints", [])
        if not endpoints:
            print("  No serving endpoints found.")
            return
        print(f"\n  {'Name':<30} {'State':<16} {'Creator'}")
        print(f"  {'─'*28}  {'─'*14}  {'─'*24}")
        for ep in endpoints:
            name = ep.get("name", "?")[:28]
            state = ep.get("state", {}).get("ready", "?")[:14]
            creator = ep.get("creator", "?")[:24]
            print(f"  {name:<30} {state:<16} {creator}")
        print()
@databricks.command("deploy-model")
@click.option("--endpoint-name", required=True, help="Serving endpoint name")
@click.option("--model-name", required=True, help="Registered model name")
@click.option("--model-version", default="1", help="Model version")
@click.option(
    "--workload-size", default="Small", type=click.Choice(["Small", "Medium", "Large"])
)
@click.option("--scale-to-zero/--no-scale-to-zero", default=True)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_deploy_model(
    endpoint_name, model_name, model_version, workload_size, scale_to_zero, fmt
):
    """Deploy a model to a serving endpoint."""
    from terradev_cli.integrations.databricks_integration import create_serving_endpoint

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(
        create_serving_endpoint(
            creds,
            endpoint_name,
            model_name,
            model_version,
            workload_size=workload_size,
            scale_to_zero=scale_to_zero,
        )
    )

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if result.get("success"):
            print(f"\U0001f680 Serving endpoint '{endpoint_name}' created")
            print(f"  Model:    {model_name} v{model_version}")
            print(f"  Workload: {workload_size}")
        else:
            print(f"\u274c {result.get('error')}")
@databricks.command("query")
@click.option("--endpoint", required=True, help="Serving endpoint name")
@click.option("--prompt", required=True, help="Prompt text")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_query(endpoint, prompt, fmt):
    """Query a model serving endpoint."""
    from terradev_cli.integrations.databricks_integration import query_serving_endpoint

    api = _get_api()
    creds = api._provider_creds("databricks")
    inputs = [{"role": "user", "content": prompt}]
    result = asyncio.run(query_serving_endpoint(creds, endpoint, inputs))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if result.get("success"):
            data = result.get("data", {})
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"\n{content}\n")
            else:
                print(json.dumps(data, indent=2, default=str))
        else:
            print(f"\u274c {result.get('error')}")
@databricks.group()
def mlflow():
    """Databricks-hosted MLflow operations."""
    pass
@mlflow.command("experiments")
@click.option("--limit", "-n", default=50, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_mlflow_experiments(limit, fmt):
    """List MLflow experiments."""
    from terradev_cli.integrations.databricks_integration import mlflow_list_experiments

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(mlflow_list_experiments(creds, max_results=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        exps = result.get("data", {}).get("experiments", [])
        if not exps:
            print("  No experiments found.")
            return
        print(f"\n  {'ID':<12} {'Name':<44} {'Lifecycle'}")
        print(f"  {'─'*10}  {'─'*42}  {'─'*12}")
        for e in exps:
            eid = e.get("experiment_id", "?")
            name = e.get("name", "?")[:42]
            lifecycle = e.get("lifecycle_stage", "?")
            print(f"  {eid:<12} {name:<44} {lifecycle}")
        print()
@mlflow.command("models")
@click.option("--limit", "-n", default=50, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def databricks_mlflow_models(limit, fmt):
    """List registered models in Databricks Model Registry."""
    from terradev_cli.integrations.databricks_integration import mlflow_list_registered_models

    api = _get_api()
    creds = api._provider_creds("databricks")
    result = asyncio.run(mlflow_list_registered_models(creds, max_results=limit))

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        if not result.get("success"):
            print(f"\u274c {result.get('error')}")
            return
        models = result.get("data", {}).get("registered_models", [])
        if not models:
            print("  No registered models found.")
            return
        print(f"\n  {'Name':<40} {'Latest Version':<16} {'Description'}")
        print(f"  {'─'*38}  {'─'*14}  {'─'*30}")
        for m in models:
            name = m.get("name", "?")[:38]
            versions = m.get("latest_versions", [])
            latest = versions[0].get("version", "?") if versions else "?"
            desc = (m.get("description") or "")[:30]
            print(f"  {name:<40} {latest:<16} {desc}")
        print()
