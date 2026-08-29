#!/usr/bin/env python3
"""ML integrations commands for the Terradev CLI."""

import json
import os  # noqa: F401
import subprocess
import sys
import time  # noqa: F401
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import click
from . import cli
from ._base import TerradevGroup as MLGroup, get_api as _get_api, run_with_timeout as _run_with_timeout


def _safe_json(raw: str, option: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        click.echo(f"ERROR: Invalid JSON in {option}: {exc}", err=True)
        raise SystemExit(1) from exc

def _parse_vllm_endpoint(endpoint: str):
    """Parse 'http://host:port' into (host, port)."""
    from urllib.parse import urlparse

    p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return p.hostname or "127.0.0.1", p.port or 8000


# ═══════════════════════════════════════════════════════════════════════
# ML Services Commands
# ═══════════════════════════════════════════════════════════════════════

@cli.group(cls=MLGroup)
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
            click.echo(get_enhanced_wandb_setup_instructions())
            return

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo(" Testing W&B connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: W&B connected successfully")
            click.echo(f"   Entity: {result['entity']}")
            click.echo(f"   Project: {result['project']}")
            click.echo(f"   Base URL: {result['base_url']}")
            click.echo(
                f"   Dashboard: {'Enabled' if creds.get('wandb_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Reports: {'Enabled' if creds.get('wandb_reports_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Alerts: {'Enabled' if creds.get('wandb_alerts_enabled') == 'true' else 'Disabled'}"
            )
        else:
            click.echo(f"ERROR: W&B connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
        raise SystemExit(1)
@wandb.command("list-projects")
def wandb_list_projects():
    """List all W&B projects."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo(" Listing W&B projects...")
        projects = _run_with_timeout(service.list_projects())

        for project in projects:
            click.echo(f"   Path {project['name']} (ID: {project['id']})")
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("create-project")
@click.argument("project_name")
def wandb_create_project(project_name):
    """Create a new W&B project."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo(f"Path Creating project: {project_name}")
        result = _run_with_timeout(
            service.create_project(project_name, "Created via Terradev CLI")
        )
        click.echo(f"OK: Project created: {result['name']}")
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("list-runs")
@click.option("--limit", "-l", default=20, help="Max runs to return")
def wandb_list_runs(limit):
    """List recent W&B runs."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo(" Listing recent runs...")
        runs = _run_with_timeout(service.list_runs(limit=limit))

        for run in runs[:limit]:
            click.echo(
                f"    {run['name'][:30]} - {run['state']} - {run['createdAt'][:10]}"
            )
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("create-dashboard")
def wandb_create_dashboard():
    """Create Terradev dashboard in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo("Status Creating Terradev dashboard...")
        result = _run_with_timeout(service.create_terradev_dashboard())

        if result["status"] == "created":
            click.echo(f"OK: Dashboard created: {result['dashboard']['id']}")
            click.echo(
                f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}"
            )
        else:
            click.echo(f"ERROR: Dashboard creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("create-report")
def wandb_create_report():
    """Generate infrastructure report in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo("Plan Generating infrastructure report...")
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

        result = _run_with_timeout(service.create_terradev_report(metrics_data))

        if result["status"] == "created":
            click.echo(f"OK: Report created: {result['report']['id']}")
            click.echo(
                f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}/reports"
            )
        else:
            click.echo(f"ERROR: Report creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("setup-alerts")
def wandb_setup_alerts():
    """Set up Terradev alerts in W&B."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo(" Setting up Terradev alerts...")
        result = _run_with_timeout(service.create_terradev_alerts())

        if result["status"] == "completed":
            click.echo(f"OK: Alerts set up: {len(result['alerts'])} alerts created")
            for alert in result["alerts"]:
                if alert["status"] == "created":
                    click.echo(f"   OK: {alert['alert']['name']}")
                else:
                    click.echo(f"   ERROR: {alert['alert']['name']}: {alert['error']}")
        else:
            click.echo(f"ERROR: Alert setup failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
@wandb.command("dashboard-status")
def wandb_dashboard_status():
    """Get comprehensive dashboard status."""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("wandb")

        if not creds.get("api_key"):
            click.echo("ERROR: W&B not configured. Run 'terradev ml wandb configure' first.", err=True)
            raise SystemExit(1)

        service = create_enhanced_wandb_service_from_credentials(creds)
        click.echo("Status Getting comprehensive dashboard status...")
        result = _run_with_timeout(service.get_dashboard_status())

        if result["status"] == "connected":
            click.echo(f"   Entity: {result['entity']}")
            click.echo(f"   Project: {result['project']}")
            click.echo(f"   Projects: {len(result['projects'])}")
            click.echo(f"   Recent Runs: {len(result['recent_runs'])}")
            click.echo(f"   Dashboards: {len(result['dashboards'])}")
            click.echo(f"   Reports: {len(result['reports'])}")
            click.echo(f"   Monitoring: {result['monitoring']}")
        else:
            click.echo(f"ERROR: Dashboard status failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced W&B service not available.", err=True)
langchain = MLGroup(
    "langchain", help="LangChain integration with workflows, LangGraph, and SGLang."
)


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
            click.echo(get_langchain_setup_instructions())
            return

        service = create_langchain_service_from_credentials(creds)
        click.echo(" Testing LangChain connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: LangChain connected successfully")
            click.echo(f"   Environment: {result['environment']}")
            click.echo(
                f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Workflow: {'Enabled' if creds.get('langchain_workflow_enabled') == 'true' else 'Disabled'}"
            )
        else:
            click.echo(f"ERROR: LangChain connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangChain service not available.", err=True)
@langchain.command("create-workflow")
@click.argument("workflow_name")
def langchain_create_workflow(workflow_name):
    """Create a LangChain workflow."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langchain_service_from_credentials(creds)
        click.echo(" Creating LangChain workflow...")
        workflow_config = {
            "name": workflow_name,
            "description": f"LangChain workflow '{workflow_name}' created via Terradev CLI",
        }
        result = _run_with_timeout(service.create_workflow(workflow_config))

        if result["status"] == "created":
            click.echo(f"OK: Workflow created: {result['workflow_id']}")
            click.echo(f"   Name: {result['name']}")
            click.echo(f"   Description: {result['description']}")
        else:
            click.echo(f"ERROR: Workflow creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangChain service not available.", err=True)
@langchain.command("create-langgraph")
@click.argument("graph_name")
def langchain_create_langgraph(graph_name):
    """Create a LangGraph workflow."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langchain_service_from_credentials(creds)
        click.echo(" Creating LangGraph workflow...")
        graph_config = {
            "name": graph_name,
            "description": f"LangGraph workflow '{graph_name}' created via Terradev CLI",
        }
        result = _run_with_timeout(service.create_langgraph_workflow(graph_config))

        if result["status"] == "created":
            click.echo(f"OK: LangGraph workflow created: {result['workflow_id']}")
            click.echo(f"   Name: {result['name']}")
            click.echo(f"   Description: {result['description']}")
        else:
            click.echo(f"ERROR: LangGraph creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangChain service not available.", err=True)
@langchain.command("create-pipeline")
@click.argument("pipeline_name")
def langchain_create_pipeline(pipeline_name):
    """Create an SGLang pipeline."""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langchain_service_from_credentials(creds)
        click.echo(" Creating SGLang pipeline...")
        pipeline_config = {
            "name": pipeline_name,
            "description": f"SGLang pipeline '{pipeline_name}' created via Terradev CLI",
        }
        result = _run_with_timeout(service.create_sglang_pipeline(pipeline_config))

        if result["status"] == "created":
            click.echo(f"OK: SGLang pipeline created: {result['pipeline_id']}")
            click.echo(f"   Name: {result['name']}")
            click.echo(f"   Description: {result['description']}")
        else:
            click.echo(f"ERROR: Pipeline creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangChain service not available.", err=True)
langgraph = MLGroup(
    "langgraph", help="LangGraph workflow orchestration with monitoring."
)


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
            click.echo(get_langgraph_setup_instructions())
            return

        service = create_langgraph_service_from_credentials(creds)
        click.echo(" Testing LangGraph connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: LangGraph connected successfully")
            click.echo(f"   Environment: {result['environment']}")
            click.echo(
                f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Deployment: {'Enabled' if creds.get('langchain_deployment_enabled') == 'true' else 'Disabled'}"
            )
            click.echo(
                f"   Observability: {'Enabled' if creds.get('langchain_observability_enabled') == 'true' else 'Disabled'}"
            )
        else:
            click.echo(f"ERROR: LangGraph connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangGraph service not available.", err=True)
@langgraph.command("create-workflow")
@click.argument("workflow_name")
@click.option("--type", "-t", "workflow_type", required=True, type=click.Choice(["orchestrator-worker", "evaluator-optimizer"]), help="Workflow type")
def langgraph_create_workflow(workflow_name, workflow_type):
    """Create a LangGraph workflow."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langgraph_service_from_credentials(creds)
        click.echo(f" Creating {workflow_type} LangGraph workflow...")
        workflow_config = {
            "name": workflow_name,
            "description": f"LangGraph {workflow_type} workflow '{workflow_name}' created via Terradev CLI",
            "type": workflow_type,
        }

        if workflow_type == "orchestrator-worker":
            result = _run_with_timeout(
                service.create_orchestrator_worker_workflow(workflow_config)
            )
        elif workflow_type == "evaluator-optimizer":
            result = _run_with_timeout(
                service.create_evaluation_workflow(workflow_config)
            )
        else:
            result = _run_with_timeout(service.create_workflow(workflow_config))

        if result["status"] == "created":
            click.echo(f"OK: {workflow_type} workflow created: {result['workflow_id']}")
            click.echo(f"   Name: {result['name']}")
            click.echo(f"   Description: {result['description']}")
        else:
            click.echo(f"ERROR: Workflow creation failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangGraph service not available.", err=True)
@langgraph.command("status")
@click.argument("workflow_id")
def langgraph_status(workflow_id):
    """Get workflow status."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langgraph_service_from_credentials(creds)
        click.echo(f"Getting workflow status: {workflow_id}")
        result = _run_with_timeout(service.get_workflow_status(workflow_id))

        if result["status"] == "running":
            click.echo(f"   Status: {result['status']}")
            click.echo(f"   Workflow ID: {result['workflow_id']}")
            click.echo(f"   Metrics: {result['metrics']}")
            click.echo(f"   Monitoring: {result['monitoring']}")
        else:
            click.echo(f"ERROR: Status check failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: Enhanced LangGraph service not available.", err=True)
@langgraph.command("deploy")
@click.argument("workflow_name")
def langgraph_deploy(workflow_name):
    """Deploy a workflow."""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("langchain")

        if not creds.get("api_key"):
            click.echo("ERROR: LangChain not configured. Run 'terradev agent langchain configure' first.", err=True)
            raise SystemExit(1)

        service = create_langgraph_service_from_credentials(creds)
        click.echo(f"Deploying workflow: {workflow_name}")
        raise click.ClickException(
            "'langgraph deploy' is not yet implemented. "
            "Use the LangGraph Cloud CLI or LangSmith UI to deploy workflows."
        )
    except ImportError:
        click.echo("ERROR: Enhanced LangGraph service not available.", err=True)
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
            click.echo(get_kserve_setup_instructions())
            return

        service = create_kserve_service_from_credentials(creds)
        click.echo(" Testing KServe connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: KServe connected successfully")
            click.echo(f"   Namespace: {result['namespace']}")
        else:
            click.echo(f"ERROR: KServe connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: KServe service not available. Install with: pip install kserve", err=True)
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
            click.echo(get_dvc_setup_instructions())
            return

        service = create_dvc_service_from_credentials(creds)
        click.echo(" Testing DVC connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: DVC connected successfully")
            click.echo(f"   Repository: {result['repo_path']}")
        else:
            click.echo(f"ERROR: DVC connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("init")
def dvc_init():
    """Initialize DVC repository."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        service = create_dvc_service_from_credentials(creds)
        click.echo("Path Initializing DVC repository...")
        result = _run_with_timeout(service.init_repo())
        click.echo(f"OK: Repository initialized: {result['repo_path']}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("add-remote")
@click.argument("remote_spec")
def dvc_add_remote(remote_spec):
    """Add remote storage (name:url)."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        if ":" not in remote_spec:
            click.echo("ERROR: Remote format should be: name:url", err=True)
            raise SystemExit(1)

        name, url = remote_spec.split(":", 1)
        service = create_dvc_service_from_credentials(creds)
        click.echo(f"PACKAGE: Adding remote: {name} -> {url}")
        result = _run_with_timeout(service.add_remote(name, url))
        click.echo(f"OK: Remote added: {result['name']}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("add-data")
@click.argument("data_path")
def dvc_add_data(data_path):
    """Add data to tracking."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        service = create_dvc_service_from_credentials(creds)
        click.echo(f"Status Adding data to tracking: {data_path}")
        result = _run_with_timeout(service.add_data(data_path))
        click.echo(f"OK: Data added: {data_path}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("push")
def dvc_push():
    """Push data to remote."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        service = create_dvc_service_from_credentials(creds)
        click.echo("UPLOAD: Pushing data to remote...")
        result = _run_with_timeout(service.push_data())
        click.echo(f"OK: Data pushed: {result['targets']}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("pull")
def dvc_pull():
    """Pull data from remote."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        service = create_dvc_service_from_credentials(creds)
        click.echo(" Pulling data from remote...")
        result = _run_with_timeout(service.pull_data())
        click.echo(f"OK: Data pulled: {result['targets']}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
@dvc.command("status")
def dvc_status():
    """Show repository status."""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("dvc")

        if not creds.get("repo_path"):
            click.echo("ERROR: DVC not configured. Run 'terradev ml dvc configure' first.", err=True)
            raise SystemExit(1)

        service = create_dvc_service_from_credentials(creds)
        click.echo("Status Repository status:")
        result = _run_with_timeout(service.get_status())
        for detail in result["details"]:
            click.echo(f"   {detail}")
    except ImportError:
        click.echo("ERROR: DVC service not available. Install with: pip install dvc", err=True)
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
            click.echo(get_mlflow_setup_instructions())
            return

        service = create_mlflow_service_from_credentials(creds)
        click.echo(" Testing MLflow connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: MLflow connected successfully")
            click.echo(f"   Tracking URI: {result['tracking_uri']}")
            click.echo(f"   Experiments: {result['experiments_count']}")
        else:
            click.echo(f"ERROR: MLflow connection failed: {result['error']}", err=True)
    except ImportError:
        click.echo("ERROR: MLflow service not available. Install with: pip install mlflow", err=True)
@mlflow_legacy.command("list-experiments")
def mlflow_legacy_list_experiments():
    """List all MLflow experiments."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            click.echo("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.", err=True)
            raise SystemExit(1)

        service = create_mlflow_service_from_credentials(creds)
        click.echo("Plan Listing MLflow experiments...")
        experiments = _run_with_timeout(service.list_experiments())

        for exp in experiments:
            click.echo(f"    {exp['name']} (ID: {exp['experiment_id']})")
    except ImportError:
        click.echo("ERROR: MLflow service not available. Install with: pip install mlflow", err=True)
@mlflow_legacy.command("create-experiment")
@click.argument("experiment_name")
def mlflow_legacy_create_experiment(experiment_name):
    """Create a new MLflow experiment."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            click.echo("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.", err=True)
            raise SystemExit(1)

        service = create_mlflow_service_from_credentials(creds)
        click.echo(f" Creating experiment: {experiment_name}")
        result = _run_with_timeout(
            service.create_experiment(experiment_name, "Created via Terradev CLI")
        )
        click.echo(f"OK: Experiment created: {result['experiment_id']}")
    except ImportError:
        click.echo("ERROR: MLflow service not available. Install with: pip install mlflow", err=True)
@mlflow_legacy.command("list-runs")
@click.argument("experiment_id")
def mlflow_legacy_list_runs(experiment_id):
    """List runs in experiment."""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("mlflow")

        if not creds.get("tracking_uri"):
            click.echo("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.", err=True)
            raise SystemExit(1)

        service = create_mlflow_service_from_credentials(creds)
        click.echo(f"Status Listing runs in experiment: {experiment_id}")
        runs = _run_with_timeout(service.list_runs([experiment_id]))

        for run in runs[:10]:
            info = run.get("info", {})
            click.echo(
                f"    {info.get('run_id', 'N/A')[:8]} - {info.get('status', 'N/A')}"
            )
    except ImportError:
        click.echo("ERROR: MLflow service not available. Install with: pip install mlflow", err=True)
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
            click.echo("ERROR: MLflow not configured. Run 'terradev ml mlflow-legacy configure' first.", err=True)
            raise SystemExit(1)

        service = create_mlflow_service_from_credentials(creds)
        click.echo("UPLOAD: Exporting experiment data...")
        data = _run_with_timeout(service.export_experiment_data(experiment_id, format))
        click.echo(data)
    except ImportError:
        click.echo("ERROR: MLflow service not available. Install with: pip install mlflow", err=True)
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
        click.echo(" Testing enhanced Ray connection...")
        result = _run_with_timeout(service.test_connection())

        if result["status"] == "connected":
            click.echo("OK: Ray connected successfully")
            click.echo(f"   Version: {result.get('ray_version', 'N/A')}")
            click.echo(f"   Cluster: {result.get('cluster_name', 'local')}")
            click.echo(f"   Dashboard: {result.get('dashboard_uri', 'N/A')}")
        elif result["status"] == "not_connected":
            click.echo("Warning  Ray installed but cluster not running")
            click.echo(f"   Version: {result.get('ray_version', 'N/A')}")
            click.echo(f"   Error: {result['error']}")
            click.echo(f"   Tip: Suggestion: {result.get('suggestion')}")
        else:
            click.echo(f"ERROR: Ray connection failed: {result['error']}", err=True)
            if "not installed" in result["error"]:
                click.echo("   Tip: Install Ray: pip install ray[default]")
                click.echo("    For full features: pip install ray[default,train]")
    except ImportError:
        click.echo(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("install")
def ray_install():
    """Show installation instructions."""
    try:
        from terradev_cli.ml_services.ray_enhanced import get_enhanced_ray_setup_instructions

        click.echo(get_enhanced_ray_setup_instructions())
    except ImportError:
        click.echo(
            "ERROR: Enhanced Ray service not available. Install with: pip install ray[default]"
        )
@ray.command("status")
def ray_status():
    """Show cluster status."""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials

        api = _get_api()
        creds = api._provider_creds("ray")
        service = create_enhanced_ray_service_from_credentials(creds)
        click.echo("Status Enhanced Ray cluster status:")
        result = _run_with_timeout(service.get_monitoring_status())

        if result.get("ray", {}).get("status") == "running":
            click.echo(f"   OK: Status: {result['ray']['status']}")
            click.echo(f"   Version: {result['ray'].get('version', 'N/A')}")
            click.echo(f"   Cluster: {result['ray'].get('cluster_name', 'local')}")
            click.echo(f"   Dashboard: {result['ray'].get('dashboard_uri', 'N/A')}")

            if result.get("metrics"):
                metrics = result["metrics"]
                click.echo(f"   Workers: {metrics.get('total_workers', 0)}")
                click.echo(f"   CPU Total: {metrics.get('cpu_total', 0)}")
                click.echo(f"   CPU Used: {metrics.get('cpu_used', 0)}")
                click.echo(f"   Memory Total: {metrics.get('memory_total', 0)}")
                click.echo(f"   Memory Used: {metrics.get('memory_used', 0)}")
                click.echo(f"   GPU Total: {metrics.get('gpu_total', 0)}")
                click.echo(f"   GPU Used: {metrics.get('gpu_used', 0)}")
        else:
            click.echo(
                f"   ERROR: Status: {result.get('ray', {}).get('status', 'Unknown')}"
            )
            click.echo(
                f"   Error: {result.get('ray', {}).get('error', 'Unknown error')}"
            )
    except ImportError:
        click.echo(
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
        click.echo(" Listing Ray nodes...")
        result = _run_with_timeout(service.get_monitoring_status())

        if result.get("ray", {}).get("status") == "running":
            metrics = result.get("metrics", {})
            total_workers = metrics.get("total_workers", 0)
            click.echo(f"   Total Workers: {total_workers}")
            click.echo(f"   Active Workers: {total_workers}")
            click.echo(f"   Head Node: {creds.get('ray_head_node_ip', 'localhost')}")
        else:
            click.echo("   INFO:  No active Ray cluster found")
    except ImportError:
        click.echo(
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
        click.echo("Deploying Starting enhanced Ray cluster...")
        result = _run_with_timeout(service.start_cluster(head_node=True))
        click.echo(f"OK: Cluster started: {result['status']}")
    except ImportError:
        click.echo(
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
        click.echo(" Stopping Ray cluster...")
        result = _run_with_timeout(service.stop_cluster())
        click.echo(f"OK: Cluster stopped: {result['status']}")
    except ImportError:
        click.echo(
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
        click.echo("Status Getting Ray dashboard URL...")
        url = _run_with_timeout(service.get_ray_dashboard_url())
        if url:
            click.echo(f" Dashboard: {url}")
        else:
            click.echo("ERROR: Dashboard URL not found", err=True)
    except ImportError:
        click.echo(
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
@click.option("--gpu-count", "-G", type=click.IntRange(1, 1000), default=1, help="Number of GPUs")
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
        click.echo(" ".join(args))
    elif output == "config":
        click.echo(
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
        click.echo(f"# Helm values for {type}-optimized vLLM")
        click.echo("serving:")
        click.echo("  vllm:")
        click.echo(f"    gpuMemoryUtilization: {config.gpu_memory_utilization}")
        click.echo(f"    maxNumBatchedTokens: {config.max_num_batched_tokens}")
        click.echo(f"    maxNumSeqs: {config.max_num_seqs}")
        click.echo(f"    enablePrefixCaching: {config.enable_prefix_caching}")
        click.echo(f"    enableChunkedPrefill: {config.enable_chunked_prefill}")
        click.echo(f"    tensorParallelSize: {config.tensor_parallel_size}")
        click.echo("resources:")
        click.echo("  requests:")
        click.echo(f'    cpu: "{config.cpu_cores}"')
        click.echo("  limits:")
        click.echo(f'    cpu: "{config.cpu_cores + 4}"  # Extra headroom')
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
@click.option("--gpu-count", "-G", type=click.IntRange(1, 1000), default=1, help="Number of GPUs available")
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
                    click.echo("ERROR: Either --endpoint or --samples must be provided", err=True)
                    raise SystemExit(1)

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
                click.echo(f"ERROR: Auto-optimization failed: {result.get('error')}", err=True)
                raise SystemExit(1)

            # Display results
            click.echo(" Workload Analysis Complete")
            click.echo("=" * 50)

            workload = result.get("workload_profile")
            if workload:
                click.echo(" Workload Profile:")
                click.echo(f"   Avg Prompt Tokens: {workload.avg_prompt_length:.0f}")
                click.echo(f"   Avg Response Tokens: {workload.avg_response_length:.0f}")
                click.echo(f"   Requests/Second: {workload.requests_per_second:.1f}")
                click.echo(f"   Concurrent Users: {workload.concurrent_users}")
                click.echo(f"   Latency Sensitivity: {workload.latency_sensitivity:.2f}")
                click.echo()

            click.echo(" Optimized Configuration:")
            optimized = result["optimized_config"]
            for key, value in optimized.items():
                click.echo(f"   {key}: {value}")

            # Show changes if comparison available
            changes = result.get("changes", [])
            if changes:
                click.echo(f"\n Recommended Changes ({len(changes)}):")
                for change in changes:
                    direction = "↑" if change["optimized"] > change["current"] else "↓"
                    click.echo(
                        f"   {change['parameter']}: {change['current']} → {change['optimized']} {direction}"
                    )
                    click.echo(f"      Impact: {change['impact']}")

            # Generate output
            if output == "config":
                click.echo("\n JSON Configuration:")
                click.echo(json.dumps(optimized, indent=2))
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
                click.echo("\n CLI Arguments:")
                click.echo(" ".join(args))
            elif output == "helm":
                click.echo("\n  Helm Values:")
                click.echo("serving:")
                click.echo("  vllm:")
                click.echo(
                    f"    gpuMemoryUtilization: {optimized['gpu_memory_utilization']}"
                )
                click.echo(f"    maxNumBatchedTokens: {optimized['max_num_batched_tokens']}")
                click.echo(f"    maxNumSeqs: {optimized['max_num_seqs']}")
                click.echo(f"    enablePrefixCaching: {optimized['enable_prefix_caching']}")
                click.echo(
                    f"    enableChunkedPrefill: {optimized['enable_chunked_prefill']}"
                )
                click.echo(
                    f"    tensorParallelSize: {optimized.get('tensor_parallel_size', 1)}"
                )
                click.echo("resources:")
                click.echo("  requests:")
                click.echo(f"    cpu: \"{optimized.get('cpu_cores', '2')}\"")
                click.echo("  limits:")
                click.echo(
                    f"    cpu: \"{optimized.get('cpu_cores', 2) + 4}\"  # Extra headroom"
                )

        except Exception as e:  # noqa: BLE001
            click.echo(f"ERROR: Error during auto-optimization: {e}", err=True)

    _run_with_timeout(run_optimization())
@vllm.command("analyze")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint to analyze")
@click.option(
    "--duration", "-d", type=click.IntRange(1, 86400), default=60, help="Analysis duration in seconds"
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

    async def run_analysis():
        try:
            host, port = _parse_vllm_endpoint(endpoint)
            config = VLLMConfig(model_name="", host=host, port=port)

            async with VLLMService(config) as svc:
                click.echo(f" Analyzing vLLM server at {endpoint} for {duration}s...")
                click.echo("=" * 60)

                result = await svc.analyze_current_workload(duration)

                if result["status"] != "success":
                    click.echo(f"ERROR: Analysis failed: {result.get('error')}", err=True)
                    raise SystemExit(1)

                # Display current workload
                workload = result["current_workload"]
                click.echo(" Current Workload:")
                click.echo(
                    f"   Avg Prompt Tokens: {workload.get('avg_prompt_tokens', 0):.0f}"
                )
                click.echo(
                    f"   Avg Generation Tokens: {workload.get('avg_generation_tokens', 0):.0f}"
                )
                click.echo(
                    f"   Requests/Second: {workload.get('requests_per_second', 0):.1f}"
                )
                click.echo(f"   Active Requests: {workload.get('active_requests', 0)}")
                click.echo(f"   Queue Size: {workload.get('queue_size', 0)}")
                click.echo()

                # Display recommendations
                recommendations = result.get("optimization_recommendations", [])
                if recommendations:
                    click.echo(
                        f"Tip: Optimization Recommendations ({len(recommendations)}):"
                    )
                    for i, rec in enumerate(recommendations, 1):
                        click.echo(f"   {i}. {rec['type'].replace('_', ' ').title()}")
                        click.echo(
                            f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}"
                        )
                        click.echo(f"      Reason: {rec['reason']}")
                        click.echo(f"      Impact: {rec['impact']}")
                        click.echo()
                else:
                    click.echo(
                        "OK: Configuration appears well-optimized for current workload"
                    )

                click.echo(f" Analysis completed at {result.get('timestamp')}")

        except Exception as e:  # noqa: BLE001
            click.echo(f"ERROR: Error during analysis: {e}", err=True)

    _run_with_timeout(run_analysis())
@vllm.command("benchmark")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint to test")
@click.option("--api-key", help="vLLM API key")
@click.option(
    "--prompt", default="Explain quantum computing in simple terms.", help="Test prompt"
)
@click.option("--concurrent", "-c", type=click.IntRange(1, 10000), default=1, help="Concurrent requests")
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
                click.echo(f"ERROR: Connection failed: {health.get('error')}", err=True)
                raise SystemExit(1)

            click.echo(f"OK: Connected to vLLM at {endpoint}")

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

            click.echo("\n Benchmark Results:")
            click.echo(f"   Concurrent requests: {concurrent}")
            click.echo(f"   Successful: {successful}/{concurrent}")
            click.echo(f"   Total time: {total_time:.2f}s")
            click.echo(f"   Throughput: {throughput:.2f} req/s")

            if successful < concurrent:
                click.echo(f"   WARNING:  {concurrent - successful} requests failed")

    _run_with_timeout(run_benchmark())


# ═══════════════════════════════════════════════════════════════════════
# vLLM Model & LoRA Adapter Import / Management
# ═══════════════════════════════════════════════════════════════════════

@vllm.command("import-adapter")
@click.argument("adapter_id")
@click.option("--local-name", "-n", help="Local name for the adapter")
@click.option("--hf-token", help="HuggingFace token for private repos")
@click.option("--no-register", is_flag=True, help="Do not register in LoRA registry")
def vllm_import_adapter(adapter_id, local_name, hf_token, no_register):
    """Import a LoRA adapter from HuggingFace for vLLM.

    Downloads the adapter, validates it, and optionally registers it in the
    central LoRA registry so it can be linked to running vLLM servers.

    Examples:
        terradev ml vllm import-adapter organization/adapter-name
        terradev ml vllm import-adapter organization/adapter-name -n customer-a
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    async def run_import():
        config = VLLMConfig(model_name="")
        async with VLLMService(config) as svc:
            result = await svc.import_peft_adapter(
                adapter_id=adapter_id,
                local_name=local_name,
                hf_token=hf_token,
                register=not no_register,
            )
            if result["status"] != "imported":
                click.echo(f"ERROR: {result.get('error')}", err=True)
                raise SystemExit(1)

            click.echo(f"OK: Imported adapter '{result['adapter_id']}'")
            click.echo(f"   Local path: {result['local_path']}")
            click.echo(f"   Base model: {result.get('base_model')}")
            click.echo(f"   Rank: {result.get('rank')}")
            if result.get("registered"):
                click.echo(
                    f"   Registered as '{result['adapter_name']}' "
                    f"(version {result['version_id'][:8]}...)"
                )

    _run_with_timeout(run_import())


@vllm.command("import-model")
@click.argument("model_id")
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Local cache directory",
)
@click.option("--hf-token", help="HuggingFace token for private repos")
def vllm_import_model(model_id, cache_dir, hf_token):
    """Import a base model from HuggingFace for vLLM serving.

    Downloads weights to a local cache and prints a ready-to-run serve command.

    Examples:
        terradev ml vllm import-model meta-llama/Llama-2-7b-hf
        terradev ml vllm import-model mistralai/Mistral-7B-v0.1 --hf-token $HF_TOKEN
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    async def run_import():
        config = VLLMConfig(model_name=model_id)
        async with VLLMService(config) as svc:
            result = await svc.import_base_model(
                model_id=model_id,
                cache_dir=Path(cache_dir) if cache_dir else None,
                hf_token=hf_token,
            )
            if result["status"] != "imported":
                click.echo(f"ERROR: {result.get('error')}", err=True)
                raise SystemExit(1)

            click.echo(f"OK: Imported model '{result['model_id']}'")
            click.echo(f"   Local path: {result['local_path']}")
            click.echo(f"   Serve command:")
            click.echo(f"   {result['serve_command']}")

    _run_with_timeout(run_import())


@vllm.group()
def lora():
    """LoRA adapter management for vLLM serving engines."""
    pass


@lora.command("list")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--api-key", help="vLLM API key")
def vllm_lora_list(endpoint, api_key):
    """List LoRA adapters currently loaded on a vLLM server."""
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)

    async def run_list():
        async with VLLMService(config) as svc:
            result = await svc.lora_list()
            if result["status"] != "success":
                click.echo(f"ERROR: {result.get('error')}", err=True)
                raise SystemExit(1)

            click.echo(f"Base models ({len(result.get('base_models', []))}):")
            for m in result.get("base_models", []):
                click.echo(f"  {m.get('id', '?')}")

            click.echo(f"LoRA adapters ({len(result.get('lora_adapters', []))}):")
            if result.get("lora_adapters"):
                for a in result.get("lora_adapters", []):
                    click.echo(f"  {a.get('id', '?')}  (parent: {a.get('parent', '-')})")
            else:
                click.echo("  (none)")

    _run_with_timeout(run_list())


@lora.command("load")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--name", "-n", required=True, help="Adapter name")
@click.option("--path", required=True, help="Local path to adapter weights")
@click.option("--api-key", help="vLLM API key")
@click.option("--register", is_flag=True, help="Register in LoRA registry before loading")
@click.option("--base-model", help="Base model name (required with --register)")
@click.option("--rank", default=64, help="LoRA rank (default: 64)")
def vllm_lora_load(endpoint, name, path, api_key, register, base_model, rank):
    """Hot-load a LoRA adapter onto a running vLLM server.

    Examples:
        terradev ml vllm lora load -e http://localhost:8000 -n customer-a --path /adapters/customer-a
        terradev ml vllm lora load -e http://localhost:8000 -n customer-a --path /adapters/customer-a --register --base-model meta-llama/Llama-2-7b-hf
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService, LoRAModule

    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)

    version_id = None
    if register:
        if not base_model:
            click.echo("ERROR: --base-model required when using --register", err=True)
            raise SystemExit(1)
        from terradev_cli.ml_services.lora_registry import get_lora_registry

        registry = get_lora_registry()
        version = registry.register_adapter(
            adapter_name=name,
            base_model=base_model,
            path=path,
            rank=rank,
        )
        version_id = version.version_id
        registry.mark_version_active(name, version_id)

    async def run_load():
        async with VLLMService(config) as svc:
            result = await svc.lora_load(
                LoRAModule(name=name, path=path),
                version_id=version_id,
            )
            if result["status"] == "loaded":
                click.echo(f"OK: Loaded adapter '{name}' on {endpoint}")
                click.echo("   Use 'model': '{name}' in API requests")
            else:
                click.echo(f"ERROR: {result.get('error')}", err=True)

    _run_with_timeout(run_load())


@lora.command("unload")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--name", "-n", required=True, help="Adapter name to unload")
@click.option("--api-key", help="vLLM API key")
def vllm_lora_unload(endpoint, name, api_key):
    """Hot-unload a LoRA adapter from a running vLLM server."""
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)

    async def run_unload():
        async with VLLMService(config) as svc:
            result = await svc.lora_unload(name)
            if result["status"] == "unloaded":
                click.echo(f"OK: Unloaded adapter '{name}' from {endpoint}")
            else:
                click.echo(f"ERROR: {result.get('error')}", err=True)

    _run_with_timeout(run_unload())


@lora.command("link")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--name", "-n", required=True, help="Registered adapter name")
@click.option("--api-key", help="vLLM API key")
def vllm_lora_link(endpoint, name, api_key):
    """Load the active registry version of an adapter onto a vLLM server.

    This links the central LoRA registry with the running serving engine.

    Examples:
        terradev ml vllm lora link -e http://localhost:8000 -n customer-a
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)

    async def run_link():
        async with VLLMService(config) as svc:
            result = await svc.lora_load_from_registry(name)
            if result["status"] == "loaded":
                click.echo(f"OK: Linked and loaded adapter '{name}' from registry to {endpoint}")
                click.echo("   Use 'model': '{name}' in API requests")
            else:
                click.echo(f"ERROR: {result.get('error')}", err=True)

    _run_with_timeout(run_link())


@lora.command("sync")
@click.option("--name", "-n", required=True, help="Registered adapter name")
@click.option("--replicas", required=True, help="Comma-separated host:port list")
def vllm_lora_sync(name, replicas):
    """Synchronize an adapter from the registry across multiple vLLM replicas."""
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    replica_list = []
    for r in replicas.split(","):
        if ":" not in r:
            click.echo(f"ERROR: invalid replica format '{r}'. Expected host:port", err=True)
            raise SystemExit(1)
        host, port = r.rsplit(":", 1)
        try:
            port = int(port)
        except ValueError:
            click.echo(f"ERROR: invalid port in replica '{r}'", err=True)
            raise SystemExit(1)
        replica_list.append({"replica_id": r, "host": host, "port": port})

    async def run_sync():
        config = VLLMConfig(model_name="")
        svc = VLLMService(config)
        result = await svc.lora_sync(name, replicas=replica_list)
        if result["status"] == "success":
            click.echo(f"OK: Synchronized adapter '{name}' across {len(replica_list)} replica(s)")
            final = result.get("final_consistency", {})
            click.echo(f"   Expected: {len(final.get('expected_replicas', []))}")
            click.echo(f"   Loaded: {len(final.get('loaded_replicas', []))}")
        else:
            click.echo(f"ERROR: {result.get('error')}", err=True)

    _run_with_timeout(run_sync())


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
        click.echo(get_phoenix_setup_instructions())
        return
    svc = create_phoenix_service_from_credentials(creds)
    result = _run_with_timeout(svc.test_connection())
    if result["status"] == "connected":
        click.echo(f"OK: Phoenix connected: {result['collector_endpoint']}")
        click.echo(f"   Projects found: {result['projects_found']}")
    else:
        click.echo(f"ERROR: Connection failed: {result.get('error')}", err=True)
@phoenix.command("projects")
@click.option("--limit", "-l", default=50, help="Max projects to return")
def phoenix_projects(limit):
    """List Phoenix projects."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    data = _run_with_timeout(svc.list_projects(limit=limit))
    projects = data.get("data", [])
    if not projects:
        click.echo("No projects found.")
        return
    for p in projects:
        click.echo(f"   {p.get('name', p.get('id', '?'))}")
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
    output = _run_with_timeout(
        view_recent_spans(
            svc, project=project, limit=limit, filter_condition=filter_cond
        )
    )
    click.echo(output)
@phoenix.command("trace")
@click.option("--trace-id", "-t", required=True, help="Trace ID to inspect")
@click.option("--project", "-p", default=None, help="Project ID or name")
def phoenix_trace(trace_id, project):
    """View full execution tree for a trace."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    from terradev_cli.core.trace_viewer import view_trace

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    output = _run_with_timeout(view_trace(svc, trace_id, project=project))
    click.echo(output)
@phoenix.command("otel-env")
@click.option("--project", "-p", default=None, help="Project name")
def phoenix_otel_env(project):
    """Print OTEL env vars to inject into serving pods."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    env = svc.generate_otel_env(project_name=project)
    for k, v in env.items():
        click.echo(f'export {k}="{v}"')
@phoenix.command("snippet")
@click.option("--project", "-p", default=None, help="Project name")
def phoenix_snippet(project):
    """Print Python instrumentation snippet."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    click.echo(svc.generate_instrumentation_snippet(project_name=project))
@phoenix.command("k8s")
@click.option("--namespace", "-n", default="observability", help="K8s namespace")
def phoenix_k8s(namespace):
    """Print K8s deployment manifest for Phoenix server."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials

    api = _get_api()
    svc = create_phoenix_service_from_credentials(api._provider_creds("phoenix"))
    click.echo(svc.generate_k8s_deployment(namespace=namespace))
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
        click.echo(get_guardrails_setup_instructions())
        return
    svc = create_guardrails_service_from_credentials(creds)
    result = _run_with_timeout(svc.test_connection())
    if result["status"] == "connected":
        click.echo(f"OK: Guardrails connected: {result['server_url']}")
    else:
        click.echo(f"ERROR: Connection failed: {result.get('error')}", err=True)
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
    result = _run_with_timeout(svc.test_rail(message, config_id=config_id))
    click.echo(f"Input:     {result['input']}")
    click.echo(f"Config:    {result['config_id']}")
    click.echo(f"Output:    {json.dumps(result['output'], indent=2)}")
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
        click.echo(f"  OK: {fpath}")
    click.echo(
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
    click.echo(svc.generate_k8s_deployment(namespace=namespace))
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
        click.echo(get_qdrant_setup_instructions())
        return
    svc = create_qdrant_service_from_credentials(creds)
    result = _run_with_timeout(svc.test_connection())
    if result["status"] == "connected":
        click.echo(f"OK: Qdrant connected: {result['url']}")
        click.echo(f"   Collections: {', '.join(result['collections']) or 'none'}")
    else:
        click.echo(f"ERROR: Connection failed: {result.get('error')}", err=True)
@qdrant.command("collections")
def qdrant_collections():
    """List all collections."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    cols = _run_with_timeout(svc.list_collections())
    if not cols:
        click.echo("No collections found.")
        return
    for c in cols:
        click.echo(f"    {c}")
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
    result = _run_with_timeout(
        svc.configure_rag_collection(name=name, embedding_model=embedding_model)
    )
    click.echo(f"OK: Collection created: {result['collection']}")
    click.echo(f"   Embedding model: {result['embedding_model']}")
    click.echo(f"   Vector size: {result['vector_size']}")
@qdrant.command("info")
@click.option("--name", "-n", default=None, help="Collection name")
def qdrant_info(name):
    """Get collection info and stats."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    info = _run_with_timeout(svc.get_collection_info(name=name))
    click.echo(json.dumps(info, indent=2))
@qdrant.command("count")
@click.option("--name", "-n", default=None, help="Collection name")
def qdrant_count(name):
    """Count points in a collection."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    count = _run_with_timeout(svc.count_points(name=name))
    click.echo(f"Points: {count}")
@qdrant.command("k8s")
@click.option("--namespace", "-n", default="vector-db", help="K8s namespace")
def qdrant_k8s(namespace):
    """Print K8s StatefulSet manifest for Qdrant."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials

    api = _get_api()
    svc = create_qdrant_service_from_credentials(api._provider_creds("qdrant"))
    click.echo(svc.generate_k8s_deployment(namespace=namespace))
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

    click.echo(" SGLang Optimization Configuration")
    click.echo(f"Model: {model_path}")
    click.echo(f"Workload Type: {summary['workload_type']}")
    click.echo(f"Hardware Detected: {summary['hardware_detected']}")
    click.echo(f"Schedule Policy: {summary['schedule_policy']}")
    click.echo(f"Attention Backend: {summary['attention_backend']}")
    click.echo()

    click.echo("Applied Optimizations:")
    for opt in summary["optimizations_applied"]:
        click.echo(f"  OK: {opt}")
    click.echo()

    if summary["performance_expectations"]:
        click.echo("Performance Expectations:")
        for key, value in summary["performance_expectations"].items():
            click.echo(f"   {key.replace('_', ' ').title()}: {value}")
        click.echo()

    if summary["hardware_tuned"]:
        click.echo(" Hardware-specific optimizations applied")
        click.echo()

    # Validate configuration
    warnings = service.validate_config(config)
    if warnings:
        click.echo("WARNING:  Configuration Warnings:")
        for warning in warnings:
            click.echo(f"  WARNING:  {warning}")
        click.echo()

    if dry_run:
        click.echo(" Dry run - configuration generated but not launched")
        return

    # Generate and display launch command
    launch_cmd = service.generate_launch_command(config)
    click.echo(" Launch Command:")
    click.echo(launch_cmd)
    click.echo()

    click.echo("Tip: To start the server, run:")
    click.echo(f"   {launch_cmd}")
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

    click.echo(" Cache-Aware Router Configuration")
    click.echo(f"Model: {model_path}")
    click.echo(f"DP Size: {dp_size}")
    click.echo(f"Workload Type: {config.workload_type.value}")
    click.echo()
    click.echo(" Router Launch Command:")
    click.echo(router_cmd)
    click.echo()

    click.echo("Tip: This router provides:")
    click.echo("   Up to 1.9x throughput increase")
    click.echo("   3.8x higher cache hit rate")
    click.echo("   Intelligent request routing based on cache predictions")
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

    click.echo(" Workload Detection Results")
    click.echo(f"Model: {model_path}")
    click.echo(f"Detected Workload Type: {detected_type.value}")

    if workload_type:
        manual_type = WorkloadType(workload_type)
        click.echo(f"Manual Workload Type: {manual_type.value}")
        if detected_type != manual_type:
            click.echo(
                "WARNING:  Manual and detected types differ - using manual specification"
            )
            final_type = manual_type
        else:
            click.echo("OK: Manual and detected types match")
            final_type = detected_type
    else:
        final_type = detected_type

    click.echo()

    # Show optimization recommendations
    config = service.create_optimized_config(
        model_path=model_path,
        workload_type=final_type,
        user_description=user_description,
    )

    summary = service.get_optimization_summary(config)

    click.echo(" Optimization Recommendations:")
    for opt in summary["optimizations_applied"]:
        click.echo(f"  OK: {opt}")
    click.echo()

    if summary["performance_expectations"]:
        click.echo(" Expected Performance:")
        for key, value in summary["performance_expectations"].items():
            click.echo(f"   {key.replace('_', ' ').title()}: {value}")

    click.echo()
    click.echo("Tip: Run 'terradev sglang optimize' to generate the full launch command")
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
        click.echo(f"PACKAGE: Installing SGLang on {instance_ip}...")
        result = _run_with_timeout(
            service.install_on_instance(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )
        )

        if result["status"] == "installed":
            click.echo("OK: SGLang installed successfully")
            click.echo(f" Output: {result['output']}")
        else:
            click.echo(f"ERROR: Installation failed: {result['error']}", err=True)
    else:
        # Local installation
        click.echo("PACKAGE: Installing SGLang locally...")
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
            click.echo("OK: SGLang installed successfully")
        except subprocess.CalledProcessError as e:
            click.echo(f"ERROR: Installation failed: {e}", err=True)
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
        click.echo(f" Starting SGLang server on {instance_ip}...")
        result = _run_with_timeout(
            service.start_server(
                instance_ip=instance_ip, ssh_user=ssh_user, ssh_key=ssh_key
            )
        )

        if result["status"] == "started":
            click.echo("OK: SGLang server started successfully")
            click.echo(f" Endpoint: http://{instance_ip}:{port}")
        else:
            click.echo(f"ERROR: Failed to start server: {result['error']}", err=True)
    else:
        # Local launch
        launch_cmd = service.generate_launch_command(config)
        click.echo(" Starting SGLang server locally...")
        click.echo(f" Endpoint: http://localhost:{port}")
        click.echo()
        click.echo("Tip: Launch command:")
        click.echo(launch_cmd)
        click.echo()
        click.echo("WARNING:  Run the command above to start the server")
@sglang.command()
def test():
    """Test SGLang installation and configuration"""
    from terradev_cli.ml_services.sglang_service import SGLangService

    service = SGLangService()

    click.echo(" Testing SGLang installation...")
    result = _run_with_timeout(service.test_connection())

    if result["status"] == "connected":
        click.echo("OK: SGLang is installed and available")
        click.echo(f"PACKAGE: Version: {result['sglang_version']}")
    else:
        click.echo(f"ERROR: SGLang test failed: {result['error']}", err=True)
        click.echo("Tip: Run 'terradev sglang install' to install SGLang")
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
    click.echo(f"\u2705 Langfuse credentials saved (host: {host})")
@langfuse.command("test")
def langfuse_test():
    """Test Langfuse connectivity."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = _run_with_timeout(svc.test_connection())
    if result["status"] == "connected":
        click.echo(f"\u2705 Connected to Langfuse at {result['base_url']}")
        click.echo(f"\U0001f4c1 Projects: {result['projects']}")
        for name in result.get("project_names", []):
            click.echo(f"   - {name}")
    else:
        click.echo(f"\u274c Connection failed: {result.get('error')}")
@langfuse.command("traces")
@click.option("--limit", "-n", default=20, type=click.IntRange(1, 10000))
@click.option("--name", default=None, help="Filter by trace name")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_traces(limit, name, fmt):
    """List recent traces."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = _run_with_timeout(svc.list_traces(limit=limit, name=name))

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        traces = result.get("data", [])
        if not traces:
            click.echo("  No traces found.")
            return
        click.echo(f"\n  {'ID':<40} {'Name':<24} {'Input':<30} {'Tokens'}")
        click.echo(f"  {'─'*38}  {'─'*22}  {'─'*28}  {'─'*8}")
        for t in traces:
            tid = t.get("id", "?")[:38]
            tname = (t.get("name") or "?")[:22]
            inp = str(t.get("input", ""))[:28]
            tokens = t.get("totalTokens") or t.get("usage", {}).get("totalTokens", "?")
            click.echo(f"  {tid:<40} {tname:<24} {inp:<30} {tokens}")
        click.echo()
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
    result = _run_with_timeout(svc.get_trace(trace_id))

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"\n  Trace: {result.get('id', '?')}")
        click.echo(f"  Name:  {result.get('name', '?')}")
        click.echo(f"  Input: {str(result.get('input', ''))[:100]}")
        click.echo(f"  Output: {str(result.get('output', ''))[:100]}")
        obs = result.get("observations", [])
        if obs:
            click.echo(f"\n  Observations ({len(obs)}):")
            for o in obs:
                click.echo(
                    f"    [{o.get('type', '?')}] {o.get('name', '?')}  "
                    f"{str(o.get('input', ''))[:60]}"
                )
        click.echo()
@langfuse.command("scores")
@click.option("--trace-id", default=None, help="Filter by trace ID")
@click.option("--name", default=None, help="Filter by score name")
@click.option("--limit", "-n", default=50, type=click.IntRange(1, 10000))
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_scores(trace_id, name, limit, fmt):
    """List scores."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = _run_with_timeout(svc.list_scores(trace_id=trace_id, name=name, limit=limit))

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        scores = result.get("data", [])
        if not scores:
            click.echo("  No scores found.")
            return
        click.echo(f"\n  {'Name':<20} {'Value':<10} {'Trace ID':<40} {'Comment'}")
        click.echo(f"  {'─'*18}  {'─'*8}  {'─'*38}  {'─'*20}")
        for s in scores:
            sname = (s.get("name") or "?")[:18]
            val = s.get("value", "?")
            tid = (s.get("traceId") or "?")[:38]
            comment = (s.get("comment") or "")[:20]
            click.echo(f"  {sname:<20} {val:<10} {tid:<40} {comment}")
        click.echo()
@langfuse.command("score")
@click.option("--trace-id", required=True, help="Trace to score")
@click.option("--name", required=True, help="Score name (e.g. accuracy, quality)")
@click.option("--value", required=True, type=click.FloatRange(0.0, 1.0), help="Score value (numeric)")
@click.option("--observation-id", default=None, help="Specific observation to score")
@click.option("--comment", default=None, help="Optional comment")
def langfuse_score(trace_id, name, value, observation_id, comment):
    """Create a score for a trace."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    _run_with_timeout(
        svc.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            observation_id=observation_id,
            comment=comment,
        )
    )
    click.echo(f"\u2705 Score created: {name}={value} on trace {trace_id[:20]}...")
@langfuse.command("datasets")
@click.option("--limit", "-n", default=20, type=click.IntRange(1, 10000))
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_datasets(limit, fmt):
    """List datasets."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = _run_with_timeout(svc.list_datasets(limit=limit))

    if fmt == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        datasets = result.get("data", [])
        if not datasets:
            click.echo("  No datasets found.")
            return
        for d in datasets:
            click.echo(f"  \U0001f4ca {d.get('name', '?')}  {d.get('description', '')[:60]}")
        click.echo()
@langfuse.command("export-training-data")
@click.option("--limit", "-n", default=500, type=click.IntRange(1, 100000), help="Max pairs to export")
@click.option("--name", default=None, help="Filter traces by name")
@click.option(
    "--min-score", default=None, type=click.FloatRange(0.0, 1.0), help="Min quality score (0.0-1.0)"
)
@click.option("--score-name", default="quality", help="Score name to filter on")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
def langfuse_export_training_data(limit, name, min_score, score_name, output):
    """Export traces as instruction/response pairs for LoRA fine-tuning."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    pairs = _run_with_timeout(
        svc.export_training_data(
            limit=limit, name_filter=name, min_score=min_score, score_name=score_name
        )
    )

    if not pairs:
        click.echo("  No training pairs extracted.")
        return

    data = json.dumps(pairs, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(data)
        click.echo(f"\u2705 Exported {len(pairs)} pairs to {output}")
    else:
        click.echo(data)
@langfuse.command("quality")
@click.option("--score-name", default="quality", help="Score name to aggregate")
@click.option("--limit", "-n", default=200, type=click.IntRange(1, 10000))
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def langfuse_quality(score_name, limit, fmt):
    """Get quality metrics for drift detection."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    result = _run_with_timeout(svc.get_quality_metrics(score_name=score_name, limit=limit))

    if fmt == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"\n  Quality Metrics ({score_name}):")
        click.echo(f"  Avg:     {result.get('avg_score', '?')}")
        click.echo(f"  Min:     {result.get('min_score', '?')}")
        click.echo(f"  Max:     {result.get('max_score', '?')}")
        click.echo(f"  Samples: {result.get('samples', 0)}\n")
@langfuse.command("otel-env")
@click.option("--project", "-p", default="default", help="Project name")
def langfuse_otel_env(project):
    """Print OTEL env vars for instrumenting LLM apps."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    env = svc.generate_otel_env(project_name=project)
    click.echo()
    for k, v in env.items():
        click.echo(f'export {k}="{v}"')
    click.echo()
@langfuse.command("k8s")
@click.option("--namespace", "-n", default="observability", help="K8s namespace")
def langfuse_k8s(namespace):
    """Print K8s deployment manifest for Langfuse."""
    from terradev_cli.ml_services.langfuse_service import create_langfuse_service_from_credentials

    api = _get_api()
    svc = create_langfuse_service_from_credentials(api._provider_creds("langfuse"))
    click.echo(svc.generate_k8s_deployment(namespace=namespace))

# ═══════════════════════════════════════════════════════════════════════
# Ollama Local Model Commands
# ═══════════════════════════════════════════════════════════════════════

def _ollama_request(endpoint: str, method: str, path: str, data: Optional[Dict] = None, timeout: int = 30):
    """Make a synchronous JSON request to the Ollama HTTP API."""
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Ollama returned {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot connect to Ollama at {endpoint}: {e.reason}")

def _ollama_stream(endpoint: str, path: str, data: Dict, timeout: int = 600):
    """Stream an NDJSON response from the Ollama HTTP API and print progress."""
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "status" in obj:
                    click.echo(f"  {obj['status']}")
                if "error" in obj:
                    raise RuntimeError(obj["error"])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Ollama returned {e.code}: {body}")

@ml.group()
def ollama():
    """Local Ollama model management and inference."""
    pass

@ollama.command("list")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
def ollama_list(endpoint):
    """List models available on the Ollama server."""
    try:
        data = _ollama_request(endpoint, "GET", "/api/tags")
        models = data.get("models", [])
        if not models:
            click.echo("No Ollama models found.")
            return
        click.echo(f"Ollama models on {endpoint}:")
        for m in models:
            size_gb = m.get("size", 0) / (1024**3)
            click.echo(f"  {m['name']} ({size_gb:.1f}GB)")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@ollama.command("pull")
@click.argument("model")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
def ollama_pull(model, endpoint):
    """Pull an Ollama model onto the local server."""
    click.echo(f"Pulling {model} from {endpoint}...")
    try:
        _ollama_stream(endpoint, "/api/pull", {"name": model, "stream": True})
        click.echo(f"OK: {model} pulled successfully")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@ollama.command("generate")
@click.argument("model")
@click.option("--prompt", "-p", required=True, help="Prompt text")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
@click.option("--options", "-o", help="JSON options for generation")
def ollama_generate(model, prompt, endpoint, options):
    """Generate text with an Ollama model."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = _safe_json(options, "--options")
        data = _ollama_request(endpoint, "POST", "/api/generate", payload, timeout=120)
        click.echo(data.get("response", ""))
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@ollama.command("chat")
@click.argument("model")
@click.option("--message", "-m", required=True, help="User message")
@click.option("--system", "-s", help="System message")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
@click.option("--options", "-o", help="JSON options for the chat request")
def ollama_chat(model, message, system, endpoint, options):
    """Chat with an Ollama model."""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        payload = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = _safe_json(options, "--options")
        data = _ollama_request(endpoint, "POST", "/api/chat", payload, timeout=120)
        reply = data.get("message", {}).get("content", "")
        click.echo(reply)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@ollama.command("info")
@click.argument("model")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
def ollama_info(model, endpoint):
    """Show detailed information about an Ollama model."""
    try:
        data = _ollama_request(endpoint, "POST", "/api/show", {"name": model})
        click.echo(json.dumps(data, indent=2, default=str))
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@ollama.command("ps")
@click.option("--endpoint", "-e", default="http://localhost:11434", help="Ollama API endpoint")
def ollama_ps(endpoint):
    """List currently running Ollama models."""
    try:
        data = _ollama_request(endpoint, "GET", "/api/ps")
        models = data.get("models", [])
        if not models:
            click.echo("No Ollama models are currently running.")
            return
        click.echo(f"Running Ollama models on {endpoint}:")
        for m in models:
            size_gb = m.get("size", 0) / (1024**3)
            until = m.get("expires_at", "unknown")
            click.echo(f"  {m['name']} ({size_gb:.1f}GB, expires {until})")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

# ═══════════════════════════════════════════════════════════════════════
# DeepEval LLM Evaluation Commands
# ═══════════════════════════════════════════════════════════════════════

DEEPEVAL_METRICS = [
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "HallucinationMetric",
    "BiasMetric",
    "ToxicityMetric",
    "SummarizationMetric",
    "RagasMetric",
    "GEval",
    "DAGMetric",
]

@ml.group()
def deepeval():
    """LLM evaluation with DeepEval."""
    pass

@deepeval.command("install")
@click.option("--upgrade", is_flag=True, help="Upgrade DeepEval")
def deepeval_install(upgrade):
    """Install the DeepEval package."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append("deepeval")
    click.echo("Installing DeepEval...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("OK: DeepEval installed")
        else:
            click.echo(f"ERROR: {result.stderr}", err=True)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@deepeval.command("init")
@click.option("--output", "-o", default="test_deepeval.py", help="Output test file path")
def deepeval_init(output):
    """Generate a starter DeepEval test file."""
    sample = '''import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_llm():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5)])
'''
    try:
        Path(output).write_text(sample, encoding="utf-8")
        click.echo(f"OK: Starter DeepEval test written to {output}")
        click.echo(f"Run it with: terradev ml deepeval run --file {output}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@deepeval.command("run")
@click.option("--file", "-f", default="test_deepeval.py", help="DeepEval test file")
def deepeval_run(file):
    """Run DeepEval tests."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "deepeval", "test", "run", "-x", file],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.echo(result.stdout)
        else:
            click.echo(f"ERROR:\n{result.stderr}", err=True)
    except FileNotFoundError:
        click.echo("ERROR: DeepEval not found. Run 'terradev ml deepeval install' first.", err=True)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)

@deepeval.command("metrics")
def deepeval_metrics():
    """List available DeepEval metrics."""
    click.echo("Available DeepEval metrics:")
    for m in DEEPEVAL_METRICS:
        click.echo(f"  {m}")

@deepeval.command("evaluate")
@click.option("--input", "-i", required=True, help="Test input/prompt")
@click.option("--actual-output", "-a", required=True, help="Actual LLM output")
@click.option(
    "--metric",
    "-m",
    required=True,
    type=click.Choice(DEEPEVAL_METRICS, case_sensitive=False),
    help="DeepEval metric to use",
)
@click.option("--expected-output", "-e", help="Expected output")
@click.option("--context", "-c", help="Ground-truth context (comma-separated)")
@click.option("--retrieval-context", "-r", help="Retrieval context (comma-separated)")
@click.option("--threshold", "-t", default=0.5, type=click.FloatRange(0.0, 1.0), help="Passing threshold")
def deepeval_evaluate(input, actual_output, metric, expected_output, context, retrieval_context, threshold):
    """Evaluate a single LLM output with a DeepEval metric."""
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError:
        click.echo("ERROR: DeepEval not installed. Run 'terradev ml deepeval install' first.", err=True)
        raise SystemExit(1)

    if metric in ("GEval", "DAGMetric"):
        click.echo(f"ERROR: {metric} requires a custom definition. Use 'deepeval run' with a test file.", err=True)
        raise SystemExit(1)

    kwargs: Dict[str, Any] = {"input": input, "actual_output": actual_output}
    if expected_output:
        kwargs["expected_output"] = expected_output
    if context:
        kwargs["context"] = [c.strip() for c in context.split(",") if c.strip()]
    if retrieval_context:
        kwargs["retrieval_context"] = [c.strip() for c in retrieval_context.split(",") if c.strip()]

    test_case = LLMTestCase(**kwargs)

    try:
        from deepeval import metrics as deepeval_metrics

        metric_cls = getattr(deepeval_metrics, metric, None)
        if metric_cls is None:
            click.echo(f"ERROR: Unknown metric {metric}", err=True)
            raise SystemExit(1)
        metric_obj = metric_cls(threshold=threshold)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to load metric: {e}", err=True)
        raise SystemExit(1)

    try:
        metric_obj.measure(test_case)
        click.echo(f"Score: {metric_obj.score}")
        click.echo(f"Reason: {metric_obj.reason}")
        click.echo(f"Passed: {metric_obj.is_successful()}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)
