#!/usr/bin/env python3
"""
Terradev CLI — compatibility entry point.

Commands are split into domain modules under `terradev_cli.commands` and
registered on the shared root `cli` group. This module keeps the remaining
command definitions that have not yet been extracted.
"""

import click
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import time
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import the root group and shared API helpers from the modular commands package.
# Commands extracted to terradev_cli.commands register themselves on `cli`.
from terradev_cli.commands import cli
from terradev_cli.commands._api import (
    TerradevAPI,
    validate_credentials,
    run_interactive_onboarding,
    _telemetry,
    TerraformWrapper,
    EnterpriseAuthManager,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest Cache + Drift Detection  CLI-native reliability
# ═══════════════════════════════════════════════════════════════════════════════


@cli.command("up")
@click.option("--job", "-j", required=True, help="Job name for manifest tracking")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
@click.option("--fix-drift", is_flag=True, help="Detect and fix drift automatically")
@click.option("--gpu-type", default="A100", help="GPU type")
@click.option("--gpu-count", type=int, default=1, help="Number of GPUs")
@click.option("--hours", type=float, default=1.0, help="Estimated runtime in hours")
@click.option("--budget", type=float, help="Budget constraint ($/hr)")
@click.option("--region", help="Preferred region")
@click.option("--dataset", help="Dataset path for drift detection")
@click.option("--ttl", default="1h", help="Time to live for nodes")
def up(
    job, cache_dir, fix_drift, gpu_type, gpu_count, hours, budget, region, dataset, ttl
):
    """CLI-native provisioning with manifest cache + drift detection"""
    import asyncio

    async def _up():
        from terradev_cli.core.manifest_cache import ManifestCache, Manifest, ManifestNode
        from terradev_cli.core.drift_detector import DriftDetector

        cache = ManifestCache(cache_dir)
        detector = DriftDetector(cache_dir)

        if fix_drift:
            # RE-PROVISION: Detect + Fix (single command)
            print(f" Detecting drift for job {job}...")

            try:
                result = await detector.fix_drift(job)

                if result["status"] == "no_drift":
                    print("OK: No drift detected - everything is in sync")
                elif result["status"] == "fixed":
                    print(" Drift fixed successfully:")
                    print(f"   Terminated: {result['terminated']} nodes")
                    print(f"   Recreated: {result['recreated']} nodes")
                else:
                    print("Warning  Partial fix - some nodes may still need attention")

                return result

            except Exception as e:  # noqa: BLE001
                print(f"ERROR: Error fixing drift: {e}")
                return

        # PROVISION: Auto-generates + stores manifest
        print(f"Deploying Provisioning job {job} with manifest cache...")

        # Get optimal deployment (existing logic)
        from terradev_cli.core.deployment_router import SmartDeploymentRouter

        router = SmartDeploymentRouter()
        user_request = {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "estimated_hours": hours,
            "budget": budget,
            "region": region,
        }

        recommendations = await router.recommend_deployments(user_request)

        if not recommendations:
            print("ERROR: No deployment options available")
            return

        # Choose best option (simplified - would normally prompt user)
        best_option = recommendations[0]

        print(f" Deploying: {best_option.provider} {best_option.instance_type}")
        print(f"   Cost: ${best_option.price_per_hour:.2f}/hr")
        print(f"   Confidence: {best_option.confidence:.1%}")

        # Execute deployment
        try:
            deployment_result = await router.execute_deployment(
                best_option, router.requirements_analyzer.analyze(user_request)
            )

            # Create manifest nodes
            nodes = []
            for i in range(gpu_count):
                node = ManifestNode(
                    provider=best_option.provider,
                    pod_id=f"{job}-node-{i+1}",
                    instance_id=deployment_result.get("instance_id", f"instance-{i+1}"),
                    gpus=1,
                    gpu_type=gpu_type,
                    region=region or "us-east-1",
                    status="running",
                    created_at=datetime.utcnow().isoformat(),
                    ttl=ttl,
                )
                nodes.append(node)

            # Create manifest
            version = f"v{len(cache.list_versions(job)) + 1}"
            dataset_hash = (
                cache.compute_dataset_hash(dataset) if dataset else "sha256:none"
            )

            manifest = Manifest(
                job=job,
                version=version,
                nodes=nodes,
                dataset_hash=dataset_hash,
                ttl=ttl,
                created_at=datetime.utcnow().isoformat(),
                metadata={
                    "deployment_id": deployment_result.get("deployment_id"),
                    "provider": best_option.provider,
                    "instance_type": best_option.instance_type,
                    "price_per_hour": best_option.price_per_hour,
                    "confidence": best_option.confidence,
                },
            )

            # Store manifest
            manifest_path = cache.store_manifest(manifest)

            print(f"OK: Job {job} provisioned successfully!")
            print(f"   Manifest: {manifest_path}")
            print(f"   Version: {version}")
            print(f"   Nodes: {len(nodes)}")
            print(f"   Fix drift: terradev up --job {job} --fix-drift")

        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Deployment failed: {e}")

    asyncio.run(_up())


@cli.command("rollback")
@click.argument("job_version", required=True)  # Format: job@v3
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
def rollback(job_version, cache_dir):
    """EXPLICIT ROLLBACK (versioned manifests)"""
    import asyncio

    async def _rollback():
        from terradev_cli.core.drift_detector import DriftDetector

        # Parse job@version
        if "@" not in job_version:
            print("ERROR: Invalid format. Use: job@version (e.g., llama3@v3)")
            return

        job, version = job_version.split("@", 1)

        print(f" Rolling back {job} to version {version}...")

        detector = DriftDetector(cache_dir)

        try:
            result = await detector.rollback(job, version)

            print("OK: Rollback completed:")
            print(f"   Target version: {result['target_version']}")
            print(f"   Terminated: {result['terminated']} nodes")
            print(f"   Recreated: {result['recreated']} nodes")

        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Rollback failed: {e}")
            sys.exit(1)

    asyncio.run(_rollback())


@cli.command("manifests")
@click.option("--job", help="Show versions for specific job")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
@click.option("--show-imported", is_flag=True, help="Show imported YAML pipelines")
@click.option("--show-recordings", is_flag=True, help="Show live recordings")
def manifests(job, cache_dir, show_imported, show_recordings):
    """List cached manifests, imported pipelines, and recordings"""
    try:
        from terradev_cli.core.manifest_cache import ManifestCache
        from pathlib import Path
    except ImportError:
        print(
            "ERROR: Manifest cache module not available. Install terradev_cli package."
        )
        sys.exit(1)

    cache = ManifestCache(cache_dir)

    if show_imported:
        # Show imported YAML pipelines
        print(" Imported YAML Pipelines:")

        # Look for manifests with yaml-import source
        manifest_files = list(Path(cache_dir).glob("*.json"))
        imported_jobs = []

        for file_path in manifest_files:
            try:
                with open(file_path, "r") as f:
                    manifest_data = json.load(f)

                if manifest_data.get("metadata", {}).get("source") == "yaml-import":
                    job_name = manifest_data["job"]
                    version = manifest_data["version"]
                    yaml_file = manifest_data["metadata"].get("yaml_file", "unknown")
                    workflow_name = manifest_data["metadata"].get(
                        "workflow_name", "unknown"
                    )

                    imported_jobs.append(
                        {
                            "name": job_name,
                            "version": version,
                            "yaml_file": yaml_file,
                            "workflow_name": workflow_name,
                            "created_at": manifest_data["created_at"],
                        }
                    )
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                continue

        if imported_jobs:
            print(
                "{'Job Name':<20} {'Version':<8} {'Workflow':<25} {'YAML File':<30} {'Created':<20}"
            )
            print("─" * 115)

            for job_info in sorted(
                imported_jobs, key=lambda x: x["created_at"], reverse=True
            ):
                yaml_name = Path(job_info["yaml_file"]).name
                created = job_info["created_at"][:19].replace("T", " ")
                print(
                    f"{job_info['name']:<20} {job_info['version']:<8} {job_info['workflow_name']:<25} {yaml_name:<30} {created:<20}"
                )
        else:
            print("   No imported YAML pipelines found")
            print("   Tip: Use 'terradev import pipeline.yaml' to register a pipeline")

        return

    if show_recordings:
        # Show live recordings
        print(" Live Recordings:")

        recordings_dir = Path("./recordings")
        if recordings_dir.exists():
            recording_files = list(recordings_dir.glob("*.recording"))

            if recording_files:
                print(f"{'Name':<20} {'Status':<12} {'Started':<20} {'Stopped':<20}")
                print("─" * 75)

                for recording_file in recording_files:
                    try:
                        with open(recording_file, "r") as f:
                            recording_data = json.load(f)

                        name = recording_data["name"]
                        status = recording_data["status"]
                        started = recording_data["started_at"][:19].replace("T", " ")
                        stopped = recording_data.get("stopped_at", "")[:19].replace(
                            "T", " "
                        )

                        print(f"{name:<20} {status:<12} {started:<20} {stopped:<20}")
                    except Exception as _exc:  # noqa: BLE001
                        logger.exception(_exc)
                        continue
            else:
                print("   No recordings found")
                print(
                    "   Tip: Use 'terradev record start --name my-recording' to start recording"
                )
        else:
            print("   No recordings directory found")
            print(
                "   Tip: Use 'terradev record start --name my-recording' to start recording"
            )

        return

    if job:
        # Show versions for specific job
        versions = cache.list_versions(job)
        if versions:
            print(f" Manifest versions for {job}:")
            for version in versions:
                manifest = cache.load_manifest(job, version)
                if manifest:
                    source = manifest.metadata.get("source", "provisioned")
                    source_icon = "" if source == "yaml-import" else ""

                    print(
                        f"   {source_icon} {version}: {len(manifest.nodes)} nodes, created {manifest.created_at}"
                    )

                    if source == "yaml-import":
                        yaml_file = manifest.metadata.get("yaml_file", "unknown")
                        print(f"      Source: {Path(yaml_file).name}")
        else:
            print(f"ERROR: No manifests found for job {job}")
    else:
        # Show all jobs
        manifest_files = list(Path(cache_dir).glob("*.json"))
        if manifest_files:
            jobs = set()
            imported_jobs = set()

            for file_path in manifest_files:
                try:
                    with open(file_path, "r") as f:
                        manifest_data = json.load(f)

                    job_name = manifest_data["job"]
                    jobs.add(job_name)

                    if manifest_data.get("metadata", {}).get("source") == "yaml-import":
                        imported_jobs.add(job_name)
                except Exception as _exc:  # noqa: BLE001
                    logger.exception(_exc)
                    parts = file_path.stem.split(".")
                    if len(parts) >= 2:
                        jobs.add(parts[0])

            if jobs:
                print(" Cached jobs:")
                for job_name in sorted(jobs):
                    versions = cache.list_versions(job_name)
                    icon = "" if job_name in imported_jobs else ""
                    print(f"   {icon} {job_name}: {len(versions)} versions")

                print("\nTip: Use --show-imported to see imported YAML pipelines")
                print("Tip: Use --show-recordings to see live recordings")
                print("Tip: Use --job <name> to see detailed versions")
            else:
                print("ERROR: No cached manifests found")
                print("   Tip: Use 'terradev provision' to create a job")
                print(
                    "   Tip: Use 'terradev import pipeline.yaml' to register a pipeline"
                )
        else:
            print("ERROR: No cached manifests found")
            print("   Tip: Use 'terradev provision' to create a job")
            print("   Tip: Use 'terradev import pipeline.yaml' to register a pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Spaces One-Click Deployment
# ═══════════════════════════════════════════════════════════════════════════════


@cli.command("hf-space")
@click.argument("space_name", required=True)
@click.option("--model-id", required=True, help="HuggingFace model ID to deploy")
@click.option(
    "--hardware",
    default="cpu-basic",
    type=click.Choice(
        ["cpu-basic", "cpu-upgrade", "t4-medium", "a10g-large", "a100-large"]
    ),
    help="Hardware tier for the Space",
)
@click.option(
    "--sdk",
    default="gradio",
    type=click.Choice(["gradio", "streamlit", "docker"]),
    help="SDK for the Space",
)
@click.option("--private", is_flag=True, help="Make the Space private")
@click.option(
    "--template",
    type=click.Choice(["llm", "embedding", "image"]),
    help="Use pre-configured template",
)
@click.option("--env", "-e", multiple=True, help="Environment variables KEY=VALUE")
@click.option("--secret", "-s", multiple=True, help="Secrets KEY=VALUE")
def hf_space(space_name, model_id, hardware, sdk, private, template, env, secret):
    """One-click HuggingFace Spaces deployment"""
    import asyncio

    async def _hf_space():
        from terradev_cli.core.hf_spaces import HFSpacesDeployer, HFSpaceConfig, HFSpaceTemplates

        # Get HF token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            print("ERROR: HF_TOKEN environment variable required")
            print("   Set it with: export HF_TOKEN=your_token")
            sys.exit(2)

        deployer = HFSpacesDeployer(hf_token)

        # Parse environment variables
        env_vars = {}
        for env_var in env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                env_vars[key] = value

        # Parse secrets
        secrets = {}
        for secret_var in secret:
            if "=" in secret_var:
                key, value = secret_var.split("=", 1)
                secrets[key] = value

        # Use template or create custom config
        if template:
            if template == "llm":
                config = HFSpaceTemplates.get_llm_template(model_id, space_name)
            elif template == "embedding":
                config = HFSpaceTemplates.get_embedding_template(model_id, space_name)
            elif template == "image":
                config = HFSpaceTemplates.get_image_model_template(model_id, space_name)

            # Override with user-specified options
            config.hardware = hardware
            config.sdk = sdk
            config.private = private
            config.env_vars.update(env_vars)
            if secrets:
                config.secrets = secrets
        else:
            config = HFSpaceConfig(
                name=space_name,
                model_id=model_id,
                hardware=hardware,
                sdk=sdk,
                python_version="3.10",
                private=private,
                env_vars=env_vars,
                secrets=secrets if secrets else None,
            )

        print(f"Deploying {model_id} to HuggingFace Spaces...")
        print(f"   Space: {space_name}")
        print(f"   Hardware: {hardware}")
        print(f"   SDK: {sdk}")

        try:
            result = await deployer.create_space(config)

            if result["status"] == "created":
                print("OK: Space created successfully!")
                print(f"    Space URL: {result['space_url']}")
                print(f"    Hardware: {result['hardware']}")
                print(f"    Model: {result['model_id']}")
                print("     Your Space will be ready in 2-5 minutes")
                print("   Status 100k+ researchers can now access your model!")
            else:
                print(f"ERROR: Failed to create space: {result['error']}")

        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Deployment failed: {e}")

    asyncio.run(_hf_space())


# GitOps Commands
# InferX Commands


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-LoRA Adapter Management (vLLM ≥0.15.0)
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_vllm_endpoint(endpoint: str):
    """Parse 'http://host:port' into (host, port)."""
    from urllib.parse import urlparse

    p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return p.hostname or "127.0.0.1", p.port or 8000


# ── Registry Commands ──


# ── Existing Commands (Updated for Registry) ──


# ── Versioning Commands ──


# ── Cost Attribution Commands ──


# ── LoRAX Integration (Predibase LoRA eXchange) ──


# ── HuggingFace PEFT Import ──


# ═══════════════════════════════════════════════════════════════════════════════
# Phoenix  LLM Trace Observability (Arize Phoenix, ELv2)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# NeMo Guardrails  Output Safety Layer (Apache 2.0)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant  Vector Database for RAG (Apache 2.0)
# ═══════════════════════════════════════════════════════════════════════════════


# Enterprise SSO Commands


# ── SGLANG COMMAND GROUP ──


# ═══════════════════════════════════════════════════════════════════════════════
# Drift-Triggered Continuous Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Langfuse  LLM Observability, Scoring & Dataset Management
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Databricks  Jobs, Clusters, Model Serving, MLflow
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Serving  KV Cache TTL, Prefix Caching, LMCache, Priority Scheduling
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Model Router  Cost/Quality Routing for Agentic Workloads
# ═══════════════════════════════════════════════════════════════════════════════


# ── MIGRATE COMMAND GROUP ──


# ── EVAL COMMAND GROUP ──


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Import/Export Commands
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Event System Commands - Triggers, Environments, and Lineage
# ═══════════════════════════════════════════════════════════════════════


# (command groups self-register via @cli.group() decorators)


# Register Karpenter and HF Spaces command groups
from terradev_cli.cli_karpenter import register_karpenter_commands

register_karpenter_commands(cli, TerradevAPI)

from terradev_cli.cli_hf_spaces import register_hf_spaces_commands

register_hf_spaces_commands(cli, TerradevAPI)


# MCP Server Command


def _register_local_pool(gpus, pool_name, host=None, user=None, key=None):
    """Write pool entry to ~/.terradev/local_pool.json"""
    import json
    import os
    import datetime

    pool_path = os.path.expanduser("~/.terradev/local_pool.json")
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    pool = {}
    if os.path.exists(pool_path):
        try:
            with open(pool_path) as f:
                pool = json.load(f)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pool = {}
    pool[pool_name] = {
        "name": pool_name,
        "gpus": gpus,
        "host": host or "localhost",
        "user": user,
        "key": key,
        "registered_at": datetime.datetime.utcnow().isoformat(),
        "price_per_hour": 0.0,
        "provider": "local",
    }
    with open(pool_path, "w") as f:
        json.dump(pool, f, indent=2)


# ── Agent Fleet Command Group ─────────────────────────────────────────────────
# terradev agent <subcommand>
# Research basis: arXiv:2605.26297 "Agentic AI Workload Characteristics"
#   - Decode dominates: 91-98% of LLM time
#   - KV cache hit rates: 84.6-99.5% (eviction = expensive recompute)
#   - Context footprint: 37K-166K tokens (P95 tail: 120K)
#   - Tool calls: 2-29% of runtime


# Gateway command for inference serving


# Observe Command - Unified Monitoring Pipeline


# Schedule Command - Spot-Aware Scheduling


if __name__ == "__main__":
    cli()
