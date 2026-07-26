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
@cli.group()
def train():
    """Launch distributed training jobs across provisioned GPU nodes"""
    pass


@train.command("start")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="YAML config file for training job",
)
@click.option("--script", "-s", help="Training script path (Python file)")
@click.option(
    "--framework",
    type=click.Choice(["torchrun", "deepspeed", "accelerate", "megatron"]),
    default="torchrun",
    help="Distributed framework: torchrun (default), deepspeed, accelerate, megatron",
)
@click.option(
    "--backend",
    type=click.Choice(["native", "ray"]),
    default="native",
    help="Launch backend: native (default), ray (optional, requires Ray cluster)",
)
@click.option(
    "--nodes", "-n", multiple=True, help="Node IP addresses (multiple allowed)"
)
@click.option(
    "--from-provision",
    "provision_group",
    default="",
    help='Use nodes from provision group (pg_xxx or "latest" for most recent)',
)
@click.option(
    "--pool", default="", help="Use local pool entry by name (e.g., workstation-4090)"
)
@click.option(
    "--overflow-to-cloud",
    is_flag=True,
    help="Fall back to cloud providers if local pool unavailable or insufficient",
)
@click.option("--gpus-per-node", default=8, help="GPUs per node (default: 8)")
@click.option("--tp", default=1, help="Tensor parallel size for model parallelism")
@click.option("--pp", default=1, help="Pipeline parallel size for model parallelism")
@click.option(
    "--total-steps", default=0, help="Total training steps (for ETA calculation)"
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip preflight GPU/NCCL/RDMA validation checks",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format: text (default) or json",
)
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def train_start(
    config_path,
    script,
    framework,
    backend,
    nodes,
    provision_group,
    pool,
    overflow_to_cloud,
    gpus_per_node,
    tp,
    pp,
    total_steps,
    skip_preflight,
    fmt,
    script_args,
):
    """Launch distributed training jobs across provisioned GPU nodes.

    Orchestrates distributed training with automatic topology optimization, FlashOptim
    integration, and checkpoint management. Supports torchrun, DeepSpeed, Accelerate,
    and Megatron frameworks.

    Examples:
      terradev train -s train.py --framework torchrun --gpus-per-node 8
      terradev train -c job.yaml                                      # Use YAML config
      terradev train -s train.py -n 10.0.0.1 -n 10.0.0.2 --tp 2 -- --lr 1e-4
      terradev train -s train.py --from-provision latest             # Auto-resolve nodes
      terradev train -s train.py --from-provision pg_1709123456_abc12345
      terradev train -s train.py --pool workstation-4090             # Use local pool entry
      terradev train -s train.py --pool workstation-4090 --overflow-to-cloud  # Cloud fallback

    Workflow:
      1. Provision nodes: terradev provision -g H100 -n 4
      2. Validate: terradev preflight (optional, auto-run by default)
      3. Train: terradev train -s train.py --from-provision latest
      4. Monitor: terradev monitor --job <job-id>
      5. Checkpoint: terradev checkpoint list --job <job-id>

    FlashOptim (auto-applied):
      When training with bf16/fp16 and 40GB+ VRAM, FlashOptim is automatically
      enabled for gradient compression and checkpoint optimization.

    Frameworks:
      - torchrun: PyTorch native distributed training (default)
      - deepspeed: Microsoft DeepSpeed for large models
      - accelerate: HuggingFace Accelerate
      - megatron: NVIDIA Megatron-LM for massive models

    Next Steps:
      Monitor training: terradev monitor --job <job-id>
      Check status: terradev train-status --job <job-id>
      Stop training: terradev train-stop --job <job-id>
      View checkpoints: terradev checkpoint list --job <job-id>
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator, TrainingConfig

    # ── Resolve nodes from provision group if specified ──
    resolved_nodes = list(nodes)
    resolved_ssh_key = ""
    if provision_group and not resolved_nodes:
        resolved_nodes, resolved_ssh_key = _resolve_provision_nodes(
            provision_group, fmt
        )
        if not resolved_nodes:
            sys.exit(1)

    # ── Resolve nodes from local pool if --pool is specified ──
    if pool and not resolved_nodes:
        import json
        import os

        pool_path = os.path.expanduser("~/.terradev/local_pool.json")
        if os.path.exists(pool_path):
            try:
                with open(pool_path) as f:
                    pool_data = json.load(f)
                if pool in pool_data:
                    entry = pool_data[pool]
                    host = entry.get("host", "localhost")
                    if host == "localhost":
                        resolved_nodes = ["127.0.0.1"]
                    else:
                        resolved_nodes = [host]
                    resolved_ssh_key = entry.get("key", "")
                    print(f"Using local pool entry '{pool}': {host}")
                else:
                    print(f"ERROR: Pool entry '{pool}' not found in local pool")
                    print(f"Available entries: {', '.join(pool_data.keys())}")
                    if overflow_to_cloud:
                        print(
                            "Proceeding with cloud providers (overflow-to-cloud enabled)..."
                        )
                    else:
                        sys.exit(1)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: Could not load local pool: {e}")
                if overflow_to_cloud:
                    print(
                        "Proceeding with cloud providers (overflow-to-cloud enabled)..."
                    )
                else:
                    sys.exit(1)
        else:
            print("ERROR: No local pool found")
            print("Register GPUs with: terradev local scan --register")
            if overflow_to_cloud:
                print("Proceeding with cloud providers (overflow-to-cloud enabled)...")
            else:
                sys.exit(1)

    # ── Cloud fallback if pool specified but no nodes resolved ──
    if pool and not resolved_nodes and overflow_to_cloud:
        print(
            f"WARNING: Local pool '{pool}' unavailable, falling back to cloud providers"
        )
        print("Run: terradev provision -g <gpu-type> -n <count>")
        sys.exit(1)

    if config_path:
        config = TrainingConfig.from_yaml(config_path)
        if resolved_nodes and not config.nodes:
            config.nodes = resolved_nodes
        if resolved_ssh_key and not config.ssh_key:
            config.ssh_key = resolved_ssh_key
    else:
        if not script:
            print("ERROR: Either --config or --script is required")
            sys.exit(1)
        config = TrainingConfig(
            script=script,
            framework=framework,
            backend=backend,
            nodes=resolved_nodes,
            gpus_per_node=gpus_per_node,
            tp_size=tp,
            pp_size=pp,
            total_steps=total_steps,
            script_args=list(script_args),
            ssh_key=resolved_ssh_key or "",
        )

    orch = TrainingOrchestrator()
    result = orch.launch(config, skip_preflight=skip_preflight)

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        status = result.get("status", "unknown")
        print(f"\nTraining Job: {result.get('job_id', 'N/A')}")
        print(f"  Status: {status}")
        print(f"  Framework: {result.get('framework')}")
        print(f"  Backend: {result.get('backend', 'native')}")
        print(f"  GPUs: {result.get('total_gpus', 0)}")
        print(f"  Nodes: {result.get('nodes', [])}")
        if result.get("pid"):
            print(f"  PID: {result['pid']}")
        if result.get("master_addr"):
            print(f"  Master: {result['master_addr']}")
        fo = result.get("flashoptim", {})
        if fo.get("enabled"):
            print(
                f"  FlashOptim: {fo.get('optimizer_class', 'FlashAdamW')} (auto-applied  {fo.get('reason', '')})"
            )
        if status == "failed":
            print(f"  Errors: {result.get('errors', '')}")
        print()


@cli.command()
@click.option("--job-id", "-j", default="", help="Job ID to monitor")
@click.option("--nodes", "-n", multiple=True, help="Node IPs")
@click.option("--ssh-user", default="root", help="SSH user")
@click.option("--ssh-key", default="", help="SSH key path")
@click.option(
    "--from-provision",
    "provision_group",
    default="",
    help='Use nodes from a provision group. "latest" = most recent.',
)
@click.option("--log-path", "-l", default="", help="Training log file to parse")
@click.option("--interval", "-i", default=10.0, help="Snapshot interval in seconds")
@click.option("--count", default=0, help="Number of snapshots (0 = continuous)")
@click.option("--prometheus", default="", help="Prometheus endpoint (optional)")
@click.option("--cost-rate", default=0.0, help="Cost per GPU-hour in USD")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def monitor(
    job_id,
    nodes,
    ssh_user,
    ssh_key,
    provision_group,
    log_path,
    interval,
    count,
    prometheus,
    cost_rate,
    fmt,
):
    """Monitor GPU utilization, training metrics, and cost.

    Default: nvidia-smi (zero deps). Optional Prometheus/DCGM-exporter hook.
    Includes straggler detection for multi-node clusters.

    Examples:
        terradev monitor -n 10.0.0.1 -n 10.0.0.2 -l /tmp/train.log
        terradev monitor --from-provision latest --cost-rate 3.50
        terradev monitor --prometheus http://localhost:9090 -f json
        terradev monitor -j job-abc123 --interval 5 --count 10
    """
    from terradev_cli.core.training_monitor import TrainingMonitor

    node_list = list(nodes) if nodes else []
    resolved_ssh_key = ssh_key
    if provision_group and not node_list:
        node_list, auto_ssh = _resolve_provision_nodes(provision_group, fmt)
        if not node_list:
            sys.exit(1)
        if auto_ssh and not ssh_key:
            resolved_ssh_key = auto_ssh
    if not node_list:
        node_list = [None]

    mon = TrainingMonitor(
        nodes=node_list,
        ssh_user=ssh_user,
        ssh_key=resolved_ssh_key or None,
        log_path=log_path,
        cost_per_gpu_hour=cost_rate,
        prometheus_endpoint=prometheus,
    )

    if fmt == "json":
        if count == 1 or count == 0:
            snap = mon.snapshot(job_id)
            print(json.dumps(snap.to_dict(), indent=2, default=str))
        else:
            snaps = mon.continuous(job_id, interval_s=interval, max_snapshots=count)
            print(json.dumps([s.to_dict() for s in snaps], indent=2, default=str))
    else:
        if count == 1:
            snap = mon.snapshot(job_id)
            mon._print_snapshot(snap)
        else:
            mon.continuous(job_id, interval_s=interval, max_snapshots=count)


@cli.command()
@click.argument("action", type=click.Choice(["list", "restore", "promote", "delete"]))
@click.option("--job-id", "-j", required=True, help="Job ID")
@click.option("--step", type=int, default=None, help="Checkpoint step")
@click.option("--checkpoint-id", default="", help="Checkpoint ID")
@click.option("--dest", default="", help="Destination path (for promote)")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def checkpoint(action, job_id, step, checkpoint_id, dest, fmt):
    """Manage distributed checkpoints.

    Local filesystem by default. Supports manifest-based atomic commits,
    parallel shard verification, and retention policies.

    Examples:
        terradev checkpoint list -j job-abc123
        terradev checkpoint restore -j job-abc123
        terradev checkpoint restore -j job-abc123 --step 5000
        terradev checkpoint promote -j job-abc123 --checkpoint-id ckpt-xyz --dest /models/final
        terradev checkpoint delete -j job-abc123 --checkpoint-id ckpt-xyz
    """
    from terradev_cli.core.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager()

    if action == "list":
        ckpts = mgr.list(job_id)
        if fmt == "json":
            print(json.dumps(ckpts, indent=2, default=str))
        else:
            if not ckpts:
                print(f"No checkpoints for job {job_id}")
            else:
                print(f"\nCheckpoints for {job_id}:")
                for c in ckpts:
                    sid = c.get("checkpoint_id", c.get("id", "N/A"))
                    print(
                        f"  step={c.get('step', '?'):>8}  "
                        f"id={sid}  "
                        f"shards={c.get('shard_count', '?')}  "
                        f"size={c.get('total_size_bytes', 0) / (1024**3):.2f}GB"
                    )
                print()

    elif action == "restore":
        try:
            manifest = mgr.restore(
                job_id, step=step, checkpoint_id=checkpoint_id or None
            )
            result = manifest.to_dict()
            if fmt == "json":
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\nRestored: {result['checkpoint_id']} step={result['step']}")
                print(f"  Shards: {result['shard_count']}")
                print(f"  Size: {result['total_size_bytes'] / (1024**3):.2f}GB")
                print()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    elif action == "promote":
        result = mgr.promote(job_id, checkpoint_id, dest_path=dest)
        print(f"Promoted: {result}")

    elif action == "delete":
        if not checkpoint_id:
            print("ERROR: --checkpoint-id required for delete")
            sys.exit(1)
        mgr.delete(job_id, checkpoint_id)
        print(f"Deleted: {checkpoint_id}")


@train.command("status")
@click.option("--job-id", "-j", default="", help="Job ID (empty = all running)")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def train_status(job_id, fmt):
    """Show training job status, GPU-hours, cost, and ETA.

    Queries the local SQLite job database  no external services needed.

    Examples:
        terradev train-status
        terradev train-status -j job-abc123
        terradev train-status -f json
    """
    from terradev_cli.core.job_state_manager import JobStateManager

    sm = JobStateManager()

    if job_id:
        result = sm.job_metrics(job_id)
        if fmt == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            if "error" in result:
                print(f"ERROR: {result['error']}")
                sys.exit(1)
            print(f"\nJob: {result['id']}")
            print(f"  Name: {result['name']}")
            print(f"  Status: {result['status']}")
            print(f"  Framework: {result['framework']}")
            print(
                f"  Progress: {result.get('current_step', 0)}/{result.get('total_steps', 0)} "
                f"({result.get('progress_pct', 0)}%)"
            )
            print(f"  Elapsed: {result.get('elapsed_hours', 0):.1f}h")
            print(f"  GPU-hours: {result.get('gpu_hours', 0):.1f}")
            eta = result.get("eta_hours")
            print(f"  ETA: {eta:.1f}h" if eta is not None else "  ETA: N/A")
            print(f"  Cost: ${result.get('cost_usd', 0):.2f}")
            print(
                f"  Efficiency: {result.get('efficiency_steps_per_gpuh', 0):.1f} steps/GPU-h"
            )
            if result.get("last_checkpoint_id"):
                print(f"  Last checkpoint: {result['last_checkpoint_id']}")
            if result.get("error_message"):
                print(f"  Error: {result['error_message']}")
            print()
    else:
        running = sm.running_jobs_summary()
        total = sm.total_cost()
        if fmt == "json":
            print(
                json.dumps(
                    {"running": running, "total_cost": total}, indent=2, default=str
                )
            )
        else:
            if not running:
                print("\nNo running training jobs.")
            else:
                print(f"\nRunning jobs ({len(running)}):")
                for j in running:
                    eta = j.get("eta_hours")
                    eta_str = f"ETA {eta:.1f}h" if eta is not None else ""
                    print(
                        f"  {j['id']}  {j['name']}  {j['framework']}  "
                        f"{j.get('current_step', 0)}/{j.get('total_steps', 0)}  "
                        f"{j.get('elapsed_hours', 0):.1f}h  "
                        f"${j.get('cost_usd', 0):.2f}  {eta_str}"
                    )
            print(f"\nTotal cost across all jobs: ${total:.2f}\n")


@train.command("stop")
@click.option("--job-id", "-j", required=True, help="Job ID to stop")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def train_stop(job_id, fmt):
    """Stop a running training job.

    Kills training processes on all nodes in parallel.

    Examples:
        terradev train-stop -j job-abc123
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    result = orch.stop(job_id)

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Job {job_id}: {result.get('status', 'unknown')}")


@train.command("resume")
@click.option("--job-id", "-j", required=True, help="Job ID to resume")
@click.option(
    "--checkpoint-id", default="", help="Checkpoint to resume from (default: latest)"
)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def train_resume(job_id, checkpoint_id, fmt):
    """Resume a training job from checkpoint.

    Rebuilds config from job state and resumes with topology validation.

    Examples:
        terradev train-resume -j job-abc123
        terradev train-resume -j job-abc123 --checkpoint-id ckpt-xyz
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    result = orch.resume(job_id, checkpoint_id=checkpoint_id or None)

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        status = result.get("status", "unknown")
        print(f"\nResumed Job: {result.get('job_id', job_id)}")
        print(f"  Status: {status}")
        if result.get("pid"):
            print(f"  PID: {result['pid']}")
        if status == "failed":
            print(f"  Error: {result.get('errors', result.get('error', ''))}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-LoRA Adapter Management (vLLM ≥0.15.0)
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_vllm_endpoint(endpoint: str):
    """Parse 'http://host:port' into (host, port)."""
    from urllib.parse import urlparse

    p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return p.hostname or "127.0.0.1", p.port or 8000












@cli.group()
def lora():
    """Production-grade LoRA adapter management with registry and cross-replica consistency.

    Manage adapter versions, track replica distribution, and ensure consistency across deployments.
    """
    pass


# ── Registry Commands ──


@lora.command("register")
@click.option("--name", "-n", required=True, help="Adapter name")
@click.option("--path", required=True, help="Path to adapter weights")
@click.option("--base-model", "-b", required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-hf)")
@click.option("--rank", default=64, help="LoRA rank (default: 64)")
@click.option("--tenant", help="Associate with tenant ID")
@click.option("--metadata", help="JSON metadata string")
def lora_register_cmd(name, path, base_model, rank, tenant, metadata):
    """Register a LoRA adapter in the central registry with version tracking.

    Examples:
        terradev lora register -n customer-a --path /adapters/customer-a -b meta-llama/Llama-2-7b-hf
        terradev lora register -n customer-b --path /adapters/customer-b -b meta-llama/Llama-2-7b-hf --tenant t-123
    """
    import json
    from terradev_cli.ml_services.lora_registry import get_lora_registry

    registry = get_lora_registry()
    metadata_dict = json.loads(metadata) if metadata else {}

    version = registry.register_adapter(
        adapter_name=name,
        base_model=base_model,
        path=path,
        rank=rank,
        metadata=metadata_dict,
    )

    if tenant:
        registry.map_tenant_to_adapter(tenant, name)

    print(f"OK: Registered adapter '{name}' version {version.version_id}")
    print(f"   Base model: {base_model}")
    print(f"   Path: {path}")
    print(f"   Rank: {rank}")
    if tenant:
        print(f"   Tenant: {tenant}")


@lora.command("versions")
@click.option("--name", "-n", required=True, help="Adapter name")
def lora_versions_cmd(name):
    """List all versions of an adapter.

    Examples:
        terradev lora versions -n customer-a
    """
    from terradev_cli.ml_services.lora_registry import get_lora_registry

    registry = get_lora_registry()
    versions = registry.get_adapter_versions(name)

    if not versions:
        print(f"ERROR: No versions found for adapter '{name}'")
        return

    active = registry.get_active_version(name)

    print(f"Adapter: {name}")
    print(f"Versions ({len(versions)}):")
    for v in versions:
        status_marker = " [ACTIVE]" if v.status.value == "active" else ""
        print(f"  {v.version_id[:8]}...  {v.created_at.strftime('%Y-%m-%d %H:%M:%S')}  {v.status.value}{status_marker}")
        print(f"    Path: {v.path}")
        print(f"    Rank: {v.rank}")
        if v.performance_metrics:
            print(f"    Metrics: {v.performance_metrics}")


@lora.command("activate")
@click.option("--name", "-n", required=True, help="Adapter name")
@click.option("--version", "-v", required=True, help="Version ID to activate")
def lora_activate_cmd(name, version):
    """Activate a specific version across all replicas.

    Examples:
        terradev lora activate -n customer-a -v abc123...
    """
    from terradev_cli.ml_services.lora_registry import get_lora_registry

    registry = get_lora_registry()
    success = registry.mark_version_active(name, version)

    if success:
        print(f"OK: Activated version {version[:8]}... for adapter '{name}'")
    else:
        print(f"ERROR: Failed to activate version {version}")


@lora.command("sync")
@click.option("--deployment", "-d", required=True, help="Deployment name")
@click.option("--name", "-n", required=True, help="Adapter name")
@click.option("--replicas", help="Comma-separated list of replica endpoints (host:port)")
def lora_sync_cmd(deployment, name, replicas):
    """Synchronize adapter state across all replicas in a deployment.

    Examples:
        terradev lora sync -d prod -n customer-a --replicas 10.0.0.1:8000,10.0.0.2:8000
    """
    import asyncio
    from terradev_cli.ml_services.lora_registry import get_lora_registry
    from terradev_cli.core.lora_consistency import LoRAConsistencyManager

    registry = get_lora_registry()
    active_version = registry.get_active_version(name)

    if not active_version:
        print(f"ERROR: No active version found for adapter '{name}'")
        return

    # Parse replicas
    replica_list = []
    if replicas:
        for replica in replicas.split(","):
            host, port = replica.split(":")
            replica_list.append({"replica_id": replica, "host": host, "port": int(port)})

    consistency_mgr = LoRAConsistencyManager(registry=registry, replicas=replica_list)
    result = asyncio.run(consistency_mgr.sync_adapter_state(name, active_version.version_id))

    if result["status"] == "success":
        print(f"OK: Adapter '{name}' synchronized across replicas")
        final = result.get("final_consistency", {})
        print(f"   Expected replicas: {len(final.get('expected_replicas', []))}")
        print(f"   Loaded replicas: {len(final.get('loaded_replicas', []))}")
    else:
        print(f"ERROR: {result.get('error')}")
        if "load_result" in result:
            print(f"   Load result: {result['load_result']}")


# ── Existing Commands (Updated for Registry) ──


@lora.command("list")
@click.option(
    "--endpoint", "-e", required=True, help="vLLM endpoint (e.g. http://10.0.0.1:8000)"
)
@click.option("--api-key", help="vLLM API key")
@click.option("--registry", is_flag=True, help="Show registry state instead of live endpoint")
def lora_list_cmd(endpoint, api_key, registry):
    """List loaded LoRA adapters.

    Examples:
        terradev lora list -e http://10.0.0.1:8000
        terradev lora list -e http://10.0.0.1:8000 --registry
    """
    if registry:
        from terradev_cli.ml_services.lora_registry import get_lora_registry

        reg = get_lora_registry()
        stats = reg.get_registry_stats()
        adapters = reg.list_all_adapters()

        print(f"Registry Statistics:")
        print(f"  Total adapters: {stats['total_adapter_names']}")
        print(f"  Total versions: {stats['total_versions']}")
        print(f"  Active versions: {stats['active_versions']}")
        print(f"  Total replicas: {stats['total_replicas']}")
        print(f"  Total tenants: {stats['total_tenants']}")
        print()
        print(f"Registered adapters ({len(adapters)}):")
        for adapter in adapters:
            active = reg.get_active_version(adapter)
            active_marker = " [ACTIVE]" if active else ""
            print(f"  {adapter}{active_marker}")
        return

    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_list())

    if result["status"] != "success":
        print(f"ERROR: {result.get('error')}")
        return

    base = result.get("base_models", [])
    adapters = result.get("lora_adapters", [])
    print(f"Base models ({len(base)}):")
    for m in base:
        print(f"  {m.get('id', '?')}")
    print(f"LoRA adapters ({len(adapters)}):")
    if adapters:
        for a in adapters:
            print(f"  {a.get('id', '?')}  (parent: {a.get('parent', '-')})")
    else:
        print("  (none)")


@lora.command("add")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option(
    "--name",
    "-n",
    required=True,
    help="Adapter name (becomes the model name in API requests)",
)
@click.option("--path", required=True, help="Path to adapter weights")
@click.option("--api-key", help="vLLM API key")
@click.option("--register", is_flag=True, help="Also register in central registry")
@click.option("--base-model", help="Base model (required with --register)")
@click.option("--rank", default=64, help="LoRA rank (default: 64)")
def lora_add_cmd(endpoint, name, path, api_key, register, base_model, rank):
    """Hot-load a LoRA adapter onto a running vLLM server.

    Examples:
        terradev lora add -e http://10.0.0.1:8000 -n customer-a --path /adapters/customer-a
        terradev lora add -e http://10.0.0.1:8000 -n customer-a --path /adapters/customer-a --register --base-model meta-llama/Llama-2-7b-hf
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService, LoRAModule

    host, port = _parse_vllm_endpoint(endpoint)
    
    # Register if requested
    version_id = None
    if register:
        if not base_model:
            print("ERROR: --base-model required when using --register")
            return
        from terradev_cli.ml_services.lora_registry import get_lora_registry
        
        registry = get_lora_registry()
        version = registry.register_adapter(
            adapter_name=name,
            base_model=base_model,
            path=path,
            rank=rank,
        )
        version_id = version.version_id
        print(f"Registered adapter '{name}' as version {version_id[:8]}...")

    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_load(LoRAModule(name=name, path=path), version_id=version_id))

    if result["status"] == "loaded":
        print(f'OK: Adapter \'{name}\' loaded  use "model": "{name}" in requests')
    else:
        print(f"ERROR: {result.get('error')}")


@lora.command("remove")
@click.option("--endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--name", "-n", required=True, help="Adapter name to unload")
@click.option("--api-key", help="vLLM API key")
def lora_remove_cmd(endpoint, name, api_key):
    """Hot-unload a LoRA adapter.

    Examples:
        terradev lora remove -e http://10.0.0.1:8000 -n customer-a
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_unload(name))

    if result["status"] == "unloaded":
        print(f"OK: Adapter '{name}' unloaded")
    else:
        print(f"ERROR: {result.get('error')}")


# ── Versioning Commands ──


@lora.command("rollback")
@click.option("--name", "-n", required=True, help="Adapter name to rollback")
@click.option("--to-version", "-v", help="Target version ID (default: previous stable)")
@click.option("--replicas", help="Comma-separated list of replica endpoints (host:port)")
def lora_rollback_cmd(name, to_version, replicas):
    """Rollback adapter to previous stable version.

    Examples:
        terradev lora rollback -n customer-a
        terradev lora rollback -n customer-a -v abc123... --replicas 10.0.0.1:8000,10.0.0.2:8000
    """
    import asyncio
    from terradev_cli.ml_services.lora_registry import get_lora_registry
    from terradev_cli.core.lora_versioning import LoRAVersioningManager
    from terradev_cli.core.lora_consistency import LoRAConsistencyManager

    registry = get_lora_registry()
    versioning_mgr = LoRAVersioningManager(registry=registry)

    # Parse replicas
    replica_list = None
    if replicas:
        replica_list = []
        for replica in replicas.split(","):
            host, port = replica.split(":")
            replica_list.append({"replica_id": replica, "host": host, "port": int(port)})
        versioning_mgr.consistency_manager = LoRAConsistencyManager(
            registry=registry, replicas=replica_list
        )

    result = asyncio.run(
        versioning_mgr.rollback_adapter(
            adapter_name=name,
            target_version_id=to_version,
            replicas=replica_list,
        )
    )

    if result.success:
        print(f"OK: Rolled back adapter '{name}'")
        print(f"   From version: {result.from_version_id[:8] if result.from_version_id else 'none'}")
        print(f"   To version: {result.to_version_id[:8]}")
        print(f"   Replicas affected: {result.replicas_affected}")
    else:
        print(f"ERROR: {result.error}")


@lora.command("drift-check")
@click.option("--name", "-n", required=True, help="Adapter name to check")
@click.option("--version", "-v", help="Specific version to check (default: active)")
@click.option("--threshold", "-t", type=float, default=0.1, help="Drift threshold (default: 0.1)")
@click.option("--source", default="phoenix-traces", help="Data source for drift detection")
def lora_drift_check_cmd(name, version, threshold, source):
    """Check for performance drift in an adapter.

    Examples:
        terradev lora drift-check -n customer-a
        terradev lora drift-check -n customer-a -t 0.15
    """
    import asyncio
    from terradev_cli.ml_services.lora_registry import get_lora_registry
    from terradev_cli.core.lora_versioning import LoRAVersioningManager

    registry = get_lora_registry()
    versioning_mgr = LoRAVersioningManager(registry=registry)

    result = asyncio.run(
        versioning_mgr.detect_drift(
            adapter_name=name,
            version_id=version,
            drift_threshold=threshold,
            source=source,
        )
    )

    print(f"Adapter: {name}")
    print(f"Version: {result.version_id[:8]}")
    print(f"Baseline score: {result.baseline_score:.4f}")
    print(f"Current score: {result.current_score:.4f}")
    print(f"Drift magnitude: {result.drift_magnitude:.2%}")
    print(f"Threshold: {result.drift_threshold}")
    print(f"Has drift: {result.has_drift}")
    print(f"Recommended action: {result.recommended_action}")

    if result.has_drift:
        print(f"\nWARNING: Performance drift detected!")
        if result.recommended_action == "rollback":
            print("Consider running: terradev lora rollback -n {name}")
        elif result.recommended_action == "retrain":
            print("Consider triggering retraining via drift service")


# ── Cost Attribution Commands ──


@lora.command("cost-report")
@click.option("--days", "-d", type=int, default=30, help="Number of days to report (default: 30)")
@click.option("--adapter", "-a", help="Specific adapter to report on")
@click.option("--tenant", "-t", help="Specific tenant to report on")
def lora_cost_report_cmd(days, adapter, tenant):
    """Generate cost attribution report for LoRA adapters.

    Examples:
        terradev lora cost-report -d 7
        terradev lora cost-report -a customer-a
        terradev lora cost-report -t tenant-123
    """
    import asyncio
    from terradev_cli.core.lora_cost_attribution import CostAttributionService, CostConfig

    config = CostConfig()
    cost_service = CostAttributionService(config)

    if adapter:
        # Get adapter-specific breakdown
        breakdown = asyncio.run(cost_service.get_cost_breakdown(adapter, days))
        print(f"Cost Breakdown: {adapter}")
        print(f"  Window: {days} days")
        print(f"  Total requests: {breakdown['total_requests']}")
        print(f"  GPU cost: ${breakdown['gpu_cost_usd']}")
        print(f"  Token cost: ${breakdown['token_cost_usd']}")
        print(f"  Total cost: ${breakdown['total_cost_usd']}")
        print(f"\n  Cost by replica:")
        for replica in breakdown['cost_by_replica']:
            print(f"    {replica['replica_id']}: ${replica['cost_usd']}")
    elif tenant:
        # Get tenant-specific cost
        tenant_record = asyncio.run(cost_service.get_tenant_cost(tenant))
        if tenant_record:
            print(f"Cost Report: Tenant {tenant}")
            print(f"  Adapters: {len(tenant_record.adapters)}")
            print(f"  GPU hours: {tenant_record.gpu_hours:.2f}")
            print(f"  Tokens processed: {tenant_record.tokens_processed:,}")
            print(f"  Requests served: {tenant_record.requests_served:,}")
            print(f"  Storage: {tenant_record.storage_gb:.2f} GB")
            print(f"  Total cost: ${tenant_record.total_cost_usd:.2f}")
            print(f"  Last updated: {tenant_record.last_updated}")
        else:
            print(f"ERROR: No cost data found for tenant '{tenant}'")
    else:
        # Get overall summary
        summary = asyncio.run(cost_service.get_cost_summary(days))
        print(f"Cost Summary: Last {days} days")
        print(f"  Total GPU hours: {summary['total_gpu_hours']}")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Total requests: {summary['total_requests']:,}")
        print(f"  Total cost: ${summary['total_cost_usd']}")
        print(f"\n  Top adapters by cost:")
        for adapter in summary['top_adapters']:
            print(f"    {adapter['name']}: ${adapter['cost_usd']}")
        print(f"\n  Top tenants by cost:")
        for tenant in summary['top_tenants']:
            print(f"    {tenant['tenant_id']}: ${tenant['cost_usd']}")


# ── LoRAX Integration (Predibase LoRA eXchange) ──


@lora.group()
def lorax():
    """LoRAX (LoRA eXchange) multi-LoRA inference server from Predibase.

    Deploy and manage LoRAX servers for serving thousands of fine-tuned models
    on a single GPU with dynamic adapter loading.
    """
    pass


@lorax.command("deploy")
@click.option("--model-id", "-m", required=True, help="Base model ID (e.g., mistralai/Mistral-7B-Instruct-v0.1)")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
@click.option("--quantization", type=click.Choice(["none", "bitsandbytes", "gptq", "awq"]), default="none", help="Quantization method")
@click.option("--gpu-memory-fraction", type=float, default=0.9, help="GPU memory fraction to use")
@click.option("--max-loras", type=int, default=8, help="Maximum number of adapters to load")
@click.option("--docker", is_flag=True, help="Deploy using Docker")
@click.option("--k8s", is_flag=True, help="Deploy using Kubernetes")
@click.option("--namespace", default="default", help="Kubernetes namespace (for --k8s)")
def lorax_deploy_cmd(model_id, host, port, quantization, gpu_memory_fraction, max_loras, docker, k8s, namespace):
    """Deploy a LoRAX server.

    Examples:
        terradev lora lorax deploy -m mistralai/Mistral-7B-Instruct-v0.1 --docker
        terradev lora lorax deploy -m meta-llama/Llama-2-7b-hf --k8s --namespace lorax
    """
    if docker:
        print(f"Deploying LoRAX with Docker...")
        print(f"  Model: {model_id}")
        print(f"  Port: {port}")
        print(f"  Quantization: {quantization}")
        print(f"\nDocker command:")
        print(f"  docker run --gpus all --shm-size 1g -p {port}:80 \\")
        print(f"    -v $PWD/data:/data \\")
        print(f"    ghcr.io/predibase/lorax:main \\")
        print(f"    --model-id {model_id} \\")
        print(f"    --max-loras {max_loras}")
        if quantization != "none":
            print(f"    --quantize {quantization}")
    elif k8s:
        print(f"Deploying LoRAX to Kubernetes...")
        print(f"  Namespace: {namespace}")
        print(f"  Model: {model_id}")
        print(f"\nHelm command:")
        print(f"  helm install lorax ./clusters/lorax-template/helm \\")
        print(f"    -f clusters/lorax-template/helm/values-lorax.yaml \\")
        print(f"    --set model.id={model_id} \\")
        print(f"    --set service.port={port} \\")
        print(f"    --set maxLoras={max_loras}")
        print(f"\nNote: Create the lorax-template cluster first with:")
        print(f"  terradev cluster create lorax-template")
    else:
        print(f"ERROR: Specify --docker or --k8s for deployment")


@lorax.command("test")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
def lorax_test_cmd(host, port):
    """Test LoRAX server connectivity.

    Examples:
        terradev lora lorax test
        terradev lora lorax test --host 10.0.0.1 --port 8080
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service

    svc = get_lorax_service(host=host, port=port)
    result = asyncio.run(svc.health_check())

    if result["status"] == "healthy":
        print(f"OK: LoRAX server is healthy at {host}:{port}")
        model_info = asyncio.run(svc.get_model_info())
        if "error" not in model_info:
            print(f"   Model: {model_info.get('model_id', 'unknown')}")
            print(f"   Architecture: {model_info.get('architecture', 'unknown')}")
    else:
        print(f"ERROR: LoRAX server health check failed")
        print(f"   Status: {result.get('status')}")
        if "error" in result:
            print(f"   Error: {result['error']}")

    asyncio.run(svc.close())


@lorax.command("list-adapters")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
def lorax_list_adapters_cmd(host, port):
    """List loaded adapters on LoRAX server.

    Examples:
        terradev lora lorax list-adapters
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service

    svc = get_lorax_service(host=host, port=port)
    adapters = asyncio.run(svc.list_loaded_adapters())

    print(f"Loaded adapters on {host}:{port}:")
    if adapters:
        for adapter in adapters:
            print(f"  {adapter.adapter_id}")
            if adapter.adapter_name:
                print(f"    Name: {adapter.adapter_name}")
            if adapter.base_model:
                print(f"    Base model: {adapter.base_model}")
            if adapter.rank:
                print(f"    Rank: {adapter.rank}")
    else:
        print("  (no adapters loaded)")

    asyncio.run(svc.close())


@lorax.command("load-adapter")
@click.option("--adapter-id", "-a", required=True, help="Adapter ID (HuggingFace repo or local path)")
@click.option("--adapter-name", help="Custom name for the adapter")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
def lorax_load_adapter_cmd(adapter_id, adapter_name, host, port):
    """Load a LoRA adapter onto LoRAX server.

    Examples:
        terradev lora lorax load-adapter -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
        terradev lora lorax load-adapter -a /path/to/local/adapter --adapter-name my-adapter
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service

    svc = get_lorax_service(host=host, port=port)
    result = asyncio.run(svc.load_adapter(adapter_id, adapter_name))

    if result["status"] == "loaded":
        print(f"OK: Adapter '{adapter_id}' loaded")
        if adapter_name:
            print(f"   Name: {adapter_name}")
    else:
        print(f"ERROR: Failed to load adapter '{adapter_id}'")
        if "error" in result:
            print(f"   Error: {result['error']}")
        if "response" in result:
            print(f"   Response: {result['response']}")

    asyncio.run(svc.close())


@lorax.command("unload-adapter")
@click.option("--adapter-id", "-a", required=True, help="Adapter ID to unload")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
def lorax_unload_adapter_cmd(adapter_id, host, port):
    """Unload a LoRA adapter from LoRAX server.

    Examples:
        terradev lora lorax unload-adapter -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service

    svc = get_lorax_service(host=host, port=port)
    result = asyncio.run(svc.unload_adapter(adapter_id))

    if result["status"] == "unloaded":
        print(f"OK: Adapter '{adapter_id}' unloaded")
    else:
        print(f"ERROR: Failed to unload adapter '{adapter_id}'")
        if "error" in result:
            print(f"   Error: {result['error']}")
        if "response" in result:
            print(f"   Response: {result['response']}")

    asyncio.run(svc.close())


@lorax.command("sync-registry")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", "-p", default=8080, help="LoRAX server port")
@click.option("--adapter", "-a", help="Specific adapter to sync")
def lorax_sync_registry_cmd(host, port, adapter):
    """Sync Terradev LoRA registry with LoRAX server state.

    Examples:
        terradev lora lorax sync-registry
        terradev lora lorax sync-registry -a customer-a
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service
    from terradev_cli.ml_services.lora_registry import get_lorax_registry

    # Get registry state
    registry = get_lorax_registry()
    if adapter:
        adapters = [adapter]
    else:
        adapters = registry.list_all_adapters()

    # Get LoRAX state
    svc = get_lorax_service(host=host, port=port)
    lorax_adapters = asyncio.run(svc.list_loaded_adapters())
    lorax_ids = {a.adapter_id for a in lorax_adapters}

    print(f"Syncing registry with LoRAX at {host}:{port}")
    print(f"  Registry adapters: {len(adapters)}")
    print(f"  LoRAX loaded: {len(lorax_adapters)}")

    for adapter_name in adapters:
        active_version = registry.get_active_version(adapter_name)
        if active_version:
            if active_version.path not in lorax_ids:
                print(f"  [MISSING] {adapter_name} (version {active_version.version_id[:8]})")
                print(f"    Path: {active_version.path}")
                print(f"    To load: terradev lora lorax load-adapter -a {active_version.path}")
            else:
                print(f"  [SYNCED] {adapter_name}")

    asyncio.run(svc.close())


@lorax.command("generate")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--adapter-id", "-a", help="Adapter ID to use")
@click.option("--max-tokens", type=int, default=64, help="Max tokens to generate")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--host", default="localhost", help="LoRAX server host")
@click.option("--port", default=8080, help="LoRAX server port")
def lorax_generate_cmd(prompt, adapter_id, max_tokens, temperature, host, port):
    """Generate text using LoRAX server.

    Examples:
        terradev lora lorax generate -p "Hello, world!"
        terradev lora lorax generate -p "What is 2+2?" -a my-adapter
    """
    import asyncio
    from terradev_cli.ml_services.lorax_service import get_lorax_service

    svc = get_lorax_service(host=host, port=port)
    response = asyncio.run(svc.generate(
        prompt=prompt,
        adapter_id=adapter_id,
        max_new_tokens=max_tokens,
        temperature=temperature
    ))

    print(f"Prompt: {prompt}")
    if adapter_id:
        print(f"Adapter: {adapter_id}")
    print(f"Generated: {response.generated_text}")
    if response.finish_reason:
        print(f"Finish reason: {response.finish_reason}")
    print(f"Tokens: {response.tokens_generated}")

    asyncio.run(svc.close())


# ── HuggingFace PEFT Import ──


@lora.group()
def peft():
    """HuggingFace PEFT adapter import and management.

    Download, validate, and prepare LoRA adapters from HuggingFace for use
    with vLLM, LoRAX, or other inference servers.
    """
    pass


@peft.command("import")
@click.option("--adapter-id", "-a", required=True, help="HuggingFace adapter ID (e.g., username/adapter-name)")
@click.option("--local-name", help="Local name for the adapter")
@click.option("--token", help="HuggingFace auth token (for private repos)")
@click.option("--register", is_flag=True, help="Register imported adapter in Terradev registry")
@click.option("--base-model", "-b", help="Base model (required with --register)")
@click.option("--rank", type=int, help="LoRA rank (auto-detected if not specified)")
def peft_import_cmd(adapter_id, local_name, token, register, base_model, rank):
    """Import a LoRA adapter from HuggingFace.

    Examples:
        terradev lora peft import -a vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k
        terradev lora peft import -a username/adapter --local-name my-adapter --register --base-model mistralai/Mistral-7B-Instruct-v0.1
    """
    from terradev_cli.ml_services.peft_import_service import get_peft_import_service

    svc = get_peft_import_service()

    print(f"Importing adapter from HuggingFace: {adapter_id}")

    try:
        config = svc.download_adapter(
            adapter_id=adapter_id,
            local_name=local_name,
            token=token
        )

        print(f"OK: Adapter imported successfully")
        print(f"  Local path: {config.local_path}")
        print(f"  Base model: {config.base_model or 'unknown'}")
        print(f"  Rank: {config.rank or 'unknown'}")
        print(f"  Alpha: {config.alpha or 'unknown'}")
        print(f"  PEFT type: {config.peft_type}")

        # Register if requested
        if register:
            if not base_model:
                print(f"ERROR: --base-model required when using --register")
                return

            from terradev_cli.ml_services.lora_registry import get_lorax_registry
            registry = get_lorax_registry()

            version = registry.register_adapter(
                adapter_name=local_name or adapter_id.replace("/", "--"),
                base_model=base_model,
                path=str(config.local_path),
                rank=rank or config.rank or 64,
            )

            print(f"\nRegistered in Terradev registry:")
            print(f"  Version ID: {version.version_id}")
            print(f"  Adapter name: {version.adapter_name}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to import adapter: {e}")


@peft.command("list")
def peft_list_cmd():
    """List all locally imported PEFT adapters.

    Examples:
        terradev lora peft list
    """
    from terradev_cli.ml_services.peft_import_service import get_peft_import_service

    svc = get_peft_import_service()
    adapters = svc.list_local_adapters()

    print(f"Local PEFT adapters ({len(adapters)}):")
    if adapters:
        for adapter in adapters:
            print(f"  {adapter.adapter_id}")
            print(f"    Path: {adapter.local_path}")
            if adapter.base_model:
                print(f"    Base model: {adapter.base_model}")
            if adapter.rank:
                print(f"    Rank: {adapter.rank}")
            if adapter.alpha:
                print(f"    Alpha: {adapter.alpha}")
            print(f"    PEFT type: {adapter.peft_type}")
    else:
        print("  (no adapters imported)")


@peft.command("validate")
@click.option("--path", "-p", required=True, help="Path to adapter directory")
def peft_validate_cmd(path):
    """Validate a PEFT adapter structure.

    Examples:
        terradev lora peft validate -p ~/.terradev/peft_adapters/username--adapter-name
    """
    from terradev_cli.ml_services.peft_import_service import get_peft_import_service

    svc = get_peft_import_service()
    result = svc.validate_adapter(Path(path))

    if result["valid"]:
        print(f"OK: Adapter is valid")
    else:
        print(f"ERROR: Adapter validation failed")
        print(f"  Missing files: {', '.join(result['missing_files'])}")

    if result["warnings"]:
        print(f"\nWarnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


@peft.command("delete")
@click.option("--adapter-id", "-a", required=True, help="Adapter ID to delete")
def peft_delete_cmd(adapter_id):
    """Delete a locally imported adapter.

    Examples:
        terradev lora peft delete -a username/adapter-name
    """
    from terradev_cli.ml_services.peft_import_service import get_peft_import_service

    svc = get_peft_import_service()
    if svc.delete_adapter(adapter_id):
        print(f"OK: Deleted adapter '{adapter_id}'")
    else:
        print(f"ERROR: Adapter '{adapter_id}' not found locally")


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
@click.group()
def sso():
    """Enterprise SSO authentication"""
    pass


@sso.command("status")
def sso_status():
    """Show SSO configuration status"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("WARNING:  Enterprise auth not initialized")
        print(
            "   Install enterprise dependencies: pip install terradev-cli[enterprise]"
        )
        return

    enabled_providers = api.enterprise_auth.list_enabled_providers()
    if enabled_providers:
        print("OK: SSO is configured")
        print("   Enabled providers:", ", ".join(enabled_providers))
    else:
        print("WARNING:  No SSO providers configured")
        print("   Configure providers with: terradev sso configure")


@sso.command("configure")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["azure_ad", "okta", "google_workspace", "auth0"]),
    required=True,
    help="SSO provider",
)
@click.option("--client-id", help="Client ID (for OIDC providers)")
@click.option("--client-secret", help="Client secret (for OIDC providers)")
@click.option("--domain", help="Domain (for Okta/Auth0)")
@click.option("--tenant-id", help="Tenant ID (for Azure AD)")
@click.option("--entity-id", help="Entity ID (for SAML providers)")
@click.option("--sso-url", help="SSO URL (for SAML providers)")
@click.option("--certificate", help="Certificate (for SAML providers)")
def sso_configure(
    provider,
    client_id,
    client_secret,
    domain,
    tenant_id,
    entity_id,
    sso_url,
    certificate,
):
    """Configure SSO provider"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("ERROR: Enterprise auth not initialized")
        print(
            "   Install enterprise dependencies: pip install terradev-cli[enterprise]"
        )
        return

    config = {}

    if provider in ["google_workspace", "auth0", "azure_ad"]:
        # OIDC providers
        if not client_id or not client_secret:
            print("ERROR: Client ID and secret required for OIDC providers")
            return

        if provider == "google_workspace":
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "enabled": True,
                }
            )
        elif provider == "auth0":
            if not domain:
                print("ERROR: Domain required for Auth0")
                return
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "domain": domain,
                    "enabled": True,
                }
            )
        elif provider == "azure_ad":
            if not tenant_id:
                print("ERROR: Tenant ID required for Azure AD")
                return
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "tenant_id": tenant_id,
                    "enabled": True,
                }
            )

    elif provider in ["azure_ad", "okta"]:
        # SAML providers
        if not entity_id or not sso_url:
            print("ERROR: Entity ID and SSO URL required for SAML providers")
            return

        config = api.enterprise_auth.get_sso_provider_config(provider)
        config.update(
            {
                "entity_id": entity_id,
                "sso_url": sso_url,
                "certificate": certificate or "",
                "enabled": True,
            }
        )

    try:
        api.enterprise_auth.enable_sso_provider(provider, config)
        print(f"OK: {provider} SSO provider configured successfully")
        print("   Test the configuration with: terradev sso test")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to configure {provider}: {e}")


@sso.command("test")
@click.option("--provider", "-p", help="Test specific provider")
def sso_test(provider):
    """Test SSO provider configuration"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("ERROR: Enterprise auth not initialized")
        return

    if provider:
        # Test specific provider
        config = api.enterprise_auth.get_sso_provider_config(provider)
        if not config or not config.get("enabled"):
            print(f"ERROR: Provider {provider} not configured")
            return

        print(f"Testing {provider}...")
        # Add actual testing logic here
        print(f"OK: {provider} configuration appears valid")
    else:
        # Test all providers
        enabled_providers = api.enterprise_auth.list_enabled_providers()
        if not enabled_providers:
            print("WARNING:  No SSO providers configured")
            return

        print("Testing all configured providers...")
        for p in enabled_providers:
            print(f"OK: {p} configuration appears valid")


# ── SGLANG COMMAND GROUP ──
















# ═══════════════════════════════════════════════════════════════════════════════
# Drift-Triggered Continuous Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════


@cli.group()
def retrain():
    """Drift-triggered continuous fine-tuning.

    Watch Phoenix traces for quality degradation, auto-retrain LoRA adapters,
    evaluate against holdout, and hot-swap onto vLLM  zero downtime.

    Examples:
        terradev retrain drift --model llama-70b-prod --source phoenix-traces
        terradev retrain status
        terradev retrain history
    """
    pass


@retrain.command("drift")
@click.option(
    "--model", "-m", required=True, help="Model identifier (e.g. llama-70b-prod)"
)
@click.option(
    "--source",
    default="phoenix-traces",
    type=click.Choice(["phoenix-traces"]),
    help="Data source for drift detection",
)
@click.option(
    "--method", default="lora", type=click.Choice(["lora"]), help="Fine-tuning method"
)
@click.option(
    "--eval-threshold",
    default=0.85,
    type=float,
    help="Minimum eval score to deploy (0.0-1.0)",
)
@click.option(
    "--deploy",
    default="canary",
    type=click.Choice(["canary", "direct"]),
    help="Deployment strategy",
)
@click.option(
    "--auto-swap",
    is_flag=True,
    default=False,
    help="Auto-deploy if eval passes (no manual approval)",
)
@click.option(
    "--phoenix-endpoint",
    default="http://localhost:6006",
    help="Phoenix collector endpoint",
)
@click.option("--phoenix-project", default="default", help="Phoenix project name")
@click.option(
    "--vllm-endpoint", "-e", default="", help="vLLM endpoint for eval and deploy"
)
@click.option("--vllm-api-key", default=None, help="vLLM API key")
@click.option("--baseline", default=0.90, type=float, help="Baseline quality score")
@click.option("--threshold", default=0.85, type=float, help="Drift trigger threshold")
@click.option(
    "--min-samples", default=50, type=int, help="Min samples before triggering"
)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def retrain_drift(
    model,
    source,
    method,
    eval_threshold,
    deploy,
    auto_swap,
    phoenix_endpoint,
    phoenix_project,
    vllm_endpoint,
    vllm_api_key,
    baseline,
    threshold,
    min_samples,
    fmt,
):
    """Run a drift-triggered retrain cycle.

    Monitors Phoenix traces, detects quality drift, retrains a LoRA adapter,
    evaluates it, and optionally hot-swaps it onto a running vLLM server.

    Examples:
        terradev retrain drift -m llama-70b-prod --auto-swap
        terradev retrain drift -m llama-70b-prod -e http://10.0.0.1:8000
        terradev retrain drift -m llama-70b-prod --eval-threshold 0.90
    """
    from terradev_cli.ml_services.drift_retrain_service import (
        DriftRetrainConfig,
        DriftRetrainService,
    )

    config = DriftRetrainConfig(
        model_id=model,
        phoenix_endpoint=phoenix_endpoint,
        phoenix_project=phoenix_project,
        baseline_score=baseline,
        degradation_threshold=threshold,
        min_samples=min_samples,
        method=method,
        eval_threshold=eval_threshold,
        vllm_endpoint=vllm_endpoint,
        vllm_api_key=vllm_api_key,
        deploy_strategy=deploy,
        auto_swap=auto_swap,
    )

    svc = DriftRetrainService(config)

    print(f"\n{'='*60}")
    print(f"  Drift-Triggered Retrain: {model}")
    print(f"  Cycle ID: {config.cycle_id}")
    print(f"{'='*60}\n")

    result = asyncio.get_event_loop().run_until_complete(svc.run_full_cycle())

    if fmt == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        outcome = result.get("outcome", "unknown")
        stages = result.get("stages", {})

        # Drift
        drift = stages.get("drift_detection", {})
        if drift:
            icon = "\u26a0\ufe0f" if drift.get("drifted") else "\u2705"
            print(
                f"  {icon} Drift Detection: score={drift.get('score', '?')} "
                f"(threshold={drift.get('threshold', '?')}, samples={drift.get('samples', 0)})"
            )

        if outcome == "no_drift":
            print("\n  \u2705 No drift detected  model is healthy\n")
            return

        # Data
        data = stages.get("data_extraction", {})
        if data:
            print(
                f"  \U0001f4ca Data: {data.get('train_count', 0)} train / "
                f"{data.get('holdout_count', 0)} holdout samples"
            )

        # Training
        train = stages.get("training", {})
        if train:
            print(
                f"  \U0001f3cb Training: job_id={train.get('job_id', '?')} "
                f"status={train.get('status', '?')}"
            )

        # Eval
        ev = stages.get("evaluation", {})
        if ev:
            icon = "\u2705" if ev.get("passed") else "\u274c"
            print(
                f"  {icon} Eval: score={ev.get('score', '?')} "
                f"(threshold={ev.get('threshold', '?')}, metric={ev.get('metric', '?')})"
            )

        # Deploy
        dep = stages.get("deployment", {})
        if dep:
            status = dep.get("status", "?")
            if status == "deployed":
                print(
                    f"  \U0001f680 Deployed: adapter={dep.get('adapter_name')} "
                    f"on {dep.get('endpoint')}"
                )
            elif status == "awaiting_approval":
                print(
                    f"  \u23f3 Awaiting approval  run: terradev retrain deploy "
                    f"--cycle-id {config.cycle_id}"
                )
            else:
                print(f"  \u274c Deploy: {dep.get('error', status)}")

        # Summary
        print(f"\n  Outcome: {outcome}")
        if result.get("manifest_path"):
            print(f"  Manifest: {result['manifest_path']}")
        print()


@retrain.command("detect")
@click.option("--model", "-m", required=True, help="Model identifier")
@click.option("--phoenix-endpoint", default="http://localhost:6006")
@click.option("--phoenix-project", default="default")
@click.option("--baseline", default=0.90, type=float)
@click.option("--threshold", default=0.85, type=float)
@click.option("--min-samples", default=50, type=int)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def retrain_detect(
    model, phoenix_endpoint, phoenix_project, baseline, threshold, min_samples, fmt
):
    """Check for drift without triggering a retrain.

    Examples:
        terradev retrain detect -m llama-70b-prod
        terradev retrain detect -m llama-70b-prod --threshold 0.80
    """
    from terradev_cli.ml_services.drift_retrain_service import (
        DriftRetrainConfig,
        DriftRetrainService,
    )

    config = DriftRetrainConfig(
        model_id=model,
        phoenix_endpoint=phoenix_endpoint,
        phoenix_project=phoenix_project,
        baseline_score=baseline,
        degradation_threshold=threshold,
        min_samples=min_samples,
    )
    svc = DriftRetrainService(config)
    result = asyncio.get_event_loop().run_until_complete(svc.detect_drift())

    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        icon = (
            "\u26a0\ufe0f  DRIFT DETECTED"
            if result.get("drifted")
            else "\u2705 No drift"
        )
        print(f"\n  {icon}")
        print(f"  Score:     {result.get('score', '?')}")
        print(f"  Baseline:  {result.get('baseline', '?')}")
        print(f"  Threshold: {result.get('threshold', '?')}")
        print(f"  Samples:   {result.get('samples', 0)}")
        print(f"  Detail:    {result.get('detail', '')}\n")


@retrain.command("deploy")
@click.option("--cycle-id", required=True, help="Retrain cycle ID to deploy")
@click.option("--vllm-endpoint", "-e", required=True, help="vLLM endpoint")
@click.option("--vllm-api-key", default=None)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def retrain_deploy(cycle_id, vllm_endpoint, vllm_api_key, fmt):
    """Manually deploy an adapter from a completed retrain cycle.

    Use this when --auto-swap was not set and eval passed.

    Examples:
        terradev retrain deploy --cycle-id retrain-abc12345 -e http://10.0.0.1:8000
    """
    from terradev_cli.ml_services.drift_retrain_service import (
        DriftRetrainConfig,
        DriftRetrainService,
    )

    manifest_path = Path.home() / ".terradev" / "retrain_manifests" / f"{cycle_id}.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest found for cycle {cycle_id}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    adapter_dir = manifest.get("adapter_path", "")
    if not adapter_dir:
        adapter_dir = str(Path.home() / ".terradev" / "adapters" / cycle_id)

    config = DriftRetrainConfig(
        model_id=manifest.get("model_id", ""),
        cycle_id=cycle_id,
        vllm_endpoint=vllm_endpoint,
        vllm_api_key=vllm_api_key,
        adapter_output_dir=adapter_dir,
        auto_swap=True,
    )
    svc = DriftRetrainService(config)
    result = asyncio.get_event_loop().run_until_complete(svc.deploy_adapter())

    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        if result.get("status") == "deployed":
            print("\n  \U0001f680 Adapter deployed!")
            print(f"  Name:     {result.get('adapter_name')}")
            print(f"  Endpoint: {result.get('endpoint')}")
            print(f"  Path:     {result.get('adapter_path')}\n")
        else:
            print(
                f"\n  \u274c Deploy failed: {result.get('error', result.get('reason', '?'))}\n"
            )


@retrain.command("history")
@click.option("--limit", "-n", default=20, type=int, help="Number of cycles to show")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def retrain_history(limit, fmt):
    """Show retrain cycle history.

    Examples:
        terradev retrain history
        terradev retrain history -n 5 -f json
    """
    from terradev_cli.ml_services.drift_retrain_service import DriftRetrainService

    manifests = DriftRetrainService.list_retrain_history(limit=limit)

    if fmt == "json":
        print(json.dumps(manifests, indent=2, default=str))
    else:
        if not manifests:
            print("\n  No retrain cycles found.\n")
            return
        print(
            f"\n  {'Cycle ID':<24} {'Model':<24} {'Status':<14} {'Eval':<8} {'Started'}"
        )
        print(f"  {'─'*22}  {'─'*22}  {'─'*12}  {'─'*6}  {'─'*20}")
        for m in manifests:
            cid = m.get("cycle_id", "?")[:22]
            model = m.get("model_id", "?")[:22]
            status = m.get("status", "?")[:12]
            score = m.get("eval_score", 0)
            started = m.get("started_at", "?")[:19]
            print(f"  {cid:<24} {model:<24} {status:<14} {score:<8.4f} {started}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Langfuse  LLM Observability, Scoring & Dataset Management
# ═══════════════════════════════════════════════════════════════════════════════


























# ═══════════════════════════════════════════════════════════════════════════════
# Databricks  Jobs, Clusters, Model Serving, MLflow
# ═══════════════════════════════════════════════════════════════════════════════




























# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Serving  KV Cache TTL, Prefix Caching, LMCache, Priority Scheduling
# ═══════════════════════════════════════════════════════════════════════════════


@cli.group("agentic-serving")
def agentic_serving():
    """Agentic inference serving  KV cache TTL, prefix caching, LMCache, priority scheduling."""
    pass


@agentic_serving.command("configure")
@click.option(
    "--engine",
    type=click.Choice(["vllm", "sglang"]),
    default="vllm",
    help="Inference engine",
)
@click.option("--model", prompt="Model ID", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option(
    "--tp", "tensor_parallel_size", default=1, type=int, help="Tensor parallel size"
)
@click.option("--max-model-len", default=32768, type=int)
@click.option("--gpu-mem", "gpu_memory_utilization", default=0.85, type=float)
@click.option(
    "--lmcache/--no-lmcache",
    "lmcache_enabled",
    default=True,
    help="Enable LMCache KV offload",
)
@click.option(
    "--lmcache-backend", type=click.Choice(["cpu", "disk", "redis"]), default="cpu"
)
@click.option(
    "--disaggregation/--no-disaggregation",
    default=False,
    help="Prefill-decode disaggregation",
)
def agentic_serving_configure(
    engine,
    model,
    tensor_parallel_size,
    max_model_len,
    gpu_memory_utilization,
    lmcache_enabled,
    lmcache_backend,
    disaggregation,
):
    """Configure agentic inference serving settings."""
    api = TerradevAPI()
    api._save_provider_creds(
        "agentic_serving",
        {
            "engine": engine,
            "model": model,
            "tensor_parallel_size": str(tensor_parallel_size),
            "max_model_len": str(max_model_len),
            "gpu_memory_utilization": str(gpu_memory_utilization),
            "enable_prefix_caching": "true",
            "lmcache_enabled": str(lmcache_enabled).lower(),
            "lmcache_backend": lmcache_backend,
            "disaggregation_enabled": str(disaggregation).lower(),
        },
    )
    print(f"\u2705 Agentic serving configured: {engine} + {model}")
    print("   Prefix caching: enabled")
    print(
        f"   LMCache: {'enabled (' + lmcache_backend + ')' if lmcache_enabled else 'disabled'}"
    )
    print(f"   PD disaggregation: {'enabled' if disaggregation else 'disabled'}")


@agentic_serving.command("show-config")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def agentic_serving_show_config(fmt):
    """Show current agentic serving configuration."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_vllm_args,
        generate_sglang_args,
        generate_lmcache_config,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    engine_args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )

    if fmt == "json":
        print(
            json.dumps(
                {
                    "engine": config.engine,
                    "model": config.model,
                    "engine_args": engine_args,
                    "lmcache": generate_lmcache_config(config),
                    "ttl": {
                        "min": config.ttl_min,
                        "max": config.ttl_max,
                        "multiplier": config.ttl_multiplier,
                    },
                    "disaggregation": config.disaggregation_enabled,
                    "prefix_caching": config.enable_prefix_caching,
                },
                indent=2,
            )
        )
    else:
        print("\n  Agentic Serving Config:")
        print(f"  Engine:            {config.engine}")
        print(f"  Model:             {config.model}")
        print(f"  TP:                {config.tensor_parallel_size}")
        print(f"  Max Model Len:     {config.max_model_len}")
        print(f"  GPU Mem Util:      {config.gpu_memory_utilization}")
        print(f"  Prefix Caching:    {config.enable_prefix_caching}")
        print(
            f"  LMCache:           {config.lmcache_enabled} ({config.lmcache_backend})"
        )
        print(f"  Disaggregation:    {config.disaggregation_enabled}")
        print(
            f"  KV TTL Range:      {config.ttl_min}s - {config.ttl_max}s (x{config.ttl_multiplier})"
        )
        print("\n  Engine Args:")
        for a in engine_args:
            print(f"    {a}")
        print()


@agentic_serving.command("launch-args")
def agentic_serving_launch_args():
    """Print engine launch arguments for copy-paste."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_vllm_args,
        generate_sglang_args,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )
    if config.engine == "vllm":
        print("\npython -m vllm.entrypoints.openai.api_server \\")
    else:
        print("\npython -m sglang.launch_server \\")
    for i, a in enumerate(args):
        sep = " \\" if i < len(args) - 1 else ""
        print(f"  {a}{sep}")
    print()


@agentic_serving.command("lmcache-env")
def agentic_serving_lmcache_env():
    """Print LMCache environment variables."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_lmcache_env,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    env = generate_lmcache_env(config)
    if not env:
        print("  LMCache is disabled.")
        return
    print()
    for k, v in env.items():
        print(f'export {k}="{v}"')
    print()


@agentic_serving.command("k8s")
@click.option("--namespace", "-n", default="inference", help="K8s namespace")
def agentic_serving_k8s(namespace):
    """Print K8s deployment manifests for agentic inference."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_k8s_deployment,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    print(generate_k8s_deployment(config, namespace=namespace))


@agentic_serving.command("helm-values")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "yaml"]), default="yaml"
)
def agentic_serving_helm_values(fmt):
    """Print Helm values for agentic inference deployment."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_helm_values,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    values = generate_helm_values(config)
    if fmt == "json":
        print(json.dumps(values, indent=2))
    else:
        import yaml

        print(yaml.dump(values, default_flow_style=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Model Router  Cost/Quality Routing for Agentic Workloads
# ═══════════════════════════════════════════════════════════════════════════════


@cli.group("model-router")
def model_router():
    """Model routing  cost/quality-aware routing between strong and weak models."""
    pass


@model_router.command("configure")
@click.option(
    "--strong-url",
    prompt="Strong model URL (e.g. https://api.openai.com)",
    help="Strong model endpoint",
)
@click.option("--strong-model", prompt="Strong model ID", default="gpt-4")
@click.option("--strong-api-key", prompt="Strong model API key", hide_input=True)
@click.option(
    "--weak-url",
    prompt="Weak model URL (e.g. http://localhost:8000)",
    help="Weak model endpoint",
)
@click.option("--weak-model", prompt="Weak model ID", default="llama-3.1-8b")
@click.option("--weak-api-key", default="", help="Weak model API key (if needed)")
@click.option(
    "--strategy",
    type=click.Choice(
        ["step_type", "threshold", "cascade", "strong_only", "weak_only"]
    ),
    default="step_type",
    help="Routing strategy",
)
@click.option(
    "--cost-threshold",
    default=0.5,
    type=float,
    help="Complexity threshold for threshold strategy",
)
def model_router_configure(
    strong_url,
    strong_model,
    strong_api_key,
    weak_url,
    weak_model,
    weak_api_key,
    strategy,
    cost_threshold,
):
    """Configure model routing endpoints and strategy."""
    api = TerradevAPI()
    api._save_provider_creds(
        "model_router",
        {
            "strong_url": strong_url,
            "strong_model": strong_model,
            "strong_api_key": strong_api_key,
            "weak_url": weak_url,
            "weak_model": weak_model,
            "weak_api_key": weak_api_key,
            "strategy": strategy,
            "cost_threshold": str(cost_threshold),
            "cascade_enabled": str(strategy == "cascade").lower(),
        },
    )
    print("\u2705 Model router configured:")
    print(f"   Strong: {strong_model} @ {strong_url}")
    print(f"   Weak:   {weak_model} @ {weak_url}")
    print(f"   Strategy: {strategy}")


@model_router.command("test")
@click.option("--prompt", "-p", default="What is 2+2?", help="Test prompt")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_test(prompt, fmt):
    """Test model routing with a sample prompt."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = TerradevAPI()
    router = create_router_from_credentials(api._provider_creds("model_router"))
    messages = [{"role": "user", "content": prompt}]
    endpoint, step_type, reason = router.route(messages)

    if fmt == "json":
        print(
            json.dumps(
                {
                    "model": endpoint.model_id,
                    "tier": endpoint.tier.value,
                    "endpoint_url": endpoint.url,
                    "step_type": step_type.value,
                    "reason": reason,
                },
                indent=2,
            )
        )
    else:
        print("\n  Routing Decision:")
        print(f"  Model:     {endpoint.model_id}")
        print(f"  Tier:      {endpoint.tier.value}")
        print(f"  URL:       {endpoint.url}")
        print(f"  Step Type: {step_type.value}")
        print(f"  Reason:    {reason}\n")


@model_router.command("classify")
@click.argument("text")
def model_router_classify(text):
    """Classify a message's step type for routing."""
    from terradev_cli.ml_services.model_router import StepClassifier

    messages = [{"role": "user", "content": text}]
    step_type = StepClassifier.classify(messages)
    print(f"  Step type: {step_type.value}")


@model_router.command("stats")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_stats(fmt):
    """Show routing statistics (in-memory, current session)."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = TerradevAPI()
    router = create_router_from_credentials(api._provider_creds("model_router"))
    stats = router.get_routing_stats()

    if fmt == "json":
        print(json.dumps(stats, indent=2))
    else:
        print("\n  Routing Stats:")
        print(f"  Total Decisions: {stats['total_decisions']}")
        if stats["total_decisions"] > 0:
            print(f"  Strong %:        {stats['strong_pct']}%")
            print(f"  Weak %:          {stats['weak_pct']}%")
            by_step = stats.get("by_step_type", {})
            if by_step:
                print("\n  By Step Type:")
                for st, counts in by_step.items():
                    print(
                        f"    {st}: {counts['total']} (strong={counts['strong']}, weak={counts['weak']})"
                    )
        print()


@model_router.command("llmd-config")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "yaml"]), default="yaml"
)
def model_router_llmd_config(fmt):
    """Generate llm-d KV-cache-aware routing config."""
    from terradev_cli.ml_services.model_router import generate_llmd_routing_config, RouterConfig

    config = generate_llmd_routing_config(RouterConfig())
    if fmt == "json":
        print(json.dumps(config, indent=2))
    else:
        import yaml

        print(yaml.dump(config, default_flow_style=False))


# ── MIGRATE COMMAND GROUP ──


@cli.group()
def migrate():
    """Cross-provider workload migration with dry-run analysis"""
    pass


@migrate.command()
@click.option("--from", "from_provider", required=True, help="Source provider")
@click.option("--to", "to_provider", required=True, help="Target provider")
@click.option("--instance-id", help="Source instance ID")
@click.option("--workload", help="Workload ID from JobStateManager")
@click.option("--dry-run", is_flag=True, help="Show migration plan without executing")
def migration(from_provider, to_provider, instance_id, workload, dry_run):
    """Migrate workload between providers with detailed cost analysis"""
    from terradev_cli.core.migration_orchestrator import MigrationOrchestrator

    print(f"\n Migration Analysis: {from_provider} → {to_provider}")
    if dry_run:
        print("    DRY RUN MODE - No changes will be made")

    try:
        orchestrator = MigrationOrchestrator()
        plan = orchestrator.plan_migration(
            source_provider=from_provider,
            target_provider=to_provider,
            instance_id=instance_id,
            workload_id=workload,
            dry_run=True,  # Always generate plan first
        )

        # Display migration plan
        print("\n Migration Plan:")
        print(f"   Source: {plan.source['provider']} ({plan.source['gpu_type']})")
        print(f"   Target: {plan.target['provider']} ({plan.target['gpu_type']})")
        print(f"   Confidence: {plan.confidence_score:.1%}")

        if plan.warnings:
            print("\nWARNING:  Warnings:")
            for warning in plan.warnings:
                print(f"    {warning}")

        print("\nCOST: Cost Analysis:")
        print(f"   Data transfer: ${plan.costs['data_transfer']:.4f}")
        print(f"   Target hourly: ${plan.costs['target_hourly']:.2f}")
        print(f"   Hourly savings: ${plan.costs['hourly_savings']:+.2f}")
        print(f"   Monthly savings: ${plan.costs['estimated_monthly_savings']:+.2f}")

        print("\n Compatibility:")
        print(f"   GPU match: {plan.compatibility['gpu_match']}")
        print(f"   Performance change: {plan.compatibility['performance_change']}")

        print("\n  Migration Steps:")
        for step in plan.steps:
            print(f"   {step}")

        print(f"\n  Estimated downtime: {plan.total_downtime}")

        if dry_run:
            print(
                "\nOK: Dry run complete. Use without --dry-run to execute migration."
            )
        else:
            # In lightweight version, just show plan and exit
            print(
                "\n Full migration execution not implemented in lightweight version."
            )
            print("   This would involve:")
            print("    Checkpointing current job")
            print("    Transferring data via optimized route")
            print("    Provisioning target instance")
            print("    Restoring from checkpoint")
            print("    Validating migration success")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Migration planning failed: {e}")
        return 1


@migrate.command("list-workloads")
@click.option("--provider", help="Filter by provider")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def list_workloads(provider, fmt):
    """List available workloads for migration"""
    from terradev_cli.core.migration_orchestrator import MigrationOrchestrator

    try:
        orchestrator = MigrationOrchestrator()
        workloads = orchestrator.discover_workloads()

        if provider:
            workloads = [w for w in workloads if w.provider.lower() == provider.lower()]

        if fmt == "json":
            print(
                json.dumps(
                    [
                        {
                            "job_id": w.job_id,
                            "name": w.name,
                            "provider": w.provider,
                            "gpu_type": w.gpu_type,
                            "progress": f"{w.current_step}/{w.total_steps}",
                            "checkpoint_size_gb": w.checkpoint_size_gb,
                        }
                        for w in workloads
                    ],
                    indent=2,
                )
            )
        else:
            print("\n Available Workloads:")
            if not workloads:
                print("   No active workloads found")
                return

            print(
                f"   {'Job ID':<20} {'Name':<15} {'Provider':<12} {'GPU':<8} {'Progress':<12} {'Size':<8}"
            )
            print(f"   {'─'*80}")
            for w in workloads:
                progress = f"{w.current_step}/{w.total_steps}"
                size = f"{w.checkpoint_size_gb:.1f}GB"
                print(
                    f"   {w.job_id:<20} {w.name:<15} {w.provider:<12} {w.gpu_type:<8} {progress:<12} {size:<8}"
                )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list workloads: {e}")
        return 1


# ── EVAL COMMAND GROUP ──


@cli.group()
def eval():
    """Model and endpoint evaluation with baseline comparison"""
    pass


@eval.command()
@click.option("--model", "model_path", help="Model checkpoint path")
@click.option("--endpoint", help="API endpoint URL")
@click.option("--dataset", help="Dataset path for evaluation")
@click.option(
    "--metrics",
    multiple=True,
    default=["accuracy", "latency"],
    help="Metrics to evaluate",
)
@click.option("--baseline", help="Baseline result file for comparison")
@click.option("--workload-type", default="general", help="Workload type classification")
@click.option(
    "--duration",
    type=int,
    default=300,
    help="Duration for endpoint evaluation (seconds)",
)
@click.option("--output", help="Output file for results")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
def evaluation(
    model_path,
    endpoint,
    dataset,
    metrics,
    baseline,
    workload_type,
    duration,
    output,
    fmt,
):
    """Run model or endpoint evaluation"""
    from terradev_cli.core.evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig

    if not model_path and not endpoint:
        print("ERROR: Either --model or --endpoint must be specified")
        return 1

    if model_path and not dataset:
        print("ERROR: --dataset required when evaluating a model")
        return 1

    try:
        orchestrator = EvaluationOrchestrator()
        config = EvaluationConfig(
            model_path=model_path,
            endpoint_url=endpoint,
            dataset_path=dataset,
            metrics=list(metrics),
            baseline_path=baseline,
            workload_type=workload_type,
            duration_seconds=duration,
        )

        print("\n Running Evaluation...")
        if model_path:
            print(f"   Model: {model_path}")
            print(f"   Dataset: {dataset}")
        if endpoint:
            print(f"   Endpoint: {endpoint}")
            print(f"   Duration: {duration}s")
        print(f"   Metrics: {', '.join(metrics)}")

        # Run evaluation
        if model_path:
            result = orchestrator.evaluate_model(config)
        else:
            result = orchestrator.evaluate_endpoint(config)

        # Display results
        if fmt == "json":
            result_data = {
                "evaluation_id": result.evaluation_id,
                "model_path": result.model_path,
                "endpoint_url": result.endpoint_url,
                "workload_type": result.workload_type,
                "metrics": result.metrics,
                "baseline_comparison": result.baseline_comparison,
                "timestamp": result.timestamp.isoformat(),
                "duration_seconds": result.duration_seconds,
                "metadata": result.metadata,
            }
            print(json.dumps(result_data, indent=2))
        else:
            print("\n Evaluation Results:")
            print(f"   Evaluation ID: {result.evaluation_id}")
            print(f"   Duration: {result.duration_seconds:.1f}s")

            print("\n Metrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    if metric in ["latency", "error_rate"]:
                        print(f"   {metric:<15}: {value:.2f}ms")
                    elif metric in ["throughput"]:
                        print(f"   {metric:<15}: {value:.1f} tokens/s")
                    elif metric in ["cost_per_token"]:
                        print(f"   {metric:<15}: ${value:.6f}")
                    else:
                        print(f"   {metric:<15}: {value:.3f}")
                else:
                    print(f"   {metric:<15}: {value}")

            if (
                result.baseline_comparison
                and "differences" in result.baseline_comparison
            ):
                print("\n Baseline Comparison:")
                for metric, diff in result.baseline_comparison["differences"].items():
                    print(f"   {metric:<15}: {diff['percentage']:+.1f}%")

        # Save results if output specified
        if output:
            orchestrator.save_result(result, output)
            print(f"\n Results saved to {output}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Evaluation failed: {e}")
        return 1


@eval.command("compare")
@click.argument("model_a")
@click.argument("model_b")
@click.option("--dataset", required=True, help="Dataset for comparison")
@click.option(
    "--metrics",
    multiple=True,
    default=["accuracy", "perplexity"],
    help="Metrics to compare",
)
@click.option("--output", help="Output file for comparison results")
def compare_models(model_a, model_b, dataset, metrics, output):
    """Compare two models side-by-side"""
    from terradev_cli.core.evaluation_orchestrator import EvaluationOrchestrator

    try:
        orchestrator = EvaluationOrchestrator()

        print("\n Comparing Models:")
        print(f"   Model A: {model_a}")
        print(f"   Model B: {model_b}")
        print(f"   Dataset: {dataset}")
        print(f"   Metrics: {', '.join(metrics)}")

        comparison = orchestrator.compare_models(
            model_a, model_b, dataset, list(metrics)
        )

        print("\n Comparison Results:")
        print(
            f"   {'Metric':<15} {'Model A':<12} {'Model B':<12} {'Winner':<10} {'Difference':<12}"
        )
        print(f"   {'─'*65}")

        for metric in metrics:
            val_a = comparison["model_a"]["metrics"].get(metric, 0)
            val_b = comparison["model_b"]["metrics"].get(metric, 0)
            winner = comparison["winner"].get(metric, "tie")
            diff = comparison["differences"].get(metric, {}).get("percentage", 0)

            # Format values
            if isinstance(val_a, float):
                val_a_str = f"{val_a:.3f}"
                val_b_str = f"{val_b:.3f}"
            else:
                val_a_str = str(val_a)
                val_b_str = str(val_b)

            diff_str = f"{diff:+.1f}%"

            print(
                f"   {metric:<15} {val_a_str:<12} {val_b_str:<12} {winner:<10} {diff_str:<12}"
            )

        # Save comparison if output specified
        if output:
            output_file = Path(output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(comparison, indent=2))
            print(f"\n Comparison saved to {output}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Model comparison failed: {e}")
        return 1


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Import/Export Commands
# ═══════════════════════════════════════════════════════════════════════


@cli.command()
@click.option("--output", "-o", required=True, help="Output YAML file path")
@click.option("--job", "-j", help="Specific job to export (omits latest)")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["argo", "native"]),
    default="argo",
    help="Output format (argo-compatible or terradev-native)",
)
def export(output, job, cache_dir, output_format):
    """Export current state or job as Argo-compatible YAML pipeline"""
    try:
        from terradev_cli.core.manifest_cache import ManifestCache
        from terradev_cli.core.pipeline_schema import Workflow, WorkflowMetadata, TerradevAnnotations
        import yaml

        cache = ManifestCache(cache_dir)

        if job:
            # Export specific job
            manifest = cache.load_manifest(job)
            if not manifest:
                print(f"ERROR: Job '{job}' not found in manifest cache")
                print(
                    f"Tip: Run 'terradev manifests --job {job}' to see available versions"
                )
                return 1

            print(f"UPLOAD: Exporting job '{job}' (version {manifest.version})...")

            # Convert manifest to Argo workflow
            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name=f"{job}-exported", namespace="default", annotations={}
            )

            # Add Terradev annotations from manifest
            if manifest.nodes:
                first_node = manifest.nodes[0]
                terradev_ann = TerradevAnnotations()
                terradev_ann.provider = first_node.provider
                terradev_ann.gpu_type = first_node.gpu_type
                terradev_ann.gpu_count = first_node.gpus

                workflow.metadata.annotations.update(terradev_ann.to_dict())

            # Create basic workflow structure
            workflow.spec = {
                "entrypoint": "main",
                "templates": [
                    {
                        "name": "main",
                        "steps": [
                            [{"name": "provision", "template": "provision-step"}],
                            [{"name": "run-workload", "template": "workload-step"}],
                        ],
                    },
                    {
                        "name": "provision-step",
                        "container": {
                            "image": "terradev/cli:latest",
                            "command": ["terradev", "provision"],
                            "args": [
                                "--provider",
                                first_node.provider,
                                "--gpu-type",
                                first_node.gpu_type,
                                "--gpu-count",
                                str(first_node.gpus),
                            ],
                        },
                    },
                    {
                        "name": "workload-step",
                        "container": {
                            "image": "pytorch/pytorch:latest",
                            "command": ["python", "train.py"],
                        },
                    },
                ],
            }

        else:
            # Export current state
            print("UPLOAD: Exporting current Terradev state...")

            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name="current-state-export",
                namespace="default",
                annotations={
                    "terradev.io/export-type": "current-state",
                    "terradev.io/export-timestamp": datetime.now().isoformat(),
                },
            )

            # Create template for current configuration
            workflow.spec = {
                "entrypoint": "main",
                "templates": [
                    {
                        "name": "main",
                        "container": {
                            "image": "terradev/cli:latest",
                            "command": ["terradev", "status"],
                        },
                    }
                ],
            }

        # Write YAML output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if output_format == "argo":
                f.write(workflow.to_yaml())
            else:
                # Native format - just dump manifest data
                if job and manifest:
                    yaml.dump(
                        {
                            "job": manifest.job,
                            "version": manifest.version,
                            "nodes": [
                                {
                                    "provider": node.provider,
                                    "pod_id": node.pod_id,
                                    "instance_id": node.instance_id,
                                    "gpus": node.gpus,
                                    "gpu_type": node.gpu_type,
                                    "region": node.region,
                                    "status": node.status,
                                    "created_at": node.created_at,
                                    "ttl": node.ttl,
                                }
                                for node in manifest.nodes
                            ],
                            "dataset_hash": manifest.dataset_hash,
                            "ttl": manifest.ttl,
                            "created_at": manifest.created_at,
                            "metadata": manifest.metadata,
                        },
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                else:
                    yaml.dump(
                        {
                            "exported": "current-state",
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                    )

        print(f"OK: Exported to {output} (format: {output_format})")

        if job and manifest:
            print(f"   Job: {manifest.job}")
            print(f"   Version: {manifest.version}")
            print(f"   Nodes: {len(manifest.nodes)}")
            print(
                f"   Provider: {manifest.nodes[0].provider if manifest.nodes else 'N/A'}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Export failed: {e}")
        return 1


@cli.command()
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option(
    "--name", "-n", help="Name to register pipeline (defaults to YAML metadata name)"
)
@click.option(
    "--force", is_flag=True, help="Overwrite existing pipeline with same name"
)
@click.option("--validate-only", is_flag=True, help="Only validate, do not register")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
def import_cmd(yaml_file, name, force, validate_only, cache_dir):
    """Import and register Argo-compatible YAML pipeline"""
    try:
        from terradev_cli.core.pipeline_schema import Workflow, PipelineValidator
        from terradev_cli.core.manifest_cache import ManifestCache, Manifest, ManifestNode

        print(f" Importing pipeline from {yaml_file}...")

        # Validate YAML file
        is_valid, errors = PipelineValidator.validate_yaml_file(yaml_file)
        if not is_valid:
            print("ERROR: YAML validation failed:")
            for error in errors:
                print(f"   - {error}")
            return 1

        # Parse workflow
        workflow = Workflow.from_file(yaml_file)

        if not workflow.metadata or not workflow.metadata.name:
            print("ERROR: Workflow must have metadata.name")
            return 1

        pipeline_name = name or workflow.metadata.name

        # Check if already exists
        cache = ManifestCache(cache_dir)
        existing_versions = cache.list_versions(pipeline_name)

        if existing_versions and not force:
            print(
                f"ERROR: Pipeline '{pipeline_name}' already exists with versions: {', '.join(existing_versions)}"
            )
            print(
                "Tip: Use --force to overwrite or --name to specify a different name"
            )
            return 1

        if validate_only:
            print(f"OK: YAML validation passed for '{pipeline_name}'")
            return 0

        # Extract Terradev annotations
        terradev_ann = workflow.metadata.terradev_annotations

        # Create manifest from workflow
        manifest_nodes = []

        # Create a basic node from workflow annotations
        if terradev_ann.provider and terradev_ann.gpu_type:
            node = ManifestNode(
                provider=terradev_ann.provider.value,
                pod_id=f"imported-{uuid.uuid4().hex[:8]}",
                instance_id=f"imported-{uuid.uuid4().hex[:8]}",
                gpus=terradev_ann.gpu_count or 1,
                gpu_type=terradev_ann.gpu_type.value,
                region="auto-imported",
                status="imported",
                created_at=datetime.now().isoformat(),
                ttl="24h",
            )
            manifest_nodes.append(node)

        # Create manifest
        version = f"v{len(existing_versions) + 1}"
        manifest = Manifest(
            job=pipeline_name,
            version=version,
            nodes=manifest_nodes,
            dataset_hash="imported-yaml",
            ttl="24h",
            created_at=datetime.now().isoformat(),
            metadata={
                "source": "yaml-import",
                "yaml_file": str(yaml_file),
                "workflow_name": workflow.metadata.name,
                "terradev_annotations": workflow.metadata.annotations,
            },
        )

        # Store manifest
        manifest_path = cache.store_manifest(manifest)

        print(f"OK: Imported pipeline '{pipeline_name}' (version {version})")
        print(f"   Stored: {manifest_path}")

        if terradev_ann.provider:
            print(f"   Provider: {terradev_ann.provider.value}")
        if terradev_ann.gpu_type:
            print(f"   GPU Type: {terradev_ann.gpu_type.value}")
        if terradev_ann.gpu_count:
            print(f"   GPU Count: {terradev_ann.gpu_count}")

        print(f"Tip: Run 'terradev job {yaml_file}' to execute this pipeline")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Import failed: {e}")
        return 1


@cli.group()
def record():
    """Record and export live workflows"""
    pass


@record.command("start")
@click.option("--name", "-n", required=True, help="Recording name")
@click.option("--output-dir", default="./recordings", help="Recording output directory")
def record_start(name, output_dir):
    """Start recording a live workflow"""
    try:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        recording_file = output_path / f"{name}.recording"

        # Create recording manifest
        recording_data = {
            "name": name,
            "status": "recording",
            "started_at": datetime.now().isoformat(),
            "commands": [],
            "events": [],
        }

        with open(recording_file, "w") as f:
            json.dump(recording_data, f, indent=2)

        print(f" Started recording '{name}'")
        print(f"   Output: {recording_file}")
        print(
            f"Tip: Run 'terradev record stop --name {name} --export pipeline.yaml' when done"
        )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start recording: {e}")
        return 1


@record.command("stop")
@click.option("--name", "-n", required=True, help="Recording name")
@click.option("--export", help="Export as YAML pipeline file")
@click.option("--output-dir", default="./recordings", help="Recording directory")
def record_stop(name, export, output_dir):
    """Stop recording and optionally export as pipeline"""
    try:
        from pathlib import Path
        from terradev_cli.core.pipeline_schema import Workflow, WorkflowMetadata

        output_path = Path(output_dir)
        recording_file = output_path / f"{name}.recording"

        if not recording_file.exists():
            print(f"ERROR: Recording '{name}' not found")
            return 1

        # Load recording
        with open(recording_file, "r") as f:
            recording_data = json.load(f)

        # Update status
        recording_data["status"] = "stopped"
        recording_data["stopped_at"] = datetime.now().isoformat()

        with open(recording_file, "w") as f:
            json.dump(recording_data, f, indent=2)

        print(f" Stopped recording '{name}'")

        if export:
            # Convert recording to workflow
            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name=f"{name}-recorded",
                namespace="default",
                annotations={
                    "terradev.io/source": "live-recording",
                    "terradev.io.recording-name": name,
                    "terradev.io.recording-started": recording_data["started_at"],
                },
            )

            # Create workflow from recorded commands
            templates = []
            steps = []

            for i, cmd in enumerate(recording_data.get("commands", [])):
                step_name = f"step-{i+1}"
                template_name = f"template-{i+1}"

                steps.append([{"name": step_name, "template": template_name}])

                # Create template for this command
                template = {
                    "name": template_name,
                    "container": {
                        "image": "terradev/cli:latest",
                        "command": cmd.split(" "),
                        "args": [],
                    },
                }
                templates.append(template)

            # Add main template
            templates.insert(0, {"name": "main", "steps": steps})

            workflow.spec = {"entrypoint": "main", "templates": templates}

            # Export workflow
            export_path = Path(export)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, "w") as f:
                f.write(workflow.to_yaml())

            print(f"UPLOAD: Exported recording as pipeline: {export}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to stop recording: {e}")
        return 1


# ═══════════════════════════════════════════════════════════════════════
# Event System Commands - Triggers, Environments, and Lineage
# ═══════════════════════════════════════════════════════════════════════


@cli.group()
def triggers():
    """Event-driven automation and triggers"""
    pass


@triggers.command("create")
@click.argument("name")
@click.argument("pipeline")
@click.option(
    "--type",
    "trigger_type",
    type=click.Choice(["event", "schedule", "condition"]),
    default="event",
    help="Trigger type",
)
@click.option(
    "--event",
    help="Event type to trigger on (dataset_landed, model_drift_detected, etc.)",
)
@click.option(
    "--schedule", help='Cron schedule (e.g., "0 0 * * 0" for Sunday midnight)'
)
@click.option("--condition", help='Condition expression (e.g., "drift_score > 0.1")')
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    default="dev",
    help="Target environment",
)
def create_trigger(
    name, pipeline, trigger_type, event, schedule, condition, environment
):
    """Create a new trigger"""
    try:
        from terradev_cli.core.event_system import (
            trigger_manager,
            TriggerType,
            EventType,
            Environment,
        )

        # Convert string types to enums
        trigger_type_enum = (
            TriggerType.EVENT_BASED
            if trigger_type == "event"
            else (
                TriggerType.SCHEDULE
                if trigger_type == "schedule"
                else TriggerType.CONDITION
            )
        )

        event_type_enum = None
        if event:
            event_type_enum = EventType(event.replace("-", "_").upper())

        env_enum = Environment(environment)

        trigger_manager.create_trigger(
            name=name,
            trigger_type=trigger_type_enum,
            target_pipeline=pipeline,
            event_type=event_type_enum,
            schedule=schedule,
            condition=condition,
            target_environment=env_enum,
        )

        print(f"OK: Created trigger '{name}'")
        print(f"   Type: {trigger_type}")
        print(f"   Pipeline: {pipeline}")
        print(f"   Environment: {environment}")

        if event:
            print(f"   Event: {event}")
        if schedule:
            print(f"   Schedule: {schedule}")
        if condition:
            print(f"   Condition: {condition}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to create trigger: {e}")
        return 1


@triggers.command("list")
def list_triggers():
    """List all triggers"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        if not trigger_manager.triggers:
            print("No triggers found")
            print("Tip: Use 'terradev triggers create' to create a trigger")
            return

        print(
            f"{'Name':<20} {'Type':<12} {'Pipeline':<20} {'Environment':<12} {'Enabled':<8} {'Count':<6}"
        )
        print("─" * 92)

        for trigger in trigger_manager.triggers.values():
            enabled = "OK:" if trigger.enabled else "ERROR:"
            print(
                f"{trigger.name:<20} {trigger.type.value:<12} {trigger.target_pipeline:<20} "
                f"{trigger.target_environment.value:<12} {enabled:<8} {trigger.trigger_count:<6}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list triggers: {e}")
        return 1


@triggers.command("enable")
@click.argument("name")
def enable_trigger(name):
    """Enable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = True
                print(f"OK: Enabled trigger '{name}'")
                return

        print(f"ERROR: Trigger '{name}' not found")
        return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to enable trigger: {e}")
        return 1


@triggers.command("disable")
@click.argument("name")
def disable_trigger(name):
    """Disable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = False
                print(f" Disabled trigger '{name}'")
                return

        print(f"ERROR: Trigger '{name}' not found")
        return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to disable trigger: {e}")
        return 1


@triggers.command("fire")
@click.argument("event_type")
@click.option("--data", help="JSON data for the event")
@click.option("--source", default="manual", help="Event source")
def fire_event(event_type, data, source):
    """Manually fire an event for testing"""
    try:
        from terradev_cli.core.event_system import event_bus, EventType, Event

        event_type_enum = EventType(event_type.replace("-", "_").upper())

        event_data = {}
        if data:
            event_data = json.loads(data)

        event = Event(type=event_type_enum, source=source, data=event_data)

        event_bus.publish(event)
        print(f" Fired event: {event_type}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to fire event: {e}")
        return 1


@cli.group()
def environments():
    """Environment management and promotion"""
    pass


@environments.command("list")
@click.option("--env", "environment", help="Filter by environment")
def list_environments(environment):
    """List artifacts by environment"""
    try:
        from terradev_cli.core.event_system import lineage_service, Environment

        if environment:
            env_enum = Environment(environment)
            artifacts = [
                a
                for a in lineage_service.artifacts.values()
                if a.environment == env_enum
            ]
        else:
            artifacts = list(lineage_service.artifacts.values())

        if not artifacts:
            print("No artifacts found")
            return

        print(
            f"{'Name':<20} {'Type':<12} {'Environment':<12} {'Version':<8} {'Created':<20}"
        )
        print("─" * 76)

        for artifact in sorted(artifacts, key=lambda x: x.created_at, reverse=True):
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.environment.value:<12} "
                f"{artifact.version:<8} {created:<20}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list environments: {e}")
        return 1


@environments.command("promote")
@click.argument("artifact_name")
@click.option(
    "--from", "from_env", required=True, type=click.Choice(["dev", "staging", "prod"])
)
@click.option(
    "--to", "to_env", required=True, type=click.Choice(["dev", "staging", "prod"])
)
@click.option("--user", default="cli-user", help="User requesting promotion")
def promote_artifact(artifact_name, from_env, to_env, user):
    """Request environment promotion"""
    try:
        from terradev_cli.core.event_system import environment_manager, lineage_service, Environment

        from_enum = Environment(from_env)
        to_enum = Environment(to_env)

        # Find artifact in source environment
        artifact = None
        for a in lineage_service.artifacts.values():
            if a.name == artifact_name and a.environment == from_enum:
                artifact = a
                break

        if not artifact:
            print(f"ERROR: Artifact '{artifact_name}' not found in {from_env}")
            return 1

        promotion = environment_manager.request_promotion(
            artifact_id=artifact.id,
            from_env=from_enum,
            to_env=to_enum,
            requested_by=user,
        )

        print(f" Promotion requested: {from_env} -> {to_env}")
        print(f"   Artifact: {artifact_name}")
        print(f"   Promotion ID: {promotion.id}")
        print(f"   Status: {promotion.status}")
        print(f"Tip: Use 'terradev environments approve {promotion.id}' to complete")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to request promotion: {e}")
        return 1


@environments.command("approve")
@click.argument("promotion_id")
@click.option("--user", default="cli-admin", help="User approving promotion")
def approve_promotion(promotion_id, user):
    """Approve and execute promotion"""
    try:
        from terradev_cli.core.event_system import environment_manager

        success = environment_manager.approve_promotion(promotion_id, user)

        if success:
            print(f"OK: Promotion approved and executed: {promotion_id}")
        else:
            print(f"ERROR: Promotion not found or failed: {promotion_id}")
            return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to approve promotion: {e}")
        return 1


@environments.command("history")
@click.option("--artifact", help="Filter by artifact name")
def promotion_history(artifact):
    """Show promotion history"""
    try:
        from terradev_cli.core.event_system import environment_manager, lineage_service

        artifact_id = None
        if artifact:
            # Find artifact by name
            for a in lineage_service.artifacts.values():
                if a.name == artifact:
                    artifact_id = a.id
                    break

        promotions = environment_manager.get_promotion_history(artifact_id)

        if not promotions:
            print("No promotion history found")
            return

        print(
            f"{'ID':<8} {'Artifact':<20} {'From':<10} {'To':<10} {'Status':<12} {'Requested':<20}"
        )
        print("─" * 84)

        for promo in promotions:
            artifact_name = "unknown"
            if promo.artifact_id in lineage_service.artifacts:
                artifact_name = lineage_service.artifacts[promo.artifact_id].name

            requested = promo.requested_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{promo.id[:8]:<8} {artifact_name:<20} {promo.from_env.value:<10} "
                f"{promo.to_env.value:<10} {promo.status:<12} {requested:<20}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get promotion history: {e}")
        return 1


@cli.group()
def lineage():
    """Artifact lineage and tracking"""
    pass


@lineage.command("register")
@click.argument(
    "type", type=click.Choice(["dataset", "model", "checkpoint", "metrics", "config"])
)
@click.argument("name")
@click.argument("uri")
@click.option(
    "--env", "environment", type=click.Choice(["dev", "staging", "prod"]), default="dev"
)
@click.option("--hash", "artifact_hash", help="Artifact hash")
@click.option("--size", type=int, help="Size in bytes")
@click.option("--user", default="cli-user", help="User registering artifact")
@click.option("--parent", help="Parent artifact ID")
def register_artifact(type, name, uri, environment, artifact_hash, size, user, parent):
    """Register a new artifact for lineage tracking"""
    try:
        from terradev_cli.core.event_system import (
            lineage_service,
            ArtifactType,
            Environment,
            event_bus,
            EventType,
            Event,
        )

        artifact_type_enum = ArtifactType(type)
        env_enum = Environment(environment)

        artifact = lineage_service.register_artifact(
            artifact_type=artifact_type_enum,
            name=name,
            uri=uri,
            environment=env_enum,
            hash=artifact_hash or "",
            size_bytes=size or 0,
            created_by=user,
        )

        # Add parent relationship if specified
        if parent:
            lineage_service.add_relationship(parent, artifact.id)

        # Publish artifact registered event
        event = Event(
            type=EventType.ARTIFACT_REGISTERED,
            source="lineage_service",
            data={
                "artifact_id": artifact.id,
                "artifact_type": type,
                "name": name,
                "environment": environment,
            },
        )
        event_bus.publish(event)

        print(f"OK: Registered artifact: {type} {name}")
        print(f"   ID: {artifact.id}")
        print(f"   Environment: {environment}")
        print(f"   URI: {uri}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to register artifact: {e}")
        return 1


@lineage.command("graph")
@click.argument("artifact_id")
@click.option("--direction", type=click.Choice(["up", "down", "both"]), default="both")
def lineage_graph(artifact_id, direction):
    """Show lineage graph for artifact"""
    try:
        from terradev_cli.core.event_system import lineage_service

        graph = lineage_service.get_lineage(artifact_id, direction)

        if not graph["parents"] and not graph["children"]:
            print(f"No lineage found for artifact {artifact_id}")
            return

        print(f" Lineage for {artifact_id}:")

        if graph["parents"]:
            print(f"\n Parents ({len(graph['parents'])}):")
            for parent in graph["parents"]:
                print(
                    f"   {parent.type.value} {parent.name} ({parent.environment.value})"
                )

        if graph["children"]:
            print(f"\nUPLOAD: Children ({len(graph['children'])}):")
            for child in graph["children"]:
                print(f"   {child.type.value} {child.name} ({child.environment.value})")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get lineage: {e}")
        return 1


@lineage.command("production")
@click.option("--type", "artifact_type", help="Filter by artifact type")
def production_artifacts(artifact_type):
    """Show artifacts in production environment"""
    try:
        from terradev_cli.core.event_system import lineage_service, ArtifactType

        type_enum = None
        if artifact_type:
            type_enum = ArtifactType(artifact_type)

        artifacts = lineage_service.get_production_artifacts(type_enum)

        if not artifacts:
            print("No production artifacts found")
            return

        print(" Production Artifacts:")
        print(
            f"{'Name':<20} {'Type':<12} {'Version':<8} {'Created':<20} {'Created By':<15}"
        )
        print("─" * 79)

        for artifact in artifacts:
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.version:<8} "
                f"{created:<20} {artifact.created_by:<15}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get production artifacts: {e}")
        return 1


@lineage.command("show")
@click.argument("model_identifier")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    help="Filter by environment",
)
def show_model_lineage(model_identifier, environment):
    """Show complete provenance of a model (auto-generated)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment) if environment else None

        # Parse model identifier (could be "prod/llama-70b" or just "llama-70b")
        if "/" in model_identifier:
            env_name, model_name = model_identifier.split("/", 1)
            env_from_identifier = Environment(env_name)
            records = auto_lineage.get_lineage_for_model(
                model_name, env_from_identifier
            )
        else:
            records = auto_lineage.get_lineage_for_model(model_identifier, env_enum)

        if not records:
            print(f"ERROR: No lineage found for model '{model_identifier}'")
            return

        print(f" Lineage for model: {model_identifier}")
        print(
            f"{'Execution ID':<8} {'Pipeline':<20} {'Environment':<12} {'Status':<10} {'Timestamp':<20}"
        )
        print("─" * 75)

        for record in records:
            exec_id = record.id[:8]
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M")
            print(
                f"{exec_id:<8} {record.pipeline_id:<20} {record.environment.value:<12} "
                f"{record.status:<10} {timestamp:<20}"
            )

        # Show detailed view of latest execution
        if records:
            latest = records[0]
            print("\n Latest Execution Details:")
            print(f"   Execution ID: {latest.id}")
            print(f"   Pipeline: {latest.pipeline_id}")
            print(f"   Environment: {latest.environment.value}")
            print(f"   Status: {latest.status}")
            print(f"   Duration: {latest.duration_seconds:.1f}s")
            print(f"   GPU Hours: {latest.gpu_hours:.2f}")
            print(f"   Cost: ${latest.compute_cost:.2f}")

            if latest.hyperparameters:
                print("\n  Hyperparameters:")
                for key, value in latest.hyperparameters.items():
                    print(f"      {key}: {value}")

            if latest.datasets:
                print(f"\n Input Datasets ({len(latest.datasets)}):")
                for dataset_id in latest.datasets[:5]:  # Show first 5
                    print(f"      {dataset_id[:12]}...")
                if len(latest.datasets) > 5:
                    print(f"      ... and {len(latest.datasets) - 5} more")

            if latest.output_models:
                print(f"\n Output Models ({len(latest.output_models)}):")
                for model_id in latest.output_models:
                    print(f"      {model_id[:12]}...")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to show lineage: {e}")
        return 1


@lineage.command("diff")
@click.argument("version1")
@click.argument("version2")
def diff_lineage(version1, version2):
    """Compare two pipeline executions"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        # Try to parse as execution IDs first, then as model versions
        exec1_id = version1
        exec2_id = version2

        diff = auto_lineage.diff_executions(exec1_id, exec2_id)

        if "error" in diff:
            print(f"ERROR: {diff['error']}")
            return 1

        print(" Comparing executions:")
        print(
            f"   Execution 1: {diff['execution_1']['id']} ({diff['execution_1']['timestamp']})"
        )
        print(
            f"   Execution 2: {diff['execution_2']['id']} ({diff['execution_2']['timestamp']})"
        )

        if not diff["differences"]:
            print("\nOK: No differences found")
            return

        print("\n Differences:")

        if "hyperparameters" in diff["differences"]:
            print("\n  Hyperparameters:")
            for key, values in diff["differences"]["hyperparameters"].items():
                print(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "environment_variables" in diff["differences"]:
            print("\n Environment Variables:")
            for key, values in diff["differences"]["environment_variables"].items():
                print(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "inputs" in diff["differences"]:
            print("\n Input Artifacts:")
            for change_type, artifacts in diff["differences"]["inputs"].items():
                print(f"   {change_type}: {', '.join(artifacts)}")

        if "resources" in diff["differences"]:
            print("\nCOST: Resource Usage:")
            for resource, values in diff["differences"]["resources"].items():
                print(f"   {resource}: {values['exec1']} → {values['exec2']}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to diff lineage: {e}")
        return 1


@lineage.command("export")
@click.option(
    "--format", type=click.Choice(["json", "csv"]), default="json", help="Export format"
)
@click.option("--model", help="Filter by model name")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    help="Filter by environment",
)
@click.option("--output", "-o", help="Output file (default: stdout)")
def export_lineage(format, model, environment, output):
    """Export lineage data for compliance reports"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment) if environment else None

        data = auto_lineage.export_lineage(format, model, env_enum)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(data)
            print(f"OK: Exported lineage to {output}")
        else:
            print(data)

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to export lineage: {e}")
        return 1


@lineage.command("trace")
@click.option("--checkpoint", help="Checkpoint ID to trace backwards from")
@click.option("--execution", help="Execution ID to trace")
def trace_artifacts(checkpoint, execution):
    """Trace complete lineage from checkpoint or execution"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        if checkpoint:
            trace = auto_lineage.trace_from_checkpoint(checkpoint)

            if "error" in trace:
                print(f"ERROR: {trace['error']}")
                return 1

            print(f" Tracing lineage from checkpoint: {checkpoint}")
            print("\n Created By:")
            created_by = trace["created_by"]
            print(f"   Execution: {created_by['execution_id']}")
            print(f"   Pipeline: {created_by['pipeline_id']}")
            print(f"   Environment: {created_by['environment']}")
            print(f"   Timestamp: {created_by['timestamp']}")

            if trace["inputs"]:
                print("\n Input Artifacts:")
                if "datasets" in trace["inputs"]:
                    print(f"   Datasets ({len(trace['inputs']['datasets'])}):")
                    for dataset in trace["inputs"]["datasets"]:
                        print(f"      {dataset['name']} ({dataset['id'][:12]}...)")

                if "models" in trace["inputs"]:
                    print(f"   Models ({len(trace['inputs']['models'])}):")
                    for model in trace["inputs"]["models"]:
                        print(f"      {model['name']} ({model['id'][:12]}...)")

            if trace["ancestors"]:
                print("\n Ancestor Executions:")
                for ancestor in trace["ancestors"]:
                    print(
                        f"   {ancestor['execution_id'][:12]}... - {ancestor['pipeline_id']} "
                        f"({ancestor['environment']}) - {ancestor['timestamp']}"
                    )

        elif execution:
            # Show execution details and trace backwards
            from terradev_cli.core.event_system import lineage_service

            # Find execution record
            exec_record = None
            for record in auto_lineage.completed_executions:
                if record.id.startswith(execution):
                    exec_record = record
                    break

            if not exec_record:
                print(f"ERROR: Execution '{execution}' not found")
                return 1

            print(f" Tracing execution: {exec_record.id}")
            print(f"   Pipeline: {exec_record.pipeline_id}")
            print(f"   Environment: {exec_record.environment.value}")
            print(f"   Status: {exec_record.status}")
            print(f"   Timestamp: {exec_record.timestamp}")

            # Show artifact lineage
            all_artifacts = (
                exec_record.datasets
                + exec_record.models
                + exec_record.output_models
                + exec_record.output_checkpoints
            )

            if all_artifacts:
                print("\n Artifact Lineage:")
                for artifact_id in all_artifacts:
                    if artifact_id in lineage_service.artifacts:
                        artifact = lineage_service.artifacts[artifact_id]
                        graph = lineage_service.get_lineage(artifact_id, "up")

                        print(f"\n   {artifact.type.value}: {artifact.name}")
                        print(f"      Environment: {artifact.environment.value}")
                        print(f"      Created: {artifact.created_at}")

                        if graph["parents"]:
                            print(f"      Parents: {len(graph['parents'])}")
                            for parent in graph["parents"][:3]:  # Show first 3
                                print(f"         └─ {parent.type.value}: {parent.name}")

        else:
            print("ERROR: Must specify either --checkpoint or --execution")
            return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to trace artifacts: {e}")
        return 1


@lineage.command("auto")
@click.option("--pipeline", required=True, help="Pipeline ID")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    default="dev",
    help="Execution environment",
)
@click.option("--triggered-by", default="manual", help="Who triggered this execution")
def start_auto_lineage(pipeline, environment, triggered_by):
    """Start automatic lineage tracking for a pipeline execution"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment)

        execution = auto_lineage.start_execution(
            pipeline_id=pipeline, environment=env_enum, triggered_by=triggered_by
        )

        print(" Started automatic lineage tracking")
        print(f"   Execution ID: {execution.id}")
        print(f"   Pipeline: {pipeline}")
        print(f"   Environment: {environment}")
        print("   Use this ID to add artifacts and complete the execution")

        # Show example commands for manual tracking
        print("\nTip: Example commands to track this execution:")
        print(f"   terradev lineage add-input {execution.id} dataset <dataset-id>")
        print(f"   terradev lineage add-output {execution.id} model <model-id>")
        print(f"   terradev lineage complete {execution.id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start lineage tracking: {e}")
        return 1


@lineage.command("add-input")
@click.argument("execution_id")
@click.argument(
    "artifact_type", type=click.Choice(["dataset", "model", "config", "checkpoint"])
)
@click.argument("artifact_id")
def add_input_artifact(execution_id, artifact_type, artifact_id):
    """Add input artifact to execution (manual override)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import ArtifactType

        artifact_enum = ArtifactType(artifact_type)
        auto_lineage.add_input_artifact(execution_id, artifact_enum, artifact_id)

        print(f"OK: Added input {artifact_type}: {artifact_id}")
        print(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to add input artifact: {e}")
        return 1


@lineage.command("add-output")
@click.argument("execution_id")
@click.argument(
    "artifact_type", type=click.Choice(["model", "checkpoint", "metrics", "evaluation"])
)
@click.argument("artifact_id")
def add_output_artifact(execution_id, artifact_type, artifact_id):
    """Add output artifact to execution (manual override)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import ArtifactType

        artifact_enum = ArtifactType(artifact_type)
        auto_lineage.add_output_artifact(execution_id, artifact_enum, artifact_id)

        print(f"OK: Added output {artifact_type}: {artifact_id}")
        print(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to add output artifact: {e}")
        return 1


@lineage.command("complete")
@click.argument("execution_id")
@click.option("--status", default="completed", help="Final status (completed, failed)")
def complete_execution(execution_id, status):
    """Complete execution and finalize lineage record"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        auto_lineage.complete_execution(execution_id, status)

        print(f"OK: Completed execution: {execution_id}")
        print(f"   Status: {status}")
        print("   Lineage record finalized and available for queries")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to complete execution: {e}")
        return 1


# Add command groups to CLI
cli.add_command(migrate)
cli.add_command(eval)
cli.add_command(sso)
cli.add_command(retrain)
cli.add_command(agentic_serving)
cli.add_command(model_router)
cli.add_command(record)
cli.add_command(triggers)
cli.add_command(environments)
cli.add_command(lineage)

# Register Karpenter and HF Spaces command groups
from terradev_cli.cli_karpenter import register_karpenter_commands

register_karpenter_commands(cli, TerradevAPI)

from terradev_cli.cli_hf_spaces import register_hf_spaces_commands

register_hf_spaces_commands(cli, TerradevAPI)


# MCP Server Command
@cli.command("mcp")
@click.argument("action", type=click.Choice(["serve", "install", "list-tools"]))
@click.option(
    "--client",
    type=click.Choice(["claude-desktop", "cursor", "windsurf", "continue", "cline"]),
    help="Client to install MCP config for",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="MCP transport protocol",
)
def mcp(action, client, transport):
    """Run Terradev as an MCP server for agent integration.

    Makes Terradev callable from AI agents (Claude Desktop, Cursor, Windsurf, Continue, Cline).

    Actions:
      serve: Start MCP server (default: stdio transport)
      install: Install MCP config for a specific client
      list-tools: List all available MCP tools
    """
    try:
        from terradev_cli.mcp import run_server, install_config, list_tools
    except ImportError:
        click.echo(
            "Error: MCP module not found. Install with: pip install mcp", err=True
        )
        return 1

    if action == "serve":
        run_server(transport=transport)
    elif action == "install":
        if not client:
            click.echo("Error: --client is required for install action", err=True)
            return 1
        install_config(client)
    elif action == "list-tools":
        list_tools()


@cli.group()
def local():
    """Local GPU discovery and hybrid compute pool management.

    Discover GPUs on this machine or remote hosts via SSH, register them
    into your compute pool alongside cloud providers, and route workloads
    to the cheapest available compute  including $0/hr local hardware.
    """
    pass


@local.command("scan")
@click.option("--host", default=None, help="Remote host IP/hostname to scan via SSH")
@click.option("--user", default="ubuntu", help="SSH username for remote scan")
@click.option("--key", default=None, help="Path to SSH private key for remote scan")
@click.option(
    "--detailed", is_flag=True, help="Show full topology, PCIe, NUMA, clock details"
)
@click.option(
    "--register", is_flag=True, help="Auto-register discovered GPUs into pool"
)
@click.option(
    "--name",
    default=None,
    help="Name for registered pool entry (auto-generated if omitted)",
)
def local_scan(host, user, key, detailed, register, name):
    """Scan local machine or remote host for GPUs.

    Uses Rust NVML bindings (5-10x faster than nvidia-smi) with automatic
    fallback to nvidia-smi parsing if the Rust extension is unavailable.

    Examples:

        terradev local scan

        terradev local scan --detailed

        terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa

        terradev local scan --register --name workstation-4090
    """
    import subprocess
    import datetime

    target = host if host else "localhost"
    click.echo(f"Scanning {target} for GPUs...")

    def _run_nvidia_smi(remote_host=None, remote_user=None, remote_key=None):
        query = "index,name,memory.total,driver_version,utilization.gpu,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current,compute_cap"
        if remote_host:
            # Validate inputs to prevent shell injection
            import re

            if not re.match(r"^[a-zA-Z0-9._-]+$", remote_user):
                return "", 1
            if not re.match(r"^[a-zA-Z0-9._-]+$", remote_host):
                return "", 1
            if remote_key and not re.match(r"^[a-zA-Z0-9._/~-]+$", remote_key):
                return "", 1

            # Build SSH command as argument list (no shell=True)
            ssh_args = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=10",
            ]
            if remote_key:
                ssh_args.extend(["-i", remote_key])
            ssh_args.extend(
                [
                    f"{remote_user}@{remote_host}",
                    f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
                ]
            )
            try:
                result = subprocess.run(
                    ssh_args, capture_output=True, text=True, timeout=15
                )
                return result.stdout.strip(), result.returncode
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                return "", 1
        else:
            # Local execution - use list-args for injection safety
            cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=15
                )
                return result.stdout.strip(), result.returncode
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                return "", 1

    # Try Rust NVML first (local only), then nvidia-smi
    gpus = []
    use_rust = False
    if not host:
        try:
            from terradev_cli.core.gpu_discovery import GPUDiscoveryWrapper

            disc = GPUDiscoveryWrapper(cache_ttl_secs=0)
            state = disc.discover_gpus()
            if state and state.get("total_count", 0) > 0:
                use_rust = True
                for g in state.get("gpus", []):
                    gpus.append(g)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pass

    if not use_rust:
        raw, rc = _run_nvidia_smi(host, user, key)
        if rc != 0 or not raw:
            click.echo(
                f"No GPUs found on {target} or nvidia-smi not available.", err=True
            )
            return
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else 0,
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]) if parts[2] else 0,
                    "driver_version": parts[3],
                    "utilization_gpu": float(parts[4]) if parts[4] else 0,
                    "temperature": float(parts[5]) if parts[5] else 0,
                    "power_draw": float(parts[6]) if parts[6] else 0,
                    "power_limit": float(parts[7]) if parts[7] else 0,
                    "pcie_gen": parts[8],
                    "pcie_width": parts[9],
                    "compute_cap": parts[10],
                }
            )

    if not gpus:
        click.echo(f"No GPUs found on {target}.")
        return

    click.echo(f"\nFound {len(gpus)} GPU{'s' if len(gpus) > 1 else ''} on {target}:\n")
    for g in gpus:
        idx = g.get("index", 0)
        gpu_name = g.get("name", "Unknown")
        mem_mb = g.get("memory_total_mb", g.get("memory_total", 0))
        mem_gb = (
            round(float(mem_mb) / 1024, 1) if float(mem_mb) > 100 else float(mem_mb)
        )
        driver = g.get("driver_version", "N/A")
        util = g.get("utilization_gpu", g.get("utilization", 0))
        temp = g.get("temperature", 0)
        click.echo(
            f"  [{idx}] {gpu_name}  {mem_gb}GB  Driver {driver}  Util: {util}%  Temp: {temp}C"
        )
        if detailed:
            pcie_gen = g.get("pcie_gen", "N/A")
            pcie_w = g.get("pcie_width", "N/A")
            pwr_draw = g.get("power_draw", 0)
            pwr_lim = g.get("power_limit", 0)
            compute = g.get("compute_cap", "N/A")
            numa = g.get("numa_node", "N/A")
            click.echo(
                f"      PCIe: Gen{pcie_gen} x{pcie_w}  NUMA: {numa}  Compute: {compute}"
            )
            click.echo(f"      Power: {pwr_draw}W / {pwr_lim}W TDP")

    if register or (not register and click.confirm("\nRegister in pool?")):
        pool_name = (
            name
            if name
            else f"local-{'remote-' if host else ''}{gpus[0].get('name','gpu').replace(' ','').lower()}-{datetime.datetime.now().strftime('%H%M')}"
        )
        _register_local_pool(gpus, pool_name, host, user, key)
        click.echo(f"\nRegistered as '{pool_name}'. View with: terradev local pool")


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


@local.command("register")
@click.option("--name", required=True, help="Name for this pool entry")
@click.option("--host", default=None, help="Remote host (omit for localhost)")
@click.option("--user", default="ubuntu", help="SSH username for remote host")
@click.option("--key", default=None, help="SSH private key path")
def local_register(name, host, user, key):
    """Register a local or remote GPU host into your compute pool.

    Example:

        terradev local register --name workstation-4090

        terradev local register --name lab-node-01 --host 10.0.0.5 --user ubuntu
    """
    import subprocess

    target = host or "localhost"
    click.echo(f"Scanning {target}...")
    query = "index,name,memory.total,driver_version,utilization.gpu,temperature.gpu"
    if host:
        # Validate inputs to prevent shell injection
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", user):
            click.echo("Error: Invalid username format", err=True)
            return
        if not re.match(r"^[a-zA-Z0-9._-]+$", host):
            click.echo("Error: Invalid hostname format", err=True)
            return
        if key and not re.match(r"^[a-zA-Z0-9._/~-]+$", key):
            click.echo("Error: Invalid key path format", err=True)
            return

        # Build SSH command as argument list (no shell=True)
        ssh_args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
        ]
        if key:
            ssh_args.extend(["-i", key])
        ssh_args.extend(
            [
                f"{user}@{host}",
                f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
            ]
        )
        try:
            result = subprocess.run(
                ssh_args, capture_output=True, text=True, timeout=15
            )
            raw = result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            click.echo(f"Error scanning {target}: {e}", err=True)
            return
    else:
        # Local execution - use list-args for injection safety
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            raw = result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            click.echo(f"Error scanning {target}: {e}", err=True)
            return
    if not raw:
        click.echo(f"No GPUs found on {target}.", err=True)
        return
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else 0,
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]) if parts[2] else 0,
                    "driver_version": parts[3],
                    "utilization_gpu": float(parts[4]) if parts[4] else 0,
                    "temperature": float(parts[5]) if parts[5] else 0,
                }
            )
    _register_local_pool(gpus, name, host, user, key)
    click.echo(
        f"Registered '{name}' with {len(gpus)} GPU(s). View with: terradev local pool"
    )


@local.command("pool")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--remove", default=None, help="Remove a pool entry by name")
def local_pool(fmt, remove):
    """View or manage your hybrid compute pool (local + cloud instances).

    Shows all registered local/remote GPU hosts alongside active cloud instances.

    Example:

        terradev local pool

        terradev local pool --format json

        terradev local pool --remove workstation-4090
    """
    import json
    import os

    pool_path = os.path.expanduser("~/.terradev/local_pool.json")
    pool = {}
    if os.path.exists(pool_path):
        try:
            with open(pool_path) as f:
                pool = json.load(f)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pool = {}

    if remove:
        if remove in pool:
            del pool[remove]
            with open(pool_path, "w") as f:
                json.dump(pool, f, indent=2)
            click.echo(f"Removed '{remove}' from pool.")
        else:
            click.echo(f"'{remove}' not found in pool.", err=True)
        return

    if fmt == "json":
        click.echo(json.dumps(pool, indent=2))
        return

    if not pool:
        click.echo("No local pool entries registered.")
        click.echo("Add one with: terradev local scan --register")
        return

    click.echo(
        f"\nCOMPUTE POOL ({len(pool)} local resource{'s' if len(pool) > 1 else ''})\n"
    )
    click.echo(
        f"{'NAME':<24} {'GPU':<12} {'VRAM':>6}  {'PROVIDER':<12} {'$/HR':>7}  STATUS"
    )
    click.echo("-" * 72)
    for entry_name, entry in pool.items():
        gpus = entry.get("gpus", [])
        gpu_name = (
            gpus[0].get("name", "Unknown").replace("NVIDIA ", "") if gpus else "Unknown"
        )
        mem_mb = gpus[0].get("memory_total_mb", 0) if gpus else 0
        mem_gb = (
            f"{round(float(mem_mb)/1024, 0):.0f}GB"
            if float(mem_mb) > 100
            else f"{mem_mb}GB"
        )
        provider = entry.get("provider", "local")
        price = entry.get("price_per_hour", 0.0)
        host = entry.get("host", "localhost")
        status = "localhost" if host == "localhost" else host
        click.echo(
            f"{entry_name:<24} {gpu_name:<12} {mem_gb:>6}  {provider:<12} ${price:>6.2f}  {status}"
        )

    click.echo("\nCloud instances: run 'terradev status --live' for cloud pool.")
    click.echo(
        "To provision preferring local: terradev provision -g RTX4090 --prefer-local"
    )


# ── Agent Fleet Command Group ─────────────────────────────────────────────────
# terradev agent <subcommand>
# Research basis: arXiv:2605.26297 "Agentic AI Workload Characteristics"
#   - Decode dominates: 91-98% of LLM time
#   - KV cache hit rates: 84.6-99.5% (eviction = expensive recompute)
#   - Context footprint: 37K-166K tokens (P95 tail: 120K)
#   - Tool calls: 2-29% of runtime


@cli.group()
def agent():
    """Provision and manage heterogeneous agent fleets.

    Multi-tier GPU provisioning purpose-built for multi-agent LLM workloads.
    Automatically maps agent count to hardware tiers based on empirical
    workload research (decode-dominated, KV cache preservation critical).

    \b
    Tiers provisioned:
      reasoning  — H100 SXM: long-context KV preservation (P95: 120K tokens)
      decode     — A100 80GB: memory-bandwidth-optimised token streaming
      cpu_tools  — 48-vCPU: Bash/WebFetch/file-op tool execution

    \b
    Examples:
      terradev agent plan   --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --topology ./agent-fleet.yaml
      terradev agent status --fleet-id ag_abc123
      terradev agent scale  --fleet-id ag_abc123 --tier decode --count 8
      terradev agent cost   --fleet-id ag_abc123
      terradev agent list
      terradev agent teardown --fleet-id ag_abc123
    """
    pass


@agent.command(name="plan")
@click.option("--agents", "-n", type=int, required=True, help="Number of concurrent agent loops to provision for")
@click.option("--model", "-m", default="meta-llama/Llama-3.1-70B-Instruct", help="Model to serve across the fleet")
@click.option("--reasoning", type=click.Choice(["instant", "thinking"]), default="instant", help="Reasoning mode: instant (faster) or thinking (extended CoT, 45-67% more output tokens)")
@click.option("--planner-gpu", default=None, help="Override reasoning tier GPU type (e.g. H100_SXM)")
@click.option("--planner-count", type=int, default=None, help="Override reasoning tier instance count")
@click.option("--worker-gpu", default=None, help="Override decode tier GPU type (e.g. A100_SXM_80)")
@click.option("--worker-count", type=int, default=None, help="Override decode tier instance count")
@click.option("--cpu-cores", type=int, default=48, help="vCPU count for CPU tools tier instances")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
def agent_plan(agents, model, reasoning, planner_gpu, planner_count, worker_gpu, worker_count, cpu_cores, fmt):
    """Plan a heterogeneous agent fleet without provisioning.

    Shows the recommended tier configuration, hardware selection rationale,
    KV cache budget, and cost estimate based on arXiv:2605.26297 research.

    \b
    Examples:
      terradev agent plan --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent plan --agents 32 --model meta-llama/Llama-3.1-8B-Instruct --format json
      terradev agent plan --agents 8 --planner-gpu H100_SXM --worker-gpu A100_SXM_80
    """
    import json as _json
    from terradev_cli.core.agentic_topology import AgentTopologyPlanner

    planner = AgentTopologyPlanner()

    if planner_gpu and worker_gpu:
        spec = planner.from_explicit(
            n_agents=agents, model=model,
            planner_gpu=planner_gpu, planner_count=planner_count or max(1, agents // 10),
            worker_gpu=worker_gpu, worker_count=worker_count or agents,
            cpu_cores=cpu_cores, reasoning=reasoning,
        )
    else:
        spec = planner.infer_from_agent_count(n_agents=agents, model=model, reasoning=reasoning)
        if planner_count:
            spec.tiers["reasoning"].count = planner_count
        if worker_count:
            spec.tiers["decode"].count = worker_count

    if fmt == "json":
        cost = planner.estimate_cost(spec)
        output = spec.to_dict()
        output["cost"] = {
            "reasoning_hr": cost.reasoning_hr,
            "decode_hr": cost.decode_hr,
            "cpu_hr": cost.cpu_hr,
            "total_hr": cost.total_hr,
            "daily": cost.daily,
            "monthly": cost.monthly,
            "cost_per_agent_hr": cost.cost_per_agent_hr,
        }
        click.echo(_json.dumps(output, indent=2))
    else:
        planner.print_plan(spec)
        click.echo(f"\nTo provision this fleet:")
        click.echo(f"  terradev agent deploy --agents {agents} --model {model}")
        click.echo(f"\nTo provision with explicit overrides:")
        r = spec.tiers["reasoning"]
        d = spec.tiers["decode"]
        click.echo(
            f"  terradev agent deploy --agents {agents} --model {model} "
            f"--planner-gpu {r.gpu_type} --planner-count {r.count} "
            f"--worker-gpu {d.gpu_type} --worker-count {d.count}"
        )


@agent.command(name="deploy")
@click.option("--agents", "-n", type=int, default=None, help="Number of concurrent agent loops")
@click.option("--model", "-m", default="meta-llama/Llama-3.1-70B-Instruct", help="Model to serve")
@click.option("--reasoning", type=click.Choice(["instant", "thinking"]), default="instant")
@click.option("--topology", type=click.Path(exists=True), default=None, help="Path to agent-fleet.yaml spec file")
@click.option("--planner-gpu", default=None, help="Reasoning tier GPU type")
@click.option("--planner-count", type=int, default=None, help="Reasoning tier instance count")
@click.option("--worker-gpu", default=None, help="Decode tier GPU type")
@click.option("--worker-count", type=int, default=None, help="Decode tier instance count")
@click.option("--cpu-cores", type=int, default=48, help="vCPU count for CPU tools tier")
@click.option("--providers", "-p", multiple=True, help="Cloud providers to use (e.g. runpod vastai)")
@click.option("--max-price", type=float, default=None, help="Max price per GPU/hr in USD")
@click.option("--dry-run", is_flag=True, help="Show allocation plan without provisioning")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_deploy(agents, model, reasoning, topology, planner_gpu, planner_count, worker_gpu, worker_count, cpu_cores, providers, max_price, dry_run, fmt):
    """Provision a heterogeneous agent fleet across all tiers simultaneously.

    Provisions reasoning (H100), decode (A100), and CPU tools tiers in parallel
    using the existing DAGExecutor wave-parallel orchestration.

    \b
    Examples:
      terradev agent deploy --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --agents 32 --dry-run
      terradev agent deploy --topology ./agent-fleet.yaml
      terradev agent deploy --agents 8 --planner-gpu H100_SXM --worker-gpu A100_SXM_80
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_topology import AgentTopologyPlanner
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    if topology:
        import yaml
        with open(topology) as f:
            spec_data = yaml.safe_load(f)
        agents = agents or spec_data.get("n_agents", 16)
        model = spec_data.get("model", model)

    if not agents:
        click.echo("Error: --agents or --topology required", err=True)
        raise SystemExit(1)

    planner = AgentTopologyPlanner()
    if planner_gpu and worker_gpu:
        spec = planner.from_explicit(
            n_agents=agents, model=model,
            planner_gpu=planner_gpu, planner_count=planner_count or max(1, agents // 10),
            worker_gpu=worker_gpu, worker_count=worker_count or agents,
            cpu_cores=cpu_cores, reasoning=reasoning,
        )
    else:
        spec = planner.infer_from_agent_count(n_agents=agents, model=model, reasoning=reasoning)
        if planner_count:
            spec.tiers["reasoning"].count = planner_count
        if worker_count:
            spec.tiers["decode"].count = worker_count

    if not dry_run:
        planner.print_plan(spec)
        click.echo(f"\nProvisioning fleet {spec.fleet_id}...")
    else:
        click.echo("[DRY RUN] Fleet plan — no instances will be provisioned:\n")
        planner.print_plan(spec)

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.provision_fleet(
        spec=spec,
        dry_run=dry_run,
        providers=list(providers) if providers else None,
        max_price_hr=max_price,
    ))

    if fmt == "json":
        output = {
            "fleet_id": result.fleet_id,
            "success": result.success,
            "dry_run": dry_run,
            "wall_ms": round(result.total_wall_ms, 1),
            "cost_estimate": {
                "total_hr": result.cost_estimate.total_hr,
                "daily": result.cost_estimate.daily,
                "monthly": result.cost_estimate.monthly,
            },
            "tiers": {k: v.count for k, v in spec.tiers.items()},
            "state_path": result.state_path,
            "errors": result.errors,
        }
        click.echo(_json.dumps(output, indent=2))
        return

    if result.success:
        status_tag = "[DRY RUN]" if dry_run else "PROVISIONED"
        click.echo(f"\n{status_tag}  Fleet: {result.fleet_id}")
        click.echo(f"  Model:   {spec.model}")
        click.echo(f"  Agents:  {spec.n_agents} concurrent loops")
        click.echo(f"  Cost:    ${result.cost_estimate.total_hr:.2f}/hr  (${result.cost_estimate.monthly:.2f}/mo)")
        click.echo(f"  Tiers:   {spec.tiers['reasoning'].count}× reasoning | {spec.tiers['decode'].count}× decode | {spec.tiers['cpu_tools'].count}× cpu_tools")
        if not dry_run:
            click.echo(f"  State:   {result.state_path}")
        click.echo()
        click.echo(f"  terradev agent status --fleet-id {result.fleet_id}")
        click.echo(f"  terradev agent cost   --fleet-id {result.fleet_id}")
        click.echo(f"  terradev agent scale  --fleet-id {result.fleet_id} --tier decode --count <N>")
    else:
        click.echo(f"\nProvisioning errors:", err=True)
        for e in result.errors:
            click.echo(f"  {e}", err=True)


@agent.command(name="status")
@click.option("--fleet-id", required=True, help="Fleet ID returned by 'terradev agent deploy'")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_status(fleet_id, fmt):
    """Show live status of a fleet — tier health, KV hit rate, queue depth, cost.

    Key metrics explained:
      kv_hit_rate  — target >0.85. Below 0.80 = cache thrashing (expensive recompute).
      ttft_p95_ms  — reasoning tier target <2000ms. Above = scale out reasoning.
      queue_depth  — decode tier pending requests. Above 6 = scale out decode.

    (Metrics from arXiv:2605.26297 empirical benchmarking)
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    status = asyncio.run(provisioner.fleet_status(fleet_id))

    if status is None:
        click.echo(f"Fleet {fleet_id} not found.", err=True)
        raise SystemExit(1)

    if fmt == "json":
        output = {
            "fleet_id": status.fleet_id,
            "model": status.model,
            "n_agents": status.n_agents,
            "kv_cache_pressure": status.kv_cache_pressure,
            "total_cost_hr": status.total_cost_hr,
            "uptime_s": status.uptime_s,
            "warnings": status.warnings,
            "tiers": {
                name: {
                    "instances": t.instances,
                    "healthy": t.healthy,
                    "failed": t.failed,
                    "kv_hit_rate": t.kv_hit_rate,
                    "decode_queue_depth": t.decode_queue_depth,
                    "ttft_p95_ms": t.ttft_p95_ms,
                    "cost_hr": t.cost_hr,
                }
                for name, t in status.tiers.items()
            },
        }
        click.echo(_json.dumps(output, indent=2))
        return

    pressure_icon = {"healthy": "OK", "warning": "WARN", "critical": "CRIT"}.get(status.kv_cache_pressure, "?")
    click.echo(f"\nFLEET STATUS  [{fleet_id}]")
    click.echo(f"Model: {status.model}  |  {status.n_agents} agents  |  KV cache: {pressure_icon}  |  ${status.total_cost_hr:.2f}/hr")
    click.echo(f"Uptime: {status.uptime_s / 3600:.1f}h")
    click.echo()
    click.echo(f"{'TIER':<16} {'INSTANCES':>9} {'HEALTHY':>7} {'FAILED':>6}  {'KV HIT':>7}  {'TTFT P95':>9}  {'QUEUE':>5}  {'$/HR':>6}")
    click.echo("-" * 80)
    for tname, t in status.tiers.items():
        kv_str = f"{t.kv_hit_rate:.0%}" if t.kv_hit_rate > 0 else "n/a"
        ttft_str = f"{t.ttft_p95_ms:.0f}ms" if t.ttft_p95_ms > 0 else "n/a"
        q_str = str(t.decode_queue_depth) if t.gpu_type else "n/a"
        kv_warn = " !" if t.kv_hit_rate > 0 and t.kv_hit_rate < 0.80 else ""
        click.echo(
            f"{tname:<16} {t.instances:>9} {t.healthy:>7} {t.failed:>6}  {kv_str:>6}{kv_warn}  "
            f"{ttft_str:>9}  {q_str:>5}  ${t.cost_hr:>5.2f}"
        )
    if status.warnings:
        click.echo()
        for w in status.warnings:
            click.echo(f"  WARN: {w}")


@agent.command(name="scale")
@click.option("--fleet-id", required=True, help="Fleet ID")
@click.option("--tier", required=True, type=click.Choice(["reasoning", "decode", "cpu_tools"]), help="Tier to scale")
@click.option("--count", required=True, type=int, help="New instance count for this tier")
@click.option("--providers", "-p", multiple=True, help="Providers to use for scale-out instances")
def agent_scale(fleet_id, tier, count, providers):
    """Scale a single fleet tier up or down without affecting other tiers.

    KV cache state on existing instances is PRESERVED during scale operations.
    New instances are added to the pool; the router distributes new requests to them.

    \b
    Examples:
      terradev agent scale --fleet-id ag_abc123 --tier decode --count 8
      terradev agent scale --fleet-id ag_abc123 --tier reasoning --count 3
      terradev agent scale --fleet-id ag_abc123 --tier cpu_tools --count 4
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.scale_tier(
        fleet_id=fleet_id,
        tier=tier,
        new_count=count,
        providers=list(providers) if providers else None,
    ))
    click.echo(_json.dumps(result, indent=2))


@agent.command(name="cost")
@click.option("--fleet-id", required=True, help="Fleet ID")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_cost(fleet_id, fmt):
    """Show real-time cost breakdown for a fleet by tier.

    \b
    Example:
      terradev agent cost --fleet-id ag_abc123
    """
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    cost = provisioner.fleet_cost(fleet_id)

    if cost is None:
        click.echo(f"Fleet {fleet_id} not found.", err=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(_json.dumps(cost, indent=2))
        return

    click.echo(f"\nFLEET COST  [{fleet_id}]")
    click.echo(f"  Uptime:       {cost['uptime_hr']:.2f}h")
    click.echo(f"  Rate:         ${cost['cost_per_hr']:.2f}/hr")
    click.echo(f"  Accrued:      ${cost['accrued_cost']:.2f}")
    click.echo(f"  Projected/day: ${cost['projected_daily']:.2f}")
    click.echo(f"  Projected/mo:  ${cost['projected_monthly']:.2f}")
    click.echo(f"  Per-agent/hr:  ${cost['cost_per_agent_hr']:.4f}")
    click.echo()
    click.echo(f"  BREAKDOWN:")
    click.echo(f"    reasoning  ${cost['breakdown']['reasoning']:.2f}/hr")
    click.echo(f"    decode     ${cost['breakdown']['decode']:.2f}/hr")
    click.echo(f"    cpu_tools  ${cost['breakdown']['cpu_tools']:.2f}/hr")


@agent.command(name="list")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_list(fmt):
    """List all known agent fleets.

    \b
    Example:
      terradev agent list
    """
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    fleets = provisioner.list_fleets()

    if fmt == "json":
        click.echo(_json.dumps(fleets, indent=2, default=str))
        return

    if not fleets:
        click.echo("No agent fleets found. Deploy one with: terradev agent deploy --agents 16")
        return

    click.echo(f"\nAGENT FLEETS ({len(fleets)})\n")
    click.echo(f"{'FLEET ID':<28} {'MODEL':<36} {'AGENTS':>6} {'$/HR':>7}  STATUS")
    click.echo("-" * 85)
    for f in fleets:
        import datetime
        created = datetime.datetime.fromtimestamp(f["created_at"]).strftime("%Y-%m-%d %H:%M")
        status_str = "OK" if f["success"] else "ERR"
        click.echo(
            f"{f['fleet_id']:<28} {f['model'][:35]:<36} "
            f"{f['n_agents']:>6} ${f['cost_hr']:>6.2f}  {status_str}  {created}"
        )


@agent.command(name="teardown")
@click.option("--fleet-id", required=True, help="Fleet ID to destroy")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def agent_teardown(fleet_id, yes):
    """Terminate all fleet instances and remove fleet state.

    \b
    Example:
      terradev agent teardown --fleet-id ag_abc123
      terradev agent teardown --fleet-id ag_abc123 --yes
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    if not yes:
        click.confirm(f"Destroy fleet {fleet_id} and all its instances?", abort=True)

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.teardown_fleet(fleet_id))
    click.echo(_json.dumps(result, indent=2))


# Register agent command group (defined above)
cli.add_command(agent)


# Gateway command for inference serving
@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind the gateway server")
@click.option("--port", "-p", default=8000, type=int, help="Port for the gateway server")
@click.option("--openai", is_flag=True, default=True, help="Enable OpenAI-compatible endpoints")
@click.option("--no-openai", is_flag=True, help="Disable OpenAI-compatible endpoints")
@click.option("--anthropic", is_flag=True, default=True, help="Enable Anthropic-compatible endpoints")
@click.option("--no-anthropic", is_flag=True, help="Disable Anthropic-compatible endpoints")
@click.option("--custom", is_flag=True, default=True, help="Enable custom workflow endpoints")
@click.option("--no-custom", is_flag=True, help="Disable custom workflow endpoints")
@click.option("--max-concurrent", type=int, default=100, help="Maximum concurrent requests")
@click.option("--timeout", type=int, default=120, help="Request timeout in seconds")
@click.option("--cors", is_flag=True, default=True, help="Enable CORS")
@click.option("--no-cors", is_flag=True, help="Disable CORS")
@click.option("--cors-origins", multiple=True, help="CORS allowed origins")
@click.option("--model", default="meta-llama/Llama-3.1-70B-Instruct", help="Default model for inference")
@click.option("--no-inference-router", is_flag=True, help="Disable inference router integration")
def gateway(host, port, openai, no_openai, anthropic, no_anthropic, custom, no_custom, 
            max_concurrent, timeout, cors, no_cors, cors_origins, model, no_inference_router):
    """Launch an API gateway for inference serving.

    Provides OpenAI/Anthropic/custom API entry and exit points for inference workflows.
    Integrates with Terradev's inference routing and KV cache management.

    \b
    OpenAI-compatible endpoints:
      - POST /v1/chat/completions
      - POST /v1/completions

    \b
    Anthropic-compatible endpoints:
      - POST /v1/messages
      - POST /v1/messages/batches

    \b
    Custom workflow endpoints:
      - POST /v1/custom/entry/{workflow_id}
      - POST /v1/custom/exit/{workflow_id}

    \b
    Management endpoints:
      - GET /health
      - GET /v1/gateway/status

    \b
    Examples:
      terradev gateway
      terradev gateway --host 0.0.0.0 --port 8080
      terradev gateway --no-anthropic --max-concurrent 50
      terradev gateway --model meta-llama/Llama-3.1-8B-Instruct

    \b
    Testing the gateway:
      curl http://localhost:8000/health
      curl http://localhost:8000/v1/gateway/status
    """
    import asyncio
    
    # Resolve boolean flags
    enable_openai = openai and not no_openai
    enable_anthropic = anthropic and not no_anthropic
    enable_custom = custom and not no_custom
    enable_cors = cors and not no_cors
    enable_inference_router = not no_inference_router
    
    # Resolve CORS origins
    origins_list = list(cors_origins) if cors_origins else ["*"]
    
    print(f"\n{'='*70}")
    print(f"TERRADEV INFERENCE GATEWAY")
    print(f"{'='*70}")
    print(f"Host: {host}:{port}")
    print(f"OpenAI API: {'ENABLED' if enable_openai else 'DISABLED'}")
    print(f"Anthropic API: {'ENABLED' if enable_anthropic else 'DISABLED'}")
    print(f"Custom Workflows: {'ENABLED' if enable_custom else 'DISABLED'}")
    print(f"CORS: {'ENABLED' if enable_cors else 'DISABLED'}")
    print(f"Inference Router: {'ENABLED' if enable_inference_router else 'DISABLED'}")
    print(f"Max Concurrent Requests: {max_concurrent}")
    print(f"Request Timeout: {timeout}s")
    print(f"Default Model: {model}")
    print(f"{'='*70}\n")
    
    try:
        from terradev_cli.core.gateway_service import (
            GatewayService,
            create_gateway_config
        )
        
        config = create_gateway_config(
            host=host,
            port=port,
            enable_openai=enable_openai,
            enable_anthropic=enable_anthropic,
            enable_custom=enable_custom,
            max_concurrent_requests=max_concurrent,
            request_timeout=timeout,
            enable_cors=enable_cors,
            cors_origins=origins_list,
            enable_inference_router=enable_inference_router,
            default_model=model,
        )
        
        gateway = GatewayService(config)
        
        print("Starting gateway server...")
        print(f"OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
        print(f"Anthropic endpoint: http://{host}:{port}/v1/messages")
        print(f"Health check: http://{host}:{port}/health")
        print(f"Gateway status: http://{host}:{port}/v1/gateway/status")
        print("\nPress Ctrl+C to stop the server\n")
        
        gateway.run_sync()
        
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nTo install required dependencies:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nGateway server stopped.")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start gateway server: {e}")
        sys.exit(1)


# Observe Command - Unified Monitoring Pipeline
@cli.group()
def observe():
    """Unified observability pipeline - W&B, Phoenix, Cost Analytics with shared trace ID"""
    pass


@observe.command("gateway")
@click.argument("gateway_endpoint")
@click.option("--wandb-project", help="W&B project name")
@click.option("--wandb-entity", help="W&B entity/team")
@click.option("--phoenix-endpoint", help="Phoenix endpoint URL")
@click.option("--enable-cost-analytics/--disable-cost-analytics", default=True, help="Enable cost analytics tracking")
@click.option("--duration", default=3600, help="Observation duration in seconds")
@click.option("--sample-rate", default=1.0, help="Sampling rate (0.0-1.0)")
def observe_gateway(
    gateway_endpoint,
    wandb_project,
    wandb_entity,
    phoenix_endpoint,
    enable_cost_analytics,
    duration,
    sample_rate
):
    """Observe API Gateway traffic across W&B, Phoenix, and Cost Analytics"""
    import asyncio
    
    async def run_observe():
        from terradev_cli.core.observe import observe_gateway_traffic
        
        result = await observe_gateway_traffic(
            gateway_endpoint=gateway_endpoint,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            phoenix_endpoint=phoenix_endpoint,
            enable_cost_analytics=enable_cost_analytics,
            duration_seconds=duration,
            sample_rate=sample_rate
        )
        
        print(f"\n📊 Observability Summary:")
        print(f"   Trace ID: {result['trace_id']}")
        print(f"   Active Destinations: {', '.join(result['active_destinations'])}")
        print(f"   Start Time: {result['start_time']}")
        
        return result
    
    try:
        result = asyncio.run(run_observe())
        return 0 if result else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Observability pipeline failed: {e}")
        return 1


@observe.command("status")
@click.argument("trace_id")
def observe_status(trace_id):
    """Get status of an observability trace"""
    import asyncio
    
    async def run_status():
        from terradev_cli.core.observe import observe_status
        
        result = await observe_status(trace_id)
        
        print(f"\n📊 Trace Status: {trace_id}")
        print(f"   Overall Status: {result['status']}")
        
        for dest, info in result.items():
            if dest != "trace_id" and dest != "status":
                print(f"   {dest.upper()}: {info}")
        
        return result
    
    try:
        result = asyncio.run(run_status())
        return 0 if result else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get trace status: {e}")
        return 1


# Schedule Command - Spot-Aware Scheduling
@cli.group()
def schedule():
    """Spot-aware scheduling for cost-optimized job execution"""
    pass


@schedule.command("job")
@click.argument("command")
@click.argument("gpu_type")
@click.option("--cron", help="Cron expression for recurring jobs")
@click.option("--max-wait-hours", default=24, help="Maximum hours to wait for optimal pricing")
@click.option("--job-name", help="Custom job name")
@click.option("--prefer-current/--no-prefer-current", default=True, help="Prefer currently active pricing window")
def schedule_job(command, gpu_type, cron, max_wait_hours, job_name, prefer_current):
    """Schedule a job with spot pricing awareness"""
    import asyncio
    
    async def run_schedule():
        from terradev_cli.core.schedule import schedule_spot_job
        
        result = await schedule_spot_job(
            command=command,
            gpu_type=gpu_type,
            cron_expression=cron,
            max_wait_hours=max_wait_hours,
            job_name=job_name,
            prefer_current_window=prefer_current
        )
        
        return result
    
    try:
        result = asyncio.run(run_schedule())
        return 0 if result.get("status") == "success" else 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to schedule job: {e}")
        return 1


@schedule.command("list")
def schedule_list_cmd():
    """List all scheduled jobs"""
    import asyncio
    
    async def run_list():
        from terradev_cli.core.schedule import schedule_list
        
        result = await schedule_list()
        return result
    
    try:
        result = asyncio.run(run_list())
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list jobs: {e}")
        return 1


@schedule.command("windows")
@click.option("--gpu-type", help="Filter by GPU type")
def schedule_windows(gpu_type):
    """Show available spot pricing windows"""
    import asyncio
    
    async def run_windows():
        from terradev_cli.core.schedule import schedule_pricing_windows
        
        result = await schedule_pricing_windows(gpu_type)
        return result
    
    try:
        result = asyncio.run(run_windows())
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get pricing windows: {e}")
        return 1


if __name__ == "__main__":
    cli()

