#!/usr/bin/env python3
"""Training / LoRA commands for the Terradev CLI."""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from . import cli
from .inference import _resolve_provision_nodes
from .ml import _parse_vllm_endpoint
from terradev_cli.core.training_stages import (
    DPO_ALGORITHM_CHOICES,
    FRAMEWORK_CHOICES,
)

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
    cost_rate,
    fmt,
):
    """Monitor GPU utilization, training metrics, and cost.

    Default: nvidia-smi (zero deps).
    Includes straggler detection for multi-node clusters.

    Examples:
        terradev monitor -n 10.0.0.1 -n 10.0.0.2 -l /tmp/train.log
        terradev monitor --from-provision latest --cost-rate 3.50
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
@cli.group()
def lora():
    """Production-grade LoRA adapter management with registry and cross-replica consistency.

    Manage adapter versions, track replica distribution, and ensure consistency across deployments.
    """
    pass
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


# ── New multi-stage training commands (SFT / DPO / GRPO / pipeline) ─────────


@train.command("sft")
@click.option("--model", required=True, help="Base model ID or path")
@click.option("--data", required=True, help="Training data path (local dir or s3://)")
@click.option(
    "--framework",
    default="unsloth",
    type=click.Choice(FRAMEWORK_CHOICES),
    help="Training framework",
)
@click.option("--provider", default="auto", help="Cloud provider, or 'auto' for cheapest quote")
@click.option("--checkpoint", default="", help="Output checkpoint directory")
@click.option("--gpu-type", default="", help="GPU type (A100, H100, etc.)")
@click.option("--gpu-count", default=1, type=int, help="Total GPUs")
@click.option("--node-count", default=1, type=int, help="Number of nodes")
@click.option("--gpus-per-node", default=8, type=int, help="GPUs per node")
@click.option("--spot/--no-spot", default=False, help="Use spot/preemptible instances")
@click.option("--max-price", default=0.0, type=float, help="Max $/hr per GPU")
@click.option("--num-train-epochs", default=1, type=int)
@click.option("--per-device-batch-size", default=1, type=int)
@click.option("--gradient-accumulation-steps", default=4, type=int)
@click.option("--learning-rate", default=2e-4, type=float)
@click.option("--warmup-ratio", default=0.1, type=float)
@click.option("--max-seq-length", default=2048, type=int)
@click.option("--lora-rank", default=64, type=int)
@click.option("--lora-alpha", default=16, type=int)
@click.option("--from-provision", default="", help='Use provision group, or "latest"')
@click.option("--nodes", "-n", multiple=True, help="Node IP addresses")
@click.option("--output-bucket", default="", help="s3:// bucket to sync checkpoint")
@click.option("--dry-run", is_flag=True, help="Print the command but do not launch")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def train_sft(
    model,
    data,
    framework,
    provider,
    checkpoint,
    gpu_type,
    gpu_count,
    node_count,
    gpus_per_node,
    spot,
    max_price,
    num_train_epochs,
    per_device_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    warmup_ratio,
    max_seq_length,
    lora_rank,
    lora_alpha,
    from_provision,
    nodes,
    output_bucket,
    dry_run,
    fmt,
    extra,
):
    """Run supervised fine-tuning (SFT) stage."""
    from terradev_cli.core.training_pipeline import TrainingPipeline
    from terradev_cli.core.training_stages import PipelineConfig, StageConfig

    resolved_nodes = list(nodes) if nodes else []
    ssh_key = ""
    if from_provision and not resolved_nodes:
        resolved_nodes, ssh_key = _resolve_provision_nodes(from_provision, fmt)

    if not resolved_nodes:
        print("ERROR: Provide --nodes, --from-provision, or let --provider auto provision")
        sys.exit(1)

    stage = StageConfig(
        type="sft",
        model=model,
        data=data,
        checkpoint=checkpoint,
        framework=framework,
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        node_count=node_count,
        gpus_per_node=gpus_per_node,
        spot=spot,
        max_price=max_price,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        output_bucket=output_bucket,
        extra_args=list(extra),
    )
    config = PipelineConfig(name=stage.name, stages=[stage], teardown=False)
    pipeline = TrainingPipeline(config)
    if dry_run:
        result = pipeline.run(dry_run=True, nodes=resolved_nodes, ssh_key=ssh_key)
        _print_stage_result(result[0] if result else None, fmt)
        return
    result = pipeline.run(dry_run=False, nodes=resolved_nodes, ssh_key=ssh_key)
    _print_stage_result(result[0] if result else None, fmt)


@train.command("dpo")
@click.option("--base-checkpoint", required=True, help="SFT checkpoint to start from")
@click.option("--data", required=True, help="Preference pairs data path")
@click.option("--model", default="", help="Optional model override (defaults to base-checkpoint)")
@click.option(
    "--algorithm",
    default="dpo",
    type=click.Choice(DPO_ALGORITHM_CHOICES),
    help="Preference optimization algorithm",
)
@click.option(
    "--framework",
    default="trl",
    type=click.Choice(FRAMEWORK_CHOICES),
    help="Training framework",
)
@click.option("--provider", default="auto", help="Cloud provider, or 'auto'")
@click.option("--checkpoint", default="", help="Output checkpoint directory")
@click.option("--gpu-type", default="", help="GPU type (A100, H100, etc.)")
@click.option("--gpu-count", default=1, type=int)
@click.option("--node-count", default=1, type=int)
@click.option("--gpus-per-node", default=8, type=int)
@click.option("--spot/--no-spot", default=False)
@click.option("--max-price", default=0.0, type=float)
@click.option("--num-train-epochs", default=1, type=int)
@click.option("--per-device-batch-size", default=1, type=int)
@click.option("--gradient-accumulation-steps", default=4, type=int)
@click.option("--learning-rate", default=2e-4, type=float)
@click.option("--warmup-ratio", default=0.1, type=float)
@click.option("--beta", default=0.1, type=float, help="DPO beta / SimPO beta")
@click.option("--max-seq-length", default=2048, type=int)
@click.option("--lora-rank", default=64, type=int)
@click.option("--lora-alpha", default=16, type=int)
@click.option("--from-provision", default="")
@click.option("--nodes", "-n", multiple=True)
@click.option("--output-bucket", default="")
@click.option("--dry-run", is_flag=True)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def train_dpo(
    base_checkpoint,
    data,
    model,
    algorithm,
    framework,
    provider,
    checkpoint,
    gpu_type,
    gpu_count,
    node_count,
    gpus_per_node,
    spot,
    max_price,
    num_train_epochs,
    per_device_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    warmup_ratio,
    beta,
    max_seq_length,
    lora_rank,
    lora_alpha,
    from_provision,
    nodes,
    output_bucket,
    dry_run,
    fmt,
    extra,
):
    """Run preference optimization (DPO / SimPO / KTO / ORPO)."""
    from terradev_cli.core.training_pipeline import TrainingPipeline
    from terradev_cli.core.training_stages import PipelineConfig, StageConfig

    resolved_nodes = list(nodes) if nodes else []
    ssh_key = ""
    if from_provision and not resolved_nodes:
        resolved_nodes, ssh_key = _resolve_provision_nodes(from_provision, fmt)
    if not resolved_nodes:
        print("ERROR: Provide --nodes, --from-provision, or let --provider auto provision")
        sys.exit(1)

    stage = StageConfig(
        type="dpo",
        model=model or base_checkpoint,
        data=data,
        base_checkpoint=base_checkpoint,
        checkpoint=checkpoint,
        framework=framework,
        algorithm=algorithm,
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        node_count=node_count,
        gpus_per_node=gpus_per_node,
        spot=spot,
        max_price=max_price,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        beta=beta,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        output_bucket=output_bucket,
        extra_args=list(extra),
    )
    config = PipelineConfig(name=stage.name, stages=[stage], teardown=False)
    pipeline = TrainingPipeline(config)
    if dry_run:
        result = pipeline.run(dry_run=True, nodes=resolved_nodes, ssh_key=ssh_key)
        _print_stage_result(result[0] if result else None, fmt)
        return
    result = pipeline.run(dry_run=False, nodes=resolved_nodes, ssh_key=ssh_key)
    _print_stage_result(result[0] if result else None, fmt)


@train.command("grpo")
@click.option("--base-checkpoint", required=True, help="DPO/SFT checkpoint to start from")
@click.option("--data", required=True, help="Prompt / rollout data path")
@click.option("--model", default="", help="Optional model override")
@click.option("--reward-fn", default="verifiable", help="Reward function name")
@click.option("--rollout-provider", default="auto", help="Provider for rollout workers")
@click.option("--trainer-provider", default="auto", help="Provider for GRPO trainer")
@click.option(
    "--framework",
    default="openrlhf",
    type=click.Choice(FRAMEWORK_CHOICES),
    help="GRPO framework (openrlhf or trl)",
)
@click.option("--provider", default="auto", help="Combined provider override")
@click.option("--checkpoint", default="", help="Output checkpoint directory")
@click.option("--gpu-type", default="", help="GPU type for trainer")
@click.option("--gpu-count", default=8, type=int)
@click.option("--node-count", default=2, type=int)
@click.option("--gpus-per-node", default=8, type=int)
@click.option("--num-generations", default=8, type=int, help="GRPO group size")
@click.option("--spot/--no-spot", default=False)
@click.option("--max-price", default=0.0, type=float)
@click.option("--num-train-epochs", default=1, type=int)
@click.option("--per-device-batch-size", default=1, type=int)
@click.option("--gradient-accumulation-steps", default=4, type=int)
@click.option("--learning-rate", default=2e-4, type=float)
@click.option("--warmup-ratio", default=0.1, type=float)
@click.option("--max-seq-length", default=2048, type=int)
@click.option("--from-provision", default="")
@click.option("--nodes", "-n", multiple=True)
@click.option("--output-bucket", default="")
@click.option("--dry-run", is_flag=True)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def train_grpo(
    base_checkpoint,
    data,
    model,
    reward_fn,
    rollout_provider,
    trainer_provider,
    framework,
    provider,
    checkpoint,
    gpu_type,
    gpu_count,
    node_count,
    gpus_per_node,
    num_generations,
    spot,
    max_price,
    num_train_epochs,
    per_device_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    warmup_ratio,
    max_seq_length,
    from_provision,
    nodes,
    output_bucket,
    dry_run,
    fmt,
    extra,
):
    """Run GRPO / RLVR stage."""
    from terradev_cli.core.training_pipeline import TrainingPipeline
    from terradev_cli.core.training_stages import PipelineConfig, StageConfig

    resolved_nodes = list(nodes) if nodes else []
    ssh_key = ""
    if from_provision and not resolved_nodes:
        resolved_nodes, ssh_key = _resolve_provision_nodes(from_provision, fmt)
    if not resolved_nodes:
        print("ERROR: Provide --nodes, --from-provision, or let --provider auto provision")
        sys.exit(1)

    stage = StageConfig(
        type="grpo",
        model=model or base_checkpoint,
        data=data,
        base_checkpoint=base_checkpoint,
        checkpoint=checkpoint,
        framework=framework,
        reward_fn=reward_fn,
        rollout_provider=rollout_provider,
        trainer_provider=trainer_provider,
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        node_count=node_count,
        gpus_per_node=gpus_per_node,
        num_generations=num_generations,
        spot=spot,
        max_price=max_price,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        output_bucket=output_bucket,
        extra_args=list(extra),
    )
    config = PipelineConfig(name=stage.name, stages=[stage], teardown=False)
    pipeline = TrainingPipeline(config)
    if dry_run:
        result = pipeline.run(dry_run=True, nodes=resolved_nodes, ssh_key=ssh_key)
        _print_stage_result(result[0] if result else None, fmt)
        return
    result = pipeline.run(dry_run=False, nodes=resolved_nodes, ssh_key=ssh_key)
    _print_stage_result(result[0] if result else None, fmt)


@train.command("pipeline")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Pipeline YAML")
@click.option("--dry-run", is_flag=True, help="Print the DAG plan without launching")
@click.option("--teardown", is_flag=True, help="Tear down provisioned nodes after each stage")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def train_pipeline(config, dry_run, teardown, fmt):
    """Run a multi-stage training pipeline from a YAML file."""
    from terradev_cli.core.training_pipeline import run_pipeline_from_yaml

    if dry_run:
        from terradev_cli.core.training_stages import PipelineConfig
        from terradev_cli.core.dag_executor import DAGExecutor

        pc = PipelineConfig.from_yaml(config)
        dag = DAGExecutor(max_workers=4, name="training_pipeline_dry_run")
        prev = None
        for i, stage in enumerate(pc.stages):
            name = stage.name or f"stage_{i}"
            if prev:
                dag.add_node(name, lambda _ctx: stage.__dict__, depends_on={prev})
            else:
                dag.add_node(name, lambda _ctx: stage.__dict__)
            prev = name
        print(json.dumps(dag.describe(), indent=2, default=str))
        return

    results = run_pipeline_from_yaml(config, dry_run=False)
    if fmt == "json":
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        print(f"\nPipeline finished: {len(results)} stage(s)")
        for r in results:
            print(f"  {r.name or 'stage'}: {r.status}  job={r.job_id}  nodes={len(r.nodes)}")


def _print_stage_result(result, fmt: str):
    if result is None:
        print("ERROR: stage produced no result")
        sys.exit(1)
    if fmt == "json":
        print(json.dumps(result.__dict__, indent=2, default=str))
    else:
        print(f"\nTraining Stage: {result.name}")
        print(f"  Status: {result.status}")
        print(f"  Job ID: {result.job_id}")
        print(f"  Output: {result.output_dir}")
        if result.description:
            print(f"  Description: {result.description}")
        if result.command:
            print(f"  Command: {' '.join(str(c) for c in result.command)}")
        if result.error:
            print(f"  Error: {result.error}")
        print()
