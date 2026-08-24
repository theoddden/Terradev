#!/usr/bin/env python3
"""
Training Pipeline — multi-stage SFT → DPO → GRPO orchestration.

Wraps the existing TrainingOrchestrator and DAGExecutor so each stage is a
DAG node. Supports provider-aware provisioning (auto or fixed), checkpoint
handoff, and optional teardown.
"""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .training_stages import (
    PipelineConfig,
    StageConfig,
    StageType,
    _normalize_path,
    build_stage_command,
    estimate_resource,
)
from .training_orchestrator import TrainingConfig, TrainingOrchestrator
from .parallel_provisioner import ParallelProvisioner
from .job_state_manager import JobStateManager, JobStatus
from .dag_executor import DAGExecutor

logger = logging.getLogger(__name__)

# Providers to query for quotes when provider='auto' and gpu_type is given.
_TRAINING_GPU_PROVIDERS = [
    "runpod",
    "vast",
    "coreweave",
    "lambda",
    "crusoe",
    "fluidstack",
    "hyperstack",
    "tensordock",
    "lambda_labs",
    "e2enetworks",
    "alibaba",
    "digitalocean",
    "aws",
    "gcp",
    "azure",
    "oracle",
]


@dataclass
class StageResult:
    status: str = "pending"
    job_id: str = ""
    provider: str = ""
    region: str = ""
    price_hr: float = 0.0
    nodes: List[str] = field(default_factory=list)
    output_dir: str = ""
    error: str = ""
    logs: List[str] = field(default_factory=list)
    name: str = ""
    command: List[str] = field(default_factory=list)
    description: str = ""


class ProviderSelector:
    """Pick the cheapest provider/instance for a given GPU type and count."""

    def __init__(self):
        from terradev_cli.commands._api import TerradevAPI

        self.api = TerradevAPI()

    def select(
        self,
        gpu_type: str,
        count: int = 1,
        preferred_providers: Optional[List[str]] = None,
        max_price: float = 0.0,
        spot: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return a ranked list of quotes for the requested GPU type."""
        candidates = preferred_providers or _TRAINING_GPU_PROVIDERS
        quotes: List[Dict[str, Any]] = []
        for provider in candidates:
            try:
                raw = asyncio.run(self.api._get_provider_quotes(provider, gpu_type))
                for q in raw:
                    q["provider"] = q.get("provider", provider).lower().replace(" ", "_")
                    q["gpu_type"] = q.get("gpu_type", gpu_type)
                    if spot and not q.get("spot", False):
                        continue
                    if max_price and q.get("price_per_hour", 999) > max_price:
                        continue
                    quotes.append(q)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Quote failed for {provider}: {e}")

        quotes.sort(key=lambda q: q.get("price_per_hour", 999))
        return quotes

    def best_allocation(
        self,
        gpu_type: str,
        count: int,
        max_price: float = 0.0,
        spot: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build a ParallelProvisioner allocation list for the cheapest spread."""
        quotes = self.select(gpu_type, count, max_price=max_price, spot=spot)
        if not quotes:
            return []

        # Build credentials map lazily from TerradevAPI _provider_creds
        creds_map: Dict[str, Dict[str, str]] = {}
        for q in quotes:
            prov = q.get("provider", "").lower().replace(" ", "_")
            if prov not in creds_map:
                try:
                    creds_map[prov] = self.api._provider_creds(prov)
                except Exception:  # noqa: BLE001
                    creds_map[prov] = {}

        pp = ParallelProvisioner()
        return pp.build_cheapest_spread(
            quotes,
            count=count,
            max_price=max_price,
            credentials_map=creds_map,
        )


class TrainingProvisioner:
    """Provision and deprovision training nodes using ParallelProvisioner."""

    def __init__(self):
        self.pp = ParallelProvisioner()

    def provision(
        self,
        gpu_type: str,
        count: int,
        max_price: float = 0.0,
        spot: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Return (group_id, provisioned_node_info)."""
        selector = ProviderSelector()
        allocations = selector.best_allocation(gpu_type, count, max_price, spot)
        if not allocations:
            raise RuntimeError(f"No available {gpu_type} instances found")
        logger.info(f"Provisioning {count} {gpu_type} from {len(allocations)} allocation(s)")
        group_id, results = asyncio.run(self.pp.provision_parallel(allocations))
        successes = [r for r in results if r.status == "active"]
        if not successes:
            errors = [f"{r.provider}({r.region}): {r.error}" for r in results]
            raise RuntimeError(f"All provisions failed: {'; '.join(errors)}")

        # Poll for IPs
        nodes = asyncio.run(self._wait_for_ips(successes, timeout_s=600))
        return group_id, nodes

    async def _wait_for_ips(
        self,
        results: List[Any],
        timeout_s: int = 600,
        poll_s: int = 10,
    ) -> List[Dict[str, Any]]:
        """Poll provider status until each instance has a usable IP/SSH address."""
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        nodes: List[Dict[str, Any]] = []
        pending = [
            {
                "provider": r.provider,
                "region": r.region,
                "instance_id": r.instance_id,
                "gpu_type": r.gpu_type,
                "price_hr": r.price_hr,
                "spot": r.spot,
            }
            for r in results
        ]
        deadline = time.time() + timeout_s
        while pending and time.time() < deadline:
            still_pending: List[Dict[str, Any]] = []
            for info in pending:
                try:
                    creds = {}
                    try:
                        from terradev_cli.commands._api import TerradevAPI

                        api = TerradevAPI()
                        creds = api._provider_creds(info["provider"])
                    except Exception:  # noqa: BLE001
                        pass
                    provider = factory.create_provider(info["provider"], creds)
                    status = await provider.get_instance_status(info["instance_id"])
                    if status and status.get("ip_address"):
                        info["ip_address"] = status["ip_address"]
                        info["status"] = status.get("status", "running")
                        nodes.append(info)
                    elif status and status.get("public_ip"):
                        info["ip_address"] = status["public_ip"]
                        info["status"] = status.get("status", "running")
                        nodes.append(info)
                    else:
                        still_pending.append(info)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"IP poll for {info['instance_id']}: {e}")
                    still_pending.append(info)
            pending = still_pending
            if pending:
                await asyncio.sleep(poll_s)

        if pending:
            raise RuntimeError(f"Timed out waiting for IPs: {pending}")
        return nodes

    def deprovision(self, nodes: List[Dict[str, Any]]) -> None:
        """Terminate all provisioned training nodes."""
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        for info in nodes:
            try:
                creds = {}
                try:
                    from terradev_cli.commands._api import TerradevAPI

                    api = TerradevAPI()
                    creds = api._provider_creds(info["provider"])
                except Exception:  # noqa: BLE001
                    pass
                provider = factory.create_provider(info["provider"], creds)
                asyncio.run(provider.terminate_instance(info["instance_id"]))
                logger.info(f"Terminated {info['instance_id']} on {info['provider']}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Teardown failed for {info.get('instance_id')}: {e}")


class TrainingPipeline:
    """Run a multi-stage training pipeline as a DAG."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.orchestrator = TrainingOrchestrator()
        self.provisioner = TrainingProvisioner()
        self.job_manager = JobStateManager()
        self.results: List[StageResult] = []

    def plan(self) -> DAGExecutor:
        """Build the execution DAG (stages are sequential by default)."""
        dag = DAGExecutor(max_workers=4, name="training_pipeline")
        prev = None
        for i, stage in enumerate(self.config.stages):
            node_name = stage.name or f"stage_{i}"

            def make_fn(s: StageConfig, idx: int):
                def fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
                    return self._run_stage(s, idx, ctx)

                return fn

            if prev:
                dag.add_node(node_name, make_fn(stage, i), depends_on={prev})
            else:
                dag.add_node(node_name, make_fn(stage, i))
            prev = node_name
        return dag

    def run(
        self,
        dry_run: bool = False,
        nodes: Optional[List[str]] = None,
        ssh_key: str = "",
    ) -> List[StageResult]:
        """Execute the pipeline."""
        dag = self.plan()
        initial = {
            "dry_run": dry_run or self.config.dry_run,
            "results": {},
            "nodes": nodes or [],
            "ssh_key": ssh_key,
        }
        result = dag.apply(initial_context=initial, fail_fast=True)
        if result.errors:
            for name, err in result.errors.items():
                logger.error(f"Stage {name} failed: {err}")
        return self.results

    def _run_stage(self, stage: StageConfig, idx: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
        dry_run = ctx.get("dry_run", False)
        stage_result = StageResult()

        # Resolve base model / checkpoint
        base_model = stage.model
        if not base_model and stage.base_checkpoint:
            base_model = stage.base_checkpoint
        if not base_model:
            raise ValueError(f"Stage {stage.name} has no model or base_checkpoint")

        # Estimate resources
        gpu_type, total_gpus, node_count, gpus_per_node, vram_gb = estimate_resource(stage)
        if not stage.gpu_type:
            stage.gpu_type = gpu_type
        if stage.type == StageType.GRPO.value and (stage.rollout_provider == "auto" or stage.trainer_provider == "auto"):
            # For GRPO, auto means use the cheapest quote for both roles
            pass

        # Determine nodes
        nodes: List[str] = list(ctx.get("nodes", []))
        provisioned: List[Dict[str, Any]] = []
        if not nodes:
            # Auto-provision
            if dry_run:
                print(f"[dry-run] {stage.name}: would provision {node_count} nodes x {gpus_per_node} {stage.gpu_type}")
            else:
                logger.info(
                    f"Stage {stage.name}: provisioning {node_count} nodes x {gpus_per_node} "
                    f"{stage.gpu_type} (est. {vram_gb:.1f}GB needed)"
                )
                _, provisioned = self.provisioner.provision(
                    stage.gpu_type,
                    node_count,
                    max_price=stage.max_price,
                    spot=stage.spot,
                )
                nodes = [n["ip_address"] for n in provisioned]

        if not nodes:
            raise RuntimeError(f"Stage {stage.name} has no nodes to run on")

        stage_result.nodes = nodes

        ssh_key = ctx.get("ssh_key", "")

        # Build command
        cmd, env, files, description = build_stage_command(stage, base_model=base_model)
        stage_result.name = stage.name
        stage_result.command = cmd
        stage_result.description = description

        if dry_run:
            print(f"[dry-run] {stage.name}: {' '.join(cmd)}")
            stage_result.status = "dry-run"
            stage_result.output_dir = _normalize_path(stage.output_dir)
            self.results.append(stage_result)
            return stage_result.__dict__

        # Add checkpoint handoff base path if applicable
        if stage.base_checkpoint:
            env.setdefault("BASE_CHECKPOINT", _normalize_path(stage.base_checkpoint))

        # Set S3 upload target
        if stage.output_bucket:
            env.setdefault("OUTPUT_BUCKET", stage.output_bucket)
        elif self.config.checkpoint_bucket:
            env.setdefault("OUTPUT_BUCKET", os.path.join(self.config.checkpoint_bucket, stage.name))

        # Create a TrainingConfig
        if cmd and len(cmd) > 1 and cmd[0] == "torchrun":
            # torchrun prefix already injected; treat the rest as the training script/args
            script = cmd[1] if len(cmd) > 1 else cmd[0]
            script_args = cmd[2:] if len(cmd) > 2 else []
        else:
            script = cmd[0]
            script_args = cmd[1:]

        # Map the executable to a framework the orchestrator understands.
        if stage.deepspeed or cmd[0] == "deepspeed":
            framework = "deepspeed"
        elif cmd[0] == "accelerate":
            framework = "accelerate"
        elif "torchrun" in cmd[0]:
            framework = "torchrun"
        else:
            # Wrapper scripts / Python training scripts are launched with torchrun
            # so torch.distributed / LOCAL_RANK are available to child CLIs.
            framework = "torchrun"

        train_config = TrainingConfig(
            name=stage.name,
            framework=framework,
            backend="native",
            script=script,
            script_args=script_args,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            tp_size=1,
            pp_size=1,
            total_steps=0,
            env_vars=env,
            ssh_key=ssh_key,
            checkpoint_dir=_normalize_path(stage.output_dir),
            data_path=_normalize_path(stage.data),
        )

        # Launch
        launch = self.orchestrator.launch(train_config, skip_preflight=False)
        stage_result.status = launch.get("status", "unknown")
        stage_result.job_id = launch.get("job_id", "")
        stage_result.output_dir = _normalize_path(stage.output_dir)

        if stage_result.status == "failed":
            stage_result.error = launch.get("error", "")
            return stage_result.__dict__

        # Wait for completion (master process on the first node)
        final = self.orchestrator.wait_for_completion(stage_result.job_id)
        stage_result.status = final.get("status", stage_result.status)
        if final.get("status") == "failed" and final.get("error"):
            stage_result.error = final["error"]

        # Re-check job status
        try:
            metrics = self.job_manager.job_metrics(stage_result.job_id)
            stage_result.status = metrics.get("status", "unknown")
            if metrics.get("error_message"):
                stage_result.error = metrics["error_message"]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not read job metrics: {e}")

        # Optional: sync checkpoint to S3
        if stage.output_bucket or self.config.checkpoint_bucket:
            self._upload_checkpoint(stage_result.output_dir, stage, self.config.checkpoint_bucket)

        # Teardown if configured
        if self.config.teardown and provisioned:
            self.provisioner.deprovision(provisioned)

        self.results.append(stage_result)
        return stage_result.__dict__

    def _wait_for_job(self, job_id: str, timeout_s: int = 0, poll_s: int = 30) -> None:
        """Poll the job until it is no longer running."""
        if not job_id:
            return
        if timeout_s <= 0:
            # No timeout by default; user can ctrl-c
            while True:
                metrics = self.job_manager.job_metrics(job_id)
                status = metrics.get("status", "unknown")
                if status in ("completed", "failed", "cancelled", "preempted"):
                    break
                time.sleep(poll_s)
        else:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                metrics = self.job_manager.job_metrics(job_id)
                status = metrics.get("status", "unknown")
                if status in ("completed", "failed", "cancelled", "preempted"):
                    break
                time.sleep(poll_s)

    def _upload_checkpoint(self, output_dir: str, stage: StageConfig, base_bucket: str) -> None:
        """Best-effort sync of output dir to S3."""
        bucket = stage.output_bucket or (os.path.join(base_bucket, stage.name) if base_bucket else "")
        if not bucket:
            return
        try:
            cmd = ["aws", "s3", "sync", output_dir, bucket]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                logger.info(f"Synced {output_dir} -> {bucket}")
            else:
                logger.warning(f"S3 sync failed: {r.stderr}")
        except FileNotFoundError:
            logger.warning("aws CLI not found; skipping S3 checkpoint upload")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"S3 sync error: {e}")


def run_pipeline_from_yaml(
    path: str,
    dry_run: bool = False,
) -> List[StageResult]:
    """Convenience entry point for CLI."""
    config = PipelineConfig.from_yaml(path)
    if dry_run:
        config.dry_run = True
    pipeline = TrainingPipeline(config)
    return pipeline.run(dry_run=dry_run)
