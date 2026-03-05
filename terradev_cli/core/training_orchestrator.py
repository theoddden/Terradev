#!/usr/bin/env python3
"""
Training Job Orchestrator — Launch, manage, and recover distributed training.

Design: zero external dependencies by default. Builds the right hostfile + env +
launch command and delegates to native tools (torchrun, deepspeed, accelerate).
Optional Ray backend behind the same interface when available.

DAG structure:
    Wave 0: preflight ∥ topology_detect               (independent)
    Wave 1: build_launch_artifacts                     (depends on topo)
    Wave 2: launch_training                            (depends on all)

All output is structured JSON for MCP consumption.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .dag_executor import DAGExecutor
from .job_state_manager import JobStateManager, JobStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Declarative training job — can come from YAML, dict, or CLI args."""
    name: str = "training-job"
    framework: str = "torchrun"  # torchrun | deepspeed | accelerate | megatron
    backend: str = "native"      # native | ray  (native = zero deps, ray = optional)
    script: str = "train.py"
    script_args: List[str] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)  # empty = localhost
    gpus_per_node: int = 8
    ssh_user: str = "root"
    ssh_key: str = ""
    # Parallelism
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 0  # 0 = auto
    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    nccl_env: Dict[str, str] = field(default_factory=dict)
    # Data
    data_path: str = ""
    # Checkpointing
    checkpoint_dir: str = ""
    checkpoint_interval_steps: int = 1000
    # Training
    total_steps: int = 0
    # Monitoring hooks (optional — empty means disabled)
    log_path: str = ""
    wandb_project: str = ""      # optional W&B hook
    prometheus_port: int = 0     # optional prometheus push gateway
    # Resume
    resume_from_checkpoint: str = ""
    # Elastic
    max_restarts: int = 3
    rdzv_port: int = 29400
    # FlashOptim (auto-applied when beneficial — user never needs to set these)
    flashoptim: str = "auto"  # auto | on | off
    flashoptim_optimizer: str = "adamw"  # adamw | adam | sgd | sgdw | lion
    flashoptim_master_weight_bits: int = 24  # 24 (default) | 32 | 0 (=None)
    flashoptim_compress_checkpoints: bool = False
    flashoptim_gradient_release: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# FlashOptim auto-detection (pure function — no side effects)
# ---------------------------------------------------------------------------

def _flashoptim_auto_config(
    config: TrainingConfig,
    topology: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide whether FlashOptim should be auto-injected into this training job.

    Returns a dict with:
      - "enabled": bool
      - "reason": str  (why it was enabled/disabled — for structured JSON output)
      - "optimizer_class": str  (e.g. "FlashAdamW")
      - "env_vars": dict  (env vars to inject)
      - "pip_install": str  (pip install command, empty if not needed)
      - "script_args_hint": list  (suggested --optimizer args for the training script)

    Decision rules (conservative — only enable when clearly beneficial):
      1. OFF if user explicitly set flashoptim="off"
      2. OFF if framework is "megatron" (Megatron has its own fused optimizer)
      3. OFF if no NVIDIA GPUs detected in topology
      4. OFF if all GPUs have <24GB VRAM (too small — overhead not worth it)
      5. ON  if user explicitly set flashoptim="on"
      6. ON  if model is being finetuned in bf16/fp16 (detected from script_args)
      7. ON  if total GPU memory across all GPUs >= 40GB (i.e., serious training)
      8. OFF otherwise (default conservative — don't inject into tiny test jobs)
    """
    result = {
        "enabled": False,
        "reason": "",
        "optimizer_class": "",
        "env_vars": {},
        "pip_install": "",
        "script_args_hint": [],
    }

    # Rule 1: explicit off
    if config.flashoptim == "off":
        result["reason"] = "disabled by user (flashoptim=off)"
        return result

    # Rule 2: Megatron has its own fused optimizer path
    if config.framework == "megatron":
        result["reason"] = "skipped: Megatron uses built-in fused optimizer"
        return result

    # Gather GPU info from topology
    nodes = topology.get("nodes", {})
    all_gpus = []
    for node_info in nodes.values():
        all_gpus.extend(node_info.get("gpus", []))

    # Rule 3: no GPUs
    if not all_gpus:
        result["reason"] = "skipped: no NVIDIA GPUs detected"
        return result

    min_vram_mb = min((g.get("memory_mb", 0) for g in all_gpus), default=0)
    total_vram_mb = sum(g.get("memory_mb", 0) for g in all_gpus)

    # Rule 4: tiny GPUs (< 24GB)
    if min_vram_mb < 24000 and config.flashoptim != "on":
        result["reason"] = f"skipped: smallest GPU has {min_vram_mb:.0f}MB VRAM (<24GB)"
        return result

    # Detect training precision from script args
    args_str = " ".join(config.script_args).lower()
    uses_reduced_precision = any(kw in args_str for kw in [
        "bf16", "bfloat16", "fp16", "float16", "--bf16", "--fp16",
        "mixed_precision", "--mixed-precision", "half",
    ])

    # Detect if user is already specifying an optimizer
    user_has_optimizer = any(kw in args_str for kw in [
        "--optimizer", "--optim ", "--optim=",
    ])

    # Map config to FlashOptim class name
    optimizer_map = {
        "adamw": "FlashAdamW",
        "adam": "FlashAdam",
        "sgd": "FlashSGD",
        "sgdw": "FlashSGDW",
        "lion": "FlashLion",
    }
    flash_class = optimizer_map.get(config.flashoptim_optimizer, "FlashAdamW")

    master_bits = config.flashoptim_master_weight_bits if config.flashoptim_master_weight_bits != 0 else None

    # Rule 5: explicit on
    if config.flashoptim == "on":
        result["enabled"] = True
        result["reason"] = "enabled by user (flashoptim=on)"
    # Rule 6: reduced precision training detected
    elif uses_reduced_precision:
        result["enabled"] = True
        result["reason"] = (
            f"auto-enabled: reduced-precision training detected in script args "
            f"(~57% optimizer memory savings with {flash_class})"
        )
    # Rule 7: large enough job to benefit
    elif total_vram_mb >= 40000:
        result["enabled"] = True
        result["reason"] = (
            f"auto-enabled: {total_vram_mb / 1000:.0f}GB total GPU memory — "
            f"FlashOptim saves ~35% peak memory"
        )
    else:
        result["reason"] = "skipped: job too small to benefit meaningfully"
        return result

    # Build injection payload
    result["optimizer_class"] = flash_class
    result["pip_install"] = "pip install -q flashoptim 2>/dev/null || true"

    # Env vars that training scripts can read to auto-configure
    result["env_vars"] = {
        "FLASHOPTIM_ENABLED": "1",
        "FLASHOPTIM_OPTIMIZER": flash_class,
        "FLASHOPTIM_MASTER_WEIGHT_BITS": str(master_bits) if master_bits else "",
        "FLASHOPTIM_COMPRESS_CHECKPOINTS": "1" if config.flashoptim_compress_checkpoints else "0",
        "FLASHOPTIM_GRADIENT_RELEASE": "1" if config.flashoptim_gradient_release else "0",
    }

    # Hint args that can be passed to training scripts that support --optim
    if not user_has_optimizer:
        result["script_args_hint"] = [
            f"# FlashOptim: replace your optimizer with {flash_class}",
            f"# from flashoptim import {flash_class}, cast_model",
        ]

    return result


# ---------------------------------------------------------------------------
# SSH helper (shared)
# ---------------------------------------------------------------------------

def _run_on(host: Optional[str], cmd: str, user: str = "root",
            key: Optional[str] = None, timeout: int = 60) -> Tuple[int, str, str]:
    if host and host not in ("localhost", "127.0.0.1"):
        ssh = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if key:
            ssh.extend(["-i", key])
        ssh.extend([f"{user}@{host}", cmd])
        try:
            r = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except Exception as e:
            return -1, "", str(e)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Launch command builders (pure functions — no side effects)
# ---------------------------------------------------------------------------

def _build_torchrun_cmd(config: TrainingConfig, n_nodes: int,
                        master_addr: str) -> List[str]:
    parts = [
        "torchrun",
        f"--nnodes={n_nodes}",
        f"--nproc_per_node={config.gpus_per_node}",
        f"--master_addr={master_addr}",
        "--master_port=29500",
    ]
    if n_nodes > 1:
        parts.extend([
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{config.rdzv_port}",
            f"--max_restarts={config.max_restarts}",
        ])
    parts.append(config.script)
    parts.extend(config.script_args)
    return parts


def _build_deepspeed_cmd(config: TrainingConfig, n_nodes: int,
                         master_addr: str, hostfile: str) -> List[str]:
    if n_nodes > 1:
        parts = [
            "deepspeed",
            f"--hostfile={hostfile}",
            f"--num_gpus={config.gpus_per_node}",
            f"--num_nodes={n_nodes}",
            f"--master_addr={master_addr}",
            "--master_port=29500",
        ]
    else:
        parts = [
            "deepspeed",
            f"--num_gpus={config.gpus_per_node}",
        ]
    parts.append(config.script)
    parts.extend(config.script_args)
    return parts


def _build_accelerate_cmd(config: TrainingConfig, n_nodes: int,
                          total_gpus: int, master_addr: str) -> List[str]:
    parts = [
        "accelerate", "launch",
        f"--num_processes={total_gpus}",
        f"--num_machines={n_nodes}",
        f"--main_process_ip={master_addr}",
        "--main_process_port=29500",
    ]
    if config.tp_size > 1:
        parts.extend(["--use_fsdp",
                       f"--fsdp_sharding_strategy=HYBRID_SHARD_{config.tp_size}"])
    parts.append(config.script)
    parts.extend(config.script_args)
    return parts


def _build_megatron_cmd(config: TrainingConfig, n_nodes: int,
                        master_addr: str) -> List[str]:
    parts = [
        "torchrun",
        f"--nnodes={n_nodes}",
        f"--nproc_per_node={config.gpus_per_node}",
        f"--master_addr={master_addr}",
        "--master_port=29500",
        config.script,
        f"--tensor-model-parallel-size={config.tp_size}",
        f"--pipeline-model-parallel-size={config.pp_size}",
    ]
    parts.extend(config.script_args)
    return parts


# ---------------------------------------------------------------------------
# DAG node functions
# ---------------------------------------------------------------------------

def _do_preflight(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Run preflight validation (imports inline so it's optional)."""
    config = ctx.get("config")
    if not isinstance(config, TrainingConfig):
        return {"passed": True, "skipped": True}
    from .preflight_validator import PreflightValidator
    validator = PreflightValidator(
        nodes=config.nodes or [None],
        ssh_user=config.ssh_user,
        ssh_key=config.ssh_key or None,
    )
    return validator.run_quick().summary()


def _detect_topology(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Detect GPU count/type on all nodes in parallel."""
    config = ctx.get("config")
    if not isinstance(config, TrainingConfig):
        return {"nodes": {}}

    node_list = config.nodes or [None]
    if len(node_list) <= 1:
        # Single node — just run locally, no need for DAG overhead
        rc, stdout, _ = _run_on(
            node_list[0],
            "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits",
            config.ssh_user, config.ssh_key or None)
        gpus = []
        if rc == 0:
            for line in stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({"index": int(parts[0]), "name": parts[1],
                                 "memory_mb": float(parts[2])})
        node_key = node_list[0] or "localhost"
        return {"nodes": {node_key: {"gpus": gpus, "count": len(gpus)}}}

    # Multi-node: DAG-parallel topology detection
    topo_dag = DAGExecutor(max_workers=len(node_list), name="topo_detect")
    for i, node in enumerate(node_list):
        def make_fn(h):
            def fn(_ctx):
                rc, stdout, _ = _run_on(
                    h, "nvidia-smi --query-gpu=index,name,memory.total "
                       "--format=csv,noheader,nounits",
                    config.ssh_user, config.ssh_key or None)
                gpus = []
                if rc == 0:
                    for line in stdout.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            gpus.append({"index": int(parts[0]), "name": parts[1],
                                         "memory_mb": float(parts[2])})
                return {"node": h or "localhost", "gpus": gpus, "count": len(gpus)}
            return fn
        topo_dag.add_node(f"topo_{i}", make_fn(node))

    result = topo_dag.apply()
    nodes = {}
    for val in result.outputs.values():
        if isinstance(val, dict) and "node" in val:
            nodes[val["node"]] = val
    return {"nodes": nodes}


def _build_artifacts(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Build hostfile, env, launch command. Pure — no remote calls."""
    deps = ctx.get("__deps__", {})
    config = ctx.get("config")
    if not isinstance(config, TrainingConfig):
        return {"error": "No config"}

    topology = deps.get("topology_detect", {})
    n_nodes = max(len(config.nodes), 1)
    total_gpus = n_nodes * config.gpus_per_node
    master_addr = config.nodes[0] if config.nodes else "localhost"

    # Hostfile (only needed for multi-node deepspeed)
    hostfile_path = ""
    if n_nodes > 1:
        hostfile_path = f"/tmp/.terradev_hostfile_{os.getpid()}"
        hostfile_content = "\n".join(
            f"{node} slots={config.gpus_per_node}"
            for node in config.nodes
        )
    else:
        hostfile_content = ""

    # Build launch command
    if config.framework == "deepspeed":
        cmd_parts = _build_deepspeed_cmd(config, n_nodes, master_addr, hostfile_path)
    elif config.framework == "torchrun":
        cmd_parts = _build_torchrun_cmd(config, n_nodes, master_addr)
    elif config.framework == "accelerate":
        cmd_parts = _build_accelerate_cmd(config, n_nodes, total_gpus, master_addr)
    elif config.framework == "megatron":
        cmd_parts = _build_megatron_cmd(config, n_nodes, master_addr)
    else:
        return {"error": f"Unknown framework: {config.framework}"}

    # Environment variables
    env = {}
    env.update(config.nccl_env)
    env.update(config.env_vars)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = "29500"
    env["WORLD_SIZE"] = str(total_gpus)
    # NCCL defaults for multi-node
    if n_nodes > 1:
        env.setdefault("NCCL_IB_DISABLE", "0")
        env.setdefault("NCCL_DEBUG", "WARN")

    # ── FlashOptim auto-injection (silent — like KV offloading for training) ──
    flashoptim_info = _flashoptim_auto_config(config, topology)
    if flashoptim_info["enabled"]:
        env.update(flashoptim_info["env_vars"])
        logger.info(f"FlashOptim: {flashoptim_info['reason']}")
    else:
        logger.debug(f"FlashOptim: {flashoptim_info['reason']}")

    return {
        "cmd_parts": cmd_parts,
        "env": env,
        "hostfile_path": hostfile_path,
        "hostfile_content": hostfile_content,
        "total_gpus": total_gpus,
        "n_nodes": n_nodes,
        "master_addr": master_addr,
        "topology": topology,
        "flashoptim": flashoptim_info,
    }


def _launch_native(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Launch via subprocess (zero deps — the default path)."""
    deps = ctx.get("__deps__", {})
    artifacts = deps.get("build_artifacts", {})
    config = ctx.get("config")
    if not isinstance(config, TrainingConfig) or "error" in artifacts:
        return {"status": "failed", "error": artifacts.get("error", "No config")}

    # Write hostfile if needed
    hf_path = artifacts.get("hostfile_path", "")
    if hf_path and artifacts.get("hostfile_content"):
        with open(hf_path, "w") as f:
            f.write(artifacts["hostfile_content"])

    # Build environment
    env = os.environ.copy()
    env.update(artifacts.get("env", {}))

    # Build full command
    cmd = " ".join(artifacts["cmd_parts"])
    if config.log_path:
        os.makedirs(os.path.dirname(config.log_path) or ".", exist_ok=True)
        cmd += f" 2>&1 | tee {config.log_path}"

    # Pre-install FlashOptim if auto-enabled (non-blocking, fail-safe)
    flashoptim_info = artifacts.get("flashoptim", {})
    if flashoptim_info.get("enabled") and flashoptim_info.get("pip_install"):
        pip_cmd = flashoptim_info["pip_install"]
        logger.info(f"FlashOptim pre-install: {pip_cmd}")
        node_list = config.nodes or [None]
        for node in node_list:
            _run_on(node, pip_cmd, config.ssh_user, config.ssh_key or None, timeout=120)

    logger.info(f"Launching: {cmd}")
    try:
        process = subprocess.Popen(
            cmd, shell=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return {
            "status": "launched",
            "pid": process.pid,
            "command": cmd,
            "total_gpus": artifacts.get("total_gpus", 0),
            "framework": config.framework,
            "backend": "native",
            "master_addr": artifacts.get("master_addr", "localhost"),
            "flashoptim": flashoptim_info,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _launch_ray(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Launch via Ray (optional — only if ray is installed)."""
    deps = ctx.get("__deps__", {})
    artifacts = deps.get("build_artifacts", {})
    config = ctx.get("config")
    if not isinstance(config, TrainingConfig):
        return {"status": "failed", "error": "No config"}

    try:
        from ..ml_services.ray_service import RayService
        ray_svc = RayService()
        # Submit as Ray job
        result = ray_svc.submit_job(
            entrypoint=" ".join(artifacts.get("cmd_parts", [])),
            runtime_env={"env_vars": artifacts.get("env", {})},
        )
        return {
            "status": "launched",
            "ray_job_id": result.get("job_id", ""),
            "backend": "ray",
            "total_gpus": artifacts.get("total_gpus", 0),
            "framework": config.framework,
        }
    except ImportError:
        logger.warning("Ray not available, falling back to native launch")
        return _launch_native(ctx)
    except Exception as e:
        logger.warning(f"Ray launch failed ({e}), falling back to native")
        return _launch_native(ctx)


# ---------------------------------------------------------------------------
# TrainingOrchestrator
# ---------------------------------------------------------------------------

class TrainingOrchestrator:
    """
    High-level training job orchestrator.

    Zero external deps by default — uses torchrun/deepspeed native.
    Set config.backend = "ray" to use Ray when available (falls back to native).

    Usage:
        orch = TrainingOrchestrator()
        result = orch.launch(TrainingConfig(script="train.py", framework="torchrun"))
        print(json.dumps(result, indent=2))
    """

    def __init__(self, state_manager: Optional[JobStateManager] = None):
        self.state_manager = state_manager or JobStateManager()

    def launch(self, config: TrainingConfig,
               skip_preflight: bool = False) -> Dict[str, Any]:
        """Launch a training job. Returns structured JSON."""
        job = self.state_manager.create_job(
            name=config.name,
            framework=config.framework,
            config=config.to_dict(),
            nodes=config.nodes or ["localhost"],
            total_steps=config.total_steps,
        )
        self.state_manager.update_job_status(job.id, JobStatus.PREFLIGHT)

        # Build DAG
        dag = DAGExecutor(max_workers=4, name=f"launch_{job.id[:8]}")

        # Wave 0: preflight ∥ topology (independent)
        if not skip_preflight:
            dag.add_node("preflight", _do_preflight)
        dag.add_node("topology_detect", _detect_topology)

        # Wave 1: build artifacts (depends on topology)
        deps_build = {"topology_detect"}
        if not skip_preflight:
            deps_build.add("preflight")
        dag.add_node("build_artifacts", _build_artifacts, depends_on=deps_build)

        # Wave 2: launch (depends on artifacts)
        launch_fn = _launch_ray if config.backend == "ray" else _launch_native
        dag.add_node("launch", launch_fn, depends_on={"build_artifacts"})

        # Execute
        self.state_manager.update_job_status(job.id, JobStatus.LAUNCHING)
        result = dag.apply(initial_context={"config": config})

        launch_out = result.outputs.get("launch", {})
        preflight_out = result.outputs.get("preflight", {})

        if not result.success:
            self.state_manager.update_job_status(
                job.id, JobStatus.FAILED,
                error_message=json.dumps(result.errors, default=str))
            return {"job_id": job.id, "status": "failed", "errors": result.errors}

        if preflight_out and not preflight_out.get("passed", True):
            self.state_manager.update_job_status(
                job.id, JobStatus.FAILED,
                error_message="Preflight failed")
            return {"job_id": job.id, "status": "preflight_failed",
                    "preflight": preflight_out}

        self.state_manager.update_job_status(job.id, JobStatus.RUNNING)
        return {
            "job_id": job.id,
            "status": launch_out.get("status", "unknown"),
            "pid": launch_out.get("pid"),
            "framework": config.framework,
            "backend": launch_out.get("backend", "native"),
            "total_gpus": launch_out.get("total_gpus", 0),
            "nodes": config.nodes or ["localhost"],
            "master_addr": launch_out.get("master_addr"),
            "preflight": preflight_out,
            "flashoptim": launch_out.get("flashoptim", {}),
        }

    def resume(self, job_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Resume from checkpoint. Rebuilds config from job state."""
        job = self.state_manager.get_job(job_id)
        if not job:
            return {"status": "failed", "error": f"Job not found: {job_id}"}

        config = TrainingConfig.from_dict(job.config)
        config.resume_from_checkpoint = checkpoint_id or "latest"

        if "--resume" not in " ".join(config.script_args):
            if config.checkpoint_dir:
                config.script_args.extend([
                    "--resume_from_checkpoint", config.checkpoint_dir])

        return self.launch(config, skip_preflight=False)

    def stop(self, job_id: str) -> Dict[str, Any]:
        """Stop a running job — kills processes on all nodes in parallel."""
        job = self.state_manager.get_job(job_id)
        if not job:
            return {"status": "failed", "error": f"Job not found: {job_id}"}

        nodes = job.nodes if hasattr(job, "nodes") else job.config.get("nodes", ["localhost"])
        if len(nodes) <= 1:
            # Single node — no DAG overhead
            _run_on(nodes[0] if nodes[0] != "localhost" else None,
                    "pkill -f 'torchrun|deepspeed|accelerate' 2>/dev/null",
                    job.config.get("ssh_user", "root"),
                    job.config.get("ssh_key"))
        else:
            dag = DAGExecutor(max_workers=len(nodes), name=f"stop_{job_id[:8]}")
            for i, node in enumerate(nodes):
                def make_fn(h):
                    def fn(_ctx):
                        _run_on(h if h != "localhost" else None,
                                "pkill -f 'torchrun|deepspeed|accelerate' 2>/dev/null",
                                job.config.get("ssh_user", "root"),
                                job.config.get("ssh_key"))
                        return {"node": h, "stopped": True}
                    return fn
                dag.add_node(f"stop_{i}", make_fn(node))
            dag.apply()

        self.state_manager.update_job_status(job_id, JobStatus.CANCELLED)
        return {"job_id": job_id, "status": "stopped"}

    def status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get job status (single job or all running)."""
        if job_id:
            job = self.state_manager.get_job(job_id)
            return job.to_dict() if job else {"error": f"Not found: {job_id}"}
        return {
            "running": self.state_manager.running_jobs_summary(),
            "total_cost": self.state_manager.total_cost(),
        }
