#!/usr/bin/env python3
"""
Training Monitor — Unified GPU + training metrics + cost view.

Default: nvidia-smi (zero deps). Hooks for:
  - Prometheus/DCGM-exporter (if endpoint configured)
  - W&B (if wandb_run configured)
  - Custom callbacks (for any other sink)

DAG per snapshot:
    Wave 0: gpu_node_0 ∥ gpu_node_1 ∥ ... ∥ training_log ∥ cost
    Wave 1: aggregate + straggler_detect  (depends on all)

All output is structured JSON for MCP consumption.
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dag_executor import DAGExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GPUMetric:
    node: str
    gpu_index: int
    gpu_name: str
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_w: float
    power_limit_w: float


@dataclass
class TrainingMetrics:
    step: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    step_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    epoch: float = 0.0


@dataclass
class StragglerInfo:
    """Detected straggler nodes — GPUs with significantly lower utilization."""
    detected: bool = False
    slow_nodes: List[str] = field(default_factory=list)
    util_spread_pct: float = 0.0  # max - min util across nodes
    message: str = ""


@dataclass
class MonitorSnapshot:
    """Complete monitoring snapshot — structured JSON for MCP."""
    timestamp: datetime = field(default_factory=datetime.now)
    job_id: str = ""
    gpus: List[GPUMetric] = field(default_factory=list)
    training: Optional[TrainingMetrics] = None
    straggler: Optional[StragglerInfo] = None
    cost_usd: float = 0.0
    gpu_hours: float = 0.0
    elapsed_hours: float = 0.0
    avg_gpu_util: float = 0.0
    avg_gpu_memory_pct: float = 0.0
    total_gpu_power_w: float = 0.0
    node_count: int = 0
    gpu_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp.isoformat(),
            "job_id": self.job_id,
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "avg_gpu_util": round(self.avg_gpu_util, 1),
            "avg_gpu_memory_pct": round(self.avg_gpu_memory_pct, 1),
            "total_gpu_power_w": round(self.total_gpu_power_w, 1),
            "cost_usd": round(self.cost_usd, 4),
            "gpu_hours": round(self.gpu_hours, 2),
            "elapsed_hours": round(self.elapsed_hours, 3),
        }
        if self.training:
            d["training"] = {
                "step": self.training.step,
                "loss": self.training.loss,
                "grad_norm": self.training.grad_norm,
                "learning_rate": self.training.learning_rate,
                "step_time_ms": self.training.step_time_ms,
                "tokens_per_sec": self.training.tokens_per_sec,
                "samples_per_sec": self.training.samples_per_sec,
            }
        if self.straggler and self.straggler.detected:
            d["straggler"] = {
                "detected": True,
                "slow_nodes": self.straggler.slow_nodes,
                "util_spread_pct": round(self.straggler.util_spread_pct, 1),
                "message": self.straggler.message,
            }
        return d


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------

def _run_on(host: Optional[str], cmd: str, user: str = "root",
            key: Optional[str] = None, timeout: int = 15) -> Tuple[int, str, str]:
    if host and host not in ("localhost", "127.0.0.1"):
        ssh = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
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


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s) if s not in ("N/A", "[N/A]", "[Not Supported]", "") else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# GPU metric collection — nvidia-smi default, Prometheus hook
# ---------------------------------------------------------------------------

def _collect_gpu_nvidia_smi(ctx: Dict[str, Any]) -> List[GPUMetric]:
    """Default: nvidia-smi query (zero deps)."""
    host = ctx.get("host")
    user = ctx.get("ssh_user", "root")
    key = ctx.get("ssh_key")
    node = host or "localhost"

    query = ("nvidia-smi --query-gpu=index,name,utilization.gpu,"
             "memory.used,memory.total,temperature.gpu,"
             "power.draw,power.limit "
             "--format=csv,noheader,nounits")
    rc, stdout, _ = _run_on(host, query, user, key)
    if rc != 0:
        return []

    metrics = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        metrics.append(GPUMetric(
            node=node, gpu_index=int(parts[0]), gpu_name=parts[1],
            utilization_pct=_safe_float(parts[2]),
            memory_used_mb=_safe_float(parts[3]),
            memory_total_mb=_safe_float(parts[4]),
            temperature_c=_safe_float(parts[5]),
            power_w=_safe_float(parts[6]),
            power_limit_w=_safe_float(parts[7]),
        ))
    return metrics


def _collect_gpu_prometheus(endpoint: str, node: str) -> List[GPUMetric]:
    """Optional: scrape Prometheus/DCGM-exporter endpoint."""
    try:
        import urllib.request
        url = f"{endpoint}/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        metrics = []
        for result in data.get("data", {}).get("result", []):
            gpu_idx = int(result.get("metric", {}).get("gpu", "0"))
            util = float(result["value"][1])
            metrics.append(GPUMetric(
                node=node, gpu_index=gpu_idx, gpu_name="",
                utilization_pct=util, memory_used_mb=0, memory_total_mb=0,
                temperature_c=0, power_w=0, power_limit_w=0,
            ))
        return metrics
    except Exception as e:
        logger.debug(f"Prometheus scrape failed ({e}), falling back to nvidia-smi")
        return []


# ---------------------------------------------------------------------------
# Training log parser
# ---------------------------------------------------------------------------

# Compiled patterns for common frameworks
_LOG_PATTERNS = {
    "step": [
        re.compile(r"(?:step|iteration|global_step)[\s:=]+(\d+)", re.I),
        re.compile(r"\b(\d+)/\d+ \["),
    ],
    "loss": [
        re.compile(r"(?:loss|train_loss|training_loss)[\s:=]+([\d.]+(?:e[+-]?\d+)?)", re.I),
    ],
    "grad_norm": [
        re.compile(r"(?:grad_norm|gradient_norm|grad\.norm)[\s:=]+([\d.]+(?:e[+-]?\d+)?)", re.I),
    ],
    "learning_rate": [
        re.compile(r"(?:lr|learning_rate)[\s:=]+([\d.]+(?:e[+-]?\d+)?)", re.I),
    ],
    "step_time_ms": [
        re.compile(r"(?:step_time|elapsed|time/step|iter_time)[\s:=]+([\d.]+)\s*(?:ms|s)?", re.I),
    ],
    "tokens_per_sec": [
        re.compile(r"(?:tokens/sec|tps|tokens_per_second|throughput)[\s:=]+([\d.]+)", re.I),
    ],
    "samples_per_sec": [
        re.compile(r"(?:samples/sec|sps|samples_per_second)[\s:=]+([\d.]+)", re.I),
    ],
}


def _parse_training_log(ctx: Dict[str, Any]) -> Optional[TrainingMetrics]:
    """Parse latest training metrics from log file (tail scan)."""
    log_path = ctx.get("log_path", "")
    if not log_path or not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        tail = lines[-50:] if len(lines) > 50 else lines
    except Exception:
        return None

    metrics = TrainingMetrics()
    found = set()

    for line in reversed(tail):
        for name, regexes in _LOG_PATTERNS.items():
            if name in found:
                continue
            for regex in regexes:
                m = regex.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                        if name == "step":
                            metrics.step = int(val)
                        else:
                            setattr(metrics, name, val)
                        found.add(name)
                    except (ValueError, IndexError):
                        pass
                    break

    return metrics if metrics.step > 0 else None


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def _compute_cost(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compute GPU-hours and cost. Uses state_manager if available, else estimates."""
    state_manager = ctx.get("state_manager")
    job_id = ctx.get("job_id", "")
    gpu_count = ctx.get("gpu_count", 0)
    cost_per_gpu_hour = ctx.get("cost_per_gpu_hour", 0.0)

    elapsed_h = 0.0
    cost_usd = 0.0

    if state_manager and job_id:
        job = state_manager.get_job(job_id)
        if job:
            if hasattr(job, "started_at") and job.started_at:
                elapsed_h = (datetime.now() - job.started_at).total_seconds() / 3600
            cost_usd = getattr(job, "cost_usd", 0.0)

    gpu_hours = elapsed_h * gpu_count
    if cost_per_gpu_hour > 0 and cost_usd == 0:
        cost_usd = gpu_hours * cost_per_gpu_hour

    return {"cost_usd": cost_usd, "elapsed_hours": elapsed_h, "gpu_hours": gpu_hours}


# ---------------------------------------------------------------------------
# Straggler detection
# ---------------------------------------------------------------------------

def _detect_stragglers(gpus: List[GPUMetric], threshold_pct: float = 30.0) -> StragglerInfo:
    """Detect nodes with significantly lower GPU utilization."""
    if not gpus:
        return StragglerInfo()

    # Per-node average utilization
    node_utils: Dict[str, List[float]] = {}
    for g in gpus:
        node_utils.setdefault(g.node, []).append(g.utilization_pct)

    if len(node_utils) < 2:
        return StragglerInfo()

    node_avg = {n: sum(u) / len(u) for n, u in node_utils.items()}
    max_util = max(node_avg.values())
    min_util = min(node_avg.values())
    spread = max_util - min_util

    if spread < threshold_pct:
        return StragglerInfo(util_spread_pct=spread)

    slow = [n for n, avg in node_avg.items() if (max_util - avg) > threshold_pct]
    return StragglerInfo(
        detected=True,
        slow_nodes=slow,
        util_spread_pct=spread,
        message=f"Nodes {slow} are {spread:.0f}% below fastest — check network/storage",
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(ctx: Dict[str, Any]) -> MonitorSnapshot:
    """Aggregate all Wave 0 outputs into a snapshot."""
    deps = ctx.get("__deps__", {})
    job_id = ctx.get("job_id", "")

    all_gpus: List[GPUMetric] = []
    training: Optional[TrainingMetrics] = None
    cost_info = {"cost_usd": 0.0, "elapsed_hours": 0.0, "gpu_hours": 0.0}
    nodes_seen = set()

    for key, val in deps.items():
        if key.startswith("gpu_") and isinstance(val, list):
            for g in val:
                if isinstance(g, GPUMetric):
                    all_gpus.append(g)
                    nodes_seen.add(g.node)
        elif key == "training_log" and isinstance(val, TrainingMetrics):
            training = val
        elif key == "cost" and isinstance(val, dict):
            cost_info = val

    avg_util = (sum(g.utilization_pct for g in all_gpus) / len(all_gpus)) if all_gpus else 0
    avg_mem = (sum(g.memory_used_mb / max(g.memory_total_mb, 1) * 100
                   for g in all_gpus) / len(all_gpus)) if all_gpus else 0
    total_power = sum(g.power_w for g in all_gpus)
    straggler = _detect_stragglers(all_gpus)

    return MonitorSnapshot(
        timestamp=datetime.now(),
        job_id=job_id,
        gpus=all_gpus,
        training=training,
        straggler=straggler,
        cost_usd=cost_info.get("cost_usd", 0.0),
        gpu_hours=cost_info.get("gpu_hours", 0.0),
        elapsed_hours=cost_info.get("elapsed_hours", 0.0),
        avg_gpu_util=avg_util,
        avg_gpu_memory_pct=avg_mem,
        total_gpu_power_w=total_power,
        node_count=len(nodes_seen),
        gpu_count=len(all_gpus),
    )


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    """
    Unified training monitor.

    Default: nvidia-smi + log file parsing (zero external deps).
    Optional hooks:
      - prometheus_endpoint: scrape DCGM-exporter / Prometheus
      - wandb_run: push snapshots to W&B
      - on_snapshot: custom callback for any sink

    Usage:
        mon = TrainingMonitor(nodes=["10.0.0.1", "10.0.0.2"])
        snap = mon.snapshot(job_id="abc")
        print(json.dumps(snap.to_dict(), indent=2))
    """

    def __init__(self, nodes: Optional[List[str]] = None,
                 ssh_user: str = "root", ssh_key: Optional[str] = None,
                 state_manager=None,
                 log_path: str = "",
                 cost_per_gpu_hour: float = 0.0,
                 # Optional hooks
                 prometheus_endpoint: str = "",
                 wandb_run=None,
                 on_snapshot: Optional[Callable] = None):
        self.nodes = nodes or [None]
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.state_manager = state_manager
        self.log_path = log_path
        self.cost_per_gpu_hour = cost_per_gpu_hour
        self.prometheus_endpoint = prometheus_endpoint
        self.wandb_run = wandb_run
        self.on_snapshot = on_snapshot
        self._history: List[MonitorSnapshot] = []

    def snapshot(self, job_id: str = "") -> MonitorSnapshot:
        """Collect a single monitoring snapshot (DAG-parallel across nodes)."""
        dag = DAGExecutor(
            max_workers=max(len(self.nodes) + 2, 4),
            name="monitor"
        )

        # Wave 0: GPU metrics per node (parallel)
        wave0 = []
        for i, node in enumerate(self.nodes):
            name = f"gpu_{i}"
            wave0.append(name)
            if self.prometheus_endpoint:
                # Try Prometheus first, nvidia-smi fallback inside
                def make_prom_fn(h, ep):
                    def fn(_ctx):
                        result = _collect_gpu_prometheus(ep, h or "localhost")
                        if result:
                            return result
                        return _collect_gpu_nvidia_smi({
                            "host": h, "ssh_user": self.ssh_user,
                            "ssh_key": self.ssh_key})
                    return fn
                dag.add_node(name, make_prom_fn(node, self.prometheus_endpoint))
            else:
                def make_smi_fn(h):
                    def fn(_ctx):
                        return _collect_gpu_nvidia_smi({
                            "host": h, "ssh_user": self.ssh_user,
                            "ssh_key": self.ssh_key})
                    return fn
                dag.add_node(name, make_smi_fn(node))

        # Training log parse
        dag.add_node("training_log", lambda _ctx: _parse_training_log({
            "log_path": self.log_path}))
        wave0.append("training_log")

        # Cost
        dag.add_node("cost", lambda _ctx: _compute_cost({
            "state_manager": self.state_manager, "job_id": job_id,
            "gpu_count": len(self.nodes) * 8,  # estimate, refined in aggregate
            "cost_per_gpu_hour": self.cost_per_gpu_hour}))
        wave0.append("cost")

        # Wave 1: Aggregate
        dag.add_node("aggregate", _aggregate, depends_on=set(wave0))

        result = dag.apply(initial_context={"job_id": job_id})
        snap = result.outputs.get("aggregate")

        if not isinstance(snap, MonitorSnapshot):
            snap = MonitorSnapshot(timestamp=datetime.now(), job_id=job_id)

        self._history.append(snap)

        # Update state manager
        if self.state_manager and job_id and snap.training and snap.training.step > 0:
            try:
                self.state_manager.update_job_step(job_id, snap.training.step)
            except Exception:
                pass

        # W&B hook (optional — only if wandb_run is set)
        if self.wandb_run and snap.training:
            try:
                self.wandb_run.log({
                    "gpu_util": snap.avg_gpu_util,
                    "gpu_memory_pct": snap.avg_gpu_memory_pct,
                    "power_w": snap.total_gpu_power_w,
                    "loss": snap.training.loss,
                    "step": snap.training.step,
                    "tokens_per_sec": snap.training.tokens_per_sec,
                    "cost_usd": snap.cost_usd,
                })
            except Exception as e:
                logger.debug(f"W&B log failed: {e}")

        # Custom callback hook
        if self.on_snapshot:
            try:
                self.on_snapshot(snap)
            except Exception as e:
                logger.debug(f"Snapshot callback failed: {e}")

        return snap

    def continuous(self, job_id: str, interval_s: float = 10.0,
                   max_snapshots: int = 0) -> List[MonitorSnapshot]:
        """Continuous monitoring loop. Returns all collected snapshots."""
        count = 0
        while True:
            snap = self.snapshot(job_id)
            count += 1
            self._print_snapshot(snap)
            if max_snapshots and count >= max_snapshots:
                break
            time.sleep(interval_s)
        return self._history

    def get_history(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._history]

    @staticmethod
    def _print_snapshot(snap: MonitorSnapshot):
        """Compact CLI output."""
        t = snap.training
        step_str = f"step={t.step} loss={t.loss:.4f} " if t and t.step else ""
        strag = " ⚠ STRAGGLER" if snap.straggler and snap.straggler.detected else ""
        print(
            f"[{snap.timestamp.strftime('%H:%M:%S')}] "
            f"{snap.gpu_count}×GPU util={snap.avg_gpu_util:.0f}% "
            f"mem={snap.avg_gpu_memory_pct:.0f}% "
            f"pwr={snap.total_gpu_power_w:.0f}W "
            f"{step_str}"
            f"cost=${snap.cost_usd:.2f} "
            f"gpuh={snap.gpu_hours:.1f}"
            f"{strag}"
        )
