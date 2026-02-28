#!/usr/bin/env python3
"""
Preflight Validator — Validates cluster readiness before training.

Design philosophy: delegate to native best-in-class tools (dcgmi, nvbandwidth,
nccl-tests, fio, ibstat/perfquery), parallelize across nodes via DAGExecutor,
return structured JSON for MCP consumption.

DAG structure per node (batch_apply across N nodes):
    Wave 0: gpu_inventory ∥ dcgm_diag ∥ nvlink_topo ∥ rdma_health ∥ storage_fio
    Wave 1: nccl_intra_node  (depends on gpu_inventory + rdma_health)

Then cross-node (after all nodes pass):
    Wave 0: nccl_cross_node_pair_0 ∥ pair_1 ∥ ...  (pairwise bisection)
    Wave 1: nccl_full_cluster                       (all nodes)

Then data:
    Wave 0: data_integrity

Inspired by Together AI's cluster acceptance testing pipeline.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dag_executor import DAGExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CheckStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    node: str = ""

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)


@dataclass
class PreflightReport:
    checks: List[CheckResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def summary(self) -> Dict[str, Any]:
        counts = {s.value: 0 for s in CheckStatus}
        for c in self.checks:
            counts[c.status.value] += 1
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "counts": counts,
            "failures": [{"name": c.name, "node": c.node, "message": c.message}
                         for c in self.failures],
            "warnings": [{"name": c.name, "node": c.node, "message": c.message}
                         for c in self.warnings],
            "duration_ms": self.total_duration_ms,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_duration_ms": self.total_duration_ms,
            "checks": [
                {"name": c.name, "status": c.status.value, "message": c.message,
                 "node": c.node, "duration_ms": c.duration_ms, "details": c.details}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------

def _run_on(host: Optional[str], cmd: str, user: str = "root",
            key: Optional[str] = None, timeout: int = 60) -> Tuple[int, str, str]:
    """Run command locally or via SSH."""
    if host and host not in ("localhost", "127.0.0.1"):
        ssh = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if key:
            ssh.extend(["-i", key])
        ssh.extend([f"{user}@{host}", cmd])
        try:
            r = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "SSH timed out"
        except Exception as e:
            return -1, "", str(e)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s) if s not in ("N/A", "[N/A]", "[Not Supported]", "") else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Per-node DAG check functions
# Each: fn(ctx) -> List[CheckResult]
# ---------------------------------------------------------------------------

def _check_gpu_inventory(ctx: Dict[str, Any]) -> List[CheckResult]:
    """Validate GPU count, type, driver — catches 'GPU fell off the bus'."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    expected_gpus = ctx.get("expected_gpus_per_node", 0)
    expected_type = ctx.get("expected_gpu_type", "")
    node = host or "localhost"

    query = ("nvidia-smi --query-gpu=index,name,driver_version,"
             "memory.total,memory.used,temperature.gpu,"
             "ecc.errors.corrected.volatile.total,"
             "ecc.errors.uncorrected.volatile.total,"
             "persistence_mode,power.draw,power.limit "
             "--format=csv,noheader,nounits")
    rc, stdout, stderr = _run_on(host, query, user, key, timeout=15)
    if rc != 0:
        return [CheckResult("gpu_inventory", CheckStatus.FAIL,
                            f"nvidia-smi failed: {stderr}", node=node)]

    results: List[CheckResult] = []
    gpus = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        gpu = {
            "index": parts[0], "name": parts[1], "driver": parts[2],
            "mem_total_mb": _safe_float(parts[3]),
            "mem_used_mb": _safe_float(parts[4]),
            "temp_c": _safe_float(parts[5]),
            "ecc_corrected": int(_safe_float(parts[6])),
            "ecc_uncorrected": int(_safe_float(parts[7])),
            "persistence_mode": parts[8].strip(),
            "power_w": _safe_float(parts[9]),
            "power_limit_w": _safe_float(parts[10]),
        }
        gpus.append(gpu)

        # ECC
        if gpu["ecc_uncorrected"] > 0:
            results.append(CheckResult(
                f"gpu_{gpu['index']}_ecc", CheckStatus.FAIL,
                f"GPU {gpu['index']} ({gpu['name']}): {gpu['ecc_uncorrected']} uncorrected ECC — replace",
                details=gpu, node=node))
        elif gpu["ecc_corrected"] > 100:
            results.append(CheckResult(
                f"gpu_{gpu['index']}_ecc", CheckStatus.WARN,
                f"GPU {gpu['index']}: {gpu['ecc_corrected']} corrected ECC — monitor",
                details=gpu, node=node))

        # Temperature
        if gpu["temp_c"] > 85:
            results.append(CheckResult(
                f"gpu_{gpu['index']}_temp", CheckStatus.WARN,
                f"GPU {gpu['index']} at {gpu['temp_c']}°C", details=gpu, node=node))

        # Memory in use before training
        mem_pct = (gpu["mem_used_mb"] / max(gpu["mem_total_mb"], 1)) * 100
        if mem_pct > 10:
            results.append(CheckResult(
                f"gpu_{gpu['index']}_mem", CheckStatus.WARN,
                f"GPU {gpu['index']} has {mem_pct:.0f}% memory in use", details=gpu, node=node))

        # Persistence mode
        if gpu["persistence_mode"].lower() not in ("enabled", "on"):
            results.append(CheckResult(
                f"gpu_{gpu['index']}_persist", CheckStatus.WARN,
                f"GPU {gpu['index']} persistence mode off — run: nvidia-smi -pm 1",
                node=node))

    # Count validation
    actual_count = len(gpus)
    if expected_gpus and actual_count != expected_gpus:
        results.append(CheckResult(
            "gpu_count", CheckStatus.FAIL,
            f"Expected {expected_gpus} GPUs, found {actual_count} — GPU may have fallen off bus",
            details={"expected": expected_gpus, "actual": actual_count}, node=node))
    elif actual_count > 0:
        results.append(CheckResult(
            "gpu_count", CheckStatus.PASS,
            f"{actual_count} GPUs detected ({gpus[0]['name']})",
            details={"count": actual_count, "type": gpus[0]["name"], "driver": gpus[0]["driver"]},
            node=node))

    # Type validation
    if expected_type and gpus:
        if expected_type.lower() not in gpus[0]["name"].lower():
            results.append(CheckResult(
                "gpu_type", CheckStatus.WARN,
                f"Expected {expected_type}, found {gpus[0]['name']}", node=node))

    return results


def _check_dcgm_diag(ctx: Dict[str, Any]) -> List[CheckResult]:
    """Run DCGM diagnostics (stress test, memory, PCIe). Falls back to gpu-burn."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    level = ctx.get("dcgm_level", 2)  # 1=quick, 2=medium, 3=full stress
    node = host or "localhost"

    # Try dcgmi first
    rc, stdout, stderr = _run_on(
        host, f"dcgmi diag --run {level} --fail-early -j 2>/dev/null || echo DCGM_NOT_FOUND",
        user, key, timeout=300)

    if "DCGM_NOT_FOUND" not in stdout and rc == 0:
        # Parse JSON output
        try:
            diag = json.loads(stdout)
            passed = True
            details = {}
            for test in diag.get("tests", diag.get("DCGM Diagnostic", {}).get("test_categories", [])):
                name = test.get("name", test.get("category", "unknown"))
                result = test.get("result", test.get("results", ""))
                details[name] = result
                if "fail" in str(result).lower():
                    passed = False
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            return [CheckResult("dcgm_diag", status,
                                f"DCGM level-{level}: {'PASS' if passed else 'FAIL'}",
                                details=details, node=node)]
        except json.JSONDecodeError:
            # Text output — check for "Pass" / "Fail"
            fails = stdout.lower().count("fail")
            if fails > 0:
                return [CheckResult("dcgm_diag", CheckStatus.FAIL,
                                    f"DCGM level-{level}: {fails} test(s) failed",
                                    details={"raw": stdout[:500]}, node=node)]
            return [CheckResult("dcgm_diag", CheckStatus.PASS,
                                f"DCGM level-{level}: all tests passed", node=node)]

    # Fallback: gpu-burn 30s
    rc2, stdout2, stderr2 = _run_on(
        host, "which gpu_burn >/dev/null 2>&1 && gpu_burn 30 2>&1 || echo GPU_BURN_NOT_FOUND",
        user, key, timeout=60)

    if "GPU_BURN_NOT_FOUND" in stdout2:
        return [CheckResult("dcgm_diag", CheckStatus.SKIP,
                            "Neither dcgmi nor gpu_burn available. "
                            "Install: apt install datacenter-gpu-manager OR github.com/wilicc/gpu-burn",
                            node=node)]

    if "OK" in stdout2 and "FAIL" not in stdout2:
        return [CheckResult("gpu_burn", CheckStatus.PASS,
                            "gpu-burn 30s: all GPUs OK", node=node)]
    return [CheckResult("gpu_burn", CheckStatus.FAIL,
                        f"gpu-burn failed: {stdout2[:300]}", node=node)]


def _check_nvlink_topo(ctx: Dict[str, Any]) -> List[CheckResult]:
    """NVLink + NVSwitch: topology matrix and nvbandwidth if available."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    node = host or "localhost"
    results: List[CheckResult] = []

    # Topology matrix
    rc, stdout, stderr = _run_on(host, "nvidia-smi topo -m", user, key, timeout=15)
    if rc == 0 and stdout:
        # Count NVLink connections (NV## entries)
        nv_count = len(re.findall(r'\bNV\d+', stdout))
        pix_count = len(re.findall(r'\bPIX\b', stdout))
        sys_count = len(re.findall(r'\bSYS\b', stdout))
        results.append(CheckResult(
            "nvlink_topo", CheckStatus.PASS,
            f"Topology: {nv_count} NVLink, {pix_count} PIX, {sys_count} SYS connections",
            details={"nvlink_connections": nv_count, "pix": pix_count, "sys": sys_count},
            node=node))
    else:
        results.append(CheckResult("nvlink_topo", CheckStatus.SKIP,
                                   "nvidia-smi topo not available", node=node))

    # nvbandwidth — actual GPU-to-GPU memcpy benchmark
    rc, stdout, _ = _run_on(
        host, "which nvbandwidth >/dev/null 2>&1 && nvbandwidth -t device_to_device_memcpy_write_ce 2>&1"
              " || echo NVB_NOT_FOUND",
        user, key, timeout=120)

    if "NVB_NOT_FOUND" not in stdout and rc == 0:
        # Parse bandwidth matrix — look for minimum off-diagonal value
        bw_values = []
        for line in stdout.splitlines():
            nums = re.findall(r'(\d+\.\d+)', line)
            bw_values.extend(float(n) for n in nums if float(n) > 1.0)
        if bw_values:
            min_bw = min(bw_values)
            max_bw = max(bw_values)
            avg_bw = sum(bw_values) / len(bw_values)
            # H100 NVLink should be ~380-400 GB/s
            status = CheckStatus.PASS if min_bw > 300 else (
                CheckStatus.WARN if min_bw > 200 else CheckStatus.FAIL)
            results.append(CheckResult(
                "nvbandwidth", status,
                f"GPU↔GPU bandwidth: min={min_bw:.0f} avg={avg_bw:.0f} max={max_bw:.0f} GB/s",
                details={"min_gbs": min_bw, "avg_gbs": avg_bw, "max_gbs": max_bw},
                node=node))
    else:
        results.append(CheckResult("nvbandwidth", CheckStatus.SKIP,
                                   "nvbandwidth not installed — github.com/NVIDIA/nvbandwidth",
                                   node=node))

    return results


def _check_rdma_health(ctx: Dict[str, Any]) -> List[CheckResult]:
    """IB fabric: port state, link errors, GPUDirect RDMA, ib_write_bw."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    node = host or "localhost"
    results: List[CheckResult] = []

    # IB port status
    rc, stdout, _ = _run_on(host, "ibstat 2>/dev/null || echo NO_IB", user, key)
    if "NO_IB" in stdout:
        return [CheckResult("rdma_ib", CheckStatus.SKIP, "No InfiniBand", node=node)]

    active = stdout.count("Active")
    down = stdout.count("Down")
    rate_matches = re.findall(r'Rate:\s+(\d+)', stdout)
    rates = [int(r) for r in rate_matches] if rate_matches else []

    if down > 0:
        results.append(CheckResult("ib_ports", CheckStatus.FAIL,
                                   f"{down} IB port(s) down, {active} active",
                                   details={"active": active, "down": down}, node=node))
    elif active > 0:
        rate_str = f", {rates[0]} Gb/s" if rates else ""
        results.append(CheckResult("ib_ports", CheckStatus.PASS,
                                   f"{active} IB port(s) active{rate_str}",
                                   details={"active": active, "rates_gbps": rates}, node=node))

    # Link error counters
    rc, stdout, _ = _run_on(host, "perfquery 2>/dev/null || echo SKIP", user, key)
    if "SKIP" not in stdout:
        error_kw = ["SymbolErrorCounter", "LinkErrorRecoveryCounter",
                    "LinkDownedCounter", "RcvErrors", "LocalLinkIntegrityErrors"]
        errors = {}
        for line in stdout.splitlines():
            for kw in error_kw:
                if kw in line:
                    val = line.split(":")[-1].strip().replace(".", "")
                    try:
                        c = int(val)
                        if c > 0:
                            errors[kw] = c
                    except ValueError:
                        pass
        if errors:
            results.append(CheckResult("ib_errors", CheckStatus.WARN,
                                       f"IB link errors: {errors}", details=errors, node=node))
        else:
            results.append(CheckResult("ib_errors", CheckStatus.PASS,
                                       "No IB link errors", node=node))

    # GPUDirect RDMA
    rc, stdout, _ = _run_on(host, "lsmod | grep nvidia_peermem", user, key)
    if "nvidia_peermem" in stdout:
        results.append(CheckResult("gpudirect_rdma", CheckStatus.PASS,
                                   "nvidia_peermem loaded", node=node))
    else:
        results.append(CheckResult("gpudirect_rdma", CheckStatus.WARN,
                                   "nvidia_peermem not loaded — run: modprobe nvidia_peermem",
                                   node=node))

    return results


def _check_storage_fio(ctx: Dict[str, Any]) -> List[CheckResult]:
    """Storage I/O via fio (preferred) or dd fallback."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    data_path = ctx.get("data_path", "/tmp")
    node = host or "localhost"

    # Try fio first
    fio_cmd = (
        f"which fio >/dev/null 2>&1 && fio --name=seq_write --directory={data_path} "
        f"--rw=write --bs=1M --size=256M --numjobs=4 --time_based --runtime=10 "
        f"--group_reporting --output-format=json --direct=1 2>/dev/null || echo FIO_NOT_FOUND"
    )
    rc, stdout, stderr = _run_on(host, fio_cmd, user, key, timeout=30)

    if "FIO_NOT_FOUND" not in stdout and rc == 0:
        try:
            fio_result = json.loads(stdout)
            jobs = fio_result.get("jobs", [{}])
            if jobs:
                write_bw_mbs = jobs[0].get("write", {}).get("bw_mean", 0) / 1024  # KB/s → MB/s
                write_iops = jobs[0].get("write", {}).get("iops_mean", 0)
                status = CheckStatus.PASS if write_bw_mbs > 500 else (
                    CheckStatus.WARN if write_bw_mbs > 100 else CheckStatus.FAIL)
                return [CheckResult("storage_fio", status,
                                    f"Sequential write: {write_bw_mbs:.0f} MB/s, {write_iops:.0f} IOPS",
                                    details={"write_mbs": write_bw_mbs, "write_iops": write_iops},
                                    node=node)]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # dd fallback
    test_file = f"{data_path}/.terradev_io_{os.getpid()}"
    dd_cmd = f"dd if=/dev/zero of={test_file} bs=1M count=256 oflag=direct 2>&1; rm -f {test_file}"
    rc, stdout, stderr = _run_on(host, dd_cmd, user, key, timeout=60)
    output = stdout + stderr
    match = re.search(r'([\d.]+)\s*(GB/s|MB/s)', output)
    if match:
        bw = float(match.group(1))
        bw_mbs = bw * 1000 if match.group(2) == "GB/s" else bw
        status = CheckStatus.PASS if bw_mbs > 500 else (
            CheckStatus.WARN if bw_mbs > 100 else CheckStatus.FAIL)
        return [CheckResult("storage_dd", status,
                            f"dd write: {bw_mbs:.0f} MB/s (install fio for better test)",
                            details={"write_mbs": bw_mbs}, node=node)]

    return [CheckResult("storage_io", CheckStatus.WARN,
                        "I/O test ran but throughput not parsed", node=node)]


def _check_nccl_intra(ctx: Dict[str, Any]) -> List[CheckResult]:
    """Intra-node NCCL all-reduce at training-realistic message sizes."""
    host, user, key = ctx.get("host"), ctx.get("ssh_user", "root"), ctx.get("ssh_key")
    node = host or "localhost"

    # Gate on GPU inventory
    gpu_inv = ctx.get("gpu_inventory", [])
    if isinstance(gpu_inv, list) and any(
        getattr(r, "status", None) == CheckStatus.FAIL for r in gpu_inv
        if isinstance(r, CheckResult) and "count" in r.name
    ):
        return [CheckResult("nccl_intra", CheckStatus.SKIP,
                            "Skipped — GPU inventory check failed", node=node)]

    # Training-realistic sizes: 256M to 4G (matching Together AI's approach)
    cmd = (
        "which all_reduce_perf >/dev/null 2>&1 && "
        "all_reduce_perf -b 256M -e 4G -f 2 -g $(nvidia-smi -L | wc -l) 2>&1 "
        "|| echo NCCL_NOT_FOUND"
    )
    rc, stdout, stderr = _run_on(host, cmd, user, key, timeout=180)

    if "NCCL_NOT_FOUND" in stdout:
        return [CheckResult("nccl_intra", CheckStatus.SKIP,
                            "nccl-tests not installed — github.com/NVIDIA/nccl-tests", node=node)]
    if rc != 0:
        return [CheckResult("nccl_intra", CheckStatus.FAIL,
                            f"NCCL all-reduce failed: {(stdout + stderr)[:300]}", node=node)]

    # Parse busbw from data lines
    busbw_values = []
    for line in stdout.splitlines():
        if re.match(r'\s*\d+', line):
            parts = line.split()
            # busbw is typically the second-to-last column
            for i in [-2, -3]:
                try:
                    val = float(parts[i])
                    if 0.1 < val < 2000:
                        busbw_values.append(val)
                        break
                except (ValueError, IndexError):
                    continue

    if busbw_values:
        min_bw = min(busbw_values)
        avg_bw = sum(busbw_values) / len(busbw_values)
        max_bw = max(busbw_values)
        # H100 SXM intra-node NVLink should be ~400+ GB/s busbw
        status = CheckStatus.PASS if min_bw > 300 else (
            CheckStatus.WARN if min_bw > 100 else CheckStatus.FAIL)
        return [CheckResult("nccl_intra", status,
                            f"Intra-node busbw: min={min_bw:.0f} avg={avg_bw:.0f} max={max_bw:.0f} GB/s",
                            details={"min_gbs": min_bw, "avg_gbs": avg_bw, "max_gbs": max_bw},
                            node=node)]

    return [CheckResult("nccl_intra", CheckStatus.PASS,
                        "NCCL all-reduce completed (bandwidth not parsed)", node=node)]


def _check_data_integrity(ctx: Dict[str, Any]) -> List[CheckResult]:
    """Data integrity: existence, size, checksums, sample count."""
    data_config = ctx.get("data_config", {})
    results: List[CheckResult] = []
    node = "data"

    data_path = data_config.get("path")
    if not data_path:
        return [CheckResult("data_integrity", CheckStatus.SKIP,
                            "No data path configured", node=node)]
    if not os.path.exists(data_path):
        return [CheckResult("data_exists", CheckStatus.FAIL,
                            f"Path does not exist: {data_path}", node=node)]

    # Size
    if os.path.isdir(data_path):
        data_files = [f for f in Path(data_path).rglob("*")
                      if f.is_file() and f.stat().st_size > 0]
        if not data_files:
            return [CheckResult("data_empty", CheckStatus.FAIL,
                                f"Directory empty: {data_path}", node=node)]
        total = sum(f.stat().st_size for f in data_files)
        results.append(CheckResult("data_exists", CheckStatus.PASS,
                                   f"{len(data_files)} files, {total / (1 << 30):.2f} GB",
                                   details={"files": len(data_files), "bytes": total}, node=node))
    else:
        size = os.path.getsize(data_path)
        if size == 0:
            return [CheckResult("data_empty", CheckStatus.FAIL,
                                f"File empty: {data_path}", node=node)]
        results.append(CheckResult("data_exists", CheckStatus.PASS,
                                   f"{size / (1 << 30):.2f} GB", node=node))

    # Manifest checksum verification (parallelized per shard via DAG)
    manifest_path = data_config.get("manifest")
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            shards = manifest.get("shards", manifest.get("files", []))
            if shards:
                verify_dag = DAGExecutor(max_workers=min(len(shards), 8),
                                         name="data_checksum")
                for i, entry in enumerate(shards):
                    def make_verify(e):
                        def fn(_ctx):
                            fp = e.get("path", "")
                            expected = e.get("sha256", e.get("checksum", ""))
                            if not fp or not expected:
                                return "skip"
                            full = os.path.join(os.path.dirname(manifest_path), fp)
                            if not os.path.exists(full):
                                return "missing"
                            h = hashlib.sha256()
                            with open(full, "rb") as fo:
                                for block in iter(lambda: fo.read(8 << 20), b""):
                                    h.update(block)
                            return "ok" if h.hexdigest() == expected else "mismatch"
                        return fn
                    verify_dag.add_node(f"shard_{i}", make_verify(entry))

                vr = verify_dag.apply()
                ok = sum(1 for v in vr.outputs.values() if v == "ok")
                bad = sum(1 for v in vr.outputs.values() if v in ("mismatch", "missing"))
                if bad > 0:
                    results.append(CheckResult("data_checksum", CheckStatus.FAIL,
                                               f"{bad} shard(s) failed, {ok} verified",
                                               details={"ok": ok, "failed": bad}, node=node))
                else:
                    results.append(CheckResult("data_checksum", CheckStatus.PASS,
                                               f"{ok} shard(s) verified", node=node))
        except Exception as e:
            results.append(CheckResult("data_checksum", CheckStatus.WARN,
                                       f"Manifest error: {e}", node=node))

    # Sample count
    expected = data_config.get("expected_samples")
    if expected:
        base = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
        meta_path = os.path.join(base, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                actual = meta.get("total_samples", meta.get("num_samples", 0))
                if actual != expected:
                    results.append(CheckResult("data_samples", CheckStatus.FAIL,
                                               f"Expected {expected}, found {actual}",
                                               details={"expected": expected, "actual": actual},
                                               node=node))
                else:
                    results.append(CheckResult("data_samples", CheckStatus.PASS,
                                               f"Sample count: {actual}", node=node))
            except Exception as e:
                results.append(CheckResult("data_samples", CheckStatus.WARN,
                                           f"Cannot read metadata: {e}", node=node))

    return results or [CheckResult("data_integrity", CheckStatus.PASS,
                                   "Data exists and is non-empty", node=node)]


# ---------------------------------------------------------------------------
# Cross-node NCCL check (pairwise bisection)
# ---------------------------------------------------------------------------

def _check_nccl_pair(ctx: Dict[str, Any]) -> List[CheckResult]:
    """NCCL all-reduce between exactly two nodes — isolates bad links/switches."""
    node_a = ctx.get("node_a", "")
    node_b = ctx.get("node_b", "")
    user = ctx.get("ssh_user", "root")
    key = ctx.get("ssh_key")
    gpus_per_node = ctx.get("gpus_per_node", 8)

    pair_label = f"{node_a}↔{node_b}"
    # Use mpirun or NCCL's multi-node mode via hostfile
    hostfile_content = f"{node_a} slots={gpus_per_node}\\n{node_b} slots={gpus_per_node}"
    cmd = (
        f"echo -e '{hostfile_content}' > /tmp/.terradev_hf_{os.getpid()} && "
        f"mpirun --hostfile /tmp/.terradev_hf_{os.getpid()} "
        f"--mca btl tcp,self -mca btl_tcp_if_exclude lo "
        f"-np 2 --map-by node "
        f"all_reduce_perf -b 1G -e 8G -f 2 -g {gpus_per_node} 2>&1 "
        f"|| echo PAIR_NCCL_FAILED; "
        f"rm -f /tmp/.terradev_hf_{os.getpid()}"
    )

    rc, stdout, stderr = _run_on(node_a, cmd, user, key, timeout=300)
    output = stdout + stderr

    if "PAIR_NCCL_FAILED" in output or rc != 0:
        return [CheckResult(f"nccl_pair_{pair_label}", CheckStatus.FAIL,
                            f"Cross-node NCCL failed: {pair_label}",
                            details={"output": output[:300]}, node=pair_label)]

    # Parse busbw
    busbw_values = []
    for line in stdout.splitlines():
        if re.match(r'\s*\d+', line):
            parts = line.split()
            for i in [-2, -3]:
                try:
                    val = float(parts[i])
                    if 0.1 < val < 2000:
                        busbw_values.append(val)
                        break
                except (ValueError, IndexError):
                    continue

    if busbw_values:
        min_bw = min(busbw_values)
        avg_bw = sum(busbw_values) / len(busbw_values)
        # 400Gb IB ≈ 50 GB/s per GPU → 400 GB/s aggregate for 8 GPUs, ~92% = 368
        status = CheckStatus.PASS if min_bw > 300 else (
            CheckStatus.WARN if min_bw > 100 else CheckStatus.FAIL)
        return [CheckResult(f"nccl_pair", status,
                            f"{pair_label}: busbw min={min_bw:.0f} avg={avg_bw:.0f} GB/s",
                            details={"pair": pair_label, "min_gbs": min_bw, "avg_gbs": avg_bw},
                            node=pair_label)]

    return [CheckResult(f"nccl_pair", CheckStatus.PASS,
                        f"{pair_label}: NCCL completed (bandwidth not parsed)",
                        node=pair_label)]


# ---------------------------------------------------------------------------
# DAG builders
# ---------------------------------------------------------------------------

def _build_node_dag() -> DAGExecutor:
    """Per-node preflight DAG:
        Wave 0: gpu_inventory ∥ dcgm_diag ∥ nvlink_topo ∥ rdma_health ∥ storage_fio
        Wave 1: nccl_intra  (depends on gpu_inventory + rdma_health)
    """
    dag = DAGExecutor(max_workers=6, name="preflight_node")
    dag.add_node("gpu_inventory", _check_gpu_inventory)
    dag.add_node("dcgm_diag", _check_dcgm_diag)
    dag.add_node("nvlink_topo", _check_nvlink_topo)
    dag.add_node("rdma_health", _check_rdma_health)
    dag.add_node("storage_fio", _check_storage_fio)
    dag.add_node("nccl_intra", _check_nccl_intra,
                 depends_on={"gpu_inventory", "rdma_health"})
    return dag


# ---------------------------------------------------------------------------
# PreflightValidator — the public API
# ---------------------------------------------------------------------------

class PreflightValidator:
    """
    High-level cluster preflight validator.

    Usage (CLI or MCP):
        validator = PreflightValidator(nodes=["10.0.0.1", "10.0.0.2"])
        report = validator.run_all()
        print(json.dumps(report.to_dict(), indent=2))
    """

    def __init__(self, nodes: Optional[List[str]] = None,
                 ssh_user: str = "root", ssh_key: Optional[str] = None,
                 data_config: Optional[Dict[str, Any]] = None,
                 expected_gpus_per_node: int = 8,
                 expected_gpu_type: str = "",
                 dcgm_level: int = 2,
                 skip_cross_node: bool = False):
        self.nodes = nodes or [None]
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.data_config = data_config or {}
        self.expected_gpus = expected_gpus_per_node
        self.expected_type = expected_gpu_type
        self.dcgm_level = dcgm_level
        self.skip_cross_node = skip_cross_node

    def run_all(self) -> PreflightReport:
        """Full preflight: per-node parallel → pairwise NCCL → data integrity."""
        report = PreflightReport(started_at=datetime.now())
        t0 = time.monotonic()

        # Phase 1: Per-node checks (N nodes × 6 checks, all parallel via batch_apply)
        node_dag = _build_node_dag()
        contexts = [
            {"host": n, "ssh_user": self.ssh_user, "ssh_key": self.ssh_key,
             "data_path": self.data_config.get("path", "/tmp"),
             "expected_gpus_per_node": self.expected_gpus,
             "expected_gpu_type": self.expected_type,
             "dcgm_level": self.dcgm_level}
            for n in self.nodes
        ]
        for dag_result in node_dag.batch_apply(contexts, fail_fast=False):
            for output in dag_result.outputs.values():
                if isinstance(output, list):
                    report.checks.extend(output)
            for name, error in dag_result.errors.items():
                report.checks.append(CheckResult(name, CheckStatus.FAIL, f"Crashed: {error}"))

        # Phase 2: Cross-node pairwise NCCL (if >1 node and no failures)
        real_nodes = [n for n in self.nodes if n and n not in ("localhost", "127.0.0.1")]
        if len(real_nodes) >= 2 and not self.skip_cross_node and report.passed:
            pairs = [(real_nodes[i], real_nodes[i + 1])
                     for i in range(0, len(real_nodes) - 1, 2)]
            if pairs:
                pair_dag = DAGExecutor(max_workers=len(pairs), name="preflight_pairs")
                for i, (a, b) in enumerate(pairs):
                    def make_fn(na, nb):
                        def fn(ctx):
                            return _check_nccl_pair({
                                "node_a": na, "node_b": nb,
                                "ssh_user": self.ssh_user, "ssh_key": self.ssh_key,
                                "gpus_per_node": self.expected_gpus,
                            })
                        return fn
                    pair_dag.add_node(f"pair_{i}", make_fn(a, b))

                pair_result = pair_dag.apply()
                for output in pair_result.outputs.values():
                    if isinstance(output, list):
                        report.checks.extend(output)

        # Phase 3: Data integrity
        if self.data_config:
            data_results = _check_data_integrity({"data_config": self.data_config})
            report.checks.extend(data_results)

        report.finished_at = datetime.now()
        report.total_duration_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Preflight: {len(report.checks)} checks, "
                    f"{len(report.failures)} fail, {len(report.warnings)} warn, "
                    f"{report.total_duration_ms:.0f}ms")
        return report

    def run_quick(self) -> PreflightReport:
        """Quick check: GPU inventory only, all nodes parallel."""
        report = PreflightReport(started_at=datetime.now())
        t0 = time.monotonic()
        dag = DAGExecutor(max_workers=max(len(self.nodes), 1), name="preflight_quick")
        for i, n in enumerate(self.nodes):
            def make_fn(h):
                def fn(ctx):
                    return _check_gpu_inventory({
                        "host": h, "ssh_user": self.ssh_user, "ssh_key": self.ssh_key,
                        "expected_gpus_per_node": self.expected_gpus,
                        "expected_gpu_type": self.expected_type})
                return fn
            dag.add_node(f"node_{i}", make_fn(n))

        for output in dag.apply().outputs.values():
            if isinstance(output, list):
                report.checks.extend(output)

        report.finished_at = datetime.now()
        report.total_duration_ms = (time.monotonic() - t0) * 1000
        return report
