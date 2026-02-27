#!/usr/bin/env python3
"""
Terradev GPU Topology Manager — v3.1.0

Abstracts the full GPU infrastructure stack for distributed LLM inference:
  1. NUMA-aware GPU-NIC pairing
  2. PCIe switch topology detection
  3. SR-IOV VF allocation
  4. RDMA configuration
  5. Kubelet Topology Manager configuration
  6. DRA / DRANET resource claim generation (K8s 1.31+)
  7. GPU-NIC pairing optimization

References:
  - Dennis Kennetz, "Hidden Infrastructure Challenges in Distributed
    LLM Inference on Kubernetes" (Feb 2025)
  - KEP-4381: DRA pcieRoot attribute
  - DRANET: Dynamic Resource Allocation for Networking (kubernetes-sigs/dranet)
"""

import json
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PCIeLocality(Enum):
    """PCIe locality levels (best -> worst)"""
    PIX = "PIX"    # Same PCIe switch -- optimal
    PXB = "PXB"    # Same root complex, different switch -- acceptable
    PHB = "PHB"    # Same NUMA node, different root complex
    SYS = "SYS"    # Cross-socket / cross-NUMA -- worst case


@dataclass
class GPUDevice:
    """Represents a GPU with full topology metadata"""
    index: int
    name: str
    pci_bus_id: str
    numa_node: int
    pcie_root: str
    pcie_switch: str
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    utilization_pct: int = 0
    xcd_count: int = 0              # intra-GPU NUMA domains (MI300X=8, H200=1)
    gpu_arch: str = ""              # e.g. "mi300x", "h100", "h200"


class IntraGPUNUMALocality(Enum):
    """Intra-GPU NUMA locality levels (arXiv:2511.02132, Nov 2025).

    AMD MI300X has 8 XCDs (Accelerated Compute Dies), each with its own
    L2 cache. Attention kernels that account for XCD boundaries see L2
    cache hit rates jump from 43% to 92%.

    NVIDIA H200 has unified HBM3e but distinct memory controller domains.
    """
    SAME_XCD = "same_xcd"       # Same XCD / L2 domain — optimal (92% L2 hit)
    ADJACENT_XCD = "adj_xcd"    # Adjacent XCD pair sharing interconnect
    REMOTE_XCD = "remote_xcd"   # Cross-XCD — worst intra-GPU locality (43% L2 hit)
    UNIFIED = "unified"         # No intra-GPU NUMA (single die / unified L2)


@dataclass
class XCDDomain:
    """Models a single XCD (Accelerated Compute Die) within a GPU.

    MI300X: 8 XCDs, each with 16 CUs, 4 MB L2, sharing 16 GB HBM3
    H200:   1 unified die (no intra-GPU NUMA), 80 SM, 50 MB L2
    """
    xcd_id: int
    gpu_index: int
    compute_units: int = 0      # CUs (AMD) or SMs (NVIDIA)
    l2_cache_mb: float = 0.0
    hbm_slice_gb: float = 0.0   # HBM3 slice attached to this XCD
    adjacent_xcds: List[int] = field(default_factory=list)


@dataclass
class IntraGPUTopology:
    """Full intra-GPU NUMA topology for a single GPU.

    References:
      - arXiv:2511.02132 "Optimizing Attention on GPUs by Exploiting
        GPU Architectural NUMA" (Nov 2025)
      - AMD AITER library for XCD-aware attention kernels
    """
    gpu_index: int
    gpu_arch: str                           # "mi300x", "h200", "h100", etc.
    xcd_count: int
    xcds: List[XCDDomain] = field(default_factory=list)
    total_l2_mb: float = 0.0
    total_hbm_gb: float = 0.0
    has_intra_numa: bool = False            # True if XCD count > 1

    def classify_xcd_locality(self, xcd_a: int, xcd_b: int) -> IntraGPUNUMALocality:
        """Classify intra-GPU locality between two XCDs."""
        if not self.has_intra_numa:
            return IntraGPUNUMALocality.UNIFIED
        if xcd_a == xcd_b:
            return IntraGPUNUMALocality.SAME_XCD
        # Check adjacency
        for xcd in self.xcds:
            if xcd.xcd_id == xcd_a and xcd_b in xcd.adjacent_xcds:
                return IntraGPUNUMALocality.ADJACENT_XCD
        return IntraGPUNUMALocality.REMOTE_XCD


# ── Known GPU architectures with intra-GPU NUMA ──

_GPU_ARCH_DB: Dict[str, Dict[str, Any]] = {
    "mi300x": {
        "xcd_count": 8,
        "cus_per_xcd": 16,
        "l2_per_xcd_mb": 4.0,
        "hbm_per_xcd_gb": 24.0,  # 192 GB / 8 XCDs
        "total_hbm_gb": 192.0,
        "total_l2_mb": 32.0,
        "adjacency": {
            0: [1, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [2, 7],
            4: [0, 5], 5: [1, 4, 6], 6: [2, 5, 7], 7: [3, 6],
        },
    },
    "mi300a": {
        "xcd_count": 6,
        "cus_per_xcd": 16,
        "l2_per_xcd_mb": 4.0,
        "hbm_per_xcd_gb": 21.33,  # 128 GB / 6 XCDs
        "total_hbm_gb": 128.0,
        "total_l2_mb": 24.0,
        "adjacency": {
            0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
            3: [0, 4], 4: [1, 3, 5], 5: [2, 4],
        },
    },
    "h200": {
        "xcd_count": 1,
        "cus_per_xcd": 132,  # SMs
        "l2_per_xcd_mb": 50.0,
        "hbm_per_xcd_gb": 141.0,
        "total_hbm_gb": 141.0,
        "total_l2_mb": 50.0,
        "adjacency": {},
    },
    "h100": {
        "xcd_count": 1,
        "cus_per_xcd": 132,
        "l2_per_xcd_mb": 50.0,
        "hbm_per_xcd_gb": 80.0,
        "total_hbm_gb": 80.0,
        "total_l2_mb": 50.0,
        "adjacency": {},
    },
}


def detect_gpu_arch(gpu_name: str) -> str:
    """Heuristic: map GPU name string to architecture key."""
    name_lower = gpu_name.lower()
    if "mi300x" in name_lower or "instinct mi300x" in name_lower:
        return "mi300x"
    if "mi300a" in name_lower or "instinct mi300a" in name_lower:
        return "mi300a"
    if "h200" in name_lower:
        return "h200"
    if "h100" in name_lower:
        return "h100"
    if "a100" in name_lower:
        return "a100"
    return "unknown"


def build_intra_gpu_topology(gpu: GPUDevice) -> IntraGPUTopology:
    """Build the intra-GPU NUMA topology for a GPU based on its architecture."""
    arch = gpu.gpu_arch or detect_gpu_arch(gpu.name)
    arch_info = _GPU_ARCH_DB.get(arch)

    if not arch_info:
        return IntraGPUTopology(
            gpu_index=gpu.index, gpu_arch=arch, xcd_count=1, has_intra_numa=False,
        )

    xcd_count = arch_info["xcd_count"]
    adjacency = arch_info.get("adjacency", {})
    xcds = []
    for i in range(xcd_count):
        xcds.append(XCDDomain(
            xcd_id=i,
            gpu_index=gpu.index,
            compute_units=arch_info["cus_per_xcd"],
            l2_cache_mb=arch_info["l2_per_xcd_mb"],
            hbm_slice_gb=arch_info["hbm_per_xcd_gb"],
            adjacent_xcds=adjacency.get(i, []),
        ))

    return IntraGPUTopology(
        gpu_index=gpu.index,
        gpu_arch=arch,
        xcd_count=xcd_count,
        xcds=xcds,
        total_l2_mb=arch_info["total_l2_mb"],
        total_hbm_gb=arch_info["total_hbm_gb"],
        has_intra_numa=xcd_count > 1,
    )


def generate_xcd_aware_env(topo: IntraGPUTopology) -> Dict[str, str]:
    """Generate XCD-aware environment variables for attention kernels.

    For MI300X: enables AMD AITER library's XCD-aware flash attention.
    For NVIDIA: no-op (unified L2, no intra-GPU NUMA).
    """
    env: Dict[str, str] = {}

    if not topo.has_intra_numa:
        return env

    if topo.gpu_arch in ("mi300x", "mi300a"):
        # AMD AITER: XCD-aware attention kernel configuration
        env["AITER_XCD_COUNT"] = str(topo.xcd_count)
        env["AITER_XCD_AWARE_ATTENTION"] = "1"
        env["AITER_L2_PARTITION"] = "1"  # partition L2 by XCD
        env["AITER_KV_CACHE_XCD_PIN"] = "1"  # pin KV cache slices to XCDs
        # NCCL intra-GPU P2P: prefer adjacent XCDs
        env["NCCL_INTRA_GPU_NUMA"] = "1"
        env["HIP_VISIBLE_DEVICES_XCD_MASK"] = ",".join(
            str(x.xcd_id) for x in topo.xcds
        )
        # CK (Composable Kernel) tuning for MI300X
        env["CK_BLOCK_MAPPING_POLICY"] = "xcd_aware"

    return env


@dataclass
class NICDevice:
    """Represents a NIC (physical or VF) with topology metadata"""
    name: str
    pci_bus_id: str
    numa_node: int
    pcie_root: str
    pcie_switch: str
    rdma_capable: bool = False
    link_speed_gbps: int = 0
    sriov_capable: bool = False
    sriov_total_vfs: int = 0
    sriov_active_vfs: int = 0
    driver: str = ""


@dataclass
class GPUNICPair:
    """An optimally paired GPU and NIC"""
    gpu: GPUDevice
    nic: NICDevice
    locality: PCIeLocality
    rdma_path: str = ""


@dataclass
class NodeTopology:
    """Full topology map for a single node"""
    hostname: str
    numa_nodes: int
    gpus: List[GPUDevice] = field(default_factory=list)
    nics: List[NICDevice] = field(default_factory=list)
    pairs: List[GPUNICPair] = field(default_factory=list)
    topology_manager_policy: str = "none"


# ---------------------------------------------------------------------------
# 1. NUMA-aware GPU-NIC pairing
# ---------------------------------------------------------------------------

class NUMADetector:
    """Detect NUMA topology on a Linux node"""

    @staticmethod
    def detect_numa_nodes() -> int:
        """Return number of NUMA nodes on this machine"""
        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "NUMA node(s):" in line:
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        return 1

    @staticmethod
    def get_device_numa(pci_bus_id: str) -> int:
        """Get the NUMA node for a PCI device"""
        numa_path = Path(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node")
        try:
            val = int(numa_path.read_text().strip())
            return val if val >= 0 else 0
        except Exception:
            return 0

    @staticmethod
    def get_numa_cpus(numa_node: int) -> List[int]:
        """Get CPU list for a NUMA node"""
        cpu_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
        try:
            raw = cpu_path.read_text().strip()
            cpus = []
            for part in raw.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    cpus.extend(range(int(lo), int(hi) + 1))
                else:
                    cpus.append(int(part))
            return cpus
        except Exception:
            return []


# ---------------------------------------------------------------------------
# 2. PCIe switch topology detection
# ---------------------------------------------------------------------------

class PCIeTopologyDetector:
    """Detect PCIe switch hierarchy using sysfs"""

    @staticmethod
    def get_pcie_root(pci_bus_id: str) -> str:
        """Determine the PCIe root complex for a device"""
        device_path = Path(f"/sys/bus/pci/devices/{pci_bus_id}")
        try:
            real_path = device_path.resolve()
            parts = str(real_path).split("/")
            for part in parts:
                if part.startswith("pci"):
                    return part
        except Exception:
            pass
        return f"pci_unknown_{pci_bus_id[:7]}"

    @staticmethod
    def get_pcie_switch(pci_bus_id: str) -> str:
        """Determine the PCIe switch a device sits behind"""
        device_path = Path(f"/sys/bus/pci/devices/{pci_bus_id}")
        try:
            real_path = device_path.resolve()
            parts = str(real_path).split("/")
            for i, part in enumerate(parts):
                if part == pci_bus_id and i >= 2:
                    return parts[i - 1]
        except Exception:
            pass
        return f"sw_unknown_{pci_bus_id[:7]}"

    @staticmethod
    def classify_locality(gpu: GPUDevice, nic: NICDevice) -> PCIeLocality:
        """Classify the PCIe locality between a GPU and NIC"""
        if gpu.pcie_switch == nic.pcie_switch:
            return PCIeLocality.PIX
        if gpu.pcie_root == nic.pcie_root:
            return PCIeLocality.PXB
        if gpu.numa_node == nic.numa_node:
            return PCIeLocality.PHB
        return PCIeLocality.SYS

    @staticmethod
    def detect_gpus() -> List[GPUDevice]:
        """Detect GPUs via nvidia-smi with topology info"""
        gpus = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return gpus

            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                idx = int(parts[0])
                pci_id = parts[2].lower()
                # Normalise PCI bus ID to sysfs format (0000:XX:XX.X)
                if not pci_id.startswith("0000:"):
                    pci_id = f"0000:{pci_id}"

                gpu = GPUDevice(
                    index=idx,
                    name=parts[1],
                    pci_bus_id=pci_id,
                    numa_node=NUMADetector.get_device_numa(pci_id),
                    pcie_root=PCIeTopologyDetector.get_pcie_root(pci_id),
                    pcie_switch=PCIeTopologyDetector.get_pcie_switch(pci_id),
                    vram_total_mb=int(float(parts[3])),
                    vram_used_mb=int(float(parts[4])),
                    utilization_pct=int(float(parts[5])),
                )
                gpus.append(gpu)
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
        return gpus


# ---------------------------------------------------------------------------
# 3. SR-IOV VF allocation
# ---------------------------------------------------------------------------

class SRIOVManager:
    """Manage SR-IOV Virtual Functions on RDMA NICs"""

    @staticmethod
    def detect_sriov_nics() -> List[NICDevice]:
        """Detect NICs with SR-IOV capability"""
        nics = []
        net_path = Path("/sys/class/net")
        if not net_path.exists():
            return nics

        for iface in net_path.iterdir():
            try:
                device_link = iface / "device"
                if not device_link.exists():
                    continue

                sriov_totalvfs = device_link / "sriov_totalvfs"
                sriov_numvfs = device_link / "sriov_numvfs"
                total_vfs = 0
                active_vfs = 0
                sriov_capable = False

                if sriov_totalvfs.exists():
                    total_vfs = int(sriov_totalvfs.read_text().strip())
                    sriov_capable = total_vfs > 0
                if sriov_numvfs.exists():
                    active_vfs = int(sriov_numvfs.read_text().strip())

                pci_bus_id = device_link.resolve().name

                rdma_capable = Path(
                    f"/sys/bus/pci/devices/{pci_bus_id}/infiniband"
                ).exists()

                driver = ""
                driver_link = device_link / "driver"
                if driver_link.exists():
                    driver = driver_link.resolve().name

                link_speed = 0
                speed_file = iface / "speed"
                if speed_file.exists():
                    try:
                        link_speed = int(speed_file.read_text().strip()) // 1000
                    except (ValueError, OSError):
                        pass

                nic = NICDevice(
                    name=iface.name,
                    pci_bus_id=pci_bus_id,
                    numa_node=NUMADetector.get_device_numa(pci_bus_id),
                    pcie_root=PCIeTopologyDetector.get_pcie_root(pci_bus_id),
                    pcie_switch=PCIeTopologyDetector.get_pcie_switch(pci_bus_id),
                    rdma_capable=rdma_capable,
                    link_speed_gbps=link_speed,
                    sriov_capable=sriov_capable,
                    sriov_total_vfs=total_vfs,
                    sriov_active_vfs=active_vfs,
                    driver=driver,
                )
                nics.append(nic)
            except Exception as e:
                logger.debug(f"Skipping interface {iface.name}: {e}")
                continue

        return nics

    @staticmethod
    def create_vfs(nic_name: str, count: int) -> bool:
        """Create SR-IOV Virtual Functions on a NIC (requires root)"""
        try:
            vf_path = Path(f"/sys/class/net/{nic_name}/device/sriov_numvfs")
            if not vf_path.exists():
                logger.error(f"SR-IOV not supported on {nic_name}")
                return False
            vf_path.write_text(str(count))
            logger.info(f"Created {count} VFs on {nic_name}")
            return True
        except PermissionError:
            logger.error(
                f"Permission denied creating VFs on {nic_name} (requires root)"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to create VFs on {nic_name}: {e}")
            return False

    @staticmethod
    def generate_sriov_network_policy(
        nic_name: str, num_vfs: int, rdma: bool = True
    ) -> Dict[str, Any]:
        """Generate SriovNetworkNodePolicy for the SR-IOV Network Operator"""
        return {
            "apiVersion": "sriovnetwork.openshift.io/v1",
            "kind": "SriovNetworkNodePolicy",
            "metadata": {
                "name": f"gpu-rdma-{nic_name}",
                "namespace": "sriov-network-operator",
            },
            "spec": {
                "nodeSelector": {
                    "feature.node.kubernetes.io/network-sriov.capable": "true"
                },
                "resourceName": f"sriov_rdma_vf_{nic_name}",
                "numVfs": num_vfs,
                "nicSelector": {"pfNames": [nic_name]},
                "deviceType": "netdevice",
                "isRdma": rdma,
            },
        }

    @staticmethod
    def generate_network_attachment(
        name: str, resource_name: str, namespace: str = "default"
    ) -> Dict[str, Any]:
        """Generate NetworkAttachmentDefinition for Multus + SR-IOV"""
        return {
            "apiVersion": "k8s.cni.cncf.io/v1",
            "kind": "NetworkAttachmentDefinition",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "config": json.dumps({
                    "cniVersion": "0.3.1",
                    "type": "sriov",
                    "name": name,
                    "ipam": {},
                    "rdma": True,
                })
            },
        }


# ---------------------------------------------------------------------------
# 4. RDMA configuration
# ---------------------------------------------------------------------------

class RDMAConfigurator:
    """Configure RDMA for GPU-NIC pairs"""

    @staticmethod
    def detect_rdma_devices() -> List[Dict[str, Any]]:
        """Detect available RDMA devices"""
        devices = []
        try:
            result = subprocess.run(
                ["rdma", "link", "show", "-j"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                devices = json.loads(result.stdout)
        except (FileNotFoundError, json.JSONDecodeError):
            rdma_path = Path("/sys/class/infiniband")
            if rdma_path.exists():
                for dev in rdma_path.iterdir():
                    devices.append({"ifname": dev.name, "port": 1, "state": "ACTIVE"})
        except Exception as e:
            logger.debug(f"RDMA detection failed: {e}")
        return devices

    @staticmethod
    def check_gpudirect_rdma() -> bool:
        """Check if GPUDirect RDMA (nvidia_peermem) is loaded"""
        try:
            result = subprocess.run(
                ["lsmod"], capture_output=True, text=True, timeout=5
            )
            return "nvidia_peermem" in result.stdout
        except Exception:
            return False

    @staticmethod
    def generate_rdma_shared_device_plugin() -> Dict[str, Any]:
        """Generate RDMA shared device plugin DaemonSet config"""
        return {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {
                "name": "rdma-shared-dp",
                "namespace": "kube-system",
            },
            "spec": {
                "selector": {"matchLabels": {"name": "rdma-shared-dp"}},
                "template": {
                    "metadata": {"labels": {"name": "rdma-shared-dp"}},
                    "spec": {
                        "hostNetwork": True,
                        "containers": [
                            {
                                "name": "rdma-shared-dp",
                                "image": "ghcr.io/mellanox/k8s-rdma-shared-dev-plugin:latest",
                                "securityContext": {"privileged": True},
                                "volumeMounts": [
                                    {
                                        "name": "device-plugin",
                                        "mountPath": "/var/lib/kubelet/device-plugins",
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "device-plugin",
                                "hostPath": {
                                    "path": "/var/lib/kubelet/device-plugins"
                                },
                            }
                        ],
                    },
                },
            },
        }

    @staticmethod
    def generate_nccl_env(pair: GPUNICPair, gpu: Optional[GPUDevice] = None) -> Dict[str, str]:
        """Generate NCCL environment variables for optimal GPU-NIC pairing.

        If gpu is provided, also generates XCD-aware env vars for MI300X.
        """
        env = {
            "NCCL_IB_HCA": pair.nic.name,
            "NCCL_SOCKET_IFNAME": pair.nic.name,
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_NET_GDR_LEVEL": (
                "PIX" if pair.locality == PCIeLocality.PIX else "PHB"
            ),
            "NCCL_P2P_LEVEL": pair.locality.value,
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_READ": "1",
        }
        if pair.locality == PCIeLocality.PIX:
            env["NCCL_TOPO_DUMP_FILE"] = "/tmp/nccl_topo.xml"

        # XCD-aware env vars for intra-GPU NUMA
        if gpu:
            intra_topo = build_intra_gpu_topology(gpu)
            xcd_env = generate_xcd_aware_env(intra_topo)
            env.update(xcd_env)

        return env


# ---------------------------------------------------------------------------
# 5. Kubelet Topology Manager configuration
# ---------------------------------------------------------------------------

class TopologyManagerConfigurator:
    """Generate kubelet Topology Manager configuration"""

    @staticmethod
    def generate_kubelet_config(
        policy: str = "restricted",
        scope: str = "container",
        prefer_closest_numa: bool = True,
    ) -> Dict[str, Any]:
        """Generate kubelet configuration for topology-aware scheduling.

        Policies:
          none       -- no topology alignment (default K8s)
          best-effort -- prefer aligned, allow misaligned
          restricted -- force aligned if possible, allow full-node requests
          single-numa-node -- strict single-NUMA (rejects cross-NUMA)
        """
        return {
            "apiVersion": "kubelet.config.k8s.io/v1beta1",
            "kind": "KubeletConfiguration",
            "topologyManagerPolicy": policy,
            "topologyManagerScope": scope,
            "topologyManagerPolicyOptions": {
                "prefer-closest-numa-nodes": str(prefer_closest_numa).lower(),
            },
            "cpuManagerPolicy": "static",
            "cpuManagerPolicyOptions": {
                "full-pcpus-only": "true",
            },
            "memoryManagerPolicy": "Static",
            "reservedMemory": [
                {"numaNode": 0, "limits": {"memory": "1Gi"}},
            ],
        }

    @staticmethod
    def generate_kubelet_patch(policy: str = "restricted") -> str:
        """Generate a JSON patch for kubelet-config ConfigMap"""
        patch = {
            "topologyManagerPolicy": policy,
            "topologyManagerScope": "container",
            "topologyManagerPolicyOptions": {
                "prefer-closest-numa-nodes": "true",
            },
            "cpuManagerPolicy": "static",
        }
        return json.dumps(patch, indent=2)

    @staticmethod
    def apply_topology_config(kubeconfig: Optional[str] = None) -> bool:
        """Apply topology manager config to the cluster kubelet-config ConfigMap"""
        env = dict(**subprocess.os.environ)
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig

        patch = TopologyManagerConfigurator.generate_kubelet_patch()
        try:
            result = subprocess.run(
                [
                    "kubectl", "patch", "configmap", "kubelet-config",
                    "-n", "kube-system",
                    "--type", "merge",
                    "-p", json.dumps({"data": {"kubelet": patch}}),
                ],
                capture_output=True, text=True, timeout=15, env=env,
            )
            if result.returncode == 0:
                logger.info("Topology Manager config applied to kubelet-config")
                return True
            else:
                logger.error(f"Failed to patch kubelet-config: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to apply topology config: {e}")
            return False


# ---------------------------------------------------------------------------
# 6. DRA / DRANET resource claim generation (K8s 1.31+)
# ---------------------------------------------------------------------------

class DRAGenerator:
    """Generate Dynamic Resource Allocation manifests for GPU-NIC pairing.

    DRA replaces device plugins with rich ResourceSlices and CEL selectors.
    DRANET (kubernetes-sigs/dranet) extends DRA to network interfaces.
    KEP-4381 proposes resource.kubernetes.io/pcieRoot for PCIe-level pairing.
    """

    @staticmethod
    def generate_device_class(
        name: str = "gpu-rdma",
        gpu_class: str = "gpu.amd.com",
        nic_class: str = "dranet",
    ) -> List[Dict[str, Any]]:
        """Generate DeviceClass manifests for GPU and RDMA NIC"""
        return [
            {
                "apiVersion": "resource.k8s.io/v1alpha3",
                "kind": "DeviceClass",
                "metadata": {"name": gpu_class.replace(".", "-")},
                "spec": {
                    "selectors": [
                        {
                            "cel": {
                                "expression": 'device.driver == "gpu.amd.com" || device.driver == "gpu.nvidia.com"'
                            }
                        }
                    ]
                },
            },
            {
                "apiVersion": "resource.k8s.io/v1alpha3",
                "kind": "DeviceClass",
                "metadata": {"name": "rdma-nic"},
                "spec": {
                    "selectors": [
                        {
                            "cel": {
                                "expression": 'device.attributes["dra.net"].rdma == true'
                            }
                        }
                    ]
                },
            },
        ]

    @staticmethod
    def generate_resource_claim_template(
        name: str = "gpu-rdma-pair",
        gpu_count: int = 1,
        nic_count: int = 1,
        pcie_aligned: bool = True,
    ) -> Dict[str, Any]:
        """Generate a ResourceClaimTemplate that pairs GPUs with RDMA NICs.

        When pcie_aligned=True, uses matchAttribute on pcieRoot to ensure
        GPU and NIC share the same PCIe root complex (PIX/PXB locality).
        """
        requests = [
            {
                "name": "gpu",
                "exactly": {
                    "deviceClassName": "gpu",
                    "count": gpu_count,
                },
            },
            {
                "name": "nic",
                "exactly": {
                    "deviceClassName": "rdma-nic",
                    "count": nic_count,
                    "selectors": [
                        {
                            "cel": {
                                "expression": 'device.attributes["dra.net"].rdma == true'
                            }
                        }
                    ],
                },
            },
        ]

        spec: Dict[str, Any] = {"devices": {"requests": requests}}

        if pcie_aligned:
            spec["devices"]["constraints"] = [
                {"matchAttribute": "resource.kubernetes.io/pcieRoot"}
            ]

        return {
            "apiVersion": "resource.k8s.io/v1alpha3",
            "kind": "ResourceClaimTemplate",
            "metadata": {"name": name},
            "spec": {"spec": spec},
        }

    @staticmethod
    def generate_pod_with_dra(
        name: str,
        image: str,
        gpu_count: int = 1,
        nic_count: int = 1,
        command: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Generate a Pod spec that uses DRA for topology-aligned GPU+NIC"""
        claim_name = f"{name}-gpu-nic"
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "containers": [
                    {
                        "name": "workload",
                        "image": image,
                        "command": command or ["sleep", "infinity"],
                        "resources": {
                            "claims": [
                                {"name": "gpu-nic", "request": "gpu"},
                                {"name": "gpu-nic", "request": "nic"},
                            ]
                        },
                    }
                ],
                "resourceClaims": [
                    {
                        "name": "gpu-nic",
                        "resourceClaimTemplateName": claim_name,
                    }
                ],
            },
        }


# ---------------------------------------------------------------------------
# 7. GPU-NIC pairing optimization
# ---------------------------------------------------------------------------

class GPUNICOptimizer:
    """Optimal GPU-NIC pairing using topology-aware matching"""

    @staticmethod
    def compute_optimal_pairs(
        gpus: List[GPUDevice], nics: List[NICDevice]
    ) -> List[GPUNICPair]:
        """Compute optimal GPU-NIC pairs minimising PCIe distance.

        Algorithm:
          1. For each GPU, score every NIC by locality (PIX=0, PXB=1, PHB=2, SYS=3)
          2. Greedy assignment: pick the best available NIC for each GPU
          3. Prefer RDMA-capable NICs
        """
        locality_score = {
            PCIeLocality.PIX: 0,
            PCIeLocality.PXB: 1,
            PCIeLocality.PHB: 2,
            PCIeLocality.SYS: 3,
        }

        available_nics = list(nics)
        pairs: List[GPUNICPair] = []

        # Sort GPUs by index for deterministic output
        for gpu in sorted(gpus, key=lambda g: g.index):
            if not available_nics:
                break

            # Score each available NIC
            scored = []
            for nic in available_nics:
                loc = PCIeTopologyDetector.classify_locality(gpu, nic)
                rdma_bonus = 0 if nic.rdma_capable else 10
                scored.append((locality_score[loc] + rdma_bonus, loc, nic))

            scored.sort(key=lambda x: x[0])
            best_score, best_loc, best_nic = scored[0]

            rdma_path = ""
            if best_nic.rdma_capable:
                if best_loc == PCIeLocality.PIX:
                    rdma_path = "GPUDirect RDMA via PIX (same PCIe switch)"
                elif best_loc == PCIeLocality.PXB:
                    rdma_path = "GPUDirect RDMA via PXB (same root complex)"
                elif best_loc == PCIeLocality.PHB:
                    rdma_path = "GPUDirect RDMA via PHB (same NUMA, cross-switch)"
                else:
                    rdma_path = "GPUDirect RDMA via SYS (cross-socket -- degraded)"

            pairs.append(GPUNICPair(
                gpu=gpu, nic=best_nic, locality=best_loc, rdma_path=rdma_path
            ))
            available_nics.remove(best_nic)

        return pairs


# ---------------------------------------------------------------------------
# Orchestrator: ties all layers together
# ---------------------------------------------------------------------------

class GPUTopologyOrchestrator:
    """High-level orchestrator that combines all topology layers.

    Usage:
        orch = GPUTopologyOrchestrator()
        report = orch.full_topology_report()
        manifests = orch.generate_k8s_manifests(gpu_count=4, topology_policy="restricted")
    """

    def __init__(self, kubeconfig: Optional[str] = None):
        self.kubeconfig = kubeconfig

    def full_topology_report(self) -> Dict[str, Any]:
        """Run full topology detection and return structured report"""
        numa_count = NUMADetector.detect_numa_nodes()
        gpus = PCIeTopologyDetector.detect_gpus()
        nics = SRIOVManager.detect_sriov_nics()
        rdma_devices = RDMAConfigurator.detect_rdma_devices()
        gpudirect = RDMAConfigurator.check_gpudirect_rdma()

        # Filter to RDMA-capable NICs for pairing
        rdma_nics = [n for n in nics if n.rdma_capable]
        all_nics_for_pairing = rdma_nics if rdma_nics else nics

        pairs = GPUNICOptimizer.compute_optimal_pairs(gpus, all_nics_for_pairing)

        # Build NUMA map
        numa_map: Dict[int, Dict[str, List[str]]] = {}
        for gpu in gpus:
            numa_map.setdefault(gpu.numa_node, {"gpus": [], "nics": []})
            numa_map[gpu.numa_node]["gpus"].append(
                f"GPU {gpu.index} ({gpu.name}) @ {gpu.pci_bus_id}"
            )
        for nic in nics:
            numa_map.setdefault(nic.numa_node, {"gpus": [], "nics": []})
            rdma_tag = " [RDMA]" if nic.rdma_capable else ""
            sriov_tag = f" [SR-IOV: {nic.sriov_total_vfs} VFs]" if nic.sriov_capable else ""
            numa_map[nic.numa_node]["nics"].append(
                f"{nic.name} @ {nic.pci_bus_id}{rdma_tag}{sriov_tag}"
            )

        # Pair summary
        pair_summary = []
        for p in pairs:
            pair_summary.append({
                "gpu": f"GPU {p.gpu.index} ({p.gpu.name})",
                "nic": p.nic.name,
                "locality": p.locality.value,
                "rdma_path": p.rdma_path,
                "optimal": p.locality in (PCIeLocality.PIX, PCIeLocality.PXB),
            })

        cross_socket_count = sum(
            1 for p in pairs if p.locality == PCIeLocality.SYS
        )

        return {
            "hostname": subprocess.run(
                ["hostname"], capture_output=True, text=True
            ).stdout.strip() if True else "unknown",
            "numa_nodes": numa_count,
            "gpu_count": len(gpus),
            "nic_count": len(nics),
            "rdma_nics": len(rdma_nics),
            "sriov_nics": len([n for n in nics if n.sriov_capable]),
            "gpudirect_rdma": gpudirect,
            "rdma_devices": len(rdma_devices),
            "numa_map": numa_map,
            "pairs": pair_summary,
            "cross_socket_pairs": cross_socket_count,
            "topology_healthy": cross_socket_count == 0,
            "recommendations": self._generate_recommendations(
                gpus, nics, pairs, gpudirect, numa_count
            ),
        }

    def _generate_recommendations(
        self,
        gpus: List[GPUDevice],
        nics: List[NICDevice],
        pairs: List[GPUNICPair],
        gpudirect: bool,
        numa_count: int,
    ) -> List[str]:
        """Generate actionable recommendations based on topology"""
        recs = []

        if not gpus:
            recs.append("No GPUs detected. Use 'terradev provision' to get cloud GPUs.")
            return recs

        cross_socket = [p for p in pairs if p.locality == PCIeLocality.SYS]
        if cross_socket:
            recs.append(
                f"{len(cross_socket)} GPU-NIC pair(s) cross NUMA sockets. "
                "Enable Topology Manager: terradev topology apply --policy restricted"
            )

        cross_switch = [p for p in pairs if p.locality == PCIeLocality.PXB]
        if cross_switch:
            recs.append(
                f"{len(cross_switch)} GPU-NIC pair(s) cross PCIe switches (PXB). "
                "For optimal RDMA, use DRA with pcieRoot constraint (K8s 1.31+): "
                "terradev topology generate-dra --pcie-aligned"
            )

        rdma_nics = [n for n in nics if n.rdma_capable]
        if gpus and not rdma_nics:
            recs.append(
                "No RDMA-capable NICs detected. Inter-node GPU communication "
                "will use TCP (10-50x slower). Consider ConnectX-7 NICs."
            )

        if rdma_nics and not gpudirect:
            recs.append(
                "RDMA NICs found but nvidia_peermem not loaded. "
                "GPUDirect RDMA disabled. Run: modprobe nvidia_peermem"
            )

        sriov_nics = [n for n in nics if n.sriov_capable]
        if sriov_nics:
            inactive = [n for n in sriov_nics if n.sriov_active_vfs == 0]
            if inactive:
                names = ", ".join(n.name for n in inactive)
                recs.append(
                    f"SR-IOV capable but no VFs active on: {names}. "
                    "Create VFs: terradev topology sriov --create-vfs"
                )

        if numa_count > 1 and len(gpus) > 1:
            recs.append(
                "Multi-NUMA node detected. Ensure kubelet Topology Manager is "
                "enabled with 'restricted' policy and prefer-closest-numa-nodes=true."
            )

        if not recs:
            recs.append("Topology looks healthy. All GPU-NIC pairs are optimally aligned.")

        return recs

    def generate_k8s_manifests(
        self,
        gpu_count: int = 1,
        nic_count: int = 1,
        topology_policy: str = "restricted",
        use_dra: bool = False,
        use_sriov: bool = True,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Generate all K8s manifests for topology-aware GPU workloads"""
        manifests: Dict[str, Any] = {}

        # Kubelet Topology Manager config
        manifests["kubelet_config"] = (
            TopologyManagerConfigurator.generate_kubelet_config(
                policy=topology_policy
            )
        )

        if use_sriov:
            # SR-IOV network policy
            nics = SRIOVManager.detect_sriov_nics()
            sriov_nics = [n for n in nics if n.sriov_capable and n.rdma_capable]
            for nic in sriov_nics:
                manifests[f"sriov_policy_{nic.name}"] = (
                    SRIOVManager.generate_sriov_network_policy(
                        nic.name, min(nic.sriov_total_vfs, gpu_count)
                    )
                )
            # Network attachment
            manifests["network_attachment"] = (
                SRIOVManager.generate_network_attachment(
                    "gpu-rdma-net", "sriov_rdma_vf", namespace
                )
            )

        # RDMA device plugin
        manifests["rdma_device_plugin"] = (
            RDMAConfigurator.generate_rdma_shared_device_plugin()
        )

        if use_dra:
            # DRA device classes
            for dc in DRAGenerator.generate_device_class():
                manifests[f"device_class_{dc['metadata']['name']}"] = dc

            # DRA resource claim template with PCIe alignment
            manifests["resource_claim_template"] = (
                DRAGenerator.generate_resource_claim_template(
                    gpu_count=gpu_count,
                    nic_count=nic_count,
                    pcie_aligned=True,
                )
            )

        # Pod spec with topology-aware resources
        pod_resources = {
            "limits": {
                "nvidia.com/gpu": str(gpu_count),
            }
        }
        if use_sriov and not use_dra:
            pod_resources["limits"]["nvidia.com/sriov-rdma-vf"] = str(nic_count)

        manifests["pod_resource_spec"] = pod_resources

        # NCCL environment variables for optimal pairing
        gpus = PCIeTopologyDetector.detect_gpus()
        nics_detected = SRIOVManager.detect_sriov_nics()
        rdma_nics = [n for n in nics_detected if n.rdma_capable]
        if gpus and rdma_nics:
            pairs = GPUNICOptimizer.compute_optimal_pairs(gpus, rdma_nics)
            if pairs:
                manifests["nccl_env"] = RDMAConfigurator.generate_nccl_env(pairs[0])

        return manifests

    def print_topology_report(self) -> None:
        """Print a human-readable topology report to stdout"""
        report = self.full_topology_report()

        print(f"\n{'='*70}")
        print(f"  Terradev GPU Topology Report — {report['hostname']}")
        print(f"{'='*70}")
        print(f"  NUMA Nodes:      {report['numa_nodes']}")
        print(f"  GPUs:            {report['gpu_count']}")
        print(f"  NICs:            {report['nic_count']}")
        print(f"  RDMA NICs:       {report['rdma_nics']}")
        print(f"  SR-IOV NICs:     {report['sriov_nics']}")
        print(f"  GPUDirect RDMA:  {'Enabled' if report['gpudirect_rdma'] else 'Disabled'}")
        print(f"  RDMA Devices:    {report['rdma_devices']}")

        # NUMA map
        print(f"\n  NUMA Topology:")
        for node_id, devices in sorted(report.get("numa_map", {}).items()):
            print(f"    NUMA Node {node_id}:")
            for g in devices.get("gpus", []):
                print(f"      GPU: {g}")
            for n in devices.get("nics", []):
                print(f"      NIC: {n}")

        # Pairs
        print(f"\n  GPU-NIC Pairs:")
        for p in report.get("pairs", []):
            status = "OK" if p["optimal"] else "WARN"
            print(
                f"    [{status}] {p['gpu']} <-> {p['nic']} "
                f"({p['locality']}) {p['rdma_path']}"
            )

        cross = report.get("cross_socket_pairs", 0)
        if cross > 0:
            print(f"\n  WARNING: {cross} cross-socket pair(s) detected!")

        # Recommendations
        print(f"\n  Recommendations:")
        for rec in report.get("recommendations", []):
            print(f"    - {rec}")

        print(f"{'='*70}\n")
