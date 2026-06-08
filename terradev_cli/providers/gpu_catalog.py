#!/usr/bin/env python3
"""
GPU Type Normalization Catalog

Single source of truth for GPU type names across all providers.
Maps provider-specific names to canonical names and provides GPU specifications.

Fixes the fragmentation where:
- Yotta Labs uses: NVIDIA_H100_80G, NVIDIA_A100_80G
- DigitalOcean uses: gpu-h100x1-80gb
- AWS uses: p5.48xlarge (instance type)
- RunPod uses: H100, A100
- Vast.ai uses: RTX 4090, A100-SXM4-80GB

All normalize to canonical: H100-80GB, A100-80GB, RTX-4090, etc.
"""

from typing import Optional, Dict
from .types import GPUDescriptor, GPUVendor


# ── GPU Aliases: Provider-specific names → Canonical names ─────────────────
# Keys are case-insensitive lookup; values are canonical names from GPU_SPECS

GPU_ALIASES: Dict[str, str] = {
    # ── H100 ─────────────────────────────────────────────────────────────
    "NVIDIA_H100_SXM5_80G": "H100-80GB",
    "NVIDIA_H100_80G": "H100-80GB",
    "H100": "H100-80GB",
    "H100-80GB": "H100-80GB",
    "H100-SXM": "H100-80GB",
    "H100-SXM5": "H100-80GB",
    "H100-SXM5-80GB": "H100-80GB",
    "H100-PCIe": "H100-80GB",
    "NVIDIA_H100_PCIe_80G": "H100-80GB",
    "h100": "H100-80GB",
    "h100-80gb": "H100-80GB",
    "h100-sxm5-80gb": "H100-80GB",
    "gpu-h100x1-80gb": "H100-80GB",  # DigitalOcean
    "gpu-h100x8-640gb": "H100-80GB",  # DigitalOcean 8x

    # ── H200 ─────────────────────────────────────────────────────────────
    "NVIDIA_H200_141G": "H200-141GB",
    "H200": "H200-141GB",
    "H200-141GB": "H200-141GB",
    "h200": "H200-141GB",

    # ── B200 / B300 ─────────────────────────────────────────────────────
    "NVIDIA_B200_180G": "B200-192GB",
    "B200": "B200-192GB",
    "B200-192GB": "B200-192GB",
    "NVIDIA_B300_262G": "B300-262GB",
    "B300": "B300-262GB",
    "B300-262GB": "B300-262GB",

    # ── A100 ─────────────────────────────────────────────────────────────
    "NVIDIA_A100_SXM4_80G": "A100-80GB",
    "NVIDIA_A100_80G": "A100-80GB",
    "A100": "A100-80GB",
    "A100-80GB": "A100-80GB",
    "A100-SXM": "A100-80GB",
    "A100-SXM4": "A100-80GB",
    "A100-SXM4-80GB": "A100-80GB",
    "a100": "A100-80GB",
    "a100-80gb": "A100-80GB",
    "A100-SXM4-80G": "A100-80GB",

    # ── A100 PCIe variants ───────────────────────────────────────────────
    "NVIDIA_A100_PCIe_80G": "A100-40GB",  # PCIe variant often 40GB in practice
    "NVIDIA_A100_PCIe_40G": "A100-40GB",
    "A100-PCIe": "A100-40GB",
    "A100-PCIe-80G": "A100-40GB",
    "A100-PCIe-40G": "A100-40GB",
    "A100-40GB": "A100-40GB",
    "a100-40gb": "A100-40GB",
    "A100-40G": "A100-40GB",

    # ── A100 80GB PCIe (less common but exists) ───────────────────────────
    "A100-PCIe-80GB": "A100-80GB",

    # ── RTX 4090 ─────────────────────────────────────────────────────────
    "NVIDIA_RTX_4090_24G": "RTX-4090",
    "RTX4090": "RTX-4090",
    "RTX-4090": "RTX-4090",
    "RTX 4090": "RTX-4090",
    "rtx4090": "RTX-4090",
    "rtx-4090": "RTX-4090",
    "GeForce RTX 4090": "RTX-4090",

    # ── RTX 5090 ─────────────────────────────────────────────────────────
    "NVIDIA_RTX_5090_32G": "RTX-5090",
    "RTX5090": "RTX-5090",
    "RTX-5090": "RTX-5090",
    "RTX 5090": "RTX-5090",

    # ── RTX 3090 ─────────────────────────────────────────────────────────
    "NVIDIA_RTX_3090_24G": "RTX-3090",
    "RTX3090": "RTX-3090",
    "RTX-3090": "RTX-3090",
    "RTX 3090": "RTX-3090",
    "rtx3090": "RTX-3090",

    # ── L40S / L40 ───────────────────────────────────────────────────────
    "NVIDIA_L40S_48G": "L40S-48GB",
    "L40S": "L40S-48GB",
    "L40S-48GB": "L40S-48GB",
    "NVIDIA_L40_48G": "L40-48GB",
    "L40": "L40-48GB",
    "L40-48GB": "L40-48GB",

    # ── RTX A6000 ─────────────────────────────────────────────────────────
    "NVIDIA_RTX_A6000_48G": "RTX-A6000-48GB",
    "RTX A6000": "RTX-A6000-48GB",
    "RTX-A6000": "RTX-A6000-48GB",
    "A6000": "RTX-A6000-48GB",

    # ── RTX 6000 Ada ─────────────────────────────────────────────────────
    "NVIDIA_RTX_6000_Ada_48G": "RTX-6000-Ada-48GB",
    "RTX 6000 Ada": "RTX-6000-Ada-48GB",
    "RTX-6000-Ada": "RTX-6000-Ada-48GB",
    "RTX6000Ada": "RTX-6000-Ada-48GB",

    # ── RTX Pro 6000 ─────────────────────────────────────────────────────
    "NVIDIA_RTX_PRO_6000_96G": "RTX-Pro-6000-96GB",
    "RTX Pro 6000": "RTX-Pro-6000-96GB",
    "RTX-Pro-6000": "RTX-Pro-6000-96GB",
    "RTXPro6000": "RTX-Pro-6000-96GB",

    # ── MI300X ────────────────────────────────────────────────────────────
    "AMD_MI300X_192G": "MI300X-192GB",
    "MI300X": "MI300X-192GB",
    "MI300X-192GB": "MI300X-192GB",
    "mi300x": "MI300X-192GB",

    # ── MI250X ────────────────────────────────────────────────────────────
    "AMD_MI250X_128G": "MI250X-128GB",
    "MI250X": "MI250X-128GB",
    "MI250X-128GB": "MI250X-128GB",

    # ── V100 ─────────────────────────────────────────────────────────────
    "NVIDIA_V100_16G": "V100-16GB",
    "NVIDIA_V100_32G": "V100-32GB",
    "V100": "V100-16GB",
    "V100-16GB": "V100-16GB",
    "V100-32GB": "V100-32GB",
    "Tesla V100": "V100-16GB",
    "v100": "V100-16GB",

    # ── T4 ───────────────────────────────────────────────────────────────
    "NVIDIA_T4_16G": "T4-16GB",
    "T4": "T4-16GB",
    "T4-16GB": "T4-16GB",
    "Tesla T4": "T4-16GB",
    "t4": "T4-16GB",

    # ── A10G ─────────────────────────────────────────────────────────────
    "NVIDIA_A10G_24G": "A10G-24GB",
    "A10G": "A10G-24GB",
    "A10G-24GB": "A10G-24GB",
    "a10g": "A10G-24GB",

    # ── A10 ───────────────────────────────────────────────────────────────
    "NVIDIA_A10_24G": "A10-24GB",
    "A10": "A10-24GB",
    "A10-24GB": "A10-24GB",

    # ── Gaudi 2 ───────────────────────────────────────────────────────────
    "Gaudi2": "Gaudi2-96GB",
    "Intel Gaudi2": "Gaudi2-96GB",
    "Gaudi2-96GB": "Gaudi2-96GB",

    # ── Gaudi 3 ───────────────────────────────────────────────────────────
    "Gaudi3": "Gaudi3-128GB",
    "Intel Gaudi3": "Gaudi3-128GB",
    "Gaudi3-128GB": "Gaudi3-128GB",

    # ── AWS Instance Types (map to GPU type) ───────────────────────────────
    "p5.48xlarge": "H100-80GB",
    "p5e.48xlarge": "H100-80GB",
    "p4d.24xlarge": "A100-80GB",
    "p4de.24xlarge": "A100-80GB",
    "p3.2xlarge": "V100-16GB",
    "p3.8xlarge": "V100-16GB",
    "p3.16xlarge": "V100-16GB",
    "g4dn.xlarge": "T4-16GB",
    "g4dn.2xlarge": "T4-16GB",
    "g4dn.4xlarge": "T4-16GB",
    "g4dn.8xlarge": "T4-16GB",
    "g4dn.12xlarge": "T4-16GB",
    "g4dn.16xlarge": "T4-16GB",
    "g5.xlarge": "A10G-24GB",
    "g5.2xlarge": "A10G-24GB",
    "g5.4xlarge": "A10G-24GB",
    "g5.8xlarge": "A10G-24GB",
    "g5.12xlarge": "A10G-24GB",
    "g5.16xlarge": "A10G-24GB",
    "g5.24xlarge": "A10G-24GB",
    "g5.48xlarge": "A10G-24GB",
    "g6.xlarge": "L4-24GB",
    "g6.2xlarge": "L4-24GB",
    "g6.4xlarge": "L4-24GB",
    "g6.8xlarge": "L4-24GB",
    "g6.12xlarge": "L4-24GB",
    "g6.16xlarge": "L4-24GB",
    "g6.24xlarge": "L4-24GB",
    "g6.48xlarge": "L4-24GB",

    # ── GCP Instance Types ─────────────────────────────────────────────────
    "a2-highgpu-1g": "A100-80GB",
    "a2-highgpu-2g": "A100-80GB",
    "a2-highgpu-4g": "A100-80GB",
    "a2-highgpu-8g": "A100-80GB",
    "a2-megagpu-16g": "A100-80GB",
    "a2-ultragpu-1g": "A100-80GB",
    "a2-ultragpu-2g": "A100-80GB",
    "a2-ultragpu-4g": "A100-80GB",
    "a2-ultragpu-8g": "A100-80GB",
    "g2-standard-4": "L4-24GB",
    "g2-standard-8": "L4-24GB",
    "g2-standard-16": "L4-24GB",
    "g2-standard-32": "L4-24GB",
    "g2-standard-96": "L4-24GB",
    "l4": "L4-24GB",

    # ── Azure Instance Types ───────────────────────────────────────────────
    "Standard_ND96amsr_A100_v4": "A100-80GB",
    "Standard_ND96asr_v4": "A100-80GB",
    "Standard_ND96rsr_v2": "V100-32GB",
    "Standard_NC6s_v3": "V100-16GB",
    "Standard_NC12s_v3": "V100-16GB",
    "Standard_NC24s_v3": "V100-16GB",
    "Standard_NC24ads_A100_v4": "A100-80GB",
    "Standard_ND96isr_H100_v5": "H100-80GB",
    "Standard_ND96amsr_H100_v5": "H100-80GB",

    # ── L4 ───────────────────────────────────────────────────────────────
    "NVIDIA_L4_24G": "L4-24GB",
    "L4": "L4-24GB",
    "L4-24GB": "L4-24GB",
    "l4": "L4-24GB",

    # ── Legacy / Consumer ─────────────────────────────────────────────────
    "RTX3080": "RTX-3080-10GB",
    "RTX 3080": "RTX-3080-10GB",
    "RTX3070": "RTX-3070-8GB",
    "RTX 3070": "RTX-3070-8GB",
    "RTX3060": "RTX-3060-12GB",
    "RTX 3060": "RTX-3060-12GB",
    "RTX2080Ti": "RTX-2080Ti-11GB",
    "RTX 2080 Ti": "RTX-2080Ti-11GB",
}


# ── GPU Specifications: Canonical names → GPUDescriptor ───────────────────
# Source: NVIDIA official specs, AMD official specs, provider documentation

GPU_SPECS: Dict[str, GPUDescriptor] = {
    # ── NVIDIA Hopper ────────────────────────────────────────────────────
    "H100-80GB": GPUDescriptor(
        name="H100-80GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=80,
        tflops_bf16=1979.0,
        tflops_fp16=989.5,
        tflops_fp32=67.0,
        bandwidth_gb_s=3350,
        nvlink=True,
        compute_capability="9.0",
    ),
    "H200-141GB": GPUDescriptor(
        name="H200-141GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=141,
        tflops_bf16=1979.0,
        tflops_fp16=989.5,
        tflops_fp32=67.0,
        bandwidth_gb_s=4800,
        nvlink=True,
        compute_capability="9.0",
    ),
    "B200-192GB": GPUDescriptor(
        name="B200-192GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=192,
        tflops_bf16=4500.0,
        tflops_fp16=2250.0,
        tflops_fp32=150.0,
        bandwidth_gb_s=8000,
        nvlink=True,
        compute_capability="10.0",
    ),
    "B300-262GB": GPUDescriptor(
        name="B300-262GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=262,
        tflops_bf16=6000.0,
        tflops_fp16=3000.0,
        tflops_fp32=200.0,
        bandwidth_gb_s=10000,
        nvlink=True,
        compute_capability="10.0",
    ),

    # ── NVIDIA Ampere ─────────────────────────────────────────────────────
    "A100-80GB": GPUDescriptor(
        name="A100-80GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=80,
        tflops_bf16=312.0,
        tflops_fp16=624.0,
        tflops_fp32=19.5,
        bandwidth_gb_s=2039,
        nvlink=True,
        compute_capability="8.0",
    ),
    "A100-40GB": GPUDescriptor(
        name="A100-40GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=40,
        tflops_bf16=312.0,
        tflops_fp16=624.0,
        tflops_fp32=19.5,
        bandwidth_gb_s=1555,
        nvlink=True,
        compute_capability="8.0",
    ),

    # ── NVIDIA Ada Lovelace ───────────────────────────────────────────────
    "RTX-4090": GPUDescriptor(
        name="RTX-4090",
        vendor=GPUVendor.NVIDIA,
        vram_gb=24,
        tflops_fp16=82.6,
        tflops_fp32=82.6,
        bandwidth_gb_s=1008,
        nvlink=False,
        compute_capability="8.9",
    ),
    "RTX-5090": GPUDescriptor(
        name="RTX-5090",
        vendor=GPUVendor.NVIDIA,
        vram_gb=32,
        tflops_fp16=150.0,
        tflops_fp32=150.0,
        bandwidth_gb_s=1800,
        nvlink=False,
        compute_capability="10.0",
    ),
    "RTX-3090": GPUDescriptor(
        name="RTX-3090",
        vendor=GPUVendor.NVIDIA,
        vram_gb=24,
        tflops_fp16=35.6,
        tflops_fp32=35.6,
        bandwidth_gb_s=936,
        nvlink=False,
        compute_capability="8.6",
    ),
    "L40S-48GB": GPUDescriptor(
        name="L40S-48GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=48,
        tflops_fp16=91.6,
        tflops_fp32=91.6,
        bandwidth_gb_s=864,
        nvlink=False,
        compute_capability="8.9",
    ),
    "L40-48GB": GPUDescriptor(
        name="L40-48GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=48,
        tflops_fp16=91.6,
        tflops_fp32=91.6,
        bandwidth_gb_s=864,
        nvlink=False,
        compute_capability="8.9",
    ),
    "RTX-A6000-48GB": GPUDescriptor(
        name="RTX-A6000-48GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=48,
        tflops_fp16=38.7,
        tflops_fp32=38.7,
        bandwidth_gb_s=768,
        nvlink=False,
        compute_capability="8.6",
    ),
    "RTX-6000-Ada-48GB": GPUDescriptor(
        name="RTX-6000-Ada-48GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=48,
        tflops_fp16=91.6,
        tflops_fp32=91.6,
        bandwidth_gb_s=864,
        nvlink=False,
        compute_capability="8.9",
    ),
    "RTX-Pro-6000-96GB": GPUDescriptor(
        name="RTX-Pro-6000-96GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=96,
        tflops_fp16=91.6,
        tflops_fp32=91.6,
        bandwidth_gb_s=864,
        nvlink=False,
        compute_capability="8.9",
    ),
    "L4-24GB": GPUDescriptor(
        name="L4-24GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=24,
        tflops_fp16=30.3,
        tflops_fp32=30.3,
        bandwidth_gb_s=300,
        nvlink=False,
        compute_capability="8.9",
    ),

    # ── AMD CDNA 3 ───────────────────────────────────────────────────────
    "MI300X-192GB": GPUDescriptor(
        name="MI300X-192GB",
        vendor=GPUVendor.AMD,
        vram_gb=192,
        tflops_bf16=1307.4,
        tflops_fp16=2614.8,
        tflops_fp32=163.4,
        bandwidth_gb_s=5300,
        nvlink=False,
        compute_capability="9.4",
    ),
    "MI250X-128GB": GPUDescriptor(
        name="MI250X-128GB",
        vendor=GPUVendor.AMD,
        vram_gb=128,
        tflops_bf16=362.0,
        tflops_fp16=724.0,
        tflops_fp32=45.3,
        bandwidth_gb_s=3280,
        nvlink=False,
        compute_capability="9.0",
    ),

    # ── NVIDIA Volta ─────────────────────────────────────────────────────
    "V100-16GB": GPUDescriptor(
        name="V100-16GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=16,
        tflops_fp16=125.0,
        tflops_fp32=15.7,
        bandwidth_gb_s=900,
        nvlink=True,
        compute_capability="7.0",
    ),
    "V100-32GB": GPUDescriptor(
        name="V100-32GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=32,
        tflops_fp16=125.0,
        tflops_fp32=15.7,
        bandwidth_gb_s=900,
        nvlink=True,
        compute_capability="7.0",
    ),

    # ── NVIDIA Turing ─────────────────────────────────────────────────────
    "T4-16GB": GPUDescriptor(
        name="T4-16GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=16,
        tflops_fp16=65.0,
        tflops_fp32=8.1,
        bandwidth_gb_s=320,
        nvlink=False,
        compute_capability="7.5",
    ),
    "A10G-24GB": GPUDescriptor(
        name="A10G-24GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=24,
        tflops_fp16=125.0,
        tflops_fp32=31.2,
        bandwidth_gb_s=600,
        nvlink=False,
        compute_capability="8.6",
    ),
    "A10-24GB": GPUDescriptor(
        name="A10-24GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=24,
        tflops_fp16=125.0,
        tflops_fp32=31.2,
        bandwidth_gb_s=600,
        nvlink=False,
        compute_capability="8.6",
    ),

    # ── Intel Gaudi ───────────────────────────────────────────────────────
    "Gaudi2-96GB": GPUDescriptor(
        name="Gaudi2-96GB",
        vendor=GPUVendor.INTEL,
        vram_gb=96,
        tflops_bf16=480.0,
        tflops_fp16=960.0,
        tflops_fp32=60.0,
        bandwidth_gb_s=2400,
        nvlink=False,
        compute_capability="N/A",
    ),
    "Gaudi3-128GB": GPUDescriptor(
        name="Gaudi3-128GB",
        vendor=GPUVendor.INTEL,
        vram_gb=128,
        tflops_bf16=1000.0,
        tflops_fp16=2000.0,
        tflops_fp32=125.0,
        bandwidth_gb_s=3600,
        nvlink=False,
        compute_capability="N/A",
    ),

    # ── Legacy Consumer ───────────────────────────────────────────────────
    "RTX-3080-10GB": GPUDescriptor(
        name="RTX-3080-10GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=10,
        tflops_fp16=29.8,
        tflops_fp32=29.8,
        bandwidth_gb_s=760,
        nvlink=False,
        compute_capability="8.6",
    ),
    "RTX-3070-8GB": GPUDescriptor(
        name="RTX-3070-8GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=8,
        tflops_fp16=20.4,
        tflops_fp32=20.4,
        bandwidth_gb_s=448,
        nvlink=False,
        compute_capability="8.6",
    ),
    "RTX-3060-12GB": GPUDescriptor(
        name="RTX-3060-12GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=12,
        tflops_fp16=13.0,
        tflops_fp32=13.0,
        bandwidth_gb_s=360,
        nvlink=False,
        compute_capability="8.6",
    ),
    "RTX-2080Ti-11GB": GPUDescriptor(
        name="RTX-2080Ti-11GB",
        vendor=GPUVendor.NVIDIA,
        vram_gb=11,
        tflops_fp16=26.9,
        tflops_fp32=13.4,
        bandwidth_gb_s=616,
        nvlink=False,
        compute_capability="7.5",
    ),
}


def normalize(provider_gpu_name: str) -> Optional[GPUDescriptor]:
    """
    Normalize a provider-specific GPU name to a canonical GPUDescriptor.

    Lookup is case-insensitive and tries multiple variations:
    - Exact match
    - Uppercase match
    - Lowercase match
    - With/without spaces and hyphens

    Args:
        provider_gpu_name: GPU name from provider (e.g., "NVIDIA_H100_80G", "gpu-h100x1-80gb")

    Returns:
        GPUDescriptor if found, None if unknown
    """
    if not provider_gpu_name:
        return None

    # Try exact match first
    canonical = GPU_ALIASES.get(provider_gpu_name)
    if canonical:
        return GPU_SPECS.get(canonical)

    # Try uppercase
    canonical = GPU_ALIASES.get(provider_gpu_name.upper())
    if canonical:
        return GPU_SPECS.get(canonical)

    # Try lowercase
    canonical = GPU_ALIASES.get(provider_gpu_name.lower())
    if canonical:
        return GPU_SPECS.get(canonical)

    # Try with common variations (spaces, hyphens, underscores)
    variants = [
        provider_gpu_name.replace("_", "-"),
        provider_gpu_name.replace("-", "_"),
        provider_gpu_name.replace(" ", ""),
        provider_gpu_name.replace(" ", "_"),
        provider_gpu_name.replace(" ", "-"),
    ]
    for variant in variants:
        canonical = GPU_ALIASES.get(variant)
        if canonical:
            return GPU_SPECS.get(canonical)
        canonical = GPU_ALIASES.get(variant.upper())
        if canonical:
            return GPU_SPECS.get(canonical)
        canonical = GPU_ALIASES.get(variant.lower())
        if canonical:
            return GPU_SPECS.get(canonical)

    # Not found
    return None


def get_canonical_name(provider_gpu_name: str) -> Optional[str]:
    """
    Get the canonical name for a provider GPU name (without full descriptor).

    Useful for logging, UI display, and API responses where only the name is needed.
    """
    descriptor = normalize(provider_gpu_name)
    return descriptor.name if descriptor else None


def list_all_canonical_gpus() -> list:
    """Return list of all canonical GPU names."""
    return sorted(GPU_SPECS.keys())


def list_providers_for_gpu(canonical_name: str) -> list:
    """
    Return list of provider-specific names that map to a canonical GPU.

    Useful for debugging and documentation.
    """
    return [k for k, v in GPU_ALIASES.items() if v == canonical_name]
