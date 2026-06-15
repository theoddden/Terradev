#!/usr/bin/env python3
"""Tests for providers/gpu_catalog.py"""

import pytest
from terradev_cli.providers.gpu_catalog import (
    normalize,
    get_canonical_name,
    list_all_canonical_gpus,
    list_providers_for_gpu,
    GPU_ALIASES,
    GPU_SPECS,
)
from terradev_cli.providers.types import GPUVendor


class TestNormalize:
    """Test normalize function"""

    def test_normalize_h100_variants(self):
        """Normalize H100 variants"""
        assert normalize("H100").name == "H100-80GB"
        assert normalize("NVIDIA_H100_80G").name == "H100-80GB"
        assert normalize("h100").name == "H100-80GB"
        assert normalize("H100-80GB").name == "H100-80GB"

    def test_normalize_a100_variants(self):
        """Normalize A100 variants"""
        assert normalize("A100").name == "A100-80GB"
        assert normalize("NVIDIA_A100_80G").name == "A100-80GB"
        assert normalize("A100-40GB").name == "A100-40GB"
        assert normalize("NVIDIA_A100_PCIe_40G").name == "A100-40GB"

    def test_normalize_rtx_variants(self):
        """Normalize RTX variants"""
        assert normalize("RTX4090").name == "RTX-4090"
        assert normalize("RTX 4090").name == "RTX-4090"
        assert normalize("RTX3090").name == "RTX-3090"

    def test_normalize_case_insensitive(self):
        """Normalize is case-insensitive"""
        assert normalize("h100").name == "H100-80GB"
        assert normalize("H100").name == "H100-80GB"
        assert normalize("a100").name == "A100-80GB"
        assert normalize("A100").name == "A100-80GB"

    def test_normalize_aws_instance_types(self):
        """Normalize AWS instance types"""
        assert normalize("p5.48xlarge").name == "H100-80GB"
        assert normalize("p4d.24xlarge").name == "A100-80GB"
        assert normalize("g4dn.xlarge").name == "T4-16GB"

    def test_normalize_gcp_instance_types(self):
        """Normalize GCP instance types"""
        assert normalize("a2-highgpu-1g").name == "A100-80GB"
        assert normalize("g2-standard-4").name == "L4-24GB"

    def test_normalize_azure_instance_types(self):
        """Normalize Azure instance types"""
        assert normalize("Standard_ND96amsr_A100_v4").name == "A100-80GB"
        assert normalize("Standard_ND96isr_H100_v5").name == "H100-80GB"

    def test_normalize_unknown_gpu(self):
        """Normalize unknown GPU returns None"""
        assert normalize("UNKNOWN_GPU") is None
        assert normalize("") is None
        assert normalize(None) is None

    def test_normalize_with_variations(self):
        """Normalize with space/hyphen/underscore variations"""
        assert normalize("RTX 4090").name == "RTX-4090"
        assert normalize("RTX-4090").name == "RTX-4090"
        assert normalize("RTX_4090").name == "RTX-4090"


class TestGetCanonicalName:
    """Test get_canonical_name function"""

    def test_get_canonical_name_h100(self):
        """Get canonical name for H100"""
        assert get_canonical_name("H100") == "H100-80GB"
        assert get_canonical_name("NVIDIA_H100_80G") == "H100-80GB"

    def test_get_canonical_name_a100(self):
        """Get canonical name for A100"""
        assert get_canonical_name("A100") == "A100-80GB"
        assert get_canonical_name("A100-40GB") == "A100-40GB"

    def test_get_canonical_name_unknown(self):
        """Get canonical name for unknown GPU returns None"""
        assert get_canonical_name("UNKNOWN_GPU") is None
        assert get_canonical_name("") is None


class TestListAllCanonicalGpus:
    """Test list_all_canonical_gpus function"""

    def test_list_all_canonical_gpus(self):
        """List all canonical GPUs"""
        gpus = list_all_canonical_gpus()
        assert isinstance(gpus, list)
        assert len(gpus) > 0
        assert "H100-80GB" in gpus
        assert "A100-80GB" in gpus
        assert "RTX-4090" in gpus

    def test_list_all_canonical_gpus_sorted(self):
        """List all canonical GPUs is sorted"""
        gpus = list_all_canonical_gpus()
        assert gpus == sorted(gpus)


class TestListProvidersForGpu:
    """Test list_providers_for_gpu function"""

    def test_list_providers_for_h100(self):
        """List provider names for H100"""
        providers = list_providers_for_gpu("H100-80GB")
        assert isinstance(providers, list)
        assert "H100" in providers
        assert "NVIDIA_H100_80G" in providers
        assert "h100" in providers

    def test_list_providers_for_a100(self):
        """List provider names for A100"""
        providers = list_providers_for_gpu("A100-80GB")
        assert isinstance(providers, list)
        assert "A100" in providers
        assert "NVIDIA_A100_80G" in providers

    def test_list_providers_for_unknown(self):
        """List provider names for unknown GPU returns empty list"""
        providers = list_providers_for_gpu("UNKNOWN_GPU")
        assert providers == []


class TestGPUAliases:
    """Test GPU_ALIASES dictionary"""

    def test_gpu_aliases_has_entries(self):
        """GPU_ALIASES has entries"""
        assert len(GPU_ALIASES) > 0

    def test_gpu_aliases_keys_are_strings(self):
        """GPU_ALIASES keys are strings"""
        for key in GPU_ALIASES.keys():
            assert isinstance(key, str)

    def test_gpu_aliases_values_are_strings(self):
        """GPU_ALIASES values are strings"""
        for value in GPU_ALIASES.values():
            assert isinstance(value, str)


class TestGPUSpecs:
    """Test GPU_SPECS dictionary"""

    def test_gpu_specs_has_entries(self):
        """GPU_SPECS has entries"""
        assert len(GPU_SPECS) > 0

    def test_gpu_specs_h100_structure(self):
        """H100 GPU spec has correct structure"""
        spec = GPU_SPECS["H100-80GB"]
        assert spec.name == "H100-80GB"
        assert spec.vendor == GPUVendor.NVIDIA
        assert spec.vram_gb == 80
        assert spec.nvlink is True
        assert spec.compute_capability == "9.0"

    def test_gpu_specs_a100_structure(self):
        """A100 GPU spec has correct structure"""
        spec = GPU_SPECS["A100-80GB"]
        assert spec.name == "A100-80GB"
        assert spec.vendor == GPUVendor.NVIDIA
        assert spec.vram_gb == 80
        assert spec.nvlink is True
        assert spec.compute_capability == "8.0"

    def test_gpu_specs_rtx4090_structure(self):
        """RTX-4090 GPU spec has correct structure"""
        spec = GPU_SPECS["RTX-4090"]
        assert spec.name == "RTX-4090"
        assert spec.vendor == GPUVendor.NVIDIA
        assert spec.vram_gb == 24
        assert spec.nvlink is False
        assert spec.compute_capability == "8.9"

    def test_gpu_specs_amd_mi300x(self):
        """AMD MI300X GPU spec has correct structure"""
        spec = GPU_SPECS["MI300X-192GB"]
        assert spec.name == "MI300X-192GB"
        assert spec.vendor == GPUVendor.AMD
        assert spec.vram_gb == 192
        assert spec.nvlink is False

    def test_gpu_specs_intel_gaudi2(self):
        """Intel Gaudi2 GPU spec has correct structure"""
        spec = GPU_SPECS["Gaudi2-96GB"]
        assert spec.name == "Gaudi2-96GB"
        assert spec.vendor == GPUVendor.INTEL
        assert spec.vram_gb == 96
        assert spec.nvlink is False
