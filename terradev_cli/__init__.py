"""
Terradev CLI - Cross-Cloud Compute Optimization Platform
Parallel provisioning and orchestration for optimized compute costs
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("terradev-cli")
except PackageNotFoundError:
    __version__ = "6.1.3"

__author__ = "Terradev Team"
__description__ = "Cross-cloud GPU infrastructure and training pipeline CLI for SFT, DPO, GRPO, and multi-stage LLM training orchestration."
