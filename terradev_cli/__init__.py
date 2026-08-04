"""
Terradev CLI - Cross-Cloud Compute Optimization Platform
Parallel provisioning and orchestration for optimized compute costs
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("terradev-cli")
except PackageNotFoundError:
    __version__ = "5.7.9"

__author__ = "Terradev Team"
__description__ = "Cross-cloud GPU infrastructure CLI for training, inference, and AI workload orchestration."
