"""
Terradev CLI - Cross-Cloud Compute Optimization Platform
Parallel provisioning and orchestration for optimized compute costs
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("terradev-cli")
except PackageNotFoundError:
    __version__ = "6.1.6"

__author__ = "Terradev Team"
__description__ = "Imperative cross-cloud CLI for AI workloads: provisioning, training, inference, and cost optimization."
