"""
Terradev CLI - Cross-Cloud Compute Optimization Platform
Parallel provisioning and orchestration for optimized compute costs
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("terradev-cli")
except PackageNotFoundError:
    __version__ = "6.2.4"

__author__ = "Terradev Team"
__description__ = "Imperative Command Line Interface for AI Workload Orchestration across 17 GPU cloud providers."
