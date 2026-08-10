"""
Terradev CLI - Cross-Cloud Compute Optimization Platform
Parallel provisioning and orchestration for optimized compute costs
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("terradev-cli")
except PackageNotFoundError:
    __version__ = "6.0.10"

__author__ = "Terradev Team"
__description__ = "Cross-cloud GPU infrastructure CLI for training, inference, and AI workload orchestration with agent sandbox, mesh, and MCP subcommands."
