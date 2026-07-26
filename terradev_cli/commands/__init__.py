#!/usr/bin/env python3
"""
Terradev CLI — root command group.

Individual command domains register themselves on `cli` in sibling modules.
Imports are at the bottom to avoid circular references while `cli` is defined.
"""

import click
import os
import sys

from terradev_cli.commands._api import TerradevAPI, run_interactive_onboarding


@click.group()
@click.version_option(version="5.1.5", prog_name="Terradev CLI")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--skip-onboarding", is_flag=True, help="Skip first-time setup")
@click.pass_context
def cli(ctx, config=None, verbose=False, skip_onboarding=False):
    """
    Terradev CLI - Cross-Cloud Compute Optimization Platform

    Parallel provisioning and orchestration for cross-cloud cost optimization.
    Save 30% on end-to-end compute provisioning costs with real-time cloud arbitrage.
    """
    ctx.ensure_object(dict)
    # Dependency injection: tests can pass obj={"api": mock_api} to avoid real API calls.
    if "api" not in ctx.obj:
        ctx.obj["api"] = TerradevAPI()
    api = ctx.obj["api"]

    # Check for first-time user and trigger onboarding (skip if --help is present,
    # or if the MCP server is being run non-interactively by an agent client)
    if not skip_onboarding and not os.environ.get("TERRADEV_SKIP_ONBOARDING"):
        # Skip onboarding if --help or -h is in arguments, or for MCP server commands
        if "--help" not in sys.argv and "-h" not in sys.argv and "mcp" not in sys.argv:
            if api.is_first_time_user():
                run_interactive_onboarding(api)




# Register domain modules. Each attaches its commands to the shared `cli` group.
from . import providers  # noqa: F401,E402
from . import compute    # noqa: F401,E402
from . import k8s             # noqa: F401,E402
from . import infrastructure  # noqa: F401,E402
from . import inference       # noqa: F401,E402
from . import ml         # noqa: F401,E402
from . import training   # noqa: F401,E402
from . import mlops      # noqa: F401,E402
from . import platform   # noqa: F401,E402
