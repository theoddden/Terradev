#!/usr/bin/env python3
"""
Terradev CLI — root command group.

Individual command domains register themselves on `cli` in sibling modules.
Imports are at the bottom to avoid circular references while `cli` is defined.
"""

import click
import os
import sys

from terradev_cli import __version__
from terradev_cli.commands._api import TerradevAPI, run_interactive_onboarding
from terradev_cli.core.output import TerradevOutput


@click.group()
@click.version_option(version=__version__, prog_name="Terradev CLI")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--format", "output_format", default=None,
              type=click.Choice(["human", "json", "jsonl"]),
              help="Output format. Defaults to TERRADEV_OUTPUT or JSON in non-TTY/CI.")
@click.option("--skip-onboarding", is_flag=True, help="Skip first-time setup")
@click.pass_context
def cli(ctx, config=None, verbose=False, output_format=None, skip_onboarding=False):
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

    # Set up the composable output collector. This is the key to Pillar IV:
    # existing commands can be piped in Docker/CI without adding new commands.
    output = TerradevOutput(format=output_format, command=ctx.invoked_subcommand)
    ctx.obj["terradev_output"] = output

    # Redirect plain ``print`` calls through the output collector while a command runs.
    capture = output.capture_print()
    capture.__enter__()
    ctx.obj["_terradev_capture_exit"] = capture.__exit__

    def _close_output():
        capture_exit = ctx.obj.get("_terradev_capture_exit")
        if capture_exit:
            capture_exit(None, None, None)
        output.close()

    ctx.call_on_close(_close_output)

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
from . import unsloth   # noqa: F401,E402
from . import mlops      # noqa: F401,E402
from . import platform   # noqa: F401,E402
from . import gateway    # noqa: F401,E402
from . import canary     # noqa: F401,E402
from . import database   # noqa: F401,E402
from . import weaviate   # noqa: F401,E402
from . import vault      # noqa: F401,E402
from . import letta   # noqa: F401,E402
from . import agent_infra  # noqa: F401,E402

# Fold agentic-serving under the agent group
platform.agent.add_command(mlops.agentic_serving)

# Attach optional tool integrations
training.train.add_command(unsloth.unsloth)
database.database.add_command(weaviate.weaviate)
platform.agent.add_command(letta.letta)

# Agent-oriented workflow / graph frameworks
platform.agent.add_command(ml.langchain)
platform.agent.add_command(ml.langgraph)
