#!/usr/bin/env python3
"""Agentic infrastructure subcommands for the Terradev CLI.

This package registers the ``sandbox``, ``mesh`` and ``mcp`` command groups
under the existing ``terradev agent`` group defined in ``platform.py``.
"""

from __future__ import annotations

from terradev_cli.commands.platform import agent

# Import subcommand modules so their command objects are created.
from . import sandbox, mesh, mcp

# Attach the three new subcommand groups to `terradev agent`.
agent.add_command(sandbox.sandbox)
agent.add_command(mesh.mesh)
agent.add_command(mcp.mcp)
