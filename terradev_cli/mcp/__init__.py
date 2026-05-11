#!/usr/bin/env python3
"""
Terradev MCP Server - Native Model Context Protocol integration

Makes Terradev callable from AI agents (Claude Desktop, Cursor, Windsurf, Continue, Cline)
"""

from .server import run_server, install_config, list_tools

__all__ = [
    "run_server",
    "install_config",
    "list_tools",
]
