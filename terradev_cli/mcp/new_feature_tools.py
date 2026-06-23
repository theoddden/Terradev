"""
MCP new-feature tool registry.

Add new experimental MCP tools here.  The module is imported by server.py
and its exports are merged into the main tool list at startup.

Exports:
    ALL_NEW_TOOLS  – list of mcp.types.Tool definitions
    COMMAND_MAP    – {tool_name: handler_callable} dict
"""

from typing import Any, Dict, List

ALL_NEW_TOOLS: List[Any] = []
COMMAND_MAP: Dict[str, Any] = {}
