"""Minimal MCP server models for Python 3.9 testing."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InitializationOptions:
    server_name: str = "terradev-mcp"
    server_version: str = "0.0.0"
    capabilities: Dict[str, Any] = field(default_factory=dict)
