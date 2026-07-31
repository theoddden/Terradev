"""Minimal MCP stdio client for Python 3.9 testing."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator


@dataclass
class StdioServerParameters:
    command: str = ""
    args: list = None
    env: dict = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


@asynccontextmanager
async def stdio_client(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
    """No-op stdio client context manager."""
    try:
        yield None
    finally:
        pass
