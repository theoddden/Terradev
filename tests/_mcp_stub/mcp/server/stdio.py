"""Minimal MCP stdio server transport for Python 3.9 testing."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Tuple


@asynccontextmanager
async def stdio_server(*args: Any, **kwargs: Any) -> AsyncGenerator[Tuple[None, None], None]:
    """No-op stdio server context manager."""
    try:
        yield (None, None)
    finally:
        pass
