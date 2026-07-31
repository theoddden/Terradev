"""Minimal MCP client session for Python 3.9 testing."""

from typing import Any


class ClientSession:
    """No-op MCP client session."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    async def __aenter__(self) -> "ClientSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def initialize(self, *args: Any, **kwargs: Any) -> Any:
        return None
