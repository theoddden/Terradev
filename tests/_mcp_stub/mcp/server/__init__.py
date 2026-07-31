"""Minimal MCP server classes for Python 3.9 testing."""

from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional


class Server:
    """Lightweight MCP server compatible with the Terradev decorator API."""

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        self.name = name
        self._tools: List[Callable] = []
        self._resources: List[Callable] = []
        self._prompts: List[Callable] = []

    def _make_decorator(self, registry: list) -> Callable:
        def decorator(func: Callable) -> Callable:
            registry.append(func)
            return func

        return decorator

    def list_tools(self) -> Callable:
        return self._make_decorator(self._tools)

    def call_tool(self) -> Callable:
        return self._make_decorator(self._tools)

    def list_resources(self) -> Callable:
        return self._make_decorator(self._resources)

    def read_resource(self) -> Callable:
        return self._make_decorator(self._resources)

    def list_prompts(self) -> Callable:
        return self._make_decorator(self._prompts)

    def get_prompt(self) -> Callable:
        return self._make_decorator(self._prompts)

    def get_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    async def run(self, *args: Any, **kwargs: Any) -> None:
        return None


class NotificationOptions:
    """MCP notification options placeholder."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass
