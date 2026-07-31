"""Minimal MCP SSE transport for Python 3.9 testing."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Tuple


@dataclass
class TransportSecuritySettings:
    """SSE transport security settings placeholder."""

    enable_dns_rebinding_protection: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)


class SseServerTransport:
    """No-op SSE server transport."""

    def __init__(self, path: str = "/messages", *args: Any, **kwargs: Any):
        self.path = path
        self.security_settings = kwargs.get("security_settings") or TransportSecuritySettings()

    async def handle_post_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    @asynccontextmanager
    async def connect_sse(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Tuple[None, None], None]:
        try:
            yield (None, None)
        finally:
            pass
