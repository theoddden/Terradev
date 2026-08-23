"""Telinea observability connector for Terradev.

Provides auth resolution, async telemetry push, and a Redis-stream bridge
that activates only when a TELINEA_API_KEY is present.
"""

from .auth import resolve_telinea_api_key
from .config import TelineaConfig
from .client import TelineaClient, get_telinea_client
from .connector import TelineaConnector

__all__ = [
    "resolve_telinea_api_key",
    "TelineaConfig",
    "TelineaClient",
    "get_telinea_client",
    "TelineaConnector",
]
