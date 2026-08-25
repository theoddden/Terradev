"""Telinea connector configuration.

All settings are overridable via environment variables so that CI/CD runners,
Docker containers, and headless agents can use Telinea without interactive
configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .auth import (
    resolve_telinea_api_key,
    resolve_telinea_project_id,
    resolve_telinea_workspace_id,
)


@dataclass
class TelineaConfig:
    """Runtime configuration for the Telinea telemetry connector."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    project_id: Optional[str] = None
    workspace_id: Optional[str] = None
    enabled: bool = True
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    max_queue_size: int = 10000

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = resolve_telinea_api_key()
        if self.base_url is None:
            self.base_url = (
                os.environ.get("TELINEA_ENDPOINT", "").strip()
                or os.environ.get("TERRADEV_API_ENDPOINT", "").strip()
                or "https://api.telinea.cloud"
            )
        if self.project_id is None:
            self.project_id = resolve_telinea_project_id()
        if self.workspace_id is None:
            self.workspace_id = resolve_telinea_workspace_id()
        if os.environ.get("TELINEA_DISABLED", "").lower() in ("1", "true", "yes"):
            self.enabled = False

        # Normalize base URL
        self.base_url = self.base_url.rstrip("/")

    @property
    def ingest_url(self) -> str:
        return f"{self.base_url}/v1/ingest"

    @property
    def auth_header(self) -> str:
        return f"Bearer {self.api_key}" if self.api_key else ""

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.api_key)
