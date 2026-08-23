"""Telinea API key and auth resolution.

Resolves credentials from environment variables first, then the encrypted
local vault, with safe fallbacks for CI/CD and headless runners.
"""

from __future__ import annotations

import os
from typing import Optional

from ..vault_adapter import VaultAdapter


def resolve_telinea_api_key(vault: Optional[VaultAdapter] = None) -> Optional[str]:
    """Resolve the Telinea API key from env or vault.

    Resolution order:
      1. ``TELINEA_API_KEY``
      2. ``TERRADEV_TELINEA_API_KEY``
      3. ``TERRADEV_API_KEY``
      4. ``vault.get("telinea", "api_key")``
      5. ``vault.get("terradev", "api_key")``

    Returns ``None`` if no key is configured. The CLI must remain fully
    functional in that case; telemetry is simply disabled.
    """
    for env_name in (
        "TELINEA_API_KEY",
        "TERRADEV_TELINEA_API_KEY",
        "TERRADEV_API_KEY",
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    v = vault or VaultAdapter()
    return v.get("telinea", "api_key") or v.get("terradev", "api_key")


def resolve_telinea_workspace_id(vault: Optional[VaultAdapter] = None) -> Optional[str]:
    """Resolve an optional Telinea workspace ID from env or vault."""
    value = os.environ.get("TELINEA_WORKSPACE_ID", "").strip()
    if value:
        return value
    v = vault or VaultAdapter()
    return v.get("telinea", "workspace_id")


def resolve_telinea_project_id(vault: Optional[VaultAdapter] = None) -> Optional[str]:
    """Resolve an optional Telinea project ID from env or vault."""
    value = os.environ.get("TELINEA_PROJECT_ID", "").strip()
    if value:
        return value
    v = vault or VaultAdapter()
    return v.get("telinea", "project_id")
