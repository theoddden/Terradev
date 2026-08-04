#!/usr/bin/env python3
"""Adapter-layer exceptions."""

from __future__ import annotations

from typing import Any, Dict, Optional


class AdapterError(Exception):
    """Base exception for adapter failures."""

    def __init__(
        self,
        message: str,
        adapter_kind: Optional[str] = None,
        adapter_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.adapter_kind = adapter_kind
        self.adapter_name = adapter_name
        self.context = context or {}


class AdapterNotFoundError(AdapterError):
    """Raised when a requested adapter is not registered."""


class AdapterConfigError(AdapterError):
    """Raised when adapter configuration is invalid or missing."""
