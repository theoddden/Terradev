#!/usr/bin/env python3
"""Runtime registry for universal adapters."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from .base import Adapter, AdapterSpec
from .capabilities import Capabilities
from .exceptions import AdapterError, AdapterNotFoundError, AdapterConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RegistryKey:
    kind: str
    name: str


class AdapterRegistry:
    """Hot-swap registry for adapter classes.

    Adapters are registered under ``(kind, name)`` and resolved from a
    typed config dict at runtime.
    """

    def __init__(self) -> None:
        self._adapters: Dict[_RegistryKey, Type[Adapter]] = {}
        self._builtin_loaded = False

    def register(
        self,
        kind: str,
        name: str,
        adapter_cls: Type[Adapter],
    ) -> "AdapterRegistry":
        """Register an adapter class."""
        key = _RegistryKey(kind, name)
        if key in self._adapters:
            logger.warning(f"Overwriting adapter registration for {kind}/{name}")
        self._adapters[key] = adapter_cls
        return self

    def is_registered(self, kind: str, name: str) -> bool:
        return _RegistryKey(kind, name) in self._adapters

    def resolve(
        self,
        kind: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Adapter:
        """Resolve an adapter by kind/name and instantiate with config."""
        key = _RegistryKey(kind, name)
        adapter_cls = self._adapters.get(key)
        if adapter_cls is None:
            raise AdapterNotFoundError(
                message=f"No adapter registered for {kind}/{name}",
                adapter_kind=kind,
                adapter_name=name,
            )

        spec = self._build_spec(adapter_cls)
        try:
            return adapter_cls(spec=spec, config=config)
        except AdapterConfigError:
            raise
        except Exception as e:
            raise AdapterError(
                message=f"Failed to instantiate {kind}/{name}: {e}",
                adapter_kind=kind,
                adapter_name=name,
                context={"config": config},
            ) from e

    def _build_spec(self, adapter_cls: Type[Adapter]) -> AdapterSpec:
        """Build a spec from adapter metadata if available."""
        kind = getattr(adapter_cls, "KIND", "unknown")
        name = getattr(adapter_cls, "NAME", adapter_cls.__name__)
        version = getattr(adapter_cls, "VERSION", "0.0.0")
        description = getattr(adapter_cls, "DESCRIPTION", "")
        capabilities = getattr(adapter_cls, "CAPABILITIES", None)
        config_schema = getattr(adapter_cls, "CONFIG_SCHEMA", {})
        return AdapterSpec(
            kind=kind,
            name=name,
            version=version,
            description=description,
            capabilities=capabilities or Capabilities(),
            config_schema=config_schema,
        )

    def load_builtins(self) -> "AdapterRegistry":
        """Load built-in adapter stubs. Safe to call repeatedly."""
        if self._builtin_loaded:
            return self
        self._builtin_loaded = True

        # Import built-in adapters dynamically so missing optional deps do not
        # break the whole registry.
        builtin_modules = [
            "terradev_cli.core.adapters.builtins.serving",
            "terradev_cli.core.adapters.builtins.compute",
            "terradev_cli.core.adapters.builtins.models",
            "terradev_cli.core.adapters.builtins.datasets",
            "terradev_cli.core.adapters.builtins.database",
        ]
        for mod in builtin_modules:
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Built-in adapter module not loaded: {mod}: {e}")

        return self

    def list(self, kind: Optional[str] = None) -> Dict[_RegistryKey, Type[Adapter]]:
        """List registered adapters, optionally filtered by kind."""
        if kind is None:
            return dict(self._adapters)
        return {k: v for k, v in self._adapters.items() if k.kind == kind}


# Global registry used by built-in adapters and third-party plugins.
REGISTRY = AdapterRegistry()
