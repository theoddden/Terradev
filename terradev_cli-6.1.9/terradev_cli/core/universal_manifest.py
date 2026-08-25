#!/usr/bin/env python3
"""Declarative universal manifests for composable Terradev stacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """A single, swappable component declaration."""

    kind: str
    name: str
    adapter: str
    version: str = "latest"
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class UniversalManifest:
    """Portable execution stack definition.

    A manifest declares the serving engine, compute module, model/dataset
    registries, and database/vector backends for a Terradev pipeline.
    """

    name: str
    version: str
    components: List[Component] = field(default_factory=list)
    globals: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Any) -> "UniversalManifest":
        """Load a manifest from a file path or file-like object."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        if suffix in (".yaml", ".yml") and yaml is not None:
            data = yaml.safe_load(text) or {}
        elif suffix in (".yaml", ".yml") and yaml is None:
            raise RuntimeError("PyYAML is required to load YAML manifests")
        else:
            data = json.loads(text)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalManifest":
        """Parse a manifest from a dict."""
        components = [
            Component(
                kind=c.get("kind", ""),
                name=c.get("name", ""),
                adapter=c.get("adapter", ""),
                version=c.get("version", "latest"),
                config=c.get("config", {}),
                depends_on=c.get("depends_on", []),
            )
            for c in data.get("components", [])
        ]
        return cls(
            name=data.get("name", "untitled"),
            version=data.get("version", "0.0.0"),
            components=components,
            globals=data.get("globals", {}),
            telemetry=data.get("telemetry", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "components": [
                {
                    "kind": c.kind,
                    "name": c.name,
                    "adapter": c.adapter,
                    "version": c.version,
                    "config": c.config,
                    "depends_on": c.depends_on,
                }
                for c in self.components
            ],
            "globals": self.globals,
            "telemetry": self.telemetry,
        }

    def save(self, path: Any, format: str = "json") -> None:
        """Save the manifest to a file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        if format == "yaml" and yaml is not None:
            p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def component(self, kind: str, name: Optional[str] = None) -> Optional[Component]:
        """Return the first matching component."""
        for c in self.components:
            if c.kind == kind and (name is None or c.name == name):
                return c
        return None
