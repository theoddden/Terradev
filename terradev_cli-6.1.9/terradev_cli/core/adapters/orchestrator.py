#!/usr/bin/env python3
"""Universal component orchestrator and manifest runner."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..telemetry import get_telemetry
from ..result import TerradevResult
from ..node_span_stream import NodeSpanStream
from ..universal_manifest import UniversalManifest, Component
from .base import Adapter, AdapterHealth
from .exceptions import AdapterError
from .registry import REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ComponentInstance:
    """A live, initialized component with its runtime metadata."""

    component: Component
    adapter: Adapter
    stream: Optional[NodeSpanStream] = None
    trace_id: str = ""


class UniversalOrchestrator:
    """Load a universal manifest, initialize adapters, and run the stack.

    Every component gets its own Redis span stream sidecar; every modular
    handoff is mirrored to both the stream and the global telemetry trace.
    """

    def __init__(
        self,
        manifest: UniversalManifest,
        enable_sidecars: bool = True,
    ) -> None:
        self.manifest = manifest
        self._enable_sidecars = enable_sidecars
        self._instances: Dict[str, ComponentInstance] = {}
        self._initialized = False

    def _instance_key(self, component: Component) -> str:
        return f"{component.kind}:{component.name}"

    def _sort_by_dependencies(self) -> List[Component]:
        """Return components in dependency order (basic topological sort)."""
        by_key = {self._instance_key(c): c for c in self.manifest.components}
        resolved: set = set()
        ordered: List[Component] = []

        def visit(c: Component):
            key = self._instance_key(c)
            if key in resolved:
                return
            for dep in c.depends_on:
                if dep in by_key and dep not in resolved:
                    visit(by_key[dep])
            ordered.append(c)
            resolved.add(key)

        for c in self.manifest.components:
            visit(c)
        return ordered

    async def initialize(self) -> "UniversalOrchestrator":
        """Initialize every component and start span streams."""
        if self._initialized:
            return self

        ordered = self._sort_by_dependencies()

        for component in ordered:
            await self._init_component(component)

        self._initialized = True
        return self

    async def _init_component(self, component: Component) -> ComponentInstance:
        """Resolve, initialize, and attach a sidecar to one component."""
        key = self._instance_key(component)
        if key in self._instances:
            return self._instances[key]

        telemetry = get_telemetry()
        telemetry.start_span(f"universal.{component.kind}.init")

        # Resolve dependencies first.
        for dep in component.depends_on:
            if dep in [self._instance_key(c) for c in self.manifest.components]:
                dep_component = next(c for c in self.manifest.components if self._instance_key(c) == dep)
                await self._init_component(dep_component)

        try:
            adapter = REGISTRY.resolve(component.kind, component.adapter, component.config)
            await adapter.initialize()
        except Exception as e:
            telemetry.end_span(status="ERROR")
            raise AdapterError(
                message=f"Failed to initialize {component.kind}/{component.name}: {e}",
                adapter_kind=component.kind,
                adapter_name=component.adapter,
            ) from e

        # Start a per-component span stream.
        stream = None
        if self._enable_sidecars:
            stream = NodeSpanStream(
                job=self.manifest.name,
                version=self.manifest.version,
                instance_id=key,
                provider=component.adapter,
            )
            stream.start({"config": component.config})
            get_telemetry().attach_stream(stream)

        instance = ComponentInstance(
            component=component,
            adapter=adapter,
            stream=stream,
        )
        instance.trace_id = get_telemetry().trace_id
        self._instances[key] = instance

        telemetry.end_span()
        return instance

    async def execute(
        self,
        kind: str,
        name: str,
        operation: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an operation against a component."""
        if not self._initialized:
            await self.initialize()

        key = f"{kind}:{name}"
        instance = self._instances.get(key)
        if instance is None:
            raise AdapterError(
                message=f"Component {key} not initialized",
                adapter_kind=kind,
                adapter_name=name,
            )

        telemetry = get_telemetry()
        telemetry.start_span(f"universal.{kind}.{name}.{operation}")
        try:
            adapter = instance.adapter
            method = getattr(adapter, operation, None)
            if method is None:
                raise AdapterError(
                    message=f"Operation '{operation}' not supported by {kind}/{name}",
                    adapter_kind=kind,
                    adapter_name=name,
                )

            if asyncio.iscoroutinefunction(method):
                result = await method(**(args or {}))
            else:
                result = method(**(args or {}))

            telemetry.end_span(status="OK")
            if instance.stream:
                instance.stream.record_command(
                    command=f"{kind}.{name}.{operation}",
                    args=[str(k) for k in (args or {}).keys()],
                    success=True,
                    returncode=0,
                    duration_ms=0.0,
                    attributes={
                        "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                        "result_type": type(result).__name__,
                    },
                )

            return result
        except Exception as e:
            telemetry.end_span(status="ERROR")
            if instance.stream:
                instance.stream.record_command(
                    command=f"{kind}.{name}.{operation}",
                    args=[str(k) for k in (args or {}).keys()],
                    success=False,
                    returncode=1,
                    duration_ms=0.0,
                    attributes={"error": str(e)},
                )
            raise

    async def health(self) -> Dict[str, AdapterHealth]:
        """Health check every initialized component."""
        result = {}
        for key, instance in self._instances.items():
            try:
                result[key] = await instance.adapter.health()
            except Exception as e:  # noqa: BLE001
                result[key] = AdapterHealth(
                    healthy=False,
                    latency_ms=0.0,
                    message=str(e),
                )
        return result

    async def teardown(self) -> None:
        """Release every component and close its sidecar."""
        telemetry = get_telemetry()
        for key, instance in list(self._instances.items()):
            telemetry.start_span(f"universal.{instance.component.kind}.teardown")
            try:
                if instance.stream:
                    instance.stream.end({"reason": "teardown"})
                    get_telemetry().detach_stream(instance.stream)
                await instance.adapter.dispose()
                telemetry.end_span()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error tearing down {key}: {e}")
                telemetry.end_span(status="ERROR")
        self._instances.clear()
        self._initialized = False

    def to_result(self) -> TerradevResult:
        """Return a structured result summarizing the orchestration."""
        result = TerradevResult(command="universal")
        result.result = {
            "manifest": self.manifest.to_dict(),
            "initialized": self._initialized,
            "components": {
                key: {
                    "kind": inst.component.kind,
                    "name": inst.component.name,
                    "adapter": inst.component.adapter,
                    "stream_key": inst.stream.stream_key if inst.stream else None,
                }
                for key, inst in self._instances.items()
            },
        }
        return result
