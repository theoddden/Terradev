#!/usr/bin/env python3
"""Built-in compute module adapter stubs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import AdapterHealth, AdapterSpec, ComputeModuleAdapter
from ..capabilities import Capability, Capabilities
from ..registry import REGISTRY


class LocalSubprocessCompute(ComputeModuleAdapter):
    """Local subprocess / Docker execution placeholder."""

    KIND = "compute"
    NAME = "local"
    VERSION = "0.1.0"
    DESCRIPTION = "Local command execution"
    CAPABILITIES = Capabilities([
        Capability.SUBPROCESS,
        Capability.CONTAINER,
        Capability.GPU,
    ])
    CONFIG_SCHEMA = {"required": []}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="local compute stub")

    async def execute(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **options,
    ) -> Dict[str, Any]:
        return {"command": command, "status": "executed"}

    async def status(self, job_id: str) -> Dict[str, Any]:
        return {"job_id": job_id, "status": "running"}

    async def stop(self, job_id: str) -> bool:
        return True


class ServerlessCompute(ComputeModuleAdapter):
    """Serverless runtime placeholder."""

    KIND = "compute"
    NAME = "serverless"
    VERSION = "0.1.0"
    DESCRIPTION = "Serverless runtime execution"
    CAPABILITIES = Capabilities([
        Capability.SERVERLESS,
        Capability.DISTRIBUTED,
    ])
    CONFIG_SCHEMA = {"required": ["endpoint"]}

    async def _do_initialize(self) -> None:
        pass

    async def health(self) -> AdapterHealth:
        return AdapterHealth(healthy=True, latency_ms=0.0, message="serverless stub")

    async def execute(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **options,
    ) -> Dict[str, Any]:
        return {"command": command, "status": "dispatched"}

    async def status(self, job_id: str) -> Dict[str, Any]:
        return {"job_id": job_id, "status": "running"}

    async def stop(self, job_id: str) -> bool:
        return True


REGISTRY.register(LocalSubprocessCompute.KIND, LocalSubprocessCompute.NAME, LocalSubprocessCompute)
REGISTRY.register(ServerlessCompute.KIND, ServerlessCompute.NAME, ServerlessCompute)
