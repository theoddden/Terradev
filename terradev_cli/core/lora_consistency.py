#!/usr/bin/env python3
"""
LoRA Cross-Replica Consistency - Ensures adapters loaded on one replica propagate to all

Uses vLLM Router for coordination and implements gossip protocol for state synchronization.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..ml_services.vllm_service import LoRAModule, VLLMService, VLLMConfig
from ..ml_services.lora_registry import AdapterRegistry, AdapterReplicaState

logger = logging.getLogger(__name__)


@dataclass
class ReplicaInfo:
    """Information about a vLLM replica"""
    replica_id: str
    host: str
    port: int
    last_heartbeat: datetime
    is_healthy: bool = True
    loaded_adapters: Set[str] = None

    def __post_init__(self):
        if self.loaded_adapters is None:
            self.loaded_adapters = set()


class LoRAConsistencyManager:
    """
    Manages cross-replica consistency for LoRA adapters.

    Provides:
    - Replica discovery (K8s or static config)
    - Broadcast load/unload operations to all replicas
    - Gossip protocol for state synchronization
    - Health check verification
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        replicas: Optional[List[Dict[str, Any]]] = None,
        k8s_namespace: Optional[str] = None,
        k8s_label_selector: Optional[str] = None,
    ):
        self.registry = registry
        self.replicas: Dict[str, ReplicaInfo] = {}
        self.k8s_namespace = k8s_namespace
        self.k8s_label_selector = k8s_label_selector

        # Static replica configuration (if not using K8s discovery)
        if replicas:
            for replica_config in replicas:
                self._add_static_replica(replica_config)

        # Gossip protocol state
        self.gossip_interval_seconds = 30
        self.gossip_rounds = 3
        self._gossip_task: Optional[asyncio.Task] = None
        self._running = False

        # Health check state
        self.health_check_interval_seconds = 60
        self._health_check_task: Optional[asyncio.Task] = None

    def _add_static_replica(self, config: Dict[str, Any]):
        """Add a statically configured replica"""
        replica_id = config.get("replica_id", f"{config['host']}:{config['port']}")
        self.replicas[replica_id] = ReplicaInfo(
            replica_id=replica_id,
            host=config["host"],
            port=config["port"],
            last_heartbeat=datetime.now(),
        )
        logger.info(f"Added static replica: {replica_id} at {config['host']}:{config['port']}")

    async def discover_replicas_k8s(self) -> List[ReplicaInfo]:
        """Discover vLLM replicas from Kubernetes"""
        try:
            from kubernetes import client, config

            # Load K8s config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            v1 = client.CoreV1Api()

            # List pods with label selector
            label_selector = self.k8s_label_selector or "app=vllm"
            pods = v1.list_namespaced_pod(
                namespace=self.k8s_namespace or "default",
                label_selector=label_selector,
            )

            discovered = []
            for pod in pods.items:
                if pod.status.phase != "Running":
                    continue

                # Get pod IP
                pod_ip = pod.status.pod_ip
                if not pod_ip:
                    continue

                # Get container port (assume 8000 for vLLM)
                port = 8000
                for container in pod.spec.containers:
                    for port_def in container.ports:
                        if port_def.name == "http" or port_def.container_port == 8000:
                            port = port_def.container_port
                            break

                replica_id = pod.metadata.name
                replica = ReplicaInfo(
                    replica_id=replica_id,
                    host=pod_ip,
                    port=port,
                    last_heartbeat=datetime.now(),
                )
                discovered.append(replica)
                self.replicas[replica_id] = replica

            logger.info(f"Discovered {len(discovered)} replicas from K8s")
            return discovered

        except ImportError:
            logger.warning("Kubernetes client not available, skipping K8s discovery")
            return []
        except Exception as e:
            logger.error(f"K8s discovery failed: {e}")
            return []

    async def discover_replicas_static(self, replicas: List[Dict[str, Any]]) -> List[ReplicaInfo]:
        """Use static replica configuration"""
        for config in replicas:
            self._add_static_replica(config)
        return list(self.replicas.values())

    async def broadcast_load_to_replicas(
        self,
        adapter: LoRAModule,
        version_id: str,
        timeout: float = 60.0,
        require_quorum: bool = True,
    ) -> Dict[str, Any]:
        """
        Load adapter on all replicas via vLLM Router or direct calls.

        Args:
            adapter: LoRA adapter to load
            version_id: Version ID from registry
            timeout: Timeout per replica
            require_quorum: If True, require majority of replicas to succeed

        Returns:
            Dict with success/failure status per replica and overall result
        """
        if not self.replicas:
            logger.warning("No replicas configured, attempting to discover from K8s")
            await self.discover_replicas_k8s()

        if not self.replicas:
            return {
                "status": "failed",
                "error": "No replicas available",
                "results": {},
            }

        logger.info(f"Broadcasting load of {adapter.name} to {len(self.replicas)} replicas")

        # Load on all replicas in parallel
        results = {}
        tasks = []

        for replica_id, replica in self.replicas.items():
            task = self._load_on_replica(replica, adapter, timeout)
            tasks.append((replica_id, task))

        # Execute all tasks
        for replica_id, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=timeout + 10)
                results[replica_id] = result

                # Update registry on success
                if result.get("status") == "loaded":
                    self.registry.record_replica_load(
                        replica_id=replica_id,
                        adapter_name=adapter.name,
                        version_id=version_id,
                    )
                    logger.info(f"Successfully loaded {adapter.name} on {replica_id}")
                else:
                    logger.error(f"Failed to load {adapter.name} on {replica_id}: {result.get('error')}")

            except asyncio.TimeoutError:
                results[replica_id] = {"status": "timeout", "error": "Operation timed out"}
                logger.error(f"Timeout loading {adapter.name} on {replica_id}")
            except Exception as e:
                results[replica_id] = {"status": "error", "error": str(e)}
                logger.error(f"Error loading {adapter.name} on {replica_id}: {e}")

        # Calculate overall result
        successful = sum(1 for r in results.values() if r.get("status") == "loaded")
        total = len(results)

        if require_quorum and successful < (total // 2 + 1):
            return {
                "status": "failed",
                "error": f"Quorum not reached: {successful}/{total} replicas succeeded",
                "results": results,
                "successful": successful,
                "total": total,
            }

        return {
            "status": "success" if successful > 0 else "failed",
            "results": results,
            "successful": successful,
            "total": total,
        }

    async def _load_on_replica(
        self, replica: ReplicaInfo, adapter: LoRAModule, timeout: float
    ) -> Dict[str, Any]:
        """Load adapter on a single replica"""
        config = VLLMConfig(
            model_name="",
            host=replica.host,
            port=replica.port,
        )

        async with VLLMService(config) as service:
            result = await service.lora_load(adapter)
            return result

    async def broadcast_unload_from_replicas(
        self, adapter_name: str, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Unload adapter from all replicas"""
        if not self.replicas:
            return {
                "status": "failed",
                "error": "No replicas available",
                "results": {},
            }

        logger.info(f"Broadcasting unload of {adapter_name} from {len(self.replicas)} replicas")

        results = {}
        tasks = []

        for replica_id, replica in self.replicas.items():
            task = self._unload_from_replica(replica, adapter_name, timeout)
            tasks.append((replica_id, task))

        for replica_id, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=timeout + 10)
                results[replica_id] = result

                # Update registry on success
                if result.get("status") == "unloaded":
                    self.registry.record_replica_unload(replica_id, adapter_name)
                    logger.info(f"Successfully unloaded {adapter_name} from {replica_id}")
                else:
                    logger.error(
                        f"Failed to unload {adapter_name} from {replica_id}: {result.get('error')}"
                    )

            except asyncio.TimeoutError:
                results[replica_id] = {"status": "timeout", "error": "Operation timed out"}
            except Exception as e:
                results[replica_id] = {"status": "error", "error": str(e)}

        successful = sum(1 for r in results.values() if r.get("status") == "unloaded")

        return {
            "status": "success" if successful > 0 else "failed",
            "results": results,
            "successful": successful,
            "total": len(results),
        }

    async def _unload_from_replica(
        self, replica: ReplicaInfo, adapter_name: str, timeout: float
    ) -> Dict[str, Any]:
        """Unload adapter from a single replica"""
        config = VLLMConfig(
            model_name="",
            host=replica.host,
            port=replica.port,
        )

        async with VLLMService(config) as service:
            result = await service.lora_unload(adapter_name)
            return result

    async def verify_consistency(
        self, adapter_name: str, version_id: str
    ) -> Dict[str, Any]:
        """
        Verify that adapter is loaded on all expected replicas.

        Returns consistency report with discrepancies.
        """
        expected_replicas = set(self.replicas.keys())
        loaded_replicas = set(
            state.replica_id
            for state in self.registry.list_replicas_with_adapter(adapter_name)
        )

        missing_replicas = expected_replicas - loaded_replicas
        extra_replicas = loaded_replicas - expected_replicas

        # Check version consistency
        version_mismatches = []
        for state in self.registry.list_replicas_with_adapter(adapter_name):
            if state.version_id != version_id:
                version_mismatches.append(
                    {
                        "replica_id": state.replica_id,
                        "expected_version": version_id,
                        "actual_version": state.version_id,
                    }
                )

        is_consistent = (
            len(missing_replicas) == 0
            and len(extra_replicas) == 0
            and len(version_mismatches) == 0
        )

        return {
            "is_consistent": is_consistent,
            "adapter_name": adapter_name,
            "expected_version": version_id,
            "expected_replicas": list(expected_replicas),
            "loaded_replicas": list(loaded_replicas),
            "missing_replicas": list(missing_replicas),
            "extra_replicas": list(extra_replicas),
            "version_mismatches": version_mismatches,
        }

    async def sync_adapter_state(self, adapter_name: str, version_id: str) -> Dict[str, Any]:
        """
        Synchronize adapter state across all replicas.

        Loads missing adapters and unloads extra/outdated versions.
        """
        # Get version details
        version = self.registry.get_version(version_id)
        if not version:
            return {"status": "failed", "error": f"Version {version_id} not found"}

        # Check current consistency
        consistency = await self.verify_consistency(adapter_name, version_id)

        if consistency["is_consistent"]:
            return {"status": "success", "message": "Adapter already consistent"}

        # Load on missing replicas
        if consistency["missing_replicas"]:
            adapter = LoRAModule(name=adapter_name, path=version.path)
            load_result = await self.broadcast_load_to_replicas(adapter, version_id)
            if load_result["status"] != "success":
                return {
                    "status": "partial",
                    "error": "Failed to load on some replicas",
                    "load_result": load_result,
                }

        # Unload from extra replicas
        if consistency["extra_replicas"]:
            unload_result = await self.broadcast_unload_from_replicas(adapter_name)
            if unload_result["status"] != "success":
                return {
                    "status": "partial",
                    "error": "Failed to unload from some replicas",
                    "unload_result": unload_result,
                }

        # Handle version mismatches
        if consistency["version_mismatches"]:
            for mismatch in consistency["version_mismatches"]:
                # Reload with correct version
                replica = self.replicas.get(mismatch["replica_id"])
                if replica:
                    adapter = LoRAModule(name=adapter_name, path=version.path)
                    await self._load_on_replica(replica, adapter, 60)

        # Verify final state
        final_consistency = await self.verify_consistency(adapter_name, version_id)

        return {
            "status": "success" if final_consistency["is_consistent"] else "partial",
            "final_consistency": final_consistency,
        }

    # ── Gossip Protocol ──

    async def start_gossip_protocol(self):
        """Start background gossip protocol for state synchronization"""
        self._running = True
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started gossip protocol and health checks")

    async def stop_gossip_protocol(self):
        """Stop background tasks"""
        self._running = False
        if self._gossip_task:
            self._gossip_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        logger.info("Stopped gossip protocol and health checks")

    async def _gossip_loop(self):
        """Background gossip loop"""
        while self._running:
            try:
                await self._gossip_round()
                await asyncio.sleep(self.gossip_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip loop error: {e}")
                await asyncio.sleep(self.gossip_interval_seconds)

    async def _gossip_round(self):
        """Execute one round of gossip protocol"""
        # For each replica, exchange state with random peers
        for replica_id, replica in self.replicas.items():
            if not replica.is_healthy:
                continue

            # Select random peers
            peers = [
                r for r_id, r in self.replicas.items() if r_id != replica_id and r.is_healthy
            ]
            if not peers:
                continue

            # Exchange state with a few peers
            for peer in peers[:3]:  # Exchange with 3 random peers
                await self._exchange_state(replica, peer)

    async def _exchange_state(self, replica_a: ReplicaInfo, replica_b: ReplicaInfo):
        """Exchange adapter state between two replicas"""
        try:
            # Get state from replica A
            state_a = await self._get_replica_state(replica_a)

            # Get state from replica B
            state_b = await self._get_replica_state(replica_b)

            # Merge states (simple union for now)
            # In production, implement more sophisticated conflict resolution
            merged_adapters = state_a.get("adapters", set()) | state_b.get("adapters", set())

            # Update local tracking
            replica_a.loaded_adapters = merged_adapters
            replica_b.loaded_adapters = merged_adapters

        except Exception as e:
            logger.debug(f"State exchange failed between {replica_a.replica_id} and {replica_b.replica_id}: {e}")

    async def _get_replica_state(self, replica: ReplicaInfo) -> Dict[str, Any]:
        """Get current adapter state from a replica"""
        try:
            config = VLLMConfig(model_name="", host=replica.host, port=replica.port)
            async with VLLMService(config) as service:
                result = await service.lora_list()
                adapters = set(a.get("id") for a in result.get("lora_adapters", []))
                return {"adapters": adapters}
        except Exception as e:
            logger.error(f"Failed to get state from {replica.replica_id}: {e}")
            return {"adapters": set()}

    # ── Health Checks ──

    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                await self._health_check_round()
                await asyncio.sleep(self.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval_seconds)

    async def _health_check_round(self):
        """Execute health check on all replicas"""
        for replica_id, replica in self.replicas.items():
            is_healthy = await self._check_replica_health(replica)
            replica.is_healthy = is_healthy
            replica.last_heartbeat = datetime.now()

            if not is_healthy:
                logger.warning(f"Replica {replica_id} marked as unhealthy")

    async def _check_replica_health(self, replica: ReplicaInfo) -> bool:
        """Check if a replica is healthy"""
        try:
            config = VLLMConfig(model_name="", host=replica.host, port=replica.port)
            async with VLLMService(config) as service:
                result = await service.test_connection()
                return result.get("status") == "connected"
        except Exception as e:
            logger.debug(f"Health check failed for {replica.replica_id}: {e}")
            return False

    def get_healthy_replicas(self) -> List[ReplicaInfo]:
        """Get list of healthy replicas"""
        return [r for r in self.replicas.values() if r.is_healthy]
