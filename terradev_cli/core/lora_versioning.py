#!/usr/bin/env python3
"""
LoRA Adapter Versioning and Rollback - Manage adapter lifecycle with drift detection

Provides:
- Rollback to previous stable versions
- Performance drift detection via Phoenix traces
- Integration with drift_retrain_service for continuous fine-tuning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..ml_services.lora_registry import (
    AdapterRegistry,
    AdapterVersion,
    AdapterStatus,
)
from ..ml_services.drift_retrain_service import DriftRetrainService
from ..core.lora_consistency import LoRAConsistencyManager

logger = logging.getLogger(__name__)


@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    adapter_name: str
    from_version_id: Optional[str]
    to_version_id: str
    replicas_affected: int
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class DriftDetectionResult:
    """Result of drift detection for an adapter"""
    adapter_name: str
    version_id: str
    has_drift: bool
    baseline_score: float
    current_score: float
    drift_magnitude: float
    drift_threshold: float
    recommended_action: str  # "rollback", "retrain", "monitor"
    timestamp: datetime


class LoRAVersioningManager:
    """
    Manages LoRA adapter versioning, rollback, and drift detection.

    Provides:
    - One-click rollback to previous stable versions
    - Performance drift detection via Phoenix traces
    - Integration with drift_retrain_service for continuous fine-tuning
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        drift_service: Optional[DriftRetrainService] = None,
        consistency_manager: Optional[LoRAConsistencyManager] = None,
    ):
        self.registry = registry
        self.drift_service = drift_service
        self.consistency_manager = consistency_manager

    def _find_previous_stable(
        self, versions: List[AdapterVersion], current_version_id: Optional[str] = None
    ) -> Optional[AdapterVersion]:
        """Find the previous stable version to rollback to.

        Strategy:
        1. If current version is active, find the most recent non-active version
        2. Prefer versions with good performance metrics
        3. Exclude FAILED versions
        """
        # Filter out current version and failed versions
        candidates = [
            v
            for v in versions
            if v.version_id != current_version_id
            and v.status != AdapterStatus.FAILED
        ]

        if not candidates:
            return None

        # Sort by creation time (most recent first)
        candidates.sort(key=lambda v: v.created_at, reverse=True)

        # Prefer versions with performance metrics
        for candidate in candidates:
            if candidate.performance_metrics:
                # Check if metrics are reasonable (e.g., quality > 0.7)
                quality = candidate.performance_metrics.get("quality", 0.0)
                if quality > 0.7:
                    return candidate

        # Fallback to most recent non-failed version
        return candidates[0]

    async def rollback_adapter(
        self,
        adapter_name: str,
        target_version_id: Optional[str] = None,
        replicas: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 60.0,
    ) -> RollbackResult:
        """
        Rollback an adapter to a previous version.

        Args:
            adapter_name: Name of the adapter to rollback
            target_version_id: Specific version to rollback to (None = auto-select)
            replicas: List of replica configs for consistency manager
            timeout: Timeout per replica operation

        Returns:
            RollbackResult with success status and details
        """
        timestamp = datetime.now()
        versions = self.registry.get_adapter_versions(adapter_name)

        if not versions:
            return RollbackResult(
                success=False,
                adapter_name=adapter_name,
                from_version_id=None,
                to_version_id="",
                replicas_affected=0,
                timestamp=timestamp,
                error="No versions found for adapter",
            )

        # Get current active version
        current_active = self.registry.get_active_version(adapter_name)
        from_version_id = current_active.version_id if current_active else None

        # Determine target version
        if target_version_id:
            target = self.registry.get_version(target_version_id)
            if not target:
                return RollbackResult(
                    success=False,
                    adapter_name=adapter_name,
                    from_version_id=from_version_id,
                    to_version_id=target_version_id,
                    replicas_affected=0,
                    timestamp=timestamp,
                    error=f"Target version {target_version_id} not found",
                )
        else:
            target = self._find_previous_stable(versions, from_version_id)
            if not target:
                return RollbackResult(
                    success=False,
                    adapter_name=adapter_name,
                    from_version_id=from_version_id,
                    to_version_id="",
                    replicas_affected=0,
                    timestamp=timestamp,
                    error="No suitable previous version found for rollback",
                )
            target_version_id = target.version_id

        # Mark target as active
        success = self.registry.mark_version_active(adapter_name, target_version_id)
        if not success:
            return RollbackResult(
                success=False,
                adapter_name=adapter_name,
                from_version_id=from_version_id,
                to_version_id=target_version_id,
                replicas_affected=0,
                timestamp=timestamp,
                error="Failed to mark version as active in registry",
            )

        # Hot-swap across all replicas if consistency manager available
        replicas_affected = 0
        if self.consistency_manager or replicas:
            if not self.consistency_manager:
                self.consistency_manager = LoRAConsistencyManager(
                    registry=self.registry, replicas=replicas
                )

            from ..ml_services.vllm_service import LoRAModule

            adapter = LoRAModule(name=adapter_name, path=target.path)
            result = await self.consistency_manager.broadcast_load_to_replicas(
                adapter=adapter, version_id=target_version_id, timeout=timeout
            )

            if result["status"] == "success":
                replicas_affected = result["successful"]
            else:
                # Registry updated but replica sync failed - partial success
                logger.warning(
                    f"Rollback registry updated but replica sync failed: {result.get('error')}"
                )

        logger.info(
            f"Rolled back adapter '{adapter_name}' from {from_version_id[:8] if from_version_id else 'none'} to {target_version_id[:8]}"
        )

        return RollbackResult(
            success=True,
            adapter_name=adapter_name,
            from_version_id=from_version_id,
            to_version_id=target_version_id,
            replicas_affected=replicas_affected,
            timestamp=timestamp,
        )

    async def detect_drift(
        self,
        adapter_name: str,
        version_id: Optional[str] = None,
        drift_threshold: float = 0.1,
        source: str = "phoenix-traces",
    ) -> DriftDetectionResult:
        """
        Detect performance drift for an adapter using Phoenix traces.

        Args:
            adapter_name: Name of the adapter to check
            version_id: Specific version to check (None = active version)
            drift_threshold: Threshold for considering drift (0.1 = 10% degradation)
            source: Data source for drift detection

        Returns:
            DriftDetectionResult with drift status and recommendations
        """
        timestamp = datetime.now()

        # Get version to check
        if version_id:
            version = self.registry.get_version(version_id)
        else:
            version = self.registry.get_active_version(adapter_name)

        if not version:
            return DriftDetectionResult(
                adapter_name=adapter_name,
                version_id=version_id or "",
                has_drift=False,
                baseline_score=0.0,
                current_score=0.0,
                drift_magnitude=0.0,
                drift_threshold=drift_threshold,
                recommended_action="monitor",
                timestamp=timestamp,
            )

        # Get baseline performance from version metrics
        baseline_score = version.performance_metrics.get("quality", 0.8)

        # Get current performance from drift service
        if not self.drift_service:
            logger.warning("DriftRetrainService not available, skipping drift detection")
            return DriftDetectionResult(
                adapter_name=adapter_name,
                version_id=version.version_id,
                has_drift=False,
                baseline_score=baseline_score,
                current_score=baseline_score,
                drift_magnitude=0.0,
                drift_threshold=drift_threshold,
                recommended_action="monitor",
                timestamp=timestamp,
            )

        try:
            # Detect drift using drift service
            drift_result = await self.drift_service.detect_drift(
                model_name=adapter_name,
                source=source,
                threshold=drift_threshold,
            )

            current_score = drift_result.get("current_score", baseline_score)
            has_drift = drift_result.get("has_drift", False)
            drift_magnitude = abs(baseline_score - current_score) / max(baseline_score, 0.01)

            # Determine recommended action
            if has_drift:
                if drift_magnitude > 0.3:  # Severe drift
                    recommended_action = "rollback"
                elif drift_magnitude > 0.15:  # Moderate drift
                    recommended_action = "retrain"
                else:  # Mild drift
                    recommended_action = "monitor"
            else:
                recommended_action = "monitor"

            return DriftDetectionResult(
                adapter_name=adapter_name,
                version_id=version.version_id,
                has_drift=has_drift,
                baseline_score=baseline_score,
                current_score=current_score,
                drift_magnitude=drift_magnitude,
                drift_threshold=drift_threshold,
                recommended_action=recommended_action,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.error(f"Drift detection failed for {adapter_name}: {e}")
            return DriftDetectionResult(
                adapter_name=adapter_name,
                version_id=version.version_id,
                has_drift=False,
                baseline_score=baseline_score,
                current_score=baseline_score,
                drift_magnitude=0.0,
                drift_threshold=drift_threshold,
                recommended_action="monitor",
                timestamp=timestamp,
            )

    async def auto_rollback_on_drift(
        self,
        adapter_name: str,
        drift_threshold: float = 0.15,
        replicas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Automatically rollback if severe drift is detected.

        This is the automation layer for continuous fine-tuning loops.
        """
        # Check for drift
        drift_result = await self.detect_drift(
            adapter_name=adapter_name, drift_threshold=drift_threshold
        )

        if not drift_result.has_drift:
            return {
                "status": "no_drift",
                "drift_result": drift_result,
                "action_taken": "none",
            }

        # If drift is severe, trigger rollback
        if drift_result.recommended_action == "rollback":
            rollback_result = await self.rollback_adapter(
                adapter_name=adapter_name, replicas=replicas
            )
            return {
                "status": "rolled_back",
                "drift_result": drift_result,
                "rollback_result": rollback_result,
                "action_taken": "rollback",
            }

        # If drift is moderate, trigger retrain
        if drift_result.recommended_action == "retrain":
            if self.drift_service:
                retrain_result = await self.drift_service.retrain_on_drift(
                    model_name=adapter_name,
                    source="phoenix-traces",
                )
                return {
                    "status": "retrain_triggered",
                    "drift_result": drift_result,
                    "retrain_result": retrain_result,
                    "action_taken": "retrain",
                }

        return {
            "status": "monitoring",
            "drift_result": drift_result,
            "action_taken": "monitor",
        }

    def get_version_history(self, adapter_name: str) -> List[Dict[str, Any]]:
        """Get version history for an adapter with performance trends."""
        versions = self.registry.get_adapter_versions(adapter_name)

        history = []
        for version in versions:
            history.append(
                {
                    "version_id": version.version_id,
                    "created_at": version.created_at.isoformat(),
                    "status": version.status.value,
                    "path": version.path,
                    "rank": version.rank,
                    "performance_metrics": version.performance_metrics,
                    "is_active": version.status == AdapterStatus.ACTIVE,
                }
            )

        return history

    def compare_versions(
        self, adapter_name: str, version_id_a: str, version_id_b: str
    ) -> Dict[str, Any]:
        """Compare two versions of an adapter."""
        version_a = self.registry.get_version(version_id_a)
        version_b = self.registry.get_version(version_id_b)

        if not version_a or not version_b:
            return {"error": "One or both versions not found"}

        comparison = {
            "adapter_name": adapter_name,
            "version_a": {
                "version_id": version_a.version_id,
                "created_at": version_a.created_at.isoformat(),
                "performance_metrics": version_a.performance_metrics,
            },
            "version_b": {
                "version_id": version_b.version_id,
                "created_at": version_b.created_at.isoformat(),
                "performance_metrics": version_b.performance_metrics,
            },
        }

        # Calculate performance delta
        metrics_a = version_a.performance_metrics or {}
        metrics_b = version_b.performance_metrics or {}

        delta = {}
        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)
            delta[key] = val_b - val_a

        comparison["performance_delta"] = delta

        return comparison
