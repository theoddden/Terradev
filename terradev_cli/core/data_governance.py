#!/usr/bin/env python3
"""
Terradev Data Governance Module
Handles explicit consent, comprehensive logging, and OPA policy enforcement for data movement
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import TerradevConfig
    from .auth import AuthManager

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of user consent for data operations"""

    DATASET_STAGING = "dataset_staging"
    CROSS_REGION = "cross_region"
    CROSS_PROVIDER = "cross_provider"
    AUTOMATED_OPTIMIZATION = "automated_optimization"
    EMERGENCY_MIGRATION = "emergency_migration"


class ConsentStatus(Enum):
    """Consent status"""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


class DataMovementType(Enum):
    """Types of data movement"""

    INITIAL_STAGING = "initial_staging"
    REGION_OPTIMIZATION = "region_optimization"
    PROVIDER_OPTIMIZATION = "provider_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    COMPLIANCE_MIGRATION = "compliance_migration"
    EMERGENCY_RELOCATION = "emergency_relocation"


@dataclass
class ConsentRequest:
    """Data movement consent request"""

    request_id: str
    user_id: str
    consent_type: ConsentType
    movement_type: DataMovementType
    dataset_name: str
    dataset_hash: str
    source_location: str
    target_locations: List[str]
    reason: str
    urgency: str  # low, medium, high, emergency
    expires_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ConsentResponse:
    """User consent response"""

    request_id: str
    user_id: str
    status: ConsentStatus
    approved_at: Optional[datetime]
    denied_reason: Optional[str]
    approved_locations: List[str]
    conditions: List[str]
    metadata: Dict[str, Any]


@dataclass
class DataMovementLog:
    """Comprehensive data movement log entry"""

    movement_id: str
    request_id: str
    user_id: str
    timestamp: datetime
    movement_type: DataMovementType
    dataset_name: str
    dataset_hash: str
    source_location: str
    target_location: str
    file_count: int
    total_size_bytes: int
    transfer_time_seconds: float
    transfer_cost_usd: float
    reason: str
    consent_id: str
    opa_policy_id: str
    policy_result: str  # allowed, denied, conditional
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class OPAPolicyEvaluation:
    """OPA policy evaluation result"""

    policy_id: str
    policy_name: str
    evaluation_time: datetime
    input_data: Dict[str, Any]
    result: bool
    decision: str
    reasons: List[str]
    conditions: List[str]
    metadata: Dict[str, Any]


class DataGovernanceManager:
    """Manages data governance, consent, and compliance"""

    def __init__(self, config=None, auth=None):
        self.config = config
        self.auth = auth
        self.consent_requests: Dict[str, ConsentRequest] = {}
        self.consent_responses: Dict[str, ConsentResponse] = {}
        self.movement_logs: List[DataMovementLog] = []
        self.opa_evaluations: List[OPAPolicyEvaluation] = []

        # Governance settings
        self.consent_required = True
        self.logging_enabled = True
        self.opa_enforcement = True
        self.audit_retention_days = 2555  # 7 years

        # Governance log file
        self._log_dir = Path.home() / ".terradev" / "governance"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._log_dir / "audit_log.jsonl"

        # Initialize OPA policies
        self._initialize_opa_policies()

        logger.info("Data Governance Manager initialized")

    def _initialize_opa_policies(self):
        """Initialize OPA policies for data governance"""
        # Derive allowed regions/providers from config if available, else use defaults
        allowed_regions = ["us-east-1", "us-west-2", "eu-west-1", "us-central1", "eastus"]
        allowed_providers = ["aws", "gcp", "azure", "runpod", "vastai", "lambda_labs",
                             "coreweave", "tensordock", "huggingface", "baseten", "oracle"]
        if self.config:
            try:
                allowed_regions = getattr(self.config, 'preferred_regions', allowed_regions)
                allowed_providers = getattr(self.config, 'default_providers', allowed_providers)
            except Exception:
                pass

        self.opa_policies = {
            "region_allowlist": {
                "policy_id": "data.region.allowlist",
                "description": "Restricts data movement to approved regions",
                "allowed_regions": allowed_regions,
                "exceptions": ["emergency_migration"],
            },
            "provider_allowlist": {
                "policy_id": "data.provider.allowlist",
                "description": "Restricts data movement to approved providers",
                "allowed_providers": allowed_providers,
                "exceptions": ["compliance_migration"],
            },
            "residency_constraints": {
                "policy_id": "data.residency.constraints",
                "description": "Enforces data residency requirements",
                "residency_rules": {"US": ["us-east-1", "us-west-2", "us-central1"],
                                     "EU": ["eu-west-1", "europe-west1"]},
                "exceptions": ["emergency_relocation"],
            },
            "data_classification": {
                "policy_id": "data.classification.restrictions",
                "description": "Restricts movement based on data sensitivity",
                "classification_rules": {"public": "any", "confidential": "same_provider",
                                          "restricted": "same_region"},
                "exceptions": [],
            },
        }

    async def request_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        movement_type: DataMovementType,
        dataset_name: str,
        source_location: str,
        target_locations: List[str],
        reason: str,
        urgency: str = "medium",
        expires_hours: int = 24,
    ) -> str:
        """Request user consent for data movement"""

        request_id = str(uuid.uuid4())

        # Calculate dataset hash
        dataset_hash = await self._calculate_dataset_hash(dataset_name, source_location)

        # Create consent request
        consent_request = ConsentRequest(
            request_id=request_id,
            user_id=user_id,
            consent_type=consent_type,
            movement_type=movement_type,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            source_location=source_location,
            target_locations=target_locations,
            reason=reason,
            urgency=urgency,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours),
            metadata={
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "user_agent": "terradev-cli",
                "ip_address": self._get_user_ip(user_id),
            },
        )

        self.consent_requests[request_id] = consent_request

        # Log consent request
        await self._log_consent_request(consent_request)

        # Send consent request to user
        await self._send_consent_notification(consent_request)

        logger.info(f"Consent request {request_id} created for user {user_id}")
        return request_id

    async def record_consent_response(
        self,
        request_id: str,
        user_id: str,
        status: ConsentStatus,
        approved_locations: Optional[List[str]] = None,
        denied_reason: Optional[str] = None,
        conditions: Optional[List[str]] = None,
    ) -> bool:
        """Record user consent response"""

        if request_id not in self.consent_requests:
            logger.error(f"Consent request {request_id} not found")
            return False

        consent_response = ConsentResponse(
            request_id=request_id,
            user_id=user_id,
            status=status,
            approved_at=(
                datetime.now(timezone.utc) if status == ConsentStatus.APPROVED else None
            ),
            denied_reason=denied_reason,
            approved_locations=approved_locations or [],
            conditions=conditions or [],
            metadata={
                "response_time": datetime.now(timezone.utc).isoformat(),
                "user_agent": "terradev-cli",
            },
        )

        self.consent_responses[request_id] = consent_response

        # Log consent response
        await self._log_consent_response(consent_response)

        logger.info(
            f"Consent response recorded for request {request_id}: {status.value}"
        )
        return True

    async def evaluate_opa_policies(
        self,
        user_id: str,
        dataset_name: str,
        source_location: str,
        target_location: str,
        movement_type: DataMovementType,
        urgency: str = "medium",
    ) -> OPAPolicyEvaluation:
        """Evaluate OPA policies for data movement"""

        # Prepare input for OPA evaluation
        input_data = {
            "user": {
                "id": user_id,
                "permissions": await self._get_user_permissions(user_id),
                "location": self._get_user_location(user_id),
            },
            "dataset": {
                "name": dataset_name,
                "classification": await self._get_dataset_classification(dataset_name),
                "hash": await self._calculate_dataset_hash(
                    dataset_name, source_location
                ),
                "source_location": source_location,
                "target_location": target_location,
            },
            "movement": {
                "type": movement_type.value,
                "urgency": urgency,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Evaluate each policy
        policy_results = []
        for policy_name, policy_config in self.opa_policies.items():
            result = await self._evaluate_single_policy(policy_config, input_data)
            policy_results.append(result)

        # Determine overall decision
        overall_decision = "allowed"
        denied_reasons = []
        conditions = []

        for result in policy_results:
            if not result["allowed"]:
                overall_decision = "denied"
                denied_reasons.append(result["reason"])
            elif result["conditions"]:
                conditions.extend(result["conditions"])

        # Create evaluation result
        evaluation = OPAPolicyEvaluation(
            policy_id=f"data.governance.bundle.{uuid.uuid4().hex[:8]}",
            policy_name="Data Governance Bundle",
            evaluation_time=datetime.now(timezone.utc),
            input_data=input_data,
            result=overall_decision == "allowed",
            decision=overall_decision,
            reasons=denied_reasons,
            conditions=conditions,
            metadata={
                "policies_evaluated": list(self.opa_policies.keys()),
                "evaluation_duration_ms": 0,  # Would be measured in real implementation
            },
        )

        self.opa_evaluations.append(evaluation)

        # Log OPA evaluation
        await self._log_opa_evaluation(evaluation)

        logger.info(
            f"OPA policy evaluation: {overall_decision} for {dataset_name} movement"
        )
        return evaluation

    async def move_data_with_governance(
        self,
        user_id: str,
        consent_request_id: str,
        dataset_name: str,
        source_location: str,
        target_location: str,
        movement_type: DataMovementType,
    ) -> DataMovementLog:
        """Move data with full governance oversight"""

        movement_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Verify consent
            consent_response = self.consent_responses.get(consent_request_id)
            if (
                not consent_response
                or consent_response.status != ConsentStatus.APPROVED
            ):
                raise Exception(
                    f"Valid consent not found for request {consent_request_id}"
                )

            # 2. Check if target location is in approved locations
            if target_location not in consent_response.approved_locations:
                raise Exception(
                    f"Target location {target_location} not in approved locations"
                )

            # 3. Evaluate OPA policies
            opa_evaluation = await self.evaluate_opa_policies(
                user_id, dataset_name, source_location, target_location, movement_type
            )

            if not opa_evaluation.result:
                raise Exception(
                    f"OPA policies denied movement: {', '.join(opa_evaluation.reasons)}"
                )

            # 4. Get dataset metadata
            dataset_metadata = await self._get_dataset_metadata(
                dataset_name, source_location
            )

            # 5. Perform data movement
            transfer_result = await self._execute_data_transfer(
                source_location, target_location, dataset_metadata
            )

            # 6. Calculate transfer cost
            transfer_cost = await self._calculate_transfer_cost(
                transfer_result["size_bytes"], source_location, target_location
            )

            # 7. Create movement log
            movement_log = DataMovementLog(
                movement_id=movement_id,
                request_id=consent_request_id,
                user_id=user_id,
                timestamp=start_time,
                movement_type=movement_type,
                dataset_name=dataset_name,
                dataset_hash=consent_response.metadata.get("dataset_hash", ""),
                source_location=source_location,
                target_location=target_location,
                file_count=transfer_result["file_count"],
                total_size_bytes=transfer_result["size_bytes"],
                transfer_time_seconds=transfer_result["duration_seconds"],
                transfer_cost_usd=transfer_cost,
                reason=self.consent_requests[consent_request_id].reason,
                consent_id=consent_request_id,
                opa_policy_id=opa_evaluation.policy_id,
                policy_result=opa_evaluation.decision,
                success=True,
                error_message=None,
                metadata={
                    "transfer_method": transfer_result["method"],
                    "encryption_used": transfer_result["encryption"],
                    "compression_ratio": transfer_result.get("compression_ratio", 1.0),
                    "checksum_verified": transfer_result["checksum_verified"],
                },
            )

            # 8. Log the movement
            await self._log_data_movement(movement_log)

            # 9. Store in audit log
            self.movement_logs.append(movement_log)

            logger.info(
                f"Data movement completed: {dataset_name} from {source_location} to {target_location}"
            )
            return movement_log

        except Exception as e:
            # Create failed movement log
            failed_log = DataMovementLog(
                movement_id=movement_id,
                request_id=consent_request_id,
                user_id=user_id,
                timestamp=start_time,
                movement_type=movement_type,
                dataset_name=dataset_name,
                dataset_hash="",
                source_location=source_location,
                target_location=target_location,
                file_count=0,
                total_size_bytes=0,
                transfer_time_seconds=0,
                transfer_cost_usd=0,
                reason="Failed",
                consent_id=consent_request_id,
                opa_policy_id="",
                policy_result="failed",
                success=False,
                error_message=str(e),
                metadata={"error_type": type(e).__name__},
            )

            await self._log_data_movement(failed_log)
            self.movement_logs.append(failed_log)

            logger.error(f"Data movement failed: {str(e)}")
            raise

    async def get_movement_history(
        self,
        user_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DataMovementLog]:
        """Get filtered movement history"""

        filtered_logs = self.movement_logs

        # Apply filters
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]

        if dataset_name:
            filtered_logs = [
                log for log in filtered_logs if log.dataset_name == dataset_name
            ]

        if start_date:
            filtered_logs = [
                log for log in filtered_logs if log.timestamp >= start_date
            ]

        if end_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]

        # Sort by timestamp (most recent first) and limit
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered_logs[:limit]

    async def generate_compliance_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""

        # Get movements in date range
        movements = await self.get_movement_history(
            start_date=start_date, end_date=end_date, limit=10000
        )

        # Calculate statistics
        total_movements = len(movements)
        successful_movements = len([m for m in movements if m.success])
        failed_movements = total_movements - successful_movements

        total_size_moved = sum(m.total_size_bytes for m in movements)
        total_cost = sum(m.transfer_cost_usd for m in movements)

        # Group by movement type
        movements_by_type = {}
        for movement in movements:
            movement_type = movement.movement_type.value
            if movement_type not in movements_by_type:
                movements_by_type[movement_type] = []
            movements_by_type[movement_type].append(movement)

        # Group by user
        movements_by_user = {}
        for movement in movements:
            user = movement.user_id
            if user not in movements_by_user:
                movements_by_user[user] = []
            movements_by_user[user].append(movement)

        # OPA policy statistics
        policy_evaluations = [
            e
            for e in self.opa_evaluations
            if start_date <= e.evaluation_time <= end_date
        ]

        allowed_evaluations = len([e for e in policy_evaluations if e.result])
        denied_evaluations = len(policy_evaluations) - allowed_evaluations

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "summary": {
                "total_movements": total_movements,
                "successful_movements": successful_movements,
                "failed_movements": failed_movements,
                "success_rate": (
                    successful_movements / total_movements if total_movements > 0 else 0
                ),
                "total_size_bytes": total_size_moved,
                "total_size_gb": total_size_moved / (1024**3),
                "total_cost_usd": total_cost,
            },
            "movements_by_type": {
                movement_type: {
                    "count": len(movements),
                    "success_rate": len([m for m in movements if m.success])
                    / len(movements),
                    "total_size_gb": sum(m.total_size_bytes for m in movements)
                    / (1024**3),
                    "total_cost_usd": sum(m.transfer_cost_usd for m in movements),
                }
                for movement_type, movements in movements_by_type.items()
            },
            "movements_by_user": {
                user: {
                    "movement_count": len(movements),
                    "total_size_gb": sum(m.total_size_bytes for m in movements)
                    / (1024**3),
                    "total_cost_usd": sum(m.transfer_cost_usd for m in movements),
                }
                for user, movements in movements_by_user.items()
            },
            "opa_policy_evaluations": {
                "total_evaluations": len(policy_evaluations),
                "allowed_evaluations": allowed_evaluations,
                "denied_evaluations": denied_evaluations,
                "allowance_rate": (
                    allowed_evaluations / len(policy_evaluations)
                    if len(policy_evaluations) > 0
                    else 0
                ),
            },
            "compliance_metrics": {
                "consent_coverage": len(self.consent_responses),
                "policy_compliance_rate": (
                    allowed_evaluations / len(policy_evaluations)
                    if len(policy_evaluations) > 0
                    else 0
                ),
                "audit_trail_complete": len(movements)
                == len([m for m in movements if m.metadata.get("audit_complete")]),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return report

    # Private helper methods

    async def _calculate_dataset_hash(self, dataset_name: str, location: str) -> str:
        """Calculate hash of dataset for integrity verification"""
        # In real implementation, this would calculate actual dataset hash
        hash_input = (
            f"{dataset_name}:{location}:{datetime.now(timezone.utc).isoformat()}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()

    async def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions for OPA evaluation"""
        # In real implementation, this would query user database
        return ["data_staging", "cross_region", "cost_optimization"]

    def _get_user_location(self, user_id: str) -> str:
        """Get user location for residency evaluation"""
        # In real implementation, this would get from user profile
        return "US"

    async def _get_dataset_classification(self, dataset_name: str) -> str:
        """Get data classification for policy evaluation"""
        # In real implementation, this would query dataset metadata
        return "confidential" if "sensitive" in dataset_name.lower() else "public"

    async def _get_dataset_metadata(
        self, dataset_name: str, location: str
    ) -> Dict[str, Any]:
        """Get dataset metadata for transfer"""
        # In real implementation, this would query storage provider
        return {
            "file_count": 100,
            "size_bytes": 1024 * 1024 * 1024,  # 1GB
            "file_types": [".csv", ".json", ".parquet"],
            "encryption": "AES-256",
        }

    async def _execute_data_transfer(
        self, source: str, target: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute actual data transfer"""
        # In real implementation, this would perform actual transfer
        return {
            "file_count": metadata["file_count"],
            "size_bytes": metadata["size_bytes"],
            "duration_seconds": 300.5,  # 5 minutes
            "method": "async_transfer",
            "encryption": metadata["encryption"],
            "compression_ratio": 0.8,
            "checksum_verified": True,
        }

    async def _calculate_transfer_cost(
        self, size_bytes: int, source: str, target: str
    ) -> float:
        """Calculate transfer cost"""
        # In real implementation, this would use provider pricing
        size_gb = size_bytes / (1024**3)
        return size_gb * 0.05  # $0.05 per GB

    def _get_user_ip(self, user_id: str) -> str:
        """Get user IP address for logging"""
        # In real implementation, this would get from request context
        return "192.168.1.100"

    async def _send_consent_notification(self, consent_request: ConsentRequest):
        """Send consent notification to user"""
        # In real implementation, this would send email/push notification
        logger.info(
            f"Consent notification sent to user {consent_request.user_id} for request {consent_request.request_id}"
        )

    async def _log_consent_request(self, consent_request: ConsentRequest):
        """Log consent request"""
        if self.logging_enabled:
            log_entry = {
                "type": "consent_request",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": asdict(consent_request),
            }
            logger.info(f"Consent request logged: {consent_request.request_id}")

    async def _log_consent_response(self, consent_response: ConsentResponse):
        """Log consent response"""
        if self.logging_enabled:
            log_entry = {
                "type": "consent_response",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": asdict(consent_response),
            }
            logger.info(f"Consent response logged: {consent_response.request_id}")

    async def _log_opa_evaluation(self, evaluation: OPAPolicyEvaluation):
        """Log OPA policy evaluation"""
        if self.logging_enabled:
            log_entry = {
                "type": "opa_evaluation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": asdict(evaluation),
            }
            logger.info(f"OPA evaluation logged: {evaluation.policy_id}")

    async def _log_data_movement(self, movement_log: DataMovementLog):
        """Log data movement"""
        if self.logging_enabled:
            log_entry = {
                "type": "data_movement",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": asdict(movement_log),
            }
            logger.info(f"Data movement logged: {movement_log.movement_id}")

    async def _evaluate_single_policy(
        self, policy_config: Dict[str, Any], input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single OPA policy"""
        # In real implementation, this would call OPA server
        policy_id = policy_config["policy_id"]

        # Mock evaluation logic
        if "region" in policy_id:
            target_region = input_data["dataset"]["target_location"].split(":")[-1]
            allowed_regions = policy_config["allowed_regions"]
            allowed = target_region in allowed_regions
            reason = (
                f"Region {target_region} not in allowlist"
                if not allowed
                else "Region allowed"
            )

        elif "provider" in policy_id:
            target_provider = input_data["dataset"]["target_location"].split(":")[0]
            allowed_providers = policy_config["allowed_providers"]
            allowed = target_provider in allowed_providers
            reason = (
                f"Provider {target_provider} not in allowlist"
                if not allowed
                else "Provider allowed"
            )

        elif "residency" in policy_id:
            user_location = input_data["user"]["location"]
            target_region = input_data["dataset"]["target_location"].split(":")[-1]
            # Simple residency rule: data must stay in same country
            allowed = user_location in target_region or target_region.startswith(
                user_location
            )
            reason = (
                f"Residency constraint violated"
                if not allowed
                else "Residency constraint satisfied"
            )

        else:
            allowed = True
            reason = "Policy passed"

        return {
            "policy_id": policy_id,
            "allowed": allowed,
            "reason": reason,
            "conditions": [] if allowed else ["Manual review required"],
        }
