"""Tests for terradev_cli.core.data_governance.

Data governance is a client trust moat: consent, audit trails, and OPA policy
evaluation. These tests cover the Python fallback manager.
"""

from datetime import datetime, timezone

import pytest

from terradev_cli.core.data_governance import (
    ConsentStatus,
    ConsentType,
    DataGovernanceManager,
    DataMovementType,
)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Create a manager that writes audit logs into a temp directory."""
    log_dir = tmp_path / "governance"
    monkeypatch.setattr(
        "terradev_cli.core.data_governance.Path.home", lambda: tmp_path
    )
    m = DataGovernanceManager()
    # Residency mock: make US targets pass residency checks.
    m._get_user_location = lambda user_id: "us"
    return m


@pytest.mark.asyncio
async def test_request_and_record_consent(manager):
    """Consent request round-trips and can be approved."""
    request_id = await manager.request_consent(
        user_id="user-1",
        consent_type=ConsentType.DATASET_STAGING,
        movement_type=DataMovementType.INITIAL_STAGING,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_locations=["runpod:us-east-1"],
        reason="training run",
    )
    assert request_id in manager.consent_requests

    ok = await manager.record_consent_response(
        request_id=request_id,
        user_id="user-1",
        status=ConsentStatus.APPROVED,
        approved_locations=["runpod:us-east-1"],
    )
    assert ok is True
    assert manager.consent_responses[request_id].status == ConsentStatus.APPROVED


@pytest.mark.asyncio
async def test_record_consent_response_missing(manager):
    """Recording a response for a missing request returns False."""
    ok = await manager.record_consent_response(
        request_id="missing",
        user_id="user-1",
        status=ConsentStatus.APPROVED,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_opa_policy_evaluation_allows_known_target(manager):
    """OPA evaluation allows a movement to an allow-listed provider/region."""
    result = await manager.evaluate_opa_policies(
        user_id="user-1",
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_location="runpod:us-east-1",
        movement_type=DataMovementType.INITIAL_STAGING,
    )
    assert result.result is True
    assert result.decision == "allowed"


@pytest.mark.asyncio
async def test_opa_policy_evaluation_denies_unknown_target(manager):
    """OPA evaluation denies a movement to an unlisted provider/region."""
    result = await manager.evaluate_opa_policies(
        user_id="user-1",
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_location="unknown:antarctica-1",
        movement_type=DataMovementType.INITIAL_STAGING,
    )
    assert result.result is False
    assert result.decision == "denied"


@pytest.mark.asyncio
async def test_move_data_with_governance_success(manager):
    """A full governed data movement succeeds with approved consent."""
    request_id = await manager.request_consent(
        user_id="user-1",
        consent_type=ConsentType.DATASET_STAGING,
        movement_type=DataMovementType.INITIAL_STAGING,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_locations=["runpod:us-east-1"],
        reason="training",
    )
    await manager.record_consent_response(
        request_id,
        "user-1",
        ConsentStatus.APPROVED,
        approved_locations=["runpod:us-east-1"],
    )

    log = await manager.move_data_with_governance(
        user_id="user-1",
        consent_request_id=request_id,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_location="runpod:us-east-1",
        movement_type=DataMovementType.INITIAL_STAGING,
    )
    assert log.success is True
    assert log.target_location == "runpod:us-east-1"
    assert log.total_size_bytes > 0


@pytest.mark.asyncio
async def test_move_data_with_governance_fails_without_approval(manager):
    """Movement fails if consent was not approved."""
    request_id = await manager.request_consent(
        user_id="user-1",
        consent_type=ConsentType.DATASET_STAGING,
        movement_type=DataMovementType.INITIAL_STAGING,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_locations=["runpod:us-east-1"],
        reason="training",
    )
    await manager.record_consent_response(
        request_id, "user-1", ConsentStatus.DENIED, denied_reason="no need"
    )

    with pytest.raises(Exception, match="Valid consent not found"):
        await manager.move_data_with_governance(
            user_id="user-1",
            consent_request_id=request_id,
            dataset_name="c4",
            source_location="aws:us-east-1",
            target_location="runpod:us-east-1",
            movement_type=DataMovementType.INITIAL_STAGING,
        )


@pytest.mark.asyncio
async def test_movement_history_and_compliance_report(manager):
    """History filters and compliance reports aggregate movement data."""
    request_id = await manager.request_consent(
        user_id="user-1",
        consent_type=ConsentType.DATASET_STAGING,
        movement_type=DataMovementType.INITIAL_STAGING,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_locations=["runpod:us-east-1"],
        reason="training",
    )
    await manager.record_consent_response(
        request_id, "user-1", ConsentStatus.APPROVED, approved_locations=["runpod:us-east-1"]
    )
    log = await manager.move_data_with_governance(
        user_id="user-1",
        consent_request_id=request_id,
        dataset_name="c4",
        source_location="aws:us-east-1",
        target_location="runpod:us-east-1",
        movement_type=DataMovementType.INITIAL_STAGING,
    )

    history = await manager.get_movement_history(user_id="user-1")
    assert len(history) == 1
    assert history[0].movement_id == log.movement_id

    report = await manager.generate_compliance_report(
        start_date=datetime.min.replace(tzinfo=timezone.utc),
        end_date=datetime.now(timezone.utc),
    )
    assert report["summary"]["total_movements"] == 1
    assert report["summary"]["successful_movements"] == 1
    assert report["summary"]["success_rate"] == 1.0


def test_opa_policies_initialized(manager):
    """The manager loads default OPA policies on construction."""
    assert "region_allowlist" in manager.opa_policies
    assert "provider_allowlist" in manager.opa_policies
    assert manager.consent_required is True
