"""Tests for terradev_cli.core.auto_lineage.

AutoLineageTracker records the lifecycle of pipeline executions and their
artifacts without manual tagging.
"""

from unittest.mock import MagicMock

import pytest

from terradev_cli.core.auto_lineage import (
    AutoLineageTracker,
    LineageRecord,
    LineageRecordType,
)
from terradev_cli.core.event_system import Artifact, ArtifactType, Environment


@pytest.fixture
def tracker(monkeypatch):
    """Provide a tracker that is wired to a fake event bus and lineage service."""
    fake_bus = MagicMock()
    fake_service = MagicMock()
    fake_service.artifacts = {}
    monkeypatch.setattr("terradev_cli.core.auto_lineage.event_bus", fake_bus)
    monkeypatch.setattr("terradev_cli.core.auto_lineage.lineage_service", fake_service)
    return AutoLineageTracker()


def test_lineage_record_defaults():
    """A LineageRecord has a UUID, default type, and running status."""
    record = LineageRecord()
    assert record.id
    assert record.type == LineageRecordType.EXECUTION
    assert record.status == "running"
    assert record.datasets == []
    assert record.output_models == []


def test_lineage_record_to_dict():
    """to_dict serializes the record and its environment."""
    record = LineageRecord(
        pipeline_id="pipe-1",
        environment=Environment.STAGING,
        status="completed",
    )
    d = record.to_dict()
    assert d["pipeline_id"] == "pipe-1"
    assert d["environment"] == "staging"
    assert d["status"] == "completed"
    assert "timestamp" in d


def test_start_execution(tracker):
    """start_execution creates and returns an active record."""
    record = tracker.start_execution("pipe-1", Environment.DEV, "tester")
    assert record.id in tracker.active_executions
    assert record.pipeline_id == "pipe-1"
    assert record.environment == Environment.DEV
    assert record.triggered_by == "tester"


def test_add_input_artifact(tracker):
    """Input artifacts are stored on the active record by type."""
    record = tracker.start_execution("pipe-1")
    tracker.add_input_artifact(record.id, ArtifactType.DATASET, "ds-1")
    tracker.add_input_artifact(record.id, ArtifactType.MODEL, "m-1")
    tracker.add_input_artifact(record.id, ArtifactType.CHECKPOINT, "cp-1")

    assert tracker.active_executions[record.id].datasets == ["ds-1"]
    assert tracker.active_executions[record.id].models == ["m-1"]
    assert tracker.active_executions[record.id].checkpoints == ["cp-1"]


def test_add_output_artifact(tracker):
    """Output artifacts are stored on the active record by type."""
    record = tracker.start_execution("pipe-1")
    tracker.add_output_artifact(record.id, ArtifactType.MODEL, "m-out")
    tracker.add_output_artifact(record.id, ArtifactType.METRICS, "met-1")

    assert tracker.active_executions[record.id].output_models == ["m-out"]
    assert tracker.active_executions[record.id].output_metrics == ["met-1"]


def test_set_hyperparameters_and_env(tracker):
    """Hyperparameters and environment variables attach to the active record."""
    record = tracker.start_execution("pipe-1")
    tracker.set_hyperparameters(record.id, {"lr": 0.01})
    tracker.set_environment_variables(record.id, {"CUDA_VISIBLE_DEVICES": "0"})

    assert tracker.active_executions[record.id].hyperparameters == {"lr": 0.01}
    assert tracker.active_executions[record.id].environment_variables == {
        "CUDA_VISIBLE_DEVICES": "0"
    }


def test_set_git_context(tracker):
    """Git commit and code hash are stored."""
    record = tracker.start_execution("pipe-1")
    tracker.set_git_context(record.id, commit="abc123", code_hash="hash-1")

    assert tracker.active_executions[record.id].git_commit == "abc123"
    assert tracker.active_executions[record.id].code_hash == "hash-1"


def test_set_resource_usage(tracker):
    """Resource usage is recorded on the active execution."""
    record = tracker.start_execution("pipe-1")
    tracker.set_resource_usage(record.id, gpu_hours=2.5, compute_cost=1.25, storage_gb=10)

    assert tracker.active_executions[record.id].gpu_hours == 2.5
    assert tracker.active_executions[record.id].compute_cost == 1.25
    assert tracker.active_executions[record.id].storage_gb == 10


def test_complete_execution(tracker):
    """complete_execution moves the record to history."""
    record = tracker.start_execution("pipe-1")
    tracker.complete_execution(record.id, "completed")

    assert record.id not in tracker.active_executions
    assert record in tracker.completed_executions
    assert record.status == "completed"
    assert record.duration_seconds >= 0


def test_event_handlers_complete_and_add_outputs(tracker, monkeypatch):
    """Event handlers drive completion and output artifact tracking."""
    record = tracker.start_execution("pipe-1")
    from terradev_cli.core.event_system import Event, EventType

    complete_event = Event(
        type=EventType.TRAINING_COMPLETED, data={"execution_id": record.id}
    )
    tracker._on_training_completed(complete_event)
    assert record.status == "completed"

    record2 = tracker.start_execution("pipe-2")
    cp_event = Event(
        type=EventType.CHECKPOINT_CREATED,
        data={"execution_id": record2.id, "checkpoint_id": "cp-2"},
    )
    tracker._on_checkpoint_created(cp_event)
    assert "cp-2" in tracker.active_executions[record2.id].output_checkpoints


def test_promotion_event_creates_record(tracker, monkeypatch):
    """A promotion requested event creates a promotion lineage record."""
    from terradev_cli.core.event_system import Event, EventType

    event = Event(
        type=EventType.PROMOTION_REQUESTED,
        data={
            "artifact_id": "m-1",
            "to_env": "staging",
            "requested_by": "admin",
        },
    )
    tracker._on_promotion_requested(event)

    promotion = [r for r in tracker.completed_executions if r.type == LineageRecordType.PROMOTION]
    assert promotion
    assert promotion[0].pipeline_id == "promotion-m-1"


def test_get_lineage_for_model(tracker, monkeypatch):
    """get_lineage_for_model returns records linked to a named model artifact."""
    from terradev_cli.core import auto_lineage

    artifact = Artifact(id="a-1", type=ArtifactType.MODEL, name="model-x", environment=Environment.DEV)
    auto_lineage.lineage_service.artifacts = {"a-1": artifact}

    record = tracker.start_execution("pipe-1")
    record.models = ["a-1"]
    tracker.complete_execution(record.id)

    results = tracker.get_lineage_for_model("model-x")
    assert len(results) == 1
    assert results[0].id == record.id


def test_diff_executions(tracker):
    """diff_executions reports hyperparameter, env, and resource differences."""
    r1 = tracker.start_execution("pipe-1")
    r1.hyperparameters = {"lr": 0.01}
    r1.environment_variables = {"FOO": "1"}
    r1.datasets = ["ds-1"]
    r1.gpu_hours = 1.0
    tracker.complete_execution(r1.id)

    r2 = tracker.start_execution("pipe-2")
    r2.hyperparameters = {"lr": 0.02}
    r2.environment_variables = {"FOO": "2"}
    r2.datasets = ["ds-1", "ds-2"]
    r2.gpu_hours = 2.0
    tracker.complete_execution(r2.id)

    diff = tracker.diff_executions(r1.id, r2.id)
    assert diff["differences"]["hyperparameters"]["lr"] == {
        "exec1": 0.01,
        "exec2": 0.02,
    }
    assert "datasets_added" in diff["differences"]["inputs"]
    assert "gpu_hours" in diff["differences"]["resources"]


def test_trace_from_checkpoint(tracker, monkeypatch):
    """trace_from_checkpoint walks back through parent executions."""
    from terradev_cli.core import auto_lineage

    artifact = Artifact(id="a-1", type=ArtifactType.DATASET, name="ds-in")
    auto_lineage.lineage_service.artifacts = {"a-1": artifact}

    parent = tracker.start_execution("parent-pipe")
    tracker.complete_execution(parent.id)

    child = tracker.start_execution("child-pipe", parent_execution_id=parent.id)
    child.output_checkpoints = ["cp-1"]
    child.datasets = ["a-1"]
    tracker.complete_execution(child.id)

    trace = tracker.trace_from_checkpoint("cp-1")
    assert trace["checkpoint_id"] == "cp-1"
    assert trace["created_by"]["execution_id"] == child.id
    assert len(trace["ancestors"]) == 1
    assert trace["ancestors"][0]["execution_id"] == parent.id
    assert "datasets" in trace["inputs"]


def test_export_lineage(tracker):
    """export_lineage produces JSON and CSV output."""
    record = tracker.start_execution("pipe-1")
    record.output_models = ["m-1"]
    tracker.complete_execution(record.id)

    json_export = tracker.export_lineage(format="json")
    assert "pipe-1" in json_export

    csv_export = tracker.export_lineage(format="csv")
    assert "id,timestamp" in csv_export
    assert record.id in csv_export
