"""Tests for terradev_cli.core.event_system.

The event system drives automation: triggers, lineage, and environment
promotion. These tests cover the Python fallback event bus and trigger
manager.
"""

import pytest

from terradev_cli.core.event_system import (
    Artifact,
    ArtifactType,
    Environment,
    Event,
    EventBus,
    EventType,
    Promotion,
    Trigger,
    TriggerManager,
    TriggerType,
)


@pytest.fixture
def bus():
    return EventBus()


def test_event_bus_subscribe_and_publish(bus):
    """Subscribers receive published events."""
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EventType.DATASET_LANDED, handler)
    event = Event(type=EventType.DATASET_LANDED, source="test", data={"x": 1})
    bus.publish(event)

    assert len(received) == 1
    assert received[0].id == event.id


def test_event_bus_get_events_filtered(bus):
    """get_events supports filtering by type and limit."""
    bus.publish(Event(type=EventType.DATASET_LANDED, source="s1"))
    bus.publish(Event(type=EventType.TRAINING_COMPLETED, source="s2"))
    bus.publish(Event(type=EventType.DATASET_LANDED, source="s3"))

    events = bus.get_events(event_type=EventType.DATASET_LANDED, limit=10)
    assert len(events) == 2
    assert all(e.type == EventType.DATASET_LANDED for e in events)


def test_event_bus_history_max(bus):
    """Old events are evicted once the in-memory history cap is reached."""
    for i in range(1001):
        bus.publish(Event(type=EventType.DATASET_LANDED, source="s", data={"i": i}))

    events = bus.get_events(limit=2000)
    assert len(events) == 1000
    assert events[0].data["i"] == 1000  # newest first


def test_trigger_manager_creates_trigger(bus):
    """Triggers are created and stored."""
    manager = TriggerManager(bus)
    trigger = manager.create_trigger(
        "nightly",
        TriggerType.SCHEDULE,
        target_pipeline="train",
        schedule="0 0 * * *",
    )
    assert trigger.name == "nightly"
    assert trigger.target_pipeline == "train"
    assert trigger in manager.triggers.values()


def test_trigger_manager_event_based_fires(bus):
    """Event-based triggers fire when the matching event is published."""
    manager = TriggerManager(bus)
    trigger = manager.create_trigger(
        "land",
        TriggerType.EVENT_BASED,
        target_pipeline="process",
        event_type=EventType.DATASET_LANDED,
    )

    bus.publish(Event(type=EventType.DATASET_LANDED, source="s"))
    assert trigger.trigger_count == 1
    assert trigger.last_triggered is not None


def test_trigger_manager_respects_enabled(bus):
    """Disabled triggers do not fire."""
    manager = TriggerManager(bus)
    trigger = manager.create_trigger(
        "land",
        TriggerType.EVENT_BASED,
        target_pipeline="process",
        event_type=EventType.DATASET_LANDED,
    )
    trigger.enabled = False

    bus.publish(Event(type=EventType.DATASET_LANDED, source="s"))
    assert trigger.trigger_count == 0


def test_artifact_and_promotion_dataclasses():
    """Artifact and Promotion records have sensible defaults and can round-trip."""
    artifact = Artifact(
        name="model-v1",
        type=ArtifactType.MODEL,
        environment=Environment.STAGING,
        parent_ids=["dataset-1"],
    )
    assert artifact.type == ArtifactType.MODEL
    assert artifact.environment == Environment.STAGING

    promotion = Promotion(
        artifact_id=artifact.id,
        from_env=Environment.DEV,
        to_env=Environment.STAGING,
    )
    assert promotion.status == "pending"
    assert promotion.from_env == Environment.DEV
