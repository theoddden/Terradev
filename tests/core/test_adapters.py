"""Tests for the universal adapter layer."""

import pytest

from terradev_cli.core.adapters import (
    REGISTRY,
    AdapterError,
    AdapterConfigError,
    AdapterNotFoundError,
    AdapterSpec,
    Capabilities,
    Capability,
    ComputeModuleAdapter,
    ServingEngineAdapter,
)
from terradev_cli.core.adapters.builtins.serving import VllmServingAdapter
from terradev_cli.core.universal_manifest import Component, UniversalManifest


def test_builtins_are_registered():
    """Built-in adapters register on package import."""
    keys = set((k.kind, k.name) for k in REGISTRY.list().keys())
    assert ("serving", "vllm") in keys
    assert ("serving", "ollama") in keys
    assert ("compute", "local") in keys
    assert ("database", "sqlite") in keys


def test_resolve_adapter():
    """Resolve a registered adapter by kind/name."""
    adapter = REGISTRY.resolve("serving", "vllm", {"model": "llama3"})
    assert adapter.kind == "serving"
    assert adapter.name == "vllm"
    assert adapter.config["model"] == "llama3"


def test_adapter_missing_required_config_raises():
    """Adapters enforce required config keys."""
    with pytest.raises(AdapterConfigError):
        REGISTRY.resolve("serving", "vllm", {})


def test_unknown_adapter_raises():
    """Unknown adapter kind/name raises AdapterNotFoundError."""
    with pytest.raises(AdapterNotFoundError):
        REGISTRY.resolve("serving", "does-not-exist", {})


def test_capabilities():
    """Capability sets support has/has_any/has_all."""
    caps = Capabilities([Capability.INFERENCE, Capability.STREAMING])
    assert caps.has(Capability.INFERENCE)
    assert caps.has_any(Capability.BATCH, Capability.STREAMING)
    assert caps.has_all(Capability.INFERENCE, Capability.STREAMING)
    assert not caps.has(Capability.GPU)


def test_universal_manifest_round_trip(tmp_path):
    """UniversalManifest loads and saves faithfully."""
    manifest = UniversalManifest(
        name="test-stack",
        version="1.0.0",
        components=[
            Component(
                kind="serving",
                name="api",
                adapter="vllm",
                version="0.1.0",
                config={"model": "llama3"},
                depends_on=["database"],
            ),
            Component(
                kind="database",
                name="db",
                adapter="sqlite",
                config={"path": str(tmp_path / "test.db")},
            ),
        ],
        globals={"region": "us-east-1"},
    )

    path = tmp_path / "manifest.json"
    manifest.save(path)
    restored = UniversalManifest.load(path)

    assert restored.name == "test-stack"
    assert len(restored.components) == 2
    assert restored.component("serving", "api").adapter == "vllm"
    assert restored.component("database", "db").config["path"]
