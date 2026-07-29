"""Tests for terradev_cli.core.config.

Terradev configuration holds provider preferences and optimization settings.
These tests cover load/save round-trips and the default config builder.
"""

from pathlib import Path

import pytest

from terradev_cli.core.config import ProviderConfig, ProviderType, TerradevConfig


def test_provider_config_defaults():
    """ProviderConfig stores provider-specific settings."""
    config = ProviderConfig(
        name="runpod",
        enabled=True,
        default_region="us-east-1",
        api_endpoint="https://api.runpod.io",
        reliability_score=0.85,
        priority=4,
        metadata={"gpu_types": ["A100"]},
    )
    assert config.name == "runpod"
    assert config.enabled is True


def test_config_create_default(tmp_path):
    """Loading a missing config creates and saves a default."""
    config_path = tmp_path / "config.json"
    config = TerradevConfig.load(str(config_path))
    assert config_path.exists()
    assert "aws" in config.providers
    assert config.parallel_queries == 6
    assert config.max_price_threshold == 10.0
    assert "us-east-1" in config.preferred_regions


def test_config_save_and_load_roundtrip(tmp_path):
    """A custom config round-trips through JSON."""
    config_path = tmp_path / "config.json"
    config = TerradevConfig(
        default_providers=["aws"],
        parallel_queries=12,
        max_price_threshold=5.0,
        preferred_regions=["us-west-2"],
        providers={
            "aws": ProviderConfig(
                name="aws",
                enabled=True,
                default_region="us-west-2",
                api_endpoint=None,
                reliability_score=0.95,
                priority=1,
                metadata={},
            )
        },
        optimization_settings={"foo": "bar"},
        analytics_settings={"enabled": True},
    )
    config.save(str(config_path))

    loaded = TerradevConfig.load(str(config_path))
    assert loaded.parallel_queries == 12
    assert loaded.max_price_threshold == 5.0
    assert loaded.default_providers == ["aws"]
    assert loaded.providers["aws"].default_region == "us-west-2"
    assert loaded.optimization_settings == {"foo": "bar"}


def test_config_load_malformed_falls_back(tmp_path):
    """Loading malformed JSON falls back to defaults."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not json")
    config = TerradevConfig.load(str(config_path))
    assert config.parallel_queries == 6


def test_provider_type_enum():
    """ProviderType enum has the expected string values."""
    assert ProviderType.AWS.value == "aws"
    assert ProviderType.RUNPOD.value == "runpod"
    assert ProviderType.HUGGINGFACE.value == "huggingface"
