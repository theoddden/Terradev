"""Tests for terradev_cli.core.config_validator.

Invalid config is a top cause of deployment failures. These tests guard the
Python fallback validation path.
"""

import json

import pytest

from terradev_cli.core.config_validator import ConfigValidator


@pytest.fixture
def schema():
    return json.dumps({
        "type": "object",
        "required": ["provider", "gpu_type"],
        "properties": {
            "provider": {"type": "string"},
            "gpu_type": {"type": "string"},
            "gpu_count": {"type": "number"},
        },
    })


def test_valid_config_passes(schema):
    """A config matching schema is reported as valid."""
    validator = ConfigValidator(schema)
    config = json.dumps({"provider": "aws", "gpu_type": "A100", "gpu_count": 2})
    result = validator.validate(config)
    assert result["is_valid"] is True
    assert result["errors"] == []


def test_missing_required_field_fails(schema):
    """A config missing a required field is invalid."""
    validator = ConfigValidator(schema)
    config = json.dumps({"provider": "aws"})
    result = validator.validate(config)
    assert result["is_valid"] is False
    assert any("gpu_type" in e for e in result["errors"])


def test_wrong_type_reported(schema):
    """A field with the wrong type is flagged."""
    validator = ConfigValidator(schema)
    config = json.dumps({"provider": "aws", "gpu_type": "A100", "gpu_count": "two"})
    result = validator.validate(config)
    assert result["is_valid"] is False
    assert any("gpu_count" in e for e in result["errors"])


def test_invalid_json_raises(schema):
    """A non-JSON config raises a ValueError."""
    validator = ConfigValidator(schema)
    with pytest.raises(ValueError):
        validator.validate("not-json")
