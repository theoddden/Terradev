"""Tests for terradev_cli.core.rust_telemetry.

RustTelemetryBackend is only functional when the Rust extension is present.
These tests cover both the import flag and the graceful ImportError fallback.
"""

import pytest

from terradev_cli.core.rust_telemetry import USE_RUST_TELEMETRY, RustTelemetryBackend


def test_use_rust_telemetry_flag():
    """USE_RUST_TELEMETRY is a boolean import flag."""
    assert isinstance(USE_RUST_TELEMETRY, bool)


def test_backend_raises_without_rust():
    """Backend raises ImportError when the Rust extension is missing."""
    if not USE_RUST_TELEMETRY:
        with pytest.raises(ImportError, match="Rust telemetry not available"):
            RustTelemetryBackend()
    else:
        # If the extension is present, instantiation should succeed
        backend = RustTelemetryBackend()
        assert backend is not None
