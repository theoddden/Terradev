"""Tests for terradev_cli.core.telemetry.

The telemetry module is a no-op stub for open source builds.
"""

from terradev_cli.core.telemetry import MandatoryTelemetryClient, TelemetryClient, get_mandatory_telemetry


def test_telemetry_client_is_noop():
    """log_action is a no-op and does not raise."""
    client = TelemetryClient()
    client.log_action("provision", {"model": "llama-7b"})
    # No exception and no side effect to assert.


def test_check_license_is_open_source():
    """License checks always allow open-source usage."""
    client = TelemetryClient()
    result = client.check_license("provision")
    assert result["allowed"] is True
    assert result["tier"] == "open-source"
    assert result["limit"] == float("inf")
    assert result["usage"] == 0


def test_get_mandatory_telemetry_singleton():
    """get_mandatory_telemetry returns a stable singleton."""
    c1 = get_mandatory_telemetry()
    c2 = get_mandatory_telemetry()
    assert c1 is c2
    assert isinstance(c1, TelemetryClient)


def test_mandatory_client_alias():
    """MandatoryTelemetryClient is an alias for TelemetryClient."""
    client = MandatoryTelemetryClient()
    assert client.check_license()["allowed"] is True
