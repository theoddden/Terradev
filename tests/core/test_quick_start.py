"""Tests for terradev_cli.core.quick_start.

QuickStart is the interactive onboarding flow. These tests exercise the
provider metadata and internal helpers with mocked input.
"""

from unittest.mock import patch

from terradev_cli.core.quick_start import QuickStart, show_quick_start


def test_quick_start_provider_metadata():
    """QuickStart exposes demo provider metadata."""
    qs = QuickStart()
    assert "runpod" in qs.demo_providers
    assert "vastai" in qs.demo_providers
    assert qs.demo_providers["runpod"]["quick_start"] is True


def test_runpod_quick_start_with_mocks(tmp_path, monkeypatch):
    """RunPod quick start completes with mocked input and provision."""
    qs = QuickStart()
    qs._save_credentials = lambda p, c: None

    def fake_provision(api, gpu_type, provider):
        return {"instance_id": "i-1", "price": 0.2, "provider": provider}

    monkeypatch.setattr(qs, "_quick_provision", fake_provision)

    inputs = iter(["1", "1", "Y"])
    with patch("builtins.input", lambda *a, **k: next(inputs)):
        result = qs._runpod_quick_start()
        assert result is True


def test_vastai_quick_start_with_mocks(tmp_path, monkeypatch):
    """Vast.ai quick start completes with mocked input and provision."""
    qs = QuickStart()
    qs._save_credentials = lambda p, c: None

    def fake_provision(api, gpu_type, provider):
        return {"instance_id": "i-2", "price": 0.15, "provider": provider}

    monkeypatch.setattr(qs, "_quick_provision", fake_provision)

    inputs = iter(["1", "1", "Y"])
    with patch("builtins.input", lambda *a, **k: next(inputs)):
        result = qs._vastai_quick_start()
        assert result is True


def test_quick_start_invalid_choice(capsys, monkeypatch):
    """An invalid menu choice reports an error."""
    qs = QuickStart()
    with patch("builtins.input", lambda *a, **k: "9"):
        result = qs.show_quick_start_guide()
        assert result is False


def test_show_quick_start_runs():
    """The module-level show_quick_start function can be called."""
    with patch("terradev_cli.core.quick_start.QuickStart") as MockQS:
        instance = MockQS.return_value
        instance.show_quick_start_guide.return_value = True
        assert show_quick_start() is True
