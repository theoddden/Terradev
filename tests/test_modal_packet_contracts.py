#!/usr/bin/env python3
"""
Modal and Packet.ai provider drift tests.

These tests mock the HTTP layer and snapshot the request payloads sent by the
new Modal and Packet providers, plus verify response parsing for quotes and
instance status.  Snapshots are stored alongside the other provider contract
snapshots in tests/snapshots/provider_contracts/.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "provider_contracts"


def _normalize_modal_name(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove timestamp suffix from Modal app names so snapshots are stable."""
    out = json.loads(json.dumps(payload, default=str))
    if isinstance(out, dict) and isinstance(out.get("json"), dict):
        if "name" in out["json"]:
            out["json"]["name"] = re.sub(r"-\d{14}$", "-<TIMESTAMP>", out["json"]["name"])
    if isinstance(out, dict) and "name" in out:
        out["name"] = re.sub(r"-\d{14}$", "-<TIMESTAMP>", out["name"])
    return out


def _write_snapshot(provider: str, label: str, data: Any):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{provider}_{label}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _read_snapshot(provider: str, label: str) -> Any:
    path = SNAPSHOT_DIR / f"{provider}_{label}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _assert_matches_snapshot(provider: str, label: str, actual: Any):
    """Compare actual request shape to snapshot; create snapshot if missing."""
    normalized = _normalize_modal_name(actual) if provider == "modal" else actual
    snapshot = _read_snapshot(provider, label)
    if snapshot is None:
        _write_snapshot(provider, label, normalized)
        pytest.skip(f"Created new snapshot for {provider}/{label}: run again to verify")
    assert normalized == snapshot, (
        f"{provider}/{label} request shape drifted from snapshot.\n"
        f"Expected: {json.dumps(snapshot, indent=2)}\n"
        f"Actual:   {json.dumps(normalized, indent=2)}"
    )


# ── Modal drift tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_modal_provision_payload_contract():
    """Verify Modal provision_instance builds the expected App creation payload."""
    from terradev_cli.providers.modal_provider import ModalProvider

    provider = ModalProvider({"token_id": "test-id", "token_secret": "test-secret"})
    captured: List[Dict[str, Any]] = []

    async def _fake_make_request(method, url, **kwargs):
        captured.append({
            "method": method,
            "url": url,
            "headers": kwargs.get("headers", {}),
            "json": kwargs.get("json"),
        })
        return {"data": {"id": "app-12345", "name": "terradev-h100-<TIMESTAMP>", "status": "provisioning"}}

    with patch.object(provider, "_make_request", _fake_make_request):
        result = await provider.provision_instance(
            instance_type="modal-h100",
            region="us-east",
            gpu_type="H100-80GB",
            ssh_public_key="ssh-rsa test",
        )

    assert result.get("instance_id") == "app-12345"
    assert len(captured) == 1
    call = captured[0]
    assert call["url"] == "https://api.modal.com/v1/apps"
    assert call["json"]["gpu"] == "H100"
    assert call["json"]["region"] == "us-east"
    _assert_matches_snapshot("modal", "provision_payload", _normalize_modal_name(call))


@pytest.mark.asyncio
async def test_modal_quote_shape_contract():
    """Verify Modal quote parsing maps canonical GPU names and returns expected fields."""
    from terradev_cli.providers.modal_provider import ModalProvider

    provider = ModalProvider({"token_id": "test-id", "token_secret": "test-secret"})

    sample_response = {
        "data": [
            {
                "gpu": "H100",
                "price_per_hour": 2.99,
                "available": True,
                "regions": ["us-east", "us-west"],
                "vcpus": 16,
                "ram_gb": 64,
                "gpu_memory_gb": 80,
            }
        ]
    }

    captured_url = None

    async def _fake_make_request(method, url, **_kwargs):
        nonlocal captured_url
        captured_url = url
        return sample_response

    with patch.object(provider, "_make_request", _fake_make_request):
        quotes = await provider.get_instance_quotes("H100-80GB", region="us-east")

    assert captured_url == "https://api.modal.com/v1/gpu-types"
    assert len(quotes) == 1
    assert quotes[0]["provider"] == "modal"
    assert quotes[0]["gpu_type"] == "H100-80GB"
    assert "price_per_hour" in quotes[0]
    assert quotes[0]["serverless"] is True


@pytest.mark.asyncio
async def test_modal_status_parsing_contract():
    """Verify Modal get_instance_status normalizes App status and extracts endpoint."""
    from terradev_cli.providers.modal_provider import ModalProvider

    provider = ModalProvider({"token_id": "test-id", "token_secret": "test-secret"})

    sample_response = {
        "data": {
            "id": "app-12345",
            "name": "test-app",
            "status": "RUNNING",
            "gpu": "H100",
            "region": "us-east",
            "endpoint": "https://test-app.modal.run",
        }
    }

    async def _fake_make_request(*_args, **_kwargs):
        return sample_response

    with patch.object(provider, "_make_request", _fake_make_request):
        status = await provider.get_instance_status("app-12345")

    assert status["status"] == "running"
    assert status["endpoint"] == "https://test-app.modal.run"
    assert status["provider"] == "modal"


# ── Packet.ai drift tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_packet_provision_payload_contract():
    """Verify Packet provision_instance builds the expected instance creation payload."""
    from terradev_cli.providers.packet_provider import PacketProvider

    provider = PacketProvider({"api_key": "test-key"})
    captured: List[Dict[str, Any]] = []

    async def _fake_make_request(method, url, **kwargs):
        captured.append({
            "method": method,
            "url": url,
            "headers": kwargs.get("headers", {}),
            "json": kwargs.get("json"),
        })
        return {
            "data": {
                "id": "inst-12345",
                "status": "provisioning",
                "gpu_type": "h100",
                "region": "us-east",
                "tier": "dedicated",
            }
        }

    with patch.object(provider, "_make_request", _fake_make_request):
        result = await provider.provision_instance(
            instance_type="packet-h100-dedicated",
            region="us-east",
            gpu_type="H100-80GB",
            ssh_public_key="ssh-rsa test",
        )

    assert result.get("instance_id") == "inst-12345"
    assert len(captured) == 1
    call = captured[0]
    assert call["url"] == "https://api.packet.ai/v1/instances"
    assert call["json"]["gpu_type"] == "h100"
    assert call["json"]["tier"] == "dedicated"
    assert call["json"]["region"] == "us-east"
    _assert_matches_snapshot("packet", "provision_payload", call)


@pytest.mark.asyncio
async def test_packet_quote_shape_contract():
    """Verify Packet quote parsing returns dynamic and dedicated tiers."""
    from terradev_cli.providers.packet_provider import PacketProvider

    provider = PacketProvider({"api_key": "test-key"})

    sample_response = {
        "data": [
            {
                "type": "h100",
                "price_per_hour": 1.95,
                "status": "available",
                "vcpus": 24,
                "ram_gb": 128,
                "vram_gb": 80,
            }
        ]
    }

    captured_url = None

    async def _fake_make_request(method, url, **_kwargs):
        nonlocal captured_url
        captured_url = url
        return sample_response

    with patch.object(provider, "_make_request", _fake_make_request):
        quotes = await provider.get_instance_quotes("H100-80GB", region="us-east")

    assert captured_url == "https://api.packet.ai/v1/gpus"
    assert len(quotes) == 2
    assert quotes[0]["provider"] == "packet"
    assert quotes[0]["gpu_type"] == "h100"
    assert "tier" in quotes[0]
    assert "spot" in quotes[0]
    assert quotes[0]["spot"] is True
    assert quotes[1]["spot"] is False


@pytest.mark.asyncio
async def test_packet_status_parsing_contract():
    """Verify Packet get_instance_status normalizes instance status and extracts SSH info."""
    from terradev_cli.providers.packet_provider import PacketProvider

    provider = PacketProvider({"api_key": "test-key"})

    sample_response = {
        "data": {
            "id": "inst-12345",
            "status": "RUNNING",
            "gpu_type": "h100",
            "region": "us-east",
            "tier": "dedicated",
            "public_ip": "203.0.113.10",
            "ssh": "ssh ubuntu@gpu-inst-12345.packet.ai -p 30122",
        }
    }

    async def _fake_make_request(*_args, **_kwargs):
        return sample_response

    with patch.object(provider, "_make_request", _fake_make_request):
        status = await provider.get_instance_status("inst-12345")

    assert status["status"] == "running"
    assert status["public_ip"] == "203.0.113.10"
    assert status["provider"] == "packet"
    assert status["spot"] is False
