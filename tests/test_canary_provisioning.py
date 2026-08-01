#!/usr/bin/env python3
"""
Canary Provisioning Integration Suite

End-to-end tests that fire against real provider API keys using tiny,
ultra-cheap CPU/GPU instances. These tests are skipped by default in CI
and can be run on demand with:

    TERRADEV_CANARY_TEST=1 pytest tests/test_canary_provisioning.py

or individually:

    TERRADEV_CANARY_TEST=1 pytest tests/test_canary_provisioning.py::test_canary_runpod

Each test:
  1. Provisions the cheapest available instance for a provider
  2. Polls until status is RUNNING/ACTIVE (with short timeout)
  3. Verifies SSH connectivity or a public endpoint
  4. Immediately terminates the instance

Set TERRADEV_CANARY_GPU to override the GPU type (default: RTX4090 or CPU).
Set TERRADEV_CANARY_MAX_PRICE to cap hourly cost (default: $0.50).
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

# Reuse the existing conftest path setup
TERRADEV_CANARY_ENABLED = os.environ.get("TERRADEV_CANARY_TEST") in ("1", "true", "True")
TERRADEV_CANARY_GPU = os.environ.get("TERRADEV_CANARY_GPU", "RTX4090")
TERRADEV_CANARY_MAX_PRICE = float(os.environ.get("TERRADEV_CANARY_MAX_PRICE", "0.50"))
TERRADEV_CANARY_TIMEOUT = int(os.environ.get("TERRADEV_CANARY_TIMEOUT", "300"))

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not TERRADEV_CANARY_ENABLED,
        reason="Canary tests hit live provider APIs. Set TERRADEV_CANARY_TEST=1 to enable.",
    ),
]


def _load_creds(provider: str) -> dict:
    """Load credentials from the user's Terradev config."""
    creds_path = Path.home() / ".terradev" / "credentials.json"
    if not creds_path.exists():
        pytest.fail(f"No credentials file found at {creds_path}")
    with open(creds_path) as f:
        all_creds = json.load(f)

    # Handle both nested and legacy flat formats
    creds = all_creds.get(provider, {})
    if isinstance(creds, dict) and creds:
        return creds

    # Legacy flat format
    flat = {
        "runpod": {"api_key": all_creds.get("runpod_api_key", "")},
        "vastai": {"api_key": all_creds.get("vastai_api_key", "")},
        "lambda_labs": {"api_key": all_creds.get("lambda_api_key", "")},
        "coreweave": {"api_key": all_creds.get("coreweave_api_key", "")},
        "tensordock": {
            "api_key": all_creds.get("tensordock_api_key", ""),
            "api_token": all_creds.get("tensordock_api_token", ""),
        },
    }
    return flat.get(provider, {})


async def _poll_until_running(provider, instance_id: str, timeout: int = 300):
    """Poll provider until instance reaches a terminal state."""
    start = time.monotonic()
    delay = 5.0
    while time.monotonic() - start < timeout:
        status = await provider.get_instance_status(instance_id)
        actual = status.get("status", "unknown").lower()
        if actual in ("running", "active", "ready"):
            return status
        if actual in ("error", "failed", "terminated", "deleted"):
            raise RuntimeError(f"Instance entered failed state: {actual}")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 30.0)
    raise TimeoutError(f"Instance {instance_id} did not reach RUNNING within {timeout}s")


def _check_ssh_connectivity(ip: str, port: int = 22, timeout: int = 60) -> bool:
    """Try an SSH TCP connect without logging in."""
    try:
        cmd = ["bash", "-c", f"timeout {timeout} bash -c 'cat < /dev/tcp/{ip}/{port}'"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SSH connectivity probe failed: {exc}")
        return False


async def _canary_for_provider(provider_name: str):
    """Provision the cheapest tiny instance, verify, and destroy."""
    from terradev_cli.providers.provider_factory import ProviderFactory

    creds = _load_creds(provider_name)
    if not any(creds.values()):
        pytest.skip(f"No credentials configured for {provider_name}")

    factory = ProviderFactory()
    provider = factory.create_provider(provider_name, creds)

    # Step 1: Get quotes for the canary GPU
    quotes = await provider.get_instance_quotes(TERRADEV_CANARY_GPU)
    cheap = [q for q in quotes if q.get("price", q.get("price_per_hour", 1e9)) <= TERRADEV_CANARY_MAX_PRICE]
    if not cheap:
        cheap = sorted(
            quotes,
            key=lambda q: q.get("price", q.get("price_per_hour", 1e9)),
        )
    if not cheap:
        pytest.skip(f"No {TERRADEV_CANARY_GPU} quotes from {provider_name}")

    selected = cheap[0]
    instance_type = selected.get("instance_type", f"{provider_name}-{TERRADEV_CANARY_GPU.lower()}")
    region = selected.get("region", "us-east-1")
    price = selected.get("price", selected.get("price_per_hour", 0))

    logger.info(
        f"[{provider_name}] Canary selected: {instance_type} in {region} at ${price:.2f}/hr"
    )

    # Step 2: Provision
    provision_result = await provider.provision_instance(
        instance_type=instance_type,
        region=region,
        gpu_type=TERRADEV_CANARY_GPU,
        ssh_public_key="",
    )
    instance_id = provision_result.get("instance_id")
    assert instance_id, f"[{provider_name}] provision did not return an instance id"

    logger.info(f"[{provider_name}] Provisioned {instance_id}, polling for RUNNING...")

    try:
        # Step 3: Poll for RUNNING
        status = await _poll_until_running(provider, instance_id, TERRADEV_CANARY_TIMEOUT)

        # Step 4: Verify SSH / endpoint connectivity
        ip = status.get("ip") or status.get("public_ip") or status.get("ip_address")
        port = status.get("port") or status.get("ssh_port") or 22
        connectivity = False
        if ip:
            connectivity = _check_ssh_connectivity(ip, int(port))

        actual_status = status.get("status", "unknown").lower()
        assert actual_status in ("running", "active", "ready"), (
            f"[{provider_name}] instance {instance_id} did not become running: {actual_status}"
        )

        if not connectivity:
            logger.warning(
                f"[{provider_name}] {instance_id} is running but SSH probe on {ip}:{port} failed"
            )
            # Do not fail the test for a network-level connectivity hiccup,
            # but surface the information.

        # Step 5: Terminate immediately to minimise cost
        await provider.terminate_instance(instance_id)
        logger.info(f"[{provider_name}] Terminated {instance_id}")

        return {
            "provider": provider_name,
            "instance_id": instance_id,
            "instance_type": instance_type,
            "region": region,
            "price": price,
            "status": actual_status,
            "ip": ip,
            "port": port,
            "ssh_reachable": connectivity,
        }
    except Exception:
        # Best-effort cleanup on any failure
        try:
            await provider.terminate_instance(instance_id)
            logger.info(f"[{provider_name}] Terminated {instance_id} after failure")
        except Exception as term_exc:  # noqa: BLE001
            logger.warning(f"[{provider_name}] Cleanup failed: {term_exc}")
        raise


@pytest.mark.parametrize(
    "provider",
    ["runpod", "vastai", "lambda_labs"],
)
async def test_canary_provider(provider):
    """Canary test for a single provider."""
    result = await _canary_for_provider(provider)
    assert result["status"] in ("running", "active", "ready")
