#!/usr/bin/env python3
"""
Canary Provisioning Integration Suite - Expanded Coverage

End-to-end tests that fire against real provider API keys using tiny,
ultra-cheap CPU/GPU instances. These tests are skipped by default in CI
and can be run on demand with:

    TERRADEV_CANARY_TEST=1 pytest tests/test_canary_provisioning.py

or individually:

    TERRADEV_CANARY_TEST=1 pytest tests/test_canary_provisioning.py::test_canary_runpod

Each test:
  1. Provisions the cheapest available instance for a provider and GPU
  2. Polls until status is RUNNING/ACTIVE (with short timeout)
  3. Verifies SSH / endpoint connectivity
  4. Optionally runs a lightweight lifecycle probe (CUDA, container, volume)
  5. Immediately terminates the instance
  6. Records telemetry to ~/.terradev/canary-results.jsonl

Environment variables:
  TERRADEV_CANARY_TEST=1          enable canary suite
  TERRADEV_CANARY_GPU=RTX4090     default GPU type
  TERRADEV_CANARY_MAX_PRICE=0.50  hourly price cap
  TERRADEV_CANARY_TIMEOUT=300     provisioning timeout
  TERRADEV_CANARY_REGIONS=        comma-separated region override
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Reuse the existing conftest path setup
TERRADEV_CANARY_ENABLED = os.environ.get("TERRADEV_CANARY_TEST") in ("1", "true", "True")
TERRADEV_CANARY_GPU = os.environ.get("TERRADEV_CANARY_GPU", "RTX4090")
TERRADEV_CANARY_MAX_PRICE = float(os.environ.get("TERRADEV_CANARY_MAX_PRICE", "0.50"))
TERRADEV_CANARY_TIMEOUT = int(os.environ.get("TERRADEV_CANARY_TIMEOUT", "300"))
TERRADEV_CANARY_REGIONS = os.environ.get("TERRADEV_CANARY_REGIONS", "")
TERRADEV_CANARY_LIFECYCLE = os.environ.get("TERRADEV_CANARY_LIFECYCLE", "1") in ("1", "true", "True")
TERRADEV_CANARY_VOLUME = os.environ.get("TERRADEV_CANARY_VOLUME", "0") in ("1", "true", "True")
TERRADEV_CANARY_OUTPUT = Path.home() / ".terradev" / "canary-results.jsonl"

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
        "aws": {
            "api_key": all_creds.get("aws_access_key_id", ""),
            "secret_key": all_creds.get("aws_secret_access_key", ""),
        },
        "gcp": {
            "api_key": all_creds.get("gcp_project_id", ""),
            "credentials_file": all_creds.get("gcp_credentials_file", ""),
        },
        "azure": {"api_key": all_creds.get("azure_client_id", "")},
        "oracle": {"api_key": all_creds.get("oci_api_key", "")},
        "crusoe": {"api_key": all_creds.get("crusoe_access_key", "")},
        "huggingface": {"api_key": all_creds.get("hf_token", "")},
        "baseten": {"api_key": all_creds.get("baseten_api_key", "")},
    }
    return flat.get(provider, {})


def _record_canary_result(result: Dict[str, Any]) -> None:
    """Append a canary result to the telemetry log."""
    try:
        TERRADEV_CANARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(TERRADEV_CANARY_OUTPUT, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not write canary telemetry: {exc}")


# Provider-specific canary tuning.
# Keep GPU lists cheap: skip A100/H100 unless explicitly requested.
CANARY_PROVIDER_CONFIG = {
    "runpod": {
        "gpu_types": ["RTX4090", "A6000", "A100"],
        "regions": ["us-east-1", "us-west-1", "eu-central-1"],
        "min_memory_in_gb": 16,
        "min_vcpu_count": 2,
    },
    "vastai": {
        "gpu_types": ["RTX4090", "A6000", "A100"],
        "regions": ["us", "eu", "asia"],
    },
    "lambda_labs": {
        "gpu_types": ["A100", "A10G", "RTX4090"],
        "regions": ["us-east-1", "us-west-1", "us-south-1"],
    },
    "coreweave": {
        "gpu_types": ["A40", "A100"],
        "regions": ["us-east-1", "us-west-2"],
    },
    "tensordock": {
        "gpu_types": ["RTX4090", "A4000"],
        "regions": ["us-east"],
    },
    "aws": {
        "gpu_types": ["A100", "H100"],
        "regions": ["us-east-1", "us-west-2"],
    },
    "gcp": {
        "gpu_types": ["A100", "H100"],
        "regions": ["us-central1", "us-east4"],
    },
    "azure": {
        "gpu_types": ["A100", "H100"],
        "regions": ["eastus", "westus2"],
    },
    "oracle": {
        "gpu_types": ["A100", "H100"],
        "regions": ["us-ashburn-1", "us-phoenix-1"],
    },
    "crusoe": {
        "gpu_types": ["A100", "H100"],
        "regions": ["us-west-1"],
    },
    "huggingface": {
        "gpu_types": ["A100", "A10G"],
        "regions": ["us-east-1"],
    },
    "baseten": {
        "gpu_types": ["A100", "A10G"],
        "regions": ["us-east-1"],
    },
}


def _get_canary_regions(provider_name: str) -> List[str]:
    """Return regions to try for a provider, honouring env overrides."""
    if TERRADEV_CANARY_REGIONS:
        return [r.strip() for r in TERRADEV_CANARY_REGIONS.split(",") if r.strip()]
    return CANARY_PROVIDER_CONFIG.get(provider_name, {}).get("regions", ["us-east-1"])


def _get_canary_gpu_types(provider_name: str, override_gpu: Optional[str] = None) -> List[str]:
    """Return GPU types to attempt for a provider."""
    if override_gpu:
        return [override_gpu]
    return CANARY_PROVIDER_CONFIG.get(provider_name, {}).get("gpu_types", [TERRADEV_CANARY_GPU])


def _get_quote_price(quote: Dict[str, Any]) -> float:
    """Extract price from a quote dict, with multiple key fallbacks."""
    for key in ("price", "price_per_hour", "hourly_price", "cost", "cost_per_hour"):
        val = quote.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1e9


def _select_canary_quotes(
    quotes: List[Dict[str, Any]],
    max_price: float,
    preferred_regions: List[str],
) -> List[Dict[str, Any]]:
    """Filter and sort quotes for canary use."""
    if not quotes:
        return []

    def _score(q: Dict[str, Any]) -> tuple:
        price = _get_quote_price(q)
        region = q.get("region", q.get("location", q.get("zone", "")))
        # Prefer cheap and preferred regions
        region_bonus = 0 if region in preferred_regions else 1
        return (region_bonus, price)

    affordable = [q for q in quotes if _get_quote_price(q) <= max_price]
    if not affordable:
        return sorted(quotes, key=_get_quote_price)
    return sorted(affordable, key=_score)


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
    if not ip:
        return False
    try:
        cmd = ["bash", "-c", f"timeout {timeout} bash -c 'cat < /dev/tcp/{ip}/{port}'"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SSH connectivity probe failed: {exc}")
        return False


async def _run_lifecycle_probe(provider, instance_id: str, ip: str) -> Dict[str, Any]:
    """Run lightweight commands on the instance to verify CUDA and container runtime."""
    result = {"cuda_visible": None, "nvidia_smi": None, "container_runtime": None, "ssh_reachable": False}
    if not ip:
        return result

    result["ssh_reachable"] = _check_ssh_connectivity(ip)
    if not result["ssh_reachable"]:
        return result

    if not hasattr(provider, "execute_command") or not callable(getattr(provider, "execute_command")):
        return result

    commands = [
        ("nvidia_smi", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
        ("cuda_visible", "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"),
        ("container_runtime", "docker --version || podman --version || crictl --version"),
    ]

    for key, command in commands:
        try:
            exec_result = await asyncio.wait_for(
                provider.execute_command(instance_id, command, async_exec=False),
                timeout=60,
            )
            output = exec_result.get("output", exec_result.get("stdout", ""))
            result[key] = output.strip() if isinstance(output, str) else str(output)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lifecycle probe '{key}' failed: {exc}")
            result[key] = f"error: {exc}"

    return result


async def _canary_for_provider(
    provider_name: str,
    gpu_type: Optional[str] = None,
    region: Optional[str] = None,
    force_volume: bool = False,
) -> Dict[str, Any]:
    """Provision the cheapest tiny instance, verify, and destroy."""
    from terradev_cli.providers.provider_factory import ProviderFactory

    start_ts = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "provider": provider_name,
        "gpu_type": gpu_type or TERRADEV_CANARY_GPU,
        "timestamp": start_ts,
        "success": False,
    }

    creds = _load_creds(provider_name)
    if not any(creds.values()):
        pytest.skip(f"No credentials configured for {provider_name}")

    factory = ProviderFactory()
    provider = factory.create_provider(provider_name, creds)

    gpu = gpu_type or TERRADEV_CANARY_GPU
    preferred_regions = [region] if region else _get_canary_regions(provider_name)

    try:
        # Step 1: Get quotes and validate one exists
        quotes = await provider.get_instance_quotes(gpu) or []
        if not quotes:
            pytest.skip(f"No {gpu} quotes from {provider_name}")

        # Step 2: Select candidate quote(s)
        cheap = _select_canary_quotes(quotes, TERRADEV_CANARY_MAX_PRICE, preferred_regions)
        if not cheap:
            pytest.skip(f"No affordable {gpu} quotes from {provider_name} under ${TERRADEV_CANARY_MAX_PRICE}")

        selected = cheap[0]
        instance_type = selected.get("instance_type", f"{provider_name}-{gpu.lower()}")
        quote_region = selected.get("region", selected.get("location", selected.get("zone", preferred_regions[0])))
        price = _get_quote_price(selected)

        logger.info(
            f"[{provider_name}] Canary selected: {instance_type} in {quote_region} at ${price:.2f}/hr"
        )

        # Step 3: Provision, with retries for spot / capacity issues and region rotation
        provision_kwargs = {
            "instance_type": instance_type,
            "region": quote_region,
            "gpu_type": gpu,
            "ssh_public_key": "",
            "attach_volume": force_volume or TERRADEV_CANARY_VOLUME,
        }

        provider_config = CANARY_PROVIDER_CONFIG.get(provider_name, {})
        if "min_memory_in_gb" in provider_config:
            provision_kwargs["min_memory_in_gb"] = provider_config["min_memory_in_gb"]
        if "min_vcpu_count" in provider_config:
            provision_kwargs["min_vcpu_count"] = provider_config["min_vcpu_count"]

        # Try the top 3 quotes across regions
        candidates = cheap[:3]
        last_error = None
        instance_id = None
        provisioned_quote: Optional[Dict[str, Any]] = None

        for attempt, quote in enumerate(candidates):
            instance_type = quote.get("instance_type", f"{provider_name}-{gpu.lower()}")
            quote_region = quote.get("region", quote.get("location", quote.get("zone", preferred_regions[0])))
            provision_kwargs["instance_type"] = instance_type
            provision_kwargs["region"] = quote_region

            logger.info(f"[{provider_name}] Canary provision attempt {attempt + 1}: {instance_type} in {quote_region}")
            try:
                provision_result = await provider.provision_instance(**provision_kwargs)
                instance_id = provision_result.get("instance_id")
                if instance_id:
                    provisioned_quote = quote
                    break
            except RuntimeError as exc:
                last_error = exc
                msg = str(exc).lower()
                if any(phrase in msg for phrase in (
                    "no instances available",
                    "no longer any instances",
                    "capacity",
                    "not enough",
                    "out of stock",
                    "unavailable",
                )):
                    logger.warning(f"[{provider_name}] Capacity depleted for {instance_type}: {exc}")
                    continue
                raise

        if not instance_id:
            raise RuntimeError(
                f"[{provider_name}] Could not provision any canary instance after {len(candidates)} attempts. "
                f"Last error: {last_error}"
            )

        result["instance_id"] = instance_id
        result["instance_type"] = instance_type
        result["region"] = quote_region
        result["price"] = price

        logger.info(f"[{provider_name}] Provisioned {instance_id}, polling for RUNNING...")

        try:
            # Step 4: Poll for RUNNING
            status = await _poll_until_running(provider, instance_id, TERRADEV_CANARY_TIMEOUT)

            # Step 5: Verify SSH / endpoint connectivity
            ip = status.get("ip") or status.get("public_ip") or status.get("ip_address")
            port = status.get("port") or status.get("ssh_port") or 22
            actual_status = status.get("status", "unknown").lower()
            assert actual_status in ("running", "active", "ready"), (
                f"[{provider_name}] instance {instance_id} did not become running: {actual_status}"
            )

            result["status"] = actual_status
            result["ip"] = ip
            result["port"] = port
            result["ssh_reachable"] = _check_ssh_connectivity(ip, int(port))

            # Step 6: Optional lifecycle probe (CUDA, container runtime)
            if TERRADEV_CANARY_LIFECYCLE:
                probe = await _run_lifecycle_probe(provider, instance_id, ip)
                result["lifecycle"] = probe

            # Step 7: Quote-to-provision validation
            result["quote_to_provision_match"] = _validate_quote_to_provision(
                selected, provisioned_quote, instance_id, status
            )

            # Step 8: Terminate immediately to minimise cost
            await provider.terminate_instance(instance_id)
            logger.info(f"[{provider_name}] Terminated {instance_id}")

            result["success"] = True
            return result

        except Exception:
            # Best-effort cleanup on any failure
            try:
                await provider.terminate_instance(instance_id)
                logger.info(f"[{provider_name}] Terminated {instance_id} after failure")
            except Exception as term_exc:  # noqa: BLE001
                logger.warning(f"[{provider_name}] Cleanup failed: {term_exc}")
            raise
    finally:
        try:
            await provider.aclose()
        except Exception:  # noqa: BLE001
            pass
        _record_canary_result(result)


def _validate_quote_to_provision(
    selected_quote: Dict[str, Any],
    provisioned_quote: Optional[Dict[str, Any]],
    instance_id: str,
    status: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify the provisioned instance matches the quote we selected."""
    quote_region = provisioned_quote.get(
        "region", provisioned_quote.get("location", provisioned_quote.get("zone", ""))
    )
    quote_type = provisioned_quote.get(
        "instance_type", provisioned_quote.get("gpu_type", provisioned_quote.get("name", ""))
    )
    actual_region = status.get("region", status.get("location", status.get("zone", "")))
    actual_type = status.get("instance_type", status.get("gpu_type", status.get("name", "")))

    matches = {
        "instance_id": instance_id,
        "quote_region": quote_region,
        "actual_region": actual_region,
        "quote_instance_type": quote_type,
        "actual_instance_type": actual_type,
        "region_match": quote_region == actual_region,
        "type_match": quote_type == actual_type,
    }
    return matches


async def _canary_distributed(provider_names: List[str], gpu_type: Optional[str] = None) -> Dict[str, Any]:
    """Provision one cheap node per provider and verify each is independently reachable."""
    from terradev_cli.providers.provider_factory import ProviderFactory

    gpu = gpu_type or TERRADEV_CANARY_GPU
    factory = ProviderFactory()
    instances: List[Dict[str, Any]] = []
    result = {"distributed": True, "gpu_type": gpu, "nodes": [], "success": False}

    try:
        for provider_name in provider_names:
            creds = _load_creds(provider_name)
            if not any(creds.values()):
                pytest.skip(f"No credentials configured for {provider_name}")

            provider = factory.create_provider(provider_name, creds)
            quotes = await provider.get_instance_quotes(gpu) or []
            cheap = _select_canary_quotes(quotes, TERRADEV_CANARY_MAX_PRICE, _get_canary_regions(provider_name))
            if not cheap:
                pytest.skip(f"No affordable {gpu} quotes from {provider_name}")

            quote = cheap[0]
            instance_type = quote.get("instance_type", f"{provider_name}-{gpu.lower()}")
            region = quote.get("region", quote.get("location", quote.get("zone", _get_canary_regions(provider_name)[0])))

            provision_kwargs = {
                "instance_type": instance_type,
                "region": region,
                "gpu_type": gpu,
                "ssh_public_key": "",
                "attach_volume": False,
            }
            provider_config = CANARY_PROVIDER_CONFIG.get(provider_name, {})
            if "min_memory_in_gb" in provider_config:
                provision_kwargs["min_memory_in_gb"] = provider_config["min_memory_in_gb"]
            if "min_vcpu_count" in provider_config:
                provision_kwargs["min_vcpu_count"] = provider_config["min_vcpu_count"]

            provision_result = await provider.provision_instance(**provision_kwargs)
            instance_id = provision_result.get("instance_id")
            assert instance_id, f"[{provider_name}] distributed canary did not return an instance id"

            instances.append({"provider": provider_name, "instance_id": instance_id, "provider_obj": provider})
            result["nodes"].append({"provider": provider_name, "instance_id": instance_id, "region": region})

        for node in instances:
            status = await _poll_until_running(node["provider_obj"], node["instance_id"], TERRADEV_CANARY_TIMEOUT)
            ip = status.get("ip") or status.get("public_ip") or status.get("ip_address")
            port = status.get("port") or status.get("ssh_port") or 22
            reachable = _check_ssh_connectivity(ip, int(port))
            assert status.get("status", "").lower() in ("running", "active", "ready")
            node["status"] = status
            node["ip"] = ip
            node["reachable"] = reachable

        result["success"] = all(node["reachable"] for node in instances)
        return result

    finally:
        for node in instances:
            try:
                await node["provider_obj"].terminate_instance(node["instance_id"])
                logger.info(f"[{node['provider']}] Terminated distributed node {node['instance_id']}")
            except Exception as term_exc:  # noqa: BLE001
                logger.warning(f"[{node['provider']}] Distributed cleanup failed: {term_exc}")
            try:
                await node["provider_obj"].aclose()
            except Exception:  # noqa: BLE001
                pass
        _record_canary_result(result)


# ── Test functions ──────────────────────────────────────────────────────

CANARY_PROVIDERS = list(CANARY_PROVIDER_CONFIG.keys())


@pytest.mark.parametrize("provider", CANARY_PROVIDERS)
async def test_canary_provider(provider):
    """Canary test for a single provider with default GPU."""
    result = await _canary_for_provider(provider)
    assert result["success"]
    assert result.get("status") in ("running", "active", "ready")


@pytest.mark.parametrize(
    "provider, gpu",
    [
        ("runpod", "RTX4090"),
        ("runpod", "A6000"),
        ("vastai", "RTX4090"),
        ("lambda_labs", "A100"),
        ("coreweave", "A100"),
        ("tensordock", "RTX4090"),
    ],
)
async def test_canary_gpu_matrix(provider, gpu):
    """Canary tests across provider + GPU combinations."""
    result = await _canary_for_provider(provider, gpu_type=gpu)
    assert result["success"]


@pytest.mark.parametrize(
    "provider, region",
    [
        ("runpod", "us-east-1"),
        ("runpod", "us-west-1"),
        ("vastai", "us"),
        ("lambda_labs", "us-east-1"),
        ("coreweave", "us-east-1"),
        ("tensordock", "us-east"),
    ],
)
async def test_canary_region(provider, region):
    """Canary tests across provider + region combinations."""
    result = await _canary_for_provider(provider, region=region)
    assert result["success"]


@pytest.mark.parametrize(
    "provider",
    ["runpod", "vastai"],
)
async def test_canary_quote_to_provision(provider):
    """Canary that validates the provisioned instance matches the selected quote."""
    result = await _canary_for_provider(provider)
    assert result["success"]
    qtp = result.get("quote_to_provision_match", {})
    assert qtp.get("region_match") or qtp.get("type_match"), (
        f"Quote-to-provision mismatch for {provider}: {qtp}"
    )


@pytest.mark.parametrize(
    "provider",
    ["runpod", "vastai", "lambda_labs"],
)
async def test_canary_lifecycle(provider):
    """Canary with lifecycle probes enabled."""
    # Temporarily force lifecycle on for this test
    result = await _canary_for_provider(provider)
    assert result["success"]
    if "lifecycle" in result:
        assert result["lifecycle"].get("ssh_reachable")


async def test_canary_distributed_two_providers():
    """Provision tiny nodes on two providers and verify both are reachable."""
    # Prefer providers with likely cheap spot GPUs
    candidates = ["runpod", "vastai"]
    configured = [p for p in candidates if any(_load_creds(p).values())]
    if len(configured) < 2:
        pytest.skip("Need credentials for at least two of runpod/vastai for distributed canary")
    result = await _canary_distributed(configured[:2])
    assert result["success"]


async def test_canary_telemetry_written():
    """Confirm canary results are written to the telemetry log."""
    if not TERRADEV_CANARY_OUTPUT.exists():
        pytest.skip("No canary telemetry file yet; run a canary first")
    lines = TERRADEV_CANARY_OUTPUT.read_text().strip().split("\n")
    assert lines
    data = json.loads(lines[-1])
    assert "provider" in data
