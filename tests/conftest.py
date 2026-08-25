#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration for the Terradev test suite.

Centralises:
  - sys.path setup (no per-file hacks needed)
  - TERRADEV_SKIP_ONBOARDING env var (prevents interactive wizard in all tests)
  - Credential / AuthManager fixtures (encrypted temp files)
  - Provider mock fixtures (AsyncMock-backed BaseProvider)
  - TerradevAPI mock fixture
  - CliRunner fixture
  - GPU catalog helpers
  - ProviderRegistry fixture with seeded health data
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── MCP fallback stub for Python 3.9 where the real SDK cannot be installed ──
import importlib.util

_mcp_spec = importlib.util.find_spec("mcp")
_mcp_ok = False
if _mcp_spec is not None and _mcp_spec.origin not in (None, ""):
    try:
        import mcp as _mcp_test  # noqa: F401

        _mcp_ok = True
    except Exception:
        # Remove any partially initialised mcp modules so the stub can load.
        for _name in list(sys.modules):
            if _name == "mcp" or _name.startswith("mcp."):
                del sys.modules[_name]

if not _mcp_ok:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "_mcp_stub"))

# ── Global env: skip interactive onboarding for every test ───────────────────
os.environ["TERRADEV_SKIP_ONBOARDING"] = "1"

# Default tests to human output so existing text assertions keep passing.
# The CLI still respects --format / TERRADEV_OUTPUT for CI/Docker usage.
os.environ.setdefault("TERRADEV_OUTPUT", "human")


# ── CLI runner ────────────────────────────────────────────────────────────────

@pytest.fixture
def runner() -> CliRunner:
    """Isolated Click test runner."""
    return CliRunner()


# ── Temporary config directory ────────────────────────────────────────────────

@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """A fresh temporary config directory for each test."""
    config = tmp_path / ".terradev"
    config.mkdir()
    return config


# ── AuthManager / credential fixtures ────────────────────────────────────────

@pytest.fixture
def auth_manager(tmp_config_dir: Path):
    """
    A real AuthManager instance backed by a temp directory.

    that need credentials don't have to configure them manually.
    """
    from terradev_cli.core.auth import AuthManager

    auth_file = tmp_config_dir / "credentials.json"
    mgr = AuthManager.load(str(auth_file))
    mgr.credentials = {
        "runpod": {"api_key": "test-runpod-key-abc123"},
        "aws": {
            "api_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        },
    }
    mgr.save(str(auth_file))
    return mgr


@pytest.fixture
def mock_credentials() -> Dict[str, str]:
    """Flat credential dict used for provider instantiation in unit tests."""
    return {
        "api_key": "test-api-key-placeholder",
        "secret_key": "test-secret-placeholder",
    }


# ── TerradevAPI mock ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_api(tmp_config_dir: Path):
    """
    A fully-mocked TerradevAPI instance.

    All async provider methods return minimal valid payloads.  Credential
    and usage I/O is replaced with in-memory stubs so no disk access occurs.
    """
    from terradev_cli.commands._api import TerradevAPI

    api = MagicMock(spec=TerradevAPI)
    api.credentials = {
        "runpod": {"api_key": "test-key"},
    }
    api._auth_manager = None
    api.config_dir = tmp_config_dir
    api.credentials_file = tmp_config_dir / "credentials.json"
    api.usage_file = tmp_config_dir / "usage.json"

    _sample_quotes = [
        {
            "provider": "RunPod",
            "price": 1.50,
            "gpu_type": "A100-80GB",
            "region": "us-east-1",
            "availability": "on-demand",
            "gpu_count": 1,
            "instance_type": "runpod-a100-80gb",
            "memory_gb": 80,
        }
    ]

    api.get_runpod_quotes = AsyncMock(return_value=_sample_quotes)
    api.get_vastai_quotes = AsyncMock(return_value=[])
    api.get_aws_quotes = AsyncMock(return_value=[])
    api.get_gcp_quotes = AsyncMock(return_value=[])
    api.get_azure_quotes = AsyncMock(return_value=[])
    api.get_tensordock_quotes = AsyncMock(return_value=[])
    api.get_crusoe_quotes = AsyncMock(return_value=[])

    api.provision_instance = AsyncMock(return_value={
        "instance_id": "test-inst-abc123",
        "provider": "RunPod",
        "gpu_type": "A100-80GB",
        "status": "running",
        "ip": "10.0.0.1",
    })
    api.get_instance_status = AsyncMock(return_value={
        "status": "running",
        "gpu_utilization": 72,
        "instance_id": "test-inst-abc123",
    })

    api.is_first_time_user = MagicMock(return_value=False)
    api.load_credentials = MagicMock()
    api.save_credentials = MagicMock()
    api.load_usage = MagicMock(return_value={"provisions_this_month": 0})
    api.save_usage = MagicMock()
    api.check_provision_limit = MagicMock(return_value=True)
    api.record_provision = MagicMock()
    return api


# ── Provider mock ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider():
    """
    A mock BaseProvider with all abstract methods stubbed as AsyncMocks.

    The health check returns healthy by default.
    """
    from terradev_cli.providers.types import HealthStatus, InstanceStatus

    provider = MagicMock()
    provider.get_instance_quotes = AsyncMock(return_value=[
        {
            "provider": "mock",
            "gpu_type": "H100-80GB",
            "region": "us-east-1",
            "price_per_hour": 3.50,
            "spot": False,
            "gpu_count": 1,
            "memory_gb": 80,
            "instance_type": "mock-h100",
        }
    ])
    provider.provision_instance = AsyncMock(return_value={
        "instance_id": "mock-inst-001",
        "status": InstanceStatus.RUNNING,
        "ip": "192.168.1.1",
    })
    provider.get_instance_status = AsyncMock(return_value={
        "instance_id": "mock-inst-001",
        "status": InstanceStatus.RUNNING,
    })
    provider.list_instances = AsyncMock(return_value=[])
    provider.terminate_instance = AsyncMock(return_value={"success": True})
    provider.start_instance = AsyncMock(return_value={"success": True})
    provider.stop_instance = AsyncMock(return_value={"success": True})
    provider.execute_command = AsyncMock(return_value={"stdout": "", "stderr": "", "exit_code": 0})
    provider.check_health = AsyncMock(return_value=HealthStatus(
        healthy=True, latency_ms=12.5, timestamp=0.0
    ))
    provider.session = None
    return provider


# ── ProviderRegistry fixture ──────────────────────────────────────────────────

@pytest.fixture
def registry():
    """A ProviderRegistry with a mock factory and clean health state."""
    from terradev_cli.providers.registry import ProviderRegistry
    from terradev_cli.providers.provider_factory import ProviderFactory

    mock_factory = MagicMock(spec=ProviderFactory)
    mock_factory.get_supported_providers.return_value = ["runpod", "vastai"]
    return ProviderRegistry(factory=mock_factory)


@pytest.fixture
def patch_registry():
    """Patch ProviderRegistry so provision/quote logic uses a fixed provider ranking."""
    with patch("terradev_cli.providers.registry.ProviderRegistry") as MockRegistry:
        instance = MagicMock()
        instance.ranked_providers.return_value = [
            "runpod",
            "vastai",
            "aws",
            "gcp",
            "azure",
            "tensordock",
            "oracle",
            "crusoe",
        ]
        MockRegistry.return_value = instance
        yield MockRegistry


# ── GPU catalog helpers ───────────────────────────────────────────────────────

@pytest.fixture
def gpu_catalog_normalize():
    """Convenience wrapper around gpu_catalog.normalize for tests."""
    from terradev_cli.providers.gpu_catalog import normalize
    return normalize


# ── Async event loop (pytest-asyncio compat) ─────────────────────────────────

@pytest.fixture(scope="session")
def _loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(autouse=True)
def event_loop(_loop_policy):
    """Provide a fresh current event loop for every test.

    This prevents Python 3.9 from raising RuntimeError when earlier tests
    call asyncio.run and leave the main thread without a current loop.
    """
    loop = _loop_policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    try:
        loop.close()
    except Exception:
        pass
    finally:
        asyncio.set_event_loop(None)


# ── Legacy telemetry fixtures (kept for backward compat with older tests) ─────

@pytest.fixture
def base_url() -> str:
    return "http://localhost:8080"


@pytest.fixture
def server_name() -> str:
    return "Primary Server"
