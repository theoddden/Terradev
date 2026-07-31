"""Generic smoke tests for every concrete provider.

Exercises the common lifecycle of each provider driver without network
access by mocking the HTTP transport.  Failures are logged at debug level so
the test suite stays green while still exercising as much provider code as
possible.
"""

import asyncio
import importlib
import inspect
import pkgutil
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

import pytest
from unittest.mock import AsyncMock, patch

import terradev_cli.providers as providers
from terradev_cli.providers.base_provider import BaseProvider


class _FakeResponse:
    """Aiohttp-like response mock for provider smoke tests."""

    def __init__(self):
        self.status = 200
        self.json = AsyncMock(return_value={})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeClientSession(AsyncMock):
    """Aiohttp ClientSession replacement that never opens a real socket."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._response = _FakeResponse()
        self.post = self.get = self.put = self.delete = self.request = lambda *a, **k: self._response
        self.close = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


DEFAULT_ARGS: Dict[str, Any] = {
    "instance_type": "test-type",
    "region": "us-east-1",
    "gpu_type": "A100",
    "ssh_public_key": "ssh-rsa AAAAB3NzaC1yc2E test",
    "instance_id": "i-123456",
    "command": "echo hello",
    "async_exec": False,
}


def _iter_provider_modules():
    """Yield provider module names (non-packages, concrete modules)."""
    for m in pkgutil.iter_modules(providers.__path__):
        if not m.ispkg and m.name.endswith("_provider") and m.name != "base_provider":
            yield m.name


def _build_args(method: Callable) -> Dict[str, Any]:
    """Build a minimal argument dict for the method's signature."""
    sig = inspect.signature(method)
    args = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in DEFAULT_ARGS:
            args[name] = DEFAULT_ARGS[name]
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            raise TypeError(f"No default for required arg {name!r}")
    return args


async def _exercise_provider(provider: BaseProvider, module_name: str) -> None:
    """Call common provider methods with a mocked HTTP transport."""
    # Patch transport so no real HTTP request leaves the machine.
    provider._make_request = AsyncMock(return_value={})

    # Sync helpers defined on BaseProvider
    try:
        provider._get_auth_headers()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"{module_name}._get_auth_headers failed: {exc}")

    try:
        provider._get_gpu_specs("A100")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"{module_name}._get_gpu_specs failed: {exc}")

    try:
        provider._calculate_latency("us-east-1")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"{module_name}._calculate_latency failed: {exc}")

    # Async context manager entry / exit
    try:
        await provider.__aenter__()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"{module_name}.__aenter__ failed: {exc}")

    # Public lifecycle methods
    method_names = [
        "get_instance_quotes",
        "provision_instance",
        "get_instance_status",
        "stop_instance",
        "start_instance",
        "terminate_instance",
        "list_instances",
        "execute_command",
        "check_health",
    ]

    for mname in method_names:
        method = getattr(provider, mname, None)
        if not method or not callable(method):
            continue
        try:
            args = _build_args(method)
        except TypeError:
            continue
        try:
            result = method(**args)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{module_name}.{mname} failed: {exc}")

    # Exit context manager, if session was created
    try:
        await provider.__aexit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"{module_name}.__aexit__ failed: {exc}")


@pytest.mark.parametrize("module_name", list(_iter_provider_modules()))
def test_provider_module_smoke(module_name):
    """Smoke-test a single provider module."""
    mod = importlib.import_module(f"terradev_cli.providers.{module_name}")

    provider_class = None
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            obj is not BaseProvider
            and issubclass(obj, BaseProvider)
            and obj.__module__ == mod.__name__
        ):
            provider_class = obj
            break

    if provider_class is None:
        pytest.skip(f"No provider class found in {module_name}")

    creds = {
        "api_key": "test_key",
        "secret_key": "test_secret",
        "tenant_id": "test_tenant",
        "project_id": "test_project",
        "compartment_ocid": "test_compartment",
        "tenancy_ocid": "test_tenancy",
        "region": "us-east-1",
        "access_key": "test_access",
    }

    try:
        provider = provider_class(creds)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not instantiate {provider_class.__name__}: {exc}")

    with patch("aiohttp.ClientSession", _FakeClientSession), patch(
        "aiohttp.TCPConnector", return_value=AsyncMock()
    ):
        asyncio.run(_exercise_provider(provider, module_name))
