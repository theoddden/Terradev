#!/usr/bin/env python3
"""
TPU integration tests.

Exercises the full GCP TPU lifecycle (quotes, provisioning, status,
termination, and CLI dry-runs) without making real GCP API calls.
"""

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure terradev root is importable for this standalone file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terradev_cli.providers.gcp_provider import GCPProvider
from terradev_cli.providers.gpu_catalog import (
    get_canonical_name,
    GPU_ALIASES,
    GPU_SPECS,
)
from terradev_cli.providers.types import GPUVendor


# ---------------------------------------------------------------------------
# Fake google.cloud.compute_v1 module (google-cloud-compute is not required)
# ---------------------------------------------------------------------------


def _build_fake_compute_v1():
    """Return a fake compute_v1 module that exercises the GCP provider."""
    mod = types.ModuleType("google.cloud.compute_v1")

    class _Capture:
        """Base for captured request objects."""

        def __init__(self, **kwargs):
            self._fields = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Instance(_Capture):
        def __init__(self):
            self.name = None
            self.machine_type = None
            self.disks = []
            self.network_interfaces = []
            self.labels = {}
            self.scheduling = None
            self.metadata = None

    class AttachedDisk(_Capture):
        def __init__(self):
            self.auto_delete = False
            self.boot = False
            self.initialize_params = None

    class AttachedDiskInitializeParams(_Capture):
        def __init__(self):
            self.source_image = None
            self.disk_size_gb = None

    class NetworkInterface(_Capture):
        def __init__(self):
            self.access_configs = []

    class AccessConfig(_Capture):
        def __init__(self):
            self.name = None
            self.type_ = None

    class Scheduling(_Capture):
        def __init__(self, *, on_host_maintenance=None, provisioning_model=None):
            self.on_host_maintenance = on_host_maintenance
            self.provisioning_model = provisioning_model

    class Metadata(_Capture):
        def __init__(self, items=None):
            self.items = items or []

    class Items(_Capture):
        def __init__(self, *, key=None, value=None):
            self.key = key
            self.value = value

    class InsertInstanceRequest(_Capture):
        pass

    class GetInstanceRequest(_Capture):
        pass

    class StopInstanceRequest(_Capture):
        pass

    class StartInstanceRequest(_Capture):
        pass

    class DeleteInstanceRequest(_Capture):
        pass

    class InstancesClient:
        def __init__(self, *args, **kwargs):
            self.insert_calls = []
            self.get_calls = []
            self.stop_calls = []
            self.start_calls = []
            self.delete_calls = []

        def insert(self, request):
            self.insert_calls.append(request)

        def get(self, request):
            self.get_calls.append(request)
            if getattr(request, "zone", None) == "us-east5-c":
                raise Exception("not found")
            inst = Instance()
            inst.status = "RUNNING"
            inst.machine_type = f"zones/{request.zone}/machineTypes/ct6e-standard-8t"
            return inst

        def stop(self, request):
            self.stop_calls.append(request)

        def start(self, request):
            self.start_calls.append(request)

        def delete(self, request):
            self.delete_calls.append(request)

    class AcceleratorTypesClient:
        def __init__(self, *args, **kwargs):
            pass

    class ReservationsClient:
        def __init__(self, *args, **kwargs):
            pass

    mod.Instance = Instance
    mod.AttachedDisk = AttachedDisk
    mod.AttachedDiskInitializeParams = AttachedDiskInitializeParams
    mod.NetworkInterface = NetworkInterface
    mod.AccessConfig = AccessConfig
    mod.Scheduling = Scheduling
    mod.Metadata = Metadata
    mod.Items = Items
    mod.InsertInstanceRequest = InsertInstanceRequest
    mod.GetInstanceRequest = GetInstanceRequest
    mod.StopInstanceRequest = StopInstanceRequest
    mod.StartInstanceRequest = StartInstanceRequest
    mod.DeleteInstanceRequest = DeleteInstanceRequest
    mod.InstancesClient = InstancesClient
    mod.AcceleratorTypesClient = AcceleratorTypesClient
    mod.ReservationsClient = ReservationsClient

    return mod


@pytest.fixture(autouse=True)
def _patch_google_cloud(monkeypatch):
    """Inject a fake google.cloud.compute_v1 so GCP tests run offline."""
    fake = _build_fake_compute_v1()
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    cloud_pkg = types.ModuleType("google.cloud")
    cloud_pkg.__path__ = []
    oauth_pkg = types.ModuleType("google.oauth2")
    oauth_pkg.__path__ = []
    sa_pkg = types.ModuleType("google.oauth2.service_account")
    sa_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud.compute_v1", fake)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth_pkg)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_pkg)
    yield


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def gcp_tpu_provider():
    """GCP provider configured for TPU in a supported zone."""
    return GCPProvider(
        {
            "project_id": "test-project",
            "zone": "us-east5-a",
        }
    )


# ---------------------------------------------------------------------------
# Catalog / normalization
# ---------------------------------------------------------------------------


class TestTPUCatalog:
    def test_tpu_keys_are_canonical(self):
        assert "TPU-V6E-8T" in GPU_SPECS
        assert "TPU-V6E-8T" in GPU_ALIASES

    def test_tpu_vendor_is_google(self):
        spec = GPU_SPECS["TPU-V6E-8T"]
        assert spec.vendor == GPUVendor.GOOGLE
        assert spec.count == 8

    def test_tpu_canonical_name_normalization(self):
        assert get_canonical_name("tpu-v6e-8t") == "TPU-V6E-8T"
        assert get_canonical_name("TPU-V6E-8") == "TPU-V6E-8T"
        assert get_canonical_name("TPU-V5P-4T") == "TPU-V5P-4T"


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


class TestTPUQuotes:
    def test_tpu_quote_in_supported_region(self, gcp_tpu_provider):
        quotes = _run_async(
            gcp_tpu_provider.get_instance_quotes("TPU-V6E-8T", region="us-east5")
        )
        assert len(quotes) == 1
        q = quotes[0]
        assert q["gpu_type"] == "TPU-V6E-8T"
        assert q["instance_type"] == "ct6e-standard-8t"
        assert q["tpu_chips"] == 8
        assert q["available"] is True
        assert q["zone_availability"]["recommended_zone"].startswith("us-east5")

    def test_tpu_quote_respects_zone_from_self_zone(self, gcp_tpu_provider):
        quotes = _run_async(
            gcp_tpu_provider.get_instance_quotes("TPU-V6E-8T")
        )
        assert quotes[0]["zone_availability"]["recommended_zone"] == "us-east5-a"

    def test_tpu_quote_returns_empty_for_unsupported_region(self, gcp_tpu_provider):
        quotes = _run_async(
            gcp_tpu_provider.get_instance_quotes("TPU-V6E-8T", region="europe-west1")
        )
        assert quotes == []

    def test_tpu_quote_v5p(self, gcp_tpu_provider):
        quotes = _run_async(
            gcp_tpu_provider.get_instance_quotes("TPU-V5P-4T", region="us-east5")
        )
        assert quotes[0]["instance_type"] == "ct5p-hightpu-4t"
        assert quotes[0]["tpu_chips"] == 4

    def test_tpu_quote_tpu7x(self):
        provider = GCPProvider({"project_id": "test-project", "zone": "us-central1-c"})
        quotes = _run_async(
            provider.get_instance_quotes("TPU-V7X-4T", region="us-central1")
        )
        assert quotes[0]["instance_type"] == "tpu7x-standard-4t"
        assert quotes[0]["zone_availability"]["recommended_zone"].startswith("us-central1")

    def test_is_tpu_type_and_normalization(self, gcp_tpu_provider):
        assert gcp_tpu_provider._is_tpu_type("tpu-v6e-8t") is True
        assert gcp_tpu_provider._is_tpu_type("TPU-V6E-8") is True
        assert gcp_tpu_provider._is_tpu_type("A100") is False
        assert gcp_tpu_provider._normalize_tpu_key("tpu-v6e-8") == "TPU-V6E-8T"


# ---------------------------------------------------------------------------
# Provisioning lifecycle
# ---------------------------------------------------------------------------


class TestTPUProvisioning:
    def test_provision_instance_tpu(self, gcp_tpu_provider):
        result = _run_async(
            gcp_tpu_provider.provision_instance(
                "ct6e-standard-8t",
                "us-east5",
                "TPU-V6E-8T",
            )
        )
        assert result["status"] == "provisioning"
        assert result["tpu_chips"] == 8
        assert result["tpu_type"] == "ct6e-standard-8t"
        assert result["metadata"]["zone"].startswith("us-east5")

        # Verify the request passed to the Compute Engine API
        client = gcp_tpu_provider.instances_client
        assert len(client.insert_calls) == 1
        req = client.insert_calls[0]
        assert req.project == "test-project"
        assert req.zone == "us-east5-a"
        inst = req.instance_resource
        assert inst.name.startswith("terradev-tpu-v6e-8t-")
        assert "tpu" in inst.labels["accelerator"]
        assert inst.scheduling.on_host_maintenance == "TERMINATE"
        assert inst.scheduling.provisioning_model == "STANDARD"

    def test_provision_uses_zone_in_region_when_self_zone_doesnt_match(
        self, gcp_tpu_provider
    ):
        # Provider is in us-east5-a, v5p also exists there
        result = _run_async(
            gcp_tpu_provider.provision_instance(
                "ct5p-hightpu-4t", "us-east5", "TPU-V5P-4T"
            )
        )
        assert result["metadata"]["zone"] == "us-east5-a"

    def test_get_instance_status_resolves_zone(self, gcp_tpu_provider):
        _run_async(
            gcp_tpu_provider.provision_instance(
                "ct6e-standard-8t", "us-east5", "TPU-V6E-8T"
            )
        )
        status = _run_async(
            gcp_tpu_provider.get_instance_status(gcp_tpu_provider.instances_client.insert_calls[0].instance_resource.name)
        )
        assert status["status"] == "running"

    def test_terminate_instance_resolves_zone(self, gcp_tpu_provider):
        _run_async(
            gcp_tpu_provider.provision_instance(
                "ct6e-standard-8t", "us-east5", "TPU-V6E-8T"
            )
        )
        name = gcp_tpu_provider.instances_client.insert_calls[0].instance_resource.name
        result = _run_async(gcp_tpu_provider.terminate_instance(name))
        assert result["status"] == "terminating"
        assert len(gcp_tpu_provider.instances_client.delete_calls) == 1


# ---------------------------------------------------------------------------
# CLI dry-runs
# ---------------------------------------------------------------------------


class TestTPUCli:
    def test_run_dry_run_shows_tpu_quote(self, runner, mock_api):
        from terradev_cli.commands import cli

        tpu_quote = {
            "provider": "GCP",
            "price": 21.60,
            "gpu_type": "TPU-V6E-8T",
            "region": "us-east5",
            "availability": "on-demand",
            "gpu_count": 8,
            "instance_type": "ct6e-standard-8t",
            "memory_gb": 1440,
            "tpu_chips": 8,
            "tpu_type": "ct6e-standard-8t",
        }

        mock_api.get_gcp_quotes = AsyncMock(return_value=[tpu_quote])
        mock_api.get_runpod_quotes = AsyncMock(return_value=[])
        mock_api.get_vastai_quotes = AsyncMock(return_value=[])
        mock_api.get_aws_quotes = AsyncMock(return_value=[])
        mock_api.get_azure_quotes = AsyncMock(return_value=[])
        mock_api.get_tensordock_quotes = AsyncMock(return_value=[])
        mock_api.get_lambda_quotes = AsyncMock(return_value=[])
        mock_api.get_coreweave_quotes = AsyncMock(return_value=[])
        mock_api.get_oracle_quotes = AsyncMock(return_value=[])
        mock_api.get_crusoe_quotes = AsyncMock(return_value=[])

        result = runner.invoke(
            cli,
            ["run", "-g", "TPU-V6E-8T", "-i", "vllm/vllm-tpu:latest", "--dry-run"],
            obj={"api": mock_api},
        )
        assert result.exit_code == 0, result.output
        assert "TPU" in result.output
        assert "21.60" in result.output
        assert "vllm/vllm-tpu" in result.output
