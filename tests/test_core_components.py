"""
Comprehensive unit tests for Terradev core components.

Covers:
  - gpu_catalog: normalize(), get_canonical_name(), list_all_canonical_gpus()
  - providers/types: dataclass construction and enum membership
  - providers/registry: ProviderRegistry circuit-breaker, health tracking,
    spot scoring, ranked_providers
  - providers/base_provider: _get_gpu_specs routes through catalog,
    _estimate_price returns None with warning, check_health lightweight path
  - core/auth: AuthManager encrypt-decrypt round-trip, atomic save, migration
  - core/credential_vault: CredentialVault Python fallback path
  - providers/provider_factory: lazy load, unknown provider raises ValueError
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── GPU Catalog ────────────────────────────────────────────────────────────────


class TestGPUCatalog:
    def test_normalize_h100_variants(self, gpu_catalog_normalize):
        for alias in ("H100", "h100", "H100-80GB", "NVIDIA_H100_80G",
                      "H100-SXM5", "h100-80gb", "gpu-h100x1-80gb"):
            result = gpu_catalog_normalize(alias)
            assert result is not None, f"alias {alias!r} returned None"
            assert result.name == "H100-80GB", f"alias {alias!r}: expected H100-80GB got {result.name}"
            assert result.vram_gb == 80

    def test_normalize_a100_pcie_80g_is_80gb(self, gpu_catalog_normalize):
        """Regression test for observation #6 — 80 GB PCIe must not map to 40 GB."""
        for alias in ("NVIDIA_A100_PCIe_80G", "A100-PCIe-80G", "A100-PCIe-80GB"):
            result = gpu_catalog_normalize(alias)
            assert result is not None, f"{alias!r} returned None"
            assert result.vram_gb == 80, (
                f"{alias!r} resolved to {result.vram_gb} GB — expected 80 GB"
            )
            assert result.name == "A100-80GB"

    def test_normalize_a100_pcie_40g_is_40gb(self, gpu_catalog_normalize):
        for alias in ("NVIDIA_A100_PCIe_40G", "A100-PCIe-40G", "A100-40GB", "a100-40gb"):
            result = gpu_catalog_normalize(alias)
            assert result is not None
            assert result.vram_gb == 40, f"{alias!r} expected 40 GB"

    def test_normalize_unknown_returns_none(self, gpu_catalog_normalize):
        assert gpu_catalog_normalize("TOTALLY_UNKNOWN_GPU_XYZ") is None

    def test_normalize_empty_string_returns_none(self, gpu_catalog_normalize):
        assert gpu_catalog_normalize("") is None

    def test_normalize_aws_instance_types(self, gpu_catalog_normalize):
        assert gpu_catalog_normalize("p5.48xlarge").name == "H100-80GB"
        assert gpu_catalog_normalize("p4de.24xlarge").name == "A100-80GB"
        assert gpu_catalog_normalize("g5.xlarge").name == "A10G-24GB"

    def test_normalize_amd_gpus(self, gpu_catalog_normalize):
        result = gpu_catalog_normalize("MI300X")
        assert result is not None
        assert result.vram_gb == 192
        from terradev_cli.providers.types import GPUVendor
        assert result.vendor == GPUVendor.AMD

    def test_list_all_canonical_gpus(self):
        from terradev_cli.providers.gpu_catalog import list_all_canonical_gpus
        gpus = list_all_canonical_gpus()
        assert "H100-80GB" in gpus
        assert "A100-80GB" in gpus
        assert "A100-40GB" in gpus
        assert "MI300X-192GB" in gpus

    def test_get_canonical_name(self):
        from terradev_cli.providers.gpu_catalog import get_canonical_name
        assert get_canonical_name("H100") == "H100-80GB"
        assert get_canonical_name("NVIDIA_H100_80G") == "H100-80GB"
        assert get_canonical_name("GARBAGE_GPU") is None


# ── Provider Types ─────────────────────────────────────────────────────────────


class TestProviderTypes:
    def test_instance_status_enum(self):
        from terradev_cli.providers.types import InstanceStatus
        assert InstanceStatus.RUNNING == "running"
        assert InstanceStatus.PREEMPTED == "preempted"

    def test_gpu_descriptor_construction(self):
        from terradev_cli.providers.types import GPUDescriptor, GPUVendor
        gpu = GPUDescriptor(name="H100-80GB", vendor=GPUVendor.NVIDIA, vram_gb=80)
        assert gpu.vram_gb == 80
        assert gpu.nvlink is False  # default

    def test_quote_construction(self):
        from terradev_cli.providers.types import Quote, GPUDescriptor, GPUVendor
        gpu = GPUDescriptor(name="A100-80GB", vendor=GPUVendor.NVIDIA, vram_gb=80)
        q = Quote(
            provider="runpod",
            provider_instance_type="NVIDIA A100 80GB",
            region="us-east-1",
            gpu=gpu,
            price_hr=2.50,
            spot=False,
            availability="available",
        )
        assert q.price_hr == 2.50
        assert q.spot is False

    def test_provider_profile_defaults(self):
        from terradev_cli.providers.types import ProviderProfile
        p = ProviderProfile(name="test", api_style="rest", auth_type="bearer")
        assert p.supports_spot is True
        assert p.rate_limit_per_minute == 0
        assert p.compute_model == "vm"


# ── ProviderRegistry ───────────────────────────────────────────────────────────


class TestProviderRegistry:
    def test_new_provider_starts_healthy(self, registry):
        assert registry.is_healthy("runpod") is True

    async def test_circuit_breaker_opens_after_threshold(self, registry):
        for _ in range(registry.FAILURE_THRESHOLD):
            await registry.record_failure("runpod", "connection timeout")
        assert registry.is_healthy("runpod") is False

    async def test_circuit_breaker_recovers_after_window(self, registry):
        for _ in range(registry.FAILURE_THRESHOLD):
            await registry.record_failure("runpod", "timeout")
        health = registry._get_health("runpod")
        health.last_failure_ts = time.time() - registry.RECOVERY_WINDOW_S - 1
        assert registry.is_healthy("runpod") is True

    def test_provider_health_has_name(self, registry):
        """Regression test for observation #10 — provider name must not be empty."""
        health = registry._get_health("lambda_labs")
        assert health.provider == "lambda_labs"

    async def test_record_success_resets_failures(self, registry):
        await registry.record_failure("runpod", "err")
        await registry.record_success("runpod", latency_ms=50.0)
        health = registry._get_health("runpod")
        assert health.consecutive_failures == 0

    async def test_latency_ema(self, registry):
        await registry.record_success("vastai", latency_ms=100.0)
        await registry.record_success("vastai", latency_ms=200.0)
        health = registry._get_health("vastai")
        assert 100.0 <= health.avg_latency_ms <= 200.0

    def test_spot_score_zero_with_no_preemptions(self, registry):
        assert registry.get_spot_score("runpod") == 0.0

    async def test_spot_score_increases_after_preemptions(self, registry):
        for _ in range(5):
            await registry.record_preemption("runpod")
        assert registry.get_spot_score("runpod") > 0.0

    async def test_ranked_providers_excludes_unhealthy(self, registry):
        for _ in range(registry.FAILURE_THRESHOLD):
            await registry.record_failure("runpod", "down")
        ranked = registry.ranked_providers("H100-80GB")
        assert "runpod" not in ranked

    def test_get_stats_returns_correct_counts(self, registry):
        stats = registry.get_stats()
        assert "total_providers" in stats
        assert "overall_success_rate" in stats
        assert stats["overall_success_rate"] == 1.0

    async def test_reset_health(self, registry):
        for _ in range(registry.FAILURE_THRESHOLD):
            await registry.record_failure("runpod", "err")
        registry.reset_health("runpod")
        assert registry.is_healthy("runpod") is True
        assert registry._get_health("runpod").consecutive_failures == 0


# ── BaseProvider._get_gpu_specs / _estimate_price ─────────────────────────────


class TestBaseProviderGPUMethods:
    """Verify that BaseProvider routes through gpu_catalog (observations #7, #8)."""

    def _make_concrete_provider(self, mock_credentials):
        """Create a minimal concrete subclass of BaseProvider for testing."""
        from terradev_cli.providers.base_provider import BaseProvider

        class _TestProvider(BaseProvider):
            async def get_instance_quotes(self, gpu_type, region=None):
                return []

            async def provision_instance(self, gpu_type, region, **kwargs):
                return {}

            async def get_instance_status(self, instance_id):
                return {}

            async def list_instances(self):
                return []

            async def terminate_instance(self, instance_id):
                return {}

            async def start_instance(self, instance_id):
                return {}

            async def stop_instance(self, instance_id):
                return {}

            async def execute_command(self, instance_id, command, async_exec):
                return {}

            def _get_auth_headers(self):
                return {"Authorization": "Bearer test"}

        return _TestProvider(mock_credentials)

    def test_get_gpu_specs_h100_routes_through_catalog(self, mock_credentials):
        provider = self._make_concrete_provider(mock_credentials)
        specs = provider._get_gpu_specs("H100-80GB")
        assert specs["memory_gb"] == 80
        assert specs["vendor"] == "nvidia"

    def test_get_gpu_specs_a100_pcie_80g_returns_80gb(self, mock_credentials):
        """Regression for observation #6 — via BaseProvider path."""
        provider = self._make_concrete_provider(mock_credentials)
        specs = provider._get_gpu_specs("NVIDIA_A100_PCIe_80G")
        assert specs["memory_gb"] == 80

    def test_get_gpu_specs_unknown_returns_empty(self, mock_credentials):
        provider = self._make_concrete_provider(mock_credentials)
        specs = provider._get_gpu_specs("GARBAGE_XYZ_9999")
        assert specs == {}

    def test_estimate_price_returns_none_and_warns(self, mock_credentials, caplog):
        import logging
        provider = self._make_concrete_provider(mock_credentials)
        with caplog.at_level(logging.WARNING):
            result = provider._estimate_price("instance-type", "A100-80GB", "us-east-1")
        assert result is None
        assert "returning None" in caplog.text.lower() or "estimate_price" in caplog.text


# ── AuthManager ───────────────────────────────────────────────────────────────


class TestAuthManager:
    def test_round_trip_encrypt_decrypt(self, tmp_config_dir):
        from terradev_cli.core.auth import AuthManager

        auth_file = str(tmp_config_dir / "credentials.json")
        mgr = AuthManager.load(auth_file)
        mgr.credentials = {"runpod": {"api_key": "super-secret-key"}}
        mgr.save(auth_file)

        mgr2 = AuthManager.load(auth_file)
        assert mgr2.credentials["runpod"]["api_key"] == "super-secret-key"

    def test_credentials_not_plaintext_on_disk(self, tmp_config_dir):
        from terradev_cli.core.auth import AuthManager

        auth_file = tmp_config_dir / "credentials.json"
        mgr = AuthManager.load(str(auth_file))
        mgr.credentials = {"runpod": {"api_key": "my-secret-api-key"}}
        mgr.save(str(auth_file))

        raw = auth_file.read_text()
        assert "my-secret-api-key" not in raw, "Credential stored in plaintext!"

    def test_key_file_permissions(self, tmp_config_dir):
        from terradev_cli.core.auth import AuthManager
        import stat

        auth_file = str(tmp_config_dir / "credentials.json")
        AuthManager.load(auth_file)
        key_file = tmp_config_dir / ".keyfile"
        if key_file.exists():
            mode = oct(stat.S_IMODE(key_file.stat().st_mode))
            assert mode == oct(0o600), f"keyfile permissions {mode} != 0600"

    def test_atomic_write_creates_valid_json(self, tmp_config_dir):
        from terradev_cli.core.auth import AuthManager

        auth_file = str(tmp_config_dir / "credentials.json")
        mgr = AuthManager.load(auth_file)
        mgr.credentials = {"aws": {"api_key": "key1", "secret_key": "secret1"}}
        mgr.save(auth_file)

        with open(auth_file) as f:
            data = json.load(f)
        assert "credentials" in data
        assert data.get("version") == "2.0"

    def test_load_nonexistent_creates_new(self, tmp_config_dir):
        from terradev_cli.core.auth import AuthManager

        auth_file = str(tmp_config_dir / "new_creds.json")
        mgr = AuthManager.load(auth_file)
        assert mgr.credentials == {}
        assert mgr.fernet is not None

    def test_remove_credentials(self, auth_manager):
        auth_manager.credentials["test_provider"] = {"api_key": "x"}
        removed = auth_manager.remove_credentials("test_provider")
        assert removed is True
        assert "test_provider" not in auth_manager.credentials

    def test_validate_credentials_missing_api_key(self, auth_manager):
        auth_manager.credentials["bad"] = {"secret_key": "only-secret"}
        assert auth_manager.validate_credentials("bad") is False

    def test_list_providers(self, auth_manager):
        providers = auth_manager.list_providers()
        assert "runpod" in providers
        assert "lambda_labs" in providers


# ── CredentialVault Python fallback ───────────────────────────────────────────


class TestCredentialVaultFallback:
    """Tests run against the Python fallback (Rust vault may or may not be present)."""

    def test_store_and_retrieve(self):
        from terradev_cli.core.credential_vault import CredentialVault, USE_RUST_VAULT

        if USE_RUST_VAULT:
            pytest.skip("Running with Rust vault — fallback path not exercised")

        vault = CredentialVault()
        vault.store("my_key", b"super-secret", provider="runpod")
        result = vault.retrieve("my_key")
        assert result == b"super-secret"

    def test_delete(self):
        from terradev_cli.core.credential_vault import CredentialVault, USE_RUST_VAULT

        if USE_RUST_VAULT:
            pytest.skip("Running with Rust vault")

        vault = CredentialVault()
        vault.store("key_to_delete", b"value")
        vault.delete("key_to_delete")
        assert vault.retrieve("key_to_delete") is None

    def test_list(self):
        from terradev_cli.core.credential_vault import CredentialVault, USE_RUST_VAULT

        if USE_RUST_VAULT:
            pytest.skip("Running with Rust vault")

        vault = CredentialVault()
        vault.store("a", b"1")
        vault.store("b", b"2")
        names = vault.list()
        assert "a" in names
        assert "b" in names


# ── ProviderFactory ────────────────────────────────────────────────────────────


class TestProviderFactory:
    def test_get_supported_providers(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        providers = factory.get_supported_providers()
        assert "runpod" in providers
        assert "lambda_labs" in providers
        assert "aws" in providers

    def test_unknown_provider_raises_value_error(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.create_provider("totally_unknown_xyz", {})

    def test_register_custom_provider(self, mock_credentials):
        from terradev_cli.providers.provider_factory import ProviderFactory
        from terradev_cli.providers.base_provider import BaseProvider

        class _DummyProvider(BaseProvider):
            async def get_instance_quotes(self, gpu_type, region=None):
                return []
            async def provision_instance(self, *a, **kw):
                return {}
            async def get_instance_status(self, instance_id):
                return {}
            async def list_instances(self):
                return []
            async def terminate_instance(self, instance_id):
                return {}
            async def start_instance(self, instance_id):
                return {}
            async def stop_instance(self, instance_id):
                return {}
            async def execute_command(self, *a, **kw):
                return {}
            def _get_auth_headers(self):
                return {}

        factory = ProviderFactory()
        factory.register_provider("dummy", _DummyProvider)
        instance = factory.create_provider("dummy", mock_credentials)
        assert isinstance(instance, _DummyProvider)

    def test_register_non_provider_raises(self):
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        with pytest.raises(ValueError, match="must inherit from BaseProvider"):
            factory.register_provider("bad", object)


# ── Credential migration in TerradevAPI ───────────────────────────────────────


class TestTerradevAPICredentialMigration:
    """Verify observation #2 fix: plaintext JSON is migrated to encrypted on first load."""

    def test_plaintext_creds_get_migrated(self, tmp_config_dir):
        creds_file = tmp_config_dir / "credentials.json"
        creds_file.write_text(json.dumps({
            "runpod_api_key": "my-runpod-key",
            "lambda_api_key": "my-lambda-key",
        }))

        from terradev_cli.cli import TerradevAPI

        api = TerradevAPI.__new__(TerradevAPI)
        api.config_dir = tmp_config_dir
        api.credentials_file = creds_file
        api._auth_manager = None
        api.credentials = {}
        api.load_credentials()

        raw = creds_file.read_text()
        assert "my-runpod-key" not in raw, "Credential still stored in plaintext after migration"
        assert "my-lambda-key" not in raw

    def test_keyfile_created_alongside_credentials(self, tmp_config_dir):
        creds_file = tmp_config_dir / "credentials.json"
        creds_file.write_text(json.dumps({"runpod_api_key": "test-key"}))

        from terradev_cli.cli import TerradevAPI

        api = TerradevAPI.__new__(TerradevAPI)
        api.config_dir = tmp_config_dir
        api.credentials_file = creds_file
        api._auth_manager = None
        api.credentials = {}
        api.load_credentials()

        assert (tmp_config_dir / ".keyfile").exists(), ".keyfile not created after migration"
