#!/usr/bin/env python3
"""
Terradev CLI - Complete Production Version
"""

import click
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import telemetry - MANDATORY FOR USAGE TRACKING
try:
    from terradev_cli.core.telemetry import get_mandatory_telemetry
    _telemetry = get_mandatory_telemetry()
except Exception as _exc:  # noqa: BLE001
    logger.exception(_exc)
    _telemetry = None

from terradev_cli.core.vault_adapter import VaultAdapter

# Import Kubernetes wrapper
try:
    from terradev_cli.k8s.terraform_wrapper import TerraformWrapper
except Exception as _exc:  # noqa: BLE001
    logger.exception(_exc)
    TerraformWrapper = None

# Import enterprise auth - OPTIONAL FOR ENTERPRISE TIERS
try:
    from terradev_cli.core.enterprise_auth import EnterpriseAuthManager
except ImportError:
    EnterpriseAuthManager = None


def validate_credentials(provider: str, credentials: Dict[str, str]) -> bool:
    """Validate that all required credentials are present for a provider"""
    required_creds = {
        "alibaba": ["access_key_id", "access_key_secret"],
        "aws": ["api_key", "secret_key"],
        "azure": ["subscription_id", "tenant_id", "client_id", "client_secret"],
        "baseten": ["api_key"],
        "coreweave": ["api_key"],
        "crusoe": ["access_key", "secret_key", "project_id"],
        "digitalocean": ["api_key"],
        "e2enetworks": ["api_key"],
        "fluidstack": ["api_key"],
        "gcp": ["project_id", "credentials_file"],
        "hetzner": ["api_token"],
        "huggingface": ["api_key", "namespace"],
        "hyperstack": ["api_key"],
        "inferx": ["api_key"],
        "lambda_labs": ["api_key"],
        "latitude": ["api_key"],
        "oracle": ["api_key", "tenancy_ocid", "compartment_ocid", "region"],
        "ovhcloud": ["application_key", "application_secret", "consumer_key", "project_id"],
        "runpod": ["api_key"],
        "siliconflow": ["api_key"],
        "tensordock": ["api_key", "api_token"],
        "vastai": ["api_key"],
        "yottalabs": ["api_key"],
    }

    provider_lower = provider.lower()
    if provider_lower not in required_creds:
        return False

    missing = []
    for req in required_creds[provider_lower]:
        if req not in credentials or not credentials[req].strip():
            missing.append(req)

    if missing:
        logger.debug(
            f"Missing credentials for {provider_lower}: {', '.join(missing)}"
        )
        return False

    return True


class TerradevAPI:
    """Open source provider API integration - no tiers, no limits"""

    def __init__(self):
        self.config_dir = Path.home() / ".terradev"
        self.config_dir.mkdir(exist_ok=True)
        self.credentials_file = self.config_dir / "credentials.json"
        self.usage_file = self.config_dir / "usage.json"
        # Tier system removed - no tier file needed

        self._vault = VaultAdapter(self.config_dir)
        self.load_credentials()
        self.load_usage()

        # Tier configuration removed - open source, unlimited access
        self.tier = None  # No tiers

        # Enterprise auth integration - available for all users
        self.enterprise_auth = None
        if EnterpriseAuthManager:
            try:
                self.enterprise_auth = EnterpriseAuthManager()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to initialize enterprise auth: {e}")

        # Initialize usage tracking
        if "inference_endpoints" not in self.usage:
            self.usage["inference_endpoints"] = []

        # GPU-hour metering sync removed - no billing

    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user with no configured credentials"""
        # Check if credentials file and any env-based credentials exist
        if not self.credentials_file.exists() and not self.credentials:
            return True

        # Check if credentials are empty or only contain default/placeholder values
        if not self.credentials or len(self.credentials) == 0:
            return True

        # Check if all credentials are still placeholder values
        placeholder_patterns = ["your_", "example_", "test_", "placeholder_", "xxx"]
        for key, value in self.credentials.items():
            if isinstance(value, dict):
                # Check nested credentials (from configure --provider)
                for nested_key, nested_value in value.items():
                    if nested_value and isinstance(nested_value, str) and not any(
                        pattern in nested_value.lower()
                        for pattern in placeholder_patterns
                    ):
                        return False  # Found a real nested credential
            elif value and isinstance(value, str) and not any(
                pattern in value.lower() for pattern in placeholder_patterns
            ):
                return False  # Found a real flat credential

        return True  # All credentials appear to be placeholders

    # Tier system removed - no tier loading, verification, or payment links
    # def _load_tier(self):
    #     """Load and cryptographically verify tier from local config - REMOVED"""
    #     pass

    # def _is_enterprise_tier(self) -> bool:
    #     """Check if current tier is enterprise or enterprise_plus - REMOVED"""
    #     return False  # No tiers

    def load_credentials(self):
        """Load cloud provider credentials via AuthManager (Fernet-encrypted).

        Migration: if a plain-JSON credentials file exists without the
        companion .keyfile, the flat-format contents are read, converted to
        the nested {provider: {key: val}} schema and re-saved encrypted so
        that subsequent runs use the secure path.

        Defensively create the vault adapter if it is missing (some tests
        construct ``TerradevAPI`` via ``__new__`` and only set a config dir).
        """
        if not hasattr(self, "_vault"):
            self._vault = VaultAdapter(getattr(self, "config_dir", None))

        from terradev_cli.core.auth import AuthManager

        key_file = self.config_dir / ".keyfile"

        if self.credentials_file.exists() and not key_file.exists():
            self._migrate_plaintext_credentials(key_file)

        try:
            auth = AuthManager.load(str(self.credentials_file))
            self._auth_manager = auth
            # File-backed credentials, then overlay any TERRADEV_* env vars for
            # known cloud providers only (avoids CI-only tokens such as
            # TERRADEV_GITHUB_TOKEN being treated as a provider credential).
            self.credentials = self._vault.load_env_credentials(auth.credentials, known_only=True)
        except Exception as e:  # noqa: BLE001
            import sys
            print(
                f"Warning: Failed to load credentials via AuthManager ({e}). "
                f"Your credentials file may be corrupted.",
                file=sys.stderr,
            )
            print(
                "   Run `terradev configure` to re-enter your keys.",
                file=sys.stderr,
            )
            # Still allow environment variables to work when the file is missing/broken.
            self.credentials = self._vault.load_env_credentials({}, known_only=True)
            self._auth_manager = None

    def _migrate_plaintext_credentials(self, key_file: Path) -> None:
        """One-time migration: convert flat plain-JSON credentials to encrypted nested format."""
        try:
            with open(self.credentials_file, "r") as f:
                flat: dict = json.load(f)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            return

        if not isinstance(flat, dict) or not flat:
            return

        from terradev_cli.core.auth import AuthManager

        auth = AuthManager()
        auth._create_new_auth_file(self.credentials_file, key_file)

        # Convert flat keys like "runpod_api_key" → {"runpod": {"api_key": value}}
        nested: dict = {}
        for raw_key, value in flat.items():
            if isinstance(value, dict):
                # Already nested (from a previous configure --provider run)
                nested[raw_key] = {k: str(v) for k, v in value.items() if v}
            elif "_" in raw_key:
                parts = raw_key.split("_", 1)
                provider, field = parts[0], parts[1]
                nested.setdefault(provider, {})[field] = str(value) if value else ""

        auth.credentials = nested
        try:
            auth.save(str(self.credentials_file))
            import sys
            print(
                "[terradev] Credentials migrated to encrypted storage.",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"Warning: credential migration failed ({e})", file=sys.stderr)

    def save_credentials(self):
        """Save cloud provider credentials via AuthManager (Fernet-encrypted)."""
        from terradev_cli.core.auth import AuthManager
        import sys

        try:
            if self._auth_manager is None:
                key_file = self.config_dir / ".keyfile"
                self._auth_manager = AuthManager.load(str(self.credentials_file))

            self._auth_manager.credentials = self.credentials
            self._auth_manager.save(str(self.credentials_file))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: Failed to save credentials: {e}", file=sys.stderr)

    def _save_provider_creds(self, provider_name: str, creds: Dict[str, str]) -> None:
        """Store/update credentials for a specific provider."""
        if not isinstance(self.credentials, dict):
            self.credentials = {}
        self.credentials[provider_name] = {
            k: str(v) if v is not None else "" for k, v in creds.items()
        }
        self.save_credentials()

    def load_usage(self):
        """Load usage tracking"""
        if self.usage_file.exists():
            with open(self.usage_file, "r") as f:
                try:
                    import fcntl

                    fcntl.flock(f, fcntl.LOCK_SH)
                    _has_fcntl = True
                except ImportError:
                    _has_fcntl = False
                try:
                    self.usage = json.load(f)
                finally:
                    if _has_fcntl:
                        fcntl.flock(f, fcntl.LOCK_UN)
        else:
            self.usage = {
                "provisions_this_month": 0,
                "month_start": datetime.now().replace(day=1).isoformat(),
                "instances_created": [],
                "inference_endpoints": [],
                "last_reset": datetime.now().isoformat(),
            }

    def save_usage(self):
        """Save usage tracking with exclusive file lock"""
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_file, "w") as f:
            try:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX)
                _has_fcntl = True
            except ImportError:
                _has_fcntl = False
            try:
                json.dump(self.usage, f, indent=2)
            finally:
                if _has_fcntl:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def check_provision_limit(self) -> bool:
        """Check if user has provisions remaining this month"""
        self._maybe_reset_monthly_usage()
        if self.tier is None:
            return True  # Open source mode: unlimited provisions
        limit = self.tier["provisions_per_month"]
        if limit == "unlimited":
            return True
        used = self.usage.get("provisions_this_month", 0)
        return used < limit

    def record_provision(self):
        """Increment the monthly provision counter"""
        self._maybe_reset_monthly_usage()
        self.usage["provisions_this_month"] = (
            self.usage.get("provisions_this_month", 0) + 1
        )
        self.save_usage()

    def _maybe_reset_monthly_usage(self):
        """Reset monthly counters if the calendar month has changed (R8 fix)"""
        month_start = datetime.fromisoformat(self.usage["month_start"])
        now = datetime.now()
        if (now.year, now.month) != (month_start.year, month_start.month):
            self.usage["provisions_this_month"] = 0
            self.usage["month_start"] = now.replace(day=1).isoformat()

    def _maybe_sync_gpu_metering(self):
        """Enterprise+ only: report accrued GPU-hours for all active instances.

        Runs on every CLI invocation (lightweight - reads local state only).
        This catches GPU-hours even if an instance was terminated outside the
        CLI (e.g. directly on the provider dashboard or via MCP subprocess).

        Billing model (metered, per-instance):
            billable_gpus = max(instance.gpu_count, 32)
            billable_gpu_hours += billable_gpus × hours_running
            Monthly minimum: 32 × 128 = 4,096 GPU-hrs ($368.64/mo)

        Examples:
          - 8 GPUs for 0.75 hrs  → max(8, 32) × 0.75  = 24 GPU-hrs
          - 72 GPUs for 32 hrs   → max(72, 32) × 32    = 2,304 GPU-hrs
          - 0 instances running  → $0 (no idle charge)
          - Month with only 500 GPU-hrs → billed 4,096 (shortfall top-up)
        """
        return  # BYOAPI: No billing reconciliation needed

    def _provider_creds(self, provider_name: str) -> Dict[str, str]:
        """Build credentials dict for a provider from stored BYOAPI keys.

        Supports two storage formats:
        1. Flat: {"runpod_api_key": "xxx"} (from onboarding wizard)
        2. Nested: {"runpod": {"api_key": "xxx"}} (from configure --provider)
        """
        if not self.credentials:
            return {}

        # Check for nested dict format first (from configure --provider)
        nested = self.credentials.get(provider_name, {})
        if isinstance(nested, dict) and nested:
            # Check if nested values are placeholders (not real credentials)
            placeholder_patterns = ["paste_", "your_", "example_", "test_", "placeholder_", "xxx"]
            has_real_creds = False
            for v in nested.values():
                if isinstance(v, str) and v:
                    v_lower = v.lower()
                    if not any(pattern in v_lower for pattern in placeholder_patterns):
                        has_real_creds = True
                        break
            
            if has_real_creds:
                # Nested dict has real credentials, use it
                return {k: str(v) for k, v in nested.items() if v}
            # Otherwise fall back to flat format

        creds: Dict[str, str] = {}
        if provider_name == "aws":
            creds["api_key"] = self.credentials.get("aws_access_key_id", "")
            creds["secret_key"] = self.credentials.get("aws_secret_access_key", "")
        elif provider_name == "gcp":
            creds["project_id"] = self.credentials.get("gcp_project_id", "")
            creds["credentials_file"] = self.credentials.get("gcp_credentials_file", "")
        elif provider_name == "azure":
            creds["subscription_id"] = self.credentials.get("azure_subscription_id", "")
            creds["tenant_id"] = self.credentials.get("azure_tenant_id", "")
            creds["client_id"] = self.credentials.get("azure_client_id", "")
            creds["client_secret"] = self.credentials.get("azure_client_secret", "")
        elif provider_name == "runpod":
            creds["api_key"] = self.credentials.get("runpod_api_key", "")
        elif provider_name == "vastai":
            creds["api_key"] = self.credentials.get("vastai_api_key", "")
        elif provider_name == "lambda_labs":
            creds["api_key"] = self.credentials.get("lambda_api_key", "")
        elif provider_name == "coreweave":
            creds["api_key"] = self.credentials.get("coreweave_api_key", "")
        elif provider_name == "tensordock":
            creds["api_key"] = self.credentials.get("tensordock_api_key", "")
            creds["api_token"] = self.credentials.get("tensordock_api_token", "")
        elif provider_name == "huggingface":
            creds["api_key"] = self.credentials.get("huggingface_api_token", "")
            creds["namespace"] = self.credentials.get("huggingface_namespace", "")
        elif provider_name == "baseten":
            creds["api_key"] = self.credentials.get("baseten_api_key", "")
        elif provider_name == "oracle":
            creds["api_key"] = self.credentials.get("oracle_api_key", "")
            creds["tenancy_ocid"] = self.credentials.get("oracle_tenancy_ocid", "")
            creds["compartment_ocid"] = self.credentials.get(
                "oracle_compartment_ocid", ""
            )
            creds["region"] = self.credentials.get("oracle_region", "us-ashburn-1")
        elif provider_name == "crusoe":
            creds["access_key"] = self.credentials.get("crusoe_access_key", "")
            creds["secret_key"] = self.credentials.get("crusoe_secret_key", "")
            creds["project_id"] = self.credentials.get("crusoe_project_id", "")
        elif provider_name == "alibaba":
            creds["access_key_id"] = self.credentials.get("alibaba_access_key_id", "")
            creds["access_key_secret"] = self.credentials.get(
                "alibaba_access_key_secret", ""
            )
            creds["region_id"] = self.credentials.get("alibaba_region_id", "cn-beijing")
            creds["security_group_id"] = self.credentials.get(
                "alibaba_security_group_id", ""
            )
            creds["vswitch_id"] = self.credentials.get("alibaba_vswitch_id", "")
        elif provider_name == "ovhcloud":
            creds["application_key"] = self.credentials.get(
                "ovhcloud_application_key", ""
            )
            creds["application_secret"] = self.credentials.get(
                "ovhcloud_application_secret", ""
            )
            creds["consumer_key"] = self.credentials.get("ovhcloud_consumer_key", "")
            creds["project_id"] = self.credentials.get("ovhcloud_project_id", "")
            creds["endpoint"] = self.credentials.get("ovhcloud_endpoint", "ovh-eu")
            creds["ssh_key_id"] = self.credentials.get("ovhcloud_ssh_key_id", "")
        elif provider_name == "fluidstack":
            creds["api_key"] = self.credentials.get("fluidstack_api_key", "")
            creds["ssh_key_name"] = self.credentials.get("fluidstack_ssh_key_name", "")
        elif provider_name == "hetzner":
            creds["api_token"] = self.credentials.get("hetzner_api_token", "")
            creds["robot_user"] = self.credentials.get("hetzner_robot_user", "")
            creds["robot_password"] = self.credentials.get("hetzner_robot_password", "")
        elif provider_name == "siliconflow":
            creds["api_key"] = self.credentials.get("siliconflow_api_key", "")
            creds["region"] = self.credentials.get("siliconflow_region", "global")
            creds["default_model"] = self.credentials.get(
                "siliconflow_default_model", ""
            )
        elif provider_name == "hyperstack":
            creds["api_key"] = self.credentials.get("hyperstack_api_key", "")
            creds["environment"] = self.credentials.get(
                "hyperstack_environment", "default-CANADA-1"
            )
            creds["ssh_key_name"] = self.credentials.get("hyperstack_ssh_key_name", "")
        elif provider_name == "digitalocean":
            creds["api_key"] = self.credentials.get("digitalocean_api_token", "")
            creds["region"] = self.credentials.get("digitalocean_region", "tor1")
        elif provider_name == "inferx":
            creds["api_key"] = self.credentials.get("inferx_api_key", "")
            creds["api_endpoint"] = self.credentials.get(
                "inferx_api_endpoint", "https://api.inferx.net"
            )
            creds["region"] = self.credentials.get("inferx_region", "us-west-2")
        elif provider_name == "e2enetworks":
            creds["api_key"] = self.credentials.get("e2enetworks_api_key", "")
            creds["project_id"] = self.credentials.get("e2enetworks_project_id", "")
        elif provider_name == "latitude":
            creds["api_key"] = self.credentials.get("latitude_api_key", "")
        elif provider_name == "yottalabs":
            creds["api_key"] = self.credentials.get("yottalabs_api_key", "")
        # ML Services
        elif provider_name == "kserve":
            creds["namespace"] = self.credentials.get("kserve_namespace", "default")
            creds["kubeconfig_path"] = self.credentials.get(
                "kserve_kubeconfig_path", ""
            )
            creds["auth_token"] = self.credentials.get("kserve_auth_token", "")
            creds["cluster_endpoint"] = self.credentials.get(
                "kserve_cluster_endpoint", ""
            )
        elif provider_name == "dvc":
            creds["repo_path"] = self.credentials.get("dvc_repo_path", ".")
            creds["remote_storage"] = self.credentials.get("dvc_remote_storage", "")
            creds["remote_type"] = self.credentials.get("dvc_remote_type", "")
            creds["aws_access_key_id"] = self.credentials.get("aws_access_key_id", "")
            creds["aws_secret_access_key"] = self.credentials.get(
                "aws_secret_access_key", ""
            )
            creds["gcp_credentials_path"] = self.credentials.get(
                "gcp_credentials_path", ""
            )
            creds["azure_connection_string"] = self.credentials.get(
                "azure_connection_string", ""
            )
        elif provider_name == "mlflow":
            creds["tracking_uri"] = self.credentials.get("mlflow_tracking_uri", "")
            creds["username"] = self.credentials.get("mlflow_username", "")
            creds["password"] = self.credentials.get("mlflow_password", "")
            creds["experiment_name"] = self.credentials.get(
                "mlflow_experiment_name", ""
            )
            creds["registry_uri"] = self.credentials.get("mlflow_registry_uri", "")
        elif provider_name == "ray":
            creds["dashboard_uri"] = self.credentials.get("ray_dashboard_uri", "")
            creds["cluster_name"] = self.credentials.get("ray_cluster_name", "")
            creds["auth_token"] = self.credentials.get("ray_auth_token", "")
            creds["head_node_ip"] = self.credentials.get("ray_head_node_ip", "")
            creds["head_node_port"] = str(
                self.credentials.get("ray_head_node_port", 6379)
            )
            creds["namespace"] = self.credentials.get("ray_namespace", "default")
        elif provider_name == "kubernetes":
            creds["kubeconfig_path"] = self.credentials.get(
                "kubernetes_kubeconfig_path", ""
            )
            creds["cluster_name"] = self.credentials.get("kubernetes_cluster_name", "")
            creds["namespace"] = self.credentials.get("kubernetes_namespace", "default")
            creds["karpenter_enabled"] = self.credentials.get(
                "kubernetes_karpenter_enabled", "false"
            )
            creds["karpenter_version"] = self.credentials.get(
                "kubernetes_karpenter_version", "v1.10.0"
            )
            creds["aws_region"] = self.credentials.get("aws_region", "us-east-1")
            creds["aws_account_id"] = self.credentials.get("aws_account_id", "")
        elif provider_name == "wandb":
            creds["api_key"] = self.credentials.get("wandb_api_key", "")
            creds["entity"] = self.credentials.get("wandb_entity", "")
            creds["project"] = self.credentials.get("wandb_project", "")
            creds["base_url"] = self.credentials.get("wandb_base_url", "")
            creds["team"] = self.credentials.get("wandb_team", "")
            creds["dashboard_enabled"] = self.credentials.get(
                "wandb_dashboard_enabled", "false"
            )
            creds["reports_enabled"] = self.credentials.get(
                "wandb_reports_enabled", "false"
            )
            creds["alerts_enabled"] = self.credentials.get(
                "wandb_alerts_enabled", "false"
            )
            creds["integration_enabled"] = self.credentials.get(
                "wandb_integration_enabled", "false"
            )

        return creds

    async def _get_provider_quotes(
        self, provider_name: str, gpu_type: str
    ) -> List[Dict[str, Any]]:
        """Get quotes from a real provider via the BYOAPI provider layer"""
        try:
            from terradev_cli.providers.provider_factory import ProviderFactory

            factory = ProviderFactory()
            creds = self._provider_creds(provider_name)

            # Skip auth-required providers when no credentials are present.
            # No-auth providers (static/public fallbacks) still run.
            if factory.requires_auth(provider_name) and not validate_credentials(
                provider_name, creds
            ):
                logger.debug(f"Skipping {provider_name} quotes: no credentials")
                return []

            provider = factory.create_provider(provider_name, creds)
            try:
                raw_quotes = await provider.get_instance_quotes(gpu_type)
            finally:
                # Close aiohttp session to avoid ResourceWarning
                if provider.session:
                    await provider.session.close()
            # Normalise to CLI display format
            quotes = []
            for q in raw_quotes:
                # Skip quotes that the provider already marked as unavailable
                # (e.g. TPU requested in an unsupported region).
                if not q.get("available", True):
                    continue
                quote = {
                    "provider": provider_name.replace("_", " ").title(),
                    "price": q.get("price_per_hour", 0),
                    "gpu_type": q.get("gpu_type", gpu_type),
                    "region": q.get("region", "unknown"),
                    "availability": "spot" if q.get("spot") else "on-demand",
                    "gpu_count": q.get("gpu_count", 1),
                    "instance_type": q.get("instance_type", "N/A"),
                    "memory_gb": q.get("memory_gb", 0),
                }
                # TPU fields: keep them so the CLI can build TPU-aware workloads
                if q.get("tpu_chips"):
                    quote["tpu_chips"] = q["tpu_chips"]
                    quote["tpu_type"] = q.get("tpu_type")
                    quote["tpu_image"] = q.get("tpu_image")
                    quote["tpu_non_cuda_warning"] = q.get("tpu_non_cuda_warning")
                quotes.append(quote)
            return quotes
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            return []

    async def get_runpod_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("runpod", gpu_type)

    async def get_vastai_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("vastai", gpu_type)

    async def get_aws_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("aws", gpu_type)

    async def get_gcp_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("gcp", gpu_type)

    async def get_azure_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("azure", gpu_type)

    async def get_tensordock_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("tensordock", gpu_type)

    async def get_lambda_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("lambda_labs", gpu_type)

    async def get_coreweave_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("coreweave", gpu_type)

    async def get_oracle_quotes(self, gpu_type: str):
        """Oracle Cloud  requires API credentials (BYOAPI requirement)"""
        # CRITICAL FIX: Don't return quotes without API credentials
        creds = self._provider_creds("oracle")
        if not creds or not creds.get("api_key"):
            return []

        oracle_prices = {
            "A100": 3.50,
            "V100": 2.50,
            "H100": 5.00,
            "T4": 0.80,
            "RTX4090": 1.50,
        }
        price = oracle_prices.get(gpu_type, 3.25)
        return [
            {
                "provider": "Oracle",
                "price": price,
                "gpu_type": gpu_type,
                "region": "us-ashburn-1",
                "availability": "on-demand",
            }
        ]

    async def get_crusoe_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("crusoe", gpu_type)

    async def get_alibaba_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("alibaba", gpu_type)

    async def get_baseten_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("baseten", gpu_type)

    async def get_digitalocean_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("digitalocean", gpu_type)

    async def get_e2enetworks_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("e2enetworks", gpu_type)

    async def get_fluidstack_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("fluidstack", gpu_type)

    async def get_hetzner_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("hetzner", gpu_type)

    async def get_huggingface_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("huggingface", gpu_type)

    async def get_hyperstack_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("hyperstack", gpu_type)

    async def get_inferx_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("inferx", gpu_type)

    async def get_latitude_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("latitude", gpu_type)

    async def get_ovhcloud_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("ovhcloud", gpu_type)

    async def get_siliconflow_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("siliconflow", gpu_type)

    async def get_yottalabs_quotes(self, gpu_type: str):
        return await self._get_provider_quotes("yottalabs", gpu_type)


def run_interactive_onboarding(api: TerradevAPI):
    """Interactive onboarding flow for first-time users."""
    import asyncio
    from terradev_cli.providers.provider_factory import ProviderFactory

    async def _test_provider(provider_key: str) -> tuple:
        """Test a provider's credentials with a lightweight quote call."""
        try:
            creds = api._provider_creds(provider_key)
            factory = ProviderFactory()
            provider = factory.create_provider(provider_key, creds)
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not initialise provider: {exc}"

        try:
            quotes = await provider.get_instance_quotes("A100")
            if quotes:
                return True, "API connection OK"
            return False, "No quotes returned. The key may be invalid, the provider has no available A100 capacity, or the API is unreachable."
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        finally:
            try:
                if provider.session and not provider.session.closed:
                    await provider.session.close()
            except Exception:  # noqa: BLE001
                pass

    def _clear_provider_creds(provider_key: str) -> None:
        """Remove in-memory credentials for a provider if the user discards them."""
        if provider_key == "aws":
            api.credentials.pop("aws_access_key_id", None)
            api.credentials.pop("aws_secret_access_key", None)
        elif provider_key == "gcp":
            api.credentials.pop("gcp_project_id", None)
            api.credentials.pop("gcp_credentials_file", None)
        else:
            api.credentials.pop(f"{provider_key}_api_key", None)

    def _collect_provider(provider_key: str, config: dict) -> bool:
        """Collect credentials for a provider. Returns True if something was entered."""
        if provider_key == "gcp":
            print()
            print("   Enter path to your service account JSON file:")
            print(f"   Example: {config['example']}")
            file_path = click.prompt(
                f"   {config['key_name']}", default="", show_default=False
            )
            if file_path and file_path.strip():
                if os.path.exists(file_path):
                    api.credentials["gcp_project_id"] = click.prompt(
                        "   GCP Project ID", default=""
                    )
                    api.credentials["gcp_credentials_file"] = file_path
                    return True
                print(f"   File not found: {file_path}")
            return False

        if provider_key == "aws":
            print()
            print("   AWS requires both Access Key ID and Secret Access Key")
            access_key = click.prompt(
                "   Access Key ID",
                default="",
                hide_input=True,
                show_default=False,
            )
            if access_key and access_key.strip():
                secret_key = click.prompt(
                    "   Secret Access Key",
                    default="",
                    hide_input=True,
                    show_default=False,
                )
                if secret_key and secret_key.strip():
                    api.credentials["aws_access_key_id"] = access_key
                    api.credentials["aws_secret_access_key"] = secret_key
                    return True
                print("   Skipped AWS (missing secret)")
            else:
                print("   Skipped AWS")
            return False

        key_value = click.prompt(
            f"   {config['key_name']}",
            default="",
            hide_input=True,
            show_default=False,
        )
        if key_value and key_value.strip():
            if provider_key == "aws":
                api.credentials["aws_access_key_id"] = key_value
            elif provider_key == "gcp":
                api.credentials["gcp_project_id"] = key_value
            else:
                api.credentials[f"{provider_key}_api_key"] = key_value
            return True
        return False

    print()
    print("=" * 70)
    print("WELCOME TO TERRADEV CLI".center(70))
    print("=" * 70)
    print()
    print("Your Cross-Cloud GPU Optimization Platform")
    print("Save 30-60% on GPU compute costs across multiple cloud providers")
    print("Real-time pricing + automated provisioning")
    print()
    print("=" * 70)

    provider_configs = {
        "runpod": {
            "name": "RunPod",
            "key_name": "API Key",
            "help": "Get from: https://runpod.io/console/settings/api-keys",
            "example": "rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "RUNPOD_API_KEY",
            "why": "Cheapest spot GPUs, perfect for training",
        },
        "vastai": {
            "name": "Vast.ai",
            "key_name": "API Key",
            "help": "Get from: https://console.vast.ai/api-keys",
            "example": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "VASTAI_API_KEY",
            "why": "Competitive spot market with great availability",
        },
        "aws": {
            "name": "AWS",
            "key_name": "Access Key ID",
            "help": "Get from: AWS IAM console → Users → Security credentials",
            "example": "AKIAEXAMPLEKEY123456",
            "env_var": "AWS_ACCESS_KEY_ID",
            "why": "Enterprise cloud, reliable on-demand instances",
        },
        "gcp": {
            "name": "Google Cloud",
            "key_name": "Service Account JSON",
            "help": "Get from: GCP Console → IAM & Admin → Service Accounts",
            "example": "path/to/service-account.json",
            "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
            "why": "ML-optimized A100/H100 instances",
        },
        "azure": {
            "name": "Azure",
            "key_name": "Client ID",
            "help": "Get from: Azure Portal → App registrations",
            "example": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "env_var": "AZURE_CLIENT_ID",
            "why": "Enterprise integration, ND-series GPUs",
        },
        "lambda_labs": {
            "name": "Lambda Labs",
            "key_name": "API Key",
            "help": "Get from: Lambda Labs dashboard → API Keys",
            "example": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "LAMBDA_API_KEY",
            "why": "Fast provisioning, good for inference",
        },
        "tensordock": {
            "name": "TensorDock",
            "key_name": "API Key",
            "help": "Get from: TensorDock dashboard → API",
            "example": "td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "TENSORDOCK_API_KEY",
            "why": "Budget-friendly, good for experiments",
        },
        "oracle": {
            "name": "Oracle Cloud",
            "key_name": "API Key",
            "help": "Get from: Oracle Cloud Console → Identity → Users → API Keys",
            "example": "ocid1.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "OCI_API_KEY",
            "why": "Reliable infrastructure, competitive pricing",
        },
        "crusoe": {
            "name": "Crusoe Cloud",
            "key_name": "Access Key",
            "help": "Get from: Crusoe dashboard → API Keys",
            "example": "crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "env_var": "CRUSOE_ACCESS_KEY",
            "why": "Sustainable computing, unique GPU options",
        },
    }

    configured_providers = []

    print()
    print("Tip: You only need to set up ONE provider to get started.")
    print("Add more later with: terradev configure --provider <name>")
    print("All keys are stored locally in ~/.terradev/credentials.json")

    while True:
        print()
        print("-" * 70)
        print("Available providers:")
        for i, (key, config) in enumerate(provider_configs.items(), 1):
            marker = " (configured)" if key in configured_providers else ""
            print(f"   {i:2d}. {config['name']:<15} - {config['why']}{marker}")
        print("    0.  Done / skip")

        choice = click.prompt(
            "Which provider would you like to set up?",
            default="runpod",
        ).strip().lower()

        if not choice or choice in ("0", "done", "skip", "no", "n"):
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(provider_configs):
                choice = list(provider_configs.keys())[idx]
            else:
                print("Invalid choice. Please pick a number from the list.")
                continue

        if choice not in provider_configs:
            print(f"Unknown provider '{choice}'. Please choose from the list.")
            continue

        provider_key = choice
        config = provider_configs[provider_key]

        print()
        print("=" * 60)
        print(f"Setting up {config['name']}")
        print(f"   {config['why']}")
        print(f"   Help: {config['help']}")
        print(f"   Environment variable: {config['env_var']}")

        existing = (
            api.credentials.get(f"{provider_key}_api_key")
            or api.credentials.get("aws_access_key_id")
            or api.credentials.get("gcp_project_id")
        )
        if (
            existing
            and isinstance(existing, str)
            and not any(pattern in existing.lower() for pattern in ["your_", "example_", "test_", "placeholder_", "xxx"])
        ):
            if not click.confirm(f"   {config['name']} is already configured. Reconfigure?", default=False):
                print(f"   Kept existing {config['name']} configuration.")
                if provider_key not in configured_providers:
                    configured_providers.append(provider_key)
                continue

        if not click.confirm(f"   Configure {config['name']}?", default=True):
            print(f"   Skipped {config['name']}")
            continue

        got_input = _collect_provider(provider_key, config)
        if not got_input:
            print(f"   Skipped {config['name']}")
            continue

        print()
        print(f"   Testing {config['name']} API connection...")
        try:
            success, msg = asyncio.run(_test_provider(provider_key))
        except Exception as exc:  # noqa: BLE001
            success, msg = False, str(exc)

        if not success:
            print(f"   Warning: API check failed: {msg}")
            if not click.confirm("   Save credentials anyway?", default=False):
                _clear_provider_creds(provider_key)
                print(f"   Discarded {config['name']} credentials.")
                continue
        else:
            print(f"   {config['name']} API connection verified")

        configured_providers.append(provider_key)

        if not click.confirm("   Set up another provider?", default=False):
            break

    if configured_providers:
        api.save_credentials()
        print()
        print("=" * 70)
        print(f"SUCCESS! Configured {len(configured_providers)} provider(s):")
        for provider in configured_providers:
            print(f"   {provider_configs[provider]['name']}")
    else:
        print()
        print("=" * 70)
        print("No providers configured. You can add them anytime with:")
        print("   terradev configure --provider runpod")

    print()
    print("NEXT STEPS:")
    if configured_providers:
        print("   1. Try it out: terradev quote -g A100")
        print("   2. Provision GPU: terradev provision -g A100 --duration 4")
        print("   3. Check status: terradev status")
    else:
        print("   1. Configure at least one provider:")
        print("      terradev configure --provider runpod")
        print("   2. Then try: terradev quote -g A100")

    print()
    print("NEED HELP?")
    print("   Documentation: https://github.com/theoddden/terradev")
    print("   Support: team@terradev.com")
    print("   Quick start guide: https://github.com/theoddden/terradev#quick-start")

    print()
    print("WELCOME TO TERRADEV! Happy GPU hunting!")
    print("=" * 70)
    print()
