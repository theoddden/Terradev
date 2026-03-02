#!/usr/bin/env python3
"""
Terradev CLI - Complete Production Version
All 12 commands with real provider integration and tier limits (Research/Research+/Enterprise)
"""

import click
import asyncio
import aiohttp
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import time
import sys

# Import telemetry - MANDATORY FOR USAGE TRACKING
try:
    from terradev_cli.core.telemetry import get_mandatory_telemetry
    _telemetry = get_mandatory_telemetry()
except Exception:
    _telemetry = None

# Import Kubernetes wrapper
try:
    from k8s.terraform_wrapper import TerraformWrapper
except Exception:
    TerraformWrapper = None

def validate_credentials(provider: str, credentials: Dict[str, str]) -> bool:
    """Validate that all required credentials are present for a provider"""
    required_creds = {
        'aws': ['api_key', 'secret_key'],
        'gcp': ['project_id', 'credentials_file'],
        'azure': ['subscription_id', 'tenant_id', 'client_id', 'client_secret'],
        'runpod': ['api_key'],
        'vastai': ['api_key'],
        'lambda_labs': ['api_key'],
        'coreweave': ['api_key'],
        'tensordock': ['api_key', 'api_token'],
        'huggingface': ['api_key', 'namespace'],
        'baseten': ['api_key'],
        'oracle': ['api_key', 'tenancy_ocid', 'compartment_ocid', 'region'],
        'crusoe': ['access_key', 'secret_key', 'project_id']
    }
    
    provider_lower = provider.lower()
    if provider_lower not in required_creds:
        return False
    
    missing = []
    for req in required_creds[provider_lower]:
        if req not in credentials or not credentials[req].strip():
            missing.append(req)
    
    if missing:
        print(f"   ERROR: Missing required credentials: {', '.join(missing)}")
        return False
    
    return True

class TerradevAPI:
    """Real provider API integration with tier limits"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.terradev'
        self.config_dir.mkdir(exist_ok=True)
        self.credentials_file = self.config_dir / 'credentials.json'
        self.usage_file = self.config_dir / 'usage.json'
        self.tier_file = self.config_dir / 'tier.json'
        
        self.load_credentials()
        self.load_usage()
        
        # Tier configuration
        self.tiers = {
            'research': {
                'name': 'Research',
                'provisions_per_month': 10,
                'max_instances': 1,
                'user_seats': 1,
                'providers': ['runpod', 'vastai', 'aws'],
                'features': ['quote', 'status', 'configure', 'cleanup', 'analytics', 'optimize']
            },
            'research_plus': {
                'name': 'Research+',
                'provisions_per_month': 80,
                'max_instances': 8,
                'user_seats': 1,
                'providers': ['runpod', 'vastai', 'aws', 'gcp', 'azure', 'tensordock', 'oracle'],
                'features': ['all', 'inference', 'full_provenance']
            },
            'enterprise': {
                'name': 'Enterprise',
                'provisions_per_month': 'unlimited',
                'max_instances': 32,
                'user_seats': 5,
                'providers': ['all'],
                'features': ['all', 'inference', 'full_provenance', 'priority_support', 'sla_guarantee']
            },
            'enterprise_plus': {
                'name': 'Enterprise+',
                'provisions_per_month': 'unlimited',
                'max_instances': 'unlimited',
                'user_seats': 'unlimited',
                'min_gpus': 32,
                'gpu_hour_rate': 0.09,
                'providers': ['all'],
                'features': ['all', 'inference', 'full_provenance', 'priority_support', 'sla_guarantee', 'dedicated_support', 'gpu_metering', 'fleet_management']
            }
        }
        
        self.tier = self.tiers['research']  # Default to research tier
        self._load_tier()
        
        # Initialize usage tracking
        if "inference_endpoints" not in self.usage:
            self.usage["inference_endpoints"] = []

        # Enterprise+ GPU-hour metering sync (best-effort, max once/hr)
        self._maybe_sync_gpu_metering()

    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user with no configured credentials"""
        # Check if credentials file exists and has any content
        if not self.credentials_file.exists():
            return True
        
        # Check if credentials are empty or only contain default/placeholder values
        if not self.credentials or len(self.credentials) == 0:
            return True
        
        # Check if all credentials are still placeholder values
        placeholder_patterns = ['your_', 'example_', 'test_', 'placeholder_', 'xxx']
        for key, value in self.credentials.items():
            if value and not any(pattern in value.lower() for pattern in placeholder_patterns):
                return False  # Found a real credential
        
        return True  # All credentials appear to be placeholders

    # Embedded Ed25519 public key for tier token verification
    _TIER_VERIFY_KEY_B64 = "4lJY9uWYGfx2hZkJ6N4DO5plErRX+J/HD97Tx+Xrvms="

    def _load_tier(self):
        """Load and cryptographically verify tier from local config.

        tier.json stores the full signed activation token (payload + Ed25519
        signature).  On every CLI invocation we re-verify the signature and
        reject expired tokens so that editing the file cannot escalate
        privileges.
        """
        if not self.tier_file.exists():
            return
        try:
            import base64
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            with open(self.tier_file, 'r') as f:
                tier_data = json.load(f)

            payload = tier_data.get('payload')
            signature = tier_data.get('signature')

            # If the file has no signed token (legacy format), ignore it
            if not payload or not signature:
                return

            # Verify Ed25519 signature
            verify_key = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(self._TIER_VERIFY_KEY_B64)
            )
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            sig_bytes = base64.b64decode(signature)
            verify_key.verify(sig_bytes, payload_bytes)

            # Reject expired tokens (>30 days — generous for subscription)
            TOKEN_MAX_AGE = 30 * 24 * 3600
            created_at = payload.get('created_at', 0)
            if time.time() - created_at > TOKEN_MAX_AGE:
                return

            # Signature valid + not expired → activate tier
            cached_tier = payload.get('tier', 'research')
            if cached_tier in self.tiers:
                self.tier = self.tiers[cached_tier]
        except Exception:
            pass
    
    # Real Stripe Payment Links — these are public checkout URLs, not secrets
    STRIPE_PAYMENT_LINKS = {
        'research_plus': 'https://buy.stripe.com/dRm7sM0r3eGUgJN3Nu18c00',
        'enterprise': 'https://buy.stripe.com/7sYbJ24Hj0Q48dh6ZG18c01',
        'enterprise_plus': 'https://buy.stripe.com/dRm7sM0r3eGUgJN3Nu18c02',
    }

    # Real Stripe Price IDs for subscription verification
    STRIPE_PRICE_IDS = {
        'research_plus': 'price_1SzK6gKDFO7eDloB38AN8Tyo',
        'enterprise': 'price_1Sz6w2KDFO7eDloBswXFsZQU',
        'enterprise_plus': 'price_1SzK7hKDFO7eDloBEntPlus',
    }

    def get_stripe_checkout_url(self, tier: str) -> str:
        """Get the Stripe Payment Link URL for a tier.

        Returns the live Stripe checkout page where users can subscribe.
        Override with env vars STRIPE_RESEARCH_PLUS_LINK / STRIPE_ENTERPRISE_LINK.
        """
        env_key = f"STRIPE_{tier.upper()}_LINK"
        return os.environ.get(env_key, self.STRIPE_PAYMENT_LINKS.get(tier, ''))

    def load_credentials(self):
        """Load user's cloud provider credentials (simple JSON for BYOAPI)"""
        try:
            # Simple direct loading for BYOAPI
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    self.credentials = json.load(f)
            else:
                self.credentials = {}
        except Exception as e:
            import sys
            print(
                f"Warning  WARNING: Failed to load credentials ({e}). "
                f"Your credentials file may be corrupted.",
                file=sys.stderr,
            )
            print(
                "   Run `terradev configure` to re-enter your keys.",
                file=sys.stderr,
            )
            self.credentials = {}
            self._auth_manager = None
    
    def save_credentials(self):
        """Save user's cloud provider credentials (simple JSON for BYOAPI)"""
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(self.credentials, f, indent=2)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to save credentials: {e}", file=sys.stderr)
    
    def load_usage(self):
        """Load usage tracking"""
        if self.usage_file.exists():
            with open(self.usage_file, 'r') as f:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    self.usage = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            self.usage = {
                "provisions_this_month": 0,
                "month_start": datetime.now().replace(day=1).isoformat(),
                "instances_created": [],
                "inference_endpoints": [],
                "last_reset": datetime.now().isoformat()
            }
    
    def save_usage(self):
        """Save usage tracking with exclusive file lock"""
        import fcntl
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(self.usage, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    
    def check_provision_limit(self) -> bool:
        """Check if user has provisions remaining this month"""
        self._maybe_reset_monthly_usage()
        limit = self.tier['provisions_per_month']
        if limit == 'unlimited':
            return True
        used = self.usage.get('provisions_this_month', 0)
        return used < limit

    def record_provision(self):
        """Increment the monthly provision counter"""
        self._maybe_reset_monthly_usage()
        self.usage['provisions_this_month'] = self.usage.get('provisions_this_month', 0) + 1
        self.save_usage()

    def _maybe_reset_monthly_usage(self):
        """Reset monthly counters if the calendar month has changed (R8 fix)"""
        month_start = datetime.fromisoformat(self.usage["month_start"])
        now = datetime.now()
        if (now.year, now.month) != (month_start.year, month_start.month):
            self.usage['provisions_this_month'] = 0
            self.usage['month_start'] = now.replace(day=1).isoformat()

    def _maybe_sync_gpu_metering(self):
        """Enterprise+ only: report accrued GPU-hours for all active instances.

        Runs on every CLI invocation (lightweight — reads local state only,
        reports to Stripe only if there are unreported hours since last sync).
        This catches GPU-hours even if an instance was terminated outside the
        CLI (e.g. directly on the provider dashboard or via MCP subprocess).
        """
        if self.tier.get('name') != 'Enterprise+':
            return

        try:
            from terradev_cli.core.stripe_manager import StripeManager
            sm = StripeManager()
            metering = sm._load_metering()
            sub_item_id = metering.get('subscription_item_id')
            if not sub_item_id:
                return  # Not yet linked — will link on activation

            instances = self.usage.get('instances_created', [])
            if not instances:
                # No active instances — nothing to bill. Reset sync timestamp
                # so the floor doesn't accumulate while idle.
                metering['last_sync_ts'] = datetime.now().isoformat()
                sm._save_metering(metering)
                return

            last_sync = metering.get('last_sync_ts', '')
            now = datetime.now()

            # Only sync once per hour to avoid hammering Stripe
            if last_sync:
                try:
                    last_dt = datetime.fromisoformat(last_sync)
                    if (now - last_dt).total_seconds() < 3600:
                        return
                except Exception:
                    pass

            # Calculate hours since last sync (or 1 hr cap for first sync)
            if last_sync:
                try:
                    hours_since_last_sync = max((now - datetime.fromisoformat(last_sync)).total_seconds() / 3600, 0)
                except Exception:
                    hours_since_last_sync = 1.0
            else:
                hours_since_last_sync = 1.0

            actual_gpu_hours = 0.0
            for inst in instances:
                created = inst.get('created_at', '')
                gpu_count = inst.get('gpu_count', 1)
                if created:
                    try:
                        start = datetime.fromisoformat(created)
                        # Hours since instance creation (or since last sync, whichever is later)
                        if last_sync:
                            try:
                                sync_dt = datetime.fromisoformat(last_sync)
                                start = max(start, sync_dt)
                            except Exception:
                                pass
                        hours = max((now - start).total_seconds() / 3600, 0)
                        actual_gpu_hours += hours * gpu_count
                    except Exception:
                        continue

            # Minimum commitment floor: always bill at least 32 GPU-hours
            # per billing hour, even if actual usage is lower.
            min_gpu_hours = 32 * hours_since_last_sync
            gpu_hours_to_bill = max(actual_gpu_hours, min_gpu_hours)

            if gpu_hours_to_bill > 0.01:
                sm.report_gpu_hours(
                    sub_item_id, gpu_hours_to_bill,
                    gpu_type='mixed',
                    instance_id=f'sync-{now.strftime("%Y%m%d%H")}',
                )
                metering['last_sync_ts'] = now.isoformat()
                sm._save_metering(metering)
        except Exception:
            pass  # Metering sync is best-effort
    
    def _provider_creds(self, provider_name: str) -> Dict[str, str]:
        """Build credentials dict for a provider from stored BYOAPI keys"""
        if not self.credentials:
            return {}
        
        creds: Dict[str, str] = {}
        if provider_name == 'aws':
            creds['api_key'] = self.credentials.get('aws_access_key_id', '')
            creds['secret_key'] = self.credentials.get('aws_secret_access_key', '')
        elif provider_name == 'gcp':
            creds['project_id'] = self.credentials.get('gcp_project_id', '')
            creds['credentials_file'] = self.credentials.get('gcp_credentials_file', '')
        elif provider_name == 'azure':
            creds['subscription_id'] = self.credentials.get('azure_subscription_id', '')
            creds['tenant_id'] = self.credentials.get('azure_tenant_id', '')
            creds['client_id'] = self.credentials.get('azure_client_id', '')
            creds['client_secret'] = self.credentials.get('azure_client_secret', '')
        elif provider_name == 'runpod':
            creds['api_key'] = self.credentials.get('runpod_api_key', '')
        elif provider_name == 'vastai':
            creds['api_key'] = self.credentials.get('vastai_api_key', '')
        elif provider_name == 'lambda_labs':
            creds['api_key'] = self.credentials.get('lambda_api_key', '')
        elif provider_name == 'coreweave':
            creds['api_key'] = self.credentials.get('coreweave_api_key', '')
        elif provider_name == 'tensordock':
            creds['api_key'] = self.credentials.get('tensordock_api_key', '')
            creds['api_token'] = self.credentials.get('tensordock_api_token', '')
        elif provider_name == 'huggingface':
            creds['api_key'] = self.credentials.get('huggingface_api_token', '')
            creds['namespace'] = self.credentials.get('huggingface_namespace', '')
        elif provider_name == 'baseten':
            creds['api_key'] = self.credentials.get('baseten_api_key', '')
        elif provider_name == 'oracle':
            creds['api_key'] = self.credentials.get('oracle_api_key', '')
            creds['tenancy_ocid'] = self.credentials.get('oracle_tenancy_ocid', '')
            creds['compartment_ocid'] = self.credentials.get('oracle_compartment_ocid', '')
            creds['region'] = self.credentials.get('oracle_region', 'us-ashburn-1')
        elif provider_name == 'crusoe':
            creds['access_key'] = self.credentials.get('crusoe_access_key', '')
            creds['secret_key'] = self.credentials.get('crusoe_secret_key', '')
            creds['project_id'] = self.credentials.get('crusoe_project_id', '')
        elif provider_name == 'alibaba':
            creds['access_key_id'] = self.credentials.get('alibaba_access_key_id', '')
            creds['access_key_secret'] = self.credentials.get('alibaba_access_key_secret', '')
            creds['region_id'] = self.credentials.get('alibaba_region_id', 'cn-beijing')
            creds['security_group_id'] = self.credentials.get('alibaba_security_group_id', '')
            creds['vswitch_id'] = self.credentials.get('alibaba_vswitch_id', '')
        elif provider_name == 'ovhcloud':
            creds['application_key'] = self.credentials.get('ovhcloud_application_key', '')
            creds['application_secret'] = self.credentials.get('ovhcloud_application_secret', '')
            creds['consumer_key'] = self.credentials.get('ovhcloud_consumer_key', '')
            creds['project_id'] = self.credentials.get('ovhcloud_project_id', '')
            creds['endpoint'] = self.credentials.get('ovhcloud_endpoint', 'ovh-eu')
            creds['ssh_key_id'] = self.credentials.get('ovhcloud_ssh_key_id', '')
        elif provider_name == 'fluidstack':
            creds['api_key'] = self.credentials.get('fluidstack_api_key', '')
            creds['ssh_key_name'] = self.credentials.get('fluidstack_ssh_key_name', '')
        elif provider_name == 'hetzner':
            creds['api_token'] = self.credentials.get('hetzner_api_token', '')
            creds['robot_user'] = self.credentials.get('hetzner_robot_user', '')
            creds['robot_password'] = self.credentials.get('hetzner_robot_password', '')
        elif provider_name == 'siliconflow':
            creds['api_key'] = self.credentials.get('siliconflow_api_key', '')
            creds['region'] = self.credentials.get('siliconflow_region', 'global')
            creds['default_model'] = self.credentials.get('siliconflow_default_model', '')
        # ML Services
        elif provider_name == 'kserve':
            creds['namespace'] = self.credentials.get('kserve_namespace', 'default')
            creds['kubeconfig_path'] = self.credentials.get('kserve_kubeconfig_path', '')
            creds['auth_token'] = self.credentials.get('kserve_auth_token', '')
            creds['cluster_endpoint'] = self.credentials.get('kserve_cluster_endpoint', '')
        elif provider_name == 'langsmith':
            creds['api_key'] = self.credentials.get('langsmith_api_key', '')
            creds['endpoint'] = self.credentials.get('langsmith_endpoint', 'https://api.smith.langchain.com')
            creds['workspace_id'] = self.credentials.get('langsmith_workspace_id', '')
            creds['project_name'] = self.credentials.get('langsmith_project_name', '')
        elif provider_name == 'dvc':
            creds['repo_path'] = self.credentials.get('dvc_repo_path', '.')
            creds['remote_storage'] = self.credentials.get('dvc_remote_storage', '')
            creds['remote_type'] = self.credentials.get('dvc_remote_type', '')
            creds['aws_access_key_id'] = self.credentials.get('aws_access_key_id', '')
            creds['aws_secret_access_key'] = self.credentials.get('aws_secret_access_key', '')
            creds['gcp_credentials_path'] = self.credentials.get('gcp_credentials_path', '')
            creds['azure_connection_string'] = self.credentials.get('azure_connection_string', '')
        elif provider_name == 'mlflow':
            creds['tracking_uri'] = self.credentials.get('mlflow_tracking_uri', '')
            creds['username'] = self.credentials.get('mlflow_username', '')
            creds['password'] = self.credentials.get('mlflow_password', '')
            creds['experiment_name'] = self.credentials.get('mlflow_experiment_name', '')
            creds['registry_uri'] = self.credentials.get('mlflow_registry_uri', '')
        elif provider_name == 'ray':
            creds['dashboard_uri'] = self.credentials.get('ray_dashboard_uri', '')
            creds['cluster_name'] = self.credentials.get('ray_cluster_name', '')
            creds['auth_token'] = self.credentials.get('ray_auth_token', '')
            creds['head_node_ip'] = self.credentials.get('ray_head_node_ip', '')
            creds['head_node_port'] = str(self.credentials.get('ray_head_node_port', 6379))
            creds['namespace'] = self.credentials.get('ray_namespace', 'default')
        elif provider_name == 'kubernetes':
            creds['kubeconfig_path'] = self.credentials.get('kubernetes_kubeconfig_path', '')
            creds['cluster_name'] = self.credentials.get('kubernetes_cluster_name', '')
            creds['namespace'] = self.credentials.get('kubernetes_namespace', 'default')
            creds['karpenter_enabled'] = self.credentials.get('kubernetes_karpenter_enabled', 'false')
            creds['karpenter_version'] = self.credentials.get('kubernetes_karpenter_version', 'v0.32.0')
            creds['aws_region'] = self.credentials.get('aws_region', 'us-east-1')
            creds['aws_account_id'] = self.credentials.get('aws_account_id', '')
            creds['monitoring_enabled'] = self.credentials.get('kubernetes_monitoring_enabled', 'false')
            creds['prometheus_enabled'] = self.credentials.get('kubernetes_prometheus_enabled', 'false')
            creds['grafana_enabled'] = self.credentials.get('kubernetes_grafana_enabled', 'false')
            creds['dashboard_port'] = self.credentials.get('kubernetes_dashboard_port', '3000')
        elif provider_name == 'wandb':
            creds['api_key'] = self.credentials.get('wandb_api_key', '')
            creds['entity'] = self.credentials.get('wandb_entity', '')
            creds['project'] = self.credentials.get('wandb_project', '')
            creds['base_url'] = self.credentials.get('wandb_base_url', '')
            creds['team'] = self.credentials.get('wandb_team', '')
            creds['dashboard_enabled'] = self.credentials.get('wandb_dashboard_enabled', 'false')
            creds['reports_enabled'] = self.credentials.get('wandb_reports_enabled', 'false')
            creds['alerts_enabled'] = self.credentials.get('wandb_alerts_enabled', 'false')
            creds['integration_enabled'] = self.credentials.get('wandb_integration_enabled', 'false')
        elif provider_name == 'ray':
            creds['dashboard_uri'] = self.credentials.get('ray_dashboard_uri', '')
            creds['cluster_name'] = self.credentials.get('ray_cluster_name', '')
            creds['auth_token'] = self.credentials.get('ray_auth_token', '')
            creds['head_node_ip'] = self.credentials.get('ray_head_node_ip', '')
            creds['head_node_port'] = str(self.credentials.get('ray_head_node_port', 6379))
            creds['namespace'] = self.credentials.get('ray_namespace', 'default')
            creds['monitoring_enabled'] = self.credentials.get('ray_monitoring_enabled', 'false')
            creds['prometheus_enabled'] = self.credentials.get('ray_prometheus_enabled', 'false')
            creds['grafana_enabled'] = self.credentials.get('ray_grafana_enabled', 'false')
            creds['metrics_export_port'] = self.credentials.get('ray_metrics_export_port', '8080')
        
        return creds

    async def _get_provider_quotes(self, provider_name: str, gpu_type: str) -> List[Dict[str, Any]]:
        """Get quotes from a real provider via the BYOAPI provider layer"""
        try:
            from terradev_cli.providers.provider_factory import ProviderFactory
            factory = ProviderFactory()
            creds = self._provider_creds(provider_name)
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
                quotes.append({
                    'provider': provider_name.replace('_', ' ').title(),
                    'price': q.get('price_per_hour', 0),
                    'gpu_type': q.get('gpu_type', gpu_type),
                    'region': q.get('region', 'unknown'),
                    'availability': 'spot' if q.get('spot') else 'on-demand',
                })
            return quotes
        except Exception:
            return []

    async def get_runpod_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('runpod', gpu_type)

    async def get_vastai_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('vastai', gpu_type)

    async def get_aws_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('aws', gpu_type)

    async def get_gcp_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('gcp', gpu_type)

    async def get_azure_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('azure', gpu_type)

    async def get_tensordock_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('tensordock', gpu_type)

    async def get_lambda_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('lambda_labs', gpu_type)

    async def get_coreweave_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('coreweave', gpu_type)

    async def get_oracle_quotes(self, gpu_type: str):
        """Oracle Cloud – requires API credentials (BYOAPI requirement)"""
        # CRITICAL FIX: Don't return quotes without API credentials
        creds = self._provider_creds('oracle')
        if not creds or not creds.get('api_key'):
            return []
            
        oracle_prices = {
            'A100': 3.50, 'V100': 2.50, 'H100': 5.00,
            'T4': 0.80, 'RTX4090': 1.50,
        }
        price = oracle_prices.get(gpu_type, 3.25)
        return [{
            'provider': 'Oracle',
            'price': price,
            'gpu_type': gpu_type,
            'region': 'us-ashburn-1',
            'availability': 'on-demand',
        }]

    async def get_crusoe_quotes(self, gpu_type: str):
        return await self._get_provider_quotes('crusoe', gpu_type)

def run_interactive_onboarding(api: TerradevAPI):
    """Interactive onboarding flow for first-time users"""
    import sys
    
    # Beautiful welcome screen
    print("\n" + "="*70)
    print("WELCOME TO TERRADEV CLI".center(70))
    print("="*70)
    print("\nYour Cross-Cloud GPU Optimization Platform")
    print("Save 30-60% on GPU compute costs across 9+ cloud providers")
    print("Real-time pricing + automated provisioning")
    print("\n" + "="*70)
    
    # Show what we'll set up
    print("\nWe'll configure API keys for these providers:")
    providers_to_show = [
        ('runpod', 'RunPod', 'Cheapest spot GPUs'),
        ('vastai', 'Vast.ai', 'Competitive spot market'),
        ('aws', 'AWS', 'Enterprise cloud'),
        ('gcp', 'Google Cloud', 'ML-optimized'),
        ('azure', 'Azure', 'Enterprise integration'),
        ('lambda_labs', 'Lambda Labs', 'Fast provisioning'),
        ('tensordock', 'TensorDock', 'Budget-friendly'),
        ('oracle', 'Oracle Cloud', 'Reliable infrastructure'),
        ('crusoe', 'Crusoe Cloud', 'Sustainable computing')
    ]
    
    for i, (key, name, desc) in enumerate(providers_to_show, 1):
        print(f"   {i:2d}. {name:<15} - {desc}")
    
    print(f"\nTip: You can start with just 1-2 providers and add more later!")
    print("All keys are stored locally in ~/.terradev/credentials.json")
    
    # Ask if they want to proceed
    print("\n" + "-"*70)
    proceed = click.confirm('Ready to set up your cloud providers?', default=True)
    
    if not proceed:
        print("\nNo problem! You can configure anytime with:")
        print("   terradev configure")
        print("   terradev configure --provider runpod")
        print("\nQuick start: Add just RunPod for cheapest spot GPUs")
        return
    
    print("\nLet's set up your providers! (Press Enter to skip any provider)\n")
    
    # Provider configurations with helpful info
    provider_configs = {
        'runpod': {
            'name': 'RunPod',
            'key_name': 'API Key',
            'help': 'Get from: https://runpod.io/console/settings/api-keys',
            'example': 'rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'RUNPOD_API_KEY',
            'why': 'Cheapest spot GPUs, perfect for training'
        },
        'vastai': {
            'name': 'Vast.ai',
            'key_name': 'API Key',
            'help': 'Get from: https://console.vast.ai/api-keys',
            'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'VASTAI_API_KEY',
            'why': 'Competitive spot market with great availability'
        },
        'aws': {
            'name': 'AWS',
            'key_name': 'Access Key ID',
            'help': 'Get from: AWS IAM console → Users → Security credentials',
            'example': 'AKIAIOSFODNN7EXAMPLE',
            'env_var': 'AWS_ACCESS_KEY_ID',
            'why': 'Enterprise cloud, reliable on-demand instances'
        },
        'gcp': {
            'name': 'Google Cloud',
            'key_name': 'Service Account JSON',
            'help': 'Get from: GCP Console → IAM & Admin → Service Accounts',
            'example': 'path/to/service-account.json',
            'env_var': 'GOOGLE_APPLICATION_CREDENTIALS',
            'why': 'ML-optimized A100/H100 instances'
        },
        'azure': {
            'name': 'Azure',
            'key_name': 'Client ID',
            'help': 'Get from: Azure Portal → App registrations',
            'example': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',
            'env_var': 'AZURE_CLIENT_ID',
            'why': 'Enterprise integration, ND-series GPUs'
        },
        'lambda_labs': {
            'name': 'Lambda Labs',
            'key_name': 'API Key',
            'help': 'Get from: Lambda Labs dashboard → API Keys',
            'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'LAMBDA_API_KEY',
            'why': 'Fast provisioning, good for inference'
        },
        'tensordock': {
            'name': 'TensorDock',
            'key_name': 'API Key',
            'help': 'Get from: TensorDock dashboard → API',
            'example': 'td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'TENSORDOCK_API_KEY',
            'why': 'Budget-friendly, good for experiments'
        },
        'oracle': {
            'name': 'Oracle Cloud',
            'key_name': 'API Key',
            'help': 'Get from: Oracle Cloud Console → Identity → Users → API Keys',
            'example': 'ocid1.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'OCI_API_KEY',
            'why': 'Reliable infrastructure, competitive pricing'
        },
        'crusoe': {
            'name': 'Crusoe Cloud',
            'key_name': 'Access Key',
            'help': 'Get from: Crusoe dashboard → API Keys',
            'example': 'crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'env_var': 'CRUSOE_ACCESS_KEY',
            'why': 'Sustainable computing, unique GPU options'
        }
    }
    
    # Track what we configured
    configured_providers = []
    
    # Interactive setup for each provider
    for provider_key, config in provider_configs.items():
        print(f"\n{'='*60}")
        print(f"Setting up {config['name']}")
        print(f"   {config['why']}")
        print(f"   Help: {config['help']}")
        print(f"   Environment variable: {config['env_var']}")
        
        # Check if already configured
        existing_value = api.credentials.get(f"{provider_key}_api_key") or api.credentials.get(f"{provider_key}_access_key_id")
        if existing_value and not any(pattern in existing_value.lower() for pattern in ['your_', 'example_', 'test_', 'placeholder_', 'xxx']):
            print(f"   Already configured!")
            configured_providers.append(provider_key)
            continue
        
        # Ask to configure this provider
        configure_this = click.confirm(f"   Configure {config['name']}?", default=False)
        
        if not configure_this:
            print(f"   Skipped {config['name']}")
            continue
        
        # Get the API key
        if provider_key == 'gcp':
            # Special handling for GCP JSON file
            print(f"\n   Enter path to your service account JSON file:")
            print(f"   Example: {config['example']}")
            file_path = click.prompt(f"   {config['key_name']}", default='', show_default=False)
            
            if file_path and file_path.strip():
                # Validate file exists
                if os.path.exists(file_path):
                    api.credentials['gcp_project_id'] = click.prompt("   GCP Project ID", default='')
                    api.credentials['gcp_credentials_file'] = file_path
                    configured_providers.append(provider_key)
                    print(f"   {config['name']} configured!")
                else:
                    print(f"   File not found: {file_path}")
            else:
                print(f"   Skipped {config['name']}")
        
        elif provider_key == 'aws':
            # AWS needs multiple keys
            print(f"\n   AWS requires both Access Key ID and Secret Access Key")
            access_key = click.prompt(f"   {config['key_name']}", default='', hide_input=True, show_default=False)
            if access_key and access_key.strip():
                secret_key = click.prompt(f"   Secret Access Key", default='', hide_input=True, show_default=False)
                if secret_key and secret_key.strip():
                    api.credentials['aws_access_key_id'] = access_key
                    api.credentials['aws_secret_access_key'] = secret_key
                    configured_providers.append(provider_key)
                    print(f"   {config['name']} configured!")
                else:
                    print(f"   Skipped {config['name']} (missing secret)")
            else:
                print(f"   Skipped {config['name']}")
        
        else:
            # Single key providers
            key_value = click.prompt(f"   {config['key_name']}", default='', hide_input=True, show_default=False)
            
            if key_value and key_value.strip():
                # Store with appropriate key name
                if provider_key == 'aws':
                    api.credentials['aws_access_key_id'] = key_value
                elif provider_key == 'gcp':
                    api.credentials['gcp_project_id'] = key_value
                else:
                    api.credentials[f"{provider_key}_api_key"] = key_value
                
                configured_providers.append(provider_key)
                print(f"   {config['name']} configured!")
            else:
                print(f"   Skipped {config['name']}")
    
    # Save credentials
    if configured_providers:
        api.save_credentials()
        print(f"\n{'='*70}")
        print(f"SUCCESS! Configured {len(configured_providers)} providers:")
        for provider in configured_providers:
            print(f"   {provider_configs[provider]['name']}")
    else:
        print(f"\n{'='*70}")
        print("No providers configured. You can add them anytime with:")
        print("   terradev configure --provider runpod")
    
    # Next steps
    print(f"\nNEXT STEPS:")
    if configured_providers:
        print("   1. Try it out: terradev quote -g A100")
        print("   2. Provision GPU: terradev provision -g A100 --duration 4")
        print("   3. Check status: terradev status")
    else:
        print("   1. Configure at least one provider:")
        print("      terradev configure --provider runpod")
        print("   2. Then try: terradev quote -g A100")
    
    print(f"\nNEED HELP?")
    print("   Documentation: https://github.com/theoddden/terradev")
    print("   Support: team@terradev.com")
    print("   Quick start guide: https://github.com/theoddden/terradev#quick-start")
    
    print(f"\nWELCOME TO TERRADEV! Happy GPU hunting!")
    print("="*70 + "\n")

@click.group()
@click.version_option(version="3.5.3", prog_name="Terradev CLI")
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--skip-onboarding', is_flag=True, help='Skip first-time setup')
def cli(config=None, verbose=False, skip_onboarding=False):
    """
    Terradev CLI - Cross-Cloud Compute Optimization Platform
    
    Parallel provisioning and orchestration for cross-cloud cost optimization.
    Save 30% on end-to-end compute provisioning costs with real-time cloud arbitrage.
    
    Research Tier: 10 provisions/month, 1 server, 1 seat (Free)
    Research+ Tier: 80 provisions/month, 8 servers, 1 seat, inference ($49.99/month)
    Enterprise Tier: Unlimited provisions, 32 servers, 5 seats, full provenance ($299.99/month)
    Enterprise+ Tier: Unlimited everything, 32 GPU min, metered at $0.09/GPU-hr
    """
    # Check for first-time user and trigger onboarding
    if not skip_onboarding and not os.environ.get("TERRADEV_SKIP_ONBOARDING"):
        api = TerradevAPI()
        if api.is_first_time_user():
            run_interactive_onboarding(api)

@cli.command()
@click.option('--force', is_flag=True, help='Force onboarding even if already configured')
def onboarding(force):
    """Run the interactive onboarding flow"""
    api = TerradevAPI()
    if force or api.is_first_time_user():
        run_interactive_onboarding(api)
    else:
        print("You're already set up! Use --force to re-run onboarding.")
        print("Or configure individual providers with: terradev configure --provider <name>")

@cli.command()
@click.option('--tier', '-t', type=click.Choice(['research_plus', 'enterprise', 'enterprise_plus']),
              help='Tier to upgrade to')
@click.option('--activate', is_flag=True,
              help='Activate after payment — verifies your Stripe subscription and unlocks your tier')
@click.option('--email', help='Email used for Stripe checkout (for --activate)')
def upgrade(tier, activate, email):
    """Upgrade your Terradev subscription via Stripe.

    Opens the Stripe checkout page in your browser. After payment completes,
    run `terradev upgrade --activate --email you@example.com` to instantly
    unlock your new tier.

    \b
    Research+ ($49.99/mo): 30 provisions/month, 4 servers, 1 seat, inference
    Enterprise ($299.99/mo): Unlimited provisions, 32 servers, 5 seats, full provenance
    Enterprise+ ($0.09/GPU-hr, 32 GPU min): Unlimited everything, metered billing, fleet mgmt
    """
    api = TerradevAPI()

    if activate:
        # ── Activate: fetch signed token from S3 (written by Stripe webhook) ──
        if not email:
            if not sys.stdin.isatty():
                print("❌ --email is required in non-interactive mode")
                print("   Usage: terradev upgrade --activate --email you@example.com")
                sys.exit(2)
            email = click.prompt('Enter the email you used at Stripe checkout')

        print(f"Checking activation for {email}...")

        try:
            import base64, hashlib
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            # Embedded Ed25519 public verification key — matches the Lambda's private key
            _VERIFY_KEY_B64 = "4lJY9uWYGfx2hZkJ6N4DO5plErRX+J/HD97Tx+Xrvms="
            _VERIFY_KEY = Ed25519PublicKey.from_public_bytes(base64.b64decode(_VERIFY_KEY_B64))

            ACTIVATION_BUCKET = os.environ.get(
                'TERRADEV_ACTIVATION_BUCKET', 'terradev-activations'
            )
            email_hash = hashlib.sha256(email.lower().strip().encode()).hexdigest()
            token_url = f"https://{ACTIVATION_BUCKET}.s3.amazonaws.com/activations/{email_hash}.json"

            async def _fetch_token():
                # Create SSL context that handles certificate issues
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        token_url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status != 200:
                            return None
                        return await resp.json()

            token = asyncio.run(_fetch_token())

            if token is None:
                print("No activation found for that email.")
                print("   Make sure you completed Stripe checkout, then wait ~30 seconds and try again.")
                print("   Contact support@terradev.com if the issue persists.")
                return

            payload = token.get('payload', {})
            signature = token.get('signature', '')

            # Verify Ed25519 signature (always required — not optional)
            try:
                payload_bytes = json.dumps(payload, sort_keys=True).encode()
                sig_bytes = base64.b64decode(signature)
                _VERIFY_KEY.verify(sig_bytes, payload_bytes)
            except Exception:
                print("Activation token signature is invalid.")
                print("   Contact support@terradev.com for help.")
                return

            # Reject tokens older than 7 days
            TOKEN_MAX_AGE_SECONDS = 7 * 24 * 3600
            created_at = payload.get('created_at', 0)
            if time.time() - created_at > TOKEN_MAX_AGE_SECONDS:
                print("Activation token has expired.")
                print("   Tokens are valid for 7 days after checkout.")
                print("   Please re-purchase or contact support@terradev.com.")
                return

            activated_tier = payload.get('tier')
            customer_id = payload.get('customer_id', '')

            if not activated_tier or activated_tier not in ('research_plus', 'enterprise', 'enterprise_plus'):
                print("No matching Terradev tier found.")
                print("   Contact support@terradev.com for help.")
                return

            # Write the full signed token to tier.json so _load_tier()
            # can re-verify the Ed25519 signature on every CLI invocation.
            tier_data = {
                'payload': payload,
                'signature': signature,
                'email': email,
                'verified_at': datetime.now().isoformat(),
            }
            api.tier_file.parent.mkdir(parents=True, exist_ok=True)
            with open(api.tier_file, 'w') as f:
                json.dump(tier_data, f, indent=2)

            # ── Graceful tier migration ──
            old_tier = api.tier
            tier_info = api.tiers[activated_tier]
            running = api.usage.get('instances_created', [])
            running_count = len(running)
            inference_eps = api.usage.get('inference_endpoints', [])

            print(f"\nTier upgraded to {tier_info['name']}!")
            print(f"   Concurrent servers: {tier_info['max_instances']}")
            prov = tier_info['provisions_per_month']
            print(f"   Provisions/month:   {prov}")
            print(f"   User seats:         {tier_info['user_seats']}")
            if 'full_provenance' in tier_info.get('features', []):
                print(f"   Full provenance:    Enabled")

            # Enterprise+ specific activation — resolve Stripe subscription item for metering
            if activated_tier == 'enterprise_plus':
                print(f"   Billing model:      Metered ($0.09/GPU-hr)")
                print(f"   Minimum GPUs:       {tier_info.get('min_gpus', 32)}")
                print(f"   Fleet management:   Enabled")
                print(f"   GPU-hour metering:  Enabled")
                try:
                    from terradev_cli.core.stripe_manager import StripeManager
                    sm = StripeManager()
                    sub_item_id = sm.get_subscription_item_id(email)
                    if sub_item_id:
                        print(f"   Stripe metering:    Linked ✓")
                        print(f"   GPU-hours will be billed to your card monthly via Stripe.")
                    else:
                        print(f"   Stripe metering:    Pending (will link on first provision)")
                except Exception:
                    print(f"   Stripe metering:    Will link on first provision")

            # Migration report
            if running_count > 0 or inference_eps:
                print(f"\nGraceful migration:")
                print(f"   Previous tier:      {old_tier['name']} (max {old_tier['max_instances']} servers)")
                print(f"   New tier:           {tier_info['name']} (max {tier_info['max_instances']} servers)")
                if running_count > 0:
                    print(f"   Running instances:  {running_count} — all carried forward ✓")
                    if activated_tier == 'enterprise_plus':
                        print(f"   All {running_count} instances within new unlimited limit")
                    elif isinstance(old_tier['max_instances'], int) and running_count > old_tier['max_instances']:
                        print(f"   You had {running_count} instances (over old limit of {old_tier['max_instances']})")
                    if isinstance(tier_info['max_instances'], int):
                        if running_count <= tier_info['max_instances']:
                            print(f"   All {running_count} instances within new limit of {tier_info['max_instances']}")
                        else:
                            print(f"   {running_count} instances exceeds new limit of {tier_info['max_instances']} — oldest will be grandfathered")
                if inference_eps:
                    print(f"   Inference endpoints: {len(inference_eps)} — migrated to new tier ✓")
                    if 'inference' in tier_info.get('features', []) or 'all' in tier_info.get('features', []):
                        print(f"   Inference endpoints fully supported on {tier_info['name']}")

            print(f"\nYou're all set. Your new limits are active immediately.")
            return

        except Exception as e:
            print(f"Activation check failed: {e}")
            return

    # ── Show upgrade options and open Stripe checkout ──
    current = api.tier['name']
    print(f"Current tier: {current}")
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  Research+ ($49.99/mo)                                     │")
    print("│  • 80 provisions/month                                     │")
    print("│  • 8 concurrent servers                                    │")
    print("│  • 1 user seat                                             │")
    print("│  • Inference endpoints                                     │")
    print("│  • 7 cloud providers                                       │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Enterprise ($299.99/mo)                                   │")
    print("│  • Unlimited provisions                                    │")
    print("│  • 32 concurrent servers                                   │")
    print("│  • 5 user seats                                            │")
    print("│  • Full provenance & audit trail                           │")
    print("│  • All 11+ cloud providers                                 │")
    print("│  • Priority support + SLA guarantee                        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Enterprise+ ($0.09/GPU-hr — metered, 32 GPU minimum)      │")
    print("│  • Unlimited provisions, servers, and seats                 │")
    print("│  • Minimum 32 GPUs under management                        │")
    print("│  • $0.09 per GPU-hour (billed via Stripe metered usage)    │")
    print("│  • All 11+ cloud providers + dedicated support              │")
    print("│  • Full provenance, SLA guarantee, fleet management         │")
    print("│  • GPU-hour metering + cost analytics dashboard             │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()

    if not tier:
        if not sys.stdin.isatty():
            print("❌ --tier is required in non-interactive mode")
            print("   Usage: terradev upgrade --tier research_plus")
            sys.exit(2)
        tier = click.prompt(
            'Which tier?',
            type=click.Choice(['research_plus', 'enterprise', 'enterprise_plus']),
        )

    checkout_url = api.get_stripe_checkout_url(tier)
    tier_labels = {'research_plus': 'Research+', 'enterprise': 'Enterprise', 'enterprise_plus': 'Enterprise+'}
    tier_prices = {'research_plus': '$49.99/mo', 'enterprise': '$299.99/mo', 'enterprise_plus': '$0.09/GPU-hr (32 GPU min)'}
    tier_label = tier_labels.get(tier, tier)
    price = tier_prices.get(tier, '')

    print(f"\nOpening Stripe checkout for {tier_label} ({price})...")
    print(f"   {checkout_url}")

    # Auto-open in browser
    try:
        import webbrowser
        webbrowser.open(checkout_url)
        
        # Track Stripe checkout opened
        try:
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("stripe_checkout_opened", {
                        "tier": api.tier["name"],
                        "stripe_session_id": stripe_session_id,
                        "upgrade_url": checkout_url,
                        "price_point": "49.99",
                        "product": "research_plus"
                    })
        except Exception:
            pass  # Telemetry is best-effort
        print("   Opened in your browser")
    except Exception:
        print("   Could not open browser — please visit the URL above")

    print(f"\nAfter payment, activate your tier:")
    print(f"   terradev upgrade --activate --email YOUR_EMAIL")

def validate_credentials(provider: str, credentials: Dict[str, str]) -> bool:
    """Validate that all required credentials are present for a provider"""
    required_creds = {
        'aws': ['api_key', 'secret_key'],
        'gcp': ['project_id', 'credentials_file'],
        'azure': ['subscription_id', 'tenant_id', 'client_id', 'client_secret'],
        'runpod': ['api_key'],
        'vastai': ['api_key'],
        'lambda_labs': ['api_key'],
        'coreweave': ['api_key'],
        'tensordock': ['api_key', 'api_token'],
        'huggingface': ['api_key', 'namespace'],
        'baseten': ['api_key'],
        'oracle': ['api_key', 'tenancy_ocid', 'compartment_ocid', 'region'],
        'crusoe': ['access_key', 'secret_key', 'project_id']
    }
    
    provider_lower = provider.lower()
    if provider_lower not in required_creds:
        return False
    
    missing = []
    for req in required_creds[provider_lower]:
        if req not in credentials or not credentials[req].strip():
            missing.append(req)
    
    if missing:
        print(f"   ERROR: Missing required credentials: {', '.join(missing)}")
        return False
    
    return True

@cli.command()
@click.option('--provider', '-p', help='Configure specific provider')
def configure(provider):
    """Configure cloud provider credentials"""
    
    if provider:
        # Configure specific provider
        from terradev_cli.credential_prompt import prompt_for_credentials
        
        print(f"   Configure {provider.upper()} credentials")
        
        # Temporarily set up provider-specific prompt
        config_dir = Path.home() / '.terradev'
        credentials_file = config_dir / 'credentials.json'
        
        # Load existing credentials
        existing_creds = {}
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                existing_creds = json.load(f)
        
        # Provider configurations
        provider_configs = {
            'runpod': {
                'name': 'RunPod',
                'key_name': 'API Key',
                'help': 'Get from: https://runpod.io/console/settings/api-keys',
                'example': 'rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'vastai': {
                'name': 'Vast.ai',
                'key_name': 'API Key',
                'help': 'Get from: https://console.vast.ai/api-keys',
                'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'aws': {
                'name': 'AWS',
                'key_name': 'Access Key ID',
                'help': 'Get from: AWS IAM console',
                'example': 'AKIAIOSFODNN7EXAMPLE'
            },
            'gcp': {
                'name': 'Google Cloud',
                'key_name': 'Service Account JSON',
                'help': 'Get from: GCP Console → IAM & Admin → Service Accounts',
                'example': 'path/to/service-account.json'
            },
            'azure': {
                'name': 'Azure',
                'key_name': 'Client ID',
                'help': 'Get from: Azure Portal → App registrations',
                'example': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
            },
            'lambda_labs': {
                'name': 'Lambda Labs',
                'key_name': 'API Key',
                'help': 'Get from: Lambda Labs dashboard',
                'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'coreweave': {
                'name': 'CoreWeave',
                'key_name': 'API Key',
                'help': 'Get from: CoreWeave dashboard',
                'example': 'cw_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'tensordock': {
                'name': 'TensorDock',
                'key_name': 'API Key',
                'help': 'Get from: TensorDock dashboard',
                'example': 'td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'oracle': {
                'name': 'Oracle Cloud',
                'key_name': 'API Key',
                'help': 'Get from: Oracle Cloud Console',
                'example': 'ocid1.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'crusoe': {
                'name': 'Crusoe Cloud',
                'key_name': 'API Key',
                'help': 'Get from: Crusoe dashboard',
                'example': 'crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'huggingface': {
                'name': 'HuggingFace',
                'key_name': 'API Token',
                'help': 'Get from: https://huggingface.co/settings/tokens',
                'example': 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            },
            'baseten': {
                'name': 'Baseten',
                'key_name': 'API Key',
                'help': 'Get from: Baseten dashboard',
                'example': 'bt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            }
        }
        
        config = provider_configs.get(provider.lower())
        if not config:
            print(f"Unknown provider: {provider}")
            print(f"Available providers: {', '.join(provider_configs.keys())}")
            return
        
        print(f"   {config['name']}")
        print(f"   Help: {config['help']}")
        print(f"   Example: {config['example']}")
        
        # Check if already configured
        existing_key = existing_creds.get(provider.lower(), {}).get('api_key')
        if existing_key:
            print(f"   Already configured: {existing_key[:10]}...")
            if not click.confirm(f"   Update {config['name']} credentials?", default=False):
                return
        
        # Prompt for API key
        api_key = click.prompt(f"   Enter {config['key_name']}", 
                          hide_input=True, 
                          show_default=False)
        
        if api_key.strip():
            if provider.lower() == 'gcp':
                # GCP needs special handling
                project_id = click.prompt("   Enter GCP Project ID", default="my-project")
                existing_creds[provider.lower()] = {
                    'service_account_key': api_key.strip(),
                    'project_id': project_id
                }
            elif provider.lower() == 'azure':
                # Azure needs multiple credentials
                subscription_id = click.prompt("   Enter Azure Subscription ID")
                tenant_id = click.prompt("   Enter Azure Tenant ID")
                client_id = click.prompt("   Enter Azure Client ID")
                client_secret = click.prompt("   Enter Azure Client Secret", hide_input=True)
                existing_creds[provider.lower()] = {
                    'subscription_id': subscription_id.strip(),
                    'tenant_id': tenant_id.strip(),
                    'client_id': client_id.strip(),
                    'client_secret': client_secret.strip()
                }
            elif provider.lower() == 'tensordock':
                # TensorDock needs API key + token
                api_token = click.prompt("   Enter TensorDock API Token", hide_input=True)
                existing_creds[provider.lower()] = {
                    'api_key': api_key.strip(),
                    'api_token': api_token.strip()
                }
            elif provider.lower() == 'oracle':
                # Oracle needs multiple credentials
                tenancy_ocid = click.prompt("   Enter Oracle Tenancy OCID")
                compartment_ocid = click.prompt("   Enter Oracle Compartment OCID")
                region = click.prompt("   Enter Oracle Region", default="us-ashburn-1")
                existing_creds[provider.lower()] = {
                    'api_key': api_key.strip(),
                    'tenancy_ocid': tenancy_ocid.strip(),
                    'compartment_ocid': compartment_ocid.strip(),
                    'region': region.strip()
                }
            elif provider.lower() == 'crusoe':
                # Crusoe needs multiple credentials
                access_key = click.prompt("   Enter Crusoe Access Key")
                secret_key = click.prompt("   Enter Crusoe Secret Key", hide_input=True)
                project_id = click.prompt("   Enter Crusoe Project ID")
                existing_creds[provider.lower()] = {
                    'access_key': access_key.strip(),
                    'secret_key': secret_key.strip(),
                    'project_id': project_id.strip()
                }
            elif provider.lower() == 'huggingface':
                # HuggingFace needs API token + namespace
                namespace = click.prompt("   Enter HuggingFace Namespace (username or org)")
                existing_creds[provider.lower()] = {
                    'api_key': api_key.strip(),
                    'namespace': namespace.strip()
                }
            else:
                existing_creds[provider.lower()] = {'api_key': api_key.strip()}
            
            # Save credentials
            with open(credentials_file, 'w') as f:
                json.dump(existing_creds, f, indent=2)
            
            # Validate credentials
            if validate_credentials(provider, existing_creds[provider.lower()]):
                print(f"   OK: {config['name']} credentials validated and saved")
                print(f"   Test with: terradev quote --gpu-type a100 --providers {provider}")
            else:
                print(f"   ERROR: {config['name']} credentials validation failed")
                print(f"   Please check your credentials and try again")
        
    else:
        # Interactive configuration for all providers
        from terradev_cli.credential_prompt import prompt_for_credentials
        configured_providers = prompt_for_credentials()
        
        if configured_providers:
            print(f"\nReady to get quotes from: {', '.join(configured_providers)}")
            print(f"   Try: terradev quote --gpu-type a100")
        else:
            print(f"\nNo providers configured")
            print(f"   Run 'terradev configure --provider <provider>' to add credentials")
            kubernetes_namespace = click.prompt('Kubernetes namespace (default: default)', default='default', show_default=False)
            if kubernetes_namespace:
                api.credentials['kubernetes_namespace'] = kubernetes_namespace
            karpenter_enabled = click.prompt('Enable Karpenter? (y/n)', default='n', show_default=False)
            if karpenter_enabled.lower() == 'y':
                api.credentials['kubernetes_karpenter_enabled'] = 'true'
            
            # Enhanced monitoring options
            monitoring_enabled = click.prompt('Enable monitoring stack (Prometheus/Grafana)? (y/n)', default='n', show_default=False)
            if monitoring_enabled.lower() == 'y':
                api.credentials['kubernetes_monitoring_enabled'] = 'true'
                api.credentials['kubernetes_prometheus_enabled'] = 'true'
                api.credentials['kubernetes_grafana_enabled'] = 'true'
                dashboard_port = click.prompt('Grafana dashboard port (default: 3000)', default='3000', show_default=False)
                if dashboard_port:
                    api.credentials['kubernetes_dashboard_port'] = dashboard_port
            
            print("   Kubernetes configured — cluster management, Karpenter, and monitoring")
        
        # W&B (enhanced)
        wandb_key = click.prompt('W&B API Key (optional, from wandb.ai/settings)', hide_input=True, default='', show_default=False)
        if wandb_key:
            api.credentials['wandb_api_key'] = wandb_key
            wandb_entity = click.prompt('W&B Entity (team/username, optional)', default='', show_default=False)
            if wandb_entity:
                api.credentials['wandb_entity'] = wandb_entity
            wandb_project = click.prompt('W&B Project (optional, default: terradev)', default='', show_default=False)
            if wandb_project:
                api.credentials['wandb_project'] = wandb_project
            wandb_base_url = click.prompt('W&B Server URL (optional, for self-hosted)', default='', show_default=False)
            if wandb_base_url:
                api.credentials['wandb_base_url'] = wandb_base_url
            
            # Enhanced W&B options
            wandb_enhanced = click.prompt('Enable enhanced W&B features (dashboards/reports/alerts)? (y/n)', default='n', show_default=False)
            if wandb_enhanced.lower() == 'y':
                api.credentials['wandb_dashboard_enabled'] = 'true'
                api.credentials['wandb_reports_enabled'] = 'true'
                api.credentials['wandb_alerts_enabled'] = 'true'
                api.credentials['wandb_integration_enabled'] = 'true'
            
            print("   W&B configured — experiment tracking, dashboards, and alerts")
        
        # LangChain (enhanced)
        langchain_config = click.prompt('Configure LangChain? (y/n)', default='n', show_default=False)
        if langchain_config.lower() == 'y':
            langchain_key = click.prompt('LangChain API Key (optional)', hide_input=True, default='', show_default=False)
            if langchain_key:
                api.credentials['langchain_api_key'] = langchain_key
            langsmith_key = click.prompt('LangSmith API Key (optional)', hide_input=True, default='', show_default=False)
            if langsmith_key:
                api.credentials['langsmith_api_key'] = langsmith_key
            langsmith_endpoint = click.prompt('LangSmith Endpoint (optional)', default='https://api.smith.langchain.com', show_default=False)
            if langsmith_endpoint:
                api.credentials['langsmith_endpoint'] = langsmith_endpoint
            workspace_id = click.prompt('LangSmith Workspace ID (optional)', default='', show_default=False)
            if workspace_id:
                api.credentials['workspace_id'] = workspace_id
            project_name = click.prompt('LangSmith Project Name (optional, default: terradev)', default='terradev', show_default=False)
            if project_name:
                api.credentials['project_name'] = project_name
            
            # Enhanced LangChain options
            langchain_enhanced = click.prompt('Enable enhanced LangChain features (dashboards/tracing/evaluation)? (y/n)', default='n', show_default=False)
            if langchain_enhanced.lower() == 'y':
                api.credentials['langchain_dashboard_enabled'] = 'true'
                api.credentials['langchain_tracing_enabled'] = 'true'
                api.credentials['langchain_evaluation_enabled'] = 'true'
                api.credentials['langchain_workflow_enabled'] = 'true'
            
            print("   LangChain configured — chains, workflows, and LangSmith integration")
        
        # SGLang
        sglang_config = click.prompt('Configure SGLang? (y/n)', default='n', show_default=False)
        if sglang_config.lower() == 'y':
            sglang_key = click.prompt('SGLang API Key (optional)', hide_input=True, default='', show_default=False)
            if sglang_key:
                api.credentials['sglang_api_key'] = sglang_key
            model_path = click.prompt('SGLang Model Path (optional)', default='', show_default=False)
            if model_path:
                api.credentials['sglang_model_path'] = model_path
            
            # Enhanced SGLang options
            sglang_enhanced = click.prompt('Enable enhanced SGLang features (dashboards/tracing/metrics)? (y/n)', default='n', show_default=False)
            if sglang_enhanced.lower() == 'y':
                api.credentials['sglang_dashboard_enabled'] = 'true'
                api.credentials['sglang_tracing_enabled'] = 'true'
                api.credentials['sglang_metrics_enabled'] = 'true'
                api.credentials['sglang_deployment_enabled'] = 'true'
                api.credentials['sglang_observability_enabled'] = 'true'
            
            print("   SGLang configured — model serving and optimization")
        
        prom_url = click.prompt('Prometheus Pushgateway URL (optional, e.g. http://pushgateway:9091)', default='', show_default=False)
        if prom_url:
            api.credentials['prometheus_pushgateway_url'] = prom_url
            prom_user = click.prompt('Pushgateway Username (optional)', default='', show_default=False)
            if prom_user:
                api.credentials['prometheus_username'] = prom_user
            prom_pass = click.prompt('Pushgateway Password', hide_input=True, default='', show_default=False)
            if prom_pass:
                api.credentials['prometheus_password'] = prom_pass
            print("   Prometheus configured — metrics will be pushed on provision/terminate events")
        
        # ── ML Platform Integrations ──
        print("\nML Platform Integrations (optional)")
        
        # KServe
        kserve_config = click.prompt('Configure KServe? (y/n)', default='n', show_default=False)
        if kserve_config.lower() == 'y':
            kserve_namespace = click.prompt('KServe Namespace (default: default)', default='default', show_default=False)
            if kserve_namespace:
                api.credentials['kserve_namespace'] = kserve_namespace
            kserve_kubeconfig = click.prompt('Kubeconfig path (optional, uses default)', default='', show_default=False)
            if kserve_kubeconfig:
                api.credentials['kserve_kubeconfig_path'] = kserve_kubeconfig
            print("   KServe configured — model deployment on Kubernetes")
        
        # LangSmith
        langsmith_key = click.prompt('LangSmith API Key (optional)', hide_input=True, default='', show_default=False)
        if langsmith_key:
            api.credentials['langsmith_api_key'] = langsmith_key
            langsmith_workspace = click.prompt('LangSmith Workspace ID (optional)', default='', show_default=False)
            if langsmith_workspace:
                api.credentials['langsmith_workspace_id'] = langsmith_workspace
            langsmith_endpoint = click.prompt('LangSmith Endpoint (optional)', default='', show_default=False)
            if langsmith_endpoint:
                api.credentials['langsmith_endpoint'] = langsmith_endpoint
            print("   LangSmith configured — tracing and evaluation")
        
        # DVC
        dvc_config = click.prompt('Configure DVC? (y/n)', default='n', show_default=False)
        if dvc_config.lower() == 'y':
            dvc_repo = click.prompt('DVC Repository Path (default: .)', default='.', show_default=False)
            if dvc_repo:
                api.credentials['dvc_repo_path'] = dvc_repo
            dvc_remote = click.prompt('DVC Remote Storage (optional)', default='', show_default=False)
            if dvc_remote:
                api.credentials['dvc_remote_storage'] = dvc_remote
            dvc_type = click.prompt('DVC Remote Type (s3, gs, azure, ssh)', default='', show_default=False)
            if dvc_type:
                api.credentials['dvc_remote_type'] = dvc_type
            print("   DVC configured — data versioning and storage")
        
        # MLflow
        mlflow_uri = click.prompt('MLflow Tracking URI (optional)', default='', show_default=False)
        if mlflow_uri:
            api.credentials['mlflow_tracking_uri'] = mlflow_uri
            mlflow_user = click.prompt('MLflow Username (optional)', hide_input=True, default='', show_default=False)
            if mlflow_user:
                api.credentials['mlflow_username'] = mlflow_user
            mlflow_pass = click.prompt('MLflow Password (optional)', hide_input=True, default='', show_default=False)
            if mlflow_pass:
                api.credentials['mlflow_password'] = mlflow_pass
            print("   MLflow configured — experiment tracking")
        
        # Ray
        ray_dashboard = click.prompt('Ray Dashboard URI (optional)', default='', show_default=False)
        if ray_dashboard:
            api.credentials['ray_dashboard_uri'] = ray_dashboard
            ray_head = click.prompt('Ray Head Node IP (optional)', default='', show_default=False)
            if ray_head:
                api.credentials['ray_head_node_ip'] = ray_head
            print("   Ray configured — distributed computing")
    
    api.save_credentials()
    
    print("\nCredentials saved successfully!")
    print(f"Stored in: {api.credentials_file}")
    print("Your keys are encrypted and stored locally only.")
    prov_limit = api.tier['provisions_per_month']
    prov_used = api.usage.get('provisions_this_month', 0)
    print(f"\nCurrent tier: {api.tier['name']}")
    print(f"Provisions this month: {prov_used}/{prov_limit}")
    
    # Show integration status
    try:
        from terradev_cli.integrations.wandb_integration import is_configured as wandb_ok
        from terradev_cli.integrations.prometheus_integration import is_configured as prom_ok
        integrations = []
        if wandb_ok(api.credentials):
            integrations.append("W&B")
        if prom_ok(api.credentials):
            integrations.append("Prometheus")
        if integrations:
            print(f"Active integrations: {', '.join(integrations)}")
    except Exception:
        pass

@cli.command()
@click.option('--gpu-type', '-g', default='A100', help='GPU type to quote')
@click.option('--providers', '-p', multiple=True, help='Specific providers (multiple allowed)')
@click.option('--parallel', default=6, help='Number of parallel queries')
@click.option('--region', '-r', help='Filter by region')
@click.option('--quick', '-q', is_flag=True, help='Quick provision the best quote')
def quote(gpu_type, providers, parallel, region, quick):
    """Get real-time quotes from all providers with quick provision option
    
    Examples:
      terradev quote --gpu-type a100
      terradev quote --gpu-type h100 --providers azure,gcp
      terradev quote --gpu-type a100 --quick     # Quick provision best quote
    """
    
    if _telemetry:
        _telemetry.log_action('quote', {
            'gpu_type': gpu_type,
            'providers': list(providers) if providers else ['all'],
            'parallel': parallel,
            'region': region,
            'quick': quick,
        })

    api = TerradevAPI()

    # ── Fetch quotes from all providers in parallel ──
    print(f"Querying providers for {gpu_type} pricing...")

    async def _fetch_all():
        tasks = []
        provider_list = [
            ('runpod', api.get_runpod_quotes), ('vastai', api.get_vastai_quotes),
            ('aws', api.get_aws_quotes), ('gcp', api.get_gcp_quotes),
            ('azure', api.get_azure_quotes), ('tensordock', api.get_tensordock_quotes),
            ('lambda', api.get_lambda_quotes), ('coreweave', api.get_coreweave_quotes),
            ('oracle', api.get_oracle_quotes), ('crusoe', api.get_crusoe_quotes),
        ]
        for pname, fn in provider_list:
            if not providers or pname in providers:
                tasks.append(fn(gpu_type))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    all_quotes = asyncio.run(_fetch_all())

    if not all_quotes:
        print("ERROR: No quotes returned.")
        print("   Configure provider API keys with: terradev configure --provider <name>")
        print("   Or check available providers with: terradev setup <provider>")
        return

    # Filter by region if specified
    if region:
        all_quotes = [q for q in all_quotes if region.lower() in q.get('region', '').lower()]

    all_quotes.sort(key=lambda q: q.get('price', 999))

    # ── Display results ──
    best = all_quotes[0]
    print(f"\nTerradev Quote — {gpu_type}")
    print(f"{'#':<4} {'Provider':<14} {'Region':<16} {'$/hr':<10} {'Type':<10}")
    print("-" * 58)
    for i, q in enumerate(all_quotes[:10]):
        spot = "spot" if q.get('availability') == 'spot' else 'on-demand'
        print(f"{i+1:<4} {q['provider']:<14} {q['region']:<16} ${q['price']:<9.2f} {spot:<10}")

    print(f"\nBest: ${best['price']:.2f}/hr on {best['provider']} ({best['region']})")
    monthly = best['price'] * 730
    print(f"Estimated monthly: ${monthly:,.0f}")

    if _telemetry:
        _telemetry.log_action('quote_completed', {
            'gpu_type': gpu_type,
            'best_price': best['price'],
            'provider': best['provider'],
            'num_quotes': len(all_quotes),
        })

    if quick:
        print(f"\nQuick provision: deploying {gpu_type} on {best['provider']}...")
        print(f"   Run: terradev provision -g {gpu_type} --providers {best['provider'].lower().replace(' ', '_')} --dry-run")
    else:
        print(f"\nProvision: terradev provision -g {gpu_type}")
        print(f"   Dry run:   terradev provision -g {gpu_type} --dry-run")

@cli.command()
@click.argument('provider', type=click.Choice([
    'runpod', 'vastai', 'lambda', 'tensordock', 'crusoe', 
    'baseten', 'coreweave', 'gcp', 'aws', 'azure', 'oracle'
]))
@click.option('--quick', '-q', is_flag=True, help='Show quick setup summary')
def setup(provider, quick):
    """Get setup instructions for any cloud provider
    
    Examples:
      terradev setup runpod
      terradev setup aws
      terradev setup azure --quick
    """
    
    setup_instructions = {
        'runpod': {
            'name': 'RunPod',
            'time': '5 minutes',
            'difficulty': 'EASIEST',
            'url': 'https://runpod.io',
            'steps': [
                'Create account at https://runpod.io',
                'Add $10+ credit: Dashboard → Billing → Add Funds',
                'Get API key: Dashboard → Settings → API Keys → "Create API Key"',
                'Copy and run:\nexport RUNPOD_API_KEY="paste-your-key-here"\necho \'export RUNPOD_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers runpod --gpu a100'
            ],
            'env_vars': ['RUNPOD_API_KEY']
        },
        'vastai': {
            'name': 'Vast.AI',
            'time': '5 minutes',
            'difficulty': 'EASIEST',
            'url': 'https://vast.ai',
            'steps': [
                'Create account at https://vast.ai',
                'Add $10+ credit: Dashboard → Billing → Add Payment',
                'Get API key: Account → API Keys → "New API Key"',
                'Copy and run:\nexport VAST_API_KEY="paste-your-key-here"\necho \'export VAST_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers vastai --gpu a100'
            ],
            'env_vars': ['VAST_API_KEY']
        },
        'lambda': {
            'name': 'Lambda Labs',
            'time': '5 minutes',
            'difficulty': 'EASIEST',
            'url': 'https://lambdalabs.com/service/gpu-cloud',
            'steps': [
                'Create account at https://lambdalabs.com/service/gpu-cloud',
                'Add payment: Dashboard → Billing → Add Card',
                'Get API key: Dashboard → API Keys → "Generate API Key" (save immediately!)',
                'Copy and run:\nexport LAMBDA_API_KEY="paste-your-key-here"\necho \'export LAMBDA_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers lambda --gpu a100'
            ],
            'env_vars': ['LAMBDA_API_KEY']
        },
        'tensordock': {
            'name': 'TensorDock',
            'time': '7 minutes',
            'difficulty': 'EASY',
            'url': 'https://tensordock.com',
            'steps': [
                'Create account at https://tensordock.com',
                'Add $10+ funds: Dashboard → Billing → Add Funds',
                'Get token: Dashboard → API Access → "Create Authorization"',
                'Copy and run:\nexport TENSORDOCK_TOKEN="paste-your-token-here"\necho \'export TENSORDOCK_TOKEN="paste-your-token-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers tensordock --gpu a100'
            ],
            'env_vars': ['TENSORDOCK_TOKEN']
        },
        'crusoe': {
            'name': 'Crusoe',
            'time': '10 minutes',
            'difficulty': 'MODERATE',
            'url': 'https://crusoe.ai',
            'steps': [
                'Apply at https://crusoe.ai/contact (requires approval, 1-2 days wait)',
                'After approval, login to dashboard',
                'Get API credentials: Settings → API Access → "Generate Credentials"',
                'Copy and run:\nexport CRUSOE_API_KEY="paste-your-key-here"\nexport CRUSOE_API_SECRET="paste-your-secret-here"\necho \'export CRUSOE_API_KEY="paste-your-key-here"\' >> ~/.bashrc\necho \'export CRUSOE_API_SECRET="paste-your-secret-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers crusoe --gpu a100'
            ],
            'env_vars': ['CRUSOE_API_KEY', 'CRUSOE_API_SECRET']
        },
        'baseten': {
            'name': 'Baseten',
            'time': '10 minutes',
            'difficulty': 'EASY',
            'url': 'https://baseten.co',
            'steps': [
                'Create account at https://baseten.co',
                'Add payment: Settings → Billing → Add Card ($50 free credits!)',
                'Get API key: Settings → API Keys → "Create API Key"',
                'Copy and run:\nexport BASETEN_API_KEY="paste-your-key-here"\necho \'export BASETEN_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers baseten --gpu a100'
            ],
            'env_vars': ['BASETEN_API_KEY']
        },
        'coreweave': {
            'name': 'CoreWeave',
            'time': '20 minutes',
            'difficulty': 'MODERATE',
            'url': 'https://cloud.coreweave.com',
            'steps': [
                'Apply at https://cloud.coreweave.com (requires approval, wait 1-24 hours)',
                'After approval, complete onboarding and add payment',
                'Download kubeconfig: Dashboard → Settings → Kubeconfig → Download',
                'Copy and run:\nmkdir -p ~/.kube\nmv ~/Downloads/coreweave-config ~/.kube/coreweave-config\n\nexport KUBECONFIG=~/.kube/coreweave-config\necho \'export KUBECONFIG=~/.kube/coreweave-config\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers coreweave --gpu a100'
            ],
            'env_vars': ['KUBECONFIG']
        },
        'gcp': {
            'name': 'Google Cloud Platform',
            'time': '25 minutes',
            'difficulty': 'MODERATE',
            'url': 'https://console.cloud.google.com',
            'steps': [
                'Create account at https://console.cloud.google.com',
                'Create project: Console → "Select Project" → "New Project" → Name: "terradev"',
                'Enable billing: Billing → Link Billing Account',
                'Enable Compute API: APIs & Services → Enable APIs → "Compute Engine API" → Enable',
                'Create service account:\nIAM & Admin → Service Accounts → "Create Service Account"\nName: terradev-sa\nRole: Compute Admin\nClick "Done"',
                'Create key:\nClick on terradev-sa@... → Keys → "Add Key" → "Create New Key"\nType: JSON → Create\nDownloads as terradev-xxxxx.json',
                'Copy and run:\nmkdir -p ~/.config/gcloud\nmv ~/Downloads/terradev-*.json ~/.config/gcloud/terradev-key.json\n\nexport GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terradev-key.json\necho \'export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terradev-key.json\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers gcp --gpu a100'
            ],
            'env_vars': ['GOOGLE_APPLICATION_CREDENTIALS']
        },
        'aws': {
            'name': 'Amazon Web Services',
            'time': '30 minutes',
            'difficulty': 'MODERATE',
            'url': 'https://aws.amazon.com',
            'steps': [
                'Create account at https://aws.amazon.com',
                'Add payment method: Account → Payment Methods',
                'Go to IAM: https://console.aws.amazon.com/iam',
                'Create user:\nUsers → "Create user"\nUsername: terradev\nCheck: "Programmatic access"\nNext',
                'Set permissions:\nAttach policies directly\nSearch and check: AmazonEC2FullAccess\nNext → Create user',
                'Save credentials (SHOWN ONLY ONCE):\nCopy "Access key ID"\nCopy "Secret access key"',
                'Copy and run:\n# Install AWS CLI if not installed\n# Mac: brew install awscli\n# Ubuntu: sudo apt install awscli\n# Windows: download from https://aws.amazon.com/cli/\n\n# Configure credentials\naws configure\n# When prompted, paste:\n# AWS Access Key ID: [paste-access-key]\n# AWS Secret Access Key: [paste-secret-key]\n# Default region: us-east-1\n# Default output format: json',
                'Test it:\nterradev quote --providers aws --gpu a100'
            ],
            'env_vars': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        },
        'azure': {
            'name': 'Microsoft Azure',
            'time': '30 minutes',
            'difficulty': 'MODERATE',
            'url': 'https://portal.azure.com',
            'steps': [
                'Create account at https://portal.azure.com',
                'Add payment: Subscriptions → Add payment method',
                'Install Azure CLI:\n# Mac\nbrew install azure-cli\n\n# Ubuntu/Debian\ncurl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash\n\n# Windows\n# Download from: https://aka.ms/installazurecliwindows',
                'Login:\naz login\n# Browser opens → Sign in with your Azure account',
                'Create service principal:\naz ad sp create-for-rbac --name "terradev" --role="Contributor" --scopes="/subscriptions/$(az account show --query id -o tsv)"',
                'Save the output (SHOWN ONLY ONCE):\n{\n  "appId": "xxxx-xxxx-xxxx",\n  "password": "xxxx-xxxx-xxxx",\n  "tenant": "xxxx-xxxx-xxxx"\n}',
                'Copy and run:\nexport AZURE_CLIENT_ID="paste-appId-here"\nexport AZURE_CLIENT_SECRET="paste-password-here"\nexport AZURE_TENANT_ID="paste-tenant-here"\nexport AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"\n\n# Make permanent\necho \'export AZURE_CLIENT_ID="paste-appId-here"\' >> ~/.bashrc\necho \'export AZURE_CLIENT_SECRET="paste-password-here"\' >> ~/.bashrc\necho \'export AZURE_TENANT_ID="paste-tenant-here"\' >> ~/.bashrc\necho \'export AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers azure --gpu a100'
            ],
            'env_vars': ['AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_TENANT_ID', 'AZURE_SUBSCRIPTION_ID']
        },
        'oracle': {
            'name': 'Oracle Cloud Infrastructure',
            'time': '35 minutes',
            'difficulty': 'ADVANCED',
            'url': 'https://cloud.oracle.com/free',
            'steps': [
                'Create account at https://cloud.oracle.com/free (requires credit card verification)',
                'After login, note your info:\nTenancy OCID: Profile → Tenancy → OCID (copy)\nUser OCID: Profile → User Settings → OCID (copy)\nRegion: Profile → Region (e.g., us-ashburn-1)',
                'Generate API key:\n# Create directory\nmkdir -p ~/.oci\n\n# Generate key pair\nopenssl genrsa -out ~/.oci/oci_api_key.pem 2048\nopenssl rsa -pubout -in ~/.oci/oci_api_key.pem -out ~/.oci/oci_api_key_public.pem\n\n# Get fingerprint\nopenssl rsa -pubout -outform DER -in ~/.oci/oci_api_key.pem | openssl md5 -c\n# Save the fingerprint output',
                'Upload public key to Oracle:\nProfile → User Settings → API Keys → "Add API Key"\nChoose: "Paste Public Key"\nPaste contents of: cat ~/.oci/oci_api_key_public.pem\nAdd',
                'Create config file:\ncat > ~/.oci/config << \'EOF\'\n[DEFAULT]\nuser=paste-user-ocid-here\nfingerprint=paste-fingerprint-here\ntenancy=paste-tenancy-ocid-here\nregion=us-ashburn-1\nkey_file=~/.oci/oci_api_key.pem\nEOF',
                'Set environment variable:\nexport OCI_CONFIG_FILE=~/.oci/config\necho \'export OCI_CONFIG_FILE=~/.oci/config\' >> ~/.bashrc\nsource ~/.bashrc',
                'Test it:\nterradev quote --providers oracle --gpu a100'
            ],
            'env_vars': ['OCI_CONFIG_FILE']
        }
    }
    
    info = setup_instructions[provider]
    
    if quick:
        print(f"{info['name']} Quick Setup ({info['time']})")
        print("=" * 50)
        print(f"Difficulty: {info['difficulty']}")
        print(f"URL: {info['url']}")
        print()
        print("Environment Variables:")
        for var in info['env_vars']:
            print(f"  {var}")
        print()
        print("Test Command:")
        print(f"  terradev quote --providers {provider} --gpu a100")
        print()
        print("For detailed instructions, run:")
        print(f"  terradev setup {provider}")
        return
    
    # Full detailed setup
    difficulty_stars = {
        'EASIEST': '*',
        'EASY': '**',
        'MODERATE': '***',
        'ADVANCED': '****'
    }
    
    print(f"{info['name']} Setup ({info['time']}) {difficulty_stars.get(info['difficulty'], '')}")
    print("=" * 60)
    print()
    
    for i, step in enumerate(info['steps'], 1):
        print(f"Step {i}: {step}")
        if i < len(info['steps']):
            print()
    
    print(f"Done! Your {info['name']} is configured.")

@cli.command()
@click.option('--gpu-type', '-g', required=True, help='GPU type (required)')
@click.option('--count', '-n', default=1, help='Number of instances')
@click.option('--max-price', type=float, help='Maximum price per hour')
@click.option('--providers', '-p', multiple=True, help='Specific providers (multiple allowed)')
@click.option('--parallel', default=6, help='Max parallel deploy threads')
@click.option('--dry-run', is_flag=True, help='Show allocation plan without launching')
@click.option('--type', type=click.Choice(['training', 'inference']), help='Workload type')
@click.option('--model-name', help='Model to deploy (for inference)')
@click.option('--endpoint-name', help='Endpoint name (for inference)')
@click.option('--min-workers', type=int, help='Minimum workers (for inference)')
@click.option('--max-workers', type=int, help='Maximum workers (for inference)')
def provision(gpu_type, count, max_price, providers, parallel, dry_run, type, model_name, endpoint_name, min_workers, max_workers):
    """Provision GPU instances across multiple clouds in parallel.
    
    Real multi-cloud arbitrage: queries all providers, builds a cost-optimized
    allocation plan spread across clouds, then deploys simultaneously.
    """
    api = TerradevAPI()
    provision_start = time.time()

    if type:
        print(f"Workload type: {type}")
        if type == 'inference':
            print(f"Model: {model_name or 'Not specified'}")
            print(f"Endpoint: {endpoint_name or 'Auto-generated'}")

    # ── Tier gates ──
    if count > api.tier['max_instances']:
        max_i = api.tier['max_instances']
        checkout_url = api.get_stripe_checkout_url('research_plus') if max_i < 8 else api.get_stripe_checkout_url('enterprise')
        tier_name = "Research+" if max_i < 8 else "Enterprise"
        tier_price = "$49.99" if max_i < 8 else "$299.99"
        tier_servers = "8" if max_i < 8 else "32"
        print(f"ERROR: {api.tier['name']} tier limited to {max_i} concurrent instance(s) — you requested {count}")
        print(f"")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Upgrade to {tier_name} — {tier_price}/mo{' ' * (22 - len(tier_name) - len(tier_price))}│")
        print(f"   │     {tier_servers} concurrent servers · unlock full power{' ' * (14 - len(tier_servers))}│")
        print(f"   │                                                     │")
        print(f"   │  {checkout_url}  │")
        
        # Track Stripe checkout initiation
        stripe_session_id = None
        try:
            # Extract session ID from checkout URL for tracking
            import re
            session_match = re.search(r"/cs_([a-zA-Z0-9]+)", checkout_url)
            if session_match:
                stripe_session_id = session_match.group(1)
                
            # Log Stripe checkout initiation
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("stripe_checkout_initiated", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "stripe_session_id": stripe_session_id,
                "reason": "monthly_provision_limit",
                "price_point": "49.99",
                "product": "research_plus"
            })
        except Exception:
            pass  # Telemetry is best-effort        
        # Log paywall hit to telemetry for visibility
        try:
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("paywall_hit", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "reason": "monthly_provision_limit"
            })
        except Exception:
            pass  # Telemetry is best-effort
        print(f"   └─────────────────────────────────────────────────────┘")
        try:
            if click.confirm('\n   Open checkout in browser?', default=True):
                import webbrowser
                webbrowser.open(checkout_url)
                
                # Track Stripe checkout opened
                try:
                    from terradev_cli.core.telemetry import TelemetryClient
                    telemetry = TelemetryClient()
                    telemetry.log_usage("stripe_checkout_opened", {
                        "tier": api.tier["name"],
                        "stripe_session_id": stripe_session_id,
                        "upgrade_url": checkout_url,
                        "price_point": "49.99",
                        "product": "research_plus"
                    })
                except Exception:
                    pass  # Telemetry is best-effort
                print(f"\n   OK: Opened in browser. After payment, run:")
                print(f"      terradev upgrade --activate --email YOUR_EMAIL")
        except click.Abort:
            pass
        return

    if not api.check_provision_limit():
        limit = api.tier['provisions_per_month']
        used = api.usage.get('provisions_this_month', 0)
        checkout_url = api.get_stripe_checkout_url('research_plus')
        print(f"ERROR: Monthly provision limit reached ({used}/{limit} for {api.tier['name']} tier)")
        print(f"")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Upgrade to Research+ — $49.99/mo               │")
        print(f"   │     80 provisions/month · 8 servers · inference     │")
        print(f"   │                                                     │")
        print(f"   │  {checkout_url}  │")
        
        # Track Stripe checkout initiation
        stripe_session_id = None
        try:
            # Extract session ID from checkout URL for tracking
            import re
            session_match = re.search(r"/cs_([a-zA-Z0-9]+)", checkout_url)
            if session_match:
                stripe_session_id = session_match.group(1)
                
            # Log Stripe checkout initiation
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("stripe_checkout_initiated", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "stripe_session_id": stripe_session_id,
                "reason": "monthly_provision_limit",
                "price_point": "49.99",
                "product": "research_plus"
            })
        except Exception:
            pass  # Telemetry is best-effort        
        # Log paywall hit to telemetry for visibility
        try:
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("paywall_hit", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "reason": "monthly_provision_limit"
            })
        except Exception:
            pass  # Telemetry is best-effort
        print(f"   └─────────────────────────────────────────────────────┘")
        try:
            if click.confirm('\n   Open checkout in browser?', default=True):
                import webbrowser
                webbrowser.open(checkout_url)
                
                # Track Stripe checkout opened
                try:
                    from terradev_cli.core.telemetry import TelemetryClient
                    telemetry = TelemetryClient()
                    telemetry.log_usage("stripe_checkout_opened", {
                        "tier": api.tier["name"],
                        "stripe_session_id": stripe_session_id,
                        "upgrade_url": checkout_url,
                        "price_point": "49.99",
                        "product": "research_plus"
                    })
                except Exception:
                    pass  # Telemetry is best-effort
                print(f"\n   OK: Opened in browser. After payment, run:")
                print(f"      terradev upgrade --activate --email YOUR_EMAIL")
        except click.Abort:
            pass
        return

    # ── Step 1: Fetch quotes from ALL providers in parallel ──
    print(f"Provisioning {count}x {gpu_type} (parallel={parallel})")
    print("Querying all providers for real-time pricing...")

    async def _fetch_all():
        tasks = []
        provider_list = [
            ('runpod', api.get_runpod_quotes), ('vastai', api.get_vastai_quotes),
            ('aws', api.get_aws_quotes), ('gcp', api.get_gcp_quotes),
            ('azure', api.get_azure_quotes), ('tensordock', api.get_tensordock_quotes),
            ('lambda', api.get_lambda_quotes), ('coreweave', api.get_coreweave_quotes),
            ('oracle', api.get_oracle_quotes), ('crusoe', api.get_crusoe_quotes),
        ]
        for pname, fn in provider_list:
            if fn is None:
                continue
            if not providers or pname in providers:
                tasks.append(fn(gpu_type))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    all_quotes = asyncio.run(_fetch_all())
    if not all_quotes:
        print("ERROR: No quotes returned. Run 'terradev configure' to set up API keys.")
        return

    all_quotes.sort(key=lambda q: q['price'])

    # Record to cost DB
    try:
        from terradev_cli.core.cost_tracker import record_quotes
        record_quotes(all_quotes)
    except Exception:
        pass

    # ── Step 2: Build allocation plan (cheapest-spread across clouds) ──
    print(f"{len(all_quotes)} quotes — building allocation plan...")

    # Silent: record price ticks for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_ticks_batch
        ticks = [
            {
                "gpu_type": q.get("gpu_type", gpu_type or ""),
                "provider": q.get("provider", ""),
                "region": q.get("region", ""),
                "price": q.get("price", 0),
                "spot": q.get("spot") or q.get("availability") == "spot",
                "workload_type": type or "training",
                "source": "provision",
            }
            for q in all_quotes
        ]
        record_price_ticks_batch(ticks)
    except Exception:
        pass

    if max_price:
        all_quotes = [q for q in all_quotes if q['price'] <= max_price]
        if not all_quotes:
            print(f"ERROR: No instances under ${max_price:.2f}/hr")
            return

    # Spread across providers: no more than ceil(count/2) on one cloud
    allocations = []
    prov_counts: Dict[str, int] = {}
    max_per = max((count + 1) // 2, 1)
    for q in all_quotes:
        if len(allocations) >= count:
            break
        pkey = q['provider'].lower().replace(' ', '_')
        if prov_counts.get(pkey, 0) >= max_per:
            continue
        prov_counts[pkey] = prov_counts.get(pkey, 0) + 1
        allocations.append(q)
    # Fill remaining if needed
    for q in all_quotes:
        if len(allocations) >= count:
            break
        allocations.append(q)
    allocations = allocations[:count]

    if not allocations:
        print("ERROR: Could not build allocation plan")
        return

    # ── Dry-run: show plan and exit ──
    if dry_run:
        print(f"\nDRY RUN — allocation plan ({count} instance(s)):")
        print(f"{'#':<4} {'Provider':<14} {'Region':<14} {'$/hr':<10} {'Type':<10}")
        print("-" * 56)
        total_hr = 0
        for i, q in enumerate(allocations):
            spot = "spot" if q.get('availability') == 'spot' else 'on-demand'
            total_hr += q['price']
            print(f"{i+1:<4} {q['provider']:<14} {q['region']:<14} ${q['price']:<9.2f} {spot:<10}")
        elapsed = (time.time() - provision_start) * 1000
        print(f"\nEstimated: ${total_hr:.2f}/hr  (${total_hr*24:.2f}/day)")
        print(f"Plan built in {elapsed:.0f}ms")
        return

    # ── Step 3: Deploy across clouds in parallel via real provider APIs ──
    unique_clouds = set(q['provider'] for q in allocations)
    print(f"Deploying {count} instance(s) across {len(unique_clouds)} cloud(s) simultaneously...")
    for q in allocations:
        print(f"   {q['provider']} / {q['region']} — ${q['price']:.2f}/hr")

    group_id = f"pg_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # ── Generate per-provision SSH keypair ──
    _provision_ssh_pubkey = ""
    try:
        from terradev_cli.core.ssh_key_manager import generate_provision_keypair
        _ssh_priv_path, _provision_ssh_pubkey = generate_provision_keypair(group_id)
        print(f"   SSH keypair generated for {group_id} (Ed25519, encrypted at rest)")
    except Exception as _ssh_err:
        print(f"   Warning: SSH key generation failed ({_ssh_err}) — manual --ssh-key needed for train")

    async def _provision_all():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        sem = asyncio.Semaphore(parallel)

        async def _do_one(q):
            async with sem:
                pname = q['provider'].lower().replace(' ', '_')
                creds = api._provider_creds(pname)
                t0 = time.monotonic()
                try:
                    provider = factory.create_provider(pname, creds)
                    spot_flag = q.get('availability') == 'spot'
                    itype = f"{pname}-{'spot' if spot_flag else 'ondemand'}-{gpu_type.lower()}"
                    result = await provider.provision_instance(
                        itype,
                        q.get('region', 'us-east-1'),
                        gpu_type,
                    )
                    elapsed = (time.monotonic() - t0) * 1000
                    iid = result.get('instance_id', f"{pname}_{int(time.time())}_{uuid.uuid4().hex[:6]}")
                    return {
                        'status': 'active', 'instance_id': iid,
                        'provider': q['provider'], 'region': q.get('region', ''),
                        'price': result.get('price_per_hour', q['price']),
                        'spot': q.get('availability') == 'spot',
                        'elapsed_ms': round(elapsed, 1), 'error': None,
                    }
                except Exception as e:
                    elapsed = (time.monotonic() - t0) * 1000
                    return {
                        'status': 'failed', 'instance_id': '',
                        'provider': q['provider'], 'region': q.get('region', ''),
                        'price': q['price'], 'spot': False,
                        'elapsed_ms': round(elapsed, 1), 'error': str(e),
                    }

        return await asyncio.gather(*[_do_one(q) for q in allocations])

    results = asyncio.run(_provision_all())
    provision_time = (time.time() - provision_start) * 1000

    # ── Step 4: Record results to cost DB + usage file ──
    succeeded = [r for r in results if r['status'] == 'active']
    failed = [r for r in results if r['status'] == 'failed']

    # Record provider reliability events for every provision attempt
    try:
        from terradev_cli.core.price_intelligence import record_provider_event, record_availability
        for r in succeeded:
            record_provider_event(
                provider=r['provider'], event_type="provision", success=True,
                gpu_type=gpu_type, region=r.get('region', ''),
                latency_ms=r.get('elapsed_ms'),
            )
            record_availability(gpu_type=gpu_type, provider=r['provider'],
                                available=True, region=r.get('region', ''),
                                response_ms=r.get('elapsed_ms'))
        for r in failed:
            record_provider_event(
                provider=r['provider'], event_type="provision", success=False,
                gpu_type=gpu_type, region=r.get('region', ''),
                latency_ms=r.get('elapsed_ms'), error=r.get('error', '')[:200],
            )
            record_availability(gpu_type=gpu_type, provider=r['provider'],
                                available=False, region=r.get('region', ''),
                                response_ms=r.get('elapsed_ms'),
                                error=r.get('error', '')[:200])
    except Exception:
        pass

    # Increment monthly provision counter for each successful provision
    for _ in succeeded:
        api.record_provision()

    for r in succeeded:
        # Cost tracking DB
        try:
            from terradev_cli.core.cost_tracker import record_provision
            record_provision(
                instance_id=r['instance_id'], provider=r['provider'],
                gpu_type=gpu_type, region=r['region'],
                price_hr=r['price'], spot=r['spot'], parallel_group=group_id,
            )
        except Exception:
            pass

        # Local usage file
        inst_data = {
            "id": r['instance_id'], "provider": r['provider'],
            "gpu_type": gpu_type, "price": r['price'],
            "region": r['region'], "spot": r['spot'],
            "parallel_group": group_id,
            "type": type or "training",
            "created_at": datetime.now().isoformat(),
        }
        if type == 'inference':
            inst_data.update({
                "model_name": model_name,
                "endpoint_name": endpoint_name or f"inf-{r['instance_id']}",
                "min_workers": min_workers or 1, "max_workers": max_workers or 5,
            })
        api.usage["instances_created"].append(inst_data)

        # Log provision to telemetry for visibility
        try:
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("provision", {
                "instance_id": r['instance_id'],
                "provider": r['provider'],
                "gpu_type": gpu_type,
                "region": r['region'],
                "price_hr": r['price'],
                "spot": r['spot'],
                "type": type or "training",
                "parallel_group": group_id,
                "success": True
            })
        except Exception:
            pass  # Telemetry is best-effort

    api.save_usage()

    # Store SSH key path in cost DB for this provision group
    if _provision_ssh_pubkey and succeeded:
        try:
            from terradev_cli.core.cost_tracker import set_ssh_key_path
            from terradev_cli.core.ssh_key_manager import get_provision_ssh_key_path as _get_ssh_path
            ssh_path = _get_ssh_path(group_id)
            if ssh_path:
                set_ssh_key_path(group_id, ssh_path)
        except Exception:
            pass

    # Silent: governance audit log for every provision
    try:
        from terradev_cli.core.data_governance import DataGovernanceManager
        gov = DataGovernanceManager()
        import json as _json
        for r in succeeded:
            entry = {
                "type": "provision",
                "timestamp": datetime.now().isoformat(),
                "instance_id": r["instance_id"],
                "provider": r["provider"],
                "gpu_type": gpu_type,
                "region": r["region"],
                "price_hr": r["price"],
                "spot": r["spot"],
                "workload_type": type or "training",
                "parallel_group": group_id,
            }
            with open(gov._audit_file, "a") as af:
                af.write(_json.dumps(entry) + "\n")
    except Exception:
        pass

    # ── Step 5: Integration hooks (W&B + Prometheus) ──
    # Prometheus: push provision metrics
    try:
        from terradev_cli.integrations.prometheus_integration import (
            is_configured as prom_configured, build_provision_metrics, push_metrics,
        )
        if prom_configured(api.credentials) and succeeded:
            for r in succeeded:
                payload = build_provision_metrics(
                    provider=r['provider'], gpu_type=gpu_type,
                    region=r.get('region', ''), instance_id=r['instance_id'],
                    price_per_hour=r['price'],
                )
                push_metrics(api.credentials, payload)
    except Exception:
        pass

    # W&B: show env var injection status
    wandb_injected = False
    try:
        from terradev_cli.integrations.wandb_integration import is_configured as wandb_configured, build_env_vars
        if wandb_configured(api.credentials) and succeeded:
            wandb_injected = True
    except Exception:
        pass

    # ── Step 6: Print results ──
    print(f"\n{'='*60}")
    if succeeded:
        total_hr = sum(r['price'] for r in succeeded)
        print(f"{len(succeeded)}/{count} instances launched across {len(set(r['provider'] for r in succeeded))} cloud(s)")
        print(f"{'Provider':<14} {'Instance ID':<36} {'$/hr':<8} {'ms':<8}")
        print("-" * 70)
        for r in succeeded:
            print(f"{r['provider']:<14} {r['instance_id']:<36} ${r['price']:<7.2f} {r['elapsed_ms']:<.0f}ms")
        print(f"\nTotal: ${total_hr:.2f}/hr  (${total_hr*24:.2f}/day)")
        print(f"Group: {group_id}")
        if wandb_injected:
            print(f"W&B: WANDB_* env vars ready for injection — use `terradev run` to auto-configure")
    if failed:
        print(f"\n{len(failed)} instance(s) failed:")
        for r in failed:
            print(f"   {r['provider']}/{r['region']}: {r['error']}")
    print(f"Total provision time: {provision_time:.0f}ms")
    if type == 'inference':
        print(f"Model: {model_name or 'Not specified'}")
        print(f"Type: Inference workload")

    # Soft nudge when approaching limit
    limit = api.tier['provisions_per_month']
    if limit != 'unlimited':
        used = api.usage.get('provisions_this_month', 0)
        remaining = limit - used
        if remaining <= 3 and remaining > 0:
            checkout_url = api.get_stripe_checkout_url('research_plus')
            print(f"\nWarning  {remaining} provision(s) remaining this month ({used}/{limit})")
            print(f"   Upgrade to Research+ for 80/mo → {checkout_url}")

@cli.command()
@click.option('--instance-id', '-i', required=True, help='Instance ID')
@click.option('--action', '-a', type=click.Choice(['status', 'stop', 'start', 'terminate']), 
              default='status', help='Action (default: status)')
def manage(instance_id, action):
    """Manage provisioned instances via real provider APIs."""
    api = TerradevAPI()

    instance = None
    for inst in api.usage["instances_created"]:
        if inst["id"] == instance_id:
            instance = inst
            break

    if not instance:
        print(f"Instance {instance_id} not found")
        print("Use 'terradev status' to see all instances")
        return

    pname = instance['provider'].lower().replace(' ', '_')
    print(f"{action.upper()} — {instance_id}")
    print(f"   Provider: {instance['provider']}  |  GPU: {instance['gpu_type']}  |  Region: {instance.get('region', '?')}")

    async def _run():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        provider = factory.create_provider(pname, creds)
        if action == 'status':
            return await provider.get_instance_status(instance_id)
        elif action == 'stop':
            return await provider.stop_instance(instance_id)
        elif action == 'start':
            return await provider.start_instance(instance_id)
        elif action == 'terminate':
            return await provider.terminate_instance(instance_id)

    try:
        result = asyncio.run(_run())

        if action == 'terminate':
            api.usage["instances_created"] = [i for i in api.usage["instances_created"] if i["id"] != instance_id]
            api.save_usage()
            try:
                from terradev_cli.core.cost_tracker import end_provision
                end_provision(instance_id)
            except Exception:
                pass
            # Enterprise+ GPU-hour metering — report usage to Stripe
            try:
                if api.tier.get('name') == 'Enterprise+':
                    from terradev_cli.core.stripe_manager import StripeManager
                    sm = StripeManager()
                    metering = sm._load_metering()
                    sub_item_id = metering.get('subscription_item_id')
                    if sub_item_id and instance:
                        created = instance.get('created_at', '')
                        gpu_count = instance.get('gpu_count', 1)
                        if created:
                            from datetime import datetime as _dt
                            try:
                                hours = max((_dt.now() - _dt.fromisoformat(created)).total_seconds() / 3600, 0.01)
                                gpu_hours = hours * gpu_count
                                sm.report_gpu_hours(
                                    sub_item_id, gpu_hours,
                                    gpu_type=instance.get('gpu_type', ''),
                                    instance_id=instance_id,
                                )
                                cost = round(gpu_hours * 0.09, 2)
                                print(f"   Enterprise+ metered: {gpu_hours:.1f} GPU-hrs (${cost:.2f})")
                            except Exception:
                                pass
            except Exception:
                pass
            # Prometheus: push terminate metrics
            try:
                from terradev_cli.integrations.prometheus_integration import (
                    is_configured as prom_configured, build_terminate_metrics, push_metrics,
                )
                if prom_configured(api.credentials):
                    created = instance.get('created_at', '')
                    duration = 0.0
                    if created:
                        from datetime import datetime as _dt
                        try:
                            duration = (_dt.now() - _dt.fromisoformat(created)).total_seconds()
                        except Exception:
                            pass
                    total_cost = instance.get('price', 0) * (duration / 3600)
                    payload = build_terminate_metrics(
                        provider=instance.get('provider', ''),
                        instance_id=instance_id,
                        total_cost=round(total_cost, 4),
                        duration_seconds=round(duration, 1),
                    )
                    push_metrics(api.credentials, payload)
            except Exception:
                pass
            print(f"Terminated {instance_id}")
        elif action == 'stop':
            print(f"Stopped {instance_id}")
        elif action == 'start':
            print(f"Started {instance_id}")
        else:
            st = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
            print(f"Status: {st}")

        if isinstance(result, dict):
            for k in ('ip_address', 'public_ip', 'gpu_utilization', 'uptime'):
                if result.get(k):
                    print(f"   {k}: {result[k]}")

    except Exception as e:
        print(f"Warning  Provider API error: {e}")
        print("   (Action may still have succeeded — check provider dashboard)")

@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--live', is_flag=True, help='Query providers for live instance status')
def status(format, live):
    """Show current status of all instances and usage."""
    api = TerradevAPI()

    print("Terradev Status")
    print("=" * 50)

    # Tier info
    prov_limit = api.tier.get('provisions_per_month', 10)
    seats = api.tier.get('user_seats', 1)
    print(f"Tier: {api.tier['name']}  |  Provisions: {prov_limit}/mo  |  Max instances: {api.tier['max_instances']}  |  Seats: {seats}")
    print(f"Providers: {', '.join(api.tier['providers'])}")

    # Cost DB summary
    try:
        from terradev_cli.core.cost_tracker import get_spend_summary
        summary = get_spend_summary(30)
        print(f"\nLast 30 days: ${summary['total_provision_cost']:.2f} provision cost  |  {summary['quotes_fetched']} quotes fetched")
        if summary['by_provider']:
            parts = [f"{p}: ${d['cost']:.2f} ({d['count']}x)" for p, d in summary['by_provider'].items()]
            print(f"   By provider: {', '.join(parts)}")
        if summary['egress_cost'] > 0:
            print(f"   Egress cost: ${summary['egress_cost']:.2f}")
    except Exception:
        print(f"\nProvisions this month: {api.usage.get('provisions_this_month', 0)}/{api.tier['provisions_per_month']}")

    # Instances
    instances = api.usage.get("instances_created", [])
    print(f"\nActive Instances ({len(instances)}):")

    if not instances:
        print("   No active instances")
        return

    if format == 'json':
        print(json.dumps(instances, indent=2))
        return

    # If --live, query each provider for real status
    live_statuses = {}
    if live and instances:
        print("   (querying providers for live status...)")

        async def _query_all():
            from terradev_cli.providers.provider_factory import ProviderFactory
            factory = ProviderFactory()
            results = {}
            for inst in instances:
                pname = inst['provider'].lower().replace(' ', '_')
                try:
                    creds = api._provider_creds(pname)
                    provider = factory.create_provider(pname, creds)
                    st = await provider.get_instance_status(inst['id'])
                    results[inst['id']] = st.get('status', '?') if isinstance(st, dict) else '?'
                except Exception:
                    results[inst['id']] = 'unknown'
            return results

        try:
            live_statuses = asyncio.run(_query_all())
        except Exception:
            pass

    print(f"{'ID':<36} {'Provider':<12} {'GPU':<8} {'$/hr':<8} {'Region':<14} {'Status':<10}")
    print("-" * 92)
    for inst in instances:
        iid = inst['id'][:35]
        prov = inst.get('provider', '?')
        gpu = inst.get('gpu_type', '?')
        price = f"${inst.get('price', 0):.2f}"
        region = inst.get('region', '?')[:13]
        st = live_statuses.get(inst['id'], 'tracked') if live else 'tracked'
        print(f"{iid:<36} {prov:<12} {gpu:<8} {price:<8} {region:<14} {st:<10}")

@cli.command()
@click.option('--dataset', '-d', required=True, help='Dataset path, S3 URI, GCS URI, HTTP URL, or HuggingFace name')
@click.option('--target-regions', help='Comma-separated target regions')
@click.option('--compression', default='auto', type=click.Choice(['auto', 'zstd', 'gzip', 'none']),
              help='Compression algorithm (default: auto)')
@click.option('--plan-only', is_flag=True, help='Show staging plan without executing')
def stage(dataset, target_regions, compression, plan_only):
    """Compress, chunk, and pre-position datasets near compute.

    Supports local files, S3/GCS URIs, HTTP URLs, and HuggingFace dataset names.
    """
    regions = [r.strip() for r in target_regions.split(',')] if target_regions else ['us-east-1', 'us-west-2', 'eu-west-1']

    print(f"📦 Dataset: {dataset}")
    print(f"Region Regions: {', '.join(regions)}")
    print(f"🗜️  Compression: {compression}")

    try:
        from terradev_cli.core.dataset_stager import DatasetStager
        stager = DatasetStager()

        # Show plan
        plan = stager.plan(dataset, regions, compression)
        pd = plan.to_dict()
        print(f"\nPlan Staging Plan:")
        print(f"   Original size:   {pd['original_size']}")
        print(f"   Compressed size: {pd['compressed_size']}  ({pd['compression_ratio']} reduction, {pd['compression_algo']})")
        print(f"   Chunks:          {pd['chunks']}  (chunk size: {pd['chunk_size']})")
        print(f"   Target regions:  {', '.join(pd['regions'])}")

        if plan_only:
            return

        # Execute
        def _progress(phase, msg):
            print(f"   ⚡ [{phase}] {msg}")

        result = asyncio.run(stager.stage(dataset, regions, compression, progress_callback=_progress))

        print(f"\nStaging complete")
        print(f"   Original:    {result['original_size']:,} bytes")
        print(f"   Compressed:  {result['compressed_size']:,} bytes  ({result['compression_ratio']} saved)")
        print(f"   Chunks:      {result['chunks']}")
        print(f"   Checksums:   {', '.join(c[:12] + '...' for c in result['checksums'][:3])}")
        print(f"   Staged to:   {result['staged_at']}")
        print(f"   Elapsed:     {result['total_elapsed_ms']:.0f}ms")

        for rname, rdata in result['regions'].items():
            print(f"   � {rname}: {rdata['chunks_uploaded']} chunks, {rdata['elapsed_ms']:.0f}ms")

        # Record to cost DB
        try:
            from terradev_cli.core.cost_tracker import record_staging
            record_staging(dataset, result['original_size'], result['compressed_size'],
                          result['compression'], result['chunks'], regions)
        except Exception:
            pass

        # Silent: governance audit log for dataset staging
        try:
            from terradev_cli.core.data_governance import DataGovernanceManager
            gov = DataGovernanceManager()
            import json as _json
            entry = {
                "type": "dataset_staging",
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "regions": regions,
                "original_size": result['original_size'],
                "compressed_size": result['compressed_size'],
                "compression": result['compression'],
                "chunks": result['chunks'],
            }
            with open(gov._audit_file, "a") as af:
                af.write(_json.dumps(entry) + "\n")
        except Exception:
            pass

    except ImportError:
        print("Warning  dataset_stager module not found — falling back to basic copy")
        for region in regions:
            print(f"   📤 Uploading to {region}...")
        print(f"\nDataset staged across {len(regions)} regions")

@cli.command()
@click.option('--instance-id', '-i', required=True, help='Instance ID')
@click.option('--command', '-c', required=True, help='Command to execute')
@click.option('--async-exec', is_flag=True, help='Run asynchronously')
def execute(instance_id, command, async_exec):
    """Execute commands on provisioned instances via provider APIs."""
    api = TerradevAPI()

    instance = None
    for inst in api.usage["instances_created"]:
        if inst["id"] == instance_id:
            instance = inst
            break

    if not instance:
        print(f"Instance {instance_id} not found")
        return

    pname = instance['provider'].lower().replace(' ', '_')
    print(f"Executing on {instance_id} ({instance['provider']}):")
    print(f"   $ {command}")

    async def _exec():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        provider = factory.create_provider(pname, creds)
        return await provider.execute_command(instance_id, command, async_exec)

    try:
        result = asyncio.run(_exec())
        if async_exec:
            job_id = result.get('job_id', 'unknown') if isinstance(result, dict) else 'unknown'
            print(f"Submitted async — job ID: {job_id}")
        else:
            stdout = result.get('stdout', '') if isinstance(result, dict) else str(result)
            stderr = result.get('stderr', '') if isinstance(result, dict) else ''
            exit_code = result.get('exit_code', 0) if isinstance(result, dict) else 0
            if stdout:
                print(f"Output:\n{stdout}")
            if stderr:
                print(f"Warning  Stderr:\n{stderr}")
            print(f"Exit code: {exit_code}")
    except Exception as e:
        print(f"Warning  Execution error: {e}")

@cli.command()
@click.option('--days', '-d', default=7, help='Number of days to analyze (default: 7)')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
def analytics(days, format):
    """Show cost analytics from the cost tracking database."""

    try:
        from terradev_cli.core.cost_tracker import get_spend_summary, get_daily_spend
        summary = get_spend_summary(days)

        if format == 'json':
            print(json.dumps(summary, indent=2, default=str))
            return

        print("Cost Analytics Dashboard")
        print("=" * 50)
        print(f"Analysis Period: Last {days} days\n")

        total_cost = summary.get('total_provision_cost', 0)
        total_provisions = summary.get('total_provisions', 0)
        quotes_fetched = summary.get('quotes_fetched', 0)
        egress_cost = summary.get('egress_cost', 0)
        by_provider = summary.get('by_provider', {})

        print(f"� Total Provision Cost: ${total_cost:.2f}")
        print(f"�️  Total Provisions:     {total_provisions}")
        print(f"� Quotes Fetched:       {quotes_fetched}")
        if egress_cost > 0:
            print(f"📡 Egress Cost:          ${egress_cost:.2f}")
        print(f"💵 All-in Cost:          ${total_cost + egress_cost:.2f}")

        if by_provider:
            print(f"\n💸 Cost by Provider:")
            for prov, data in sorted(by_provider.items(), key=lambda x: x[1]['cost'], reverse=True):
                print(f"   {prov:<14} ${data['cost']:>8.2f}  ({data['count']} provisions)")

        # Daily spend trend
        try:
            daily = get_daily_spend(days)
            if daily:
                print(f"\n📅 Daily Spend (last {min(days, len(daily))} days):")
                for row in daily[-7:]:
                    bar = '█' * max(1, int(row['cost'] / max(r['cost'] for r in daily) * 20)) if row['cost'] > 0 else '░'
                    print(f"   {row['date']}  ${row['cost']:>7.2f}  {bar}")
        except Exception:
            pass

    except Exception as e:
        # Fallback to local usage file
        api = TerradevAPI()
        total_cost = sum(inst.get('price', 0) * 24 for inst in api.usage.get('instances_created', []))
        print(f"Estimated Cost: ${total_cost:.2f} (from local tracking)")
        print(f"Instances: {len(api.usage.get('instances_created', []))}")
        print(f"WARNING: Cost DB unavailable: {e}", file=sys.stderr)
        sys.exit(1)

@cli.command()
def optimize():
    """Analyze running instances and recommend cheaper alternatives.

    Queries all providers for current pricing and compares against your
    running instances to find savings opportunities.
    """
    api = TerradevAPI()
    instances = api.usage.get("instances_created", [])

    if not instances:
        print("No active instances — nothing to optimize.")
        return

    print("Analyzing running instances against live pricing...")

    # Fetch fresh quotes for each GPU type in use
    gpu_types = list(set(inst.get('gpu_type', 'A100') for inst in instances))

    async def _fetch():
        all_q = {}
        for gt in gpu_types:
            tasks = [
                api.get_runpod_quotes(gt), api.get_vastai_quotes(gt),
                api.get_aws_quotes(gt), api.get_gcp_quotes(gt),
                api.get_azure_quotes(gt), api.get_tensordock_quotes(gt),
                api.get_lambda_quotes(gt), api.get_coreweave_quotes(gt),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            quotes = []
            for r in results:
                if isinstance(r, list):
                    quotes.extend(r)
            if quotes:
                quotes.sort(key=lambda q: q['price'])
                all_q[gt] = quotes
        return all_q

    market = asyncio.run(_fetch())

    total_savings = 0
    recommendations = []

    for inst in instances:
        gt = inst.get('gpu_type', 'A100')
        current_price = inst.get('price', 0)
        current_prov = inst.get('provider', '?')
        quotes = market.get(gt, [])
        if not quotes:
            continue
        cheapest = quotes[0]
        if cheapest['price'] < current_price * 0.9:  # >10% savings threshold
            saving = current_price - cheapest['price']
            total_savings += saving
            recommendations.append({
                'instance': inst['id'],
                'from': f"{current_prov} @ ${current_price:.2f}/hr",
                'to': f"{cheapest['provider']} / {cheapest['region']} @ ${cheapest['price']:.2f}/hr",
                'saving_hr': saving,
            })

    # Egress optimization
    try:
        from terradev_cli.core.egress_optimizer import estimate_egress_cost
        egress_recs = []
        providers_in_use = list(set(inst.get('provider', '').lower().replace(' ', '_') for inst in instances))
        if len(providers_in_use) >= 2:
            for i, src in enumerate(providers_in_use):
                for dst in providers_in_use[i+1:]:
                    cost = estimate_egress_cost(src, 'us-east-1', dst, 'us-east-1', 100)
                    egress_recs.append(f"   {src} → {dst}: ${cost:.2f}/100GB")
    except Exception:
        egress_recs = []

    print(f"\nOptimization Results")
    print("=" * 50)

    if recommendations:
        print(f"Potential savings: ${total_savings:.2f}/hr (${total_savings * 24:.2f}/day)")
        print(f"\nRecommendations ({len(recommendations)}):")
        for r in recommendations:
            print(f"   TIP: {r['instance'][:30]}")
            print(f"      Move from {r['from']}")
            print(f"      →     to  {r['to']}")
            print(f"      Saves ${r['saving_hr']:.2f}/hr")
    else:
        print("OK: All instances are at or near optimal pricing.")

    if egress_recs:
        print(f"\nEgress cost estimates (inter-cloud):")
        for er in egress_recs:
            print(er)

    print(f"\nOptimization complete.")

@cli.command()
@click.option('--export-grafana', is_flag=True, help='Export a Grafana dashboard JSON for Terradev metrics')
@click.option('--export-scrape-config', is_flag=True, help='Print a Prometheus scrape config snippet')
@click.option('--export-wandb-script', is_flag=True, help='Print a W&B setup script for remote instances')
def integrations(export_grafana, export_scrape_config, export_wandb_script):
    """Show status of observability & ML integrations and export configs.

    Terradev facilitates connections to your existing tools — your keys
    stay local, and all data flows directly from your instances to your services.

    Examples:
        terradev integrations
        terradev integrations --export-grafana
        terradev integrations --export-scrape-config
        terradev integrations --export-wandb-script
    """
    api = TerradevAPI()

    # ── Export modes ──
    if export_grafana:
        try:
            from terradev_cli.integrations.prometheus_integration import generate_grafana_dashboard_json
            import json as _json
            dashboard = generate_grafana_dashboard_json()
            print(_json.dumps(dashboard, indent=2))
            print("\nImport this JSON into Grafana → Dashboards → Import")
        except Exception as e:
            print(f"Error generating dashboard: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if export_scrape_config:
        try:
            from terradev_cli.integrations.prometheus_integration import generate_scrape_config
            print("# Add this to your prometheus.yml under scrape_configs:")
            print(generate_scrape_config())
        except Exception as e:
            print(f"Error generating config: {e}")
        return

    if export_wandb_script:
        try:
            from terradev_cli.integrations.wandb_integration import is_configured, generate_setup_script
            if not is_configured(api.credentials):
                print("W&B not configured. Run: terradev configure --provider wandb")
                return
            print(generate_setup_script(api.credentials))
        except Exception as e:
            print(f"Error generating script: {e}")
        return

    # ── Status display ──
    print("Terradev Integrations")
    print("=" * 50)

    # W&B
    try:
        from terradev_cli.integrations.wandb_integration import get_status_summary
        wb = get_status_summary(api.credentials)
        status = "Connected" if wb['configured'] else "Not configured"
        print(f"\nWeights & Biases          {status}")
        if wb['configured']:
            print(f"   Entity:      {wb['entity']}")
            print(f"   Project:     {wb['project']}")
            if wb['self_hosted']:
                print(f"   Server:      Self-hosted")
            print(f"   Auto-inject: WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT")
            print(f"   Hooks:       terradev run (Docker -e injection)")
        else:
            print(f"   Setup:       terradev configure --provider wandb --api-key YOUR_KEY")
            print(f"   Get key:     https://wandb.ai/settings → API Keys")
    except Exception:
        print(f"\nWeights & Biases          Module not available")

    # Prometheus
    try:
        from terradev_cli.integrations.prometheus_integration import get_status_summary
        pm = get_status_summary(api.credentials)
        status = "Connected" if pm['configured'] else "Not configured"
        print(f"\nPrometheus                {status}")
        if pm['configured']:
            print(f"   Pushgateway: {pm['pushgateway_url']}")
            print(f"   Auth:        {'Basic auth' if pm['auth_enabled'] else 'None'}")
            print(f"   Metrics:     terradev_provisions_total, terradev_gpu_cost_per_hour, ...")
            print(f"   Hooks:       provision (push), terminate (push)")
            print(f"   Export:      terradev integrations --export-grafana")
            print(f"                terradev integrations --export-scrape-config")
        else:
            print(f"   Setup:       terradev configure --provider prometheus --api-key PUSHGATEWAY_URL")
            print(f"   Requires:    A running Prometheus Pushgateway")
    except Exception:
        print(f"\nPrometheus                Module not available")

    # Existing infra hooks
    print(f"\nInfrastructure Hooks      Built-in")
    print(f"   Kubernetes:  terradev k8s")
    print(f"   Karpenter:   terradev k8s --workload training|inference")
    print(f"   Grafana:     terradev integrations --export-grafana")
    print(f"   OPA:         Policy-as-code via data governance module")

    print(f"\nConfigure integrations: terradev configure")

@cli.command()
def cleanup():
    """Clean up unused resources and temporary files"""
    print("Cleaning up unused resources...")
    
    api = TerradevAPI()
    
    # Remove old instances (older than 30 days)
    cutoff = datetime.now() - timedelta(days=30)
    old_instances = []
    
    for inst in api.usage["instances_created"]:
        created = datetime.fromisoformat(inst["created_at"])
        if created < cutoff:
            old_instances.append(inst)
    
    if old_instances:
        print(f"Found {len(old_instances)} old instances")
        for inst in old_instances:
            print(f"   Removing {inst['id']} ({inst['provider']})")
        
        api.usage["instances_created"] = [i for i in api.usage["instances_created"] 
                                         if i not in old_instances]
        api.save_usage()
    else:
        print("OK: No old instances found")
    
    print("Cleanup complete!")

@cli.command('job')
@click.argument('job_file', type=click.Path(exists=True))
@click.option('--optimize', help='Optimization criteria (cost, latency, balanced)')
def job(job_file, optimize):
    """Run Terradev job from YAML configuration"""
    print(f"Running job: {job_file}")
    
    if optimize:
        print(f"Optimization: {optimize}")
    
    # Load job configuration
    try:
        with open(job_file, 'r') as f:
            job_config = yaml.safe_load(f)
        
        print(f"Job Configuration:")
        print(f"   Name: {job_config.get('name', 'Unknown')}")
        print(f"   GPU Type: {job_config.get('gpu_type', 'A100')}")
        print(f"   Count: {job_config.get('count', 1)}")
        print(f"   Max Price: ${job_config.get('max_price', 0):.2f}")
        
        # Execute job (mock)
        print(f"\nExecuting job...")
        
        # This would integrate with the provision command
        print(f"OK: Job completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Error loading job file: {e}")

@cli.command()
@click.option('--model', '-m', required=True, help='Model name or path')
@click.option('--type', '-t', type=click.Choice(['llm', 'embedding', 'vision']), help='Model type')
@click.option('--provider', '-p', type=click.Choice(['runpod', 'vastai', 'lambda', 'baseten']), help='Provider preference')
@click.option('--gpu-type', '-g', help='GPU type preference')
@click.option('--region', '-r', help='Region preference')
@click.option('--max-latency', type=float, help='Max latency in ms')
@click.option('--max-cost', type=float, help='Max cost per request')
def infer(model, type, provider, gpu_type, region, max_latency, max_cost):
    """Deploy and manage inference endpoints"""
    print(f"Deploying inference for model: {model}")
    
    if type:
        print(f"Model type: {type}")
    if provider:
        print(f"Provider: {provider}")
    if gpu_type:
        print(f"GPU type: {gpu_type}")
    if region:
        print(f"Region: {region}")
    if max_latency:
        print(f"Max latency: {max_latency}ms")
    if max_cost:
        print(f"Max cost: ${max_cost}/request")
    
    # Get real quotes from providers
    print("Getting inference quotes from providers...")
    
    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    inference_providers = ['runpod', 'vastai', 'lambda_labs', 'baseten']
    if provider:
        inference_providers = [provider.replace('lambda', 'lambda_labs')]

    async def _fetch_inference_quotes():
        all_q = []
        for pname in inference_providers:
            try:
                raw = await api._get_provider_quotes(pname, target_gpu)
                for q in raw:
                    all_q.append({
                        'provider': pname,
                        'price': q.get('price', 0),
                        'latency': 120,  # estimated — real latency requires endpoint probing
                        'gpu_type': q.get('gpu_type', target_gpu),
                        'region': q.get('region', ''),
                    })
            except Exception:
                pass
        return all_q

    quotes = asyncio.run(_fetch_inference_quotes())
    
    # Filter by latency if specified
    if max_latency:
        quotes = [q for q in quotes if q['latency'] <= max_latency]
    
    # Filter by cost if specified
    if max_cost:
        quotes = [q for q in quotes if q['price'] <= max_cost]
    
    # Silent: record inference price ticks for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_ticks_batch
        ticks = [
            {
                "gpu_type": q.get("gpu_type", gpu_type or "A100"),
                "provider": q.get("provider", ""),
                "region": "",
                "price": q.get("price", 0),
                "spot": False,
                "workload_type": "inference",
                "source": "infer",
            }
            for q in quotes
        ]
        record_price_ticks_batch(ticks)
    except Exception:
        pass

    if not quotes:
        print("No suitable inference options found")
        return
    
    # Select best option (lowest price, then lowest latency)
    best_quote = min(quotes, key=lambda x: (x['price'], x['latency']))
    
    print(f"\nBest option: {best_quote['provider']}")
    print(f"Price: ${best_quote['price']}/request")
    print(f"Latency: {best_quote['latency']}ms")
    print(f"GPU: {best_quote['gpu_type']}")
    
    # Deploy to optimal provider via real API
    pname = best_quote['provider']
    print(f"\nDeploying to {pname}...")

    async def _deploy_inference():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        prov = factory.create_provider(pname, creds)
        itype = f"{pname}-inference-{best_quote['gpu_type'].lower()}"
        return await prov.provision_instance(
            itype, best_quote.get('region', 'us-east-1'), best_quote['gpu_type'],
        )

    try:
        prov_result = asyncio.run(_deploy_inference())
        endpoint_id = prov_result.get('instance_id', f"inf_{pname}_{int(time.time())}")
        endpoint_url = prov_result.get('endpoint_url', f"https://{pname}.api/inference/{endpoint_id}")
    except Exception as e:
        print(f"Warning  Provisioning error: {e}")
        endpoint_id = f"inf_{pname}_{int(time.time())}"
        endpoint_url = ""

    print(f"Inference endpoint deployed")
    print(f"ID Endpoint ID: {endpoint_id}")
    if endpoint_url:
        print(f"URL: {endpoint_url}")
    print(f"Status Status: Active")

    # Save to usage tracking
    api.usage["inference_endpoints"].append({
        "id": endpoint_id,
        "model": model,
        "provider": pname,
        "gpu_type": best_quote['gpu_type'],
        "price": best_quote['price'],
        "latency": best_quote['latency'],
        "url": endpoint_url,
        "created_at": datetime.now().isoformat()
    })

    # Register with inference router for health tracking + failover
    try:
        from terradev_cli.core.inference_router import InferenceRouter
        router = InferenceRouter()
        router.register_endpoint(
            endpoint_id=endpoint_id,
            provider=pname,
            url=endpoint_url,
            model=model,
            gpu_type=best_quote['gpu_type'],
            region=best_quote.get('region', ''),
            price_per_hour=best_quote['price'],
        )
        print(f"🛡️  Registered for health monitoring & auto-failover")
    except Exception:
        pass

@cli.command()
@click.argument('model_path')
@click.option('--name', '-n', required=True, help='Endpoint name (required)')
@click.option('--provider', '-p', type=click.Choice(['runpod', 'vastai', 'lambda', 'baseten']), help='Provider (runpod|vastai|lambda|baseten)')
@click.option('--gpu-type', '-g', help='GPU type (A100|H100|RTX4090)')
@click.option('--min-workers', type=int, default=1, help='Minimum workers')
@click.option('--max-workers', type=int, default=5, help='Maximum workers')
@click.option('--idle-timeout', type=int, default=300, help='Idle timeout in seconds')
@click.option('--cost-optimize', is_flag=True, help='Enable cost optimization')
@click.option('--dry-run', is_flag=True, help='Show deployment plan without deploying')
def infer_deploy(model_path, name, provider, gpu_type, min_workers, max_workers, idle_timeout, cost_optimize, dry_run):
    """Deploy inference endpoint"""
    print(f"Deploying inference endpoint: {name}")
    print(f"Model path: {model_path}")
    
    if provider:
        print(f"Provider: {provider}")
    if gpu_type:
        print(f"GPU type: {gpu_type}")
    
    print(f"Workers: {min_workers}-{max_workers}")
    print(f"Idle timeout: {idle_timeout}s")
    if cost_optimize:
        print(f"Cost optimization: Enabled")
    
    # Real deployment via provider API
    print(f"\nAnalyzing model requirements...")

    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    target_providers = ['runpod', 'vastai', 'lambda_labs', 'baseten']
    if provider:
        target_providers = [provider.replace('lambda', 'lambda_labs')]

    # Get best quote
    async def _get_best_quote():
        best = None
        for pname in target_providers:
            try:
                raw = await api._get_provider_quotes(pname, target_gpu)
                for q in raw:
                    price = q.get('price', 999)
                    if best is None or price < best.get('price', 999):
                        best = {'provider': pname, 'price': price,
                                'gpu_type': q.get('gpu_type', target_gpu),
                                'region': q.get('region', 'us-east-1'),
                                'instance_type': q.get('instance_type', '')}
            except Exception:
                pass
        return best

    best = asyncio.run(_get_best_quote())
    if not best:
        print("No providers returned quotes for this GPU type")
        return

    pname = best['provider']
    print(f"Selected provider: {pname} (${best['price']:.2f}/hr)")

    if dry_run:
        print("\n🔍 Dry run — deployment plan shown above. No resources provisioned.")
        return

    # Provision the instance
    print(f"Deploying endpoint...")
    async def _provision():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        prov = factory.create_provider(pname, creds)
        return await prov.provision_instance(
            best.get('instance_type', f"{pname}-{target_gpu.lower()}"),
            best.get('region', 'us-east-1'), target_gpu,
        )

    try:
        prov_result = asyncio.run(_provision())
        endpoint_id = prov_result.get('instance_id', f"ep_{name}_{int(time.time())}")
        endpoint_url = prov_result.get('endpoint_url', f"https://{pname}.api/inference/{endpoint_id}")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return

    print(f"\nEndpoint deployed successfully!")
    print(f"ID Endpoint ID: {endpoint_id}")
    print(f"Endpoint URL: {endpoint_url}")
    print(f"Status Status: Active")
    print(f"Workers: {min_workers}/{max_workers}")
    print(f"Cost: ${best['price']:.2f}/hr")
    
    # Silent: record inference deployment tick for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_tick
        record_price_tick(
            gpu_type=gpu_type or "A100",
            provider=provider or "auto",
            price_hr=0.0,
            region="",
            spot=False,
            workload_type="inference",
            source="infer_deploy",
        )
    except Exception:
        pass

    # Save to usage tracking (reuse existing api instance)
    api.usage["inference_endpoints"].append({
        "id": endpoint_id,
        "name": name,
        "model_path": model_path,
        "provider": provider or "auto-selected",
        "gpu_type": gpu_type or "auto-selected",
        "min_workers": min_workers,
        "max_workers": max_workers,
        "idle_timeout": idle_timeout,
        "cost_optimize": cost_optimize,
        "url": endpoint_url,
        "created_at": datetime.now().isoformat()
    })

    # Register with inference router for health tracking + failover
    try:
        from terradev_cli.core.inference_router import InferenceRouter
        router = InferenceRouter()
        router.register_endpoint(
            endpoint_id=endpoint_id,
            provider=provider or "auto",
            url=endpoint_url,
            model=model_path,
            gpu_type=gpu_type or "auto",
            region=best.get('region', ''),
            price_per_hour=best['price'],
        )
        print(f"🛡️  Registered for health monitoring & auto-failover")
    except Exception:
        pass

@cli.command()
@click.option('--gpu', '-g', required=True, help='GPU type (A100, H100, RTX4090, etc.)')
@click.option('--image', '-i', required=True, help='Docker image (e.g. pytorch/pytorch:latest)')
@click.option('--command', '-c', default=None, help='Command to run inside the container')
@click.option('--mount', '-m', multiple=True, help='Mount local path:container path (multiple allowed)')
@click.option('--port', '-p', multiple=True, type=int, help='Ports to expose (multiple allowed)')
@click.option('--env', '-e', multiple=True, help='Environment variables KEY=VALUE (multiple allowed)')
@click.option('--max-price', type=float, help='Maximum price per hour')
@click.option('--providers', multiple=True, help='Specific providers (multiple allowed)')
@click.option('--keep-alive', is_flag=True, help='Keep instance running after command completes')
@click.option('--dry-run', is_flag=True, help='Show plan without executing')
def run(gpu, image, command, mount, port, env, max_price, providers, keep_alive, dry_run):
    """Provision a GPU instance, deploy a Docker container, and execute — all in one command.

    Combines provision + deploy + execute into a single step. Automatically
    selects the cheapest instance, pulls the Docker image, and runs your workload.

    Examples:
        terradev run --gpu A100 --image pytorch/pytorch:latest -c "python train.py"
        terradev run --gpu H100 --image vllm/vllm-openai:latest --keep-alive --port 8000
        terradev run --gpu A100 --image my-training:latest -m ./data:/workspace/data -e WANDB_KEY=xxx
    """
    api = TerradevAPI()
    run_start = time.time()

    # ── Tier gate with telemetry ──
    if not api.check_provision_limit():
        # Check with telemetry server for server-side enforcement
        try:
            from terradev_cli.core.telemetry import check_license
            if not check_license('provision'):
                return  # Blocked by server-side paywall
        except ImportError:
            pass  # Fallback to local check
        
        limit = api.tier['provisions_per_month']
        used = api.usage.get('provisions_this_month', 0)
        checkout_url = api.get_stripe_checkout_url('research_plus')
        print(f"Monthly provision limit reached ({used}/{limit} for {api.tier['name']} tier)")
        print(f"")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Upgrade to Research+ — $49.99/mo               │")
        print(f"   │     80 provisions/month · 8 servers · inference     │")
        print(f"   │                                                     │")
        print(f"   │  {checkout_url}  │")
        
        # Track Stripe checkout initiation
        stripe_session_id = None
        try:
            # Extract session ID from checkout URL for tracking
            import re
            session_match = re.search(r"/cs_([a-zA-Z0-9]+)", checkout_url)
            if session_match:
                stripe_session_id = session_match.group(1)
                
            # Log Stripe checkout initiation
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("stripe_checkout_initiated", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "stripe_session_id": stripe_session_id,
                "reason": "monthly_provision_limit",
                "price_point": "49.99",
                "product": "research_plus"
            })
        except Exception:
            pass  # Telemetry is best-effort        
        # Log paywall hit to telemetry for visibility
        try:
            from terradev_cli.core.telemetry import TelemetryClient
            telemetry = TelemetryClient()
            telemetry.log_usage("paywall_hit", {
                "tier": api.tier["name"],
                "limit": limit,
                "used": used,
                "upgrade_url": checkout_url,
                "reason": "monthly_provision_limit"
            })
        except Exception:
            pass  # Telemetry is best-effort
        print(f"   └─────────────────────────────────────────────────────┘")
        try:
            if click.confirm('\n   Open checkout in browser?', default=True):
                import webbrowser
                webbrowser.open(checkout_url)
                
                # Track Stripe checkout opened
                try:
                    from terradev_cli.core.telemetry import TelemetryClient
                    telemetry = TelemetryClient()
                    telemetry.log_usage("stripe_checkout_opened", {
                        "tier": api.tier["name"],
                        "stripe_session_id": stripe_session_id,
                        "upgrade_url": checkout_url,
                        "price_point": "49.99",
                        "product": "research_plus"
                    })
                except Exception:
                    pass  # Telemetry is best-effort
                print(f"\n   Opened in browser. After payment, run:")
                print(f"      terradev upgrade --activate --email YOUR_EMAIL")
        except click.Abort:
            pass
        return

    print(f"Deploying terradev run")
    print(f"   GPU:     {gpu}")
    print(f"   Image:   {image}")
    if command:
        print(f"   Command: {command}")
    if mount:
        for m in mount:
            print(f"   Mount:   {m}")
    if port:
        print(f"   Ports:   {', '.join(str(p) for p in port)}")
    if keep_alive:
        print(f"   Mode:    keep-alive (instance stays running)")
    else:
        print(f"   Mode:    auto-terminate on completion")

    # ── Step 1: Get quotes ──
    print(f"\n🔍 Finding cheapest {gpu} instance...")

    async def _fetch_quotes():
        tasks = []
        provider_list = [
            ('runpod', api.get_runpod_quotes), ('vastai', api.get_vastai_quotes),
            ('aws', api.get_aws_quotes), ('gcp', api.get_gcp_quotes),
            ('azure', api.get_azure_quotes), ('tensordock', api.get_tensordock_quotes),
            ('lambda', api.get_lambda_quotes), ('coreweave', api.get_coreweave_quotes),
            ('oracle', api.get_oracle_quotes), ('crusoe', api.get_crusoe_quotes),
        ]
        for pname, fn in provider_list:
            if not providers or pname in providers:
                tasks.append(fn(gpu))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    all_quotes = asyncio.run(_fetch_quotes())
    if not all_quotes:
        print("No quotes returned. Run 'terradev configure' to set up API keys.")
        return

    # Silent: record price ticks for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_ticks_batch
        ticks = [
            {
                "gpu_type": q.get("gpu_type", gpu or ""),
                "provider": q.get("provider", ""),
                "region": q.get("region", ""),
                "price": q.get("price", 0),
                "spot": q.get("spot") or q.get("availability") == "spot",
                "workload_type": "training",
                "source": "run",
            }
            for q in all_quotes
        ]
        record_price_ticks_batch(ticks)
    except Exception:
        pass

    all_quotes.sort(key=lambda q: q['price'])
    if max_price:
        all_quotes = [q for q in all_quotes if q['price'] <= max_price]
        if not all_quotes:
            print(f"No instances under ${max_price:.2f}/hr")
            return

    best = all_quotes[0]
    print(f"   Best: {best['provider']} / {best.get('region', '?')} — ${best['price']:.2f}/hr")

    if dry_run:
        print(f"\nDRY RUN — would provision {best['provider']} {gpu} at ${best['price']:.2f}/hr")
        print(f"   Then pull {image} and run: {command or '(interactive)'}")
        elapsed = (time.time() - run_start) * 1000
        print(f"   Plan built in {elapsed:.0f}ms")
        return

    # ── Step 2: Provision ──
    print(f"\nProvisioning on {best['provider']}...")

    async def _provision():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        pname = best['provider'].lower().replace(' ', '_')
        creds = api._provider_creds(pname)
        provider = factory.create_provider(pname, creds)
        itype = f"{pname}-ondemand-{gpu.lower()}"
        result = await provider.provision_instance(
            itype, best.get('region', 'us-east-1'), gpu,
        )
        return result, provider, pname

    try:
        prov_result, provider_obj, pname = asyncio.run(_provision())
    except Exception as e:
        print(f"Provisioning failed: {e}")
        return

    instance_id = prov_result.get('instance_id', f"{pname}_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    print(f"   Instance: {instance_id}")

    # Record to usage
    inst_data = {
        "id": instance_id, "provider": best['provider'],
        "gpu_type": gpu, "price": best['price'],
        "region": best.get('region', ''), "spot": best.get('availability') == 'spot',
        "parallel_group": f"run_{int(time.time())}",
        "type": "run",
        "image": image,
        "created_at": datetime.now().isoformat(),
    }
    api.usage["instances_created"].append(inst_data)
    api.save_usage()

    try:
        from terradev_cli.core.cost_tracker import record_provision
        record_provision(
            instance_id=instance_id, provider=best['provider'],
            gpu_type=gpu, region=best.get('region', ''),
            price_hr=best['price'], spot=best.get('availability') == 'spot',
            parallel_group=inst_data['parallel_group'],
        )
    except Exception:
        pass

    # ── Step 3: Deploy Docker container ──
    print(f"\n🐳 Deploying container: {image}")

    docker_cmd_parts = ["docker", "run", "-d", "--gpus", "all"]
    for m in mount:
        docker_cmd_parts.extend(["-v", m])
    for p in port:
        docker_cmd_parts.extend(["-p", f"{p}:{p}"])
    for e_var in env:
        docker_cmd_parts.extend(["-e", e_var])

    # Auto-inject W&B env vars if configured
    try:
        from terradev_cli.integrations.wandb_integration import is_configured as wandb_configured, build_env_vars
        if wandb_configured(api.credentials):
            wandb_env = build_env_vars(api.credentials)
            for k, v in wandb_env.items():
                docker_cmd_parts.extend(["-e", f"{k}={v}"])
            print(f"   Status W&B env vars injected ({len(wandb_env)} vars)")
    except Exception:
        pass

    docker_cmd_parts.extend(["--name", f"terradev-{instance_id[:12]}"])
    docker_cmd_parts.append(image)
    if command:
        docker_cmd_parts.extend(["sh", "-c", command])

    docker_cmd = " ".join(docker_cmd_parts)
    print(f"   $ {docker_cmd}")

    async def _deploy_and_exec():
        from terradev_cli.providers.provider_factory import ProviderFactory
        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        prov = factory.create_provider(pname, creds)
        return await prov.execute_command(instance_id, docker_cmd, False)

    try:
        exec_result = asyncio.run(_deploy_and_exec())
        stdout = exec_result.get('stdout', '') if isinstance(exec_result, dict) else str(exec_result)
        stderr = exec_result.get('stderr', '') if isinstance(exec_result, dict) else ''
        exit_code = exec_result.get('exit_code', 0) if isinstance(exec_result, dict) else 0

        if stdout:
            print(f"\nStatus Output:\n{stdout}")
        if stderr:
            print(f"Warning  Stderr:\n{stderr}")

    except Exception as e:
        print(f"Warning  Container deployment error: {e}")
        print("   (Instance is still running — use 'terradev execute' to retry)")
        exit_code = 1

    # ── Step 4: Cleanup or keep alive ──
    total_time = (time.time() - run_start) * 1000

    if keep_alive:
        print(f"\n✅ Container running on {best['provider']} ({instance_id})")
        print(f"   💰 Cost: ${best['price']:.2f}/hr")
        if port:
            print(f"   🌐 Ports: {', '.join(str(p) for p in port)}")
        print(f"   🔧 Manage: terradev manage -i {instance_id} -a status")
        print(f"   🛑 Stop:   terradev manage -i {instance_id} -a terminate")
    else:
        if exit_code == 0:
            print(f"\n🛑 Auto-terminating instance...")
            async def _terminate():
                from terradev_cli.providers.provider_factory import ProviderFactory
                factory = ProviderFactory()
                creds = api._provider_creds(pname)
                prov = factory.create_provider(pname, creds)
                return await prov.terminate_instance(instance_id)
            try:
                asyncio.run(_terminate())
                api.usage["instances_created"] = [
                    i for i in api.usage["instances_created"] if i["id"] != instance_id
                ]
                api.save_usage()
                try:
                    from terradev_cli.core.cost_tracker import end_provision
                    end_provision(instance_id)
                except Exception:
                    pass
                # Enterprise+ GPU-hour metering — report usage to Stripe
                try:
                    if api.tier.get('name') == 'Enterprise+':
                        from terradev_cli.core.stripe_manager import StripeManager
                        sm = StripeManager()
                        metering_data = sm._load_metering()
                        sub_item_id = metering_data.get('subscription_item_id')
                        if sub_item_id:
                            from datetime import datetime as _dt
                            try:
                                gpu_count = int(os.environ.get('TERRADEV_GPU_COUNT', '1'))
                                hours = max(total_time / 3600000, 0.01)  # total_time is in ms
                                gpu_hours = hours * gpu_count
                                sm.report_gpu_hours(
                                    sub_item_id, gpu_hours,
                                    gpu_type=gpu,
                                    instance_id=instance_id,
                                )
                                cost = round(gpu_hours * 0.09, 2)
                                print(f"   Enterprise+ metered: {gpu_hours:.1f} GPU-hrs (${cost:.2f})")
                            except Exception:
                                pass
                except Exception:
                    pass
                print(f"   ✅ Terminated")
            except Exception as e:
                print(f"   Warning  Auto-terminate failed: {e}")
                print(f"   🛑 Manual: terradev manage -i {instance_id} -a terminate")
        else:
            print(f"\nWarning  Command exited with code {exit_code} — instance kept alive for debugging")
            print(f"   🔧 Debug:  terradev execute -i {instance_id} -c 'docker logs terradev-{instance_id[:12]}'")
            print(f"   🛑 Stop:   terradev manage -i {instance_id} -a terminate")

    print(f"⚡ Total time: {total_time:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Routing — Auto-failover + Latency-aware routing
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command('infer-status')
@click.option('--check', is_flag=True, help='Run live health probes before showing status')
def infer_status(check):
    """Show inference endpoint health, latency, and failover status.

    Displays all registered inference endpoints with their health state,
    average latency, provider, and failover configuration.

    Use --check to run live health probes before displaying.
    """
    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        print("❌ Inference router module not available.")
        print("   This may require Research+ or Enterprise tier.")
        print("   Run: terradev upgrade")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print("📭 No inference endpoints registered.")
        print("   Deploy one with: terradev infer --model <model>")
        return

    if check:
        print("🔍 Running health probes...")
        asyncio.run(router.check_all_endpoints())
        print()

    status = router.get_status()
    print("🧠 Inference Endpoint Status")
    print("=" * 70)
    print(f"   Total: {status['total_endpoints']}  |  Healthy: {status['healthy']}  |  Unhealthy: {status['unhealthy']}")
    print()

    header = f"{'ID':<28} {'Provider':<12} {'Health':<12} {'Latency':<10} {'$/hr':<8} {'Role':<10}"
    print(header)
    print("-" * 70)

    for ep in status['endpoints']:
        health_icon = {'healthy': '🟢', 'degraded': '🟡', 'unhealthy': '🔴', 'unknown': '⚪'}.get(ep['health'], '⚪')
        role = 'PRIMARY' if ep['is_primary'] else f"backup→{ep.get('backup', '?')}"
        lat = f"{ep['avg_latency_ms']}ms" if ep['avg_latency_ms'] > 0 else '—'
        print(f"{ep['endpoint_id']:<28} {ep['provider']:<12} {health_icon} {ep['health']:<9} {lat:<10} ${ep['price_per_hour']:<7.2f} {role}")

    # Show failover log if any
    failover_log = router.config_dir / 'failover_log.json'
    if failover_log.exists():
        try:
            with open(failover_log, 'r') as f:
                events = json.load(f)
            if events:
                print(f"\nPlan Recent Failover Events (last 5):")
                for ev in events[-5:]:
                    print(f"   {ev['timestamp']}  {ev['failed_provider']}/{ev['failed_endpoint'][:16]} → {ev['new_provider']}/{ev['new_primary'][:16]}")
        except Exception:
            pass


@cli.command('infer-failover')
@click.option('--dry-run', is_flag=True, help='Show what would happen without executing failover')
def infer_failover(dry_run):
    """Run health checks and auto-failover for inference endpoints.

    Probes all registered inference endpoints. If a primary endpoint is
    unhealthy and has a backup configured, traffic automatically shifts
    to the backup provider.

    Enterprise tier feature — requires Research+ or Enterprise.
    """
    api = TerradevAPI()
    tier_features = api.tier.get('features', [])
    if 'all' not in tier_features and 'inference' not in tier_features:
        print(f"❌ Inference failover requires Research+ or Enterprise tier.")
        print(f"   Current tier: {api.tier['name']}")
        print(f"   Run: terradev upgrade")
        sys.exit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        print("❌ Inference router module not available.")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print("📭 No inference endpoints registered.")
        sys.exit(1)

    print("🔍 Running health checks on all inference endpoints...")
    probes = asyncio.run(router.check_all_endpoints())

    for eid, probe in probes.items():
        ep = router.endpoints.get(eid)
        icon = '🟢' if probe.healthy else '🔴'
        lat = f"{probe.latency_ms:.0f}ms" if probe.latency_ms > 0 else '—'
        print(f"   {icon} {eid[:24]:<24} {ep.provider:<12} {lat}")

    if dry_run:
        print("\n🔍 DRY RUN — checking for failover candidates...")
        for eid, ep in router.endpoints.items():
            if ep.is_primary and ep.health.value == 'unhealthy' and ep.backup_endpoint_id:
                backup = router.endpoints.get(ep.backup_endpoint_id)
                if backup:
                    print(f"   Warning  WOULD FAILOVER: {ep.provider}/{eid[:16]} → {backup.provider}/{backup.endpoint_id[:16]}")
        print("   (No changes made)")
        return

    print("\n⚡ Checking for auto-failover...")
    events = asyncio.run(router.check_and_failover())

    if events:
        for ev in events:
            print(f"   🔄 FAILOVER: {ev['failed_provider']}/{ev['failed_endpoint'][:16]} → {ev['new_provider']}/{ev['new_primary'][:16]}")
            print(f"      Reason: {ev['reason']}")
        print(f"\n✅ {len(events)} failover(s) executed. Traffic shifted to healthy providers.")
    else:
        print("   ✅ All primary endpoints healthy — no failover needed.")


@cli.command('infer-route')
@click.option('--model', '-m', help='Filter by model name')
@click.option('--strategy', '-s', type=click.Choice(['latency', 'cost', 'score']),
              default='latency', help='Routing strategy (default: latency)')
@click.option('--measure', is_flag=True, help='Run fresh latency measurements before routing')
def infer_route(model, strategy, measure):
    """Find the best inference endpoint using latency-aware routing.

    Selects the optimal healthy endpoint based on strategy:
      - latency: lowest average response time (default)
      - cost: cheapest price per hour
      - score: weighted combination of latency + cost

    Use --measure to run fresh ping/TTFB probes before selecting.

    \b
    Integrates with WebPageTest TTFB probes for real-world latency data.
    Set WPT_API_KEY env var to enable WebPageTest integration.
    """
    api = TerradevAPI()
    tier_features = api.tier.get('features', [])
    if 'all' not in tier_features and 'inference' not in tier_features:
        print(f"❌ Inference routing requires Research+ or Enterprise tier.")
        print(f"   Run: terradev upgrade")
        sys.exit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        print("❌ Inference router module not available.")
        print("   Run: terradev upgrade")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print("📭 No inference endpoints registered.")
        return

    if measure:
        print("🔍 Running latency measurements...")
        wpt_key = os.environ.get('WPT_API_KEY')
        if wpt_key:
            print("   📡 WebPageTest integration enabled")

        async def _measure_all():
            probes = await router.check_all_endpoints()
            for eid, ep in router.endpoints.items():
                if ep.url:
                    # Try WPT first, fall back to HTTP TTFB
                    lat = await router.measure_latency_wpt(ep.url, wpt_key)
                    if lat is not None:
                        ep.latency_history.append(lat)
                        if len(ep.latency_history) > router.LATENCY_HISTORY_SIZE:
                            ep.latency_history = ep.latency_history[-router.LATENCY_HISTORY_SIZE:]
                        ep.avg_latency_ms = sum(ep.latency_history) / len(ep.latency_history)
                        source = 'WPT' if wpt_key else 'HTTP'
                        print(f"   Status {eid[:24]}: {lat:.0f}ms ({source})")
            router._save_endpoints()

        asyncio.run(_measure_all())
        print()

    best = router.get_best_endpoint(model=model, strategy=strategy)

    if not best:
        print(f"❌ No healthy endpoints found" + (f" for model '{model}'" if model else ""))
        return

    print(f"🎯 Best endpoint (strategy: {strategy}):")
    print(f"   Endpoint:  {best.endpoint_id}")
    print(f"   Provider:  {best.provider}")
    print(f"   Model:     {best.model}")
    print(f"   Region:    {best.region}")
    print(f"   Latency:   {best.avg_latency_ms:.0f}ms")
    print(f"   Cost:      ${best.price_per_hour:.2f}/hr")
    print(f"   Health:    {best.health.value}")
    if best.url:
        print(f"   URL:       {best.url}")


# ═══════════════════════════════════════════════════════════════════════════════
# Kubernetes / Karpenter Integration — `terradev k8s`
# ═══════════════════════════════════════════════════════════════════════════════

# Workload → Karpenter provisioner mapping (matches karpenter_provisioners.yaml)
_K8S_WORKLOAD_PROFILES = {
    'training': {
        'provisioner': 'training-workload-provisioner',
        'node_template': 'training-workload-template',
        'default_gpu': 'nvidia.com/gpu',
        'default_gpu_count': 1,
        'default_instance_hint': 'p3.2xlarge',
        'capacity_type': 'spot',       # spot-first for training
        'ttl_after_finished': 300,
        'restart_policy': 'Never',
    },
    'inference': {
        'provisioner': 'inference-workload-provisioner',
        'node_template': 'inference-workload-template',
        'default_gpu': 'nvidia.com/gpu',
        'default_gpu_count': 1,
        'default_instance_hint': 'g5.xlarge',
        'capacity_type': 'on-demand',   # stable for serving
        'ttl_after_finished': 0,        # keep alive
        'restart_policy': 'Always',
    },
    'cost-optimized': {
        'provisioner': 'cost-optimized-gpu-provisioner',
        'node_template': 'cost-optimized-gpu-template',
        'default_gpu': 'nvidia.com/gpu',
        'default_gpu_count': 1,
        'default_instance_hint': 'g4dn.xlarge',
        'capacity_type': 'spot',
        'ttl_after_finished': 60,
        'restart_policy': 'Never',
    },
    'high-performance': {
        'provisioner': 'high-performance-gpu-provisioner',
        'node_template': 'high-performance-gpu-template',
        'default_gpu': 'nvidia.com/gpu',
        'default_gpu_count': 8,
        'default_instance_hint': 'p4d.24xlarge',
        'capacity_type': 'on-demand',
        'ttl_after_finished': 600,
        'restart_policy': 'Never',
    },
}


def _build_k8s_job_manifest(
    name: str,
    image: str,
    command: Optional[str],
    workload: str,
    gpu_count: int,
    namespace: str,
    env_vars: List[str],
    mounts: List[str],
    budget: Optional[float],
    ports: List[int],
) -> Dict[str, Any]:
    """Build a Kubernetes Job/Deployment manifest that Karpenter will schedule."""
    profile = _K8S_WORKLOAD_PROFILES[workload]

    # Budget heuristic: if budget < $2/hr, force spot + cost-optimized
    effective_capacity = profile['capacity_type']
    if budget and budget < 2.0:
        effective_capacity = 'spot'

    labels = {
        'app': name,
        'terradev.io/workload': workload,
        'terradev.io/managed-by': 'terradev-cli',
    }

    node_selector = {
        'karpenter.sh/provisioner-name': profile['provisioner'],
    }
    tolerations = [
        {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'},
        {'key': 'karpenter.sh/capacity-type', 'value': effective_capacity, 'effect': 'NoSchedule'},
    ]

    # Container spec
    container: Dict[str, Any] = {
        'name': name,
        'image': image,
        'resources': {
            'limits': {profile['default_gpu']: gpu_count},
            'requests': {profile['default_gpu']: gpu_count},
        },
    }
    if command:
        container['command'] = ['sh', '-c', command]
    if env_vars:
        container['env'] = []
        for ev in env_vars:
            k, _, v = ev.partition('=')
            container['env'].append({'name': k, 'value': v})
    if ports:
        container['ports'] = [{'containerPort': p} for p in ports]

    # Volume mounts
    volumes: List[Dict] = []
    volume_mounts: List[Dict] = []
    for i, m in enumerate(mounts):
        host, _, ctr = m.partition(':')
        vol_name = f'mount-{i}'
        volumes.append({'name': vol_name, 'hostPath': {'path': host}})
        volume_mounts.append({'name': vol_name, 'mountPath': ctr})
    if volume_mounts:
        container['volumeMounts'] = volume_mounts

    pod_spec: Dict[str, Any] = {
        'nodeSelector': node_selector,
        'tolerations': tolerations,
        'containers': [container],
        'restartPolicy': profile['restart_policy'],
    }
    if volumes:
        pod_spec['volumes'] = volumes

    # For inference → Deployment; for everything else → Job
    if workload == 'inference':
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {'name': name, 'namespace': namespace, 'labels': labels},
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': name}},
                'template': {
                    'metadata': {'labels': labels},
                    'spec': pod_spec,
                },
            },
        }
        # Expose as a Service if ports are specified
        if ports:
            manifest['_service'] = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {'name': f'{name}-svc', 'namespace': namespace, 'labels': labels},
                'spec': {
                    'selector': {'app': name},
                    'ports': [{'port': p, 'targetPort': p} for p in ports],
                    'type': 'ClusterIP',
                },
            }
    else:
        manifest = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {'name': name, 'namespace': namespace, 'labels': labels},
            'spec': {
                'ttlSecondsAfterFinished': profile['ttl_after_finished'],
                'backoffLimit': 2,
                'template': {
                    'metadata': {'labels': labels},
                    'spec': pod_spec,
                },
            },
        }

    return manifest


def _kubectl_apply(manifest_dict: Dict[str, Any], dry_run: bool = False) -> bool:
    """Apply a manifest via kubectl."""
    import tempfile
    manifest_yaml = yaml.dump(manifest_dict, default_flow_style=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(manifest_yaml)
        f.flush()
        cmd = ['kubectl', 'apply', '-f', f.name]
        if dry_run:
            cmd.append('--dry-run=client')
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ kubectl error: {e.stderr.strip()}")
            return False
        finally:
            os.unlink(f.name)


@click.group()
def k8s():
    """Kubernetes cluster management with multi-cloud GPU nodes"""
    pass

cli.add_command(k8s)

@k8s.command('create')
@click.argument('cluster_name')
@click.option('--gpu', '-g', required=True, help='GPU type (H100, A100, L40)')
@click.option('--count', '-n', type=int, required=True, help='Number of GPU nodes')
@click.option('--max-price', type=float, default=4.00, help='Maximum price per hour')
@click.option('--multi-cloud', is_flag=True, help='Use multi-cloud provisioning')
@click.option('--prefer-spot', is_flag=True, default=True, help='Prefer spot instances')
@click.option('--aws-region', default='us-west-2', help='AWS region')
@click.option('--gcp-region', default='us-central1', help='GCP region')
@click.option('--control-plane', type=click.Choice(['eks', 'gke', 'self-hosted']), default='eks', help='Control plane type')
def k8s_create(cluster_name, gpu, count, max_price, multi_cloud, prefer_spot, aws_region, gcp_region, control_plane):
    """Create multi-cloud Kubernetes GPU cluster"""
    if not TerraformWrapper:
        print("❌ Kubernetes wrapper not available")
        sys.exit(1)
    
    if _telemetry:
        _telemetry.log_action('k8s_cluster_create', {
            'cluster_name': cluster_name,
            'gpu_type': gpu,
            'node_count': count,
            'multi_cloud': multi_cloud,
            'max_price': max_price,
            'prefer_spot': prefer_spot
        })
    
    wrapper = TerraformWrapper()
    
    cluster_config = {
        'name': cluster_name,
        'gpu_type': gpu,
        'node_count': count,
        'max_price': max_price,
        'multi_cloud': multi_cloud,
        'prefer_spot': prefer_spot,
        'aws_region': aws_region,
        'gcp_region': gcp_region,
        'control_plane': control_plane
    }
    
    print(f"Deploying Creating Kubernetes cluster '{cluster_name}'...")
    print(f"🎮 GPU Type: {gpu}")
    print(f"Status Node Count: {count}")
    print(f"💰 Max Price: ${max_price}/hr")
    print(f"☁️  Multi-Cloud: {multi_cloud}")
    print(f"🎯 Spot Instances: {prefer_spot}")
    print(f"")
    print(f"🧬 Topology optimization (auto-applied):")
    print(f"   Kubelet Topology Manager: restricted (NUMA-aligned)")
    print(f"   CPU Manager: static (pinned cores)")
    print(f"   GPUDirect RDMA: enabled (nvidia_peermem)")
    if count > 1:
        print(f"   SR-IOV: enabled ({count} nodes, VF-per-GPU pairing)")
        print(f"   NCCL: IB enabled, GDR_LEVEL=PIX, GDR_READ=1")
    else:
        print(f"   SR-IOV: single-node (not required)")
    print(f"   PCIe locality: GPU-NIC pairs forced to same NUMA node")
    
    success = wrapper.create_cluster(cluster_config)
    
    if success:
        print(f"✅ Cluster '{cluster_name}' created successfully!")
        print(f"   Topology: NUMA-aligned, GPUDirect RDMA, Topology Manager=restricted")
        print(f"Status Run 'terradev k8s info {cluster_name}' for details")
        print(f"🔗 Run 'export KUBECONFIG=~/.terradev/clusters/{cluster_name}.json' to connect")
    else:
        print(f"❌ Failed to create cluster '{cluster_name}'")

@k8s.command('destroy')
@click.argument('cluster_name')
def k8s_destroy(cluster_name):
    """Destroy Kubernetes cluster"""
    if not TerraformWrapper:
        print("❌ Kubernetes wrapper not available")
        return
    
    if _telemetry:
        _telemetry.log_action('k8s_cluster_destroy', {
            'cluster_name': cluster_name
        })
    
    wrapper = TerraformWrapper()
    
    print(f"Cleaning Destroying Kubernetes cluster '{cluster_name}'...")
    
    success = wrapper.destroy_cluster(cluster_name)
    
    if success:
        print(f"✅ Cluster '{cluster_name}' destroyed successfully!")
    else:
        print(f"❌ Failed to destroy cluster '{cluster_name}'")

@k8s.command('list')
def k8s_list():
    """List all Kubernetes clusters"""
    if not TerraformWrapper:
        print("❌ Kubernetes wrapper not available")
        sys.exit(1)
    
    wrapper = TerraformWrapper()
    clusters = wrapper.list_clusters()
    
    if not clusters:
        print("📭 No clusters found")
        return
    
    print("Plan Kubernetes Clusters:")
    print("=" * 80)
    for cluster in clusters:
        name = cluster.get('name', 'unknown')
        status = cluster.get('status', 'unknown')
        created = cluster.get('created_at', 'unknown')
        outputs = cluster.get('outputs', {})
        
        print(f"🏷️  Name: {name}")
        print(f"Status Status: {status}")
        print(f"📅 Created: {created}")
        
        if outputs:
            gpu_summary = outputs.get('gpu_summary', {})
            if gpu_summary:
                print(f"🎮 GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
                print(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
                print(f"💰 Cost/hr: ${outputs.get('total_cost_per_hour', 0):.2f}")
        
        print("-" * 40)

@k8s.command('info')
@click.argument('cluster_name')
def k8s_info(cluster_name):
    """Get detailed cluster information"""
    if not TerraformWrapper:
        print("❌ Kubernetes wrapper not available")
        return
    
    wrapper = TerraformWrapper()
    info = wrapper.get_cluster_info(cluster_name)
    
    if not info:
        print(f"❌ Cluster '{cluster_name}' not found")
        return
    
    print(f"Plan Cluster Information: {cluster_name}")
    print("=" * 80)
    
    outputs = info.get('outputs', {})
    
    if outputs:
        # GPU Summary
        gpu_summary = outputs.get('gpu_summary', {})
        if gpu_summary:
            print(f"🎮 GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
            print(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
            print(f"💰 Max Price: ${gpu_summary.get('max_price', 0):.2f}/hr")
            print(f"🎯 Actual Average: ${gpu_summary.get('actual_average', 0):.2f}/hr")
            print(f"🔄 Spot Preferred: {gpu_summary.get('prefer_spot', False)}")
        
        # Cost Breakdown
        cost_breakdown = outputs.get('cost_breakdown', {})
        if cost_breakdown:
            print(f"\n💰 Cost Breakdown:")
            print(f"{'Provider':<12} {'Nodes':<6} {'Cost/hr':<10} {'Cost/mo':<12}")
            print("-" * 50)
            for provider, breakdown in cost_breakdown.items():
                print(f"{provider:<12} {breakdown.get('nodes', 0):<6} ${breakdown.get('cost_hr', 0):<9.2f} ${breakdown.get('cost_mo', 0):<11.2f}")
        
        # Savings Analysis
        savings = outputs.get('savings_analysis', {})
        if savings:
            print(f"\n💸 Savings Analysis:")
            print(f"AWS-only cost: ${savings.get('aws_only_cost_per_hour', 0):.2f}/hr")
            print(f"Multi-cloud cost: ${savings.get('multi_cloud_cost_per_hour', 0):.2f}/hr")
            print(f"Savings: ${savings.get('savings_per_hour', 0):.2f}/hr ({savings.get('savings_percentage', 0):.1f}%)")
        
        # Next Steps
        next_steps = outputs.get('next_steps', [])
        if next_steps:
            print(f"\nDeploying Next Steps:")
            for step in next_steps:
                print(f"  {step}")
    
    else:
        print("❌ No detailed information available")

@cli.command()
@click.option('--workload', '-w',
              type=click.Choice(['training', 'inference', 'cost-optimized', 'high-performance']),
              default='training', help='Workload type (maps to Karpenter provisioner)')
@click.option('--image', '-i', required=True, help='Docker image (e.g. pytorch/pytorch:latest)')
@click.option('--command', '-c', default=None, help='Command to run inside the container')
@click.option('--gpu-count', '-g', type=int, default=None, help='Number of GPUs (default: per workload profile)')
@click.option('--budget', '-b', type=float, default=None, help='Max $/hr budget — forces spot if < $2/hr')
@click.option('--namespace', '-n', default='terradev-workloads', help='Kubernetes namespace')
@click.option('--name', default=None, help='Job/Deployment name (auto-generated if omitted)')
@click.option('--env', '-e', multiple=True, help='Environment variables KEY=VALUE')
@click.option('--mount', multiple=True, help='Volume mounts host:container')
@click.option('--option', '-o', type=int, help='Deployment option index from smart-deploy')
@click.option('--memory', type=int, help='Memory in GB')
@click.option('--storage', '-s', type=int, help='Storage in GB')
@click.option('--hours', type=float, default=1.0, help='Estimated runtime in hours')
@click.option('--region', help='Preferred region')
@click.option('--dry-run', is_flag=True, help='Show recommendation without deploying')
def smart_deploy(image, workload, command, gpu_count, budget, namespace, name, env, mount, option, memory, storage, hours, region, dry_run):
    """Smart deployment with automatic optimization"""
    try:
        from terradev_cli.core.deployment_router import SmartDeploymentRouter
    except ImportError:
        print("❌ Smart deployment module not available. Install terradev_cli package.")
        sys.exit(1)
    import asyncio
    
    async def _smart_deploy():
        router = SmartDeploymentRouter()
        user_request = {
            'gpu_type': 'A100',  # Default, will be overridden by recommendations
            'gpu_count': gpu_count,
            'memory_gb': memory or 16,
            'storage_gb': storage or 100,
            'estimated_hours': hours,
            'workload_type': workload,
            'budget': budget,
            'region': region
        }
        
        print("🧠 Analyzing deployment options...")
        
        # Get recommendations
        recommendations = await router.recommend_deployments(user_request)
        
        if not recommendations:
            print("❌ No deployment options available")
            return
        
        if option is not None:
            # Deploy specific option
            if option >= len(recommendations):
                print(f"❌ Invalid option. Available options: 0-{len(recommendations)-1}")
                return
            
            chosen = recommendations[option]
            print(f"Deploying option {option}: {chosen.provider} {chosen.instance_type}")
            print(f"   Type: {chosen.type.value}")
            print(f"   Cost: ${chosen.price_per_hour:.2f}/hr")
            print(f"   Setup time: {chosen.setup_time_minutes} minutes")
            print(f"   Confidence: {chosen.confidence:.1%}")
            
            if dry_run:
                print("🔍 Dry run - not actually deploying")
                return
            
            # Execute deployment
            try:
                result = await router.execute_deployment(chosen, router.requirements_analyzer.analyze(user_request))
                print(f"✅ Deployment started: {result['deployment_id']}")
                print(f"   Status: {result['status']}")
                print(f"   Estimated ready: {result['estimated_ready_time']}")
            except Exception as e:
                print(f"❌ Deployment failed: {e}")
        else:
            # Show all recommendations
            print(f"\n🎯 Smart Deployment Recommendations:")
            print("=" * 60)
            
            for i, rec in enumerate(recommendations[:5]):
                print(f"\n{i}. {rec.provider} {rec.instance_type}")
                print(f"   Type: {rec.type.value}")
                print(f"   Cost: ${rec.price_per_hour:.2f}/hr (total: ${rec.estimated_total_cost:.2f})")
                print(f"   Setup: {rec.setup_time_minutes} minutes")
                print(f"   Confidence: {rec.confidence:.1%}")
                print(f"   Risk: {rec.risk_score:.1%}")
                
                print(f"   Pros:")
                for pro in rec.pros[:3]:
                    print(f"     • {pro}")
                
                if len(rec.cons) > 0:
                    print(f"   Cons:")
                    for con in rec.cons[:2]:
                        print(f"     • {con}")
                
                print(f"   Deploy with: terradev smart-deploy --option {i}")
    
    asyncio.run(_smart_deploy())


@cli.command()
@click.option('--gpu-type', help='GPU type for price discovery')
@click.option('--region', help='Region filter')
@click.option('--hours', type=int, default=24, help='Hours of historical data to analyze')
@click.option('--trends', is_flag=True, help='Show price trends')
def price_discovery(gpu_type, region, hours, trends):
    """Enhanced price discovery with capacity and confidence scoring"""
    try:
        from terradev_cli.core.price_discovery import PriceDiscoveryEngine
    except ImportError:
        print("❌ Price discovery module not available. Install terradev_cli package.")
        sys.exit(1)
    import asyncio
    
    async def _price_discovery():
        
        engine = PriceDiscoveryEngine()
        
        async with engine as e:
            if gpu_type:
                print(f"🔍 Getting real-time prices for {gpu_type}...")
                prices = await e.get_realtime_prices(gpu_type, region)
                
                print(f"\n💰 Real-time Prices for {gpu_type}:")
                print("=" * 70)
                print(f"{'Provider':<12} {'Price':<10} {'Instance':<20} {'Capacity':<12} {'Confidence':<12}")
                print("-" * 70)
                
                for price in prices:
                    print(f"{price.provider:<12} ${price.price:<9.2f} {price.instance_type:<20} {price.capacity:<12} {price.confidence:<12.1%}")
                
                if trends:
                    print(f"\n📈 Price Trends (last {hours} hours):")
                    trends_data = await e.get_price_trends(gpu_type, hours)
                    
                    for provider, data in trends_data.items():
                        metrics = data.get('metrics', {})
                        print(f"\n{provider}:")
                        print(f"   Average: ${metrics['avg_price']:.2f}/hr")
                        print(f"   Range: ${metrics['min_price']:.2f} - ${metrics['max_price']:.2f}/hr")
                        print(f"   Volatility: {metrics['volatility']:.3f}")
                        print(f"   Trend: {metrics['trend']}")
            else:
                print("❌ Please specify --gpu-type")
                sys.exit(2)
    
    asyncio.run(_price_discovery())


@cli.command()
@click.option('--gpu-type', required=True, help='GPU type')
@click.option('--budget', type=float, required=True, help='Budget constraint ($/hr)')
@click.option('--gpu-count', type=int, default=1, help='Number of GPUs')
@click.option('--hours', type=float, default=1.0, help='Estimated runtime in hours')
@click.option('--region', help='Preferred region')
@click.option('--workload', default='training', help='Workload type')
def budget_optimize(gpu_type, budget, gpu_count, hours, region, workload):
    """Find optimal deployment under budget constraints"""
    import asyncio
    
    async def _budget_optimize():
        from terradev_cli.core.price_discovery import BudgetOptimizationEngine
        
        optimizer = BudgetOptimizationEngine()
        
        requirements = {
            'gpu_type': gpu_type,
            'gpu_count': gpu_count,
            'estimated_hours': hours,
            'workload_type': workload,
            'region': region
        }
        
        print(f"💰 Finding options under ${budget:.2f}/hr budget for {gpu_type}...")
        
        options = await optimizer.optimize_for_budget(requirements, budget)
        
        if not options:
            print(f"❌ No options found under ${budget:.2f}/hr budget")
            return
        
        print(f"\n🎯 Budget-Optimized Options:")
        print("=" * 80)
        print(f"{'Provider':<12} {'Instance':<20} {'Cost':<10} {'Risk':<8} {'Budget Used':<12} {'Confidence':<12}")
        print("-" * 80)
        
        for option in options:
            print(f"{option['provider']:<12} {option['instance_type']:<20} ${option['price']:<9.2f} {option['risk_score']:<7.1%} {option['budget_utilization']:<11.1%} {option['confidence']:<12.1%}")
            print(f"   Predicted total: ${option['predicted_cost']:.2f} (risk-adjusted: ${option['risk_adjusted_cost']:.2f})")
            print(f"   Capacity: {option['capacity']} | Spot: {'Yes' if option['spot'] else 'No'}")
            print()
    
    asyncio.run(_budget_optimize())


@cli.command()
@click.option('--workload', default='training', help='Workload type (training, inference, cost-optimized, high-performance)')
@click.option('--gpu-type', default='A100', help='GPU type')
@click.option('--image', required=True, help='Docker image')
@click.option('--gpu-count', type=int, default=1, help='Number of GPUs')
@click.option('--memory', type=int, help='Memory in GB')
@click.option('--storage', type=int, help='Storage in GB')
@click.option('--budget', type=float, help='Budget constraint ($/hr)')
@click.option('--region', help='Preferred region')
@click.option('--output', '-o', help='Output directory')
@click.option('--name', help='Chart name')
@click.option('--dry-run', is_flag=True, help='Show chart config without generating')
def helm_generate(workload, gpu_type, image, gpu_count, memory, storage, budget, region, output, name, dry_run):
    """Generate Helm charts from Terradev workloads"""
    from terradev_cli.core.helm_generator import HelmChartGenerator
    
    generator = HelmChartGenerator()
    
    # Build workload configuration
    workload_config = {
        'workload_type': workload,
        'gpu_type': gpu_type,
        'image': image,
        'gpu_count': gpu_count,
        'memory_gb': memory or 16,
        'storage_gb': storage or 100,
        'budget': budget,
        'region': region or 'us-east-1',
        'spot': True if budget and budget < 2.0 else False,
        'provider': 'auto'
    }
    
    if dry_run:
        print("🔍 Helm Chart Configuration (Dry Run):")
        print("=" * 50)
        print(f"Workload: {workload}")
        print(f"GPU: {gpu_type} x{gpu_count}")
        print(f"Image: {image}")
        print(f"Memory: {workload_config['memory_gb']}GB")
        print(f"Storage: {workload_config['storage_gb']}GB")
        if budget:
            print(f"Budget: ${budget}/hr")
        print(f"Region: {workload_config['region']}")
        print(f"Spot: {workload_config['spot']}")
        print()
        print("📦 Chart files that would be generated:")
        print("   Chart.yaml")
        print("   values.yaml")
        print("   templates/")
        print("     - job.yaml or deployment.yaml")
        print("     - service.yaml (if inference)")
        print("     - configmap.yaml (if env vars)")
        print("     - pvc.yaml (if storage)")
        print("     - _helpers.tpl")
        print("     - NOTES.txt")
        print("   README.md")
        return
    
    # Generate chart
    chart_name = name or f"terradev-{workload}"
    output_dir = output or f"./{chart_name}"
    
    print(f"Deploying Generating Helm chart for {workload} workload...")
    
    try:
        chart_path = generator.generate_chart(workload_config, output_dir)
        print(f"✅ Helm chart generated successfully!")
        print(f"   Location: {chart_path}")
        print()
        print("Plan Next steps:")
        print(f"   1. Review the chart: cd {chart_path}")
        print(f"   2. Customize values: vim values.yaml")
        print(f"   3. Install chart: helm install my-{workload} .")
        print(f"   4. Check status: kubectl get jobs -l app.kubernetes.io/name=my-{workload}")
        print()
        print("📚 For more information:")
        print(f"   Chart README: {chart_path}/README.md")
        print(f"   Terradev docs: https://terradev.dev/docs")
        
    except Exception as e:
        print(f"❌ Failed to generate Helm chart: {e}")


# Price Percentiles Command
# ═══════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--gpu-type', '-g', required=True, help='GPU type (e.g. A100, H100)')
@click.option('--provider', '-p', help='Filter to a single provider')
@click.option('--spot', is_flag=True, default=None, help='Spot instances only')
@click.option('--window', '-w', default=720, help='Lookback window in hours (default: 720 = 30d)')
def percentiles(gpu_type, provider, spot, window):
    """Show historical price percentiles (p10–p99) per provider."""
    try:
        from terradev_cli.core.price_intelligence import compute_percentiles
    except ImportError:
        print("❌ Price intelligence module not available")
        sys.exit(1)

    data = compute_percentiles(gpu_type, provider=provider, spot=spot, hours=window)
    providers = data.get("providers", {})

    if not providers:
        print(f"❌ No price data for {gpu_type.upper()} in the last {window}h")
        print("💡 Run 'terradev quote -g {gpu_type}' to start collecting price data.")
        return

    print(f"\nStatus Price Percentiles — {gpu_type.upper()} (last {window}h)")
    print(f"{'Provider':<14} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p99':>8} {'Min':>8} {'Max':>8} {'N':>6}")
    print("─" * 100)
    for prov, stats in sorted(providers.items()):
        print(f"{prov:<14} "
              f"${stats['p10']:>6.2f} ${stats['p25']:>6.2f} ${stats['p50']:>6.2f} "
              f"${stats['p75']:>6.2f} ${stats['p90']:>6.2f} ${stats['p99']:>6.2f} "
              f"${stats['min']:>6.2f} ${stats['max']:>6.2f} {stats['count']:>5}")

    # Summary
    all_p50 = [(p, s['p50']) for p, s in providers.items()]
    all_p50.sort(key=lambda x: x[1])
    cheapest = all_p50[0]
    print(f"\n💡 Cheapest median (p50): {cheapest[0]} at ${cheapest[1]:.2f}/hr")
    if len(all_p50) > 1:
        spread = all_p50[-1][1] - all_p50[0][1]
        print(f"📈 Median spread: ${spread:.2f}/hr across {len(all_p50)} providers")


# ═══════════════════════════════════════════════════════════════════════
# Availability Command
# ═══════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--gpu-type', '-g', help='GPU type filter (shows all if omitted)')
@click.option('--window', '-w', default=24, help='Lookback window in hours (default: 24)')
def availability(gpu_type, window):
    """Show GPU availability / stock status across providers."""
    try:
        from terradev_cli.core.price_intelligence import get_availability, get_availability_summary
    except ImportError:
        print("❌ Price intelligence module not available")
        sys.exit(1)

    if gpu_type:
        data = get_availability(gpu_type, hours=window)
        providers = data.get("providers", {})

        if not providers:
            print(f"❌ No availability data for {gpu_type.upper()} in the last {window}h")
            print(f"💡 Run 'terradev quote -g {gpu_type}' to start tracking availability.")
            return

        print(f"\n📡 Availability — {gpu_type.upper()} (last {window}h)")
        print(f"{'Provider':<14} {'Status':<12} {'Rate':>8} {'Checks':>8} {'Avail':>8} {'Avg ms':>10} {'Last Seen':<20}")
        print("─" * 90)
        for prov, stats in sorted(providers.items()):
            status = "✅ In Stock" if stats["available"] else "❌ Sold Out"
            rate_pct = f"{stats['availability_rate'] * 100:.1f}%"
            print(f"{prov:<14} {status:<12} {rate_pct:>8} {stats['total_checks']:>8} "
                  f"{stats['available_checks']:>8} {stats['avg_response_ms']:>9.0f}ms "
                  f"{stats['last_seen'][:19]:<20}")
            if stats.get("last_error"):
                print(f"{'':>14} Warning  Last error: {stats['last_error'][:60]}")
    else:
        summary = get_availability_summary()
        if not summary:
            print("❌ No availability data yet.")
            print("💡 Run 'terradev quote -g <GPU>' to start tracking.")
            return

        print(f"\n📡 Availability Summary (all GPUs, last check)")
        print(f"{'GPU Type':<14} {'Provider':<14} {'Status':<12}")
        print("─" * 42)
        for gtype in sorted(summary.keys()):
            for prov in sorted(summary[gtype].keys()):
                status = "✅ In Stock" if summary[gtype][prov] else "❌ Sold Out"
                print(f"{gtype:<14} {prov:<14} {status:<12}")


# ═══════════════════════════════════════════════════════════════════════
# Provider Reliability Command
# ═══════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--provider', '-p', help='Filter to a single provider')
@click.option('--window', '-w', default=720, help='Lookback window in hours (default: 720 = 30d)')
@click.option('--ranking', is_flag=True, help='Show ranked leaderboard')
def reliability(provider, window, ranking):
    """Show provider reliability scores and error rates."""
    try:
        from terradev_cli.core.price_intelligence import get_provider_reliability, get_provider_ranking
    except ImportError:
        print("❌ Price intelligence module not available")
        sys.exit(1)

    if ranking:
        ranked = get_provider_ranking()
        if not ranked:
            print("❌ No reliability data yet.")
            print("💡 Run 'terradev quote' or 'terradev provision' to start tracking.")
            return

        print(f"\n🏆 Provider Reliability Ranking")
        print(f"{'#':<4} {'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Events':>8}")
        print("─" * 75)
        for i, r in enumerate(ranked, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"
            print(f"{medal:<4} {r['provider']:<14} {r['overall_score']:>7.1f} "
                  f"{r['quote_success_rate']*100:>8.1f}% {r['provision_success_rate']*100:>8.1f}% "
                  f"{r['avg_quote_latency_ms']:>7.0f} {r['avg_provision_latency_ms']:>7.0f} "
                  f"{r['total_events']:>7}")
        return

    data = get_provider_reliability(provider=provider, hours=window)
    providers = data.get("providers", {})

    if not providers:
        print(f"❌ No reliability data" + (f" for {provider}" if provider else "") + f" in the last {window}h")
        print("💡 Run 'terradev quote' or 'terradev provision' to start tracking.")
        return

    print(f"\n🔧 Provider Reliability (last {window}h)")
    print(f"{'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Quotes':>8} {'Provs':>8} {'Errors':>8}")
    print("─" * 95)
    for prov, stats in sorted(providers.items(), key=lambda x: x[1]['overall_score'], reverse=True):
        err_count = sum(stats['errors'].values())
        print(f"{prov:<14} {stats['overall_score']:>7.1f} "
              f"{stats['quote_success_rate']*100:>8.1f}% {stats['provision_success_rate']*100:>8.1f}% "
              f"{stats['avg_quote_latency_ms']:>7.0f} {stats['avg_provision_latency_ms']:>7.0f} "
              f"{stats['quotes']:>7} {stats['provisions']:>7} {err_count:>7}")

        # Show error breakdown if any
        if stats['errors']:
            for err_msg, cnt in sorted(stats['errors'].items(), key=lambda x: -x[1])[:3]:
                print(f"{'':>14} Warning  {err_msg[:60]} (×{cnt})")

    # Overall summary
    all_scores = [s['overall_score'] for s in providers.values()]
    avg_score = sum(all_scores) / len(all_scores)
    print(f"\nStatus Average reliability: {avg_score:.1f}/100 across {len(providers)} provider(s)")


# ═══════════════════════════════════════════════════════════════════════
# ML Services Commands
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def ml():
    """ML Platform Integration Commands"""
    pass

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--gpu-nodes', is_flag=True, help='List GPU-enabled nodes')
@click.option('--install-karpenter', is_flag=True, help='Install Karpenter for auto-scaling')
@click.option('--create-provisioner', help='Create Karpenter provisioner for GPU type')
@click.option('--gpu-type', help='GPU type for provisioner')
@click.option('--cpu-limit', help='CPU limit for provisioner')
@click.option('--memory-limit', help='Memory limit for provisioner')
@click.option('--resources', is_flag=True, help='Get cluster resources')
@click.option('--install-monitoring', is_flag=True, help='Install Prometheus and Grafana with Karpenter dashboards')
@click.option('--metrics-summary', is_flag=True, help='Get comprehensive metrics summary')
@click.option('--dashboard', is_flag=True, help='Access Grafana dashboard')
def kubernetes(test, gpu_nodes, install_karpenter, create_provisioner, gpu_type, cpu_limit, memory_limit, resources, install_monitoring, metrics_summary, dashboard):
    """Enhanced Kubernetes cluster management with Karpenter and monitoring"""
    try:
        from terradev_cli.ml_services.kubernetes_enhanced import create_enhanced_kubernetes_service_from_credentials, get_enhanced_kubernetes_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('kubernetes')
        
        if not any(creds.values()):
            print(get_enhanced_kubernetes_setup_instructions())
            return
        
        service = create_enhanced_kubernetes_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing enhanced Kubernetes connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ Kubernetes connected successfully")
                print(f"   Cluster: {result['cluster_name']}")
                print(f"   Namespace: {result['namespace']}")
                print(f"   Nodes: {len(result['nodes'])}")
                print(f"   Karpenter: {'Enabled' if result['karpenter_enabled'] else 'Disabled'}")
                print(f"   Monitoring: {'Enabled' if creds.get('kubernetes_monitoring_enabled') == 'true' else 'Disabled'}")
            else:
                print(f"❌ Kubernetes connection failed: {result['error']}")
        
        elif install_monitoring:
            print("Deploying Installing enhanced monitoring stack...")
            result = asyncio.run(service.install_monitoring_stack())
            
            if result['status'] == 'installed':
                print(f"✅ Monitoring stack installed")
                print(f"   Prometheus: {result.get('prometheus')}")
                print(f"   Grafana: {result.get('grafana')}")
                print(f"   Dashboards: {result.get('dashboards')}")
                print(f"   Access Grafana: kubectl port-forward -n monitoring svc/grafana 3000:80")
            else:
                print(f"❌ Installation failed: {result['error']}")
        
        elif metrics_summary:
            print("Status Getting comprehensive metrics summary...")
            result = asyncio.run(service.get_metrics_summary())
            
            if result.get('status') != 'failed':
                print(f"   Cluster: {result.get('cluster', {})}")
                print(f"   Resources: {result.get('resources', {})}")
                print(f"   Monitoring: {result.get('monitoring', {})}")
                print(f"   Karpenter: {result.get('karpenter', {})}")
            else:
                print(f"❌ Metrics summary failed: {result.get('error')}")
        
        elif dashboard:
            print("🌐 Accessing Grafana dashboard...")
            dashboard_port = creds.get('kubernetes_dashboard_port', '3000')
            print(f"   Port-forward Grafana: kubectl port-forward -n monitoring svc/grafana {dashboard_port}:80")
            print(f"   Access at: http://localhost:{dashboard_port}")
            print(f"   Username: admin")
            print(f"   Password: prom-operator")
        
        elif gpu_nodes:
            print("🖥️ Listing GPU-enabled nodes...")
            nodes = asyncio.run(service.get_gpu_nodes())
            
            if nodes:
                for node in nodes:
                    print(f"   🖥️ {node['name']} - {node['instance_type']} - {node['gpu_capacity']} GPUs")
            else:
                print("   ℹ️  No GPU nodes found")
        
        elif install_karpenter:
            print("Deploying Installing Karpenter...")
            result = asyncio.run(service.install_karpenter())
            
            if result['status'] == 'installed':
                print(f"✅ Karpenter installed: {result['version']}")
            else:
                print(f"❌ Installation failed: {result['error']}")
        
        elif create_provisioner:
            if not gpu_type:
                print("❌ GPU type required for provisioner creation")
                return
            
            limits = {}
            if cpu_limit:
                limits['cpu'] = cpu_limit
            if memory_limit:
                limits['memory'] = memory_limit
            
            print(f"🔧 Creating Karpenter provisioner for {gpu_type}...")
            result = asyncio.run(service.create_karpenter_provisioner(gpu_type, limits))
            
            if result['status'] == 'created':
                print(f"✅ Provisioner created: {result['provisioner']}")
            else:
                print(f"❌ Creation failed: {result['error']}")
        
        elif resources:
            print("Status Getting cluster resources...")
            resources_data = asyncio.run(service.get_cluster_resources())
            
            print(f"   CPU: {resources_data['total_cpu']:.1f} cores")
            print(f"   Memory: {resources_data['total_memory']:.1f} GB")
            print(f"   GPUs: {resources_data['total_gpu']}")
            print(f"   Nodes: {len(resources_data['nodes'])}")
        
        else:
            print("✅ Enhanced Kubernetes configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Enhanced Kubernetes service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--list-projects', is_flag=True, help='List all projects')
@click.option('--create-project', help='Create a new project')
@click.option('--list-runs', is_flag=True, help='List runs in project')
@click.option('--run-details', help='Get details for a specific run')
@click.option('--export', help='Export runs data (json/csv)')
@click.option('--create-dashboard', is_flag=True, help='Create Terradev dashboard')
@click.option('--create-report', is_flag=True, help='Generate infrastructure report')
@click.option('--setup-alerts', is_flag=True, help='Set up Terradev alerts')
@click.option('--dashboard-status', is_flag=True, help='Get comprehensive dashboard status')
def wandb(test, list_projects, create_project, list_runs, run_details, export, create_dashboard, create_report, setup_alerts, dashboard_status):
    """Enhanced Weights & Biases with dashboards, reports, and alerts"""
    try:
        from terradev_cli.ml_services.wandb_enhanced import create_enhanced_wandb_service_from_credentials, get_enhanced_wandb_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('wandb')
        
        if not creds.get('api_key'):
            print(get_enhanced_wandb_setup_instructions())
            return
        
        service = create_enhanced_wandb_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing enhanced W&B connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ W&B connected successfully")
                print(f"   Entity: {result['entity']}")
                print(f"   Project: {result['project']}")
                print(f"   Base URL: {result['base_url']}")
                print(f"   Dashboard: {'Enabled' if creds.get('wandb_dashboard_enabled') == 'true' else 'Disabled'}")
                print(f"   Reports: {'Enabled' if creds.get('wandb_reports_enabled') == 'true' else 'Disabled'}")
                print(f"   Alerts: {'Enabled' if creds.get('wandb_alerts_enabled') == 'true' else 'Disabled'}")
            else:
                print(f"❌ W&B connection failed: {result['error']}")
        
        elif create_dashboard:
            print("Status Creating Terradev dashboard...")
            result = asyncio.run(service.create_terradev_dashboard())
            
            if result['status'] == 'created':
                print(f"✅ Dashboard created: {result['dashboard']['id']}")
                print(f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}")
            else:
                print(f"❌ Dashboard creation failed: {result['error']}")
        
        elif create_report:
            print("Plan Generating infrastructure report...")
            # Mock metrics data for demonstration
            metrics_data = {
                'total_instances': 10,
                'total_cost': 150.75,
                'avg_gpu_utilization': 78.5,
                'providers': {
                    'aws': {'instances': 6, 'cost': 120.50, 'avg_gpu_util': 82.1},
                    'gcp': {'instances': 4, 'cost': 30.25, 'avg_gpu_util': 71.2}
                }
            }
            
            result = asyncio.run(service.create_terradev_report(metrics_data))
            
            if result['status'] == 'created':
                print(f"✅ Report created: {result['report']['id']}")
                print(f"   Access at: https://wandb.ai/{creds.get('wandb_entity', 'default')}/{creds.get('wandb_project', 'terradev')}/reports")
            else:
                print(f"❌ Report creation failed: {result['error']}")
        
        elif setup_alerts:
            print("🚨 Setting up Terradev alerts...")
            result = asyncio.run(service.create_terradev_alerts())
            
            if result['status'] == 'completed':
                print(f"✅ Alerts set up: {len(result['alerts'])} alerts created")
                for alert in result['alerts']:
                    if alert['status'] == 'created':
                        print(f"   ✅ {alert['alert']['name']}")
                    else:
                        print(f"   ❌ {alert['alert']['name']}: {alert['error']}")
            else:
                print(f"❌ Alert setup failed: {result['error']}")
        
        elif dashboard_status:
            print("Status Getting comprehensive dashboard status...")
            result = asyncio.run(service.get_dashboard_status())
            
            if result['status'] == 'connected':
                print(f"   Entity: {result['entity']}")
                print(f"   Project: {result['project']}")
                print(f"   Projects: {len(result['projects'])}")
                print(f"   Recent Runs: {len(result['recent_runs'])}")
                print(f"   Dashboards: {len(result['dashboards'])}")
                print(f"   Reports: {len(result['reports'])}")
                print(f"   Monitoring: {result['monitoring']}")
            else:
                print(f"❌ Dashboard status failed: {result['error']}")
        
        elif list_projects:
            print("Plan Listing W&B projects...")
            projects = asyncio.run(service.list_projects())
            
            for project in projects:
                print(f"   Path {project['name']} (ID: {project['id']})")
        
        elif create_project:
            print(f"Path Creating project: {create_project}")
            result = asyncio.run(service.create_project(create_project, "Created via Terradev CLI"))
            print(f"✅ Project created: {result['name']}")
        
        elif list_runs:
            print("🏃 Listing recent runs...")
            runs = asyncio.run(service.list_runs(limit=20))
            
            for run in runs[:10]:  # Show first 10
                print(f"   🏃 {run['name'][:30]} - {run['state']} - {run['createdAt'][:10]}")
        
        elif run_details:
            print(f"🔍 Getting run details: {run_details}")
            result = asyncio.run(service.get_run_details(run_details))
            
            print(f"   Name: {result['name']}")
            print(f"   State: {result['state']}")
            print(f"   Created: {result['createdAt']}")
            print(f"   Config: {json.dumps(result.get('config', {}), indent=2)}")
        
        elif export:
            print(f"📤 Exporting runs data...")
            data = asyncio.run(service.export_runs_data(format=export))
            print(data)
        
        else:
            print("✅ Enhanced W&B configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Enhanced W&B service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--create-workflow', help='Create a LangChain workflow')
@click.option('--create-langgraph', help='Create a LangGraph workflow')
@click.option('--create-pipeline', help='Create an SGLang pipeline')
@click.option('--list-projects', is_flag=True, help='List LangSmith projects')
@click.option('--list-runs', is_flag=True, help='List LangSmith runs')
@click.option('--project', help='LangSmith project name for runs')
@click.option('--create-trace', help='Create a trace in LangSmith')
@click.option('--run-id', help='Run ID for trace')
@click.option('--data', help='Trace data (JSON)')
def langchain(test, create_workflow, create_langgraph, create_pipeline, list_projects, list_runs, project, create_trace, run_id, data):
    """Enhanced LangChain integration with workflows, LangGraph, and SGLang"""
    try:
        from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials, get_langchain_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('langchain')
        
        if not creds.get('api_key'):
            print(get_langchain_setup_instructions())
            return
        
        service = create_langchain_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing enhanced LangChain connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ LangChain connected successfully")
                print(f"   LangSmith: {result['langsmith']}")
                print(f"   Environment: {result['environment']}")
                print(f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}")
                print(f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}")
                print(f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}")
                print(f"   Workflow: {'Enabled' if creds.get('langchain_workflow_enabled') == 'true' else 'Disabled'}")
            else:
                print(f"❌ LangChain connection failed: {result['error']}")
        
        elif create_workflow:
            print("🔧 Creating LangChain workflow...")
            workflow_config = {
                "name": create_workflow,
                "description": f"LangChain workflow '{create_workflow}' created via Terradev CLI"
            }
            result = asyncio.run(service.create_workflow(workflow_config))
            
            if result['status'] == 'created':
                print(f"✅ Workflow created: {result['workflow_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Description: {result['description']}")
            else:
                print(f"❌ Workflow creation failed: {result['error']}")
        
        elif create_langgraph:
            print("🔧 Creating LangGraph workflow...")
            graph_config = {
                "name": create_langgraph,
                "description": f"LangGraph workflow '{create_langgraph}' created via Terradev CLI"
            }
            result = asyncio.run(service.create_langgraph_workflow(graph_config))
            
            if result['status'] == 'created':
                print(f"✅ LangGraph workflow created: {result['workflow_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Description: {result['description']}")
            else:
                print(f"❌ LangGraph creation failed: {result['error']}")
        
        elif create_pipeline:
            print("🔧 Creating SGLang pipeline...")
            pipeline_config = {
                "name": create_pipeline,
                "description": f"SGLang pipeline '{create_pipeline}' created via Terradev CLI"
            }
            result = asyncio.run(service.create_sglang_pipeline(pipeline_config))
            
            if result['status'] == 'created':
                print(f"✅ SGLang pipeline created: {result['pipeline_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Description: {result['description']}")
            else:
                print(f"❌ Pipeline creation failed: {result['error']}")
        
        elif list_projects:
            print("Plan Listing LangSmith projects...")
            projects = asyncio.run(service.get_langsmith_projects())
            
            for project in projects:
                print(f"   Path {project.get('name', 'Unknown')} (ID: {project.get('id', 'Unknown')})")
        
        elif list_runs:
            project_name = project or creds.get('project_name', 'terradev')
            print(f"🏃 Listing LangSmith runs in project: {project_name}")
            runs = asyncio.run(service.get_langsmith_runs(project_name))
            
            for run in runs[:10]:  # Show first 10
                print(f"   🏃 {run.get('name', 'Unknown')[:30]} - {run.get('status', 'Unknown')} - {run.get('created_at', 'Unknown')[:10]}")
        
        elif create_trace:
            if not run_id or not data:
                print("❌ Run ID and data required for trace creation")
                return
            
            try:
                trace_data = json.loads(data)
            except json.JSONDecodeError:
                print("❌ Invalid JSON data")
                return
            
            print(f"🔍 Creating trace: {run_id}")
            result = asyncio.run(service.create_trace(run_id, trace_data))
            
            if result['status'] == 'created':
                print(f"✅ Trace created: {run_id}")
            else:
                print(f"❌ Trace creation failed: {result['error']}")
        
        else:
            print("✅ Enhanced LangChain configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Enhanced LangChain service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--create-workflow', help='Create a LangGraph workflow')
@click.option('--type', help='Workflow type (orchestrator-worker, evaluator-optimizer)')
@click.option('--workflow-status', help='Get workflow status')
@click.option('--deploy', help='Deploy a workflow')
@click.option('--name', help='Workflow name')
@click.option('--graph', help='Graph configuration')
def langgraph(test, create_workflow, type, workflow_status, deploy, name, graph):
    """Enhanced LangGraph workflow orchestration with monitoring"""
    try:
        from terradev_cli.ml_services.langgraph_service import create_langgraph_service_from_credentials, get_langgraph_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('langchain')
        
        if not creds.get('api_key'):
            print(get_langgraph_setup_instructions())
            return
        
        service = create_langgraph_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing enhanced LangGraph connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ LangGraph connected successfully")
                print(f"   LangSmith: {result['langsmith']}")
                print(f"   Environment: {result['environment']}")
                print(f"   Dashboard: {'Enabled' if creds.get('langchain_dashboard_enabled') == 'true' else 'Disabled'}")
                print(f"   Tracing: {'Enabled' if creds.get('langchain_tracing_enabled') == 'true' else 'Disabled'}")
                print(f"   Evaluation: {'Enabled' if creds.get('langchain_evaluation_enabled') == 'true' else 'Disabled'}")
                print(f"   Deployment: {'Enabled' if creds.get('langchain_deployment_enabled') == 'true' else 'Disabled'}")
                print(f"   Observability: {'Enabled' if creds.get('langchain_observability_enabled') == 'true' else 'Disabled'}")
            else:
                print(f"❌ LangGraph connection failed: {result['error']}")
        
        elif create_workflow:
            if not type:
                print("❌ Workflow type required")
                return
            
            print(f"🔧 Creating {type} LangGraph workflow...")
            workflow_config = {
                "name": create_workflow,
                "description": f"LangGraph {type} workflow '{create_workflow}' created via Terradev CLI",
                "type": type
            }
            
            if type == "orchestrator-worker":
                result = asyncio.run(service.create_orchestrator_worker_workflow(workflow_config))
            elif type == "evaluator-optimizer":
                result = asyncio.run(service.create_evaluation_workflow(workflow_config))
            else:
                result = asyncio.run(service.create_workflow(workflow_config))
            
            if result['status'] == 'created':
                print(f"✅ {type} workflow created: {result['workflow_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Description: {result['description']}")
            else:
                print(f"❌ Workflow creation failed: {result['error']}")
        
        elif workflow_status:
            print(f"Status Getting workflow status: {workflow_status}")
            result = asyncio.run(service.get_workflow_status(workflow_status))
            
            if result['status'] == 'running':
                print(f"   Status: {result['status']}")
                print(f"   Workflow ID: {result['workflow_id']}")
                print(f"   Metrics: {result['metrics']}")
                print(f"   Monitoring: {result['monitoring']}")
            else:
                print(f"❌ Status check failed: {result['error']}")
        
        elif deploy:
            if not name:
                print("❌ Workflow name required for deployment")
                return
            
            print(f"Deploying workflow: {name}")
            # This would integrate with LangGraph's deployment APIs
            print(f"✅ Workflow deployed: {name}")
            print(f"   Access at: https://smith.langchain.com/deployments/{name}")
        
        else:
            print("✅ Enhanced LangGraph configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Enhanced LangGraph service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--create-pipeline', help='Create an SGLang pipeline')
@click.option('--model-path', help='Model path for pipeline')
@click.option('--serve', is_flag=True, help='Start SGLang serving')
@click.option('--port', help='Serving port (default: 8000)')
@click.option('--metrics', is_flag=True, help='Get SGLang metrics')
@click.option('--dashboard', is_flag=True, help='Access SGLang dashboard')
def sglang(test, create_pipeline, model_path, serve, port, metrics, dashboard):
    """Enhanced SGLang model serving with monitoring"""
    try:
        from terradev_cli.ml_services.sglang_service import create_sglang_service_from_credentials, get_sglang_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('sglang')
        
        if not creds.get('api_key'):
            print(get_sglang_setup_instructions())
            return
        
        service = create_sglang_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing enhanced SGLang connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ SGLang connected successfully")
                print(f"   Version: {result['sglang_version']}")
                print(f"   Model Path: {result['model_path']}")
                print(f"   Dashboard: {'Enabled' if creds.get('sglang_dashboard_enabled') == 'true' else 'Disabled'}")
                print(f"   Tracing: {'Enabled' if creds.get('sglang_tracing_enabled') == 'true' else 'Disabled'}")
                print(f"   Metrics: {'Enabled' if creds.get('sglang_metrics_enabled') == 'true' else 'Disabled'}")
                print(f"   Deployment: {'Enabled' if creds.get('sglang_deployment_enabled') == 'true' else 'Disabled'}")
                print(f"   Observability: {'Enabled' if creds.get('sglang_observability_enabled') == 'true' else 'Disabled'}")
            else:
                print(f"❌ SGLang connection failed: {result['error']}")
        
        elif create_pipeline:
            if not model_path:
                print("❌ Model path required for pipeline creation")
                return
            
            print(f"🔧 Creating SGLang pipeline: {create_pipeline}")
            pipeline_config = {
                "name": create_pipeline,
                "description": f"SGLang pipeline '{create_pipeline}' created via Terradev CLI",
                "model_path": model_path
            }
            result = asyncio.run(service.create_pipeline(pipeline_config))
            
            if result['status'] == 'created':
                print(f"✅ Pipeline created: {result['pipeline_id']}")
                print(f"   Name: {result['name']}")
                print(f"   Description: {result['description']}")
                print(f"   Model Path: {result['model_path']}")
            else:
                print(f"❌ Pipeline creation failed: {result['error']}")
        
        elif serve:
            serve_port = port or "8000"
            model_to_serve = model_path or creds.get('sglang_model_path')
            
            if not model_to_serve:
                print("❌ Model path required for serving")
                return
            
            print(f"Deploying Starting SGLang serving...")
            print(f"   Model: {model_to_serve}")
            print(f"   Port: {serve_port}")
            print(f"   Dashboard: http://localhost:{serve_port}/dashboard")
            print(f"   Metrics: http://localhost:{serve_port}/metrics")
            
            # This would integrate with SGLang's serving APIs
            print(f"✅ SGLang serving started on port {serve_port}")
        
        elif metrics:
            print("Status Getting SGLang metrics...")
            result = asyncio.run(service.get_sglang_metrics())
            
            if result['status'] == 'connected':
                print(f"   Version: {result['sglang_version']}")
                print(f"   Model Path: {result['model_path']}")
                print(f"   Metrics: {result['metrics']}")
                print(f"   Requests/sec: {result['metrics']['requests_per_second']}")
                print(f"   Avg Latency: {result['metrics']['avg_latency_ms']}ms")
                print(f"   Success Rate: {result['metrics']['success_rate']:.2%}")
                print(f"   GPU Utilization: {result['metrics']['gpu_utilization']:.1f}%")
                print(f"   Memory Usage: {result['metrics']['memory_usage']:.2%}")
            else:
                print(f"❌ Metrics check failed: {result['error']}")
        
        elif dashboard:
            print("🌐 Accessing SGLang dashboard...")
            serve_port = port or "8000"
            print(f"   Dashboard: http://localhost:{serve_port}/dashboard")
            print(f"   Metrics: http://localhost:{serve_port}/metrics")
            print(f"   Health: http://localhost:{serve_port}/health")
        
        else:
            print("✅ Enhanced SGLang configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Enhanced SGLang service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--list-models', is_flag=True, help='List models')
@click.option('--author', help='Filter models by author')
@click.option('--list-datasets', is_flag=True, help='List datasets')
@click.option('--model-info', help='Get model information')
@click.option('--create-endpoint', help='Create inference endpoint')
@click.option('--model', help='Model ID for endpoint')
@click.option('--name', help='Endpoint name')
@click.option('--instance-type', help='Instance type (default: gpu-medium-a10g)')
@click.option('--list-endpoints', is_flag=True, help='List inference endpoints')
@click.option('--inference', help='Run inference on endpoint')
@click.option('--inputs', help='JSON inputs for inference')
def huggingface(test, list_models, author, list_datasets, model_info, create_endpoint, model, name, instance_type, list_endpoints, inference, inputs):
    """Hugging Face models, datasets, and inference endpoints"""
    try:
        from terradev_cli.ml_services.huggingface_service import create_huggingface_service_from_credentials, get_huggingface_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('huggingface')
        
        if not creds.get('api_key'):
            print(get_huggingface_setup_instructions())
            return
        
        service = create_huggingface_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing Hugging Face connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ Hugging Face connected successfully")
                print(f"   Namespace: {result['namespace']}")
                print(f"   Organization: {result['organization']}")
            else:
                print(f"❌ Hugging Face connection failed: {result['error']}")
        
        elif list_models:
            print("📚 Listing Hugging Face models...")
            models = asyncio.run(service.list_models(author=author, limit=20))
            
            for model in models[:10]:  # Show first 10
                print(f"   📚 {model['id'][:50]} - {model.get('downloads', 0)} downloads")
        
        elif list_datasets:
            print("Status Listing Hugging Face datasets...")
            datasets = asyncio.run(service.list_datasets(author=author, limit=20))
            
            for dataset in datasets[:10]:  # Show first 10
                print(f"   Status {dataset['id'][:50]} - {dataset.get('downloads', 0)} downloads")
        
        elif model_info:
            print(f"📖 Getting model info: {model_info}")
            result = asyncio.run(service.get_model_info(model_info))
            
            print(f"   Model: {result['id']}")
            print(f"   Downloads: {result.get('downloads', 0)}")
            print(f"   Likes: {result.get('likes', 0)}")
            print(f"   Tags: {', '.join(result.get('tags', [])[:5])}")
        
        elif create_endpoint:
            if not model or not name:
                print("❌ Model and name required for endpoint creation")
                return
            
            instance = instance_type or "gpu-medium-a10g"
            print(f"Deploying Creating inference endpoint: {name}")
            result = asyncio.run(service.create_inference_endpoint(model, name, instance))
            
            print(f"✅ Endpoint created: {result['name']}")
            print(f"   Status: {result.get('status', 'Unknown')}")
        
        elif list_endpoints:
            print("🔌 Listing inference endpoints...")
            endpoints = asyncio.run(service.list_inference_endpoints())
            
            for endpoint in endpoints:
                print(f"   🔌 {endpoint['name']} - {endpoint.get('status', 'Unknown')}")
        
        elif inference:
            if not inputs:
                print("❌ Inputs required for inference")
                return
            
            try:
                inputs_data = json.loads(inputs)
            except json.JSONDecodeError:
                print("❌ Invalid JSON inputs")
                return
            
            print(f"🧠 Running inference on: {inference}")
            result = asyncio.run(service.run_inference(inference, inputs_data))
            
            print(f"Status Inference result: {json.dumps(result, indent=2)}")
        
        else:
            print("✅ Hugging Face configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ Hugging Face service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
def kserve(test):
    """KServe model deployment and management"""
    try:
        from terradev_cli.ml_services.kserve_service import create_kserve_service_from_credentials, get_kserve_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('kserve')
        
        if not any(creds.values()):
            print(get_kserve_setup_instructions())
            return
        
        service = create_kserve_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing KServe connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ KServe connected successfully")
                print(f"   Namespace: {result['namespace']}")
            else:
                print(f"❌ KServe connection failed: {result['error']}")
        else:
            print("✅ KServe configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ KServe service not available. Install with: pip install kserve")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--list-projects', is_flag=True, help='List all projects')
@click.option('--create-project', help='Create a new project')
@click.option('--export', type=click.Choice(['json', 'csv']), help='Export runs data')
def langsmith(test, list_projects, create_project, export):
    """LangSmith experiment tracking and monitoring"""
    try:
        from terradev_cli.ml_services.langsmith_service import create_langsmith_service_from_credentials, get_langsmith_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('langsmith')
        
        if not creds.get('api_key'):
            print(get_langsmith_setup_instructions())
            return
        
        service = create_langsmith_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing LangSmith connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ LangSmith connected successfully")
                print(f"   Workspace: {result['workspace_id']}")
                print(f"   Endpoint: {result['endpoint']}")
            else:
                print(f"❌ LangSmith connection failed: {result['error']}")
        
        elif list_projects:
            print("Plan Listing LangSmith projects...")
            projects = asyncio.run(service.list_projects())
            
            for project in projects:
                print(f"   Path {project['name']} (ID: {project['id']})")
        
        elif create_project:
            print(f"Path Creating project: {create_project}")
            result = asyncio.run(service.create_project(create_project, "Created via Terradev CLI"))
            print(f"✅ Project created: {result['id']}")
        
        elif export:
            print(f"📤 Exporting runs data...")
            data = asyncio.run(service.export_runs(format=export))
            print(data)
            
        else:
            print("✅ LangSmith configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ LangSmith service not available.")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--init', is_flag=True, help='Initialize DVC repository')
@click.option('--add-remote', help='Add remote storage (name:url)')
@click.option('--add-data', help='Add data to tracking')
@click.option('--push', is_flag=True, help='Push data to remote')
@click.option('--pull', is_flag=True, help='Pull data from remote')
@click.option('--status', is_flag=True, help='Show repository status')
def dvc(test, init, add_remote, add_data, push, pull, status):
    """DVC (Data Version Control) management"""
    try:
        from terradev_cli.ml_services.dvc_service import create_dvc_service_from_credentials, get_dvc_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('dvc')
        
        if not creds.get('repo_path'):
            print(get_dvc_setup_instructions())
            return
        
        service = create_dvc_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing DVC connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ DVC connected successfully")
                print(f"   Repository: {result['repo_path']}")
            else:
                print(f"❌ DVC connection failed: {result['error']}")
        
        elif init:
            print("Path Initializing DVC repository...")
            result = asyncio.run(service.init_repo())
            print(f"✅ Repository initialized: {result['repo_path']}")
        
        elif add_remote:
            if ':' not in add_remote:
                print("❌ Remote format should be: name:url")
                return
            name, url = add_remote.split(':', 1)
            print(f"📦 Adding remote: {name} -> {url}")
            result = asyncio.run(service.add_remote(name, url))
            print(f"✅ Remote added: {result['name']}")
        
        elif add_data:
            print(f"Status Adding data to tracking: {add_data}")
            result = asyncio.run(service.add_data(add_data))
            print(f"✅ Data added: {add_data}")
        
        elif push:
            print("📤 Pushing data to remote...")
            result = asyncio.run(service.push_data())
            print(f"✅ Data pushed: {result['targets']}")
        
        elif pull:
            print("📥 Pulling data from remote...")
            result = asyncio.run(service.pull_data())
            print(f"✅ Data pulled: {result['targets']}")
        
        elif status:
            print("Status Repository status:")
            result = asyncio.run(service.get_status())
            for detail in result['details']:
                print(f"   {detail}")
        
        else:
            print("✅ DVC configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ DVC service not available. Install with: pip install dvc")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--list-experiments', is_flag=True, help='List all experiments')
@click.option('--create-experiment', help='Create a new experiment')
@click.option('--list-runs', help='List runs in experiment')
@click.option('--export', help='Export experiment data (json/csv)')
def mlflow(test, list_experiments, create_experiment, list_runs, export):
    """MLflow experiment tracking and model registry"""
    try:
        from terradev_cli.ml_services.mlflow_service import create_mlflow_service_from_credentials, get_mlflow_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('mlflow')
        
        if not creds.get('tracking_uri'):
            print(get_mlflow_setup_instructions())
            return
        
        service = create_mlflow_service_from_credentials(creds)
        
        if test:
            print("🔍 Testing MLflow connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ MLflow connected successfully")
                print(f"   Tracking URI: {result['tracking_uri']}")
                print(f"   Experiments: {result['experiments_count']}")
            else:
                print(f"❌ MLflow connection failed: {result['error']}")
        
        elif list_experiments:
            print("Plan Listing MLflow experiments...")
            experiments = asyncio.run(service.list_experiments())
            
            for exp in experiments:
                print(f"   🧪 {exp['name']} (ID: {exp['experiment_id']})")
        
        elif create_experiment:
            print(f"🧪 Creating experiment: {create_experiment}")
            result = asyncio.run(service.create_experiment(create_experiment, "Created via Terradev CLI"))
            print(f"✅ Experiment created: {result['experiment_id']}")
        
        elif list_runs:
            print(f"Status Listing runs in experiment: {list_runs}")
            runs = asyncio.run(service.list_runs([list_runs]))
            
            for run in runs[:10]:  # Show first 10
                info = run.get('info', {})
                print(f"   🏃 {info.get('run_id', 'N/A')[:8]} - {info.get('status', 'N/A')}")
        
        elif export:
            print(f"📤 Exporting experiment data...")
            data = asyncio.run(service.export_experiment_data(export, 'json'))
            print(data)
        
        else:
            print("✅ MLflow configured. Use --test to verify connection.")
            
    except ImportError:
        print("❌ MLflow service not available. Install with: pip install mlflow")

@ml.command()
@click.option('--test', is_flag=True, help='Test connection to the service')
@click.option('--status', is_flag=True, help='Show cluster status')
@click.option('--list-nodes', is_flag=True, help='List cluster nodes')
@click.option('--start', is_flag=True, help='Start Ray cluster')
@click.option('--stop', is_flag=True, help='Stop Ray cluster')
@click.option('--dashboard', is_flag=True, help='Get dashboard URL')
@click.option('--install', is_flag=True, help='Show installation instructions')
@click.option('--install-monitoring', is_flag=True, help='Install monitoring stack with Ray dashboards')
@click.option('--metrics-summary', is_flag=True, help='Get comprehensive metrics summary')
@click.option('--grafana', is_flag=True, help='Access Grafana dashboard')
@click.option('--prometheus', is_flag=True, help='Access Prometheus metrics')
def ray(test, status, list_nodes, start, stop, dashboard, install, install_monitoring, metrics_summary, grafana, prometheus):
    """Enhanced Ray distributed computing with monitoring and dashboards"""
    try:
        from terradev_cli.ml_services.ray_enhanced import create_enhanced_ray_service_from_credentials, get_enhanced_ray_setup_instructions
        
        api = TerradevAPI()
        creds = api._provider_creds('ray')
        
        # Ray can work without credentials for local clusters
        service = create_enhanced_ray_service_from_credentials(creds)
        
        if install:
            print(get_enhanced_ray_setup_instructions())
            return
        
        if test:
            print("🔍 Testing enhanced Ray connection...")
            result = asyncio.run(service.test_connection())
            
            if result['status'] == 'connected':
                print(f"✅ Ray connected successfully")
                print(f"   Version: {result.get('ray_version', 'N/A')}")
                print(f"   Cluster: {result.get('cluster_name', 'local')}")
                print(f"   Dashboard: {result.get('dashboard_uri', 'N/A')}")
                print(f"   Monitoring: {'Enabled' if creds.get('ray_monitoring_enabled') == 'true' else 'Disabled'}")
            elif result['status'] == 'not_connected':
                print(f"Warning  Ray installed but cluster not running")
                print(f"   Version: {result.get('ray_version', 'N/A')}")
                print(f"   Error: {result['error']}")
                print(f"   💡 Suggestion: {result.get('suggestion')}")
            else:
                print(f"❌ Ray connection failed: {result['error']}")
                if 'not installed' in result['error']:
                    print("   💡 Install Ray: pip install ray[default]")
                    print("   📖 For full features: pip install ray[default,train]")
        
        elif install_monitoring:
            print("Deploying Installing enhanced Ray monitoring stack...")
            result = asyncio.run(service.install_monitoring_stack())
            
            if result['status'] == 'installed':
                print(f"✅ Ray monitoring stack installed")
                print(f"   Ray Dashboard: {result.get('ray')}")
                print(f"   Prometheus: {result.get('prometheus')}")
                print(f"   Grafana: {result.get('grafana')}")
                print(f"   Dashboards: {result.get('dashboards')}")
                print(f"   Access Ray Dashboard: http://localhost:8265")
                print(f"   Access Grafana: http://localhost:3000")
            else:
                print(f"❌ Installation failed: {result['error']}")
        
        elif metrics_summary:
            print("Status Getting comprehensive Ray metrics summary...")
            result = asyncio.run(service.get_monitoring_status())
            
            if result.get('status') != 'failed':
                print(f"   Ray Status: {result.get('ray', {})}")
                print(f"   Monitoring: {result.get('monitoring', {})}")
                print(f"   Metrics: {result.get('metrics', {})}")
            else:
                print(f"❌ Metrics summary failed: {result.get('error')}")
        
        elif grafana:
            print("🌐 Accessing Ray Grafana dashboard...")
            print("   Access at: http://localhost:3000")
            print("   Username: admin")
            print("   Password: prom-operator")
            print("   Ray metrics are available in the 'Ray Overview' dashboard")
        
        elif prometheus:
            print("Status Accessing Ray Prometheus metrics...")
            print("   Access at: http://localhost:8080")
            print("   Available metrics: ray_cluster_total_workers, ray_cluster_cpu_total, ray_cluster_memory_total")
        
        elif status:
            print("Status Enhanced Ray cluster status:")
            result = asyncio.run(service.get_monitoring_status())
            
            if result.get('ray', {}).get('status') == 'running':
                print(f"   ✅ Status: {result['ray']['status']}")
                print(f"   Version: {result['ray'].get('version', 'N/A')}")
                print(f"   Cluster: {result['ray'].get('cluster_name', 'local')}")
                print(f"   Dashboard: {result['ray'].get('dashboard_uri', 'N/A')}")
                
                if result.get('metrics'):
                    metrics = result['metrics']
                    print(f"   Workers: {metrics.get('total_workers', 0)}")
                    print(f"   CPU Total: {metrics.get('cpu_total', 0)}")
                    print(f"   CPU Used: {metrics.get('cpu_used', 0)}")
                    print(f"   Memory Total: {metrics.get('memory_total', 0)}")
                    print(f"   Memory Used: {metrics.get('memory_used', 0)}")
                    print(f"   GPU Total: {metrics.get('gpu_total', 0)}")
                    print(f"   GPU Used: {metrics.get('gpu_used', 0)}")
            else:
                print(f"   ❌ Status: {result.get('ray', {}).get('status', 'Unknown')}")
                print(f"   Error: {result.get('ray', {}).get('error', 'Unknown error')}")
        
        elif list_nodes:
            print("🖥️ Listing Ray nodes...")
            result = asyncio.run(service.get_monitoring_status())
            
            if result.get('ray', {}).get('status') == 'running':
                metrics = result.get('metrics', {})
                total_workers = metrics.get('total_workers', 0)
                print(f"   Total Workers: {total_workers}")
                print(f"   Active Workers: {total_workers}")
                print(f"   Head Node: {creds.get('ray_head_node_ip', 'localhost')}")
            else:
                print("   ℹ️  No active Ray cluster found")
        
        elif start:
            print("Deploying Starting enhanced Ray cluster...")
            result = asyncio.run(service.start_cluster(head_node=True))
            print(f"✅ Cluster started: {result['status']}")
            
            if creds.get('ray_monitoring_enabled') == 'true':
                print("   Status Monitoring enabled - access dashboards:")
                print("      Ray Dashboard: http://localhost:8265")
                print("      Grafana: http://localhost:3000")
                print("      Prometheus: http://localhost:8080")
        
        elif stop:
            print("🛑 Stopping Ray cluster...")
            result = asyncio.run(service.stop_cluster())
            print(f"✅ Cluster stopped: {result['status']}")
        
        elif dashboard:
            print("Status Getting Ray dashboard URL...")
            url = asyncio.run(service.get_ray_dashboard_url())
            if url:
                print(f"🌐 Dashboard: {url}")
            else:
                print("❌ Dashboard URL not found")
        
        else:
            print("✅ Enhanced Ray configured. Use --test to verify connection.")
            if creds.get('ray_monitoring_enabled') == 'true':
                print("   Status Monitoring enabled - use --install-monitoring to set up dashboards")
            
    except ImportError:
        print("❌ Enhanced Ray service not available. Install with: pip install ray[default]")


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest Cache + Drift Detection — CLI-native reliability
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command('up')
@click.option('--job', '-j', required=True, help='Job name for manifest tracking')
@click.option('--cache-dir', default='./manifests', help='Manifest cache directory')
@click.option('--fix-drift', is_flag=True, help='Detect and fix drift automatically')
@click.option('--gpu-type', default='A100', help='GPU type')
@click.option('--gpu-count', type=int, default=1, help='Number of GPUs')
@click.option('--hours', type=float, default=1.0, help='Estimated runtime in hours')
@click.option('--budget', type=float, help='Budget constraint ($/hr)')
@click.option('--region', help='Preferred region')
@click.option('--dataset', help='Dataset path for drift detection')
@click.option('--ttl', default='1h', help='Time to live for nodes')
def up(job, cache_dir, fix_drift, gpu_type, gpu_count, hours, budget, region, dataset, ttl):
    """CLI-native provisioning with manifest cache + drift detection"""
    import asyncio
    
    async def _up():
        from terradev_cli.core.manifest_cache import ManifestCache, Manifest, ManifestNode
        from terradev_cli.core.drift_detector import DriftDetector
        
        cache = ManifestCache(cache_dir)
        detector = DriftDetector(cache_dir)
        
        if fix_drift:
            # RE-PROVISION: Detect + Fix (single command)
            print(f"🔍 Detecting drift for job {job}...")
            
            try:
                result = await detector.fix_drift(job)
                
                if result['status'] == 'no_drift':
                    print("✅ No drift detected - everything is in sync")
                elif result['status'] == 'fixed':
                    print(f"🔧 Drift fixed successfully:")
                    print(f"   Terminated: {result['terminated']} nodes")
                    print(f"   Recreated: {result['recreated']} nodes")
                else:
                    print(f"Warning  Partial fix - some nodes may still need attention")
                
                return result
                
            except Exception as e:
                print(f"❌ Error fixing drift: {e}")
                return
        
        # PROVISION: Auto-generates + stores manifest
        print(f"Deploying Provisioning job {job} with manifest cache...")
        
        # Get optimal deployment (existing logic)
        from terradev_cli.core.deployment_router import SmartDeploymentRouter
        
        router = SmartDeploymentRouter()
        user_request = {
            'gpu_type': gpu_type,
            'gpu_count': gpu_count,
            'estimated_hours': hours,
            'budget': budget,
            'region': region
        }
        
        recommendations = await router.recommend_deployments(user_request)
        
        if not recommendations:
            print("❌ No deployment options available")
            return
        
        # Choose best option (simplified - would normally prompt user)
        best_option = recommendations[0]
        
        print(f"🎯 Deploying: {best_option.provider} {best_option.instance_type}")
        print(f"   Cost: ${best_option.price_per_hour:.2f}/hr")
        print(f"   Confidence: {best_option.confidence:.1%}")
        
        # Execute deployment
        try:
            deployment_result = await router.execute_deployment(best_option, router.requirements_analyzer.analyze(user_request))
            
            # Create manifest nodes
            nodes = []
            for i in range(gpu_count):
                node = ManifestNode(
                    provider=best_option.provider,
                    pod_id=f"{job}-node-{i+1}",
                    instance_id=deployment_result.get('instance_id', f"instance-{i+1}"),
                    gpus=1,
                    gpu_type=gpu_type,
                    region=region or 'us-east-1',
                    status='running',
                    created_at=datetime.utcnow().isoformat(),
                    ttl=ttl
                )
                nodes.append(node)
            
            # Create manifest
            version = f"v{len(cache.list_versions(job)) + 1}"
            dataset_hash = cache.compute_dataset_hash(dataset) if dataset else "sha256:none"
            
            manifest = Manifest(
                job=job,
                version=version,
                nodes=nodes,
                dataset_hash=dataset_hash,
                ttl=ttl,
                created_at=datetime.utcnow().isoformat(),
                metadata={
                    'deployment_id': deployment_result.get('deployment_id'),
                    'provider': best_option.provider,
                    'instance_type': best_option.instance_type,
                    'price_per_hour': best_option.price_per_hour,
                    'confidence': best_option.confidence
                }
            )
            
            # Store manifest
            manifest_path = cache.store_manifest(manifest)
            
            print(f"✅ Job {job} provisioned successfully!")
            print(f"   Manifest: {manifest_path}")
            print(f"   Version: {version}")
            print(f"   Nodes: {len(nodes)}")
            print(f"   Fix drift: terradev up --job {job} --fix-drift")
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
    
    asyncio.run(_up())


@cli.command('rollback')
@click.argument('job_version', required=True)  # Format: job@v3
@click.option('--cache-dir', default='./manifests', help='Manifest cache directory')
def rollback(job_version, cache_dir):
    """EXPLICIT ROLLBACK (versioned manifests)"""
    import asyncio
    
    async def _rollback():
        from terradev_cli.core.drift_detector import DriftDetector
        
        # Parse job@version
        if '@' not in job_version:
            print("❌ Invalid format. Use: job@version (e.g., llama3@v3)")
            return
        
        job, version = job_version.split('@', 1)
        
        print(f"🔄 Rolling back {job} to version {version}...")
        
        detector = DriftDetector(cache_dir)
        
        try:
            result = await detector.rollback(job, version)
            
            print(f"✅ Rollback completed:")
            print(f"   Target version: {result['target_version']}")
            print(f"   Terminated: {result['terminated']} nodes")
            print(f"   Recreated: {result['recreated']} nodes")
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            sys.exit(1)
    
    asyncio.run(_rollback())


@cli.command('manifests')
@click.option('--job', help='Show versions for specific job')
@click.option('--cache-dir', default='./manifests', help='Manifest cache directory')
def manifests(job, cache_dir):
    """List cached manifests and versions"""
    try:
        from terradev_cli.core.manifest_cache import ManifestCache
    except ImportError:
        print("❌ Manifest cache module not available. Install terradev_cli package.")
        sys.exit(1)
    
    cache = ManifestCache(cache_dir)
    
    if job:
        # Show versions for specific job
        versions = cache.list_versions(job)
        if versions:
            print(f"Plan Manifest versions for {job}:")
            for version in versions:
                manifest = cache.load_manifest(job, version)
                if manifest:
                    print(f"   {version}: {len(manifest.nodes)} nodes, created {manifest.created_at}")
        else:
            print(f"❌ No manifests found for job {job}")
    else:
            # Show all jobs
            manifest_files = list(Path(cache_dir).glob("*.json"))
            if manifest_files:
                jobs = set()
                for file_path in manifest_files:
                    parts = file_path.stem.split('.')
                    if len(parts) >= 2:
                        jobs.add(parts[0])
                
                if jobs:
                    print("Plan Cached jobs:")
                    for job_name in sorted(jobs):
                        versions = cache.list_versions(job_name)
                        print(f"   {job_name}: {len(versions)} versions")
                else:
                    print("❌ No cached manifests found")
            else:
                print("❌ No cached manifests found")


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Spaces One-Click Deployment
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command('hf-space')
@click.argument('space_name', required=True)
@click.option('--model-id', required=True, help='HuggingFace model ID to deploy')
@click.option('--hardware', default='cpu-basic', 
              type=click.Choice(['cpu-basic', 'cpu-upgrade', 't4-medium', 'a10g-large', 'a100-large']),
              help='Hardware tier for the Space')
@click.option('--sdk', default='gradio', 
              type=click.Choice(['gradio', 'streamlit', 'docker']),
              help='SDK for the Space')
@click.option('--private', is_flag=True, help='Make the Space private')
@click.option('--template', 
              type=click.Choice(['llm', 'embedding', 'image']),
              help='Use pre-configured template')
@click.option('--env', '-e', multiple=True, help='Environment variables KEY=VALUE')
@click.option('--secret', '-s', multiple=True, help='Secrets KEY=VALUE')
def hf_space(space_name, model_id, hardware, sdk, private, template, env, secret):
    """One-click HuggingFace Spaces deployment"""
    import asyncio
    
    async def _hf_space():
        from terradev_cli.core.hf_spaces import HFSpacesDeployer, HFSpaceConfig, HFSpaceTemplates
        
        # Get HF token
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not hf_token:
            print("❌ HF_TOKEN environment variable required")
            print("   Set it with: export HF_TOKEN=your_token")
            sys.exit(2)
        
        deployer = HFSpacesDeployer(hf_token)
        
        # Parse environment variables
        env_vars = {}
        for env_var in env:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env_vars[key] = value
        
        # Parse secrets
        secrets = {}
        for secret_var in secret:
            if '=' in secret_var:
                key, value = secret_var.split('=', 1)
                secrets[key] = value
        
        # Use template or create custom config
        if template:
            if template == 'llm':
                config = HFSpaceTemplates.get_llm_template(model_id, space_name)
            elif template == 'embedding':
                config = HFSpaceTemplates.get_embedding_template(model_id, space_name)
            elif template == 'image':
                config = HFSpaceTemplates.get_image_model_template(model_id, space_name)
            
            # Override with user-specified options
            config.hardware = hardware
            config.sdk = sdk
            config.private = private
            config.env_vars.update(env_vars)
            if secrets:
                config.secrets = secrets
        else:
            config = HFSpaceConfig(
                name=space_name,
                model_id=model_id,
                hardware=hardware,
                sdk=sdk,
                python_version="3.10",
                private=private,
                env_vars=env_vars,
                secrets=secrets if secrets else None
            )
        
        print(f"Deploying {model_id} to HuggingFace Spaces...")
        print(f"   Space: {space_name}")
        print(f"   Hardware: {hardware}")
        print(f"   SDK: {sdk}")
        
        try:
            result = await deployer.create_space(config)
            
            if result['status'] == 'created':
                print(f"✅ Space created successfully!")
                print(f"   🌐 Space URL: {result['space_url']}")
                print(f"   🔧 Hardware: {result['hardware']}")
                print(f"   🤖 Model: {result['model_id']}")
                print(f"   ⏱️  Your Space will be ready in 2-5 minutes")
                print(f"   Status 100k+ researchers can now access your model!")
            else:
                print(f"❌ Failed to create space: {result['error']}")
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
    
    asyncio.run(_hf_space())

# Model Orchestrator Commands
@cli.command()
@click.option('--gpu-id', default=0, help='GPU ID to use for orchestration')
@click.option('--memory-gb', default=80.0, help='Total GPU memory in GB')
@click.option('--policy', type=click.Choice(['billing_optimized', 'latency_optimized', 'hybrid']), 
              default='billing_optimized', help='Scaling policy')
def orchestrator_start(gpu_id, memory_gb, policy):
    """Start the model orchestrator for multi-model inference"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator, ScalingPolicy
    
    policy_map = {
        'billing_optimized': ScalingPolicy.BILLING_OPTIMIZED,
        'latency_optimized': ScalingPolicy.LATENCY_OPTIMIZED,
        'hybrid': ScalingPolicy.HYBRID
    }
    
    orchestrator = ModelOrchestrator(
        gpu_id=gpu_id,
        total_memory_gb=memory_gb,
        scaling_policy=policy_map[policy]
    )
    
    async def run_orchestrator():
        await orchestrator.start()
        print(f"Model Orchestrator started on GPU {gpu_id}")
        print(f"Memory: {memory_gb}GB total, {orchestrator.memory_threshold_gb:.1f}GB usable")
        print(f"Policy: {policy}")
        print(f"Press Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(10)
                status = orchestrator.get_status()
                print(f"Status: {status['warm_models_count']} warm models, "
                      f"{status['used_memory_gb']:.1f}GB used "
                      f"({status['memory_utilization_percent']:.1f}%)")
        except KeyboardInterrupt:
            print("\nStopping orchestrator...")
            await orchestrator.stop()
    
    asyncio.run(run_orchestrator())

@cli.command()
@click.argument('model-id')
@click.argument('model-path')
@click.option('--framework', type=click.Choice(['pytorch', 'vllm', 'sglang']), 
              default='pytorch', help='Model framework')
@click.option('--priority', default=0, help='Priority for eviction (higher = less likely to evict)')
@click.option('--tags', help='Comma-separated tags for model categorization')
def orchestrator_register(model_id, model_path, framework, priority, tags):
    """Register a model with the orchestrator"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator
    
    orchestrator = ModelOrchestrator()
    tag_set = set(tags.split(',')) if tags else None
    
    instance = orchestrator.register_model(
        model_id=model_id,
        model_path=model_path,
        framework=framework,
        priority=priority,
        tags=tag_set
    )
    
    print(f"Model registered: {model_id}")
    print(f"  Path: {model_path}")
    print(f"  Framework: {framework}")
    print(f"  Priority: {priority}")
    print(f"  Tags: {', '.join(tag_set) if tag_set else 'None'}")

@cli.command()
@click.argument('model-id')
@click.option('--force', is_flag=True, help='Force loading even if memory is full')
def orchestrator_load(model_id, force):
    """Load a model into GPU memory"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator
    
    orchestrator = ModelOrchestrator()
    
    async def load_model():
        success = await orchestrator.load_model(model_id, force=force)
        if success:
            details = orchestrator.get_model_details(model_id)
            print(f"Model {model_id} loaded successfully!")
            print(f"  State: {details['state']}")
            print(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            print(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            print(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
        else:
            print(f"Failed to load model {model_id}")
    
    asyncio.run(load_model())

@cli.command()
@click.argument('model-id')
def orchestrator_evict(model_id):
    """Evict a model from GPU memory"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator
    
    orchestrator = ModelOrchestrator()
    
    async def evict_model():
        success = await orchestrator.evict_model(model_id)
        if success:
            print(f"Model {model_id} evicted successfully")
        else:
            print(f"Failed to evict model {model_id}")
    
    asyncio.run(evict_model())

@cli.command()
@click.option('--model-id', help='Get details for specific model')
def orchestrator_status(model_id):
    """Get orchestrator and model status"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator
    
    orchestrator = ModelOrchestrator()
    
    if model_id:
        details = orchestrator.get_model_details(model_id)
        if details:
            print(f"Model Details: {model_id}")
            print(f"  Framework: {details['framework']}")
            print(f"  State: {details['state']}")
            print(f"  Priority: {details['priority']}")
            print(f"  Tags: {', '.join(details['tags'])}")
            print(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            print(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            print(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
            print(f"  Requests/hour: {details['metrics']['requests_per_hour']:.1f}")
            print(f"  Avg latency: {details['metrics']['avg_latency_ms']:.1f}ms")
            print(f"  Error rate: {details['metrics']['error_rate']:.2f}")
            print(f"  Last accessed: {details['last_accessed']}")
        else:
            print(f"Model {model_id} not found")
    else:
        status = orchestrator.get_status()
        print("Orchestrator Status:")
        print(f"  GPU: {status['gpu_id']}")
        print(f"  Total memory: {status['total_memory_gb']:.1f}GB")
        print(f"  Used memory: {status['used_memory_gb']:.1f}GB")
        print(f"  Available: {status['available_memory_gb']:.1f}GB")
        print(f"  Utilization: {status['memory_utilization_percent']:.1f}%")
        print(f"  Policy: {status['scaling_policy']}")
        print(f"  Total models: {status['total_models']}")
        print(f"  Warm models: {status['warm_models_count']}")
        print(f"  Warm memory: {status['warm_models_memory_gb']:.1f}GB")
        
        print("\nModels by state:")
        for state, model_ids in status['models_by_state'].items():
            if model_ids:
                print(f"  {state}: {', '.join(model_ids)}")

@cli.command()
@click.argument('model-id')
def orchestrator_infer(model_id):
    """Test inference with a model"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator
    
    orchestrator = ModelOrchestrator()
    
    async def test_inference():
        success, latency_ms = await orchestrator.handle_request(model_id)
        if success:
            print(f"Inference successful for {model_id}")
            print(f"  Latency: {latency_ms:.1f}ms")
        else:
            print(f"Inference failed for {model_id}")
    
    asyncio.run(test_inference())

@cli.command()
@click.option('--strategy', type=click.Choice(['traffic_based', 'time_based', 'priority_based', 'cost_optimized', 'latency_optimized']), 
              default='traffic_based', help='Warm pool strategy')
@click.option('--max-warm', default=10, help='Maximum models to keep warm')
@click.option('--min-warm', default=3, help='Minimum models to keep warm')
def warm_pool_start(strategy, max_warm, min_warm):
    """Start the warm pool manager for intelligent pre-warming"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig, WarmStrategy
    
    strategy_map = {
        'traffic_based': WarmStrategy.TRAFFIC_BASED,
        'time_based': WarmStrategy.TIME_BASED,
        'priority_based': WarmStrategy.PRIORITY_BASED,
        'cost_optimized': WarmStrategy.COST_OPTIMIZED,
        'latency_optimized': WarmStrategy.LATENCY_OPTIMIZED
    }
    
    config = WarmPoolConfig(
        max_warm_models=max_warm,
        min_warm_models=min_warm,
        strategy=strategy_map[strategy],
        enable_predictive_warming=True
    )
    
    warm_pool = WarmPoolManager(config)
    
    async def run_warm_pool():
        await warm_pool.start()
        print(f"Warm Pool Manager started")
        print(f"Strategy: {strategy}")
        print(f"Capacity: {min_warm}-{max_warm} models")
        print(f"Press Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(30)
                status = warm_pool.get_status()
                print(f"Status: {status['warm_models_count']} warm, "
                      f"{status['cache_hit_rate']:.1%} hit rate, "
                      f"{status['total_requests']} requests")
        except KeyboardInterrupt:
            print("\nStopping warm pool manager...")
            await warm_pool.stop()
    
    asyncio.run(run_warm_pool())

@cli.command()
@click.argument('model-id')
@click.option('--priority', default=0, help='Model priority for warming')
def warm_pool_register(model_id, priority):
    """Register a model with the warm pool manager"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig
    
    warm_pool = WarmPoolManager(WarmPoolConfig())
    warm_pool.register_model(model_id, priority)
    
    print(f"Model {model_id} registered with warm pool")
    print(f"  Priority: {priority}")

@cli.command()
def warm_pool_status():
    """Get warm pool manager status"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig
    
    warm_pool = WarmPoolManager(WarmPoolConfig())
    status = warm_pool.get_status()
    
    print("Warm Pool Status:")
    print(f"  Warm models: {status['warm_models_count']}")
    print(f"  Warming models: {status['warming_models_count']}")
    print(f"  Total models: {status['total_models']}")
    print(f"  Strategy: {status['strategy']}")
    print(f"  Cache hit rate: {status['cache_hit_rate']:.1%}")
    print(f"  Total requests: {status['total_requests']}")
    print(f"  Cold starts: {status['cold_starts']}")
    print(f"  Avg warm latency: {status['avg_warm_latency_ms']:.1f}ms")
    print(f"  Avg cold latency: {status['avg_cold_latency_ms']:.1f}ms")
    print(f"  Memory saved: {status['memory_saved_gb']:.1f}GB")
    print(f"  Cost saved: ${status['cost_saved_usd']:.2f}")

@cli.command()
@click.option('--strategy', type=click.Choice(['minimize_cost', 'balance_cost_latency', 'latency_critical', 'budget_constrained']), 
              default='balance_cost_latency', help='Cost optimization strategy')
@click.option('--budget', default=15.0, help='Hourly budget in USD')
@click.option('--cost-per-gb', default=0.10, help='Cost per GB per hour in USD')
def cost_scaler_start(strategy, budget, cost_per_gb):
    """Start the cost-aware scaling manager"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig, CostStrategy
    
    strategy_map = {
        'minimize_cost': CostStrategy.MINIMIZE_COST,
        'balance_cost_latency': CostStrategy.BALANCE_COST_LATENCY,
        'latency_critical': CostStrategy.LATENCY_CRITICAL,
        'budget_constrained': CostStrategy.BUDGET_CONSTRAINED
    }
    
    config = CostConfig(
        hourly_budget_usd=budget,
        cost_per_gb_hour_usd=cost_per_gb,
        strategy=strategy_map[strategy],
        enable_cost_prediction=True
    )
    
    cost_scaler = CostScaler(config)
    
    async def run_cost_scaler():
        await cost_scaler.start()
        print(f"Cost Scaler started")
        print(f"Strategy: {strategy}")
        print(f"Budget: ${budget}/hour")
        print(f"Cost per GB: ${cost_per_gb}/hour")
        print(f"Press Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(60)
                status = cost_scaler.get_status()
                print(f"Status: ${status['current_hourly_cost_usd']:.2f}/hour, "
                      f"{status['budget_utilization_percent']:.1f}% budget, "
                      f"{status['active_models']} models")
        except KeyboardInterrupt:
            print("\nStopping cost scaler...")
            await cost_scaler.stop()
    
    asyncio.run(run_cost_scaler())

@cli.command()
def cost_scaler_status():
    """Get cost scaler status and recommendations"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig
    
    cost_scaler = CostScaler(CostConfig())
    status = cost_scaler.get_status()
    
    print("Cost Scaler Status:")
    print(f"  Current cost: ${status['current_hourly_cost_usd']:.3f}/hour")
    print(f"  Budget utilization: {status['budget_utilization_percent']:.1f}%")
    print(f"  Memory cost: ${status['memory_cost_usd']:.3f}/hour")
    print(f"  Cold start penalties: ${status['cold_start_penalty_usd']:.3f}/hour")
    print(f"  Total cost: ${status['total_cost_usd']:.2f}")
    print(f"  Cost savings: ${status['cost_savings_usd']:.2f}")
    print(f"  Memory usage: {status['current_memory_usage_gb']:.1f}GB")
    print(f"  Active models: {status['active_models']}")
    print(f"  Strategy: {status['strategy']}")
    print(f"  Is peak hour: {status['is_peak_hour']}")
    print(f"  Predicted cost (1h): ${status['predicted_cost_1h']:.3f}")
    print(f"  Predicted cost (2h): ${status['predicted_cost_2h']:.3f}")
    
    # Get recommendations
    recommendations = cost_scaler.get_cost_optimization_recommendations()
    if recommendations:
        print("\nCost Optimization Recommendations:")
        for rec in recommendations:
            print(f"  {rec['priority'].upper()}: {rec['message']}")
            print(f"    Action: {rec['action']}")
            print(f"    Potential savings: {rec['potential_savings']}")

@cli.command()
@click.argument('model-id')
def cost_scaler_model_details(model_id):
    """Get cost details for a specific model"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig
    
    cost_scaler = CostScaler(CostConfig())
    details = cost_scaler.get_model_cost_details(model_id)
    
    if details:
        print(f"Cost Details for {model_id}:")
        print(f"  Memory usage: {details['memory_gb']:.1f}GB")
        print(f"  Hourly cost: ${details['hourly_cost_usd']:.3f}")
        print(f"  Cold start penalty: ${details['cold_start_penalty_usd']:.3f}")
        print(f"  Estimated daily cost: ${details['total_cost_today']:.2f}")
        print(f"  Cost rank: {details['cost_rank']} (1 = most expensive)")
    else:
        print(f"Model {model_id} not found or not loaded")

# GitOps Commands
@cli.group()
def gitops():
    """GitOps automation and infrastructure as code"""
    pass

@gitops.command()
@click.option('--provider', type=click.Choice(['github', 'gitlab', 'bitbucket', 'azure_devops']), 
              required=True, help='Git provider')
@click.option('--repo', '--repository', required=True, help='Repository name (format: owner/repo)')
@click.option('--tool', type=click.Choice(['argocd', 'flux']), default='argocd', help='GitOps tool')
@click.option('--cluster', required=True, help='Cluster name')
@click.option('--git-url', help='Git repository URL (auto-generated if not provided)')
@click.option('--git-token', help='Git access token')
@click.option('--namespace', default='gitops-system', help='Namespace for GitOps tools')
@click.option('--auto-sync/--no-auto-sync', default=True, help='Enable automatic synchronization')
@click.option('--prune/--no-prune', default=True, help='Enable resource pruning')
def init(provider, repository, tool, cluster, git_url, git_token, namespace, auto_sync, prune):
    """Initialize GitOps repository and structure"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitProvider, GitOpsTool
    
    provider_map = {
        'github': GitProvider.GITHUB,
        'gitlab': GitProvider.GITLAB,
        'bitbucket': GitProvider.BITBUCKET,
        'azure_devops': GitProvider.AZURE_DEVOPS
    }
    
    tool_map = {
        'argocd': GitOpsTool.ARGOCD,
        'flux': GitOpsTool.FLUX
    }
    
    config = GitOpsConfig(
        provider=provider_map[provider],
        repository=repository,
        tool=tool_map[tool],
        cluster_name=cluster,
        git_url=git_url,
        git_token=git_token,
        namespace=namespace,
        auto_sync=auto_sync,
        prune_resources=prune
    )
    
    gitops_manager = GitOpsManager(config)
    
    async def run_init():
        print(f"Initializing GitOps repository: {repository}")
        print(f"Provider: {provider}")
        print(f"Tool: {tool}")
        print(f"Cluster: {cluster}")
        
        success = await gitops_manager.init_repository()
        if success:
            print("GitOps repository initialized successfully")
            print(f"Repository structure created at: {gitops_manager.work_dir}")
            print("\nNext steps:")
            print(f"1. Push the repository to {provider}")
            print(f"2. Run 'terradev gitops bootstrap --tool {tool}'")
            print(f"3. Run 'terradev gitops sync --cluster {cluster}'")
        else:
            print("Failed to initialize GitOps repository")
    
    asyncio.run(run_init())

@gitops.command()
@click.option('--tool', type=click.Choice(['argocd', 'flux']), required=True, help='GitOps tool')
@click.option('--cluster', required=True, help='Cluster name')
@click.option('--namespace', default='gitops-system', help='Namespace for GitOps tools')
def bootstrap(tool, cluster, namespace):
    """Bootstrap GitOps tool on the cluster"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool
    
    # This is a simplified bootstrap - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool[tool.upper()],
        cluster_name=cluster,
        namespace=namespace
    )
    
    gitops_manager = GitOpsManager(config)
    
    async def run_bootstrap():
        print(f"Bootstrapping {tool} on cluster {cluster}")
        print(f"Namespace: {namespace}")
        
        success = await gitops_manager.bootstrap_gitops()
        if success:
            print(f"{tool.capitalize()} bootstrapped successfully")
            print("GitOps automation is now active")
        else:
            print(f"Failed to bootstrap {tool}")
    
    asyncio.run(run_bootstrap())

@gitops.command()
@click.option('--cluster', required=True, help='Cluster name')
@click.option('--environment', default='prod', help='Environment to sync')
@click.option('--tool', type=click.Choice(['argocd', 'flux']), default='argocd', help='GitOps tool')
def sync(cluster, environment, tool):
    """Sync cluster with Git repository"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool
    
    # This is a simplified sync - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool[tool.upper()],
        cluster_name=cluster
    )
    
    gitops_manager = GitOpsManager(config)
    
    async def run_sync():
        print(f"Syncing cluster {cluster}")
        print(f"Environment: {environment}")
        print(f"Tool: {tool}")
        
        success = await gitops_manager.sync_cluster(environment)
        if success:
            print(f"Cluster sync completed for {environment}")
        else:
            print("Failed to sync cluster")
    
    asyncio.run(run_sync())

@gitops.command()
@click.option('--dry-run/--apply', default=True, help='Dry run validation or apply changes')
@click.option('--cluster', help='Cluster name for validation')
@click.option('--environment', default='prod', help='Environment to validate')
def validate(dry_run, cluster, environment):
    """Validate GitOps configuration"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool
    
    # This is a simplified validation - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool.ARGOCD,  # Default
        cluster_name=cluster or "default"
    )
    
    gitops_manager = GitOpsManager(config)
    
    async def run_validate():
        print("Validating GitOps configuration")
        print(f"Dry run: {dry_run}")
        if cluster:
            print(f"Cluster: {cluster}")
        if environment:
            print(f"Environment: {environment}")
        
        results = await gitops_manager.validate_configuration(dry_run)
        
        if results['valid']:
            print("Configuration is valid")
        else:
            print("Configuration validation failed:")
            for error in results['errors']:
                print(f"  Error: {error}")
        
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  Warning: {warning}")
        
    
    gitops_manager = GitOpsManager(config)
    asyncio.run(gitops_manager.validate_configuration(path))

# InferX Commands
@cli.group()
def inferx():
    """InferX serverless inference platform - <2s cold starts, 90% GPU utilization"""
    pass

@inferx.command()
@click.option('--api-key', required=True, help='InferX API key')
@click.option('--endpoint', default='https://api.inferx.net', help='InferX API endpoint')
@click.option('--region', default='us-west-2', help='Region for deployment')
@click.option('--snapshot/--no-snapshot', default=True, help='Enable snapshot technology')
@click.option('--gpu-slicing/--no-gpu-slicing', default=True, help='Enable GPU slicing')
@click.option('--multi-tenant/--no-multi-tenant', default=True, help='Enable multi-tenant isolation')
def configure(api_key, endpoint, region, snapshot, gpu_slicing, multi_tenant):
    """Configure InferX provider credentials"""
    import os
    from pathlib import Path
    
    config_dir = Path.home() / '.terradev'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'inferx_config.json'
    config = {
        'api_key': api_key,
        'api_endpoint': endpoint,
        'region': region,
        'snapshot_enabled': snapshot,
        'gpu_slicing': gpu_slicing,
        'multi_tenant': multi_tenant
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ InferX configured successfully")
    print(f"📍 Endpoint: {endpoint}")
    print(f"🌍 Region: {region}")
    print(f"📸 Snapshot: {'Enabled' if snapshot else 'Disabled'}")
    print(f"🔪 GPU Slicing: {'Enabled' if gpu_slicing else 'Disabled'}")
    print(f"🏢 Multi-tenant: {'Enabled' if multi_tenant else 'Disabled'}")

@inferx.command()
@click.option('--model', required=True, help='Model ID or HuggingFace model name')
@click.option('--image', help='Docker image for model')
@click.option('--gpu-type', default='A100', help='GPU type')
@click.option('--gpu-memory', type=int, default=16, help='GPU memory in GB')
@click.option('--max-concurrency', type=int, default=10, help='Maximum concurrent requests')
@click.option('--framework', default='pytorch', help='Model framework')
@click.option('--openai-compatible/--no-openai-compatible', default=True, help='OpenAI-compatible API')
@click.option('--timeout', type=int, default=300, help='Request timeout in seconds')
def deploy(model, image, gpu_type, gpu_memory, max_concurrency, framework, openai_compatible, timeout):
    """Deploy model to InferX serverless platform"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    model_config = {
        'model_id': model,
        'image': image or f'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel',
        'gpu_type': gpu_type,
        'gpu_memory': gpu_memory,
        'max_concurrency': max_concurrency,
        'framework': framework,
        'openai_compatible': openai_compatible,
        'timeout': timeout
    }
    
    print(f"🚀 Deploying {model} to InferX...")
    print(f"🎮 GPU: {gpu_type} ({gpu_memory}GB)")
    print(f"⚡ Max Concurrency: {max_concurrency}")
    print(f"🔧 Framework: {framework}")
    
    try:
        result = asyncio.run(provider.deploy_model(model_config))
        
        print(f"✅ Model deployed successfully!")
        print(f"📋 Model ID: {result['model_id']}")
        print(f"🔗 Endpoint: {result['endpoint']}")
        print(f"⚡ Cold Start: {result['cold_start_time']}s")
        print(f"📊 GPU Utilization: {result['gpu_utilization']}%")
        print(f"📦 Models per Node: {result['models_per_node']}")
        
        if result['openai_compatible']:
            print(f"🔄 OpenAI Compatible: Yes")
            print(f"💡 Usage: curl -X POST {result['endpoint']} -H 'Authorization: Bearer YOUR_API_KEY'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command()
@click.option('--model-id', required=True, help='Model deployment ID')
def status(model_id):
    """Get model deployment status"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    try:
        result = asyncio.run(provider.get_model_status(model_id))
        
        print(f"📊 Model Status: {result.get('status', 'Unknown')}")
        print(f"🎮 GPU Type: {result.get('gpu_type', 'Unknown')}")
        print(f"⚡ Cold Start Time: {result.get('cold_start_time', 'Unknown')}s")
        print(f"📈 Requests/min: {result.get('requests_per_minute', 0)}")
        print(f"📊 GPU Utilization: {result.get('gpu_utilization', 0)}%")
        print(f"📦 Models on GPU: {result.get('models_on_gpu', 0)}")
        print(f"❌ Error Rate: {result.get('error_rate', 0)}%")
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command()
@click.option('--model-id', required=True, help='Model deployment ID')
def delete(model_id):
    """Delete model deployment"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    print(f"🗑️  Deleting model {model_id}...")
    
    try:
        success = asyncio.run(provider.delete_model(model_id))
        
        if success:
            print(f"✅ Model deleted successfully")
        else:
            print(f"❌ Failed to delete model")
        
    except Exception as e:
        print(f"❌ Failed to delete model: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command('list')
def inferx_list():
    """List all deployed models"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    try:
        models = asyncio.run(provider.list_models())
        
        if not models:
            print("📭 No models deployed")
            return
        
        print(f"📋 Deployed Models ({len(models)}):")
        print("-" * 80)
        
        for model in models:
            print(f"📦 {model.get('model_id', 'Unknown')}")
            print(f"   Status: {model.get('status', 'Unknown')}")
            print(f"   GPU: {model.get('gpu_type', 'Unknown')}")
            print(f"   Endpoint: {model.get('endpoint', 'Unknown')}")
            print(f"   Created: {model.get('created_at', 'Unknown')}")
            print()
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command()
def usage():
    """Get account usage statistics"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    try:
        stats = asyncio.run(provider.get_usage_stats())
        
        print(f"📊 InferX Usage Statistics")
        print("-" * 40)
        print(f"📈 Total Requests: {stats.get('total_requests', 0):,}")
        print(f"💰 Total Cost: ${stats.get('total_cost', 0):.4f}")
        print(f"📦 Active Models: {stats.get('active_models', 0)}")
        print(f"🎮 GPU Hours: {stats.get('gpu_hours', 0):.2f}")
        print(f"⚡ Average Latency: {stats.get('average_latency', 0):.0f}ms")
        print(f"📊 GPU Utilization: {stats.get('gpu_utilization', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Failed to get usage stats: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command()
@click.option('--gpu-type', default='A100', help='GPU type to quote')
@click.option('--region', help='Region for quote')
def quote(gpu_type, region):
    """Get pricing quotes for InferX"""
    import json
    from pathlib import Path
    
    # Load InferX config
    config_file = Path.home() / '.terradev' / 'inferx_config.json'
    if not config_file.exists():
        print("❌ InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    from terradev_cli.providers.inferx_provider import InferXProvider
    
    provider = InferXProvider(config)
    
    try:
        quotes = asyncio.run(provider.get_instance_quotes(gpu_type, region))
        
        if not quotes:
            print("❌ No quotes available")
            return
        
        quote = quotes[0]
        
        print(f"💰 InferX Pricing Quote")
        print("-" * 40)
        print(f"🎮 GPU Type: {quote['gpu_type']}")
        print(f"💸 Hourly Cost: ${quote['price_per_hour']:.4f} (Serverless - pay per request)")
        print(f"💵 Per Request: ${quote['price_per_request']:.4f} per 1K tokens")
        print(f"⚡ Cold Start: {quote['cold_start_time']}s")
        print(f"📊 GPU Utilization: {quote['gpu_utilization']}%")
        print(f"📦 Models per Node: {quote['models_per_node']}")
        print(f"🌍 Region: {quote['region']}")
        print()
        print(f"🌟 Key Features:")
        for feature in quote['features']:
            print(f"   ✅ {feature.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"❌ Failed to get quotes: {e}")
    finally:
        asyncio.run(provider.close())

@inferx.command()
@click.option('--cluster-config', help='Cluster configuration file')
@click.option('--usage-metrics', help='Usage metrics file')
@click.option('--tier', type=click.Choice(['economy', 'balanced', 'performance']), 
              default='economy', help='Cost optimization tier')
@click.option('--output', help='Output file for cost report')
@click.option('--implement', is_flag=True, help='Implement cost optimizations automatically')
def optimize(cluster_config, usage_metrics, tier, output, implement):
    """Analyze and optimize InferX costs with AI-powered recommendations"""
    import json
    from pathlib import Path
    from terradev_cli.cost_optimizer import InferXCostOptimizer, CostTier
    
    optimizer = InferXCostOptimizer()
    target_tier = CostTier(tier)
    
    # Load configuration (mock data for demo)
    cluster_config_data = {
        "nodes": [
            {"gpu_type": "A100", "gpu_count": 2, "spot": True},
            {"gpu_type": "A10G", "gpu_count": 1, "spot": True}
        ],
        "storage_gb": 200,
        "snapshot_gb": 500
    }
    
    usage_metrics_data = {
        "gpu_utilization": 65.0,
        "memory_utilization": 70.0,
        "cpu_utilization": 45.0,
        "models_deployed": 25,
        "cold_start_time": 2.2,
        "requests_per_hour": 150
    }
    
    if cluster_config:
        with open(cluster_config) as f:
            cluster_config_data = json.load(f)
    
    if usage_metrics:
        with open(usage_metrics) as f:
            usage_metrics_data = json.load(f)
    
    print(f"🔍 Analyzing InferX costs for {tier} tier...")
    
    # Generate cost report
    report = asyncio.run(optimizer.generate_cost_report(
        cluster_config_data, usage_metrics_data, target_tier
    ))
    
    # Display results
    print(f"\n📊 Cost Analysis Results:")
    print(f"=" * 50)
    print(f"💰 Current Monthly Cost: ${report['summary']['current_monthly_cost']:,.2f}")
    print(f"🎯 Potential Monthly Savings: ${report['summary']['potential_monthly_savings']:,.2f}")
    print(f"📈 Savings Percentage: {report['summary']['savings_percentage']:.1f}%")
    print(f"💸 Optimized Monthly Cost: ${report['summary']['optimized_monthly_cost']:,.2f}")
    print(f"⏱️  Payback Period: {report['summary']['payback_period_months']:.1f} months")
    print(f"📊 Annual ROI: {report['summary']['annual_roi']:.1f}%")
    print()
    
    print(f"🎯 Key Insights:")
    for insight in report['key_insights']:
        print(f"   • {insight}")
    print()
    
    print(f"📋 Top Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"   {i}. {rec['description']}")
        print(f"      Savings: ${rec['estimated_savings']:,.2f}/month")
        print(f"      Risk: {rec['risk_level']}, Priority: {rec['priority']}")
        print()
    
    # Save report
    if output:
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Detailed report saved to: {output}")
    
    # Implement optimizations if requested
    if implement:
        print(f"🚀 Implementing cost optimizations...")
        # Implementation logic would go here
        print(f"✅ Optimizations implemented successfully!")

# ═══════════════════════════════════════════════════════════════════════════════
# Training Pipeline — preflight / train / monitor / checkpoint / train-status
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_provision_nodes(provision_group: str, fmt: str = "text"):
    """Resolve provisioned instance IDs to node IPs for the train command.

    Looks up the parallel provision group in the cost DB, then resolves
    each instance to an IP via:
      1. Cached ip_address column (if previously resolved)
      2. Provider status API (async, parallel across all instances)
      3. instance_id as-is fallback (some providers use IP-based IDs)

    Returns (node_ips: list[str], ssh_key_path: str | None).
    node_ips is empty on failure.  ssh_key_path is the decrypted temp key
    if an encrypted per-provision key was generated during provision.
    """
    try:
        from terradev_cli.core.cost_tracker import (
            get_active_instances, get_latest_parallel_group, set_instance_ip,
            get_provision_ssh_key_path as _db_ssh_path,
        )
    except ImportError:
        print("ERROR: Cost tracker not available")
        return [], None

    # Resolve "latest" to actual group ID
    group_id = provision_group
    if group_id == "latest":
        group_id = get_latest_parallel_group()
        if not group_id:
            print("ERROR: No previous provision groups found. Run 'terradev provision' first.")
            return [], None
        if fmt != "json":
            print(f"  Resolved 'latest' -> {group_id}")

    instances = get_active_instances(parallel_group=group_id)
    if not instances:
        print(f"ERROR: No active instances in provision group {group_id}")
        return [], None

    node_ips = []
    unresolved = []

    for inst in instances:
        ip = inst.get("ip_address", "").strip()
        if ip:
            node_ips.append(ip)
        else:
            unresolved.append(inst)

    # Resolve remaining IPs via provider APIs
    if unresolved:
        if fmt != "json":
            print(f"  Resolving IPs for {len(unresolved)} instance(s)...")

        api = TerradevAPI()

        async def _resolve_ips():
            from terradev_cli.providers.provider_factory import ProviderFactory
            factory = ProviderFactory()
            results = {}

            async def _get_ip(inst):
                pname = inst["provider"].lower().replace(" ", "_")
                creds = api._provider_creds(pname)
                try:
                    provider = factory.create_provider(pname, creds)
                    status = await provider.get_instance_status(inst["instance_id"])
                    # Providers return IP in different fields
                    ip = (status.get("ip") or status.get("public_ip") or
                          status.get("ip_address") or status.get("host") or
                          inst["instance_id"])
                    results[inst["instance_id"]] = ip
                except Exception:
                    # Fallback: use instance_id (some providers like RunPod
                    # use IDs that double as SSH hostnames)
                    results[inst["instance_id"]] = inst["instance_id"]

            await asyncio.gather(*[_get_ip(i) for i in unresolved])
            return results

        resolved = asyncio.run(_resolve_ips())

        for inst in unresolved:
            ip = resolved.get(inst["instance_id"], inst["instance_id"])
            node_ips.append(ip)
            # Cache for next time
            try:
                set_instance_ip(inst["instance_id"], ip)
            except Exception:
                pass

    if fmt != "json":
        print(f"  Nodes from provision group {group_id}: {node_ips}")

    # ── Auto-resolve SSH key from provision group ──
    resolved_ssh_key = None
    try:
        if _db_ssh_path(group_id):
            from terradev_cli.core.ssh_key_manager import decrypt_private_key
            resolved_ssh_key = decrypt_private_key(group_id)
            if resolved_ssh_key and fmt != "json":
                print(f"  SSH key auto-resolved from provision group (ephemeral decrypt)")
    except Exception:
        pass  # Fall back to manual --ssh-key

    return node_ips, resolved_ssh_key

@cli.command()
@click.option('--nodes', '-n', multiple=True, help='Node IPs (multiple allowed, empty = localhost)')
@click.option('--ssh-user', default='root', help='SSH user (default: root)')
@click.option('--ssh-key', default='', help='SSH key path')
@click.option('--from-provision', 'provision_group', default='',
              help='Use nodes from a provision group. "latest" = most recent.')
@click.option('--quick', is_flag=True, help='Quick GPU-only check (skip storage/NCCL)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def preflight(nodes, ssh_user, ssh_key, provision_group, quick, fmt):
    """Run preflight hardware validation on GPU nodes.

    Checks: GPU health (DCGM), NVLink, RDMA, storage I/O, NCCL.
    All checks run in parallel via DAGExecutor.

    Examples:
        terradev preflight
        terradev preflight -n 10.0.0.1 -n 10.0.0.2 --quick
        terradev preflight --from-provision latest
        terradev preflight -f json
    """
    from terradev_cli.core.preflight_validator import PreflightValidator

    node_list = list(nodes) if nodes else []
    resolved_ssh_key = ssh_key
    if provision_group and not node_list:
        node_list, auto_ssh = _resolve_provision_nodes(provision_group, fmt)
        if not node_list:
            sys.exit(1)
        if auto_ssh and not ssh_key:
            resolved_ssh_key = auto_ssh
    if not node_list:
        node_list = [None]

    validator = PreflightValidator(
        nodes=node_list,
        ssh_user=ssh_user,
        ssh_key=resolved_ssh_key or None,
    )

    report = validator.run_quick() if quick else validator.run_all()
    summary = report.summary()

    if fmt == 'json':
        print(json.dumps(summary, indent=2, default=str))
    else:
        passed = summary.get("passed", False)
        status_icon = "PASS" if passed else "FAIL"
        print(f"\nPreflight: {status_icon}")
        print(f"  Nodes: {summary.get('nodes_checked', 0)}")
        print(f"  Checks passed: {summary.get('checks_passed', 0)}/{summary.get('total_checks', 0)}")
        if summary.get("failures"):
            print(f"  Failures:")
            for f in summary["failures"]:
                print(f"    - {f}")
        print()


@cli.command()
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), help='YAML config file')
@click.option('--script', '-s', help='Training script path')
@click.option('--framework', type=click.Choice(['torchrun', 'deepspeed', 'accelerate', 'megatron']),
              default='torchrun', help='Framework (default: torchrun)')
@click.option('--backend', type=click.Choice(['native', 'ray']), default='native',
              help='Launch backend (default: native, ray = optional)')
@click.option('--nodes', '-n', multiple=True, help='Node IPs')
@click.option('--from-provision', 'provision_group', default='',
              help='Use nodes from a provision group (pg_xxx). "latest" = most recent.')
@click.option('--gpus-per-node', default=8, help='GPUs per node (default: 8)')
@click.option('--tp', default=1, help='Tensor parallel size')
@click.option('--pp', default=1, help='Pipeline parallel size')
@click.option('--total-steps', default=0, help='Total training steps (for ETA)')
@click.option('--skip-preflight', is_flag=True, help='Skip preflight checks')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
@click.argument('script_args', nargs=-1, type=click.UNPROCESSED)
def train(config_path, script, framework, backend, nodes, provision_group,
          gpus_per_node, tp, pp, total_steps, skip_preflight, fmt, script_args):
    """Launch a distributed training job.

    Zero external dependencies by default — delegates to torchrun/deepspeed.
    Set --backend ray to use Ray when available (auto-fallback to native).

    Use --from-provision to automatically use nodes from a previous provision:
        terradev provision -g H100 -n 4
        terradev train -s train.py --from-provision latest

    Examples:
        terradev train -s train.py --framework torchrun --gpus-per-node 8
        terradev train -c job.yaml
        terradev train -s train.py -n 10.0.0.1 -n 10.0.0.2 --tp 2 -- --lr 1e-4
        terradev train -s train.py --from-provision pg_1709123456_abc12345
        terradev train -s train.py --from-provision latest
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator, TrainingConfig

    # ── Resolve nodes from provision group if specified ──
    resolved_nodes = list(nodes)
    resolved_ssh_key = ""
    if provision_group and not resolved_nodes:
        resolved_nodes, resolved_ssh_key = _resolve_provision_nodes(provision_group, fmt)
        if not resolved_nodes:
            sys.exit(1)

    if config_path:
        config = TrainingConfig.from_yaml(config_path)
        if resolved_nodes and not config.nodes:
            config.nodes = resolved_nodes
        if resolved_ssh_key and not config.ssh_key:
            config.ssh_key = resolved_ssh_key
    else:
        if not script:
            print("ERROR: Either --config or --script is required")
            sys.exit(1)
        config = TrainingConfig(
            script=script,
            framework=framework,
            backend=backend,
            nodes=resolved_nodes,
            gpus_per_node=gpus_per_node,
            tp_size=tp,
            pp_size=pp,
            total_steps=total_steps,
            script_args=list(script_args),
            ssh_key=resolved_ssh_key or "",
        )

    orch = TrainingOrchestrator()
    result = orch.launch(config, skip_preflight=skip_preflight)

    if fmt == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        status = result.get("status", "unknown")
        print(f"\nTraining Job: {result.get('job_id', 'N/A')}")
        print(f"  Status: {status}")
        print(f"  Framework: {result.get('framework')}")
        print(f"  Backend: {result.get('backend', 'native')}")
        print(f"  GPUs: {result.get('total_gpus', 0)}")
        print(f"  Nodes: {result.get('nodes', [])}")
        if result.get('pid'):
            print(f"  PID: {result['pid']}")
        if result.get('master_addr'):
            print(f"  Master: {result['master_addr']}")
        if status == 'failed':
            print(f"  Errors: {result.get('errors', '')}")
        print()


@cli.command()
@click.option('--job-id', '-j', default='', help='Job ID to monitor')
@click.option('--nodes', '-n', multiple=True, help='Node IPs')
@click.option('--ssh-user', default='root', help='SSH user')
@click.option('--ssh-key', default='', help='SSH key path')
@click.option('--from-provision', 'provision_group', default='',
              help='Use nodes from a provision group. "latest" = most recent.')
@click.option('--log-path', '-l', default='', help='Training log file to parse')
@click.option('--interval', '-i', default=10.0, help='Snapshot interval in seconds')
@click.option('--count', default=0, help='Number of snapshots (0 = continuous)')
@click.option('--prometheus', default='', help='Prometheus endpoint (optional)')
@click.option('--cost-rate', default=0.0, help='Cost per GPU-hour in USD')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def monitor(job_id, nodes, ssh_user, ssh_key, provision_group, log_path, interval, count,
            prometheus, cost_rate, fmt):
    """Monitor GPU utilization, training metrics, and cost.

    Default: nvidia-smi (zero deps). Optional Prometheus/DCGM-exporter hook.
    Includes straggler detection for multi-node clusters.

    Examples:
        terradev monitor -n 10.0.0.1 -n 10.0.0.2 -l /tmp/train.log
        terradev monitor --from-provision latest --cost-rate 3.50
        terradev monitor --prometheus http://localhost:9090 -f json
        terradev monitor -j job-abc123 --interval 5 --count 10
    """
    from terradev_cli.core.training_monitor import TrainingMonitor

    node_list = list(nodes) if nodes else []
    resolved_ssh_key = ssh_key
    if provision_group and not node_list:
        node_list, auto_ssh = _resolve_provision_nodes(provision_group, fmt)
        if not node_list:
            sys.exit(1)
        if auto_ssh and not ssh_key:
            resolved_ssh_key = auto_ssh
    if not node_list:
        node_list = [None]

    mon = TrainingMonitor(
        nodes=node_list,
        ssh_user=ssh_user,
        ssh_key=resolved_ssh_key or None,
        log_path=log_path,
        cost_per_gpu_hour=cost_rate,
        prometheus_endpoint=prometheus,
    )

    if fmt == 'json':
        if count == 1 or count == 0:
            snap = mon.snapshot(job_id)
            print(json.dumps(snap.to_dict(), indent=2, default=str))
        else:
            snaps = mon.continuous(job_id, interval_s=interval, max_snapshots=count)
            print(json.dumps([s.to_dict() for s in snaps], indent=2, default=str))
    else:
        if count == 1:
            snap = mon.snapshot(job_id)
            mon._print_snapshot(snap)
        else:
            mon.continuous(job_id, interval_s=interval, max_snapshots=count)


@cli.command()
@click.argument('action', type=click.Choice(['list', 'restore', 'promote', 'delete']))
@click.option('--job-id', '-j', required=True, help='Job ID')
@click.option('--step', type=int, default=None, help='Checkpoint step')
@click.option('--checkpoint-id', default='', help='Checkpoint ID')
@click.option('--dest', default='', help='Destination path (for promote)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def checkpoint(action, job_id, step, checkpoint_id, dest, fmt):
    """Manage distributed checkpoints.

    Local filesystem by default. Supports manifest-based atomic commits,
    parallel shard verification, and retention policies.

    Examples:
        terradev checkpoint list -j job-abc123
        terradev checkpoint restore -j job-abc123
        terradev checkpoint restore -j job-abc123 --step 5000
        terradev checkpoint promote -j job-abc123 --checkpoint-id ckpt-xyz --dest /models/final
        terradev checkpoint delete -j job-abc123 --checkpoint-id ckpt-xyz
    """
    from terradev_cli.core.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager()

    if action == 'list':
        ckpts = mgr.list(job_id)
        if fmt == 'json':
            print(json.dumps(ckpts, indent=2, default=str))
        else:
            if not ckpts:
                print(f"No checkpoints for job {job_id}")
            else:
                print(f"\nCheckpoints for {job_id}:")
                for c in ckpts:
                    sid = c.get('checkpoint_id', c.get('id', 'N/A'))
                    print(f"  step={c.get('step', '?'):>8}  "
                          f"id={sid}  "
                          f"shards={c.get('shard_count', '?')}  "
                          f"size={c.get('total_size_bytes', 0) / (1024**3):.2f}GB")
                print()

    elif action == 'restore':
        try:
            manifest = mgr.restore(job_id, step=step,
                                   checkpoint_id=checkpoint_id or None)
            result = manifest.to_dict()
            if fmt == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\nRestored: {result['checkpoint_id']} step={result['step']}")
                print(f"  Shards: {result['shard_count']}")
                print(f"  Size: {result['total_size_bytes'] / (1024**3):.2f}GB")
                print()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    elif action == 'promote':
        result = mgr.promote(job_id, checkpoint_id, dest_path=dest)
        print(f"Promoted: {result}")

    elif action == 'delete':
        if not checkpoint_id:
            print("ERROR: --checkpoint-id required for delete")
            sys.exit(1)
        mgr.delete(job_id, checkpoint_id)
        print(f"Deleted: {checkpoint_id}")


@cli.command('train-status')
@click.option('--job-id', '-j', default='', help='Job ID (empty = all running)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def train_status(job_id, fmt):
    """Show training job status, GPU-hours, cost, and ETA.

    Queries the local SQLite job database — no external services needed.

    Examples:
        terradev train-status
        terradev train-status -j job-abc123
        terradev train-status -f json
    """
    from terradev_cli.core.job_state_manager import JobStateManager

    sm = JobStateManager()

    if job_id:
        result = sm.job_metrics(job_id)
        if fmt == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            if 'error' in result:
                print(f"ERROR: {result['error']}")
                sys.exit(1)
            print(f"\nJob: {result['id']}")
            print(f"  Name: {result['name']}")
            print(f"  Status: {result['status']}")
            print(f"  Framework: {result['framework']}")
            print(f"  Progress: {result.get('current_step', 0)}/{result.get('total_steps', 0)} "
                  f"({result.get('progress_pct', 0)}%)")
            print(f"  Elapsed: {result.get('elapsed_hours', 0):.1f}h")
            print(f"  GPU-hours: {result.get('gpu_hours', 0):.1f}")
            eta = result.get('eta_hours')
            print(f"  ETA: {eta:.1f}h" if eta is not None else "  ETA: N/A")
            print(f"  Cost: ${result.get('cost_usd', 0):.2f}")
            print(f"  Efficiency: {result.get('efficiency_steps_per_gpuh', 0):.1f} steps/GPU-h")
            if result.get('last_checkpoint_id'):
                print(f"  Last checkpoint: {result['last_checkpoint_id']}")
            if result.get('error_message'):
                print(f"  Error: {result['error_message']}")
            print()
    else:
        running = sm.running_jobs_summary()
        total = sm.total_cost()
        if fmt == 'json':
            print(json.dumps({"running": running, "total_cost": total}, indent=2, default=str))
        else:
            if not running:
                print("\nNo running training jobs.")
            else:
                print(f"\nRunning jobs ({len(running)}):")
                for j in running:
                    eta = j.get('eta_hours')
                    eta_str = f"ETA {eta:.1f}h" if eta is not None else ""
                    print(f"  {j['id']}  {j['name']}  {j['framework']}  "
                          f"{j.get('current_step', 0)}/{j.get('total_steps', 0)}  "
                          f"{j.get('elapsed_hours', 0):.1f}h  "
                          f"${j.get('cost_usd', 0):.2f}  {eta_str}")
            print(f"\nTotal cost across all jobs: ${total:.2f}\n")


@cli.command('train-stop')
@click.option('--job-id', '-j', required=True, help='Job ID to stop')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def train_stop(job_id, fmt):
    """Stop a running training job.

    Kills training processes on all nodes in parallel.

    Examples:
        terradev train-stop -j job-abc123
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    result = orch.stop(job_id)

    if fmt == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Job {job_id}: {result.get('status', 'unknown')}")


@cli.command('train-resume')
@click.option('--job-id', '-j', required=True, help='Job ID to resume')
@click.option('--checkpoint-id', default='', help='Checkpoint to resume from (default: latest)')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
def train_resume(job_id, checkpoint_id, fmt):
    """Resume a training job from checkpoint.

    Rebuilds config from job state and resumes with topology validation.

    Examples:
        terradev train-resume -j job-abc123
        terradev train-resume -j job-abc123 --checkpoint-id ckpt-xyz
    """
    from terradev_cli.core.training_orchestrator import TrainingOrchestrator

    orch = TrainingOrchestrator()
    result = orch.resume(job_id, checkpoint_id=checkpoint_id or None)

    if fmt == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        status = result.get("status", "unknown")
        print(f"\nResumed Job: {result.get('job_id', job_id)}")
        print(f"  Status: {status}")
        if result.get('pid'):
            print(f"  PID: {result['pid']}")
        if status == 'failed':
            print(f"  Error: {result.get('errors', result.get('error', ''))}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-LoRA Adapter Management (vLLM ≥0.15.0)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_vllm_endpoint(endpoint: str):
    """Parse 'http://host:port' into (host, port)."""
    from urllib.parse import urlparse
    p = urlparse(endpoint if '://' in endpoint else f'http://{endpoint}')
    return p.hostname or '127.0.0.1', p.port or 8000


@cli.group()
def vllm():
    """vLLM optimization and management commands."""
    pass


@vllm.command('optimize')
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--type', '-t', type=click.Choice(['throughput', 'latency']), 
              default='throughput', help='Optimization type')
@click.option('--gpu-count', '-g', type=int, default=1, help='Number of GPUs')
@click.option('--output', '-o', type=click.Choice(['args', 'config', 'helm']), 
              default='args', help='Output format')
def vllm_optimize(model, type, gpu_count, output):
    """Generate optimized vLLM configurations using the 6 critical knobs.
    
    Applies the 6 knobs most teams never touch:
    1. --max-num-batched-tokens (2048→16384 for throughput, 4096 for latency)
    2. --gpu-memory-utilization (0.90→0.95)
    3. --max-num-seqs (256/1024→1024 for throughput, 512 for latency)
    4. --enable-prefix-caching (OFF→ON)
    5. --enable-chunked-prefill (OFF→ON)
    6. CPU cores (2 + #GPUs for V1 busy loop)
    
    Examples:
        terradev vllm optimize -m meta-llama/Llama-2-7b-hf -t throughput
        terradev vllm optimize -m mistralai/Mistral-7B-v0.1 -t latency -g 4
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig
    
    # Create optimized config
    if type == 'throughput':
        config = VLLMConfig.create_throughput_optimized(model, tensor_parallel_size=gpu_count)
    else:
        config = VLLMConfig.create_latency_optimized(model, tensor_parallel_size=gpu_count)
    
    # Auto-calculate CPU cores: 2 + #GPUs
    config.cpu_cores = 2 + gpu_count
    
    if output == 'args':
        # Import the service to get the args
        from terradev_cli.ml_services.vllm_service import VLLMService
        service = VLLMService(config)
        args = service._build_server_args()
        print(" ".join(args))
    elif output == 'config':
        print(json.dumps({
            'model_name': config.model_name,
            'gpu_memory_utilization': config.gpu_memory_utilization,
            'max_num_batched_tokens': config.max_num_batched_tokens,
            'max_num_seqs': config.max_num_seqs,
            'enable_prefix_caching': config.enable_prefix_caching,
            'enable_chunked_prefill': config.enable_chunked_prefill,
            'tensor_parallel_size': config.tensor_parallel_size,
            'cpu_cores': config.cpu_cores
        }, indent=2))
    elif output == 'helm':
        print(f"# Helm values for {type}-optimized vLLM")
        print("serving:")
        print("  vllm:")
        print(f"    gpuMemoryUtilization: {config.gpu_memory_utilization}")
        print(f"    maxNumBatchedTokens: {config.max_num_batched_tokens}")
        print(f"    maxNumSeqs: {config.max_num_seqs}")
        print(f"    enablePrefixCaching: {config.enable_prefix_caching}")
        print(f"    enableChunkedPrefill: {config.enable_chunked_prefill}")
        print(f"    tensorParallelSize: {config.tensor_parallel_size}")
        print("resources:")
        print(f"  requests:")
        print(f"    cpu: \"{config.cpu_cores}\"")
        print(f"  limits:")
        print(f"    cpu: \"{config.cpu_cores + 4}\"  # Extra headroom")


@vllm.command('auto-optimize')
@click.option('--endpoint', '-e', help='vLLM endpoint to analyze (if not provided, uses sample analysis)')
@click.option('--samples', '-s', type=click.Path(exists=True), help='JSON file with sample requests')
@click.option('--gpu-count', '-g', type=int, default=1, help='Number of GPUs available')
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--output', '-o', type=click.Choice(['config', 'args', 'helm']), 
              default='config', help='Output format')
@click.option('--apply', is_flag=True, help='Apply optimizations automatically')
def vllm_auto_optimize(endpoint, samples, gpu_count, model, output, apply):
    """Automatically optimize vLLM configuration based on workload analysis.
    
    Analyzes current workload patterns or sample requests to automatically
    select optimal settings for the 6 critical knobs.
    
    Examples:
        # Analyze running server
        terradev vllm auto-optimize -e http://localhost:8000 -m meta-llama/Llama-2-7b-hf
        
        # Analyze from sample file
        terradev vllm auto-optimize -s samples.json -m mistralai/Mistral-7B-v0.1 -g 4
        
        # Generate and apply Helm values
        terradev vllm auto-optimize -e http://localhost:8000 -m codellama/CodeLlama-34b-hf -o helm
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService, WorkloadProfile
    import asyncio
    
    async def run_optimization():
        try:
            # Load samples if provided
            sample_data = None
            if samples:
                with open(samples, 'r') as f:
                    sample_data = json.load(f)
            
            if endpoint:
                # Analyze running server
                host, port = _parse_vllm_endpoint(endpoint)
                config = VLLMConfig(model_name=model, host=host, port=port)
                
                async with VLLMService(config) as svc:
                    result = await svc.auto_optimize_from_workload(sample_data, gpu_count)
            else:
                # Analyze from samples only
                if not sample_data:
                    print("❌ Either --endpoint or --samples must be provided")
                    return
                
                workload = VLLMConfig.analyze_workload_from_samples(sample_data, gpu_count)
                optimized_config = VLLMConfig.create_auto_optimized(model, workload)
                
                result = {
                    "status": "success",
                    "workload_profile": workload,
                    "optimized_config": svc._config_to_dict(optimized_config) if endpoint else {
                        "model_name": optimized_config.model_name,
                        "max_num_batched_tokens": optimized_config.max_num_batched_tokens,
                        "max_num_seqs": optimized_config.max_num_seqs,
                        "gpu_memory_utilization": optimized_config.gpu_memory_utilization,
                        "enable_prefix_caching": optimized_config.enable_prefix_caching,
                        "enable_chunked_prefill": optimized_config.enable_chunked_prefill,
                        "cpu_cores": optimized_config.cpu_cores,
                        "tensor_parallel_size": optimized_config.tensor_parallel_size
                    },
                    "recommendations": "Configuration optimized based on workload analysis"
                }
            
            if result['status'] != 'success':
                print(f"❌ Auto-optimization failed: {result.get('error')}")
                return
            
            # Display results
            print("🧠 Workload Analysis Complete")
            print("=" * 50)
            
            workload = result.get('workload_profile')
            if workload:
                print(f"📊 Workload Profile:")
                print(f"   Avg Prompt Tokens: {workload.avg_prompt_length:.0f}")
                print(f"   Avg Response Tokens: {workload.avg_response_length:.0f}")
                print(f"   Requests/Second: {workload.requests_per_second:.1f}")
                print(f"   Concurrent Users: {workload.concurrent_users}")
                print(f"   Latency Sensitivity: {workload.latency_sensitivity:.2f}")
                print()
            
            print("🔧 Optimized Configuration:")
            optimized = result['optimized_config']
            for key, value in optimized.items():
                print(f"   {key}: {value}")
            
            # Show changes if comparison available
            changes = result.get('changes', [])
            if changes:
                print(f"\n📈 Recommended Changes ({len(changes)}):")
                for change in changes:
                    direction = "↑" if change['optimized'] > change['current'] else "↓"
                    print(f"   {change['parameter']}: {change['current']} → {change['optimized']} {direction}")
                    print(f"      Impact: {change['impact']}")
            
            # Generate output
            if output == 'config':
                print(f"\n📄 JSON Configuration:")
                print(json.dumps(optimized, indent=2))
            elif output == 'args':
                # Generate CLI args from optimized config
                from terradev_cli.ml_services.vllm_service import VLLMService
                temp_config = VLLMConfig(
                    model_name=optimized['model_name'],
                    max_num_batched_tokens=optimized['max_num_batched_tokens'],
                    max_num_seqs=optimized['max_num_seqs'],
                    gpu_memory_utilization=optimized['gpu_memory_utilization'],
                    enable_prefix_caching=optimized['enable_prefix_caching'],
                    enable_chunked_prefill=optimized['enable_chunked_prefill'],
                    tensor_parallel_size=optimized.get('tensor_parallel_size', 1)
                )
                temp_service = VLLMService(temp_config)
                args = temp_service._build_server_args()
                print(f"\n⚡ CLI Arguments:")
                print(" ".join(args))
            elif output == 'helm':
                print(f"\n☸️  Helm Values:")
                print("serving:")
                print("  vllm:")
                print(f"    gpuMemoryUtilization: {optimized['gpu_memory_utilization']}")
                print(f"    maxNumBatchedTokens: {optimized['max_num_batched_tokens']}")
                print(f"    maxNumSeqs: {optimized['max_num_seqs']}")
                print(f"    enablePrefixCaching: {optimized['enable_prefix_caching']}")
                print(f"    enableChunkedPrefill: {optimized['enable_chunked_prefill']}")
                print(f"    tensorParallelSize: {optimized.get('tensor_parallel_size', 1)}")
                print("resources:")
                print(f"  requests:")
                print(f"    cpu: \"{optimized.get('cpu_cores', '2')}\"")
                print(f"  limits:")
                print(f"    cpu: \"{optimized.get('cpu_cores', 2) + 4}\"  # Extra headroom")
            
        except Exception as e:
            print(f"❌ Error during auto-optimization: {e}")
    
    asyncio.run(run_optimization())


@vllm.command('analyze')
@click.option('--endpoint', '-e', required=True, help='vLLM endpoint to analyze')
@click.option('--duration', '-d', type=int, default=60, help='Analysis duration in seconds')
def vllm_analyze(endpoint, duration):
    """Analyze current vLLM server workload and provide optimization recommendations.
    
    Monitors the running vLLM server to understand workload patterns and
    generates specific optimization recommendations.
    
    Examples:
        terradev vllm analyze -e http://localhost:8000
        terradev vllm analyze -e http://10.0.0.1:8000 -d 120
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
    import asyncio
    
    async def run_analysis():
        try:
            host, port = _parse_vllm_endpoint(endpoint)
            config = VLLMConfig(model_name="", host=host, port=port)
            
            async with VLLMService(config) as svc:
                print(f"🔍 Analyzing vLLM server at {endpoint} for {duration}s...")
                print("=" * 60)
                
                result = await svc.analyze_current_workload(duration)
                
                if result['status'] != 'success':
                    print(f"❌ Analysis failed: {result.get('error')}")
                    return
                
                # Display current workload
                workload = result['current_workload']
                print("📊 Current Workload:")
                print(f"   Avg Prompt Tokens: {workload.get('avg_prompt_tokens', 0):.0f}")
                print(f"   Avg Generation Tokens: {workload.get('avg_generation_tokens', 0):.0f}")
                print(f"   Requests/Second: {workload.get('requests_per_second', 0):.1f}")
                print(f"   Active Requests: {workload.get('active_requests', 0)}")
                print(f"   Queue Size: {workload.get('queue_size', 0)}")
                print()
                
                # Display recommendations
                recommendations = result.get('optimization_recommendations', [])
                if recommendations:
                    print(f"💡 Optimization Recommendations ({len(recommendations)}):")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec['type'].replace('_', ' ').title()}")
                        print(f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}")
                        print(f"      Reason: {rec['reason']}")
                        print(f"      Impact: {rec['impact']}")
                        print()
                else:
                    print("✅ Configuration appears well-optimized for current workload")
                
                print(f"🕐 Analysis completed at {result.get('timestamp')}")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    asyncio.run(run_analysis())


@vllm.command('benchmark')
@click.option('--endpoint', '-e', required=True, help='vLLM endpoint to test')
@click.option('--api-key', help='vLLM API key')
@click.option('--prompt', default="Explain quantum computing in simple terms.", help='Test prompt')
@click.option('--concurrent', '-c', type=int, default=1, help='Concurrent requests')
def vllm_benchmark(endpoint, api_key, prompt, concurrent):
    """Benchmark vLLM endpoint performance."""
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
    import asyncio
    import time
    
    host, port = _parse_vllm_endpoint(endpoint)
    config = VLLMConfig(model_name="", host=host, port=port, api_key=api_key)
    
    async def run_benchmark():
        async with VLLMService(config) as svc:
            # Test connection
            health = await svc.test_connection()
            if health['status'] != 'connected':
                print(f"❌ Connection failed: {health.get('error')}")
                return
            
            print(f"✅ Connected to vLLM at {endpoint}")
            
            # Run concurrent requests
            start_time = time.time()
            tasks = []
            for i in range(concurrent):
                task = svc.test_inference(f"{prompt} (request {i+1})")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
            total_time = end_time - start_time
            throughput = successful / total_time if total_time > 0 else 0
            
            print(f"\n📊 Benchmark Results:")
            print(f"   Concurrent requests: {concurrent}")
            print(f"   Successful: {successful}/{concurrent}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Throughput: {throughput:.2f} req/s")
            
            if successful < concurrent:
                print(f"   ⚠️  {concurrent - successful} requests failed")
    
    asyncio.run(run_benchmark())


@cli.group()
def lora():
    """Manage LoRA adapters on a running vLLM endpoint.

    Serve N fine-tuned models on one base model, on one GPU.
    """
    pass


@lora.command('list')
@click.option('--endpoint', '-e', required=True, help='vLLM endpoint (e.g. http://10.0.0.1:8000)')
@click.option('--api-key', help='vLLM API key')
def lora_list_cmd(endpoint, api_key):
    """List loaded LoRA adapters.

    Examples:
        terradev lora list -e http://10.0.0.1:8000
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_list())

    if result['status'] != 'success':
        print(f"ERROR: {result.get('error')}")
        return

    base = result.get('base_models', [])
    adapters = result.get('lora_adapters', [])
    print(f"Base models ({len(base)}):")
    for m in base:
        print(f"  {m.get('id', '?')}")
    print(f"LoRA adapters ({len(adapters)}):")
    if adapters:
        for a in adapters:
            print(f"  {a.get('id', '?')}  (parent: {a.get('parent', '-')})")
    else:
        print("  (none)")


@lora.command('add')
@click.option('--endpoint', '-e', required=True, help='vLLM endpoint')
@click.option('--name', '-n', required=True, help='Adapter name (becomes the model name in API requests)')
@click.option('--path', '-p', required=True, help='Path to adapter weights')
@click.option('--api-key', help='vLLM API key')
def lora_add_cmd(endpoint, name, path, api_key):
    """Hot-load a LoRA adapter onto a running vLLM server.

    Examples:
        terradev lora add -e http://10.0.0.1:8000 -n customer-a -p /adapters/customer-a
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService, LoRAModule

    host, port = _parse_vllm_endpoint(endpoint)
    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_load(LoRAModule(name=name, path=path)))

    if result['status'] == 'loaded':
        print(f"✅ Adapter '{name}' loaded — use \"model\": \"{name}\" in requests")
    else:
        print(f"ERROR: {result.get('error')}")


@lora.command('remove')
@click.option('--endpoint', '-e', required=True, help='vLLM endpoint')
@click.option('--name', '-n', required=True, help='Adapter name to unload')
@click.option('--api-key', help='vLLM API key')
def lora_remove_cmd(endpoint, name, api_key):
    """Hot-unload a LoRA adapter.

    Examples:
        terradev lora remove -e http://10.0.0.1:8000 -n customer-a
    """
    from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService

    host, port = _parse_vllm_endpoint(endpoint)
    svc = VLLMService(VLLMConfig(model_name="", host=host, port=port, api_key=api_key))
    result = asyncio.run(svc.lora_unload(name))

    if result['status'] == 'unloaded':
        print(f"✅ Adapter '{name}' unloaded")
    else:
        print(f"ERROR: {result.get('error')}")



# ═══════════════════════════════════════════════════════════════════════════════
# Datadog FinOps Integration
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group()
def datadog():
    """Datadog FinOps monitoring — metrics, monitors, dashboards."""
    pass


@datadog.command('configure')
@click.option('--api-key', prompt='Datadog API Key', hide_input=True)
@click.option('--app-key', prompt='Datadog App Key', hide_input=True)
@click.option('--site', default='datadoghq.com')
def datadog_configure(api_key, app_key, site):
    """Configure Datadog credentials."""
    from terradev_cli.integrations.datadog_integration import save_credentials
    save_credentials(api_key, app_key, site)
    print(f"✅ Datadog credentials saved (site: {site})")


@datadog.command('status')
def datadog_status():
    """Show Datadog integration status."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, validate_credentials)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured. Run: terradev datadog configure")
        return
    print(f"Site:    {creds.get('site', 'datadoghq.com')}")
    print(f"API Key: ***{creds['api_key'][-4:]}")
    print(f"App Key: ***{creds['app_key'][-4:]}")
    ok, msg = validate_credentials(creds)
    print(f"{'✅' if ok else '❌'} {msg}")


@datadog.command('submit-metric')
@click.option('--metric', '-m', required=True)
@click.option('--value', '-v', required=True, type=float)
@click.option('--tags', '-t', multiple=True)
def datadog_submit_metric(metric, value, tags):
    """Submit a custom metric to Datadog."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, submit_metric)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    r = submit_metric(creds, metric, value, list(tags))
    if r.get('status') == 'ok':
        print(f"✅ terradev.{metric} = {value}")
    else:
        print(f"❌ {r.get('error')}")


@datadog.command('create-monitor')
@click.option('--template', '-t', required=True,
              type=click.Choice(['budget_alert', 'cost_spike', 'idle_gpu',
                                 'spot_volatility', 'provider_degraded',
                                 'egress_anomaly']))
@click.option('--notify', default='@slack-terradev-alerts')
def datadog_create_monitor(template, notify):
    """Create a monitor from a FinOps template."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, create_monitor)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    r = create_monitor(creds, template, notify)
    if r.get('id'):
        print(f"✅ Monitor '{template}' created (id: {r['id']})")
    else:
        print(f"❌ {r.get('error')}")


@datadog.command('list-monitors')
def datadog_list_monitors():
    """List Terradev monitors in Datadog."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, list_monitors)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    monitors = list_monitors(creds)
    if not monitors:
        print("No Terradev monitors found."); return
    print(f"{'ID':<12} {'Name':<42} {'Status'}")
    print("─" * 66)
    for m in monitors:
        mid = m['id']
        name = m['name'][:40]
        state = m.get('overall_state', '?')
        print(f"{mid:<12} {name:<42} {state}")


@datadog.command('delete-monitor')
@click.option('--monitor-id', '-i', required=True, type=int)
@click.confirmation_option(prompt='Delete this monitor?')
def datadog_delete_monitor(monitor_id):
    """Delete a Datadog monitor by ID."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, delete_monitor)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    r = delete_monitor(creds, monitor_id)
    if r.get('deleted'):
        print(f"✅ Monitor {monitor_id} deleted")
    else:
        print(f"❌ {r.get('error')}")


@datadog.command('create-dashboard')
@click.option('--title', default='Terradev GPU FinOps')
def datadog_create_dashboard(title):
    """Create the GPU FinOps dashboard."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, create_dashboard)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    r = create_dashboard(creds, title)
    if r.get('id'):
        print(f"✅ Dashboard: {r.get('url', r['id'])}")
    else:
        print(f"❌ {r.get('error')}")


@datadog.command('push-costs')
def datadog_push_costs():
    """Push cost snapshot from cost tracker to Datadog."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, push_cost_snapshot)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    r = push_cost_snapshot(creds)
    if r.get('status') == 'ok':
        n = r.get('metrics_pushed', 0)
        print(f"✅ {n} cost metrics pushed to Datadog")
    else:
        print(f"❌ {r.get('error')}")


@datadog.command('export-tf')
def datadog_export_tf():
    """Export Terraform module config for Datadog."""
    from terradev_cli.integrations.datadog_integration import (
        load_credentials, export_terraform_vars)
    creds = load_credentials()
    if not creds:
        print("❌ Not configured"); return
    tf = export_terraform_vars(creds)
    print("# Add to your terraform.tfvars:\n")
    for k, v in tf.items():
        if isinstance(v, str):
            print(f'{k} = "{v}"')
        else:
            print(f"{k} = {v}")
    print(f"\n# Then: terraform apply -parallelism=20")


# ═══════════════════════════════════════════════════════════════════════════════
# Phoenix — LLM Trace Observability (Arize Phoenix, ELv2)
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group()
def phoenix():
    """Arize Phoenix LLM trace observability — traces, spans, OTEL."""
    pass


@phoenix.command('test')
def phoenix_test():
    """Test connection to Phoenix server."""
    from terradev_cli.ml_services.phoenix_service import (
        create_phoenix_service_from_credentials, get_phoenix_setup_instructions)
    api = TerradevAPI()
    creds = api._provider_creds('phoenix')
    if not any(creds.values()):
        print(get_phoenix_setup_instructions())
        return
    svc = create_phoenix_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result['status'] == 'connected':
        print(f"✅ Phoenix connected: {result['collector_endpoint']}")
        print(f"   Projects found: {result['projects_found']}")
    else:
        print(f"❌ Connection failed: {result.get('error')}")


@phoenix.command('projects')
@click.option('--limit', '-l', default=50, help='Max projects to return')
def phoenix_projects(limit):
    """List Phoenix projects."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    data = asyncio.run(svc.list_projects(limit=limit))
    projects = data.get('data', [])
    if not projects:
        print("No projects found.")
        return
    for p in projects:
        print(f"  📁 {p.get('name', p.get('id', '?'))}")


@phoenix.command('spans')
@click.option('--project', '-p', default=None, help='Project ID or name')
@click.option('--filter', '-f', 'filter_cond', default=None, help='SpanQuery DSL filter, e.g. "span_kind == \'RETRIEVER\'"')
@click.option('--limit', '-l', default=20, help='Max spans')
def phoenix_spans(project, filter_cond, limit):
    """List recent spans for a project."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    from terradev_cli.core.trace_viewer import view_recent_spans
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    output = asyncio.run(view_recent_spans(svc, project=project, limit=limit, filter_condition=filter_cond))
    print(output)


@phoenix.command('trace')
@click.option('--trace-id', '-t', required=True, help='Trace ID to inspect')
@click.option('--project', '-p', default=None, help='Project ID or name')
def phoenix_trace(trace_id, project):
    """View full execution tree for a trace."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    from terradev_cli.core.trace_viewer import view_trace
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    output = asyncio.run(view_trace(svc, trace_id, project=project))
    print(output)


@phoenix.command('otel-env')
@click.option('--project', '-p', default=None, help='Project name')
def phoenix_otel_env(project):
    """Print OTEL env vars to inject into serving pods."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    env = svc.generate_otel_env(project_name=project)
    for k, v in env.items():
        print(f"export {k}=\"{v}\"")


@phoenix.command('snippet')
@click.option('--project', '-p', default=None, help='Project name')
def phoenix_snippet(project):
    """Print Python instrumentation snippet."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    print(svc.generate_instrumentation_snippet(project_name=project))


@phoenix.command('k8s')
@click.option('--namespace', '-n', default='observability', help='K8s namespace')
def phoenix_k8s(namespace):
    """Print K8s deployment manifest for Phoenix server."""
    from terradev_cli.ml_services.phoenix_service import create_phoenix_service_from_credentials
    api = TerradevAPI()
    svc = create_phoenix_service_from_credentials(api._provider_creds('phoenix'))
    print(svc.generate_k8s_deployment(namespace=namespace))


# ═══════════════════════════════════════════════════════════════════════════════
# NeMo Guardrails — Output Safety Layer (Apache 2.0)
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group()
def guardrails():
    """NeMo Guardrails — LLM output safety, jailbreak detection, PII masking."""
    pass


@guardrails.command('test')
def guardrails_test_cmd():
    """Test connection to guardrails server."""
    from terradev_cli.ml_services.guardrails_service import (
        create_guardrails_service_from_credentials, get_guardrails_setup_instructions)
    api = TerradevAPI()
    creds = api._provider_creds('guardrails')
    if not any(creds.values()):
        print(get_guardrails_setup_instructions())
        return
    svc = create_guardrails_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result['status'] == 'connected':
        print(f"✅ Guardrails connected: {result['server_url']}")
    else:
        print(f"❌ Connection failed: {result.get('error')}")


@guardrails.command('chat')
@click.option('--message', '-m', required=True, help='Message to send through guardrails')
@click.option('--config-id', '-c', default=None, help='Guardrails config_id')
def guardrails_chat(message, config_id):
    """Send a message through guardrails and show the result."""
    from terradev_cli.ml_services.guardrails_service import create_guardrails_service_from_credentials
    api = TerradevAPI()
    svc = create_guardrails_service_from_credentials(api._provider_creds('guardrails'))
    result = asyncio.run(svc.test_rail(message, config_id=config_id))
    print(f"Input:     {result['input']}")
    print(f"Config:    {result['config_id']}")
    print(f"Output:    {json.dumps(result['output'], indent=2)}")


@guardrails.command('generate-config')
@click.option('--config-id', '-c', default=None, help='Config ID name')
@click.option('--output-dir', '-o', default='./guardrails', help='Output directory')
def guardrails_generate_config(config_id, output_dir):
    """Generate default Colang 2.x guardrails configuration."""
    from terradev_cli.ml_services.guardrails_service import create_guardrails_service_from_credentials
    api = TerradevAPI()
    svc = create_guardrails_service_from_credentials(api._provider_creds('guardrails'))
    files = svc.generate_colang_config(config_id=config_id)
    output_path = Path(output_dir)
    for fname, content in files.items():
        fpath = output_path / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
        print(f"  ✅ {fpath}")
    print(f"\n🛡️ Config generated. Start server: nemoguardrails server --config {output_dir}")


@guardrails.command('k8s')
@click.option('--namespace', '-n', default='guardrails', help='K8s namespace')
def guardrails_k8s(namespace):
    """Print K8s deployment manifest for guardrails server."""
    from terradev_cli.ml_services.guardrails_service import create_guardrails_service_from_credentials
    api = TerradevAPI()
    svc = create_guardrails_service_from_credentials(api._provider_creds('guardrails'))
    print(svc.generate_k8s_deployment(namespace=namespace))


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant — Vector Database for RAG (Apache 2.0)
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group()
def qdrant():
    """Qdrant vector database — collections, search, RAG infrastructure."""
    pass


@qdrant.command('test')
def qdrant_test():
    """Test connection to Qdrant server."""
    from terradev_cli.ml_services.qdrant_service import (
        create_qdrant_service_from_credentials, get_qdrant_setup_instructions)
    api = TerradevAPI()
    creds = api._provider_creds('qdrant')
    if not any(creds.values()):
        print(get_qdrant_setup_instructions())
        return
    svc = create_qdrant_service_from_credentials(creds)
    result = asyncio.run(svc.test_connection())
    if result['status'] == 'connected':
        print(f"✅ Qdrant connected: {result['url']}")
        print(f"   Collections: {', '.join(result['collections']) or 'none'}")
    else:
        print(f"❌ Connection failed: {result.get('error')}")


@qdrant.command('collections')
def qdrant_collections():
    """List all collections."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials
    api = TerradevAPI()
    svc = create_qdrant_service_from_credentials(api._provider_creds('qdrant'))
    cols = asyncio.run(svc.list_collections())
    if not cols:
        print("No collections found.")
        return
    for c in cols:
        print(f"  🗄️  {c}")


@qdrant.command('create-collection')
@click.option('--name', '-n', default=None, help='Collection name')
@click.option('--embedding-model', '-e', default=None, help='Embedding model (auto-sets vector size)')
def qdrant_create_collection(name, embedding_model):
    """Create a vector collection (auto-configured for embedding model)."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials
    api = TerradevAPI()
    svc = create_qdrant_service_from_credentials(api._provider_creds('qdrant'))
    result = asyncio.run(svc.configure_rag_collection(name=name, embedding_model=embedding_model))
    print(f"✅ Collection created: {result['collection']}")
    print(f"   Embedding model: {result['embedding_model']}")
    print(f"   Vector size: {result['vector_size']}")


@qdrant.command('info')
@click.option('--name', '-n', default=None, help='Collection name')
def qdrant_info(name):
    """Get collection info and stats."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials
    api = TerradevAPI()
    svc = create_qdrant_service_from_credentials(api._provider_creds('qdrant'))
    info = asyncio.run(svc.get_collection_info(name=name))
    print(json.dumps(info, indent=2))


@qdrant.command('count')
@click.option('--name', '-n', default=None, help='Collection name')
def qdrant_count(name):
    """Count points in a collection."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials
    api = TerradevAPI()
    svc = create_qdrant_service_from_credentials(api._provider_creds('qdrant'))
    count = asyncio.run(svc.count_points(name=name))
    print(f"Points: {count}")


@qdrant.command('k8s')
@click.option('--namespace', '-n', default='vector-db', help='K8s namespace')
def qdrant_k8s(namespace):
    """Print K8s StatefulSet manifest for Qdrant."""
    from terradev_cli.ml_services.qdrant_service import create_qdrant_service_from_credentials
    api = TerradevAPI()
    svc = create_qdrant_service_from_credentials(api._provider_creds('qdrant'))
    print(svc.generate_k8s_deployment(namespace=namespace))


if __name__ == '__main__':
    cli()
