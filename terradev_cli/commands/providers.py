#!/usr/bin/env python3
"""Provider configuration and profile management commands."""

import asyncio
import json
from pathlib import Path
import logging

import click

from . import cli
from ._api import (
    validate_credentials,
    run_interactive_onboarding,
    _telemetry,
)

logger = logging.getLogger(__name__)


class ProvidersCommand(click.Command):
    """Click command that catches runtime failures and returns non-zero on errors."""

    def invoke(self, ctx):
        try:
            rv = super().invoke(ctx)
        except (click.ClickException, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            click.echo(f"ERROR: {exc}", err=True)
            raise click.exceptions.Exit(1) from exc

        output = ctx.obj.get("terradev_output") if ctx.obj else None
        if output is not None and (rv is None or rv == 0):
            messages = getattr(output, "_messages", [])
            if any(m.level == "error" for m in messages):
                raise click.exceptions.Exit(1)
        return rv


class ProvidersGroup(click.Group):
    """Click group that uses ProvidersCommand for leaf subcommands and ProvidersGroup for nested groups."""

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", ProvidersCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", ProvidersGroup)
        return super().group(*args, **kwargs)



# Upgrade command removed - tier system eliminated (open source CLI)
# @cli.command()
# @click.option('--tier', '-t', type=click.Choice(['research_plus', 'enterprise', 'enterprise_plus']),
#               help='Tier to upgrade to')
# @click.option('--activate', is_flag=True,
#               help='Activate after payment')
# @click.option('--email', help='Email used for checkout (for --activate)')
# def upgrade(tier, activate, email):
#     """Upgrade your Terradev subscription - REMOVED (tier system eliminated)"""
#     pass

# Entire upgrade function body removed - tier system eliminated (open source CLI)


@cli.command(cls=ProvidersCommand)
@click.option(
    "--provider", "-p", help="Configure specific provider (e.g., runpod, vastai, aws)"
)
def configure(provider):
    """Configure cloud provider credentials for GPU provisioning.

    Stores API keys locally at ~/.terradev/credentials.json (never sent to Terradev servers).

    Examples:
      terradev configure --provider runpod
      terradev configure --provider aws
      terradev configure              # Interactive mode for all providers

    Quick Start:
      RunPod is the easiest to set up (5 minutes): terradev setup runpod --quick
    """

    # Initialize API object for credential management
    # Must be instantiated before branching to ensure it's in scope for save_credentials()
    api = click.get_current_context().obj["api"]

    if provider:
        # Configure specific provider
        from terradev_cli.credential_prompt import prompt_for_credentials

        click.echo(f"   Configure {provider.upper()} credentials")

        # Use the injected API's config paths so tests can isolate credentials
        config_dir = getattr(api, "config_dir", None) or (Path.home() / ".terradev")
        credentials_file = getattr(api, "credentials_file", None) or (config_dir / "credentials.json")

        # Load existing credentials
        existing_creds = {}
        if credentials_file.exists():
            with open(credentials_file, "r") as f:
                existing_creds = json.load(f)

        # Provider configurations
        provider_configs = {
            "runpod": {
                "name": "RunPod",
                "key_name": "API Key",
                "help": "Get from: https://runpod.io/console/settings/api-keys",
                "example": "rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "vastai": {
                "name": "Vast.ai",
                "key_name": "API Key",
                "help": "Get from: https://console.vast.ai/api-keys",
                "example": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "aws": {
                "name": "AWS",
                "key_name": "Access Key ID",
                "help": "Get from: AWS IAM console",
                "example": "AKIAIOSFODNN7EXAMPLE",
            },
            "gcp": {
                "name": "Google Cloud",
                "key_name": "Service Account JSON",
                "help": "Get from: GCP Console → IAM & Admin → Service Accounts",
                "example": "path/to/service-account.json",
            },
            "azure": {
                "name": "Azure",
                "key_name": "Client ID",
                "help": "Azure Portal → Azure Active Directory → App Registrations → New Registration. Then Certificates & Secrets → New client secret. Subscription ID is under Subscriptions. Assign Contributor role.",
                "example": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            },
            "tensordock": {
                "name": "TensorDock",
                "key_name": "API Key",
                "help": "Get from: TensorDock dashboard",
                "example": "td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "crusoe": {
                "name": "Crusoe Cloud",
                "key_name": "API Key",
                "help": "Get from: Crusoe dashboard",
                "example": "crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "huggingface": {
                "name": "HuggingFace",
                "key_name": "API Token",
                "help": "Get from: https://huggingface.co/settings/tokens",
                "example": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "baseten": {
                "name": "Baseten",
                "key_name": "API Key",
                "help": "Get from: Baseten dashboard",
                "example": "bt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "siliconflow": {
                "name": "SiliconFlow",
                "key_name": "API Key",
                "help": "Get from: https://cloud.siliconflow.cn/account/ak",
                "example": "sf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "hyperstack": {
                "name": "Hyperstack",
                "key_name": "API Key",
                "help": "Get from: Hyperstack dashboard → API Keys",
                "example": "hs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "digitalocean": {
                "name": "DigitalOcean",
                "key_name": "API Token",
                "help": "Get from: DigitalOcean → API → Tokens",
                "example": "dop_v1_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "inferx": {
                "name": "InferX",
                "key_name": "API Key",
                "help": "Get from: InferX Console → endpoint Client Setup",
                "example": "ix_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "e2enetworks": {
                "name": "E2E Networks",
                "key_name": "API Key",
                "help": "Get from: https://e2enetworks.com",
                "example": "e2e_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "latitude": {
                "name": "Latitude.sh",
                "key_name": "API Key",
                "help": "Get from: https://latitude.sh/account/api-keys",
                "example": "lat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "yottalabs": {
                "name": "Yotta Labs",
                "key_name": "API Key",
                "help": "Get from: https://yottalabs.ai",
                "example": "yotta_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
            "gcore": {
                "name": "Gcore",
                "key_name": "API Token",
                "help": "Get from: Gcore Customer Portal → Profile → API Tokens",
                "example": "12345_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            },
        }

        config = provider_configs.get(provider.lower())
        if not config:
            click.echo(f"ERROR: Unknown provider '{provider}'", err=True)
            click.echo("\nAvailable providers:")
            for i, (key, val) in enumerate(provider_configs.items(), 1):
                click.echo(f"  {i}. {val['name']}")
            click.echo("\nUse 'terradev setup <provider>' for step-by-step instructions")
            click.echo("   Example: terradev setup runpod")
            raise SystemExit(1)

        click.echo(f"   {config['name']}")
        click.echo(f"   Help: {config['help']}")
        click.echo(f"   Example: {config['example']}")

        # Check if already configured
        existing_key = existing_creds.get(provider.lower(), {}).get("api_key")
        if existing_key:
            click.echo(f"   Already configured: {existing_key[:10]}...")
            if not click.confirm(
                f"   Update {config['name']} credentials?", default=False
            ):
                click.echo("   Your existing credentials will be used")
                return

        # Prompt for API key
        api_key = click.prompt(
            f"   Enter {config['key_name']}", hide_input=True, show_default=False
        )

        if api_key.strip():
            if provider.lower() == "gcp":
                # GCP needs special handling
                project_id = click.prompt(
                    "   Enter GCP Project ID", default="my-project"
                )
                existing_creds[provider.lower()] = {
                    "credentials_file": api_key.strip(),
                    "project_id": project_id,
                }
            elif provider.lower() == "aws":
                # AWS needs both access key and secret key
                secret_key = click.prompt(
                    "   Enter AWS Secret Access Key", hide_input=True
                )
                existing_creds[provider.lower()] = {
                    "api_key": api_key.strip(),
                    "secret_key": secret_key.strip(),
                }
            elif provider.lower() == "azure":
                # Azure needs multiple credentials
                subscription_id = click.prompt("   Enter Azure Subscription ID")
                tenant_id = click.prompt("   Enter Azure Tenant ID")
                client_id = click.prompt("   Enter Azure Client ID")
                client_secret = click.prompt(
                    "   Enter Azure Client Secret", hide_input=True
                )
                existing_creds[provider.lower()] = {
                    "subscription_id": subscription_id.strip(),
                    "tenant_id": tenant_id.strip(),
                    "client_id": client_id.strip(),
                    "client_secret": client_secret.strip(),
                }
            elif provider.lower() == "tensordock":
                # TensorDock needs API key + token
                api_token = click.prompt(
                    "   Enter TensorDock API Token", hide_input=True
                )
                existing_creds[provider.lower()] = {
                    "api_key": api_key.strip(),
                    "api_token": api_token.strip(),
                }
            elif provider.lower() == "crusoe":
                # Crusoe needs multiple credentials
                access_key = click.prompt("   Enter Crusoe Access Key")
                secret_key = click.prompt("   Enter Crusoe Secret Key", hide_input=True)
                project_id = click.prompt("   Enter Crusoe Project ID")
                existing_creds[provider.lower()] = {
                    "access_key": access_key.strip(),
                    "secret_key": secret_key.strip(),
                    "project_id": project_id.strip(),
                }
            elif provider.lower() == "huggingface":
                # HuggingFace needs API token + namespace
                namespace = click.prompt(
                    "   Enter HuggingFace Namespace (username or org)"
                )
                existing_creds[provider.lower()] = {
                    "api_key": api_key.strip(),
                    "namespace": namespace.strip(),
                }
            elif provider.lower() == "e2enetworks":
                # E2E Networks optionally supports a project ID
                project_id = click.prompt(
                    "   Enter E2E Networks Project ID (optional)", default=""
                )
                existing_creds[provider.lower()] = {
                    "api_key": api_key.strip(),
                    "project_id": project_id.strip(),
                }
            elif provider.lower() == "inferx":
                # InferX needs the base endpoint and default model from Console
                api_endpoint = click.prompt(
                    "   Enter InferX API Base URL",
                    default="https://model.inferx.net/endpoints/v1",
                )
                model = click.prompt(
                    "   Enter InferX default model",
                    default="Qwen3.8-27B-FP8",
                )
                existing_creds[provider.lower()] = {
                    "api_key": api_key.strip(),
                    "api_endpoint": api_endpoint.strip(),
                    "model": model.strip(),
                }
            else:
                existing_creds[provider.lower()] = {"api_key": api_key.strip()}

            # Save credentials
            with open(credentials_file, "w") as f:
                json.dump(existing_creds, f, indent=2)

            # Validate credentials
            if validate_credentials(provider, existing_creds[provider.lower()]):
                click.echo(f"   OK: {config['name']} credentials validated and saved")
                click.echo(
                    f"   Test with: terradev quote --gpu-type a100 --providers {provider}"
                )
                click.echo("   Then provision: terradev provision -g a100")
            else:
                click.echo(f"   ERROR: {config['name']} credentials validation failed")
                click.echo("   Please check your credentials and try again")
                click.echo(
                    f"   Use 'terradev setup {provider}' for step-by-step setup instructions"
                )

    else:
        # Interactive configuration for all providers
        from terradev_cli.credential_prompt import prompt_for_credentials

        configured_providers = prompt_for_credentials()

        # Reload credentials from file after prompt_for_credentials saves them
        # This ensures api.credentials includes the provider credentials
        api.load_credentials()

        if configured_providers:
            click.echo(f"\nReady to get quotes from: {', '.join(configured_providers)}")
            click.echo("   Try: terradev quote --gpu-type a100")
            click.echo("   Then provision: terradev provision -g a100")
        else:
            click.echo("\nNo providers configured")
            click.echo(
                "   Run 'terradev configure --provider <provider>' to add credentials"
            )
            click.echo(
                "   Or use 'terradev setup runpod' for the easiest setup (5 minutes)"
            )

        # Kubernetes configuration (optional)
        kubernetes_config = click.prompt(
            "Configure Kubernetes? (y/n)", default="n", show_default=False
        )
        if kubernetes_config.lower() == "y":
            kubernetes_namespace = click.prompt(
                "Kubernetes namespace (default: default)",
                default="default",
                show_default=False,
            )
            if kubernetes_namespace:
                api.credentials["kubernetes_namespace"] = kubernetes_namespace
            karpenter_enabled = click.prompt(
                "Enable Karpenter? (y/n)", default="n", show_default=False
            )
            if karpenter_enabled.lower() == "y":
                api.credentials["kubernetes_karpenter_enabled"] = "true"

            click.echo(
                "   Kubernetes configured  cluster management and Karpenter"
            )

        # W&B (enhanced)
        wandb_key = click.prompt(
            "W&B API Key (optional, from wandb.ai/settings)",
            hide_input=True,
            default="",
            show_default=False,
        )
        if wandb_key:
            api.credentials["wandb_api_key"] = wandb_key
            wandb_entity = click.prompt(
                "W&B Entity (team/username, optional)", default="", show_default=False
            )
            if wandb_entity:
                api.credentials["wandb_entity"] = wandb_entity
            wandb_project = click.prompt(
                "W&B Project (optional, default: terradev)",
                default="",
                show_default=False,
            )
            if wandb_project:
                api.credentials["wandb_project"] = wandb_project
            wandb_base_url = click.prompt(
                "W&B Server URL (optional, for self-hosted)",
                default="",
                show_default=False,
            )
            if wandb_base_url:
                api.credentials["wandb_base_url"] = wandb_base_url

            # Enhanced W&B options
            wandb_enhanced = click.prompt(
                "Enable enhanced W&B features (dashboards/reports/alerts)? (y/n)",
                default="n",
                show_default=False,
            )
            if wandb_enhanced.lower() == "y":
                api.credentials["wandb_dashboard_enabled"] = "true"
                api.credentials["wandb_reports_enabled"] = "true"
                api.credentials["wandb_alerts_enabled"] = "true"
                api.credentials["wandb_integration_enabled"] = "true"

            click.echo("   W&B configured  experiment tracking, dashboards, and alerts")

        # LangChain (enhanced)
        langchain_config = click.prompt(
            "Configure LangChain? (y/n)", default="n", show_default=False
        )
        if langchain_config.lower() == "y":
            langchain_key = click.prompt(
                "LangChain API Key (optional)",
                hide_input=True,
                default="",
                show_default=False,
            )
            if langchain_key:
                api.credentials["langchain_api_key"] = langchain_key

            # Enhanced LangChain options
            langchain_enhanced = click.prompt(
                "Enable enhanced LangChain features (dashboards/tracing/evaluation)? (y/n)",
                default="n",
                show_default=False,
            )
            if langchain_enhanced.lower() == "y":
                api.credentials["langchain_dashboard_enabled"] = "true"
                api.credentials["langchain_tracing_enabled"] = "true"
                api.credentials["langchain_evaluation_enabled"] = "true"
                api.credentials["langchain_workflow_enabled"] = "true"

            click.echo(
                "   LangChain configured  chains and workflows"
            )

        # SGLang
        sglang_config = click.prompt(
            "Configure SGLang? (y/n)", default="n", show_default=False
        )
        if sglang_config.lower() == "y":
            sglang_key = click.prompt(
                "SGLang API Key (optional)",
                hide_input=True,
                default="",
                show_default=False,
            )
            if sglang_key:
                api.credentials["sglang_api_key"] = sglang_key
            model_path = click.prompt(
                "SGLang Model Path (optional)", default="", show_default=False
            )
            if model_path:
                api.credentials["sglang_model_path"] = model_path

            # Enhanced SGLang options
            sglang_enhanced = click.prompt(
                "Enable enhanced SGLang features (dashboards/tracing/metrics)? (y/n)",
                default="n",
                show_default=False,
            )
            if sglang_enhanced.lower() == "y":
                api.credentials["sglang_dashboard_enabled"] = "true"
                api.credentials["sglang_tracing_enabled"] = "true"
                api.credentials["sglang_metrics_enabled"] = "true"
                api.credentials["sglang_deployment_enabled"] = "true"
                api.credentials["sglang_observability_enabled"] = "true"

            click.echo("   SGLang configured  model serving and optimization")

        # ── ML Platform Integrations ──
        click.echo("\nML Platform Integrations (optional)")

        # KServe
        kserve_config = click.prompt(
            "Configure KServe? (y/n)", default="n", show_default=False
        )
        if kserve_config.lower() == "y":
            kserve_namespace = click.prompt(
                "KServe Namespace (default: default)",
                default="default",
                show_default=False,
            )
            if kserve_namespace:
                api.credentials["kserve_namespace"] = kserve_namespace
            kserve_kubeconfig = click.prompt(
                "Kubeconfig path (optional, uses default)",
                default="",
                show_default=False,
            )
            if kserve_kubeconfig:
                api.credentials["kserve_kubeconfig_path"] = kserve_kubeconfig
            click.echo("   KServe configured  model deployment on Kubernetes")

        # DVC
        dvc_config = click.prompt(
            "Configure DVC? (y/n)", default="n", show_default=False
        )
        if dvc_config.lower() == "y":
            dvc_repo = click.prompt(
                "DVC Repository Path (default: .)", default=".", show_default=False
            )
            if dvc_repo:
                api.credentials["dvc_repo_path"] = dvc_repo
            dvc_remote = click.prompt(
                "DVC Remote Storage (optional)", default="", show_default=False
            )
            if dvc_remote:
                api.credentials["dvc_remote_storage"] = dvc_remote
            dvc_type = click.prompt(
                "DVC Remote Type (s3, gs, azure, ssh)", default="", show_default=False
            )
            if dvc_type:
                api.credentials["dvc_remote_type"] = dvc_type
            click.echo("   DVC configured  data versioning and storage")

        # MLflow
        mlflow_uri = click.prompt(
            "MLflow Tracking URI (optional)", default="", show_default=False
        )
        if mlflow_uri:
            api.credentials["mlflow_tracking_uri"] = mlflow_uri
            mlflow_user = click.prompt(
                "MLflow Username (optional)",
                hide_input=True,
                default="",
                show_default=False,
            )
            if mlflow_user:
                api.credentials["mlflow_username"] = mlflow_user
            mlflow_pass = click.prompt(
                "MLflow Password (optional)",
                hide_input=True,
                default="",
                show_default=False,
            )
            if mlflow_pass:
                api.credentials["mlflow_password"] = mlflow_pass
            click.echo("   MLflow configured  experiment tracking")

        # Ray
        ray_dashboard = click.prompt(
            "Ray Dashboard URI (optional)", default="", show_default=False
        )
        if ray_dashboard:
            api.credentials["ray_dashboard_uri"] = ray_dashboard
            ray_head = click.prompt(
                "Ray Head Node IP (optional)", default="", show_default=False
            )
            if ray_head:
                api.credentials["ray_head_node_ip"] = ray_head
            click.echo("   Ray configured  distributed computing")

        # Save all credentials (provider + integrations) to disk
        # Only in interactive mode - single-provider mode saves directly to file
        api.save_credentials()

        click.echo("\nCredentials saved successfully!")
        click.echo(f"Stored in: {api.credentials_file}")
        click.echo("Your keys are encrypted and stored locally only.")
        click.echo("\nOpen Source Mode: Unlimited access")


@cli.command(cls=ProvidersCommand)
@click.option(
    "--gpu-type",
    "-g",
    default="A100",
    help="GPU type to quote (A100, H100, RTX4090, L40S, etc.)",
)
@click.option(
    "--providers",
    "-p",
    multiple=True,
    help="Filter to specific providers (multiple allowed, e.g., runpod,vastai)",
)
@click.option("--parallel", default=6, type=click.IntRange(1, 100), help="Number of parallel queries (default: 6)")
@click.option("--region", "-r", help="Filter by region (e.g., us-east-1, eu-west-1)")
@click.option(
    "--quick", "-q", is_flag=True, help="Show quick provision command for best quote"
)
@click.option(
    "--include-local",
    is_flag=True,
    help="Include local GPUs from your registered pool (priced at $0/hr)",
)
def quote(gpu_type, providers, parallel, region, quick, include_local):
    """Get real-time GPU pricing quotes from all configured providers.

    Queries all configured cloud providers in parallel and displays pricing sorted
    by cost (cheapest first). Shows spot vs on-demand availability and estimated
    monthly costs.

    Examples:
      terradev quote -g A100                    # Quote A100 across all providers
      terradev quote -g H100 -p runpod,vastai   # Quote H100 from specific providers
      terradev quote -g RTX4090 -r us-east-1     # Quote RTX4090 in specific region
      terradev quote -g A100 -q                  # Show quick provision command
      terradev quote -g RTX4090 --include-local  # Include local GPUs from your pool

    Next Steps:
      After quoting, use: terradev provision -g <gpu-type>
      Or use --quick flag to auto-generate provision command

    Common GPUs:
      A100, H100, RTX4090, L40S, V100, L4, T4
    """

    # Validate GPU type parameter
    if not gpu_type or gpu_type.strip() == "":
        click.echo("ERROR: GPU type is required", err=True)
        click.echo("\nUsage: terradev quote -g <gpu-type>")
        click.echo("\nExample GPU types:")
        click.echo("  terradev quote -g A100")
        click.echo("  terradev quote -g H100")
        click.echo("  terradev quote -g RTX4090")
        click.echo("\nCommon GPUs: A100, H100, RTX4090, L40S, V100, L4, T4")
        raise SystemExit(1)

    if _telemetry:
        _telemetry.log_action(
            "quote",
            {
                "gpu_type": gpu_type,
                "providers": list(providers) if providers else ["all"],
                "parallel": parallel,
                "region": region,
                "quick": quick,
            },
        )

    api = click.get_current_context().obj["api"]

    # ── Load local pool if --include-local is set ──
    local_quotes = []
    if include_local:
        import json
        import os

        pool_path = os.path.expanduser("~/.terradev/local_pool.json")
        if os.path.exists(pool_path):
            try:
                with open(pool_path) as f:
                    pool = json.load(f)
                for pool_name, entry in pool.items():
                    gpus = entry.get("gpus", [])
                    for gpu in gpus:
                        gpu_name = gpu.get("name", "")
                        # Fuzzy match GPU type (e.g., "RTX 4090" matches "RTX4090")
                        gpu_normalized = (
                            gpu_name.replace(" ", "").replace("NVIDIA", "").upper()
                        )
                        gpu_type_normalized = gpu_type.replace(" ", "").upper()
                        if (
                            gpu_type_normalized in gpu_normalized
                            or gpu_normalized in gpu_type_normalized
                        ):
                            local_quotes.append(
                                {
                                    "provider": "local",
                                    "region": entry.get("host", "localhost"),
                                    "price": 0.0,
                                    "availability": "on-demand",
                                    "gpu_name": gpu_name,
                                    "pool_name": pool_name,
                                }
                            )
                if local_quotes:
                    click.echo(f"Included {len(local_quotes)} local GPU(s) from your pool")
            except Exception as e:  # noqa: BLE001
                click.echo(f"Warning: Could not load local pool: {e}")

    # ── Fetch quotes from all providers in parallel ──
    click.echo(f"Querying providers for {gpu_type} pricing...")

    async def _fetch_one(pname, fn):
        try:
            return await asyncio.wait_for(fn(gpu_type), timeout=30)
        except asyncio.TimeoutError:
            click.echo(f"WARNING: {pname} quote timed out")
            return []
        except Exception as exc:  # noqa: BLE001
            click.echo(f"WARNING: {pname} quote failed: {exc}")
            return []

    async def _fetch_all():
        tasks = []
        provider_list = [
            ("runpod", api.get_runpod_quotes),
            ("vastai", api.get_vastai_quotes),
            ("aws", api.get_aws_quotes),
            ("gcp", api.get_gcp_quotes),
            ("azure", api.get_azure_quotes),
            ("tensordock", api.get_tensordock_quotes),
            ("crusoe", api.get_crusoe_quotes),
            ("baseten", api.get_baseten_quotes),
            ("digitalocean", api.get_digitalocean_quotes),
            ("e2enetworks", api.get_e2enetworks_quotes),
            ("huggingface", api.get_huggingface_quotes),
            ("hyperstack", api.get_hyperstack_quotes),
            ("inferx", api.get_inferx_quotes),
            ("latitude", api.get_latitude_quotes),
            ("siliconflow", api.get_siliconflow_quotes),
            ("yottalabs", api.get_yottalabs_quotes),
            ("gcore", api.get_gcore_quotes),
        ]
        for pname, fn in provider_list:
            if not providers or pname in providers:
                tasks.append(asyncio.create_task(_fetch_one(pname, fn)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
            elif isinstance(r, Exception):
                click.echo(f"WARNING: Provider quote task failed: {r}")
        return out

    try:
        all_quotes = asyncio.run(asyncio.wait_for(_fetch_all(), timeout=120))
    except asyncio.TimeoutError:
        click.echo("ERROR: Quote request timed out after 120s. Try a smaller provider list or check your network.", err=True)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: Quote request failed: {exc}", err=True)
        raise SystemExit(1)

    # Merge local quotes with cloud quotes
    all_quotes = local_quotes + all_quotes

    if not all_quotes:
        click.echo("ERROR: No quotes returned from any provider", err=True)
        click.echo("\nTo fix this:")
        click.echo(
            "   1. Configure provider credentials: terradev configure --provider <name>"
        )
        click.echo("      Example: terradev configure --provider runpod")
        click.echo("   2. Or get setup instructions: terradev setup <provider>")
        click.echo("      Example: terradev setup runpod")
        click.echo(
            "   3. RunPod is the easiest to set up (5 minutes): terradev setup runpod --quick"
        )
        raise SystemExit(1)

    # Filter by region if specified
    if region:
        all_quotes = [
            q for q in all_quotes if region.lower() in q.get("region", "").lower()
        ]
        if not all_quotes:
            click.echo(f"ERROR: No quotes returned for {gpu_type} in region {region}", err=True)
            click.echo("\nTip: Try a different region or remove the -r/--region filter")
            raise SystemExit(1)

    all_quotes.sort(key=lambda q: q.get("price", 999))

    # ── Display results ──
    best = all_quotes[0]
    click.echo(f"\nTerradev Quote  {gpu_type}")
    click.echo(
        f"{'#':<4} {'Provider':<14} {'Region':<16} {'$/hr':<10} {'GPUs':<6} {'Instance':<20} {'Spot':<8}"
    )
    click.echo("-" * 84)
    for i, q in enumerate(all_quotes[:10]):
        spot = "✓" if q.get("availability") == "spot" else ""
        gpu_count = q.get("gpu_count", 1)
        gpu_display = f"x{gpu_count}" if gpu_count > 1 else "x1"
        instance_type = q.get("instance_type", "N/A")
        provider_display = q.get("provider", q.get("gpu_name", "unknown"))
        if q.get("provider") == "local":
            provider_display = f"{q['pool_name']}"
        click.echo(
            f"{i+1:<4} {provider_display:<14} {q['region']:<16} ${q['price']:<9.2f} {gpu_display:<6} {instance_type:<20} {spot:<8}"
        )

    click.echo(f"\nBest: ${best['price']:.2f}/hr on {best['provider']} ({best['region']})")
    monthly = best["price"] * 730
    click.echo(f"Estimated monthly: ${monthly:,.0f}")

    if _telemetry:
        _telemetry.log_action(
            "quote_completed",
            {
                "gpu_type": gpu_type,
                "best_price": best["price"],
                "provider": best["provider"],
                "num_quotes": len(all_quotes),
            },
        )

    if quick:
        click.echo(f"\nQuick provision: deploying {gpu_type} on {best['provider']}...")
        click.echo(
            f"   Run: terradev provision -g {gpu_type} --providers {best['provider'].lower().replace(' ', '_')} --dry-run"
        )
    else:
        click.echo(f"\nProvision: terradev provision -g {gpu_type}")
        click.echo(f"   Dry run:   terradev provision -g {gpu_type} --dry-run")


@cli.command(cls=ProvidersCommand)
@click.argument(
    "provider",
    type=click.Choice(
        [
            "runpod",
            "vastai",
            "aws",
            "gcp",
            "azure",
            "tensordock",
            "crusoe",
            "baseten",
            "huggingface",
            "siliconflow",
            "hyperstack",
            "digitalocean",
            "inferx",
            "e2enetworks",
            "latitude",
            "yottalabs",
            "gcore",
        ]
    ),
)
@click.option("--quick", "-q", is_flag=True, help="Show quick setup summary")
def setup(provider, quick):
    """Get step-by-step setup instructions for any cloud provider.

    Shows detailed setup steps including account creation, API key generation,
    environment variable configuration, and testing.

    Examples:
      terradev setup runpod           # Easiest - 5 minutes
      terradev setup vastai           # Easiest - 5 minutes
      terradev setup aws              # Moderate - 30 minutes
      terradev setup azure --quick    # Quick summary

    Quick Start:
      RunPod and Vast.ai are the fastest to set up (5 minutes each)
      Use --quick to see just the essential steps
    """

    setup_instructions = {
        "runpod": {
            "name": "RunPod",
            "time": "5 minutes",
            "difficulty": "EASIEST",
            "url": "https://runpod.io",
            "steps": [
                "Create account at https://runpod.io",
                "Add $10+ credit: Dashboard → Billing → Add Funds",
                'Get API key: Dashboard → Settings → API Keys → "Create API Key"',
                'Copy and run:\nexport RUNPOD_API_KEY="paste-your-key-here"\necho \'export RUNPOD_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers runpod --gpu a100",
            ],
            "env_vars": ["RUNPOD_API_KEY"],
        },
        "vastai": {
            "name": "Vast.AI",
            "time": "5 minutes",
            "difficulty": "EASIEST",
            "url": "https://vast.ai",
            "steps": [
                "Create account at https://vast.ai",
                "Add $10+ credit: Dashboard → Billing → Add Payment",
                'Get API key: Account → API Keys → "New API Key"',
                'Copy and run:\nexport VAST_API_KEY="paste-your-key-here"\necho \'export VAST_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers vastai --gpu a100",
            ],
            "env_vars": ["VAST_API_KEY"],
        },
        "tensordock": {
            "name": "TensorDock",
            "time": "7 minutes",
            "difficulty": "EASY",
            "url": "https://tensordock.com",
            "steps": [
                "Create account at https://tensordock.com",
                "Add $10+ funds: Dashboard → Billing → Add Funds",
                'Get token: Dashboard → API Access → "Create Authorization"',
                'Copy and run:\nexport TENSORDOCK_TOKEN="paste-your-token-here"\necho \'export TENSORDOCK_TOKEN="paste-your-token-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers tensordock --gpu a100",
            ],
            "env_vars": ["TENSORDOCK_TOKEN"],
        },
        "crusoe": {
            "name": "Crusoe",
            "time": "10 minutes",
            "difficulty": "MODERATE",
            "url": "https://crusoe.ai",
            "steps": [
                "Apply at https://crusoe.ai/contact (requires approval, 1-2 days wait)",
                "After approval, login to dashboard",
                'Get API credentials: Settings → API Access → "Generate Credentials"',
                'Copy and run:\nexport CRUSOE_API_KEY="paste-your-key-here"\nexport CRUSOE_API_SECRET="paste-your-secret-here"\necho \'export CRUSOE_API_KEY="paste-your-key-here"\' >> ~/.bashrc\necho \'export CRUSOE_API_SECRET="paste-your-secret-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers crusoe --gpu a100",
            ],
            "env_vars": ["CRUSOE_API_KEY", "CRUSOE_API_SECRET"],
        },
        "baseten": {
            "name": "Baseten",
            "time": "10 minutes",
            "difficulty": "EASY",
            "url": "https://baseten.co",
            "steps": [
                "Create account at https://baseten.co",
                "Add payment: Settings → Billing → Add Card ($50 free credits!)",
                'Get API key: Settings → API Keys → "Create API Key"',
                'Copy and run:\nexport BASETEN_API_KEY="paste-your-key-here"\necho \'export BASETEN_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers baseten --gpu a100",
            ],
            "env_vars": ["BASETEN_API_KEY"],
        },
        "gcp": {
            "name": "Google Cloud Platform",
            "time": "25 minutes",
            "difficulty": "MODERATE",
            "url": "https://console.cloud.google.com",
            "steps": [
                "Create account at https://console.cloud.google.com",
                'Create project: Console → "Select Project" → "New Project" → Name: "terradev"',
                "Enable billing: Billing → Link Billing Account",
                'Enable Compute API: APIs & Services → Enable APIs → "Compute Engine API" → Enable',
                'Create service account:\nIAM & Admin → Service Accounts → "Create Service Account"\nName: terradev-sa\nRole: Compute Admin\nClick "Done"',
                'Create key:\nClick on terradev-sa@... → Keys → "Add Key" → "Create New Key"\nType: JSON → Create\nDownloads as terradev-xxxxx.json',
                "Copy and run:\nmkdir -p ~/.config/gcloud\nmv ~/Downloads/terradev-*.json ~/.config/gcloud/terradev-key.json\n\nexport GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terradev-key.json\necho 'export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terradev-key.json' >> ~/.bashrc\nsource ~/.bashrc",
                "Test it:\nterradev quote --providers gcp --gpu a100",
            ],
            "env_vars": ["GOOGLE_APPLICATION_CREDENTIALS"],
        },
        "aws": {
            "name": "Amazon Web Services",
            "time": "30 minutes",
            "difficulty": "MODERATE",
            "url": "https://aws.amazon.com",
            "steps": [
                "Create account at https://aws.amazon.com",
                "Add payment method: Account → Payment Methods",
                "Go to IAM: https://console.aws.amazon.com/iam",
                'Create user:\nUsers → "Create user"\nUsername: terradev\nCheck: "Programmatic access"\nNext',
                "Set permissions:\nAttach policies directly\nSearch and check: AmazonEC2FullAccess\nNext → Create user",
                'Save credentials (SHOWN ONLY ONCE):\nCopy "Access key ID"\nCopy "Secret access key"',
                "Copy and run:\n# Install AWS CLI if not installed\n# Mac: brew install awscli\n# Ubuntu: sudo apt install awscli\n# Windows: download from https://aws.amazon.com/cli/\n\n# Configure credentials\naws configure\n# When prompted, paste:\n# AWS Access Key ID: [paste-access-key]\n# AWS Secret Access Key: [paste-secret-key]\n# Default region: us-east-1\n# Default output format: json",
                "Test it:\nterradev quote --providers aws --gpu a100",
            ],
            "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        },
        "azure": {
            "name": "Microsoft Azure",
            "time": "30 minutes",
            "difficulty": "MODERATE",
            "url": "https://portal.azure.com",
            "steps": [
                "Create account at https://portal.azure.com",
                "Add payment: Subscriptions → Add payment method",
                "Install Azure CLI:\n# Mac\nbrew install azure-cli\n\n# Ubuntu/Debian\ncurl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash\n\n# Windows\n# Download from: https://aka.ms/installazurecliwindows",
                "Login:\naz login\n# Browser opens → Sign in with your Azure account",
                'Create service principal:\naz ad sp create-for-rbac --name "terradev" --role="Contributor" --scopes="/subscriptions/$(az account show --query id -o tsv)"',
                'Save the output (SHOWN ONLY ONCE):\n{\n  "appId": "xxxx-xxxx-xxxx",\n  "password": "xxxx-xxxx-xxxx",\n  "tenant": "xxxx-xxxx-xxxx"\n}',
                'Copy and run:\nexport AZURE_CLIENT_ID="paste-appId-here"\nexport AZURE_CLIENT_SECRET="paste-password-here"\nexport AZURE_TENANT_ID="paste-tenant-here"\nexport AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"\n\n# Make permanent\necho \'export AZURE_CLIENT_ID="paste-appId-here"\' >> ~/.bashrc\necho \'export AZURE_CLIENT_SECRET="paste-password-here"\' >> ~/.bashrc\necho \'export AZURE_TENANT_ID="paste-tenant-here"\' >> ~/.bashrc\necho \'export AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"\' >> ~/.bashrc\nsource ~/.bashrc',
                "Test it:\nterradev quote --providers azure --gpu a100",
            ],
            "env_vars": [
                "AZURE_CLIENT_ID",
                "AZURE_CLIENT_SECRET",
                "AZURE_TENANT_ID",
                "AZURE_SUBSCRIPTION_ID",
            ],
        },
        "oracle": {
            "name": "Oracle Cloud Infrastructure",
            "time": "35 minutes",
            "difficulty": "ADVANCED",
            "url": "https://cloud.oracle.com/free",
            "steps": [
                "Create account at https://cloud.oracle.com/free (requires credit card verification)",
                "After login, note your info:\nTenancy OCID: Profile → Tenancy → OCID (copy)\nUser OCID: Profile → User Settings → OCID (copy)\nRegion: Profile → Region (e.g., us-ashburn-1)",
                "Generate API key:\n# Create directory\nmkdir -p ~/.oci\n\n# Generate key pair\nopenssl genrsa -out ~/.oci/oci_api_key.pem 2048\nopenssl rsa -pubout -in ~/.oci/oci_api_key.pem -out ~/.oci/oci_api_key_public.pem\n\n# Get fingerprint\nopenssl rsa -pubout -outform DER -in ~/.oci/oci_api_key.pem | openssl md5 -c\n# Save the fingerprint output",
                'Upload public key to Oracle:\nProfile → User Settings → API Keys → "Add API Key"\nChoose: "Paste Public Key"\nPaste contents of: cat ~/.oci/oci_api_key_public.pem\nAdd',
                "Create config file:\ncat > ~/.oci/config << 'EOF'\n[DEFAULT]\nuser=paste-user-ocid-here\nfingerprint=paste-fingerprint-here\ntenancy=paste-tenancy-ocid-here\nregion=us-ashburn-1\nkey_file=~/.oci/oci_api_key.pem\nEOF",
                "Set environment variable:\nexport OCI_CONFIG_FILE=~/.oci/config\necho 'export OCI_CONFIG_FILE=~/.oci/config' >> ~/.bashrc\nsource ~/.bashrc",
                "Test it:\nterradev quote --providers oracle --gpu a100",
            ],
            "env_vars": ["OCI_CONFIG_FILE"],
        },
    }

    info = setup_instructions.get(provider)
    if not info:
        name = provider.replace("_", " ").title()
        info = {
            "name": name,
            "time": "10 minutes",
            "difficulty": "MODERATE",
            "url": f"https://{provider}.com",
            "steps": [
                f"Create an account at {provider}'s website",
                "Generate an API key from the dashboard",
                f'Copy and run:\nexport {provider.upper()}_API_KEY="paste-your-key-here"\necho \'export {provider.upper()}_API_KEY="paste-your-key-here"\' >> ~/.bashrc\nsource ~/.bashrc',
                f"Test it:\nterradev quote --providers {provider} --gpu a100",
            ],
            "env_vars": [f"{provider.upper()}_API_KEY"],
        }

    if quick:
        click.echo(f"{info['name']} Quick Setup ({info['time']})")
        click.echo("=" * 50)
        click.echo(f"Difficulty: {info['difficulty']}")
        click.echo(f"URL: {info['url']}")
        click.echo()
        click.echo("Environment Variables:")
        for var in info["env_vars"]:
            click.echo(f"  {var}")
        click.echo()
        click.echo("Test Command:")
        click.echo(f"  terradev quote --providers {provider} --gpu a100")
        click.echo()
        click.echo("For detailed instructions, run:")
        click.echo(f"  terradev setup {provider}")
        return

    # Full detailed setup
    difficulty_stars = {
        "EASIEST": "*",
        "EASY": "**",
        "MODERATE": "***",
        "ADVANCED": "****",
    }

    click.echo(
        f"{info['name']} Setup ({info['time']}) {difficulty_stars.get(info['difficulty'], '')}"
    )
    click.echo("=" * 60)
    click.echo()

    for i, step in enumerate(info["steps"], 1):
        click.echo(f"Step {i}: {step}")
        if i < len(info["steps"]):
            click.echo()

    click.echo(f"Done! Your {info['name']} is configured.")


# Provider Profiles Commands
@cli.group(cls=ProvidersGroup)
def providers():
    """Manage custom provider profiles for intelligent routing"""
    pass


@providers.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    help="Path to YAML or JSON file containing provider profiles",
)
@click.option(
    "--override",
    is_flag=True,
    help="Override existing profiles with same name (default: skip existing)",
)
def load_profiles(path, override):
    """Load custom provider profiles from a YAML or JSON file.

    Example:
      terradev providers load-profiles ~/.terradev/custom_providers.yaml
      terradev providers load-profiles profiles.json --override

    Profile file format (YAML):
      profiles:
        my_provider:
          api_style: rest
          auth_type: bearer
          egress_cost: 0.05
          supports_spot: true
    """
    from terradev_cli.providers import load_profiles_from_file

    if not path:
        # Try default location
        default_path = Path.home() / ".terradev" / "custom_providers.yaml"
        if default_path.exists():
            path = str(default_path)
        else:
            click.echo("Error: No path specified and default file not found", err=True)
            click.echo(f"Expected: {default_path}")
            click.echo("Use --path to specify a profile file")
            raise SystemExit(1)

    try:
        load_profiles_from_file(path, override=override)
        click.echo(f"✓ Loaded provider profiles from: {path}")
        if override:
            click.echo("  (existing profiles were overridden)")
        else:
            click.echo("  (existing profiles were preserved)")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install PyYAML for YAML support: pip install pyyaml")
    except Exception as e:  # noqa: BLE001
        click.echo(f"Error loading profiles: {e}", err=True)


@providers.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format (default: table)",
)
def list_profiles(format):
    """List all registered provider profiles (built-in and custom).

    Example:
      terradev providers list-profiles
      terradev providers list-profiles --format json
    """
    from terradev_cli.providers import list_all_profiles

    profiles = list_all_profiles()

    if format == "json":
        import json

        output = {}
        for name, profile in profiles.items():
            output[name] = {
                "api_style": profile.api_style,
                "auth_type": profile.auth_type,
                "egress_cost": profile.egress_cost,
                "supports_spot": profile.supports_spot,
                "compute_model": profile.compute_model,
            }
        click.echo(json.dumps(output, indent=2))

    elif format == "yaml":
        try:
            import yaml
        except ImportError:
            click.echo("Error: PyYAML required for YAML output. Install with: pip install pyyaml", err=True)
            raise SystemExit(1)

        output = {}
        for name, profile in profiles.items():
            output[name] = {
                "api_style": profile.api_style,
                "auth_type": profile.auth_type,
                "egress_cost": profile.egress_cost,
                "supports_spot": profile.supports_spot,
                "compute_model": profile.compute_model,
            }
        click.echo(yaml.dump(output, default_flow_style=False))

    else:  # table format
        click.echo(f"Registered Provider Profiles ({len(profiles)} total)")
        click.echo("=" * 80)
        click.echo(f"{'Name':<20} {'API Style':<10} {'Auth':<12} {'Egress':<8} {'Spot':<6} {'Model':<8}")
        click.echo("-" * 80)
        for name, profile in sorted(profiles.items()):
            click.echo(
                f"{name:<20} {profile.api_style:<10} {profile.auth_type:<12} "
                f"${profile.egress_cost:<7.2f} {'Yes' if profile.supports_spot else 'No':<6} "
                f"{profile.compute_model:<8}"
            )


@providers.command()
@click.argument("name")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format (default: table)",
)
def show_profile(name, format):
    """Show details for a specific provider profile.

    Example:
      terradev providers show-profile runpod
      terradev providers show-profile my_custom_provider --format json
    """
    from terradev_cli.providers import get_profile

    profile = get_profile(name)

    if format == "json":
        import json

        output = {
            "name": profile.name,
            "api_style": profile.api_style,
            "auth_type": profile.auth_type,
            "egress_cost": profile.egress_cost,
            "supports_spot": profile.supports_spot,
            "compute_model": profile.compute_model,
            "isolation_level": profile.isolation_level,
            "requires_instance_type_mapping": profile.requires_instance_type_mapping,
            "quote_method": profile.quote_method,
            "rate_limit_per_minute": profile.rate_limit_per_minute,
        }
        click.echo(json.dumps(output, indent=2))

    elif format == "yaml":
        try:
            import yaml
        except ImportError:
            click.echo("Error: PyYAML required for YAML output. Install with: pip install pyyaml", err=True)
            raise SystemExit(1)

        output = {
            "name": profile.name,
            "api_style": profile.api_style,
            "auth_type": profile.auth_type,
            "egress_cost": profile.egress_cost,
            "supports_spot": profile.supports_spot,
            "compute_model": profile.compute_model,
            "isolation_level": profile.isolation_level,
        }
        click.echo(yaml.dump(output, default_flow_style=False))

    else:  # table format
        click.echo(f"Provider Profile: {profile.name}")
        click.echo("=" * 50)
        click.echo(f"API Style:        {profile.api_style}")
        click.echo(f"Auth Type:        {profile.auth_type}")
        click.echo(f"Egress Cost:      ${profile.egress_cost:.2f}/GB")
        click.echo(f"Supports Spot:    {'Yes' if profile.supports_spot else 'No'}")
        click.echo(f"Compute Model:    {profile.compute_model}")
        click.echo(f"Isolation Level:  {profile.isolation_level}")
        click.echo(f"Quote Method:     {profile.quote_method}")
        click.echo(f"Rate Limit:       {profile.rate_limit_per_minute or 'None'} req/min")
        if profile.has_fallback_routing:
            click.echo(f"Fallback Routing: {', '.join(profile.fallback_providers)}")


@providers.command()
@click.argument("name")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Remove without confirmation",
)
def remove_profile(name, force):
    """Remove a custom provider profile from the registry.

    Example:
      terradev providers remove-profile my_custom_provider
      terradev providers remove-profile my_custom_provider --force
    """
    from terradev_cli.providers import unregister_profile

    if not force:
        click.confirm(f"Remove provider profile '{name}'?", abort=True)

    if unregister_profile(name):
        click.echo(f"✓ Removed provider profile: {name}")
    else:
        click.echo(f"Profile '{name}' not found (may be a built-in profile)", err=True)
        raise SystemExit(1)


@providers.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: stdout)",
)
def export_example(output):
    """Export an example provider profiles YAML file.

    Example:
      terradev providers export-example
      terradev providers export-example -o ~/.terradev/custom_providers.yaml
    """
    example_yaml = """# Custom Provider Profiles Example
# Copy this file to ~/.terradev/custom_providers.yaml and add your custom providers
# Then load with: terradev providers load-profiles ~/.terradev/custom_providers.yaml

profiles:
  # Example: Internal GPU cluster
  my_internal_cluster:
    api_style: rest
    auth_type: bearer
    egress_cost: 0.0
    supports_spot: false
    compute_model: vm
    isolation_level: vm
    supports_stop_start: true
    region_specific_availability: true

  # Example: Custom cloud provider
  my_cloud_provider:
    api_style: rest
    auth_type: x_api_key
    egress_cost: 0.02
    supports_spot: true
    spot_interruption_notice_minutes: 5
    rate_limit_per_minute: 60
    compute_model: vm
    isolation_level: vm
    supports_stop_start: true
    provision_requires_location_id: true

  # Example: GPU marketplace with SSH quirks
  my_gpu_marketplace:
    api_style: rest
    auth_type: bearer
    egress_cost: 0.01
    supports_spot: true
    ssh_port_fixed: false  # Dynamic SSH ports
    compute_model: vm
    isolation_level: vm
    supports_stop_start: true
"""

    if output:
        with open(output, "w") as f:
            f.write(example_yaml)
        click.echo(f"✓ Exported example profiles to: {output}")
    else:
        click.echo(example_yaml)


