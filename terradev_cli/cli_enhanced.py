#!/usr/bin/env python3
"""
Terradev CLI Enhanced - Interactive Setup Wizard
Improved user experience with guided configuration and provider selection
"""

import click
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from terradev_cli.core.terradev_engine import TerradevEngine
from terradev_cli.core.config import TerradevConfig
from terradev_cli.core.auth import AuthManager
from terradev_cli.utils.formatters import (
    format_table,
    format_json,
    format_success,
    format_error,
    format_warning,
)
from terradev_cli.providers.provider_factory import ProviderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global configuration
CONFIG_FILE = Path.home() / ".terradev" / "config.json"
AUTH_FILE = Path.home() / ".terradev" / "auth.json"


@click.group()
@click.version_option(version="1.0.0", prog_name="Terradev CLI")
@click.option(
    "--config", "-c", default=str(CONFIG_FILE), help="Configuration file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config, verbose):
    """
    Terradev CLI - Cross-Cloud Compute Optimization Platform

    Parallel provisioning and orchestration for optimized compute costs.
    Save 20%+ on compute costs with parallel cloud provider optimization.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure config directory exists
    Path(config).parent.mkdir(parents=True, exist_ok=True)

    # Initialize configuration
    ctx.obj["config"] = TerradevConfig.load(config)
    ctx.obj["auth"] = AuthManager.load(AUTH_FILE)


@cli.command()
@click.option(
    "--guided/--quick", default=True, help="Guided setup wizard vs quick setup"
)
def setup(guided):
    """Interactive setup wizard for new users"""
    if guided:
        _run_setup_wizard()
    else:
        _run_quick_setup()


def _run_setup_wizard():
    """Step-by-step setup wizard"""
    print("🌟 Welcome to Terradev CLI!")
    print("Let's get you set up for cloud cost optimization.")
    print()

    # Step 1: Provider selection
    providers = _select_providers_interactive()

    # Step 2: Credential configuration
    for provider in providers:
        _configure_provider_interactive(provider)

    # Step 3: Validation
    _validate_all_credentials()

    # Step 4: First quote demonstration
    _show_first_quote()

    print("🎉 Setup complete! You're ready to start saving on cloud costs.")


def _run_quick_setup():
    """Quick setup for advanced users"""
    print("⚡ Quick Setup Mode")
    print()

    # Get provider list
    provider_list = click.prompt(
        "Enter providers (comma-separated)", type=str, default="runpod,vastai"
    )
    providers = [p.strip() for p in provider_list.split(",")]

    # Configure each provider
    for provider in providers:
        api_key = click.prompt(f"Enter {provider} API key", hide_input=True)

        # Save credentials
        auth = AuthManager.load(AUTH_FILE)
        auth.set_credentials(provider, api_key, "")
        auth.save(AUTH_FILE)

        # Add to config
        config = TerradevConfig.load(CONFIG_FILE)
        config.add_provider(provider, "us-east-1")
        config.save(CONFIG_FILE)

        format_success(f"✅ {provider} configured")

    format_success("🚀 Quick setup complete!")


def _select_providers_interactive() -> List[str]:
    """Interactive provider selection with descriptions"""
    providers = {
        "aws": {
            "name": "Amazon Web Services",
            "description": "Largest cloud provider with extensive GPU offerings",
            "difficulty": "Medium",
            "savings": "15-25%",
            "credentials": ["access_key", "secret_key"],
            "help_url": "https://console.aws.amazon.com/iam/home",
        },
        "runpod": {
            "name": "RunPod",
            "description": "Specialized GPU cloud with competitive pricing",
            "difficulty": "Easy",
            "savings": "60-70%",
            "credentials": ["api_key"],
            "help_url": "https://runpod.io/console/user",
        },
        "vastai": {
            "name": "VastAI",
            "description": "GPU marketplace with variable pricing",
            "difficulty": "Easy",
            "savings": "40-60%",
            "credentials": ["api_key"],
            "help_url": "https://cloud.vast.ai/api-keys",
        },
        "huggingface": {
            "name": "HuggingFace",
            "description": "ML Model Hub and Inference API",
            "difficulty": "Easy",
            "savings": "50-80%",
            "credentials": ["api_token"],
            "help_url": "https://huggingface.co/settings/tokens",
        },
        "lambda_labs": {
            "name": "Lambda Labs",
            "description": "GPU cloud with focus on ML workloads",
            "difficulty": "Easy",
            "savings": "30-50%",
            "credentials": ["api_key"],
            "help_url": "https://lambdalabs.com/console",
        },
        "coreweave": {
            "name": "CoreWeave",
            "description": "Specialized GPU cloud for Kubernetes",
            "difficulty": "Medium",
            "savings": "20-40%",
            "credentials": ["api_key"],
            "help_url": "https://www.coreweave.com/",
        },
    }

    print("📋 Select cloud providers to configure:")
    print()

    for key, info in providers.items():
        print(f"  [{key}] {info['name']}")
        print(f"       {info['description']}")
        print(f"       Difficulty: {info['difficulty']} | Savings: {info['savings']}")
        print()

    selection = click.prompt(
        "Enter provider keys (comma-separated)", type=str, default="runpod,vastai"
    )
    selected_providers = [p.strip() for p in selection.split(",")]

    # Validate selections
    valid_providers = []
    for provider in selected_providers:
        if provider in providers:
            valid_providers.append(provider)
        else:
            format_warning(f"⚠️ Unknown provider: {provider}")

    return valid_providers


def _configure_provider_interactive(provider: str):
    """Enhanced credential configuration with help"""
    provider_configs = {
        "aws": {
            "credentials": ["access_key", "secret_key"],
            "help_url": "https://console.aws.amazon.com/iam/home",
            "instructions": [
                "1. Go to AWS IAM Console",
                "2. Create a new user or use existing one",
                "3. Generate access keys",
                "4. Copy Access Key ID and Secret Access Key",
            ],
        },
        "runpod": {
            "credentials": ["api_key"],
            "help_url": "https://runpod.io/console/user",
            "instructions": [
                "1. Go to RunPod Console",
                "2. Go to Settings > API Keys",
                "3. Generate new API key",
                "4. Copy the API key",
            ],
        },
        "vastai": {
            "credentials": ["api_key"],
            "help_url": "https://cloud.vast.ai/api-keys",
            "instructions": [
                "1. Go to VastAI Console",
                "2. Go to API Keys section",
                "3. Create new API key",
                "4. Copy the API key",
            ],
        },
        "huggingface": {
            "credentials": ["api_token"],
            "help_url": "https://huggingface.co/settings/tokens",
            "instructions": [
                "1. Go to HuggingFace Settings",
                "2. Go to Access Tokens",
                "3. Create new token",
                "4. Copy the token (starts with hf_)",
            ],
        },
        "lambda_labs": {
            "credentials": ["api_key"],
            "help_url": "https://lambdalabs.com/console",
            "instructions": [
                "1. Go to Lambda Labs Console",
                "2. Go to API settings",
                "3. Generate API key",
                "4. Copy the API key",
            ],
        },
        "coreweave": {
            "credentials": ["api_key"],
            "help_url": "https://www.coreweave.com/",
            "instructions": [
                "1. Go to CoreWeave Console",
                "2. Go to API settings",
                "3. Generate API key",
                "4. Copy the API key",
            ],
        },
    }

    config = provider_configs.get(provider, provider_configs["runpod"])

    print(f"🔧 Configuring {provider.upper()}:")
    print(f"📖 Help: {config['help_url']}")
    print("📋 Instructions:")
    for i, instruction in enumerate(config["instructions"], 1):
        print(f"   {instruction}")
    print()

    credentials = {}
    for cred in config["credentials"]:
        if cred == "api_token" and provider == "huggingface":
            # Special handling for HuggingFace token
            value = click.prompt(
                f"Enter {cred.replace('_', ' ').title()}",
                hide_input=True,
                confirmation_prompt=True,
            )
            # Validate HF token format
            if not value.startswith("hf_"):
                format_warning("⚠️ HuggingFace tokens usually start with 'hf_'")
                if not click.confirm("Continue anyway?"):
                    return _configure_provider_interactive(provider)
        else:
            value = click.prompt(
                f"Enter {cred.replace('_', ' ').title()}",
                hide_input=True,
                confirmation_prompt=True,
            )
        credentials[cred] = value

    # Save credentials
    auth = AuthManager.load(AUTH_FILE)

    if provider == "aws":
        auth.set_credentials(
            provider, credentials["access_key"], credentials["secret_key"]
        )
    else:
        api_key = credentials.get("api_key") or credentials.get("api_token")
        auth.set_credentials(provider, api_key, "")

    auth.save(AUTH_FILE)

    # Add to config
    config_obj = TerradevConfig.load(CONFIG_FILE)
    region = click.prompt(
        f"Default region for {provider}", type=str, default="us-east-1"
    )
    config_obj.add_provider(provider, region)
    config_obj.save(CONFIG_FILE)

    format_success(f"✅ {provider} configured successfully")


def _validate_all_credentials():
    """Validate all configured credentials"""
    print("🔍 Validating credentials...")

    auth = AuthManager.load(AUTH_FILE)
    config = TerradevConfig.load(CONFIG_FILE)

    factory = ProviderFactory()
    validation_results = []

    for provider in config.get_providers():
        try:
            credentials = auth.get_credentials(provider)
            provider_instance = factory.create_provider(provider, credentials)

            # Test connection (async)
            async def test_connection():
                instances = await provider_instance.get_available_instances()
                return len(instances) > 0

            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_valid = loop.run_until_complete(test_connection())
            loop.close()

            validation_results.append(
                {
                    "provider": provider,
                    "status": "✅ Valid" if is_valid else "❌ Invalid",
                    "message": (
                        "Connection successful" if is_valid else "Connection failed"
                    ),
                }
            )

        except Exception as e:
            validation_results.append(
                {"provider": provider, "status": "❌ Error", "message": str(e)}
            )

    # Display results
    print("\n📊 Credential Validation Results:")
    for result in validation_results:
        print(f"   {result['status']} {result['provider']}: {result['message']}")


def _show_first_quote():
    """Show first quote demonstration"""
    print("\n🎯 Let's get your first quote to see the savings!")

    try:
        config = TerradevConfig.load(CONFIG_FILE)
        auth = AuthManager.load(AUTH_FILE)
        engine = TerradevEngine(config, auth)

        # Get quotes for popular GPU
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        quotes = loop.run_until_complete(
            engine.get_quotes(gpu_type="A100", parallel_queries=3)
        )
        loop.close()

        if quotes:
            print(f"\n✅ Found {len(quotes)} quotes for A100 GPUs:")

            # Show top 3 cheapest
            quotes_sorted = sorted(quotes, key=lambda x: x["price_per_hour"])[:3]

            headers = ["Provider", "Instance Type", "Price/Hour", "Region"]
            rows = []
            for quote in quotes_sorted:
                rows.append(
                    [
                        quote["provider"],
                        quote["instance_type"],
                        f"${quote['price_per_hour']:.4f}",
                        quote["region"],
                    ]
                )

            print(format_table(headers, rows))

            # Calculate savings
            if quotes_sorted:
                cheapest = quotes_sorted[0]
                aws_price = 32.77  # AWS A100 price
                savings = ((aws_price - cheapest["price_per_hour"]) / aws_price) * 100

                print(f"\n💰 Potential savings: {savings:.1f}% vs AWS")
                print(
                    f"   Monthly savings: ${(aws_price - cheapest['price_per_hour']) * 730:.2f}"
                )

    except Exception as e:
        format_error(f"❌ Failed to get quotes: {str(e)}")


@cli.command()
@click.option("--provider", "-p", multiple=True, help="Cloud providers to configure")
@click.option("--api-key", help="API key for provider")
@click.option("--secret-key", help="Secret key for provider")
@click.option("--region", "-r", help="Default region")
@click.pass_context
def configure(ctx, provider, api_key, secret_key, region):
    """Configure cloud provider credentials"""
    config = ctx.obj["config"]
    auth = ctx.obj["auth"]

    if not provider:
        # Show current configuration
        format_success("Current Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    for p in provider:
        if not api_key:
            api_key = click.prompt(f"Enter API key for {p}", hide_input=True)
        if not secret_key and p == "aws":
            secret_key = click.prompt(f"Enter secret key for {p}", hide_input=True)

        auth.set_credentials(p, api_key, secret_key)
        config.add_provider(p, region or "us-east-1")

        format_success(f"✅ {p} configured successfully")

    # Save configuration
    config.save(ctx.obj["config_path"])
    auth.save(AUTH_FILE)

    format_success("Configuration saved!")


@cli.command()
@click.pass_context
def config_status(ctx):
    """Show configuration status and health"""
    config = ctx.obj["config"]
    auth = ctx.obj["auth"]

    print("📊 Terradev Configuration Status:")
    print()

    # Provider status
    providers = config.get_providers()
    if providers:
        print("🔧 Configured Providers:")
        for provider in providers:
            credentials = auth.get_credentials(provider)
            has_creds = bool(credentials and any(credentials.values()))
            status = "✅" if has_creds else "❌"
            region = config.get_provider_region(provider)
            print(f"   {status} {provider} ({region})")
    else:
        print("❌ No providers configured")

    print()

    # File locations
    print("📁 Configuration Files:")
    print(f"   Config: {ctx.obj['config_path']}")
    print(f"   Auth: {AUTH_FILE}")
    print(f"   Config exists: {Path(ctx.obj['config_path']).exists()}")
    print(f"   Auth exists: {AUTH_FILE.exists()}")

    print()

    # Quick actions
    if not providers:
        print("🚀 Quick Start:")
        print("   terradev setup          # Interactive setup wizard")
        print("   terradev configure -p runpod  # Quick configure")
    else:
        print("⚡ Next Steps:")
        print("   terradev quote          # Get price quotes")
        print("   terradev provision      # Provision instances")


@cli.command()
@click.option("--providers", "-p", multiple=True, help="Specific providers to query")
@click.option("--parallel", default=6, help="Number of parallel queries")
@click.option("--gpu-type", "-g", help="Filter by GPU type")
@click.option("--region", "-r", help="Filter by region")
@click.pass_context
def quote(ctx, providers, parallel, gpu_type, region):
    """Get real-time quotes from all providers"""
    config = ctx.obj["config"]
    auth = ctx.obj["auth"]

    if not auth.has_credentials():
        format_error(
            "❌ No cloud provider credentials configured. Run 'terradev setup' first."
        )
        return

    format_success("🔍 Getting real-time quotes...")

    engine = TerradevEngine(config, auth)

    try:
        quotes = asyncio.run(
            engine.get_quotes(
                providers=list(providers) if providers else None,
                parallel_queries=parallel,
                gpu_type=gpu_type,
                region=region,
            )
        )

        _display_quotes(quotes)

    except Exception as e:
        format_error(f"❌ Quote retrieval failed: {str(e)}")
        if ctx.obj["verbose"]:
            import traceback

            traceback.print_exc()


def _display_quotes(quotes):
    """Display quote results"""
    format_success(f"✅ Retrieved {len(quotes)} quotes")

    headers = ["Provider", "Instance Type", "GPU", "Price/Hour", "Region", "Available"]
    rows = []

    for quote in quotes:
        rows.append(
            [
                quote["provider"],
                quote["instance_type"],
                quote["gpu_type"],
                f"${quote['price_per_hour']:.4f}",
                quote["region"],
                str(quote["available"]),
            ]
        )

    print(format_table(headers, rows))


if __name__ == "__main__":
    cli()
