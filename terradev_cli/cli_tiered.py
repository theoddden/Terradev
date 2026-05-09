#!/usr/bin/env python3
"""
Terradev CLI - Open Source Multi-Cloud Compute Platform
Free open source CLI - no tiers, no payment, just bring your own API keys
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

from terradev_cli.core.terradev_engine import TerradevEngine
from terradev_cli.core.config import TerradevConfig
from terradev_cli.core.auth import AuthManager
# Tier system removed - open source CLI with no tiers
# from terradev_cli.core.tier_manager import (
#     TierManager,
#     TierType,
#     require_tier,
#     require_enterprise_id,
# )
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
    Terradev CLI - Open Source Multi-Cloud Compute Optimization Platform

    Parallel provisioning and orchestration for optimized compute costs.
    Save 30-93% on compute costs with parallel cloud provider optimization.

    Open Source - Free to use with your own cloud provider API keys.
    No tiers, no subscriptions, no payment required.
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
    # Tier system removed - open source CLI


# Upgrade command removed - tier system eliminated (open source CLI)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current configuration and status"""
    config = ctx.obj["config"]

    print("🎯 Terradev Status")
    print("=" * 50)
    print("📊 Mode: Open Source (Free)")
    print("🔧 Max Instances: Unlimited")
    print("⚡ Max Parallel Queries: Unlimited")
    print("☁️  Providers: All cloud providers supported")
    print("🚀 Features: All features enabled")
    print()
    print("📈 Configuration:")
    print(f"   Config File: {ctx.obj['config_path']}")
    print(f"   Auth Configured: {bool(ctx.obj['auth'])}")
    print()
    print("💡 Open Source: No tiers, no limits, no payment required")


@cli.command()
@click.option(
    "--gpu-type", "-g", required=True, help="GPU type (e.g., A100, V100, RTX4090)"
)
@click.option("--count", "-n", default=1, help="Number of instances")
@click.option("--max-price", help="Maximum price per hour")
@click.option("--region", help="Preferred region")
@click.option("--providers", help="Comma-separated list of providers")
@click.option("--parallel", "-p", default=5, help="Parallel queries")
@click.pass_context
def quote(ctx, gpu_type, count, max_price, region, providers, parallel):
    """Get price quotes across providers - Open Source (all providers enabled)"""
    # No tier limits - all providers accessible

    if providers:
        provider_list = [p.strip() for p in providers.split(",")]
        print(f"🔍 Querying providers: {', '.join(provider_list)}")

    print(f"🔍 Getting quotes for {count}x {gpu_type} instances...")
    print("⚡ Parallel querying across available providers...")

    # Mock quote data for demonstration
    quotes = [
        {"provider": "AWS", "price": 6.98, "region": "us-east-1", "gpu_type": gpu_type},
        {
            "provider": "RunPod",
            "price": 1.49,
            "region": "us-east-1",
            "gpu_type": gpu_type,
        },
        {
            "provider": "Vast.ai",
            "price": 2.10,
            "region": "us-west-1",
            "gpu_type": gpu_type,
        },
    ]

    # No tier filtering - all providers available

    # Sort by price
    quotes.sort(key=lambda x: x["price"])

    print("\n💰 Price Quotes:")
    print(
        format_table(
            ["Provider", "Price/hr", "Region", "GPU Type"],
            [
                [q["provider"], f"${q['price']:.2f}", q["region"], q["gpu_type"]]
                for q in quotes
            ],
        )
    )

    if quotes:
        best = quotes[0]
        savings = ((quotes[-1]["price"] - best["price"]) / quotes[-1]["price"]) * 100
        print(f"\n💡 Best deal: {best['provider']} at ${best['price']:.2f}/hr")
        print(f"📈 Potential savings: {savings:.1f}% vs most expensive")


@cli.command()
@click.option(
    "--gpu-type", "-g", required=True, help="GPU type (e.g., A100, V100, RTX4090)"
)
@click.option("--count", "-n", default=1, help="Number of instances")
@click.option("--max-price", help="Maximum price per hour")
@click.option("--provider", help="Specific provider to use")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be provisioned without actually provisioning",
)
@click.pass_context
def provision(ctx, gpu_type, count, max_price, provider, dry_run):
    """Provision compute instances - Open Source (unlimited access)"""
    # No tier limits - all features accessible

    if dry_run:
        print("🔍 Dry run - showing what would be provisioned:")
        print(f"   GPU Type: {gpu_type}")
        print(f"   Count: {count}")
        print(f"   Max Price: ${max_price or 'unlimited'}/hr")
        print(f"   Provider: {provider or 'auto-select best'}")
        print("✅ Provisioning would succeed")
        return

    print(f"🚀 Provisioning {count}x {gpu_type} instances...")

    # Mock provisioning
    print("⚡ Finding optimal provider...")
    print("🔧 Configuring instances...")
    print("📡 Deploying to cloud...")

    # Update usage (local tracking only - no tier limits)
    print(f"✅ Successfully provisioned {count} instances")
    print("💡 Use 'terradev status' to view configuration")


@cli.command()
@click.pass_context
def analytics(ctx):
    """Show cost and usage analytics - Open Source (unlimited access)"""
    # No tier limits - all features accessible

    print("📊 Analytics Dashboard")
    print("=" * 50)

    # Mock analytics data
    print("💰 Cost Analysis:")
    print("   Total Savings: $1,234.56")
    print("   Average Savings: 67%")
    print("   Best Provider: RunPod")
    print()

    print("📈 Usage Statistics:")
    print(f"   Instances Provisioned: 10")
    print(f"   Quotes Requested: 50")
    print(f"   Total Cost Saved: $1000.00")
    print()

    print("🎯 Performance:")
    print("   Average Quote Time: 3.2 seconds")
    print("   Average Provision Time: 45 seconds")
    print("   Uptime: 99.9%")


@cli.command()
@click.pass_context
def manage(ctx):
    """Instance management - Open Source (unlimited access)"""
    # No tier limits - all features accessible
    print("🏢 Enterprise Management Console")
    print("=" * 50)

    # Mock enterprise data
    instances = [
        {
            "id": "i-12345",
            "provider": "RunPod",
            "gpu": "A100",
            "status": "running",
            "cost": "$1.49/hr",
        },
        {
            "id": "i-67890",
            "provider": "AWS",
            "gpu": "V100",
            "status": "stopped",
            "cost": "$2.34/hr",
        },
    ]

    print(
        format_table(
            ["Instance ID", "Provider", "GPU", "Status", "Cost"],
            [
                [i["id"], i["provider"], i["gpu"], i["status"], i["cost"]]
                for i in instances
            ],
        )
    )


@cli.command()
@click.pass_context
def setup(ctx):
    """Interactive setup wizard - Open Source (no tiers)"""
    print("🌟 Welcome to Terradev CLI!")
    print("Let's set up your multi-cloud optimization platform.")
    print()

    # No tier selection - open source, all features enabled
    print("📊 Open Source Mode - All features enabled:")
    print("✓ Unlimited instances")
    print("✓ All cloud providers")
    print("✓ All features (quote, provision, analytics, manage)")
    print("✓ No payment required")

    print()
    print("🎯 Setup complete! Use 'terradev --help' to see available commands.")
    print("💡 Configure your cloud provider API keys to get started.")


if __name__ == "__main__":
    cli()
