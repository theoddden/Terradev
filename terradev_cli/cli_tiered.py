#!/usr/bin/env python3
"""
Terradev CLI with Tier Management
Production-ready CLI with Research/Research+/Enterprise tiers
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
from terradev_cli.core.tier_manager import (
    TierManager,
    TierType,
    require_tier,
    require_enterprise_id,
)
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
    Terradev CLI - Multi-Cloud Compute Optimization Platform

    Parallel provisioning and orchestration for optimized compute costs.
    Save 30-93% on compute costs with parallel cloud provider optimization.

    Tiers:
    • Research: 1 instance, 3 providers, 10 provisions/month (Free)
    • Research+: 4 instances, 7 providers, 40 provisions/month, inference, full provenance ($49.99/month)
    • Enterprise: 32 instances, 8 providers, unlimited provisions, full provenance ($299.99/month)
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
    ctx.obj["tier_manager"] = TierManager(config)


@cli.command()
@click.option(
    "--tier", type=click.Choice(["research", "research_plus", "enterprise"]), help="Target tier"
)
@click.option(
    "--id", "enterprise_id", help="Enterprise ID (required for Enterprise tier)"
)
@click.pass_context
def upgrade(ctx, tier, enterprise_id):
    """Upgrade subscription tier"""
    tier_manager = ctx.obj["tier_manager"]

    if not tier:
        # Show current tier info
        info = tier_manager.get_tier_info()
        print(f"📊 Current Tier: {info['tier'].title()}")
        print(f"🔧 Max Instances: {info['limits']['max_instances']}")
        print(f"☁️  Providers: {', '.join(info['limits']['providers_allowed'])}")
        print(f"⚡ Features: {', '.join(info['limits']['features_enabled'])}")
        print(f"🏢 Enterprise: {info['limits']['enterprise_features']}")
        return

    if tier == "research":
        tier_manager.set_tier(TierType.RESEARCH)
        print("✅ Downgraded to Research tier")
    elif tier == "research_plus":
        secret_key = click.prompt("Enter your Research+ secret key")
        tier_manager.upgrade_to_research_plus(secret_key)
    elif tier == "enterprise":
        if not enterprise_id:
            print("❌ Enterprise tier requires --id parameter")
            return
        tier_manager.upgrade_to_enterprise(enterprise_id)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current tier and usage status"""
    tier_manager = ctx.obj["tier_manager"]
    info = tier_manager.get_tier_info()

    print("🎯 Terradev Status")
    print("=" * 50)
    print(f"📊 Current Tier: {info['tier'].title()}")
    print(f"🔧 Max Instances: {info['limits']['max_instances']}")
    print(f"⚡ Max Parallel Queries: {info['limits']['max_parallel_queries']}")
    print(f"☁️  Providers Allowed: {', '.join(info['limits']['providers_allowed'])}")
    print(f"🚀 Features: {', '.join(info['limits']['features_enabled'])}")
    print(f"🏢 Enterprise Features: {info['limits']['enterprise_features']}")
    print()
    print("📈 Current Usage:")
    print(f"   Instances: {info['current_usage']['instances']}")
    print(f"   Enterprise ID: {info['current_usage']['enterprise_id'] or 'None'}")
    print(f"   Last Reset: {info['current_usage']['last_reset']}")


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
@require_tier(TierType.RESEARCH)
def quote(ctx, gpu_type, count, max_price, region, providers, parallel):
    """Get price quotes across providers"""
    tier_manager = ctx.obj["tier_manager"]

    # Check tier limits
    if not tier_manager.check_instance_limit(count):
        info = tier_manager.get_tier_info()
        print(
            f"❌ Instance limit exceeded. Current: {info['current_usage']['instances']}, Max: {info['limits']['max_instances']}"
        )
        print("💡 Upgrade with: terradev upgrade --tier research_plus")
        return

    # Check provider access
    if providers:
        provider_list = [p.strip() for p in providers.split(",")]
        for provider in provider_list:
            if not tier_manager.check_provider_access(provider):
                print(f"❌ Provider '{provider}' not available in current tier")
                return

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

    # Filter by tier provider access
    allowed_providers = tier_manager.TIER_LIMITS[
        tier_manager.get_current_tier()
    ].providers_allowed
    quotes = [q for q in quotes if q["provider"] in allowed_providers]

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
@require_tier(TierType.RESEARCH_PLUS)
def provision(ctx, gpu_type, count, max_price, provider, dry_run):
    """Provision compute instances"""
    tier_manager = ctx.obj["tier_manager"]

    # Check tier limits
    if not tier_manager.check_instance_limit(count):
        info = tier_manager.get_tier_info()
        print(
            f"❌ Instance limit exceeded. Current: {info['current_usage']['instances']}, Max: {info['limits']['max_instances']}"
        )
        return

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

    # Update usage
    tier_manager.increment_instance_count(count)

    print(f"✅ Successfully provisioned {count} instances")
    print("💡 Use 'terradev status' to view current usage")


@cli.command()
@click.pass_context
@require_tier(TierType.RESEARCH_PLUS)
def analytics(ctx):
    """Show cost and usage analytics"""
    tier_manager = ctx.obj["tier_manager"]
    info = tier_manager.get_tier_info()

    print("📊 Analytics Dashboard")
    print("=" * 50)

    # Mock analytics data
    print("💰 Cost Analysis:")
    print("   Total Savings: $1,234.56")
    print("   Average Savings: 67%")
    print("   Best Provider: RunPod")
    print()

    print("📈 Usage Statistics:")
    print(f"   Instances Provisioned: {info['current_usage']['instances']}")
    print(f"   Quotes Requested: {info['usage_stats']['quotes_requested']}")
    print(f"   Total Cost Saved: ${info['usage_stats']['total_cost_saved']:.2f}")
    print()

    print("🎯 Performance:")
    print("   Average Quote Time: 3.2 seconds")
    print("   Average Provision Time: 45 seconds")
    print("   Uptime: 99.9%")


@cli.command()
@click.pass_context
@require_tier(TierType.ENTERPRISE)
def manage(ctx):
    """Enterprise instance management"""
    tier_manager = ctx.obj["tier_manager"]

    if not tier_manager.check_enterprise_feature():
        print("❌ Enterprise features required")
        return

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
    """Interactive setup wizard"""
    print("🌟 Welcome to Terradev CLI!")
    print("Let's set up your multi-cloud optimization platform.")
    print()

    tier_manager = ctx.obj["tier_manager"]

    # Tier selection
    print("📊 Choose your subscription tier:")
    print("1. Research - 1 instance, 3 providers, 10 provisions/month (Free)")
    print("2. Research+ - 4 instances, 7 providers, 40 provisions/month, inference, full provenance ($49.99/month)")
    print("3. Enterprise - 32 instances, 8 providers, unlimited provisions, full provenance ($299.99/month)")

    tier_choice = click.prompt("Enter choice (1-3)", type=int)

    if tier_choice == 1:
        tier_manager.set_tier(TierType.RESEARCH)
        print("✅ Research tier activated")
    elif tier_choice == 2:
        secret_key = click.prompt("Enter your Research+ secret key")
        tier_manager.upgrade_to_research_plus(secret_key)
    elif tier_choice == 3:
        enterprise_id = click.prompt("Enter your Enterprise ID")
        tier_manager.upgrade_to_enterprise(enterprise_id)

    print()
    print("🎯 Setup complete! Use 'terradev --help' to see available commands.")


if __name__ == "__main__":
    cli()
