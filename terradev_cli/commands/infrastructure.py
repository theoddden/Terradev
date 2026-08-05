#!/usr/bin/env python3
"""Infrastructure intelligence commands for the Terradev CLI."""

import logging
import sys

import click
from . import cli

logger = logging.getLogger(__name__)

@cli.command()
@click.option("--gpu-type", help="GPU type for price discovery")
@click.option("--region", help="Region filter")
@click.option(
    "--hours", type=int, default=24, help="Hours of historical data to analyze"
)
@click.option("--trends", is_flag=True, help="Show price trends")
def price_discovery(gpu_type, region, hours, trends):
    """Enhanced price discovery with capacity and confidence scoring"""
    try:
        from terradev_cli.core.price_discovery import PriceDiscoveryEngine
    except ImportError:
        print(
            "ERROR: Price discovery module not available. Install terradev_cli package."
        )
        sys.exit(1)
    import asyncio

    async def _price_discovery():

        engine = PriceDiscoveryEngine()

        async with engine as e:
            if gpu_type:
                print(f" Getting real-time prices for {gpu_type}...")
                prices = await e.get_realtime_prices(gpu_type, region)

                print(f"\nCOST: Real-time Prices for {gpu_type}:")
                print("=" * 70)
                print(
                    f"{'Provider':<12} {'Price':<10} {'Instance':<20} {'Capacity':<12} {'Confidence':<12}"
                )
                print("-" * 70)

                for price in prices:
                    print(
                        f"{price.provider:<12} ${price.price:<9.2f} {price.instance_type:<20} {price.capacity:<12} {price.confidence:<12.1%}"
                    )

                if trends:
                    print(f"\n Price Trends (last {hours} hours):")
                    trends_data = await e.get_price_trends(gpu_type, hours)

                    for provider, data in trends_data.items():
                        metrics = data.get("metrics", {})
                        print(f"\n{provider}:")
                        print(f"   Average: ${metrics['avg_price']:.2f}/hr")
                        print(
                            f"   Range: ${metrics['min_price']:.2f} - ${metrics['max_price']:.2f}/hr"
                        )
                        print(f"   Volatility: {metrics['volatility']:.3f}")
                        print(f"   Trend: {metrics['trend']}")
            else:
                print("ERROR: Please specify --gpu-type")
                sys.exit(2)

    asyncio.run(_price_discovery())
@cli.command()
@click.option("--gpu-type", required=True, help="GPU type")
@click.option("--budget", type=float, required=True, help="Budget constraint ($/hr)")
@click.option("--gpu-count", type=int, default=1, help="Number of GPUs")
@click.option("--hours", type=float, default=1.0, help="Estimated runtime in hours")
@click.option("--region", help="Preferred region")
@click.option("--workload", default="training", help="Workload type")
def budget_optimize(gpu_type, budget, gpu_count, hours, region, workload):
    """Find optimal deployment under budget constraints"""
    import asyncio

    async def _budget_optimize():
        from terradev_cli.core.price_discovery import BudgetOptimizationEngine

        optimizer = BudgetOptimizationEngine()

        requirements = {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "estimated_hours": hours,
            "workload_type": workload,
            "region": region,
        }

        print(f"COST: Finding options under ${budget:.2f}/hr budget for {gpu_type}...")

        options = await optimizer.optimize_for_budget(requirements, budget)

        if not options:
            print(f"ERROR: No options found under ${budget:.2f}/hr budget")
            return

        print("\n Budget-Optimized Options:")
        print("=" * 80)
        print(
            f"{'Provider':<12} {'Instance':<20} {'Cost':<10} {'Risk':<8} {'Budget Used':<12} {'Confidence':<12}"
        )
        print("-" * 80)

        for option in options:
            print(
                f"{option['provider']:<12} {option['instance_type']:<20} ${option['price']:<9.2f} {option['risk_score']:<7.1%} {option['budget_utilization']:<11.1%} {option['confidence']:<12.1%}"
            )
            print(
                f"   Predicted total: ${option['predicted_cost']:.2f} (risk-adjusted: ${option['risk_adjusted_cost']:.2f})"
            )
            print(
                f"   Capacity: {option['capacity']} | Spot: {'Yes' if option['spot'] else 'No'}"
            )
            print()

    asyncio.run(_budget_optimize())
@cli.command()
@click.option(
    "--workload",
    default="training",
    help="Workload type (training, inference, cost-optimized, high-performance, moe-inference, rag, vllm-optimized)",
)
@click.option(
    "--gpu-type",
    default="A100",
    help="GPU type (A100, H100, V100, L4, L40S, RTX 4090, T4, etc.)",
)
@click.option("--image", required=True, help="Docker image")
@click.option("--gpu-count", type=int, default=1, help="Number of GPUs")
@click.option("--memory", type=int, help="Memory in GB")
@click.option("--storage", type=int, help="Storage in GB")
@click.option("--budget", type=float, help="Budget constraint ($/hr)")
@click.option("--region", help="Preferred region")
@click.option(
    "--port", type=int, multiple=True, help="Expose port(s) via Service (repeatable)"
)
@click.option(
    "--stack",
    "-s",
    multiple=True,
    help="Stack integrations: qdrant, phoenix, guardrails (repeatable)",
)
@click.option("--output", "-o", help="Output directory")
@click.option("--name", help="Chart name")
@click.option("--dry-run", is_flag=True, help="Show chart config without generating")
def helm_generate(
    workload,
    gpu_type,
    image,
    gpu_count,
    memory,
    storage,
    budget,
    region,
    port,
    stack,
    output,
    name,
    dry_run,
):
    """Generate Helm charts from Terradev workloads"""
    from terradev_cli.core.helm_generator import HelmChartGenerator

    generator = HelmChartGenerator()

    # Build workload configuration
    workload_config = {
        "workload_type": workload,
        "gpu_type": gpu_type,
        "image": image,
        "gpu_count": gpu_count,
        "memory_gb": memory or 16,
        "storage_gb": storage or 100,
        "budget": budget,
        "region": region or "us-east-1",
        "spot": True if budget and budget < 2.0 else False,
        "provider": "auto",
        "ports": list(port) if port else [],
        "stack": list(stack) if stack else [],
    }

    if dry_run:
        print("Helm Chart Configuration (Dry Run):")
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
        if port:
            print(f"Ports: {list(port)}")
        if stack:
            print(f"Stack: {list(stack)}")
        print()
        print("Chart files that would be generated:")
        print("   Chart.yaml")
        print("   values.yaml")
        print("   templates/")
        if workload in ("training", "cost-optimized"):
            print("     - job.yaml")
        else:
            print("     - deployment.yaml")
            print("     - service.yaml")
            print("     - hpa.yaml")
            print("     - pdb.yaml")
        print("     - configmap.yaml")
        print("     - pvc.yaml (if storage)")
        print("     - serviceaccount.yaml")
        print("     - _helpers.tpl")
        print("     - NOTES.txt")
        print("   README.md")
        return

    # Generate chart
    chart_name = name or f"terradev-{workload}"
    output_dir = output or f"./{chart_name}"

    print(f"Generating Helm chart for {workload} workload...")

    try:
        chart_path = generator.generate_chart(workload_config, output_dir)
        print("Helm chart generated successfully!")
        print(f"   Location: {chart_path}")
        print()
        print("Next steps:")
        print(f"   1. Review the chart: cd {chart_path}")
        print("   2. Customize values: vim values.yaml")
        print(f"   3. Install chart: helm install my-{workload} .")
        print(
            f"   4. Check status: kubectl get all -l app.kubernetes.io/name=my-{workload}"
        )
        print()
        print(f"   Chart README: {chart_path}/README.md")
        print("   Terradev docs: https://terradev.dev/docs")

    except Exception as e:  # noqa: BLE001
        print(f"Failed to generate Helm chart: {e}")
@cli.command()
@click.option("--gpu-type", "-g", required=True, help="GPU type (e.g. A100, H100)")
@click.option("--provider", "-p", help="Filter to a single provider")
@click.option("--spot", is_flag=True, default=None, help="Spot instances only")
@click.option(
    "--window", "-w", default=720, help="Lookback window in hours (default: 720 = 30d)"
)
def percentiles(gpu_type, provider, spot, window):
    """Show historical price percentiles (p10p99) per provider."""
    try:
        from terradev_cli.core.price_intelligence import compute_percentiles
    except ImportError:
        print("ERROR: Price intelligence module not available")
        sys.exit(1)

    data = compute_percentiles(gpu_type, provider=provider, spot=spot, hours=window)
    providers = data.get("providers", {})

    if not providers:
        print(f"ERROR: No price data for {gpu_type.upper()} in the last {window}h")
        print("Tip: Run 'terradev quote -g {gpu_type}' to start collecting price data.")
        return

    print(f"\nStatus Price Percentiles  {gpu_type.upper()} (last {window}h)")
    print(
        f"{'Provider':<14} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p99':>8} {'Min':>8} {'Max':>8} {'N':>6}"
    )
    print("─" * 100)
    for prov, stats in sorted(providers.items()):
        print(
            f"{prov:<14} "
            f"${stats['p10']:>6.2f} ${stats['p25']:>6.2f} ${stats['p50']:>6.2f} "
            f"${stats['p75']:>6.2f} ${stats['p90']:>6.2f} ${stats['p99']:>6.2f} "
            f"${stats['min']:>6.2f} ${stats['max']:>6.2f} {stats['count']:>5}"
        )

    # Summary
    all_p50 = [(p, s["p50"]) for p, s in providers.items()]
    all_p50.sort(key=lambda x: x[1])
    cheapest = all_p50[0]
    print(f"\nTip: Cheapest median (p50): {cheapest[0]} at ${cheapest[1]:.2f}/hr")
    if len(all_p50) > 1:
        spread = all_p50[-1][1] - all_p50[0][1]
        print(f" Median spread: ${spread:.2f}/hr across {len(all_p50)} providers")
@cli.command()
@click.option("--gpu-type", "-g", help="GPU type filter (shows all if omitted)")
@click.option(
    "--window", "-w", default=24, help="Lookback window in hours (default: 24)"
)
def availability(gpu_type, window):
    """Show GPU availability / stock status across providers."""
    try:
        from terradev_cli.core.price_intelligence import get_availability, get_availability_summary
    except ImportError:
        print("ERROR: Price intelligence module not available")
        sys.exit(1)

    if gpu_type:
        data = get_availability(gpu_type, hours=window)
        providers = data.get("providers", {})

        if not providers:
            print(
                f"ERROR: No availability data for {gpu_type.upper()} in the last {window}h"
            )
            print(
                f"Tip: Run 'terradev quote -g {gpu_type}' to start tracking availability."
            )
            return

        print(f"\n Availability  {gpu_type.upper()} (last {window}h)")
        print(
            f"{'Provider':<14} {'Status':<12} {'Rate':>8} {'Checks':>8} {'Avail':>8} {'Avg ms':>10} {'Last Seen':<20}"
        )
        print("─" * 90)
        for prov, stats in sorted(providers.items()):
            status = "OK: In Stock" if stats["available"] else "ERROR: Sold Out"
            rate_pct = f"{stats['availability_rate'] * 100:.1f}%"
            print(
                f"{prov:<14} {status:<12} {rate_pct:>8} {stats['total_checks']:>8} "
                f"{stats['available_checks']:>8} {stats['avg_response_ms']:>9.0f}ms "
                f"{stats['last_seen'][:19]:<20}"
            )
            if stats.get("last_error"):
                print(f"{'':>14} Warning  Last error: {stats['last_error'][:60]}")
    else:
        summary = get_availability_summary()
        if not summary:
            print("ERROR: No availability data yet.")
            print("Tip: Run 'terradev quote -g <GPU>' to start tracking.")
            return

        print("\n Availability Summary (all GPUs, last check)")
        print(f"{'GPU Type':<14} {'Provider':<14} {'Status':<12}")
        print("─" * 42)
        for gtype in sorted(summary.keys()):
            for prov in sorted(summary[gtype].keys()):
                status = "OK: In Stock" if summary[gtype][prov] else "ERROR: Sold Out"
                print(f"{gtype:<14} {prov:<14} {status:<12}")
@cli.command()
@click.option("--provider", "-p", help="Filter to a single provider")
@click.option(
    "--window", "-w", default=720, help="Lookback window in hours (default: 720 = 30d)"
)
@click.option("--ranking", is_flag=True, help="Show ranked leaderboard")
def reliability(provider, window, ranking):
    """Show provider reliability scores and error rates."""
    try:
        from terradev_cli.core.price_intelligence import (
            get_provider_reliability,
            get_provider_ranking,
        )
    except ImportError:
        print("ERROR: Price intelligence module not available")
        sys.exit(1)

    if ranking:
        ranked = get_provider_ranking()
        if not ranked:
            print("ERROR: No reliability data yet.")
            print(
                "Tip: Run 'terradev quote' or 'terradev provision' to start tracking."
            )
            return

        print("\n Provider Reliability Ranking")
        print(
            f"{'#':<4} {'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Events':>8}"
        )
        print("─" * 75)
        for i, r in enumerate(ranked, 1):
            medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else f" {i}"
            print(
                f"{medal:<4} {r['provider']:<14} {r['overall_score']:>7.1f} "
                f"{r['quote_success_rate']*100:>8.1f}% {r['provision_success_rate']*100:>8.1f}% "
                f"{r['avg_quote_latency_ms']:>7.0f} {r['avg_provision_latency_ms']:>7.0f} "
                f"{r['total_events']:>7}"
            )
        return

    data = get_provider_reliability(provider=provider, hours=window)
    providers = data.get("providers", {})

    if not providers:
        print(
            "ERROR: No reliability data"
            + (f" for {provider}" if provider else "")
            + f" in the last {window}h"
        )
        print("Tip: Run 'terradev quote' or 'terradev provision' to start tracking.")
        return

    print(f"\n Provider Reliability (last {window}h)")
    print(
        f"{'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Quotes':>8} {'Provs':>8} {'Errors':>8}"
    )
    print("─" * 95)
    for prov, stats in sorted(
        providers.items(), key=lambda x: x[1]["overall_score"], reverse=True
    ):
        err_count = sum(stats["errors"].values())
        print(
            f"{prov:<14} {stats['overall_score']:>7.1f} "
            f"{stats['quote_success_rate']*100:>8.1f}% {stats['provision_success_rate']*100:>8.1f}% "
            f"{stats['avg_quote_latency_ms']:>7.0f} {stats['avg_provision_latency_ms']:>7.0f} "
            f"{stats['quotes']:>7} {stats['provisions']:>7} {err_count:>7}"
        )

        # Show error breakdown if any
        if stats["errors"]:
            for err_msg, cnt in sorted(stats["errors"].items(), key=lambda x: -x[1])[
                :3
            ]:
                print(f"{'':>14} Warning  {err_msg[:60]} (×{cnt})")

    # Overall summary
    all_scores = [s["overall_score"] for s in providers.values()]
    avg_score = sum(all_scores) / len(all_scores)
    print(
        f"\nStatus Average reliability: {avg_score:.1f}/100 across {len(providers)} provider(s)"
    )
