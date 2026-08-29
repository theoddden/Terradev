#!/usr/bin/env python3
"""Infrastructure intelligence commands for the Terradev CLI."""

import logging

import click
from . import cli

logger = logging.getLogger(__name__)

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
@click.option("--gpu-count", type=click.IntRange(1, 1000), default=1, help="Number of GPUs")
@click.option("--memory", type=click.IntRange(1, 10000), help="Memory in GB")
@click.option("--storage", type=click.IntRange(1, 100000), help="Storage in GB")
@click.option("--budget", type=click.FloatRange(0.0, 10000.0), help="Budget constraint ($/hr)")
@click.option("--region", help="Preferred region")
@click.option(
    "--port", type=click.IntRange(1, 65535), multiple=True, help="Expose port(s) via Service (repeatable)"
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
        click.echo("Helm Chart Configuration (Dry Run):")
        click.echo("=" * 50)
        click.echo(f"Workload: {workload}")
        click.echo(f"GPU: {gpu_type} x{gpu_count}")
        click.echo(f"Image: {image}")
        click.echo(f"Memory: {workload_config['memory_gb']}GB")
        click.echo(f"Storage: {workload_config['storage_gb']}GB")
        if budget:
            click.echo(f"Budget: ${budget}/hr")
        click.echo(f"Region: {workload_config['region']}")
        click.echo(f"Spot: {workload_config['spot']}")
        if port:
            click.echo(f"Ports: {list(port)}")
        if stack:
            click.echo(f"Stack: {list(stack)}")
        click.echo()
        click.echo("Chart files that would be generated:")
        click.echo("   Chart.yaml")
        click.echo("   values.yaml")
        click.echo("   templates/")
        if workload in ("training", "cost-optimized"):
            click.echo("     - job.yaml")
        else:
            click.echo("     - deployment.yaml")
            click.echo("     - service.yaml")
            click.echo("     - hpa.yaml")
            click.echo("     - pdb.yaml")
        click.echo("     - configmap.yaml")
        click.echo("     - pvc.yaml (if storage)")
        click.echo("     - serviceaccount.yaml")
        click.echo("     - _helpers.tpl")
        click.echo("     - NOTES.txt")
        click.echo("   README.md")
        return

    # Generate chart
    chart_name = name or f"terradev-{workload}"
    output_dir = output or f"./{chart_name}"

    click.echo(f"Generating Helm chart for {workload} workload...")

    try:
        chart_path = generator.generate_chart(workload_config, output_dir)
        click.echo("Helm chart generated successfully!")
        click.echo(f"   Location: {chart_path}")
        click.echo()
        click.echo("Next steps:")
        click.echo(f"   1. Review the chart: cd {chart_path}")
        click.echo("   2. Customize values: vim values.yaml")
        click.echo(f"   3. Install chart: helm install my-{workload} .")
        click.echo(
            f"   4. Check status: kubectl get all -l app.kubernetes.io/name=my-{workload}"
        )
        click.echo()
        click.echo(f"   Chart README: {chart_path}/README.md")
        click.echo("   Terradev docs: https://terradev.dev/docs")

    except Exception as e:  # noqa: BLE001
        click.echo(f"Failed to generate Helm chart: {e}")
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
        click.echo("ERROR: Price intelligence module not available", err=True)
        raise SystemExit(1)

    if gpu_type:
        data = get_availability(gpu_type, hours=window)
        providers = data.get("providers", {})

        if not providers:
            click.echo(
                f"ERROR: No availability data for {gpu_type.upper()} in the last {window}h", err=True
            )
            click.echo(
                f"Tip: Run 'terradev quote -g {gpu_type}' to start tracking availability.", err=True
            )
            raise SystemExit(1)

        click.echo(f"\n Availability  {gpu_type.upper()} (last {window}h)")
        click.echo(
            f"{'Provider':<14} {'Status':<12} {'Rate':>8} {'Checks':>8} {'Avail':>8} {'Avg ms':>10} {'Last Seen':<20}"
        )
        click.echo("─" * 90)
        for prov, stats in sorted(providers.items()):
            status = "OK: In Stock" if stats["available"] else "ERROR: Sold Out"
            rate_pct = f"{stats['availability_rate'] * 100:.1f}%"
            click.echo(
                f"{prov:<14} {status:<12} {rate_pct:>8} {stats['total_checks']:>8} "
                f"{stats['available_checks']:>8} {stats['avg_response_ms']:>9.0f}ms "
                f"{stats['last_seen'][:19]:<20}"
            )
            if stats.get("last_error"):
                click.echo(f"{'':>14} Warning  Last error: {stats['last_error'][:60]}")
    else:
        summary = get_availability_summary()
        if not summary:
            click.echo("ERROR: No availability data yet.", err=True)
            click.echo("Tip: Run 'terradev quote -g <GPU>' to start tracking.", err=True)
            raise SystemExit(1)

        click.echo("\n Availability Summary (all GPUs, last check)")
        click.echo(f"{'GPU Type':<14} {'Provider':<14} {'Status':<12}")
        click.echo("─" * 42)
        for gtype in sorted(summary.keys()):
            for prov in sorted(summary[gtype].keys()):
                status = "OK: In Stock" if summary[gtype][prov] else "ERROR: Sold Out"
                click.echo(f"{gtype:<14} {prov:<14} {status:<12}")
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
        click.echo("ERROR: Price intelligence module not available", err=True)
        raise SystemExit(1)

    if ranking:
        ranked = get_provider_ranking()
        if not ranked:
            click.echo("ERROR: No reliability data yet.", err=True)
            click.echo(
                "Tip: Run 'terradev quote' or 'terradev provision' to start tracking.", err=True
            )
            raise SystemExit(1)

        click.echo("\n Provider Reliability Ranking")
        click.echo(
            f"{'#':<4} {'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Events':>8}"
        )
        click.echo("─" * 75)
        for i, r in enumerate(ranked, 1):
            medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else f" {i}"
            click.echo(
                f"{medal:<4} {r['provider']:<14} {r['overall_score']:>7.1f} "
                f"{r['quote_success_rate']*100:>8.1f}% {r['provision_success_rate']*100:>8.1f}% "
                f"{r['avg_quote_latency_ms']:>7.0f} {r['avg_provision_latency_ms']:>7.0f} "
                f"{r['total_events']:>7}"
            )
        return

    data = get_provider_reliability(provider=provider, hours=window)
    providers = data.get("providers", {})

    if not providers:
        click.echo(
            "ERROR: No reliability data"
            + (f" for {provider}" if provider else "")
            + f" in the last {window}h"
        , err=True)
        click.echo("Tip: Run 'terradev quote' or 'terradev provision' to start tracking.", err=True)
        raise SystemExit(1)

    click.echo(f"\n Provider Reliability (last {window}h)")
    click.echo(
        f"{'Provider':<14} {'Score':>8} {'Quote %':>9} {'Prov %':>9} {'Q ms':>8} {'P ms':>8} {'Quotes':>8} {'Provs':>8} {'Errors':>8}"
    )
    click.echo("─" * 95)
    for prov, stats in sorted(
        providers.items(), key=lambda x: x[1]["overall_score"], reverse=True
    ):
        err_count = sum(stats["errors"].values())
        click.echo(
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
                click.echo(f"{'':>14} Warning  {err_msg[:60]} (×{cnt})")

    # Overall summary
    all_scores = [s["overall_score"] for s in providers.values()]
    avg_score = sum(all_scores) / len(all_scores)
    click.echo(
        f"\nStatus Average reliability: {avg_score:.1f}/100 across {len(providers)} provider(s)"
    )
