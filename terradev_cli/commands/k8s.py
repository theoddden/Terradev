#!/usr/bin/env python3
"""Kubernetes / GitOps commands for the Terradev CLI."""

import asyncio
import sys

import click
from . import cli
from . import _api

TerraformWrapper = _api.TerraformWrapper
_telemetry = _api._telemetry

@click.group()
def k8s():
    """Kubernetes cluster management with multi-cloud GPU nodes"""
    pass


cli.add_command(k8s)


@k8s.command("create")
@click.argument("cluster_name")
@click.option("--gpu", "-g", required=True, help="GPU type (H100, A100, L40)")
@click.option("--count", "-n", type=int, required=True, help="Number of GPU nodes")
@click.option("--max-price", type=float, default=4.00, help="Maximum price per hour")
@click.option("--multi-cloud", is_flag=True, help="Use multi-cloud provisioning")
@click.option("--prefer-spot", is_flag=True, default=True, help="Prefer spot instances")
@click.option("--aws-region", default="us-west-2", help="AWS region")
@click.option("--gcp-region", default="us-central1", help="GCP region")
@click.option(
    "--control-plane",
    type=click.Choice(["eks", "gke", "self-hosted"]),
    default="eks",
    help="Control plane type",
)
def k8s_create(
    cluster_name,
    gpu,
    count,
    max_price,
    multi_cloud,
    prefer_spot,
    aws_region,
    gcp_region,
    control_plane,
):
    """Create multi-cloud Kubernetes GPU cluster"""
    if not TerraformWrapper:
        print("ERROR: Kubernetes wrapper not available")
        sys.exit(1)

    if _telemetry:
        _telemetry.log_action(
            "k8s_cluster_create",
            {
                "cluster_name": cluster_name,
                "gpu_type": gpu,
                "node_count": count,
                "multi_cloud": multi_cloud,
                "max_price": max_price,
                "prefer_spot": prefer_spot,
            },
        )

    wrapper = TerraformWrapper()

    cluster_config = {
        "name": cluster_name,
        "gpu_type": gpu,
        "node_count": count,
        "max_price": max_price,
        "multi_cloud": multi_cloud,
        "prefer_spot": prefer_spot,
        "aws_region": aws_region,
        "gcp_region": gcp_region,
        "control_plane": control_plane,
    }

    print(f" Creating Kubernetes cluster '{cluster_name}'...")
    print(f" GPU Type: {gpu}")
    print(f" Node Count: {count}")
    print(f"COST: Max Price: ${max_price}/hr")
    print(f"  Multi-Cloud: {multi_cloud}")
    print(f" Spot Instances: {prefer_spot}")
    print("")
    print(" Topology optimization (auto-applied):")
    print("   Kubelet Topology Manager: restricted (NUMA-aligned)")
    print("   CPU Manager: static (pinned cores)")
    print("   GPUDirect RDMA: enabled (nvidia_peermem)")
    if count > 1:
        print(f"   SR-IOV: enabled ({count} nodes, VF-per-GPU pairing)")
        print("   NCCL: IB enabled, GDR_LEVEL=PIX, GDR_READ=1")
    else:
        print("   SR-IOV: single-node (not required)")
    print("   PCIe locality: GPU-NIC pairs forced to same NUMA node")

    success = wrapper.create_cluster(cluster_config)

    if success:
        print(f"OK: Cluster '{cluster_name}' created successfully!")
        print("   Topology: NUMA-aligned, GPUDirect RDMA, Topology Manager=restricted")
        print(f"INFO:  Run 'terradev k8s info {cluster_name}' for details")
        print(
            f" Run 'export KUBECONFIG=~/.terradev/clusters/{cluster_name}.json' to connect"
        )
    else:
        print(f"ERROR: Failed to create cluster '{cluster_name}'")


@k8s.command("destroy")
@click.argument("cluster_name")
def k8s_destroy(cluster_name):
    """Destroy Kubernetes cluster"""
    if not TerraformWrapper:
        print("ERROR: Kubernetes wrapper not available")
        return

    if _telemetry:
        _telemetry.log_action("k8s_cluster_destroy", {"cluster_name": cluster_name})

    wrapper = TerraformWrapper()

    print(f"  Destroying Kubernetes cluster '{cluster_name}'...")

    success = wrapper.destroy_cluster(cluster_name)

    if success:
        print(f"OK: Cluster '{cluster_name}' destroyed successfully!")
    else:
        print(f"ERROR: Failed to destroy cluster '{cluster_name}'")


@k8s.command("list")
def k8s_list():
    """List all Kubernetes clusters"""
    if not TerraformWrapper:
        print("ERROR: Kubernetes wrapper not available")
        sys.exit(1)

    wrapper = TerraformWrapper()
    clusters = wrapper.list_clusters()

    if not clusters:
        print(" No clusters found")
        return

    print("Plan Kubernetes Clusters:")
    print("=" * 80)
    for cluster in clusters:
        name = cluster.get("name", "unknown")
        status = cluster.get("status", "unknown")
        created = cluster.get("created_at", "unknown")
        outputs = cluster.get("outputs", {})

        print(f"  Name: {name}")
        print(f"Status Status: {status}")
        print(f" Created: {created}")

        if outputs:
            gpu_summary = outputs.get("gpu_summary", {})
            if gpu_summary:
                print(f" GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
                print(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
                print(f"COST: Cost/hr: ${outputs.get('total_cost_per_hour', 0):.2f}")

        print("-" * 40)


@k8s.command("info")
@click.argument("cluster_name")
def k8s_info(cluster_name):
    """Get detailed cluster information"""
    if not TerraformWrapper:
        print("ERROR: Kubernetes wrapper not available")
        return

    wrapper = TerraformWrapper()
    info = wrapper.get_cluster_info(cluster_name)

    if not info:
        print(f"ERROR: Cluster '{cluster_name}' not found")
        return

    print(f"Plan Cluster Information: {cluster_name}")
    print("=" * 80)

    outputs = info.get("outputs", {})

    if outputs:
        # GPU Summary
        gpu_summary = outputs.get("gpu_summary", {})
        if gpu_summary:
            print(f" GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
            print(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
            print(f"COST: Max Price: ${gpu_summary.get('max_price', 0):.2f}/hr")
            print(f" Actual Average: ${gpu_summary.get('actual_average', 0):.2f}/hr")
            print(f" Spot Preferred: {gpu_summary.get('prefer_spot', False)}")

        # Cost Breakdown
        cost_breakdown = outputs.get("cost_breakdown", {})
        if cost_breakdown:
            print("\nCOST: Cost Breakdown:")
            print(f"{'Provider':<12} {'Nodes':<6} {'Cost/hr':<10} {'Cost/mo':<12}")
            print("-" * 50)
            for provider, breakdown in cost_breakdown.items():
                print(
                    f"{provider:<12} {breakdown.get('nodes', 0):<6} ${breakdown.get('cost_hr', 0):<9.2f} ${breakdown.get('cost_mo', 0):<11.2f}"
                )

        # Savings Analysis
        savings = outputs.get("savings_analysis", {})
        if savings:
            print("\n Savings Analysis:")
            print(f"AWS-only cost: ${savings.get('aws_only_cost_per_hour', 0):.2f}/hr")
            print(
                f"Multi-cloud cost: ${savings.get('multi_cloud_cost_per_hour', 0):.2f}/hr"
            )
            print(
                f"Savings: ${savings.get('savings_per_hour', 0):.2f}/hr ({savings.get('savings_percentage', 0):.1f}%)"
            )

        # Next Steps
        next_steps = outputs.get("next_steps", [])
        if next_steps:
            print("\nDeploying Next Steps:")
            for step in next_steps:
                print(f"  {step}")

    else:
        print("ERROR: No detailed information available")


@cli.command()
@click.option(
    "--workload",
    "-w",
    type=click.Choice(["training", "inference", "cost-optimized", "high-performance"]),
    default="training",
    help="Workload type (maps to Karpenter provisioner)",
)
@click.option(
    "--image", required=True, help="Docker image (e.g. pytorch/pytorch:latest)"
)
@click.option("--cmd", default=None, help="Command to run inside the container")
@click.option(
    "--gpu-count",
    "-G",
    type=int,
    default=None,
    help="Number of GPUs (default: per workload profile)",
)
@click.option(
    "--budget",
    "-b",
    type=float,
    default=None,
    help="Max $/hr budget  forces spot if < $2/hr",
)
@click.option(
    "--namespace", "-n", default="terradev-workloads", help="Kubernetes namespace"
)
@click.option(
    "--name", default=None, help="Job/Deployment name (auto-generated if omitted)"
)
@click.option("--env", "-e", multiple=True, help="Environment variables KEY=VALUE")
@click.option("--mount", multiple=True, help="Volume mounts host:container")
@click.option(
    "--option", "-o", type=int, help="Deployment option index from smart-deploy"
)
@click.option("--memory", type=int, help="Memory in GB")
@click.option("--storage", "-s", type=int, help="Storage in GB")
@click.option("--hours", type=float, default=1.0, help="Estimated runtime in hours")
@click.option("--region", help="Preferred region")
@click.option("--dry-run", is_flag=True, help="Show recommendation without deploying")
def smart_deploy(
    image,
    workload,
    cmd,
    gpu_count,
    budget,
    namespace,
    name,
    env,
    mount,
    option,
    memory,
    storage,
    hours,
    region,
    dry_run,
):
    """Smart deployment with automatic optimization"""
    try:
        from terradev_cli.core.deployment_router import SmartDeploymentRouter
    except ImportError:
        print(
            "ERROR: Smart deployment module not available. Install terradev_cli package."
        )
        sys.exit(1)
    import asyncio

    async def _smart_deploy():
        router = SmartDeploymentRouter()
        user_request = {
            "gpu_type": "A100",  # Default, will be overridden by recommendations
            "gpu_count": gpu_count,
            "memory_gb": memory or 16,
            "storage_gb": storage or 100,
            "estimated_hours": hours,
            "workload_type": workload,
            "budget": budget,
            "region": region,
        }

        print(" Analyzing deployment options...")

        # Get recommendations
        recommendations = await router.recommend_deployments(user_request)

        if not recommendations:
            print("ERROR: No deployment options available")
            return

        if option is not None:
            # Deploy specific option
            if option >= len(recommendations):
                print(
                    f"ERROR: Invalid option. Available options: 0-{len(recommendations)-1}"
                )
                return

            chosen = recommendations[option]
            print(
                f"Deploying option {option}: {chosen.provider} {chosen.instance_type}"
            )
            print(f"   Type: {chosen.type.value}")
            print(f"   Cost: ${chosen.price_per_hour:.2f}/hr")
            print(f"   Setup time: {chosen.setup_time_minutes} minutes")
            print(f"   Confidence: {chosen.confidence:.1%}")

            if dry_run:
                print(" Dry run - not actually deploying")
                return

            # Execute deployment
            try:
                result = await router.execute_deployment(
                    chosen, router.requirements_analyzer.analyze(user_request)
                )
                print(f"OK: Deployment started: {result['deployment_id']}")
                print(f"   Status: {result['status']}")
                print(f"   Estimated ready: {result['estimated_ready_time']}")
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: Deployment failed: {e}")
        else:
            # Show all recommendations
            print("\n Smart Deployment Recommendations:")
            print("=" * 60)

            for i, rec in enumerate(recommendations[:5]):
                print(f"\n{i}. {rec.provider} {rec.instance_type}")
                print(f"   Type: {rec.type.value}")
                print(
                    f"   Cost: ${rec.price_per_hour:.2f}/hr (total: ${rec.estimated_total_cost:.2f})"
                )
                print(f"   Setup: {rec.setup_time_minutes} minutes")
                print(f"   Confidence: {rec.confidence:.1%}")
                print(f"   Risk: {rec.risk_score:.1%}")

                print("   Pros:")
                for pro in rec.pros[:3]:
                    print(f"      {pro}")

                if len(rec.cons) > 0:
                    print("   Cons:")
                    for con in rec.cons[:2]:
                        print(f"      {con}")

                print(f"   Deploy with: terradev smart-deploy --option {i}")

    asyncio.run(_smart_deploy())


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


# Price Percentiles Command
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# Availability Command
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# Provider Reliability Command
# ═══════════════════════════════════════════════════════════════════════


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


@cli.group()
def gitops():
    """GitOps automation and infrastructure as code"""
    pass


@gitops.command()
@click.option(
    "--provider",
    type=click.Choice(["github", "gitlab", "bitbucket", "azure_devops"]),
    required=True,
    help="Git provider",
)
@click.option(
    "--repo",
    "--repository",
    "repository",
    required=True,
    help="Repository name (format: owner/repo)",
)
@click.option(
    "--tool",
    type=click.Choice(["argocd", "flux"]),
    default="argocd",
    help="GitOps tool",
)
@click.option("--cluster", required=True, help="Cluster name")
@click.option("--git-url", help="Git repository URL (auto-generated if not provided)")
@click.option("--git-token", help="Git access token")
@click.option("--namespace", default="gitops-system", help="Namespace for GitOps tools")
@click.option(
    "--auto-sync/--no-auto-sync", default=True, help="Enable automatic synchronization"
)
@click.option("--prune/--no-prune", default=True, help="Enable resource pruning")
def init(
    provider, repository, tool, cluster, git_url, git_token, namespace, auto_sync, prune
):
    """Initialize GitOps repository and structure"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitProvider, GitOpsTool

    provider_map = {
        "github": GitProvider.GITHUB,
        "gitlab": GitProvider.GITLAB,
        "bitbucket": GitProvider.BITBUCKET,
        "azure_devops": GitProvider.AZURE_DEVOPS,
    }

    tool_map = {"argocd": GitOpsTool.ARGOCD, "flux": GitOpsTool.FLUX}

    config = GitOpsConfig(
        provider=provider_map[provider],
        repository=repository,
        tool=tool_map[tool],
        cluster_name=cluster,
        git_url=git_url,
        git_token=git_token,
        namespace=namespace,
        auto_sync=auto_sync,
        prune_resources=prune,
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
@click.option(
    "--tool", type=click.Choice(["argocd", "flux"]), required=True, help="GitOps tool"
)
@click.option("--cluster", required=True, help="Cluster name")
@click.option("--namespace", default="gitops-system", help="Namespace for GitOps tools")
def bootstrap(tool, cluster, namespace):
    """Bootstrap GitOps tool on the cluster"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool, GitProvider

    # This is a simplified bootstrap - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool[tool.upper()],
        cluster_name=cluster,
        namespace=namespace,
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
@click.option("--cluster", required=True, help="Cluster name")
@click.option("--environment", default="prod", help="Environment to sync")
@click.option(
    "--tool",
    type=click.Choice(["argocd", "flux"]),
    default="argocd",
    help="GitOps tool",
)
def sync(cluster, environment, tool):
    """Sync cluster with Git repository"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool, GitProvider

    # This is a simplified sync - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool[tool.upper()],
        cluster_name=cluster,
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
@click.option(
    "--dry-run/--apply", default=True, help="Dry run validation or apply changes"
)
@click.option("--cluster", help="Cluster name for validation")
@click.option("--environment", default="prod", help="Environment to validate")
def validate(dry_run, cluster, environment):
    """Validate GitOps configuration"""
    from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitOpsTool, GitProvider

    # This is a simplified validation - in practice, you'd load config from previous init
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,  # Default
        repository="terradev/infra",  # Default
        tool=GitOpsTool.ARGOCD,  # Default
        cluster_name=cluster or "default",
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

        if results["valid"]:
            print("Configuration is valid")
        else:
            print("Configuration validation failed:")
            for error in results["errors"]:
                print(f"  Error: {error}")

        if results["warnings"]:
            print("Warnings:")
            for warning in results["warnings"]:
                print(f"  Warning: {warning}")

    asyncio.run(run_validate())


