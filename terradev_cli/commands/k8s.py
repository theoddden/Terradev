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
@click.option("--count", "-n", type=click.IntRange(1, 1000), required=True, help="Number of GPU nodes")
@click.option("--max-price", type=click.FloatRange(0.0, 10000.0), default=4.00, help="Maximum price per hour")
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
        click.echo("ERROR: Kubernetes wrapper not available", err=True)
        raise SystemExit(1)

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

    click.echo(f" Creating Kubernetes cluster '{cluster_name}'...")
    click.echo(f" GPU Type: {gpu}")
    click.echo(f" Node Count: {count}")
    click.echo(f"COST: Max Price: ${max_price}/hr")
    click.echo(f"  Multi-Cloud: {multi_cloud}")
    click.echo(f" Spot Instances: {prefer_spot}")
    click.echo("")
    click.echo(" Topology optimization (auto-applied):")
    click.echo("   Kubelet Topology Manager: restricted (NUMA-aligned)")
    click.echo("   CPU Manager: static (pinned cores)")
    click.echo("   GPUDirect RDMA: enabled (nvidia_peermem)")
    if count > 1:
        click.echo(f"   SR-IOV: enabled ({count} nodes, VF-per-GPU pairing)")
        click.echo("   NCCL: IB enabled, GDR_LEVEL=PIX, GDR_READ=1")
    else:
        click.echo("   SR-IOV: single-node (not required)")
    click.echo("   PCIe locality: GPU-NIC pairs forced to same NUMA node")

    success = wrapper.create_cluster(cluster_config)

    if success:
        click.echo(f"OK: Cluster '{cluster_name}' created successfully!")
        click.echo("   Topology: NUMA-aligned, GPUDirect RDMA, Topology Manager=restricted")
        click.echo(f"INFO:  Run 'terradev k8s info {cluster_name}' for details")
        click.echo(
            f" Run 'export KUBECONFIG=~/.terradev/clusters/{cluster_name}.json' to connect"
        )
    else:
        click.echo(f"ERROR: Failed to create cluster '{cluster_name}'", err=True)


@k8s.command("destroy")
@click.argument("cluster_name")
def k8s_destroy(cluster_name):
    """Destroy Kubernetes cluster"""
    if not TerraformWrapper:
        click.echo("ERROR: Kubernetes wrapper not available", err=True)
        return

    if _telemetry:
        _telemetry.log_action("k8s_cluster_destroy", {"cluster_name": cluster_name})

    wrapper = TerraformWrapper()

    click.echo(f"  Destroying Kubernetes cluster '{cluster_name}'...")

    success = wrapper.destroy_cluster(cluster_name)

    if success:
        click.echo(f"OK: Cluster '{cluster_name}' destroyed successfully!")
    else:
        click.echo(f"ERROR: Failed to destroy cluster '{cluster_name}'", err=True)


@k8s.command("list")
def k8s_list():
    """List all Kubernetes clusters"""
    if not TerraformWrapper:
        click.echo("ERROR: Kubernetes wrapper not available", err=True)
        raise SystemExit(1)

    wrapper = TerraformWrapper()
    clusters = wrapper.list_clusters()

    if not clusters:
        click.echo(" No clusters found")
        return

    click.echo("Plan Kubernetes Clusters:")
    click.echo("=" * 80)
    for cluster in clusters:
        name = cluster.get("name", "unknown")
        status = cluster.get("status", "unknown")
        created = cluster.get("created_at", "unknown")
        outputs = cluster.get("outputs", {})

        click.echo(f"  Name: {name}")
        click.echo(f"Status Status: {status}")
        click.echo(f" Created: {created}")

        if outputs:
            gpu_summary = outputs.get("gpu_summary", {})
            if gpu_summary:
                click.echo(f" GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
                click.echo(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
                click.echo(f"COST: Cost/hr: ${outputs.get('total_cost_per_hour', 0):.2f}")

        click.echo("-" * 40)


@k8s.command("info")
@click.argument("cluster_name")
def k8s_info(cluster_name):
    """Get detailed cluster information"""
    if not TerraformWrapper:
        click.echo("ERROR: Kubernetes wrapper not available", err=True)
        return

    wrapper = TerraformWrapper()
    info = wrapper.get_cluster_info(cluster_name)

    if not info:
        click.echo(f"ERROR: Cluster '{cluster_name}' not found", err=True)
        return

    click.echo(f"Plan Cluster Information: {cluster_name}")
    click.echo("=" * 80)

    outputs = info.get("outputs", {})

    if outputs:
        # GPU Summary
        gpu_summary = outputs.get("gpu_summary", {})
        if gpu_summary:
            click.echo(f" GPU Type: {gpu_summary.get('gpu_type', 'unknown')}")
            click.echo(f"Status Total Nodes: {gpu_summary.get('total_gpus', 0)}")
            click.echo(f"COST: Max Price: ${gpu_summary.get('max_price', 0):.2f}/hr")
            click.echo(f" Actual Average: ${gpu_summary.get('actual_average', 0):.2f}/hr")
            click.echo(f" Spot Preferred: {gpu_summary.get('prefer_spot', False)}")

        # Cost Breakdown
        cost_breakdown = outputs.get("cost_breakdown", {})
        if cost_breakdown:
            click.echo("\nCOST: Cost Breakdown:")
            click.echo(f"{'Provider':<12} {'Nodes':<6} {'Cost/hr':<10} {'Cost/mo':<12}")
            click.echo("-" * 50)
            for provider, breakdown in cost_breakdown.items():
                click.echo(
                    f"{provider:<12} {breakdown.get('nodes', 0):<6} ${breakdown.get('cost_hr', 0):<9.2f} ${breakdown.get('cost_mo', 0):<11.2f}"
                )

        # Savings Analysis
        savings = outputs.get("savings_analysis", {})
        if savings:
            click.echo("\n Savings Analysis:")
            click.echo(f"AWS-only cost: ${savings.get('aws_only_cost_per_hour', 0):.2f}/hr")
            click.echo(
                f"Multi-cloud cost: ${savings.get('multi_cloud_cost_per_hour', 0):.2f}/hr"
            )
            click.echo(
                f"Savings: ${savings.get('savings_per_hour', 0):.2f}/hr ({savings.get('savings_percentage', 0):.1f}%)"
            )

        # Next Steps
        next_steps = outputs.get("next_steps", [])
        if next_steps:
            click.echo("\nDeploying Next Steps:")
            for step in next_steps:
                click.echo(f"  {step}")

    else:
        click.echo("ERROR: No detailed information available", err=True)









# Price Percentiles Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Availability Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Provider Reliability Command
# ═══════════════════════════════════════════════════════════════════════




