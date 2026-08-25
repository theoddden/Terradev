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









# Price Percentiles Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Availability Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Provider Reliability Command
# ═══════════════════════════════════════════════════════════════════════




