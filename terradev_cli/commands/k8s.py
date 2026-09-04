#!/usr/bin/env python3
"""Kubernetes / GitOps commands for the Terradev CLI."""

import asyncio
import sys
import uuid
from pathlib import Path

import click
from . import cli
from . import _api

TerraformWrapper = _api.TerraformWrapper
_telemetry = _api._telemetry

_KNROPS_PROVIDERS = [
    "aws", "azure", "baseten", "crusoe", "digitalocean", "e2enetworks",
    "gcore", "gcp", "huggingface", "hyperstack", "inferx", "latitude",
    "runpod", "siliconflow", "tensordock", "vastai", "yottalabs",
]

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

# ═══════════════════════════════════════════════════════════════════════
# k8s node — knr-ops GitOps bridge
# ═══════════════════════════════════════════════════════════════════════

@k8s.group("node")
def node():
    """Provision GPU nodes via Terradev providers and register them in a knr-ops repository.

    Each subcommand interacts with the GitOps repository at --repo. Committed
    manifests are picked up by Flux automatically; k0smotron SSHes into the
    provisioned VM and joins it to the gpu-workers CAPI cluster.

    \b
    Typical workflow:
      1. terradev k8s node add --provider runpod --gpu H100 --repo ./knr-ops
      2. git -C ./knr-ops push origin main          # Flux reconciles within ~60s
      3. kubectl get machines -n default             # watch the node join
      4. terradev k8s node list                      # show the fleet
      5. terradev k8s node rm gpu-runpod-a1b2c3 --repo ./knr-ops
    """
    pass


@node.command("add")
@click.option(
    "--provider", "-p", required=True,
    type=click.Choice(_KNROPS_PROVIDERS, case_sensitive=False),
    help="Terradev provider to provision the GPU VM from",
)
@click.option("--gpu", "-g", required=True, help="GPU type (H100, A100, RTX4090, ...)")
@click.option(
    "--repo", "-r", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to knr-ops Git repository",
)
@click.option("--region", default=None, help="Provider region (provider default if omitted)")
@click.option("--spot", is_flag=True, help="Request a spot/interruptible instance")
@click.option(
    "--ssh-key-secret", default="gpu-ssh-key", show_default=True,
    help="Name of the Kubernetes Secret holding the SSH private key for k0smotron",
)
@click.option(
    "--k0s-version", default="v1.33.0+k0s.0", show_default=True,
    help="k0s version string to install on the worker node",
)
@click.option(
    "--ssh-public-key", default="",
    help="SSH public key to inject at provision time (provider-dependent)",
)
@click.pass_context
def node_add(ctx, provider, gpu, repo, region, spot, ssh_key_secret, k0s_version, ssh_public_key):
    """Provision a GPU VM via a Terradev provider and commit it to knr-ops.

    Calls the provider's API to create the instance, waits up to 60 s for an
    IP address, then writes RemoteMachine / K0sWorkerConfig / Machine manifests
    and commits them to the GitOps repo. Push the repo to trigger Flux.

    \b
    Examples:
      terradev k8s node add --provider runpod --gpu H100 --repo ~/knr-ops
      terradev k8s node add --provider vastai --gpu A100 --repo ~/knr-ops --spot
    """
    from terradev_cli.core.gitops_manager import KnrOpsNodeBridge
    from terradev_cli.providers.provider_factory import ProviderFactory

    api = ctx.obj["api"]
    creds = api.credentials.get(provider, {})
    if not creds:
        click.echo(
            f"ERROR: No credentials configured for '{provider}'.\n"
            f"  Run: terradev configure --provider {provider}",
            err=True,
        )
        raise SystemExit(1)

    node_id = f"gpu-{provider}-{uuid.uuid4().hex[:6]}"
    repo_path = Path(repo)

    click.echo(f"Provisioning {gpu} node via {provider}...")

    async def _provision():
        factory = ProviderFactory()
        p = factory.create_provider(provider, creds)
        try:
            result = await p.provision_instance(
                instance_type=gpu,
                region=region or "us-east-1",
                gpu_type=gpu,
                ssh_public_key=ssh_public_key,
            )
            instance_id = result.get(
                "instance_id", f"{provider}_{uuid.uuid4().hex[:8]}"
            )
            address = (
                result.get("ip_address")
                or result.get("public_ip")
                or result.get("address")
                or result.get("host")
            )
            if not address:
                click.echo("  Waiting for instance IP address (up to 60 s)...")
                for _ in range(12):
                    await asyncio.sleep(5)
                    status = await p.get_instance_status(instance_id)
                    address = (
                        status.get("ip_address")
                        or status.get("public_ip")
                        or status.get("address")
                        or status.get("host")
                    )
                    if address:
                        break
            return instance_id, address
        finally:
            await p.aclose()

    try:
        instance_id, address = asyncio.run(_provision())
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: Provisioning failed: {exc}", err=True)
        raise SystemExit(1)

    click.echo(f"  Instance: {instance_id}")

    if not address:
        click.echo("WARNING: Instance provisioned but IP not yet available.")
        click.echo("  Once the instance is ready, run:")
        click.echo(
            f"    terradev k8s node ready {node_id} --address <IP> "
            f"--provider {provider} --gpu {gpu} --instance-id {instance_id} --repo {repo}"
        )
        return

    click.echo(f"  Address:  {address}")

    bridge = KnrOpsNodeBridge(repo_path)
    manifests = bridge.generate_manifests(
        node_id, address, provider, gpu, ssh_key_secret, instance_id, k0s_version
    )
    bridge.commit_node(node_id, manifests)

    click.echo(f"OK: Node {node_id} committed to {repo}")
    click.echo("  Push the repo to trigger Flux reconciliation:")
    click.echo(f"    git -C {repo} push origin main")
    click.echo("  Then watch: kubectl get machines -n default")


@node.command("ready")
@click.argument("node_id")
@click.option("--address", required=True, help="Public IP address of the provisioned instance")
@click.option("--provider", "-p", required=True,
              type=click.Choice(_KNROPS_PROVIDERS, case_sensitive=False))
@click.option("--gpu", "-g", required=True, help="GPU type (H100, A100, ...)")
@click.option("--instance-id", default="", help="Provider instance ID (for later termination)")
@click.option(
    "--repo", "-r", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to knr-ops Git repository",
)
@click.option("--ssh-key-secret", default="gpu-ssh-key", show_default=True)
@click.option("--k0s-version", default="v1.33.0+k0s.0", show_default=True)
def node_ready(node_id, address, provider, gpu, instance_id, repo, ssh_key_secret, k0s_version):
    """Complete registration of a pending node once its IP address is known.

    Use this when 'node add' exited without committing because the provider
    had not yet assigned an IP to the instance.
    """
    from terradev_cli.core.gitops_manager import KnrOpsNodeBridge

    repo_path = Path(repo)
    bridge = KnrOpsNodeBridge(repo_path)
    manifests = bridge.generate_manifests(
        node_id, address, provider, gpu, ssh_key_secret, instance_id, k0s_version
    )
    bridge.commit_node(node_id, manifests)

    click.echo(f"OK: Node {node_id} ({address}) committed to {repo}")
    click.echo(f"  Push to trigger Flux: git -C {repo} push origin main")


@node.command("list")
@click.option(
    "--repo", "-r", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to knr-ops Git repository",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def node_list(repo, as_json):
    """List GPU nodes currently committed to the knr-ops repository."""
    import json as _json
    from terradev_cli.core.gitops_manager import KnrOpsNodeBridge

    nodes = KnrOpsNodeBridge.read_nodes(Path(repo))
    if not nodes:
        click.echo(f"No nodes found in {repo}. Run: terradev k8s node add ...")
        return

    if as_json:
        click.echo(_json.dumps(nodes, indent=2))
        return

    header = f"{'NODE ID':<30} {'PROVIDER':<14} {'GPU':<10} {'ADDRESS':<18} {'PROVISIONED'}"
    click.echo(header)
    click.echo("-" * 90)
    for n in nodes:
        click.echo(
            f"{n['id']:<30} {n['provider'] or '':<14} {n['gpu_type'] or '':<10} "
            f"{n.get('address', '') or '(pending)':<18} {n.get('provisioned_at', '')[:19]}"
        )


@node.command("rm")
@click.argument("node_id")
@click.option(
    "--repo", "-r", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to knr-ops Git repository",
)
@click.option("--keep-instance", is_flag=True, help="Remove from GitOps only; do not terminate the VM")
@click.pass_context
def node_rm(ctx, node_id, repo, keep_instance):
    """Remove a GPU node from knr-ops and optionally terminate the VM.

    Commits a deletion to the GitOps repo; Flux removes the Machine and
    k0smotron drains the node. The VM is terminated unless --keep-instance
    is set.

    \b
    Example:
      terradev k8s node rm gpu-runpod-a1b2c3 --repo ~/knr-ops
    """
    from terradev_cli.core.gitops_manager import KnrOpsNodeBridge
    from terradev_cli.providers.provider_factory import ProviderFactory

    repo_path = Path(repo)
    nodes = KnrOpsNodeBridge.read_nodes(repo_path)
    record = next((n for n in nodes if n["id"] == node_id), None)
    if not record:
        click.echo(f"ERROR: Node '{node_id}' not found in {repo}.", err=True)
        raise SystemExit(1)

    bridge = KnrOpsNodeBridge(repo_path)
    bridge.remove_node_files(node_id)
    click.echo(f"  Removed manifests for {node_id} from {repo}")

    if not keep_instance and record.get("instance_id"):
        api = ctx.obj["api"]
        creds = api.credentials.get(record["provider"], {})
        if creds:
            async def _terminate():
                factory = ProviderFactory()
                p = factory.create_provider(record["provider"], creds)
                try:
                    await p.terminate_instance(record["instance_id"])
                finally:
                    await p.aclose()

            try:
                asyncio.run(_terminate())
                click.echo(f"  Terminated instance {record['instance_id']} on {record['provider']}")
            except Exception as exc:  # noqa: BLE001
                click.echo(f"WARNING: Could not terminate instance: {exc}", err=True)
        else:
            click.echo(
                f"WARNING: No credentials for '{record['provider']}'; instance not terminated.",
                err=True,
            )

    click.echo(f"OK: Node {node_id} removed")
    click.echo(f"  Push to trigger Flux: git -C {repo} push origin main")




