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








# Price Percentiles Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Availability Command
# ═══════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════
# Provider Reliability Command
# ═══════════════════════════════════════════════════════════════════════




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


