#!/usr/bin/env python3
"""Commands for the Terradev CLI."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from . import cli
from terradev_cli.commands._api import TerradevAPI
from terradev_cli.core.universal_manifest import UniversalManifest, Component
from terradev_cli.core.adapters.orchestrator import UniversalOrchestrator

logger = logging.getLogger(__name__)

@cli.group()
def sso():
    """Enterprise SSO authentication"""
    pass
@sso.command("status")
def sso_status():
    """Show SSO configuration status"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("WARNING:  Enterprise auth not initialized")
        print(
            "   Install enterprise dependencies: pip install terradev-cli[enterprise]"
        )
        return

    enabled_providers = api.enterprise_auth.list_enabled_providers()
    if enabled_providers:
        print("OK: SSO is configured")
        print("   Enabled providers:", ", ".join(enabled_providers))
    else:
        print("WARNING:  No SSO providers configured")
        print("   Configure providers with: terradev sso configure")
@sso.command("configure")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["azure_ad", "okta", "google_workspace", "auth0"]),
    required=True,
    help="SSO provider",
)
@click.option("--client-id", help="Client ID (for OIDC providers)")
@click.option("--client-secret", help="Client secret (for OIDC providers)")
@click.option("--domain", help="Domain (for Okta/Auth0)")
@click.option("--tenant-id", help="Tenant ID (for Azure AD)")
@click.option("--entity-id", help="Entity ID (for SAML providers)")
@click.option("--sso-url", help="SSO URL (for SAML providers)")
@click.option("--certificate", help="Certificate (for SAML providers)")
def sso_configure(
    provider,
    client_id,
    client_secret,
    domain,
    tenant_id,
    entity_id,
    sso_url,
    certificate,
):
    """Configure SSO provider"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("ERROR: Enterprise auth not initialized")
        print(
            "   Install enterprise dependencies: pip install terradev-cli[enterprise]"
        )
        return

    config = {}

    is_oidc = client_id and client_secret
    is_saml = entity_id and sso_url

    if is_oidc and provider in ["google_workspace", "auth0", "azure_ad"]:
        # OIDC providers
        if provider == "auth0":
            if not domain:
                print("ERROR: Domain required for Auth0")
                return
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "domain": domain,
                    "enabled": True,
                }
            )
        elif provider == "azure_ad":
            if not tenant_id:
                print("ERROR: Tenant ID required for Azure AD")
                return
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "tenant_id": tenant_id,
                    "enabled": True,
                }
            )
        else:
            # google_workspace
            config = api.enterprise_auth.get_sso_provider_config(provider)
            config.update(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "enabled": True,
                }
            )

    elif is_saml and provider in ["okta", "azure_ad"]:
        # SAML providers
        config = api.enterprise_auth.get_sso_provider_config(provider)
        config.update(
            {
                "entity_id": entity_id,
                "sso_url": sso_url,
                "certificate": certificate or "",
                "enabled": True,
            }
        )

    else:
        if provider in ["google_workspace", "auth0"] or (
            provider == "azure_ad" and not is_saml
        ):
            print("ERROR: Client ID and secret required for OIDC providers")
        elif provider in ["okta"] or (provider == "azure_ad" and not is_oidc):
            print("ERROR: Entity ID and SSO URL required for SAML providers")
        else:
            print(f"ERROR: {provider} not supported with the provided credentials")
        return

    try:
        api.enterprise_auth.enable_sso_provider(provider, config)
        print(f"OK: {provider} SSO provider configured successfully")
        print("   Test the configuration with: terradev sso test")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to configure {provider}: {e}")
@sso.command("test")
@click.option("--provider", "-p", help="Test specific provider")
def sso_test(provider):
    """Test SSO provider configuration"""
    api = TerradevAPI()

    if not api.enterprise_auth:
        print("ERROR: Enterprise auth not initialized")
        return

    if provider:
        # Test specific provider
        config = api.enterprise_auth.get_sso_provider_config(provider)
        if not config or not config.get("enabled"):
            print(f"ERROR: Provider {provider} not configured")
            return

        print(f"Testing {provider}...")
        if api.enterprise_auth.test_sso_provider(provider):
            print(f"OK: {provider} configuration appears valid")
        else:
            print(f"WARNING: {provider} configuration test failed")
    else:
        # Test all providers
        enabled_providers = api.enterprise_auth.list_enabled_providers()
        if not enabled_providers:
            print("WARNING:  No SSO providers configured")
            return

        print("Testing all configured providers...")
        for p in enabled_providers:
            if api.enterprise_auth.test_sso_provider(p):
                print(f"OK: {p} configuration appears valid")
            else:
                print(f"WARNING: {p} configuration test failed")
@cli.command("mcp")
@click.argument("action", type=click.Choice(["serve", "install", "list-tools"]))
@click.option(
    "--client",
    type=click.Choice(["claude-desktop", "cursor", "windsurf", "continue", "cline"]),
    help="Client to install MCP config for",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="MCP transport protocol",
)
def mcp(action, client, transport):
    """Run Terradev as an MCP server for agent integration.

    Makes Terradev callable from AI agents (Claude Desktop, Cursor, Windsurf, Continue, Cline).

    Actions:
      serve: Start MCP server (default: stdio transport)
      install: Install MCP config for a specific client
      list-tools: List all available MCP tools
    """
    try:
        from terradev_cli.mcp import run_server, install_config, list_tools
    except ImportError:
        click.echo(
            "Error: MCP module not found. Install with: pip install mcp", err=True
        )
        raise SystemExit(1)

    if action == "serve":
        run_server(transport=transport)
    elif action == "install":
        if not client:
            click.echo("Error: --client is required for install action", err=True)
            raise SystemExit(1)
        install_config(client)
    elif action == "list-tools":
        list_tools()
@cli.group()
def local():
    """Local GPU discovery and hybrid compute pool management.

    Discover GPUs on this machine or remote hosts via SSH, register them
    into your compute pool alongside cloud providers, and route workloads
    to the cheapest available compute  including $0/hr local hardware.
    """
    pass
@local.command("scan")
@click.option("--host", default=None, help="Remote host IP/hostname to scan via SSH")
@click.option("--user", default="ubuntu", help="SSH username for remote scan")
@click.option("--key", default=None, help="Path to SSH private key for remote scan")
@click.option(
    "--detailed", is_flag=True, help="Show full topology, PCIe, NUMA, clock details"
)
@click.option(
    "--register", is_flag=True, help="Auto-register discovered GPUs into pool"
)
@click.option(
    "--name",
    default=None,
    help="Name for registered pool entry (auto-generated if omitted)",
)
def local_scan(host, user, key, detailed, register, name):
    """Scan local machine or remote host for GPUs.

    Uses Rust NVML bindings (5-10x faster than nvidia-smi) with automatic
    fallback to nvidia-smi parsing if the Rust extension is unavailable.

    Examples:

        terradev local scan

        terradev local scan --detailed

        terradev local scan --host 192.168.1.50 --user ubuntu --key ~/.ssh/id_rsa

        terradev local scan --register --name workstation-4090
    """
    import subprocess
    import datetime

    target = host if host else "localhost"
    click.echo(f"Scanning {target} for GPUs...")

    def _run_nvidia_smi(remote_host=None, remote_user=None, remote_key=None):
        query = "index,name,memory.total,driver_version,utilization.gpu,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current,compute_cap"
        if remote_host:
            # Validate inputs to prevent shell injection
            import re

            if not re.match(r"^[a-zA-Z0-9._-]+$", remote_user):
                return "", 1
            if not re.match(r"^[a-zA-Z0-9._-]+$", remote_host):
                return "", 1
            if remote_key and not re.match(r"^[a-zA-Z0-9._/~-]+$", remote_key):
                return "", 1

            # Build SSH command as argument list (no shell=True)
            ssh_args = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=10",
            ]
            if remote_key:
                ssh_args.extend(["-i", remote_key])
            ssh_args.extend(
                [
                    f"{remote_user}@{remote_host}",
                    f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
                ]
            )
            try:
                result = subprocess.run(
                    ssh_args, capture_output=True, text=True, timeout=15
                )
                return result.stdout.strip(), result.returncode
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                return "", 1
        else:
            # Local execution - use list-args for injection safety
            cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=15
                )
                return result.stdout.strip(), result.returncode
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                return "", 1

    # Try Rust NVML first (local only), then nvidia-smi
    gpus = []
    use_rust = False
    if not host:
        try:
            from terradev_cli.core.gpu_discovery import GPUDiscoveryWrapper

            disc = GPUDiscoveryWrapper(cache_ttl_secs=0)
            state = disc.discover_gpus()
            if state and state.get("total_count", 0) > 0:
                use_rust = True
                for g in state.get("gpus", []):
                    gpus.append(g)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pass

    if not use_rust:
        raw, rc = _run_nvidia_smi(host, user, key)
        if rc != 0 or not raw:
            click.echo(
                f"No GPUs found on {target} or nvidia-smi not available.", err=True
            )
            return
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else 0,
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]) if parts[2] else 0,
                    "driver_version": parts[3],
                    "utilization_gpu": float(parts[4]) if parts[4] else 0,
                    "temperature": float(parts[5]) if parts[5] else 0,
                    "power_draw": float(parts[6]) if parts[6] else 0,
                    "power_limit": float(parts[7]) if parts[7] else 0,
                    "pcie_gen": parts[8],
                    "pcie_width": parts[9],
                    "compute_cap": parts[10],
                }
            )

    if not gpus:
        click.echo(f"No GPUs found on {target}.")
        return

    click.echo(f"\nFound {len(gpus)} GPU{'s' if len(gpus) > 1 else ''} on {target}:\n")
    for g in gpus:
        idx = g.get("index", 0)
        gpu_name = g.get("name", "Unknown")
        mem_mb = g.get("memory_total_mb", g.get("memory_total", 0))
        mem_gb = (
            round(float(mem_mb) / 1024, 1) if float(mem_mb) > 100 else float(mem_mb)
        )
        driver = g.get("driver_version", "N/A")
        util = g.get("utilization_gpu", g.get("utilization", 0))
        temp = g.get("temperature", 0)
        click.echo(
            f"  [{idx}] {gpu_name}  {mem_gb}GB  Driver {driver}  Util: {util}%  Temp: {temp}C"
        )
        if detailed:
            pcie_gen = g.get("pcie_gen", "N/A")
            pcie_w = g.get("pcie_width", "N/A")
            pwr_draw = g.get("power_draw", 0)
            pwr_lim = g.get("power_limit", 0)
            compute = g.get("compute_cap", "N/A")
            numa = g.get("numa_node", "N/A")
            click.echo(
                f"      PCIe: Gen{pcie_gen} x{pcie_w}  NUMA: {numa}  Compute: {compute}"
            )
            click.echo(f"      Power: {pwr_draw}W / {pwr_lim}W TDP")

    if register or (not register and click.confirm("\nRegister in pool?")):
        pool_name = (
            name
            if name
            else f"local-{'remote-' if host else ''}{gpus[0].get('name','gpu').replace(' ','').lower()}-{datetime.datetime.now().strftime('%H%M')}"
        )
        _register_local_pool(gpus, pool_name, host, user, key)
        click.echo(f"\nRegistered as '{pool_name}'. View with: terradev local pool")
@local.command("register")
@click.option("--name", required=True, help="Name for this pool entry")
@click.option("--host", default=None, help="Remote host (omit for localhost)")
@click.option("--user", default="ubuntu", help="SSH username for remote host")
@click.option("--key", default=None, help="SSH private key path")
def local_register(name, host, user, key):
    """Register a local or remote GPU host into your compute pool.

    Example:

        terradev local register --name workstation-4090

        terradev local register --name lab-node-01 --host 10.0.0.5 --user ubuntu
    """
    import subprocess

    target = host or "localhost"
    click.echo(f"Scanning {target}...")
    query = "index,name,memory.total,driver_version,utilization.gpu,temperature.gpu"
    if host:
        # Validate inputs to prevent shell injection
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", user):
            click.echo("Error: Invalid username format", err=True)
            return
        if not re.match(r"^[a-zA-Z0-9._-]+$", host):
            click.echo("Error: Invalid hostname format", err=True)
            return
        if key and not re.match(r"^[a-zA-Z0-9._/~-]+$", key):
            click.echo("Error: Invalid key path format", err=True)
            return

        # Build SSH command as argument list (no shell=True)
        ssh_args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
        ]
        if key:
            ssh_args.extend(["-i", key])
        ssh_args.extend(
            [
                f"{user}@{host}",
                f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
            ]
        )
        try:
            result = subprocess.run(
                ssh_args, capture_output=True, text=True, timeout=15
            )
            raw = result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            click.echo(f"Error scanning {target}: {e}", err=True)
            return
    else:
        # Local execution - use list-args for injection safety
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            raw = result.stdout.strip()
        except Exception as e:  # noqa: BLE001
            click.echo(f"Error scanning {target}: {e}", err=True)
            return
    if not raw:
        click.echo(f"No GPUs found on {target}.", err=True)
        return
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else 0,
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]) if parts[2] else 0,
                    "driver_version": parts[3],
                    "utilization_gpu": float(parts[4]) if parts[4] else 0,
                    "temperature": float(parts[5]) if parts[5] else 0,
                }
            )
    _register_local_pool(gpus, name, host, user, key)
    click.echo(
        f"Registered '{name}' with {len(gpus)} GPU(s). View with: terradev local pool"
    )
@local.command("pool")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--remove", default=None, help="Remove a pool entry by name")
def local_pool(fmt, remove):
    """View or manage your hybrid compute pool (local + cloud instances).

    Shows all registered local/remote GPU hosts alongside active cloud instances.

    Example:

        terradev local pool

        terradev local pool --format json

        terradev local pool --remove workstation-4090
    """
    import json
    import os

    pool_path = os.path.expanduser("~/.terradev/local_pool.json")
    pool = {}
    if os.path.exists(pool_path):
        try:
            with open(pool_path) as f:
                pool = json.load(f)
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pool = {}

    if remove:
        if remove in pool:
            del pool[remove]
            with open(pool_path, "w") as f:
                json.dump(pool, f, indent=2)
            click.echo(f"Removed '{remove}' from pool.")
        else:
            click.echo(f"'{remove}' not found in pool.", err=True)
        return

    if fmt == "json":
        click.echo(json.dumps(pool, indent=2))
        return

    if not pool:
        click.echo("No local pool entries registered.")
        click.echo("Add one with: terradev local scan --register")
        return

    click.echo(
        f"\nCOMPUTE POOL ({len(pool)} local resource{'s' if len(pool) > 1 else ''})\n"
    )
    click.echo(
        f"{'NAME':<24} {'GPU':<12} {'VRAM':>6}  {'PROVIDER':<12} {'$/HR':>7}  STATUS"
    )
    click.echo("-" * 72)
    for entry_name, entry in pool.items():
        gpus = entry.get("gpus", [])
        gpu_name = (
            gpus[0].get("name", "Unknown").replace("NVIDIA ", "") if gpus else "Unknown"
        )
        mem_mb = gpus[0].get("memory_total_mb", 0) if gpus else 0
        mem_gb = (
            f"{round(float(mem_mb)/1024, 0):.0f}GB"
            if float(mem_mb) > 100
            else f"{mem_mb}GB"
        )
        provider = entry.get("provider", "local")
        price = entry.get("price_per_hour", 0.0)
        host = entry.get("host", "localhost")
        status = "localhost" if host == "localhost" else host
        click.echo(
            f"{entry_name:<24} {gpu_name:<12} {mem_gb:>6}  {provider:<12} ${price:>6.2f}  {status}"
        )

    click.echo("\nCloud instances: run 'terradev status --live' for cloud pool.")
    click.echo(
        "To provision preferring local: terradev provision -g RTX4090 --prefer-local"
    )
@cli.group()
def agent():
    """Provision and manage heterogeneous agent fleets.

    Multi-tier GPU provisioning purpose-built for multi-agent LLM workloads.
    Automatically maps agent count to hardware tiers based on empirical
    workload research (decode-dominated, KV cache preservation critical).

    \b
    Tiers provisioned:
      reasoning  — H100 SXM: long-context KV preservation (P95: 120K tokens)
      decode     — A100 80GB: memory-bandwidth-optimised token streaming
      cpu_tools  — 48-vCPU: Bash/WebFetch/file-op tool execution

    \b
    Examples:
      terradev agent plan   --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --topology ./agent-fleet.yaml
      terradev agent status --fleet-id ag_abc123
      terradev agent scale  --fleet-id ag_abc123 --tier decode --count 8
      terradev agent cost   --fleet-id ag_abc123
      terradev agent list
      terradev agent teardown --fleet-id ag_abc123
    """
    pass
@agent.command(name="plan")
@click.option("--agents", "-n", type=int, required=True, help="Number of concurrent agent loops to provision for")
@click.option("--model", "-m", default="meta-llama/Llama-3.1-70B-Instruct", help="Model to serve across the fleet")
@click.option("--reasoning", type=click.Choice(["instant", "thinking"]), default="instant", help="Reasoning mode: instant (faster) or thinking (extended CoT, 45-67% more output tokens)")
@click.option("--planner-gpu", default=None, help="Override reasoning tier GPU type (e.g. H100_SXM)")
@click.option("--planner-count", type=int, default=None, help="Override reasoning tier instance count")
@click.option("--worker-gpu", default=None, help="Override decode tier GPU type (e.g. A100_SXM_80)")
@click.option("--worker-count", type=int, default=None, help="Override decode tier instance count")
@click.option("--cpu-cores", type=int, default=48, help="vCPU count for CPU tools tier instances")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
def agent_plan(agents, model, reasoning, planner_gpu, planner_count, worker_gpu, worker_count, cpu_cores, fmt):
    """Plan a heterogeneous agent fleet without provisioning.

    Shows the recommended tier configuration, hardware selection rationale,
    KV cache budget, and cost estimate based on arXiv:2605.26297 research.

    \b
    Examples:
      terradev agent plan --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent plan --agents 32 --model meta-llama/Llama-3.1-8B-Instruct --format json
      terradev agent plan --agents 8 --planner-gpu H100_SXM --worker-gpu A100_SXM_80
    """
    import json as _json
    from terradev_cli.core.agentic_topology import AgentTopologyPlanner

    planner = AgentTopologyPlanner()

    if planner_gpu and worker_gpu:
        spec = planner.from_explicit(
            n_agents=agents, model=model,
            planner_gpu=planner_gpu, planner_count=planner_count or max(1, agents // 10),
            worker_gpu=worker_gpu, worker_count=worker_count or agents,
            cpu_cores=cpu_cores, reasoning=reasoning,
        )
    else:
        spec = planner.infer_from_agent_count(n_agents=agents, model=model, reasoning=reasoning)
        if planner_count:
            spec.tiers["reasoning"].count = planner_count
        if worker_count:
            spec.tiers["decode"].count = worker_count

    if fmt == "json":
        cost = planner.estimate_cost(spec)
        output = spec.to_dict()
        output["cost"] = {
            "reasoning_hr": cost.reasoning_hr,
            "decode_hr": cost.decode_hr,
            "cpu_hr": cost.cpu_hr,
            "total_hr": cost.total_hr,
            "daily": cost.daily,
            "monthly": cost.monthly,
            "cost_per_agent_hr": cost.cost_per_agent_hr,
        }
        click.echo(_json.dumps(output, indent=2))
    else:
        planner.print_plan(spec)
        click.echo(f"\nTo provision this fleet:")
        click.echo(f"  terradev agent deploy --agents {agents} --model {model}")
        click.echo(f"\nTo provision with explicit overrides:")
        r = spec.tiers["reasoning"]
        d = spec.tiers["decode"]
        click.echo(
            f"  terradev agent deploy --agents {agents} --model {model} "
            f"--planner-gpu {r.gpu_type} --planner-count {r.count} "
            f"--worker-gpu {d.gpu_type} --worker-count {d.count}"
        )
@agent.command(name="deploy")
@click.option("--agents", "-n", type=int, default=None, help="Number of concurrent agent loops")
@click.option("--model", "-m", default="meta-llama/Llama-3.1-70B-Instruct", help="Model to serve")
@click.option("--reasoning", type=click.Choice(["instant", "thinking"]), default="instant")
@click.option("--topology", type=click.Path(exists=True), default=None, help="Path to agent-fleet.yaml spec file")
@click.option("--planner-gpu", default=None, help="Reasoning tier GPU type")
@click.option("--planner-count", type=int, default=None, help="Reasoning tier instance count")
@click.option("--worker-gpu", default=None, help="Decode tier GPU type")
@click.option("--worker-count", type=int, default=None, help="Decode tier instance count")
@click.option("--cpu-cores", type=int, default=48, help="vCPU count for CPU tools tier")
@click.option("--providers", "-p", multiple=True, help="Cloud providers to use (e.g. runpod vastai)")
@click.option("--max-price", type=float, default=None, help="Max price per GPU/hr in USD")
@click.option("--dry-run", is_flag=True, help="Show allocation plan without provisioning")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_deploy(agents, model, reasoning, topology, planner_gpu, planner_count, worker_gpu, worker_count, cpu_cores, providers, max_price, dry_run, fmt):
    """Provision a heterogeneous agent fleet across all tiers simultaneously.

    Provisions reasoning (H100), decode (A100), and CPU tools tiers in parallel
    using the existing DAGExecutor wave-parallel orchestration.

    \b
    Examples:
      terradev agent deploy --agents 16 --model meta-llama/Llama-3.1-70B-Instruct
      terradev agent deploy --agents 32 --dry-run
      terradev agent deploy --topology ./agent-fleet.yaml
      terradev agent deploy --agents 8 --planner-gpu H100_SXM --worker-gpu A100_SXM_80
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_topology import AgentTopologyPlanner
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    if topology:
        import yaml
        with open(topology) as f:
            spec_data = yaml.safe_load(f)
        agents = agents or spec_data.get("n_agents", 16)
        model = spec_data.get("model", model)

    if not agents:
        click.echo("Error: --agents or --topology required", err=True)
        raise SystemExit(1)

    planner = AgentTopologyPlanner()
    if planner_gpu and worker_gpu:
        spec = planner.from_explicit(
            n_agents=agents, model=model,
            planner_gpu=planner_gpu, planner_count=planner_count or max(1, agents // 10),
            worker_gpu=worker_gpu, worker_count=worker_count or agents,
            cpu_cores=cpu_cores, reasoning=reasoning,
        )
    else:
        spec = planner.infer_from_agent_count(n_agents=agents, model=model, reasoning=reasoning)
        if planner_count:
            spec.tiers["reasoning"].count = planner_count
        if worker_count:
            spec.tiers["decode"].count = worker_count

    if not dry_run:
        planner.print_plan(spec)
        click.echo(f"\nProvisioning fleet {spec.fleet_id}...")
    else:
        click.echo("[DRY RUN] Fleet plan — no instances will be provisioned:\n")
        planner.print_plan(spec)

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.provision_fleet(
        spec=spec,
        dry_run=dry_run,
        providers=list(providers) if providers else None,
        max_price_hr=max_price,
    ))

    if fmt == "json":
        output = {
            "fleet_id": result.fleet_id,
            "success": result.success,
            "dry_run": dry_run,
            "wall_ms": round(result.total_wall_ms, 1),
            "cost_estimate": {
                "total_hr": result.cost_estimate.total_hr,
                "daily": result.cost_estimate.daily,
                "monthly": result.cost_estimate.monthly,
            },
            "tiers": {k: v.count for k, v in spec.tiers.items()},
            "state_path": result.state_path,
            "errors": result.errors,
        }
        click.echo(_json.dumps(output, indent=2))
        return

    if result.success:
        status_tag = "[DRY RUN]" if dry_run else "PROVISIONED"
        click.echo(f"\n{status_tag}  Fleet: {result.fleet_id}")
        click.echo(f"  Model:   {spec.model}")
        click.echo(f"  Agents:  {spec.n_agents} concurrent loops")
        click.echo(f"  Cost:    ${result.cost_estimate.total_hr:.2f}/hr  (${result.cost_estimate.monthly:.2f}/mo)")
        click.echo(f"  Tiers:   {spec.tiers['reasoning'].count}× reasoning | {spec.tiers['decode'].count}× decode | {spec.tiers['cpu_tools'].count}× cpu_tools")
        if not dry_run:
            click.echo(f"  State:   {result.state_path}")
        click.echo()
        click.echo(f"  terradev agent status --fleet-id {result.fleet_id}")
        click.echo(f"  terradev agent cost   --fleet-id {result.fleet_id}")
        click.echo(f"  terradev agent scale  --fleet-id {result.fleet_id} --tier decode --count <N>")
    else:
        click.echo(f"\nProvisioning errors:", err=True)
        for e in result.errors:
            click.echo(f"  {e}", err=True)
@agent.command(name="status")
@click.option("--fleet-id", required=True, help="Fleet ID returned by 'terradev agent deploy'")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_status(fleet_id, fmt):
    """Show live status of a fleet — tier health, KV hit rate, queue depth, cost.

    Key metrics explained:
      kv_hit_rate  — target >0.85. Below 0.80 = cache thrashing (expensive recompute).
      ttft_p95_ms  — reasoning tier target <2000ms. Above = scale out reasoning.
      queue_depth  — decode tier pending requests. Above 6 = scale out decode.

    (Metrics from arXiv:2605.26297 empirical benchmarking)
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    status = asyncio.run(provisioner.fleet_status(fleet_id))

    if status is None:
        click.echo(f"Fleet {fleet_id} not found.", err=True)
        raise SystemExit(1)

    if fmt == "json":
        output = {
            "fleet_id": status.fleet_id,
            "model": status.model,
            "n_agents": status.n_agents,
            "kv_cache_pressure": status.kv_cache_pressure,
            "total_cost_hr": status.total_cost_hr,
            "uptime_s": status.uptime_s,
            "warnings": status.warnings,
            "tiers": {
                name: {
                    "instances": t.instances,
                    "healthy": t.healthy,
                    "failed": t.failed,
                    "kv_hit_rate": t.kv_hit_rate,
                    "decode_queue_depth": t.decode_queue_depth,
                    "ttft_p95_ms": t.ttft_p95_ms,
                    "cost_hr": t.cost_hr,
                }
                for name, t in status.tiers.items()
            },
        }
        click.echo(_json.dumps(output, indent=2))
        return

    pressure_icon = {"healthy": "OK", "warning": "WARN", "critical": "CRIT"}.get(status.kv_cache_pressure, "?")
    click.echo(f"\nFLEET STATUS  [{fleet_id}]")
    click.echo(f"Model: {status.model}  |  {status.n_agents} agents  |  KV cache: {pressure_icon}  |  ${status.total_cost_hr:.2f}/hr")
    click.echo(f"Uptime: {status.uptime_s / 3600:.1f}h")
    click.echo()
    click.echo(f"{'TIER':<16} {'INSTANCES':>9} {'HEALTHY':>7} {'FAILED':>6}  {'KV HIT':>7}  {'TTFT P95':>9}  {'QUEUE':>5}  {'$/HR':>6}")
    click.echo("-" * 80)
    for tname, t in status.tiers.items():
        kv_str = f"{t.kv_hit_rate:.0%}" if t.kv_hit_rate > 0 else "n/a"
        ttft_str = f"{t.ttft_p95_ms:.0f}ms" if t.ttft_p95_ms > 0 else "n/a"
        q_str = str(t.decode_queue_depth) if t.gpu_type else "n/a"
        kv_warn = " !" if t.kv_hit_rate > 0 and t.kv_hit_rate < 0.80 else ""
        click.echo(
            f"{tname:<16} {t.instances:>9} {t.healthy:>7} {t.failed:>6}  {kv_str:>6}{kv_warn}  "
            f"{ttft_str:>9}  {q_str:>5}  ${t.cost_hr:>5.2f}"
        )
    if status.warnings:
        click.echo()
        for w in status.warnings:
            click.echo(f"  WARN: {w}")
@agent.command(name="scale")
@click.option("--fleet-id", required=True, help="Fleet ID")
@click.option("--tier", required=True, type=click.Choice(["reasoning", "decode", "cpu_tools"]), help="Tier to scale")
@click.option("--count", required=True, type=int, help="New instance count for this tier")
@click.option("--providers", "-p", multiple=True, help="Providers to use for scale-out instances")
def agent_scale(fleet_id, tier, count, providers):
    """Scale a single fleet tier up or down without affecting other tiers.

    KV cache state on existing instances is PRESERVED during scale operations.
    New instances are added to the pool; the router distributes new requests to them.

    \b
    Examples:
      terradev agent scale --fleet-id ag_abc123 --tier decode --count 8
      terradev agent scale --fleet-id ag_abc123 --tier reasoning --count 3
      terradev agent scale --fleet-id ag_abc123 --tier cpu_tools --count 4
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.scale_tier(
        fleet_id=fleet_id,
        tier=tier,
        new_count=count,
        providers=list(providers) if providers else None,
    ))
    click.echo(_json.dumps(result, indent=2))
@agent.command(name="cost")
@click.option("--fleet-id", required=True, help="Fleet ID")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_cost(fleet_id, fmt):
    """Show real-time cost breakdown for a fleet by tier.

    \b
    Example:
      terradev agent cost --fleet-id ag_abc123
    """
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    cost = provisioner.fleet_cost(fleet_id)

    if cost is None:
        click.echo(f"Fleet {fleet_id} not found.", err=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(_json.dumps(cost, indent=2))
        return

    click.echo(f"\nFLEET COST  [{fleet_id}]")
    click.echo(f"  Uptime:       {cost['uptime_hr']:.2f}h")
    click.echo(f"  Rate:         ${cost['cost_per_hr']:.2f}/hr")
    click.echo(f"  Accrued:      ${cost['accrued_cost']:.2f}")
    click.echo(f"  Projected/day: ${cost['projected_daily']:.2f}")
    click.echo(f"  Projected/mo:  ${cost['projected_monthly']:.2f}")
    click.echo(f"  Per-agent/hr:  ${cost['cost_per_agent_hr']:.4f}")
    click.echo()
    click.echo(f"  BREAKDOWN:")
    click.echo(f"    reasoning  ${cost['breakdown']['reasoning']:.2f}/hr")
    click.echo(f"    decode     ${cost['breakdown']['decode']:.2f}/hr")
    click.echo(f"    cpu_tools  ${cost['breakdown']['cpu_tools']:.2f}/hr")
@agent.command(name="list")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def agent_list(fmt):
    """List all known agent fleets.

    \b
    Example:
      terradev agent list
    """
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    provisioner = AgenticProvisioner()
    fleets = provisioner.list_fleets()

    if fmt == "json":
        click.echo(_json.dumps(fleets, indent=2, default=str))
        return

    if not fleets:
        click.echo("No agent fleets found. Deploy one with: terradev agent deploy --agents 16")
        return

    click.echo(f"\nAGENT FLEETS ({len(fleets)})\n")
    click.echo(f"{'FLEET ID':<28} {'MODEL':<36} {'AGENTS':>6} {'$/HR':>7}  STATUS")
    click.echo("-" * 85)
    for f in fleets:
        import datetime
        created = datetime.datetime.fromtimestamp(f["created_at"]).strftime("%Y-%m-%d %H:%M")
        status_str = "OK" if f["success"] else "ERR"
        click.echo(
            f"{f['fleet_id']:<28} {f['model'][:35]:<36} "
            f"{f['n_agents']:>6} ${f['cost_hr']:>6.2f}  {status_str}  {created}"
        )
@agent.command(name="teardown")
@click.option("--fleet-id", required=True, help="Fleet ID to destroy")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def agent_teardown(fleet_id, yes):
    """Terminate all fleet instances and remove fleet state.

    \b
    Example:
      terradev agent teardown --fleet-id ag_abc123
      terradev agent teardown --fleet-id ag_abc123 --yes
    """
    import asyncio
    import json as _json
    from terradev_cli.core.agentic_provisioner import AgenticProvisioner

    if not yes:
        click.confirm(f"Destroy fleet {fleet_id} and all its instances?", abort=True)

    provisioner = AgenticProvisioner()
    result = asyncio.run(provisioner.teardown_fleet(fleet_id))
    click.echo(_json.dumps(result, indent=2))
def _agent_vector_db_manifest(name: str, adapter: str, config: str) -> UniversalManifest:
    """Build a manifest for an agent-facing vector database."""
    cfg = json.loads(config) if config else {}
    component = Component(
        kind="vector_store",
        name=name,
        adapter=adapter,
        config=cfg,
    )
    return UniversalManifest(
        name=f"agent-vdb-{name}",
        version="0.1.0",
        components=[component],
    )


@agent.group("vector-db")
def vector_db():
    """Provision vector databases for agent memory and retrieval.

    Examples:
      terradev agent vector-db up --name agent-memory --adapter qdrant
      terradev agent vector-db up --name docs-weaviate --adapter weaviate
      terradev agent vector-db down --name agent-memory
    """
    pass


@vector_db.command("up")
@click.option("--name", "-n", default="agent-memory", help="Vector DB name")
@click.option(
    "--adapter",
    "-a",
    default="qdrant",
    type=click.Choice(["qdrant", "weaviate"]),
    help="Vector DB adapter",
)
@click.option("--config", "-c", default="{}", help="JSON adapter config")
@click.option("--manifest", "-m", type=click.Path(exists=False), help="Path to universal manifest")
def vector_db_up(name, adapter, config, manifest):
    """Provision a vector database for an agent fleet."""
    if manifest:
        m = UniversalManifest.load(Path(manifest))
    else:
        m = _agent_vector_db_manifest(name, adapter, config)

    orchestrator = UniversalOrchestrator(m)

    async def _main():
        await orchestrator.initialize()
        return orchestrator.to_result().result

    try:
        result = asyncio.run(_main())
        click.echo(f"OK: Vector DB '{name}' ({adapter}) is ready")
        click.echo(json.dumps(result, indent=2, default=str))
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to provision vector DB: {e}", err=True)
        raise SystemExit(1)


@vector_db.command("down")
@click.option("--name", "-n", default="agent-memory", help="Vector DB name")
@click.option(
    "--adapter",
    "-a",
    default="qdrant",
    type=click.Choice(["qdrant", "weaviate"]),
    help="Vector DB adapter",
)
@click.option("--config", "-c", default="{}", help="JSON adapter config")
@click.option("--manifest", "-m", type=click.Path(exists=False), help="Path to universal manifest")
def vector_db_down(name, adapter, config, manifest):
    """Teardown a vector database provisioned for an agent fleet."""
    if manifest:
        m = UniversalManifest.load(Path(manifest))
    else:
        m = _agent_vector_db_manifest(name, adapter, config)

    orchestrator = UniversalOrchestrator(m)

    try:
        asyncio.run(orchestrator.teardown())
        click.echo(f"OK: Vector DB '{name}' torn down")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to teardown vector DB: {e}", err=True)
        raise SystemExit(1)


@agent.group("skill")
def skill():
    """Manage skill.md files and attach them to Letta agents."""
    pass


@skill.command("init")
@click.option("--name", "-n", required=True, help="Skill name")
@click.option("--output", "-o", default=None, help="Output path (default: <name>.skill.md)")
@click.option("--description", "-d", default="", help="Short description")
@click.option("--tools", default="", help="Comma-separated tool names")
def skill_init(name, output, description, tools):
    """Create a skill.md template for an agent."""
    path = Path(output) if output else Path(f"{name}.skill.md")
    tool_list = [t.strip() for t in tools.split(",") if t.strip()]

    content = f"""# Skill: {name}

## Description
{description or "Describe what this skill does."}

## Tools
{chr(10).join(f"- {t}" for t in tool_list) or "- "}

## Instructions
Write the step-by-step instructions the agent should follow when this skill is active.

## Example
Show a short example interaction or expected output.
"""
    path.write_text(content)
    click.echo(f"OK: Created skill template: {path}")


@skill.command("attach")
@click.option("--agent-id", "-a", required=True, help="Letta agent ID")
@click.option(
    "--skill",
    "-s",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to skill.md",
)
@click.option("--label", "-l", default="skill", help="Memory block label")
@click.option(
    "--environment",
    "-e",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
def skill_attach(agent_id, skill, label, environment):
    """Attach a skill.md to a Letta agent as a durable memory block."""
    from . import letta

    try:
        client = letta._get_client(environment)
    except SystemExit:
        return

    content = Path(skill).read_text()
    try:
        client.agents.core_memory.append(
            agent_id=agent_id,
            label=label,
            value=content,
        )
    except AttributeError:
        client.agents.messages.create(
            agent_id=agent_id,
            input=f"Skill '{label}' instructions:\n{content}",
        )

    click.echo(f"OK: Attached skill '{skill}' to agent {agent_id} under '{label}'")


