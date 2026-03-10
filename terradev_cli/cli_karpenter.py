"""Karpenter CLI commands for Terradev."""
import os, json, asyncio, click, sys


def register_karpenter_commands(cli, TerradevAPI):
    @cli.group()
    def karpenter():
        """Karpenter GPU node auto-provisioning — install, nodepools, status, events."""
        pass

    @karpenter.command('install')
    @click.option('--version', '-v', 'ver', default='v1.1.1')
    @click.option('--cluster-name', required=True)
    def install(ver, cluster_name):
        """Install or upgrade Karpenter via Helm."""
        from ml_services.kubernetes_service import create_kubernetes_service_from_credentials
        creds = TerradevAPI()._provider_creds('kubernetes')
        creds.update({'karpenter_enabled': 'true', 'karpenter_version': ver, 'cluster_name': cluster_name})
        svc = create_kubernetes_service_from_credentials(creds)
        print(f"Installing Karpenter {ver} on {cluster_name}...")
        r = asyncio.run(svc.install_karpenter())
        print(f"\u2705 Karpenter {r['version']} installed" if r['status'] == 'installed' else f"\u274c {r['error']}")

    @karpenter.command('status')
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
    def status(fmt):
        """Show Karpenter health, node pools, GPU summary."""
        from ml_services.kubernetes_enhanced import create_enhanced_kubernetes_service_from_credentials
        svc = create_enhanced_kubernetes_service_from_credentials(TerradevAPI()._provider_creds('kubernetes'))
        result = asyncio.run(svc.get_monitoring_status())
        if fmt == 'json':
            print(json.dumps(result, indent=2, default=str)); return
        karp = result.get('karpenter', {}); k8s = result.get('kubernetes', {})
        print(f"\n  Karpenter: {karp.get('status', '?')}")
        print(f"  Nodes: {k8s.get('total_nodes', '?')} total, {k8s.get('gpu_nodes', '?')} GPU")
        for name, info in k8s.get('node_pools', {}).items():
            print(f"    {name}: {info['count']} nodes, {info.get('gpu_count', 0)} GPUs")
        print()

    @karpenter.command('nodepools')
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
    def nodepools(fmt):
        """List all Karpenter NodePools."""
        import subprocess as sp
        env = os.environ.copy()
        kp = TerradevAPI()._provider_creds('kubernetes').get('kubeconfig_path')
        if kp: env['KUBECONFIG'] = kp
        r = sp.run(['kubectl', 'get', 'nodepools.karpenter.sh', '-o', 'json'], capture_output=True, text=True, timeout=15, env=env)
        if r.returncode != 0: print(f"\u274c {r.stderr.strip()}"); return
        items = json.loads(r.stdout).get('items', [])
        if fmt == 'json': print(json.dumps(items, indent=2)); return
        if not items: print("  No NodePools found."); return
        for np_item in items:
            spec = np_item.get('spec', {})
            lim = ', '.join(f"{k}={v}" for k, v in spec.get('limits', {}).items()) or 'none'
            print(f"  {np_item['metadata']['name']:<24} {spec.get('disruption', {}).get('consolidationPolicy', '?'):<20} {lim}")

    @karpenter.command('create-nodepool')
    @click.option('--gpu-type', '-g', required=True, help='H100, A100, A10G, L40S, L4, T4, V100')
    @click.option('--cpu-limit', default='1000')
    @click.option('--memory-limit', default='1000Gi')
    def create_nodepool(gpu_type, cpu_limit, memory_limit):
        """Create topology-optimized NodePool + EC2NodeClass."""
        from ml_services.kubernetes_service import create_kubernetes_service_from_credentials
        creds = TerradevAPI()._provider_creds('kubernetes'); creds['karpenter_enabled'] = 'true'
        svc = create_kubernetes_service_from_credentials(creds)
        r = asyncio.run(svc.create_karpenter_provisioner(gpu_type, {'cpu': cpu_limit, 'memory': memory_limit}))
        if r['status'] == 'created':
            t = r.get('topology', {})
            print(f"\u2705 NodePool '{r['provisioner']}' — {', '.join(t.get('instance_families', []))}, RDMA={t.get('gpudirect_rdma')}")
        else: print(f"\u274c {r['error']}")

    @karpenter.command('delete-nodepool')
    @click.argument('name')
    @click.option('--yes', '-y', is_flag=True)
    def delete_nodepool(name, yes):
        """Delete a NodePool and its EC2NodeClass."""
        if not yes: click.confirm(f"Delete '{name}'?", abort=True)
        import subprocess as sp
        env = os.environ.copy()
        kp = TerradevAPI()._provider_creds('kubernetes').get('kubeconfig_path')
        if kp: env['KUBECONFIG'] = kp
        for kind in ['nodepool.karpenter.sh', 'ec2nodeclass.karpenter.k8s.aws']:
            r = sp.run(['kubectl', 'delete', kind, name, '--ignore-not-found'], capture_output=True, text=True, timeout=30, env=env)
            print(f"\u2705 {kind}/{name}" if r.returncode == 0 else f"\u274c {r.stderr.strip()}")

    @karpenter.command('events')
    @click.option('--limit', '-n', default=20, type=int)
    def events(limit):
        """Show recent Karpenter provisioning events."""
        import subprocess as sp
        env = os.environ.copy()
        kp = TerradevAPI()._provider_creds('kubernetes').get('kubeconfig_path')
        if kp: env['KUBECONFIG'] = kp
        r = sp.run(['kubectl', 'get', 'events', '-n', 'karpenter', '--sort-by=.lastTimestamp', '-o', 'json'], capture_output=True, text=True, timeout=15, env=env)
        if r.returncode != 0: print(f"\u274c {r.stderr.strip()}"); return
        for ev in json.loads(r.stdout).get('items', [])[-limit:]:
            ts = (ev.get('lastTimestamp') or '?')[:19]
            print(f"  {ts}  {ev.get('type','?'):<8} {(ev.get('reason') or '?')[:20]:<22} {(ev.get('message') or '')[:60]}")

    @karpenter.command('logs')
    @click.option('--lines', '-n', default=50, type=int)
    def logs(lines):
        """Tail Karpenter controller logs."""
        import subprocess as sp
        env = os.environ.copy()
        kp = TerradevAPI()._provider_creds('kubernetes').get('kubeconfig_path')
        if kp: env['KUBECONFIG'] = kp
        sp.run(['kubectl', 'logs', '-n', 'karpenter', '-l', 'app.kubernetes.io/name=karpenter', '--tail', str(lines), '-c', 'controller'], env=env, timeout=15)

    @karpenter.command('gpu-nodes')
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
    def gpu_nodes(fmt):
        """List GPU nodes with capacity details."""
        from ml_services.kubernetes_service import create_kubernetes_service_from_credentials
        svc = create_kubernetes_service_from_credentials(TerradevAPI()._provider_creds('kubernetes'))
        nodes = asyncio.run(svc.get_gpu_nodes())
        if fmt == 'json': print(json.dumps(nodes, indent=2, default=str)); return
        if not nodes: print("  No GPU nodes found."); return
        for n in nodes:
            print(f"  {n['name'][:34]:<36} {n['instance_type'][:18]:<20} {n['gpu_capacity']} GPUs  {n['status']}")

    @karpenter.command('resources')
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'text']), default='text')
    def resources(fmt):
        """Show cluster CPU, memory, GPU usage."""
        from ml_services.kubernetes_service import create_kubernetes_service_from_credentials
        svc = create_kubernetes_service_from_credentials(TerradevAPI()._provider_creds('kubernetes'))
        res = asyncio.run(svc.get_cluster_resources())
        if fmt == 'json': print(json.dumps(res, indent=2, default=str)); return
        print(f"\n  CPU: {res['total_cpu']:.1f} cores | Memory: {res['total_memory']:.1f} GB | GPUs: {res['total_gpu']} | Nodes: {len(res['nodes'])}\n")

    cli.add_command(karpenter)
