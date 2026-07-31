#!/usr/bin/env python3
"""Commands for the Terradev CLI."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from . import cli
from terradev_cli.commands._api import TerradevAPI

logger = logging.getLogger(__name__)

from .inference import _resolve_provision_nodes
from .ml import _parse_vllm_endpoint
@cli.group("agentic-serving")
def agentic_serving():
    """Agentic inference serving  KV cache TTL, prefix caching, LMCache, priority scheduling."""
    pass
@agentic_serving.command("configure")
@click.option(
    "--engine",
    type=click.Choice(["vllm", "sglang"]),
    default="vllm",
    help="Inference engine",
)
@click.option("--model", prompt="Model ID", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option(
    "--tp", "tensor_parallel_size", default=1, type=int, help="Tensor parallel size"
)
@click.option("--max-model-len", default=32768, type=int)
@click.option("--gpu-mem", "gpu_memory_utilization", default=0.85, type=float)
@click.option(
    "--lmcache/--no-lmcache",
    "lmcache_enabled",
    default=True,
    help="Enable LMCache KV offload",
)
@click.option(
    "--lmcache-backend", type=click.Choice(["cpu", "disk", "redis"]), default="cpu"
)
@click.option(
    "--disaggregation/--no-disaggregation",
    default=False,
    help="Prefill-decode disaggregation",
)
def agentic_serving_configure(
    engine,
    model,
    tensor_parallel_size,
    max_model_len,
    gpu_memory_utilization,
    lmcache_enabled,
    lmcache_backend,
    disaggregation,
):
    """Configure agentic inference serving settings."""
    api = TerradevAPI()
    api._save_provider_creds(
        "agentic_serving",
        {
            "engine": engine,
            "model": model,
            "tensor_parallel_size": str(tensor_parallel_size),
            "max_model_len": str(max_model_len),
            "gpu_memory_utilization": str(gpu_memory_utilization),
            "enable_prefix_caching": "true",
            "lmcache_enabled": str(lmcache_enabled).lower(),
            "lmcache_backend": lmcache_backend,
            "disaggregation_enabled": str(disaggregation).lower(),
        },
    )
    print(f"\u2705 Agentic serving configured: {engine} + {model}")
    print("   Prefix caching: enabled")
    print(
        f"   LMCache: {'enabled (' + lmcache_backend + ')' if lmcache_enabled else 'disabled'}"
    )
    print(f"   PD disaggregation: {'enabled' if disaggregation else 'disabled'}")
@agentic_serving.command("show-config")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def agentic_serving_show_config(fmt):
    """Show current agentic serving configuration."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_vllm_args,
        generate_sglang_args,
        generate_lmcache_config,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    engine_args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )

    if fmt == "json":
        print(
            json.dumps(
                {
                    "engine": config.engine,
                    "model": config.model,
                    "engine_args": engine_args,
                    "lmcache": generate_lmcache_config(config),
                    "ttl": {
                        "min": config.ttl_min,
                        "max": config.ttl_max,
                        "multiplier": config.ttl_multiplier,
                    },
                    "disaggregation": config.disaggregation_enabled,
                    "prefix_caching": config.enable_prefix_caching,
                },
                indent=2,
            )
        )
    else:
        print("\n  Agentic Serving Config:")
        print(f"  Engine:            {config.engine}")
        print(f"  Model:             {config.model}")
        print(f"  TP:                {config.tensor_parallel_size}")
        print(f"  Max Model Len:     {config.max_model_len}")
        print(f"  GPU Mem Util:      {config.gpu_memory_utilization}")
        print(f"  Prefix Caching:    {config.enable_prefix_caching}")
        print(
            f"  LMCache:           {config.lmcache_enabled} ({config.lmcache_backend})"
        )
        print(f"  Disaggregation:    {config.disaggregation_enabled}")
        print(
            f"  KV TTL Range:      {config.ttl_min}s - {config.ttl_max}s (x{config.ttl_multiplier})"
        )
        print("\n  Engine Args:")
        for a in engine_args:
            print(f"    {a}")
        print()
@agentic_serving.command("launch-args")
def agentic_serving_launch_args():
    """Print engine launch arguments for copy-paste."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_vllm_args,
        generate_sglang_args,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )
    if config.engine == "vllm":
        print("\npython -m vllm.entrypoints.openai.api_server \\")
    else:
        print("\npython -m sglang.launch_server \\")
    for i, a in enumerate(args):
        sep = " \\" if i < len(args) - 1 else ""
        print(f"  {a}{sep}")
    print()
@agentic_serving.command("lmcache-env")
def agentic_serving_lmcache_env():
    """Print LMCache environment variables."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_lmcache_env,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    env = generate_lmcache_env(config)
    if not env:
        print("  LMCache is disabled.")
        return
    print()
    for k, v in env.items():
        print(f'export {k}="{v}"')
    print()
@agentic_serving.command("k8s")
@click.option("--namespace", "-n", default="inference", help="K8s namespace")
def agentic_serving_k8s(namespace):
    """Print K8s deployment manifests for agentic inference."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_k8s_deployment,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    print(generate_k8s_deployment(config, namespace=namespace))
@agentic_serving.command("helm-values")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "yaml"]), default="yaml"
)
def agentic_serving_helm_values(fmt):
    """Print Helm values for agentic inference deployment."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_helm_values,
    )

    api = TerradevAPI()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    values = generate_helm_values(config)
    if fmt == "json":
        print(json.dumps(values, indent=2))
    else:
        import yaml

        print(yaml.dump(values, default_flow_style=False))
@cli.group("model-router")
def model_router():
    """Model routing  cost/quality-aware routing between strong and weak models."""
    pass
@model_router.command("configure")
@click.option(
    "--strong-url",
    prompt="Strong model URL (e.g. https://api.openai.com)",
    help="Strong model endpoint",
)
@click.option("--strong-model", prompt="Strong model ID", default="gpt-4")
@click.option("--strong-api-key", prompt="Strong model API key", hide_input=True)
@click.option(
    "--weak-url",
    prompt="Weak model URL (e.g. http://localhost:8000)",
    help="Weak model endpoint",
)
@click.option("--weak-model", prompt="Weak model ID", default="llama-3.1-8b")
@click.option("--weak-api-key", default="", help="Weak model API key (if needed)")
@click.option(
    "--strategy",
    type=click.Choice(
        ["step_type", "threshold", "cascade", "strong_only", "weak_only"]
    ),
    default="step_type",
    help="Routing strategy",
)
@click.option(
    "--cost-threshold",
    default=0.5,
    type=float,
    help="Complexity threshold for threshold strategy",
)
def model_router_configure(
    strong_url,
    strong_model,
    strong_api_key,
    weak_url,
    weak_model,
    weak_api_key,
    strategy,
    cost_threshold,
):
    """Configure model routing endpoints and strategy."""
    api = TerradevAPI()
    api._save_provider_creds(
        "model_router",
        {
            "strong_url": strong_url,
            "strong_model": strong_model,
            "strong_api_key": strong_api_key,
            "weak_url": weak_url,
            "weak_model": weak_model,
            "weak_api_key": weak_api_key,
            "strategy": strategy,
            "cost_threshold": str(cost_threshold),
            "cascade_enabled": str(strategy == "cascade").lower(),
        },
    )
    print("\u2705 Model router configured:")
    print(f"   Strong: {strong_model} @ {strong_url}")
    print(f"   Weak:   {weak_model} @ {weak_url}")
    print(f"   Strategy: {strategy}")
@model_router.command("test")
@click.option("--prompt", "-p", default="What is 2+2?", help="Test prompt")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_test(prompt, fmt):
    """Test model routing with a sample prompt."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = TerradevAPI()
    router = create_router_from_credentials(api._provider_creds("model_router"))
    messages = [{"role": "user", "content": prompt}]
    endpoint, step_type, reason = router.route(messages)

    if fmt == "json":
        print(
            json.dumps(
                {
                    "model": endpoint.model_id,
                    "tier": endpoint.tier.value,
                    "endpoint_url": endpoint.url,
                    "step_type": step_type.value,
                    "reason": reason,
                },
                indent=2,
            )
        )
    else:
        print("\n  Routing Decision:")
        print(f"  Model:     {endpoint.model_id}")
        print(f"  Tier:      {endpoint.tier.value}")
        print(f"  URL:       {endpoint.url}")
        print(f"  Step Type: {step_type.value}")
        print(f"  Reason:    {reason}\n")
@model_router.command("classify")
@click.argument("text")
def model_router_classify(text):
    """Classify a message's step type for routing."""
    from terradev_cli.ml_services.model_router import StepClassifier

    messages = [{"role": "user", "content": text}]
    step_type = StepClassifier.classify(messages)
    print(f"  Step type: {step_type.value}")
@model_router.command("stats")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_stats(fmt):
    """Show routing statistics (in-memory, current session)."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = TerradevAPI()
    router = create_router_from_credentials(api._provider_creds("model_router"))
    stats = router.get_routing_stats()

    if fmt == "json":
        print(json.dumps(stats, indent=2))
    else:
        print("\n  Routing Stats:")
        print(f"  Total Decisions: {stats['total_decisions']}")
        if stats["total_decisions"] > 0:
            print(f"  Strong %:        {stats['strong_pct']}%")
            print(f"  Weak %:          {stats['weak_pct']}%")
            by_step = stats.get("by_step_type", {})
            if by_step:
                print("\n  By Step Type:")
                for st, counts in by_step.items():
                    print(
                        f"    {st}: {counts['total']} (strong={counts['strong']}, weak={counts['weak']})"
                    )
        print()
@model_router.command("llmd-config")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "yaml"]), default="yaml"
)
def model_router_llmd_config(fmt):
    """Generate llm-d KV-cache-aware routing config."""
    from terradev_cli.ml_services.model_router import generate_llmd_routing_config, RouterConfig

    config = generate_llmd_routing_config(RouterConfig())
    if fmt == "json":
        print(json.dumps(config, indent=2))
    else:
        import yaml

        print(yaml.dump(config, default_flow_style=False))
@cli.group()
def migrate():
    """Cross-provider workload migration with dry-run analysis"""
    pass
@migrate.command()
@click.option("--from", "from_provider", required=True, help="Source provider")
@click.option("--to", "to_provider", required=True, help="Target provider")
@click.option("--instance-id", help="Source instance ID")
@click.option("--workload", help="Workload ID from JobStateManager")
@click.option("--dry-run", is_flag=True, help="Show migration plan without executing")
def migration(from_provider, to_provider, instance_id, workload, dry_run):
    """Migrate workload between providers with detailed cost analysis"""
    from terradev_cli.core.migration_orchestrator import MigrationOrchestrator

    print(f"\n Migration Analysis: {from_provider} → {to_provider}")
    if dry_run:
        print("    DRY RUN MODE - No changes will be made")

    try:
        orchestrator = MigrationOrchestrator()
        plan = orchestrator.plan_migration(
            source_provider=from_provider,
            target_provider=to_provider,
            instance_id=instance_id,
            workload_id=workload,
            dry_run=True,  # Always generate plan first
        )

        # Display migration plan
        print("\n Migration Plan:")
        print(f"   Source: {plan.source['provider']} ({plan.source['gpu_type']})")
        print(f"   Target: {plan.target['provider']} ({plan.target['gpu_type']})")
        print(f"   Confidence: {plan.confidence_score:.1%}")

        if plan.warnings:
            print("\nWARNING:  Warnings:")
            for warning in plan.warnings:
                print(f"    {warning}")

        print("\nCOST: Cost Analysis:")
        print(f"   Data transfer: ${plan.costs['data_transfer']:.4f}")
        print(f"   Target hourly: ${plan.costs['target_hourly']:.2f}")
        print(f"   Hourly savings: ${plan.costs['hourly_savings']:+.2f}")
        print(f"   Monthly savings: ${plan.costs['estimated_monthly_savings']:+.2f}")

        print("\n Compatibility:")
        print(f"   GPU match: {plan.compatibility['gpu_match']}")
        print(f"   Performance change: {plan.compatibility['performance_change']}")

        print("\n  Migration Steps:")
        for step in plan.steps:
            print(f"   {step}")

        print(f"\n  Estimated downtime: {plan.total_downtime}")

        if dry_run:
            print(
                "\nOK: Dry run complete. Use without --dry-run to execute migration."
            )
        else:
            # In lightweight version, just show plan and exit
            print(
                "\n Full migration execution not implemented in lightweight version."
            )
            print("   This would involve:")
            print("    Checkpointing current job")
            print("    Transferring data via optimized route")
            print("    Provisioning target instance")
            print("    Restoring from checkpoint")
            print("    Validating migration success")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Migration planning failed: {e}")
        return 1
@migrate.command("list-workloads")
@click.option("--provider", help="Filter by provider")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def list_workloads(provider, fmt):
    """List available workloads for migration"""
    from terradev_cli.core.migration_orchestrator import MigrationOrchestrator

    try:
        orchestrator = MigrationOrchestrator()
        workloads = orchestrator.discover_workloads()

        if provider:
            workloads = [w for w in workloads if w.provider.lower() == provider.lower()]

        if fmt == "json":
            print(
                json.dumps(
                    [
                        {
                            "job_id": w.job_id,
                            "name": w.name,
                            "provider": w.provider,
                            "gpu_type": w.gpu_type,
                            "progress": f"{w.current_step}/{w.total_steps}",
                            "checkpoint_size_gb": w.checkpoint_size_gb,
                        }
                        for w in workloads
                    ],
                    indent=2,
                )
            )
        else:
            print("\n Available Workloads:")
            if not workloads:
                print("   No active workloads found")
                return

            print(
                f"   {'Job ID':<20} {'Name':<15} {'Provider':<12} {'GPU':<8} {'Progress':<12} {'Size':<8}"
            )
            print(f"   {'─'*80}")
            for w in workloads:
                progress = f"{w.current_step}/{w.total_steps}"
                size = f"{w.checkpoint_size_gb:.1f}GB"
                print(
                    f"   {w.job_id:<20} {w.name:<15} {w.provider:<12} {w.gpu_type:<8} {progress:<12} {size:<8}"
                )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list workloads: {e}")
        return 1
@cli.group()
def eval():
    """Model and endpoint evaluation with baseline comparison"""
    pass
@eval.command()
@click.option("--model", "model_path", help="Model checkpoint path")
@click.option("--endpoint", help="API endpoint URL")
@click.option("--dataset", help="Dataset path for evaluation")
@click.option(
    "--metrics",
    multiple=True,
    default=["accuracy", "latency"],
    help="Metrics to evaluate",
)
@click.option("--baseline", help="Baseline result file for comparison")
@click.option("--workload-type", default="general", help="Workload type classification")
@click.option(
    "--duration",
    type=int,
    default=300,
    help="Duration for endpoint evaluation (seconds)",
)
@click.option("--output", help="Output file for results")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table")
def evaluation(
    model_path,
    endpoint,
    dataset,
    metrics,
    baseline,
    workload_type,
    duration,
    output,
    fmt,
):
    """Run model or endpoint evaluation"""
    from terradev_cli.core.evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig

    if not model_path and not endpoint:
        print("ERROR: Either --model or --endpoint must be specified")
        return 1

    if model_path and not dataset:
        print("ERROR: --dataset required when evaluating a model")
        return 1

    try:
        orchestrator = EvaluationOrchestrator()
        config = EvaluationConfig(
            model_path=model_path,
            endpoint_url=endpoint,
            dataset_path=dataset,
            metrics=list(metrics),
            baseline_path=baseline,
            workload_type=workload_type,
            duration_seconds=duration,
        )

        print("\n Running Evaluation...")
        if model_path:
            print(f"   Model: {model_path}")
            print(f"   Dataset: {dataset}")
        if endpoint:
            print(f"   Endpoint: {endpoint}")
            print(f"   Duration: {duration}s")
        print(f"   Metrics: {', '.join(metrics)}")

        # Run evaluation
        if model_path:
            result = orchestrator.evaluate_model(config)
        else:
            result = orchestrator.evaluate_endpoint(config)

        # Display results
        if fmt == "json":
            result_data = {
                "evaluation_id": result.evaluation_id,
                "model_path": result.model_path,
                "endpoint_url": result.endpoint_url,
                "workload_type": result.workload_type,
                "metrics": result.metrics,
                "baseline_comparison": result.baseline_comparison,
                "timestamp": result.timestamp.isoformat(),
                "duration_seconds": result.duration_seconds,
                "metadata": result.metadata,
            }
            print(json.dumps(result_data, indent=2))
        else:
            print("\n Evaluation Results:")
            print(f"   Evaluation ID: {result.evaluation_id}")
            print(f"   Duration: {result.duration_seconds:.1f}s")

            print("\n Metrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    if metric in ["latency", "error_rate"]:
                        print(f"   {metric:<15}: {value:.2f}ms")
                    elif metric in ["throughput"]:
                        print(f"   {metric:<15}: {value:.1f} tokens/s")
                    elif metric in ["cost_per_token"]:
                        print(f"   {metric:<15}: ${value:.6f}")
                    else:
                        print(f"   {metric:<15}: {value:.3f}")
                else:
                    print(f"   {metric:<15}: {value}")

            if (
                result.baseline_comparison
                and "differences" in result.baseline_comparison
            ):
                print("\n Baseline Comparison:")
                for metric, diff in result.baseline_comparison["differences"].items():
                    print(f"   {metric:<15}: {diff['percentage']:+.1f}%")

        # Save results if output specified
        if output:
            orchestrator.save_result(result, output)
            print(f"\n Results saved to {output}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Evaluation failed: {e}")
        return 1
@eval.command("compare")
@click.argument("model_a")
@click.argument("model_b")
@click.option("--dataset", required=True, help="Dataset for comparison")
@click.option(
    "--metrics",
    multiple=True,
    default=["accuracy", "perplexity"],
    help="Metrics to compare",
)
@click.option("--output", help="Output file for comparison results")
def compare_models(model_a, model_b, dataset, metrics, output):
    """Compare two models side-by-side"""
    from terradev_cli.core.evaluation_orchestrator import EvaluationOrchestrator

    try:
        orchestrator = EvaluationOrchestrator()

        print("\n Comparing Models:")
        print(f"   Model A: {model_a}")
        print(f"   Model B: {model_b}")
        print(f"   Dataset: {dataset}")
        print(f"   Metrics: {', '.join(metrics)}")

        comparison = orchestrator.compare_models(
            model_a, model_b, dataset, list(metrics)
        )

        print("\n Comparison Results:")
        print(
            f"   {'Metric':<15} {'Model A':<12} {'Model B':<12} {'Winner':<10} {'Difference':<12}"
        )
        print(f"   {'─'*65}")

        for metric in metrics:
            val_a = comparison["model_a"]["metrics"].get(metric, 0)
            val_b = comparison["model_b"]["metrics"].get(metric, 0)
            winner = comparison["winner"].get(metric, "tie")
            diff = comparison["differences"].get(metric, {}).get("percentage", 0)

            # Format values
            if isinstance(val_a, float):
                val_a_str = f"{val_a:.3f}"
                val_b_str = f"{val_b:.3f}"
            else:
                val_a_str = str(val_a)
                val_b_str = str(val_b)

            diff_str = f"{diff:+.1f}%"

            print(
                f"   {metric:<15} {val_a_str:<12} {val_b_str:<12} {winner:<10} {diff_str:<12}"
            )

        # Save comparison if output specified
        if output:
            output_file = Path(output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(comparison, indent=2))
            print(f"\n Comparison saved to {output}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Model comparison failed: {e}")
        return 1
@cli.command()
@click.option("--output", "-o", required=True, help="Output YAML file path")
@click.option("--job", "-j", help="Specific job to export (omits latest)")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["argo", "native"]),
    default="argo",
    help="Output format (argo-compatible or terradev-native)",
)
def export(output, job, cache_dir, output_format):
    """Export current state or job as Argo-compatible YAML pipeline"""
    try:
        from terradev_cli.core.manifest_cache import ManifestCache
        from terradev_cli.core.pipeline_schema import Workflow, WorkflowMetadata, TerradevAnnotations
        import yaml

        cache = ManifestCache(cache_dir)

        if job:
            # Export specific job
            manifest = cache.load_manifest(job)
            if not manifest:
                print(f"ERROR: Job '{job}' not found in manifest cache")
                print(
                    f"Tip: Run 'terradev manifests --job {job}' to see available versions"
                )
                return 1

            print(f"UPLOAD: Exporting job '{job}' (version {manifest.version})...")

            # Convert manifest to Argo workflow
            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name=f"{job}-exported", namespace="default", annotations={}
            )

            # Add Terradev annotations from manifest
            if manifest.nodes:
                first_node = manifest.nodes[0]
                terradev_ann = TerradevAnnotations()
                terradev_ann.provider = first_node.provider
                terradev_ann.gpu_type = first_node.gpu_type
                terradev_ann.gpu_count = first_node.gpus

                workflow.metadata.annotations.update(terradev_ann.to_dict())

            # Create basic workflow structure
            workflow.spec = {
                "entrypoint": "main",
                "templates": [
                    {
                        "name": "main",
                        "steps": [
                            [{"name": "provision", "template": "provision-step"}],
                            [{"name": "run-workload", "template": "workload-step"}],
                        ],
                    },
                    {
                        "name": "provision-step",
                        "container": {
                            "image": "terradev/cli:latest",
                            "command": ["terradev", "provision"],
                            "args": [
                                "--provider",
                                first_node.provider,
                                "--gpu-type",
                                first_node.gpu_type,
                                "--gpu-count",
                                str(first_node.gpus),
                            ],
                        },
                    },
                    {
                        "name": "workload-step",
                        "container": {
                            "image": "pytorch/pytorch:latest",
                            "command": ["python", "train.py"],
                        },
                    },
                ],
            }

        else:
            # Export current state
            print("UPLOAD: Exporting current Terradev state...")

            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name="current-state-export",
                namespace="default",
                annotations={
                    "terradev.io/export-type": "current-state",
                    "terradev.io/export-timestamp": datetime.now().isoformat(),
                },
            )

            # Create template for current configuration
            workflow.spec = {
                "entrypoint": "main",
                "templates": [
                    {
                        "name": "main",
                        "container": {
                            "image": "terradev/cli:latest",
                            "command": ["terradev", "status"],
                        },
                    }
                ],
            }

        # Write YAML output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if output_format == "argo":
                f.write(workflow.to_yaml())
            else:
                # Native format - just dump manifest data
                if job and manifest:
                    yaml.dump(
                        {
                            "job": manifest.job,
                            "version": manifest.version,
                            "nodes": [
                                {
                                    "provider": node.provider,
                                    "pod_id": node.pod_id,
                                    "instance_id": node.instance_id,
                                    "gpus": node.gpus,
                                    "gpu_type": node.gpu_type,
                                    "region": node.region,
                                    "status": node.status,
                                    "created_at": node.created_at,
                                    "ttl": node.ttl,
                                }
                                for node in manifest.nodes
                            ],
                            "dataset_hash": manifest.dataset_hash,
                            "ttl": manifest.ttl,
                            "created_at": manifest.created_at,
                            "metadata": manifest.metadata,
                        },
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                else:
                    yaml.dump(
                        {
                            "exported": "current-state",
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                    )

        print(f"OK: Exported to {output} (format: {output_format})")

        if job and manifest:
            print(f"   Job: {manifest.job}")
            print(f"   Version: {manifest.version}")
            print(f"   Nodes: {len(manifest.nodes)}")
            print(
                f"   Provider: {manifest.nodes[0].provider if manifest.nodes else 'N/A'}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Export failed: {e}")
        return 1
@cli.command("import")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option(
    "--name", "-n", help="Name to register pipeline (defaults to YAML metadata name)"
)
@click.option(
    "--force", is_flag=True, help="Overwrite existing pipeline with same name"
)
@click.option("--validate-only", is_flag=True, help="Only validate, do not register")
@click.option("--cache-dir", default="./manifests", help="Manifest cache directory")
def import_cmd(yaml_file, name, force, validate_only, cache_dir):
    """Import and register Argo-compatible YAML pipeline"""
    try:
        from terradev_cli.core.pipeline_schema import Workflow, PipelineValidator
        from terradev_cli.core.manifest_cache import ManifestCache, Manifest, ManifestNode

        print(f" Importing pipeline from {yaml_file}...")

        # Validate YAML file
        is_valid, errors = PipelineValidator.validate_yaml_file(yaml_file)
        if not is_valid:
            print("ERROR: YAML validation failed:")
            for error in errors:
                print(f"   - {error}")
            return 1

        # Parse workflow
        workflow = Workflow.from_file(yaml_file)

        if not workflow.metadata or not workflow.metadata.name:
            print("ERROR: Workflow must have metadata.name")
            return 1

        pipeline_name = name or workflow.metadata.name

        # Check if already exists
        cache = ManifestCache(cache_dir)
        existing_versions = cache.list_versions(pipeline_name)

        if existing_versions and not force:
            print(
                f"ERROR: Pipeline '{pipeline_name}' already exists with versions: {', '.join(existing_versions)}"
            )
            print(
                "Tip: Use --force to overwrite or --name to specify a different name"
            )
            return 1

        if validate_only:
            print(f"OK: YAML validation passed for '{pipeline_name}'")
            return 0

        # Extract Terradev annotations
        terradev_ann = workflow.metadata.terradev_annotations

        # Create manifest from workflow
        manifest_nodes = []

        # Create a basic node from workflow annotations
        if terradev_ann.provider and terradev_ann.gpu_type:
            node = ManifestNode(
                provider=terradev_ann.provider.value,
                pod_id=f"imported-{uuid.uuid4().hex[:8]}",
                instance_id=f"imported-{uuid.uuid4().hex[:8]}",
                gpus=terradev_ann.gpu_count or 1,
                gpu_type=terradev_ann.gpu_type.value,
                region="auto-imported",
                status="imported",
                created_at=datetime.now().isoformat(),
                ttl="24h",
            )
            manifest_nodes.append(node)

        # Create manifest
        version = f"v{len(existing_versions) + 1}"
        manifest = Manifest(
            job=pipeline_name,
            version=version,
            nodes=manifest_nodes,
            dataset_hash="imported-yaml",
            ttl="24h",
            created_at=datetime.now().isoformat(),
            metadata={
                "source": "yaml-import",
                "yaml_file": str(yaml_file),
                "workflow_name": workflow.metadata.name,
                "terradev_annotations": workflow.metadata.annotations,
            },
        )

        # Store manifest
        manifest_path = cache.store_manifest(manifest)

        print(f"OK: Imported pipeline '{pipeline_name}' (version {version})")
        print(f"   Stored: {manifest_path}")

        if terradev_ann.provider:
            print(f"   Provider: {terradev_ann.provider.value}")
        if terradev_ann.gpu_type:
            print(f"   GPU Type: {terradev_ann.gpu_type.value}")
        if terradev_ann.gpu_count:
            print(f"   GPU Count: {terradev_ann.gpu_count}")

        print(f"Tip: Run 'terradev job {yaml_file}' to execute this pipeline")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Import failed: {e}")
        return 1
@cli.group()
def record():
    """Record and export live workflows"""
    pass
@record.command("start")
@click.option("--name", "-n", required=True, help="Recording name")
@click.option("--output-dir", default="./recordings", help="Recording output directory")
def record_start(name, output_dir):
    """Start recording a live workflow"""
    try:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        recording_file = output_path / f"{name}.recording"

        # Create recording manifest
        recording_data = {
            "name": name,
            "status": "recording",
            "started_at": datetime.now().isoformat(),
            "commands": [],
            "events": [],
        }

        with open(recording_file, "w") as f:
            json.dump(recording_data, f, indent=2)

        print(f" Started recording '{name}'")
        print(f"   Output: {recording_file}")
        print(
            f"Tip: Run 'terradev record stop --name {name} --export pipeline.yaml' when done"
        )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start recording: {e}")
        return 1
@record.command("stop")
@click.option("--name", "-n", required=True, help="Recording name")
@click.option("--export", help="Export as YAML pipeline file")
@click.option("--output-dir", default="./recordings", help="Recording directory")
def record_stop(name, export, output_dir):
    """Stop recording and optionally export as pipeline"""
    try:
        from pathlib import Path
        from terradev_cli.core.pipeline_schema import Workflow, WorkflowMetadata

        output_path = Path(output_dir)
        recording_file = output_path / f"{name}.recording"

        if not recording_file.exists():
            print(f"ERROR: Recording '{name}' not found")
            return 1

        # Load recording
        with open(recording_file, "r") as f:
            recording_data = json.load(f)

        # Update status
        recording_data["status"] = "stopped"
        recording_data["stopped_at"] = datetime.now().isoformat()

        with open(recording_file, "w") as f:
            json.dump(recording_data, f, indent=2)

        print(f" Stopped recording '{name}'")

        if export:
            # Convert recording to workflow
            workflow = Workflow()
            workflow.metadata = WorkflowMetadata(
                name=f"{name}-recorded",
                namespace="default",
                annotations={
                    "terradev.io/source": "live-recording",
                    "terradev.io.recording-name": name,
                    "terradev.io.recording-started": recording_data["started_at"],
                },
            )

            # Create workflow from recorded commands
            templates = []
            steps = []

            for i, cmd in enumerate(recording_data.get("commands", [])):
                step_name = f"step-{i+1}"
                template_name = f"template-{i+1}"

                steps.append([{"name": step_name, "template": template_name}])

                # Create template for this command
                template = {
                    "name": template_name,
                    "container": {
                        "image": "terradev/cli:latest",
                        "command": cmd.split(" "),
                        "args": [],
                    },
                }
                templates.append(template)

            # Add main template
            templates.insert(0, {"name": "main", "steps": steps})

            workflow.spec = {"entrypoint": "main", "templates": templates}

            # Export workflow
            export_path = Path(export)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, "w") as f:
                f.write(workflow.to_yaml())

            print(f"UPLOAD: Exported recording as pipeline: {export}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to stop recording: {e}")
        return 1
@cli.group()
def triggers():
    """Event-driven automation and triggers"""
    pass
@triggers.command("create")
@click.argument("name")
@click.argument("pipeline")
@click.option(
    "--type",
    "trigger_type",
    type=click.Choice(["event", "schedule", "condition"]),
    default="event",
    help="Trigger type",
)
@click.option(
    "--event",
    help="Event type to trigger on (dataset_landed, model_drift_detected, etc.)",
)
@click.option(
    "--schedule", help='Cron schedule (e.g., "0 0 * * 0" for Sunday midnight)'
)
@click.option("--condition", help='Condition expression (e.g., "drift_score > 0.1")')
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    default="dev",
    help="Target environment",
)
def create_trigger(
    name, pipeline, trigger_type, event, schedule, condition, environment
):
    """Create a new trigger"""
    try:
        from terradev_cli.core.event_system import (
            trigger_manager,
            TriggerType,
            EventType,
            Environment,
        )

        # Convert string types to enums
        trigger_type_enum = (
            TriggerType.EVENT_BASED
            if trigger_type == "event"
            else (
                TriggerType.SCHEDULE
                if trigger_type == "schedule"
                else TriggerType.CONDITION
            )
        )

        event_type_enum = None
        if event:
            event_type_enum = EventType(event.replace("-", "_").upper())

        env_enum = Environment(environment)

        trigger_manager.create_trigger(
            name=name,
            trigger_type=trigger_type_enum,
            target_pipeline=pipeline,
            event_type=event_type_enum,
            schedule=schedule,
            condition=condition,
            target_environment=env_enum,
        )

        print(f"OK: Created trigger '{name}'")
        print(f"   Type: {trigger_type}")
        print(f"   Pipeline: {pipeline}")
        print(f"   Environment: {environment}")

        if event:
            print(f"   Event: {event}")
        if schedule:
            print(f"   Schedule: {schedule}")
        if condition:
            print(f"   Condition: {condition}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to create trigger: {e}")
        return 1
@triggers.command("list")
def list_triggers():
    """List all triggers"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        if not trigger_manager.triggers:
            print("No triggers found")
            print("Tip: Use 'terradev triggers create' to create a trigger")
            return

        print(
            f"{'Name':<20} {'Type':<12} {'Pipeline':<20} {'Environment':<12} {'Enabled':<8} {'Count':<6}"
        )
        print("─" * 92)

        for trigger in trigger_manager.triggers.values():
            enabled = "OK:" if trigger.enabled else "ERROR:"
            print(
                f"{trigger.name:<20} {trigger.type.value:<12} {trigger.target_pipeline:<20} "
                f"{trigger.target_environment.value:<12} {enabled:<8} {trigger.trigger_count:<6}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list triggers: {e}")
        return 1
@triggers.command("enable")
@click.argument("name")
def enable_trigger(name):
    """Enable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = True
                print(f"OK: Enabled trigger '{name}'")
                return

        print(f"ERROR: Trigger '{name}' not found")
        return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to enable trigger: {e}")
        return 1
@triggers.command("disable")
@click.argument("name")
def disable_trigger(name):
    """Disable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = False
                print(f" Disabled trigger '{name}'")
                return

        print(f"ERROR: Trigger '{name}' not found")
        return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to disable trigger: {e}")
        return 1
@triggers.command("fire")
@click.argument("event_type")
@click.option("--data", help="JSON data for the event")
@click.option("--source", default="manual", help="Event source")
def fire_event(event_type, data, source):
    """Manually fire an event for testing"""
    try:
        from terradev_cli.core.event_system import event_bus, EventType, Event

        event_type_enum = EventType(event_type.replace("-", "_").upper())

        event_data = {}
        if data:
            event_data = json.loads(data)

        event = Event(type=event_type_enum, source=source, data=event_data)

        event_bus.publish(event)
        print(f" Fired event: {event_type}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to fire event: {e}")
        return 1
@cli.group()
def environments():
    """Environment management and promotion"""
    pass
@environments.command("list")
@click.option("--env", "environment", help="Filter by environment")
def list_environments(environment):
    """List artifacts by environment"""
    try:
        from terradev_cli.core.event_system import lineage_service, Environment

        if environment:
            env_enum = Environment(environment)
            artifacts = [
                a
                for a in lineage_service.artifacts.values()
                if a.environment == env_enum
            ]
        else:
            artifacts = list(lineage_service.artifacts.values())

        if not artifacts:
            print("No artifacts found")
            return

        print(
            f"{'Name':<20} {'Type':<12} {'Environment':<12} {'Version':<8} {'Created':<20}"
        )
        print("─" * 76)

        for artifact in sorted(artifacts, key=lambda x: x.created_at, reverse=True):
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.environment.value:<12} "
                f"{artifact.version:<8} {created:<20}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list environments: {e}")
        return 1
@environments.command("promote")
@click.argument("artifact_name")
@click.option(
    "--from", "from_env", required=True, type=click.Choice(["dev", "staging", "prod"])
)
@click.option(
    "--to", "to_env", required=True, type=click.Choice(["dev", "staging", "prod"])
)
@click.option("--user", default="cli-user", help="User requesting promotion")
def promote_artifact(artifact_name, from_env, to_env, user):
    """Request environment promotion"""
    try:
        from terradev_cli.core.event_system import environment_manager, lineage_service, Environment

        from_enum = Environment(from_env)
        to_enum = Environment(to_env)

        # Find artifact in source environment
        artifact = None
        for a in lineage_service.artifacts.values():
            if a.name == artifact_name and a.environment == from_enum:
                artifact = a
                break

        if not artifact:
            print(f"ERROR: Artifact '{artifact_name}' not found in {from_env}")
            return 1

        promotion = environment_manager.request_promotion(
            artifact_id=artifact.id,
            from_env=from_enum,
            to_env=to_enum,
            requested_by=user,
        )

        print(f" Promotion requested: {from_env} -> {to_env}")
        print(f"   Artifact: {artifact_name}")
        print(f"   Promotion ID: {promotion.id}")
        print(f"   Status: {promotion.status}")
        print(f"Tip: Use 'terradev environments approve {promotion.id}' to complete")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to request promotion: {e}")
        return 1
@environments.command("approve")
@click.argument("promotion_id")
@click.option("--user", default="cli-admin", help="User approving promotion")
def approve_promotion(promotion_id, user):
    """Approve and execute promotion"""
    try:
        from terradev_cli.core.event_system import environment_manager

        success = environment_manager.approve_promotion(promotion_id, user)

        if success:
            print(f"OK: Promotion approved and executed: {promotion_id}")
        else:
            print(f"ERROR: Promotion not found or failed: {promotion_id}")
            return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to approve promotion: {e}")
        return 1
@environments.command("history")
@click.option("--artifact", help="Filter by artifact name")
def promotion_history(artifact):
    """Show promotion history"""
    try:
        from terradev_cli.core.event_system import environment_manager, lineage_service

        artifact_id = None
        if artifact:
            # Find artifact by name
            for a in lineage_service.artifacts.values():
                if a.name == artifact:
                    artifact_id = a.id
                    break

        promotions = environment_manager.get_promotion_history(artifact_id)

        if not promotions:
            print("No promotion history found")
            return

        print(
            f"{'ID':<8} {'Artifact':<20} {'From':<10} {'To':<10} {'Status':<12} {'Requested':<20}"
        )
        print("─" * 84)

        for promo in promotions:
            artifact_name = "unknown"
            if promo.artifact_id in lineage_service.artifacts:
                artifact_name = lineage_service.artifacts[promo.artifact_id].name

            requested = promo.requested_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{promo.id[:8]:<8} {artifact_name:<20} {promo.from_env.value:<10} "
                f"{promo.to_env.value:<10} {promo.status:<12} {requested:<20}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get promotion history: {e}")
        return 1
@cli.group()
def lineage():
    """Artifact lineage and tracking"""
    pass
@lineage.command("register")
@click.argument(
    "type", type=click.Choice(["dataset", "model", "checkpoint", "metrics", "config"])
)
@click.argument("name")
@click.argument("uri")
@click.option(
    "--env", "environment", type=click.Choice(["dev", "staging", "prod"]), default="dev"
)
@click.option("--hash", "artifact_hash", help="Artifact hash")
@click.option("--size", type=int, help="Size in bytes")
@click.option("--user", default="cli-user", help="User registering artifact")
@click.option("--parent", help="Parent artifact ID")
def register_artifact(type, name, uri, environment, artifact_hash, size, user, parent):
    """Register a new artifact for lineage tracking"""
    try:
        from terradev_cli.core.event_system import (
            lineage_service,
            ArtifactType,
            Environment,
            event_bus,
            EventType,
            Event,
        )

        artifact_type_enum = ArtifactType(type)
        env_enum = Environment(environment)

        artifact = lineage_service.register_artifact(
            artifact_type=artifact_type_enum,
            name=name,
            uri=uri,
            environment=env_enum,
            hash=artifact_hash or "",
            size_bytes=size or 0,
            created_by=user,
        )

        # Add parent relationship if specified
        if parent:
            lineage_service.add_relationship(parent, artifact.id)

        # Publish artifact registered event
        event = Event(
            type=EventType.ARTIFACT_REGISTERED,
            source="lineage_service",
            data={
                "artifact_id": artifact.id,
                "artifact_type": type,
                "name": name,
                "environment": environment,
            },
        )
        event_bus.publish(event)

        print(f"OK: Registered artifact: {type} {name}")
        print(f"   ID: {artifact.id}")
        print(f"   Environment: {environment}")
        print(f"   URI: {uri}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to register artifact: {e}")
        return 1
@lineage.command("graph")
@click.argument("artifact_id")
@click.option("--direction", type=click.Choice(["up", "down", "both"]), default="both")
def lineage_graph(artifact_id, direction):
    """Show lineage graph for artifact"""
    try:
        from terradev_cli.core.event_system import lineage_service

        graph = lineage_service.get_lineage(artifact_id, direction)

        if not graph["parents"] and not graph["children"]:
            print(f"No lineage found for artifact {artifact_id}")
            return

        print(f" Lineage for {artifact_id}:")

        if graph["parents"]:
            print(f"\n Parents ({len(graph['parents'])}):")
            for parent in graph["parents"]:
                print(
                    f"   {parent.type.value} {parent.name} ({parent.environment.value})"
                )

        if graph["children"]:
            print(f"\nUPLOAD: Children ({len(graph['children'])}):")
            for child in graph["children"]:
                print(f"   {child.type.value} {child.name} ({child.environment.value})")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get lineage: {e}")
        return 1
@lineage.command("production")
@click.option("--type", "artifact_type", help="Filter by artifact type")
def production_artifacts(artifact_type):
    """Show artifacts in production environment"""
    try:
        from terradev_cli.core.event_system import lineage_service, ArtifactType

        type_enum = None
        if artifact_type:
            type_enum = ArtifactType(artifact_type)

        artifacts = lineage_service.get_production_artifacts(type_enum)

        if not artifacts:
            print("No production artifacts found")
            return

        print(" Production Artifacts:")
        print(
            f"{'Name':<20} {'Type':<12} {'Version':<8} {'Created':<20} {'Created By':<15}"
        )
        print("─" * 79)

        for artifact in artifacts:
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            print(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.version:<8} "
                f"{created:<20} {artifact.created_by:<15}"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get production artifacts: {e}")
        return 1
@lineage.command("show")
@click.argument("model_identifier")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    help="Filter by environment",
)
def show_model_lineage(model_identifier, environment):
    """Show complete provenance of a model (auto-generated)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment) if environment else None

        # Parse model identifier (could be "prod/llama-70b" or just "llama-70b")
        if "/" in model_identifier:
            env_name, model_name = model_identifier.split("/", 1)
            env_from_identifier = Environment(env_name)
            records = auto_lineage.get_lineage_for_model(
                model_name, env_from_identifier
            )
        else:
            records = auto_lineage.get_lineage_for_model(model_identifier, env_enum)

        if not records:
            print(f"ERROR: No lineage found for model '{model_identifier}'")
            return

        print(f" Lineage for model: {model_identifier}")
        print(
            f"{'Execution ID':<8} {'Pipeline':<20} {'Environment':<12} {'Status':<10} {'Timestamp':<20}"
        )
        print("─" * 75)

        for record in records:
            exec_id = record.id[:8]
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M")
            print(
                f"{exec_id:<8} {record.pipeline_id:<20} {record.environment.value:<12} "
                f"{record.status:<10} {timestamp:<20}"
            )

        # Show detailed view of latest execution
        if records:
            latest = records[0]
            print("\n Latest Execution Details:")
            print(f"   Execution ID: {latest.id}")
            print(f"   Pipeline: {latest.pipeline_id}")
            print(f"   Environment: {latest.environment.value}")
            print(f"   Status: {latest.status}")
            print(f"   Duration: {latest.duration_seconds:.1f}s")
            print(f"   GPU Hours: {latest.gpu_hours:.2f}")
            print(f"   Cost: ${latest.compute_cost:.2f}")

            if latest.hyperparameters:
                print("\n  Hyperparameters:")
                for key, value in latest.hyperparameters.items():
                    print(f"      {key}: {value}")

            if latest.datasets:
                print(f"\n Input Datasets ({len(latest.datasets)}):")
                for dataset_id in latest.datasets[:5]:  # Show first 5
                    print(f"      {dataset_id[:12]}...")
                if len(latest.datasets) > 5:
                    print(f"      ... and {len(latest.datasets) - 5} more")

            if latest.output_models:
                print(f"\n Output Models ({len(latest.output_models)}):")
                for model_id in latest.output_models:
                    print(f"      {model_id[:12]}...")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to show lineage: {e}")
        return 1
@lineage.command("diff")
@click.argument("version1")
@click.argument("version2")
def diff_lineage(version1, version2):
    """Compare two pipeline executions"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        # Try to parse as execution IDs first, then as model versions
        exec1_id = version1
        exec2_id = version2

        diff = auto_lineage.diff_executions(exec1_id, exec2_id)

        if "error" in diff:
            print(f"ERROR: {diff['error']}")
            return 1

        print(" Comparing executions:")
        print(
            f"   Execution 1: {diff['execution_1']['id']} ({diff['execution_1']['timestamp']})"
        )
        print(
            f"   Execution 2: {diff['execution_2']['id']} ({diff['execution_2']['timestamp']})"
        )

        if not diff["differences"]:
            print("\nOK: No differences found")
            return

        print("\n Differences:")

        if "hyperparameters" in diff["differences"]:
            print("\n  Hyperparameters:")
            for key, values in diff["differences"]["hyperparameters"].items():
                print(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "environment_variables" in diff["differences"]:
            print("\n Environment Variables:")
            for key, values in diff["differences"]["environment_variables"].items():
                print(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "inputs" in diff["differences"]:
            print("\n Input Artifacts:")
            for change_type, artifacts in diff["differences"]["inputs"].items():
                print(f"   {change_type}: {', '.join(artifacts)}")

        if "resources" in diff["differences"]:
            print("\nCOST: Resource Usage:")
            for resource, values in diff["differences"]["resources"].items():
                print(f"   {resource}: {values['exec1']} → {values['exec2']}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to diff lineage: {e}")
        return 1
@lineage.command("export")
@click.option(
    "--format", type=click.Choice(["json", "csv"]), default="json", help="Export format"
)
@click.option("--model", help="Filter by model name")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    help="Filter by environment",
)
@click.option("--output", "-o", help="Output file (default: stdout)")
def export_lineage(format, model, environment, output):
    """Export lineage data for compliance reports"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment) if environment else None

        data = auto_lineage.export_lineage(format, model, env_enum)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(data)
            print(f"OK: Exported lineage to {output}")
        else:
            print(data)

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to export lineage: {e}")
        return 1
@lineage.command("trace")
@click.option("--checkpoint", help="Checkpoint ID to trace backwards from")
@click.option("--execution", help="Execution ID to trace")
def trace_artifacts(checkpoint, execution):
    """Trace complete lineage from checkpoint or execution"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        if checkpoint:
            trace = auto_lineage.trace_from_checkpoint(checkpoint)

            if "error" in trace:
                print(f"ERROR: {trace['error']}")
                return 1

            print(f" Tracing lineage from checkpoint: {checkpoint}")
            print("\n Created By:")
            created_by = trace["created_by"]
            print(f"   Execution: {created_by['execution_id']}")
            print(f"   Pipeline: {created_by['pipeline_id']}")
            print(f"   Environment: {created_by['environment']}")
            print(f"   Timestamp: {created_by['timestamp']}")

            if trace["inputs"]:
                print("\n Input Artifacts:")
                if "datasets" in trace["inputs"]:
                    print(f"   Datasets ({len(trace['inputs']['datasets'])}):")
                    for dataset in trace["inputs"]["datasets"]:
                        print(f"      {dataset['name']} ({dataset['id'][:12]}...)")

                if "models" in trace["inputs"]:
                    print(f"   Models ({len(trace['inputs']['models'])}):")
                    for model in trace["inputs"]["models"]:
                        print(f"      {model['name']} ({model['id'][:12]}...)")

            if trace["ancestors"]:
                print("\n Ancestor Executions:")
                for ancestor in trace["ancestors"]:
                    print(
                        f"   {ancestor['execution_id'][:12]}... - {ancestor['pipeline_id']} "
                        f"({ancestor['environment']}) - {ancestor['timestamp']}"
                    )

        elif execution:
            # Show execution details and trace backwards
            from terradev_cli.core.event_system import lineage_service

            # Find execution record
            exec_record = None
            for record in auto_lineage.completed_executions:
                if record.id.startswith(execution):
                    exec_record = record
                    break

            if not exec_record:
                print(f"ERROR: Execution '{execution}' not found")
                return 1

            print(f" Tracing execution: {exec_record.id}")
            print(f"   Pipeline: {exec_record.pipeline_id}")
            print(f"   Environment: {exec_record.environment.value}")
            print(f"   Status: {exec_record.status}")
            print(f"   Timestamp: {exec_record.timestamp}")

            # Show artifact lineage
            all_artifacts = (
                exec_record.datasets
                + exec_record.models
                + exec_record.output_models
                + exec_record.output_checkpoints
            )

            if all_artifacts:
                print("\n Artifact Lineage:")
                for artifact_id in all_artifacts:
                    if artifact_id in lineage_service.artifacts:
                        artifact = lineage_service.artifacts[artifact_id]
                        graph = lineage_service.get_lineage(artifact_id, "up")

                        print(f"\n   {artifact.type.value}: {artifact.name}")
                        print(f"      Environment: {artifact.environment.value}")
                        print(f"      Created: {artifact.created_at}")

                        if graph["parents"]:
                            print(f"      Parents: {len(graph['parents'])}")
                            for parent in graph["parents"][:3]:  # Show first 3
                                print(f"         └─ {parent.type.value}: {parent.name}")

        else:
            print("ERROR: Must specify either --checkpoint or --execution")
            return 1

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to trace artifacts: {e}")
        return 1
@lineage.command("auto")
@click.option("--pipeline", required=True, help="Pipeline ID")
@click.option(
    "--env",
    "environment",
    type=click.Choice(["dev", "staging", "prod"]),
    default="dev",
    help="Execution environment",
)
@click.option("--triggered-by", default="manual", help="Who triggered this execution")
def start_auto_lineage(pipeline, environment, triggered_by):
    """Start automatic lineage tracking for a pipeline execution"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import Environment

        env_enum = Environment(environment)

        execution = auto_lineage.start_execution(
            pipeline_id=pipeline, environment=env_enum, triggered_by=triggered_by
        )

        print(" Started automatic lineage tracking")
        print(f"   Execution ID: {execution.id}")
        print(f"   Pipeline: {pipeline}")
        print(f"   Environment: {environment}")
        print("   Use this ID to add artifacts and complete the execution")

        # Show example commands for manual tracking
        print("\nTip: Example commands to track this execution:")
        print(f"   terradev lineage add-input {execution.id} dataset <dataset-id>")
        print(f"   terradev lineage add-output {execution.id} model <model-id>")
        print(f"   terradev lineage complete {execution.id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start lineage tracking: {e}")
        return 1
@lineage.command("add-input")
@click.argument("execution_id")
@click.argument(
    "artifact_type", type=click.Choice(["dataset", "model", "config", "checkpoint"])
)
@click.argument("artifact_id")
def add_input_artifact(execution_id, artifact_type, artifact_id):
    """Add input artifact to execution (manual override)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import ArtifactType

        artifact_enum = ArtifactType(artifact_type)
        auto_lineage.add_input_artifact(execution_id, artifact_enum, artifact_id)

        print(f"OK: Added input {artifact_type}: {artifact_id}")
        print(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to add input artifact: {e}")
        return 1
@lineage.command("add-output")
@click.argument("execution_id")
@click.argument(
    "artifact_type", type=click.Choice(["model", "checkpoint", "metrics", "evaluation"])
)
@click.argument("artifact_id")
def add_output_artifact(execution_id, artifact_type, artifact_id):
    """Add output artifact to execution (manual override)"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage
        from terradev_cli.core.event_system import ArtifactType

        artifact_enum = ArtifactType(artifact_type)
        auto_lineage.add_output_artifact(execution_id, artifact_enum, artifact_id)

        print(f"OK: Added output {artifact_type}: {artifact_id}")
        print(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to add output artifact: {e}")
        return 1
@lineage.command("complete")
@click.argument("execution_id")
@click.option("--status", default="completed", help="Final status (completed, failed)")
def complete_execution(execution_id, status):
    """Complete execution and finalize lineage record"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        auto_lineage.complete_execution(execution_id, status)

        print(f"OK: Completed execution: {execution_id}")
        print(f"   Status: {status}")
        print("   Lineage record finalized and available for queries")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to complete execution: {e}")
        return 1
