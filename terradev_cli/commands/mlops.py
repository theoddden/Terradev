#!/usr/bin/env python3
"""Commands for the Terradev CLI."""

import json
import logging
from pathlib import Path

import click
from . import cli
from terradev_cli.commands._base import get_api as _get_api, run_with_timeout as _run_with_timeout

logger = logging.getLogger(__name__)

agentic_serving = click.Group(
    name="agentic-serving",
    help="Agentic inference serving  KV cache TTL, prefix caching, LMCache, priority scheduling.",
)
@agentic_serving.command("configure")
@click.option(
    "--engine",
    type=click.Choice(["vllm", "sglang"]),
    default="vllm",
    help="Inference engine",
)
@click.option("--model", prompt="Model ID", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option(
    "--tp", "tensor_parallel_size", default=1, type=click.IntRange(1, 1000000), help="Tensor parallel size"
)
@click.option("--max-model-len", default=32768, type=click.IntRange(1, 1000000), help="Maximum sequence length for the model")
@click.option("--gpu-mem", "gpu_memory_utilization", default=0.85, type=click.FloatRange(0.0, 10000.0), help="Fraction of GPU memory to use (0.0–1.0)")
@click.option(
    "--lmcache/--no-lmcache",
    "lmcache_enabled",
    default=True,
    help="Enable LMCache KV offload",
)
@click.option(
    "--lmcache-backend", type=click.Choice(["cpu", "disk", "redis"]), default="cpu",
    help="LMCache storage backend",
)
@click.option(
    "--disaggregation/--no-disaggregation",
    default=False,
    help="Enable prefill-decode disaggregation",
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
    api = _get_api()
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
    click.echo(f"\u2705 Agentic serving configured: {engine} + {model}")
    click.echo("   Prefix caching: enabled")
    click.echo(
        f"   LMCache: {'enabled (' + lmcache_backend + ')' if lmcache_enabled else 'disabled'}"
    )
    click.echo(f"   PD disaggregation: {'enabled' if disaggregation else 'disabled'}")
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

    api = _get_api()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    engine_args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )

    if fmt == "json":
        click.echo(
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
        click.echo("\n  Agentic Serving Config:")
        click.echo(f"  Engine:            {config.engine}")
        click.echo(f"  Model:             {config.model}")
        click.echo(f"  TP:                {config.tensor_parallel_size}")
        click.echo(f"  Max Model Len:     {config.max_model_len}")
        click.echo(f"  GPU Mem Util:      {config.gpu_memory_utilization}")
        click.echo(f"  Prefix Caching:    {config.enable_prefix_caching}")
        click.echo(
            f"  LMCache:           {config.lmcache_enabled} ({config.lmcache_backend})"
        )
        click.echo(f"  Disaggregation:    {config.disaggregation_enabled}")
        click.echo(
            f"  KV TTL Range:      {config.ttl_min}s - {config.ttl_max}s (x{config.ttl_multiplier})"
        )
        click.echo("\n  Engine Args:")
        for a in engine_args:
            click.echo(f"    {a}")
        click.echo("")
@agentic_serving.command("launch-args")
def agentic_serving_launch_args():
    """Print engine launch arguments for copy-paste."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_vllm_args,
        generate_sglang_args,
    )

    api = _get_api()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    args = (
        generate_vllm_args(config)
        if config.engine == "vllm"
        else generate_sglang_args(config)
    )
    if config.engine == "vllm":
        click.echo("\npython -m vllm.entrypoints.openai.api_server \\")
    else:
        click.echo("\npython -m sglang.launch_server \\")
    for i, a in enumerate(args):
        sep = " \\" if i < len(args) - 1 else ""
        click.echo(f"  {a}{sep}")
    click.echo("")
@agentic_serving.command("lmcache-env")
def agentic_serving_lmcache_env():
    """Print LMCache environment variables."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_lmcache_env,
    )

    api = _get_api()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    env = generate_lmcache_env(config)
    if not env:
        click.echo("  LMCache is disabled.")
        return
    click.echo("")
    for k, v in env.items():
        click.echo(f'export {k}="{v}"')
    click.echo("")
@agentic_serving.command("k8s")
@click.option("--namespace", "-n", default="inference", help="K8s namespace")
def agentic_serving_k8s(namespace):
    """Print K8s deployment manifests for agentic inference."""
    from terradev_cli.ml_services.agentic_serving import (
        create_agentic_serving_from_credentials,
        generate_k8s_deployment,
    )

    api = _get_api()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    click.echo(generate_k8s_deployment(config, namespace=namespace))
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

    api = _get_api()
    config, _ = create_agentic_serving_from_credentials(
        api._provider_creds("agentic_serving")
    )
    values = generate_helm_values(config)
    if fmt == "json":
        click.echo(json.dumps(values, indent=2))
    else:
        import yaml

        click.echo(yaml.dump(values, default_flow_style=False))
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
    type=click.FloatRange(0.0, 10000.0),
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
    api = _get_api()
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
    click.echo("\u2705 Model router configured:")
    click.echo(f"   Strong: {strong_model} @ {strong_url}")
    click.echo(f"   Weak:   {weak_model} @ {weak_url}")
    click.echo(f"   Strategy: {strategy}")
@model_router.command("test")
@click.option("--prompt", "-p", default="What is 2+2?", help="Test prompt")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_test(prompt, fmt):
    """Test model routing with a sample prompt."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = _get_api()
    creds = api._provider_creds("model_router")
    if not creds:
        click.echo("ERROR: model-router not configured. Run `terradev model-router configure` first.", err=True)
        raise SystemExit(1)

    try:
        router = create_router_from_credentials(creds)
        messages = [{"role": "user", "content": prompt}]
        endpoint, step_type, reason = router.route(messages)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Model routing failed: {e}", err=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(
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
        click.echo("\n  Routing Decision:")
        click.echo(f"  Model:     {endpoint.model_id}")
        click.echo(f"  Tier:      {endpoint.tier.value}")
        click.echo(f"  URL:       {endpoint.url}")
        click.echo(f"  Step Type: {step_type.value}")
        click.echo(f"  Reason:    {reason}\n")
@model_router.command("classify")
@click.argument("text")
def model_router_classify(text):
    """Classify a message's step type for routing."""
    from terradev_cli.ml_services.model_router import StepClassifier

    messages = [{"role": "user", "content": text}]
    try:
        step_type = StepClassifier.classify(messages)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Could not classify message: {e}", err=True)
        raise SystemExit(1)
    click.echo(f"  Step type: {step_type.value}")
@model_router.command("stats")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def model_router_stats(fmt):
    """Show routing statistics (in-memory, current session)."""
    from terradev_cli.ml_services.model_router import create_router_from_credentials

    api = _get_api()
    creds = api._provider_creds("model_router")
    if not creds:
        click.echo("ERROR: model-router not configured. Run `terradev model-router configure` first.", err=True)
        raise SystemExit(1)

    try:
        router = create_router_from_credentials(creds)
        stats = router.get_routing_stats()
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Could not load routing stats: {e}", err=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo("\n  Routing Stats:")
        click.echo(f"  Total Decisions: {stats['total_decisions']}")
        if stats["total_decisions"] > 0:
            click.echo(f"  Strong %:        {stats['strong_pct']}%")
            click.echo(f"  Weak %:          {stats['weak_pct']}%")
            by_step = stats.get("by_step_type", {})
            if by_step:
                click.echo("\n  By Step Type:")
                for st, counts in by_step.items():
                    click.echo(
                        f"    {st}: {counts['total']} (strong={counts['strong']}, weak={counts['weak']})"
                    )
        click.echo()
@model_router.command("llmd-config")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "yaml"]), default="yaml"
)
def model_router_llmd_config(fmt):
    """Generate llm-d KV-cache-aware routing config."""
    from terradev_cli.ml_services.model_router import generate_llmd_routing_config, RouterConfig

    try:
        config = generate_llmd_routing_config(RouterConfig())
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Could not generate LLM-d routing config: {e}", err=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(config, indent=2))
    else:
        import yaml

        click.echo(yaml.dump(config, default_flow_style=False))
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

    click.echo(f"\n Migration Analysis: {from_provider} → {to_provider}")
    if dry_run:
        click.echo("    DRY RUN MODE - No changes will be made")

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
        click.echo("\n Migration Plan:")
        click.echo(f"   Source: {plan.source['provider']} ({plan.source['gpu_type']})")
        click.echo(f"   Target: {plan.target['provider']} ({plan.target['gpu_type']})")
        click.echo(f"   Confidence: {plan.confidence_score:.1%}")

        if plan.warnings:
            click.echo("\nWARNING:  Warnings:")
            for warning in plan.warnings:
                click.echo(f"    {warning}")

        click.echo("\nCOST: Cost Analysis:")
        click.echo(f"   Data transfer: ${plan.costs['data_transfer']:.4f}")
        click.echo(f"   Target hourly: ${plan.costs['target_hourly']:.2f}")
        click.echo(f"   Hourly savings: ${plan.costs['hourly_savings']:+.2f}")
        click.echo(f"   Monthly savings: ${plan.costs['estimated_monthly_savings']:+.2f}")

        click.echo("\n Compatibility:")
        click.echo(f"   GPU match: {plan.compatibility['gpu_match']}")
        click.echo(f"   Performance change: {plan.compatibility['performance_change']}")

        click.echo("\n  Migration Steps:")
        for step in plan.steps:
            click.echo(f"   {step}")

        click.echo(f"\n  Estimated downtime: {plan.total_downtime}")

        if dry_run:
            click.echo(
                "\nOK: Dry run complete. Use without --dry-run to execute migration."
            )
        else:
            click.echo("\n Executing migration...")
            try:
                api = _get_api()
                result = orchestrator.execute_migration(
                    plan, api, _run_with_timeout, dry_run=False
                )
                click.echo("OK: Migration initiated")
                click.echo(f"   Source: {result['source_id']}")
                click.echo(f"   Target: {result['target_id']}")
                click.echo(f"   Target provider: {result['target_provider']}")
                click.echo(f"   Target GPU: {result['target_gpu']}")
                click.echo(f"   Cost: ${result['target_hourly']:.2f}/hr")
                click.echo("\n Next steps:")
                click.echo(f"   - Transfer data from {result['source_id']} to {result['target_id']}")
                click.echo("   - Verify workload on target: terradev status --live")
                click.echo(
                    f"   - Terminate source when ready: terradev manage -i {result['source_id']} -a terminate"
                )
            except Exception as exec_err:  # noqa: BLE001
                click.echo(f"ERROR: Migration execution failed: {exec_err}", err=True)
                raise SystemExit(1)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Migration planning failed: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(
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
            click.echo("\n Available Workloads:")
            if not workloads:
                click.echo("   No active workloads found")
                return

            click.echo(
                f"   {'Job ID':<20} {'Name':<15} {'Provider':<12} {'GPU':<8} {'Progress':<12} {'Size':<8}"
            )
            click.echo(f"   {'─'*80}")
            for w in workloads:
                progress = f"{w.current_step}/{w.total_steps}"
                size = f"{w.checkpoint_size_gb:.1f}GB"
                click.echo(
                    f"   {w.job_id:<20} {w.name:<15} {w.provider:<12} {w.gpu_type:<8} {progress:<12} {size:<8}"
                )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to list workloads: {e}", err=True)
        raise SystemExit(1)
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
    type=click.IntRange(1, 1000000),
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
        click.echo("ERROR: Either --model or --endpoint must be specified", err=True)
        raise SystemExit(1)

    if model_path and not dataset:
        click.echo("ERROR: --dataset required when evaluating a model", err=True)
        raise SystemExit(1)

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

        click.echo("\n Running Evaluation...")
        if model_path:
            click.echo(f"   Model: {model_path}")
            click.echo(f"   Dataset: {dataset}")
        if endpoint:
            click.echo(f"   Endpoint: {endpoint}")
            click.echo(f"   Duration: {duration}s")
        click.echo(f"   Metrics: {', '.join(metrics)}")

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
            click.echo(json.dumps(result_data, indent=2))
        else:
            click.echo("\n Evaluation Results:")
            click.echo(f"   Evaluation ID: {result.evaluation_id}")
            click.echo(f"   Duration: {result.duration_seconds:.1f}s")

            click.echo("\n Metrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    if metric in ["latency", "error_rate"]:
                        click.echo(f"   {metric:<15}: {value:.2f}ms")
                    elif metric in ["throughput"]:
                        click.echo(f"   {metric:<15}: {value:.1f} tokens/s")
                    elif metric in ["cost_per_token"]:
                        click.echo(f"   {metric:<15}: ${value:.6f}")
                    else:
                        click.echo(f"   {metric:<15}: {value:.3f}")
                else:
                    click.echo(f"   {metric:<15}: {value}")

            if (
                result.baseline_comparison
                and "differences" in result.baseline_comparison
            ):
                click.echo("\n Baseline Comparison:")
                for metric, diff in result.baseline_comparison["differences"].items():
                    click.echo(f"   {metric:<15}: {diff['percentage']:+.1f}%")

        # Save results if output specified
        if output:
            orchestrator.save_result(result, output)
            click.echo(f"\n Results saved to {output}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Evaluation failed: {e}", err=True)
        raise SystemExit(1)
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

        click.echo("\n Comparing Models:")
        click.echo(f"   Model A: {model_a}")
        click.echo(f"   Model B: {model_b}")
        click.echo(f"   Dataset: {dataset}")
        click.echo(f"   Metrics: {', '.join(metrics)}")

        comparison = orchestrator.compare_models(
            model_a, model_b, dataset, list(metrics)
        )

        click.echo("\n Comparison Results:")
        click.echo(
            f"   {'Metric':<15} {'Model A':<12} {'Model B':<12} {'Winner':<10} {'Difference':<12}"
        )
        click.echo(f"   {'─'*65}")

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

            click.echo(
                f"   {metric:<15} {val_a_str:<12} {val_b_str:<12} {winner:<10} {diff_str:<12}"
            )

        # Save comparison if output specified
        if output:
            output_file = Path(output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(comparison, indent=2))
            click.echo(f"\n Comparison saved to {output}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Model comparison failed: {e}", err=True)
        raise SystemExit(1)
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
                click.echo(f"ERROR: Job '{job}' not found in manifest cache", err=True)
                click.echo(
                    f"Tip: Run 'terradev manifests --job {job}' to see available versions"
                )
                raise SystemExit(1)

            click.echo(f"UPLOAD: Exporting job '{job}' (version {manifest.version})...")

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
            click.echo("UPLOAD: Exporting current Terradev state...")

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

        click.echo(f"OK: Exported to {output} (format: {output_format})")

        if job and manifest:
            click.echo(f"   Job: {manifest.job}")
            click.echo(f"   Version: {manifest.version}")
            click.echo(f"   Nodes: {len(manifest.nodes)}")
            click.echo(
                f"   Provider: {manifest.nodes[0].provider if manifest.nodes else 'N/A'}"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Export failed: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(f" Importing pipeline from {yaml_file}...")

        # Validate YAML file
        is_valid, errors = PipelineValidator.validate_yaml_file(yaml_file)
        if not is_valid:
            click.echo("ERROR: YAML validation failed:", err=True)
            for error in errors:
                click.echo(f"   - {error}")
            raise SystemExit(1)

        # Parse workflow
        workflow = Workflow.from_file(yaml_file)

        if not workflow.metadata or not workflow.metadata.name:
            click.echo("ERROR: Workflow must have metadata.name", err=True)
            raise SystemExit(1)

        pipeline_name = name or workflow.metadata.name

        # Check if already exists
        cache = ManifestCache(cache_dir)
        existing_versions = cache.list_versions(pipeline_name)

        if existing_versions and not force:
            click.echo(
                f"ERROR: Pipeline '{pipeline_name}' already exists with versions: {', '.join(existing_versions)}"
            )
            click.echo(
                "Tip: Use --force to overwrite or --name to specify a different name"
            )
            raise SystemExit(1)

        if validate_only:
            click.echo(f"OK: YAML validation passed for '{pipeline_name}'")
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

        click.echo(f"OK: Imported pipeline '{pipeline_name}' (version {version})")
        click.echo(f"   Stored: {manifest_path}")

        if terradev_ann.provider:
            click.echo(f"   Provider: {terradev_ann.provider.value}")
        if terradev_ann.gpu_type:
            click.echo(f"   GPU Type: {terradev_ann.gpu_type.value}")
        if terradev_ann.gpu_count:
            click.echo(f"   GPU Count: {terradev_ann.gpu_count}")

        click.echo(f"Tip: Run 'terradev job {yaml_file}' to execute this pipeline")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Import failed: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(f" Started recording '{name}'")
        click.echo(f"   Output: {recording_file}")
        click.echo(
            f"Tip: Run 'terradev record stop --name {name} --export pipeline.yaml' when done"
        )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to start recording: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(f"ERROR: Recording '{name}' not found", err=True)
            raise SystemExit(1)

        # Load recording
        with open(recording_file, "r") as f:
            recording_data = json.load(f)

        # Update status
        recording_data["status"] = "stopped"
        recording_data["stopped_at"] = datetime.now().isoformat()

        with open(recording_file, "w") as f:
            json.dump(recording_data, f, indent=2)

        click.echo(f" Stopped recording '{name}'")

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

            click.echo(f"UPLOAD: Exported recording as pipeline: {export}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to stop recording: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(f"OK: Created trigger '{name}'")
        click.echo(f"   Type: {trigger_type}")
        click.echo(f"   Pipeline: {pipeline}")
        click.echo(f"   Environment: {environment}")

        if event:
            click.echo(f"   Event: {event}")
        if schedule:
            click.echo(f"   Schedule: {schedule}")
        if condition:
            click.echo(f"   Condition: {condition}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to create trigger: {e}", err=True)
        raise SystemExit(1)
@triggers.command("list")
def list_triggers():
    """List all triggers"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        if not trigger_manager.triggers:
            click.echo("No triggers found")
            click.echo("Tip: Use 'terradev triggers create' to create a trigger")
            return

        click.echo(
            f"{'Name':<20} {'Type':<12} {'Pipeline':<20} {'Environment':<12} {'Enabled':<8} {'Count':<6}"
        )
        click.echo("─" * 92)

        for trigger in trigger_manager.triggers.values():
            enabled = "OK:" if trigger.enabled else "ERROR:"
            click.echo(
                f"{trigger.name:<20} {trigger.type.value:<12} {trigger.target_pipeline:<20} "
                f"{trigger.target_environment.value:<12} {enabled:<8} {trigger.trigger_count:<6}"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to list triggers: {e}", err=True)
        raise SystemExit(1)
@triggers.command("enable")
@click.argument("name")
def enable_trigger(name):
    """Enable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = True
                click.echo(f"OK: Enabled trigger '{name}'")
                return

        click.echo(f"ERROR: Trigger '{name}' not found", err=True)
        raise SystemExit(1)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to enable trigger: {e}", err=True)
        raise SystemExit(1)
@triggers.command("disable")
@click.argument("name")
def disable_trigger(name):
    """Disable a trigger"""
    try:
        from terradev_cli.core.event_system import trigger_manager

        for trigger in trigger_manager.triggers.values():
            if trigger.name == name:
                trigger.enabled = False
                click.echo(f" Disabled trigger '{name}'")
                return

        click.echo(f"ERROR: Trigger '{name}' not found", err=True)
        raise SystemExit(1)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to disable trigger: {e}", err=True)
        raise SystemExit(1)
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
            event_data = _safe_json(data, "data")

        event = Event(type=event_type_enum, source=source, data=event_data)

        event_bus.publish(event)
        click.echo(f" Fired event: {event_type}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to fire event: {e}", err=True)
        raise SystemExit(1)
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
            click.echo("No artifacts found")
            return

        click.echo(
            f"{'Name':<20} {'Type':<12} {'Environment':<12} {'Version':<8} {'Created':<20}"
        )
        click.echo("─" * 76)

        for artifact in sorted(artifacts, key=lambda x: x.created_at, reverse=True):
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            click.echo(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.environment.value:<12} "
                f"{artifact.version:<8} {created:<20}"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to list environments: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(f"ERROR: Artifact '{artifact_name}' not found in {from_env}", err=True)
            raise SystemExit(1)

        promotion = environment_manager.request_promotion(
            artifact_id=artifact.id,
            from_env=from_enum,
            to_env=to_enum,
            requested_by=user,
        )

        click.echo(f" Promotion requested: {from_env} -> {to_env}")
        click.echo(f"   Artifact: {artifact_name}")
        click.echo(f"   Promotion ID: {promotion.id}")
        click.echo(f"   Status: {promotion.status}")
        click.echo(f"Tip: Use 'terradev environments approve {promotion.id}' to complete")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to request promotion: {e}", err=True)
        raise SystemExit(1)
@environments.command("approve")
@click.argument("promotion_id")
@click.option("--user", default="cli-admin", help="User approving promotion")
def approve_promotion(promotion_id, user):
    """Approve and execute promotion"""
    try:
        from terradev_cli.core.event_system import environment_manager

        success = environment_manager.approve_promotion(promotion_id, user)

        if success:
            click.echo(f"OK: Promotion approved and executed: {promotion_id}")
        else:
            click.echo(f"ERROR: Promotion not found or failed: {promotion_id}", err=True)
            raise SystemExit(1)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to approve promotion: {e}", err=True)
        raise SystemExit(1)
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
            click.echo("No promotion history found")
            return

        click.echo(
            f"{'ID':<8} {'Artifact':<20} {'From':<10} {'To':<10} {'Status':<12} {'Requested':<20}"
        )
        click.echo("─" * 84)

        for promo in promotions:
            artifact_name = "unknown"
            if promo.artifact_id in lineage_service.artifacts:
                artifact_name = lineage_service.artifacts[promo.artifact_id].name

            requested = promo.requested_at.strftime("%Y-%m-%d %H:%M")
            click.echo(
                f"{promo.id[:8]:<8} {artifact_name:<20} {promo.from_env.value:<10} "
                f"{promo.to_env.value:<10} {promo.status:<12} {requested:<20}"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get promotion history: {e}", err=True)
        raise SystemExit(1)
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
@click.option("--size", type=click.IntRange(1, 1000000), help="Size in bytes")
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

        click.echo(f"OK: Registered artifact: {type} {name}")
        click.echo(f"   ID: {artifact.id}")
        click.echo(f"   Environment: {environment}")
        click.echo(f"   URI: {uri}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to register artifact: {e}", err=True)
        raise SystemExit(1)
@lineage.command("graph")
@click.argument("artifact_id")
@click.option("--direction", type=click.Choice(["up", "down", "both"]), default="both")
def lineage_graph(artifact_id, direction):
    """Show lineage graph for artifact"""
    try:
        from terradev_cli.core.event_system import lineage_service

        graph = lineage_service.get_lineage(artifact_id, direction)

        if not graph["parents"] and not graph["children"]:
            click.echo(f"No lineage found for artifact {artifact_id}")
            return

        click.echo(f" Lineage for {artifact_id}:")

        if graph["parents"]:
            click.echo(f"\n Parents ({len(graph['parents'])}):")
            for parent in graph["parents"]:
                click.echo(
                    f"   {parent.type.value} {parent.name} ({parent.environment.value})"
                )

        if graph["children"]:
            click.echo(f"\nUPLOAD: Children ({len(graph['children'])}):")
            for child in graph["children"]:
                click.echo(f"   {child.type.value} {child.name} ({child.environment.value})")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get lineage: {e}", err=True)
        raise SystemExit(1)
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
            click.echo("No production artifacts found")
            return

        click.echo(" Production Artifacts:")
        click.echo(
            f"{'Name':<20} {'Type':<12} {'Version':<8} {'Created':<20} {'Created By':<15}"
        )
        click.echo("─" * 79)

        for artifact in artifacts:
            created = artifact.created_at.strftime("%Y-%m-%d %H:%M")
            click.echo(
                f"{artifact.name:<20} {artifact.type.value:<12} {artifact.version:<8} "
                f"{created:<20} {artifact.created_by:<15}"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get production artifacts: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(f"ERROR: No lineage found for model '{model_identifier}'", err=True)
            return

        click.echo(f" Lineage for model: {model_identifier}")
        click.echo(
            f"{'Execution ID':<8} {'Pipeline':<20} {'Environment':<12} {'Status':<10} {'Timestamp':<20}"
        )
        click.echo("─" * 75)

        for record in records:
            exec_id = record.id[:8]
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M")
            click.echo(
                f"{exec_id:<8} {record.pipeline_id:<20} {record.environment.value:<12} "
                f"{record.status:<10} {timestamp:<20}"
            )

        # Show detailed view of latest execution
        if records:
            latest = records[0]
            click.echo("\n Latest Execution Details:")
            click.echo(f"   Execution ID: {latest.id}")
            click.echo(f"   Pipeline: {latest.pipeline_id}")
            click.echo(f"   Environment: {latest.environment.value}")
            click.echo(f"   Status: {latest.status}")
            click.echo(f"   Duration: {latest.duration_seconds:.1f}s")
            click.echo(f"   GPU Hours: {latest.gpu_hours:.2f}")
            click.echo(f"   Cost: ${latest.compute_cost:.2f}")

            if latest.hyperparameters:
                click.echo("\n  Hyperparameters:")
                for key, value in latest.hyperparameters.items():
                    click.echo(f"      {key}: {value}")

            if latest.datasets:
                click.echo(f"\n Input Datasets ({len(latest.datasets)}):")
                for dataset_id in latest.datasets[:5]:  # Show first 5
                    click.echo(f"      {dataset_id[:12]}...")
                if len(latest.datasets) > 5:
                    click.echo(f"      ... and {len(latest.datasets) - 5} more")

            if latest.output_models:
                click.echo(f"\n Output Models ({len(latest.output_models)}):")
                for model_id in latest.output_models:
                    click.echo(f"      {model_id[:12]}...")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to show lineage: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(f"ERROR: {diff['error']}", err=True)
            raise SystemExit(1)

        click.echo(" Comparing executions:")
        click.echo(
            f"   Execution 1: {diff['execution_1']['id']} ({diff['execution_1']['timestamp']})"
        )
        click.echo(
            f"   Execution 2: {diff['execution_2']['id']} ({diff['execution_2']['timestamp']})"
        )

        if not diff["differences"]:
            click.echo("\nOK: No differences found")
            return

        click.echo("\n Differences:")

        if "hyperparameters" in diff["differences"]:
            click.echo("\n  Hyperparameters:")
            for key, values in diff["differences"]["hyperparameters"].items():
                click.echo(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "environment_variables" in diff["differences"]:
            click.echo("\n Environment Variables:")
            for key, values in diff["differences"]["environment_variables"].items():
                click.echo(f"   {key}: {values['exec1']} → {values['exec2']}")

        if "inputs" in diff["differences"]:
            click.echo("\n Input Artifacts:")
            for change_type, artifacts in diff["differences"]["inputs"].items():
                click.echo(f"   {change_type}: {', '.join(artifacts)}")

        if "resources" in diff["differences"]:
            click.echo("\nCOST: Resource Usage:")
            for resource, values in diff["differences"]["resources"].items():
                click.echo(f"   {resource}: {values['exec1']} → {values['exec2']}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to diff lineage: {e}", err=True)
        raise SystemExit(1)
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
            click.echo(f"OK: Exported lineage to {output}")
        else:
            click.echo(data)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to export lineage: {e}", err=True)
        raise SystemExit(1)
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
                click.echo(f"ERROR: {trace['error']}", err=True)
                raise SystemExit(1)

            click.echo(f" Tracing lineage from checkpoint: {checkpoint}")
            click.echo("\n Created By:")
            created_by = trace["created_by"]
            click.echo(f"   Execution: {created_by['execution_id']}")
            click.echo(f"   Pipeline: {created_by['pipeline_id']}")
            click.echo(f"   Environment: {created_by['environment']}")
            click.echo(f"   Timestamp: {created_by['timestamp']}")

            if trace["inputs"]:
                click.echo("\n Input Artifacts:")
                if "datasets" in trace["inputs"]:
                    click.echo(f"   Datasets ({len(trace['inputs']['datasets'])}):")
                    for dataset in trace["inputs"]["datasets"]:
                        click.echo(f"      {dataset['name']} ({dataset['id'][:12]}...)")

                if "models" in trace["inputs"]:
                    click.echo(f"   Models ({len(trace['inputs']['models'])}):")
                    for model in trace["inputs"]["models"]:
                        click.echo(f"      {model['name']} ({model['id'][:12]}...)")

            if trace["ancestors"]:
                click.echo("\n Ancestor Executions:")
                for ancestor in trace["ancestors"]:
                    click.echo(
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
                click.echo(f"ERROR: Execution '{execution}' not found", err=True)
                raise SystemExit(1)

            click.echo(f" Tracing execution: {exec_record.id}")
            click.echo(f"   Pipeline: {exec_record.pipeline_id}")
            click.echo(f"   Environment: {exec_record.environment.value}")
            click.echo(f"   Status: {exec_record.status}")
            click.echo(f"   Timestamp: {exec_record.timestamp}")

            # Show artifact lineage
            all_artifacts = (
                exec_record.datasets
                + exec_record.models
                + exec_record.output_models
                + exec_record.output_checkpoints
            )

            if all_artifacts:
                click.echo("\n Artifact Lineage:")
                for artifact_id in all_artifacts:
                    if artifact_id in lineage_service.artifacts:
                        artifact = lineage_service.artifacts[artifact_id]
                        graph = lineage_service.get_lineage(artifact_id, "up")

                        click.echo(f"\n   {artifact.type.value}: {artifact.name}")
                        click.echo(f"      Environment: {artifact.environment.value}")
                        click.echo(f"      Created: {artifact.created_at}")

                        if graph["parents"]:
                            click.echo(f"      Parents: {len(graph['parents'])}")
                            for parent in graph["parents"][:3]:  # Show first 3
                                click.echo(f"         └─ {parent.type.value}: {parent.name}")

        else:
            click.echo("ERROR: Must specify either --checkpoint or --execution", err=True)
            raise SystemExit(1)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to trace artifacts: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(" Started automatic lineage tracking")
        click.echo(f"   Execution ID: {execution.id}")
        click.echo(f"   Pipeline: {pipeline}")
        click.echo(f"   Environment: {environment}")
        click.echo("   Use this ID to add artifacts and complete the execution")

        # Show example commands for manual tracking
        click.echo("\nTip: Example commands to track this execution:")
        click.echo(f"   terradev lineage add-input {execution.id} dataset <dataset-id>")
        click.echo(f"   terradev lineage add-output {execution.id} model <model-id>")
        click.echo(f"   terradev lineage complete {execution.id}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to start lineage tracking: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(f"OK: Added input {artifact_type}: {artifact_id}")
        click.echo(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to add input artifact: {e}", err=True)
        raise SystemExit(1)
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

        click.echo(f"OK: Added output {artifact_type}: {artifact_id}")
        click.echo(f"   Execution: {execution_id}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to add output artifact: {e}", err=True)
        raise SystemExit(1)
@lineage.command("complete")
@click.argument("execution_id")
@click.option("--status", default="completed", help="Final status (completed, failed)")
def complete_execution(execution_id, status):
    """Complete execution and finalize lineage record"""
    try:
        from terradev_cli.core.auto_lineage import auto_lineage

        auto_lineage.complete_execution(execution_id, status)

        click.echo(f"OK: Completed execution: {execution_id}")
        click.echo(f"   Status: {status}")
        click.echo("   Lineage record finalized and available for queries")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to complete execution: {e}", err=True)
        raise SystemExit(1)
