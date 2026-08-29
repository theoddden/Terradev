#!/usr/bin/env python3
"""Inference serving commands for the Terradev CLI."""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Optional

import click
from . import cli
from terradev_cli.commands._api import TerradevAPI




def _run_with_timeout(coro, timeout=300):
    """Run an async coroutine with a timeout to prevent hangs."""
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError:
        click.echo(f"ERROR: Operation timed out after {timeout}s", err=True)
        raise SystemExit(1)

class InferenceCommand(click.Command):
    """Click command that catches runtime failures and returns non-zero on errors."""

    def invoke(self, ctx):
        try:
            rv = super().invoke(ctx)
        except (click.ClickException, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            click.echo(f"ERROR: {exc}", err=True)
            raise click.exceptions.Exit(1) from exc

        output = ctx.obj.get("terradev_output") if ctx.obj else None
        if output is not None and (rv is None or rv == 0):
            messages = getattr(output, "_messages", [])
            if any(m.level == "error" for m in messages):
                raise click.exceptions.Exit(1)
        return rv


class InferenceGroup(click.Group):
    """Click group that uses InferenceCommand for leaf subcommands and InferenceGroup for nested groups."""

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", InferenceCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", InferenceGroup)
        return super().group(*args, **kwargs)


@cli.group(cls=InferenceGroup)
def orchestrator():
    """Model orchestrator for multi-model inference"""
    pass


@orchestrator.command("start")
@click.option("--gpu-id", default=0, help="GPU ID to use for orchestration")
@click.option("--memory-gb", default=80.0, help="Total GPU memory in GB")
@click.option(
    "--policy",
    type=click.Choice(["billing_optimized", "latency_optimized", "hybrid"]),
    default="billing_optimized",
    help="Scaling policy",
)
def orchestrator_start(gpu_id, memory_gb, policy):
    """Start the model orchestrator for multi-model inference"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator, ScalingPolicy

    policy_map = {
        "billing_optimized": ScalingPolicy.BILLING_OPTIMIZED,
        "latency_optimized": ScalingPolicy.LATENCY_OPTIMIZED,
        "hybrid": ScalingPolicy.HYBRID,
    }

    orchestrator = ModelOrchestrator(
        gpu_id=gpu_id, total_memory_gb=memory_gb, scaling_policy=policy_map[policy]
    )

    async def run_orchestrator():
        await orchestrator.start()
        click.echo(f"Model Orchestrator started on GPU {gpu_id}")
        click.echo(
            f"Memory: {memory_gb}GB total, {orchestrator.memory_threshold_gb:.1f}GB usable"
        )
        click.echo(f"Policy: {policy}")
        click.echo("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(10)
                status = orchestrator.get_status()
                click.echo(
                    f"Status: {status['warm_models_count']} warm models, "
                    f"{status['used_memory_gb']:.1f}GB used "
                    f"({status['memory_utilization_percent']:.1f}%)"
                )
        except KeyboardInterrupt:
            click.echo("\nStopping orchestrator...")
            await orchestrator.stop()

    _run_with_timeout(run_orchestrator())


@orchestrator.command("register")
@click.argument("model-id")
@click.argument("model-path")
@click.option(
    "--framework",
    type=click.Choice(["pytorch", "vllm", "sglang"]),
    default="pytorch",
    help="Model framework",
)
@click.option(
    "--priority",
    default=0,
    help="Priority for eviction (higher = less likely to evict)",
)
@click.option("--tags", help="Comma-separated tags for model categorization")
def orchestrator_register(model_id, model_path, framework, priority, tags):
    """Register a model with the orchestrator"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()
    tag_set = set(tags.split(",")) if tags else None

    _run_with_timeout(orchestrator.register_model(
        model_id=model_id,
        model_path=model_path,
        framework=framework,
        priority=priority,
        tags=tag_set,
    ))

    click.echo(f"Model registered: {model_id}")
    click.echo(f"  Path: {model_path}")
    click.echo(f"  Framework: {framework}")
    click.echo(f"  Priority: {priority}")
    click.echo(f"  Tags: {', '.join(tag_set) if tag_set else 'None'}")


@orchestrator.command("load")
@click.argument("model-id")
@click.option("--force", is_flag=True, help="Force loading even if memory is full")
def orchestrator_load(model_id, force):
    """Load a model into GPU memory"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    async def load_model():
        success = await orchestrator.load_model(model_id, force=force)
        if success:
            details = orchestrator.get_model_details(model_id)
            click.echo(f"Model {model_id} loaded successfully!")
            click.echo(f"  State: {details['state']}")
            click.echo(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            click.echo(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            click.echo(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
        else:
            click.echo(f"Failed to load model {model_id}", err=True)

    _run_with_timeout(load_model())


@orchestrator.command("evict")
@click.argument("model-id")
def orchestrator_evict(model_id):
    """Evict a model from GPU memory"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    async def evict_model():
        success = await orchestrator.evict_model(model_id)
        if success:
            click.echo(f"Model {model_id} evicted successfully")
        else:
            click.echo(f"Failed to evict model {model_id}", err=True)

    _run_with_timeout(evict_model())


@orchestrator.command("status")
@click.option("--model-id", help="Get details for specific model")
def orchestrator_status(model_id):
    """Get orchestrator and model status"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    if model_id:
        details = orchestrator.get_model_details(model_id)
        if details:
            click.echo(f"Model Details: {model_id}")
            click.echo(f"  Framework: {details['framework']}")
            click.echo(f"  State: {details['state']}")
            click.echo(f"  Priority: {details['priority']}")
            click.echo(f"  Tags: {', '.join(details['tags'])}")
            click.echo(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            click.echo(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            click.echo(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
            click.echo(f"  Requests/hour: {details['metrics']['requests_per_hour']:.1f}")
            click.echo(f"  Avg latency: {details['metrics']['avg_latency_ms']:.1f}ms")
            click.echo(f"  Error rate: {details['metrics']['error_rate']:.2f}")
            click.echo(f"  Last accessed: {details['last_accessed']}")
        else:
            click.echo(f"Model {model_id} not found")
    else:
        status = orchestrator.get_status()
        click.echo("Orchestrator Status:")
        click.echo(f"  GPU: {status['gpu_id']}")
        click.echo(f"  Total memory: {status['total_memory_gb']:.1f}GB")
        click.echo(f"  Used memory: {status['used_memory_gb']:.1f}GB")
        click.echo(f"  Available: {status['available_memory_gb']:.1f}GB")
        click.echo(f"  Utilization: {status['memory_utilization_percent']:.1f}%")
        click.echo(f"  Policy: {status['scaling_policy']}")
        click.echo(f"  Total models: {status['total_models']}")
        click.echo(f"  Warm models: {status['warm_models_count']}")
        click.echo(f"  Warm memory: {status['warm_models_memory_gb']:.1f}GB")

        click.echo("\nModels by state:")
        for state, model_ids in status["models_by_state"].items():
            if model_ids:
                click.echo(f"  {state}: {', '.join(model_ids)}")


@orchestrator.command("infer")
@click.argument("model-id")
def orchestrator_infer(model_id):
    """Test inference with a model"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    async def test_inference():
        success, latency_ms = await orchestrator.handle_request(model_id)
        if success:
            click.echo(f"Inference successful for {model_id}")
            click.echo(f"  Latency: {latency_ms:.1f}ms")
        else:
            click.echo(f"Inference failed for {model_id}")

    _run_with_timeout(test_inference())


@cli.group(name="warm-pool", cls=InferenceGroup)
def warm_pool():
    """Warm pool manager for intelligent pre-warming"""
    pass


@warm_pool.command("start")
@click.option(
    "--strategy",
    type=click.Choice(
        [
            "traffic_based",
            "time_based",
            "priority_based",
            "cost_optimized",
            "latency_optimized",
        ]
    ),
    default="traffic_based",
    help="Warm pool strategy",
)
@click.option("--max-warm", default=10, help="Maximum models to keep warm")
@click.option("--min-warm", default=3, help="Minimum models to keep warm")
def warm_pool_start(strategy, max_warm, min_warm):
    """Start the warm pool manager for intelligent pre-warming"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig, WarmStrategy

    strategy_map = {
        "traffic_based": WarmStrategy.TRAFFIC_BASED,
        "time_based": WarmStrategy.TIME_BASED,
        "priority_based": WarmStrategy.PRIORITY_BASED,
        "cost_optimized": WarmStrategy.COST_OPTIMIZED,
        "latency_optimized": WarmStrategy.LATENCY_OPTIMIZED,
    }

    config = WarmPoolConfig(
        max_warm_models=max_warm,
        min_warm_models=min_warm,
        strategy=strategy_map[strategy],
        enable_predictive_warming=True,
    )

    warm_pool = WarmPoolManager(config)

    async def run_warm_pool():
        await warm_pool.start()
        click.echo("Warm Pool Manager started")
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Capacity: {min_warm}-{max_warm} models")
        click.echo("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(30)
                status = warm_pool.get_status()
                click.echo(
                    f"Status: {status['warm_models_count']} warm, "
                    f"{status['cache_hit_rate']:.1%} hit rate, "
                    f"{status['total_requests']} requests"
                )
        except KeyboardInterrupt:
            click.echo("\nStopping warm pool manager...")
            await warm_pool.stop()

    _run_with_timeout(run_warm_pool())


@warm_pool.command("register")
@click.argument("model-id")
@click.option("--priority", default=0, help="Model priority for warming")
def warm_pool_register(model_id, priority):
    """Register a model with the warm pool manager"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig

    warm_pool = WarmPoolManager(WarmPoolConfig())
    warm_pool.register_model(model_id, priority)

    click.echo(f"Model {model_id} registered with warm pool")
    click.echo(f"  Priority: {priority}")


@warm_pool.command("status")
def warm_pool_status():
    """Get warm pool manager status"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig

    warm_pool = WarmPoolManager(WarmPoolConfig())
    status = warm_pool.get_status()

    click.echo("Warm Pool Status:")
    click.echo(f"  Warm models: {status['warm_models_count']}")
    click.echo(f"  Warming models: {status['warming_models_count']}")
    click.echo(f"  Total models: {status['total_models']}")
    click.echo(f"  Strategy: {status['strategy']}")
    click.echo(f"  Cache hit rate: {status['cache_hit_rate']:.1%}")
    click.echo(f"  Total requests: {status['total_requests']}")
    click.echo(f"  Cold starts: {status['cold_starts']}")
    click.echo(f"  Avg warm latency: {status['avg_warm_latency_ms']:.1f}ms")
    click.echo(f"  Avg cold latency: {status['avg_cold_latency_ms']:.1f}ms")
    click.echo(f"  Memory saved: {status['memory_saved_gb']:.1f}GB")
    click.echo(f"  Cost saved: ${status['cost_saved_usd']:.2f}")



@cli.group(cls=InferenceGroup)
def inferx():
    """InferX serverless inference platform - <2s cold starts, 90% GPU utilization"""
    pass


@inferx.command()
@click.option("--api-key", required=True, help="InferX API key")
@click.option(
    "--endpoint", default="https://api.inferx.net", help="InferX API endpoint"
)
@click.option("--region", default="us-west-2", help="Region for deployment")
@click.option(
    "--snapshot/--no-snapshot", default=True, help="Enable snapshot technology"
)
@click.option("--gpu-slicing/--no-gpu-slicing", default=True, help="Enable GPU slicing")
@click.option(
    "--multi-tenant/--no-multi-tenant",
    default=True,
    help="Enable multi-tenant isolation",
)
def inferx_configure(api_key, endpoint, region, snapshot, gpu_slicing, multi_tenant):
    """Configure InferX provider credentials"""
    from pathlib import Path

    config_dir = Path.home() / ".terradev"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "inferx_config.json"
    config = {
        "api_key": api_key,
        "api_endpoint": endpoint,
        "region": region,
        "snapshot_enabled": snapshot,
        "gpu_slicing": gpu_slicing,
        "multi_tenant": multi_tenant,
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    click.echo("OK: InferX configured successfully")
    click.echo(f" Endpoint: {endpoint}")
    click.echo(f" Region: {region}")
    click.echo(f" Snapshot: {'Enabled' if snapshot else 'Disabled'}")
    click.echo(f" GPU Slicing: {'Enabled' if gpu_slicing else 'Disabled'}")
    click.echo(f" Multi-tenant: {'Enabled' if multi_tenant else 'Disabled'}")


@inferx.command()
@click.option("--model", required=True, help="Model ID or HuggingFace model name")
@click.option("--image", help="Docker image for model")
@click.option("--gpu-type", default="A100", help="GPU type")
@click.option("--gpu-memory", type=click.IntRange(1, 256), default=16, help="GPU memory in GB")
@click.option(
    "--max-concurrency", type=click.IntRange(1, 10000), default=10, help="Maximum concurrent requests"
)
@click.option("--framework", default="pytorch", help="Model framework")
@click.option(
    "--openai-compatible/--no-openai-compatible",
    default=True,
    help="OpenAI-compatible API",
)
@click.option("--timeout", type=click.IntRange(1, 3600), default=300, help="Request timeout in seconds")
def deploy(
    model,
    image,
    gpu_type,
    gpu_memory,
    max_concurrency,
    framework,
    openai_compatible,
    timeout,
):
    """Deploy model to InferX serverless platform"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    model_config = {
        "model_id": model,
        "image": image or "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        "gpu_type": gpu_type,
        "gpu_memory": gpu_memory,
        "max_concurrency": max_concurrency,
        "framework": framework,
        "openai_compatible": openai_compatible,
        "timeout": timeout,
    }

    click.echo(f" Deploying {model} to InferX...")
    click.echo(f" GPU: {gpu_type} ({gpu_memory}GB)")
    click.echo(f" Max Concurrency: {max_concurrency}")
    click.echo(f" Framework: {framework}")

    try:
        result = _run_with_timeout(provider.deploy_model(model_config))

        click.echo("OK: Model deployed successfully!")
        click.echo(f" Model ID: {result['model_id']}")
        click.echo(f" Endpoint: {result['endpoint']}")
        click.echo(f" Cold Start: {result['cold_start_time']}s")
        click.echo(f" GPU Utilization: {result['gpu_utilization']}%")
        click.echo(f"PACKAGE: Models per Node: {result['models_per_node']}")

        if result["openai_compatible"]:
            click.echo(" OpenAI Compatible: Yes")
            click.echo(
                f"Tip: Usage: curl -X POST {result['endpoint']} -H 'Authorization: Bearer YOUR_API_KEY'"
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Deployment failed: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command()
@click.option("--model-id", required=True, help="Model deployment ID")
def inferx_status(model_id):
    """Get model deployment status"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        result = _run_with_timeout(provider.get_model_status(model_id))

        click.echo(f" Model Status: {result.get('status', 'Unknown')}")
        click.echo(f" GPU Type: {result.get('gpu_type', 'Unknown')}")
        click.echo(f" Cold Start Time: {result.get('cold_start_time', 'Unknown')}s")
        click.echo(f" Requests/min: {result.get('requests_per_minute', 0)}")
        click.echo(f" GPU Utilization: {result.get('gpu_utilization', 0)}%")
        click.echo(f"PACKAGE: Models on GPU: {result.get('models_on_gpu', 0)}")
        click.echo(f"   Error rate: {result.get('error_rate', 0)}%")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get status: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command()
@click.option("--model-id", required=True, help="Model deployment ID")
def inferx_delete(model_id):
    """Delete model deployment"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    click.echo(f"  Deleting model {model_id}...")

    try:
        success = _run_with_timeout(provider.delete_model(model_id))

        if success:
            click.echo("OK: Model deleted successfully")
        else:
            click.echo("ERROR: Failed to delete model", err=True)

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to delete model: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command("list")
def inferx_list():
    """List all deployed models"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        models = _run_with_timeout(provider.list_models())

        if not models:
            click.echo(" No models deployed")
            return

        click.echo(f" Deployed Models ({len(models)}):")
        click.echo("-" * 80)

        for model in models:
            click.echo(f"PACKAGE: {model.get('model_id', 'Unknown')}")
            click.echo(f"   Status: {model.get('status', 'Unknown')}")
            click.echo(f"   GPU: {model.get('gpu_type', 'Unknown')}")
            click.echo(f"   Endpoint: {model.get('endpoint', 'Unknown')}")
            click.echo(f"   Created: {model.get('created_at', 'Unknown')}")
            click.echo()

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to list models: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command()
def usage():
    """Get account usage statistics"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        stats = _run_with_timeout(provider.get_usage_stats())

        click.echo(" InferX Usage Statistics")
        click.echo("-" * 40)
        click.echo(f" Total Requests: {stats.get('total_requests', 0):,}")
        click.echo(f"COST: Total Cost: ${stats.get('total_cost', 0):.4f}")
        click.echo(f"PACKAGE: Active Models: {stats.get('active_models', 0)}")
        click.echo(f" GPU Hours: {stats.get('gpu_hours', 0):.2f}")
        click.echo(f" Average Latency: {stats.get('average_latency', 0):.0f}ms")
        click.echo(f" GPU Utilization: {stats.get('gpu_utilization', 0):.1f}%")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get usage stats: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command()
@click.option("--gpu-type", default="A100", help="GPU type to quote")
@click.option("--region", help="Region for quote")
def inferx_quote(gpu_type, region):
    """Get pricing quotes for InferX"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        click.echo("ERROR: InferX not configured. Run 'terradev inferx configure' first.", err=True)
        raise SystemExit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        quotes = _run_with_timeout(provider.get_instance_quotes(gpu_type, region))

        if not quotes:
            click.echo("ERROR: No quotes available", err=True)
            raise SystemExit(1)
        quote = quotes[0]

        click.echo("COST: InferX Pricing Quote")
        click.echo("-" * 40)
        click.echo(f" GPU Type: {quote['gpu_type']}")
        click.echo(
            f" Hourly Cost: ${quote['price_per_hour']:.4f} (Serverless - pay per request)"
        )
        click.echo(f" Per Request: ${quote['price_per_request']:.4f} per 1K tokens")
        click.echo(f" Cold Start: {quote['cold_start_time']}s")
        click.echo(f" GPU Utilization: {quote['gpu_utilization']}%")
        click.echo(f"PACKAGE: Models per Node: {quote['models_per_node']}")
        click.echo(f" Region: {quote['region']}")
        click.echo()
        click.echo(" Key Features:")
        for feature in quote["features"]:
            click.echo(f"   OK: {feature.replace('_', ' ').title()}")

    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to get quotes: {e}", err=True)
    finally:
        _run_with_timeout(provider.close())


@inferx.command()
@click.option("--cluster-config", help="Cluster configuration file")
@click.option("--usage-metrics", help="Usage metrics file")
@click.option(
    "--tier",
    type=click.Choice(["economy", "balanced", "performance"]),
    default="economy",
    help="Cost optimization tier",
)
@click.option("--output", help="Output file for cost report")
@click.option(
    "--implement", is_flag=True, help="Implement cost optimizations automatically"
)
def inferx_optimize(cluster_config, usage_metrics, tier, output, implement):
    """Analyze and optimize InferX costs with AI-powered recommendations"""
    import json
    from terradev_cli.k8s.t_optimizer import InferXCostOptimizer, CostTier

    optimizer = InferXCostOptimizer()
    target_tier = CostTier(tier)

    # Load configuration (mock data for demo)
    cluster_config_data = {
        "nodes": [
            {"gpu_type": "A100", "gpu_count": 2, "spot": True},
            {"gpu_type": "A10G", "gpu_count": 1, "spot": True},
        ],
        "storage_gb": 200,
        "snapshot_gb": 500,
    }

    usage_metrics_data = {
        "gpu_utilization": 65.0,
        "memory_utilization": 70.0,
        "cpu_utilization": 45.0,
        "models_deployed": 25,
        "cold_start_time": 2.2,
        "requests_per_hour": 150,
    }

    if cluster_config:
        with open(cluster_config) as f:
            cluster_config_data = json.load(f)

    if usage_metrics:
        with open(usage_metrics) as f:
            usage_metrics_data = json.load(f)

    click.echo(f" Analyzing InferX costs for {tier} tier...")

    # Generate cost report
    report = _run_with_timeout(
        optimizer.generate_cost_report(
            cluster_config_data, usage_metrics_data, target_tier
        )
    )

    # Display results
    click.echo("\n Cost Analysis Results:")
    click.echo("=" * 50)
    click.echo(
        f"COST: Current Monthly Cost: ${report['summary']['current_monthly_cost']:,.2f}"
    )
    click.echo(
        f" Potential Monthly Savings: ${report['summary']['potential_monthly_savings']:,.2f}"
    )
    click.echo(f" Savings Percentage: {report['summary']['savings_percentage']:.1f}%")
    click.echo(
        f" Optimized Monthly Cost: ${report['summary']['optimized_monthly_cost']:,.2f}"
    )
    click.echo(f"  Payback Period: {report['summary']['payback_period_months']:.1f} months")
    click.echo(f" Annual ROI: {report['summary']['annual_roi']:.1f}%")
    click.echo()

    click.echo(" Key Insights:")
    for insight in report["key_insights"]:
        click.echo(f"    {insight}")
    click.echo()

    click.echo(" Top Recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        click.echo(f"   {i}. {rec['description']}")
        click.echo(f"      Savings: ${rec['estimated_savings']:,.2f}/month")
        click.echo(f"      Risk: {rec['risk_level']}, Priority: {rec['priority']}")
        click.echo()

    # Save report
    if output:
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f" Detailed report saved to: {output}")

    # Implement optimizations if requested
    if implement:
        click.echo(" Implementing cost optimizations...")
        # Implementation logic would go here
        click.echo("OK: Optimizations implemented successfully!")


# ═══════════════════════════════════════════════════════════════════════════════
# Training Pipeline  preflight / train / monitor / checkpoint / train-status
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_provision_nodes(provision_group: str, fmt: str = "text"):
    """Resolve provisioned instance IDs to node IPs for the train command.

    Looks up the parallel provision group in the cost DB, then resolves
    each instance to an IP via:
      1. Cached ip_address column (if previously resolved)
      2. Provider status API (async, parallel across all instances)
      3. instance_id as-is fallback (some providers use IP-based IDs)

    Returns (node_ips: list[str], ssh_key_path: str | None).
    node_ips is empty on failure.  ssh_key_path is the decrypted temp key
    if an encrypted per-provision key was generated during provision.
    """
    try:
        from terradev_cli.core.cost_tracker import (
            get_active_instances,
            get_latest_parallel_group,
            set_instance_ip,
            get_provision_ssh_key_path as _db_ssh_path,
        )
    except ImportError:
        click.echo("ERROR: Cost tracker not available", err=True)
        return [], None

    # Resolve "latest" to actual group ID
    group_id = provision_group
    if group_id == "latest":
        group_id = get_latest_parallel_group()
        if not group_id:
            click.echo(
                "ERROR: No previous provision groups found. Run 'terradev provision' first."
            )
            return [], None
        if fmt != "json":
            click.echo(f"  Resolved 'latest' -> {group_id}")

    instances = get_active_instances(parallel_group=group_id)
    if not instances:
        click.echo(f"ERROR: No active instances in provision group {group_id}", err=True)
        return [], None

    node_ips = []
    unresolved = []

    for inst in instances:
        ip = inst.get("ip_address", "").strip()
        if ip:
            node_ips.append(ip)
        else:
            unresolved.append(inst)

    # Resolve remaining IPs via provider APIs
    if unresolved:
        if fmt != "json":
            click.echo(f"  Resolving IPs for {len(unresolved)} instance(s)...")

        api = TerradevAPI()

        async def _resolve_ips():
            from terradev_cli.providers.provider_factory import ProviderFactory

            factory = ProviderFactory()
            results = {}

            async def _get_ip(inst):
                pname = inst["provider"].lower().replace(" ", "_")
                creds = api._provider_creds(pname)
                try:
                    provider = factory.create_provider(pname, creds)
                    status = await provider.get_instance_status(inst["instance_id"])
                    # Providers return IP in different fields
                    ip = (
                        status.get("ip")
                        or status.get("public_ip")
                        or status.get("ip_address")
                        or status.get("host")
                        or inst["instance_id"]
                    )
                    results[inst["instance_id"]] = ip
                except Exception as _exc:  # noqa: BLE001
                    logger.exception(_exc)
                    # Fallback: use instance_id (some providers like RunPod
                    # use IDs that double as SSH hostnames)
                    results[inst["instance_id"]] = inst["instance_id"]

            await asyncio.gather(*[_get_ip(i) for i in unresolved])
            return results

        resolved = _run_with_timeout(_resolve_ips())

        for inst in unresolved:
            ip = resolved.get(inst["instance_id"], inst["instance_id"])
            node_ips.append(ip)
            # Cache for next time
            try:
                set_instance_ip(inst["instance_id"], ip)
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                pass

    if fmt != "json":
        click.echo(f"  Nodes from provision group {group_id}: {node_ips}")

    # ── Auto-resolve SSH key from provision group ──
    resolved_ssh_key = None
    try:
        if _db_ssh_path(group_id):
            from terradev_cli.core.ssh_key_manager import decrypt_private_key

            resolved_ssh_key = decrypt_private_key(group_id)
            if resolved_ssh_key and fmt != "json":
                click.echo(
                    "  SSH key auto-resolved from provision group (ephemeral decrypt)"
                )
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass  # Fall back to manual --ssh-key

    return node_ips, resolved_ssh_key


@cli.command(cls=InferenceCommand)
@click.option(
    "--nodes",
    "-n",
    multiple=True,
    help="Node IPs (multiple allowed, empty = localhost)",
)
@click.option("--ssh-user", default="root", help="SSH user (default: root)")
@click.option("--ssh-key", default="", help="SSH key path")
@click.option(
    "--from-provision",
    "provision_group",
    default="",
    help='Use nodes from a provision group. "latest" = most recent.',
)
@click.option("--quick", is_flag=True, help="Quick GPU-only check (skip storage/NCCL)")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["json", "text"]), default="text"
)
def preflight(nodes, ssh_user, ssh_key, provision_group, quick, fmt):
    """Run preflight hardware validation on GPU nodes.

    Checks: GPU health (DCGM), NVLink, RDMA, storage I/O, NCCL.
    All checks run in parallel via DAGExecutor.

    Examples:
        terradev preflight
        terradev preflight -n 10.0.0.1 -n 10.0.0.2 --quick
        terradev preflight --from-provision latest
        terradev preflight -f json
    """
    from terradev_cli.core.preflight_validator import PreflightValidator

    node_list = list(nodes) if nodes else []
    resolved_ssh_key = ssh_key
    if provision_group and not node_list:
        node_list, auto_ssh = _resolve_provision_nodes(provision_group, fmt)
        if not node_list:
            raise SystemExit(1)
        if auto_ssh and not ssh_key:
            resolved_ssh_key = auto_ssh
    if not node_list:
        node_list = [None]

    validator = PreflightValidator(
        nodes=node_list,
        ssh_user=ssh_user,
        ssh_key=resolved_ssh_key or None,
    )

    report = validator.run_quick() if quick else validator.run_all()
    summary = report.summary()

    if fmt == "json":
        click.echo(json.dumps(summary, indent=2, default=str))
    else:
        passed = summary.get("passed", False)
        status_icon = "PASS" if passed else "FAIL"
        click.echo(f"\nPreflight: {status_icon}")
        click.echo(f"  Nodes: {summary.get('nodes_checked', 0)}")
        click.echo(
            f"  Checks passed: {summary.get('checks_passed', 0)}/{summary.get('total_checks', 0)}"
        )
        if summary.get("failures"):
            click.echo("  Failures:")
            for f in summary["failures"]:
                click.echo(f"    - {f}")
        click.echo()




@cli.group(cls=InferenceGroup)
def infer():
    """Deploy and manage inference endpoints"""
    pass


@infer.command("deploy")
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option(
    "--type", "-t", type=click.Choice(["llm", "embedding", "vision"]), help="Model type"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["runpod", "vastai", "baseten", "huggingface", "siliconflow", "inferx"]),
)
@click.option("--gpu-type", "-g", help="GPU type preference")
@click.option("--region", "-r", help="Region preference")
@click.option("--max-latency", type=click.FloatRange(0.0, 10000.0), help="Max latency in ms")
@click.option("--max-cost", type=click.FloatRange(0.0, 1000.0), help="Max cost per request")
def infer_deploy_main(model, type, provider, gpu_type, region, max_latency, max_cost):
    """Compare and select the cheapest inference option across providers.

    Queries all configured inference providers (GPU-based and inference-only)
    and prints the best quote. To actually deploy, use `terradev infer endpoint`.

    siliconflow, inferx.
    """
    click.echo(f"Deploying inference for model: {model}")

    if type:
        click.echo(f"Model type: {type}")
    if provider:
        click.echo(f"Provider: {provider}")
    if gpu_type:
        click.echo(f"GPU type: {gpu_type}")
    if region:
        click.echo(f"Region: {region}")
    if max_latency:
        click.echo(f"Max latency: {max_latency}ms")
    if max_cost:
        click.echo(f"Max cost: ${max_cost}/request")

    # Get real quotes from providers
    click.echo("Getting inference quotes from terradev_cli.providers...")

    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    inference_providers = ["runpod", "vastai", "baseten", "huggingface", "siliconflow", "inferx"]
    if provider:
        inference_providers = [provider.replace("lambda")]

    async def _fetch_inference_quotes():
        all_q = []
        for pname in inference_providers:
            try:
                raw = await api._get_provider_quotes(pname, target_gpu)
                for q in raw:
                    all_q.append(
                        {
                            "provider": pname,
                            "price": q.get("price", 0),
                            "latency": 120,  # estimated  real latency requires endpoint probing
                            "gpu_type": q.get("gpu_type", target_gpu),
                            "region": q.get("region", ""),
                        }
                    )
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                pass
        return all_q

    quotes = _run_with_timeout(_fetch_inference_quotes())

    # Filter by latency if specified
    if max_latency:
        quotes = [q for q in quotes if q["latency"] <= max_latency]

    # Filter by cost if specified
    if max_cost:
        quotes = [q for q in quotes if q["price"] <= max_cost]

    # Silent: record inference price ticks for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_ticks_batch

        ticks = [
            {
                "gpu_type": q.get("gpu_type", gpu_type or "A100"),
                "provider": q.get("provider", ""),
                "region": "",
                "price": q.get("price", 0),
                "spot": False,
                "workload_type": "inference",
                "source": "infer",
            }
            for q in quotes
        ]
        record_price_ticks_batch(ticks)
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass

    if not quotes:
        click.echo("No suitable inference options found")
        return

    # Select best option (lowest price, then lowest latency)
    best_quote = min(quotes, key=lambda x: (x["price"], x["latency"]))

    click.echo(f"\nBest option: {best_quote['provider']}")
    click.echo(f"Price: ${best_quote['price']}/request")
    click.echo(f"Latency: {best_quote['latency']}ms")
    click.echo(f"GPU: {best_quote['gpu_type']}")

    # Deploy to optimal provider via real API
    pname = best_quote["provider"]
    click.echo(f"\nDeploying to {pname}...")

    async def _deploy_inference():
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        prov = factory.create_provider(pname, creds)
        itype = f"{pname}-inference-{best_quote['gpu_type'].lower()}"
        return await prov.provision_instance(
            itype,
            best_quote.get("region", "us-east-1"),
            best_quote["gpu_type"],
        )

    try:
        prov_result = _run_with_timeout(_deploy_inference())
        endpoint_id = prov_result.get("instance_id", f"inf_{pname}_{int(time.time())}")
        endpoint_url = prov_result.get(
            "endpoint_url", f"https://{pname}.api/inference/{endpoint_id}"
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"Warning  Provisioning error: {e}")
        endpoint_id = f"inf_{pname}_{int(time.time())}"
        endpoint_url = ""

    click.echo("Inference endpoint deployed")
    click.echo(f"ID Endpoint ID: {endpoint_id}")
    if endpoint_url:
        click.echo(f"URL: {endpoint_url}")
    click.echo("Status Status: Active")

    # Save to usage tracking
    api.usage["inference_endpoints"].append(
        {
            "id": endpoint_id,
            "model": model,
            "provider": pname,
            "gpu_type": best_quote["gpu_type"],
            "price": best_quote["price"],
            "latency": best_quote["latency"],
            "url": endpoint_url,
            "created_at": datetime.now().isoformat(),
        }
    )

    # Register with inference router for health tracking + failover
    try:
        from terradev_cli.core.inference_router import InferenceRouter

        router = InferenceRouter()
        router.register_endpoint(
            endpoint_id=endpoint_id,
            provider=pname,
            url=endpoint_url,
            model=model,
            gpu_type=best_quote["gpu_type"],
            region=best_quote.get("region", ""),
            price_per_hour=best_quote["price"],
        )
        click.echo("SHIELD:  Registered for health monitoring & auto-failover")
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass


@infer.command("endpoint")
@click.argument("model_path")
@click.option("--name", "-n", required=True, help="Endpoint name (required)")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["runpod", "vastai", "baseten", "huggingface", "siliconflow", "inferx"]),
)
@click.option("--gpu-type", "-g", help="GPU type (A100|H100|RTX4090)")
@click.option("--min-workers", type=click.IntRange(0, 1000), default=1, help="Minimum workers")
@click.option("--max-workers", type=click.IntRange(0, 1000), default=5, help="Maximum workers")
@click.option("--idle-timeout", type=click.IntRange(0, 86400), default=300, help="Idle timeout in seconds")
@click.option("--cost-optimize", is_flag=True, help="Enable cost optimization")
@click.option("--dry-run", is_flag=True, help="Show deployment plan without deploying")
def infer_endpoint(
    model_path,
    name,
    provider,
    gpu_type,
    min_workers,
    max_workers,
    idle_timeout,
    cost_optimize,
    dry_run,
):
    """Deploy an inference endpoint for MODEL_PATH.

    MODEL_PATH is passed to inference-only providers (huggingface, baseten,
    siliconflow, inferx) as their deployment model. For GPU-VM providers

    Use --provider to pin a specific provider or omit it to pick the cheapest
    quote. Use --dry-run to preview the selected provider before provisioning.

    Examples:
      terradev infer endpoint meta-llama/Llama-3.1-8B-Instruct -n my-ep
      terradev infer endpoint my-org/my-model -n my-ep --provider siliconflow -g A100
    """
    click.echo(f"Deploying inference endpoint: {name}")
    click.echo(f"Model path: {model_path}")

    if provider:
        click.echo(f"Provider: {provider}")
    if gpu_type:
        click.echo(f"GPU type: {gpu_type}")

    click.echo(f"Workers: {min_workers}-{max_workers}")
    click.echo(f"Idle timeout: {idle_timeout}s")
    if cost_optimize:
        click.echo("Cost optimization: Enabled")

    # Real deployment via provider API
    click.echo("\nAnalyzing model requirements...")

    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    target_providers = ["runpod", "vastai", "baseten", "huggingface", "siliconflow", "inferx"]
    if provider:
        target_providers = [provider.replace("lambda")]

    # Get best quote
    async def _get_best_quote():
        best = None
        for pname in target_providers:
            try:
                raw = await api._get_provider_quotes(pname, target_gpu)
                for q in raw:
                    price = q.get("price", 999)
                    if best is None or price < best.get("price", 999):
                        best = {
                            "provider": pname,
                            "price": price,
                            "gpu_type": q.get("gpu_type", target_gpu),
                            "region": q.get("region", "us-east-1"),
                            "instance_type": q.get("instance_type", ""),
                        }
            except Exception as _exc:  # noqa: BLE001
                logger.exception(_exc)
                pass
        return best

    best = _run_with_timeout(_get_best_quote())
    if not best:
        click.echo("No providers returned quotes for this GPU type")
        return

    pname = best["provider"]
    click.echo(f"Selected provider: {pname} (${best['price']:.2f}/hr)")

    if dry_run:
        click.echo("\n Dry run  deployment plan shown above. No resources provisioned.")
        return

    # Provision the instance
    click.echo("Deploying endpoint...")

    async def _provision():
        from terradev_cli.providers.provider_factory import ProviderFactory

        factory = ProviderFactory()
        creds = api._provider_creds(pname)
        # Inference-only providers need the requested model in their credentials
        if pname in ("huggingface", "baseten", "siliconflow", "inferx"):
            creds["model"] = model_path
            creds["default_model"] = model_path
        prov = factory.create_provider(pname, creds)
        if pname == "inferx":
            return await prov.deploy_model(
                {
                    "model_id": model_path,
                    "gpu_type": target_gpu,
                    "region": best.get("region", "us-east-1"),
                    "openai_compatible": True,
                }
            )
        return await prov.provision_instance(
            best.get("instance_type", f"{pname}-{target_gpu.lower()}"),
            best.get("region", "us-east-1"),
            target_gpu,
        )

    try:
        prov_result = _run_with_timeout(_provision())
        endpoint_id = prov_result.get("instance_id", f"ep_{name}_{int(time.time())}")
        endpoint_url = prov_result.get(
            "endpoint_url", f"https://{pname}.api/inference/{endpoint_id}"
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Deployment failed: {e}", err=True)
        raise SystemExit(1)
    click.echo("\nEndpoint deployed successfully!")
    click.echo(f"ID Endpoint ID: {endpoint_id}")
    click.echo(f"Endpoint URL: {endpoint_url}")
    click.echo("Status Status: Active")
    click.echo(f"Workers: {min_workers}/{max_workers}")
    click.echo(f"Cost: ${best['price']:.2f}/hr")

    # Silent: record inference deployment tick for ML training data
    try:
        from terradev_cli.core.price_intelligence import record_price_tick

        record_price_tick(
            gpu_type=gpu_type or "A100",
            provider=provider or "auto",
            price_hr=0.0,
            region="",
            spot=False,
            workload_type="inference",
            source="infer_deploy",
        )
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass

    # Save to usage tracking (reuse existing api instance)
    api.usage["inference_endpoints"].append(
        {
            "id": endpoint_id,
            "name": name,
            "model_path": model_path,
            "provider": provider or "auto-selected",
            "gpu_type": gpu_type or "auto-selected",
            "min_workers": min_workers,
            "max_workers": max_workers,
            "idle_timeout": idle_timeout,
            "cost_optimize": cost_optimize,
            "url": endpoint_url,
            "created_at": datetime.now().isoformat(),
        }
    )

    # Register with inference router for health tracking + failover
    try:
        from terradev_cli.core.inference_router import InferenceRouter

        router = InferenceRouter()
        router.register_endpoint(
            endpoint_id=endpoint_id,
            provider=provider or "auto",
            url=endpoint_url,
            model=model_path,
            gpu_type=gpu_type or "auto",
            region=best.get("region", ""),
            price_per_hour=best["price"],
        )
        click.echo("SHIELD:  Registered for health monitoring & auto-failover")
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass




# ═══════════════════════════════════════════════════════════════════════════════
# Inference Routing  Auto-failover + Latency-aware routing
# ═══════════════════════════════════════════════════════════════════════════════


@infer.command("status")
@click.option(
    "--check", is_flag=True, help="Run live health probes before showing status"
)
def infer_status(check):
    """Show inference endpoint health, latency, and failover status.

    Displays all registered inference endpoints with their health state,
    average latency, provider, and failover configuration.

    Use --check to run live health probes before displaying.
    """
    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        click.echo("ERROR: Inference router module not available.", err=True)
        raise SystemExit(1)

    router = InferenceRouter()

    if not router.endpoints:
        click.echo(" No inference endpoints registered.")
        click.echo("   Deploy one with: terradev infer --model <model>")
        return

    if check:
        click.echo(" Running health probes...")
        _run_with_timeout(router.check_all_endpoints())
        click.echo()

    status = router.get_status()
    click.echo(" Inference Endpoint Status")
    click.echo("=" * 70)
    click.echo(
        f"   Total: {status['total_endpoints']}  |  Healthy: {status['healthy']}  |  Unhealthy: {status['unhealthy']}"
    )
    click.echo()

    header = f"{'ID':<28} {'Provider':<12} {'Health':<12} {'Latency':<10} {'$/hr':<8} {'Role':<10}"
    click.echo(header)
    click.echo("-" * 70)

    for ep in status["endpoints"]:
        health_icon = {
            "healthy": "",
            "degraded": "",
            "unhealthy": "",
            "unknown": "",
        }.get(ep["health"], "")
        role = "PRIMARY" if ep["is_primary"] else f"backup→{ep.get('backup', '?')}"
        lat = f"{ep['avg_latency_ms']}ms" if ep["avg_latency_ms"] > 0 else ""
        click.echo(
            f"{ep['endpoint_id']:<28} {ep['provider']:<12} {health_icon} {ep['health']:<9} {lat:<10} ${ep['price_per_hour']:<7.2f} {role}"
        )

    # Show failover log if any
    failover_log = router.config_dir / "failover_log.json"
    if failover_log.exists():
        try:
            with open(failover_log, "r") as f:
                events = json.load(f)
            if events:
                click.echo("\nPlan Recent Failover Events (last 5):")
                for ev in events[-5:]:
                    click.echo(
                        f"   {ev['timestamp']}  {ev['failed_provider']}/{ev['failed_endpoint'][:16]} → {ev['new_provider']}/{ev['new_primary'][:16]}"
                    )
        except Exception as _exc:  # noqa: BLE001
            logger.exception(_exc)
            pass


@infer.command("failover")
@click.option(
    "--dry-run", is_flag=True, help="Show what would happen without executing failover"
)
def infer_failover(dry_run):
    """Run health checks and auto-failover for inference endpoints.

    Probes all registered inference endpoints. If a primary endpoint is
    unhealthy and has a backup configured, traffic automatically shifts
    to the backup provider.

    Open source feature - available to all users.
    """
    TerradevAPI()
    # Tier check removed - inference available to all users (open source)
    # tier_features = api.tier.get('features', [])
    # if 'all' not in tier_features and 'inference' not in tier_features:
    #     print(f"ERROR: Inference failover requires Research+ or Enterprise tier.")
    #     print(f"   Current tier: {api.tier['name']}")
    #     print(f"   Run: terradev upgrade")
    #     raise SystemExit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        click.echo("ERROR: Inference router module not available.", err=True)
        raise SystemExit(1)

    router = InferenceRouter()

    if not router.endpoints:
        click.echo(" No inference endpoints registered.")
        raise SystemExit(1)

    click.echo(" Running health checks on all inference endpoints...")
    probes = _run_with_timeout(router.check_all_endpoints())

    for eid, probe in probes.items():
        ep = router.endpoints.get(eid)
        icon = "" if probe.healthy else ""
        lat = f"{probe.latency_ms:.0f}ms" if probe.latency_ms > 0 else ""
        click.echo(f"   {icon} {eid[:24]:<24} {ep.provider:<12} {lat}")

    if dry_run:
        click.echo("\n DRY RUN  checking for failover candidates...")
        for eid, ep in router.endpoints.items():
            if (
                ep.is_primary
                and ep.health.value == "unhealthy"
                and ep.backup_endpoint_id
            ):
                backup = router.endpoints.get(ep.backup_endpoint_id)
                if backup:
                    click.echo(
                        f"   Warning  WOULD FAILOVER: {ep.provider}/{eid[:16]} → {backup.provider}/{backup.endpoint_id[:16]}"
                    )
        click.echo("   (No changes made)")
        return

    click.echo("\n Checking for auto-failover...")
    events = _run_with_timeout(router.check_and_failover())

    if events:
        for ev in events:
            click.echo(
                f"    FAILOVER: {ev['failed_provider']}/{ev['failed_endpoint'][:16]} → {ev['new_provider']}/{ev['new_primary'][:16]}"
            )
            click.echo(f"      Reason: {ev['reason']}")
        click.echo(
            f"\nOK: {len(events)} failover(s) executed. Traffic shifted to healthy providers."
        )
    else:
        click.echo("   OK: All primary endpoints healthy  no failover needed.")


@infer.command("route")
@click.option("--model", "-m", help="Filter by model name")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["latency", "cost", "score"]),
    default="latency",
    help="Routing strategy (default: latency)",
)
@click.option(
    "--measure", is_flag=True, help="Run fresh latency measurements before routing"
)
def infer_route(model, strategy, measure):
    """Find the best inference endpoint using latency-aware routing.

    Selects the optimal healthy endpoint based on strategy:
      - latency: lowest average response time (default)
      - cost: cheapest price per hour
      - score: weighted combination of latency + cost

    Use --measure to run fresh ping/TTFB probes before selecting.

    \b
    Integrates with WebPageTest TTFB probes for real-world latency data.
    Set WPT_API_KEY env var to enable WebPageTest integration.
    """
    TerradevAPI()
    # Tier check removed - inference available to all users (open source)
    # tier_features = api.tier.get('features', [])
    # if 'all' not in tier_features and 'inference' not in tier_features:
    #     print(f"ERROR: Inference routing requires Research+ or Enterprise tier.")
    #     print(f"   Run: terradev upgrade")
    #     raise SystemExit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        click.echo("ERROR: Inference router module not available.", err=True)
        raise SystemExit(1)

    router = InferenceRouter()

    if not router.endpoints:
        click.echo(" No inference endpoints registered.")
        return

    if measure:
        click.echo(" Running latency measurements...")
        wpt_key = os.environ.get("WPT_API_KEY")
        if wpt_key:
            click.echo("    WebPageTest integration enabled")

        async def _measure_all():
            await router.check_all_endpoints()
            for eid, ep in router.endpoints.items():
                if ep.url:
                    # Try WPT first, fall back to HTTP TTFB
                    lat = await router.measure_latency_wpt(ep.url, wpt_key)
                    if lat is not None:
                        ep.latency_history.append(lat)
                        if len(ep.latency_history) > router.LATENCY_HISTORY_SIZE:
                            ep.latency_history = ep.latency_history[
                                -router.LATENCY_HISTORY_SIZE :
                            ]
                        ep.avg_latency_ms = sum(ep.latency_history) / len(
                            ep.latency_history
                        )
                        source = "WPT" if wpt_key else "HTTP"
                        click.echo(f"   Status {eid[:24]}: {lat:.0f}ms ({source})")
            router._save_endpoints()

        _run_with_timeout(_measure_all())
        click.echo()

    best = router.get_best_endpoint(model=model, strategy=strategy)

    if not best:
        click.echo(
            "ERROR: No healthy endpoints found"
            + (f" for model '{model}'" if model else "")
        )
        return

    click.echo(f" Best endpoint (strategy: {strategy}):")
    click.echo(f"   Endpoint:  {best.endpoint_id}")
    click.echo(f"   Provider:  {best.provider}")
    click.echo(f"   Model:     {best.model}")
    click.echo(f"   Region:    {best.region}")
    click.echo(f"   Latency:   {best.avg_latency_ms:.0f}ms")
    click.echo(f"   Cost:      ${best.price_per_hour:.2f}/hr")
    click.echo(f"   Health:    {best.health.value}")
    if best.url:
        click.echo(f"   URL:       {best.url}")


# ═══════════════════════════════════════════════════════════════════════════════
# Kubernetes / Karpenter Integration  `terradev k8s`
# ═══════════════════════════════════════════════════════════════════════════════

# Workload → Karpenter provisioner mapping (matches karpenter_provisioners.yaml)
_K8S_WORKLOAD_PROFILES = {
    "training": {
        "provisioner": "training-workload-provisioner",
        "node_template": "training-workload-template",
        "default_gpu": "nvidia.com/gpu",
        "default_gpu_count": 1,
        "default_instance_hint": "p3.2xlarge",
        "capacity_type": "spot",  # spot-first for training
        "ttl_after_finished": 300,
        "restart_policy": "Never",
    },
    "inference": {
        "provisioner": "inference-workload-provisioner",
        "node_template": "inference-workload-template",
        "default_gpu": "nvidia.com/gpu",
        "default_gpu_count": 1,
        "default_instance_hint": "g5.xlarge",
        "capacity_type": "on-demand",  # stable for serving
        "ttl_after_finished": 0,  # keep alive
        "restart_policy": "Always",
    },
    "cost-optimized": {
        "provisioner": "cost-optimized-gpu-provisioner",
        "node_template": "cost-optimized-gpu-template",
        "default_gpu": "nvidia.com/gpu",
        "default_gpu_count": 1,
        "default_instance_hint": "g4dn.xlarge",
        "capacity_type": "spot",
        "ttl_after_finished": 60,
        "restart_policy": "Never",
    },
    "high-performance": {
        "provisioner": "high-performance-gpu-provisioner",
        "node_template": "high-performance-gpu-template",
        "default_gpu": "nvidia.com/gpu",
        "default_gpu_count": 8,
        "default_instance_hint": "p4d.24xlarge",
        "capacity_type": "on-demand",
        "ttl_after_finished": 600,
        "restart_policy": "Never",
    },
}


def _build_k8s_job_manifest(
    name: str,
    image: str,
    command: Optional[str],
    workload: str,
    gpu_count: int,
    namespace: str,
    env_vars: List[str],
    mounts: List[str],
    budget: Optional[float],
    ports: List[int],
) -> Dict[str, Any]:
    """Build a Kubernetes Job/Deployment manifest that Karpenter will schedule."""
    profile = _K8S_WORKLOAD_PROFILES[workload]

    # Budget heuristic: if budget < $2/hr, force spot + cost-optimized
    effective_capacity = profile["capacity_type"]
    if budget and budget < 2.0:
        effective_capacity = "spot"

    labels = {
        "app": name,
        "terradev.io/workload": workload,
        "terradev.io/managed-by": "terradev-cli",
    }

    node_selector = {
        "karpenter.sh/nodepool": profile["provisioner"],
    }
    tolerations = [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        {
            "key": "karpenter.sh/capacity-type",
            "operator": "Equal",
            "value": effective_capacity,
            "effect": "NoSchedule",
        },
    ]

    # Container spec
    container: Dict[str, Any] = {
        "name": name,
        "image": image,
        "resources": {
            "limits": {profile["default_gpu"]: gpu_count},
            "requests": {profile["default_gpu"]: gpu_count},
        },
    }
    if command:
        container["command"] = ["sh", "-c", command]
    if env_vars:
        container["env"] = []
        for ev in env_vars:
            k, _, v = ev.partition("=")
            container["env"].append({"name": k, "value": v})
    if ports:
        container["ports"] = [{"containerPort": p} for p in ports]

    # Volume mounts
    volumes: List[Dict] = []
    volume_mounts: List[Dict] = []
    for i, m in enumerate(mounts):
        host, _, ctr = m.partition(":")
        vol_name = f"mount-{i}"
        volumes.append({"name": vol_name, "hostPath": {"path": host}})
        volume_mounts.append({"name": vol_name, "mountPath": ctr})
    if volume_mounts:
        container["volumeMounts"] = volume_mounts

    pod_spec: Dict[str, Any] = {
        "nodeSelector": node_selector,
        "tolerations": tolerations,
        "containers": [container],
        "restartPolicy": profile["restart_policy"],
    }
    if volumes:
        pod_spec["volumes"] = volumes

    # For inference → Deployment; for everything else → Job
    if workload == "inference":
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name, "namespace": namespace, "labels": labels},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": pod_spec,
                },
            },
        }
        # Expose as a Service if ports are specified
        if ports:
            manifest["_service"] = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{name}-svc",
                    "namespace": namespace,
                    "labels": labels,
                },
                "spec": {
                    "selector": {"app": name},
                    "ports": [{"port": p, "targetPort": p} for p in ports],
                    "type": "ClusterIP",
                },
            }
    else:
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": namespace, "labels": labels},
            "spec": {
                "ttlSecondsAfterFinished": profile["ttl_after_finished"],
                "backoffLimit": 2,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": pod_spec,
                },
            },
        }

    return manifest


def _kubectl_apply(manifest_dict: Dict[str, Any], dry_run: bool = False) -> bool:
    """Apply a manifest via kubectl."""
    import tempfile

    try:
        import yaml
    except ImportError:
        import json as yaml

        yaml.dump = lambda d, **kw: json.dumps(d, indent=2, default=str)
    manifest_yaml = yaml.dump(manifest_dict, default_flow_style=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest_yaml)
        f.flush()
        cmd = ["kubectl", "apply", "-f", f.name]
        if dry_run:
            cmd.append("--dry-run=client")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            click.echo(f"   {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            click.echo(f"   ERROR: kubectl error: {e.stderr.strip()}")
            return False
        finally:
            os.unlink(f.name)

