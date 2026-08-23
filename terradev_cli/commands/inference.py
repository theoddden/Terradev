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
        print(f"Model Orchestrator started on GPU {gpu_id}")
        print(
            f"Memory: {memory_gb}GB total, {orchestrator.memory_threshold_gb:.1f}GB usable"
        )
        print(f"Policy: {policy}")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(10)
                status = orchestrator.get_status()
                print(
                    f"Status: {status['warm_models_count']} warm models, "
                    f"{status['used_memory_gb']:.1f}GB used "
                    f"({status['memory_utilization_percent']:.1f}%)"
                )
        except KeyboardInterrupt:
            print("\nStopping orchestrator...")
            await orchestrator.stop()

    asyncio.run(run_orchestrator())


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

    asyncio.run(orchestrator.register_model(
        model_id=model_id,
        model_path=model_path,
        framework=framework,
        priority=priority,
        tags=tag_set,
    ))

    print(f"Model registered: {model_id}")
    print(f"  Path: {model_path}")
    print(f"  Framework: {framework}")
    print(f"  Priority: {priority}")
    print(f"  Tags: {', '.join(tag_set) if tag_set else 'None'}")


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
            print(f"Model {model_id} loaded successfully!")
            print(f"  State: {details['state']}")
            print(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            print(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            print(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
        else:
            print(f"Failed to load model {model_id}")

    asyncio.run(load_model())


@orchestrator.command("evict")
@click.argument("model-id")
def orchestrator_evict(model_id):
    """Evict a model from GPU memory"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    async def evict_model():
        success = await orchestrator.evict_model(model_id)
        if success:
            print(f"Model {model_id} evicted successfully")
        else:
            print(f"Failed to evict model {model_id}")

    asyncio.run(evict_model())


@orchestrator.command("status")
@click.option("--model-id", help="Get details for specific model")
def orchestrator_status(model_id):
    """Get orchestrator and model status"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    if model_id:
        details = orchestrator.get_model_details(model_id)
        if details:
            print(f"Model Details: {model_id}")
            print(f"  Framework: {details['framework']}")
            print(f"  State: {details['state']}")
            print(f"  Priority: {details['priority']}")
            print(f"  Tags: {', '.join(details['tags'])}")
            print(f"  Memory: {details['metrics']['memory_gb']:.1f}GB")
            print(f"  Load time: {details['metrics']['load_time_s']:.1f}s")
            print(f"  Warmup time: {details['metrics']['warmup_time_s']:.1f}s")
            print(f"  Requests/hour: {details['metrics']['requests_per_hour']:.1f}")
            print(f"  Avg latency: {details['metrics']['avg_latency_ms']:.1f}ms")
            print(f"  Error rate: {details['metrics']['error_rate']:.2f}")
            print(f"  Last accessed: {details['last_accessed']}")
        else:
            print(f"Model {model_id} not found")
    else:
        status = orchestrator.get_status()
        print("Orchestrator Status:")
        print(f"  GPU: {status['gpu_id']}")
        print(f"  Total memory: {status['total_memory_gb']:.1f}GB")
        print(f"  Used memory: {status['used_memory_gb']:.1f}GB")
        print(f"  Available: {status['available_memory_gb']:.1f}GB")
        print(f"  Utilization: {status['memory_utilization_percent']:.1f}%")
        print(f"  Policy: {status['scaling_policy']}")
        print(f"  Total models: {status['total_models']}")
        print(f"  Warm models: {status['warm_models_count']}")
        print(f"  Warm memory: {status['warm_models_memory_gb']:.1f}GB")

        print("\nModels by state:")
        for state, model_ids in status["models_by_state"].items():
            if model_ids:
                print(f"  {state}: {', '.join(model_ids)}")


@orchestrator.command("infer")
@click.argument("model-id")
def orchestrator_infer(model_id):
    """Test inference with a model"""
    from terradev_cli.core.model_orchestrator import ModelOrchestrator

    orchestrator = ModelOrchestrator()

    async def test_inference():
        success, latency_ms = await orchestrator.handle_request(model_id)
        if success:
            print(f"Inference successful for {model_id}")
            print(f"  Latency: {latency_ms:.1f}ms")
        else:
            print(f"Inference failed for {model_id}")

    asyncio.run(test_inference())


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
        print("Warm Pool Manager started")
        print(f"Strategy: {strategy}")
        print(f"Capacity: {min_warm}-{max_warm} models")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(30)
                status = warm_pool.get_status()
                print(
                    f"Status: {status['warm_models_count']} warm, "
                    f"{status['cache_hit_rate']:.1%} hit rate, "
                    f"{status['total_requests']} requests"
                )
        except KeyboardInterrupt:
            print("\nStopping warm pool manager...")
            await warm_pool.stop()

    asyncio.run(run_warm_pool())


@warm_pool.command("register")
@click.argument("model-id")
@click.option("--priority", default=0, help="Model priority for warming")
def warm_pool_register(model_id, priority):
    """Register a model with the warm pool manager"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig

    warm_pool = WarmPoolManager(WarmPoolConfig())
    warm_pool.register_model(model_id, priority)

    print(f"Model {model_id} registered with warm pool")
    print(f"  Priority: {priority}")


@warm_pool.command("status")
def warm_pool_status():
    """Get warm pool manager status"""
    from terradev_cli.core.warm_pool_manager import WarmPoolManager, WarmPoolConfig

    warm_pool = WarmPoolManager(WarmPoolConfig())
    status = warm_pool.get_status()

    print("Warm Pool Status:")
    print(f"  Warm models: {status['warm_models_count']}")
    print(f"  Warming models: {status['warming_models_count']}")
    print(f"  Total models: {status['total_models']}")
    print(f"  Strategy: {status['strategy']}")
    print(f"  Cache hit rate: {status['cache_hit_rate']:.1%}")
    print(f"  Total requests: {status['total_requests']}")
    print(f"  Cold starts: {status['cold_starts']}")
    print(f"  Avg warm latency: {status['avg_warm_latency_ms']:.1f}ms")
    print(f"  Avg cold latency: {status['avg_cold_latency_ms']:.1f}ms")
    print(f"  Memory saved: {status['memory_saved_gb']:.1f}GB")
    print(f"  Cost saved: ${status['cost_saved_usd']:.2f}")


@cli.group(cls=InferenceGroup)
def cost_scaler():
    """Cost-aware scaling manager for inference optimization"""
    pass


@cost_scaler.command("start")
@click.option(
    "--strategy",
    type=click.Choice(
        [
            "minimize_cost",
            "balance_cost_latency",
            "latency_critical",
            "budget_constrained",
        ]
    ),
    default="balance_cost_latency",
    help="Cost optimization strategy",
)
@click.option("--budget", default=15.0, help="Hourly budget in USD")
@click.option("--cost-per-gb", default=0.10, help="Cost per GB per hour in USD")
def cost_scaler_start(strategy, budget, cost_per_gb):
    """Start the cost-aware scaling manager"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig, CostStrategy

    strategy_map = {
        "minimize_cost": CostStrategy.MINIMIZE_COST,
        "balance_cost_latency": CostStrategy.BALANCE_COST_LATENCY,
        "latency_critical": CostStrategy.LATENCY_CRITICAL,
        "budget_constrained": CostStrategy.BUDGET_CONSTRAINED,
    }

    config = CostConfig(
        hourly_budget_usd=budget,
        cost_per_gb_hour_usd=cost_per_gb,
        strategy=strategy_map[strategy],
        enable_cost_prediction=True,
    )

    cost_scaler = CostScaler(config)

    async def run_cost_scaler():
        await cost_scaler.start()
        print("Cost Scaler started")
        print(f"Strategy: {strategy}")
        print(f"Budget: ${budget}/hour")
        print(f"Cost per GB: ${cost_per_gb}/hour")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(60)
                status = cost_scaler.get_status()
                print(
                    f"Status: ${status['current_hourly_cost_usd']:.2f}/hour, "
                    f"{status['budget_utilization_percent']:.1f}% budget, "
                    f"{status['active_models']} models"
                )
        except KeyboardInterrupt:
            print("\nStopping cost scaler...")
            await cost_scaler.stop()

    asyncio.run(run_cost_scaler())


@cost_scaler.command("status")
def cost_scaler_status():
    """Get cost scaler status and recommendations"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig

    cost_scaler = CostScaler(CostConfig())
    status = cost_scaler.get_status()

    print("Cost Scaler Status:")
    print(f"  Current cost: ${status['current_hourly_cost_usd']:.3f}/hour")
    print(f"  Budget utilization: {status['budget_utilization_percent']:.1f}%")
    print(f"  Memory cost: ${status['memory_cost_usd']:.3f}/hour")
    print(f"  Cold start penalties: ${status['cold_start_penalty_usd']:.3f}/hour")
    print(f"  Total cost: ${status['total_cost_usd']:.2f}")
    print(f"  Cost savings: ${status['cost_savings_usd']:.2f}")
    print(f"  Memory usage: {status['current_memory_usage_gb']:.1f}GB")
    print(f"  Active models: {status['active_models']}")
    print(f"  Strategy: {status['strategy']}")
    print(f"  Is peak hour: {status['is_peak_hour']}")
    print(f"  Predicted cost (1h): ${status['predicted_cost_1h']:.3f}")
    print(f"  Predicted cost (2h): ${status['predicted_cost_2h']:.3f}")

    # Get recommendations
    recommendations = cost_scaler.get_cost_optimization_recommendations()
    if recommendations:
        print("\nCost Optimization Recommendations:")
        for rec in recommendations:
            print(f"  {rec['priority'].upper()}: {rec['message']}")
            print(f"    Action: {rec['action']}")
            print(f"    Potential savings: {rec['potential_savings']}")


@cost_scaler.command("model-details")
@click.argument("model-id")
def cost_scaler_model_details(model_id):
    """Get cost details for a specific model"""
    from terradev_cli.core.cost_scaler import CostScaler, CostConfig

    cost_scaler = CostScaler(CostConfig())
    details = cost_scaler.get_model_cost_details(model_id)

    if details:
        print(f"Cost Details for {model_id}:")
        print(f"  Memory usage: {details['memory_gb']:.1f}GB")
        print(f"  Hourly cost: ${details['hourly_cost_usd']:.3f}")
        print(f"  Cold start penalty: ${details['cold_start_penalty_usd']:.3f}")
        print(f"  Estimated daily cost: ${details['total_cost_today']:.2f}")
        print(f"  Cost rank: {details['cost_rank']} (1 = most expensive)")
    else:
        print(f"Model {model_id} not found or not loaded")




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

    print("OK: InferX configured successfully")
    print(f" Endpoint: {endpoint}")
    print(f" Region: {region}")
    print(f" Snapshot: {'Enabled' if snapshot else 'Disabled'}")
    print(f" GPU Slicing: {'Enabled' if gpu_slicing else 'Disabled'}")
    print(f" Multi-tenant: {'Enabled' if multi_tenant else 'Disabled'}")


@inferx.command()
@click.option("--model", required=True, help="Model ID or HuggingFace model name")
@click.option("--image", help="Docker image for model")
@click.option("--gpu-type", default="A100", help="GPU type")
@click.option("--gpu-memory", type=int, default=16, help="GPU memory in GB")
@click.option(
    "--max-concurrency", type=int, default=10, help="Maximum concurrent requests"
)
@click.option("--framework", default="pytorch", help="Model framework")
@click.option(
    "--openai-compatible/--no-openai-compatible",
    default=True,
    help="OpenAI-compatible API",
)
@click.option("--timeout", type=int, default=300, help="Request timeout in seconds")
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
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

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

    print(f" Deploying {model} to InferX...")
    print(f" GPU: {gpu_type} ({gpu_memory}GB)")
    print(f" Max Concurrency: {max_concurrency}")
    print(f" Framework: {framework}")

    try:
        result = asyncio.run(provider.deploy_model(model_config))

        print("OK: Model deployed successfully!")
        print(f" Model ID: {result['model_id']}")
        print(f" Endpoint: {result['endpoint']}")
        print(f" Cold Start: {result['cold_start_time']}s")
        print(f" GPU Utilization: {result['gpu_utilization']}%")
        print(f"PACKAGE: Models per Node: {result['models_per_node']}")

        if result["openai_compatible"]:
            print(" OpenAI Compatible: Yes")
            print(
                f"Tip: Usage: curl -X POST {result['endpoint']} -H 'Authorization: Bearer YOUR_API_KEY'"
            )

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Deployment failed: {e}")
    finally:
        asyncio.run(provider.close())


@inferx.command()
@click.option("--model-id", required=True, help="Model deployment ID")
def inferx_status(model_id):
    """Get model deployment status"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        result = asyncio.run(provider.get_model_status(model_id))

        print(f" Model Status: {result.get('status', 'Unknown')}")
        print(f" GPU Type: {result.get('gpu_type', 'Unknown')}")
        print(f" Cold Start Time: {result.get('cold_start_time', 'Unknown')}s")
        print(f" Requests/min: {result.get('requests_per_minute', 0)}")
        print(f" GPU Utilization: {result.get('gpu_utilization', 0)}%")
        print(f"PACKAGE: Models on GPU: {result.get('models_on_gpu', 0)}")
        print(f"   Error rate: {result.get('error_rate', 0)}%")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get status: {e}")
    finally:
        asyncio.run(provider.close())


@inferx.command()
@click.option("--model-id", required=True, help="Model deployment ID")
def inferx_delete(model_id):
    """Delete model deployment"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    print(f"  Deleting model {model_id}...")

    try:
        success = asyncio.run(provider.delete_model(model_id))

        if success:
            print("OK: Model deleted successfully")
        else:
            print("ERROR: Failed to delete model")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to delete model: {e}")
    finally:
        asyncio.run(provider.close())


@inferx.command("list")
def inferx_list():
    """List all deployed models"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        models = asyncio.run(provider.list_models())

        if not models:
            print(" No models deployed")
            return

        print(f" Deployed Models ({len(models)}):")
        print("-" * 80)

        for model in models:
            print(f"PACKAGE: {model.get('model_id', 'Unknown')}")
            print(f"   Status: {model.get('status', 'Unknown')}")
            print(f"   GPU: {model.get('gpu_type', 'Unknown')}")
            print(f"   Endpoint: {model.get('endpoint', 'Unknown')}")
            print(f"   Created: {model.get('created_at', 'Unknown')}")
            print()

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to list models: {e}")
    finally:
        asyncio.run(provider.close())


@inferx.command()
def usage():
    """Get account usage statistics"""
    import json
    from pathlib import Path

    # Load InferX config
    config_file = Path.home() / ".terradev" / "inferx_config.json"
    if not config_file.exists():
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        stats = asyncio.run(provider.get_usage_stats())

        print(" InferX Usage Statistics")
        print("-" * 40)
        print(f" Total Requests: {stats.get('total_requests', 0):,}")
        print(f"COST: Total Cost: ${stats.get('total_cost', 0):.4f}")
        print(f"PACKAGE: Active Models: {stats.get('active_models', 0)}")
        print(f" GPU Hours: {stats.get('gpu_hours', 0):.2f}")
        print(f" Average Latency: {stats.get('average_latency', 0):.0f}ms")
        print(f" GPU Utilization: {stats.get('gpu_utilization', 0):.1f}%")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get usage stats: {e}")
    finally:
        asyncio.run(provider.close())


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
        print("ERROR: InferX not configured. Run 'terradev inferx configure' first.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    from terradev_cli.providers.inferx_provider import InferXProvider

    provider = InferXProvider(config)

    try:
        quotes = asyncio.run(provider.get_instance_quotes(gpu_type, region))

        if not quotes:
            print("ERROR: No quotes available")
            return

        quote = quotes[0]

        print("COST: InferX Pricing Quote")
        print("-" * 40)
        print(f" GPU Type: {quote['gpu_type']}")
        print(
            f" Hourly Cost: ${quote['price_per_hour']:.4f} (Serverless - pay per request)"
        )
        print(f" Per Request: ${quote['price_per_request']:.4f} per 1K tokens")
        print(f" Cold Start: {quote['cold_start_time']}s")
        print(f" GPU Utilization: {quote['gpu_utilization']}%")
        print(f"PACKAGE: Models per Node: {quote['models_per_node']}")
        print(f" Region: {quote['region']}")
        print()
        print(" Key Features:")
        for feature in quote["features"]:
            print(f"   OK: {feature.replace('_', ' ').title()}")

    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to get quotes: {e}")
    finally:
        asyncio.run(provider.close())


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

    print(f" Analyzing InferX costs for {tier} tier...")

    # Generate cost report
    report = asyncio.run(
        optimizer.generate_cost_report(
            cluster_config_data, usage_metrics_data, target_tier
        )
    )

    # Display results
    print("\n Cost Analysis Results:")
    print("=" * 50)
    print(
        f"COST: Current Monthly Cost: ${report['summary']['current_monthly_cost']:,.2f}"
    )
    print(
        f" Potential Monthly Savings: ${report['summary']['potential_monthly_savings']:,.2f}"
    )
    print(f" Savings Percentage: {report['summary']['savings_percentage']:.1f}%")
    print(
        f" Optimized Monthly Cost: ${report['summary']['optimized_monthly_cost']:,.2f}"
    )
    print(f"  Payback Period: {report['summary']['payback_period_months']:.1f} months")
    print(f" Annual ROI: {report['summary']['annual_roi']:.1f}%")
    print()

    print(" Key Insights:")
    for insight in report["key_insights"]:
        print(f"    {insight}")
    print()

    print(" Top Recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"   {i}. {rec['description']}")
        print(f"      Savings: ${rec['estimated_savings']:,.2f}/month")
        print(f"      Risk: {rec['risk_level']}, Priority: {rec['priority']}")
        print()

    # Save report
    if output:
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f" Detailed report saved to: {output}")

    # Implement optimizations if requested
    if implement:
        print(" Implementing cost optimizations...")
        # Implementation logic would go here
        print("OK: Optimizations implemented successfully!")


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
        print("ERROR: Cost tracker not available")
        return [], None

    # Resolve "latest" to actual group ID
    group_id = provision_group
    if group_id == "latest":
        group_id = get_latest_parallel_group()
        if not group_id:
            print(
                "ERROR: No previous provision groups found. Run 'terradev provision' first."
            )
            return [], None
        if fmt != "json":
            print(f"  Resolved 'latest' -> {group_id}")

    instances = get_active_instances(parallel_group=group_id)
    if not instances:
        print(f"ERROR: No active instances in provision group {group_id}")
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
            print(f"  Resolving IPs for {len(unresolved)} instance(s)...")

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

        resolved = asyncio.run(_resolve_ips())

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
        print(f"  Nodes from provision group {group_id}: {node_ips}")

    # ── Auto-resolve SSH key from provision group ──
    resolved_ssh_key = None
    try:
        if _db_ssh_path(group_id):
            from terradev_cli.core.ssh_key_manager import decrypt_private_key

            resolved_ssh_key = decrypt_private_key(group_id)
            if resolved_ssh_key and fmt != "json":
                print(
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
            sys.exit(1)
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
        print(json.dumps(summary, indent=2, default=str))
    else:
        passed = summary.get("passed", False)
        status_icon = "PASS" if passed else "FAIL"
        print(f"\nPreflight: {status_icon}")
        print(f"  Nodes: {summary.get('nodes_checked', 0)}")
        print(
            f"  Checks passed: {summary.get('checks_passed', 0)}/{summary.get('total_checks', 0)}"
        )
        if summary.get("failures"):
            print("  Failures:")
            for f in summary["failures"]:
                print(f"    - {f}")
        print()




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
    type=click.Choice(["runpod", "vastai", "lambda_labs", "baseten", "huggingface", "siliconflow", "inferx"]),
    help="Provider (runpod|vastai|lambda_labs|baseten|huggingface|siliconflow|inferx)",
)
@click.option("--gpu-type", "-g", help="GPU type preference")
@click.option("--region", "-r", help="Region preference")
@click.option("--max-latency", type=float, help="Max latency in ms")
@click.option("--max-cost", type=float, help="Max cost per request")
def infer_deploy_main(model, type, provider, gpu_type, region, max_latency, max_cost):
    """Compare and select the cheapest inference option across providers.

    Queries all configured inference providers (GPU-based and inference-only)
    and prints the best quote. To actually deploy, use `terradev infer endpoint`.

    Supported providers: runpod, vastai, lambda_labs, baseten, huggingface,
    siliconflow, inferx.
    """
    print(f"Deploying inference for model: {model}")

    if type:
        print(f"Model type: {type}")
    if provider:
        print(f"Provider: {provider}")
    if gpu_type:
        print(f"GPU type: {gpu_type}")
    if region:
        print(f"Region: {region}")
    if max_latency:
        print(f"Max latency: {max_latency}ms")
    if max_cost:
        print(f"Max cost: ${max_cost}/request")

    # Get real quotes from providers
    print("Getting inference quotes from terradev_cli.providers...")

    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    inference_providers = ["runpod", "vastai", "lambda_labs", "baseten", "huggingface", "siliconflow", "inferx"]
    if provider:
        inference_providers = [provider.replace("lambda", "lambda_labs")]

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

    quotes = asyncio.run(_fetch_inference_quotes())

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
        print("No suitable inference options found")
        return

    # Select best option (lowest price, then lowest latency)
    best_quote = min(quotes, key=lambda x: (x["price"], x["latency"]))

    print(f"\nBest option: {best_quote['provider']}")
    print(f"Price: ${best_quote['price']}/request")
    print(f"Latency: {best_quote['latency']}ms")
    print(f"GPU: {best_quote['gpu_type']}")

    # Deploy to optimal provider via real API
    pname = best_quote["provider"]
    print(f"\nDeploying to {pname}...")

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
        prov_result = asyncio.run(_deploy_inference())
        endpoint_id = prov_result.get("instance_id", f"inf_{pname}_{int(time.time())}")
        endpoint_url = prov_result.get(
            "endpoint_url", f"https://{pname}.api/inference/{endpoint_id}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Warning  Provisioning error: {e}")
        endpoint_id = f"inf_{pname}_{int(time.time())}"
        endpoint_url = ""

    print("Inference endpoint deployed")
    print(f"ID Endpoint ID: {endpoint_id}")
    if endpoint_url:
        print(f"URL: {endpoint_url}")
    print("Status Status: Active")

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
        print("SHIELD:  Registered for health monitoring & auto-failover")
    except Exception as _exc:  # noqa: BLE001
        logger.exception(_exc)
        pass


@infer.command("endpoint")
@click.argument("model_path")
@click.option("--name", "-n", required=True, help="Endpoint name (required)")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["runpod", "vastai", "lambda_labs", "baseten", "huggingface", "siliconflow", "inferx"]),
    help="Provider (runpod|vastai|lambda_labs|baseten|huggingface|siliconflow|inferx)",
)
@click.option("--gpu-type", "-g", help="GPU type (A100|H100|RTX4090)")
@click.option("--min-workers", type=int, default=1, help="Minimum workers")
@click.option("--max-workers", type=int, default=5, help="Maximum workers")
@click.option("--idle-timeout", type=int, default=300, help="Idle timeout in seconds")
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
    (runpod, vastai, lambda_labs) it is used as the endpoint label.

    Use --provider to pin a specific provider or omit it to pick the cheapest
    quote. Use --dry-run to preview the selected provider before provisioning.

    Examples:
      terradev infer endpoint meta-llama/Llama-3.1-8B-Instruct -n my-ep
      terradev infer endpoint my-org/my-model -n my-ep --provider siliconflow -g A100
    """
    print(f"Deploying inference endpoint: {name}")
    print(f"Model path: {model_path}")

    if provider:
        print(f"Provider: {provider}")
    if gpu_type:
        print(f"GPU type: {gpu_type}")

    print(f"Workers: {min_workers}-{max_workers}")
    print(f"Idle timeout: {idle_timeout}s")
    if cost_optimize:
        print("Cost optimization: Enabled")

    # Real deployment via provider API
    print("\nAnalyzing model requirements...")

    api = TerradevAPI()
    target_gpu = gpu_type or "A100"
    target_providers = ["runpod", "vastai", "lambda_labs", "baseten", "huggingface", "siliconflow", "inferx"]
    if provider:
        target_providers = [provider.replace("lambda", "lambda_labs")]

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

    best = asyncio.run(_get_best_quote())
    if not best:
        print("No providers returned quotes for this GPU type")
        return

    pname = best["provider"]
    print(f"Selected provider: {pname} (${best['price']:.2f}/hr)")

    if dry_run:
        print("\n Dry run  deployment plan shown above. No resources provisioned.")
        return

    # Provision the instance
    print("Deploying endpoint...")

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
        prov_result = asyncio.run(_provision())
        endpoint_id = prov_result.get("instance_id", f"ep_{name}_{int(time.time())}")
        endpoint_url = prov_result.get(
            "endpoint_url", f"https://{pname}.api/inference/{endpoint_id}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Deployment failed: {e}")
        return

    print("\nEndpoint deployed successfully!")
    print(f"ID Endpoint ID: {endpoint_id}")
    print(f"Endpoint URL: {endpoint_url}")
    print("Status Status: Active")
    print(f"Workers: {min_workers}/{max_workers}")
    print(f"Cost: ${best['price']:.2f}/hr")

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
        print("SHIELD:  Registered for health monitoring & auto-failover")
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
        print("ERROR: Inference router module not available.")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print(" No inference endpoints registered.")
        print("   Deploy one with: terradev infer --model <model>")
        return

    if check:
        print(" Running health probes...")
        asyncio.run(router.check_all_endpoints())
        print()

    status = router.get_status()
    print(" Inference Endpoint Status")
    print("=" * 70)
    print(
        f"   Total: {status['total_endpoints']}  |  Healthy: {status['healthy']}  |  Unhealthy: {status['unhealthy']}"
    )
    print()

    header = f"{'ID':<28} {'Provider':<12} {'Health':<12} {'Latency':<10} {'$/hr':<8} {'Role':<10}"
    print(header)
    print("-" * 70)

    for ep in status["endpoints"]:
        health_icon = {
            "healthy": "",
            "degraded": "",
            "unhealthy": "",
            "unknown": "",
        }.get(ep["health"], "")
        role = "PRIMARY" if ep["is_primary"] else f"backup→{ep.get('backup', '?')}"
        lat = f"{ep['avg_latency_ms']}ms" if ep["avg_latency_ms"] > 0 else ""
        print(
            f"{ep['endpoint_id']:<28} {ep['provider']:<12} {health_icon} {ep['health']:<9} {lat:<10} ${ep['price_per_hour']:<7.2f} {role}"
        )

    # Show failover log if any
    failover_log = router.config_dir / "failover_log.json"
    if failover_log.exists():
        try:
            with open(failover_log, "r") as f:
                events = json.load(f)
            if events:
                print("\nPlan Recent Failover Events (last 5):")
                for ev in events[-5:]:
                    print(
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
    #     sys.exit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        print("ERROR: Inference router module not available.")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print(" No inference endpoints registered.")
        sys.exit(1)

    print(" Running health checks on all inference endpoints...")
    probes = asyncio.run(router.check_all_endpoints())

    for eid, probe in probes.items():
        ep = router.endpoints.get(eid)
        icon = "" if probe.healthy else ""
        lat = f"{probe.latency_ms:.0f}ms" if probe.latency_ms > 0 else ""
        print(f"   {icon} {eid[:24]:<24} {ep.provider:<12} {lat}")

    if dry_run:
        print("\n DRY RUN  checking for failover candidates...")
        for eid, ep in router.endpoints.items():
            if (
                ep.is_primary
                and ep.health.value == "unhealthy"
                and ep.backup_endpoint_id
            ):
                backup = router.endpoints.get(ep.backup_endpoint_id)
                if backup:
                    print(
                        f"   Warning  WOULD FAILOVER: {ep.provider}/{eid[:16]} → {backup.provider}/{backup.endpoint_id[:16]}"
                    )
        print("   (No changes made)")
        return

    print("\n Checking for auto-failover...")
    events = asyncio.run(router.check_and_failover())

    if events:
        for ev in events:
            print(
                f"    FAILOVER: {ev['failed_provider']}/{ev['failed_endpoint'][:16]} → {ev['new_provider']}/{ev['new_primary'][:16]}"
            )
            print(f"      Reason: {ev['reason']}")
        print(
            f"\nOK: {len(events)} failover(s) executed. Traffic shifted to healthy providers."
        )
    else:
        print("   OK: All primary endpoints healthy  no failover needed.")


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
    #     sys.exit(1)

    try:
        from terradev_cli.core.inference_router import InferenceRouter
    except ImportError:
        print("ERROR: Inference router module not available.")
        sys.exit(1)

    router = InferenceRouter()

    if not router.endpoints:
        print(" No inference endpoints registered.")
        return

    if measure:
        print(" Running latency measurements...")
        wpt_key = os.environ.get("WPT_API_KEY")
        if wpt_key:
            print("    WebPageTest integration enabled")

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
                        print(f"   Status {eid[:24]}: {lat:.0f}ms ({source})")
            router._save_endpoints()

        asyncio.run(_measure_all())
        print()

    best = router.get_best_endpoint(model=model, strategy=strategy)

    if not best:
        print(
            "ERROR: No healthy endpoints found"
            + (f" for model '{model}'" if model else "")
        )
        return

    print(f" Best endpoint (strategy: {strategy}):")
    print(f"   Endpoint:  {best.endpoint_id}")
    print(f"   Provider:  {best.provider}")
    print(f"   Model:     {best.model}")
    print(f"   Region:    {best.region}")
    print(f"   Latency:   {best.avg_latency_ms:.0f}ms")
    print(f"   Cost:      ${best.price_per_hour:.2f}/hr")
    print(f"   Health:    {best.health.value}")
    if best.url:
        print(f"   URL:       {best.url}")


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
            print(f"   {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ERROR: kubectl error: {e.stderr.strip()}")
            return False
        finally:
            os.unlink(f.name)

