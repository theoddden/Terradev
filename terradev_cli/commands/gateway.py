#!/usr/bin/env python3
"""Gateway command group and inference provider subcommands for Terradev."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

import click

from . import cli
from terradev_cli.commands._api import TerradevAPI
from terradev_cli.core.gateway_service import create_gateway_config, GatewayService
from terradev_cli.core.inference_router import InferenceRouter

logger = logging.getLogger(__name__)

INFERENCE_PROVIDERS = {"huggingface", "baseten", "siliconflow", "inferx"}


class GatewayCommand(click.Command):
    """Click command that catches unexpected runtime failures and exits cleanly."""

    def invoke(self, ctx):
        try:
            rv = super().invoke(ctx)
        except (click.ClickException, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            click.echo(f"ERROR: {exc}", err=True)
            raise SystemExit(1) from exc

        output = ctx.obj.get("terradev_output") if ctx.obj else None
        if output is not None and (rv is None or rv == 0):
            messages = getattr(output, "_messages", [])
            if any(m.level == "error" for m in messages):
                raise SystemExit(1)
        return rv


class GatewayGroup(click.Group):
    """Click group that uses GatewayCommand for leaf subcommands."""

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", GatewayCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", GatewayGroup)
        return super().group(*args, **kwargs)


async def _run_with_timeout(coro):
    try:
        return await asyncio.wait_for(coro, timeout=120)
    except asyncio.TimeoutError:
        click.echo("ERROR: Gateway operation timed out", err=True)
        raise SystemExit(1)


def _run_async(coro):
    return asyncio.run(_run_with_timeout(coro))


def _start_gateway_service(config: Dict[str, Any]) -> None:
    """Start the Terradev inference gateway server."""
    try:
        cfg = create_gateway_config(**config)
        gateway = GatewayService(cfg)
    except ImportError as e:
        click.echo(f"ERROR: {e}", err=True)
        click.echo("To install required dependencies:", err=True)
        click.echo("  pip install fastapi uvicorn", err=True)
        raise SystemExit(1)

    host = config["host"]
    port = config["port"]

    click.echo(f"\n{'='*70}")
    click.echo("TERRADEV INFERENCE GATEWAY")
    click.echo(f"{'='*70}")
    click.echo(f"Host: {host}:{port}")
    click.echo(f"OpenAI API: {'ENABLED' if config.get('enable_openai') else 'DISABLED'}")
    click.echo(f"Anthropic API: {'ENABLED' if config.get('enable_anthropic') else 'DISABLED'}")
    click.echo(f"Custom Workflows: {'ENABLED' if config.get('enable_custom') else 'DISABLED'}")
    click.echo(f"CORS: {'ENABLED' if config.get('enable_cors') else 'DISABLED'}")
    click.echo(f"Inference Router: {'ENABLED' if config.get('enable_inference_router') else 'DISABLED'}")
    click.echo(f"Max Concurrent Requests: {config['max_concurrent_requests']}")
    click.echo(f"Request Timeout: {config['request_timeout']}s")
    click.echo(f"Default Model: {config['default_model']}")
    click.echo(f"{'='*70}\n")
    click.echo("Starting gateway server...")
    click.echo(f"OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    click.echo(f"Anthropic endpoint: http://{host}:{port}/v1/messages")
    click.echo(f"Health check: http://{host}:{port}/health")
    click.echo(f"Gateway status: http://{host}:{port}/v1/gateway/status")
    click.echo("\nPress Ctrl+C to stop the server\n")

    try:
        gateway.run_sync()
    except KeyboardInterrupt:
        click.echo("\n\nGateway server stopped.")
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: Failed to start gateway server: {e}", err=True)
        raise SystemExit(1)


def _load_provider(provider: str, api: TerradevAPI, overrides: Optional[Dict[str, str]] = None):
    """Create a provider instance from stored credentials and CLI overrides."""
    from terradev_cli.providers.provider_factory import ProviderFactory

    creds = api._provider_creds(provider) or {}
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                creds[key] = value

    if not any(v for v in creds.values() if v and isinstance(v, str)):
        raise click.ClickException(
            f"No credentials configured for {provider}. "
            f"Run 'terradev gateway {provider} configure' first."
        )

    factory = ProviderFactory()
    return factory.create_provider(provider, creds)


def _register_endpoint(
    api: TerradevAPI,
    provider: str,
    result: Dict[str, Any],
    model: str,
    gpu_type: str,
    region: Optional[str],
) -> tuple:
    """Register a deployed inference endpoint with the router and usage store."""
    endpoint_id = (
        result.get("instance_id")
        or result.get("model_id")
        or result.get("id")
        or f"{provider}-{int(datetime.now().timestamp())}"
    )
    url = result.get("endpoint_url") or result.get("endpoint") or ""
    price = result.get("price_per_hour") or result.get("price") or 0.0

    try:
        price_per_hour = float(price)
    except (TypeError, ValueError):
        price_per_hour = 0.0

    router = InferenceRouter()
    router.register_endpoint(
        endpoint_id=endpoint_id,
        provider=provider,
        url=url,
        model=model,
        gpu_type=gpu_type,
        region=region or "",
        price_per_hour=price_per_hour,
    )

    if "inference_endpoints" not in api.usage:
        api.usage["inference_endpoints"] = []
    # Idempotent: replace any existing record with the same endpoint id.
    api.usage["inference_endpoints"] = [
        e for e in api.usage["inference_endpoints"] if e.get("id") != endpoint_id
    ]
    api.usage["inference_endpoints"].append(
        {
            "id": endpoint_id,
            "provider": provider,
            "model": model,
            "gpu_type": gpu_type,
            "region": region,
            "url": url,
            "price": price,
            "created_at": datetime.now().isoformat(),
        }
    )
    api.save_usage()
    return endpoint_id, url


class ProviderAdapter:
    """Base adapter for inference-provider subcommands."""

    def __init__(self, provider: str, api: TerradevAPI, overrides: Optional[Dict[str, str]] = None):
        self.provider = provider
        self.api = api
        self.instance = _load_provider(provider, api, overrides)

    async def deploy(self, model: str, gpu_type: str, region: Optional[str]) -> Dict[str, Any]:
        raise NotImplementedError

    async def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError

    async def list(self) -> List[Dict[str, Any]]:
        return await self.instance.list_instances()

    async def status(self, endpoint_id: str) -> Dict[str, Any]:
        return await self.instance.get_instance_status(endpoint_id)

    async def delete(self, endpoint_id: str) -> Dict[str, Any]:
        return await self.instance.terminate_instance(endpoint_id)

    async def models(self) -> List[Dict[str, Any]]:
        if hasattr(self.instance, "list_models"):
            return await self.instance.list_models()
        raise NotImplementedError(f"{self.provider} does not support listing models")


class HuggingFaceAdapter(ProviderAdapter):
    async def deploy(self, model: str, gpu_type: str, region: Optional[str]) -> Dict[str, Any]:
        self.instance.credentials["model"] = model
        info = self.instance.GPU_PRICING.get(gpu_type.upper())
        if not info:
            raise click.ClickException(f"Unsupported GPU type for HuggingFace: {gpu_type}")
        return await self.instance.provision_instance(
            info["instance_type"], region or "us-east-1", gpu_type
        )

    async def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        result = await self.instance.run_inference(model, {"inputs": prompt})
        if result.get("status") == "error":
            raise click.ClickException(result.get("error", "HuggingFace inference failed"))
        return json.dumps(result.get("result"), indent=2, default=str)

    async def models(self) -> List[Dict[str, Any]]:
        # HuggingFace endpoints are the closest equivalent to a model list
        return await self.instance.list_instances()


class BasetenAdapter(ProviderAdapter):
    async def deploy(self, model: str, gpu_type: str, region: Optional[str]) -> Dict[str, Any]:
        self.instance.credentials["model"] = model
        return await self.instance.provision_instance(
            f"baseten-{gpu_type.lower()}", region or "us-east-1", gpu_type
        )

    async def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        result = await self.instance.execute_command(model, prompt, False)
        if result.get("exit_code"):
            raise click.ClickException(result.get("output", "Baseten chat failed"))
        return str(result.get("output", ""))

    async def models(self) -> List[Dict[str, Any]]:
        return await self.instance.list_instances()


class SiliconFlowAdapter(ProviderAdapter):
    async def deploy(self, model: str, gpu_type: str, region: Optional[str]) -> Dict[str, Any]:
        self.instance.default_model = model
        self.instance.credentials["default_model"] = model
        return await self.instance.provision_instance(
            f"dedicated-{gpu_type.lower()}", region or "auto", gpu_type
        )

    async def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        result = await self.instance.chat_completion(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return json.dumps(result, indent=2, default=str)


class InferXAdapter(ProviderAdapter):
    async def deploy(self, model: str, gpu_type: str, region: Optional[str]) -> Dict[str, Any]:
        return await self.instance.deploy_model(
            {
                "model_id": model,
                "gpu_type": gpu_type,
                "region": region or "us-west-2",
                "max_concurrency": 10,
                "openai_compatible": True,
            }
        )

    async def chat(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        result = await self.instance.chat_completion(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return json.dumps(result, indent=2, default=str)


ADAPTERS = {
    "huggingface": HuggingFaceAdapter,
    "baseten": BasetenAdapter,
    "siliconflow": SiliconFlowAdapter,
    "inferx": InferXAdapter,
}


@cli.group(name="gateway", invoke_without_command=True, cls=GatewayGroup)
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind the gateway server")
@click.option("--port", "-p", default=8000, type=click.IntRange(1, 65535), help="Port for the gateway server")
@click.option("--openai", is_flag=True, default=True, help="Enable OpenAI-compatible endpoints")
@click.option("--no-openai", is_flag=True, help="Disable OpenAI-compatible endpoints")
@click.option("--anthropic", is_flag=True, default=True, help="Enable Anthropic-compatible endpoints")
@click.option("--no-anthropic", is_flag=True, help="Disable Anthropic-compatible endpoints")
@click.option("--custom", is_flag=True, default=True, help="Enable custom workflow endpoints")
@click.option("--no-custom", is_flag=True, help="Disable custom workflow endpoints")
@click.option("--max-concurrent", type=click.IntRange(1, 65535), default=100, help="Maximum concurrent requests")
@click.option("--timeout", type=click.IntRange(1, 3600), default=120, help="Request timeout in seconds")
@click.option("--cors", is_flag=True, default=True, help="Enable CORS")
@click.option("--no-cors", is_flag=True, help="Disable CORS")
@click.option("--cors-origins", multiple=True, help="CORS allowed origins")
@click.option("--model", default="meta-llama/Llama-3.1-70B-Instruct", help="Default model for inference")
@click.option("--no-inference-router", is_flag=True, help="Disable inference router integration")
@click.pass_context
def gateway(
    ctx,
    host,
    port,
    openai,
    no_openai,
    anthropic,
    no_anthropic,
    custom,
    no_custom,
    max_concurrent,
    timeout,
    cors,
    no_cors,
    cors_origins,
    model,
    no_inference_router,
):
    """Launch an API gateway for inference serving.

    Run without subcommands to start the gateway server. Use the provider
    subcommands to configure, deploy and manage inference-only endpoints.

    Supported inference providers:
      - huggingface   (HuggingFace Inference Endpoints)
      - baseten       (Baseten model deployments)
      - siliconflow   (SiliconFlow model API)
      - inferx        (InferX serverless inference)

    Typical workflow:
      1. terradev gateway <provider> configure --api-key <key>
      2. terradev gateway <provider> deploy --model <model> --gpu-type A100
      3. terradev gateway <provider> status <endpoint-id>
      4. terradev gateway <provider> chat --model <model> --prompt "Hello"

    \b
    Examples:
      terradev gateway
      terradev gateway --host 0.0.0.0 --port 8080
      terradev gateway --port 8080 serve
      terradev gateway huggingface configure --api-key $HF_TOKEN --namespace hf-user
      terradev gateway huggingface deploy --model meta-llama/Llama-3.1-8B-Instruct
    """
    if ctx.resilient_parsing:
        return

    ctx.ensure_object(dict)
    config = {
        "host": host,
        "port": port,
        "enable_openai": openai and not no_openai,
        "enable_anthropic": anthropic and not no_anthropic,
        "enable_custom": custom and not no_custom,
        "max_concurrent_requests": max_concurrent,
        "request_timeout": timeout,
        "enable_cors": cors and not no_cors,
        "cors_origins": list(cors_origins) if cors_origins else ["*"],
        "enable_inference_router": not no_inference_router,
        "default_model": model,
    }
    ctx.obj["gateway_config"] = config

    if ctx.invoked_subcommand is None:
        _start_gateway_service(config)


@gateway.command("serve")
def serve():
    """Start the API gateway server explicitly."""
    ctx = click.get_current_context()
    _start_gateway_service(ctx.obj["gateway_config"])


@gateway.command("status")
@click.option("--host", "-h", default="0.0.0.0", help="Gateway host")
@click.option("--port", "-p", default=8000, type=click.IntRange(1, 65535), help="Gateway port")
@click.pass_context
def gateway_status(ctx, host, port):
    """Show the running gateway server status."""
    url = f"http://{host}:{port}/v1/gateway/status"
    try:
        with urlopen(url, timeout=5) as resp:
            data = json.load(resp)
            click.echo(json.dumps(data, indent=2, default=str))
    except URLError as e:
        click.echo(f"ERROR: Gateway not reachable at {url}: {e}", err=True)
        click.echo("Start a gateway with: terradev gateway serve", err=True)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)


def _build_provider_group(provider: str):
    """Build a Click group for a single inference provider."""
    group = GatewayGroup(
        name=provider,
        help=f"{provider.title()} inference provider commands",
    )
    Adapter = ADAPTERS[provider]

    @group.command("configure")
    @click.option("--api-key", prompt=True, hide_input=True, help="API key / token")
    @click.option("--namespace", help="HuggingFace namespace (huggingface only)")
    @click.option("--region", help="Default region")
    @click.option("--default-model", help="Default model (siliconflow / inferx)")
    @click.option("--endpoint", help="InferX API endpoint (inferx only)")
    @click.pass_context
    def configure(ctx, api_key, namespace, region, default_model, endpoint):
        """Configure provider credentials.

        Required/optional fields by provider:
          - huggingface: --api-key and --namespace
          - baseten:     --api-key
          - siliconflow: --api-key, optional --default-model
          - inferx:      --api-key, optional --default-model and --endpoint
        """
        api = ctx.obj["api"]
        creds: Dict[str, str] = {"api_key": api_key or ""}

        if not api_key:
            raise click.ClickException("API key is required")

        if provider == "huggingface" and not namespace:
            raise click.ClickException("HuggingFace requires --namespace")

        if namespace:
            creds["namespace"] = namespace
        if region:
            creds["region"] = region
        if default_model:
            creds["default_model"] = default_model
        if endpoint:
            creds["api_endpoint"] = endpoint

        api._save_provider_creds(provider, creds)
        click.echo(f"OK: {provider} credentials saved.")

    @group.command("deploy")
    @click.option("--model", "-m", required=True, help="Model to deploy / serve")
    @click.option("--gpu-type", "-g", default="A100", help="GPU type")
    @click.option("--region", "-r", help="Region or vendor location")
    @click.pass_context
    def deploy(ctx, model, gpu_type, region):
        """Deploy an inference endpoint with this provider.

        Provisions the requested MODEL on the selected GPU and registers the
        resulting endpoint with the Terradev InferenceRouter for health/failover.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        result = _run_async(adapter.deploy(model, gpu_type, region))
        endpoint_id, url = _register_endpoint(api, provider, result, model, gpu_type, region)

        click.echo(f"OK: {provider} endpoint deployed")
        click.echo(f"  ID: {endpoint_id}")
        if url:
            click.echo(f"  URL: {url}")
        click.echo(f"  Status: {result.get('status', 'unknown')}")

    @group.command("list")
    @click.pass_context
    def list_cmd(ctx):
        """List deployed endpoints for this provider.

        Shows all active or recently provisioned endpoints/models.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        items = _run_async(adapter.list())
        if not items:
            click.echo(f"No {provider} endpoints found")
            return

        click.echo(f"{provider} endpoints:")
        for item in items:
            iid = item.get("instance_id") or item.get("id") or item.get("model_id") or "unknown"
            click.echo(f"  {iid}: {item.get('status', 'unknown')}")

    @group.command("status")
    @click.argument("endpoint-id")
    @click.pass_context
    def status_cmd(ctx, endpoint_id):
        """Get status of a deployed endpoint.

        ENDPOINT-ID can be the provider instance id, model id, or endpoint url.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        result = _run_async(adapter.status(endpoint_id))
        click.echo(json.dumps(result, indent=2, default=str))

    @group.command("delete")
    @click.argument("endpoint-id")
    @click.pass_context
    def delete_cmd(ctx, endpoint_id):
        """Delete/terminate a deployed endpoint.

        This operation is provider-dependent and may be irreversible.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        result = _run_async(adapter.delete(endpoint_id))
        click.echo(f"OK: {endpoint_id} deletion initiated")
        click.echo(json.dumps(result, indent=2, default=str))

    @group.command("chat")
    @click.option("--model", "-m", required=True, help="Model or endpoint ID to query")
    @click.option("--prompt", "-p", required=True, help="Prompt text")
    @click.option("--max-tokens", default=2048, type=click.IntRange(1, 8192), help="Maximum tokens")
    @click.option("--temperature", default=0.7, type=click.FloatRange(0.0, 2.0), help="Sampling temperature")
    @click.pass_context
    def chat_cmd(ctx, model, prompt, max_tokens, temperature):
        """Send a chat/prompt to this provider.

        MODEL can be a model id, endpoint id, or full model path depending on the
        provider. The provider must already have a deployed endpoint.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        response = _run_async(adapter.chat(model, prompt, max_tokens, temperature))
        click.echo(response)

    @group.command("models")
    @click.pass_context
    def models_cmd(ctx):
        """List available models or deployed endpoints for this provider.

        For providers without a public model catalog this lists your deployments.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        try:
            items = _run_async(adapter.models())
        except NotImplementedError as e:
            click.echo(f"ERROR: {e}", err=True)
            raise SystemExit(1)

        if not items:
            click.echo(f"No {provider} models found")
            return

        click.echo(f"{provider} models:")
        for item in items:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("model") or item.get("name") or item.get("model_id", "unknown")
                click.echo(f"  {mid}")
            else:
                click.echo(f"  {item}")

    return group


# Attach provider subcommand groups to the gateway group
for _provider in ADAPTERS:
    gateway.add_command(_build_provider_group(_provider))
