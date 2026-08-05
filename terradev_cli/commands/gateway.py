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


def _run_async(coro):
    return asyncio.run(coro)


def _start_gateway_service(config: Dict[str, Any]) -> None:
    """Start the Terradev inference gateway server."""
    try:
        cfg = create_gateway_config(**config)
        gateway = GatewayService(cfg)
    except ImportError as e:
        print(f"ERROR: {e}")
        print("To install required dependencies:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    host = config["host"]
    port = config["port"]

    print(f"\n{'='*70}")
    print("TERRADEV INFERENCE GATEWAY")
    print(f"{'='*70}")
    print(f"Host: {host}:{port}")
    print(f"OpenAI API: {'ENABLED' if config.get('enable_openai') else 'DISABLED'}")
    print(f"Anthropic API: {'ENABLED' if config.get('enable_anthropic') else 'DISABLED'}")
    print(f"Custom Workflows: {'ENABLED' if config.get('enable_custom') else 'DISABLED'}")
    print(f"CORS: {'ENABLED' if config.get('enable_cors') else 'DISABLED'}")
    print(f"Inference Router: {'ENABLED' if config.get('enable_inference_router') else 'DISABLED'}")
    print(f"Max Concurrent Requests: {config['max_concurrent_requests']}")
    print(f"Request Timeout: {config['request_timeout']}s")
    print(f"Default Model: {config['default_model']}")
    print(f"{'='*70}\n")
    print("Starting gateway server...")
    print(f"OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    print(f"Anthropic endpoint: http://{host}:{port}/v1/messages")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Gateway status: http://{host}:{port}/v1/gateway/status")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        gateway.run_sync()
    except KeyboardInterrupt:
        print("\n\nGateway server stopped.")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to start gateway server: {e}")
        sys.exit(1)


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

    router = InferenceRouter()
    router.register_endpoint(
        endpoint_id=endpoint_id,
        provider=provider,
        url=url,
        model=model,
        gpu_type=gpu_type,
        region=region or "",
        price_per_hour=float(price),
    )

    if "inference_endpoints" not in api.usage:
        api.usage["inference_endpoints"] = []
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


@cli.group(name="gateway", invoke_without_command=True)
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind the gateway server")
@click.option("--port", "-p", default=8000, type=int, help="Port for the gateway server")
@click.option("--openai", is_flag=True, default=True, help="Enable OpenAI-compatible endpoints")
@click.option("--no-openai", is_flag=True, help="Disable OpenAI-compatible endpoints")
@click.option("--anthropic", is_flag=True, default=True, help="Enable Anthropic-compatible endpoints")
@click.option("--no-anthropic", is_flag=True, help="Disable Anthropic-compatible endpoints")
@click.option("--custom", is_flag=True, default=True, help="Enable custom workflow endpoints")
@click.option("--no-custom", is_flag=True, help="Disable custom workflow endpoints")
@click.option("--max-concurrent", type=int, default=100, help="Maximum concurrent requests")
@click.option("--timeout", type=int, default=120, help="Request timeout in seconds")
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
@click.option("--port", "-p", default=8000, type=int, help="Gateway port")
@click.pass_context
def gateway_status(ctx, host, port):
    """Show the running gateway server status."""
    url = f"http://{host}:{port}/v1/gateway/status"
    try:
        with urlopen(url, timeout=5) as resp:
            data = json.load(resp)
            print(json.dumps(data, indent=2, default=str))
    except URLError as e:
        print(f"ERROR: Gateway not reachable at {url}: {e}")
        print("Start a gateway with: terradev gateway serve")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        sys.exit(1)


def _build_provider_group(provider: str):
    """Build a Click group for a single inference provider."""
    group = click.Group(
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
        print(f"OK: {provider} credentials saved.")

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

        print(f"OK: {provider} endpoint deployed")
        print(f"  ID: {endpoint_id}")
        if url:
            print(f"  URL: {url}")
        print(f"  Status: {result.get('status', 'unknown')}")

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
            print(f"No {provider} endpoints found")
            return

        print(f"{provider} endpoints:")
        for item in items:
            iid = item.get("instance_id") or item.get("id") or item.get("model_id") or "unknown"
            print(f"  {iid}: {item.get('status', 'unknown')}")

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
        print(json.dumps(result, indent=2, default=str))

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
        print(f"OK: {endpoint_id} deletion initiated")
        print(json.dumps(result, indent=2, default=str))

    @group.command("chat")
    @click.option("--model", "-m", required=True, help="Model or endpoint ID to query")
    @click.option("--prompt", "-p", required=True, help="Prompt text")
    @click.option("--max-tokens", default=2048, type=int, help="Maximum tokens")
    @click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
    @click.pass_context
    def chat_cmd(ctx, model, prompt, max_tokens, temperature):
        """Send a chat/prompt to this provider.

        MODEL can be a model id, endpoint id, or full model path depending on the
        provider. The provider must already have a deployed endpoint.
        """
        api = ctx.obj["api"]
        adapter = Adapter(provider, api)
        response = _run_async(adapter.chat(model, prompt, max_tokens, temperature))
        print(response)

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
            print(f"ERROR: {e}")
            sys.exit(1)

        if not items:
            print(f"No {provider} models found")
            return

        print(f"{provider} models:")
        for item in items:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("model") or item.get("name") or item.get("model_id", "unknown")
                print(f"  {mid}")
            else:
                print(f"  {item}")

    return group


# Attach provider subcommand groups to the gateway group
for _provider in ADAPTERS:
    gateway.add_command(_build_provider_group(_provider))
