#!/usr/bin/env python3
"""
Weights & Biases Integration — BYOAPI facilitation layer

Stores the user's W&B API key locally and provides helpers to:
  - Inject WANDB_* env vars into provisioned instances
  - Build run configs with Terradev GPU cost metadata
  - Generate setup scripts for remote instances

Terradev never calls the W&B SDK directly — it facilitates the connection
so the user's training code can seamlessly log to their own W&B account.
"""

from typing import Dict, Any, Optional, List


# ── Credential keys the user provides via `terradev configure` ────────────
REQUIRED_CREDENTIALS = {
    "api_key": "W&B API key (from wandb.ai/settings)",
}

OPTIONAL_CREDENTIALS = {
    "entity": "W&B entity (team or username — defaults to your W&B default)",
    "project": "W&B project name (defaults to 'terradev')",
    "base_url": "W&B server URL (only for self-hosted W&B Server instances)",
}


def get_credential_prompts() -> List[Dict[str, str]]:
    """Return the list of credential prompts for interactive configure."""
    return [
        {"key": "wandb_api_key", "prompt": "W&B API Key", "required": True, "hide": True},
        {"key": "wandb_entity", "prompt": "W&B Entity (team/username, optional)", "required": False, "hide": False},
        {"key": "wandb_project", "prompt": "W&B Project (optional, default: terradev)", "required": False, "hide": False},
        {"key": "wandb_base_url", "prompt": "W&B Server URL (optional, for self-hosted)", "required": False, "hide": False},
    ]


def build_env_vars(credentials: Dict[str, str]) -> Dict[str, str]:
    """
    Build WANDB_* environment variables from stored credentials.

    These are injected into provisioned instances so the user's training
    code can call `wandb.init()` without any manual key setup.
    """
    env = {}

    api_key = credentials.get("wandb_api_key", "")
    if api_key:
        env["WANDB_API_KEY"] = api_key

    entity = credentials.get("wandb_entity", "")
    if entity:
        env["WANDB_ENTITY"] = entity

    project = credentials.get("wandb_project", "")
    if project:
        env["WANDB_PROJECT"] = project
    else:
        env["WANDB_PROJECT"] = "terradev"

    base_url = credentials.get("wandb_base_url", "")
    if base_url:
        env["WANDB_BASE_URL"] = base_url

    return env


def build_run_config(
    gpu_type: str,
    provider: str,
    price_per_hour: float,
    region: str,
    instance_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a W&B run config dict with Terradev GPU cost metadata.

    Users can pass this to `wandb.init(config=...)` to automatically
    tag their runs with provisioning details for cost-to-value tracking.
    """
    config = {
        "terradev_gpu_type": gpu_type,
        "terradev_provider": provider,
        "terradev_price_per_hour": price_per_hour,
        "terradev_region": region,
        "terradev_instance_id": instance_id,
    }
    if extra:
        config.update(extra)
    return config


def generate_setup_script(credentials: Dict[str, str]) -> str:
    """
    Generate a shell snippet that sets up W&B on a remote instance.

    This is injected into the instance's startup or passed via
    `terradev run --command` so the user's environment is ready.
    """
    lines = ["#!/bin/bash", "# Terradev — W&B environment setup"]

    env_vars = build_env_vars(credentials)
    for key, value in env_vars.items():
        lines.append(f'export {key}="{value}"')

    lines.append("")
    lines.append("# Verify W&B connection (optional — requires wandb pip package)")
    lines.append("if command -v wandb &> /dev/null; then")
    lines.append("    wandb login --relogin $WANDB_API_KEY 2>/dev/null && echo '✅ W&B connected'")
    lines.append("else")
    lines.append("    echo '💡 Install wandb: pip install wandb'")
    lines.append("fi")

    return "\n".join(lines)


def is_configured(credentials: Dict[str, str]) -> bool:
    """Check if W&B credentials are stored."""
    return bool(credentials.get("wandb_api_key", ""))


def get_status_summary(credentials: Dict[str, str]) -> Dict[str, Any]:
    """Return a summary of W&B integration status for `terradev status`."""
    configured = is_configured(credentials)
    return {
        "integration": "wandb",
        "name": "Weights & Biases",
        "configured": configured,
        "entity": credentials.get("wandb_entity", "(default)"),
        "project": credentials.get("wandb_project", "terradev"),
        "self_hosted": bool(credentials.get("wandb_base_url", "")),
    }
