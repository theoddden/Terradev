#!/usr/bin/env python3
"""
Simple credential prompt system - no demo mode, just input credentials
"""

import json
from pathlib import Path
from typing import Dict

import click


def prompt_for_credentials():
    """Prompt users to input their API credentials"""

    config_dir = Path.home() / ".terradev"
    config_dir.mkdir(exist_ok=True)

    credentials_file = config_dir / "credentials.json"

    # Load existing credentials
    existing_creds = {}
    if credentials_file.exists():
        with open(credentials_file, "r") as f:
            existing_creds = json.load(f)

    print("Configure Cloud Provider Credentials")
    print("=" * 50)
    print("Enter your API keys for the providers you want to use.")
    print("Press Enter to skip a provider.")
    print()

    # Provider configurations: name, help text, example, and required fields
    PROVIDER_PROMPTS = {
        "runpod": {
            "name": "RunPod",
            "help": "Get from: https://runpod.io/console/settings/api-keys",
            "example": "rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "vastai": {
            "name": "Vast.ai",
            "help": "Get from: https://console.vast.ai/api-keys",
            "example": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "aws": {
            "name": "AWS",
            "help": "Get from: AWS IAM console",
            "example": "AKIAEXAMPLEKEY123456",
            "fields": [
                {"key": "api_key", "label": "Access Key ID", "hide_input": True},
                {"key": "secret_key", "label": "Secret Access Key", "hide_input": True},
            ],
        },
        "gcp": {
            "name": "Google Cloud",
            "help": "Get from: GCP Console → IAM & Admin → Service Accounts. Provide the path to your downloaded service account JSON.",
            "example": "/path/to/service-account.json",
            "fields": [
                {"key": "credentials_file", "label": "Service Account JSON file path", "hide_input": False},
                {"key": "project_id", "label": "GCP Project ID", "hide_input": False},
            ],
        },
        "azure": {
            "name": "Azure",
            "help": "Azure Portal → Azure AD → App Registrations → New Registration. Create client secret under Certificates & Secrets. Get subscription ID and assign Contributor role.",
            "example": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "fields": [
                {"key": "subscription_id", "label": "Subscription ID", "hide_input": False},
                {"key": "tenant_id", "label": "Tenant ID", "hide_input": False},
                {"key": "client_id", "label": "Client ID", "hide_input": False},
                {"key": "client_secret", "label": "Client Secret", "hide_input": True},
            ],
        },
        "tensordock": {
            "name": "TensorDock",
            "help": "Get from: TensorDock dashboard",
            "example": "td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
                {"key": "api_token", "label": "API Token", "hide_input": True},
            ],
        },
        "crusoe": {
            "name": "Crusoe Cloud",
            "help": "Get from: Crusoe dashboard",
            "example": "crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "access_key", "label": "Access Key", "hide_input": False},
                {"key": "secret_key", "label": "Secret Key", "hide_input": True},
                {"key": "project_id", "label": "Project ID", "hide_input": False},
            ],
        },
        "huggingface": {
            "name": "HuggingFace",
            "help": "Get from: https://huggingface.co/settings/tokens",
            "example": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Token", "hide_input": True},
                {"key": "namespace", "label": "Namespace (username or org)", "hide_input": False},
            ],
        },
        "baseten": {
            "name": "Baseten",
            "help": "Get from: Baseten dashboard",
            "example": "bt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "hyperstack": {
            "name": "Hyperstack",
            "help": "Get from: Hyperstack dashboard → API Keys",
            "example": "hs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
                {"key": "environment", "label": "Environment", "hide_input": False, "default": "default-CANADA-1"},
                {"key": "ssh_key_name", "label": "SSH Key Name", "hide_input": False, "default": ""},
            ],
        },
        "digitalocean": {
            "name": "DigitalOcean",
            "help": "Get from: DigitalOcean → API → Tokens",
            "example": "dop_v1_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Token", "hide_input": True},
                {"key": "region", "label": "Region", "hide_input": False, "default": "tor1"},
            ],
        },
        "e2enetworks": {
            "name": "E2E Networks",
            "help": "Get from: https://e2enetworks.com",
            "example": "e2e_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
                {"key": "project_id", "label": "Project ID (optional)", "hide_input": False, "default": ""},
            ],
        },
        "inferx": {
            "name": "InferX",
            "help": "Get from: InferX Console → endpoint Client Setup",
            "example": "ix_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
                {"key": "api_endpoint", "label": "API Endpoint", "hide_input": False, "default": "https://model.inferx.net/endpoints/v1"},
                {"key": "model", "label": "Default model", "hide_input": False, "default": "Qwen3.8-27B-FP8"},
            ],
        },
        "latitude": {
            "name": "Latitude.sh",
            "help": "Get from: https://latitude.sh/account/api-keys",
            "example": "lat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "siliconflow": {
            "name": "SiliconFlow",
            "help": "Get from: https://cloud.siliconflow.cn/account/ak",
            "example": "sf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "yottalabs": {
            "name": "Yotta Labs",
            "help": "Get from: https://yottalabs.ai",
            "example": "yotta_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Key", "hide_input": True},
            ],
        },
        "gcore": {
            "name": "Gcore",
            "help": "Get from: Gcore Customer Portal → Profile → API Tokens",
            "example": "12345_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "fields": [
                {"key": "api_key", "label": "API Token", "hide_input": True},
                {"key": "project_id", "label": "Project ID (optional)", "hide_input": False, "default": ""},
                {"key": "region_id", "label": "Region ID (optional)", "hide_input": False, "default": ""},
            ],
        },
    }

    updated_creds = existing_creds.copy()

    for provider_id, config in PROVIDER_PROMPTS.items():
        print(f"\n{config['name']}")
        print(f"   Help: {config['help']}")
        print(f"   Example: {config['example']}")

        # Check if already configured
        existing = existing_creds.get(provider_id, {})
        if existing and any(str(v).strip() for v in existing.values() if isinstance(v, str)):
            first_key = next(iter(existing))
            print(f"   Already configured: {first_key}={existing[first_key][:10]}...")
            if not click.confirm(f"   Update {config['name']} credentials?", default=False):
                continue

        # Prompt for each required field
        collected: Dict[str, str] = {}
        for field in config["fields"]:
            default = field.get("default", "")
            show_default = bool(default)
            value = click.prompt(
                f"   Enter {field['label']}",
                default=default,
                hide_input=field.get("hide_input", False),
                show_default=show_default,
            )
            collected[field["key"]] = value.strip()

        if any(collected.values()):
            updated_creds[provider_id] = collected
            print(f"   {config['name']} credentials saved")
        else:
            # Remove if user skipped and it existed
            if provider_id in updated_creds:
                del updated_creds[provider_id]
                print(f"   {config['name']} credentials removed")

    # Save credentials
    with open(credentials_file, "w") as f:
        json.dump(updated_creds, f, indent=2)

    print(f"\nCredentials saved to: {credentials_file}")

    # Show configured providers
    configured_providers = list(updated_creds.keys())
    if configured_providers:
        print(f"\nConfigured providers: {', '.join(configured_providers)}")
        print("\nGet quotes:")
        print("   terradev quote --gpu-type a100")
        print(
            f"   terradev quote --gpu-type h100 --providers {','.join(configured_providers[:3])}"
        )
    else:
        print("\nNo providers configured")
        print("   Run 'terradev configure' to add credentials")

    return configured_providers


def check_configured_providers():
    """Check which providers are configured"""

    config_dir = Path.home() / ".terradev"
    credentials_file = config_dir / "credentials.json"

    if not credentials_file.exists():
        return []

    with open(credentials_file, "r") as f:
        credentials = json.load(f)

    return list(credentials.keys())
