#!/usr/bin/env python3
"""
Simple credential prompt system - no demo mode, just input credentials
"""

import json
from pathlib import Path
import click

def prompt_for_credentials():
    """Prompt users to input their API credentials"""
    
    config_dir = Path.home() / '.terradev'
    config_dir.mkdir(exist_ok=True)
    
    credentials_file = config_dir / 'credentials.json'
    
    # Load existing credentials
    existing_creds = {}
    if credentials_file.exists():
        with open(credentials_file, 'r') as f:
            existing_creds = json.load(f)
    
    print("Configure Cloud Provider Credentials")
    print("=" * 50)
    print("Enter your API keys for the providers you want to use.")
    print("Press Enter to skip a provider.")
    print()
    
    # Provider configurations
    providers = {
        'runpod': {
            'name': 'RunPod',
            'key_name': 'API Key',
            'help': 'Get from: https://runpod.io/console/settings/api-keys',
            'example': 'rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'vastai': {
            'name': 'Vast.ai',
            'key_name': 'API Key',
            'help': 'Get from: https://console.vast.ai/api-keys',
            'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'aws': {
            'name': 'AWS',
            'key_name': 'Access Key ID',
            'help': 'Get from: AWS IAM console',
            'example': 'AKIAIOSFODNN7EXAMPLE'
        },
        'gcp': {
            'name': 'Google Cloud',
            'key_name': 'Service Account JSON',
            'help': 'Get from: GCP Console → IAM & Admin → Service Accounts',
            'example': 'path/to/service-account.json'
        },
        'azure': {
            'name': 'Azure',
            'key_name': 'Client ID',
            'help': 'Get from: Azure Portal → App registrations',
            'example': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
        },
        'lambda_labs': {
            'name': 'Lambda Labs',
            'key_name': 'API Key',
            'help': 'Get from: Lambda Labs dashboard',
            'example': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'coreweave': {
            'name': 'CoreWeave',
            'key_name': 'API Key',
            'help': 'Get from: CoreWeave dashboard',
            'example': 'cw_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'tensordock': {
            'name': 'TensorDock',
            'key_name': 'API Key',
            'help': 'Get from: TensorDock dashboard',
            'example': 'td_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'oracle': {
            'name': 'Oracle Cloud',
            'key_name': 'API Key',
            'help': 'Get from: Oracle Cloud Console',
            'example': 'ocid1.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        },
        'crusoe': {
            'name': 'Crusoe Cloud',
            'key_name': 'API Key',
            'help': 'Get from: Crusoe dashboard',
            'example': 'crusoe_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        }
    }
    
    updated_creds = existing_creds.copy()
    
    for provider_id, config in providers.items():
        print(f"\n{config['name']}")
        print(f"   Help: {config['help']}")
        print(f"   Example: {config['example']}")
        
        # Check if already configured
        existing_key = existing_creds.get(provider_id, {}).get('api_key')
        if existing_key:
            print(f"   Already configured: {existing_key[:10]}...")
            if click.confirm(f"   Update {config['name']} credentials?", default=False):
                pass  # User wants to update
            else:
                continue  # Skip this provider
        
        # Prompt for API key
        api_key = click.prompt(f"   Enter {config['key_name']}", 
                              default='', 
                              hide_input=True, 
                              show_default=False)
        
        if api_key.strip():
            if provider_id == 'gcp':
                # GCP needs special handling for service account JSON
                updated_creds[provider_id] = {
                    'service_account_key': api_key.strip(),
                    'project_id': click.prompt("   Enter GCP Project ID", default="my-project")
                }
            else:
                updated_creds[provider_id] = {'api_key': api_key.strip()}
            
            print(f"   {config['name']} credentials saved")
        else:
            # Remove if user skipped and it existed
            if provider_id in updated_creds:
                del updated_creds[provider_id]
                print(f"   {config['name']} credentials removed")
    
    # Save credentials
    with open(credentials_file, 'w') as f:
        json.dump(updated_creds, f, indent=2)
    
    print(f"\nCredentials saved to: {credentials_file}")
    
    # Show configured providers
    configured_providers = list(updated_creds.keys())
    if configured_providers:
        print(f"\nConfigured providers: {', '.join(configured_providers)}")
        print(f"\nGet quotes:")
        print(f"   terradev quote --gpu-type a100")
        print(f"   terradev quote --gpu-type h100 --providers {','.join(configured_providers[:3])}")
    else:
        print(f"\nNo providers configured")
        print(f"   Run 'terradev configure' to add credentials")
    
    return configured_providers

def check_configured_providers():
    """Check which providers are configured"""
    
    config_dir = Path.home() / '.terradev'
    credentials_file = config_dir / 'credentials.json'
    
    if not credentials_file.exists():
        return []
    
    with open(credentials_file, 'r') as f:
        credentials = json.load(f)
    
    return list(credentials.keys())
