#!/usr/bin/env python3
"""
Terradev CLI - GitHub Action Entry Point
"""

import sys
import json
import subprocess
from pathlib import Path

def main():
    """Main entry point for GitHub Action"""
    
    # Parse inputs from environment variables (GitHub Actions standard)
    command = sys.argv[1] if len(sys.argv) > 1 else 'quote'
    gpu_type = sys.argv[2] if len(sys.argv) > 2 else 'A100'
    duration_hours = sys.argv[3] if len(sys.argv) > 3 else '1'
    cloud_provider = sys.argv[4] if len(sys.argv) > 4 else 'auto'
    region = sys.argv[5] if len(sys.argv) > 5 else 'auto'
    huggingface_model = sys.argv[6] if len(sys.argv) > 6 else ''
    
    print(f"🚀 Terradev CLI - GitHub Action")
    print(f"Command: {command}")
    print(f"GPU Type: {gpu_type}")
    print(f"Duration: {duration_hours} hours")
    print(f"Provider: {cloud_provider}")
    print(f"Region: {region}")
    
    # Build command
    cmd = ['python', '-m', 'terradev_cli', command]
    
    if command == 'quote':
        cmd.extend(['-g', gpu_type])
    elif command == 'provision':
        cmd.extend(['-g', gpu_type, '--duration', duration_hours])
        if cloud_provider != 'auto':
            cmd.extend(['--provider', cloud_provider])
        if region != 'auto':
            cmd.extend(['--region', region])
    elif command == 'hf-space':
        if huggingface_model:
            cmd.extend(['--model-id', huggingface_model])
        else:
            cmd.extend(['--model-id', 'meta-llama/Llama-2-7b-hf'])
    
    # Execute command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Command executed successfully")
        print("STDOUT:")
        print(result.stdout)
        
        # Parse output for GitHub Actions outputs
        outputs = parse_output(result.stdout, command)
        
        # Set GitHub Actions outputs
        for key, value in outputs.items():
            print(f"::set-output name={key}::{value}")
            
    except subprocess.CalledProcessError as e:
        print("❌ Command failed")
        print("STDERR:")
        print(e.stderr)
        sys.exit(1)

def parse_output(output, command):
    """Parse CLI output and extract key information"""
    outputs = {}
    
    if command == 'quote':
        # Extract pricing data
        lines = output.split('\n')
        prices = []
        providers = []
        
        for line in lines:
            if '$' in line and any(provider in line for provider in ['AWS', 'GCP', 'Azure', 'RunPod', 'Lambda']):
                parts = line.split()
                for part in parts:
                    if '$' in part:
                        price = part.replace('$', '').replace('/hr', '')
                        try:
                            prices.append(float(price))
                        except ValueError:
                            pass
                        break
        
        if prices:
            outputs['pricing_data'] = json.dumps(prices)
            outputs['optimal_provider'] = f"${min(prices):.2f}/hr (lowest price)"
            outputs['estimated_cost'] = f"${min(prices)} for 1 hour"
    
    elif command == 'provision':
        # Extract deployment info
        outputs['deployment_url'] = "https://dashboard.terradev.cloud"  # Placeholder
        outputs['estimated_cost'] = output.split('\n')[0] if output else "Unknown"
    
    elif command == 'hf-space':
        # Extract HuggingFace Space info
        outputs['deployment_url'] = "https://huggingface.co/spaces/terradev"  # Placeholder
        outputs['estimated_cost'] = "Free tier available"
    
    return outputs

if __name__ == "__main__":
    main()
