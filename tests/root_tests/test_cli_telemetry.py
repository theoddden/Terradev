#!/usr/bin/env python3
"""
Terradev CLI Telemetry Integration Test
Tests the actual CLI telemetry against our running backend servers
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Add CLI to path
cli_path = Path(__file__).parent / "terradev_cli"
sys.path.insert(0, str(cli_path))

def test_cli_telemetry():
    """Test CLI telemetry integration"""
    print("🧪 Testing CLI Telemetry Integration")
    print("=" * 50)
    
    # Set environment variables to point to our local servers
    os.environ['TERRADEV_API_URL'] = 'http://localhost:8080'
    os.environ['TERRADEV_FALLBACK_URL'] = 'http://localhost:8081'
    
    try:
        # Import and test telemetry client
        from terradev_cli.core.telemetry import get_mandatory_telemetry
        
        print("✅ Successfully imported telemetry client")
        
        # Get telemetry client
        telemetry = get_mandatory_telemetry()
        print(f"✅ Telemetry client initialized")
        print(f"   Machine ID: {telemetry.machine_id[:16]}...")
        print(f"   Install ID: {telemetry.install_id[:16]}...")
        
        # Test logging an action
        print("\n📊 Testing telemetry logging...")
        telemetry.log_action('test_action', {
            'test': True,
            'timestamp': time.time(),
            'source': 'cli_integration_test'
        })
        print("✅ Test action logged")
        
        # Test license check
        print("\n📋 Testing license check...")
        license_result = telemetry.check_license('test')
        print(f"✅ License check result: {license_result}")
        
        # Wait a moment for async logging
        time.sleep(2)
        
        # Check stats on primary server
        print("\n📈 Checking primary server stats...")
        try:
            import requests
            response = requests.get('http://localhost:8080/user-stats', timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Primary Server Stats:")
                print(f"   Total Users: {stats['total_users']}")
                print(f"   Active Today: {stats['active_users_today']}")
                print(f"   Installations: {stats['total_installations']}")
            else:
                print(f"❌ Failed to get stats: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import telemetry client: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing CLI telemetry: {e}")
        return False

def simulate_cli_usage():
    """Simulate various CLI actions"""
    print("\n🎭 Simulating CLI Usage...")
    
    try:
        from terradev_cli.core.telemetry import get_mandatory_telemetry
        telemetry = get_mandatory_telemetry()
        
        # Simulate different CLI actions
        actions = [
            ('quote', {'gpu_type': 'A100', 'providers': ['AWS', 'GCP']}),
            ('provision', {'gpu_type': 'H100', 'duration_hours': 4}),
            ('heartbeat', {'uptime': 3600, 'commands_run': 15}),
            ('daily_active_users', {'session_start': time.time()}),
            ('price_check', {'gpu_type': 'V100', 'spot_price': 1.25})
        ]
        
        for action, details in actions:
            telemetry.log_action(action, details)
            print(f"✅ Logged: {action}")
            time.sleep(0.5)  # Small delay between actions
        
        print("✅ CLI usage simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Error simulating CLI usage: {e}")
        return False

def main():
    print("🚀 Terradev CLI Telemetry Integration Test")
    print("=" * 60)
    
    # Test basic CLI telemetry
    if test_cli_telemetry():
        print("\n✅ Basic CLI telemetry test passed")
        
        # Simulate CLI usage
        if simulate_cli_usage():
            print("\n✅ CLI usage simulation passed")
        else:
            print("\n❌ CLI usage simulation failed")
    else:
        print("\n❌ Basic CLI telemetry test failed")
        return
    
    # Final stats check
    print("\n📊 Final Server Stats:")
    try:
        import requests
        
        # Primary server
        response = requests.get('http://localhost:8080/user-stats', timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"  Primary Server (8080):")
            print(f"    Users: {stats['total_users']}")
            print(f"    Active Today: {stats['active_users_today']}")
            print(f"    Installations: {stats['total_installations']}")
        
        # Fallback server
        response = requests.get('http://localhost:8081/user-stats', timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"  Fallback Server (8081):")
            print(f"    Users: {stats['total_users']}")
            print(f"    Active Today: {stats['active_users_today']}")
            print(f"    Installations: {stats['total_installations']}")
        
    except Exception as e:
        print(f"❌ Error getting final stats: {e}")
    
    print("\n🎉 CLI telemetry integration test completed!")

if __name__ == '__main__':
    main()
