#!/usr/bin/env python3
"""
Test Terradev Telemetry Integration
Simulates CLI telemetry data to test the backend servers
"""

import json
import time
import random
import requests
from datetime import datetime

def test_telemetry_server(base_url, server_name):
    """Test telemetry server endpoints"""
    print(f"\n🧪 Testing {server_name} at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Generate test telemetry data
    test_machine_id = f"test_machine_{random.randint(1000, 9999)}"
    test_install_id = f"test_install_{random.randint(1000, 9999)}"
    
    # Test heartbeat
    heartbeat_data = {
        "machine_id": test_machine_id,
        "install_id": test_install_id,
        "api_key": f"tdv_mandatory_{test_machine_id[:32]}",
        "action": "heartbeat",
        "timestamp": datetime.now().isoformat(),
        "details": {
            "version": "2.9.2",
            "platform": "macOS",
            "python_version": "3.9.0",
            "mandatory": True,
            "opt_out": False,
            "compliance": "enforced"
        }
    }
    
    try:
        response = requests.post(f"{base_url}/log-usage", json=heartbeat_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Heartbeat logged: {result['message']}")
        else:
            print(f"❌ Heartbeat failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Heartbeat error: {e}")
    
    # Test quote event
    quote_data = {
        "machine_id": test_machine_id,
        "install_id": test_install_id,
        "api_key": f"tdv_mandatory_{test_machine_id[:32]}",
        "action": "quote",
        "timestamp": datetime.now().isoformat(),
        "details": {
            "gpu_type": "A100",
            "providers": ["AWS", "GCP", "Azure"],
            "best_price": 2.35,
            "provider": "VastAI",
            "region": "us-west",
            "spot": False,
            "workload_type": "training"
        }
    }
    
    try:
        response = requests.post(f"{base_url}/log-usage", json=quote_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Quote event logged: {result['message']}")
        else:
            print(f"❌ Quote event failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Quote event error: {e}")
    
    # Test license check
    license_data = {
        "machine_id": test_machine_id,
        "install_id": test_install_id,
        "api_key": f"tdv_mandatory_{test_machine_id[:32]}",
        "action": "provision",
        "timestamp": datetime.now().isoformat(),
        "details": {
            "mandatory": True,
            "enforcement": "active"
        }
    }
    
    try:
        response = requests.post(f"{base_url}/check-license", json=license_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ License check: {result['tier']} tier - {result['reason']}")
        else:
            print(f"❌ License check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ License check error: {e}")
    
    # Test user stats
    try:
        response = requests.get(f"{base_url}/user-stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ User stats: {stats['total_users']} users, {stats['total_installations']} installations")
        else:
            print(f"❌ User stats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ User stats error: {e}")
    
    return True

def simulate_multiple_users(base_url, num_users=5):
    """Simulate multiple users sending telemetry"""
    print(f"\n👥 Simulating {num_users} users...")
    
    for i in range(num_users):
        machine_id = f"sim_machine_{i+1}_{random.randint(1000, 9999)}"
        install_id = f"sim_install_{i+1}_{random.randint(1000, 9999)}"
        
        # Random actions
        actions = ["heartbeat", "quote", "provision", "daily_active_users"]
        action = random.choice(actions)
        
        data = {
            "machine_id": machine_id,
            "install_id": install_id,
            "api_key": f"tdv_mandatory_{machine_id[:32]}",
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "version": "2.9.2",
                "platform": random.choice(["macOS", "Linux", "Windows"]),
                "gpu_type": random.choice(["A100", "H100", "V100", "RTX4090"]),
                "mandatory": True
            }
        }
        
        try:
            response = requests.post(f"{base_url}/log-usage", json=data, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ User {i+1}: {action}")
            else:
                print(f"  ❌ User {i+1}: failed")
        except Exception as e:
            print(f"  ❌ User {i+1}: error - {e}")
        
        time.sleep(0.1)  # Small delay between requests

def main():
    print("🚀 Terradev Telemetry Backend Test Suite")
    print("=" * 50)
    
    # Test primary server (port 8080)
    primary_success = test_telemetry_server("http://localhost:8080", "Primary Server")
    
    # Test fallback server (port 8081)
    fallback_success = test_telemetry_server("http://localhost:8081", "Fallback Server")
    
    if primary_success:
        print("\n📊 Simulating traffic to primary server...")
        simulate_multiple_users("http://localhost:8080", 3)
    
    if fallback_success:
        print("\n📊 Simulating traffic to fallback server...")
        simulate_multiple_users("http://localhost:8081", 2)
    
    print("\n📈 Final Stats:")
    
    if primary_success:
        try:
            response = requests.get("http://localhost:8080/user-stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"  Primary Server: {stats['total_users']} users, {stats['total_installations']} installations")
        except:
            pass
    
    if fallback_success:
        try:
            response = requests.get("http://localhost:8081/user-stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"  Fallback Server: {stats['total_users']} users, {stats['total_installations']} installations")
        except:
            pass
    
    print("\n✅ Test suite completed!")

if __name__ == '__main__':
    main()
