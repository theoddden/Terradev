#!/usr/bin/env python3
"""
Check Server Logs and Fix Deployment Issues
"""

import subprocess
import time

def check_server_logs():
    print('🔍 CHECKING SERVER LOGS AND FIXING ISSUES')
    print('=' * 50)
    
    # Commands to check logs and restart servers
    commands = [
        "cd /home/ubuntu/terradev",
        "echo '=== Production Server Log ==='",
        "tail -20 production.log || echo 'No production log'",
        "echo '=== Fallback Server Log ==='",
        "tail -20 fallback.log || echo 'No fallback log'",
        "echo '=== Stripe Server Log ==='",
        "tail -20 stripe.log || echo 'No stripe log'",
        "echo '=== Running Processes ==='",
        "ps aux | grep python3 | grep -v grep || echo 'No Python processes'",
        "echo '=== Port Usage ==='",
        "netstat -tlnp | grep ':808[012]' || echo 'No ports bound'",
        "echo '=== Kill All Servers ==='",
        "pkill -f production_server.py || echo 'No production server'",
        "pkill -f fallback_server.py || echo 'No fallback server'",
        "pkill -f stripe_server.py || echo 'No stripe server'",
        "echo '=== Start Production Server (port 8080) ==='",
        "nohup python3 production_server.py > production.log 2>&1 &",
        "sleep 5",
        "echo '=== Start Fallback Server (port 8081) ==='",
        "nohup python3 fallback_server.py > fallback.log 2>&1 &",
        "sleep 5",
        "echo '=== Start Stripe Server (port 8082) ==='",
        "nohup python3 stripe_server.py > stripe.log 2>&1 &",
        "sleep 10",
        "echo '=== Final Status ==='",
        "ps aux | grep python3 | grep server || echo 'No server processes'",
        "netstat -tlnp | grep ':808[012]' || echo 'No ports bound'"
    ]
    
    for cmd in commands:
        print(f"📤 Running: {cmd[:50]}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.stdout.strip():
                print(f"✅ Output: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"⚠️  Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("⏰ Command timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()
    
    return True

def test_all_servers():
    """Test all three servers"""
    print('🧪 TESTING ALL THREE SERVERS')
    print('=' * 30)
    
    servers = [
        ('Production', 'http://3.235.193.19:8080/health'),
        ('Fallback', 'http://3.235.193.19:8081/health'),
        ('Stripe', 'http://3.235.193.19:8082/pricing')
    ]
    
    for name, url in servers:
        print(f"🔍 Testing {name}: {url}")
        try:
            result = subprocess.run(f'curl -s {url}', shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                print(f"✅ {name} is responding")
                if name == 'Stripe':
                    # Check if pricing is correct
                    if 'research_plus' in result.stdout or '4999' in result.stdout:
                        print("✅ Stripe has correct Terradev pricing")
                    else:
                        print("⚠️  Stripe pricing may not be correct")
            else:
                print(f"❌ {name} not responding")
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
        print()

if __name__ == '__main__':
    check_server_logs()
    test_all_servers()
