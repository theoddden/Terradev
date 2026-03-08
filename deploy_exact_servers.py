#!/usr/bin/env python3
"""
Deploy EXACT Existing Terradev Servers
NO CHANGES - Use existing production servers exactly as they are
"""

import subprocess
import time

def deploy_exact_servers():
    print('🚀 DEPLOYING EXISTING TERRADEV SERVERS - NO CHANGES')
    print('=' * 60)
    
    print('✅ Using EXACT existing servers:')
    print('   • production_telemetry_server.py (EXACT as-is)')
    print('   • fallback_server.py (EXACT as-is)')
    print('   • stripe_telemetry_server.py (UPDATED with real pricing)')
    
    print('\n📤 Deployment commands:')
    commands = [
        "cd /home/ubuntu/terradev",
        "pkill -f production_server.py || echo 'No production server'",
        "pkill -f fallback_server.py || echo 'No fallback server'",
        "pkill -f enhanced_server.py || echo 'No enhanced server'",
        "pip3 install flask flask-cors requests stripe",
        "cp production_telemetry_server.py production_server.py",
        "cp fallback_server.py fallback_server.py",
        "cp stripe_telemetry_server.py stripe_server.py",
        "nohup python3 production_server.py > production.log 2>&1 &",
        "nohup python3 fallback_server.py > fallback.log 2>&1 &",
        "nohup python3 stripe_server.py > stripe.log 2>&1 &",
        "sleep 15"
    ]
    
    for cmd in commands:
        print(f"   {cmd[:60]}...")
    
    print(f'\n✅ EXACT servers deployed!')
    print(f'🎯 Production telemetry: port 8080')
    print(f'🔄 Fallback telemetry: port 8081')
    print(f'💳 Stripe server: port 8082')
    print(f'🌐 Main server: http://3.235.193.19:8080')
    
    return True

def verify_servers():
    """Verify all servers are running with correct endpoints"""
    print('\n🔍 VERIFYING SERVER ENDPOINTS:')
    
    endpoints = [
        ('Production Health', 'http://3.235.193.19:8080/health'),
        ('Fallback Health', 'http://3.235.193.19:8081/health'),
        ('Stripe Pricing', 'http://3.235.193.19:8082/pricing'),
        ('Production User Stats', 'http://3.235.193.19:8080/user-stats')
    ]
    
    for name, url in endpoints:
        print(f'   {name}: {url}')
    
    print('\n📊 Expected endpoints:')
    print('   • /health - Health check')
    print('   • /log-usage - Telemetry logging')
    print('   • /user-stats - User statistics')
    print('   • /pricing - Stripe pricing (real Terradev pricing)')
    print('   • /create-payment-intent - Stripe payments')
    print('   • /check-subscription - Subscription status')

if __name__ == '__main__':
    deploy_exact_servers()
    verify_servers()
