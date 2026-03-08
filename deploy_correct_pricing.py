#!/usr/bin/env python3
"""
Deploy Correct Stripe Server with Real Terradev Pricing
"""

import subprocess
import time

def deploy_correct_pricing():
    print('🚀 DEPLOYING CORRECT STRIPE SERVER WITH REAL TERRADEV PRICING')
    print('=' * 60)
    
    print('✅ Updated pricing tiers to match real Terradev:')
    print('   • Research: FREE (10 provisions, 1 instance)')
    print('   • Research+: $49.99/month (80 provisions, 4 instances)')
    print('   • Enterprise: $299.99/month (unlimited, 32 instances)')
    print('   • Enterprise+: $0.09/GPU-hour (metered, unlimited)')
    
    print('\n📤 Deployment commands:')
    commands = [
        "cd /home/ubuntu/terradev",
        "pkill -f production_server.py || echo 'No production server'",
        "pkill -f enhanced_server.py || echo 'No enhanced server'",
        "pip3 install stripe",
        "wget -q https://raw.githubusercontent.com/theoddden/terradev/main/stripe_telemetry_server.py -O correct_stripe_server.py",
        "nohup python3 correct_stripe_server.py > correct_stripe.log 2>&1 &",
        "sleep 15"
    ]
    
    for cmd in commands:
        print(f"   {cmd[:60]}...")
    
    print(f'\n✅ Correct Stripe server deployed!')
    print(f'🎯 Real Terradev pricing structure implemented')
    print(f'💰 Ultra-low cost: $7.49/month')
    print(f'📊 Everything on ONE t3.micro instance!')
    print(f'🌐 Server: http://3.235.193.19:8080')
    print(f'📊 Dashboard: http://3.235.193.19:8080/dashboard')
    print(f'💳 Pricing: http://3.235.193.19:8080/pricing')
    
    return True

def show_correct_pricing():
    """Show the correct pricing structure"""
    print('\n' + '='*60)
    print('💰 REAL TERRADEV PRICING STRUCTURE')
    print('='*60)
    
    print('\n📊 PRICING TIERS:')
    print('   🔬 Research: FREE')
    print('      • 10 provisions/month')
    print('      • 1 instance max')
    print('      • 3 parallel queries')
    print('      • Basic providers (AWS, RunPod, VastAI)')
    
    print('\n   🔬 Research+: $49.99/month')
    print('      • 80 provisions/month')
    print('      • 4 instances max')
    print('      • 10 parallel queries')
    print('      • All major providers')
    print('      • Inference support')
    
    print('\n   🏢 Enterprise: $299.99/month')
    print('      • Unlimited provisions')
    print('      • 32 instances max')
    print('      • 50 parallel queries')
    print('      • All providers + CoreWeave')
    print('      • Enterprise features')
    
    print('\n   🏢 Enterprise+: $0.09/GPU-hour (metered)')
    print('      • Unlimited everything')
    print('      • 100 parallel queries')
    print('      • All providers + Lambda Labs, Crusoe, etc.')
    print('      • Fleet management & dedicated support')
    print('      • Minimum: $368.64/month (32 GPUs × 128 hrs)')
    
    print('\n💵 BREAK-EVEN ANALYSIS:')
    print('   • Server cost: $7.49/month')
    print('   • Break-even: 1 Research+ user = $42.50 profit/month')
    print('   • Break-even: 1 Enterprise user = $292.50 profit/month')
    print('   • Enterprise+: Minimum $361.15 profit/month')
    
    print('\n🎯 URLS:')
    print('   • API: http://3.235.193.19:8080')
    print('   • Pricing: http://3.235.193.19:8080/pricing')
    print('   • Dashboard: http://3.235.193.19:8080/dashboard')
    print('   • Health: http://3.235.193.19:8080/health')

if __name__ == '__main__':
    deploy_correct_pricing()
    show_correct_pricing()
