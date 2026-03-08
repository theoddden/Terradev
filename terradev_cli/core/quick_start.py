#!/usr/bin/env python3
"""
Quick Start - Get users to provision SOMETHING in 60 seconds
Remove all friction, show immediate value, hit paywall fast
"""

import json
import time
from typing import Dict, List, Any
from pathlib import Path

class QuickStart:
    """Get users from download to provision in 60 seconds"""
    
    def __init__(self):
        self.demo_providers = {
            'runpod': {
                'name': 'RunPod',
                'description': 'Easiest GPU cloud - start in 30 seconds',
                'setup_url': 'https://runpod.io/console/auth',
                'api_key_url': 'https://runpod.io/console/user/settings',
                'quick_start': True,
                'free_credit': True
            },
            'vastai': {
                'name': 'Vast.ai',
                'description': 'Cheap GPU rentals - marketplace model',
                'setup_url': 'https://vast.ai/console/auth',
                'api_key_url': 'https://vast.ai/console/account/api-keys',
                'quick_start': True,
                'free_credit': False
            }
        }
    
    def show_quick_start_guide(self):
        """Show ultra-simple getting started guide"""
        print("🚀 QUICK START - Provision in 60 seconds")
        print("=" * 50)
        print()
        
        print("📋 STEP 1: Choose Your Cloud (30 seconds)")
        print()
        
        for i, (provider_id, info) in enumerate(self.demo_providers.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   📝 {info['description']}")
            print(f"   🔗 Setup: {info['setup_url']}")
            if info.get('free_credit'):
                print(f"   💰 Free credit available!")
            print()
        
        choice = input("Choose provider (1-2): ").strip()
        
        if choice == '1':
            return self._runpod_quick_start()
        elif choice == '2':
            return self._vastai_quick_start()
        else:
            print("❌ Invalid choice")
            return False
    
    def _runpod_quick_start(self):
        """RunPod ultra-simple setup"""
        print("🎯 RUNPOD QUICK START")
        print("=" * 30)
        print()
        
        print("📋 STEP 2: Get API Key (20 seconds)")
        print("1. Go to: https://runpod.io/console/auth")
        print("2. Sign up/login (Google/GitHub works)")
        print("3. Go to: https://runpod.io/console/user/settings")
        print("4. Copy your API Key")
        print()
        
        api_key = input("🔑 Paste your RunPod API Key: ").strip()
        
        if not api_key:
            print("❌ API Key required")
            return False
        
        print("✅ API Key saved!")
        print()
        
        print("📋 STEP 3: Provision Your First GPU (10 seconds)")
        print()
        
        # Save credentials
        self._save_credentials('runpod', {'api_key': api_key})
        
        # Show cheapest options
        print("💰 CHEAPEST OPTIONS RIGHT NOW:")
        print("1. RTX 3090 - ~$0.20/hour - Great for testing")
        print("2. RTX 4090 - ~$0.40/hour - Fast performance")
        print("3. A100 - ~$1.50/hour - Enterprise GPU")
        print()
        
        gpu_choice = input("Choose GPU (1-3, default 1): ").strip() or "1"
        
        gpu_map = {
            "1": "RTX 3090",
            "2": "RTX 4090", 
            "3": "A100"
        }
        
        gpu_type = gpu_map.get(gpu_choice, "RTX 3090")
        
        print(f"🚀 Provisioning {gpu_type}...")
        print("   This will cost ~\$0.20-1.50/hour")
        print("   You can stop it anytime")
        print()
        
        if input("Proceed? (Y/n): ").lower() == 'n':
            print("❌ Cancelled")
            return False
        
        # Actually provision
        try:
            from terradev_cli.cli import TerradevAPI
            api = TerradevAPI()
            
            # Quick provision using existing logic
            result = self._quick_provision(api, gpu_type, 'runpod')
            
            if result:
                print("🎉 SUCCESS! Your GPU is ready!")
                print(f"   Instance ID: {result.get('instance_id', 'N/A')}")
                print(f"   Cost: ~${result.get('price', 0.20)}/hour")
                print()
                print("💡 NEXT STEPS:")
                print("   1. SSH into your instance")
                print("   2. Run your ML workload")
                print("   3. Stop when done to save money")
                print()
                print("💰 UPGRADE FOR MORE:")
                print("   - 8x more provisions/month")
                print("   - 8 parallel instances")
                print("   - Priority support")
                print("   terradev upgrade --tier research_plus")
                
                return True
            else:
                print("❌ Provision failed")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _vastai_quick_start(self):
        """Vast.ai ultra-simple setup"""
        print("🎯 VAST.AI QUICK START")
        print("=" * 30)
        print()
        
        print("📋 STEP 2: Get API Key (20 seconds)")
        print("1. Go to: https://vast.ai/console/auth")
        print("2. Sign up/login")
        print("3. Go to: https://vast.ai/console/account/api-keys")
        print("4. Create new API Key")
        print("5. Copy your API Key")
        print()
        
        api_key = input("🔑 Paste your Vast.ai API Key: ").strip()
        
        if not api_key:
            print("❌ API Key required")
            return False
        
        print("✅ API Key saved!")
        print()
        
        print("📋 STEP 3: Provision Your First GPU (10 seconds)")
        print()
        
        # Save credentials
        self._save_credentials('vastai', {'api_key': api_key})
        
        # Show cheapest options
        print("💰 CHEAPEST OPTIONS RIGHT NOW:")
        print("1. RTX 3090 - ~$0.15/hour - Great for testing")
        print("2. RTX 4090 - ~$0.35/hour - Fast performance")
        print("3. A100 - ~$1.20/hour - Enterprise GPU")
        print()
        
        gpu_choice = input("Choose GPU (1-3, default 1): ").strip() or "1"
        
        gpu_map = {
            "1": "RTX 3090",
            "2": "RTX 4090",
            "3": "A100"
        }
        
        gpu_type = gpu_map.get(gpu_choice, "RTX 3090")
        
        print(f"🚀 Provisioning {gpu_type}...")
        print("   This will cost ~\$0.15-1.20/hour")
        print("   You can stop it anytime")
        print()
        
        if input("Proceed? (Y/n): ").lower() == 'n':
            print("❌ Cancelled")
            return False
        
        # Actually provision
        try:
            from terradev_cli.cli import TerradevAPI
            api = TerradevAPI()
            
            # Quick provision using existing logic
            result = self._quick_provision(api, gpu_type, 'vastai')
            
            if result:
                print("🎉 SUCCESS! Your GPU is ready!")
                print(f"   Instance ID: {result.get('instance_id', 'N/A')}")
                print(f"   Cost: ~${result.get('price', 0.15)}/hour")
                print()
                print("💡 NEXT STEPS:")
                print("   1. SSH into your instance")
                print("   2. Run your ML workload")
                print("   3. Stop when done to save money")
                print()
                print("💰 UPGRADE FOR MORE:")
                print("   - 8x more provisions/month")
                print("   - 8 parallel instances")
                print("   - Priority support")
                print("   terradev upgrade --tier research_plus")
                
                return True
            else:
                print("❌ Provision failed")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _save_credentials(self, provider: str, credentials: Dict[str, str]):
        """Save credentials for provider"""
        from terradev_cli.cli import TerradevAPI
        api = TerradevAPI()
        
        if not api.credentials:
            api.credentials = {}
        
        api.credentials[provider] = credentials
        api.save_credentials()
    
    def _quick_provision(self, api, gpu_type: str, provider: str) -> Dict[str, Any]:
        """Quick provision using existing logic"""
        try:
            # Get quotes for specific GPU and provider
            from terradev_cli.providers.provider_factory import ProviderFactory
            
            provider_factory = ProviderFactory()
            provider_instance = provider_factory.get_provider(provider, api.credentials.get(provider, {}))
            
            # Get single quote
            quotes = provider_instance.get_instance_quotes(gpu_type)
            
            if not quotes:
                return {}
            
            # Choose cheapest
            best_quote = min(quotes, key=lambda x: x.get('price', float('inf')))
            
            # Provision
            result = provider_instance.provision_instance(
                gpu_type=gpu_type,
                region=best_quote.get('region', ''),
                spot=best_quote.get('availability') == 'spot'
            )
            
            if result and result.get('success'):
                # Record provision
                api.record_provision()
                
                # Log to cost tracker
                try:
                    from terradev_cli.core.cost_tracker import record_provision
                    record_provision(
                        instance_id=result.get('instance_id'),
                        provider=provider,
                        gpu_type=gpu_type,
                        region=best_quote.get('region', ''),
                        price_hr=best_quote.get('price', 0),
                        spot=best_quote.get('availability') == 'spot'
                    )
                except:
                    pass
                
                return {
                    'instance_id': result.get('instance_id'),
                    'price': best_quote.get('price', 0),
                    'provider': provider,
                    'gpu_type': gpu_type
                }
            
            return {}
            
        except Exception as e:
            print(f"Provision error: {e}")
            return {}

def show_quick_start():
    """Show quick start guide"""
    quick_start = QuickStart()
    return quick_start.show_quick_start_guide()
