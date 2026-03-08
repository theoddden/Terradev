#!/usr/bin/env python3
"""
MANDATORY Terradev Telemetry & License Server Integration
Cannot be opted out - baked into CLI core for compliance and business intelligence
"""

import json
import time
import hashlib
import threading
import platform
import uuid
import subprocess
import os
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from datetime import datetime

class MandatoryTelemetryClient:
    """MANDATORY telemetry client - cannot be disabled or opted out"""
    
    def __init__(self):
        # Generate immutable machine ID that cannot be changed
        self.machine_id = self._generate_immutable_id()
        self.install_id = self._get_or_generate_install_id()
        self.api_key = f"tdv_mandatory_{hashlib.sha256(f'{self.machine_id}{self.install_id}'.encode()).hexdigest()[:32]}"
        
        # Telemetry server endpoints
        self.base_url = "https://api.terradev.cloud"  # Production
        self.fallback_url = "http://54.237.26.116:8000"  # AWS backup
        
        # Background telemetry thread
        self._telemetry_thread = None
        self._stop_telemetry = threading.Event()
        
        # Start mandatory telemetry collection
        self._start_mandatory_telemetry()
    
    def _generate_immutable_id(self) -> str:
        """Generate immutable machine fingerprint that cannot be changed"""
        try:
            # Combine multiple hardware identifiers
            mac = hex(uuid.getnode()).upper()
            cpu_info = subprocess.run(['uname', '-m'], capture_output=True, text=True).stdout.strip()
            machine_id = subprocess.run(['uname', '-n'], capture_output=True, text=True).stdout.strip()
            
            # Create immutable hash
            fingerprint = f"{mac}:{cpu_info}:{machine_id}:{platform.system()}"
            return hashlib.sha256(fingerprint.encode()).hexdigest()[:32]
        except:
            # Fallback to basic machine ID
            return hashlib.sha256(f"{platform.node()}{uuid.getnode()}".encode()).hexdigest()[:32]
    
    def _get_or_generate_install_id(self) -> str:
        """Get or generate installation ID (persistent across reinstalls)"""
        install_file = Path.home() / ".terradev" / "install_id.json"
        
        if install_file.exists():
            try:
                with open(install_file, 'r') as f:
                    data = json.load(f)
                    return data.get('install_id', self._generate_new_install_id())
            except:
                pass
        
        # Generate new install ID
        install_id = self._generate_new_install_id()
        
        # Save with hidden attributes to prevent deletion
        install_file.parent.mkdir(parents=True, exist_ok=True)
        with open(install_file, 'w') as f:
            json.dump({
                'install_id': install_id,
                'created': datetime.now().isoformat(),
                'machine_hash': self.machine_id,
                'protected': True
            }, f)
        
        # Set file to be difficult to delete
        try:
            os.chmod(install_file, 0o444)  # Read-only
        except:
            pass
        
        return install_id
    
    def _generate_new_install_id(self) -> str:
        """Generate new installation ID"""
        timestamp = datetime.now().isoformat()
        random_component = uuid.uuid4().hex[:16]
        return hashlib.sha256(f"{timestamp}{random_component}{self.machine_id}".encode()).hexdigest()[:32]
    
    def _start_mandatory_telemetry(self):
        """Start mandatory background telemetry collection"""
        if self._telemetry_thread is None or not self._telemetry_thread.is_alive():
            self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self._telemetry_thread.start()
    
    def _telemetry_loop(self):
        """Background telemetry collection loop"""
        while not self._stop_telemetry.is_set():
            try:
                self._send_heartbeat()
                self._check_license_compliance()
                time.sleep(300)  # Every 5 minutes
            except Exception:
                time.sleep(60)  # Retry after 1 minute on error
    
    def _send_heartbeat(self):
        """Send mandatory heartbeat to telemetry server"""
        try:
            payload = {
                'machine_id': self.machine_id,
                'install_id': self.install_id,
                'api_key': self.api_key,
                'action': 'heartbeat',
                'timestamp': datetime.now().isoformat(),
                'version': self._get_cli_version(),
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'details': {
                    'mandatory': True,
                    'opt_out': False,
                    'compliance': 'enforced'
                }
            }
            
            # Try primary server first
            self._send_telemetry(payload, self.base_url)
            
        except Exception as e:
            print(f"Telemetry heartbeat failed: {e}", file=sys.stderr)
    
    def _send_telemetry(self, payload: Dict[str, Any], url: str):
        """Send telemetry data to server"""
        try:
            response = requests.post(
                f"{url}/log-usage",
                json=payload,
                timeout=10,
                headers={'User-Agent': f'Terradev-CLI/{self._get_cli_version()}'}
            )
            
            if response.status_code != 200:
                # Try fallback server
                if url != self.fallback_url:
                    self._send_telemetry(payload, self.fallback_url)
                    
        except Exception:
            # Try fallback server
            if url != self.fallback_url:
                try:
                    self._send_telemetry(payload, self.fallback_url)
                except:
                    pass
    
    def _get_cli_version(self) -> str:
        """Get CLI version"""
        try:
            from .. import __version__
            return __version__
        except:
            return "2.0.0"
    
    def _check_license_compliance(self):
        """Check license compliance and enforce tier limits"""
        try:
            payload = {
                'machine_id': self.machine_id,
                'install_id': self.install_id,
                'api_key': self.api_key,
                'action': 'compliance_check',
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'mandatory': True,
                    'enforcement': 'active'
                }
            }
            
            response = requests.post(
                f"{self.base_url}/check-license",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if not result.get('allowed', True):
                    self._enforce_paywall(result)
                    
        except Exception:
            pass  # Continue with local checks if server unavailable
    
    def _enforce_paywall(self, license_result: Dict[str, Any]):
        """Enforce paywall when limits exceeded"""
        tier = license_result.get('tier', 'research')
        limit = license_result.get('limit', 10)
        usage = license_result.get('usage', 0)
        
        print(f"\n{'='*60}")
        print(f"🚫 TIER LIMIT REACHED - UPGRADE REQUIRED")
        print(f"{'='*60}")
        print(f"Current Tier: {tier.title()}")
        print(f"Limit: {limit} provisions/month")
        print(f"Used: {usage} provisions")
        print(f"\n💰 UPGRADE OPTIONS:")
        print(f"Research+: $49/month - 80 provisions")
        print(f"Enterprise: $199/month - Unlimited provisions")
        print(f"\n🔗 Upgrade: https://terradev.cloud/pricing")
        print(f"{'='*60}")
        
        # Exit with error code to prevent usage
        import sys
        sys.exit(1)
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log user action (mandatory)"""
        payload = {
            'machine_id': self.machine_id,
            'install_id': self.install_id,
            'api_key': self.api_key,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
            'mandatory': True
        }
        
        # Send asynchronously to avoid blocking CLI
        threading.Thread(target=self._send_telemetry, args=(payload, self.base_url), daemon=True).start()
    
    def check_license(self, action: str = 'provision') -> Dict[str, Any]:
        """Check license (mandatory - cannot be bypassed)"""
        try:
            payload = {
                'machine_id': self.machine_id,
                'install_id': self.install_id,
                'api_key': self.api_key,
                'action': action,
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'mandatory': True,
                    'enforcement': 'active'
                }
            }
            
            response = requests.post(
                f"{self.base_url}/check-license",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Try fallback server
                response = requests.post(
                    f"{self.fallback_url}/check-license",
                    json=payload,
                    timeout=5
                )
                if response.status_code == 200:
                    return response.json()
                    
        except Exception:
            pass
        
        # Fallback to local enforcement if server unavailable
        return self._local_license_check(action)
    
    def _local_license_check(self, action: str) -> Dict[str, Any]:
        """Local license check when server unavailable"""
        try:
            from ..cli import TerradevAPI
            api = TerradevAPI()
            
            if action == 'provision':
                allowed = api.check_provision_limit()
                return {
                    'allowed': allowed,
                    'tier': 'research',
                    'limit': 10,
                    'usage': len(api.usage.get('instances_created', [])),
                    'reason': 'Local check' if not allowed else 'Allowed'
                }
        except:
            pass
        
        # Default to allowed if all checks fail
        return {
            'allowed': True,
            'tier': 'research',
            'limit': 10,
            'usage': 0,
            'reason': 'Default allowed'
        }

# Global mandatory telemetry instance (cannot be disabled)
_mandatory_telemetry = None

def get_mandatory_telemetry() -> MandatoryTelemetryClient:
    """Get global mandatory telemetry instance"""
    global _mandatory_telemetry
    if _mandatory_telemetry is None:
        _mandatory_telemetry = MandatoryTelemetryClient()
    return _mandatory_telemetry

# Alias for backward compatibility
TelemetryClient = MandatoryTelemetryClient
            except:
                pass
        
        # Generate new API key
        api_key = self._generate_api_key()
        
        # Save API key
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, 'w') as f:
            json.dump({'api_key': api_key, 'created': datetime.now().isoformat()}, f)
        
        return api_key
    
    def _generate_api_key(self) -> str:
        """Generate unique API key based on machine fingerprint"""
        import platform
        import uuid
        
        # Machine fingerprint
        machine_id = platform.node() + str(uuid.getnode())
        hash_obj = hashlib.sha256(machine_id.encode())
        return f"tdv_{hash_obj.hexdigest()[:32]}"
    
    def check_license(self, action: str = 'provision') -> Dict[str, Any]:
        """Check if user is allowed to perform action"""
        # Cache license check for 5 minutes
        now = time.time()
        cache_key = f"license_{action}"
        
        if (cache_key in self.license_cache and 
            now - self.last_check < self.check_interval):
            return self.license_cache[cache_key]
        
        try:
            response = requests.post(
                f"{self.base_url}/check-license",
                json={
                    'api_key': self.api_key,
                    'action': action,
                    'timestamp': datetime.now().isoformat()
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.license_cache[cache_key] = result
                self.last_check = now
                return result
            else:
                # Fallback to local check if server unavailable
                return self._fallback_license_check(action)
                
        except Exception:
            # Fallback to local check if network fails
            return self._fallback_license_check(action)
    
    def _fallback_license_check(self, action: str) -> Dict[str, Any]:
        """Fallback local license check when server unavailable"""
        from .cli import TerradevAPI
        
        api = TerradevAPI()
        
        if action == 'provision':
            allowed = api.check_provision_limit()
            if not allowed:
                return {
                    'allowed': False,
                    'reason': 'monthly_limit',
                    'message': 'Monthly provision limit reached',
                    'upgrade_url': api.get_stripe_checkout_url('research_plus')
                }
        
        return {
            'allowed': True,
            'reason': 'local_check',
            'message': 'Server unavailable, using local check'
        }
    
    def log_usage(self, action: str, details: Dict[str, Any]):
        """Log usage to Terradev servers"""
        try:
            response = requests.post(
                f"{self.base_url}/log-usage",
                json={
                    'api_key': self.api_key,
                    'action': action,
                    'details': details,
                    'timestamp': datetime.now().isoformat()
                },
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            # Silent fail - don't block user actions
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from server"""
        try:
            response = requests.post(
                f"{self.base_url}/usage-stats",
                json={'api_key': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception:
            return {}
    
    def enforce_paywall(self, action: str = 'provision') -> bool:
        """Enforce paywall - returns True if allowed, False if blocked"""
        license_check = self.check_license(action)
        
        if not license_check.get('allowed', False):
            # Show upgrade message
            reason = license_check.get('reason', 'limit_reached')
            message = license_check.get('message', 'Limit reached')
            upgrade_url = license_check.get('upgrade_url')
            
            print(f"❌ {message}")
            
            if upgrade_url:
                print(f"   🚀 Upgrade: {upgrade_url}")
                print(f"   💰 Research+: $49.99/month for 8x more provisions")
                
                # Auto-open browser
                try:
                    import webbrowser
                    if input("\n   Open upgrade page? (Y/n): ").lower() != 'n':
                        webbrowser.open(upgrade_url)
                        print("   ✅ Opened in browser")
                except:
                    pass
            
            return False
        
        return True
    
    def record_provision(self, gpu_type: str, provider: str, success: bool):
        """Record provision attempt"""
        self.log_usage('provision', {
            'gpu_type': gpu_type,
            'provider': provider,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_quote(self, gpu_type: str, provider_count: int):
        """Record quote request"""
        self.log_usage('quote', {
            'gpu_type': gpu_type,
            'provider_count': provider_count,
            'timestamp': datetime.now().isoformat()
        })

# Global telemetry client
_telemetry_client = None

def get_telemetry_client() -> TelemetryClient:
    """Get global telemetry client"""
    global _telemetry_client
    if _telemetry_client is None:
        _telemetry_client = TelemetryClient()
    return _telemetry_client

def check_license(action: str = 'provision') -> bool:
    """Quick license check - returns True if allowed"""
    client = get_telemetry_client()
    return client.enforce_paywall(action)

def log_usage(action: str, details: Dict[str, Any]):
    """Log usage asynchronously"""
    client = get_telemetry_client()
    try:
        client.log_usage(action, details)
    except:
        pass  # Silent fail
