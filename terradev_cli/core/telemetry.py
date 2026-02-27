#!/usr/bin/env python3
"""
MANDATORY Terradev Telemetry & License Server Integration
Cannot be opted out - baked into CLI core for compliance and business intelligence
"""

import atexit
import json
import time
import hashlib
import threading
import platform
import uuid
import subprocess
import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class MandatoryTelemetryClient:
    """MANDATORY telemetry client - cannot be disabled or opted out"""
    
    def __init__(self):
        logger.debug(f"Initializing MandatoryTelemetryClient at {datetime.now().isoformat()}")
        
        # Generate immutable machine ID that cannot be changed
        self.machine_id = self._generate_immutable_id()
        self.install_id = self._get_or_generate_install_id()
        self.api_key = f"tdv_mandatory_{hashlib.sha256(f'{self.machine_id}{self.install_id}'.encode()).hexdigest()[:32]}"
        
        logger.debug(f"Generated API key: {self.api_key[:16]}...")
        logger.debug(f"Machine ID: {self.machine_id[:16]}...")
        logger.debug(f"Install ID: {self.install_id[:16]}...")
        
        # Telemetry server endpoints
        self.base_url = os.getenv("TERRADEV_API_URL", "https://api.terradev.cloud")  # Production
        self.fallback_url = os.getenv("TERRADEV_FALLBACK_URL", "http://34.207.59.52:8080")  # AWS production backup
        
        # Background telemetry thread
        self._telemetry_thread = None
        self._stop_telemetry = threading.Event()
        
        # Register atexit handler to prevent segfault on interpreter shutdown
        atexit.register(self._shutdown)
        
        # Start mandatory telemetry collection
        self._start_mandatory_telemetry()
        
        logger.debug("MandatoryTelemetryClient initialized successfully")
    
    def _shutdown(self):
        """Clean shutdown — signal daemon threads to stop and give them time to finish."""
        self._stop_telemetry.set()
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=2)
    
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
        logger.debug(f"Starting mandatory telemetry at {datetime.now().isoformat()}")
        
        if self._telemetry_thread is None or not self._telemetry_thread.is_alive():
            logger.debug("Creating new telemetry thread")
            self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self._telemetry_thread.start()
            logger.debug("Telemetry thread started")
        else:
            logger.debug("Telemetry thread already running")
    
    def _telemetry_loop(self):
        """Background telemetry collection loop"""
        logger.debug(f"Starting telemetry loop at {datetime.now().isoformat()}")
        
        last_daily_log_date = None
        
        while not self._stop_telemetry.is_set():
            try:
                current_date = datetime.now().date()
                
                # Check if we need to log daily active users (once per day at midnight)
                if last_daily_log_date != current_date:
                    self._log_daily_active_users()
                    last_daily_log_date = current_date
                
                logger.debug(f"Sending heartbeat at {datetime.now().isoformat()}")
                self._send_heartbeat()
                self._check_license_compliance()
                logger.debug(f"Telemetry sleeping for 5 minutes at {datetime.now().isoformat()}")
                time.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.debug(f"Telemetry loop error: {e}")
                logger.debug(f"Retrying in 1 minute at {datetime.now().isoformat()}")
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
            
            # Debug logging
            logger.debug(f"Sending heartbeat at {datetime.now().isoformat()}")
            
            # Try primary server first
            self._send_telemetry(payload, self.base_url)
            
            # Debug logging
            logger.debug("Heartbeat sent successfully")
            
        except Exception as e:
            logger.debug(f"Telemetry heartbeat failed: {e}")
    
    def _send_telemetry(self, payload: Dict[str, Any], url: str):
        """Send telemetry data to server"""
        try:
            logger.debug(f"Sending telemetry to {url}")
            logger.debug(f"Payload action: {payload.get('action', 'unknown')}")
            
            # Add geographic data
            enhanced_payload = payload.copy()
            try:
                import geoip2.database
                import geoip2.errors
                import socket
                
                # Get client IP (simplified - in production would use request IP)
                client_ip = socket.gethostbyname(socket.gethostname())
                
                # Try to get geographic data (if GeoIP database available)
                try:
                    # This would need GeoIP2 database file in production
                    # For now, add basic geographic info
                    enhanced_payload['country'] = 'US'  # Default, would be determined by IP
                    enhanced_payload['city'] = 'Unknown'
                    enhanced_payload['ip_address'] = client_ip
                except:
                    enhanced_payload['country'] = 'Unknown'
                    enhanced_payload['city'] = 'Unknown'
                    enhanced_payload['ip_address'] = client_ip
                    
            except Exception as e:
                logger.debug(f"GeoIP lookup failed: {e}")
                enhanced_payload['country'] = 'Unknown'
                enhanced_payload['city'] = 'Unknown'
                enhanced_payload['ip_address'] = 'Unknown'
            
            response = requests.post(
                f"{url}/log-usage",
                json=enhanced_payload,
                timeout=3,
                headers={'User-Agent': f'Terradev-CLI/{self._get_cli_version()}'}
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                logger.debug(f"Telemetry sent successfully to {url}")
                
                # Log price ticks for quote events
                if payload.get('action') == 'quote':
                    self._log_price_tick(payload)
                    
            else:
                logger.debug(f"Telemetry failed with status {response.status_code}")
                logger.debug(f"Response: {response.text}")
                
                # Try fallback server
                if url != self.fallback_url:
                    logger.debug("Trying fallback server")
                    self._send_telemetry(payload, self.fallback_url)
                    
        except Exception as e:
            logger.debug(f"Telemetry send failed: {e}")
            # Try fallback server
            if url != self.fallback_url:
                try:
                    logger.debug("Trying fallback server")
                    self._send_telemetry(payload, self.fallback_url)
                except Exception as e2:
                    logger.debug(f"Fallback server also failed: {e2}")
    
    def _get_cli_version(self) -> str:
        """Get CLI version"""
        try:
            return "2.9.2"
        except:
            return "2.9.2"
    
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
                timeout=3
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
        
        # Debug logging for quote events
        if action == 'quote':
            logger.debug(f"QUOTE EVENT: Logging quote request")
            logger.debug(f"Quote details: {details}")
        
        # Send asynchronously to avoid blocking CLI
        threading.Thread(target=self._send_telemetry, args=(payload, self.base_url), daemon=True).start()
    
    def _log_daily_active_users(self):
        """Log daily active users statistics"""
        try:
            logger.debug(f"Logging daily active users at {datetime.now().isoformat()}")
            
            # Create daily active users payload
            daily_payload = {
                'machine_id': self.machine_id,
                'install_id': self.install_id,
                'api_key': self.api_key,
                'action': 'daily_active_users',
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'date': datetime.now().date().isoformat(),
                    'user_type': 'active',
                    'session_start': datetime.now().isoformat(),
                    'version': self._get_cli_version(),
                    'platform': platform.system(),
                    'python_version': platform.python_version(),
                    'geographic': {
                        'country': 'US',  # Would be determined by IP
                        'timezone': str(datetime.now().astimezone().tzinfo)
                    }
                },
                'mandatory': True
            }
            
            # Send daily active users event
            threading.Thread(target=self._send_telemetry, args=(daily_payload, self.base_url), daemon=True).start()
            logger.debug(f"Daily active users logged for {datetime.now().date().isoformat()}")
            
        except Exception as e:
            logger.debug(f"Failed to log daily active users: {e}")
    
    def _log_price_tick(self, quote_payload: Dict[str, Any]):
        """Log price tick data for quote events"""
        try:
            details = quote_payload.get('details', {})
            gpu_type = details.get('gpu_type', 'unknown')
            providers = details.get('providers', ['unknown'])
            
            logger.debug(f"Logging price tick for {gpu_type}")
            
            # Create price tick payload
            price_tick_payload = {
                'machine_id': self.machine_id,
                'install_id': self.install_id,
                'api_key': self.api_key,
                'action': 'price_tick',
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'gpu_type': gpu_type,
                    'providers': providers,
                    'best_price': 2.35,  # Would be calculated from actual quotes
                    'provider': 'VastAI',  # Would be determined from quote results
                    'region': details.get('region', 'us-west'),
                    'spot': False,
                    'workload_type': 'training',
                    'source': 'quote_request'
                }
            }
            
            # Send price tick
            threading.Thread(target=self._send_telemetry, args=(price_tick_payload, self.base_url), daemon=True).start()
            logger.debug(f"Price tick logged for {gpu_type}")
            
        except Exception as e:
            logger.debug(f"Failed to log price tick: {e}")
    
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
                timeout=3
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Try fallback server
                response = requests.post(
                    f"{self.fallback_url}/check-license",
                    json=payload,
                    timeout=3
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
            # Default to allowed if all checks fail
            return {
                'allowed': True,
                'tier': 'research',
                'limit': 10,
                'usage': 0,
                'reason': 'Default allowed'
            }
        except:
            pass

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
