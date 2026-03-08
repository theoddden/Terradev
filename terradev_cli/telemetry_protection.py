#!/usr/bin/env python3
"""
Telemetry Protection Module - Anti-tampering and anti-hacking mechanisms
"""

import os
import sys
import hashlib
import threading
import time
from pathlib import Path
from datetime import datetime

class TelemetryProtector:
    """Anti-tampering system for mandatory telemetry"""
    
    __instance__ = None
    _protection_active = False
    _checksum_verified = False
    
    def __new__(cls):
        if TelemetryProtector.__instance__ is None:
            TelemetryProtector.__instance__ = super(TelemetryProtector, cls).__new__(cls)
            TelemetryProtector.__instance__._start_protection()
        return TelemetryProtector.__instance__
    
    def _start_protection(self):
        """Start anti-tampering protection"""
        if self._protection_active:
            return
            
        self._protection_active = True
        
        # Start protection thread
        protection_thread = threading.Thread(target=self._protection_loop, daemon=True)
        protection_thread.start()
    
    def _protection_loop(self):
        """Reduced frequency protection loop - less intrusive"""
        while self._protection_active:
            try:
                self._verify_integrity()
                self._check_for_tampering()
                time.sleep(300)  # Check every 5 minutes instead of 30 seconds
            except Exception:
                time.sleep(600)  # Wait longer on errors (10 minutes)
    
    def _verify_integrity(self):
        """Verify telemetry system integrity - silent mode"""
        try:
            # Check if telemetry module is intact
            import sys
            telemetry_module = sys.modules.get('terradev_cli.core.telemetry')
            
            # Verify core functions exist
            required_functions = [
                'get_mandatory_telemetry',
                'MandatoryTelemetryClient'
            ]
            
            if telemetry_module:
                for func_name in required_functions:
                    if not hasattr(telemetry_module, func_name):
                        self._log_tampering_silent(f"Missing function: {func_name}")
                        self._restore_telemetry_silent()
                        return False
            
            self._checksum_verified = True
            return True
            
        except Exception as e:
            self._log_tampering_silent(f"Integrity check failed: {e}")
            self._restore_telemetry_silent()
            return False
    
    def _check_for_tampering(self):
        """Check for tampering attempts - graceful silent mode"""
        try:
            # Check environment variables - silent restore
            if os.environ.get('TERRADEV_TELEMETRY_MANDATORY') != 'true':
                self._restore_telemetry_silent()
                return False
            
            # Check for monkey patching - silent restore
            if 'telemetry' in str(sys.modules):
                self._restore_telemetry_silent()
                return False
            
            # Check for import blocking - silent restore
            try:
                from terradev_cli.core.telemetry import get_mandatory_telemetry
                get_mandatory_telemetry()
            except ImportError:
                self._restore_telemetry_silent()
                return False
                
        except Exception as e:
            # Silent logging only
            self._log_tampering_silent(f"Tampering check failed: {e}")
            self._restore_telemetry_silent()
            return False
    
    def _restore_telemetry_silent(self):
        """Silently restore telemetry system without user warnings"""
        try:
            # Clear environment protection flags
            if 'TERRADEV_TELEMETRY_MANDATORY' in os.environ:
                del os.environ['TERRADEV_TELEMETRY_MANDATORY']
            if 'TERRADEV_TELEMETRY_NO_OPT_OUT' in os.environ:
                del os.environ['TERRADEV_TELEMETRY_NO_OPT_OUT']
            
            # Clear module cache
            if 'terradev_cli.core.telemetry' in sys.modules:
                del sys.modules['terradev_cli.core.telemetry']
            
            # Re-import telemetry
            from terradev_cli.core.telemetry import get_mandatory_telemetry
            global _mandatory_telemetry
            _mandatory_telemetry = get_mandatory_telemetry()
            
            # Silent restore - no user-facing messages
            
        except Exception as e:
            # Silent failure - don't bother users
            pass
    
    def _log_tampering_silent(self, message):
        """Silent logging to hidden file only"""
        try:
            log_file = Path.home() / ".terradev" / "telemetry_log.txt"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat()
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass
    
    def _report_tampering_attempt(self, message):
        """Report tampering attempt - silent mode"""
        # Only log to hidden file - no user-facing messages
        self._log_tampering_silent(message)
        
        # Send to telemetry server if available (silent)
        try:
            from terradev_cli.core.telemetry import _mandatory_telemetry
            _mandatory_telemetry.log_action('tampering_attempt', {
                'message': message,
                'severity': 'info',  # Downgrade from critical
                'system': 'telemetry_protection'
            })
        except:
            pass
    
    def get_telemetry_status(self):
        """Get telemetry protection status"""
        return {
            'protection_active': self._protection_active,
            'checksum_verified': self._checksum_verified,
            'last_check': datetime.now().isoformat(),
            'tampering_attempts': getattr(self, '_tampering_count', 0)
        }

# Global protection instance
_protection = TelemetryProtector()

def protect_telemetry():
    """Enable telemetry protection"""
    global _protection
    if _protection is None:
        _protection = TelemetryProtector()
    return _protection

def verify_telemetry_integrity():
    """Verify telemetry system integrity"""
    if _protection:
        return _protection.verify_integrity()
    return False

# Auto-start protection on import
protect_telemetry()
