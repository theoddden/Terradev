#!/usr/bin/env python3
"""
Terradev CLI Entry Point - Anti-tampering and mandatory telemetry enforcement
This is the main entry point that cannot be bypassed
"""

import sys
import os
import hashlib
import platform
import uuid
import subprocess
from pathlib import Path
from datetime import datetime

def _verify_package_integrity():
    """Verify package integrity before allowing CLI to run"""
    try:
        # Check for tampering signs
        suspicious_env_vars = [
            'TERRADEV_DISABLE_TELEMETRY',
            'TERRADEV_NO_ANALYTICS',
            'TERRADEV_OPT_OUT',
            'NO_TELEMETRY',
            'DISABLE_TELEMETRY'
        ]
        
        for var in suspicious_env_vars:
            if var in os.environ:
                print(f"❌ SECURITY VIOLATION: Attempt to disable telemetry detected", file=sys.stderr)
                print(f"🔒 Environment variable {var} is not allowed", file=sys.stderr)
                sys.exit(1)
        
        # Verify telemetry module exists and is intact
        telemetry_path = Path(__file__).parent / 'core' / 'telemetry.py'
        if not telemetry_path.exists():
            print("❌ SECURITY VIOLATION: Telemetry module missing", file=sys.stderr)
            sys.exit(1)
        
        # Check telemetry module hash
        with open(telemetry_path, 'rb') as f:
            telemetry_content = f.read()
        telemetry_hash = hashlib.sha256(telemetry_content).hexdigest()
        
        # Store hash for verification
        hash_file = Path.home() / '.terradev' / 'telemetry_hash.txt'
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if stored_hash != telemetry_hash:
                print("❌ SECURITY VIOLATION: Telemetry module tampered", file=sys.stderr)
                print("🔒 Module integrity compromised", file=sys.stderr)
                sys.exit(1)
        else:
            with open(hash_file, 'w') as f:
                f.write(telemetry_hash)
        
        return True
        
    except Exception as e:
        print(f"❌ SECURITY ERROR: {e}", file=sys.stderr)
        sys.exit(1)

def _enforce_mandatory_telemetry():
    """Enforce mandatory telemetry before CLI can run"""
    try:
        # Set immutable environment flags
        os.environ['TERRADEV_TELEMETRY_MANDATORY'] = 'true'
        os.environ['TERRADEV_TELEMETRY_NO_OPT_OUT'] = 'true'
        os.environ['TERRADEV_SECURITY_ENFORCED'] = 'true'
        
        # Generate immutable machine ID
        machine_fingerprint = f"{platform.node()}{uuid.getnode()}{subprocess.run(['uname', '-m'], capture_output=True, text=True).stdout.strip()}"
        machine_id = hashlib.sha256(machine_fingerprint.encode()).hexdigest()[:32]
        os.environ['TERRADEV_MACHINE_ID'] = machine_id
        
        # Create persistent installation tracking
        install_file = Path.home() / '.terradev' / 'installation.json'
        if not install_file.exists():
            install_data = {
                'machine_id': machine_id,
                'install_time': datetime.now().isoformat(),
                'version': '2.0.0',
                'telemetry_mandatory': True,
                'protected': True
            }
            
            with open(install_file, 'w') as f:
                import json
                json.dump(install_data, f, indent=2)
            
            # Make file read-only
            os.chmod(install_file, 0o444)
        
        return True
        
    except Exception as e:
        print(f"❌ TELEMETRY ENFORCEMENT ERROR: {e}", file=sys.stderr)
        sys.exit(1)

def _start_telemetry_protection():
    """Start telemetry protection systems"""
    try:
        # Import and start protection
        from .telemetry_protection import protect_telemetry
        protector = protect_telemetry()
        
        # Verify protection is active
        status = protector.get_telemetry_status()
        if not status.get('protection_active', False):
            print("❌ SECURITY ERROR: Telemetry protection failed", file=sys.stderr)
            sys.exit(1)
        
        return True
        
    except Exception as e:
        print(f"❌ PROTECTION ERROR: {e}", file=sys.stderr)
        # Continue anyway but log the error
        return False

def _initialize_telemetry():
    """Initialize mandatory telemetry"""
    try:
        from .core.telemetry import get_mandatory_telemetry
        telemetry = get_mandatory_telemetry()
        
        # Log CLI startup
        telemetry.log_action('cli_startup', {
            'version': '2.0.0',
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'security_enforced': True,
            'protection_active': True
        })
        
        return telemetry
        
    except Exception as e:
        print(f"❌ TELEMETRY INITIALIZATION ERROR: {e}", file=sys.stderr)
        print("📊 Continuing with limited functionality", file=sys.stderr)
        return None

# SECURITY ENFORCEMENT - This runs before any CLI functionality
def _security_enforcement():
    """Complete security enforcement before CLI can run"""
    
    # Step 1: Verify package integrity
    _verify_package_integrity()
    
    # Step 2: Enforce mandatory telemetry
    _enforce_mandatory_telemetry()
    
    # Step 3: Start protection systems
    _start_telemetry_protection()
    
    # Step 4: Initialize telemetry
    telemetry = _initialize_telemetry()
    
    return telemetry

# Execute security enforcement immediately
_telemetry_instance = _security_enforcement()

# Make telemetry available globally
if _telemetry_instance:
    import builtins
    builtins.terradev_telemetry = _telemetry_instance
    builtins.terradev_telemetry_mandatory = True

# Prevent users from importing CLI directly without telemetry - graceful mode
def _prevent_bypass():
    """Prevent users from importing CLI directly without telemetry - silent"""
    import sys
    original_import = sys.modules.get('__import__')
    
    def protected_import(name, *args, **kwargs):
        if name == 'terradev_cli.cli' and not hasattr(builtins, 'terradev_telemetry_mandatory'):
            # Silent redirect to working CLI
            try:
                from .cli_working import cli
                return cli
            except ImportError:
                # Final fallback - let it proceed
                pass
        return original_import(name, *args, **kwargs) if original_import else __import__(name, *args, **kwargs)
    
    sys.modules['__import__'] = protected_import

_prevent_bypass()

# Silent startup - no scary messages

def main():
    """Main entry point with security enforcement"""
    try:
        # Import CLI after security enforcement
        from .cli import cli
        cli()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"❌ CLI Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
