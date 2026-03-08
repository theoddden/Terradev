#!/usr/bin/env python3
"""
Terradev Server Management Script
Manages telemetry backend servers and provides status updates
"""

import subprocess
import time
import requests
import signal
import sys
import os
from datetime import datetime

class TerradevServerManager:
    def __init__(self):
        self.servers = {
            'production': {'port': 8080, 'script': 'production_telemetry_server.py', 'process': None},
            'fallback': {'port': 8081, 'script': 'fallback_server.py', 'process': None}
        }
        self.status_cache = {}
    
    def start_server(self, server_name):
        """Start a specific server"""
        if server_name not in self.servers:
            print(f"❌ Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server['process'] and server['process'].poll() is None:
            print(f"✅ {server_name.title()} server already running on port {server['port']}")
            return True
        
        print(f"🚀 Starting {server_name.title()} server...")
        
        try:
            # Start server process
            server['process'] = subprocess.Popen(
                [sys.executable, server['script']],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is responsive
            if self.check_server_health(server_name):
                print(f"✅ {server_name.title()} server started successfully on port {server['port']}")
                return True
            else:
                print(f"❌ {server_name.title()} server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting {server_name.title()} server: {e}")
            return False
    
    def stop_server(self, server_name):
        """Stop a specific server"""
        if server_name not in self.servers:
            print(f"❌ Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if not server['process'] or server['process'].poll() is not None:
            print(f"ℹ️  {server_name.title()} server not running")
            return True
        
        print(f"🛑 Stopping {server_name.title()} server...")
        
        try:
            server['process'].terminate()
            server['process'].wait(timeout=5)
            print(f"✅ {server_name.title()} server stopped")
            return True
        except subprocess.TimeoutExpired:
            print(f"⚠️  Force killing {server_name.title()} server...")
            server['process'].kill()
            server['process'].wait()
            print(f"✅ {server_name.title()} server force killed")
            return True
        except Exception as e:
            print(f"❌ Error stopping {server_name.title()} server: {e}")
            return False
    
    def check_server_health(self, server_name):
        """Check if server is healthy"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        url = f"http://localhost:{server['port']}/health"
        
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.status_cache[server_name] = {
                    'healthy': True,
                    'status': data.get('status', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                return True
        except:
            pass
        
        self.status_cache[server_name] = {
            'healthy': False,
            'status': 'unreachable',
            'timestamp': datetime.now().isoformat()
        }
        return False
    
    def get_server_stats(self, server_name):
        """Get detailed server statistics"""
        if server_name not in self.servers:
            return None
        
        server = self.servers[server_name]
        url = f"http://localhost:{server['port']}/user-stats"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return None
    
    def show_status(self):
        """Show status of all servers"""
        print("🚀 Terradev Server Status")
        print("=" * 50)
        
        for server_name, server in self.servers.items():
            is_healthy = self.check_server_health(server_name)
            
            if is_healthy:
                status = "🟢 RUNNING"
                stats = self.get_server_stats(server_name)
                if stats:
                    print(f"{server_name.title()} ({server['port']}): {status}")
                    print(f"  Users: {stats.get('total_users', 0)}")
                    print(f"  Active Today: {stats.get('active_users_today', 0)}")
                    print(f"  Installations: {stats.get('total_installations', 0)}")
                else:
                    print(f"{server_name.title()} ({server['port']}): {status}")
            else:
                status = "🔴 STOPPED"
                print(f"{server_name.title()} ({server['port']}): {status}")
        
        print()
    
    def start_all(self):
        """Start all servers"""
        print("🚀 Starting all Terradev servers...")
        
        # Start production server first
        if self.start_server('production'):
            # Start fallback server
            self.start_server('fallback')
        
        self.show_status()
    
    def stop_all(self):
        """Stop all servers"""
        print("🛑 Stopping all Terradev servers...")
        
        for server_name in self.servers:
            self.stop_server(server_name)
        
        self.show_status()
    
    def restart_server(self, server_name):
        """Restart a specific server"""
        print(f"🔄 Restarting {server_name.title()} server...")
        self.stop_server(server_name)
        time.sleep(2)
        self.start_server(server_name)
        self.show_status()
    
    def test_integration(self):
        """Test CLI integration with servers"""
        print("🧪 Testing CLI Integration...")
        
        # Test production server
        if self.check_server_health('production'):
            print("✅ Production server reachable")
            
            # Send test telemetry
            try:
                test_data = {
                    'machine_id': 'test_integration',
                    'install_id': 'test_integration',
                    'action': 'integration_test',
                    'timestamp': datetime.now().isoformat(),
                    'details': {'test': True}
                }
                
                response = requests.post(
                    f"http://localhost:{self.servers['production']['port']}/log-usage",
                    json=test_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    print("✅ Telemetry logging works")
                else:
                    print("❌ Telemetry logging failed")
                    
            except Exception as e:
                print(f"❌ Integration test error: {e}")
        else:
            print("❌ Production server not reachable")
    
    def cleanup(self):
        """Clean up all processes"""
        print("🧹 Cleaning up...")
        self.stop_all()

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print("\n🛑 Shutting down...")
    manager.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Terradev Server Manager')
    parser.add_argument('command', choices=[
        'start', 'stop', 'restart', 'status', 'test', 'start-all', 'stop-all'
    ], help='Command to execute')
    parser.add_argument('--server', choices=['production', 'fallback'], 
                       help='Specific server to operate on')
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    manager = TerradevServerManager()
    
    try:
        if args.command == 'start':
            if args.server:
                manager.start_server(args.server)
            else:
                manager.start_all()
        
        elif args.command == 'stop':
            if args.server:
                manager.stop_server(args.server)
            else:
                manager.stop_all()
        
        elif args.command == 'restart':
            if args.server:
                manager.restart_server(args.server)
            else:
                manager.restart_server('production')
                manager.restart_server('fallback')
        
        elif args.command == 'status':
            manager.show_status()
        
        elif args.command == 'test':
            manager.test_integration()
        
        elif args.command == 'start-all':
            manager.start_all()
        
        elif args.command == 'stop-all':
            manager.stop_all()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        manager.cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.cleanup()
