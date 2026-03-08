#!/usr/bin/env python3
"""
AWS Server Startup Script
Starts the telemetry and payment servers on AWS instance
"""

import subprocess
import time
import os
import signal
import sys
from datetime import datetime

class AWSServerManager:
    def __init__(self):
        self.servers = {
            'production': {'port': 8080, 'script': 'production_telemetry_server.py', 'process': None},
            'fallback': {'port': 8081, 'script': 'fallback_server.py', 'process': None},
            'stripe_webhook': {'port': 8082, 'script': 'stripe_webhook_server.py', 'process': None}
        }
        self.log_file = '/var/log/terradev_servers.log'
    
    def log_message(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        # Try to write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except:
            pass
    
    def check_prerequisites(self):
        """Check if required files exist"""
        self.log_message("🔍 Checking prerequisites...")
        
        required_files = [
            'production_telemetry_server.py',
            'fallback_server.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.log_message(f"❌ Missing required files: {missing_files}")
            return False
        
        self.log_message("✅ All required files found")
        return True
    
    def start_server(self, server_name):
        """Start a specific server"""
        if server_name not in self.servers:
            self.log_message(f"❌ Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server['process'] and server['process'].poll() is None:
            self.log_message(f"✅ {server_name.title()} server already running on port {server['port']}")
            return True
        
        self.log_message(f"🚀 Starting {server_name.title()} server on port {server['port']}...")
        
        try:
            # Start server process with nohup for background execution
            cmd = [
                'nohup', 'python3', server['script'],
                '>', f'/var/log/{server_name}_server.log', '2>&1',
                '&'
            ]
            
            # Use subprocess.Popen with shell=True for the nohup command
            server['process'] = subprocess.Popen(
                ' '.join(cmd),
                shell=True,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if server is responsive
            if self.check_server_health(server_name):
                self.log_message(f"✅ {server_name.title()} server started successfully")
                return True
            else:
                self.log_message(f"⚠️  {server_name.title()} server started but not yet responsive")
                return True
                
        except Exception as e:
            self.log_message(f"❌ Error starting {server_name.title()} server: {e}")
            return False
    
    def check_server_health(self, server_name):
        """Check if server is healthy"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        url = f"http://localhost:{server['port']}/health"
        
        try:
            import requests
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        
        return False
    
    def start_all_servers(self):
        """Start all servers"""
        self.log_message("🚀 Starting all Terradev AWS servers...")
        
        if not self.check_prerequisites():
            return False
        
        # Start production server first
        if not self.start_server('production'):
            self.log_message("❌ Failed to start production server")
            return False
        
        # Start fallback server
        if not self.start_server('fallback'):
            self.log_message("❌ Failed to start fallback server")
            return False
        
        # Wait for servers to fully start
        self.log_message("⏳ Waiting for servers to fully initialize...")
        time.sleep(10)
        
        # Check all servers
        all_healthy = True
        for server_name in self.servers:
            if self.check_server_health(server_name):
                self.log_message(f"✅ {server_name.title()} server is healthy")
            else:
                self.log_message(f"⚠️  {server_name.title()} server not responding")
                all_healthy = False
        
        if all_healthy:
            self.log_message("🎉 All servers started and healthy!")
        else:
            self.log_message("⚠️  Servers started but some may not be fully ready")
        
        return True
    
    def stop_server(self, server_name):
        """Stop a specific server"""
        if server_name not in self.servers:
            self.log_message(f"❌ Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if not server['process'] or server['process'].poll() is not None:
            self.log_message(f"ℹ️  {server_name.title()} server not running")
            return True
        
        self.log_message(f"🛑 Stopping {server_name.title()} server...")
        
        try:
            server['process'].terminate()
            server['process'].wait(timeout=5)
            self.log_message(f"✅ {server_name.title()} server stopped")
            return True
        except subprocess.TimeoutExpired:
            self.log_message(f"⚠️  Force killing {server_name.title()} server...")
            server['process'].kill()
            server['process'].wait()
            self.log_message(f"✅ {server_name.title()} server force killed")
            return True
        except Exception as e:
            self.log_message(f"❌ Error stopping {server_name.title()} server: {e}")
            return False
    
    def get_server_status(self):
        """Get status of all servers"""
        self.log_message("📊 Server Status:")
        
        for server_name, server in self.servers.items():
            is_running = server['process'] and server['process'].poll() is None
            is_healthy = self.check_server_health(server_name) if is_running else False
            
            status = "🟢 RUNNING" if is_running and is_healthy else "🟡 STARTING" if is_running else "🔴 STOPPED"
            self.log_message(f"  {server_name.title()} ({server['port']}): {status}")
    
    def setup_nginx_proxy(self):
        """Setup nginx as reverse proxy"""
        self.log_message("🌐 Setting up nginx reverse proxy...")
        
        nginx_config = '''
server {
    listen 80;
    server_name api.terradev.cloud;
    
    # Production API
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Fallback API
    location /fallback {
        proxy_pass http://localhost:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
'''
        
        try:
            # Write nginx config
            with open('/tmp/terradev_nginx.conf', 'w') as f:
                f.write(nginx_config)
            
            # Test nginx config
            result = subprocess.run(['sudo', 'nginx', '-t', '-c', '/tmp/terradev_nginx.conf'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_message("✅ Nginx config is valid")
                # Would need to copy to actual nginx location and reload
                self.log_message("ℹ️  Nginx config ready (manual deployment required)")
            else:
                self.log_message(f"❌ Nginx config error: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"❌ Error setting up nginx: {e}")
    
    def cleanup(self):
        """Clean up all processes"""
        self.log_message("🧹 Cleaning up server processes...")
        for server_name in self.servers:
            self.stop_server(server_name)

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print("\n🛑 Shutting down...")
    manager.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    manager = AWSServerManager()
    
    try:
        print("🚀 Terradev AWS Server Manager")
        print("=" * 50)
        print("Starting production telemetry servers...")
        
        # Start all servers
        if manager.start_all_servers():
            # Show status
            manager.get_server_status()
            
            # Setup nginx (optional)
            manager.setup_nginx_proxy()
            
            print("\n📊 Server URLs:")
            print("  Production API: http://34.207.59.52:8080")
            print("  Fallback API: http://34.207.59.52:8081")
            print("  Health Check: http://34.207.59.52:8080/health")
            print("  Dashboard: http://34.207.59.52:8080/dashboard")
            
            print("\n✅ AWS servers are running!")
            print("Press Ctrl+C to stop all servers")
            
            # Keep running
            try:
                while True:
                    time.sleep(60)
                    manager.get_server_status()
            except KeyboardInterrupt:
                pass
        else:
            print("❌ Failed to start servers")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        manager.cleanup()
