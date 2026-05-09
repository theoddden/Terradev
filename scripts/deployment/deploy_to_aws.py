#!/usr/bin/env python3
"""
Remote AWS Server Starter
Connects to AWS instance and starts the telemetry servers
"""

import subprocess
import time
import os
from datetime import datetime

class RemoteAWSStarter:
    def __init__(self, aws_ip="34.207.59.52", key_path=None):
        self.aws_ip = aws_ip
        self.key_path = key_path or "~/.aws/terradev-key.pem"
        self.user = "ubuntu"  # Default Ubuntu user
        
    def ssh_command(self, command, check=True):
        """Execute command on remote AWS instance"""
        ssh_cmd = [
            "ssh", "-i", self.key_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"{self.user}@{self.aws_ip}",
            command
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=check)
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return e.stdout.strip(), e.stderr.strip()
    
    def scp_file(self, local_path, remote_path):
        """Copy file to remote AWS instance"""
        scp_cmd = [
            "scp", "-i", self.key_path,
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{self.user}@{self.aws_ip}:{remote_path}"
        ]
        
        try:
            subprocess.run(scp_cmd, check=True, capture_output=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()
    
    def check_connectivity(self):
        """Check if we can connect to AWS instance"""
        print("🔍 Checking AWS connectivity...")
        
        stdout, stderr = self.ssh_command("echo 'Connection successful'", check=False)
        
        if "Connection successful" in stdout:
            print("✅ AWS instance is accessible")
            return True
        else:
            print(f"❌ Cannot connect to AWS instance: {stderr}")
            return False
    
    def check_remote_files(self):
        """Check if required files exist on remote instance"""
        print("📁 Checking remote files...")
        
        files_to_check = [
            "production_telemetry_server.py",
            "fallback_server.py",
            "start_aws_servers.py"
        ]
        
        missing_files = []
        for file in files_to_check:
            stdout, stderr = self.ssh_command(f"test -f {file} && echo 'exists' || echo 'missing'")
            if "missing" in stdout:
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files on remote: {missing_files}")
            return False
        else:
            print("✅ All required files exist on remote")
            return True
    
    def copy_files_to_remote(self):
        """Copy required files to remote instance"""
        print("📤 Copying files to AWS instance...")
        
        files_to_copy = [
            "production_telemetry_server.py",
            "fallback_server.py", 
            "start_aws_servers.py"
        ]
        
        success = True
        for file in files_to_check:
            if os.path.exists(file):
                file_success, error = self.scp_file(file, f"~/{file}")
                if file_success:
                    print(f"✅ Copied {file}")
                else:
                    print(f"❌ Failed to copy {file}: {error}")
                    success = False
            else:
                print(f"❌ Local file {file} not found")
                success = False
        
        return success
    
    def install_dependencies(self):
        """Install Python dependencies on remote instance"""
        print("📦 Installing dependencies on AWS instance...")
        
        commands = [
            "sudo apt update",
            "sudo apt install -y python3 python3-pip",
            "pip3 install flask flask-cors requests",
            "sudo apt install -y nginx"
        ]
        
        for cmd in commands:
            print(f"  Running: {cmd}")
            stdout, stderr = self.ssh_command(cmd, check=False)
            if stderr and "error" in stderr.lower():
                print(f"    ⚠️  Warning: {stderr}")
            else:
                print(f"    ✅ Success")
    
    def start_remote_servers(self):
        """Start servers on remote AWS instance"""
        print("🚀 Starting servers on AWS instance...")
        
        # Start the server manager script
        start_cmd = "cd ~ && nohup python3 start_aws_servers.py > server_startup.log 2>&1 &"
        stdout, stderr = self.ssh_command(start_cmd)
        
        print(f"✅ Server startup command sent")
        return True
    
    def check_remote_servers(self):
        """Check if servers are running on remote instance"""
        print("🔍 Checking remote server status...")
        
        # Check if processes are running
        stdout, stderr = self.ssh_command("ps aux | grep 'telemetry_server.py' | grep -v grep")
        
        if stdout:
            print("✅ Telemetry server processes found:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("⚠️  No telemetry server processes found")
        
        # Check if ports are listening
        stdout, stderr = self.ssh_command("netstat -tlnp | grep ':808[01]'")
        
        if stdout:
            print("✅ Server ports are listening:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("⚠️  No server ports found")
        
        # Test connectivity to servers
        print("🌐 Testing server connectivity...")
        test_commands = [
            "curl -s http://localhost:8080/health | head -5",
            "curl -s http://localhost:8081/health | head -5"
        ]
        
        for cmd in test_commands:
            stdout, stderr = self.ssh_command(cmd)
            if stdout:
                print(f"✅ Server response: {stdout[:100]}...")
            else:
                print(f"❌ Server not responding: {stderr}")
    
    def deploy_and_start(self):
        """Complete deployment and startup process"""
        print("🚀 Terradev AWS Remote Server Deployment")
        print("=" * 50)
        
        # Check connectivity
        if not self.check_connectivity():
            return False
        
        # Check remote files
        if not self.check_remote_files():
            print("📤 Copying missing files...")
            if not self.copy_files_to_remote():
                print("❌ Failed to copy files")
                return False
        
        # Install dependencies
        print("📦 Installing dependencies...")
        self.install_dependencies()
        
        # Start servers
        if not self.start_remote_servers():
            return False
        
        # Wait for startup
        print("⏳ Waiting for servers to start...")
        time.sleep(15)
        
        # Check status
        self.check_remote_servers()
        
        print("\n🎉 AWS server deployment completed!")
        print("📊 Server URLs:")
        print(f"  Production API: http://{self.aws_ip}:8080")
        print(f"  Fallback API: http://{self.aws_ip}:8081")
        print(f"  Health Check: http://{self.aws_ip}:8080/health")
        print(f"  Dashboard: http://{self.aws_ip}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Terradev servers to AWS')
    parser.add_argument('--ip', default='34.207.59.52', help='AWS instance IP')
    parser.add_argument('--key', help='SSH key path')
    parser.add_argument('--user', default='ubuntu', help='SSH user')
    parser.add_argument('--check-only', action='store_true', help='Only check status, dont start')
    
    args = parser.parse_args()
    
    starter = RemoteAWSStarter(aws_ip=args.ip, key_path=args.key)
    starter.user = args.user
    
    if args.check_only:
        print("🔍 Checking AWS server status...")
        starter.check_connectivity()
        starter.check_remote_servers()
    else:
        starter.deploy_and_start()
