#!/usr/bin/env python3
"""
Direct SSH Server Starter
Uses SSH to directly connect and start servers
"""

import subprocess
import time
import os
from datetime import datetime

class DirectSSHStarter:
    def __init__(self, ip='54.237.26.116', user='ubuntu', key_path=None):
        self.ip = ip
        self.user = user
        self.key_path = key_path or '~/.ssh/id_rsa'
        
    def ssh_command(self, command, timeout=60):
        """Execute command via SSH"""
        ssh_cmd = [
            'ssh', '-i', self.key_path,
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=30',
            f'{self.user}@{self.ip}',
            command
        ]
        
        try:
            print(f"📤 SSH: {command[:50]}...")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    print(f"✅ {output}")
                return True
            else:
                error = result.stderr.strip()
                if error:
                    print(f"❌ Error: {error}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ SSH command timeout")
            return False
        except Exception as e:
            print(f"❌ SSH error: {e}")
            return False
    
    def scp_file(self, local_file, remote_path):
        """Copy file to remote server"""
        scp_cmd = [
            'scp', '-i', self.key_path,
            '-o', 'StrictHostKeyChecking=no',
            local_file,
            f'{self.user}@{self.ip}:{remote_path}'
        ]
        
        try:
            print(f"📤 SCP: {local_file} -> {remote_path}")
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ File copied successfully")
                return True
            else:
                error = result.stderr.strip()
                if error:
                    print(f"❌ SCP error: {error}")
                return False
                
        except Exception as e:
            print(f"❌ SCP error: {e}")
            return False
    
    def test_ssh_connectivity(self):
        """Test SSH connectivity"""
        print("🔍 Testing SSH connectivity...")
        
        if self.ssh_command("echo 'SSH connection successful'", timeout=10):
            print("✅ SSH connection works")
            return True
        else:
            print("❌ SSH connection failed")
            return False
    
    def start_servers(self):
        """Start telemetry servers via SSH"""
        print("🚀 Starting telemetry servers via SSH...")
        
        # Setup commands
        commands = [
            ("cd /home/ubuntu && pwd", "Check directory"),
            ("sudo apt update -y", "Update packages"),
            ("sudo apt install -y python3 python3-pip curl", "Install Python"),
            ("pip3 install flask flask-cors requests", "Install dependencies"),
            ("mkdir -p /home/ubuntu/terradev && cd /home/ubuntu/terradev", "Create directory")
        ]
        
        for cmd, desc in commands:
            print(f"  {desc}")
            if not self.ssh_command(cmd, timeout=120):
                print(f"❌ Failed: {desc}")
        
        # Copy server files
        print("📤 Copying server files...")
        
        # Create production server
        prod_server = '''#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS telemetry_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT, action TEXT, timestamp TEXT, details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0-aws', 'server': 'production'})

@app.route('/log-usage', methods=['POST'])
def log_usage():
    try:
        data = request.get_json()
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO telemetry_logs (machine_id, action, timestamp, details) VALUES (?, ?, ?, ?)", (data.get('machine_id'), data.get('action'), data.get('timestamp'), json.dumps(data.get('details', {}))))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Telemetry logged successfully', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def user_stats():
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT machine_id) FROM telemetry_logs')
    total_users = cursor.fetchone()[0] or 0
    conn.close()
    return jsonify({'total_users': total_users, 'active_users_today': 0, 'server_info': {'version': '1.0.0-aws', 'timestamp': datetime.now().isoformat()}})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8080)
'''
        
        # Write production server remotely
        write_prod_cmd = f"cat > /home/ubuntu/terradev/production_server.py << 'EOF'\n{prod_server}\nEOF"
        if not self.ssh_command(write_prod_cmd, timeout=30):
            print("❌ Failed to create production server")
            return False
        
        # Create fallback server
        fallback_server = '''#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('telemetry_fallback.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS telemetry_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT, action TEXT, timestamp TEXT, details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0-fallback', 'server': 'fallback'})

@app.route('/log-usage', methods=['POST'])
def log_usage():
    try:
        data = request.get_json()
        conn = sqlite3.connect('telemetry_fallback.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO telemetry_logs (machine_id, action, timestamp, details) VALUES (?, ?, ?, ?)", (data.get('machine_id'), data.get('action'), data.get('timestamp'), json.dumps(data.get('details', {}))))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Telemetry logged successfully (fallback)', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8081)
'''
        
        # Write fallback server remotely
        write_fallback_cmd = f"cat > /home/ubuntu/terradev/fallback_server.py << 'EOF'\n{fallback_server}\nEOF"
        if not self.ssh_command(write_fallback_cmd, timeout=30):
            print("❌ Failed to create fallback server")
            return False
        
        # Start servers
        start_commands = [
            ("cd /home/ubuntu/terradev && chmod +x *.py", "Make executable"),
            ("cd /home/ubuntu/terradev && nohup python3 production_server.py > production.log 2>&1 &", "Start production"),
            ("cd /home/ubuntu/terradev && nohup python3 fallback_server.py > fallback.log 2>&1 &", "Start fallback"),
            ("sleep 15", "Wait for startup"),
            ("ps aux | grep python3 | grep server || echo 'No processes'", "Check processes"),
            ("netstat -tlnp | grep ':808[01]' || echo 'Ports not bound'", "Check ports"),
            ("curl -s http://localhost:8080/health || echo 'Production not ready'", "Test production"),
            ("curl -s http://localhost:8081/health || echo 'Fallback not ready'", "Test fallback"),
            ("sudo ufw allow 8080 8081 || echo 'UFW not available'", "Open firewall")
        ]
        
        for cmd, desc in start_commands:
            print(f"  {desc}")
            self.ssh_command(cmd, timeout=60)
            time.sleep(2)
        
        return True
    
    def test_servers(self):
        """Test servers from local"""
        print("🔍 Testing server connectivity...")
        
        try:
            import requests
            
            # Test production
            try:
                response = requests.get(f"http://{self.ip}:8080/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Production server running!")
                    data = response.json()
                    print(f"   Status: {data.get('status')}")
                    print(f"   Version: {data.get('version')}")
                else:
                    print(f"⚠️  Production responded: {response.status_code}")
            except Exception as e:
                print(f"❌ Production not accessible: {e}")
            
            # Test fallback
            try:
                response = requests.get(f"http://{self.ip}:8081/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Fallback server running!")
                    data = response.json()
                    print(f"   Status: {data.get('status')}")
                    print(f"   Version: {data.get('version')}")
                else:
                    print(f"⚠️  Fallback responded: {response.status_code}")
            except Exception as e:
                print(f"❌ Fallback not accessible: {e}")
                
        except ImportError:
            print("❌ requests not available for testing")
    
    def deploy_and_start(self):
        """Complete deployment"""
        print("🚀 Direct SSH Server Deployment")
        print("=" * 50)
        print(f"📍 Target: {self.user}@{self.ip}")
        print(f"🔑 Key: {self.key_path}")
        
        # Test SSH connectivity
        if not self.test_ssh_connectivity():
            print("❌ Cannot connect via SSH")
            print("💡 Make sure:")
            print("   - SSH key is correct")
            print("   - Key permissions are 600")
            print("   - Instance allows SSH access")
            return False
        
        # Start servers
        print("\n🚀 Starting telemetry servers...")
        if not self.start_servers():
            print("❌ Failed to start servers")
            return False
        
        # Wait for startup
        print("\n⏳ Waiting 20 seconds for servers to start...")
        time.sleep(20)
        
        # Test servers
        print("\n🔍 Testing server connectivity...")
        self.test_servers()
        
        print(f"\n🎉 SSH deployment completed!")
        print(f"📊 Server URLs:")
        print(f"  Production API: http://{self.ip}:8080")
        print(f"  Fallback API: http://{self.ip}:8081")
        print(f"  Health Check: http://{self.ip}:8080/health")
        print(f"  Dashboard: http://{self.ip}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Terradev servers via SSH')
    parser.add_argument('--ip', default='54.237.26.116', help='Instance IP')
    parser.add_argument('--user', default='ubuntu', help='SSH user')
    parser.add_argument('--key', help='SSH key path')
    
    args = parser.parse_args()
    
    starter = DirectSSHStarter(ip=args.ip, user=args.user, key_path=args.key)
    starter.deploy_and_start()
