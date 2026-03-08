#!/usr/bin/env python3
"""
Simple AWS SSM Server Starter
Direct SSM commands to start telemetry servers
"""

import boto3
import time
from datetime import datetime

class SimpleSSMStarter:
    def __init__(self, instance_id='i-0b072b78a8c21dada', region='us-east-1'):
        self.instance_id = instance_id
        self.region = region
        self.ssm_client = None
        self.ec2_client = None
        
    def initialize_clients(self):
        """Initialize AWS clients"""
        try:
            self.ssm_client = boto3.client('ssm', region_name=self.region)
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            print("✅ AWS clients initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AWS clients: {e}")
            return False
    
    def get_instance_ip(self):
        """Get instance public IP"""
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance.get('PublicIpAddress')
        except Exception as e:
            print(f"❌ Error getting instance IP: {e}")
        
        return None
    
    def send_command(self, command, timeout=60):
        """Send command via SSM"""
        try:
            print(f"📤 Sending: {command[:50]}...")
            
            response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': [command]}
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result = self.ssm_client.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=self.instance_id
                    )
                    
                    status = result['Status']
                    if status in ['Success', 'Failed', 'TimedOut', 'Cancelled']:
                        if status == 'Success':
                            output = result.get('StandardOutputContent', '').strip()
                            error = result.get('StandardErrorContent', '').strip()
                            
                            if output:
                                print(f"✅ {output}")
                            if error:
                                print(f"⚠️  {error}")
                        else:
                            print(f"❌ Command failed: {status}")
                        
                        return result
                    
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"❌ Error checking status: {e}")
                    time.sleep(3)
            
            print("⏰ Command timeout")
            return None
            
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return None
    
    def start_servers(self):
        """Start telemetry servers"""
        print("🚀 Starting telemetry servers...")
        
        commands = [
            "cd /home/ubuntu && pwd",
            "sudo apt update -y",
            "sudo apt install -y python3 python3-pip",
            "pip3 install flask flask-cors requests",
            "mkdir -p /home/ubuntu/terradev && cd /home/ubuntu/terradev"
        ]
        
        # Setup commands
        for cmd in commands:
            self.send_command(cmd, timeout=120)
            time.sleep(2)
        
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
        
        # Create production server file
        create_prod = f"cat > /home/ubuntu/terradev/production_server.py << 'EOF'\n{prod_server}\nEOF"
        self.send_command(create_prod, timeout=30)
        
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
        
        # Create fallback server file
        create_fallback = f"cat > /home/ubuntu/terradev/fallback_server.py << 'EOF'\n{fallback_server}\nEOF"
        self.send_command(create_fallback, timeout=30)
        
        # Start servers
        start_commands = [
            "cd /home/ubuntu/terradev && nohup python3 production_server.py > production.log 2>&1 &",
            "cd /home/ubuntu/terradev && nohup python3 fallback_server.py > fallback.log 2>&1 &",
            "sleep 10",
            "ps aux | grep python3 | grep server",
            "netstat -tlnp | grep ':808[01]' || echo 'Ports not bound yet'",
            "curl -s http://localhost:8080/health || echo 'Production not ready'",
            "curl -s http://localhost:8081/health || echo 'Fallback not ready'",
            "sudo ufw allow 8080 && sudo ufw allow 8081"
        ]
        
        for cmd in start_commands:
            self.send_command(cmd, timeout=60)
            time.sleep(3)
    
    def test_servers(self, ip):
        """Test servers from local"""
        print(f"🔍 Testing servers at {ip}...")
        
        try:
            import requests
            
            # Test production
            try:
                response = requests.get(f"http://{ip}:8080/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Production server is running!")
                    data = response.json()
                    print(f"   Status: {data.get('status')}")
                    print(f"   Version: {data.get('version')}")
                else:
                    print(f"⚠️  Production server responded with {response.status_code}")
            except Exception as e:
                print(f"❌ Production server not accessible: {e}")
            
            # Test fallback
            try:
                response = requests.get(f"http://{ip}:8081/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Fallback server is running!")
                    data = response.json()
                    print(f"   Status: {data.get('status')}")
                    print(f"   Version: {data.get('version')}")
                else:
                    print(f"⚠️  Fallback server responded with {response.status_code}")
            except Exception as e:
                print(f"❌ Fallback server not accessible: {e}")
                
        except ImportError:
            print("❌ requests module not available for testing")
    
    def deploy_and_start(self):
        """Complete deployment"""
        print("🚀 Simple SSM Server Deployment")
        print("=" * 50)
        
        # Initialize clients
        if not self.initialize_clients():
            return False
        
        # Get instance IP
        ip = self.get_instance_ip()
        if not ip:
            print("❌ Could not get instance IP")
            return False
        
        print(f"📍 Instance: {self.instance_id}")
        print(f"🌐 IP: {ip}")
        
        # Start servers
        print("\n🚀 Starting telemetry servers...")
        self.start_servers()
        
        # Wait for startup
        print("\n⏳ Waiting 20 seconds for servers to start...")
        time.sleep(20)
        
        # Test servers
        print("\n🔍 Testing server connectivity...")
        self.test_servers(ip)
        
        print(f"\n🎉 Deployment completed!")
        print(f"📊 Server URLs:")
        print(f"  Production API: http://{ip}:8080")
        print(f"  Fallback API: http://{ip}:8081")
        print(f"  Health Check: http://{ip}:8080/health")
        print(f"  Dashboard: http://{ip}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    starter = SimpleSSMStarter()
    starter.deploy_and_start()
