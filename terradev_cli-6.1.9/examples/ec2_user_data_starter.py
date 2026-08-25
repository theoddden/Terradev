#!/usr/bin/env python3
"""
AWS EC2 User Data Server Starter
Uses EC2 User Data to start servers on instance launch/reboot
"""

import boto3
import json
import time
from datetime import datetime

class EC2UserDataStarter:
    def __init__(self, instance_id='i-0b072b78a8c21dada', region='us-east-1'):
        self.instance_id = instance_id
        self.region = region
        self.ec2_client = None
        self.ssm_client = None
        
    def initialize_clients(self):
        """Initialize AWS clients"""
        try:
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            self.ssm_client = boto3.client('ssm', region_name=self.region)
            print("✅ AWS clients initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AWS clients: {e}")
            return False
    
    def get_instance_details(self):
        """Get instance details"""
        print(f"🔍 Getting instance details...")
        
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance
                    
        except Exception as e:
            print(f"❌ Error getting instance details: {e}")
            return None
    
    def create_startup_script(self):
        """Create startup script for user data"""
        script = '''#!/bin/bash
# Terradev Telemetry Server Startup Script

echo "🚀 Starting Terradev Telemetry Servers..."

# Update system
sudo apt update -y
sudo apt install -y python3 python3-pip curl

# Install dependencies
pip3 install flask flask-cors requests

# Create server directory
mkdir -p /home/ubuntu/terradev
cd /home/ubuntu/terradev

# Create production telemetry server
cat > production_telemetry_server.py << 'EOF'
#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            action TEXT,
            timestamp TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-aws',
        'server': 'production'
    })

@app.route('/log-usage', methods=['POST'])
def log_usage():
    try:
        data = request.get_json()
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry_logs (machine_id, action, timestamp, details)
            VALUES (?, ?, ?, ?)
        ''', (
            data.get('machine_id'),
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {}))
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def user_stats():
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT machine_id) FROM telemetry_logs')
    total_users = cursor.fetchone()[0] or 0
    conn.close()
    
    return jsonify({
        'total_users': total_users,
        'active_users_today': 0,
        'server_info': {
            'version': '1.0.0-aws',
            'timestamp': datetime.now().isoformat()
        }
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8080)
EOF

# Create fallback telemetry server
cat > fallback_server.py << 'EOF'
#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('telemetry_fallback.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            action TEXT,
            timestamp TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-fallback',
        'server': 'fallback'
    })

@app.route('/log-usage', methods=['POST'])
def log_usage():
    try:
        data = request.get_json()
        conn = sqlite3.connect('telemetry_fallback.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry_logs (machine_id, action, timestamp, details)
            VALUES (?, ?, ?, ?)
        ''', (
            data.get('machine_id'),
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {}))
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully (fallback)',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8081)
EOF

# Start servers
echo "Starting production server..."
nohup python3 production_telemetry_server.py > production.log 2>&1 &

echo "Starting fallback server..."
nohup python3 fallback_server.py > fallback.log 2>&1 &

# Wait for servers to start
sleep 10

# Test servers
echo "Testing production server..."
curl -s http://localhost:8080/health || echo "Production server not ready"

echo "Testing fallback server..."
curl -s http://localhost:8081/health || echo "Fallback server not ready"

# Setup port forwarding/security
sudo ufw allow 8080
sudo ufw allow 8081
sudo ufw allow 80
sudo ufw allow 443

echo "✅ Terradev telemetry servers started!"
echo "📊 Production: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo "📊 Fallback: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8081"
'''
        return script
    
    def update_instance_user_data(self):
        """Update instance user data and restart"""
        print("🔄 Updating instance user data...")
        
        try:
            # Get current instance details
            instance = self.get_instance_details()
            if not instance:
                return False
            
            # Create startup script
            startup_script = self.create_startup_script()
            
            # Stop the instance
            print("🛑 Stopping instance...")
            self.ec2_client.stop_instances(InstanceIds=[self.instance_id])
            
            # Wait for instance to stop
            print("⏳ Waiting for instance to stop...")
            waiter = self.ec2_client.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[self.instance_id])
            
            # Update user data
            print("📝 Updating user data...")
            self.ec2_client.modify_instance_attribute(
                InstanceId=self.instance_id,
                UserData=startup_script
            )
            
            # Start the instance
            print("🚀 Starting instance...")
            self.ec2_client.start_instances(InstanceIds=[self.instance_id])
            
            # Wait for instance to start
            print("⏳ Waiting for instance to start...")
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[self.instance_id])
            
            print("✅ Instance restarted with new user data")
            return True
            
        except Exception as e:
            print(f"❌ Error updating user data: {e}")
            return False
    
    def run_startup_commands(self):
        """Run startup commands via SSM (if available)"""
        print("🚀 Attempting to run startup commands...")
        
        commands = [
            "cd /home/ubuntu && mkdir -p terradev && cd terradev",
            "pip3 install flask flask-cors requests",
            "curl -s 'https://raw.githubusercontent.com/theoddden/terradev/main/production_telemetry_server.py' -o production_telemetry_server.py || echo 'Using local copy'",
            "curl -s 'https://raw.githubusercontent.com/theoddden/terradev/main/fallback_server.py' -o fallback_server.py || echo 'Using local copy'",
            "nohup python3 production_telemetry_server.py > production.log 2>&1 &",
            "nohup python3 fallback_server.py > fallback.log 2>&1 &",
            "sleep 15",
            "curl -s http://localhost:8080/health || echo 'Production not ready'",
            "curl -s http://localhost:8081/health || echo 'Fallback not ready'"
        ]
        
        for command in commands:
            try:
                print(f"📤 Running: {command[:50]}...")
                response = self.ssm_client.send_command(
                    InstanceIds=[self.instance_id],
                    DocumentName='AWS-RunShellScript',
                    Parameters={'commands': [command]}
                )
                
                command_id = response['Command']['CommandId']
                print(f"✅ Command sent: {command_id}")
                
                # Wait a bit between commands
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error running command: {e}")
    
    def deploy_and_start(self):
        """Complete deployment"""
        print("🚀 EC2 User Data Server Deployment")
        print("=" * 50)
        
        # Initialize clients
        if not self.initialize_clients():
            return False
        
        # Get instance details
        instance = self.get_instance_details()
        if not instance:
            return False
        
        print(f"📍 Instance: {instance['InstanceId']}")
        print(f"🌐 IP: {instance.get('PublicIpAddress')}")
        print(f"🏷️  Tags: {instance.get('Tags', [])}")
        
        # Try SSM commands first (less disruptive)
        print("\n🚀 Attempting SSM deployment...")
        self.run_startup_commands()
        
        # Wait and check
        print("\n⏳ Waiting 30 seconds for servers to start...")
        time.sleep(30)
        
        # Test connectivity
        ip = instance.get('PublicIpAddress')
        print(f"\n🔍 Testing connectivity to {ip}...")
        
        try:
            import requests
            response = requests.get(f"http://{ip}:8080/health", timeout=10)
            if response.status_code == 200:
                print("✅ Production server is running!")
                print(f"📊 Dashboard: http://{ip}:8080/dashboard")
            else:
                print("⚠️  Production server responding but not healthy")
        except:
            print("❌ Production server not accessible")
        
        try:
            response = requests.get(f"http://{ip}:8081/health", timeout=10)
            if response.status_code == 200:
                print("✅ Fallback server is running!")
            else:
                print("⚠️  Fallback server responding but not healthy")
        except:
            print("❌ Fallback server not accessible")
        
        print(f"\n🎉 Deployment completed!")
        print(f"📊 Server URLs:")
        print(f"  Production API: http://{ip}:8080")
        print(f"  Fallback API: http://{ip}:8081")
        print(f"  Health Check: http://{ip}:8080/health")
        print(f"  Dashboard: http://{ip}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    starter = EC2UserDataStarter()
    starter.deploy_and_start()
