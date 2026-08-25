#!/usr/bin/env python3
"""
AWS SSM Server Starter
Uses AWS SSM to start telemetry servers on the EC2 instance
"""

import boto3
import json
import time
from datetime import datetime

class AWSSSMStarter:
    def __init__(self, instance_id=None, region='us-east-1'):
        self.instance_id = instance_id or 'i-0123456789abcdef0'  # Default instance ID
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
    
    def find_terradev_instance(self):
        """Find the Terradev EC2 instance"""
        print("🔍 Finding Terradev EC2 instance...")
        
        try:
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:Project', 'Values': ['terradev']},
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'id': instance['InstanceId'],
                        'ip': instance.get('PublicIpAddress'),
                        'private_ip': instance.get('PrivateIpAddress'),
                        'state': instance['State']['Name'],
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    })
            
            if instances:
                print(f"✅ Found {len(instances)} Terradev instances:")
                for instance in instances:
                    print(f"  {instance['id']} - {instance['ip']} ({instance['state']})")
                
                # Use the first running instance
                for instance in instances:
                    if instance['state'] == 'running':
                        self.instance_id = instance['id']
                        print(f"📍 Using instance: {self.instance_id}")
                        return instance
            
            print("❌ No running Terradev instances found")
            return None
            
        except Exception as e:
            print(f"❌ Error finding instances: {e}")
            return None
    
    def check_instance_connectivity(self):
        """Check if SSM can connect to the instance"""
        print(f"🔍 Checking SSM connectivity to {self.instance_id}...")
        
        try:
            response = self.ssm_client.describe_instance_information(
                InstanceInformationFilterList=[
                    {'Key': 'InstanceIds', 'ValueSet': [self.instance_id]}
                ]
            )
            
            if response['InstanceInformationList']:
                instance_info = response['InstanceInformationList'][0]
                print(f"✅ SSM agent connected: {instance_info.get('PingStatus', 'unknown')}")
                return True
            else:
                print("❌ SSM agent not connected")
                return False
                
        except Exception as e:
            print(f"❌ Error checking SSM connectivity: {e}")
            return False
    
    def send_command(self, command, comment=""):
        """Send command via SSM"""
        print(f"📤 Sending command: {comment}")
        
        try:
            response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': [command]},
                Comment=comment
            )
            
            command_id = response['Command']['CommandId']
            print(f"✅ Command sent: {command_id}")
            return command_id
            
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return None
    
    def wait_for_command(self, command_id, timeout=300):
        """Wait for command to complete"""
        print(f"⏳ Waiting for command {command_id} to complete...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.ssm_client.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=self.instance_id
                )
                
                status = response['Status']
                print(f"  Status: {status}")
                
                if status in ['Success', 'Failed', 'TimedOut', 'Cancelled']:
                    return response
                
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error checking command status: {e}")
                time.sleep(5)
        
        print("⏰ Timeout waiting for command")
        return None
    
    def start_telemetry_servers(self):
        """Start telemetry servers on the instance"""
        print("🚀 Starting telemetry servers via SSM...")
        
        # Commands to run
        commands = [
            ("cd /home/ubuntu && ls -la", "List home directory"),
            ("cd /home/ubuntu && python3 --version", "Check Python version"),
            ("cd /home/ubuntu && pip3 install flask flask-cors requests", "Install dependencies"),
            ("cd /home/ubuntu && nohup python3 production_telemetry_server.py > production.log 2>&1 &", "Start production server"),
            ("cd /home/ubuntu && nohup python3 fallback_server.py > fallback.log 2>&1 &", "Start fallback server"),
            ("sleep 5 && netstat -tlnp | grep ':808[01]'", "Check if servers are listening"),
            ("curl -s http://localhost:8080/health || echo 'Production server not ready yet'", "Test production server"),
            ("curl -s http://localhost:8081/health || echo 'Fallback server not ready yet'", "Test fallback server")
        ]
        
        results = []
        
        for command, comment in commands:
            command_id = self.send_command(command, comment)
            if command_id:
                result = self.wait_for_command(command_id, timeout=60)
                results.append(result)
                time.sleep(2)  # Brief pause between commands
        
        return results
    
    def check_server_status(self):
        """Check if servers are running"""
        print("🔍 Checking server status...")
        
        commands = [
            ("ps aux | grep telemetry_server | grep -v grep", "Check server processes"),
            ("netstat -tlnp | grep ':808[01]'", "Check listening ports"),
            ("curl -s http://localhost:8080/health | head -3", "Test production health"),
            ("curl -s http://localhost:8081/health | head -3", "Test fallback health")
        ]
        
        for command, comment in commands:
            command_id = self.send_command(command, comment)
            if command_id:
                result = self.wait_for_command(command_id, timeout=30)
                if result and result['Status'] == 'Success':
                    print(f"✅ {comment}:")
                    output = result.get('StandardOutputContent', '').strip()
                    if output:
                        print(f"  {output}")
                else:
                    print(f"❌ {comment} failed")
    
    def deploy_and_start(self):
        """Complete deployment process"""
        print("🚀 Terradev AWS SSM Server Deployment")
        print("=" * 50)
        
        # Initialize clients
        if not self.initialize_clients():
            return False
        
        # Find instance
        instance = self.find_terradev_instance()
        if not instance:
            return False
        
        # Check connectivity
        if not self.check_instance_connectivity():
            return False
        
        # Start servers
        print("\n🚀 Starting telemetry servers...")
        results = self.start_telemetry_servers()
        
        # Wait for servers to fully start
        print("\n⏳ Waiting 30 seconds for servers to initialize...")
        time.sleep(30)
        
        # Check status
        print("\n📊 Checking server status...")
        self.check_server_status()
        
        print(f"\n🎉 Deployment completed!")
        print(f"📊 Server URLs:")
        print(f"  Production API: http://{instance.get('ip', '34.207.59.52')}:8080")
        print(f"  Fallback API: http://{instance.get('ip', '34.207.59.52')}:8081")
        print(f"  Health Check: http://{instance.get('ip', '34.207.59.52')}:8080/health")
        print(f"  Dashboard: http://{instance.get('ip', '34.207.59.52')}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Terradev servers via AWS SSM')
    parser.add_argument('--instance-id', help='EC2 Instance ID')
    parser.add_argument('--region', default='us-east-1', help='AWS Region')
    parser.add_argument('--check-only', action='store_true', help='Only check status')
    
    args = parser.parse_args()
    
    starter = AWSSSMStarter(instance_id=args.instance_id, region=args.region)
    
    if args.check_only:
        starter.initialize_clients()
        instance = starter.find_terradev_instance()
        if instance:
            starter.check_instance_connectivity()
            starter.check_server_status()
    else:
        starter.deploy_and_start()
