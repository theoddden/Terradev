#!/usr/bin/env python3
"""
Direct AWS Server Starter
Starts telemetry servers on the specific AWS instance
"""

import boto3
import time
from datetime import datetime

class DirectAWSStarter:
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
    
    def check_instance(self):
        """Check specific instance details"""
        print(f"🔍 Checking instance {self.instance_id}...")
        
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    print(f"✅ Instance found:")
                    print(f"  ID: {instance['InstanceId']}")
                    print(f"  IP: {instance.get('PublicIpAddress')}")
                    print(f"  State: {instance['State']['Name']}")
                    print(f"  Tags: {instance.get('Tags', [])}")
                    
                    return instance
                    
        except Exception as e:
            print(f"❌ Error checking instance: {e}")
            return None
    
    def check_ssm_connectivity(self):
        """Check if SSM can connect to the instance"""
        print(f"🔍 Checking SSM connectivity...")
        
        try:
            response = self.ssm_client.describe_instance_information(
                InstanceInformationFilterList=[
                    {'key': 'InstanceIds', 'valueSet': [self.instance_id]}
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
    
    def send_command(self, command, comment="", timeout=60):
        """Send command via SSM and wait for completion"""
        print(f"📤 Sending command: {comment}")
        
        try:
            # Send command
            response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': [command]},
                Comment=comment
            )
            
            command_id = response['Command']['CommandId']
            print(f"✅ Command sent: {command_id}")
            
            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result_response = self.ssm_client.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=self.instance_id
                    )
                    
                    status = result_response['Status']
                    print(f"  Status: {status}")
                    
                    if status in ['Success', 'Failed', 'TimedOut', 'Cancelled']:
                        if status == 'Success':
                            output = result_response.get('StandardOutputContent', '').strip()
                            error = result_response.get('StandardErrorContent', '').strip()
                            
                            if output:
                                print(f"✅ Output: {output}")
                            if error:
                                print(f"⚠️  Error: {error}")
                        
                        return result_response
                    
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"❌ Error checking command status: {e}")
                    time.sleep(5)
            
            print("⏰ Timeout waiting for command")
            return None
            
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return None
    
    def start_servers(self):
        """Start telemetry servers"""
        print("🚀 Starting telemetry servers...")
        
        # Commands to execute
        commands = [
            ("whoami && pwd", "Check current user and directory"),
            ("ls -la", "List files"),
            ("python3 --version", "Check Python version"),
            ("pip3 install flask flask-cors requests", "Install dependencies"),
            ("nohup python3 production_telemetry_server.py > production.log 2>&1 &", "Start production server"),
            ("nohup python3 fallback_server.py > fallback.log 2>&1 &", "Start fallback server"),
            ("sleep 10", "Wait for servers to start"),
            ("ps aux | grep telemetry_server | grep -v grep", "Check server processes"),
            ("netstat -tlnp | grep ':808[01]' || echo 'Ports not bound yet'", "Check listening ports"),
            ("curl -s http://localhost:8080/health || echo 'Production server not ready'", "Test production"),
            ("curl -s http://localhost:8081/health || echo 'Fallback server not ready'", "Test fallback")
        ]
        
        for command, comment in commands:
            result = self.send_command(command, comment, timeout=120)
            if result and result['Status'] != 'Success':
                print(f"⚠️  Command failed, continuing anyway...")
            time.sleep(2)
    
    def check_servers(self):
        """Check server status"""
        print("🔍 Checking server status...")
        
        commands = [
            ("ps aux | grep telemetry_server | grep -v grep", "Check processes"),
            ("netstat -tlnp | grep ':808[01]'", "Check ports"),
            ("curl -s http://localhost:8080/health | head -3", "Test production health"),
            ("curl -s http://localhost:8081/health | head -3", "Test fallback health")
        ]
        
        for command, comment in commands:
            self.send_command(command, comment, timeout=30)
    
    def deploy_and_start(self):
        """Complete deployment"""
        print("🚀 Direct AWS Server Starter")
        print("=" * 50)
        
        # Initialize clients
        if not self.initialize_clients():
            return False
        
        # Check instance
        instance = self.check_instance()
        if not instance:
            return False
        
        # Check SSM connectivity
        if not self.check_ssm_connectivity():
            print("❌ Cannot connect via SSM")
            return False
        
        # Start servers
        print("\n🚀 Starting telemetry servers...")
        self.start_servers()
        
        # Wait and check
        print("\n⏳ Waiting 30 seconds for full startup...")
        time.sleep(30)
        
        # Final status check
        print("\n📊 Final server status:")
        self.check_servers()
        
        print(f"\n🎉 Server deployment completed!")
        print(f"📊 Server URLs:")
        print(f"  Production API: http://{instance.get('PublicIpAddress')}:8080")
        print(f"  Fallback API: http://{instance.get('PublicIpAddress')}:8081")
        print(f"  Health Check: http://{instance.get('PublicIpAddress')}:8080/health")
        print(f"  Dashboard: http://{instance.get('PublicIpAddress')}:8080/dashboard")
        
        return True

if __name__ == '__main__':
    starter = DirectAWSStarter()
    starter.deploy_and_start()
