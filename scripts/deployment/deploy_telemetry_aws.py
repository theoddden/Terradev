#!/usr/bin/env python3
"""
Deploy Terradev Telemetry to AWS Cloud
Deploys the production telemetry server to AWS infrastructure
"""

import subprocess
import os
import json
import time
from datetime import datetime

class AWSTelemetryDeployer:
    def __init__(self):
        self.region = "us-east-1"
        self.cluster_name = "terradev-cluster"
        self.namespace = "terradev-system"
        
    def check_prerequisites(self):
        """Check if AWS CLI and kubectl are available"""
        print("🔍 Checking prerequisites...")
        
        # Check AWS CLI
        try:
            result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
            print(f"✅ AWS CLI: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ AWS CLI not found. Please install AWS CLI")
            return False
        
        # Check kubectl
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], capture_output=True, text=True)
            print(f"✅ kubectl: {result.stdout.split()[4] if len(result.stdout.split()) > 4 else 'installed'}")
        except FileNotFoundError:
            print("❌ kubectl not found. Please install kubectl")
            return False
        
        # Check AWS credentials
        try:
            result = subprocess.run(['aws', 'sts', 'get-caller-identity'], capture_output=True, text=True)
            if result.returncode == 0:
                identity = json.loads(result.stdout)
                print(f"✅ AWS Identity: {identity['UserId']} ({identity['Account']})")
            else:
                print("❌ AWS credentials not configured")
                return False
        except Exception as e:
            print(f"❌ Error checking AWS credentials: {e}")
            return False
        
        return True
    
    def deploy_telemetry_to_eks(self):
        """Deploy telemetry server to existing EKS cluster"""
        print("\n🚀 Deploying Terradev Telemetry to EKS...")
        
        # Create namespace if it doesn't exist
        print("📦 Creating namespace...")
        try:
            subprocess.run([
                'kubectl', 'create', 'namespace', self.namespace,
                '--dry-run=client', '-o', 'yaml'
            ], check=True, capture_output=True)
            
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=subprocess.run([
                'kubectl', 'create', 'namespace', self.namespace,
                '--dry-run=client', '-o', 'yaml'
            ], check=True, capture_output=True, text=True).stdout, text=True)
            
            print(f"✅ Namespace {self.namespace} ready")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating namespace: {e}")
            return False
        
        # Create ConfigMap for telemetry configuration
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "telemetry-config",
                "namespace": self.namespace,
                "labels": {
                    "app": "terradev-telemetry",
                    "component": "config"
                }
            },
            "data": {
                "DATABASE_URL": "postgresql://terradev_admin:password@terradev-postgres:5432/terradev",
                "REDIS_URL": "redis://terradev-redis:6379",
                "LOG_LEVEL": "INFO",
                "ENVIRONMENT": "production"
            }
        }
        
        print("⚙️  Creating ConfigMap...")
        try:
            import yaml
            configmap_yaml = yaml.dump(configmap)
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=configmap_yaml, text=True, check=True)
            print("✅ ConfigMap created")
        except Exception as e:
            print(f"❌ Error creating ConfigMap: {e}")
            return False
        
        # Create Secret for sensitive data
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "telemetry-secrets",
                "namespace": self.namespace,
                "labels": {
                    "app": "terradev-telemetry",
                    "component": "secrets"
                }
            },
            "type": "Opaque",
            "data": {
                "postgres-password": "cGFzc3dvcmQ=",  # base64 encoded "password"
                "redis-password": "dGVsZW1ldHJ5LXNlY3JldA==",  # base64 encoded "telemetry-secret"
                "jwt-secret": "dGVsZW1ldHJ5LWp3dC1zZWNyZXQ="  # base64 encoded "telemetry-jwt-secret"
            }
        }
        
        print("🔐 Creating Secret...")
        try:
            secret_yaml = yaml.dump(secret)
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=secret_yaml, text=True, check=True)
            print("✅ Secret created")
        except Exception as e:
            print(f"❌ Error creating Secret: {e}")
            return False
        
        # Create Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "terradev-telemetry",
                "namespace": self.namespace,
                "labels": {
                    "app": "terradev-telemetry",
                    "component": "api"
                }
            },
            "spec": {
                "replicas": 2,
                "selector": {
                    "matchLabels": {
                        "app": "terradev-telemetry"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "terradev-telemetry",
                            "component": "api"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "telemetry-api",
                            "image": "theoddden/terradev-cli:2.9.2",
                            "ports": [{
                                "containerPort": 8080,
                                "protocol": "TCP"
                            }],
                            "env": [
                                {
                                    "name": "PORT",
                                    "value": "8080"
                                },
                                {
                                    "name": "DATABASE_URL",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": "telemetry-config",
                                            "key": "DATABASE_URL"
                                        }
                                    }
                                },
                                {
                                    "name": "REDIS_URL",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": "telemetry-config",
                                            "key": "REDIS_URL"
                                        }
                                    }
                                },
                                {
                                    "name": "POSTGRES_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "telemetry-secrets",
                                            "key": "postgres-password"
                                        }
                                    }
                                },
                                {
                                    "name": "REDIS_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "telemetry-secrets",
                                            "key": "redis-password"
                                        }
                                    }
                                },
                                {
                                    "name": "JWT_SECRET",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "telemetry-secrets",
                                            "key": "jwt-secret"
                                        }
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "256Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        print("🚀 Creating Deployment...")
        try:
            deployment_yaml = yaml.dump(deployment)
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=deployment_yaml, text=True, check=True)
            print("✅ Deployment created")
        except Exception as e:
            print(f"❌ Error creating Deployment: {e}")
            return False
        
        # Create Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "terradev-telemetry-service",
                "namespace": self.namespace,
                "labels": {
                    "app": "terradev-telemetry",
                    "component": "api"
                }
            },
            "spec": {
                "selector": {
                    "app": "terradev-telemetry"
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8080
                }],
                "type": "LoadBalancer"
            }
        }
        
        print("🌐 Creating Service...")
        try:
            service_yaml = yaml.dump(service)
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=service_yaml, text=True, check=True)
            print("✅ Service created")
        except Exception as e:
            print(f"❌ Error creating Service: {e}")
            return False
        
        # Create Ingress for external access
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "terradev-telemetry-ingress",
                "namespace": self.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": ["api.terradev.cloud"],
                    "secretName": "telemetry-tls"
                }],
                "rules": [{
                    "host": "api.terradev.cloud",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "terradev-telemetry-service",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        print("🔒 Creating Ingress...")
        try:
            ingress_yaml = yaml.dump(ingress)
            subprocess.run([
                'kubectl', 'apply', '-f', '-'
            ], input=ingress_yaml, text=True, check=True)
            print("✅ Ingress created")
        except Exception as e:
            print(f"❌ Error creating Ingress: {e}")
            return False
        
        return True
    
    def wait_for_deployment(self):
        """Wait for deployment to be ready"""
        print("\n⏳ Waiting for deployment to be ready...")
        
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                result = subprocess.run([
                    'kubectl', 'get', 'deployment', 'terradev-telemetry',
                    '-n', self.namespace, '-o', 'jsonpath={.status.readyReplicas}'
                ], capture_output=True, text=True)
                
                ready_replicas = result.stdout.strip()
                if ready_replicas == "2":
                    print("✅ Deployment is ready!")
                    return True
                
                print(f"⏳ Waiting... Ready replicas: {ready_replicas}/2")
                time.sleep(10)
                
            except Exception as e:
                print(f"❌ Error checking deployment status: {e}")
                time.sleep(10)
        
        print("⏰ Timeout waiting for deployment")
        return False
    
    def get_service_url(self):
        """Get the external URL for the service"""
        print("\n🔗 Getting service URL...")
        
        try:
            result = subprocess.run([
                'kubectl', 'get', 'service', 'terradev-telemetry-service',
                '-n', self.namespace, '-o', 'jsonpath={.status.loadBalancer.ingress[0].hostname}'
            ], capture_output=True, text=True)
            
            hostname = result.stdout.strip()
            if hostname:
                url = f"http://{hostname}"
                print(f"✅ Service URL: {url}")
                return url
            else:
                print("⚠️  LoadBalancer not ready yet")
                return None
                
        except Exception as e:
            print(f"❌ Error getting service URL: {e}")
            return None
    
    def test_deployment(self, url):
        """Test the deployed telemetry service"""
        print(f"\n🧪 Testing deployment at {url}...")
        
        try:
            import requests
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed!")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Users: {data.get('users', {})}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing deployment: {e}")
            return False
    
    def deploy(self):
        """Complete deployment process"""
        print("🚀 Terradev Telemetry AWS Deployment")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Deploy to EKS
        if not self.deploy_telemetry_to_eks():
            return False
        
        # Wait for deployment
        if not self.wait_for_deployment():
            return False
        
        # Get service URL
        url = self.get_service_url()
        if not url:
            print("⚠️  Deployment succeeded but URL not available")
            return True
        
        # Test deployment
        self.test_deployment(url)
        
        print(f"\n🎉 Deployment completed successfully!")
        print(f"📊 Dashboard: {url}/dashboard")
        print(f"🔍 Health: {url}/health")
        print(f"📈 Stats: {url}/user-stats")
        
        return True

if __name__ == '__main__':
    deployer = AWSTelemetryDeployer()
    deployer.deploy()
