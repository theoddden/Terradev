#!/usr/bin/env python3
"""
GitOps Automation - Infrastructure as Code with GitOps workflows

Implements GitOps patterns for Kubernetes infrastructure management:
1. Git repository initialization and structure
2. ArgoCD/Flux bootstrap automation
3. Sync and validation workflows
4. Policy as Code integration
5. Multi-environment support

Based on production lessons: "GitOps isn't optional, it's survival"
"""

import asyncio
import json
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class GitProvider(Enum):
    """Supported Git providers"""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"


class GitOpsTool(Enum):
    """Supported GitOps tools"""
    ARGOCD = "argocd"
    FLUX = "flux"
    KAPP_CONTROLLER = "kapp_controller"


@dataclass
class GitOpsConfig:
    """Configuration for GitOps setup"""
    provider: GitProvider
    repository: str
    tool: GitOpsTool
    cluster_name: str
    environments: List[str] = field(default_factory=lambda: ["dev", "staging", "prod"])
    git_url: Optional[str] = None
    git_token: Optional[str] = None
    namespace: str = "gitops-system"
    auto_sync: bool = True
    prune_resources: bool = True


@dataclass
class GitRepoStructure:
    """Git repository structure for GitOps"""
    base_dirs: List[str] = field(default_factory=lambda: [
        "clusters",
        "apps",
        "infra",
        "policies",
        "monitoring"
    ])
    
    def create_structure(self, base_path: Path) -> None:
        """Create the GitOps repository structure"""
        for directory in self.base_dirs:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
            
        # Create cluster-specific directories
        for env in ["dev", "staging", "prod"]:
            (base_path / "clusters" / env).mkdir(parents=True, exist_ok=True)


class GitOpsManager:
    """Main GitOps automation manager"""
    
    def __init__(self, config: GitOpsConfig):
        self.config = config
        self.repo_structure = GitRepoStructure()
        self.work_dir = Path.home() / ".terradev" / "gitops" / config.repository
        
    async def init_repository(self) -> bool:
        """Initialize GitOps repository with proper structure"""
        try:
            logger.info(f"Initializing GitOps repository: {self.config.repository}")
            
            # Create working directory
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize git repository
            if not (self.work_dir / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.work_dir, check=True)
                
            # Create repository structure
            self.repo_structure.create_structure(self.work_dir)
            
            # Generate initial configuration files
            await self._generate_gitops_files()
            
            # Create initial commit
            subprocess.run(["git", "add", "."], cwd=self.work_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial GitOps setup"], cwd=self.work_dir, check=True)
            
            logger.info("GitOps repository initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GitOps repository: {e}")
            return False
    
    async def _generate_gitops_files(self) -> None:
        """Generate GitOps configuration files"""
        
        # Generate ArgoCD ApplicationSet
        if self.config.tool == GitOpsTool.ARGOCD:
            await self._generate_argocd_config()
        
        # Generate Flux configuration
        elif self.config.tool == GitOpsTool.FLUX:
            await self._generate_flux_config()
        
        # Generate cluster configurations
        await self._generate_cluster_configs()
        
        # Generate application templates
        await self._generate_app_templates()
        
        # Generate policy configurations
        await self._generate_policy_configs()
    
    async def _generate_argocd_config(self) -> None:
        """Generate ArgoCD configuration files"""
        
        # ArgoCD namespace
        argocd_ns = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "name": self.config.namespace
                }
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "argocd-namespace.yaml", argocd_ns)
        
        # ArgoCD ApplicationSet for environments
        appset = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "ApplicationSet",
            "metadata": {
                "name": "cluster-apps",
                "namespace": self.config.namespace
            },
            "spec": {
                "generators": [
                    {
                        "git": {
                            "repoURL": self.config.git_url or f"https://github.com/{self.config.repository}",
                            "revision": "HEAD",
                            "directories": [
                                {
                                    "path": "clusters/*"
                                }
                            ]
                        }
                    }
                ],
                "template": {
                    "metadata": {
                        "name": "{{path.basename}}-apps",
                        "namespace": self.config.namespace
                    },
                    "spec": {
                        "project": "default",
                        "source": {
                            "repoURL": self.config.git_url or f"https://github.com/{self.config.repository}",
                            "targetRevision": "HEAD",
                            "path": "clusters/{{path.basename}}"
                        },
                        "destination": {
                            "server": "https://kubernetes.default.svc",
                            "namespace": "{{path.basename}}"
                        },
                        "syncPolicy": {
                            "automated": {
                                "prune": self.config.prune_resources,
                                "selfHeal": self.config.auto_sync
                            },
                            "syncOptions": [
                                "CreateNamespace=true"
                            ]
                        }
                    }
                }
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "argocd-appset.yaml", appset)
        
        # ArgoCD installation
        argocd_install = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": "argocd",
                "namespace": self.config.namespace
            },
            "spec": {
                "project": "default",
                "source": {
                    "repoURL": "https://argoproj.github.io/argo-helm",
                    "chart": "argo-cd",
                    "targetRevision": "5.51.4",
                    "helm": {
                        "parameters": [
                            {"name": "server.service.type", "value": "LoadBalancer"},
                            {"name": "configs.credentialTemplates.git-creds.url", "value": self.config.git_url or f"https://github.com/{self.config.repository}"},
                            {"name": "configs.credentialTemplates.git-creds.username", "value": "git"},
                            {"name": "configs.credentialTemplates.git-creds.password", "value": self.config.git_token or ""}
                        ]
                    }
                },
                "destination": {
                    "server": "https://kubernetes.default.svc",
                    "namespace": self.config.namespace
                },
                "syncPolicy": {
                    "automated": {
                        "prune": True,
                        "selfHeal": True
                    },
                    "syncOptions": [
                        "CreateNamespace=true"
                    ]
                }
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "argocd-install.yaml", argocd_install)
    
    async def _generate_flux_config(self) -> None:
        """Generate Flux configuration files"""
        
        # Flux namespace
        flux_ns = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "flux-system"
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "flux-namespace.yaml", flux_ns)
        
        # Flux GitRepository
        git_repo = {
            "apiVersion": "source.toolkit.fluxcd.io/v1beta2",
            "kind": "GitRepository",
            "metadata": {
                "name": "infra-repo",
                "namespace": "flux-system"
            },
            "spec": {
                "interval": "1m0s",
                "url": self.config.git_url or f"https://github.com/{self.config.repository}",
                "secretRef": {
                    "name": "infra-repo-creds"
                },
                "gitImplementation": "go-git"
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "flux-gitrepository.yaml", git_repo)
        
        # Flux Kustomization
        kustomization = {
            "apiVersion": "kustomize.toolkit.fluxcd.io/v1beta2",
            "kind": "Kustomization",
            "metadata": {
                "name": "infra-kustomization",
                "namespace": "flux-system"
            },
            "spec": {
                "interval": "10m0s",
                "sourceRef": {
                    "kind": "GitRepository",
                    "name": "infra-repo"
                },
                "path": "./clusters/prod",
                "prune": self.config.prune_resources,
                "validation": "client"
            }
        }
        
        self._write_yaml(self.work_dir / "infra" / "flux-kustomization.yaml", kustomization)
    
    async def _generate_cluster_configs(self) -> None:
        """Generate cluster-specific configurations"""
        
        for env in self.config.environments:
            cluster_dir = self.work_dir / "clusters" / env
            
            # Cluster configuration
            cluster_config = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "cluster-config",
                    "namespace": env
                },
                "data": {
                    "environment": env,
                    "cluster": self.config.cluster_name,
                    "region": "us-west-2",
                    "gitops": "enabled"
                }
            }
            
            self._write_yaml(cluster_dir / "cluster-config.yaml", cluster_config)
            
            # Namespace configuration
            namespace = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": env,
                    "labels": {
                        "environment": env,
                        "managed-by": "gitops"
                    }
                }
            }
            
            self._write_yaml(cluster_dir / "namespace.yaml", namespace)
            
            # Resource quotas
            resource_quota = {
                "apiVersion": "v1",
                "kind": "ResourceQuota",
                "metadata": {
                    "name": f"{env}-quota",
                    "namespace": env
                },
                "spec": {
                    "hard": {
                        "requests.cpu": "10",
                        "requests.memory": "20Gi",
                        "limits.cpu": "20",
                        "limits.memory": "40Gi",
                        "persistentvolumeclaims": "10",
                        "pods": "20",
                        "services": "10",
                        "secrets": "10",
                        "configmaps": "10"
                    }
                }
            }
            
            self._write_yaml(cluster_dir / "resource-quota.yaml", resource_quota)
            
            # Network policies
            network_policy = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"{env}-default-deny-all",
                    "namespace": env
                },
                "spec": {
                    "podSelector": {},
                    "policyTypes": ["Ingress", "Egress"]
                }
            }
            
            self._write_yaml(cluster_dir / "network-policy.yaml", network_policy)
    
    async def _generate_app_templates(self) -> None:
        """Generate application templates"""
        
        apps_dir = self.work_dir / "apps"
        
        # Sample application template
        app_template = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": "{{app_name}}",
                "namespace": self.config.namespace
            },
            "spec": {
                "project": "default",
                "source": {
                    "repoURL": self.config.git_url or f"https://github.com/{self.config.repository}",
                    "targetRevision": "HEAD",
                    "path": "apps/{{app_name}}"
                },
                "destination": {
                    "server": "https://kubernetes.default.svc",
                    "namespace": "{{environment}}"
                },
                "syncPolicy": {
                    "automated": {
                        "prune": self.config.prune_resources,
                        "selfHeal": self.config.auto_sync
                    },
                    "syncOptions": [
                        "CreateNamespace=true"
                    ]
                }
            }
        }
        
        self._write_yaml(apps_dir / "application-template.yaml", app_template)
        
        # Sample deployment template
        deployment_template = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "{{app_name}}",
                "namespace": "{{environment}}",
                "labels": {
                    "app": "{{app_name}}",
                    "version": "{{version}}"
                }
            },
            "spec": {
                "replicas": "{{replicas}}",
                "selector": {
                    "matchLabels": {
                        "app": "{{app_name}}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "{{app_name}}",
                            "version": "{{version}}"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "{{app_name}}",
                                "image": "{{image}}:{{version}}",
                                "ports": [
                                    {
                                        "containerPort": 8080
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "100m",
                                        "memory": "128Mi"
                                    },
                                    "limits": {
                                        "cpu": "500m",
                                        "memory": "512Mi"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        self._write_yaml(apps_dir / "deployment-template.yaml", deployment_template)
    
    async def _generate_policy_configs(self) -> None:
        """Generate policy configurations"""
        
        policies_dir = self.work_dir / "policies"
        
        # Sample policy using Gatekeeper
        gatekeeper_policy = {
            "apiVersion": "templates.gatekeeper.sh/v1beta1",
            "kind": "ConstraintTemplate",
            "metadata": {
                "name": "k8srequiredlabels"
            },
            "spec": {
                "crd": {
                    "spec": {
                        "names": {
                            "kind": "K8sRequiredLabels"
                        }
                    }
                },
                "targets": [
                    {
                        "target": "admission.k8s.gatekeeper.sh",
                        "rego": """
package k8srequiredlabels

deny[msg] {
    input.review.object.metadata.labels
    required := {"environment", "app", "team"}
    missing := {x | x = required[_] - input.review.object.metadata.labels[_]}
    count(missing) > 0
    msg := sprintf("missing required labels: %v", [missing])
}
                        """
                    }
                ]
            }
        }
        
        self._write_yaml(policies_dir / "required-labels.yaml", gatekeeper_policy)
        
        # Sample constraint
        constraint = {
            "apiVersion": "constraints.gatekeeper.sh/v1beta1",
            "kind": "K8sRequiredLabels",
            "metadata": {
                "name": "all-must-have-required-labels"
            },
            "spec": {
                "match": {
                    "kinds": [
                        {
                            "apiGroups": ["*"],
                            "kinds": ["*"]
                        }
                    ]
                },
                "parameters": {
                    "labels": ["environment", "app", "team"]
                }
            }
        }
        
        self._write_yaml(policies_dir / "required-labels-constraint.yaml", constraint)
    
    async def bootstrap_gitops(self) -> bool:
        """Bootstrap GitOps tool on the cluster"""
        try:
            logger.info(f"Bootstrapping {self.config.tool.value} on cluster {self.config.cluster_name}")
            
            if self.config.tool == GitOpsTool.ARGOCD:
                return await self._bootstrap_argocd()
            elif self.config.tool == GitOpsTool.FLUX:
                return await self._bootstrap_flux()
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to bootstrap GitOps: {e}")
            return False
    
    async def _bootstrap_argocd(self) -> bool:
        """Bootstrap ArgoCD on the cluster"""
        try:
            # Install ArgoCD using kubectl
            install_cmd = [
                "kubectl", "apply", "-f", 
                "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
            ]
            
            subprocess.run(install_cmd, check=True)
            
            # Wait for ArgoCD to be ready
            await self._wait_for_deployment("argocd-server", self.config.namespace)
            
            logger.info("ArgoCD bootstrapped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bootstrap ArgoCD: {e}")
            return False
    
    async def _bootstrap_flux(self) -> bool:
        """Bootstrap Flux on the cluster"""
        try:
            # Install Flux using flux CLI
            bootstrap_cmd = [
                "flux", "bootstrap", "git",
                "--url", self.config.git_url or f"https://github.com/{self.config.repository}",
                "--branch", "main",
                "--path", "clusters/prod",
                "--namespace", "flux-system"
            ]
            
            if self.config.git_token:
                bootstrap_cmd.extend(["--token-auth"])
            
            subprocess.run(bootstrap_cmd, check=True)
            
            logger.info("Flux bootstrapped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bootstrap Flux: {e}")
            return False
    
    async def sync_cluster(self, environment: str = "prod") -> bool:
        """Sync cluster with Git repository"""
        try:
            logger.info(f"Syncing cluster {self.config.cluster_name} with environment {environment}")
            
            if self.config.tool == GitOpsTool.ARGOCD:
                return await self._sync_argocd(environment)
            elif self.config.tool == GitOpsTool.FLUX:
                return await self._sync_flux(environment)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to sync cluster: {e}")
            return False
    
    async def _sync_argocd(self, environment: str) -> bool:
        """Sync ArgoCD applications"""
        try:
            # Get applications for the environment
            apps_cmd = [
                "kubectl", "get", "applications", "-n", self.config.namespace,
                "-l", f"environment={environment}", "-o", "name"
            ]
            
            result = subprocess.run(apps_cmd, capture_output=True, text=True, check=True)
            
            # Sync each application
            for app in result.stdout.strip().split('\n'):
                if app:
                    app_name = app.replace('application.', '')
                    sync_cmd = [
                        "argocd", "app", "sync", app_name,
                        "--namespace", self.config.namespace
                    ]
                    subprocess.run(sync_cmd, check=True)
            
            logger.info(f"ArgoCD sync completed for environment {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync ArgoCD: {e}")
            return False
    
    async def _sync_flux(self, environment: str) -> bool:
        """Sync Flux Kustomizations"""
        try:
            # Force reconciliation
            reconcile_cmd = [
                "flux", "reconcile", "kustomization", "infra-kustomization",
                "--namespace", "flux-system"
            ]
            
            subprocess.run(reconcile_cmd, check=True)
            
            logger.info(f"Flux sync completed for environment {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync Flux: {e}")
            return False
    
    async def validate_configuration(self, dry_run: bool = True) -> Dict[str, Any]:
        """Validate GitOps configuration"""
        try:
            logger.info("Validating GitOps configuration")
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Validate YAML syntax
            for yaml_file in self.work_dir.rglob("*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    validation_results["errors"].append(f"YAML syntax error in {yaml_file}: {e}")
                    validation_results["valid"] = False
            
            # Validate Kubernetes manifests
            if not dry_run:
                for yaml_file in self.work_dir.rglob("*.yaml"):
                    try:
                        validate_cmd = [
                            "kubectl", "apply", "--dry-run=client", "-f", str(yaml_file)
                        ]
                        subprocess.run(validate_cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        validation_results["errors"].append(f"K8s validation error in {yaml_file}: {e.stderr.decode()}")
                        validation_results["valid"] = False
            
            # Check for best practices
            validation_results["recommendations"].extend([
                "Consider implementing resource quotas for all namespaces",
                "Add network policies to restrict traffic between namespaces",
                "Implement pod security policies",
                "Set up monitoring and alerting for GitOps operations"
            ])
            
            logger.info(f"Validation completed. Valid: {validation_results['valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    async def _wait_for_deployment(self, deployment_name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""
        try:
            wait_cmd = [
                "kubectl", "wait", f"deployment/{deployment_name}",
                f"--namespace={namespace}",
                "--for=condition=available",
                f"--timeout={timeout}s"
            ]
            
            subprocess.run(wait_cmd, check=True)
            return True
            
        except subprocess.CalledProcessError:
            return False
    
    def _write_yaml(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write data to YAML file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


# Global GitOps manager instance
_gitops_manager = None

def get_gitops_manager(config: GitOpsConfig) -> GitOpsManager:
    """Get or create GitOps manager instance"""
    global _gitops_manager
    if _gitops_manager is None:
        _gitops_manager = GitOpsManager(config)
    return _gitops_manager
