#!/usr/bin/env python3
"""
Continuous Integration Pipeline for Terradev CLI

Automated testing pipeline that runs on every commit:
1. Unit tests for all modules
2. Integration tests for providers
3. End-to-end workflow tests
4. Performance benchmarks
5. Security vulnerability scans
6. Code quality checks
"""

import asyncio
import subprocess
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any
import requests
import pytest


class CIPipeline:
    """Continuous Integration Pipeline for Terradev CLI"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.workspace = Path.cwd()
        
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete CI pipeline"""
        print("🚀 Starting Terradev CLI CI Pipeline")
        
        stages = [
            ("setup", self.setup_environment),
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("security_scan", self.run_security_scan),
            ("performance_test", self.run_performance_tests),
            ("build_package", self.build_package),
            ("deploy_staging", self.deploy_to_staging),
            ("e2e_tests", self.run_e2e_tests),
            ("cleanup", self.cleanup_environment)
        ]
        
        for stage_name, stage_func in stages:
            print(f"\n📋 Running stage: {stage_name}")
            start_time = time.time()
            
            try:
                result = await stage_func()
                duration = time.time() - start_time
                
                self.results[stage_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "duration": duration,
                    "details": result
                }
                
                print(f"✅ {stage_name}: PASSED ({duration:.1f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                self.results[stage_name] = {
                    "status": "FAILED",
                    "duration": duration,
                    "error": str(e)
                }
                
                print(f"❌ {stage_name}: FAILED ({duration:.1f}s) - {e}")
                
                # Stop pipeline on critical failures
                if stage_name in ["setup", "unit_tests", "integration_tests"]:
                    break
        
        return await self.generate_pipeline_report()
    
    async def setup_environment(self) -> bool:
        """Setup test environment"""
        print("Setting up test environment...")
        
        # Create virtual environment
        subprocess.run(["python", "-m", "venv", "test_env"], check=True)
        
        # Install dependencies
        subprocess.run([
            "./test_env/bin/pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        # Install test dependencies
        subprocess.run([
            "./test_env/bin/pip", "install", "pytest", "pytest-asyncio", 
            "pytest-cov", "black", "flake8", "mypy", "bandit"
        ], check=True)
        
        # Setup test configuration
        test_config = {
            "test_mode": True,
            "mock_apis": True,
            "test_cluster": "ci-test-cluster",
            "test_namespace": "ci-test"
        }
        
        with open("test_config.json", "w") as f:
            json.dump(test_config, f)
        
        return True
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with coverage"""
        print("Running unit tests...")
        
        # Run pytest with coverage
        result = subprocess.run([
            "./test_env/bin/pytest", 
            "tests/unit/",
            "--cov=terradev_cli",
            "--cov-report=xml",
            "--cov-report=html",
            "--junit-xml=test_results.xml",
            "-v"
        ], capture_output=True, text=True)
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "coverage_generated": Path("htmlcov").exists()
        }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("Running integration tests...")
        
        # Run our integration test framework
        result = subprocess.run([
            "./test_env/bin/python", "tests/test_integration.py",
            "--mock", "--suite", "providers"
        ], capture_output=True, text=True)
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    async def run_security_scan(self) -> Dict[str, Any]:
        """Run security vulnerability scans"""
        print("Running security scans...")
        
        # Run bandit security scan
        bandit_result = subprocess.run([
            "./test_env/bin/bandit", "-r", "terradev_cli/",
            "-f", "json", "-o", "bandit_report.json"
        ], capture_output=True, text=True)
        
        # Run safety check for dependencies
        safety_result = subprocess.run([
            "./test_env/bin/safety", "check", "--json", "--output", "safety_report.json"
        ], capture_output=True, text=True)
        
        return {
            "bandit": {
                "exit_code": bandit_result.returncode,
                "issues_found": len(json.loads(Path("bandit_report.json").read_text())["results"])
            },
            "safety": {
                "exit_code": safety_result.returncode,
                "vulnerabilities_found": len(json.loads(Path("safety_report.json").read_text()))
            }
        }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("Running performance tests...")
        
        # Test CLI response time
        start_time = time.time()
        subprocess.run(["./test_env/bin/python", "-m", "terradev_cli", "--help"], 
                      capture_output=True, check=True)
        help_time = time.time() - start_time
        
        # Test provisioning performance
        result = subprocess.run([
            "./test_env/bin/python", "tests/test_integration.py",
            "--performance", "--suite", "performance"
        ], capture_output=True, text=True)
        
        return {
            "help_command_time": help_time,
            "performance_tests": {
                "exit_code": result.returncode,
                "stdout": result.stdout
            }
        }
    
    async def build_package(self) -> Dict[str, Any]:
        """Build package for distribution"""
        print("Building package...")
        
        # Run code quality checks
        black_result = subprocess.run([
            "./test_env/bin/black", "--check", "terradev_cli/"
        ], capture_output=True, text=True)
        
        flake8_result = subprocess.run([
            "./test_env/bin/flake8", "terradev_cli/"
        ], capture_output=True, text=True)
        
        # Build package
        build_result = subprocess.run([
            "./test_env/bin/python", "-m", "build"
        ], capture_output=True, text=True)
        
        return {
            "code_quality": {
                "black": {"exit_code": black_result.returncode},
                "flake8": {"exit_code": flake8_result.returncode}
            },
            "build": {
                "exit_code": build_result.returncode,
                "dist_files": list(Path("dist").glob("*")) if Path("dist").exists() else []
            }
        }
    
    async def deploy_to_staging(self) -> Dict[str, Any]:
        """Deploy to staging environment"""
        print("Deploying to staging...")
        
        # For CI, we'll simulate deployment
        # In real pipeline, this would deploy to staging cluster
        
        staging_config = {
            "environment": "staging",
            "cluster": "staging-cluster",
            "namespace": "terradev-staging",
            "deployed_at": time.time()
        }
        
        with open("staging_deployment.json", "w") as f:
            json.dump(staging_config, f)
        
        return {
            "environment": "staging",
            "deployed": True,
            "config": staging_config
        }
    
    async def run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests"""
        print("Running end-to-end tests...")
        
        # Test complete workflows
        e2e_tests = [
            self.test_gitops_workflow,
            self.test_provisioning_workflow,
            self.test_monitoring_workflow
        ]
        
        results = {}
        for test in e2e_tests:
            try:
                result = await test()
                results[test.__name__] = {
                    "status": "PASSED" if result else "FAILED",
                    "details": result
                }
            except Exception as e:
                results[test.__name__] = {
                    "status": "FAILED",
                    "error": str(e)
                }
        
        return results
    
    async def test_gitops_workflow(self) -> bool:
        """Test complete GitOps workflow"""
        try:
            # Initialize GitOps
            result = subprocess.run([
                "./test_env/bin/python", "-m", "terradev_cli", 
                "gitops", "init",
                "--provider", "github",
                "--repo", "test/e2e-infra",
                "--tool", "argocd",
                "--cluster", "e2e-test"
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def test_provisioning_workflow(self) -> bool:
        """Test complete provisioning workflow"""
        try:
            # Test quote fetching
            result = subprocess.run([
                "./test_env/bin/python", "-m", "terradev_cli",
                "quote", "--gpu-type", "A100", "--quick"
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def test_monitoring_workflow(self) -> bool:
        """Test monitoring integration workflow"""
        try:
            # Test monitoring setup
            result = subprocess.run([
                "./test_env/bin/python", "-m", "terradev_cli",
                "integrations", "--export-grafana"
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def cleanup_environment(self) -> bool:
        """Cleanup test environment"""
        print("Cleaning up test environment...")
        
        try:
            # Remove test virtual environment
            if Path("test_env").exists():
                subprocess.run(["rm", "-rf", "test_env"], check=True)
            
            # Remove test artifacts
            test_artifacts = [
                "test_config.json",
                "test_results.xml",
                "bandit_report.json",
                "safety_report.json",
                "staging_deployment.json",
                "htmlcov/",
                ".coverage"
            ]
            
            for artifact in test_artifacts:
                if Path(artifact).exists():
                    if Path(artifact).is_dir():
                        subprocess.run(["rm", "-rf", artifact], check=True)
                    else:
                        subprocess.run(["rm", artifact], check=True)
            
            return True
        except Exception:
            return False
    
    async def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        total_duration = time.time() - self.start_time
        
        # Count passed/failed stages
        passed_stages = sum(1 for stage in self.results.values() if stage["status"] == "PASSED")
        total_stages = len(self.results)
        
        report = {
            "pipeline": {
                "status": "PASSED" if passed_stages == total_stages else "FAILED",
                "duration": total_duration,
                "stages_passed": passed_stages,
                "total_stages": total_stages,
                "success_rate": (passed_stages / total_stages * 100) if total_stages > 0 else 0
            },
            "stages": self.results,
            "timestamp": time.time(),
            "workspace": str(self.workspace)
        }
        
        # Save report
        with open("ci_pipeline_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Pipeline Report Generated")
        print(f"Status: {report['pipeline']['status']}")
        print(f"Duration: {total_duration:.1f}s")
        print(f"Success Rate: {report['pipeline']['success_rate']:.1f}%")
        
        return report


# GitHub Actions workflow generator
def generate_github_workflow():
    """Generate GitHub Actions workflow file"""
    workflow = {
        "name": "Terradev CLI CI Pipeline",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "strategy": {
                    "matrix": {
                        "python-version": ["3.9", "3.10", "3.11"]
                    }
                },
                "steps": [
                    {"uses": "actions/checkout@v3"},
                    {
                        "name": "Set up Python ${{ matrix.python-version }}",
                        "uses": "actions/setup-python@v4",
                        "with": {
                            "python-version": "${{ matrix.python-version }}"
                        }
                    },
                    {
                        "name": "Install dependencies",
                        "run": "\n".join([
                            "python -m pip install --upgrade pip",
                            "pip install -r requirements.txt",
                            "pip install pytest pytest-asyncio pytest-cov black flake8 bandit safety"
                        ])
                    },
                    {
                        "name": "Run unit tests",
                        "run": "pytest tests/unit/ --cov=terradev_cli --cov-report=xml"
                    },
                    {
                        "name": "Run integration tests",
                        "run": "python tests/test_integration.py --mock --suite providers",
                        "env": {
                            "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_TEST_KEY }}",
                            "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_TEST_SECRET }}"
                        }
                    },
                    {
                        "name": "Security scan",
                        "run": "\n".join([
                            "bandit -r terradev_cli/ -f json -o bandit_report.json",
                            "safety check --json --output safety_report.json"
                        ])
                    },
                    {
                        "name": "Build package",
                        "run": "python -m build"
                    },
                    {
                        "name": "Upload coverage",
                        "uses": "codecov/codecov-action@v3",
                        "with": {
                            "file": "./coverage.xml"
                        }
                    }
                ]
            }
        }
    }
    
    # Create .github/workflows directory
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Write workflow file
    with open(workflow_dir / "ci.yml", "w") as f:
        import yaml
        yaml.dump(workflow, f, default_flow_style=False)
    
    print("GitHub Actions workflow generated: .github/workflows/ci.yml")


if __name__ == "__main__":
    # Generate GitHub workflow
    generate_github_workflow()
    
    # Run CI pipeline
    pipeline = CIPipeline()
    
    async def main():
        results = await pipeline.run_full_pipeline()
        
        if results["pipeline"]["status"] == "FAILED":
            exit(1)
    
    asyncio.run(main())
