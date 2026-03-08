#!/usr/bin/env python3
"""
Simple Integration Test for Terradev CLI
"""

import asyncio
import subprocess
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import click

@click.command()
@click.option('--mock/--live', default=True, help='Use mock responses or live APIs')
@click.option('--suite', help='Run specific test suite only')
def test_integration(mock, suite):
    """Run comprehensive integration tests"""
    print(f"🚀 Starting Terradev CLI Integration Test Suite")
    print(f"Test Mode: {'Mock' if mock else 'Live'}")
    
    if suite == 'core':
        success = test_core_functionality()
    elif suite == 'gitops':
        success = test_gitops_workflow()
    elif suite == 'cli':
        success = test_cli_commands()
    else:
        success = test_all_basic()
    
    if success:
        print("✅ Integration tests passed!")
    else:
        print("❌ Integration tests failed!")
        exit(1)

def test_core_functionality() -> bool:
    """Test core CLI functionality"""
    try:
        print("Testing CLI help command...")
        result = subprocess.run(
            ["./test_env/bin/python", "-m", "terradev_cli", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        assert result.returncode == 0
        assert "Terradev CLI" in result.stdout
        assert "gitops" in result.stdout
        print("✅ CLI help test passed")
        
        print("Testing version command...")
        result = subprocess.run(
            ["./test_env/bin/python", "-m", "terradev_cli", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        assert result.returncode == 0
        assert "2.9.5" in result.stdout
        print("✅ Version test passed")
        
        return True
    except Exception as e:
        print(f"Core functionality test failed: {e}")
        return False

def test_gitops_workflow() -> bool:
    """Test GitOps workflow"""
    try:
        print("Testing GitOps commands...")
        
        # Test GitOps help
        result = subprocess.run(
            ["./test_env/bin/python", "-m", "terradev_cli", "gitops", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        assert result.returncode == 0
        assert "GitOps automation" in result.stdout
        assert "init" in result.stdout
        assert "bootstrap" in result.stdout
        print("✅ GitOps help test passed")
        
        # Test GitOps init (dry run with temp dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                "./test_env/bin/python", "-c", 
                f"""
import asyncio
import sys
sys.path.insert(0, '{Path.cwd()}')
from terradev_cli.core.gitops_manager import GitOpsManager, GitOpsConfig, GitProvider, GitOpsTool
from pathlib import Path

async def test():
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,
        repository="test/infra",
        tool=GitOpsTool.ARGOCD,
        cluster_name="test-cluster"
    )

    manager = GitOpsManager(config)
    manager.work_dir = Path('{temp_dir}')

    success = await manager.init_repository()
    print(f"GitOps init result: {{success}}")
    assert success is True

    # Check repository structure
    assert (manager.work_dir / "clusters").exists()
    assert (manager.work_dir / "apps").exists()
    assert (manager.work_dir / "infra").exists()
    assert (manager.work_dir / "policies").exists()

    print("✅ GitOps repository structure created successfully")

asyncio.run(test())
                """
            ], capture_output=True, text=True, timeout=60)
            
            assert result.returncode == 0
            assert "GitOps init result: True" in result.stdout
            print("✅ GitOps init test passed")
        
        return True
    except Exception as e:
        print(f"GitOps workflow test failed: {e}")
        return False

def test_cli_commands() -> bool:
    """Test CLI commands"""
    try:
        print("Testing CLI commands...")
        
        commands = [
            ["--help"],
            ["--version"],
            ["gitops", "--help"],
            ["quote", "--help"],
            ["status", "--help"]
        ]
        
        for cmd in commands:
            full_cmd = ["./test_env/bin/python", "-m", "terradev_cli"] + cmd
            result = subprocess.run(
                full_cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            assert result.returncode == 0, f"Command failed: {' '.join(cmd)}"
            print(f"✅ Command {' '.join(cmd)} passed")
        
        return True
    except Exception as e:
        print(f"CLI commands test failed: {e}")
        return False

def test_all_basic() -> bool:
    """Run all basic tests"""
    tests = [
        test_core_functionality,
        test_gitops_workflow,
        test_cli_commands
    ]
    
    for test in tests:
        if not test():
            return False
    
    return True

if __name__ == '__main__':
    test_integration()
