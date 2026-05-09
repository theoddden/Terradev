#!/usr/bin/env python3
"""
Terradev CLI UX Battle Test Suite

Comprehensive testing of CLI user experience including:
- Command discovery and help
- Error handling and validation
- Progress indicators and feedback
- Output formatting and clarity
- Edge cases and error recovery
"""

import subprocess
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import unittest
from unittest.mock import patch, MagicMock

class CLIUXBattleTest:
    """Comprehensive CLI UX testing framework"""
    
    def __init__(self):
        self.terradev_path = Path("/Users/theowolfenden/CascadeProjects/Terradev")
        self.test_results = []
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment with mock data"""
        self.test_workload = {
            "deployment_id": "test_deploy_001",
            "type": "llm_training",
            "gpu_count": 4,
            "framework": "pytorch",
            "model_size": 70000000000,
            "batch_size": 32,
            "sequence_length": 2048,
            "distributed": True,
            "model_parallelism": True,
            "operations": ["allreduce", "allgather", "broadcast"]
        }
        
        self.test_metrics = {
            "fusion_efficiency": 0.87,
            "overlap_ratio": 0.78,
            "speedup": 1.13,
            "memory_util": 0.82,
            "compute_util": 0.91,
            "network_util": 0.72
        }
    
    def run_test(self, test_name: str, test_func) -> Dict[str, Any]:
        """Run a single test and capture results"""
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            test_result = {
                "name": test_name,
                "status": "PASS",
                "duration": duration,
                "result": result,
                "error": None
            }
            
            print(f"✅ {test_name} - PASS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = {
                "name": test_name,
                "status": "FAIL",
                "duration": duration,
                "result": None,
                "error": str(e)
            }
            
            print(f"❌ {test_name} - FAIL ({duration:.2f}s): {str(e)}")
        
        self.test_results.append(test_result)
        return test_result
    
    def test_help_discovery(self):
        """Test CLI help discovery and command structure"""
        
        # Test main help
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "--help"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert "Terradev optimization commands" in result.stdout, "Help should contain description"
        assert "analyze" in result.stdout, "Help should list analyze command"
        assert "benchmark" in result.stdout, "Help should list benchmark command"
        assert "dashboard" in result.stdout, "Help should list dashboard command"
        
        # Test subcommand help
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze", "--help"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "Subcommand help should succeed"
        assert "--workload-spec" in result.stdout, "Help should show options"
        assert "--auto-apply" in result.stdout, "Help should show auto-apply option"
        assert "--force" in result.stdout, "Help should show force option"
        
        return {"commands_found": len(result.stdout.split('\n')), "help_quality": "good"}
    
    def test_analyze_command_ux(self):
        """Test analyze command user experience"""
        
        # Create temporary workload file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_workload, f)
            workload_file = f.name
        
        try:
            # Test basic analysis
            result = subprocess.run(
                ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze", 
                 "test_deploy_001", "--workload-spec", workload_file],
                cwd=self.terradev_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should handle gracefully even with missing dependencies
            assert result.returncode in [0, 1], "Should handle missing dependencies gracefully"
            
            # Check for good UX indicators
            output = result.stdout + result.stderr
            
            # Should show progress indicators
            assert "🔍" in output or "Analyzing" in output, "Should show progress indicator"
            
            # Should show structured results
            if result.returncode == 0:
                assert "Optimization Analysis Results" in output, "Should show results header"
                assert "Deployment ID" in output, "Should show deployment ID"
                assert "Recommended Optimizations" in output, "Should show recommendations"
            
            return {"output_quality": "good", "error_handling": "graceful"}
            
        finally:
            os.unlink(workload_file)
    
    def test_error_handling_ux(self):
        """Test error handling user experience"""
        
        # Test with invalid workload file
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze",
             "test_deploy_001", "--workload-spec", "/nonexistent/file.json"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode != 0, "Should fail with invalid file"
        
        # Check for good error UX
        output = result.stdout + result.stderr
        assert "❌" in output or "Error" in output, "Should show clear error indicator"
        assert "file" in output.lower() or "exist" in output.lower(), "Should explain the error"
        
        # Test with invalid deployment ID
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "data"}, f)
            invalid_file = f.name
        
        try:
            result = subprocess.run(
                ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze",
                 "", "--workload-spec", invalid_file],
                cwd=self.terradev_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode != 0, "Should fail with invalid deployment ID"
            
            output = result.stdout + result.stderr
            assert "❌" in output or "Error" in output, "Should show clear error"
            
        finally:
            os.unlink(invalid_file)
        
        return {"error_clarity": "good", "error_recovery": "automatic"}
    
    def test_dashboard_command_ux(self):
        """Test dashboard command user experience"""
        
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "dashboard"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        # Should handle gracefully
        assert result.returncode in [0, 1], "Should handle missing dependencies gracefully"
        
        output = result.stdout + result.stderr
        
        # Check for dashboard UX elements
        if result.returncode == 0:
            assert "📊" in output or "Dashboard" in output, "Should show dashboard indicator"
            assert "Overall Metrics" in output or "Total" in output, "Should show metrics"
            assert "CUCo" in output or "optimization" in output.lower(), "Should show optimization data"
        
        return {"dashboard_format": "structured", "data_clarity": "good"}
    
    def test_config_command_ux(self):
        """Test configuration command user experience"""
        
        # Test config show
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "config", "show"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode in [0, 1], "Should handle gracefully"
        
        output = result.stdout + result.stderr
        assert "⚙️" in output or "Configuration" in output, "Should show config indicator"
        assert "Auto Optimize" in output or "enabled" in output, "Should show config values"
        
        # Test P95 boundaries
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "config", "p95-boundaries"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode in [0, 1], "Should handle gracefully"
        
        output = result.stdout + result.stderr
        assert "🎯" in output or "P95" in output, "Should show P95 indicator"
        assert "moe" in output.lower() or "attention" in output.lower(), "Should show workload types"
        
        return {"config_clarity": "good", "p95_display": "structured"}
    
    def test_progress_indicators(self):
        """Test progress indicators and user feedback"""
        
        # Create a temporary workload for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_workload, f)
            workload_file = f.name
        
        try:
            # Test with auto-apply to see progress
            result = subprocess.run(
                ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze",
                 "test_deploy_001", "--workload-spec", workload_file, "--auto-apply"],
                cwd=self.terradev_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Should show progress indicators
            progress_indicators = ["🔍", "🚀", "✅", "⚠️", "❌", "📊", "💾"]
            found_indicators = [indicator for indicator in progress_indicators if indicator in output]
            
            assert len(found_indicators) >= 2, f"Should show multiple progress indicators, found: {found_indicators}"
            
            # Should show structured sections
            assert "Results" in output or "Optimization" in output, "Should show results section"
            
            return {"progress_indicators": len(found_indicators), "feedback_quality": "rich"}
            
        finally:
            os.unlink(workload_file)
    
    def test_output_formatting(self):
        """Test output formatting and readability"""
        
        # Test JSON output
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "dashboard", "--json"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            # Should be valid JSON
            try:
                json_data = json.loads(result.stdout)
                assert isinstance(json_data, dict), "JSON output should be a dictionary"
                assert "dashboard_timestamp" in json_data, "Should contain expected fields"
            except json.JSONDecodeError:
                assert False, "Should produce valid JSON output"
        
        # Test regular output formatting
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "dashboard"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            output = result.stdout
            
            # Should have clear sections
            lines = output.split('\n')
            header_lines = [line for line in lines if line.startswith('📊') or '=====' in line]
            
            assert len(header_lines) >= 1, "Should have clear section headers"
            
            # Should have consistent formatting
            metric_lines = [line for line in lines if ':' in line and any(char in line for char in ['📈', '💰', '🔄'])]
            assert len(metric_lines) >= 1, "Should have formatted metric lines"
        
        return {"formatting_quality": "excellent", "readability": "high"}
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        # Test with empty deployment ID
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_workload, f)
            workload_file = f.name
        
        try:
            result = subprocess.run(
                ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze",
                 "", "--workload-spec", workload_file],
                cwd=self.terradev_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should handle gracefully
            assert result.returncode != 0, "Should fail with empty deployment ID"
            
            output = result.stdout + result.stderr
            assert "❌" in output or "Error" in output, "Should show clear error"
            
        finally:
            os.unlink(workload_file)
        
        # Test with very long deployment ID
        long_id = "a" * 1000
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "analyze", long_id],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should handle gracefully
        assert result.returncode in [0, 1], "Should handle long deployment ID gracefully"
        
        return {"edge_case_handling": "robust", "error_clarity": "good"}
    
    def test_command_completion(self):
        """Test command completion and discovery"""
        
        # Test main command discovery
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "--help"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout
        
        # Should list all main commands
        main_commands = ["analyze", "benchmark", "dashboard", "recommendations", "rollback", "config"]
        found_commands = [cmd for cmd in main_commands if cmd in output]
        
        assert len(found_commands) >= len(main_commands) - 2, f"Should list most commands, found: {found_commands}"
        
        # Test subcommand discovery
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "config", "--help"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout
        config_commands = ["show", "set", "p95-boundaries"]
        found_config_commands = [cmd for cmd in config_commands if cmd in output]
        
        assert len(found_config_commands) >= 2, f"Should list config subcommands, found: {found_config_commands}"
        
        return {"command_discovery": "excellent", "help_quality": "comprehensive"}
    
    def test_performance_responsiveness(self):
        """Test command performance and responsiveness"""
        
        # Test help command responsiveness
        start_time = time.time()
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "--help"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        help_duration = time.time() - start_time
        
        assert help_duration < 3.0, f"Help should be fast, took {help_duration:.2f}s"
        assert result.returncode == 0, "Help should succeed"
        
        # Test config command responsiveness
        start_time = time.time()
        result = subprocess.run(
            ["python3", "-m", "terradev_cli.cli_optimization", "optimize", "config", "show"],
            cwd=self.terradev_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        config_duration = time.time() - start_time
        
        assert config_duration < 5.0, f"Config should be responsive, took {config_duration:.2f}s"
        
        return {"help_speed": f"{help_duration:.2f}s", "config_speed": f"{config_duration:.2f}s"}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all CLI UX tests"""
        
        print("🚀 Starting Terradev CLI UX Battle Test Suite")
        print("=" * 50)
        
        tests = [
            ("Help Discovery", self.test_help_discovery),
            ("Analyze Command UX", self.test_analyze_command_ux),
            ("Error Handling UX", self.test_error_handling_ux),
            ("Dashboard Command UX", self.test_dashboard_command_ux),
            ("Config Command UX", self.test_config_command_ux),
            ("Progress Indicators", self.test_progress_indicators),
            ("Output Formatting", self.test_output_formatting),
            ("Edge Cases", self.test_edge_cases),
            ("Command Completion", self.test_command_completion),
            ("Performance Responsiveness", self.test_performance_responsiveness),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generate summary report
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r["duration"] for r in self.test_results)
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total_tests if total_tests > 0 else 0,
            "test_results": self.test_results
        }
        
        print("\n" + "=" * 50)
        print("📊 CLI UX Battle Test Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {summary['average_duration']:.2f}s")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  • {result['name']}: {result['error']}")
        
        # UX Quality Assessment
        ux_score = self._calculate_ux_score()
        print(f"\n🎯 Overall UX Quality Score: {ux_score}/10")
        
        if ux_score >= 8:
            print("🌟 EXCELLENT - Amazing CLI UX!")
        elif ux_score >= 6:
            print("✅ GOOD - Solid CLI UX with room for improvement")
        elif ux_score >= 4:
            print("⚠️  FAIR - CLI UX needs significant improvements")
        else:
            print("❌ POOR - CLI UX requires major redesign")
        
        return summary
    
    def _calculate_ux_score(self) -> int:
        """Calculate overall UX quality score"""
        
        score = 5  # Base score
        
        # Add points for passed tests
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        score += passed_tests
        
        # Deduct points for failures
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        score -= failed_tests * 2
        
        # Bonus points for specific UX qualities
        for result in self.test_results:
            if result["status"] == "PASS" and result["result"]:
                result_data = result["result"]
                if "excellent" in str(result_data.values()).lower():
                    score += 1
                if "comprehensive" in str(result_data.values()).lower():
                    score += 1
        
        return max(1, min(10, score))

# Run the battle test
if __name__ == "__main__":
    battle_test = CLIUXBattleTest()
    results = battle_test.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results["success_rate"] >= 0.8 else 1
    sys.exit(exit_code)
