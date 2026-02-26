#!/usr/bin/env python3
"""
Makefile for Terradev CLI Testing

Provides convenient commands for running different test suites
and managing the testing infrastructure.
"""

.PHONY: help test-setup test-unit test-integration test-all test-performance test-security test-e2e clean

# Default target
help:
	@echo "Terradev CLI Testing Commands:"
	@echo ""
	@echo "  test-setup      - Setup test environment"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests (requires API keys)"
	@echo "  test-all        - Run complete test suite"
	@echo "  test-performance - Run performance benchmarks"
	@echo "  test-security   - Run security scans"
	@echo "  test-e2e        - Run end-to-end tests"
	@echo "  clean           - Clean test artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test-unit"
	@echo "  make test-integration MOCK=true"
	@echo "  make test-all"

# Setup test environment
test-setup:
	@echo "Setting up test environment..."
	python -m venv test_env
	./test_env/bin/pip install --upgrade pip
	./test_env/bin/pip install -r requirements.txt
	./test_env/bin/pip install pytest pytest-asyncio pytest-cov black flake8 mypy bandit safety
	@echo "Test environment setup complete!"

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	./test_env/bin/python -m pytest tests/unit/ --cov=terradev_cli --cov-report=html --cov-report=xml -v

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	@if [ "$(MOCK)" = "true" ]; then \
		./test_env/bin/python tests/test_integration.py --mock --suite providers; \
	else \
		./test_env/bin/python tests/test_integration.py --suite providers; \
	fi

# Run security scans
test-security:
	@echo "Running security scans..."
	./test_env/bin/bandit -r terradev_cli/ -f json -o bandit_report.json
	./test_env/bin/safety check --json --output safety_report.json
	@echo "Security scans complete!"

# Run performance tests
test-performance:
	@echo "Running performance tests..."
	./test_env/bin/python tests/test_integration.py --performance --suite performance

# Run end-to-end tests
test-e2e:
	@echo "Running end-to-end tests..."
	./test_env/bin/python tests/ci_pipeline.py

# Run complete test suite
test-all: test-setup
	@echo "Running complete test suite..."
	./test_env/bin/python -m pytest tests/unit/ --cov=terradev_cli --cov-report=html --cov-report=xml
	./test_env/bin/python tests/test_integration.py --mock --suite providers
	./test_env/bin/python tests/ci_pipeline.py
	@echo "Complete test suite finished!"

# Code quality checks
quality-check:
	@echo "Running code quality checks..."
	./test_env/bin/black --check terradev_cli/
	./test_env/bin/flake8 terradev_cli/
	./test_env/bin/mypy terradev_cli/
	@echo "Code quality checks complete!"

# Build package
build-package:
	@echo "Building package..."
	./test_env/bin/python -m build
	@echo "Package build complete!"

# Clean test artifacts
clean:
	@echo "Cleaning test artifacts..."
	rm -rf test_env/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf test_results.xml
	rm -rf bandit_report.json
	rm -rf safety_report.json
	rm -rf ci_pipeline_report.json
	rm -rf integration_test_report.json
	rm -rf dist/
	@echo "Clean complete!"

# Generate test coverage report
coverage-report:
	@echo "Generating coverage report..."
	./test_env/bin/python -m pytest tests/unit/ --cov=terradev_cli --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Run tests with specific Python version
test-python39:
	@echo "Running tests with Python 3.9..."
	python3.9 -m venv test_env_py39
	./test_env_py39/bin/pip install --upgrade pip
	./test_env_py39/bin/pip install -r requirements.txt
	./test_env_py39/bin/pip install pytest pytest-asyncio pytest-cov
	./test_env_py39/bin/python -m pytest tests/unit/ --cov=terradev_cli
	rm -rf test_env_py39/

# Run tests with specific Python version
test-python310:
	@echo "Running tests with Python 3.10..."
	python3.10 -m venv test_env_py310
	./test_env_py310/bin/pip install --upgrade pip
	./test_env_py310/bin/pip install -r requirements.txt
	./test_env_py310/bin/pip install pytest pytest-asyncio pytest-cov
	./test_env_py310/bin/python -m pytest tests/unit/ --cov=terradev_cli
	rm -rf test_env_py310/

# Run tests with specific Python version
test-python311:
	@echo "Running tests with Python 3.11..."
	python3.11 -m venv test_env_py311
	./test_env_py311/bin/pip install --upgrade pip
	./test_env_py311/bin/pip install -r requirements.txt
	./test_env_py311/bin/pip install pytest pytest-asyncio pytest-cov
	./test_env_py311/bin/python -m pytest tests/unit/ --cov=terradev_cli
	rm -rf test_env_py311/

# Quick test for development
quick-test:
	@echo "Running quick development tests..."
	./test_env/bin/python -c "import terradev_cli; print('Import successful')"
	./test_env/bin/python -m terradev_cli --help
	@echo "Quick tests passed!"

# Monitor test execution
monitor-tests:
	@echo "Monitoring test execution..."
	watch -n 5 'ps aux | grep pytest | grep -v grep'

# Parallel test execution
parallel-tests:
	@echo "Running tests in parallel..."
	./test_env/bin/python -m pytest tests/unit/ -n auto --dist loadscope

# Generate test report
test-report:
	@echo "Generating test report..."
	./test_env/bin/python -m pytest tests/unit/ --cov=terradev_cli --cov-report=html --junit-xml=test-results.xml
	@echo "Test report generated!"

# Docker-based testing
docker-test:
	@echo "Running tests in Docker..."
	docker build -t terradev-cli-test -f Dockerfile.test .
	docker run --rm terradev-cli-test

# Integration test with real APIs (requires credentials)
live-integration-test:
	@echo "Running live integration tests..."
	@echo "Warning: This will use real cloud APIs and may incur costs!"
	@read -p "Continue? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	./test_env/bin/python tests/test_integration.py --suite providers

# Stress testing
stress-test:
	@echo "Running stress tests..."
	./test_env/bin/python tests/test_integration.py --performance --suite performance --parallel 8

# Regression testing
regression-test:
	@echo "Running regression tests..."
	./test_env/bin/python tests/test_integration.py --suite regression

# Memory leak testing
memory-test:
	@echo "Running memory leak tests..."
	./test_env/bin/python -m pytest tests/unit/ -v --tb=short | grep -E "(PASSED|FAILED|ERROR)"

# Documentation tests
docs-test:
	@echo "Testing documentation examples..."
	./test_env/bin/python -m doctest terradev_cli/*.py

# API contract testing
contract-test:
	@echo "Running API contract tests..."
	./test_env/bin/python tests/test_integration.py --suite contracts
