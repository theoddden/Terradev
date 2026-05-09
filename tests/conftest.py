#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration for Terradev test suite.
"""

import os
import sys
import pytest

# Ensure the project root is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def base_url():
    """Base URL for telemetry server testing."""
    return "http://localhost:8080"

@pytest.fixture
def server_name():
    """Server name for telemetry server testing."""
    return "Primary Server"
