#!/usr/bin/env python3
"""
Shared pytest fixtures and configuration for Terradev test suite.
"""

import os
import sys

# Ensure the project root is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
