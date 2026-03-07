#!/bin/bash

# Build script for Terradev CLI v3.7.1
# This script prepares the package for PyPI upload

set -e

echo "🚀 Building Terradev CLI v3.7.1 for PyPI upload..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Update version in all files (already done)
echo "✅ Version updated to 3.7.1 in all files"

# Copy README to terradev_cli directory for PyPI
echo "📋 Copying README to terradev_cli directory..."
cp README.md terradev_cli/README.md

# Build the package
echo "📦 Building package..."
python -m build

# Check the package
echo "🔍 Checking package..."
twine check dist/*

echo "✅ Build complete! Ready to upload to PyPI with:"
echo "   twine upload dist/*"
echo ""
echo "📋 Package contents:"
ls -la dist/

echo ""
echo "🎯 v3.7.1 Features included:"
echo "  • CUDA Graph optimization with NUMA awareness"
echo "  • Passive background optimization (no CLI commands needed)"
echo "  • Model-specific optimization (transformers, CNNs, MoE)"
echo "  • NUMA topology scoring (PIX, PXB, PHB, SYS)"
echo "  • Warm pool enhancement for CUDA Graph models"
echo "  • Automatic endpoint selection for graph performance"
