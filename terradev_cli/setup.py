#!/usr/bin/env python3
"""
Setup script for Terradev CLI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="terradev-cli",
    version="3.1.6",
    author="Terradev Team",
    author_email="team@terradev.com",
    description="Cross-cloud GPU provisioning with GitOps automation and HuggingFace Spaces deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theoddden/terradev",
    packages=find_packages(),
    py_modules=["cli"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "aws": ["boto3>=1.34.0"],
        "gcp": ["google-cloud-compute>=1.8.0"],
        "azure": ["azure-mgmt-compute>=29.0.0", "azure-identity>=1.12.0"],
        "oracle": ["oci>=2.118.0"],
        "hf": ["huggingface-hub>=0.19.0"],
        "all": [
            "boto3>=1.34.0",
            "google-cloud-compute>=1.8.0",
            "azure-mgmt-compute>=29.0.0",
            "azure-identity>=1.12.0",
            "oci>=2.118.0",
            "huggingface-hub>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "terradev=cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
