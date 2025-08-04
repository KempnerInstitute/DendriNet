"""
Setup script for dendritic-modeling package.

This setup.py provides backwards compatibility and allows the package to use pyproject.toml
for configuration while maintaining compatibility with older pip versions.

Usage:
    pip install -e .                    # Development install
    pip install -e .[dev]               # Install with development dependencies
    pip install -e .[test]              # Install with testing dependencies
    pip install -e .[all]               # Install with all optional dependencies
"""

from setuptools import setup

# All configuration is now handled by pyproject.toml
# This setup.py exists for backwards compatibility
setup() 