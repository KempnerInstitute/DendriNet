"""
Setup script for dendritic-modeling package.

This setup.py allows the dendritic-modeling project to be installed as a Python package,
making it easier to manage dependencies and distribute the code.

Usage:
    pip install -e .                    # Development install
    pip install -e .[dev]               # Install with development dependencies
    pip install -e .[test]              # Install with testing dependencies
    pip install -e .[all]               # Install with all optional dependencies
"""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from a file, filtering out comments and empty lines."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Handle inline comments
                    req = line.split('#')[0].strip()
                    if req:
                        requirements.append(req)
    return requirements


# Read the main requirements
install_requires = read_requirements('requirements.txt')

# Filter out development/testing dependencies for core install
core_packages = []
dev_packages = []
test_packages = []

for req in install_requires:
    if any(pkg in req for pkg in ['pytest', 'mypy', 'ruff', 'black', 'isort']):
        test_packages.append(req)
    elif any(pkg in req for pkg in ['ipython', 'jupyter']):
        dev_packages.append(req)
    else:
        core_packages.append(req)

# Read long description from README if available
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="dendritic-modeling",
    version="0.1.0",
    author="Dendritic Modeling Team",
    author_email="hsafaai@fas.harvard.edu",
    description="A comprehensive framework for dendritic neural network modeling and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kempner-institute/dendritic-modeling",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "dendritic_modeling": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    
    # Core dependencies
    install_requires=core_packages,
    
    # Optional dependencies
    extras_require={
        "dev": dev_packages + [
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "test": test_packages + [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "mypy>=0.900",
            "ruff>=0.0.200",
        ],
        "profiling": [
            "tensorboard>=2.10.0",
            "py-spy>=0.3.0",  # Additional profiling tool
        ],
        "distributed": [
            # PyTorch distributed is included with torch
            # Listed here for documentation
        ],
        "all": dev_packages + test_packages + [
            "tensorboard>=2.10.0",
            "py-spy>=0.3.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8,<4.0",
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "dendritic-train=dendritic_modeling.scripts.train_experiments:main",
            "dendritic-sweep=scripts.sweeps.unified.sweep_manager:main",
            "dendritic-analyze=scripts.sweeps.unified.result_analyzer:main",
        ],
    },
    
    # Project classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords for discovery
    keywords="dendritic neural-networks pytorch machine-learning neuroscience",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/kempner-institute/dendritic-modeling/issues",
        "Source": "https://github.com/kempner-institute/dendritic-modeling",
        "Documentation": "https://github.com/kempner-institute/dendritic-modeling/blob/main/README.md",
    },
) 