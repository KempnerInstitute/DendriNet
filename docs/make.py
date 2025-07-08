#!/usr/bin/env python3
"""
Documentation Generation Helper Script

This script helps generate documentation figures and build the documentation.
Run this script before building the documentation to ensure all figures are up-to-date.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def generate_figures():
    """Generate all documentation figures."""
    print("Generating documentation figures...")
    
    # Create _static directory if it doesn't exist
    static_dir = Path("_static")
    static_dir.mkdir(exist_ok=True)
    
    # Generate placeholder SVG files for now
    # In a real implementation, these would be generated from actual code
    
    figures = {
        "shunting_inhibition_diagram.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Shunting Inhibition: E/(E+I+ε)
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Dendritic computation with divisive normalization
            </text>
        </svg>
        """,
        
        "basic_dendritic_neuron.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Basic Dendritic Neuron
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Multi-branch dendritic architecture
            </text>
        </svg>
        """,
        
        "multilayer_ei_network.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Multi-Layer E/I Network
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Excitatory and inhibitory populations
            </text>
        </svg>
        """,
        
        "input_network_architecture.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Input Network Architecture
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Input transformation to E/I signals
            </text>
        </svg>
        """,
        
        "training_workflow_diagram.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Training Workflow
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Complete training pipeline
            </text>
        </svg>
        """,
        
        "analysis_workflow_diagram.svg": """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="50" width="300" height="200" fill="none" stroke="black" stroke-width="2"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16">
                Analysis Workflow
            </text>
            <text x="200" y="180" text-anchor="middle" font-family="Arial" font-size="12">
                Comprehensive analysis pipeline
            </text>
        </svg>
        """
    }
    
    for filename, content in figures.items():
        filepath = static_dir / filename
        with open(filepath, 'w') as f:
            f.write(content.strip())
        print(f"Generated {filename}")
    
    print(f"Generated {len(figures)} figures in {static_dir}")

def build_docs():
    """Build the documentation using Sphinx."""
    print("Building documentation...")
    
    try:
        # Run sphinx-build
        result = subprocess.run([
            "sphinx-build", 
            "-b", "html", 
            ".", 
            "_build/html"
        ], check=True, capture_output=True, text=True)
        
        print("Documentation built successfully!")
        print(f"HTML documentation available at: _build/html/index.html")
        
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    return True

def clean_docs():
    """Clean the documentation build directory."""
    print("Cleaning documentation build directory...")
    
    build_dir = Path("_build")
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("Cleaned _build directory")
    else:
        print("_build directory does not exist")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = ["sphinx", "sphinx-rtd-theme"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies are installed!")
    return True

def main():
    """Main function to handle command line arguments and execute tasks."""
    parser = argparse.ArgumentParser(description="Documentation generation helper")
    parser.add_argument("--figures", action="store_true", help="Generate figures only")
    parser.add_argument("--build", action="store_true", help="Build documentation only")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    parser.add_argument("--all", action="store_true", help="Generate figures and build docs")
    
    args = parser.parse_args()
    
    # Change to docs directory if not already there
    if not os.path.exists("conf.py"):
        docs_dir = Path(__file__).parent
        os.chdir(docs_dir)
        print(f"Changed to docs directory: {docs_dir}")
    
    success = True
    
    if args.check or args.all:
        success &= check_dependencies()
    
    if args.clean:
        clean_docs()
    
    if args.figures or args.all:
        generate_figures()
    
    if args.build or args.all:
        success &= build_docs()
    
    if not any([args.figures, args.build, args.clean, args.check, args.all]):
        # Default behavior: generate figures and build docs
        print("No specific action specified, running default: generate figures and build docs")
        generate_figures()
        success &= build_docs()
    
    if success:
        print("\n✓ Documentation generation completed successfully!")
    else:
        print("\n✗ Documentation generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
