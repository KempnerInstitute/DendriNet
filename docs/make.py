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
    
    # Generate improved SVG figures
    figures = {
        "shunting_inhibition_diagram.svg": """
        <svg width="500" height="350" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="black" />
                </marker>
            </defs>
            
            <!-- Title -->
            <text x="250" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">
                Shunting Inhibition in Dendritic Computation
            </text>
            
            <!-- Soma -->
            <circle cx="250" cy="200" r="20" fill="#8B4513" stroke="black" stroke-width="2"/>
            <text x="250" y="240" text-anchor="middle" font-family="Arial" font-size="12">Soma</text>
            
            <!-- Primary dendrite -->
            <line x1="250" y1="180" x2="250" y2="120" stroke="black" stroke-width="4"/>
            
            <!-- Dendritic branches -->
            <!-- Branch 1 -->
            <line x1="250" y1="140" x2="180" y2="80" stroke="black" stroke-width="3"/>
            <circle cx="180" cy="80" r="8" fill="#4CAF50" stroke="black" stroke-width="1"/>
            <text x="180" y="65" text-anchor="middle" font-family="Arial" font-size="10">D1</text>
            
            <!-- Branch 2 -->
            <line x1="250" y1="140" x2="320" y2="80" stroke="black" stroke-width="3"/>
            <circle cx="320" cy="80" r="8" fill="#4CAF50" stroke="black" stroke-width="1"/>
            <text x="320" y="65" text-anchor="middle" font-family="Arial" font-size="10">D2</text>
            
            <!-- Excitatory inputs (red) -->
            <line x1="120" y1="70" x2="172" y2="80" stroke="red" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="115" y="65" text-anchor="middle" font-family="Arial" font-size="10" fill="red">E</text>
            
            <line x1="380" y1="70" x2="328" y2="80" stroke="red" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="385" y="65" text-anchor="middle" font-family="Arial" font-size="10" fill="red">E</text>
            
            <!-- Inhibitory inputs (blue) -->
            <line x1="120" y1="90" x2="172" y2="80" stroke="blue" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="115" y="95" text-anchor="middle" font-family="Arial" font-size="10" fill="blue">I</text>
            
            <line x1="380" y1="90" x2="328" y2="80" stroke="blue" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="385" y="95" text-anchor="middle" font-family="Arial" font-size="10" fill="blue">I</text>
            
            <!-- Upstream dendritic connections -->
            <line x1="200" y1="100" x2="188" y2="80" stroke="purple" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="195" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="purple">D→D</text>
            
            <line x1="300" y1="100" x2="312" y2="80" stroke="purple" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="305" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="purple">D→D</text>
            
            <!-- Computation formula -->
            <rect x="150" y="270" width="200" height="50" fill="#f0f0f0" stroke="black" stroke-width="1" rx="5"/>
            <text x="250" y="290" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold">
                V = E/(E+I+ε)
            </text>
            <text x="250" y="310" text-anchor="middle" font-family="Arial" font-size="12">
                Divisive Normalization
            </text>
        </svg>
        """,
        
        "basic_dendritic_neuron.svg": """
        <svg width="450" height="350" xmlns="http://www.w3.org/2000/svg">
            <!-- Title -->
            <text x="225" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">
                Basic Dendritic Neuron
            </text>
            
            <!-- Soma -->
            <circle cx="225" cy="200" r="25" fill="#8B4513" stroke="black" stroke-width="2"/>
            <text x="225" y="240" text-anchor="middle" font-family="Arial" font-size="12">Soma</text>
            
            <!-- Axon -->
            <line x1="225" y1="225" x2="225" y2="300" stroke="black" stroke-width="4"/>
            <text x="240" y="270" font-family="Arial" font-size="12">Axon</text>
            
            <!-- Dendrites -->
            <!-- Main dendritic trunk -->
            <line x1="225" y1="175" x2="225" y2="100" stroke="black" stroke-width="4"/>
            
            <!-- Left dendrites -->
            <line x1="225" y1="130" x2="150" y2="80" stroke="black" stroke-width="3"/>
            <line x1="225" y1="150" x2="170" y2="120" stroke="black" stroke-width="3"/>
            <line x1="225" y1="170" x2="180" y2="160" stroke="black" stroke-width="3"/>
            
            <!-- Right dendrites -->
            <line x1="225" y1="130" x2="300" y2="80" stroke="black" stroke-width="3"/>
            <line x1="225" y1="150" x2="280" y2="120" stroke="black" stroke-width="3"/>
            <line x1="225" y1="170" x2="270" y2="160" stroke="black" stroke-width="3"/>
            
            <!-- Secondary branches -->
            <line x1="150" y1="80" x2="120" y2="60" stroke="black" stroke-width="2"/>
            <line x1="150" y1="80" x2="130" y2="50" stroke="black" stroke-width="2"/>
            <line x1="300" y1="80" x2="330" y2="60" stroke="black" stroke-width="2"/>
            <line x1="300" y1="80" x2="320" y2="50" stroke="black" stroke-width="2"/>
            
            <!-- Synapses (small circles) -->
            <circle cx="120" cy="60" r="3" fill="red"/>
            <circle cx="130" cy="50" r="3" fill="blue"/>
            <circle cx="140" cy="70" r="3" fill="red"/>
            <circle cx="160" cy="100" r="3" fill="blue"/>
            <circle cx="180" cy="140" r="3" fill="red"/>
            
            <circle cx="330" cy="60" r="3" fill="blue"/>
            <circle cx="320" cy="50" r="3" fill="red"/>
            <circle cx="310" cy="70" r="3" fill="blue"/>
            <circle cx="290" cy="100" r="3" fill="red"/>
            <circle cx="270" cy="140" r="3" fill="blue"/>
            
            <!-- Legend -->
            <text x="50" y="320" font-family="Arial" font-size="12" font-weight="bold">Legend:</text>
            <circle cx="80" cy="330" r="3" fill="red"/>
            <text x="90" y="335" font-family="Arial" font-size="11">Excitatory Synapses</text>
            <circle cx="220" cy="330" r="3" fill="blue"/>
            <text x="230" y="335" font-family="Arial" font-size="11">Inhibitory Synapses</text>
            
            <!-- Dendrites label -->
            <text x="100" y="40" font-family="Arial" font-size="12" font-weight="bold">Dendrites</text>
        </svg>
        """,
        
        "multilayer_ei_network.svg": """
        <svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">
            <!-- Title -->
            <text x="250" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">
                Multi-Layer Excitation-Inhibition Network
            </text>
            
            <!-- Layer 1 -->
            <text x="100" y="50" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Layer 1</text>
            <!-- Excitatory neurons -->
            <circle cx="80" cy="80" r="8" fill="red" stroke="black"/>
            <circle cx="80" cy="110" r="8" fill="red" stroke="black"/>
            <circle cx="80" cy="140" r="8" fill="red" stroke="black"/>
            <!-- Inhibitory neurons -->
            <circle cx="120" cy="95" r="8" fill="blue" stroke="black"/>
            <circle cx="120" cy="125" r="8" fill="blue" stroke="black"/>
            
            <!-- Layer 2 -->
            <text x="250" y="50" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Layer 2</text>
            <!-- Excitatory neurons -->
            <circle cx="230" cy="90" r="8" fill="red" stroke="black"/>
            <circle cx="230" cy="120" r="8" fill="red" stroke="black"/>
            <!-- Inhibitory neurons -->
            <circle cx="270" cy="105" r="8" fill="blue" stroke="black"/>
            
            <!-- Layer 3 -->
            <text x="400" y="50" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Layer 3</text>
            <!-- Output neurons -->
            <circle cx="380" cy="105" r="8" fill="green" stroke="black"/>
            
            <!-- Connections -->
            <!-- Layer 1 to Layer 2 -->
            <line x1="88" y1="80" x2="222" y2="90" stroke="gray" stroke-width="1"/>
            <line x1="88" y1="110" x2="222" y2="120" stroke="gray" stroke-width="1"/>
            <line x1="128" y1="95" x2="262" y2="105" stroke="gray" stroke-width="1"/>
            
            <!-- Layer 2 to Layer 3 -->
            <line x1="238" y1="90" x2="372" y2="105" stroke="gray" stroke-width="1"/>
            <line x1="238" y1="120" x2="372" y2="105" stroke="gray" stroke-width="1"/>
            
            <!-- Legend -->
            <text x="50" y="220" font-family="Arial" font-size="12" font-weight="bold">Legend:</text>
            <circle cx="80" cy="240" r="8" fill="red" stroke="black"/>
            <text x="95" y="245" font-family="Arial" font-size="11">Excitatory Neurons</text>
            <circle cx="80" cy="260" r="8" fill="blue" stroke="black"/>
            <text x="95" y="265" font-family="Arial" font-size="11">Inhibitory Neurons</text>
            <circle cx="250" cy="240" r="8" fill="green" stroke="black"/>
            <text x="265" y="245" font-family="Arial" font-size="11">Output Neurons</text>
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
