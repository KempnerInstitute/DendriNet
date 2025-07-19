#!/usr/bin/env python3
"""
Documentation Generation Helper Script

Simple script to generate documentation figures and build the documentation.
"""

import os
import sys
import subprocess
from pathlib import Path


def generate_figures():
    """Generate documentation figures using the proper script."""
    print("Generating documentation figures...")

    # Path to the figure generation script
    script_path = Path(__file__).parent / "generate_figures.py"

    if not script_path.exists():
        print(f"Warning: Figure generation script not found at {script_path}")
        return False

    try:
        # Run the figure generation script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], check=True, capture_output=True, text=True)

        print("Figures generated successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error generating figures: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


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
        print("HTML documentation available at: _build/html/index.html")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


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


def main():
    """Main function to build documentation with figures."""
    # Change to docs directory if not already there
    if not os.path.exists("conf.py"):
        docs_dir = Path(__file__).parent
        os.chdir(docs_dir)
        print(f"Changed to docs directory: {docs_dir}")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            clean_docs()
            return
        elif sys.argv[1] == "figures":
            generate_figures()
            return
        elif sys.argv[1] == "build":
            build_docs()
            return

    # Default: generate figures and build docs
    print("Generating figures and building documentation...")

    success = True
    success &= generate_figures()
    success &= build_docs()

    if success:
        print("\nDocumentation generation completed successfully!")
    else:
        print("\nDocumentation generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
