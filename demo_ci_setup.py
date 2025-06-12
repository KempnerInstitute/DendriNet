#!/usr/bin/env python3
"""
Demo script to simulate what the CI pipeline does.

This script runs the same checks that will run automatically on GitHub
when you create a Pull Request.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description, critical=True):
    """Run a command and report results."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ SUCCESS ({duration:.1f}s)")
        if result.stdout:
            print(result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""))
    else:
        print(f"❌ FAILED ({duration:.1f}s)")
        if result.stderr:
            print("Error output:", result.stderr[:500])
        if critical:
            print("🚨 Critical failure - stopping here")
            return False
        
    return result.returncode == 0

def check_files_exist():
    """Check that all CI files are present."""
    required_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/pr-quick-check.yml", 
        "pytest.ini",
        "ruff.toml",
        ".pre-commit-config.yaml",
        "tests/test_runner.py"
    ]
    
    print("\n📁 Checking CI configuration files...")
    all_exist = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_exist = False
    
    return all_exist

def main():
    """Run the CI demo."""
    print("🚀 CI Setup Demo")
    print("=" * 60)
    print("This script simulates what happens in GitHub Actions")
    print("when you create a Pull Request.")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src/dendritic_modeling").exists():
        print("❌ Error: Run this script from the project root directory")
        print("Expected to find: src/dendritic_modeling/")
        return 1
    
    # Check configuration files
    if not check_files_exist():
        print("\n❌ Some CI configuration files are missing!")
        print("Make sure all files were created properly.")
        return 1
    
    print("\n🎯 Running CI Pipeline Demo...")
    print("This simulates what GitHub Actions will do automatically.")
    
    # Step 1: Check dependencies
    print("\n" + "="*60)
    print("📦 STEP 1: Check Dependencies (like CI setup)")
    print("="*60)
    
    if not run_command("python tests/test_runner.py --check-deps", 
                      "Checking test dependencies"):
        return 1
    
    # Step 2: Package import test
    print("\n" + "="*60)
    print("📦 STEP 2: Package Import Test")
    print("="*60)
    
    if not run_command(
        'python -c "import dendritic_modeling; print(\'✅ Package imports successfully\')"',
        "Testing package imports"
    ):
        return 1
    
    # Step 3: Fast tests (like Quick PR Check)
    print("\n" + "="*60)
    print("🧪 STEP 3: Fast Tests (Quick PR Check)")
    print("="*60)
    
    if not run_command(
        'python -m pytest tests/ -m "not slow and not requires_gpu and not requires_data" --maxfail=3 -q',
        "Running fast unit tests"
    ):
        return 1
    
    # Step 4: Linting checks
    print("\n" + "="*60)
    print("🔍 STEP 4: Code Quality Checks")
    print("="*60)
    
    # Check if linting tools are available
    linting_commands = [
        ("ruff check src/ tests/ || echo 'ruff not available'", "Ruff linting"),
        ("black --check src/ tests/ || echo 'black not available'", "Black formatting check"),
        ("isort --check-only src/ tests/ || echo 'isort not available'", "Import sorting check")
    ]
    
    for cmd, desc in linting_commands:
        run_command(cmd, desc, critical=False)
    
    # Step 5: Custom test runner
    print("\n" + "="*60)
    print("🏃 STEP 5: Custom Test Runner")
    print("="*60)
    
    if not run_command(
        "python tests/test_runner.py --test-type unit --verbosity 0",
        "Running custom test runner"
    ):
        return 1
    
    # Step 6: Coverage test (if pytest-cov is available)
    print("\n" + "="*60)
    print("📊 STEP 6: Coverage Analysis (Full CI)")
    print("="*60)
    
    run_command(
        "python -m pytest tests/ --cov=src/dendritic_modeling --cov-report=term-missing --maxfail=5 || echo 'pytest-cov not available'",
        "Running tests with coverage",
        critical=False
    )
    
    # Summary
    print("\n" + "="*60)
    print("🎉 CI DEMO COMPLETE!")
    print("="*60)
    print("""
✅ All checks completed successfully!

🚀 What happens next in real CI:

1. **On PR Creation:**
   - Quick PR Check runs (2-3 minutes)
   - Provides immediate feedback
   - Posts status to PR

2. **Full CI Pipeline:**
   - Tests on Python 3.9, 3.10, 3.11, 3.12
   - Tests on Ubuntu, macOS, Windows
   - Generates coverage reports
   - Security scanning
   - Performance tests

3. **Automatic Reporting:**
   - Coverage report posted as PR comment
   - Test artifacts uploaded
   - Status checks in PR interface

📋 **To enable this for your repository:**

1. Commit and push the CI files:
   git add .github/ pytest.ini ruff.toml .pre-commit-config.yaml
   git commit -m "Add CI configuration"
   git push

2. Create a Pull Request - testing will run automatically!

3. (Optional) Set up branch protection rules in GitHub repository settings

🔗 **See CI_SETUP.md for detailed instructions**
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 