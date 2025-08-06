"""
Test runner for dendritic_modeling package.

This script provides a unified interface for running tests with various configurations:
- Run all tests or specific modules
- Generate coverage reports
- Run with different verbosity levels
- Integration with CI/CD pipelines
"""

import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def discover_tests(test_dir="tests", pattern="test_*.py"):
    """
    Discover all test cases in the test directory.

    Args:
        test_dir: Directory containing test files
        pattern: Pattern to match test files

    Returns:
        Test suite containing all discovered tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    return suite


def run_tests_basic(verbosity=2, failfast=False):
    """
    Run tests using basic unittest runner.

    Args:
        verbosity: Level of verbosity (0-2)
        failfast: Stop on first failure

    Returns:
        TestResult object
    """
    suite = discover_tests()
    runner = unittest.TextTestRunner(
        verbosity=verbosity, failfast=failfast, stream=sys.stdout
    )
    result = runner.run(suite)
    return result


def run_tests_with_coverage(verbosity=2, coverage_report=True, html_report=False):
    """
    Run tests with coverage analysis using pytest-cov.

    Args:
        verbosity: Level of verbosity
        coverage_report: Whether to generate coverage report
        html_report: Whether to generate HTML coverage report

    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest", "tests/", f"-{'v' * verbosity}", "--tb=short"]

    if coverage_report:
        cmd.extend(["--cov=src/dendritic_modeling", "--cov-report=term-missing"])

        if html_report:
            cmd.append("--cov-report=html")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_specific_tests(test_modules, verbosity=2):
    """
    Run specific test modules.

    Args:
        test_modules: list of test module names (without .py extension)
        verbosity: Level of verbosity

    Returns:
        TestResult object
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add current directory to path for imports
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    for module_name in test_modules:
        try:
            # Import the test module directly
            if os.path.exists(f"tests/{module_name}.py"):
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    module_name, f"tests/{module_name}.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Load tests from the module
                module_suite = loader.loadTestsFromModule(module)
                suite.addTest(module_suite)
            else:
                print(f"Warning: Test file 'tests/{module_name}.py' not found")
        except Exception as e:
            print(f"Warning: Could not import test module '{module_name}': {e}")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result


def run_integration_tests(verbosity=2):
    """
    Run integration tests that test interactions between modules.

    Args:
        verbosity: Level of verbosity

    Returns:
        TestResult object
    """
    # Integration tests are typically marked with a pattern or in specific files
    integration_modules = ["test_train_experiments", "test_end_to_end"]  # If it exists

    return run_specific_tests(integration_modules, verbosity)


def run_unit_tests(verbosity=2):
    """
    Run unit tests (excludes integration tests).

    Args:
        verbosity: Level of verbosity

    Returns:
        TestResult object
    """
    unit_test_modules = [
        "test_analysis",
        "test_datasets",
        "test_training_strategies",
        "test_plotting",
        "test_config",
        "test_models",
        "test_networks",
        "test_utils",
    ]

    return run_specific_tests(unit_test_modules, verbosity)


def check_dependencies():
    """
    Check if required testing dependencies are available.

    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {
        "pytest": False,
        "pytest-cov": False,
        "coverage": False,
        "unittest": True,  # Built-in
    }

    # Check pytest
    try:
        import pytest
        dependencies["pytest"] = True
    except ImportError:
        pass

    # Check coverage
    try:
        import coverage
        dependencies["coverage"] = True
    except ImportError:
        pass

    # Check pytest-cov (direct import)
    try:
        import pytest_cov
        dependencies["pytest-cov"] = True
    except ImportError:
        pass

    return dependencies


def print_test_summary(result):
    """
    Print a summary of test results.

    Args:
        result: TestResult object or exit code
    """
    if isinstance(result, int):
        if result == 0:
            print("\nAll tests passed!")
            return 0
        else:
            print(f"\nTests failed with exit code {result}")
            return result

    # Handle unittest.TestResult
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_tests}")
    print(f"Successes: {total_tests - failures - errors - skipped}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")

    if failures > 0:
        print(f"\n{failures} test(s) failed")
    if errors > 0:
        print(f"\n{errors} test(s) had errors")
    if failures == 0 and errors == 0:
        print("\nAll tests passed!")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for dendritic_modeling package"
    )

    parser.add_argument(
        "--test-type",
        choices=["all", "unit", "integration", "coverage"],
        default="all",
        help="Type of tests to run",
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific test modules to run (without test_ prefix or .py extension)",
    )

    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Verbosity level (0=quiet, 1=normal, 2=verbose)",
    )

    parser.add_argument(
        "--failfast", "-f", action="store_true", help="Stop on first failure"
    )

    parser.add_argument(
        "--html-coverage",
        action="store_true",
        help="Generate HTML coverage report (requires pytest-cov)",
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Check testing dependencies and exit"
    )

    parser.add_argument(
        "--use-pytest",
        action="store_true",
        help="Use pytest instead of unittest (recommended for coverage)",
    )

    args = parser.parse_args()

    # Check dependencies if requested
    if args.check_deps:
        deps = check_dependencies()
        print("Testing Dependencies:")
        for dep, available in deps.items():
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            print(f"  {dep}: {status}")
        return 0 if all(deps.values()) else 1

    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print(
            "Error: tests directory not found. Run this script from the project root."
        )
        return 1

    # Run specific modules if requested
    if args.modules:
        # Add test_ prefix if not present
        modules = [
            mod if mod.startswith("test_") else f"test_{mod}" for mod in args.modules
        ]
        result = run_specific_tests(modules, args.verbosity)
        print_test_summary(result)
        return 0 if result.wasSuccessful() else 1

    # Run tests based on type
    if args.test_type == "coverage" or args.use_pytest:
        deps = check_dependencies()
        if not deps["pytest"]:
            print(
                "Error: pytest not available. Install with: pip install pytest pytest-cov"
            )
            return 1

        result = run_tests_with_coverage(
            verbosity=args.verbosity,
            coverage_report=True,
            html_report=args.html_coverage,
        )
        print_test_summary(result)
        return result

    elif args.test_type == "unit":
        result = run_unit_tests(args.verbosity)
        print_test_summary(result)
        return 0 if result.wasSuccessful() else 1

    elif args.test_type == "integration":
        result = run_integration_tests(args.verbosity)
        print_test_summary(result)
        return 0 if result.wasSuccessful() else 1

    else:  # "all"
        result = run_tests_basic(args.verbosity, args.failfast)
        print_test_summary(result)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
