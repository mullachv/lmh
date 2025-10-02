#!/usr/bin/env python3
"""
Test runner for hallucination detection tests.

This script provides a convenient way to run tests with different configurations
and generate reports.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type: str = "all", coverage: bool = False, verbose: bool = True):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "manifold")
        coverage: Whether to generate coverage report
        verbose: Whether to run in verbose mode
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection based on type
    if test_type == "unit":
        cmd.extend(["-m", "not slow and not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "manifold":
        cmd.append("tests/test_manifold_detection.py")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Run the tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False


def main():
    """Main function for the test runner."""
    parser = argparse.ArgumentParser(description="Run hallucination detection tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "manifold"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run in quiet mode (less verbose output)"
    )
    
    args = parser.parse_args()
    
    success = run_tests(
        test_type=args.type,
        coverage=args.coverage,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
