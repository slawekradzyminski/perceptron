#!/usr/bin/env python3
"""Unified backend check script.

Runs all quality checks: ruff, mypy, and pytest.
Run with: poetry run check

This is the single command to verify backend code quality.
Expected runtime: ~2-3 minutes on first run, faster on subsequent runs.
"""

from __future__ import annotations

import subprocess
import sys
import time


def run_command(name: str, cmd: list[str]) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running {name}...")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {name} passed ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n✗ {name} failed ({elapsed:.1f}s)")
        return False


def main() -> int:
    """Run all backend checks.

    Returns:
        0 if all checks pass, 1 if any fail
    """
    print("=" * 60)
    print("Backend Check Suite")
    print("Expected runtime: ~2-3 minutes")
    print("=" * 60)

    start_total = time.time()
    results: dict[str, bool] = {}

    # 1. Ruff (linting)
    results["Ruff (lint)"] = run_command(
        "Ruff (lint)",
        ["ruff", "check", "backend/"],
    )

    # 2. Mypy (type checking)
    results["Mypy (types)"] = run_command(
        "Mypy (types)",
        ["mypy", "backend/"],
    )

    # 3. Pytest (tests)
    results["Pytest (tests)"] = run_command(
        "Pytest (tests)",
        ["pytest", "-q"],
    )

    # Summary
    elapsed_total = time.time() - start_total
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print(f"\nTotal time: {elapsed_total:.1f}s")

    if all_passed:
        print("\n🎉 All checks passed!")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
