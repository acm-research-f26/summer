"""
Unified test runner - runs every test_*.py suite in this directory and
prints a single pass/fail summary, instead of invoking each file by hand.

Usage:
    python run_tests.py

Exits with status 0 if every suite passed, 1 if any failed (handy for CI or
just checking $? after a run).
"""

import glob
import os
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def discover_test_files():
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(here, "test_*.py")))


def run_one(path):
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, env=os.environ,
    )
    passed = result.returncode == 0
    return passed, result.stdout, result.stderr


def main():
    test_files = discover_test_files()
    if not test_files:
        print("No test_*.py files found.")
        return 1

    print(f"Running {len(test_files)} test suite(s)...\n")
    all_passed = True
    total_ok = []
    total_fail = []

    for path in test_files:
        name = os.path.basename(path)
        passed, stdout, stderr = run_one(path)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if passed:
            total_ok.append(name)
        else:
            all_passed = False
            total_fail.append(name)
            # show the failure detail immediately, since it's the useful part
            print("  --- stdout (tail) ---")
            for line in stdout.strip().splitlines()[-10:]:
                print("  " + line)
            print("  --- stderr (tail) ---")
            for line in stderr.strip().splitlines()[-15:]:
                print("  " + line)
            print()

    print()
    print("=" * 60)
    print(f"{len(total_ok)}/{len(test_files)} suites passed.")
    if total_fail:
        print("Failed: " + ", ".join(total_fail))
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
