"""
run_all.py
==========
End-to-end pipeline: preprocess -> train (both variants) -> evaluate.

Usage:
  python run_all.py
"""
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    run([sys.executable, "preprocess.py"])
    run([sys.executable, "train.py", "--variant", "both"])
    run([sys.executable, "evaluate.py"])


if __name__ == "__main__":
    main()
