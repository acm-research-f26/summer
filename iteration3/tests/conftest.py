import sys
from pathlib import Path

# tests run from anywhere: put iteration3 root first so `config` and `src`
# resolve to iteration3's modules (same shim as iteration2/tests/conftest.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
