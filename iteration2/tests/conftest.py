import sys
from pathlib import Path

# tests run from anywhere: put iteration2 root first so `config` and `src`
# resolve to iteration2's modules (mirrors running `python -m src.x` from root)
sys.path.insert(0, str(Path(__file__).parent.parent))
