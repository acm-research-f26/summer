# IMP3 runs/ -- shared plumbing for every variant's runner script:
# resolving research/ onto sys.path (so `import orchestration.*`/`agents.*`
# works regardless of the caller's cwd), a run_id/results-dir convention every
# variant follows identically, and json read/write helpers. Kept dependency-
# free (stdlib only) so importing it never requires the `verl` conda env.

import json
import os
import sys
import time

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPLEMENTATION3_DIR = os.path.dirname(RUNS_DIR)
RESEARCH_DIR = os.path.join(IMPLEMENTATION3_DIR, "research")
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

EXPERIMENT_LOG_PATH = os.path.join(IMPLEMENTATION3_DIR, "EXPERIMENT_LOG.md")


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_dir(variant: str, run_id: str) -> str:
    """runs/<variant>/results/<run_id>/ -- every variant's own output
    directory for one run_id, created on first use. A single full run_all.sh
    invocation shares one run_id across all four variants so their results
    correlate; a standalone `python run_<variant>.py` (no --run-id) mints its
    own timestamp instead."""
    d = os.path.join(RUNS_DIR, variant, "results", run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
