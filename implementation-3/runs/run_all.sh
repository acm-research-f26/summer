#!/usr/bin/env bash
# IMP3 -- Full ALFWorld orchestration run: Baseline A, Baseline B, the
# sub-agent credit trainer, and SEAL. Each variant is launched as its own
# independent process (deliberately -- every variant loads a fresh base model
# + zero-init LoRA adapters, so none of them silently inherits another
# variant's in-memory weight changes; that would corrupt the comparison).
#
# Results land under runs/<variant>/results/<run_id>/result.json (+ per-variant
# extras: round_timings.jsonl, forgetting_curve.json for SEAL). After all four
# finish (or fail -- a failure in one does not stop the others),
# aggregate_and_log.py appends one dated section to ../EXPERIMENT_LOG.md
# summarizing whichever variants actually produced results.
#
# Usage:
#   bash run_all.sh                 # full-scale run -- see each variant's
#                                    # run_<variant>.py --help for exact sizes
#   bash run_all.sh --quick         # tiny smoke-test sizes, to sanity-check
#                                    # the pipeline/timing on this hardware
#                                    # before committing to the full run
set -uo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

if [ "${CONDA_DEFAULT_ENV:-}" != "verl" ]; then
  echo "WARNING: conda env 'verl' does not appear to be active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset})." >&2
  echo "         Run 'conda activate verl' first -- see PROJECT_GUIDE.md section 13/14." >&2
fi

echo "=== Full ALFWorld orchestration run: $RUN_ID ==="
echo "Args forwarded to every variant: $*"

STATUSES=()
for variant in baseline_a baseline_b sub_agent seal; do
  echo ""
  echo "--- $variant ---"
  python "$RUNS_DIR/$variant/run_$variant.py" --run-id "$RUN_ID" "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    STATUSES+=("$variant: OK")
  else
    STATUSES+=("$variant: FAILED (exit $status)")
    echo "WARNING: $variant failed -- continuing with remaining variants" >&2
  fi
done

echo ""
echo "--- aggregating + logging ---"
python "$RUNS_DIR/aggregate_and_log.py" --run-id "$RUN_ID"

echo ""
echo "=== Summary for run $RUN_ID ==="
printf '%s\n' "${STATUSES[@]}"
echo "See $(dirname "$RUNS_DIR")/EXPERIMENT_LOG.md for the cumulative log entry."
