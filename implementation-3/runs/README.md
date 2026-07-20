# `runs/` — full-scale ALFWorld experiment runners

This is the run infrastructure for the four variants described in
[`../PROJECT_GUIDE.md`](../PROJECT_GUIDE.md) sections 7-8 (Orchestrator System: the
hand-rolled orchestration prototypes). It does **not** touch Paper Baseline
(`framework/`, launched separately via `run_alfworld_lite.sh`) — see
PROJECT_GUIDE.md section 9 for why those two are not comparable.

## Layout

```
runs/
  common.py                   # shared run_id/results-dir/json helpers (stdlib only)
  run_all.sh                   # runs all four variants back to back, then aggregates
  aggregate_and_log.py          # appends one run's summary to ../EXPERIMENT_LOG.md
  baseline_a/
    run_baseline_a.py            # fixed heuristic, no learning
    results/<run_id>/result.json
  baseline_b/
    run_baseline_b.py            # GRPO-trained orchestrator stop policy
    results/<run_id>/{result.json, round_timings.jsonl}
  sub_agent/
    run_sub_agent.py              # sub-agent credit trainer (specialists' shared adapter)
    results/<run_id>/{result.json, round_timings.jsonl}
  seal/
    run_seal.py                   # SEAL self-editing variant
    results/<run_id>/{result.json, round_timings.jsonl, forgetting_curve.json}
```

Each variant is its own independent, standalone script — run any one alone,
or all four via `run_all.sh`. Every variant loads its own fresh copy of the
shared base model + zero-init LoRA adapters (see `model_pool.py`), so running
them as separate processes (rather than importing all four into one script)
is deliberate: no variant can silently inherit another's in-memory weight
changes and contaminate the comparison.

## Running

```bash
conda activate verl
cd implementation-3/runs

# one variant at a time, e.g.:
python baseline_a/run_baseline_a.py --num-episodes 30
python baseline_b/run_baseline_b.py --num-groups 10 --group-size 4
python sub_agent/run_sub_agent.py --num-groups 10 --group-size 4
python seal/run_seal.py --num-rounds 5

# or everything, correlated under one run_id, with a cumulative log entry at the end:
bash run_all.sh
```

Every script takes `--quick` (tiny smoke-test sizes — the same scale each
module's own `__main__` block used) to sanity-check the pipeline and get a
rough per-episode wall-clock reading on your hardware *before* committing to
a full run. `--run-id` lets you re-use an existing run_id (e.g. to re-run one
variant that failed, then re-aggregate); `run_all.sh` generates and shares one
automatically across all four.

Run `python <variant>/run_<variant>.py --help` for the full flag list and
defaults (group sizes, learning rates, cost coefficient, etc.).

## Results

Each variant's own `results/<run_id>/result.json` is self-contained (full
episode/round history, `_config` used, `_elapsed_s` wall-clock). Nothing here
is overwritten by a later run — every run_id gets its own subdirectory, so old
results stay around for comparison. These per-run JSON/JSONL files are
git-ignored (regenerable, and can get large); only the cumulative log is
tracked.

## The cumulative log

[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) is the one file in the main
`implementation-3/` folder that accumulates every `run_all.sh` invocation:
one dated Markdown section per run, with a summary table across all four
variants, the SEAL-vs-Baseline-B verdict from `compare_variants.py` (honest
`PENDING` if either didn't complete), and pointers to that run's result
files. It's append-only — read it top-to-bottom for the project's actual
experimental history, newest at the bottom.

## Known caveats (carried over from PROJECT_GUIDE.md — not fixed by this
runner infrastructure, just not hidden by it)

- **Sub-agent trainer's `keep_penalty`**: `run_sub_agent.py` passes
  `keep_penalty=-0.3` explicitly (the ALFWorld-tuned value `run_alfworld_lite.sh`
  actually uses) instead of `SubAgentGRPOConfig`'s own class default of `-3.5`
  (WebShop-tuned) — see PROJECT_GUIDE.md §7.3/§12b. This is the one behavior
  change these runners make versus calling the underlying modules' own
  `__main__` blocks unchanged.
- **The seeded-replay assumption is still unverified** (PROJECT_GUIDE.md
  §12a) — GRPO's group-normalization in the sub-agent trainer and SEAL both
  depend on it. Not addressed here; still the top item in PROJECT_GUIDE.md
  §15's priority list.
- **The cost coefficient `c=0.5`** and all group sizes/learning rates are
  still the same untuned starting points PROJECT_GUIDE.md documents — this
  infrastructure makes them CLI-configurable but doesn't tune them.
