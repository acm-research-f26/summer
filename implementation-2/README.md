# Implementation 2 — Full Framework Build (contains Implementation 1's work too)


## What we built (what changed since Implementation 1)

Implementation 1 left us with a working environment and a confirmed lightweight training config, but no orchestration logic at all — HiPER-agent's own repo has no concept of a multi-agent orchestrator delegating to sub-agents. Everything below is new:

**Part 1 — Sub-agent execution credit** (`agents/sub_agent_credit.py`): a critic-free, GRPO-style replacement for HiPER's own Hierarchical Advantage Estimation. HiPER learns a three-head PPO-style critic for sub-agent credit; this drops the critic entirely and computes three separate group-normalized advantages instead — action tokens get in-segment return-to-go, subgoal tokens get the segment-aggregate return, switch tokens get the full-trajectory return — while keeping HiPER's `keep_penalty`/`keep_consistency_penalty` reward shaping unchanged, since that's what prevents the sub-agent from collapsing into always-KEEP behavior. Validated with a synthetic-batch unit test; not yet wired into a live training run (that needs real WebShop rollouts, same as Implementation 1's Step 6).

**Part 2 — Two-role WebShop orchestration** (`orchestration/`), entirely new:
- `model_pool.py` — one shared, frozen base model (Qwen2.5-0.5B-Instruct) carrying **two separate LoRA adapters**: `"shared"` for both specialists, `"orchestrator"` for the orchestrator. This is a deliberate deviation from the spec's literal wording (which suggested all three roles might share one adapter) — it's what makes Part 4's "freeze sub-agents, train only the orchestrator" requirement actually possible.
- `search_specialist.py`, `compare_specialist.py`, `sub_agent.py` — the two delegated roles, sharing one Plan-Execute execution loop (they differ only in their starting subgoal and step budget).
- `orchestrator.py` — the new `{delegate-search, delegate-compare, stop}` decision, with a recorder hook so training code (Baseline B) can capture exact prompts/responses without duplicating the decision logic.
- `episode_runner.py` — wires all three roles together against a real WebShop environment, used by both baselines.

**Part 3 — Baseline A** (`orchestration/baseline_a_heuristic.py`): fixed heuristic — delegate-search, then delegate-compare, then stop. Reuses `episode_runner` with a scripted action sequence rather than a separate execution path.

**Part 4 — Baseline B** (`orchestration/baseline_b_rl_stop.py`): GRPO training of just the orchestrator's stop-policy, reward `R_episode = 10·success − c·num_delegation_rounds` (`c = 0.5`, an untuned starting point — tuning it is Implementation 3's job). Sub-agents held frozen. Implemented as a small custom PPO-clipped training loop (not verl's Ray trainer — wiring a second RL training path through verl for a 3-way categorical decision was out of scope here) that freezes everything except the `"orchestrator"` LoRA adapter.

## Validation done so far

Everything above is smoke-tested end-to-end against the real WebShop environment. A synthetic-reward test explicitly confirmed Baseline B's gradients flow into the orchestrator adapter only (168 tensors changed) while the shared sub-agent adapter stays exactly frozen (0 tensors changed). A full Baseline A run (8 episodes, the default) completes in ~2 minutes. What hasn't been done: an actual multi-iteration training run of either the sub-agent credit mechanism or Baseline B (both need real GPU time beyond what this phase used), and any tuning of `c`, LoRA rank, or group size.

**Known fragility**: during one Baseline A run, pyserini's embedded JVM (used for WebShop's search index) segfaulted in a background thread (`SIGSEGV` inside `libtorch_python.so`, called from native JVM code) without affecting that run's correctness — output was still produced and correct. This points at a real but so far non-blocking instability where PyTorch and the JVM share the same process (via pyjnius); if it starts causing actual failures, worth revisiting whether WebShop's search index really needs to live in-process alongside the model.

## How to reproduce

```bash
conda activate verl
cd .   # this folder

python -m orchestration.baseline_a_heuristic   # ~2 min for 8 episodes; writes baseline_a_results.json
python -m orchestration.baseline_b_rl_stop     # GRPO update on the orchestrator adapter only
```

## What Implementation 3 should look at first

1. The `critic/grad_norm` instability flagged in Implementation 1's results (3599 after iteration 2).
2. Whether Baseline B's stop-policy training is actually stable over more than a couple of GRPO update steps.
3. Running enough iterations to see a real reward trend — not attempted in either Implementation 1 or 2, since it needs hours of GPU time at the observed ~4 min/iteration pace.
4. The cost coefficient `c = 0.5` is an untuned placeholder, per the original spec.
