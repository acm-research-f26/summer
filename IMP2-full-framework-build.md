# IMPLEMENTATION 2 — Build the Full Framework Structure (Baseline A + Baseline B), Lightweight Version

Copy everything below this line into Claude Code as your instruction for this session. **Run this only after Implementation 1 (LITE) is confirmed complete** — you should have a working `acm-research-credit-rl/` project (detached from the original upstream repo, now our own codebase — see Implementation 1 LITE's Step 4.5) and a confirmed-working ultralight training config (`0.5B` model + LoRA, or whatever fallback level Implementation 1 LITE landed on) from that session. Use that config as-is here; **this session stays lightweight throughout — do not scale back up to the 1.5B/full-fine-tune setup, and do not spend this session tuning numbers.**

**Goal of this session:** get the entire framework structurally built and running end-to-end, at the lightweight scale — sub-agent credit mechanism, the 2-role WebShop orchestration architecture, and both baselines (fixed heuristic and RL-trained). It's fine, expected even, if the numbers aren't good yet and Baseline B isn't fully converged — **the goal is a complete, correctly-wired pipeline that runs without crashing at our actual compute scale, not tuned performance.** Tuning is Implementation 3, later.

---

## Work inside our own project, not the upstream repo
All work in this session happens inside `acm-research-credit-rl/` (the detached, locally-owned copy from Implementation 1 LITE's Step 4.5) — new files, new modules, everything. Do not re-clone `JonP07/HiPER-agent` or work against it directly; this is our own codebase now, and we're free to restructure it as needed (e.g., adding new top-level directories like `orchestration/` for the multi-agent code that doesn't exist in the original repo at all).

## Context and end goal
End target: an RL-trained "when should a multi-agent orchestrator stop delegating" policy — a decision a May 2026 survey (arXiv:2605.02801, "Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces," Zhang) confirmed had zero existing RL training methods. This session builds the whole scaffold that later sessions will tune and extend.

## Primary sources
1. **HiPER** — arXiv:2602.16165. Source of the Plan-Execute framework and HAE credit mechanism (Part 1). The reference implementation is now inside our own `acm-research-credit-rl/` (originally from https://github.com/JonP07/HiPER-agent, detached in Implementation 1 LITE) — read `hiper_hae/core_hae.py` there directly as the ground-truth reference, but write our *new* sub-agent/orchestrator code in new files, adapting rather than editing that reference file in place, so it stays available as a clean comparison point.
2. **GRPO** — "DeepSeekMath," Shao et al., 2024, arXiv:2402.03300 — source of the group-relative advantage formula used both for sub-agent credit and the orchestrator's decision.
3. **Orchestration-traces survey** — arXiv:2605.02801 — the paper whose gap this project fills.

## Use the ultralight config from Implementation 1 LITE as-is
Whatever final working config Implementation 1 LITE's Step 6 landed on (model size — `0.5B` unless you had to go smaller, LoRA rank, batch sizes, sequence lengths) — use those values throughout this session without re-tuning them. If adding the orchestrator layer on top pushes memory over budget, reduce the same way Implementation 1 LITE did (LoRA rank down → micro-batch to 1 → sequence length → group size, in that order) and note what changed, but don't spend session time optimizing beyond "it runs."

## Part 1 — Sub-agent execution credit (HAE, critic-free + LoRA adaptation)
Build the credit mechanism each sub-agent uses for its own multi-step execution, as a **new module** in our project (e.g., `agents/sub_agent_credit.py`), referencing but not editing the vendored `hiper_hae/core_hae.py`:
- Plan-Execute prompt structure per turn: `<switch>SWITCH|KEEP</switch><subgoal>...</subgoal><action>...</action>`.
- Keep the `keep_penalty` / `keep_consistency_penalty` reward-shaping terms from the reference implementation — these prevent degenerate switching behavior in practice, not optional extras.
- **Deliberate simplification from HiPER's own method:** replace HiPER's learned three-head PPO-style critic with group-relative (GRPO-style) advantages — sample a group of `G` rollouts (size from Implementation 1 LITE's confirmed config, likely small — 2-4), compute segment-aggregate and full-trajectory returns, normalize within-group: `A = (R − mean(group)) / (std(group) + ε)`, applied separately to switch/subgoal/action decisions (three separate `∇log π · A` terms, not one flat advantage per output).
- Train with LoRA (per Implementation 1 LITE's confirmed `lora_rank`/`lora_alpha`/`target_modules`) rather than full fine-tuning throughout — this stays lightweight, same as the compute check.
- **Document the critic-free deviation clearly in code comments** — traceable for the eventual writeup.
- Get this training (briefly, doesn't need convergence) on one sub-agent role alone before Part 2.

## Part 2 — System architecture: WebShop 2-role orchestration
```
acm-research-credit-rl/
  orchestration/                  <- new directory, doesn't exist in the original repo
    search_specialist.py      (Qwen2.5-0.5B-Instruct + LoRA, Part 1's mechanism)
    compare_specialist.py     (Qwen2.5-0.5B-Instruct + LoRA, Part 1's mechanism;
                                share weights/adapter with search-specialist as the
                                starting default — separate LoRA adapters per role
                                is a stretch goal, not required here)
    orchestrator.py           (Qwen2.5-0.5B-Instruct + LoRA; action space each round:
                                {delegate-search, delegate-compare, stop})
    episode_runner.py         (wires the three roles together end-to-end)
```
Get a full episode running end-to-end (orchestrator decides, delegates, sub-agent executes with Part 1's mechanism, result returns to orchestrator, repeat until stop or step cap) before automating training. Confirm you can read back a full episode log — instruction, each round's delegation + sub-agent output, final outcome — in a reusable format for both baselines below.

## Part 3 — Baseline A: fixed heuristic orchestrator
New file, e.g. `orchestration/baseline_a_heuristic.py`. Implement a simple, explicit, clearly-documented stopping rule (e.g., fixed round cap: delegate-search, then delegate-compare, then stop). No RL — the naive standard-practice floor. Run enough episodes for a stable success-rate and average-rounds-used estimate at our lightweight model/LoRA scale.

## Part 4 — Baseline B: RL-trained stop policy, outcome-only reward
New file, e.g. `orchestration/baseline_b_rl_stop.py`. Reward: `R_episode = 10 · success − c · (num_delegation_rounds)`, start `c = 0.5` — don't tune this yet, that's Implementation 3.

Train the orchestrator's `{delegate-search, delegate-compare, stop}` decision via GRPO, LoRA, sub-agents held **frozen** (no co-training in this pass — stretch goal). Sample a group of `G` orchestrator rollouts per task (small group size, per our compute reality), compute cost-adjusted return per rollout, normalize within-group, update with a standard PPO-style clipped objective. Single-level decision, no HAE-style split needed here.

**Metrics, same format for both baselines:** task success rate, average delegation rounds, cost-adjusted return (mean ± std).

## Deliverables at the end of this session
1. Sub-agent credit mechanism (Part 1) implemented as new project code, confirmed training without crashing at the lightweight scale.
2. Full 2-role WebShop architecture (Part 2) wired and running end-to-end, as new project code inside `orchestration/`.
3. Baseline A running with logged metrics.
4. Baseline B training with the same metrics logged in the same format — report actual progress honestly even if far from converged.
5. Confirm the project directory (`acm-research-credit-rl/`) is a clean, self-owned git repo at this point — commit the new `orchestration/` code with a clear commit message, separate from the initial detached-import commit from Implementation 1 LITE, so the history shows what's original HiPER code vs. what our team built.
6. A short written summary: what you had to deviate from this spec and why, what's structurally done vs. still rough, and what Implementation 3 should tune first.

## What NOT to do in this session
- No Variant C/D/E — later session, gated on Baseline A/B being structurally complete here.
- No multi-seed runs — one seed of each baseline, structurally working, is the target.
- No hyperparameter tuning beyond what's needed to avoid crashing.
- No scaling back up to 1.5B or full fine-tuning — stay at whatever lightweight scale Implementation 1 LITE confirmed.
- No ALFWorld.
- Don't work directly against a re-cloned upstream `JonP07/HiPER-agent` — everything happens in our own detached `acm-research-credit-rl/` project.

---

*(End of Implementation 2 prompt.)*
