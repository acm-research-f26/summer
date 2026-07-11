# IMPLEMENTATION 3 — Tuning From the Starting Numbers (For Later, Not Today)

Copy everything below this line into Claude Code as your instruction for this session. **Run this only after Implementation 2 is confirmed structurally complete** — you should have a working WebShop 2-role architecture with both Baseline A and Baseline B running end-to-end inside our own `acm-research-credit-rl/` project (detached from upstream in Implementation 1 LITE), at the lightweight scale (0.5B + LoRA, or whatever level Implementation 1 LITE landed on), even if Baseline B's numbers are rough.

**This session stays at the same lightweight scale as Implementations 1 and 2 unless the tuning results genuinely show we have real headroom to spare** — tuning means getting better numbers out of the same lightweight setup, not an excuse to scale back up to a heavier model or drop LoRA, unless Implementation 2's summary specifically flagged that the lightweight scale itself is the bottleneck (in which case, note that as a decision to make together before spending compute on a bigger run, not something to do unilaterally).

**Before starting, review Implementation 2's own end-of-session summary** — it should have flagged what looked wrong or unstable, which is your starting point for what to tune first, rather than guessing fresh.

---

## What "tuning" means in this session, specifically

1. **The cost penalty `c`** in `R_episode = 10 · success − c · num_delegation_rounds`. Implementation 2 used `c=0.5` as a placeholder. Look at Baseline A's actual round-count distribution (should be logged from Implementation 2) and Baseline B's behavior under `c=0.5` — is the orchestrator stopping too early (premature-stop dominates failures) or too late (over-iteration dominates)? Adjust `c` accordingly and re-run.

2. **LoRA rank and learning rate for the orchestrator's GRPO training.** If Implementation 2 flagged instability (reward not trending, KL exploding, loss spikes), start here — this could mean the LoRA rank is too low to express what's needed (try increasing it modestly, watching memory), or the learning rate carried over from the reference config doesn't suit our critic-free, LoRA-adapted setup.

3. **Group size `G` and `train_data_size`**, if Implementation 2's compute-constrained values look like they're limiting the advantage signal quality (very noisy/high-variance training curves) rather than the underlying mechanism being wrong — check whether a modest increase (only if real compute headroom remains) meaningfully stabilizes things before concluding the compute ceiling is the actual bottleneck.

4. **Multi-seed runs.** Once single-seed Baseline A vs. Baseline B looks stable and sensible, run each across 3 seeds (per the project's pre-registered evaluation plan) and report mean ± variance rather than point estimates.

5. **Convergence, not just stability.** Implementation 2's goal was "doesn't crash." This session's goal is "the numbers actually mean something" — push training long enough (informed by Implementation 1 LITE's wall-clock-per-iteration figure, so you can estimate real training-time budget at our lightweight scale) to see whether Baseline B's success rate and cost-adjusted return genuinely stabilize above Baseline A's.

## Keep working inside our own project
All tuning and new runs happen inside `acm-research-credit-rl/` — commit meaningful checkpoints (e.g., "tuned cost penalty to c=X based on Baseline A round distribution") as their own git commits in our repo's history, so the tuning process itself is traceable later, not just the final numbers.

## What comes after this session (context only, not part of this prompt)
Once Baseline A and B are properly tuned, measured, and stable, the next phase is Variant C (counterfactual credit via a simplified C3-style leave-one-out mechanism) and Variant D (LLM-generated reward shaping via potential-based shaping), compared against these now-tuned baselines — a separate prompt, written once this session's results are in, and staying at the same lightweight scale unless we've explicitly decided otherwise by then.

## Deliverables at the end of this session
1. A tuned cost penalty `c`, with a brief justification (what you observed at the old value, what changed at the new one).
2. Stable training curves for both sub-agent and orchestrator training — reward trending sensibly, no NaNs/KL blowups, documented what you changed to get there.
3. Multi-seed (3 seeds) results for Baseline A vs. Baseline B, reported as mean ± variance, not point estimates.
4. A clear statement of whether Baseline B beats Baseline A, by how much, and whether that gap looks real given the variance across seeds — treat the writeup of this accordingly, it's the project's first real result.
5. An honest list of anything still unresolved, including whether the lightweight (0.5B + LoRA) scale itself is now looking like a limiting factor worth revisiting before moving on to Variants C/D/E.

---

*(End of Implementation 3 prompt. Do not run this before Implementation 2 is confirmed complete.)*
