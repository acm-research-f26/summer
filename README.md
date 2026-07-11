# Credit Assignment for Multi-Agent LLM Orchestration

## 📌 Project Summary

This project studies how a multi-agent LLM system should decide **when to stop delegating**. A central orchestrator delegates sub-tasks to specialist agents (a search specialist and a compare/purchase specialist) inside the [WebShop](https://github.com/princeton-nlp/WebShop) simulated e-commerce environment, and must itself learn — via reinforcement learning — when further delegation is worth its cost and when to stop and act. The work builds on [HiPER](https://arxiv.org/abs/2602.16165) (Hierarchical Plan-Execute RL with explicit credit assignment) for the sub-agent layer and GRPO ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) for group-relative policy optimization, adapting both to run at a lightweight, single-GPU compute scale (Qwen2.5-0.5B-Instruct + LoRA).

**Current status**: Implementation 1 (environment setup + ultralight compute check, `implementation-1/`) and Implementation 2 (full orchestration framework: sub-agent credit mechanism, both baselines, `implementation-2/`) are both structurally complete — see each folder's README for exactly what's validated vs. still rough. Implementation 3 (tuning, multi-seed evaluation, and an actual multi-hour training run) has not started yet.

## 🎯 Motivation

Multi-agent LLM systems increasingly rely on a central orchestrator that delegates to specialist sub-agents or tools. In practice, the decision of *when to stop delegating and commit to an answer* is almost always hand-coded — a fixed round cap, a heuristic, or the orchestrator simply running out of budget — rather than learned. A May 2026 survey of RL methods for multi-agent LLM orchestration (arXiv:2605.02801) confirmed that **zero existing RL training methods** target this stopping decision directly. Over-delegation wastes inference cost and latency; under-delegation leaves tasks unfinished. This project trains that stopping decision directly, rather than leaving it to a fixed rule.

## 🧩 Novelty

- **Critic-free hierarchical credit assignment**: HiPER's own mechanism learns a three-head PPO-style critic (low/high/termination value heads) for sub-agent credit. We replace it with a critic-free, GRPO-style variant — group-relative advantages computed from sampled rollout groups instead of a learned value function — cutting the memory overhead a critic requires and making training feasible at a lightweight (0.5B + LoRA), single-GPU scale.
- **A learned stop-policy, not a heuristic**: the orchestrator's `{delegate-search, delegate-compare, stop}` decision is trained via GRPO against an outcome-only, cost-adjusted episode reward, and compared directly against a fixed-heuristic baseline.
- **Two-adapter architecture on one frozen base model**: the two sub-agent specialists share one LoRA adapter; the orchestrator has its own, separate LoRA adapter on the same base model. This is what makes it possible to train the stop-policy while holding the sub-agents genuinely frozen, rather than co-training everything at once.

## 🧠 Methodology

**Dataset**: Uses the [WebShop](https://github.com/princeton-nlp/WebShop) simulated e-commerce environment — real-world scraped product listings and human-crowdsourced shopping instructions, run at a 1,000-product working subset for this lightweight phase (drawn from the full ~1.18M-product corpus).

**Architecture**:
- LLM backbone: Qwen2.5-0.5B-Instruct, with LoRA (rank 32) for all trainable roles
- Sub-agents: search specialist + compare/purchase specialist, sharing one LoRA adapter, executing HiPER's Plan-Execute loop (`<switch>/<subgoal>/<action>`)
- Orchestrator: a separate LoRA adapter on the same frozen base model, choosing `{delegate-search, delegate-compare, stop}` each round
- Trainer: [veRL](https://github.com/volcengine/verl) / [verl-agent](https://github.com/langfengQ/verl-agent) (Ray-based) for the sub-agent mechanism; a small custom PPO-clipped loop for the orchestrator's stop-policy

**Evaluation**: Baseline A (fixed heuristic orchestrator: delegate-search → delegate-compare → stop) vs. Baseline B (GRPO-trained stop policy), run on the exact same episode-execution path for a fair comparison.

**Metrics**:
- Task success rate
- Average delegation rounds per episode
- Cost-adjusted return `R = 10·success − c·rounds` (mean ± std across episodes/seeds), `c = 0.5` as an untuned starting point
- Training stability: loss/gradient-norm curves, wall-clock per iteration, peak GPU memory

**Additional Methodology**: Intentionally lightweight compute scale throughout (0.5B model, LoRA, small batch/group sizes) — the goal through Implementation 2 is a structurally correct, end-to-end pipeline that runs without crashing, not tuned or converged results. Reproducibility: setup steps, findings, and honest results for each phase live in `implementation-1/README.md` and `implementation-2/README.md`.

## 🌍 Impact

A stopping decision that's learned rather than hard-coded generalizes beyond WebShop to any multi-agent or agent-plus-tools system where a central controller must decide how much delegation is enough: coding agents deciding when to stop consulting sub-tools, research agents deciding when to stop searching, customer-support systems deciding when to stop escalating between specialist bots. Getting this decision right directly trades off task success against inference cost and latency — the two things that determine whether a multi-agent system is actually practical to deploy at scale.

## Future Work

- Implementation 3: tune the cost coefficient `c`, investigate the `critic/grad_norm` instability flagged in Implementation 1, and run enough training iterations (hours, not the couple of minutes used so far) to see a real reward trend.
- Multi-seed evaluation (3 seeds) of Baseline A vs. Baseline B, reported as mean ± variance rather than point estimates.
- Co-training sub-agents alongside the orchestrator's stop-policy (currently frozen by design — see `implementation-2/README.md`).
- Variant C (counterfactual/leave-one-out credit) and Variant D (LLM-generated potential-based reward shaping), gated on Baseline A/B being stable, per the original project plan.
- Revisit the lightweight (0.5B + LoRA) scale if Implementation 3's results suggest it's the actual bottleneck rather than the credit-assignment method itself.

## Additional Sources

- [HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for LLM Agents](https://arxiv.org/abs/2602.16165) — source of the Plan-Execute framework and HAE credit mechanism this project adapts.
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) — source of the group-relative advantage formula used for both sub-agent and orchestrator credit.
- Orchestration-traces survey, arXiv:2605.02801 — the paper whose gap (no RL methods for learned stop-policies) this project addresses.
- [WebShop](https://github.com/princeton-nlp/WebShop) — the environment used throughout.
- [veRL](https://github.com/volcengine/verl) / [verl-agent](https://github.com/langfengQ/verl-agent) — the underlying distributed RL trainer this project's environment and sub-agent training are built on.

## Repository layout

```
summer/
├── README.md                       # this file
├── implementation-1/                # environment setup + ultralight compute check
│   ├── README.md                   # what Implementation 1 did, findings, honest results
│   ├── UPSTREAM_README.md          # HiPER-agent's own original README (citation, license notes)
│   ├── example_scripts/            # includes our run_webshop_ultralight.sh lightweight config
│   └── verl/, agent_system/, hiper_hae/, ...   # vendored HiPER-agent codebase, unmodified
├── implementation-2/                # our own code only -- built on top of implementation-1/
│   ├── README.md                   # what Implementation 2 built, deviations, what's rough
│   ├── agents/                     # sub-agent credit mechanism (Part 1)
│   └── orchestration/              # orchestrator + specialists + both baselines (Parts 2-4)
└── data/                           # WebShop raw corpus (gitignored, not tracked)
```

`implementation-2/` imports across the folder boundary into `implementation-1/` (`verl`, `agent_system`, `hiper_hae`) — both folders are on the conda env's Python path, so this works regardless of which folder you run a script from. See `implementation-2/README.md` for exactly how that's wired.

## Quick start

```bash
conda activate verl   # the single environment that has both the training stack and WebShop's runtime deps

# 1. Environment setup (conda envs, WebShop data, dependency fixes) --
#    see implementation-1/README.md for the full list of what's needed and why.
cd implementation-1

# 2. Confirm the base pipeline: model load + one WebShop episode, no training (Implementation 1, Step 5)
python agent_system/environments/env_package/webshop/webshop/run_web_agent_text_env.py

# 3. Run the lightweight training config (Implementation 1, Step 6)
bash example_scripts/HiPER_trainer/run_webshop_ultralight.sh

# 4. Run the two orchestration baselines (Implementation 2, Parts 3-4)
cd ../implementation-2
python -m orchestration.baseline_a_heuristic   # fixed heuristic; writes baseline_a_results.json
python -m orchestration.baseline_b_rl_stop     # GRPO-trained stop policy
```

See `implementation-1/README.md` and `implementation-2/README.md` for full setup details, findings, and what each phase actually validated.
