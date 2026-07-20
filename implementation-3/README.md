# Reward Design for Delegation in Agentic Systems — ALFWorld

## Summary

Multi-agent LLM systems often need an orchestrator to decide when to
delegate a task to a specialist versus act itself. Learning that decision
usually means training a full critic, which roughly doubles training cost.
This project trains delegation with a lightweight, critic-free RL signal
instead, and compares it against a reproduction of an existing published
single-agent baseline, both on ALFWorld (an embodied household-task
benchmark).

## Research question

Can a critic-free, group-relative RL signal teach an orchestrator *when* to
delegate, *to which* specialist ("navigate" vs. "interact"), and *when to
stop*, and how does the resulting system compare to a single-agent policy
trained with a full critic-based advantage estimator on the same benchmark?

## Methodology

One shared Qwen2.5 model (LoRA-adapted) plays three roles: an orchestrator
and two specialists. Two systems are trained and compared:

- **Paper Baseline** — single-agent, critic-based Hierarchical Advantage
  Estimation, reproducing an existing published pipeline. No delegation.
- **Orchestrator System** — our own approach, in four variants: a fixed
  heuristic floor, an RL-trained stop/delegate policy (GRPO), specialist
  training via a three-way critic-free credit split, and a self-editing
  (SEAL) extension.

Both use group-relative sampling and PPO-clipped updates. Evaluation is
ALFWorld task success rate, plus a cost-adjusted return that penalizes
excess delegation.

## Results

Both pipelines run end-to-end and train correctly. Paper Baseline's success
rate moved from 0% to a real, nonzero rate within its first training steps,
with a high (85–97%) valid-action rate from the start. Orchestrator System
required fixing two implementation issues before it could learn at all
(malformed output silencing the orchestrator's decisions, and an exploration
gap that never let it consider stopping); after fixing both, it now shows
real gradient signal and a shifting delegation policy across training
rounds. Both systems are still early in training — success rates are
directionally positive but not yet converged.

## Future work

- Run both systems to full scale for converged, directly comparable success
  rates.
- Specialist credit training (variant 3) is mechanically verified but not
  yet run long enough to produce a meaningful result.
- SEAL (variant 4) is fully implemented but not yet run at a scale sufficient
  to determine whether its self-editing signal is worth its compute overhead.

See `PROJECT_GUIDE.md` for full technical detail.
