"""
losses.py
=========
Trajectory Balance (TB) loss for the pHGFN GFlowNet.

Intuition
---------
Picture the generator as a river system. The source holds a total amount of
water Z (the partition function). Water flows through partial molecules to
terminal molecules. The amount reaching each terminal state should be
proportional to that molecule's reward R. TB enforces, for every complete
trajectory tau (source -> terminal):

    log Z + sum_t log P_F(a_t | s_t)  =  log R(x)  +  sum_t log P_B(s_{t-1} | s_t)

If this holds for all trajectories, the policy samples molecules in proportion to
reward — giving DIVERSE high-reward molecules rather than collapsing to a single
mode (the key advantage over standard RL for drug design).

Backward policy
---------------
We build molecules as token sequences, so each state has exactly ONE parent (drop
the last token). The construction DAG is a tree and the backward policy is
deterministic: P_B = 1, hence log P_B = 0. Callers may pass a non-zero
`backward_log_probs` for non-tree environments, but the default is 0.

Loss
----
    L = ( log Z + sum log P_F  -  log R  -  sum log P_B )^2
averaged over the batch. We work in log-reward space; rewards must be positive
(the trainer applies a softplus shaping), and `log R = log(max(R, eps))`.
"""

from __future__ import annotations

from typing import Optional

import torch


def trajectory_balance_loss(
    log_Z: torch.Tensor,
    forward_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    backward_log_probs: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Trajectory Balance loss.

    Args:
        log_Z:              learnable scalar log partition function, shape [1] or [].
        forward_log_probs:  sum_t log P_F(a_t|s_t) per trajectory, shape [B].
        rewards:            terminal reward R(x) per trajectory (POSITIVE), shape [B].
        backward_log_probs: sum_t log P_B per trajectory, shape [B]; default 0 (tree).
        epsilon:            floor for the reward inside the log.

    Returns:
        Scalar loss tensor.
    """
    if backward_log_probs is None:
        backward_log_probs = torch.zeros_like(forward_log_probs)

    log_reward = torch.log(rewards.clamp_min(epsilon))
    # log Z + sum log P_F - log R - sum log P_B   (should be 0 at the optimum)
    residual = log_Z.squeeze() + forward_log_probs - log_reward - backward_log_probs
    return (residual ** 2).mean()
