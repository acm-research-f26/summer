# IMP2 Part 1 -- sub-agent execution credit (critic-free, GRPO-style HAE variant).
#
# HiPER's own mechanism (hiper_hae/core_hae.py, vendored reference -- read, not
# edited) learns a three-head PPO-style critic (V_low, V_high, V_term) and
# computes bootstrapped GAE at both the per-turn (low) and per-segment (high)
# level, plus a termination advantage for the switch decision.
#
# Deliberate simplification for this project: we drop all three critics and
# replace them with GRPO-style group-relative advantages, computed from a
# group of G rollouts sampled for the same task/prompt. No value network is
# trained. This trades away HiPER's bootstrapped, per-step temporal credit for
# something that trains under LoRA at the lightweight (0.5B) scale confirmed
# in Implementation 1 LITE -- a critic doubles the trainable-model memory
# footprint, which is the one thing that ultralight config can't absorb.
#
# Concretely, three separate group-normalized advantages are computed and
# assigned to their own token spans (three separate d/d(log pi) * A terms,
# not one flat advantage per output):
#   - subgoal tokens <- segment-aggregate return (reward summed across the
#     turns belonging to that subgoal segment), at the segment's boundary turn.
#   - action tokens  <- return-to-go *within* the segment (how the execution
#     played out from this turn to the end of the current subgoal), i.e. a
#     finer-grained cut of the same segment-aggregate signal.
#   - switch tokens  <- full-trajectory return (the whole episode's outcome),
#     assigned at every turn -- whether to KEEP or SWITCH is judged against
#     the final result, the same way Baseline B's stop-policy reward (IMP2
#     Part 4) is outcome-only.
#
# keep_penalty / keep_consistency_penalty are carried over unchanged from
# core_hae.py's reward shaping -- HiPER found these necessary in practice to
# stop the sub-agent collapsing into always-KEEP degenerate behavior, and nothing
# about removing the critic changes that failure mode.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hiper_hae.hier_mask import HAEMaskConfig, make_hae_masks_and_switch


def _group_indices(ids: np.ndarray) -> Dict[object, List[int]]:
    groups: Dict[object, List[int]] = defaultdict(list)
    for i, gid in enumerate(ids):
        groups[gid].append(i)
    return groups


@dataclass
class SubAgentGRPOConfig:
    group_key: str = "uid"  # non_tensor_batch key shared by the G rollouts sampled per task
    epsilon: float = 1e-6
    keep_penalty: float = -3.5  # same default as HiPER's reference config (run_webshop.sh)
    keep_penalty_mode: str = "normalized"  # "fixed" or "normalized" -- see core_hae.py
    keep_consistency_penalty: float = -0.3
    keep_consistency_requires_both: bool = True
    norm_adv: bool = True


def _apply_keep_penalties(
    r: torch.Tensor,
    switch: np.ndarray,
    subgoal_mask: torch.Tensor,
    responses: torch.Tensor,
    groups: Dict[object, List[int]],
    turn_idx: np.ndarray,
    cfg: SubAgentGRPOConfig,
) -> torch.Tensor:
    """Reward shaping ported from hiper_hae.core_hae.compute_hae_advantage, unchanged in
    behavior -- reimplemented here rather than imported since it operates on a plain
    reward tensor, not the PPO batch that function expects."""
    r = r.clone()
    if cfg.keep_penalty != 0.0:
        if cfg.keep_penalty_mode == "fixed":
            keep_mask = torch.as_tensor(~switch, device=r.device, dtype=torch.bool)
            r = r + keep_mask.to(torch.float32) * cfg.keep_penalty
        elif cfg.keep_penalty_mode == "normalized":
            for tid, idxs in groups.items():
                idxs = sorted(idxs, key=lambda i: turn_idx[i])
                keep_count = sum(1 for i in idxs if not bool(switch[i]))
                if keep_count > 0:
                    keep_ratio = keep_count / len(idxs)
                    per_keep_penalty = cfg.keep_penalty * keep_ratio / keep_count
                    for i in idxs:
                        if not bool(switch[i]):
                            r[i] = r[i] + per_keep_penalty
        else:
            raise ValueError(f"Unknown keep_penalty_mode: {cfg.keep_penalty_mode}")

    if cfg.keep_consistency_penalty != 0.0:
        for tid, idxs in groups.items():
            idxs = sorted(idxs, key=lambda i: turn_idx[i])
            for pos in range(1, len(idxs)):
                i = idxs[pos]
                if bool(switch[i]):
                    continue
                prev_i = idxs[pos - 1]
                curr_toks = responses[i][subgoal_mask[i]]
                prev_toks = responses[prev_i][subgoal_mask[prev_i]]
                if cfg.keep_consistency_requires_both and (curr_toks.numel() == 0 or prev_toks.numel() == 0):
                    continue
                if (curr_toks.numel() != prev_toks.numel()) or curr_toks.numel() == 0 or not torch.equal(curr_toks, prev_toks):
                    r[i] = r[i] + cfg.keep_consistency_penalty
    return r


def _build_segments(idxs_sorted: List[int], switch: np.ndarray) -> List[Tuple[int, int]]:
    """Segment boundaries within one trajectory's turns (already sorted by turn_idx).
    Returns list of (start_pos, end_pos) index-into-idxs_sorted pairs."""
    T = len(idxs_sorted)
    boundary_pos = sorted({0} | {pos for pos in range(1, T) if bool(switch[idxs_sorted[pos]])})
    segs = []
    for k, s_pos in enumerate(boundary_pos):
        e_pos = (boundary_pos[k + 1] - 1) if (k + 1 < len(boundary_pos)) else (T - 1)
        segs.append((s_pos, e_pos))
    return segs


def _group_normalize(values: torch.Tensor, active: torch.Tensor, group_id: np.ndarray, epsilon: float) -> torch.Tensor:
    """Normalize `values` to zero-mean/unit-std within each `group_id`, only over
    entries where `active` is True (mirrors core_hae.py's norm_adv, but per-group
    instead of over the whole batch -- that's the GRPO part)."""
    out = values.clone()
    groups = _group_indices(group_id)
    for _, idxs in groups.items():
        idxs = [i for i in idxs if active[i]]
        if len(idxs) < 2:
            continue
        idx_t = torch.as_tensor(idxs, device=values.device, dtype=torch.long)
        v = values[idx_t]
        mean = v.mean()
        std = v.std(unbiased=False) + epsilon
        out[idx_t] = (v - mean) / std
    return out


@torch.no_grad()
def compute_subagent_grpo_advantage(batch, cfg: SubAgentGRPOConfig, tokenizer):
    """
    Expects the same batch layout as hiper_hae.core_hae.compute_hae_advantage:
      batch.batch:            responses (N,L), response_mask (N,L)
      batch.non_tensor_batch: traj_uid (N,), turn_idx (N,), dones (N,), rewards (N,),
                               and cfg.group_key (N,) -- the id shared by the G
                               rollouts sampled for the same task/prompt.

    Writes into batch.batch (same names as core_hae.py, for drop-in trainer
    compatibility): advantages_low (action), advantages_high (subgoal),
    advantages_term (switch), and their sum as "advantages".
    """
    device = batch.batch["response_mask"].device
    response_mask = batch.batch["response_mask"].to(torch.bool)
    responses = batch.batch["responses"]
    N, L = responses.shape

    action_mask, subgoal_mask, switch_mask, is_new_subgoal = make_hae_masks_and_switch(
        batch, tokenizer, include_tags_mask=False, cfg=HAEMaskConfig()
    )
    switch = is_new_subgoal  # True == SWITCH decision, False == KEEP

    traj_uid = batch.non_tensor_batch["traj_uid"]
    turn_idx = np.asarray(batch.non_tensor_batch["turn_idx"]).astype(np.int64)
    group_id = np.asarray(batch.non_tensor_batch[cfg.group_key])
    r = torch.as_tensor(np.asarray(batch.non_tensor_batch["rewards"], dtype=np.float32), device=device)

    traj_groups = _group_indices(traj_uid)
    r = _apply_keep_penalties(r, switch, subgoal_mask, responses, traj_groups, turn_idx, cfg)

    ret_action = torch.zeros((N,), device=device, dtype=torch.float32)
    ret_subgoal = torch.zeros((N,), device=device, dtype=torch.float32)
    ret_switch = torch.zeros((N,), device=device, dtype=torch.float32)
    is_boundary = torch.zeros((N,), device=device, dtype=torch.bool)

    for tid, idxs in traj_groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        if len(idxs) == 0:
            continue

        # full-trajectory return: same value at every turn, feeds the switch advantage.
        traj_return = sum(r[i].item() for i in idxs)
        for i in idxs:
            ret_switch[i] = traj_return

        for s_pos, e_pos in _build_segments(idxs, switch):
            seg_idxs = idxs[s_pos : e_pos + 1]
            seg_return = sum(r[i].item() for i in seg_idxs)
            boundary_i = idxs[s_pos]
            ret_subgoal[boundary_i] = seg_return
            is_boundary[boundary_i] = True

            # return-to-go within the segment, for the action advantage.
            running = 0.0
            for i in reversed(seg_idxs):
                running += r[i].item()
                ret_action[i] = running

    lo_active = action_mask.any(dim=1)
    hi_active = subgoal_mask.any(dim=1) & is_boundary
    term_active = switch_mask.any(dim=1)

    if cfg.norm_adv:
        adv_action = _group_normalize(ret_action, lo_active, group_id, cfg.epsilon)
        adv_subgoal = _group_normalize(ret_subgoal, hi_active, group_id, cfg.epsilon)
        adv_switch = _group_normalize(ret_switch, term_active, group_id, cfg.epsilon)
    else:
        adv_action, adv_subgoal, adv_switch = ret_action, ret_subgoal, ret_switch

    hi_mask = subgoal_mask & is_boundary.unsqueeze(-1)
    advantages_low = adv_action.unsqueeze(-1) * action_mask.to(torch.float32)
    advantages_high = adv_subgoal.unsqueeze(-1) * hi_mask.to(torch.float32)
    advantages_term = adv_switch.unsqueeze(-1) * switch_mask.to(torch.float32)

    batch.batch["advantages_low"] = advantages_low
    batch.batch["advantages_high"] = advantages_high
    batch.batch["advantages_term"] = advantages_term
    batch.batch["advantages"] = advantages_low + advantages_high + advantages_term
    batch.batch["hae_lo_mask"] = action_mask
    batch.batch["hae_hi_mask"] = hi_mask
    batch.batch["hae_term_mask"] = switch_mask
    return batch
