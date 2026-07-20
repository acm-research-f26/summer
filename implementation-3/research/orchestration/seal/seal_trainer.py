# IMP3 SEAL variant -- the outer RL loop over the note-writing policy (SEAL
# arXiv:2506.10943's own ReST-EM-style "favor self-edits that produce real
# improvement" approach, adapted to this project's GRPO-style critic-free
# credit style used elsewhere -- e.g. baseline_b_rl_stop.py, sub_agent_credit.py).
#
# Per round:
#   1. Collect a small batch of recent episodes (navigate/interact specialists
#      acting on real ALFWorld tasks, orchestrator bypassed via the fixed
#      delegation sequence -- same pattern sub_agent_trainer.py uses).
#   2. Sample K candidate notes from NoteWriter conditioned on that batch.
#   3. Evaluate a shared "before" success rate once on a small fixed eval batch.
#   4. For each candidate: snapshot the shared adapter, apply its self-edit
#      (run_self_edit_step, which -- see self_edit.py -- correctly targets
#      SHARED_ADAPTER, not self_editor), re-eval on the SAME eval batch for
#      "after", reward_k = after_k - before, then restore the shared adapter
#      from snapshot before evaluating the next candidate.
#   5. Group-normalize the K rewards into advantages; PPO-clipped update on
#      NoteWriter's note log-probs, gradients restricted to SELF_EDITOR_ADAPTER
#      only -- the shared adapter must already be back at its pre-edit state
#      at this point, so this update cannot also perturb sub-agent behavior.
#   6. Apply the round's best-performing candidate "for real" (no rollback)
#      so the shared adapter actually accumulates the winning edit -- this is
#      what makes rounds compound instead of resetting every time.
#
# Wraps each round with timing.RoundTimer (episode_collection / note_generation
# / finetune_step / eval phases) and, every N rounds, forgetting_tracker's
# held-out re-evaluation -- both SEAL paper warnings this project was asked to
# track honestly, not asserted.

import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from orchestration.seal.forgetting_tracker import ForgettingCheckpoint, ForgettingTracker
from orchestration.seal.note_writer import NoteCandidate, NoteWriter
from orchestration.seal.self_edit import note_to_training_examples, restore_adapter, run_self_edit_step, snapshot_adapter
from orchestration.seal.timing import RoundTimer

# 3 full navigate+interact cycles, not 1 -- see baseline_a_heuristic.py's
# FIXED_ACTION_SEQUENCE comment: a single pair structurally can't complete
# almost any real ALFWorld task, which meant this collected episodes (and
# self-edit training examples derived from them) that could basically never
# reflect a real success. Exactly `round_cap`-many elements (no trailing
# "stop") so the override never runs out and falls through to the real
# (untrained) orchestrator's own decision.
FIXED_DELEGATION_SEQUENCE = [
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
]
# Distinct from the low-seed ranges training/eval batches use elsewhere in
# this file, so held-out forgetting-eval tasks are never accidentally trained on.
HELD_OUT_SEED_BASE = 9000


def _group_normalize(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    mean = values.mean()
    std = values.std() + epsilon
    return (values - mean) / std


@dataclass
class SealRoundResult:
    round_idx: int
    candidate_rewards: List[float]
    applied_note: str
    before_success_rate: float
    after_success_rate_of_applied: float
    self_edit_stats: dict
    forgetting_checkpoint: Optional[dict] = None


class SealTrainer:
    def __init__(
        self,
        config_path: str,
        train_batch_size: int = 3,
        eval_batch_size: int = 3,
        num_candidates: int = 2,
        held_out_size: int = 3,
        forgetting_check_every: int = 2,
        note_lr: float = 1e-5,
        clip_eps: float = 0.2,
        self_edit_lr: float = 5e-5,
        self_edit_steps: int = 3,
    ):
        from orchestration.interact_specialist import InteractSpecialist
        from orchestration.model_pool import SELF_EDITOR_ADAPTER, SHARED_ADAPTER, set_adapter_trainable, use_adapter
        from orchestration.navigate_specialist import NavigateSpecialist
        from orchestration.orchestrator import Orchestrator

        self._self_editor_adapter = SELF_EDITOR_ADAPTER
        self._shared_adapter = SHARED_ADAPTER
        self._use_adapter = use_adapter
        self.config_path = config_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_candidates = num_candidates
        self.clip_eps = clip_eps
        self.self_edit_lr = self_edit_lr
        self.self_edit_steps = self_edit_steps

        self.orchestrator = Orchestrator()
        self.navigate = NavigateSpecialist()
        self.interact = InteractSpecialist()
        self.note_writer = NoteWriter()
        self.model, self.tokenizer = self.navigate.model, self.navigate.tokenizer

        held_out_seeds = list(range(HELD_OUT_SEED_BASE, HELD_OUT_SEED_BASE + held_out_size))
        self.forgetting_tracker = ForgettingTracker(held_out_seeds, config_path, check_every_n_rounds=forgetting_check_every)
        self.timer = RoundTimer()

        for p in self.model.parameters():
            p.requires_grad = False
        set_adapter_trainable(self.model, SELF_EDITOR_ADAPTER, True)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found on the self_editor adapter -- check adapter naming.")
        self.note_optimizer = torch.optim.AdamW(trainable, lr=note_lr)

    def _collect_episodes(self, seed_base: int, n: int) -> list:
        from orchestration.alfworld_env import AlfworldSingleEpisodeEnv
        from orchestration.episode_runner import run_episode

        logs = []
        for i in range(n):
            env = AlfworldSingleEpisodeEnv(self.config_path, seed=seed_base + i, is_train=True)
            log = run_episode(
                env, self.orchestrator, self.navigate, self.interact,
                orchestrator_override=FIXED_DELEGATION_SEQUENCE,
            )
            logs.append(log)
        return logs

    def _eval_success_rate(self, seed_base: int, n: int) -> float:
        logs = self._collect_episodes(seed_base, n)
        return sum(l.success for l in logs) / len(logs) if logs else 0.0

    def _note_logprob(self, candidate: NoteCandidate, requires_grad: bool) -> torch.Tensor:
        messages = [{"role": "system", "content": candidate.system_prompt}, {"role": "user", "content": candidate.user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        response_ids = self.tokenizer(candidate.raw_output, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        self._use_adapter(self.model, self._self_editor_adapter)
        with torch.enable_grad() if requires_grad else torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            response_len = response_ids.shape[1]
            return token_logprobs[:, -response_len:].sum()

    def run_round(self, round_idx: int, train_seed_base: int, eval_seed_base: int) -> SealRoundResult:
        with self.timer.round(round_idx):
            with self.timer.phase("episode_collection"):
                recent_logs = self._collect_episodes(train_seed_base, self.train_batch_size)

            with self.timer.phase("note_generation"):
                candidates = [self.note_writer.write_note(recent_logs) for _ in range(self.num_candidates)]

            with self.timer.phase("eval"):
                before_rate = self._eval_success_rate(eval_seed_base, self.eval_batch_size)

            rewards = []
            self_edit_stats_per_candidate = []
            shared_snapshot = snapshot_adapter(self.model, self._shared_adapter)
            for cand in candidates:
                with self.timer.phase("finetune_step"):
                    examples = note_to_training_examples(cand.note, recent_logs)
                    stats = run_self_edit_step(self.model, self.tokenizer, examples, lr=self.self_edit_lr, num_steps=self.self_edit_steps)
                    self_edit_stats_per_candidate.append(stats)
                with self.timer.phase("eval"):
                    after_rate = self._eval_success_rate(eval_seed_base, self.eval_batch_size)
                rewards.append(after_rate - before_rate)
                restore_adapter(self.model, self._shared_adapter, shared_snapshot)

            advantages = _group_normalize(np.array(rewards))
            old_logprobs = [self._note_logprob(c, requires_grad=False).detach() for c in candidates]

            self.note_optimizer.zero_grad()
            for cand, old_lp, adv in zip(candidates, old_logprobs, advantages):
                new_lp = self._note_logprob(cand, requires_grad=True)
                ratio = torch.exp(new_lp - old_lp)
                adv_t = torch.tensor(float(adv), device=self.model.device)
                unclipped = ratio * adv_t
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                loss = -torch.min(unclipped, clipped) / len(candidates)
                loss.backward()
            self.note_optimizer.step()

            # Apply the round's best-performing candidate "for real" (no rollback) so
            # the shared adapter accumulates the winning edit into the next round.
            best_idx = int(np.argmax(rewards))
            with self.timer.phase("finetune_step"):
                best_examples = note_to_training_examples(candidates[best_idx].note, recent_logs)
                run_self_edit_step(self.model, self.tokenizer, best_examples, lr=self.self_edit_lr, num_steps=self.self_edit_steps)
            with self.timer.phase("eval"):
                after_rate_applied = self._eval_success_rate(eval_seed_base, self.eval_batch_size)

            forgetting_cp = None
            if self.forgetting_tracker.should_check(round_idx):
                with self.timer.phase("eval"):
                    cp = self.forgetting_tracker.evaluate(round_idx, self.orchestrator, self.navigate, self.interact)
                    forgetting_cp = {"round_idx": cp.round_idx, "held_out_success_rate": cp.held_out_success_rate, "num_tasks": cp.num_tasks}

        return SealRoundResult(
            round_idx=round_idx,
            candidate_rewards=rewards,
            applied_note=candidates[best_idx].note,
            before_success_rate=before_rate,
            after_success_rate_of_applied=after_rate_applied,
            self_edit_stats=self_edit_stats_per_candidate[best_idx],
            forgetting_checkpoint=forgetting_cp,
        )


def train_seal(num_rounds: int = 1, config_path: str = None, **trainer_kwargs) -> List[dict]:
    from dataclasses import asdict

    from orchestration.episode_runner import _ALFWORLD_CONFIG

    config_path = config_path or _ALFWORLD_CONFIG
    trainer = SealTrainer(config_path=config_path, **trainer_kwargs)
    history = []
    for round_idx in range(num_rounds):
        # disjoint train/eval seed ranges per round so a round's eval batch
        # isn't also something the same round trained on
        result = trainer.run_round(round_idx, train_seed_base=round_idx * 100, eval_seed_base=round_idx * 100 + 50)
        history.append(asdict(result))
    trainer.timer.save("seal_round_timings.jsonl")
    trainer.forgetting_tracker.save("forgetting_curve.json")
    return history


if __name__ == "__main__":
    history = train_seal(num_rounds=1, train_batch_size=2, eval_batch_size=2, num_candidates=2, held_out_size=2)
    print(json.dumps(history, indent=2))
