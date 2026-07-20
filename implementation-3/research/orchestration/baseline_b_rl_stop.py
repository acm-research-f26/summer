# IMP3 -- Baseline B: RL-trained stop policy, outcome-only reward.
#
# R_episode = 10 * success - c * num_delegation_rounds, c=0.5 starting point
# (still untuned here -- out of scope for this ALFWorld-port pass, same as
# implementation-2 left it). Trains the orchestrator's
# {delegate-navigate, delegate-interact, stop} decision via GRPO: sample a
# group of G orchestrator rollouts per task, compute cost-adjusted return per
# rollout, normalize within-group, update with a standard PPO-style clipped
# objective. Single-level decision, no HAE-style split needed here (unlike
# Part 1's sub-agent credit, which does need that split).
#
# Sub-agents stay frozen -- only the orchestrator's own LoRA adapter is
# trainable (see model_pool.py's two-adapter design). No co-training of
# sub-agents in this pass (that's a stretch goal per the spec).
#
# Reuses episode_runner.run_episode unchanged (not reimplemented here) by
# having Orchestrator record each decision's exact (prompt, response) via
# set_recorder(), which is what a training loop needs but a plain baseline
# run doesn't.
#
# Ported from implementation-2's WebShop version: no logic changes, only
# renamed specialist imports/identifiers following the ALFWorld port.

import json
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

from orchestration.episode_runner import EpisodeLog, make_env, run_episode
from orchestration.interact_specialist import InteractSpecialist
from orchestration.model_pool import ORCHESTRATOR_ADAPTER, set_adapter_trainable, use_adapter
from orchestration.navigate_specialist import NavigateSpecialist
from orchestration.orchestrator import Orchestrator

DEFAULT_COST_COEF = 0.5  # c in R_episode = 10*success - c*rounds; tuning is IMP3's job, not this one


@dataclass
class DecisionRecord:
    system_prompt: str
    user_prompt: str
    response_text: str


@dataclass
class RolloutRecord:
    decisions: List[DecisionRecord] = field(default_factory=list)
    episode: EpisodeLog = None
    reward: float = 0.0


def cost_adjusted_reward(log: EpisodeLog, cost_coef: float) -> float:
    return 10.0 * float(log.success) - cost_coef * log.num_delegation_rounds


class BaselineBTrainer:
    def __init__(
        self,
        group_size: int = 4,
        cost_coef: float = DEFAULT_COST_COEF,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        round_cap: int = 6,
    ):
        self.group_size = group_size
        self.cost_coef = cost_coef
        self.clip_eps = clip_eps
        self.round_cap = round_cap

        self.orchestrator = Orchestrator()
        self.navigate = NavigateSpecialist()
        self.interact = InteractSpecialist()
        self.model, self.tokenizer = self.orchestrator.model, self.orchestrator.tokenizer

        for p in self.model.parameters():
            p.requires_grad = False
        set_adapter_trainable(self.model, ORCHESTRATOR_ADAPTER, True)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found on the orchestrator adapter -- check adapter naming.")
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    def collect_group(self, env) -> List[RolloutRecord]:
        rollouts = []
        for _ in range(self.group_size):
            decisions: List[DecisionRecord] = []
            self.orchestrator.set_recorder(
                lambda sp, up, raw: decisions.append(DecisionRecord(sp, up, raw))
            )
            log = run_episode(env, self.orchestrator, self.navigate, self.interact, round_cap=self.round_cap)
            self.orchestrator.set_recorder(None)

            reward = cost_adjusted_reward(log, self.cost_coef)
            rollouts.append(RolloutRecord(decisions=decisions, episode=log, reward=reward))
        return rollouts

    def _sequence_logprob(self, record: DecisionRecord, requires_grad: bool) -> torch.Tensor:
        messages = [{"role": "system", "content": record.system_prompt}, {"role": "user", "content": record.user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        response_ids = self.tokenizer(record.response_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        use_adapter(self.model, ORCHESTRATOR_ADAPTER)
        with torch.enable_grad() if requires_grad else torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            response_len = response_ids.shape[1]
            return token_logprobs[:, -response_len:].sum()

    def _rollout_logprob(self, rollout: RolloutRecord, requires_grad: bool) -> torch.Tensor:
        if not rollout.decisions:
            return torch.tensor(0.0, device=self.model.device)
        return sum(self._sequence_logprob(d, requires_grad) for d in rollout.decisions)

    def update(self, rollouts: List[RolloutRecord]) -> dict:
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        old_logprobs = [self._rollout_logprob(r, requires_grad=False).detach() for r in rollouts]

        self.optimizer.zero_grad()
        pg_losses = []  # per-rollout policy loss, kept unsummed for reporting -- summing signed
        # per-rollout losses (as the actual backward-accumulated loss does) can trivially
        # cancel to ~0 across a batch even when every rollout produced a real gradient, which
        # would be a misleading number to watch for a loss curve (IMP3 will be watching this).
        for rollout, old_lp, adv in zip(rollouts, old_logprobs, advantages):
            new_lp = self._rollout_logprob(rollout, requires_grad=True)
            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            loss = -torch.min(unclipped, clipped) / len(rollouts)
            loss.backward()
            pg_losses.append(loss.item())
        self.optimizer.step()

        return {
            "pg_loss_mean": sum(pg_losses) / len(pg_losses),
            "pg_loss_abs_mean": sum(abs(l) for l in pg_losses) / len(pg_losses),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "success_rate": sum(r.episode.success for r in rollouts) / len(rollouts),
            "avg_delegation_rounds": sum(r.episode.num_delegation_rounds for r in rollouts) / len(rollouts),
        }


def train_baseline_b(num_groups: int = 4, group_size: int = 4, cost_coef: float = DEFAULT_COST_COEF, timer=None) -> List[dict]:
    """`timer`, if given (an orchestration.seal.timing.RoundTimer), wraps each
    round's phases so compare_variants.py can compute SEAL's per-round
    overhead ratio against this exact same measurement, not a hand estimate."""
    env = make_env()
    trainer = BaselineBTrainer(group_size=group_size, cost_coef=cost_coef)
    history = []
    for round_idx in range(num_groups):
        if timer is not None:
            with timer.round(round_idx):
                with timer.phase("episode_collection"):
                    rollouts = trainer.collect_group(env)
                with timer.phase("update"):
                    history.append(trainer.update(rollouts))
        else:
            rollouts = trainer.collect_group(env)
            history.append(trainer.update(rollouts))
    return history


if __name__ == "__main__":
    history = train_baseline_b(num_groups=2, group_size=2)
    print(json.dumps(history, indent=2))
