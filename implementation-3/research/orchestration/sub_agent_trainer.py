# IMP3 -- wires agents/sub_agent_credit.py's critic-free GRPO sub-agent credit
# mechanism (including its keep_penalty/keep_consistency_penalty degenerate-
# behavior guard) into an actual training loop. In implementation-2,
# compute_subagent_grpo_advantage() was correctly implemented but never
# called from anywhere -- dead code. This file is what makes it real: it
# trains the "shared" adapter (both specialists) against real ALFWorld
# rollouts, mirroring baseline_b_rl_stop.py's custom PPO-clipped loop shape
# (not verl's Ray trainer) but using the three-way action/subgoal/switch
# credit split instead of a single flat episode reward per rollout.
#
# Episode collection bypasses the orchestrator's own policy (delegates
# navigate -> interact -> stop on a fixed schedule, reusing Baseline A's
# `orchestrator_override` mechanism via episode_runner.run_episode) since
# this loop trains only the specialists' shared adapter, not the
# orchestrator's stop-policy.
#
# OPEN RISK, not resolved by testing here (ALFWorld's game data is not
# downloaded in this environment -- see implementation-3/README.md): GRPO
# needs G rollouts of the *same* underlying task per group so group-relative
# advantages are meaningful. The vendored AlfredTWEnv gym env doesn't expose
# a direct "replay this exact game" argument in this codebase; the design
# here assumes re-seeding a *fresh* AlfworldSingleEpisodeEnv instance to the
# same seed before its first reset() reproducibly selects the same starting
# game (standard seeded-RNG behavior) -- so each rollout in a group gets its
# own freshly-constructed, identically-seeded env, rather than reusing one
# instance's reset() repeatedly (which would walk through different games
# from the registered pool). Verify this against real ALFWorld data before
# trusting group-normalized advantages from this trainer.

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# NOTE: only agents.sub_agent_credit is imported at module level (pure
# tensor/numpy logic, no environment dependencies) so that
# _smoke_test_synthetic_batch() below can run without ALFWorld/gymnasium
# installed -- that's the whole point of a synthetic-batch check. Everything
# that touches a live env or the model pool is imported lazily inside
# SubAgentTrainer, only needed for the real-rollout path (train_sub_agents).
from agents.sub_agent_credit import SubAgentGRPOConfig, compute_subagent_grpo_advantage

# Bypasses the orchestrator's own decisions -- this trainer only updates the
# specialists' shared adapter, so episodes are driven on a fixed schedule
# (same pattern Baseline A uses for the same reason). 3 full navigate+interact
# cycles, not 1 -- see baseline_a_heuristic.py's FIXED_ACTION_SEQUENCE comment
# for why: a single pair is structurally incapable of completing almost any
# real ALFWorld task, which meant this trainer could never observe a success
# to learn from at all (confirmed today: 0 nonzero-reward turns across 400+
# collected turns). Exactly `DEFAULT_ROUND_CAP` elements, no trailing "stop" --
# see the same comment for why a shorter override falls through to the real
# (untrained) orchestrator instead of ending cleanly.
FIXED_DELEGATION_SEQUENCE = [
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
]
DEFAULT_ROUND_CAP = 6


class _SimpleBatch:
    """Minimal DataProto-like shim. compute_subagent_grpo_advantage() only
    needs dict-style .batch / .non_tensor_batch read+write access, not verl's
    full Ray-oriented DataProto machinery -- keeps this trainer decoupled
    from verl's distributed trainer, same as baseline_b_rl_stop.py."""

    def __init__(self):
        self.batch: Dict[str, torch.Tensor] = {}
        self.non_tensor_batch: Dict[str, np.ndarray] = {}


@dataclass
class RecordedTurn:
    traj_uid: str
    turn_idx: int
    reward: float
    done: bool
    prompt: str
    raw_output: str


def _flatten_episode_turns(traj_uid: str, log) -> List[RecordedTurn]:
    """Extracts every sub-agent turn across an episode's rounds, in the order
    they happened, as the flat per-turn records compute_subagent_grpo_advantage()
    expects. Each SubAgentTurn already carries its own per-turn reward and the
    exact (prompt, raw_output) it was generated from -- reused directly, not
    re-derived or reconstructed."""
    turns: List[RecordedTurn] = []
    turn_idx = 0
    for round_ in log.rounds:
        result = round_.sub_agent_result
        if result is None:  # "stop" round, no sub-agent ran
            continue
        for t in result["turns"]:
            turns.append(RecordedTurn(
                traj_uid=traj_uid, turn_idx=turn_idx, reward=t["reward"], done=t["done"],
                prompt=t["prompt"], raw_output=t["raw_output"],
            ))
            turn_idx += 1
    return turns


class SubAgentTrainer:
    def __init__(
        self,
        group_size: int = 2,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        round_cap: int = DEFAULT_ROUND_CAP,
        credit_cfg: SubAgentGRPOConfig = None,
    ):
        from orchestration.interact_specialist import InteractSpecialist
        from orchestration.model_pool import SHARED_ADAPTER, set_adapter_trainable, use_adapter
        from orchestration.navigate_specialist import NavigateSpecialist
        from orchestration.orchestrator import Orchestrator
        from orchestration.sub_agent import SUB_AGENT_SYSTEM_PROMPT

        self._shared_adapter = SHARED_ADAPTER
        self._use_adapter = use_adapter
        self._system_prompt = SUB_AGENT_SYSTEM_PROMPT
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.round_cap = round_cap
        self.credit_cfg = credit_cfg or SubAgentGRPOConfig()

        self.orchestrator = Orchestrator()  # unused for decisions, kept only so run_episode's signature matches
        self.navigate = NavigateSpecialist()
        self.interact = InteractSpecialist()
        self.model, self.tokenizer = self.navigate.model, self.navigate.tokenizer

        for p in self.model.parameters():
            p.requires_grad = False
        set_adapter_trainable(self.model, SHARED_ADAPTER, True)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found on the shared adapter -- check adapter naming.")
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    def collect_group(self, seed: int, config_path: str) -> List[RecordedTurn]:
        """Runs `group_size` episodes of the *same* seeded task (see the
        module-level open-risk note) and returns every sub-agent turn across
        all of them, flattened, ready for compute_subagent_grpo_advantage()."""
        from orchestration.alfworld_env import AlfworldSingleEpisodeEnv
        from orchestration.episode_runner import run_episode

        all_turns: List[RecordedTurn] = []
        for episode_idx in range(self.group_size):
            env = AlfworldSingleEpisodeEnv(config_path, seed=seed, is_train=True)
            log = run_episode(
                env, self.orchestrator, self.navigate, self.interact,
                round_cap=self.round_cap, orchestrator_override=FIXED_DELEGATION_SEQUENCE,
            )
            all_turns.extend(_flatten_episode_turns(traj_uid=f"seed{seed}-ep{episode_idx}", log=log))
        return all_turns

    def _build_batch(self, turns: List[RecordedTurn], group_id) -> _SimpleBatch:
        response_ids_list = [
            self.tokenizer(t.raw_output, return_tensors="pt", add_special_tokens=False).input_ids[0]
            for t in turns
        ]
        N = len(turns)
        L = max(1, max(ids.shape[0] for ids in response_ids_list))
        pad_id = self.tokenizer.eos_token_id

        responses = torch.full((N, L), pad_id, dtype=torch.long, device=self.model.device)
        response_mask = torch.zeros((N, L), dtype=torch.long, device=self.model.device)
        for i, ids in enumerate(response_ids_list):
            vl = ids.shape[0]
            responses[i, :vl] = ids.to(self.model.device)
            response_mask[i, :vl] = 1

        batch = _SimpleBatch()
        batch.batch["responses"] = responses
        batch.batch["response_mask"] = response_mask
        batch.non_tensor_batch["traj_uid"] = np.array([t.traj_uid for t in turns])
        batch.non_tensor_batch["turn_idx"] = np.array([t.turn_idx for t in turns])
        batch.non_tensor_batch["dones"] = np.array([t.done for t in turns])
        batch.non_tensor_batch["rewards"] = np.array([t.reward for t in turns], dtype=np.float32)
        batch.non_tensor_batch[self.credit_cfg.group_key] = np.array([group_id] * N)
        return batch

    def _turn_logprobs(self, turn: RecordedTurn, response_ids: torch.Tensor, valid_len: int, requires_grad: bool) -> torch.Tensor:
        """Per-token log-probs of `turn`'s response, conditioned on its exact
        prompt -- same tokenize-prompt-separately/concat-then-forward pattern
        baseline_b_rl_stop.py uses, but returning the per-token sequence
        (not summed) since compute_subagent_grpo_advantage()'s advantages are
        per-token, not per-sequence."""
        # must match the system prompt sub_agent.py's SubAgentSpecialist.run() actually
        # generated `turn.raw_output` under -- a mismatch here would recompute log-probs
        # against a different conditioning context than the one that was sampled from.
        messages = [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": turn.prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        input_ids = torch.cat([prompt_ids, response_ids[:valid_len].unsqueeze(0)], dim=1)

        self._use_adapter(self.model, self._shared_adapter)
        with torch.enable_grad() if requires_grad else torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return token_logprobs[0, -valid_len:]

    def update(self, turns: List[RecordedTurn], group_id) -> dict:
        if not turns:
            return {"skipped": True, "reason": "empty group"}

        batch = self._build_batch(turns, group_id)
        compute_subagent_grpo_advantage(batch, self.credit_cfg, self.tokenizer)
        advantages = batch.batch["advantages"]      # (N, L)
        response_mask = batch.batch["response_mask"].to(torch.bool)  # (N, L)
        responses = batch.batch["responses"]         # (N, L)
        valid_lens = response_mask.long().sum(dim=1)

        old_logprobs = [
            self._turn_logprobs(t, responses[i], int(valid_lens[i]), requires_grad=False).detach()
            for i, t in enumerate(turns)
        ]

        self.optimizer.zero_grad()
        pg_losses = []  # per-turn losses kept unsummed for reporting, same rationale as baseline_b_rl_stop.py
        for i, (turn, old_lp) in enumerate(zip(turns, old_logprobs)):
            vl = int(valid_lens[i])
            if vl == 0:
                continue
            new_lp = self._turn_logprobs(turn, responses[i], vl, requires_grad=True)
            adv = advantages[i, :vl]
            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            loss = -torch.min(unclipped, clipped).sum() / len(turns)
            loss.backward()
            pg_losses.append(loss.item())
        self.optimizer.step()

        return {
            "pg_loss_mean": sum(pg_losses) / len(pg_losses) if pg_losses else 0.0,
            "num_turns": len(turns),
            "reward_mean": float(np.mean(batch.non_tensor_batch["rewards"])),
            # NOTE: this is a step-reward diagnostic, NOT the actual KEEP/SWITCH
            # ratio -- ALFWorld only pays a nonzero reward on the terminal winning
            # step, so most steps read 0.0 regardless of switch behavior. The real
            # keep-collapse signal is `switch` inside compute_subagent_grpo_advantage
            # (not surfaced here to avoid a second decode pass); watch pg_loss_mean
            # and the applied keep_penalty's effect on reward trend instead.
            "zero_reward_step_ratio": float(np.mean([t.reward == 0.0 for t in turns])) if turns else 0.0,
        }


def train_sub_agents(num_groups: int = 2, group_size: int = 2, timer=None) -> List[dict]:
    """`timer`, if given (an orchestration.seal.timing.RoundTimer), wraps each
    round's phases -- see baseline_b_rl_stop.train_baseline_b's identical use
    for why (apples-to-apples overhead comparison in compare_variants.py)."""
    from orchestration.episode_runner import _ALFWORLD_CONFIG

    trainer = SubAgentTrainer(group_size=group_size)
    history = []
    for group_id in range(num_groups):
        if timer is not None:
            with timer.round(group_id):
                with timer.phase("episode_collection"):
                    turns = trainer.collect_group(seed=group_id, config_path=_ALFWORLD_CONFIG)
                with timer.phase("update"):
                    history.append(trainer.update(turns, group_id))
        else:
            turns = trainer.collect_group(seed=group_id, config_path=_ALFWORLD_CONFIG)
            history.append(trainer.update(turns, group_id))
    return history


def _smoke_test_synthetic_batch() -> dict:
    """Structural check with no live env or model calls: builds a tiny
    synthetic batch directly and confirms compute_subagent_grpo_advantage()
    runs and produces correctly-shaped, finite outputs. Extends the same kind
    of synthetic-batch check implementation-2's README references having run
    once for this function (never committed) -- this version is committed so
    it can be re-run before trusting live rollouts."""
    cfg = SubAgentGRPOConfig()
    # One "token" == one character of this fixed text, so a stub decode can map
    # position -> character deterministically without needing a real vocabulary
    # (make_hae_masks_and_switch decodes token-by-token to build char offsets).
    text = "<switch>KEEP</switch><subgoal>x</subgoal><action>go to shelf 1</action>"
    N, L = 4, len(text)
    batch = _SimpleBatch()
    batch.batch["responses"] = torch.arange(L).unsqueeze(0).repeat(N, 1)  # position IS the token id here
    batch.batch["response_mask"] = torch.ones((N, L), dtype=torch.long)
    batch.non_tensor_batch["traj_uid"] = np.array(["a", "a", "b", "b"])
    batch.non_tensor_batch["turn_idx"] = np.array([0, 1, 0, 1])
    batch.non_tensor_batch["dones"] = np.array([False, True, False, True])
    batch.non_tensor_batch["rewards"] = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    batch.non_tensor_batch[cfg.group_key] = np.array([0, 0, 0, 0])

    class _StubTokenizer:
        def batch_decode(self, ids, **kwargs):
            # `ids` is a list of single-token-id lists; each id is a character position.
            return [text[int(tok[0])] for tok in ids]

    compute_subagent_grpo_advantage(batch, cfg, _StubTokenizer())
    adv = batch.batch["advantages"]
    assert adv.shape == (N, L), f"unexpected advantage shape: {adv.shape}"
    assert torch.isfinite(adv).all(), "non-finite advantage values"
    return {"ok": True, "advantages_shape": list(adv.shape)}


if __name__ == "__main__":
    print(json.dumps(_smoke_test_synthetic_batch(), indent=2))
