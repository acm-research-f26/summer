"""
PPO (Proximal Policy Optimization) training for the Limbus game, with an
entropy-gated MCPS fallback at inference time.

At inference: the policy runs normally for low-entropy (confident) decisions.
When entropy exceeds a threshold, it falls back to MCPS to search for a
better move - this means you only pay the MCPS compute cost when the model
is genuinely uncertain, which in practice is mostly early in training or in
unusual game states the model hasn't seen much.

Usage:
    python ppo_train.py                  # train from scratch
    python ppo_train.py --resume ckpt.pt # resume from checkpoint
    python ppo_train.py --eval ckpt.pt   # evaluate a saved model

All hyperparameters are at the top of the file. The most important ones to
tune first are LR, NUM_EPISODES_PER_UPDATE, and ENTROPY_THRESHOLD.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ai_interface import run_headless_battle
from mcps import make_mcps_policy

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
INPUT_DIM         = 52        # from returnUnitParams + returnOtherUnitParams + returnBossParams
HIDDEN            = 128       # MLP hidden layer width
LR                = 3e-5      # Adam learning rate
GAMMA             = 0.99      # discount factor for returns
CLIP_EPS          = 0.2       # PPO clip epsilon (standard value)
VALUE_COEF        = 0.5       # value loss weight in combined loss
ENTROPY_COEF      = 0.01      # entropy bonus weight (encourages exploration)
PPO_EPOCHS        = 4         # optimisation passes over each collected batch
MINIBATCH_SIZE    = 64        # transitions per gradient step
NUM_EPISODES_PER_UPDATE = 16  # episodes to collect before each PPO update
MAX_TURNS         = 30        # episode time limit
TOTAL_UPDATES     = 500       # training length; ~8000 episodes total at 16/update

# Entropy-gated MCPS: run MCPS when the model's average per-decision entropy
# exceeds this threshold (in nats, since we use natural log). A freshly
# initialised model has entropy ≈ ln(8) ≈ 2.08 (uniform over 8 action
# combos). Tune this so MCPS fires on the genuinely uncertain decisions
# (~10-20% of the time in a well-trained model) without firing constantly.
ENTROPY_THRESHOLD = 0.5
MCPS_PLAYOUTS     = 50        # MCPS budget when the gate fires (tradeoff: quality vs speed)
MCPS_RHO          = 5         # GRAVE reference threshold for MCPS

# ----------------------------------------------------------------------------
# Feature extraction  (mirrors the user's existing parameter functions)
# ----------------------------------------------------------------------------
def _unit_params(i, unit):
    return [
        unit["hp"], int(unit["is_dead"]), int(unit["is_staggered"]),
        unit["num_stagger_thresholds_remaining"],
        unit["burn_potency"], unit["burn_count"],
        unit["tremor_potency"], unit["tremor_count"],
        int(unit["tremor_scorch_active"]),
        unit["magic_bullets"], unit["clash_power_level"],
        unit["damage_reduction_active"],
        *unit["skills"],   # 4 bits: current bottom/top, next bottom/top
        i,                 # unit index so a shared model knows which unit it is
    ]


def _other_unit_params(j, other):
    return [
        int(other["is_dead"]), int(other["is_staggered"]),
        other["num_stagger_thresholds_remaining"],
        other["clash_power_level"],
        j, *other["skills"],
    ]


def _boss_params(boss, commands_so_far):
    def already_clashing(slot_idx):
        return int(any(c[1] == slot_idx + 1 for c in commands_so_far))
    params = [
        boss["hp"], int(boss["is_staggered"]),
        boss["num_stagger_thresholds_remaining"],
        boss["burn_potency"], boss["burn_count"],
        boss["tremor_potency"], boss["tremor_count"],
        int(boss["tremor_scorch_active"]),
    ]
    for slot_idx in range(3):
        skill_num, target = boss["chosen_skills"][slot_idx]
        params += [skill_num, target, already_clashing(slot_idx)]
    return params


def build_features(state, unit_index, commands_so_far):
    """Build the 53-dimensional input vector for one unit's decision."""
    unit = state["units"][unit_index]
    params = _unit_params(unit_index, unit)
    for j, other in enumerate(state["units"]):
        if j != unit_index:
            params += _other_unit_params(j, other)
    params += _boss_params(state["boss"], commands_so_far)
    return torch.tensor(params, dtype=torch.float32)


# ----------------------------------------------------------------------------
# Policy + Value network (shared trunk, separate heads)
# ----------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Two action heads (two independent discrete choices per unit):
        # skill_idx in {0=bottom, 1=top}, target_idx in {0..3}
        self.skill_head  = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, 4)
        # Scalar state-value estimate (for PPO's advantage calculation)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.skill_head(h), self.target_head(h), self.value_head(h).squeeze(-1)

    def act(self, x):
        """Sample an action and return (skill, target, log_prob, entropy, value)."""
        skill_logits, target_logits, value = self(x)
        skill_dist  = torch.distributions.Categorical(logits=skill_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)
        skill  = skill_dist.sample()
        target = target_dist.sample()
        log_prob = skill_dist.log_prob(skill) + target_dist.log_prob(target)
        entropy  = (skill_dist.entropy() + target_dist.entropy()) / 2
        return skill.item(), target.item(), log_prob, entropy, value

    def log_prob_and_entropy(self, x, skills, targets):
        """Re-evaluate stored actions (used during PPO update pass)."""
        skill_logits, target_logits, values = self(x)
        skill_dist  = torch.distributions.Categorical(logits=skill_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)
        log_probs = skill_dist.log_prob(skills) + target_dist.log_prob(targets)
        entropy   = (skill_dist.entropy() + target_dist.entropy()) / 2
        return log_probs, entropy, values


# ----------------------------------------------------------------------------
# Episode rollout (one full game, collecting PPO transition data)
# ----------------------------------------------------------------------------
def run_episode(model, mcps_policy, seed=None):
    """
    Run one full episode. For each unit decision, the model either acts
    directly (low entropy) or defers to MCPS (high entropy). Either way,
    the TRANSITION is stored using the model's own log_prob for the chosen
    action (so PPO always learns from the model's own distribution, even
    when MCPS overrides the final move).

    Returns:
        transitions: list of dicts with keys
            x, skill, target, log_prob, value, reward (filled in later)
        outcome: "win" | "loss" | "draw"
    """
    rng = random.Random(seed)
    transitions = []

    # pending: transitions from the current turn, awaiting the reward that
    # comes from seeing what state results at the START of the next turn.
    pending_transitions = []
    prev_boss_hp    = None
    prev_party_hp   = None

    def policy_fn(state, battle, boss_slots):
        nonlocal prev_boss_hp, prev_party_hp, pending_transitions

        # --- assign reward to last turn's transitions ---
        boss_hp_now   = battle.boss.hp
        party_hp_now  = sum(u.hp for u in battle.units.values())
        if prev_boss_hp is not None:
            r = (prev_boss_hp - boss_hp_now) - (prev_party_hp - party_hp_now)
            # scale to a roughly ±1 range so it's comparable to the terminal bonus
            r /= 200.0
            for t in pending_transitions:
                t["reward"] = r
            transitions.extend(pending_transitions)
        prev_boss_hp  = boss_hp_now
        prev_party_hp = party_hp_now
        pending_transitions = []

        # --- decide each unit's action ---
        commands = []
        this_turn = []
        claimed_slots = set()  # target_idxs already committed this turn
        num_real_slots = sum(1 for p in state["boss"]["chosen_skills"] if p != [-1, -1])
        model.eval()
        with torch.no_grad():
            for i, unit in enumerate(state["units"]):
                if unit["is_dead"] or unit["is_staggered"]:
                    continue

                x = build_features(state, i, commands)
                skill, target, log_prob, entropy, value = model.act(x)

                # defensive: if the model picks an already-claimed slot or
                # one that doesn't exist this turn, fall back to unopposed
                if target != 0 and (target in claimed_slots or target > num_real_slots):
                    target = 0
                    # re-evaluate log_prob under the actual chosen action
                    s_t = torch.tensor(skill)
                    t_t = torch.tensor(target)
                    log_prob, _, _ = model.log_prob_and_entropy(
                        x.unsqueeze(0), s_t.unsqueeze(0), t_t.unsqueeze(0)
                    )
                    log_prob = log_prob.squeeze(0)

                if entropy.item() > ENTROPY_THRESHOLD and mcps_policy is not None:
                    # model is uncertain - ask MCPS for a better move, but
                    # keep the model's own log_prob for this state/action pair
                    # so PPO still trains on what the MODEL would have done
                    # (MCPS is just a better supervisor label for this step).
                    mcps_commands = mcps_policy(state, battle, boss_slots)
                    if len(mcps_commands) > len(commands):
                        mcps_move = mcps_commands[len(commands)]
                        mcps_skill, mcps_target = mcps_move[0], mcps_move[1]
                        # only take the MCPS suggestion if it's actually valid
                        # (not already claimed by an earlier unit this turn)
                        if mcps_target == 0 or (mcps_target not in claimed_slots and mcps_target <= num_real_slots):
                            skill, target = mcps_skill, mcps_target
                    # re-evaluate log_prob for the MCPS-chosen action under
                    # the current model, so the gradient points toward it
                    skill_t  = torch.tensor(skill)
                    target_t = torch.tensor(target)
                    log_prob, _, _ = model.log_prob_and_entropy(
                        x.unsqueeze(0), skill_t.unsqueeze(0), target_t.unsqueeze(0)
                    )
                    log_prob = log_prob.squeeze(0)

                if target != 0:
                    claimed_slots.add(target)
                commands.append([skill, target])
                this_turn.append({
                    "x":        x,
                    "skill":    torch.tensor(skill),
                    "target":   torch.tensor(target),
                    "log_prob": log_prob.detach(),
                    "value":    value.detach(),
                    "reward":   None,
                })

        pending_transitions = this_turn
        return commands

    result = run_headless_battle(policy_fn, seed=rng.randint(0, 2**31), max_turns=MAX_TURNS)

    # --- terminal reward for the final turn's transitions ---
    terminal = {"win": 1.0, "loss": -1.0, "draw": 0.0}[result["outcome"]]
    for t in pending_transitions:
        t["reward"] = terminal
    transitions.extend(pending_transitions)

    return transitions, result["outcome"]


# ----------------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------------
def compute_returns(transitions, gamma=GAMMA):
    """Discounted returns, computed backwards through the episode, then
    normalized to zero mean / unit variance so the value head doesn't have
    to learn an arbitrary scale from scratch."""
    returns = []
    G = 0.0
    for t in reversed(transitions):
        G = t["reward"] + gamma * G
        returns.append(G)
    returns.reverse()

    # normalize - keeps value loss and advantage estimates on a sane scale
    # regardless of how many turns the episode lasted or how large the
    # shaping rewards happened to be
    ret_t = torch.tensor(returns, dtype=torch.float32)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    for t, G in zip(transitions, ret_t.tolist()):
        t["return"] = G
    return transitions


def ppo_update(model, optimizer, all_transitions, clip_eps=CLIP_EPS,
               value_coef=VALUE_COEF, entropy_coef=ENTROPY_COEF,
               ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE):
    """One full PPO update: multiple epochs of minibatch gradient steps."""
    xs        = torch.stack([t["x"]        for t in all_transitions])
    skills    = torch.stack([t["skill"]    for t in all_transitions])
    targets   = torch.stack([t["target"]   for t in all_transitions])
    old_lps   = torch.stack([t["log_prob"] for t in all_transitions])
    old_vals  = torch.stack([t["value"]    for t in all_transitions])
    returns   = torch.tensor([t["return"]  for t in all_transitions], dtype=torch.float32)

    # advantages: how much better was the actual return vs. the value estimate?
    advantages = returns - old_vals
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(all_transitions)
    total_loss = total_policy_loss = total_value_loss = total_entropy = 0.0
    steps = 0

    model.train()
    for _ in range(ppo_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            x_mb   = xs[idx]
            s_mb   = skills[idx]
            t_mb   = targets[idx]
            adv_mb = advantages[idx]
            ret_mb = returns[idx]
            old_mb = old_lps[idx]

            new_lps, ent, vals = model.log_prob_and_entropy(x_mb, s_mb, t_mb)

            ratio  = (new_lps - old_mb).exp()
            p_loss = -torch.min(
                ratio * adv_mb,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb,
            ).mean()
            v_loss = (vals - ret_mb).pow(2).mean()
            e_loss = ent.mean()

            loss = p_loss + value_coef * v_loss - entropy_coef * e_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss        += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss  += v_loss.item()
            total_entropy     += e_loss.item()
            steps += 1

    return {
        "loss":         total_loss / steps,
        "policy_loss":  total_policy_loss / steps,
        "value_loss":   total_value_loss / steps,
        "entropy":      total_entropy / steps,
    }


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(resume_path=None):
    model     = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    start_update = 0

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 0)
        print(f"Resumed from {resume_path} at update {start_update}")

    for update in range(start_update, TOTAL_UPDATES):
        all_transitions = []
        outcomes = defaultdict(int)

        for ep in range(NUM_EPISODES_PER_UPDATE):
            seed = update * NUM_EPISODES_PER_UPDATE + ep
            # no MCPS during training - model learns entirely from its own
            # decisions. MCPS is only used at inference (see evaluate()).
            transitions, outcome = run_episode(model, mcps_policy=None, seed=seed)
            transitions = compute_returns(transitions)
            all_transitions.extend(transitions)
            outcomes[outcome] += 1

        if not all_transitions:
            continue

        stats = ppo_update(model, optimizer, all_transitions)

        n = NUM_EPISODES_PER_UPDATE
        win_rate = outcomes["win"] / n
        print(
            f"update {update+1:4d}/{TOTAL_UPDATES}  "
            f"wins {outcomes['win']:2d}/{n}  ({win_rate*100:5.1f}%)  "
            f"loss {stats['loss']:.4f}  "
            f"ent {stats['entropy']:.3f}"
        )

        if (update + 1) % 50 == 0:
            path = f"SecondImplementation/Checkpoints/checkpoint_{update+1}.pt"
            torch.save({"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "update": update + 1}, path)
            print(f"  saved {path}")

    torch.save({"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "update": TOTAL_UPDATES}, "SecondImplementation/finalModel.pt")
    print("Training complete. Saved final.pt")
    return model


# ----------------------------------------------------------------------------
# Evaluation (deterministic policy, no MCPS)
# ----------------------------------------------------------------------------
def evaluate(model_path, n_episodes=100):
    """Pure model evaluation - no MCPS, greedy argmax only."""
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()

    def greedy_policy(state, battle, boss_slots):
        commands      = []
        claimed_slots = set()
        num_real_slots = sum(1 for p in state["boss"]["chosen_skills"] if p != [-1, -1])
        with torch.no_grad():
            for i, unit in enumerate(state["units"]):
                if unit["is_dead"] or unit["is_staggered"]:
                    continue
                x = build_features(state, i, commands)
                skill_logits, target_logits, _ = model(x)
                skill  = skill_logits.argmax().item()
                target = target_logits.argmax().item()
                if target != 0 and (target in claimed_slots or target > num_real_slots):
                    target = 0
                if target != 0:
                    claimed_slots.add(target)
                commands.append([skill, target])
        return commands

    outcomes = defaultdict(int)
    for seed in range(n_episodes):
        result = run_headless_battle(greedy_policy, seed=seed, max_turns=MAX_TURNS)
        outcomes[result["outcome"]] += 1

    print(f"Evaluation (model only) over {n_episodes} episodes:")
    print(f"  wins:   {outcomes['win']:3d}  ({outcomes['win']/n_episodes*100:.1f}%)")
    print(f"  losses: {outcomes['loss']:3d}  ({outcomes['loss']/n_episodes*100:.1f}%)")
    print(f"  draws:  {outcomes['draw']:3d}  ({outcomes['draw']/n_episodes*100:.1f}%)")


def test_with_mcps(model_path, n_episodes=100):
    """
    Hybrid evaluation: the model plays normally when confident (entropy below
    threshold), but defers to MCPS when uncertain. This is the intended
    production mode - best of both worlds between the model's speed and
    MCPS's accuracy on hard decisions.
    """
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()

    mcps = make_mcps_policy(num_playouts=MCPS_PLAYOUTS, rho=MCPS_RHO, seed=0)

    outcomes        = defaultdict(int)
    mcps_fired      = 0
    total_decisions = 0

    def hybrid_policy(state, battle, boss_slots):
        nonlocal mcps_fired, total_decisions
        commands      = []
        claimed_slots = set()
        num_real_slots = sum(1 for p in state["boss"]["chosen_skills"] if p != [-1, -1])

        with torch.no_grad():
            for i, unit in enumerate(state["units"]):
                if unit["is_dead"] or unit["is_staggered"]:
                    continue

                x = build_features(state, i, commands)
                skill_logits, target_logits, _ = model(x)
                skill_dist  = torch.distributions.Categorical(logits=skill_logits)
                target_dist = torch.distributions.Categorical(logits=target_logits)
                entropy = (skill_dist.entropy() + target_dist.entropy()) / 2
                total_decisions += 1

                if entropy.item() > ENTROPY_THRESHOLD:
                    # uncertain - let MCPS decide
                    mcps_fired += 1
                    mcps_commands = mcps(state, battle, boss_slots)
                    if len(mcps_commands) > len(commands):
                        mcps_move  = mcps_commands[len(commands)]
                        skill, target = mcps_move[0], mcps_move[1]
                        if target != 0 and (target in claimed_slots or target > num_real_slots):
                            target = 0
                    else:
                        skill, target = skill_logits.argmax().item(), 0
                else:
                    # confident - use the model's greedy choice
                    skill  = skill_logits.argmax().item()
                    target = target_logits.argmax().item()
                    if target != 0 and (target in claimed_slots or target > num_real_slots):
                        target = 0

                if target != 0:
                    claimed_slots.add(target)
                commands.append([skill, target])
        return commands

    for seed in range(n_episodes):
        result = run_headless_battle(hybrid_policy, seed=seed, max_turns=MAX_TURNS)
        outcomes[result["outcome"]] += 1

    gate_rate = mcps_fired / total_decisions if total_decisions else 0
    print(f"Hybrid test (model + MCPS fallback, threshold={ENTROPY_THRESHOLD}) over {n_episodes} episodes:")
    print(f"  wins:   {outcomes['win']:3d}  ({outcomes['win']/n_episodes*100:.1f}%)")
    print(f"  losses: {outcomes['loss']:3d}  ({outcomes['loss']/n_episodes*100:.1f}%)")
    print(f"  draws:  {outcomes['draw']:3d}  ({outcomes['draw']/n_episodes*100:.1f}%)")
    print(f"  MCPS gate fired on {mcps_fired}/{total_decisions} decisions ({gate_rate*100:.1f}%)")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="CKPT",
                        help="resume training from a checkpoint")
    parser.add_argument("--eval",   metavar="CKPT",
                        help="evaluate a checkpoint (pure model, no MCPS)")
    parser.add_argument("--test",   metavar="CKPT",
                        help="test a checkpoint with entropy-gated MCPS fallback")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes for --eval or --test (default: 100)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, n_episodes=args.episodes)
    elif args.test:
        test_with_mcps(args.test, n_episodes=args.episodes)
    else:
        train(resume_path=args.resume)

'''
3 options to run:
python SecondImplementation/ppo_train.py                        # train from scratch
python SecondImplementation/ppo_train.py --resume checkpoint_50.pt  # resume training
python SecondImplementation/ppo_train.py --eval finalModel.pt        # pure model, no MCPS
python SecondImplementation/ppo_train.py --test finalModel.pt        # entropy-gated MCPS fallback
python SecondImplementation/ppo_train.py --test finalModel.pt --episodes 200  # more episodes
'''