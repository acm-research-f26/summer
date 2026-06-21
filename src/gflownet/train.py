"""
train.py  (GFlowNet training)
=============================
Train the pH-conditioned GFlowNet to generate diverse, tumour-selective molecules.

GPU layout
----------
* Oracle (proxy reward) runs on cuda:0 (it was trained there).
* Policy trains on cuda:1.
Reward tensors are computed on cuda:0 (no grad through the oracle) and moved to
cuda:1 for the Trajectory Balance update.

Reward
------
For each sampled molecule:
  1. Decode SELFIES -> SMILES (SELFIES guarantees validity).
  2. ADMET filter: non-drug-like (or undecodable) -> tiny reward (eps), so the
     policy learns to avoid them.
  3. Drug-like -> raw = oracle.differential_reward(mol)  (proxy GNINA differential,
     a structurally-grounded selectivity signal), shaped to a POSITIVE reward via
     softplus(raw / temperature). TB needs positive rewards (it takes log R).

The oracle is ALWAYS eval()/no_grad — gradients never flow into it.

Replay buffer
-------------
High-reward trajectories are stored and replayed alongside fresh on-policy samples
to avoid forgetting good molecules discovered early.

Run (after the oracle is trained):
    conda activate phgfn
    python -c "from src.gflownet.train import train_gflownet; train_gflownet()"
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import cfg
from src.gflownet.environment import SELFIESEnvironment
from src.gflownet.losses import trajectory_balance_loss
from src.gflownet.policy import pHConditionedPolicy
from src.oracle.model import pHGFNOracle
from src.utils.seeding import set_seed

_EPS = 1e-3  # reward floor for invalid / non-drug-like molecules


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Capacity-bounded store of (reward, action_list) keeping the highest rewards."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._heap: list[tuple[float, int, list]] = []   # min-heap on reward
        self._counter = 0

    def push(self, action_list: list[int], reward: float) -> None:
        self._counter += 1
        item = (reward, self._counter, action_list)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        elif reward > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sample(self, n: int, rng: np.random.RandomState):
        if not self._heap:
            return []
        n = min(n, len(self._heap))
        idx = rng.choice(len(self._heap), size=n, replace=False)
        return [self._heap[i] for i in idx]

    def __len__(self) -> int:
        return len(self._heap)


# --------------------------------------------------------------------------- #
# Batching helpers
# --------------------------------------------------------------------------- #
def _collate(action_lists: list[list[int]], pad_idx: int, device):
    """Pad a list of variable-length action sequences to [B, Tmax] + lengths."""
    lengths = torch.tensor([len(a) for a in action_lists], device=device)
    T = max(1, int(lengths.max().item()))
    actions = torch.full((len(action_lists), T), pad_idx, dtype=torch.long, device=device)
    for i, a in enumerate(action_lists):
        if a:
            actions[i, : len(a)] = torch.tensor(a, device=device)
    return actions, lengths


def _actions_to_lists(actions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Inverse of _collate for storing fresh samples in the replay buffer."""
    out = []
    for i in range(actions.size(0)):
        out.append(actions[i, : int(lengths[i].item())].tolist())
    return out


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_rewards(oracle, env, smiles, policy_device, temperature: float):
    """
    Map a batch of SMILES (some may be None) to positive rewards [B] on the policy
    device, plus per-item drug-like flags.

    GRADED reward (avoids a sparse cold-start under the strict ADMET filter):
      * invalid SMILES                  -> _EPS (0.001)
      * valid but NOT drug-like         -> 0.02 + 0.08*QED   (climb toward drug-likeness)
      * drug-like                       -> 0.10 + exp(clamp(differential)/T)
                                           (drug-likeness floor + selectivity on top)
    The 0.10 floor guarantees any drug-like molecule outscores the best non-drug-like
    one, while the exp term drives selectivity among drug-like molecules.
    """
    n = len(smiles)
    rewards = np.full(n, _EPS, dtype=np.float32)
    drug_like = np.zeros(n, dtype=bool)

    dl_idx, dl_smiles = [], []
    for i, s in enumerate(smiles):
        if not s:
            continue
        admet = env.compute_all_admet(s)
        if admet is None:
            continue
        if admet["is_drug_like"]:
            dl_idx.append(i)
            dl_smiles.append(s)
            drug_like[i] = True
        else:
            # Componentised guidance: reward partial progress toward EACH drug-like
            # sub-criterion (size, rings, MW, QED), not just QED — otherwise the
            # policy maximises QED among tiny ring-less molecules and never reaches
            # the drug-like manifold. progress in [0,1] -> partial in [0.02, 0.10].
            e = cfg.eval
            mw_ok = min(admet["MW"] / e.mw_min, 1.0)
            heavy_ok = min(admet.get("n_heavy_atoms", 0) / e.min_heavy_atoms, 1.0)
            ring_ok = min(admet.get("n_rings", 0) / max(1, e.min_rings), 1.0)
            qed_ok = min(admet["QED"] / e.qed_min, 1.0)
            carbon_ok = 1.0 if admet.get("has_carbon", True) else 0.0
            progress = 0.25 * mw_ok + 0.25 * heavy_ok + 0.25 * ring_ok + 0.20 * qed_ok + 0.05 * carbon_ok
            rewards[i] = max(_EPS, 0.02 + 0.08 * progress)

    if dl_smiles:
        raw = oracle.differential_reward(dl_smiles).clamp(-10.0, 5.0)  # [V], proxy GNINA diff
        shaped = torch.exp(raw / temperature).float().cpu().numpy()
        for j, i in enumerate(dl_idx):
            rewards[i] = 0.10 + float(shaped[j])
    return torch.tensor(rewards, device=policy_device), drug_like


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_gflownet(epochs: int = None, verbose: bool = True, oracle_ckpt: str = None):
    """Train the GFlowNet; returns the trained policy. Saves checkpoints/policy_best.pt."""
    set_seed(cfg.system.seed)
    cfg.ensure_dirs()
    gc = cfg.gflownet
    epochs = epochs or gc.epochs
    rng = np.random.RandomState(cfg.system.seed)

    oracle_dev = f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu"
    policy_dev = f"cuda:{cfg.system.secondary_gpu}" if torch.cuda.device_count() > 1 else oracle_dev

    # ---- oracle (frozen reward) ----
    oracle = pHGFNOracle(device=oracle_dev)
    ckpt = Path(oracle_ckpt or (cfg.system.checkpoint_dir / "oracle_best.pt"))
    if ckpt.exists():
        oracle.load(ckpt)
        if verbose:
            print(f"loaded oracle <- {ckpt}")
    else:
        print(f"WARNING: no oracle checkpoint at {ckpt}; using UNTRAINED oracle "
              "(rewards will be meaningless — train the oracle first).")
    oracle.eval()

    # ---- policy (initialise from the behaviour-cloned checkpoint if available) ----
    env = SELFIESEnvironment()
    policy = pHConditionedPolicy(env=env).to(policy_dev)
    pretrained = cfg.system.checkpoint_dir / "policy_pretrained.pt"
    if pretrained.exists():
        policy.load_state_dict(
            torch.load(pretrained, map_location=policy_dev, weights_only=True)["policy"]
        )
        if verbose:
            print(f"initialised policy from pretrained {pretrained}")
    else:
        print("WARNING: no pretrained policy; GFlowNet starts from random init "
              "(may not reach drug-like space — run src.gflownet.pretrain first).")
    optimizer = torch.optim.Adam(
        [
            {"params": [p for n, p in policy.named_parameters() if n != "log_Z"], "lr": gc.lr},
            {"params": [policy.log_Z], "lr": gc.lr * 100},   # log_Z must adapt quickly to track Z
        ]
    )
    use_amp = gc.use_fp16 and policy_dev.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    buffer = ReplayBuffer(gc.replay_buffer_size)

    best_reward, best_smiles = -1.0, None
    history = {"mean_reward": [], "frac_drug_like": [], "best_reward": []}

    for epoch in range(1, epochs + 1):
        # ---- sample fresh trajectories at the tumour pH ----
        policy.eval()
        actions, lengths, smiles = policy.sample(gc.n_samples_per_epoch, gc.target_ph, device=policy_dev)
        rewards, drug_like = compute_rewards(oracle, env, smiles, policy_dev, gc.reward_temperature)

        # ---- push good trajectories to the replay buffer ----
        action_lists = _actions_to_lists(actions, lengths)
        for al, r, dl in zip(action_lists, rewards.tolist(), drug_like):
            if dl:
                buffer.push(al, r)
        # track best
        bidx = int(torch.argmax(rewards).item())
        if drug_like[bidx] and rewards[bidx].item() > best_reward:
            best_reward, best_smiles = rewards[bidx].item(), smiles[bidx]

        # ---- TB update on fresh batch (+ replay minibatch) ----
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            lp = policy.trajectory_log_prob(actions, lengths, gc.target_ph)
            loss = trajectory_balance_loss(policy.log_Z, lp, rewards)
            replay = buffer.sample(gc.batch_size, rng)
            if replay:
                r_rewards = torch.tensor([it[0] for it in replay], device=policy_dev)
                r_actions, r_lengths = _collate([it[2] for it in replay], env.pad_idx, policy_dev)
                r_lp = policy.trajectory_log_prob(r_actions, r_lengths, gc.target_ph)
                loss = loss + trajectory_balance_loss(policy.log_Z, r_lp, r_rewards)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _policy_params = [p for n, p in policy.named_parameters() if n != "log_Z"]
        torch.nn.utils.clip_grad_norm_(_policy_params, gc.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ---- logging ----
        frac_dl = float(drug_like.mean())
        history["mean_reward"].append(float(rewards.mean()))
        history["frac_drug_like"].append(frac_dl)
        history["best_reward"].append(best_reward)
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"epoch {epoch:4d}/{epochs} | loss {loss.item():9.2f} "
                  f"| mean_R {float(rewards.mean()):.3f} | best_R {best_reward:.3f} "
                  f"| drug-like {frac_dl*100:4.1f}% | logZ {policy.log_Z.item():.2f} "
                  f"| buf {len(buffer)} | best: {best_smiles}")

        if epoch % 50 == 0 or epoch == epochs:
            _save(policy, cfg.system.checkpoint_dir / "policy_best.pt", best_smiles, best_reward)

    _save(policy, cfg.system.checkpoint_dir / "policy_best.pt", best_smiles, best_reward)
    _plot(history)
    if verbose:
        print(f"\nDONE. best reward {best_reward:.3f} | best molecule {best_smiles}")
    return policy


def _save(policy, path: Path, best_smiles, best_reward) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"policy": policy.state_dict(), "log_Z": policy.log_Z.detach().cpu(),
         "best_smiles": best_smiles, "best_reward": best_reward},
        path,
    )


def _plot(history: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["mean_reward"], label="mean reward")
    ax[0].plot(history["best_reward"], label="best reward")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("reward"); ax[0].legend(); ax[0].set_title("Reward")
    ax[1].plot(history["frac_drug_like"])
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("fraction"); ax[1].set_title("Drug-like fraction")
    fig.tight_layout()
    out = cfg.system.results_dir / "gflownet_training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved curves -> {out}")


if __name__ == "__main__":
    train_gflownet()
