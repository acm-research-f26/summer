"""
pretrain.py
===========
Behaviour-clone (MLE pretrain) the pH-conditioned policy on real drug-like
molecules, so the GFlowNet STARTS on the drug-like manifold instead of having to
discover it from random SELFIES (which decode to tiny, ring-less, non-drug-like
fragments — confirmed 0/5000 drug-like).

Pipeline
--------
1. build_corpus(): from ZINC250k (a standard drug-like SMILES benchmark), keep
   molecules that are (a) valid, (b) expressible in our SELFIES vocab, (c) fit the
   max length, and (d) drug-like-structured (>= min heavy atoms, >= 1 ring). Save
   the SMILES corpus to data/processed/druglike_corpus.smi.

2. pretrain_policy(): maximum-likelihood next-token training (minimise the negative
   per-token log-prob = behaviour cloning). pH is sampled uniformly from
   {target, comparison} so the pH pathway is exercised; drug-likeness itself is
   pH-independent. Saves checkpoints/policy_pretrained.pt.

The GFlowNet trainer then initialises from this checkpoint and fine-tunes for
selectivity with Trajectory Balance.

Run:
    python -c "from src.gflownet.pretrain import build_corpus, pretrain_policy; build_corpus(); pretrain_policy()"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import cfg
from src.gflownet.environment import SELFIESEnvironment
from src.gflownet.policy import pHConditionedPolicy
from src.utils.seeding import set_seed

_CORPUS = cfg.data.processed_dir / "druglike_corpus.smi"
_ZINC = cfg.data.processed_dir / "zinc250k.csv"


# --------------------------------------------------------------------------- #
# SMILES <-> action indices
# --------------------------------------------------------------------------- #
def smiles_to_actions(smiles: str, env: SELFIESEnvironment):
    """SMILES -> list of action indices (SELFIES tokens + EOS) or None if it
    doesn't fit our vocab / max length."""
    import selfies as sf

    try:
        enc = sf.encoder(smiles)
    except Exception:
        return None
    toks = list(sf.split_selfies(enc))
    if len(toks) + 1 > env.max_length:
        return None
    idx = []
    for t in toks:
        j = env.token_to_idx.get(t)
        if j is None:
            return None
        idx.append(j)
    idx.append(env.eos_idx)
    return idx


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #
def build_corpus(target: int = 30000, scan_limit: int = 200000, verbose: bool = True) -> Path:
    """Build a drug-like SMILES corpus from ZINC250k that fits our SELFIES vocab."""
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if not _ZINC.exists():
        raise FileNotFoundError(
            f"{_ZINC} missing. Download ZINC250k first (see README / pretrain docs)."
        )
    env = SELFIESEnvironment()
    df = pd.read_csv(_ZINC, nrows=scan_limit)
    df["smiles"] = df["smiles"].astype(str).str.strip()

    kept: list[str] = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Cheap drug-like-structure gate (full ADMET is enforced later by the reward).
        if mol.GetNumHeavyAtoms() < cfg.eval.min_heavy_atoms or rdMolDescriptors.CalcNumRings(mol) < cfg.eval.min_rings:
            continue
        cano = Chem.MolToSmiles(mol)
        if smiles_to_actions(cano, env) is None:   # must fit vocab + length
            continue
        kept.append(cano)
        if len(kept) >= target:
            break

    kept = list(dict.fromkeys(kept))               # dedup, preserve order
    _CORPUS.write_text("\n".join(kept))
    if verbose:
        print(f"[corpus] kept {len(kept)} drug-like molecules -> {_CORPUS}")
    return _CORPUS


# --------------------------------------------------------------------------- #
# MLE pretraining
# --------------------------------------------------------------------------- #
def _collate(action_lists, pad_idx, device):
    lengths = torch.tensor([len(a) for a in action_lists], device=device)
    T = int(lengths.max().item())
    actions = torch.full((len(action_lists), T), pad_idx, dtype=torch.long, device=device)
    for i, a in enumerate(action_lists):
        actions[i, : len(a)] = torch.tensor(a, device=device)
    return actions, lengths


def pretrain_policy(epochs: int = 15, batch_size: int = 256, lr: float = 5e-4, verbose: bool = True):
    """Behaviour-clone the policy on the drug-like corpus; save policy_pretrained.pt."""
    set_seed(cfg.system.seed)
    cfg.ensure_dirs()
    if not _CORPUS.exists():
        build_corpus()
    device = f"cuda:{cfg.system.secondary_gpu}" if torch.cuda.device_count() > 1 else \
        ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = SELFIESEnvironment()
    smiles = [s for s in _CORPUS.read_text().splitlines() if s]
    corpus = [a for a in (smiles_to_actions(s, env) for s in smiles) if a is not None]
    if verbose:
        print(f"[pretrain] corpus: {len(corpus)} molecules | device {device}")

    policy = pHConditionedPolicy(env=env).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.RandomState(cfg.system.seed)
    phs = [cfg.gflownet.target_ph, cfg.gflownet.comparison_ph]

    for epoch in range(1, epochs + 1):
        policy.train()
        perm = rng.permutation(len(corpus))
        total, nb = 0.0, 0
        for start in range(0, len(corpus), batch_size):
            batch = [corpus[i] for i in perm[start:start + batch_size]]
            actions, lengths = _collate(batch, env.pad_idx, device)
            ph = float(rng.choice(phs))
            # MLE = maximise log P(trajectory) = minimise negative per-token log-prob.
            lp = policy.trajectory_log_prob(actions, lengths, ph)
            loss = -(lp / lengths.clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); nb += 1

        # quick generation check
        policy.eval()
        _, _, gen = policy.sample(128, cfg.gflownet.target_ph, device=device)
        dl = sum(1 for s in gen if s and (lambda a: a and a["is_drug_like"])(env.compute_all_admet(s)))
        if verbose:
            print(f"[pretrain] epoch {epoch:2d}/{epochs} | nll {total/max(1,nb):.3f} "
                  f"| sampled drug-like {dl}/128 ({100*dl/128:.0f}%)")

    ckpt = cfg.system.checkpoint_dir / "policy_pretrained.pt"
    torch.save({"policy": policy.state_dict()}, ckpt)
    if verbose:
        print(f"[pretrain] saved -> {ckpt}")
    return policy


if __name__ == "__main__":
    build_corpus()
    pretrain_policy()
