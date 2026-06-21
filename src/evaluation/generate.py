"""
generate.py
===========
Sample candidate molecules from the trained pHGFN policy, score them with the
proxy oracle, and (optionally) validate the best ones with REAL GNINA docking.

Outputs
-------
* `results/candidates.csv` — unique generated molecules with proxy scores:
  smiles, proxy_acidic, proxy_neutral, differential.
* `results/candidates_gnina_validated.csv` — top-K re-docked with real GNINA so we
  can report true (not proxy) selectivity for the headline molecules and quantify
  proxy↔GNINA agreement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.config import cfg
from src.gflownet.environment import SELFIESEnvironment
from src.gflownet.policy import pHConditionedPolicy
from src.oracle.model import pHGFNOracle


@torch.no_grad()
def generate_candidates(
    n: int | None = None,
    policy_ckpt: Path | None = None,
    oracle_ckpt: Path | None = None,
    out_csv: Path | None = None,
    batch: int = 256,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample `n` molecules from the policy and score them with the proxy oracle."""
    n = n or cfg.eval.n_final_candidates
    out_csv = Path(out_csv or (cfg.system.results_dir / "candidates.csv"))
    cfg.ensure_dirs()

    oracle_dev = f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu"
    policy_dev = f"cuda:{cfg.system.secondary_gpu}" if torch.cuda.device_count() > 1 else oracle_dev

    env = SELFIESEnvironment()
    policy = pHConditionedPolicy(env=env).to(policy_dev)
    pol_ckpt = Path(policy_ckpt or (cfg.system.checkpoint_dir / "policy_best.pt"))
    policy.load_state_dict(torch.load(pol_ckpt, map_location=policy_dev, weights_only=True)["policy"])
    policy.eval()

    oracle = pHGFNOracle(device=oracle_dev)
    orc_ckpt = Path(oracle_ckpt or (cfg.system.checkpoint_dir / "oracle_best.pt"))
    if orc_ckpt.exists():
        oracle.load(orc_ckpt)
    oracle.eval()

    # Sample (at the tumour pH) and keep unique valid molecules.
    seen: set[str] = set()
    smiles_all: list[str] = []
    while len(smiles_all) < n:
        _, _, smiles = policy.sample(min(batch, n), cfg.gflownet.target_ph, device=policy_dev)
        for s in smiles:
            if s and s not in seen:
                seen.add(s)
                smiles_all.append(s)
        if len(seen) > 0 and len(smiles_all) >= n:
            break
        # Guard against a degenerate policy that can't produce enough uniques.
        if len(smiles_all) == 0 and len(seen) == 0:
            break
    smiles_all = smiles_all[:n]

    # Proxy scores in real units.
    acidic, neutral = [], []
    for start in range(0, len(smiles_all), batch):
        chunk = smiles_all[start:start + batch]
        acidic.append(oracle.predict_score(chunk, cfg.gflownet.target_ph).cpu())
        neutral.append(oracle.predict_score(chunk, cfg.gflownet.comparison_ph).cpu())
    a = torch.cat(acidic).numpy() if acidic else []
    nn = torch.cat(neutral).numpy() if neutral else []
    lam = cfg.docking.selectivity_lambda
    df = pd.DataFrame({
        "smiles": smiles_all,
        "proxy_acidic": a,
        "proxy_neutral": nn,
        "differential": [ai - lam * ni for ai, ni in zip(a, nn)],
    }).sort_values("differential", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"generated {len(df)} unique candidates -> {out_csv}")
    return df


def validate_with_gnina(
    in_csv: Path | None = None,
    out_csv: Path | None = None,
    top_k: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-dock the top-K candidates (by proxy differential) with REAL GNINA."""
    from src.docking.gnina import GninaDocker

    in_csv = Path(in_csv or (cfg.system.results_dir / "candidates_filtered.csv"))
    out_csv = Path(out_csv or (cfg.system.results_dir / "candidates_gnina_validated.csv"))
    top_k = top_k or cfg.eval.n_gnina_validate

    df = pd.read_csv(in_csv).sort_values("differential", ascending=False).head(top_k).reset_index(drop=True)
    docker = GninaDocker()
    rows = []
    for i, smi in enumerate(df["smiles"]):
        d = docker.differential(smi, gpu_id=i % 2)
        rows.append({
            "smiles": smi,
            "proxy_differential": df.loc[i, "differential"] if "differential" in df else None,
            "gnina_acidic": d["acidic_score"],
            "gnina_neutral": d["neutral_score"],
            "gnina_differential": d["differential"],
            "ok": d["ok"],
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  gnina-validated {i+1}/{len(df)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    if verbose:
        ok = out[out["ok"] == True]  # noqa: E712
        if len(ok) > 1:
            corr = ok["proxy_differential"].corr(ok["gnina_differential"])
            print(f"proxy<->GNINA differential correlation (n={len(ok)}): {corr:.3f}")
        print(f"saved -> {out_csv}")
    return out


if __name__ == "__main__":
    generate_candidates()
    from src.evaluation.admet import filter_candidates
    filter_candidates()
    validate_with_gnina()
