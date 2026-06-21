"""
diversity.py
============
Diversity metrics for pHGFN-generated molecules. Diversity is the whole point of
using a GFlowNet (vs. RL, which collapses to one mode), so we quantify it three ways.

1. Scaffold diversity  -- # unique Bemis-Murcko scaffolds / # molecules.
                          Higher = broader coverage of distinct chemical frameworks.
2. Internal diversity  -- mean (1 - Tanimoto) over random molecule pairs.
                          Higher = the set spreads out across chemical space.
3. Novelty             -- fraction of molecules whose max Tanimoto to any TRAINING
                          molecule is < threshold. Higher = genuinely new chemistry,
                          not memorised HARIBOSS ligands.

All similarities use Morgan fingerprints (radius 2, 2048 bits) — the standard for
drug-likeness comparisons.

Writes `results/diversity_metrics.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg


def _morgan(smiles_list):
    """Return list of (smiles, fingerprint) for valid molecules (Morgan r=2, 2048 bits)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out.append((smi, gen.GetFingerprint(mol)))
    return out


def _tanimoto(fp1, fp2) -> float:
    from rdkit import DataStructs

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def scaffold_diversity(smiles_list) -> float:
    """Unique Bemis-Murcko scaffolds / number of molecules."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = set()
    n = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n += 1
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    return len(scaffolds) / max(1, n)


def internal_diversity(fps, n_pairs: int = 10000, seed: int = 42) -> float:
    """Mean (1 - Tanimoto) over up to n_pairs random molecule pairs."""
    if len(fps) < 2:
        return 0.0
    rng = np.random.RandomState(seed)
    dists = []
    for _ in range(min(n_pairs, len(fps) * (len(fps) - 1) // 2)):
        i, j = rng.randint(0, len(fps), size=2)
        if i != j:
            dists.append(1.0 - _tanimoto(fps[i][1], fps[j][1]))
    return float(np.mean(dists)) if dists else 0.0


def novelty(gen_fps, train_fps, threshold: float) -> float:
    """Fraction of generated molecules with max Tanimoto to training set < threshold."""
    if not gen_fps:
        return 0.0
    if not train_fps:
        return 1.0
    novel = 0
    for _, g in gen_fps:
        max_sim = max(_tanimoto(g, t) for _, t in train_fps)
        if max_sim < threshold:
            novel += 1
    return novel / len(gen_fps)


def compute_all_metrics(
    filtered_csv: Path | None = None,
    training_csv: Path | None = None,
    out_json: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Compute scaffold/internal/novelty metrics and write the JSON summary."""
    filtered_csv = Path(filtered_csv or (cfg.system.results_dir / "candidates_filtered.csv"))
    training_csv = Path(training_csv or cfg.data.hariboss_csv)
    out_json = Path(out_json or (cfg.system.results_dir / "diversity_metrics.json"))

    gen_smiles = pd.read_csv(filtered_csv)["smiles"].tolist()
    train_smiles = (
        pd.read_csv(training_csv)["ligand_smiles"].tolist() if training_csv.exists() else []
    )
    gen_fps = _morgan(gen_smiles)
    train_fps = _morgan(train_smiles)

    metrics = {
        "n_molecules": len(gen_smiles),
        "scaffold_diversity": scaffold_diversity(gen_smiles),
        "internal_diversity": internal_diversity(gen_fps),
        "novelty": novelty(gen_fps, train_fps, cfg.eval.tanimoto_novelty_threshold),
        "novelty_threshold": cfg.eval.tanimoto_novelty_threshold,
        "n_training_reference": len(train_fps),
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print("========== Diversity metrics ==========")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v}")
        print(f"  saved -> {out_json}")
    return metrics


if __name__ == "__main__":
    compute_all_metrics()
