"""
label_library.py
=================
Build the **GNINA differential-label dataset** that trains the oracle's proxy head
(the fast in-loop reward predictor).

Why
---
Calling real GNINA inside the GFlowNet loop is intractable (~weeks). Instead we
dock a representative molecule library OFFLINE into both KRAS conformers (acidic
i-motif + neutral ssRNA), record every score component, and later train the proxy
to predict them. The proxy then supplies a millisecond reward during generation.

Library composition
-------------------
* All HARIBOSS ligands (real RNA binders) from `hariboss_processed.csv`.
* Random molecules sampled from the **SELFIES** semantic alphabet — the SAME
  generative space the GFlowNet draws from — so the proxy's training distribution
  matches the molecules it will later score. SELFIES guarantees validity.

Output
------
`data/processed/gnina_labels.csv`, one row per molecule with ALL score components
(acidic/neutral affinity + CNN + unified score + differential). Storing the raw
components — not just the final differential — lets us reshape the reward later
WITHOUT re-docking (e.g. switch lambda, gate on absolute binding, use the pure gap).

The run is resumable: re-invoking skips molecules already present in the CSV.

Run (full library, both GPUs):
    conda activate phgfn
    python -m src.docking.label_library                 # uses cfg.data.proxy_library_size
    python -m src.docking.label_library 50              # small smoke batch
"""

from __future__ import annotations

import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.config import cfg

# Columns written to gnina_labels.csv (order matters for the header).
_FIELDS = [
    "smiles", "source",
    "acidic_affinity", "neutral_affinity",
    "acidic_cnn", "neutral_cnn",
    "acidic_score", "neutral_score",
    "differential", "ok", "error",
]


# --------------------------------------------------------------------------- #
# Library construction
# --------------------------------------------------------------------------- #
def _canon(smiles: str):
    """Canonical SMILES with a sanity size filter; None if invalid/oversized."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 6 or n_heavy > 60:
        return None
    if Descriptors.MolWt(mol) > 600:
        return None
    return Chem.MolToSmiles(mol)


def _random_selfies_smiles(rng, max_tokens: int):
    """Decode a random SELFIES token string to a (valid) SMILES."""
    import selfies as sf

    alphabet = sorted(sf.get_semantic_robust_alphabet())
    n = rng.randint(5, max_tokens)
    tokens = "".join(rng.choice(alphabet) for _ in range(n))
    try:
        return sf.decoder(tokens)
    except Exception:
        return None


def build_library(n_total: int, seed: int = 42) -> list[tuple[str, str]]:
    """
    Assemble up to `n_total` unique (canonical_smiles, source) molecules:
    HARIBOSS ligands first, then random SELFIES molecules to fill the rest.
    """
    import random

    rng = random.Random(seed)
    seen: set[str] = set()
    library: list[tuple[str, str]] = []

    # 1) HARIBOSS ligands (if preprocessing has been run).
    if Path(cfg.data.hariboss_csv).exists():
        hb = pd.read_csv(cfg.data.hariboss_csv)
        for smi in hb["ligand_smiles"].tolist():
            c = _canon(smi)
            if c and c not in seen:
                seen.add(c)
                library.append((c, "hariboss"))

    # 2) Fill with random SELFIES molecules (same space as the GFlowNet).
    max_tokens = cfg.gflownet.selfies_max_length
    attempts = 0
    while len(library) < n_total and attempts < n_total * 50:
        attempts += 1
        smi = _random_selfies_smiles(rng, max_tokens)
        c = _canon(smi) if smi else None
        if c and c not in seen:
            seen.add(c)
            library.append((c, "random_selfies"))
    return library[:n_total]


# --------------------------------------------------------------------------- #
# Docking driver (parallel across GPUs, resumable)
# --------------------------------------------------------------------------- #
def dock_library(
    n_total: int | None = None,
    out_csv: Path | None = None,
    n_parallel: int | None = None,
    gpu_ids: tuple[int, ...] = (0, 1),
    seed: int = 42,
) -> Path:
    """Dock the library into both conformers and write/extend gnina_labels.csv."""
    from src.docking.gnina import GninaDocker

    n_total = n_total or cfg.data.proxy_library_size
    out_csv = Path(out_csv or cfg.data.proxy_library_csv)
    n_parallel = n_parallel or cfg.docking.n_parallel_docks

    cfg.ensure_dirs()
    docker = GninaDocker()

    # Resume: skip molecules already docked.
    done: set[str] = set()
    if out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv)["smiles"].tolist())
        except Exception:
            done = set()

    library = [(s, src) for (s, src) in build_library(n_total, seed) if s not in done]
    print(
        f"[label] library size={n_total} | already done={len(done)} | "
        f"to dock now={len(library)} | parallel={n_parallel} GPUs={gpu_ids}"
    )
    if not library:
        print("[label] nothing to do.")
        return out_csv

    # Open CSV for append; write header only if new.
    new_file = not out_csv.exists()
    fh = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=_FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()
    write_lock = threading.Lock()
    counter = {"n": 0, "ok": 0}

    def _job(idx: int, smiles: str, source: str) -> dict:
        gpu = gpu_ids[idx % len(gpu_ids)]
        try:
            d = docker.differential(smiles, gpu_id=gpu)
        except Exception as exc:                     # never let one molecule crash the run
            d = {"smiles": smiles, "ok": False, "error": f"exception:{type(exc).__name__}:{exc}"}
        d["source"] = source
        return d

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = [
            pool.submit(_job, i, smi, src) for i, (smi, src) in enumerate(library)
        ]
        for fut in as_completed(futures):
            try:
                d = fut.result()
            except Exception as exc:                 # defense in depth
                d = {"ok": False, "error": f"future:{type(exc).__name__}:{exc}"}
            row = {k: d.get(k) for k in _FIELDS}
            with write_lock:
                writer.writerow(row)
                fh.flush()
                counter["n"] += 1
                counter["ok"] += int(bool(d.get("ok")))
                if counter["n"] % 10 == 0 or counter["n"] == len(library):
                    print(
                        f"[label] {counter['n']}/{len(library)} docked "
                        f"({counter['ok']} ok)  last diff="
                        f"{d.get('differential')}"
                    )
    fh.close()
    print(f"[label] wrote -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    dock_library(n_total=n)
