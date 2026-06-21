"""
admet.py
========
ADMET filtering of generated pHGFN candidates.

ADMET = Absorption, Distribution, Metabolism, Excretion, Toxicity. These filters
remove molecules that — no matter how selectively they bind — would never work as
oral drugs in the human body.

Filters (Lipinski's Rule of Five + extensions, all via RDKit):
  * Molecular weight   <= 500 Da
  * LogP               <= 5
  * H-bond donors      <= 5
  * H-bond acceptors   <= 10
  * Rotatable bonds    <= 10
  * QED                >= 0.5      (overall drug-likeness, 0-1)
  * SA score           <= 6.0      (synthetic accessibility; skipped if unavailable)
  * No PAINS alerts                (Pan-Assay INterference compounds — frequent
                                    false-positives in screens)

Thresholds come from `cfg.eval`. Reads `results/candidates.csv`, writes
`results/candidates_filtered.csv`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import cfg
from src.gflownet.environment import SELFIESEnvironment

_ENV = SELFIESEnvironment()


@lru_cache(maxsize=1)
def _pains_catalog():
    """Build (once) the RDKit PAINS filter catalog."""
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def has_pains(smiles: str) -> bool:
    """True if the molecule matches any PAINS alert."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return _pains_catalog().HasMatch(mol)


def filter_candidates(
    in_csv: Path | None = None, out_csv: Path | None = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Filter `candidates.csv` to drug-like, PAINS-free molecules.

    Adds ADMET columns (MW, LogP, HBD, HBA, QED, SA_score, rotatable_bonds, TPSA,
    pains) and keeps rows passing every filter. Writes the filtered CSV.
    """
    in_csv = Path(in_csv or (cfg.system.results_dir / "candidates.csv"))
    out_csv = Path(out_csv or (cfg.system.results_dir / "candidates_filtered.csv"))
    df = pd.read_csv(in_csv).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    records = []
    for smi in df["smiles"]:
        admet = _ENV.compute_all_admet(smi)
        if admet is None:
            records.append({"keep": False})
            continue
        admet = dict(admet)
        admet["pains"] = has_pains(smi)
        admet["keep"] = bool(admet["is_drug_like"] and not admet["pains"])
        records.append(admet)
    admet_df = pd.DataFrame(records)
    merged = pd.concat([df.reset_index(drop=True), admet_df], axis=1)
    kept = merged[merged["keep"] == True].drop(columns=["keep"]).reset_index(drop=True)  # noqa: E712
    kept.to_csv(out_csv, index=False)

    if verbose:
        print("========== ADMET filtering ==========")
        print(f"  input candidates : {len(df)}")
        print(f"  passed ADMET+PAINS: {len(kept)} ({100*len(kept)/max(1,len(df)):.1f}%)")
        print(f"  saved -> {out_csv}")
    return kept


if __name__ == "__main__":
    filter_candidates()
