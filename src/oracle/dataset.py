"""
dataset.py
==========
Turn the raw HARIBOSS PDB complexes into a tabular dataset for training the
oracle's **binding-affinity head** (the fast general RNA-ligand filter).

Pipeline (per PDB in `hariboss/`)
---------------------------------
1. Parse the structure with BioPython.
2. Extract RNA chains (standard residues A/U/G/C) and keep the longest as the
   receptor sequence (RNA-FM consumes one sequence).
3. Extract small-molecule ligands: HETATM residues that are NOT water, NOT
   standard nucleotides, NOT common ions/buffers, and have MW > 100 Da.
4. Convert each ligand's 3D coordinates to a SMILES with RDKit (PDB bond
   perception).
5. Compute a proxy binding label:  label = -0.5 * (#RNA-ligand atom contacts < 4.5 A).
   More contacts -> more negative -> "stronger" proxy binding. This is only a
   weak surrogate; the *structurally grounded* selectivity signal comes from
   GNINA (see src/docking/gnina.py), not from this label.
6. Write one row per (PDB, ligand) to `data/processed/hariboss_processed.csv`.

Tokenisation is intentionally NOT done here: the encoders in
`src/oracle/encoders.py` accept raw strings and tokenise internally, so the CSV
stays human-readable and the torch `Dataset` below just yields strings + label.

Run:
    conda activate phgfn
    python -c "from src.oracle.dataset import preprocess_hariboss; preprocess_hariboss()"
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import cfg

# Standard RNA residue names (single-letter, as they appear in PDB ATOM records).
_RNA_RESIDUES = {"A", "U", "G", "C"}
# DNA, in case a complex carries a DNA chain (we ignore these for the RNA seq).
_DNA_RESIDUES = {"DA", "DT", "DG", "DC", "DU"}
# Common non-drug HETATMs to reject even if MW slips past the threshold
# (ions, buffers, cryoprotectants, polyamines, sugars).
_HETATM_BLOCKLIST = {
    "HOH", "WAT", "DOD",                          # water
    "NA", "K", "MG", "MN", "ZN", "CA", "CL", "BR", "IOD", "CD", "NI", "CO", "CU",
    "SO4", "PO4", "PI", "ACT", "FMT", "EDO", "GOL", "PEG", "PG4", "DMS", "MPD",
    "TRS", "EPE", "IPA", "BME", "NH4", "NCO", "SPM", "SPD", "PUT", "MES",
}


# --------------------------------------------------------------------------- #
# Low-level extraction helpers
# --------------------------------------------------------------------------- #
def _three_to_one_rna(resname: str) -> Optional[str]:
    """Return 'A'/'U'/'G'/'C' for an RNA residue name, else None."""
    rn = resname.strip()
    if rn in _RNA_RESIDUES:
        return rn
    # Some files use 'RA','RU','RG','RC' or modified-residue parents; be liberal.
    if len(rn) >= 1 and rn[-1] in _RNA_RESIDUES and rn not in _DNA_RESIDUES:
        return rn[-1]
    return None


def _extract_rna_chains(model) -> list[tuple[str, str, list]]:
    """
    Return [(chain_id, sequence, [residues])] for chains that look like RNA.
    A chain is RNA if a majority of its residues are A/U/G/C.
    """
    out = []
    for chain in model:
        residues, seq = [], []
        for res in chain:
            one = _three_to_one_rna(res.get_resname())
            if one is not None:
                seq.append(one)
                residues.append(res)
        # Require at least a few RNA residues to call it an RNA chain.
        if len(seq) >= 3:
            out.append((chain.id, "".join(seq), residues))
    return out


def _ligand_pdb_block(residue) -> str:
    """Serialise a single HETATM residue to a minimal PDB block for RDKit."""
    lines = []
    for i, atom in enumerate(residue.get_atoms(), start=1):
        x, y, z = atom.coord
        element = (atom.element or "").strip().rjust(2)
        name = atom.get_name()
        # PDB-ish HETATM line; columns are approximate but RDKit's PDB reader is tolerant.
        lines.append(
            f"HETATM{i:>5} {name:<4}{residue.get_resname():>3} A{1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}"
        )
    lines.append("END")
    return "\n".join(lines)


def _canonicalize(smiles: Optional[str]) -> Optional[str]:
    """Validate a SMILES with RDKit and return its canonical form (or None)."""
    if not smiles:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Keep only the largest fragment (drops counter-ions packaged in the CCD entry).
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _ccd_smiles(code: str, cache: dict) -> Optional[str]:
    """
    Look up a ligand's canonical SMILES by its 3-letter PDB Chemical Component
    Dictionary code via the RCSB data API. Results (including misses, stored as
    "") are cached on disk so we hit the network at most once per code.

    This is a small metadata lookup keyed by component code -- it does NOT
    re-download any of the provided datasets; it just fetches the *correct* bond
    orders/aromaticity that cannot be recovered from coordinates alone.
    """
    code = code.strip().upper()
    if code in cache:                       # "" means known-miss
        return cache[code] or None
    smi = None
    try:
        import requests

        r = requests.get(
            f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=15
        )
        if r.ok:
            d = r.json().get("rcsb_chem_comp_descriptor", {}) or {}
            smi = _canonicalize(d.get("SMILES_stereo") or d.get("SMILES"))
    except Exception:
        smi = None
    cache[code] = smi or ""                 # cache the miss to avoid re-querying
    return smi


def _ligand_to_smiles(residue) -> Optional[str]:
    """
    Convert a HETATM residue's 3D coordinates to a canonical SMILES via RDKit.
    Fallback only -- bond orders perceived from coordinates are unreliable.
    Returns None if bond perception / sanitisation fails.
    """
    from rdkit import Chem

    block = _ligand_pdb_block(residue)
    for kwargs in ({"sanitize": True}, {"sanitize": False}):
        mol = Chem.MolFromPDBBlock(block, removeHs=True, proximityBonding=True, **kwargs)
        if mol is None:
            continue
        try:
            if not kwargs["sanitize"]:
                Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            if smi and "." not in smi:          # reject fragmented / disconnected perceptions
                return smi
            if smi:                              # keep largest fragment if disconnected
                frags = sorted(smi.split("."), key=len, reverse=True)
                return frags[0]
        except Exception:
            continue
    return None


def _ligand_mw(residue) -> float:
    """Rough molecular weight from element masses (no H needed for the filter)."""
    masses = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974,
        "S": 32.06, "F": 18.998, "CL": 35.45, "BR": 79.904, "I": 126.90,
    }
    return sum(masses.get((a.element or "").strip().upper(), 12.0) for a in residue.get_atoms())


def _count_contacts(rna_residues: list, ligand_residue, cutoff: float = 4.5) -> int:
    """Count RNA-atom / ligand-atom pairs within `cutoff` Angstroms."""
    rna_coords = np.array(
        [a.coord for res in rna_residues for a in res.get_atoms()], dtype=float
    )
    lig_coords = np.array([a.coord for a in ligand_residue.get_atoms()], dtype=float)
    if len(rna_coords) == 0 or len(lig_coords) == 0:
        return 0
    # Pairwise distances; small molecules so this is cheap.
    d = np.linalg.norm(rna_coords[:, None, :] - lig_coords[None, :, :], axis=-1)
    return int((d < cutoff).sum())


# --------------------------------------------------------------------------- #
# Main preprocessing
# --------------------------------------------------------------------------- #
def preprocess_hariboss(verbose: bool = True) -> pd.DataFrame:
    """
    Parse all HARIBOSS PDBs into `hariboss_processed.csv` and return the DataFrame.
    """
    from Bio.PDB import PDBParser

    cfg.ensure_dirs()
    parser = PDBParser(QUIET=True)
    pdb_files = sorted(Path(cfg.data.hariboss_dir).glob("*.pdb"))

    # Disk-cached map: CCD code -> canonical SMILES (or "" for known-misses).
    ccd_cache_path = cfg.data.processed_dir / "ccd_smiles_cache.json"
    ccd_cache: dict = (
        json.loads(ccd_cache_path.read_text()) if ccd_cache_path.exists() else {}
    )

    rows: list[dict] = []
    n_files = n_with_rna = n_pairs = n_failed = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence BioPython discontinuity warnings
        for pdb in pdb_files:
            n_files += 1
            try:
                structure = parser.get_structure(pdb.stem, str(pdb))
                model = next(iter(structure))           # first model only
            except Exception as exc:                     # unparyable file
                n_failed += 1
                if verbose:
                    print(f"  [skip] {pdb.name}: parse error ({exc})")
                continue

            rna_chains = _extract_rna_chains(model)
            if not rna_chains:
                continue
            n_with_rna += 1
            # Longest RNA chain = receptor sequence context.
            chain_id, rna_seq, rna_residues = max(rna_chains, key=lambda t: len(t[1]))

            # Iterate HETATM residues across all chains looking for drug-like ligands.
            for chain in model:
                for res in chain:
                    hetflag = res.id[0].strip()          # '' for ATOM, 'W'/'H_xxx' for HETATM
                    if hetflag == "":
                        continue
                    resname = res.get_resname().strip()
                    if resname in _HETATM_BLOCKLIST or _three_to_one_rna(resname):
                        continue
                    n_heavy = sum(1 for _ in res.get_atoms())
                    if n_heavy < 6 or n_heavy > 120:     # too small / unreasonably large
                        continue
                    if _ligand_mw(res) <= 100.0:
                        continue
                    # Correct SMILES by CCD code first; coordinate perception as fallback.
                    smi = _ccd_smiles(resname, ccd_cache) or _ligand_to_smiles(res)
                    if smi is None:
                        n_failed += 1
                        continue
                    contacts = _count_contacts(rna_residues, res, cutoff=4.5)
                    if contacts == 0:                    # not actually bound to the RNA
                        continue
                    rows.append({
                        "pdb_id": pdb.stem,
                        "rna_chain": chain_id,
                        "rna_len": len(rna_seq),
                        "rna_sequence": rna_seq,
                        "ligand_resname": resname,
                        "ligand_smiles": smi,
                        "n_contacts": contacts,
                        "binding_label": -0.5 * contacts,
                    })
                    n_pairs += 1

    df = pd.DataFrame(rows).drop_duplicates(subset=["pdb_id", "ligand_smiles"]).reset_index(drop=True)
    out_csv = cfg.data.hariboss_csv
    df.to_csv(out_csv, index=False)
    ccd_cache_path.write_text(json.dumps(ccd_cache, indent=0))  # persist lookups

    if verbose:
        print("\n========== HARIBOSS preprocessing ==========")
        print(f"  Files parsed            : {n_files}")
        print(f"  Files with RNA chain    : {n_with_rna}")
        print(f"  Valid RNA+ligand pairs  : {len(df)} (raw {n_pairs} before dedup)")
        print(f"  Ligand/parse failures   : {n_failed}")
        print(f"  Saved -> {out_csv}")
        if len(df):
            print("\n  First rows:")
            with pd.option_context("display.max_colwidth", 40, "display.width", 160):
                print(df[["pdb_id", "rna_len", "ligand_resname", "ligand_smiles",
                          "n_contacts", "binding_label"]].head().to_string(index=False))
            print("\n  binding_label distribution:")
            print(df["binding_label"].describe().to_string())
    return df


# --------------------------------------------------------------------------- #
# torch Dataset + splitting (used by the oracle trainer)
# --------------------------------------------------------------------------- #
class HaribossDataset:
    """
    Lightweight map-style dataset over the processed CSV.
    Yields raw strings + label; the encoders tokenise internally, so no torch
    tensors live here except the scalar label.
    """

    def __init__(self, df: pd.DataFrame):
        self.rna = df["rna_sequence"].tolist()
        self.smiles = df["ligand_smiles"].tolist()
        self.labels = df["binding_label"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"rna": self.rna[i], "smiles": self.smiles[i], "label": self.labels[i]}


def load_splits(seed: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the processed CSV and split into train/val/test per cfg.data."""
    seed = cfg.system.seed if seed is None else seed
    df = pd.read_csv(cfg.data.hariboss_csv).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(cfg.data.train_split * n)
    n_val = int(cfg.data.val_split * n)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val:].reset_index(drop=True),
    )


if __name__ == "__main__":
    preprocess_hariboss()
