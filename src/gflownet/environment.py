"""
environment.py
==============
The SELFIES molecule-generation environment for the pHGFN GFlowNet.

Background
----------
**SELFIES** (SELF-referencIng Embedded Strings) is a string representation of
molecules where *every* syntactically valid string maps to a *chemically valid*
molecule. Generating molecules token-by-token in SELFIES therefore guarantees
validity — no wasted samples on broken SMILES, which is ideal for RL/GFlowNet.

Terminology
-----------
* **state**       : the partial molecule = the sequence of SELFIES token indices
                    chosen so far.
* **action**      : the next SELFIES token to append (or the EOS token to stop).
* **trajectory**  : the full sequence of actions from the empty state to a
                    terminal (complete) molecule.
* **terminal**    : reached when EOS is chosen OR `selfies_max_length` is hit.

This module owns the token vocabulary, state<->molecule conversion, and the
RDKit-based ADMET computation used both as a generation filter and in evaluation.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

import selfies as sf

from src.config import cfg

# --- Optional synthetic-accessibility scorer (RDKit Contrib) --------------- #
# sascorer lives in RDKit's Contrib dir; it may or may not ship with the wheel.
_SASCORER = None
try:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer as _sascorer_mod

    _SASCORER = _sascorer_mod
except Exception:
    _SASCORER = None  # SA score will be reported as None and not gate drug-likeness


class SELFIESEnvironment:
    """
    SELFIES generation environment + ADMET utilities.

    The vocabulary is: index 0 = [PAD], index 1 = [EOS], then the SELFIES
    semantic-robust alphabet. Only PAD is never a legal action.
    """

    def __init__(self):
        self.max_length: int = cfg.gflownet.selfies_max_length
        alphabet = sorted(sf.get_semantic_robust_alphabet())
        self.PAD, self.EOS = "[PAD]", "[EOS]"
        self.vocab: list[str] = [self.PAD, self.EOS] + alphabet
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.pad_idx = 0
        self.eos_idx = 1

    # ------------------------------------------------------------------ #
    # Vocabulary
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # State <-> molecule
    # ------------------------------------------------------------------ #
    def state_to_selfies(self, token_indices) -> str:
        """Join chosen token indices into a SELFIES string (PAD/EOS dropped)."""
        toks = []
        for i in token_indices:
            i = int(i)
            if i == self.eos_idx:
                break
            if i == self.pad_idx:
                continue
            toks.append(self.vocab[i])
        return "".join(toks)

    def state_to_mol(self, selfies_str: str) -> tuple[Optional[str], object]:
        """
        Convert a SELFIES string to (canonical SMILES, RDKit mol).
        Returns (None, None) on failure (rare — SELFIES guarantees validity, but
        empty strings / odd edge cases are handled gracefully).
        """
        from rdkit import Chem

        if not selfies_str:
            return None, None
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            return None, None
        if not smiles:
            return None, None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return Chem.MolToSmiles(mol), mol

    def state_to_smiles(self, token_indices) -> Optional[str]:
        """Convenience: token indices -> canonical SMILES (or None)."""
        smiles, _ = self.state_to_mol(self.state_to_selfies(token_indices))
        return smiles

    # ------------------------------------------------------------------ #
    # ADMET
    # ------------------------------------------------------------------ #
    @lru_cache(maxsize=20000)
    def compute_all_admet(self, smiles: str) -> Optional[dict]:
        """
        Compute drug-likeness properties for a molecule.

        Returns dict(MW, LogP, HBD, HBA, QED, SA_score, rotatable_bonds, TPSA,
        is_drug_like). Returns None if the SMILES is invalid. Cached by SMILES.
        """
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        sa = float(_SASCORER.calculateScore(mol)) if _SASCORER is not None else None
        props = {
            "MW": float(Descriptors.MolWt(mol)),
            "LogP": float(Crippen.MolLogP(mol)),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "QED": float(QED.qed(mol)),
            "SA_score": sa,
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            # Lower-bound descriptors (reject trivial fragments).
            "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "n_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "has_carbon": any(a.GetSymbol() == "C" for a in mol.GetAtoms()),
        }
        props["is_drug_like"] = self.is_drug_like(props)
        return props

    def is_drug_like(self, props: dict) -> bool:
        """Apply the Lipinski-plus thresholds (upper AND lower bounds) from `cfg.eval`."""
        e = cfg.eval
        ok = (
            # Ro5 upper bounds
            props["MW"] <= e.mw_max
            and props["LogP"] <= e.logp_max
            and props["HBD"] <= e.hbd_max
            and props["HBA"] <= e.hba_max
            and props["rotatable_bonds"] <= e.rotatable_bonds_max
            and props["QED"] >= e.qed_min
            # Lower bounds — reject trivial fragments
            and props["MW"] >= e.mw_min
            and props.get("n_heavy_atoms", 0) >= e.min_heavy_atoms
            and props.get("n_rings", 0) >= e.min_rings
            and (props.get("has_carbon", True) or not e.require_carbon)
        )
        if props.get("SA_score") is not None:
            ok = ok and props["SA_score"] <= e.sa_max
        return bool(ok)


if __name__ == "__main__":
    env = SELFIESEnvironment()
    print(f"vocab_size = {env.vocab_size} (PAD={env.pad_idx}, EOS={env.eos_idx})")
    print(f"max_length = {env.max_length}")
    print("first 8 action tokens:", env.vocab[2:10])
    # Round-trip a known molecule through SELFIES.
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    enc = sf.encoder(aspirin)
    idx = [env.token_to_idx[t] for t in sf.split_selfies(enc)] + [env.eos_idx]
    smi = env.state_to_smiles(idx)
    print("round-trip aspirin ->", smi)
    print("ADMET:", env.compute_all_admet(smi))
