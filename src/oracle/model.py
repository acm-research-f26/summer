"""
model.py
========
`pHGFNOracle` — the complete oracle: frozen encoders + the trainable pH fusion head.

Responsibilities
----------------
* `forward(rna, smiles, ph)`         -> head dict (used during training).
* `predict_score(smiles, ph)`        -> GNINA conformer score in REAL units (the proxy).
* `differential_reward(smiles)`      -> score(pH 6.7) - lambda * score(pH 7.4),
                                        the fast structurally-grounded selectivity reward.
* `predict_binding(rna, smiles)`     -> HARIBOSS binding affinity (auxiliary filter).
* save/load, parameter_summary.

Only the fusion head trains; RNA-FM and ChemBERTa stay frozen. Target
standardisation stats (mean/std of the GNINA score) are stored as buffers so
`predict_score` / `differential_reward` always return real, interpretable units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from src.config import cfg
from src.oracle.encoders import MolEncoder, RNAEncoder
from src.oracle.fusion import pHFusionLayer


class pHGFNOracle(nn.Module):
    """Frozen RNA-FM + ChemBERTa encoders with a trainable pH-conditioned fusion head."""

    def __init__(self, device: Union[str, torch.device] = None):
        super().__init__()
        self.device = torch.device(
            device or (f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu")
        )
        oc = cfg.oracle

        # Frozen encoders (load locally, freeze inside).
        self.rna_encoder = RNAEncoder(device=self.device)
        self.mol_encoder = MolEncoder(device=self.device)

        # Trainable fusion head.
        self.fusion = pHFusionLayer(
            rna_dim=oc.rna_embed_dim, mol_dim=oc.mol_embed_dim, ph_dim=oc.ph_embed_dim,
            hidden_dim=oc.fusion_hidden_dim, n_heads=oc.fusion_heads,
            n_layers=oc.fusion_layers, dropout=oc.dropout,
        ).to(self.device)

        # Standardisation stats for the GNINA score target (set by the trainer).
        self.register_buffer("score_mean", torch.zeros(1))
        self.register_buffer("score_std", torch.ones(1))

        # Move buffers (and anything not already placed) onto the target device.
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Core forward (training)
    # ------------------------------------------------------------------ #
    def forward(self, rna_seqs: list[str], smiles: list[str], ph: torch.Tensor) -> dict:
        """
        Encode (frozen, no-grad) then fuse (trainable). `ph` is a [B] float tensor.
        Returns the fusion head dict with standardised 'score' and 'binding'.
        """
        # Encoders are frozen; run under no_grad in full precision.
        with torch.no_grad():
            rna_emb, rna_mask = self.rna_encoder(rna_seqs, return_mask=True)
            mol_emb, mol_mask = self.mol_encoder(smiles, return_mask=True)
        ph = ph.to(self.device)
        return self.fusion(rna_emb, rna_mask, mol_emb, mol_mask, ph)

    # ------------------------------------------------------------------ #
    # Inference helpers (real units)
    # ------------------------------------------------------------------ #
    def _destandardize(self, z: torch.Tensor) -> torch.Tensor:
        """Map a standardised score back to real GNINA-score units."""
        return z * self.score_std + self.score_mean

    @torch.no_grad()
    def predict_score(self, smiles: list[str], ph: float, rna_seq: Optional[str] = None) -> torch.Tensor:
        """
        Predicted GNINA conformer score (real units, higher = better binding) for a
        batch of SMILES at a given pH. RNA defaults to the KRAS target sequence.
        """
        self.eval()
        rna_seq = rna_seq or cfg.data.kras_sequence
        rna_seqs = [rna_seq] * len(smiles)
        ph_t = torch.full((len(smiles),), float(ph), device=self.device)
        out = self.forward(rna_seqs, smiles, ph_t)
        return self._destandardize(out["score"])

    @torch.no_grad()
    def differential_reward(
        self,
        smiles: list[str],
        acidic_ph: Optional[float] = None,
        neutral_ph: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Structurally-grounded selectivity reward (real units):

            reward = score(acidic_ph) - lam * score(neutral_ph)

        Positive => molecule prefers the acidic i-motif (tumour-selective). This is
        the fast proxy for GNINA's differential; the proxy was trained on real
        GNINA docking scores, so the signal is physically grounded.
        """
        acidic_ph = cfg.gflownet.target_ph if acidic_ph is None else acidic_ph
        neutral_ph = cfg.gflownet.comparison_ph if neutral_ph is None else neutral_ph
        lam = cfg.docking.selectivity_lambda if lam is None else lam
        s_acidic = self.predict_score(smiles, acidic_ph)
        s_neutral = self.predict_score(smiles, neutral_ph)
        return s_acidic - lam * s_neutral

    @torch.no_grad()
    def predict_binding(self, rna_seqs: list[str], smiles: list[str], ph: float = 7.0) -> torch.Tensor:
        """Auxiliary HARIBOSS binding-affinity prediction (general fast filter)."""
        self.eval()
        ph_t = torch.full((len(smiles),), float(ph), device=self.device)
        return self.forward(rna_seqs, smiles, ph_t)["binding"]

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        """Only the fusion head trains — feed exactly these to the optimizer."""
        return self.fusion.parameters()

    def set_score_stats(self, mean: float, std: float) -> None:
        """Store target standardisation stats (called by the trainer)."""
        self.score_mean.fill_(float(mean))
        self.score_std.fill_(float(std) if std > 1e-6 else 1.0)

    def parameter_summary(self) -> None:
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("pHGFNOracle parameter summary:")
        print(f"  frozen (encoders) : {frozen:,}")
        print(f"  trainable (fusion): {train:,}")
        print(f"  total             : {frozen + train:,}")

    def save(self, path: Union[str, Path]) -> None:
        """Persist ONLY the trainable fusion head + standardisation stats."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fusion": self.fusion.state_dict(),
                "score_mean": self.score_mean,
                "score_std": self.score_std,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> "pHGFNOracle":
        """Load a saved fusion head + stats (encoders are reloaded from disk)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.fusion.load_state_dict(ckpt["fusion"])
        self.score_mean.copy_(ckpt["score_mean"].to(self.device))
        self.score_std.copy_(ckpt["score_std"].to(self.device))
        return self

    def __repr__(self) -> str:
        return (
            f"pHGFNOracle(device={self.device}, hidden={cfg.oracle.fusion_hidden_dim}, "
            f"layers={cfg.oracle.fusion_layers}, heads={cfg.oracle.fusion_heads})"
        )


if __name__ == "__main__":
    oracle = pHGFNOracle()
    oracle.parameter_summary()
    print(oracle)
    # Frozen-encoder assertions (the spec's safety check).
    assert all(not p.requires_grad for p in oracle.rna_encoder.parameters()), "RNA enc not frozen"
    assert all(not p.requires_grad for p in oracle.mol_encoder.parameters()), "Mol enc not frozen"
    print("Frozen encoder check: PASSED")
    # Tiny end-to-end inference.
    smi = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    print("score@6.7 :", oracle.predict_score(smi, 6.7).tolist())
    print("reward    :", oracle.differential_reward(smi).tolist())
