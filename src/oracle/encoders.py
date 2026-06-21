"""
encoders.py
===========
Frozen pre-trained encoders for the pHGFN oracle.

Two wrappers, both with **all weights frozen** (no gradients, eval mode):

  RNAEncoder  -- RNA-FM (multimolecule.RnaFmModel), 640-d per-residue embeddings.
  MolEncoder  -- ChemBERTa (transformers RoBERTa), 768-d per-token embeddings.

Both load strictly from the LOCAL on-disk checkpoints (no internet), handle
batching / padding / truncation internally, and return per-position embeddings
plus the attention mask so the trainable fusion head can pool correctly.

Why frozen: the pre-trained encoders already capture RNA / molecule structure.
We only train the small fusion head on top. Freezing saves compute and avoids
overfitting the tiny HARIBOSS dataset.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from src.config import cfg


class _FrozenEncoder(nn.Module):
    """Shared plumbing for the two frozen encoders (freeze + verify utilities)."""

    embed_dim: int

    def __init__(self, device: Union[str, torch.device]):
        super().__init__()
        self.device = torch.device(device)

    def _freeze(self) -> None:
        """Disable gradients and switch to eval mode for every sub-parameter."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def verify_frozen(self) -> None:
        """Assert that no parameter is trainable. Raises AssertionError otherwise."""
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert n_trainable == 0, f"{type(self).__name__} has {n_trainable} trainable params!"
        total = sum(p.numel() for p in self.parameters())
        print(f"  {type(self).__name__}: frozen OK ({total:,} params, 0 trainable)")

    def train(self, mode: bool = True):  # noqa: D401
        """Override: a frozen encoder must never leave eval mode."""
        return super().train(False)


class RNAEncoder(_FrozenEncoder):
    """
    Wraps RNA-FM. `forward(sequences)` -> [batch, seq_len, 640].

    Args:
        device: where to place the model and run inference (e.g. 'cuda:0').
    """

    def __init__(self, device: Union[str, torch.device] = "cuda:0"):
        super().__init__(device)
        from multimolecule import RnaFmModel, RnaTokenizer

        self.tokenizer = RnaTokenizer.from_pretrained(
            str(cfg.data.rnafm_dir), local_files_only=True
        )
        self.model = RnaFmModel.from_pretrained(
            str(cfg.data.rnafm_dir), local_files_only=True
        ).to(self.device)
        self.embed_dim = cfg.oracle.rna_embed_dim  # 640
        self._freeze()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"RNA-FM loaded from local. Parameters: {n:,}. Frozen: True.")

    @torch.no_grad()
    def forward(self, sequences: list[str], return_mask: bool = False):
        """
        Embed a batch of RNA sequences.

        Returns [batch, seq_len, 640]; if `return_mask`, also the [batch, seq_len]
        attention mask (1 = real residue, 0 = padding).
        """
        enc = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.oracle.max_rna_len,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state
        if return_mask:
            return out, enc["attention_mask"]
        return out


class MolEncoder(_FrozenEncoder):
    """
    Wraps ChemBERTa. `forward(smiles_list)` -> [batch, seq_len, 768].

    Args:
        device: where to place the model and run inference (e.g. 'cuda:0').
    """

    def __init__(self, device: Union[str, torch.device] = "cuda:0"):
        super().__init__(device)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.data.chemberta_dir), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(cfg.data.chemberta_dir), local_files_only=True
        ).to(self.device)
        self.embed_dim = cfg.oracle.mol_embed_dim  # 768
        self._freeze()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"ChemBERTa loaded from local. Parameters: {n:,}. Frozen: True.")

    @torch.no_grad()
    def forward(self, smiles_list: list[str], return_mask: bool = False):
        """
        Embed a batch of SMILES.

        Returns [batch, seq_len, 768]; if `return_mask`, also the [batch, seq_len]
        attention mask (1 = real token, 0 = padding).
        """
        enc = self.tokenizer(
            list(smiles_list),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.oracle.max_mol_len,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state
        if return_mask:
            return out, enc["attention_mask"]
        return out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dev = f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu"
    rna_enc = RNAEncoder(device=dev)
    mol_enc = MolEncoder(device=dev)

    rna_out, rna_mask = rna_enc(["AUCGAUCGAUCG", "CCCCGCCCC"], return_mask=True)
    print("RNA encoder output:", tuple(rna_out.shape), "mask", tuple(rna_mask.shape))

    mol_out, mol_mask = mol_enc(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"], return_mask=True)
    print("Mol encoder output:", tuple(mol_out.shape), "mask", tuple(mol_mask.shape))

    rna_enc.verify_frozen()
    mol_enc.verify_frozen()
    print("All encoder tests passed")
