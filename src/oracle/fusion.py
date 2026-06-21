"""
fusion.py
=========
The **only trainable part** of the pHGFN oracle: a pH-conditioned cross-attention
head that fuses a frozen RNA embedding, a frozen molecule embedding, and a pH
value into binding predictions.

Key idea — pH is the conformer switch
-------------------------------------
RNA-FM embeds *sequence*, so the acidic (i-motif) and neutral (unfolded) states
share an identical RNA embedding. We make pH carry the conformer information:

  * at pH 6.7 the head is trained to reproduce GNINA's score against the ACIDIC
    i-motif conformer;
  * at pH 7.4 it reproduces GNINA's score against the NEUTRAL conformer.

So the differential reward is just the head queried at two pH values:
    reward(mol) = score(mol, pH=6.7) - lambda * score(mol, pH=7.4)
Because the targets come from real 3D GNINA docking, pH now means something
concrete (which conformer), unlike the original sequence-only design.

Architecture (per `OracleConfig`)
---------------------------------
  1. pH scalar -> 2-layer MLP -> pH embedding (64) -> projected to hidden (512).
  2. RNA (640) and molecule (768) embeddings are each projected to hidden (512).
  3. Context = [molecule tokens ; pH token]  (the pH token rides alongside the
     molecule so attention can modulate binding by pH).
  4. `n_layers` cross-attention blocks: RNA queries attend over the context
     (molecule+pH), each block = MHA + residual/LN + FFN + residual/LN.
  5. Masked mean-pool over RNA positions -> a single 512-d complex embedding.
  6. Two MLP heads on the pooled vector:
        score_head   -> predicted GNINA conformer score (the proxy / reward).
        binding_head -> predicted HARIBOSS binding affinity (auxiliary filter).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import cfg


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over sequence dim using a [B, L] mask (1 = valid). x: [B, L, D]."""
    m = mask.unsqueeze(-1).to(x.dtype)          # [B, L, 1]
    summed = (x * m).sum(dim=1)                  # [B, D]
    counts = m.sum(dim=1).clamp_min(1.0)         # [B, 1]
    return summed / counts


class _CrossAttnBlock(nn.Module):
    """One cross-attention block: RNA (query) attends over context (key/value)."""

    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, query, context, key_padding_mask):
        # key_padding_mask: [B, Lc] with True at padded positions (ignored).
        attn_out, _ = self.attn(
            query, context, context, key_padding_mask=key_padding_mask, need_weights=False
        )
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query


class pHFusionLayer(nn.Module):
    """
    The trainable fusion head. See module docstring for the architecture.

    forward(rna_emb, rna_mask, mol_emb, mol_mask, ph) -> dict with:
        'score'   [B]  predicted GNINA conformer score (proxy/reward target)
        'binding' [B]  predicted HARIBOSS binding affinity (auxiliary)
        'pooled'  [B, hidden]  the fused complex embedding (for reuse/inspection)
    """

    def __init__(
        self,
        rna_dim: int = 640,
        mol_dim: int = 768,
        ph_dim: int = 64,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # (1) pH scalar -> embedding -> hidden-space token.
        self.ph_mlp = nn.Sequential(
            nn.Linear(1, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim)
        )
        self.ph_proj = nn.Linear(ph_dim, hidden_dim)

        # (2) project frozen embeddings into the shared hidden space.
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.mol_proj = nn.Linear(mol_dim, hidden_dim)
        self.in_drop = nn.Dropout(dropout)

        # (4) stacked cross-attention blocks.
        self.blocks = nn.ModuleList(
            [_CrossAttnBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )

        # (6) two prediction heads (3-layer MLPs).
        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, 1),
            )

        self.score_head = _head()     # GNINA conformer score (proxy / reward)
        self.binding_head = _head()   # HARIBOSS binding affinity (auxiliary)

    def forward(
        self,
        rna_emb: torch.Tensor,    # [B, Lr, rna_dim]
        rna_mask: torch.Tensor,   # [B, Lr]
        mol_emb: torch.Tensor,    # [B, Lm, mol_dim]
        mol_mask: torch.Tensor,   # [B, Lm]
        ph: torch.Tensor,         # [B] or [B, 1] float
    ) -> dict:
        B = rna_emb.size(0)

        # (1) pH token.
        ph = ph.view(B, 1).to(rna_emb.dtype)
        ph_token = self.ph_proj(self.ph_mlp(ph)).unsqueeze(1)        # [B, 1, hidden]

        # (2) project embeddings.
        rna_h = self.in_drop(self.rna_proj(rna_emb))                 # [B, Lr, hidden]
        mol_h = self.in_drop(self.mol_proj(mol_emb))                 # [B, Lm, hidden]

        # (3) context = molecule tokens + pH token.
        context = torch.cat([mol_h, ph_token], dim=1)               # [B, Lm+1, hidden]
        ones = torch.ones(B, 1, device=mol_mask.device, dtype=mol_mask.dtype)
        context_mask = torch.cat([mol_mask, ones], dim=1)           # [B, Lm+1]
        key_padding_mask = context_mask == 0                        # True = ignore

        # (4) RNA attends over context through the stacked blocks.
        query = rna_h
        for block in self.blocks:
            query = block(query, context, key_padding_mask)

        # (5) pool over valid RNA positions.
        pooled = _masked_mean(query, rna_mask)                      # [B, hidden]

        # (6) heads.
        return {
            "score": self.score_head(pooled).squeeze(-1),
            "binding": self.binding_head(pooled).squeeze(-1),
            "pooled": pooled,
        }


# --------------------------------------------------------------------------- #
# Smoke test: parameter count + a forward pass on random tensors.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    oc = cfg.oracle
    fusion = pHFusionLayer(
        rna_dim=oc.rna_embed_dim, mol_dim=oc.mol_embed_dim, ph_dim=oc.ph_embed_dim,
        hidden_dim=oc.fusion_hidden_dim, n_heads=oc.fusion_heads,
        n_layers=oc.fusion_layers, dropout=oc.dropout,
    )
    n_trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    B, Lr, Lm = 4, 20, 30
    rna = torch.randn(B, Lr, oc.rna_embed_dim)
    mol = torch.randn(B, Lm, oc.mol_embed_dim)
    rna_mask = torch.ones(B, Lr)
    mol_mask = torch.ones(B, Lm)
    ph = torch.tensor([6.7, 7.4, 6.7, 7.4])
    out = fusion(rna, rna_mask, mol, mol_mask, ph)
    print("score", tuple(out["score"].shape), "| binding", tuple(out["binding"].shape),
          "| pooled", tuple(out["pooled"].shape))
