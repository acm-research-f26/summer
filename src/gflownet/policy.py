"""
policy.py
=========
The pH-conditioned transformer policy for the pHGFN GFlowNet.

Role
----
Given a partial SELFIES sequence and a target pH, output a distribution over the
next SELFIES token (including EOS). Conditioning on pH lets the SAME network
design *for* the acidic i-motif (pH 6.7) versus *against* the neutral state.

Architecture
------------
A decoder-style (causal) transformer language model over SELFIES tokens:

  1. The pH scalar -> 2-layer MLP -> a 512-d "pH conditioning token" placed at
     position 0. Every generated token attends back to it through the causal
     mask, so pH influences every decision.
  2. SELFIES token embeddings follow the pH token.
  3. Sinusoidal positional encodings are added.
  4. `policy_layers` causal Transformer-encoder layers (self-attention).
  5. A linear head maps each position's hidden state to logits over the vocab;
     position i predicts token i, so one teacher-forced pass scores a whole
     trajectory's forward log-prob efficiently.

Also holds a learned scalar **log_Z** (log partition function) required by the
Trajectory Balance objective.

Design note vs. the original spec: the spec described reading logits from the
fixed pH token (position 0). That makes the next-token distribution independent
of how much has been generated, which is incorrect for autoregressive sampling.
We use the standard causal-LM read-out (position i -> token i) with pH as the
position-0 conditioning token, which preserves the "pH influences every token"
intent while being correct and efficient.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg
from src.gflownet.environment import SELFIESEnvironment


class _SinusoidalPositional(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class pHConditionedPolicy(nn.Module):
    """pH-conditioned causal transformer over SELFIES tokens (+ learned log_Z)."""

    def __init__(self, env: Optional[SELFIESEnvironment] = None):
        super().__init__()
        self.env = env or SELFIESEnvironment()
        gc = cfg.gflownet
        H = gc.policy_hidden_dim
        self.hidden = H
        self.max_length = gc.selfies_max_length
        self.vocab_size = self.env.vocab_size
        self.pad_idx = self.env.pad_idx
        self.eos_idx = self.env.eos_idx

        self.token_embed = nn.Embedding(self.vocab_size, H, padding_idx=self.pad_idx)
        self.ph_mlp = nn.Sequential(
            nn.Linear(1, gc.ph_embed_dim), nn.ReLU(), nn.Linear(gc.ph_embed_dim, H)
        )
        self.pos_enc = _SinusoidalPositional(H, max_len=self.max_length + 2)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=gc.policy_heads, dim_feedforward=4 * H,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=gc.policy_layers)
        self.head = nn.Linear(H, self.vocab_size)

        # Learned log partition function for Trajectory Balance.
        self.log_Z = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ph_token(self, ph: torch.Tensor) -> torch.Tensor:
        """[B] pH -> [B, 1, H] conditioning token."""
        return self.ph_mlp(ph.view(-1, 1).float()).unsqueeze(1)

    def _logits(self, tokens: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
        """
        Causal forward pass.
        tokens: [B, L] token indices already generated (may be empty, L=0).
        Returns logits [B, L+1, vocab]; position i predicts the (i-th) next token.
        PAD logits are masked to -inf so PAD is never sampled.
        """
        B = ph.size(0)
        ph_tok = self._ph_token(ph)                                 # [B,1,H]
        if tokens.numel() > 0 and tokens.size(1) > 0:
            tok_emb = self.token_embed(tokens)                      # [B,L,H]
            x = torch.cat([ph_tok, tok_emb], dim=1)                 # [B,L+1,H]
        else:
            x = ph_tok                                              # [B,1,H]
        x = self.pos_enc(x)
        Lp = x.size(1)
        causal = torch.triu(
            torch.full((Lp, Lp), float("-inf"), device=x.device), diagonal=1
        )
        h = self.transformer(x, mask=causal)
        logits = self.head(h)                                       # [B,Lp,vocab]
        logits[..., self.pad_idx] = float("-inf")                   # never emit PAD
        return logits

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, batch_size: int, ph: float, device=None, temperature: float = 1.0):
        """
        Autoregressively sample `batch_size` trajectories at the given pH.

        Returns:
          actions    : LongTensor [B, T] padded with PAD (T = longest trajectory)
          lengths    : LongTensor [B] number of real actions (incl. EOS)
          smiles     : list[str|None] decoded molecules
        Forward log-probs are recomputed (with grad) by `trajectory_log_prob`.
        """
        device = device or next(self.parameters()).device
        ph_t = torch.full((batch_size,), float(ph), device=device)
        seqs = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cur = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        for _ in range(self.max_length):
            logits = self._logits(cur, ph_t)[:, -1, :]              # next-token logits
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(1)            # [B]
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_idx), nxt)
            for b in range(batch_size):
                if not finished[b]:
                    seqs[b].append(int(nxt[b]))
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            finished = finished | (nxt == self.eos_idx)
            if bool(finished.all()):
                break

        lengths = torch.tensor([len(s) for s in seqs], device=device)
        T = int(lengths.max().item()) if len(seqs) else 0
        actions = torch.full((batch_size, max(T, 1)), self.pad_idx, dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            if s:
                actions[b, : len(s)] = torch.tensor(s, device=device)
        smiles = [self.env.state_to_smiles(s) for s in seqs]
        return actions, lengths, smiles

    # ------------------------------------------------------------------ #
    # Trajectory forward log-prob (teacher forced, differentiable)
    # ------------------------------------------------------------------ #
    def trajectory_log_prob(self, actions: torch.Tensor, lengths: torch.Tensor, ph: float) -> torch.Tensor:
        """
        Sum of log P_forward(a_t | s_t) for each trajectory (differentiable).

        actions: [B, T] padded; lengths: [B]; ph: scalar.
        One causal pass scores every step: logits[:, i] predicts actions[:, i].
        """
        device = actions.device
        B, T = actions.shape
        ph_t = torch.full((B,), float(ph), device=device)
        logits = self._logits(actions, ph_t)[:, :T, :]             # [B,T,vocab]
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B,T]
        # Zero out padding positions. Use `where` (not multiply): padded actions are
        # PAD whose logit is -inf, and -inf * 0 = NaN, so masking must replace, not scale.
        step = torch.arange(T, device=device).unsqueeze(0)          # [1,T]
        valid = step < lengths.unsqueeze(1)                         # [B,T]
        return torch.where(valid, chosen, torch.zeros_like(chosen)).sum(dim=1)  # [B]


if __name__ == "__main__":
    dev = f"cuda:{cfg.system.secondary_gpu}" if torch.cuda.is_available() else "cpu"
    policy = pHConditionedPolicy().to(dev)
    n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"policy trainable params: {n:,} | vocab={policy.vocab_size} | log_Z={policy.log_Z.item():.3f}")
    actions, lengths, smiles = policy.sample(batch_size=6, ph=cfg.gflownet.target_ph, device=dev)
    print("sampled lengths:", lengths.tolist())
    valid = [s for s in smiles if s]
    print(f"valid molecules: {len(valid)}/6")
    for s in valid[:3]:
        print("  ", s)
    lp = policy.trajectory_log_prob(actions, lengths, cfg.gflownet.target_ph)
    print("forward log-probs:", [round(v, 2) for v in lp.tolist()])
