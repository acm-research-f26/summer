"""
model.py
========
CNN + LSTM hybrid detector, following the Niu et al. (2019) architecture
shape: a 1D-CNN encodes the spatial structure of the measurement vector at
each timestep, and an LSTM consumes the resulting embedding sequence to
capture temporal dynamics across the sliding window. A final MLP head
produces a binary attack / no-attack decision for the current timestep.

TWIST: `use_action=True` adds a small side-MLP branch that encodes the
GridLock LLM-agent's proposed action at each timestep and fuses it with
the CNN measurement embedding before the LSTM. This lets the model learn
that a stealthy FDIA -- invisible in the raw measurement residual -- still
shows up as an inconsistency between "what the sensors say" and "what the
agent decided to do about it".
"""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """1D-CNN over the measurement feature axis at a single timestep.

    Input:  (B, meas_dim)   -- treated as a length-`meas_dim` 1-channel signal
    Output: (B, embed_dim)
    """

    def __init__(self, meas_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x):
        # x: (B, meas_dim) -> (B, 1, meas_dim)
        x = x.unsqueeze(1)
        h = self.net(x).squeeze(-1)  # (B, 32)
        return self.proj(h)          # (B, embed_dim)


class ActionEncoder(nn.Module):
    """Small MLP encoder for the proposed-action feature vector."""

    def __init__(self, action_dim, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, a):
        return self.net(a)


class CNNLSTMDetector(nn.Module):
    """
    Args:
      meas_dim:   number of measurement features per timestep
      action_dim: number of proposed-action features per timestep
      use_action: if False, the model only ever sees measurements
                  (reproduces the paper's original setup, used as the
                  ablation baseline in evaluate.py)
    """

    def __init__(self, meas_dim, action_dim, cnn_embed=64, action_embed=16,
                 lstm_hidden=64, lstm_layers=1, use_action=True, dropout=0.2):
        super().__init__()
        self.use_action = use_action
        self.cnn = CNNEncoder(meas_dim, cnn_embed)
        fused_dim = cnn_embed
        if use_action:
            self.action_enc = ActionEncoder(action_dim, action_embed)
            fused_dim += action_embed

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x_meas, x_action=None):
        # x_meas: (B, T, meas_dim), x_action: (B, T, action_dim)
        B, T, F = x_meas.shape
        cnn_in = x_meas.reshape(B * T, F)
        cnn_out = self.cnn(cnn_in).reshape(B, T, -1)

        if self.use_action:
            if x_action is None:
                raise ValueError("use_action=True but no action tensor was given")
            act_out = self.action_enc(x_action.reshape(B * T, -1)).reshape(B, T, -1)
            fused = torch.cat([cnn_out, act_out], dim=-1)
        else:
            fused = cnn_out

        lstm_out, _ = self.lstm(fused)
        last = lstm_out[:, -1, :]  # representation at the current timestep
        logits = self.head(last)
        return logits


def build_model(meas_dim, action_dim, use_action=True):
    return CNNLSTMDetector(meas_dim=meas_dim, action_dim=action_dim, use_action=use_action)
