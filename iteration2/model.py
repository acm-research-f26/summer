"""
model.py
Autoencoder for ICS actuator command anomaly detection.

Architecture: flatten the window -> encode to bottleneck -> decode back.
Anomaly score = MSE reconstruction error.
Normal behavior reconstructs well (low error).
Anomalous commands deviate from learned normal patterns (high error).
"""

import torch
import torch.nn as nn


class CommandAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int = 30, latent_dim: int = 16):
        super().__init__()
        input_dim = n_features * window_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # features are MinMax scaled to [0,1]
        )

    def forward(self, x):
        # x: (batch, window_size, n_features)
        batch = x.size(0)
        flat = x.view(batch, -1)
        z = self.encoder(flat)
        out = self.decoder(z)
        return out.view(batch, x.size(1), x.size(2))

    def reconstruction_error(self, x):
        """Per-sample MSE reconstruction error -- this is the anomaly score."""
        with torch.no_grad():
            recon = self.forward(x)
            error = ((recon - x) ** 2).mean(dim=(1, 2))
        return error
