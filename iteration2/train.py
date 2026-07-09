"""
train.py
Train the autoencoder on NORMAL windows only (y_train == 0).
This is the standard unsupervised anomaly detection setup:
the model learns to reconstruct normal actuator command patterns.
Anomalous windows (attacks) will have high reconstruction error at test time.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from preprocess import load_and_preprocess
from model import CommandAutoencoder

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "data", "autoencoder.pt")


def train():
    X_train, y_train, X_test, y_test, n_features = load_and_preprocess()

    # Train ONLY on normal windows -- key to unsupervised anomaly detection
    normal_mask = y_train == 0
    X_normal = torch.tensor(X_train[normal_mask])
    print(f"Training on {len(X_normal)} normal windows")

    loader = DataLoader(
        TensorDataset(X_normal),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = CommandAutoencoder(n_features=n_features, window_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg = total_loss / len(X_normal)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | loss: {avg:.6f}")

    os.makedirs("data", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_features": n_features,
        "window_size": X_train.shape[1],
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, X_train, y_train, X_test, y_test, n_features


if __name__ == "__main__":
    train()
