"""
train.py
========
Supervised training of the CNN+LSTM FDIA detector on data/dataset.npz.

Trains two variants so evaluate.py can compare them:
  - with_action:    model sees (grid state, proposed action) jointly [the twist]
  - without_action:  model sees grid state only [reproduces the paper's setup]

Usage:
  python train.py --variant with_action
  python train.py --variant without_action
  python train.py --variant both        (default)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
VAL_FRAC = 0.15
TEST_FRAC = 0.15


class WindowDataset(Dataset):
    def __init__(self, X_meas, X_action, y):
        self.X_meas = torch.from_numpy(X_meas)
        self.X_action = torch.from_numpy(X_action)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_meas[idx], self.X_action[idx], self.y[idx]


def load_splits():
    d = np.load("data/dataset.npz")
    X_meas, X_action, y = d["X_meas"], d["X_action"], d["y"]
    meas_dim, action_dim = int(d["meas_dim"]), int(d["action_dim"])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * TEST_FRAC)
    n_val = int(len(y) * VAL_FRAC)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    splits = {}
    for name, ind in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = (X_meas[ind], X_action[ind], y[ind])
    return splits, meas_dim, action_dim


def run_epoch(model, loader, optimizer, criterion, use_action, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xm, xa, yb in loader:
            xm, xa, yb = xm.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
            logits = model(xm, xa if use_action else None)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
    return total_loss / n


def train_variant(use_action, splits, meas_dim, action_dim, tag):
    torch.manual_seed(SEED)
    train_ds = WindowDataset(*splits["train"])
    val_ds = WindowDataset(*splits["val"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(meas_dim, action_dim, use_action=use_action).to(DEVICE)

    # class-imbalance-aware weighting (attacks are the minority class)
    y_train = splits["train"][2]
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, use_action, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, use_action, train=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"[{tag}] epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{tag}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "meas_dim": meas_dim,
        "action_dim": action_dim,
        "use_action": use_action,
    }, ckpt_path)
    print(f"[{tag}] saved {ckpt_path} (best val_loss={best_val:.4f})")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["with_action", "without_action", "both"], default="both")
    args = parser.parse_args()

    splits, meas_dim, action_dim = load_splits()
    print(f"[train] device={DEVICE}  meas_dim={meas_dim}  action_dim={action_dim}")
    print(f"[train] split sizes: "
          f"train={len(splits['train'][2])} val={len(splits['val'][2])} test={len(splits['test'][2])}")

    # persist the test split so evaluate.py uses an identical held-out set for both variants
    os.makedirs("data", exist_ok=True)
    np.savez("data/test_split.npz",
             X_meas=splits["test"][0], X_action=splits["test"][1], y=splits["test"][2])

    if args.variant in ("with_action", "both"):
        train_variant(True, splits, meas_dim, action_dim, "with_action")
    if args.variant in ("without_action", "both"):
        train_variant(False, splits, meas_dim, action_dim, "without_action")


if __name__ == "__main__":
    main()
