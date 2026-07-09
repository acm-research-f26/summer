"""
evaluate.py
Evaluate the trained autoencoder on the test set.
Sets anomaly threshold at the 95th percentile of normal training errors,
then reports precision, recall, F1, and attack success detection rate.
Also plots reconstruction error distributions for the poster figure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score,
                             roc_auc_score)

from preprocess import load_and_preprocess
from model import CommandAutoencoder

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "data", "autoencoder.pt")
THRESHOLD_PERCENTILE = 95  # set on normal training errors


def load_model():
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    model = CommandAutoencoder(
        n_features=ckpt["n_features"],
        window_size=ckpt["window_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_errors(model, X):
    X_t = torch.tensor(X)
    errors = []
    batch_size = 256
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i: i + batch_size]
        errors.append(model.reconstruction_error(batch).numpy())
    return np.concatenate(errors)


def evaluate():
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    model = load_model()

    # Get reconstruction errors on training normal windows to set threshold
    normal_mask = y_train == 0
    train_errors = get_errors(model, X_train[normal_mask])
    threshold = np.percentile(train_errors, THRESHOLD_PERCENTILE)
    print(f"\nAnomaly threshold ({THRESHOLD_PERCENTILE}th pct of normal errors): {threshold:.6f}")

    # Evaluate on test set
    test_errors = get_errors(model, X_test)
    y_pred = (test_errors > threshold).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, test_errors)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        auc = None

    print(f"\nF1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribution of reconstruction errors by class
    ax = axes[0]
    ax.hist(test_errors[y_test == 0], bins=60, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(test_errors[y_test == 1], bins=60, alpha=0.6, label="Attack", color="tomato")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({THRESHOLD_PERCENTILE}th pct)")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Count")
    ax.set_title("Actuator Command Reconstruction Error by Class")
    ax.legend()

    # Error over time
    ax2 = axes[1]
    ax2.plot(test_errors, alpha=0.7, color="steelblue", linewidth=0.8, label="Recon error")
    ax2.axhline(threshold, color="black", linestyle="--", label="Threshold")
    # shade attack windows
    in_attack = False
    start = None
    for i, label in enumerate(y_test):
        if label == 1 and not in_attack:
            start = i
            in_attack = True
        elif label == 0 and in_attack:
            ax2.axvspan(start, i, alpha=0.2, color="tomato")
            in_attack = False
    if in_attack:
        ax2.axvspan(start, len(y_test), alpha=0.2, color="tomato")
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Reconstruction Error")
    ax2.set_title("Anomaly Detection Over Time (red = attack windows)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("data/results.png", dpi=150)
    print("\nPlot saved to data/results.png")

    return {"f1": f1, "precision": prec, "recall": rec, "auc": auc, "threshold": threshold}


if __name__ == "__main__":
    evaluate()
