"""
evaluate.py
===========
Loads the held-out test split and both trained checkpoints (with_action /
without_action), reports precision / recall / F1 / confusion matrix for
each, and prints a side-by-side comparison table showing what the
proposed-action feature buys us.
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_set():
    d = np.load("data/test_split.npz")
    return d["X_meas"], d["X_action"], d["y"]


def load_model(tag):
    ckpt = torch.load(f"checkpoints/{tag}.pt", map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["meas_dim"], ckpt["action_dim"], use_action=ckpt["use_action"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["use_action"]


@torch.no_grad()
def predict(model, use_action, X_meas, X_action, batch_size=128):
    preds = []
    for i in range(0, len(X_meas), batch_size):
        xm = torch.from_numpy(X_meas[i:i + batch_size]).to(DEVICE)
        xa = torch.from_numpy(X_action[i:i + batch_size]).to(DEVICE)
        logits = model(xm, xa if use_action else None)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def report(tag, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== {tag} ===")
    print(f"precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")
    print("confusion matrix [rows=true, cols=pred] (0=normal, 1=attack):")
    print(cm)
    return {"precision": p, "recall": r, "f1": f1, "cm": cm}


def main():
    X_meas, X_action, y = load_test_set()
    print(f"[evaluate] test set: {len(y)} windows, positive rate={y.mean():.3f}")

    results = {}
    for tag in ["with_action", "without_action"]:
        model, use_action = load_model(tag)
        y_pred = predict(model, use_action, X_meas, X_action)
        results[tag] = report(tag, y, y_pred)

    print("\n=== comparison: does the proposed-action feature help? ===")
    header = f"{'metric':<12}{'without_action':>16}{'with_action':>16}{'delta':>10}"
    print(header)
    for metric in ["precision", "recall", "f1"]:
        wo = results["without_action"][metric]
        w = results["with_action"][metric]
        print(f"{metric:<12}{wo:>16.4f}{w:>16.4f}{(w - wo):>+10.4f}")


if __name__ == "__main__":
    main()
