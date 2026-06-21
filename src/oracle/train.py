"""
train.py  (oracle / proxy training)
===================================
Train the pHGFN oracle's trainable fusion head on GPU 0.

Two objectives, trained jointly (the fusion trunk is shared):
  * PROXY (primary)   -- predict GNINA's per-conformer score from (molecule, pH).
                         Trained on `gnina_labels.csv`, where each molecule yields
                         two examples: (mol, pH 6.7) -> acidic_score and
                         (mol, pH 7.4) -> neutral_score. This head IS the fast
                         in-loop reward used by the GFlowNet.
  * BINDING (aux)     -- predict the HARIBOSS contact-based binding proxy. Gives
                         the shared trunk extra signal and a general binding filter.

The encoders stay frozen (asserted before training). Mixed precision (fp16),
early stopping on the metric that actually matters for generation: the validation
correlation of the *differential* (acidic - lambda*neutral) with the GNINA truth.

Run (after labeling finishes):
    conda activate phgfn
    python -c "from src.oracle.train import train_oracle; train_oracle()"
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import cfg
from src.oracle.model import pHGFNOracle
from src.utils.seeding import set_seed


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 for degenerate (constant) inputs."""
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_proxy_frames(seed: int):
    """
    Build train/val/test proxy tables from gnina_labels.csv.

    Returns DataFrames with columns [smiles, ph, target] where each input molecule
    contributes two rows (acidic @ target_ph, neutral @ comparison_ph). Splitting
    is done BY MOLECULE so a molecule's two rows never straddle splits.
    """
    df = pd.read_csv(cfg.data.proxy_library_csv)
    df = df[df["ok"] == True].reset_index(drop=True)  # noqa: E712
    if len(df) == 0:
        raise RuntimeError(
            f"No usable rows in {cfg.data.proxy_library_csv}. Run the labeling first."
        )

    rng = np.random.RandomState(seed)
    mols = df["smiles"].tolist()
    perm = rng.permutation(len(mols))
    n_train = int(cfg.data.train_split * len(mols))
    n_val = int(cfg.data.val_split * len(mols))
    split_of = {}
    for i, idx in enumerate(perm):
        split_of[mols[idx]] = (
            "train" if i < n_train else "val" if i < n_train + n_val else "test"
        )

    acidic_ph = cfg.gflownet.target_ph
    neutral_ph = cfg.gflownet.comparison_ph
    rows = {"train": [], "val": [], "test": []}
    for _, r in df.iterrows():
        s = split_of[r["smiles"]]
        rows[s].append({"smiles": r["smiles"], "ph": acidic_ph, "target": r["acidic_score"]})
        rows[s].append({"smiles": r["smiles"], "ph": neutral_ph, "target": r["neutral_score"]})
    return (
        pd.DataFrame(rows["train"]),
        pd.DataFrame(rows["val"]),
        pd.DataFrame(rows["test"]),
        df,  # the raw per-molecule frame, for differential evaluation
        split_of,
    )


def _iter_batches(n: int, batch_size: int, rng: np.random.RandomState, shuffle=True):
    """Yield index arrays for mini-batches."""
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_scores(oracle, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Predict standardised scores for every row of `frame` (real units)."""
    oracle.eval()
    preds = []
    smiles = frame["smiles"].tolist()
    phs = frame["ph"].tolist()
    for start in range(0, len(frame), batch_size):
        sl = slice(start, start + batch_size)
        ph_t = torch.tensor(phs[sl], dtype=torch.float32, device=oracle.device)
        rna = [cfg.data.kras_sequence] * len(phs[sl])
        out = oracle.forward(rna, smiles[sl], ph_t)
        preds.append(oracle._destandardize(out["score"]).float().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


@torch.no_grad()
def _eval_differential(oracle, mol_frame: pd.DataFrame, split_of: dict, which: str,
                       batch_size: int, lam: float) -> tuple[float, float]:
    """
    Correlation + sign-accuracy of predicted vs true GNINA differential on a split.
    True differential uses the stored components; predicted uses the proxy head.
    """
    sub = mol_frame[mol_frame["smiles"].map(lambda s: split_of.get(s) == which)]
    if len(sub) < 2:
        return 0.0, 0.0
    smiles = sub["smiles"].tolist()
    s_acidic = oracle.predict_score(smiles, cfg.gflownet.target_ph).float().cpu().numpy()
    s_neutral = oracle.predict_score(smiles, cfg.gflownet.comparison_ph).float().cpu().numpy()
    pred_diff = s_acidic - lam * s_neutral
    true_diff = (sub["acidic_score"] - lam * sub["neutral_score"]).to_numpy()
    r = _pearson(pred_diff, true_diff)
    sign_acc = float(np.mean(np.sign(pred_diff) == np.sign(true_diff)))
    return r, sign_acc


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_oracle(epochs: int = None, verbose: bool = True) -> pHGFNOracle:
    """Train the oracle (proxy + auxiliary binding head) and save the best checkpoint."""
    set_seed(cfg.system.seed)
    cfg.ensure_dirs()
    device = f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu"
    epochs = epochs or cfg.proxy.epochs
    bs = cfg.proxy.batch_size
    lam = cfg.docking.selectivity_lambda
    rng = np.random.RandomState(cfg.system.seed)

    # ---- data ----
    tr, va, te, mol_frame, split_of = _load_proxy_frames(cfg.system.seed)
    have_binding = Path(cfg.data.hariboss_csv).exists()
    if have_binding:
        hb = pd.read_csv(cfg.data.hariboss_csv)
        hb_mean, hb_std = hb["binding_label"].mean(), hb["binding_label"].std() or 1.0

    # ---- model ----
    oracle = pHGFNOracle(device=device)
    # SAFETY: encoders must be frozen (the spec's mandated check).
    assert all(not p.requires_grad for p in oracle.rna_encoder.parameters()), \
        "RNA encoder is NOT frozen — this wastes compute and degrades performance"
    assert all(not p.requires_grad for p in oracle.mol_encoder.parameters()), \
        "ChemBERTa is NOT frozen — this wastes compute"
    if verbose:
        print("Frozen encoder check: PASSED")
        print(f"proxy rows: train={len(tr)} val={len(va)} test={len(te)} "
              f"| molecules={len(mol_frame)} | binding={'yes' if have_binding else 'no'}")

    # Standardise the GNINA score target on the training split; store on the model.
    t_mean, t_std = float(tr["target"].mean()), float(tr["target"].std() or 1.0)
    oracle.set_score_stats(t_mean, t_std)

    optimizer = torch.optim.AdamW(
        oracle.trainable_parameters(), lr=cfg.proxy.lr, weight_decay=cfg.proxy.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = cfg.proxy.use_fp16 and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tr_smiles, tr_ph, tr_tgt = tr["smiles"].tolist(), tr["ph"].to_numpy(), tr["target"].to_numpy()
    history = {"train_loss": [], "val_r": [], "val_diff_r": []}
    best_diff_r, best_state, patience = -2.0, None, 0

    for epoch in range(1, epochs + 1):
        oracle.fusion.train()
        running = 0.0
        for bidx in _iter_batches(len(tr_smiles), bs, rng):
            smiles = [tr_smiles[i] for i in bidx]
            ph_t = torch.tensor(tr_ph[bidx], dtype=torch.float32, device=device)
            tgt = torch.tensor((tr_tgt[bidx] - t_mean) / t_std, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = oracle.forward([cfg.data.kras_sequence] * len(smiles), smiles, ph_t)
                loss = F.mse_loss(out["score"], tgt)
                # Auxiliary HARIBOSS binding objective (shared trunk).
                if have_binding and len(hb) >= 2:
                    hidx = rng.randint(0, len(hb), size=min(bs, len(hb)))
                    hrows = hb.iloc[hidx]
                    bph = torch.full((len(hrows),), 7.0, device=device)
                    bout = oracle.forward(hrows["rna_sequence"].tolist(),
                                          hrows["ligand_smiles"].tolist(), bph)
                    btgt = torch.tensor(
                        ((hrows["binding_label"].to_numpy() - hb_mean) / hb_std),
                        dtype=torch.float32, device=device,
                    )
                    loss = loss + 0.3 * F.mse_loss(bout["binding"], btgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(oracle.trainable_parameters(), cfg.proxy.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        # ---- validation ----
        val_pred = _eval_scores(oracle, va, bs)
        val_r = _pearson(val_pred, va["target"].to_numpy())
        val_diff_r, val_sign = _eval_differential(oracle, mol_frame, split_of, "val", bs, lam)
        history["train_loss"].append(running / max(1, math.ceil(len(tr_smiles) / bs)))
        history["val_r"].append(val_r)
        history["val_diff_r"].append(val_diff_r)
        if verbose:
            print(f"epoch {epoch:3d} | train_loss {history['train_loss'][-1]:.4f} "
                  f"| val score_r {val_r:.3f} | val diff_r {val_diff_r:.3f} "
                  f"| diff sign-acc {val_sign:.2f}")

        # Early stopping on the differential correlation (the reward-relevant metric).
        if val_diff_r > best_diff_r:
            best_diff_r = val_diff_r
            best_state = {k: v.detach().cpu().clone() for k, v in oracle.fusion.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.proxy.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val diff_r={best_diff_r:.3f})")
                break

    # ---- restore best, evaluate on test, save ----
    if best_state is not None:
        oracle.fusion.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_pred = _eval_scores(oracle, te, bs)
    test_r = _pearson(test_pred, te["target"].to_numpy())
    test_diff_r, test_sign = _eval_differential(oracle, mol_frame, split_of, "test", bs, lam)
    if verbose:
        print(f"\nTEST | score_r {test_r:.3f} | diff_r {test_diff_r:.3f} | diff sign-acc {test_sign:.2f}")

    ckpt = cfg.system.checkpoint_dir / "oracle_best.pt"
    oracle.save(ckpt)
    if verbose:
        print(f"saved best oracle -> {ckpt}")
    _plot_curves(history)
    return oracle


def _plot_curves(history: dict) -> None:
    """Save training-curve figure to results/oracle_training_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Training loss")
    ax[1].plot(history["val_r"], label="val score r")
    ax[1].plot(history["val_diff_r"], label="val differential r")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Pearson r"); ax[1].legend()
    ax[1].set_title("Validation correlation")
    fig.tight_layout()
    out = cfg.system.results_dir / "oracle_training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved curves -> {out}")


if __name__ == "__main__":
    train_oracle()
