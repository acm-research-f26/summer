"""
visualize.py  —  SleepGuard v3
Figures (all overwrite same files each run):
  1. model_comparison.png     — AUROC bar chart across 3 models
  2. roc_curves.png           — ROC curves per patient, best model
  3. feature_importance.png   — RF feature importances
  4. ablation.png             — AUROC drop per removed feature
  5. confusion_matrices.png   — per-patient confusion matrix grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GRAY   = "#999999"
BG     = "#F7F7F7"
GRID   = "#E0E0E0"
MODEL_COLORS = [BLUE, ORANGE, GREEN]


def _style(ax, axis="y"):
    ax.set_facecolor(BG)
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_model_comparison(summaries):
    names   = list(summaries.keys())
    aurocs  = [summaries[n]["auroc_mean"] for n in names]
    stds    = [summaries[n]["auroc_std"]  for n in names]
    f1s     = [summaries[n]["f1_mean"]    for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    _style(ax)

    b1 = ax.bar(x - w/2, aurocs, w, yerr=stds, capsize=5,
                color=[MODEL_COLORS[i] for i in range(len(names))],
                label="AUROC", zorder=3, alpha=0.9)
    b2 = ax.bar(x + w/2, f1s,    w,
                color=[MODEL_COLORS[i] for i in range(len(names))],
                label="F1", zorder=3, alpha=0.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — AUROC & F1\n(Leave-One-Patient-Out CV, Threshold Optimized)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_roc_curves(roc_data, model_name, mean_auroc):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    cmap = plt.cm.get_cmap("tab10", len(roc_data))
    for i, fold in enumerate(roc_data):
        ax.plot(fold["fpr"], fold["tpr"], color=cmap(i),
                linewidth=1.5, alpha=0.85, label=f"P{fold['patient']}")
    ax.plot([0,1],[0,1], color=GRAY, linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curves — {model_name}\nMean AUROC = {mean_auroc:.3f}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    path = os.path.join(RESULTS_DIR, "roc_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_feature_importance(imp, top_n=13):
    top    = imp.head(top_n)
    colors = [ORANGE if any(k in n for k in ["pct","modd","cv","accel"])
              else BLUE for n in top.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], zorder=3)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title("Feature Importances — Random Forest",
                 fontsize=12, fontweight="bold", pad=12)
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_ablation(full_auroc, ablation_df):
    top = ablation_df.head(13)
    colors = [ORANGE if row["auroc_drop"] > 0.005 else BLUE
              for _, row in top.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.barh(top["feature"][::-1], top["auroc_drop"][::-1],
            color=colors[::-1], zorder=3)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("AUROC Drop When Feature Removed", fontsize=10)
    ax.set_title(f"Feature Ablation Study (Full AUROC = {full_auroc:.3f})\n"
                 f"Orange = meaningful contribution",
                 fontsize=12, fontweight="bold", pad=12)
    path = os.path.join(RESULTS_DIR, "ablation.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_confusion_matrices(all_details, best_model_name):
    detail = all_details[best_model_name]
    pids   = detail["patient"].tolist()
    n      = len(pids)
    cols   = 3
    rows   = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3.5))
    axes = axes.flatten()

    for i, (_, row) in enumerate(detail.iterrows()):
        ax  = axes[i]
        cm  = np.array([[row["tn"], row["fp"]],
                         [row["fn"], row["tp"]]])
        im  = ax.imshow(cm, interpolation="nearest",
                        cmap=plt.cm.Blues, vmin=0)
        ax.set_title(f"Patient {int(row['patient'])}\n"
                     f"AUROC={row['auroc']:.3f}  thresh={row['threshold']}",
                     fontsize=9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred Normal","Pred Hypo"], fontsize=8)
        ax.set_yticklabels(["True Normal","True Hypo"], fontsize=8)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[r,c] > cm.max()/2 else "black")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {best_model_name}\n(Threshold Optimized for Recall)",
                 fontsize=12, fontweight="bold")
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def generate_all(summaries, all_details, all_roc, imp, full_auroc, ablation_df):
    print("\nGenerating figures...")
    plot_model_comparison(summaries)

    best_model = max(summaries, key=lambda n: summaries[n]["auroc_mean"])
    plot_roc_curves(all_roc[best_model], best_model,
                    summaries[best_model]["auroc_mean"])
    plot_feature_importance(imp)
    plot_ablation(full_auroc, ablation_df)
    plot_confusion_matrices(all_details, best_model)
    print("All figures saved to results/")
