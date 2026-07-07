"""
visualize.py  —  SleepGuard v2
Saves 3 figures to results/ (always overwrites same files):
  1. per_patient_auroc.png   — per-patient AUROC bar chart
  2. feature_importance.png  — top feature importances
  3. roc_curves.png          — ROC curve per patient overlaid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BLUE  = "#4C72B0"
ORANGE= "#DD8452"
GRAY  = "#999999"
BG    = "#F7F7F7"
GRID  = "#E0E0E0"

def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_per_patient(detail):
    fig, ax = plt.subplots(figsize=(9, 5))
    _style(ax)
    pids = [str(p) for p in detail["patient"]]
    x    = np.arange(len(pids))
    bars = ax.bar(x, detail["auroc"], color=BLUE, zorder=3, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.2, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in detail["patient"]], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Per-Patient AUROC — Glucose Feature Model\n(Leave-One-Patient-Out CV)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    path = os.path.join(RESULTS_DIR, "per_patient_auroc.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_feature_importance(imp, top_n=13):
    top    = imp.head(top_n)
    colors = [ORANGE if "pct" in n or n in ("g_min","g_last","g_cv","g_modd")
              else BLUE for n in top.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], zorder=3)
    ax.set_xlabel("Feature Importance (mean decrease impurity)", fontsize=10)
    ax.set_title("Feature Importances — Glucose Feature Model",
                 fontsize=13, fontweight="bold", pad=12)
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_roc_curves(roc_data, mean_auroc):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    cmap = plt.cm.get_cmap("tab10", len(roc_data))
    for i, fold in enumerate(roc_data):
        ax.plot(fold["fpr"], fold["tpr"],
                color=cmap(i), linewidth=1.5, alpha=0.8,
                label=f"P{fold['patient']}")
    ax.plot([0,1],[0,1], color=GRAY, linestyle="--", linewidth=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves per Patient\nMean AUROC = {mean_auroc:.3f}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    path = os.path.join(RESULTS_DIR, "roc_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def generate_all(summary, detail, roc_data, imp):
    print("\nGenerating figures...")
    plot_per_patient(detail)
    plot_feature_importance(imp)
    plot_roc_curves(roc_data, summary["auroc_mean"])
    print("All figures saved to results/")
