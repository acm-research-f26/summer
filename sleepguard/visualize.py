"""
visualize.py
Generates all result figures saved to results/
  1. results_comparison.png  — AUROC & F1 bar chart (A vs B)
  2. feature_importance.png  — top-10 feature importances for Model B
  3. per_patient_auroc.png   — per-patient AUROC for both models
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "baseline": "#4C72B0",
    "hrv":      "#DD8452",
    "bg":       "#F7F7F7",
    "grid":     "#E0E0E0",
}


def _style_ax(ax):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_comparison(sum_a, sum_b):
    metrics = ["AUROC", "F1", "Recall", "Precision"]
    vals_a  = [sum_a["auroc_mean"], sum_a["f1_mean"],
               sum_a["recall_mean"], sum_a["prec_mean"]]
    vals_b  = [sum_b["auroc_mean"], sum_b["f1_mean"],
               sum_b["recall_mean"], sum_b["prec_mean"]]
    errs_a  = [sum_a["auroc_std"],  sum_a["f1_std"], 0, 0]
    errs_b  = [sum_b["auroc_std"],  sum_b["f1_std"], 0, 0]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_ax(ax)

    bars_a = ax.bar(x - w/2, vals_a, w, yerr=errs_a, capsize=4,
                    color=COLORS["baseline"], label="Baseline", zorder=3)
    bars_b = ax.bar(x + w/2, vals_b, w, yerr=errs_b, capsize=4,
                    color=COLORS["hrv"],      label="Baseline + HRV", zorder=3)

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SleepGuard — Model Comparison\n(Leave-One-Patient-Out CV)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)

    path = os.path.join(RESULTS_DIR, "results_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_feature_importance(imp_series, top_n=10):
    top = imp_series.head(top_n)
    colors = [COLORS["hrv"] if name.startswith("hrv") or name.startswith("hr_")
              else COLORS["baseline"] for name in top.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_ax(ax)

    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], zorder=3)
    ax.set_xlabel("Feature Importance (mean decrease impurity)", fontsize=10)
    ax.set_title("Feature Importances — Model B (Baseline + HRV)",
                 fontsize=13, fontweight="bold", pad=12)

    patches = [
        mpatches.Patch(color=COLORS["hrv"],      label="HRV feature"),
        mpatches.Patch(color=COLORS["baseline"], label="Baseline feature"),
    ]
    ax.legend(handles=patches, fontsize=9)

    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_per_patient(detail_a, detail_b):
    merged = detail_a[["patient", "auroc"]].merge(
        detail_b[["patient", "auroc"]], on="patient", suffixes=("_base", "_hrv")
    )

    x = np.arange(len(merged))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_ax(ax)

    ax.bar(x - w/2, merged["auroc_base"], w,
           color=COLORS["baseline"], label="Baseline", zorder=3)
    ax.bar(x + w/2, merged["auroc_hrv"],  w,
           color=COLORS["hrv"],      label="Baseline + HRV", zorder=3)

    ax.axhline(0.5, color="#999", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in merged["patient"]], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Per-Patient AUROC — Baseline vs Baseline+HRV",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)

    path = os.path.join(RESULTS_DIR, "per_patient_auroc.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def generate_all(sum_a, sum_b, detail_a, detail_b, imp_series):
    print("\nGenerating figures...")
    plot_comparison(sum_a, sum_b)
    plot_feature_importance(imp_series)
    plot_per_patient(detail_a, detail_b)
    print("All figures saved to results/")