#!/usr/bin/env python3
"""PHASE 8 — figures and the final results summary."""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.figo import TYPE_MEANINGS, components

FIG = REPO_ROOT / "reports" / "figures"

# Tokens from the data-viz reference palette. The two categorical hues were run
# through scripts/validate_palette.js (light, surface #fcfcfb): all six checks
# PASS, worst adjacent CVD deltaE 24.7, normal-vision 33.6.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # text-primary
INK_2 = "#52514e"        # text-secondary
MUTED = "#898781"        # axis / tick labels
RULE = "#c3c2b7"         # baseline, gridlines
S1 = "#2a78d6"           # categorical slot 1 - blue
S2 = "#eb6834"           # categorical slot 2 - orange


def _style(ax, title="", xlabel="", ylabel=""):
    """Recessive chrome: solid hairline grid and axes, one shade off the surface."""
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=RULE, alpha=0.55, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_distribution(df, cfg):
    """One series, so no legend: the title names it."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(df.percent_intramural, bins=40, color=S1, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(50, color=INK, linewidth=1.2)
    ax.annotate("50% — the FIGO 1/2 and 5/6 boundary",
                xy=(50, ax.get_ylim()[1] * 0.95), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=INK_2, va="top")
    _style(ax, "Percent-intramural across the cohort",
           "percent intramural", "fibroids")
    _save(fig, "percent_intramural_distribution.png")


SHORT = {"0": "pedunculated intracavitary", "1": "submucosal, <50% intramural",
         "2": "submucosal, ≥50% intramural", "3": "intramural, touches endometrium",
         "4": "intramural, no contact", "5": "subserosal, ≥50% intramural",
         "6": "subserosal, <50% intramural", "7": "subserosal pedunculated"}


def fig_types(df, cfg):
    """Horizontal bars: eight descriptive category labels do not fit on an x-axis.

    Vertically they collided into each other at any readable size. Rotating them
    would have worked too, but horizontal bars keep the labels level and give the
    counts a natural place to sit at the bar ends.
    """
    counts = {}
    for t in df.figo:
        for c in components(t, cfg):
            counts[c] = counts.get(c, 0) + 1
    ks = sorted(counts, reverse=True)
    vals = [counts[k] for k in ks]
    decided = {"1", "2", "5", "6"}

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.barh(range(len(ks)), vals, height=0.74,
            color=[S2 if k in decided else S1 for k in ks])
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, str(v), va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{k}  {SHORT[k]}" for k in ks], fontsize=8, color=MUTED)
    ax.set_xlim(0, max(vals) * 1.10)
    handles = [Patch(facecolor=S2, label="decided by the 50% threshold"),
               Patch(facecolor=S1, label="determined by contact alone")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right",
              labelcolor=INK_2)
    _style(ax, "FIGO types  (hybrids counted in both families)", "fibroids", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, alpha=0.55, linewidth=0.6)
    _save(fig, "figo_distribution.png")


def fig_uncertainty(df):
    """Uncertainty vs size and sampling, separating saturated from live fibroids.

    Pooling the two makes the sampling panel *non-monotonic* and contradicts its
    own title: 60% of fibroids are pinned at 0% or 100% intramural, where
    perturbation cannot move the answer, so their intervals are narrow by
    saturation rather than precision. Plotting the pooled median understated the
    uncertainty of every fibroid whose value is actually in play.
    """
    if "ci_width" not in df or df.ci_width.isna().all():
        return
    d = df.dropna(subset=["ci_width"]).copy()
    d["saturated"] = (d.percent_intramural > 99.5) | (d.percent_intramural < 0.5)
    sat, live = d[d.saturated], d[~d.saturated]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axes[0].scatter(sat.volume_mm3 / 1000, sat.ci_width, s=16, alpha=0.25,
                    color=S1, linewidths=0, label=f"pinned at 0% or 100%  (n={len(sat)})")
    axes[0].scatter(live.volume_mm3 / 1000, live.ci_width, s=22, alpha=0.55,
                    color=S2, linewidths=0, label=f"value in play  (n={len(live)})")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK_2)
    _style(axes[0], "Most narrow intervals are saturation, not precision",
           "volume (cm³, log scale)", "CI width (points)")

    by = live.assign(s=live.n_slices_spanned.clip(upper=6)).groupby("s").ci_width.median()
    axes[1].bar([str(int(i)) if i < 6 else "6+" for i in by.index], by.values,
                width=0.78, color=S2)
    rho = live.n_slices_spanned.corr(live.ci_width, method="spearman")
    axes[1].annotate(f"Spearman ρ = {rho:+.2f}", xy=(0.97, 0.94),
                     xycoords="axes fraction", ha="right", fontsize=8, color=INK_2)
    _style(axes[1], "Fibroids whose value is in play: uncertainty falls with sampling",
           "slices spanned", "median CI width (points)")
    _save(fig, "uncertainty.png")


def fig_borderline(df, cfg):
    if "is_borderline" not in df:
        return
    d = df.dropna(subset=["percent_intramural"])
    ok, bad = d[~d.is_borderline], d[d.is_borderline]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.scatter(ok.percent_intramural, ok.volume_mm3 / 1000, s=18, alpha=0.30,
               color=S1, linewidths=0, label=f"determined  (n={len(ok)})")
    ax.scatter(bad.percent_intramural, bad.volume_mm3 / 1000, s=34, alpha=0.90,
               color=S2, linewidths=0.5, edgecolors=SURFACE,
               label=f"ambiguous — needs a second look  (n={len(bad)})")
    ax.axvline(50, color=INK, linewidth=1.2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK_2)
    _style(ax, "Where the FIGO call is not determined by the imaging",
           "percent intramural", "volume (cm³, log scale)")
    _save(fig, "borderline.png")


def main() -> int:
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    for name in ("fibroids_borderline", "fibroids_uncertainty", "fibroids_figo"):
        path = REPO_ROOT / "results" / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"using results/{name}.parquet")
            break
    else:
        print("no results table found; run the pipeline first")
        return 1

    rel = df[df.reliable] if "reliable" in df else df
    fig_distribution(rel, cfg)
    fig_types(rel, cfg)
    fig_uncertainty(rel)
    fig_borderline(rel, cfg)

    figs = sorted(p.name for p in FIG.glob("*.png"))
    print(f"\nwrote {len(figs)} figures to reports/figures/:")
    for f in figs:
        print(f"  {f}")
    qc = sorted((REPO_ROOT / "reports" / "body_qc").glob("*.png"))
    print(f"\nplus {len(qc)} reconstruction QC overlays in reports/body_qc/")
    print("\nPHASE 8: figures written. RESULTS.md is maintained by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
