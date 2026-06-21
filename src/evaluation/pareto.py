"""
pareto.py
=========
Pareto-frontier analysis of pHGFN candidates.

We optimise two objectives simultaneously:
  1. Selectivity differential  (proxy GNINA: acidic - lambda*neutral) -> maximise.
     Higher = more tumour-selective.
  2. QED drug-likeness          -> maximise. Higher = more viable as a drug.

A molecule is **Pareto-optimal** if no other molecule beats it on BOTH objectives.
The frontier is the set of best achievable trade-offs, letting a chemist pick by
whichever property matters most for their application.

Reads `results/candidates_filtered.csv`; writes `results/pareto_optimal.csv` and
`results/pareto_frontier.png`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """
    Boolean mask of Pareto-optimal rows for a MAXIMISATION problem.
    objectives: [N, K] array (here K=2: selectivity, QED).
    """
    n = objectives.shape[0]
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        # i is dominated if some j is >= in all objectives and > in at least one.
        dominates = np.all(objectives >= objectives[i], axis=1) & np.any(
            objectives > objectives[i], axis=1
        )
        if np.any(dominates):
            is_optimal[i] = False
    return is_optimal


def compute_pareto_frontier(
    filtered_csv: Path | None = None,
    out_csv: Path | None = None,
    out_png: Path | None = None,
    sel_col: str = "differential",
    qed_col: str = "QED",
    verbose: bool = True,
) -> pd.DataFrame:
    """Find Pareto-optimal molecules on (selectivity, QED) and plot the frontier."""
    filtered_csv = Path(filtered_csv or (cfg.system.results_dir / "candidates_filtered.csv"))
    out_csv = Path(out_csv or (cfg.system.results_dir / "pareto_optimal.csv"))
    out_png = Path(out_png or (cfg.system.results_dir / "pareto_frontier.png"))

    df = pd.read_csv(filtered_csv)
    if sel_col not in df or qed_col not in df or len(df) == 0:
        raise ValueError(
            f"Need columns '{sel_col}' and '{qed_col}' in {filtered_csv} (got {list(df.columns)})."
        )

    obj = df[[sel_col, qed_col]].to_numpy(dtype=float)
    mask = pareto_mask(obj)
    pareto = df[mask].sort_values(sel_col, ascending=False).reset_index(drop=True)
    pareto.to_csv(out_csv, index=False)

    _plot(df, mask, sel_col, qed_col, out_png)
    if verbose:
        print("========== Pareto frontier ==========")
        print(f"  candidates        : {len(df)}")
        print(f"  Pareto-optimal    : {len(pareto)}")
        print(f"  saved -> {out_csv}")
        print(f"  saved -> {out_png}")
        if len(pareto):
            print("\n  top Pareto molecules (by selectivity):")
            cols = [c for c in ["smiles", sel_col, qed_col] if c in pareto.columns]
            print(pareto[cols].head(8).to_string(index=False))
    return pareto


def _plot(df, mask, sel_col, qed_col, out_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df[sel_col][~mask], df[qed_col][~mask], s=14, c="lightgray",
               label="dominated", alpha=0.6)
    pareto = df[mask].sort_values(sel_col)
    ax.scatter(pareto[sel_col], pareto[qed_col], s=42, c="crimson",
               edgecolors="k", label="Pareto-optimal", zorder=3)
    ax.plot(pareto[sel_col], pareto[qed_col], c="crimson", lw=1, alpha=0.5, zorder=2)
    ax.set_xlabel("Selectivity differential (proxy GNINA: acidic - λ·neutral)")
    ax.set_ylabel("QED drug-likeness")
    ax.set_title("pHGFN candidate Pareto frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    compute_pareto_frontier()
