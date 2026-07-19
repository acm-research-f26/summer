"""Compare the v3 (hedged) and v4 (calibrated) vision prompts on shared screens:
hedge mass at/near 0.5, severity spread, and the detection-quality guardrail
(per-category F1 vs ground truth at the fixed threshold). Reads the v4-based
corpus_scores.csv plus the frozen v3 aggregates; never calls a model."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import (
    BASELINE_VISION_AGGREGATES,
    CATEGORIES,
    DETECTION_THRESHOLD,
    IT2_PROMPT_VERSION,
    IT2_PROMPT_VERSION_BASELINE,
    PATHS,
    POTENCY_WEIGHTS,
)
from src import plot_style
from src.plot_style import AQUA, BLUE, INK, INK_2
from src.severity import severity

HEDGE_BAND = (0.45, 0.55)  # tolerant of 0.49/0.51 noise around the 0.5 hedge
VERSIONS = (IT2_PROMPT_VERSION_BASELINE, IT2_PROMPT_VERSION)  # (v3, v4)


def load_baseline_vision() -> dict[str, dict]:
    """Merged v3 screen_id -> prediction record across both datasets."""
    import json

    merged: dict[str, dict] = {}
    for ds, path in BASELINE_VISION_AGGREGATES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"baseline vision aggregate missing: {path} — the frozen "
                f"{IT2_PROMPT_VERSION_BASELINE} results must come from the "
                f"Drive cache; never re-run the old prompt"
            )
        merged.update(json.loads(path.read_text()))
    return merged


def comparison_table(corpus: pd.DataFrame, baseline: dict[str, dict]) -> pd.DataFrame:
    """One row per screen scored by both versions: gt, v3/v4 scores, severities.

    The corpus's vision_* columns are the v4 run; v3 comes from the baseline
    aggregates. Screens missing on either side are dropped.
    """
    rows = []
    has_v4 = corpus[f"vision_{CATEGORIES[0]}"].notna()
    for t in corpus[has_v4].itertuples():
        base = baseline.get(t.screen_id)
        if base is None:
            continue
        row = {"screen_id": t.screen_id, "source_dataset": t.source_dataset}
        for c in CATEGORIES:
            row[f"gt_{c}"] = getattr(t, f"gt_{c}")
            row[f"v3_{c}"] = base["categories"][c]
            row[f"v4_{c}"] = getattr(t, f"vision_{c}")
        row["gt_is_dark"] = t.gt_is_dark
        row["v3_is_dark"] = base["is_dark"]
        row["v4_is_dark"] = t.vision_is_dark
        row["severity_v3"] = severity(base["categories"], POTENCY_WEIGHTS)
        row["severity_v4"] = t.severity_vision
        rows.append(row)
    return pd.DataFrame(rows)


def _cat_scores(cmp: pd.DataFrame, ver: str) -> pd.Series:
    """All category scores of one version pooled into a single series."""
    return pd.concat([cmp[f"{ver}_{c}"] for c in CATEGORIES], ignore_index=True)


def hedge_table(cmp: pd.DataFrame) -> pd.DataFrame:
    """Per category and pooled: how much score mass sits at/near 0.5."""
    lo, hi = HEDGE_BAND
    rows = []
    for c in CATEGORIES + ["ALL"]:
        row = {"category": c}
        for ver in ("v3", "v4"):
            s = _cat_scores(cmp, ver) if c == "ALL" else cmp[f"{ver}_{c}"]
            row[f"{ver}_at_0.5"] = float((s == 0.5).mean())
            row[f"{ver}_in_band"] = float(((s >= lo) & (s <= hi)).mean())
            row[f"{ver}_std"] = float(s.std())
        rows.append(row)
    return pd.DataFrame(rows)


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def detection_table(cmp: pd.DataFrame) -> pd.DataFrame:
    """Guardrail: per-category P/R/F1 vs ground truth at the fixed threshold."""
    rows = []
    for c in CATEGORIES:
        row = {"category": c, "n_gt": int(cmp[f"gt_{c}"].sum())}
        for ver in ("v3", "v4"):
            pred = (cmp[f"{ver}_{c}"] >= DETECTION_THRESHOLD).astype(int).to_numpy()
            p, r, f1 = _prf(cmp[f"gt_{c}"].to_numpy(), pred)
            row.update({f"{ver}_precision": p, f"{ver}_recall": r, f"{ver}_f1": f1})
        rows.append(row)
    return pd.DataFrame(rows)


def severity_summary(cmp: pd.DataFrame) -> pd.DataFrame:
    desc = cmp[["severity_v3", "severity_v4"]].describe().T
    desc["range"] = desc["max"] - desc["min"]
    return desc.round(3).reset_index(names="score")


def _plot(cmp: pd.DataFrame, out_path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), facecolor=plot_style.SURFACE
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(plot_style.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(plot_style.MUTED)
        ax.tick_params(colors=plot_style.MUTED, labelcolor=INK_2, labelsize=9)
        ax.grid(True, color=plot_style.GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    bins = np.linspace(0, 1, 21)
    ax1.hist(_cat_scores(cmp, "v3"), bins=bins, color=BLUE, alpha=0.55,
             label=f"{IT2_PROMPT_VERSION_BASELINE} (hedged)")
    ax1.hist(_cat_scores(cmp, "v4"), bins=bins, color=AQUA, alpha=0.55,
             label=f"{IT2_PROMPT_VERSION} (calibrated)")
    ax1.set_xlabel("category score (5 categories pooled)", color=INK_2)
    ax1.set_ylabel("count", color=INK_2)
    ax1.set_title("Category-score distribution", color=INK, fontsize=11)
    ax1.legend(fontsize=8, frameon=False, labelcolor=INK_2)

    ax2.hist(cmp["severity_v3"], bins=bins, color=BLUE, alpha=0.55,
             label=f"severity ({IT2_PROMPT_VERSION_BASELINE})")
    ax2.hist(cmp["severity_v4"], bins=bins, color=AQUA, alpha=0.55,
             label=f"severity ({IT2_PROMPT_VERSION})")
    ax2.set_xlabel("severity_vision", color=INK_2)
    ax2.set_title("Severity distribution", color=INK, fontsize=11)
    ax2.legend(fontsize=8, frameon=False, labelcolor=INK_2)

    fig.suptitle(
        f"VLM calibration: {IT2_PROMPT_VERSION_BASELINE} vs "
        f"{IT2_PROMPT_VERSION} (n={len(cmp)} screens)",
        color=INK, fontsize=12,
    )
    plot_style.save(fig, out_path)


def _verdict(hedge: pd.DataFrame, det: pd.DataFrame, sev: pd.DataFrame) -> list[str]:
    h = hedge.set_index("category").loc["ALL"]
    s = sev.set_index("score")
    macro_v3 = det["v3_f1"].mean()
    macro_v4 = det["v4_f1"].mean()

    lines = ["## Verdict", ""]
    hedge_dir = "DOWN" if h["v4_in_band"] < h["v3_in_band"] else "NOT down"
    lines.append(
        f"- **Hedging {hedge_dir}**: category-score mass in "
        f"[{HEDGE_BAND[0]}, {HEDGE_BAND[1]}] went "
        f"{h['v3_in_band']:.1%} → {h['v4_in_band']:.1%} "
        f"(exactly 0.5: {h['v3_at_0.5']:.1%} → {h['v4_at_0.5']:.1%})."
    )
    spread_dir = (
        "WIDER"
        if s.loc["severity_v4", "std"] > s.loc["severity_v3", "std"]
        else "NOT wider"
    )
    lines.append(
        f"- **Severity spread {spread_dir}**: std "
        f"{s.loc['severity_v3', 'std']:.3f} → {s.loc['severity_v4', 'std']:.3f}, "
        f"range {s.loc['severity_v3', 'range']:.3f} → "
        f"{s.loc['severity_v4', 'range']:.3f}, max "
        f"{s.loc['severity_v3', 'max']:.3f} → {s.loc['severity_v4', 'max']:.3f}."
    )
    guard = "HELD" if macro_v4 >= macro_v3 - 0.05 else "REGRESSED"
    lines.append(
        f"- **Detection guardrail {guard}**: macro-F1 vs ground truth "
        f"{macro_v3:.3f} → {macro_v4:.3f} at threshold {DETECTION_THRESHOLD} "
        f"(calibration should move confidence, not correctness; a drop beyond "
        f"0.05 would mean the model got confidently wrong)."
    )
    lines.append("")
    return lines


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [
            float_fmt.format(v) if isinstance(v, float) else str(v) for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    corpus = pd.read_csv(PATHS["corpus_scores"])
    baseline = load_baseline_vision()
    cmp = comparison_table(corpus, baseline)

    n_v4 = int(corpus[f"vision_{CATEGORIES[0]}"].notna().sum())
    n_only_v4 = n_v4 - len(cmp)

    hedge = hedge_table(cmp)
    det = detection_table(cmp)
    sev = severity_summary(cmp)

    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    det.to_csv(PATHS["reports"] / "calibration_detection.csv", index=False)
    _plot(cmp, PATHS["reports"] / "calibration_v3_v4.png")

    md = [
        f"# VLM calibration: {IT2_PROMPT_VERSION_BASELINE} vs {IT2_PROMPT_VERSION}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(cmp)} screens "
        f"scored by both prompts ({n_only_v4} scored only by "
        f"{IT2_PROMPT_VERSION}, excluded here) · same model, same images, "
        f"same parser — only the prompt's calibration block differs.",
        "",
    ] + _verdict(hedge, det, sev) + [
        "## Hedge mass (share of category scores at/near 0.5)",
        "",
        _md_table(hedge),
        "",
        "## Severity_vision distribution",
        "",
        _md_table(sev),
        "",
        f"## Detection vs ground truth at threshold {DETECTION_THRESHOLD} (guardrail)",
        "",
        _md_table(det),
        "",
        "![calibration](calibration_v3_v4.png)",
        "",
    ]
    out = PATHS["reports"] / "calibration_v3_v4.md"
    out.write_text("\n".join(md))
    print(f"[calibration] wrote {out}, .png, calibration_detection.csv")

    h = hedge.set_index("category").loc["ALL"]
    print(
        f"[calibration] hedge band {h['v3_in_band']:.1%} -> "
        f"{h['v4_in_band']:.1%} | severity std "
        f"{cmp['severity_v3'].std():.3f} -> {cmp['severity_v4'].std():.3f} | "
        f"macro-F1 {det['v3_f1'].mean():.3f} -> {det['v4_f1'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
