
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES, PATHS, THRESHOLD


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def score_system(truth: pd.DataFrame, preds: dict[str, dict],
                 threshold: float = THRESHOLD) -> dict:
    """Metrics for one system against a truth DataFrame (dataset_* format).

    `preds` maps filename -> screen result ({"is_dark", "categories", ...}).
    Screens missing from `preds` raise — nothing is silently dropped.
    """
    missing = [f for f in truth["filename"] if f not in preds]
    if missing:
        raise KeyError(f"{len(missing)} screens missing predictions, e.g. {missing[:3]}")

    y_true = truth[CATEGORIES].to_numpy()
    y_score = np.array([
        [preds[f]["categories"][c] for c in CATEGORIES] for f in truth["filename"]
    ])
    y_pred = (y_score >= threshold).astype(int)

    per_cat = {}
    for j, c in enumerate(CATEGORIES):
        support = int(y_true[:, j].sum())
        if support == 0:
            per_cat[c] = {"precision": None, "recall": None, "f1": None,
                          "support": 0,
                          "false_positives": int(y_pred[:, j].sum())}
        else:
            p, r, f1 = _prf(y_true[:, j], y_pred[:, j])
            per_cat[c] = {"precision": p, "recall": r, "f1": f1, "support": support}

    populated = [c for c in CATEGORIES if per_cat[c]["support"] > 0]
    macro_f1 = float(np.mean([per_cat[c]["f1"] for c in populated]))
    macro_p = float(np.mean([per_cat[c]["precision"] for c in populated]))
    macro_r = float(np.mean([per_cat[c]["recall"] for c in populated]))
    micro_p, micro_r, micro_f1 = _prf(y_true.ravel(), y_pred.ravel())

    dark_true = truth["is_dark"].to_numpy()
    dark_pred = np.array([
        int(preds[f]["is_dark"] >= threshold) for f in truth["filename"]
    ])
    is_dark_acc = float((dark_true == dark_pred).mean())
    is_dark_p, is_dark_r, is_dark_f1 = _prf(dark_true, dark_pred)

    return {
        "per_category": per_cat,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_over": populated,
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "is_dark_accuracy": is_dark_acc,
        "is_dark_precision": is_dark_p,
        "is_dark_recall": is_dark_r,
        "is_dark_f1": is_dark_f1,
        "n_screens": len(truth),
        "threshold": threshold,
    }


def _fmt(x) -> str:
    # None becomes NaN once it passes through a DataFrame
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


def _delta(t_f1, v_f1):
    return None if (t_f1 is None or v_f1 is None) else v_f1 - t_f1


def headline_rows(text_m: dict, vision_m: dict) -> pd.DataFrame:
    """Precision / recall / F1 side by side, text vs vision, plus ΔF1.

    Rows: the 5 categories, then micro, macro* (over populated categories),
    and is_dark (binary presence). is_dark accuracy is reported separately
    below the table since it is not a P/R/F1 quantity.
    """
    rows = []
    for c in CATEGORIES:
        t, v = text_m["per_category"][c], vision_m["per_category"][c]
        rows.append({"category": c, "support": t["support"],
                     "text_p": t["precision"], "text_r": t["recall"], "text_f1": t["f1"],
                     "vision_p": v["precision"], "vision_r": v["recall"], "vision_f1": v["f1"],
                     "delta_f1": _delta(t["f1"], v["f1"])})
    rows.append({"category": "micro", "support": "",
                 "text_p": text_m["micro_precision"], "text_r": text_m["micro_recall"],
                 "text_f1": text_m["micro_f1"],
                 "vision_p": vision_m["micro_precision"], "vision_r": vision_m["micro_recall"],
                 "vision_f1": vision_m["micro_f1"],
                 "delta_f1": _delta(text_m["micro_f1"], vision_m["micro_f1"])})
    rows.append({"category": "macro*", "support": "",
                 "text_p": text_m["macro_precision"], "text_r": text_m["macro_recall"],
                 "text_f1": text_m["macro_f1"],
                 "vision_p": vision_m["macro_precision"], "vision_r": vision_m["macro_recall"],
                 "vision_f1": vision_m["macro_f1"],
                 "delta_f1": _delta(text_m["macro_f1"], vision_m["macro_f1"])})
    rows.append({"category": "is_dark", "support": "",
                 "text_p": text_m["is_dark_precision"], "text_r": text_m["is_dark_recall"],
                 "text_f1": text_m["is_dark_f1"],
                 "vision_p": vision_m["is_dark_precision"], "vision_r": vision_m["is_dark_recall"],
                 "vision_f1": vision_m["is_dark_f1"],
                 "delta_f1": _delta(text_m["is_dark_f1"], vision_m["is_dark_f1"])})
    return pd.DataFrame(rows)


def _table_md(df: pd.DataFrame) -> str:
    lines = [
        "| Category | Support | Text P | Text R | Text F1 | Vision P | Vision R | "
        "Vision F1 | ΔF1 (vision − text) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.category} | {r.support} | {_fmt(r.text_p)} | {_fmt(r.text_r)} | "
            f"{_fmt(r.text_f1)} | {_fmt(r.vision_p)} | {_fmt(r.vision_r)} | "
            f"{_fmt(r.vision_f1)} | {_fmt(r.delta_f1)} |"
        )
    return "\n".join(lines)


def _takeaway(text_m: dict, vision_m: dict) -> str:
    """One plain-language line derived from the MACRO-averaged precision/recall
    (the per-category average shown in the 'macro*' row), not micro — micro is
    dominated by the largest category (`other`) and hides per-category
    over-firing. The wording follows the actual gap and always cites the
    numbers it is based on, so it cannot drift from the data.
    """
    tp, tr = text_m["macro_precision"], text_m["macro_recall"]
    vp, vr = vision_m["macro_precision"], vision_m["macro_recall"]

    def shape(p, r):
        d = r - p
        if d >= 0.2:
            return (f"over-fires (recall {r:.2f} far above precision {p:.2f} — "
                    f"catches most positives but with many false positives)")
        if d >= 0.1:
            return f"leans toward over-firing (recall {r:.2f} above precision {p:.2f})"
        if d <= -0.2:
            return (f"is conservative (precision {p:.2f} far above recall {r:.2f} — "
                    f"rarely fires, misses most positives)")
        if d <= -0.1:
            return f"leans conservative (precision {p:.2f} above recall {r:.2f})"
        return f"is balanced (precision {p:.2f}, recall {r:.2f})"

    # "more selective" only when recalls are comparable and one holds higher precision
    lead = ""
    if abs(tr - vr) <= 0.15:
        if vp - tp >= 0.05:
            lead = (f"Vision is the more selective detector — at comparable recall "
                    f"({vr:.2f} vs {tr:.2f}) it holds higher precision ({vp:.2f} vs {tp:.2f}). ")
        elif tp - vp >= 0.05:
            lead = (f"Text is the more precise detector — at comparable recall "
                    f"({tr:.2f} vs {vr:.2f}) it holds higher precision ({tp:.2f} vs {vp:.2f}). ")
    return f"**Takeaway (macro-averaged):** {lead}Text {shape(tp, tr)}; vision {shape(vp, vr)}."


def _threshold_section(truth: pd.DataFrame, preds_text: dict, preds_vision: dict,
                       thresholds=(0.4, 0.5, 0.6)) -> list[str]:
    """Robustness check from the SAME cached scores — no re-run, prompt frozen."""
    lines = [
        "## Threshold sensitivity (robustness check)",
        "",
        "Same cached predictions, scored at three cutoffs. Confirms the "
        "text-vs-vision ordering is not an artifact of the 0.5 threshold.",
        "",
        "| Threshold | Text micro-F1 | Vision micro-F1 | Text macro-F1* | "
        "Vision macro-F1* | Text is_dark-acc | Vision is_dark-acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for th in thresholds:
        tm = score_system(truth, preds_text, th)
        vm = score_system(truth, preds_vision, th)
        lines.append(
            f"| {th:.1f} | {tm['micro_f1']:.3f} | {vm['micro_f1']:.3f} | "
            f"{tm['macro_f1']:.3f} | {vm['macro_f1']:.3f} | "
            f"{tm['is_dark_accuracy']:.3f} | {vm['is_dark_accuracy']:.3f} |"
        )
    lines.append("")
    return lines


def write_headline_report(truth: pd.DataFrame, preds_text: dict, preds_vision: dict,
                          dataset_name: str, threshold: float = THRESHOLD) -> Path:
    """The deliverable: reports/text_vs_vision.md (+.csv), full set and
    web/mobile cuts."""
    reports = PATHS["reports"]
    reports.mkdir(parents=True, exist_ok=True)

    text_m = score_system(truth, preds_text, threshold)
    vision_m = score_system(truth, preds_vision, threshold)
    full = headline_rows(text_m, vision_m)
    full.to_csv(reports / "text_vs_vision.csv", index=False)

    md = [
        "# Does seeing the page beat reading it?",
        "",
        f"Dataset: **{dataset_name}** ({len(truth)} screens) · threshold {threshold} · "
        f"generated {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Screen-level multi-label scores: each category is an independent presence",
        "probability (NOT a softmax). Text = OCR → frozen iteration-1 classifier;",
        "Vision = prompted Qwen2-VL (see config.py for versions). Precision, recall",
        "and F1 shown side by side so over-firing vs selectivity is visible directly.",
        "",
        "## Headline — precision / recall / F1 (full set)",
        "",
        _table_md(full),
        "",
        f"is_dark accuracy: text {text_m['is_dark_accuracy']:.3f}, "
        f"vision {vision_m['is_dark_accuracy']:.3f} "
        f"(non-DP base rate {1 - truth['is_dark'].mean():.3f}).",
        "",
        _takeaway(text_m, vision_m),
        "",
        f"\\* macro over categories with >0 positives: "
        f"{', '.join(text_m['macro_over'])}. Categories with 0 positives are n/a "
        f"(their false positives still count in micro: "
        f"text {sum(text_m['per_category'][c].get('false_positives', 0) for c in CATEGORIES if text_m['per_category'][c]['support'] == 0)}, "
        f"vision {sum(vision_m['per_category'][c].get('false_positives', 0) for c in CATEGORIES if vision_m['per_category'][c]['support'] == 0)}).",
        "",
    ]

    for plat in sorted(truth["platform"].unique()):
        if plat == "own":
            continue
        cut = truth[truth["platform"] == plat]
        tm = score_system(cut, preds_text, threshold)
        vm = score_system(cut, preds_vision, threshold)
        md += [f"## Cut: {plat} only ({len(cut)} screens)", "",
               _table_md(headline_rows(tm, vm)), "",
               f"is_dark accuracy: text {tm['is_dark_accuracy']:.3f}, "
               f"vision {vm['is_dark_accuracy']:.3f}.", "",
               _takeaway(tm, vm), ""]
        if plat == "mobile":
            md += ["Note: CONTEXTDP mobile has zero urgency/scarcity/social_proof "
                   "instances — only is_dark and `other` are informative here.", ""]

    md += _threshold_section(truth, preds_text, preds_vision)

    out = reports / "text_vs_vision.md"
    out.write_text("\n".join(md))
    print(f"[eval] wrote {out} and text_vs_vision.csv")
    return out


def write_error_gallery(truth: pd.DataFrame, preds: dict, system_name: str,
                        extra: dict[str, str] | None = None,
                        threshold: float = THRESHOLD, n: int = 10) -> Path:
    """10 worst failures: filename, truth flags, predicted scores (+ per-system
    extra context: OCR text for A, raw reply for B)."""
    scored = []
    for row in truth.itertuples():
        p = preds[row.filename]
        err = sum(
            abs(getattr(row, c) - (p["categories"][c] >= threshold))
            for c in CATEGORIES
        ) + abs(row.is_dark - (p["is_dark"] >= threshold))
        if err > 0:
            margin = sum(abs(getattr(row, c) - p["categories"][c]) for c in CATEGORIES)
            scored.append((err, margin, row, p))
    scored.sort(key=lambda t: (-t[0], -t[1]))

    lines = [f"# Error gallery — {system_name}",
             f"\n{len(scored)} screens with ≥1 wrong flag "
             f"(threshold {threshold}); showing worst {min(n, len(scored))}.\n"]
    for err, _, row, p in scored[:n]:
        truth_flags = [c for c in CATEGORIES if getattr(row, c)] or ["(none — non-DP)"]
        pred_scores = ", ".join(f"{c}={p['categories'][c]:.2f}" for c in CATEGORIES)
        lines += [f"## {row.filename} ({row.platform}) — {err} wrong flag(s)",
                  f"- truth: {', '.join(truth_flags)} | is_dark={row.is_dark}",
                  f"- predicted: {pred_scores} | is_dark={p['is_dark']:.2f}"]
        if extra and row.filename in extra:
            lines.append(f"- context:\n```\n{extra[row.filename][:800]}\n```")
        lines.append("")

    out = PATHS["reports"] / f"error_gallery_{system_name}.md"
    out.write_text("\n".join(lines))
    print(f"[eval] wrote {out}")
    return out
