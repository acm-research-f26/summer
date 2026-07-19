
from __future__ import annotations

import time

import pandas as pd

from config import (
    CATEGORIES,
    DETECTION_THRESHOLD,
    PATHS,
    POTENCY_WEIGHTS,
    POTENCY_WEIGHTS_ALT,
)
from src import plot_style
from src.plot_style import AQUA, BLUE, INK, INK_2


def prevalence_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """Per category: literature potency (both vectors) + three prevalences.

    prev_gt (primary) = fraction of the corpus whose ground-truth labels
    include the category. prev_text / prev_vision = fraction where the cached
    detector score clears the fixed threshold; the vision denominator
    excludes screens with no cached vision prediction.
    """
    vision_ok = corpus[f"vision_{CATEGORIES[0]}"].notna()
    rows = []
    for c in CATEGORIES:
        rows.append({
            "category": c,
            "potency": POTENCY_WEIGHTS[c],
            "potency_alt": POTENCY_WEIGHTS_ALT[c],
            "prev_gt": corpus[f"gt_{c}"].mean(),
            "n_gt": int(corpus[f"gt_{c}"].sum()),
            "prev_text": (corpus[f"text_{c}"] >= DETECTION_THRESHOLD).mean(),
            "prev_vision": (
                corpus.loc[vision_ok, f"vision_{c}"] >= DETECTION_THRESHOLD
            ).mean(),
        })
    return pd.DataFrame(rows)


def modality_gap_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """Per category, how often each modality fires at the fixed threshold,
    among screens with a vision prediction; includes vision-only examples."""
    v = corpus[corpus[f"vision_{CATEGORIES[0]}"].notna()]
    rows = []
    for c in CATEGORIES:
        text_fire = v[f"text_{c}"] >= DETECTION_THRESHOLD
        vision_fire = v[f"vision_{c}"] >= DETECTION_THRESHOLD
        vision_only = v[vision_fire & ~text_fire]
        rows.append({
            "category": c,
            "both": int((text_fire & vision_fire).sum()),
            "text_only": int((text_fire & ~vision_fire).sum()),
            "vision_only": int(len(vision_only)),
            "neither": int((~text_fire & ~vision_fire).sum()),
            "vision_only_examples": ", ".join(vision_only["screen_id"].head(3)),
        })
    return pd.DataFrame(rows)


def _plot_map(table: pd.DataFrame, n_corpus: int, out_path) -> None:
    fig, ax = plot_style.new_axes()

    ax.scatter(table["prev_gt"], table["potency"], s=90, color=BLUE, zorder=3,
               label="primary weights")
    # the alt vector moves only "other" — plot just the moved point
    moved = table[table["potency"] != table["potency_alt"]]
    if not moved.empty:
        ax.scatter(moved["prev_gt"], moved["potency_alt"], s=90, zorder=3,
                   facecolors="none", edgecolors=AQUA, linewidths=1.8,
                   label="alt weights (sensitivity)")
        for r in moved.itertuples():
            ax.annotate("", xy=(r.prev_gt, r.potency_alt),
                        xytext=(r.prev_gt, r.potency),
                        arrowprops={"arrowstyle": "-", "color": AQUA,
                                    "linestyle": ":", "linewidth": 1.0})

    for r in table.itertuples():
        ax.annotate(r.category, (r.prev_gt, r.potency),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=9, color=INK_2)

    ax.set_xlabel(f"prevalence in corpus (n={n_corpus}) — ground truth",
                  color=INK_2)
    ax.set_ylabel("potency (literature-derived weight)", color=INK_2)
    ax.set_title("Prevalence vs potency, 5 dark-pattern categories",
                 color=INK, fontsize=12)
    ax.set_xlim(0, max(0.5, table["prev_gt"].max() * 1.25))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    plot_style.save(fig, out_path)


def _findings(table: pd.DataFrame, gap: pd.DataFrame, corpus: pd.DataFrame) -> list[str]:
    n_corpus = len(corpus)
    t = table.set_index("category")
    most_prevalent = t["prev_gt"].idxmax()
    most_potent = t["potency"].idxmax()
    most_potent_alt = t["potency_alt"].idxmax()

    lines = ["## Findings", ""]
    if most_prevalent != most_potent:
        lines.append(
            f"- **The corpus's most common pattern is not its most potent.** "
            f"`{most_prevalent}` is the most prevalent category "
            f"({t.loc[most_prevalent, 'prev_gt']:.1%} of screens) but carries a "
            f"potency weight of {t.loc[most_prevalent, 'potency']:.1f}, while "
            f"`{most_potent}` holds the highest potency "
            f"({t.loc[most_potent, 'potency']:.1f}) at a prevalence of "
            f"{t.loc[most_potent, 'prev_gt']:.1%}."
        )
    else:
        lines.append(
            f"- **Under the primary weights, the most prevalent category "
            f"(`{most_prevalent}`, {t.loc[most_prevalent, 'prev_gt']:.1%}) is "
            f"also the most potent** ({t.loc[most_prevalent, 'potency']:.1f})."
        )
    if most_potent_alt != most_potent:
        lines.append(
            f"- **Weight sensitivity: the headline flips under the alt vector.** "
            f"With `other` at {t.loc['other', 'potency_alt']:.1f} (naive L&S "
            f"mapping), `{most_potent_alt}` becomes the most potent category, "
            f"so any prevalence-potency claim must be stated per weight vector."
        )
    else:
        lines.append(
            "- Weight sensitivity: the potency ordering at the top does not "
            "change under the alt vector; the headline is robust to the "
            "`other` weight choice."
        )

    weakest = t["potency"].idxmin()
    lines.append(
        f"- The weakest-weighted category (`{weakest}`, "
        f"{t.loc[weakest, 'potency']:.1f}) appears on "
        f"{t.loc[weakest, 'prev_gt']:.1%} of screens — if sites leaned on "
        f"patterns in proportion to their effectiveness this would be near "
        f"the bottom of the prevalence range too."
    )

    g = gap.set_index("category")
    total_vision_only = int(g["vision_only"].sum())
    top_gap = g["vision_only"].idxmax()
    lines.append(
        f"- **Invisible to text-only tools:** across the corpus, "
        f"{total_vision_only} category-level detections fire for vision but "
        f"not text at the fixed 0.5 threshold, led by `{top_gap}` "
        f"({g.loc[top_gap, 'vision_only']} screens). See the modality-gap "
        f"table."
    )
    n_guilt = int(t.loc["guilt_wording", "n_gt"])
    n_guilt_ctx = int(
        corpus.loc[corpus["source_dataset"] == "contextdp", "gt_guilt_wording"].sum()
    )
    origin = (
        "all from the curated own set (no public dataset labels it)"
        if n_guilt_ctx == 0
        else f"{n_guilt_ctx} from CONTEXTDP, the rest from the own set"
    )
    lines.append(
        f"- `guilt_wording` has ground truth on only {n_guilt} of {n_corpus} "
        f"screens, {origin} — its prevalence point is the least certain on "
        f"the map."
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
    n_corpus = len(corpus)
    n_vision = int(corpus[f"vision_{CATEGORIES[0]}"].notna().sum())

    table = prevalence_table(corpus)
    gap = modality_gap_table(corpus)

    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    table.to_csv(PATHS["reports"] / "prevalence_potency.csv", index=False)
    _plot_map(table, n_corpus, PATHS["reports"] / "prevalence_potency.png")

    md = [
        "# Prevalence vs potency",
        "",
        f"Corpus: {n_corpus} screens (CONTEXTDP 501 + own 45) · detection "
        f"threshold {DETECTION_THRESHOLD} · generated "
        f"{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Prevalence (primary) = fraction of screens whose **ground-truth** "
        "labels include the category; detector-based prevalences are shown "
        "alongside (text over-fires and vision under-fires social_proof, so "
        "neither is an unbiased prevalence estimate). Potency = "
        "literature-derived weight from the Luguri & Strahilevitz (2021) "
        "ordering — an assumption, not a measurement (see caveats).",
        "",
        "## Caveats (read first)",
        "",
        f"- **Prevalence here is prevalence in this corpus (n={n_corpus}), "
        "not on the web.** CONTEXTDP is a curated evaluation set roughly half "
        "non-dark by construction, and the own 45 were deliberately "
        "stratified; these counts do not estimate real-world frequency.",
        "- **`other` in this corpus is mostly preselection and nagging** "
        "(CONTEXTDP: DEFAULT CHOICE 111, NAGGING 57, DISGUISED ADS 21, "
        "ATTENTION DISTRACTION 13, GAMIFICATION 11), NOT the hidden-"
        "information / obstruction / trick-question patterns Luguri & "
        "Strahilevitz found most potent. Its primary weight (0.5) is a "
        "floor-level assumption for what the bucket actually contains; the "
        "alt vector (0.9) shows what the naive L&S mapping would claim.",
        "- The map has 5 points and its potency axis is an ordinal literature "
        "ranking mapped to [0,1]; it is descriptive — no correlation "
        "statistic would be meaningful at n=5.",
        "",
        "## Table",
        "",
        _md_table(table),
        "",
        f"## Modality gap — detections vision makes that text misses "
        f"(n={n_vision} screens with vision predictions)",
        "",
        _md_table(gap),
        "",
    ] + _findings(table, gap, corpus) + [
        "![prevalence vs potency](prevalence_potency.png)",
        "",
    ]
    out = PATHS["reports"] / "prevalence_potency.md"
    out.write_text("\n".join(md))
    print(f"[prevalence] wrote {out}, .csv, .png")


if __name__ == "__main__":
    main()
