
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import (
    CATEGORIES,
    MIN_SEVERITY_SPREAD,
    N_SEVERITY_BINS,
    N_STIMULI,
    PATHS,
    SEED,
    STIMULUS_SEVERITY_COLUMN,
    VISUAL_ONLY_PREFIX,
)

SEV = STIMULUS_SEVERITY_COLUMN


def _vision_top_category(row: pd.Series) -> str:
    scores = {c: row[f"vision_{c}"] for c in CATEGORIES}
    return max(scores, key=scores.get)


def select_stimuli(corpus: pd.DataFrame, n: int = N_STIMULI) -> pd.DataFrame:
    """Pick n stimuli spanning low -> high severity.

    Hard constraints (error if unsatisfiable): at least one ground-truth
    guilt_wording screen and one purely-visual own-set screen (FIVE_*), each
    taken as the highest-severity exemplar of its pool. Remaining slots fill
    5 equal-width severity bins, 2 per bin (seeded), preferring unrepresented
    vision top-categories and platforms; shortfalls fill greedily by max-min
    severity distance.
    """
    eligible = corpus[corpus[SEV].notna()].copy()
    eligible["top_category_vision"] = eligible.apply(_vision_top_category, axis=1)

    guilt_pool = eligible[eligible["gt_guilt_wording"] == 1]
    visual_pool = eligible[eligible["screen_id"].str.startswith(VISUAL_ONLY_PREFIX)]
    if guilt_pool.empty:
        raise RuntimeError(
            "hard constraint unsatisfiable: no eligible screen with "
            "gt_guilt_wording=1 (do the own-set caches cover the THREE_* "
            "screens?)"
        )
    if visual_pool.empty:
        raise RuntimeError(
            f"hard constraint unsatisfiable: no eligible {VISUAL_ONLY_PREFIX}* "
            f"visual-only screen has a cached vision prediction"
        )

    picked_ids = []
    constraint_tag = {}

    guilt_pick = guilt_pool.sort_values([SEV, "screen_id"], ascending=[False, True]).iloc[0]
    picked_ids.append(guilt_pick["screen_id"])
    constraint_tag[guilt_pick["screen_id"]] = "guilt_wording (hard constraint)"

    visual_pool = visual_pool[~visual_pool["screen_id"].isin(picked_ids)]
    visual_pick = visual_pool.sort_values([SEV, "screen_id"], ascending=[False, True]).iloc[0]
    picked_ids.append(visual_pick["screen_id"])
    constraint_tag[visual_pick["screen_id"]] = "visual-only (hard constraint)"

    # severity bins over the eligible range
    lo, hi = eligible[SEV].min(), eligible[SEV].max()
    edges = np.linspace(lo, hi, N_SEVERITY_BINS + 1)

    def bin_of(x: float) -> int:
        return int(min(np.searchsorted(edges, x, side="right") - 1,
                       N_SEVERITY_BINS - 1))

    per_bin_target = n // N_SEVERITY_BINS
    rng = np.random.default_rng(SEED)
    shortfalls = []

    for b in range(N_SEVERITY_BINS):
        already = sum(1 for sid in picked_ids
                      if bin_of(float(eligible.set_index("screen_id").loc[sid, SEV])) == b)
        need = per_bin_target - already
        if need <= 0:
            continue
        cands = eligible[
            (eligible[SEV].apply(bin_of) == b)
            & (~eligible["screen_id"].isin(picked_ids))
        ].copy()
        if len(cands) < need:
            shortfalls.append((b, need - len(cands)))
        # diversity preference: unseen top-category, then unseen platform
        picked_df = eligible[eligible["screen_id"].isin(picked_ids)]
        seen_cats = set(picked_df["top_category_vision"])
        seen_plats = set(picked_df["platform"])
        cands["rank_cat"] = cands["top_category_vision"].isin(seen_cats).astype(int)
        cands["rank_plat"] = cands["platform"].isin(seen_plats).astype(int)
        cands["tiebreak"] = rng.random(len(cands))
        cands = cands.sort_values(["rank_cat", "rank_plat", "tiebreak"])
        for sid in cands["screen_id"].head(need):
            picked_ids.append(sid)
            constraint_tag[sid] = f"severity bin {b + 1}/{N_SEVERITY_BINS}"

    # fill any remaining slots by max-min severity distance (greedy)
    while len(picked_ids) < n:
        pool = eligible[~eligible["screen_id"].isin(picked_ids)]
        if pool.empty:
            break
        picked_sev = eligible.set_index("screen_id").loc[picked_ids, SEV].to_numpy()
        dist = pool[SEV].apply(lambda x: np.abs(picked_sev - x).min())
        best = pool.loc[dist.idxmax(), "screen_id"]
        picked_ids.append(best)
        constraint_tag[best] = "spread fill (max-min distance)"

    chosen = (
        eligible[eligible["screen_id"].isin(picked_ids)]
        .set_index("screen_id")
        .loc[picked_ids]
        .reset_index()
        .sort_values(SEV)
        .reset_index(drop=True)
    )
    chosen["selection_reason"] = chosen["screen_id"].map(constraint_tag)
    chosen.attrs["shortfalls"] = shortfalls
    return chosen


def main() -> None:
    corpus = pd.read_csv(PATHS["corpus_scores"])
    chosen = select_stimuli(corpus)

    stimuli = pd.DataFrame({
        "screen_id": chosen["screen_id"],
        "source_dataset": chosen["source_dataset"],
        "platform": chosen["platform"],
        "manipulated_path": chosen["path"],
        # neutralized images are produced later; placeholder paths until then
        "neutralized_path": chosen["screen_id"].map(
            lambda s: str(PATHS["neutralized"] / s)
        ),
        "severity_vision": chosen["severity_vision"],
        "severity_text": chosen["severity_text"],
        "severity_vision_alt": chosen["severity_vision_alt"],
        "top_category_vision": chosen["top_category_vision"],
        "gt_guilt_wording": chosen["gt_guilt_wording"],
        "selection_reason": chosen["selection_reason"],
    })
    PATHS["data"].mkdir(parents=True, exist_ok=True)
    stimuli.to_csv(PATHS["stimuli"], index=False)

    spread = chosen[SEV].max() - chosen[SEV].min()
    n_vals = chosen[SEV].round(3).nunique()
    degenerate = spread < MIN_SEVERITY_SPREAD

    md = [
        "# Stimulus set for the behavioral experiment",
        "",
        f"{len(chosen)} screens · selected {time.strftime('%Y-%m-%d %H:%M')} · "
        f"seed {SEED}",
        "",
        "## Selection rule",
        "",
        f"Selected on **{SEV}** (the score the experiment validates), over the "
        f"{int(corpus[SEV].notna().sum())} screens with cached vision "
        "predictions. Hard constraints: at least one ground-truth "
        "guilt_wording screen and at least one purely-visual own-set screen "
        f"({VISUAL_ONLY_PREFIX}*) — without them, two of the five categories "
        "would be absent from the experiment entirely (guilt_wording exists "
        "on 12 screens of the corpus, the visual-only tricks on 6). Remaining "
        f"slots: {N_SEVERITY_BINS} equal-width severity bins, 2 per bin, "
        "seeded, preferring unrepresented vision top-categories and "
        "platforms; shortfalls fill by max-min severity distance.",
        "",
        "**Why the model chooses the stimuli:** severity for every stimulus "
        "is fixed by the frozen iteration-1/2 detectors plus the "
        "literature-derived weights *before* any behavior is measured. The "
        "fall regression of behavioral delta on severity is therefore a "
        "genuine out-of-sample prediction test of the score, not a fit to "
        "hand-picked examples.",
        "",
        "## Chosen screens (low -> high severity)",
        "",
        "| screen_id | dataset | platform | severity_vision | severity_text | "
        "top vision category | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in chosen.itertuples():
        md.append(
            f"| {r.screen_id} | {r.source_dataset} | {r.platform} | "
            f"{getattr(r, SEV):.3f} | {r.severity_text:.3f} | "
            f"{r.top_category_vision} | {r.selection_reason} |"
        )
    md += [
        "",
        f"Severity spread: min {chosen[SEV].min():.3f}, max "
        f"{chosen[SEV].max():.3f}, range {spread:.3f}, {n_vals} distinct "
        f"values (rounded to 3 dp).",
    ]
    if degenerate:
        md += [
            "",
            f"**⚠ DEGENERATE SPREAD:** the picked severities span less than "
            f"{MIN_SEVERITY_SPREAD}. This is the VLM's known 0.5-confidence "
            "hedge surfacing in the severity score — report it as a finding "
            "and consider the calibration fix before running participants.",
        ]
    if chosen.attrs.get("shortfalls"):
        md += ["", "Bin shortfalls (filled by spread-fill): " + ", ".join(
            f"bin {b + 1} short {k}" for b, k in chosen.attrs["shortfalls"]
        )]
    md += [
        "",
        "Neutralized counterparts are NOT yet produced; `stimuli.csv` carries "
        "placeholder paths under `data/neutralized/` and the instrument "
        "renders a gray placeholder until the real images exist.",
        "",
    ]
    out = PATHS["reports"] / "stimulus_set.md"
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"[stimuli] wrote {PATHS['stimuli']} and {out}")
    if degenerate:
        print(f"[stimuli] WARNING: severity spread {spread:.3f} < {MIN_SEVERITY_SPREAD}")


if __name__ == "__main__":
    main()
