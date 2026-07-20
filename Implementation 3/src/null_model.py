

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_components(path: Path, role: str = "device"):
    """circuit_id -> {drawing: Counter(label)}"""
    data: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["role"] != role:
                continue
            cid, drw = int(row["circuit_id"]), int(row["drawing"])
            data[cid][drw][row["label"]] += 1
    return data


def histogram_distance(a: Counter, b: Counter) -> int:
    keys = set(a) | set(b)
    surplus = sum(max(0, a[k] - b[k]) for k in keys)
    deficit = sum(max(0, b[k] - a[k]) for k in keys)
    return max(surplus, deficit)


def similarity(a: Counter, b: Counter) -> float:
    d = histogram_distance(a, b)
    denom = (sum(a.values()) + sum(b.values())) / 2.0
    return math.exp(-d / denom) if denom else 0.0


def evaluate(data, k: int = 5, seed: int = 0):
    circuits = sorted(c for c, v in data.items() if 1 in v and 2 in v)
    if len(circuits) < 2:
        raise SystemExit("Need at least 2 circuits with both drawings.")

    ranks = []
    for q in circuits:
        qh = data[q][1]
        scored = [(similarity(qh, data[g][2]), g) for g in circuits]
        # Ties are common with coarse histograms. Break them randomly rather
        # than by circuit id, or sorting order silently inflates the score.
        rng = random.Random(seed + q)
        scored.sort(key=lambda t: (-t[0], rng.random()))
        rank = next(i for i, (_, g) in enumerate(scored, 1) if g == q)
        ranks.append(rank)

    n = len(ranks)
    return {
        "circuits": n,
        "top1": sum(r == 1 for r in ranks) / n,
        f"recall@{k}": sum(r <= k for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "median_rank": sorted(ranks)[n // 2],
        "random_recall": k / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path, help="folder containing components.csv")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    data = load_components(args.derived / "components.csv")
    res = evaluate(data, k=args.k)

    print("Null model: component counts only, no topology\n")
    print(f"  circuits          {res['circuits']}")
    print(f"  top-1 accuracy    {res['top1']:.3f}")
    print(f"  recall@{args.k}          {res[f'recall@{args.k}']:.3f}")
    print(f"  MRR               {res['mrr']:.3f}")
    print(f"  median rank       {res['median_rank']}")
    print(f"\n  random baseline recall@{args.k} = {res['random_recall']:.3f}")
    print("\nRead this as: how much of retrieval is explained WITHOUT topology.")
    print("Your graph-based results must beat it to justify the graph pipeline.")


if __name__ == "__main__":
    main()