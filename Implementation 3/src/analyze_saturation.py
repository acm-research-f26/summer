

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_histograms(path: Path, coarse: bool):
    """circuit_id -> {drawing: frozen histogram}"""
    hist = defaultdict(lambda: defaultdict(Counter))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["role"] != "device":
                continue
            label = row["label"].split(".")[0] if coarse else row["label"]
            hist[int(row["circuit_id"])][int(row["drawing"])][label] += 1
    return hist


def freeze(c: Counter):
    return tuple(sorted(c.items()))


def report(hist, coarse: bool, label: str) -> None:
    # Use drawing 2 (the database side) as the identity of each circuit.
    keys = {c: freeze(v[2]) for c, v in hist.items() if 2 in v}
    groups = defaultdict(list)
    for cid, k in keys.items():
        groups[k].append(cid)

    n = len(keys)
    unique = sum(1 for g in groups.values() if len(g) == 1)
    colliding = n - unique
    biggest = sorted(groups.values(), key=len, reverse=True)[:5]

    print(f"\n--- {label} ---")
    print(f"  circuits                     {n}")
    print(f"  distinct parts lists         {len(groups)}")
    print(f"  uniquely identified          {unique}  ({unique/n:.1%})")
    print(f"  in a collision group         {colliding}  ({colliding/n:.1%})")
    if biggest and len(biggest[0]) > 1:
        print("  largest collision groups:")
        for g in biggest:
            if len(g) > 1:
                print(f"      {len(g)} circuits: {sorted(g)[:8]}")


def drawing_agreement(hist) -> None:
    """How often do the two drawings of one circuit have identical parts lists?"""
    both = [(v[1], v[2]) for v in hist.values() if 1 in v and 2 in v]
    same = sum(1 for a, b in both if freeze(a) == freeze(b))
    print(f"\n--- redraw consistency ---")
    print(f"  circuits with both drawings  {len(both)}")
    print(f"  identical parts lists        {same}  ({same/len(both):.1%})")
    print("\n  This is the number that explains the saturated benchmark: when a")
    print("  person redraws a circuit they draw the same parts, so matching")
    print("  D1 to D2 barely requires topology at all.")


def class_discrimination(path: Path) -> None:
    """Which device classes actually separate circuits?"""
    per_circuit = defaultdict(Counter)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["role"] == "device" and int(row["drawing"]) == 2:
                per_circuit[int(row["circuit_id"])][row["label"]] += 1

    presence = Counter()
    for counts in per_circuit.values():
        for lab in counts:
            presence[lab] += 1

    n = len(per_circuit)
    print(f"\n--- device class prevalence (drawing 2, {n} circuits) ---")
    print("  a class present in ~50% of circuits is maximally informative;")
    print("  one present in 95% or 2% tells you almost nothing.\n")
    for lab, k in presence.most_common(15):
        bar = "#" * int(40 * k / n)
        print(f"  {lab:<28} {k/n:5.1%} {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    args = ap.parse_args()
    comp = args.derived / "components.csv"

    fine = load_histograms(comp, coarse=False)
    coarse = load_histograms(comp, coarse=True)

    report(fine, False, "fine labels (capacitor.polarized separate)")
    report(coarse, True, "coarse labels (all capacitors merged)")
    drawing_agreement(fine)
    class_discrimination(comp)


if __name__ == "__main__":
    main()