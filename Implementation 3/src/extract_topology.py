
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

STRUCTURAL = {"junction", "crossover", "terminal"}


# ---------------------------------------------------------------- binarize --
def binarize(gray: np.ndarray) -> np.ndarray:
    """Return uint8 mask, 255 = ink. Adaptive, then oriented so ink is minority."""
    blur = cv2.medianBlur(gray, 5)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12
    )
    # Ink should be the minority of pixels; if not, we picked the wrong polarity.
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def keep_large_domains(mask: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """First connectivity filter: keep domains >= frac of the largest."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {i + 1 for i, a in enumerate(areas) if a >= frac * areas.max()}
    return np.isin(labels, list(keep)).astype(np.uint8) * 255


# ------------------------------------------------------------------- boxes --
def to_px(box, w, h, pad=0):
    x0, y0, x1, y1 = box
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(w, int(x1) + pad), min(h, int(y1) + pad))


def overlaps(labels: np.ndarray, box, w, h, pad) -> set[int]:
    """Labels of fragments intersecting a (padded) box."""
    x0, y0, x1, y1 = to_px(box, w, h, pad)
    if x1 <= x0 or y1 <= y0:
        return set()
    vals = np.unique(labels[y0:y1, x0:x1])
    return {int(v) for v in vals if v != 0}


# --------------------------------------------------------------- union-find --
class DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def pair_across_crossover(labels, box, w, h, pad):
    """At a crossover, wires pass over each other. Pair opposite stubs."""
    x0, y0, x1, y1 = to_px(box, w, h, pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sides = {"L": [], "R": [], "T": [], "B": []}
    for lab in overlaps(labels, box, w, h, pad):
        ys, xs = np.nonzero(labels == lab)
        # Only the part of the fragment near this crossover matters.
        near = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
        if not near.any():
            continue
        mx, my = xs[near].mean(), ys[near].mean()
        if abs(mx - cx) > abs(my - cy):
            sides["L" if mx < cx else "R"].append(lab)
        else:
            sides["T" if my < cy else "B"].append(lab)
    pairs = []
    if len(sides["L"]) == 1 and len(sides["R"]) == 1:
        pairs.append((sides["L"][0], sides["R"][0]))
    if len(sides["T"]) == 1 and len(sides["B"]) == 1:
        pairs.append((sides["T"][0], sides["B"][0]))
    return pairs


# ------------------------------------------------------------------- main ---
def extract(image_path: Path, comps: list[dict], pad: int, frac: float,
            debug_path: Path | None = None):
    """comps: dicts with cls, role, xmin..ymax. Returns list[set[int]] of nets."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "could not read image"
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = keep_large_domains(binarize(gray), frac)

    # Step 3: mask every annotated box out of the ink.
    carved = mask.copy()
    for c in comps:
        x0, y0, x1, y1 = to_px((c["xmin"], c["ymin"], c["xmax"], c["ymax"]), w, h)
        carved[y0:y1, x0:x1] = 0

    n_frag, labels = cv2.connectedComponents(carved, connectivity=8)
    if n_frag <= 1:
        return [], "no wire fragments after masking"

    # Step 5: merge fragments that are electrically the same net.
    dsu = DSU()
    for lab in range(1, n_frag):
        dsu.find(lab)
    for c in comps:
        box = (c["xmin"], c["ymin"], c["xmax"], c["ymax"])
        if c["cls"] == "junction":
            touching = sorted(overlaps(labels, box, w, h, pad))
            for other in touching[1:]:
                dsu.union(touching[0], other)
        elif c["cls"] == "crossover":
            for a, b in pair_across_crossover(labels, box, w, h, pad):
                dsu.union(a, b)

    # Step 6: which components touch which net.
    net_members: dict[int, set[int]] = defaultdict(set)
    for idx, c in enumerate(comps):
        if c["role"] != "device":
            continue
        box = (c["xmin"], c["ymin"], c["xmax"], c["ymax"])
        for lab in overlaps(labels, box, w, h, pad):
            net_members[dsu.find(lab)].add(idx)

    nets = [members for members in net_members.values() if members]

    if debug_path is not None:
        vis = img.copy()
        vis[mask > 0] = (0, 200, 255)
        vis[carved > 0] = (255, 0, 255)
        for c in comps:
            x0, y0, x1, y1 = to_px((c["xmin"], c["ymin"], c["xmax"], c["ymax"]), w, h)
            colour = (0, 255, 0) if c["role"] == "device" else (255, 128, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 3)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), vis)

    return nets, None


def load_grouped(components_csv: Path):
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with components_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["circuit_id"]), int(row["drawing"]))
            groups[key].append({
                "cls": row["cls"], "role": row["role"], "label": row["label"],
                "xmin": float(row["xmin"]), "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]), "ymax": float(row["ymax"]),
            })
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("--pad", type=int, default=6,
                    help="pixels to dilate boxes when testing contact")
    ap.add_argument("--domain-frac", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N circuit-drawings")
    ap.add_argument("--debug-dir", type=Path, default=None)
    args = ap.parse_args()

    groups = load_grouped(args.derived / "components.csv")

    paths: dict[tuple[int, int], Path] = {}
    with (args.derived / "manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths[(int(row["circuit_id"]), int(row["drawing"]))] = Path(row["image_path"])

    keys = sorted(groups)
    if args.limit:
        keys = keys[: args.limit]

    rows, failures, net_counts, deg_counts = [], [], [], []
    for cid, drw in keys:
        img_path = paths.get((cid, drw))
        if img_path is None or not img_path.exists():
            failures.append((cid, drw, "missing image"))
            continue
        dbg = args.debug_dir / f"C{cid}_D{drw}.jpg" if args.debug_dir else None
        nets, err = extract(img_path, groups[(cid, drw)], args.pad,
                            args.domain_frac, dbg)
        if err:
            failures.append((cid, drw, err))
            continue
        net_counts.append(len(nets))
        n_dev = sum(1 for c in groups[(cid, drw)] if c["role"] == "device")
        connected = {i for members in nets for i in members}
        deg_counts.append(len(connected) / n_dev if n_dev else 0.0)
        for nid, members in enumerate(nets):
            rows.append({
                "circuit_id": cid, "drawing": drw, "net_id": nid,
                "n_components": len(members),
                "components": ";".join(str(i) for i in sorted(members)),
            })

    out = args.derived / "nets.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["circuit_id", "drawing", "net_id",
                                          "n_components", "components"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out}  ({len(rows)} nets over {len(net_counts)} drawings)")
    if net_counts:
        nc = sorted(net_counts)
        print(f"\nNets per drawing:  mean {sum(nc)/len(nc):.1f}  "
              f"min {nc[0]}  median {nc[len(nc)//2]}  max {nc[-1]}")
        dc = sorted(deg_counts)
        print(f"Fraction of devices attached to >=1 net:  "
              f"mean {sum(dc)/len(dc):.2f}  min {dc[0]:.2f}  median {dc[len(dc)//2]:.2f}")
        print("\n  Sanity: that fraction should be close to 1.0.")
        print("  Well below 1.0 means masking or contact detection is off --")
        print("  raise --pad, or inspect the debug overlays.")
    for cid, drw, why in failures[:10]:
        print(f"  FAILED C{cid}_D{drw}: {why}")
    if failures:
        print(f"  ({len(failures)} failures total)")


if __name__ == "__main__":
    main()