# Build a manifest of all annotated images and their associated annotations.

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

STEM_RE = re.compile(r"^C(-?\d+)_D(\d+)_P(\d+)$", re.IGNORECASE)

EXCLUDED_DRAFTERS = {"drafter_0", "drafter_31"}

# Annotations that are not part of the circuit at all.
DROP_CLASSES = {"text", "explanatory"}

# Annotations that carry CONNECTIVITY information but are not devices.
#   junction  - CGHD uses this for real junctions AND plain wire corners,
#               which is why there are ~79k of them. Not a device.
#   crossover - wires that cross WITHOUT connecting. Critical for topology:
#               this is the distinction Chen's method cannot make.
#   terminal  - circuit input/output port.
STRUCTURAL_CLASSES = {"junction", "crossover", "terminal"}


def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())   # (circuit, drawing, photo)


def parse_voc(xml_path: Path):
    """Yield (class_name, xmin, ymin, xmax, ymax)."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return
    for obj in root.findall("object"):
        name_el = obj.find("name")
        box = obj.find("bndbox")
        if name_el is None or box is None or not name_el.text:
            continue
        try:
            coords = [float(box.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
        except (AttributeError, TypeError, ValueError):
            continue
        yield (name_el.text.strip(), *coords)


def role_of(cls: str) -> str:
    if cls in DROP_CLASSES:
        return "drop"
    if cls in STRUCTURAL_CLASSES:
        return "structural"
    return "device"


def node_label(cls: str, coarse: bool) -> str:
    """Label used for graph nodes. Coarse merges 'capacitor.polarized' etc."""
    return cls.split(".")[0] if coarse else cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/derived"))
    ap.add_argument("--drafters", nargs="*", default=None,
                    help="restrict to these drafter numbers, e.g. --drafters 1 2")
    ap.add_argument("--coarse-labels", action="store_true",
                    help="merge subtypes: capacitor.polarized -> capacitor")
    ap.add_argument("--allow-negative", action="store_true",
                    help="include drafter_-1 (different naming convention)")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {f"drafter_{d}" for d in args.drafters} if args.drafters else None

    # ---- pass 1: index every annotated image we can use --------------------
    # found[circuit][drawing] = (drafter, image_path, xml_path)
    found: dict[int, dict[int, tuple]] = defaultdict(dict)
    skipped = Counter()

    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("drafter"):
            continue
        if d.name in EXCLUDED_DRAFTERS:
            skipped["excluded drafter"] += 1
            continue
        if d.name == "drafter_-1" and not args.allow_negative:
            skipped["excluded drafter"] += 1
            continue
        if wanted and d.name not in wanted:
            continue

        img_dir, ann_dir = d / "images", d / "annotations"
        if not img_dir.is_dir() or not ann_dir.is_dir():
            skipped["missing images/ or annotations/"] += 1
            continue

        for img in sorted(img_dir.glob("*.jp*g")):
            parsed = parse_stem(img.stem)
            if parsed is None:
                skipped["unparseable filename"] += 1
                continue
            circuit, drawing, photo = parsed
            if photo != 1:          # one representative photo per drawing
                continue
            xml = ann_dir / f"{img.stem}.xml"
            if not xml.exists():
                skipped["image without annotation"] += 1
                continue
            found[circuit][drawing] = (d.name, img, xml)

    # ---- pass 2: keep only circuits with BOTH drawing 1 and drawing 2 ------
    usable = {c: v for c, v in found.items() if 1 in v and 2 in v}
    dropped_unpaired = len(found) - len(usable)

    # ---- pass 3: extract components ---------------------------------------
    man_rows, comp_rows = [], []
    class_counts, role_counts = Counter(), Counter()
    devices_per_circuit = []

    for circuit in sorted(usable):
        for drawing in (1, 2):
            drafter, img, xml = usable[circuit][drawing]
            counts = Counter()
            for cls, x0, y0, x1, y1 in parse_voc(xml):
                role = role_of(cls)
                class_counts[cls] += 1
                role_counts[role] += 1
                counts[role] += 1
                if role == "drop":
                    continue
                comp_rows.append({
                    "circuit_id": circuit,
                    "drawing": drawing,
                    "cls": cls,
                    "role": role,
                    "label": node_label(cls, args.coarse_labels) if role == "device" else cls,
                    "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                })
            if drawing == 1:
                devices_per_circuit.append(counts["device"])
            man_rows.append({
                "circuit_id": circuit,
                "drafter": drafter,
                "drawing": drawing,
                "split": "query" if drawing == 1 else "database",
                "n_devices": counts["device"],
                "n_structural": counts["structural"],
                "n_dropped": counts["drop"],
                "image_path": str(img),
                "xml_path": str(xml),
            })

    # ---- write ------------------------------------------------------------
    man_path = args.out / "manifest.csv"
    with man_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0]))
        w.writeheader()
        w.writerows(man_rows)

    comp_path = args.out / "components.csv"
    with comp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0]))
        w.writeheader()
        w.writerows(comp_rows)

    # ---- report -----------------------------------------------------------
    n = len(devices_per_circuit)
    mean_dev = sum(devices_per_circuit) / n if n else 0
    smallest = sorted(devices_per_circuit)

    print(f"Wrote {man_path}  ({len(man_rows)} rows)")
    print(f"Wrote {comp_path}  ({len(comp_rows)} rows)")
    print(f"\nUsable circuits: {len(usable)}   "
          f"(dropped {dropped_unpaired} without both drawings)")
    for reason, k in skipped.items():
        print(f"  skipped: {reason} x{k}")

    print(f"\nDevices per circuit: mean {mean_dev:.1f}, "
          f"min {smallest[0]}, median {smallest[n // 2]}, max {smallest[-1]}")
    print(f"  circuits with <=8 devices: {sum(1 for d in smallest if d <= 8)}"
          f"   <-- exact GED is tractable on these")
    print(f"  circuits with <=12 devices: {sum(1 for d in smallest if d <= 12)}")

    print("\nAnnotation roles:")
    for role, k in role_counts.most_common():
        print(f"  {role:<12} {k:>8}")

    print(f"\nDevice classes kept: "
          f"{len({c for c in class_counts if role_of(c) == 'device'})}")
    for cls, k in class_counts.most_common(15):
        print(f"  {cls:<32} {k:>7}  [{role_of(cls)}]")


if __name__ == "__main__":
    main()