""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# Matches C11_D1_P2  ->  circuit=11, drawing=1, picture=2
NAME_RE = re.compile(r"^C(\d+)_D(\d+)_P(\d+)$", re.IGNORECASE)


def parse_stem(stem: str):
    """Return (circuit, drawing, picture) or None if the name doesn't match."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def parse_voc(xml_path: Path):
    """Return list of (class_name, xmin, ymin, xmax, ymax) from a VOC file."""
    objects = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        print(f"  ! could not parse {xml_path.name}: {e}")
        return objects

    for obj in root.findall("object"):
        name_el = obj.find("name")
        box = obj.find("bndbox")
        if name_el is None or box is None:
            continue

        def val(tag):
            el = box.find(tag)
            return float(el.text) if el is not None and el.text else None

        coords = [val(t) for t in ("xmin", "ymin", "xmax", "ymax")]
        if any(c is None for c in coords):
            continue
        objects.append((name_el.text.strip(), *coords))
    return objects


def main(root: Path) -> None:
    drafters = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("drafter")),
        key=lambda p: p.name,
    )
    if not drafters:
        print(f"No drafter_* folders found in {root}")
        print("Contents:", [p.name for p in list(root.iterdir())[:20]])
        return

    print(f"Found {len(drafters)} drafter folders in {root}\n")

    class_counts = Counter()
    circuits = defaultdict(set)        # circuit_id -> set of drafter names
    drawings = defaultdict(set)        # circuit_id -> set of drawing numbers
    n_images = n_annots = n_seg = n_asc = 0
    bad_names = []

    for d in drafters:
        img_dir, ann_dir, seg_dir = d / "images", d / "annotations", d / "segmentation"

        imgs = sorted(img_dir.glob("*.jp*g")) if img_dir.is_dir() else []
        anns = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else []
        segs = sorted(seg_dir.glob("*.jp*g")) if seg_dir.is_dir() else []
        ascs = sorted(d.rglob("*.asc"))

        n_images += len(imgs)
        n_annots += len(anns)
        n_seg += len(segs)
        n_asc += len(ascs)

        for img in imgs:
            parsed = parse_stem(img.stem)
            if parsed is None:
                bad_names.append(f"{d.name}/{img.name}")
                continue
            circuit, drawing, _pic = parsed
            circuits[circuit].add(d.name)
            drawings[circuit].add(drawing)

        for ann in anns:
            for cls, *_ in parse_voc(ann):
                class_counts[cls] += 1

        print(f"  {d.name:<14} {len(imgs):>5} images  {len(anns):>5} xml  "
              f"{len(segs):>4} seg  {len(ascs):>3} asc")

    print(f"\nTotals: {n_images} images, {n_annots} annotations, "
          f"{n_seg} segmentation maps, {n_asc} netlists")
    print(f"Distinct circuits: {len(circuits)}")

    # --- assumption checks -------------------------------------------------
    print("\n--- assumption checks ---")

    shared = {c: ds for c, ds in circuits.items() if len(ds) > 1}
    if shared:
        print(f"  {len(shared)} circuits appear under MORE THAN ONE drafter.")
        print("  (If this is large, the 'each circuit belongs to one drafter'")
        print("   assumption is wrong and the query/database split must change.)")
        for c, ds in list(shared.items())[:5]:
            print(f"      C{c}: {sorted(ds)}")
    else:
        print("  OK: every circuit belongs to exactly one drafter.")

    two_drawings = sum(1 for c, ds in drawings.items() if len(ds) >= 2)
    print(f"  {two_drawings}/{len(circuits)} circuits have 2+ drawings "
          f"(needed for the query/database split).")

    if bad_names:
        print(f"  {len(bad_names)} filenames did NOT match CX_DY_PZ, e.g.:")
        for b in bad_names[:5]:
            print(f"      {b}")
    else:
        print("  OK: all filenames match the CX_DY_PZ convention.")

    # --- classes -----------------------------------------------------------
    print(f"\n--- {len(class_counts)} component classes, most common ---")
    for cls, n in class_counts.most_common(25):
        print(f"  {cls:<32} {n:>7}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser())