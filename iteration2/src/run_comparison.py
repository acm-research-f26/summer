
from __future__ import annotations

import argparse

from config import DEV_SUBSET_SIZE, PATHS, THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["contextdp", "own"], default="contextdp")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"run on the fixed dev subset of N screens "
                         f"(stratified, seeded; {DEV_SUBSET_SIZE} = the documented dev set)")
    ap.add_argument("--system", choices=["a", "b", "both"], default="both")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    if args.dataset == "contextdp":
        from src.dataset_contextdp import label_distribution, load_contextdp, select_dev_subset
        truth = load_contextdp()
        print(label_distribution(truth).to_string(index=False))
        if args.limit:
            truth = select_dev_subset(truth, size=args.limit)
            dev_file = PATHS["reports"] / "dev_subset.txt"
            dev_file.parent.mkdir(parents=True, exist_ok=True)
            dev_file.write_text("\n".join(truth["filename"]) + "\n")
            print(f"[run] dev subset of {len(truth)} screens (list -> {dev_file})")
    else:
        from src.dataset_own import load_own
        truth = load_own()
        if args.limit:
            truth = truth.head(args.limit)
        print(f"[run] own dataset: {len(truth)} screens")

    name = args.dataset if not args.limit else f"{args.dataset}_dev{len(truth)}"

    preds_a = preds_b = None
    if args.system in ("a", "both"):
        from src.ocr_baseline import run_system_a
        preds_a = run_system_a(truth, name)
    if args.system in ("b", "both"):
        from src.vlm_model import run_system_b
        preds_b = run_system_b(truth, name)

    if preds_a and preds_b:
        from src.evaluate import write_error_gallery, write_headline_report
        from src.ocr_baseline import filter_blocks, ocr_image

        write_headline_report(truth, preds_a, preds_b, name, args.threshold)

        # gallery context: OCR'd text for A (cache hit, free), raw reply for B
        ocr_text = {
            r.filename: "\n".join(filter_blocks(ocr_image(r.path)))
            for r in truth.itertuples()
        }
        raw_b = {f: p.get("raw_response", "") for f, p in preds_b.items()}
        write_error_gallery(truth, preds_a, "text", extra=ocr_text,
                            threshold=args.threshold)
        write_error_gallery(truth, preds_b, "vision", extra=raw_b,
                            threshold=args.threshold)
    elif preds_a or preds_b:
        from src.evaluate import score_system
        only, label = (preds_a, "text") if preds_a else (preds_b, "vision")
        m = score_system(truth, only, args.threshold)
        print(f"[run] {label}-only metrics: micro-F1 {m['micro_f1']:.3f}, "
              f"macro-F1 {m['macro_f1']:.3f} (over {m['macro_over']}), "
              f"is_dark acc {m['is_dark_accuracy']:.3f}")


if __name__ == "__main__":
    main()
