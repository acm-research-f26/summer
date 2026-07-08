
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from config import (
    IT1_ROOT,
    OCR_MIN_ALPHA_CHARS,
    OCR_MIN_CHARS,
    OCR_MIN_CONFIDENCE,
    PATHS,
    ROOT,
)
from src.aggregate import aggregate_snippets

_READER = None  # easyocr model, loaded once


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ocr_image(path: str) -> list[dict]:
    """OCR one screenshot -> [{"text", "confidence"}, ...], disk-cached."""
    cache_file = PATHS["cache_ocr"] / f"{_file_sha256(path)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["blocks"]

    global _READER
    if _READER is None:
        import easyocr  # heavy import; only when a cache miss forces OCR

        _READER = easyocr.Reader(["en"])

    results = _READER.readtext(path, detail=1, paragraph=False)
    blocks = [{"text": text, "confidence": float(conf)} for _, text, conf in results]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"source": Path(path).name, "blocks": blocks}))
    return blocks


def filter_blocks(blocks: list[dict]) -> list[str]:
    """Drop empty/garbage blocks: too short, too few letters, low confidence."""
    kept = []
    for b in blocks:
        text = b["text"].strip()
        n_alpha = sum(ch.isalpha() for ch in text)
        if (
            len(text) >= OCR_MIN_CHARS
            and n_alpha >= OCR_MIN_ALPHA_CHARS
            and b["confidence"] >= OCR_MIN_CONFIDENCE
        ):
            kept.append(text)
    return kept


def run_it1(texts: list[str]) -> list[dict]:
    """One subprocess call to iteration 1's frozen predict_batch for all texts."""
    if not texts:
        return []
    with tempfile.TemporaryDirectory() as td:
        snippets = Path(td) / "snippets.json"
        predictions = Path(td) / "predictions.json"
        snippets.write_text(json.dumps({"texts": texts}))
        subprocess.run(
            [sys.executable, str(ROOT / "src/it1_worker.py"), str(snippets), str(predictions)],
            cwd=IT1_ROOT,
            check=True,
        )
        return json.loads(predictions.read_text())["predictions"]


def run_system_a(df: pd.DataFrame, dataset_name: str) -> dict[str, dict]:
    """Text baseline over all screens in `df`. Returns {filename: screen result}.

    Screen results follow the multi-label semantics documented in
    src/aggregate.py (independent per-category presence, NOT softmax).
    """
    t0 = time.time()
    texts_per_screen: dict[str, list[str]] = {}
    for i, row in enumerate(df.itertuples(), 1):
        texts_per_screen[row.filename] = filter_blocks(ocr_image(row.path))
        if i % 25 == 0 or i == len(df):
            rate = i / max(time.time() - t0, 1e-9) * 60
            print(f"[system A / ocr] {i}/{len(df)} screens ({rate:.1f} img/min)")

    # flatten -> one it1 call -> unflatten
    flat: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for fname, texts in texts_per_screen.items():
        spans[fname] = (len(flat), len(flat) + len(texts))
        flat.extend(texts)

    print(f"[system A / it1] classifying {len(flat)} snippets via {IT1_ROOT}")
    preds = run_it1(flat)

    results = {
        fname: aggregate_snippets(preds[lo:hi]) for fname, (lo, hi) in spans.items()
    }

    n_empty = sum(1 for texts in texts_per_screen.values() if not texts)
    print(f"[system A] done: {len(results)} screens, {n_empty} had no usable OCR text")

    out_file = PATHS["cache_ocr"] / f"system_a_{dataset_name}.json"
    out_file.write_text(json.dumps(results, indent=1))
    print(f"[system A] wrote {out_file}")
    return results
