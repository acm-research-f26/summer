
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from config import (
    CATEGORIES,
    PATHS,
    PROMPT_VERSION,
    VLM_LOAD_IN_4BIT,
    VLM_MAX_NEW_TOKENS,
    VLM_MAX_PIXELS,
    VLM_MIN_PIXELS,
    VLM_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# The rubric. Editing ANY of this text requires bumping config.PROMPT_VERSION.
# ---------------------------------------------------------------------------
RUBRIC = """\
You are auditing a UI screenshot for dark patterns - design tricks that pressure or \
mislead users. Look at BOTH the text and the visual design (colors, sizes, layout, \
preselected controls, what is emphasized vs hidden).

Score each category independently from 0.0 to 1.0 for whether it is PRESENT anywhere \
on this screen:
- "urgency": countdown timers, deal clocks, "ends tonight" / "limited time" deadline text.
- "scarcity": low-stock warnings ("only 2 left!"), stock meters, high-demand claims \
("selling fast", "in 20 carts", "N people viewing").
- "social_proof": recent-activity popups ("Alex from Ohio just bought this"), \
purchase-count or testimonial banners used as pressure.
- "guilt_wording": shame-laden decline options ("No thanks, I hate saving money"), \
guilt-tripping opt-out or unsubscribe phrasing.
- "other": any other manipulation - preselected checkboxes or plans, nagging popups \
blocking content, ads disguised as content or as system messages, false hierarchy \
(huge bright accept button vs tiny grey/low-contrast decline), hidden costs or fine \
print, forced signup, gamified reward wheels.

Also output "is_dark": the probability (0.0-1.0) that at least one dark pattern is \
present. Screens with none of these get low scores everywhere.

First, in one short sentence, note the most relevant thing you actually observe about \
dark patterns on THIS screen (or that you see none). Then, on a new line, output ONLY \
a JSON object in exactly this key structure, replacing every <...> placeholder with a \
real float between 0.0 and 1.0 based on YOUR OWN observation above - the angle-bracket \
placeholders are not valid values and must never appear literally in your answer:
{"is_dark": <is_dark_score>, "categories": {"urgency": <urgency_score>, \
"scarcity": <scarcity_score>, "social_proof": <social_proof_score>, \
"guilt_wording": <guilt_wording_score>, "other": <other_score>}}"""

RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object, "
    "nothing else."
)

_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load the VLM once; resident in GPU memory for the whole run."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    quant = None
    if VLM_LOAD_IN_4BIT:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    print(f"[system B] loading {VLM_MODEL_NAME} (4bit={VLM_LOAD_IN_4BIT}) ...")
    _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME, quantization_config=quant, device_map="auto"
    )
    # min/max_pixels bound the image-token count so ~1800px screenshots
    # don't blow past Colab memory or crawl.
    _PROCESSOR = AutoProcessor.from_pretrained(
        VLM_MODEL_NAME, min_pixels=VLM_MIN_PIXELS, max_pixels=VLM_MAX_PIXELS
    )
    print("[system B] model resident")


def _generate(image_path: str, prompt: str) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = _PROCESSOR(text=[text], images=image_inputs, return_tensors="pt").to(_MODEL.device)
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return _PROCESSOR.decode(trimmed, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Defensive parsing: fences -> json.loads -> (caller retries) -> regex -> None
# ---------------------------------------------------------------------------

def _clamp(x) -> float:
    return min(1.0, max(0.0, float(x)))


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # keep only the outermost {...} if there's prose around it
    lo, hi = s.find("{"), s.rfind("}")
    if lo != -1 and hi > lo:
        s = s[lo : hi + 1]
    return s


def parse_vlm_response(raw: str) -> dict | None:

    try:
        data = json.loads(_strip_fences(raw))
        cats_in = data.get("categories", {})
        cats = {c: _clamp(cats_in.get(c, 0.0)) for c in CATEGORIES}
        return {
            "is_dark": _clamp(data.get("is_dark", max(cats.values()))),
            "categories": cats,
            "top_category": max(cats, key=cats.get),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # regex fallback: '"urgency": 0.7' anywhere in the raw text
    cats, found = {}, 0
    for c in CATEGORIES:
        m = re.search(rf'"?{c}"?\s*[:=]\s*([01](?:\.\d+)?|\.\d+)', raw)
        cats[c] = _clamp(m.group(1)) if m else 0.0
        found += bool(m)
    if not found:
        return None
    m = re.search(r'"?is_dark"?\s*[:=]\s*([01](?:\.\d+)?|\.\d+)', raw)
    return {
        "is_dark": _clamp(m.group(1)) if m else max(cats.values()),
        "categories": cats,
        "top_category": max(cats, key=cats.get),
    }


# ---------------------------------------------------------------------------
# Cached per-image prediction + full-run driver
# ---------------------------------------------------------------------------

def _cache_file(image_path: str) -> Path:
    img_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    model_slug = VLM_MODEL_NAME.split("/")[-1].lower()
    return PATHS["cache_vlm"] / f"{img_hash}_{PROMPT_VERSION}_{model_slug}.json"


def predict_image(image_path: str) -> dict:

    cache = _cache_file(image_path)
    if cache.exists():
        return json.loads(cache.read_text())

    _load_model()
    raw = _generate(image_path, RUBRIC)
    result = parse_vlm_response(raw)
    raw_retry = None
    if result is None:
        raw_retry = _generate(image_path, RUBRIC + RETRY_SUFFIX)
        result = parse_vlm_response(raw_retry)

    if result is None:
        result = {
            "is_dark": 0.5,
            "categories": {c: 0.5 for c in CATEGORIES},
            "top_category": "other",
        }
        parse_failed = True
    else:
        parse_failed = False

    record = {
        **result,
        "parse_failed": parse_failed,
        "raw_response": raw,
        **({"raw_retry_response": raw_retry} if raw_retry is not None else {}),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, indent=1))
    return record


def run_system_b(df: pd.DataFrame, dataset_name: str) -> dict[str, dict]:

    results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    t0, n_fresh = time.time(), 0

    for i, row in enumerate(df.itertuples(), 1):
        was_cached = _cache_file(row.path).exists()
        rec = predict_image(row.path)
        results[row.filename] = rec
        if rec.get("parse_failed"):
            failures.append((row.filename, rec.get("raw_response", "")))

        n_fresh += not was_cached
        elapsed = time.time() - t0
        if not was_cached or i == len(df):
            rate = n_fresh / max(elapsed, 1e-9) * 60 if n_fresh else float("inf")
            remaining = len(df) - i
            eta_min = remaining / rate if rate > 0 else 0
            print(
                f"[system B] {i}/{len(df)} {row.filename}"
                f"{' (cached)' if was_cached else ''} | {rate:.1f} img/min"
                f" | ~{eta_min:.0f} min left"
            )

    if failures:
        report = PATHS["reports"] / "vlm_parse_failures.md"
        with open(report, "a") as f:
            f.write(f"\n## {dataset_name} — prompt {PROMPT_VERSION} — {time.ctime()}\n\n")
            for fname, raw in failures:
                f.write(f"### {fname}\n```\n{raw[:1500]}\n```\n\n")
        print(f"[system B] {len(failures)} parse failures logged to {report}")

    out_file = PATHS["cache_vlm"] / f"system_b_{dataset_name}_{PROMPT_VERSION}.json"
    out_file.write_text(json.dumps(results, indent=1))
    print(f"[system B] done: {len(results)} screens -> {out_file}")
    return results
