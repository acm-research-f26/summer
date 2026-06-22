"""Frozen inference contract for iteration 1.

>>> from src.inference import predict
>>> predict("Only 2 left in stock!")
{
  "is_dark": 0.97,
  "categories": {
    "urgency": 0.05, "scarcity": 0.82, "social_proof": 0.04,
    "guilt_wording": 0.02, "other": 0.07
  },
  "top_category": "scarcity"
}

The three top-level keys, the five snake_case `categories` keys, and the value
types are FROZEN. Iteration 2 imports `predict` / `predict_batch` and must not
have to change. Renaming requires a contract bump.

Both fine-tuned DistilBERTs are lazy-loaded on the first call and cached at
module level (one binary, one 5-class). The label-int -> snake_case mapping
is read from `data/processed/label_maps.json`, never hardcoded, and asserted
on load against the frozen key set.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_LENGTH, PATHS

EXPECTED_CATEGORY_KEYS = {
    "urgency", "scarcity", "social_proof", "guilt_wording", "other",
}


class CategoryProbs(TypedDict):
    urgency: float
    scarcity: float
    social_proof: float
    guilt_wording: float
    other: float


class PredictResult(TypedDict):
    is_dark: float
    categories: CategoryProbs
    top_category: str


@lru_cache(maxsize=1)
def _resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(PATHS["label_maps"]) as f:
        label_maps = json.load(f)
    mc = label_maps["multiclass"]
    keys_by_id = [mc[str(i)]["key"] for i in range(len(mc))]
    found = set(keys_by_id)
    if found != EXPECTED_CATEGORY_KEYS:
        raise RuntimeError(
            f"label_maps.json multiclass keys {sorted(found)} != "
            f"frozen contract {sorted(EXPECTED_CATEGORY_KEYS)}"
        )

    bin_dir = PATHS["models"] / "distilbert_binary"
    mc_dir  = PATHS["models"] / "distilbert_multiclass"
    if not bin_dir.exists() or not mc_dir.exists():
        raise FileNotFoundError(
            f"Expected fine-tuned checkpoints at {bin_dir} and {mc_dir}. "
            "Run `python -m src.train_transformer` first."
        )

    bin_tok = AutoTokenizer.from_pretrained(str(bin_dir))
    bin_mdl = AutoModelForSequenceClassification.from_pretrained(str(bin_dir)).to(device).eval()
    mc_tok  = AutoTokenizer.from_pretrained(str(mc_dir))
    mc_mdl  = AutoModelForSequenceClassification.from_pretrained(str(mc_dir)).to(device).eval()

    return {
        "device": device,
        "keys_by_id": keys_by_id,
        "bin_tok": bin_tok, "bin_mdl": bin_mdl,
        "mc_tok": mc_tok,   "mc_mdl": mc_mdl,
    }


def _softmax(model, tok, texts: list[str], device: str) -> torch.Tensor:
    enc = tok(texts, truncation=True, padding=True,
              max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1)


def predict_batch(texts: list[str]) -> list[PredictResult]:
    """Batched form of `predict`. Same per-element output shape."""
    if not texts:
        return []
    r = _resources()
    bin_probs = _softmax(r["bin_mdl"], r["bin_tok"], list(texts), r["device"]).cpu().numpy()
    mc_probs  = _softmax(r["mc_mdl"],  r["mc_tok"],  list(texts), r["device"]).cpu().numpy()

    out: list[PredictResult] = []
    keys = r["keys_by_id"]
    for i in range(len(texts)):
        cats = {k: float(mc_probs[i][j]) for j, k in enumerate(keys)}
        out.append({  # type: ignore[typeddict-item]
            "is_dark": float(bin_probs[i][1]),
            "categories": cats,  # type: ignore[typeddict-item]
            "top_category": keys[int(mc_probs[i].argmax())],
        })
    return out


def predict(text: str) -> PredictResult:
    """Run both fine-tuned DistilBERTs and return the frozen contract dict."""
    return predict_batch([text])[0]
