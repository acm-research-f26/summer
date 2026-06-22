"""Fine-tune DistilBERT for both tasks.

Run:  python -m src.train_transformer

For multi-class, trains BOTH an unweighted and an inverse-frequency-weighted
variant and picks whichever wins val macro-F1; that variant is copied to the
canonical `models/distilbert_multiclass/` directory consumed by inference.
Both numbers are recorded in `winner.json` so the choice is documented.

The binary task is balanced post-dedup, so no weighting variant.

GPU is used when available; otherwise falls back to CPU (slow).
"""
from __future__ import annotations

import json
import shutil
from inspect import signature

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import (
    EPOCHS,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    MULTICLASS_LABELS,
    PATHS,
    SEED,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)

set_seed(SEED)


def _load_split(task: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PATHS["processed"] / f"{task}_{split}.parquet")


def _to_hf(df: pd.DataFrame, tok) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


class WeightedTrainer(Trainer):
    """Trainer that swaps the default loss for weighted CrossEntropyLoss."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _train_one(
    task: str,
    num_labels: int,
    class_weights: torch.Tensor | None,
    out_subdir: str,
) -> float:
    print(f"\n== DistilBERT: task={task} weights={'yes' if class_weights is not None else 'no'} ==")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = _to_hf(_load_split(task, "train"), tok)
    val_ds = _to_hf(_load_split(task, "val"), tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
    )

    out_dir = PATHS["models"] / out_subdir
    args = TrainingArguments(
        output_dir=str(out_dir / "_runs"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
        save_total_limit=1,
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics,
    )
    if "tokenizer" in signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tok

    if class_weights is not None:
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    print(f"  val metrics: {metrics}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    runs_dir = out_dir / "_runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir, ignore_errors=True)
    return float(metrics["eval_macro_f1"])


def train_binary() -> None:
    _train_one("binary", num_labels=2, class_weights=None,
               out_subdir="distilbert_binary")


def train_multiclass() -> None:
    n = len(MULTICLASS_LABELS)
    train = _load_split("multiclass", "train")
    counts = np.array([(train["label"] == i).sum() for i in range(n)], dtype=float)
    inv = counts.sum() / (n * counts)
    weights = torch.tensor(inv, dtype=torch.float32)
    print(f"\nmulticlass inverse-freq weights: "
          f"{ {k: round(float(inv[i]), 3) for i, k, _ in MULTICLASS_LABELS} }")

    f1_unw = _train_one("multiclass", n, None,
                        "distilbert_multiclass_unweighted")
    f1_wei = _train_one("multiclass", n, weights,
                        "distilbert_multiclass_weighted")
    winner = "weighted" if f1_wei > f1_unw else "unweighted"
    print(f"\nmulticlass val macro-F1: "
          f"unweighted={f1_unw:.4f}  weighted={f1_wei:.4f}  -> {winner}")

    src = PATHS["models"] / f"distilbert_multiclass_{winner}"
    dst = PATHS["models"] / "distilbert_multiclass"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with open(dst / "winner.json", "w") as f:
        json.dump({
            "winner": winner,
            "unweighted_val_macro_f1": f1_unw,
            "weighted_val_macro_f1": f1_wei,
        }, f, indent=2)
    print(f"copied {src.name} -> {dst.name}")


def main() -> None:
    train_binary()
    train_multiclass()


if __name__ == "__main__":
    main()
