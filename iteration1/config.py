"""Single source of truth for paths, seed, label schema, and hyperparameters.

Swapping the transformer checkpoint is a one-line change to MODEL_NAME.
"""
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

SEED = 42

PATHS = {
    "raw_mathur": ROOT / "data/raw/dark-patterns/data/final-dark-patterns/dark-patterns.csv",
    "raw_dataset_tsv": ROOT / "data/raw/ec-darkpattern/dataset/dataset.tsv",
    "augmentation": ROOT / "data/augmentation.csv",
    "processed": ROOT / "data/processed",
    "label_maps": ROOT / "data/processed/label_maps.json",
    "models": ROOT / "models",
    "reports": ROOT / "reports",
    "confusion_matrices": ROOT / "reports/confusion_matrices",
}

# Binary task: integer id -> snake_case key.
BINARY_LABELS = {0: "not_dark", 1: "dark"}

# Multi-class task: (integer id, snake_case key, display name).
# The snake_case keys are part of the FROZEN inference contract; do not rename.
MULTICLASS_LABELS = [
    (0, "urgency",       "Urgency"),
    (1, "scarcity",      "Scarcity"),
    (2, "social_proof",  "Social Proof"),
    (3, "guilt_wording", "Guilt-wording"),
    (4, "other",         "Other"),
]

# Transformer config — single knob to swap to roberta-base, etc.
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Stratified split fractions (per-task, applied independently).
TEST_FRAC = 0.15
VAL_FRAC = 0.15

# Baseline LogReg C grid for the val-only sweep.
BASELINE_C_GRID = [0.25, 1.0, 4.0]
