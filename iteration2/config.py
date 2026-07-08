
from pathlib import Path
import os

ROOT = Path(__file__).parent.resolve()
IT1_ROOT = ROOT.parent / "iteration1"

SEED = 42


CONTEXTDP_ROOT = Path(
    os.environ.get("CONTEXTDP_ROOT", ROOT / "AidUI/evaluation/evaluation_dataset")
)
OWN_SCREENSHOTS_ROOT = ROOT / "data/own_screenshots"

PATHS = {
    "cache_ocr": ROOT / "data/cache/ocr",
    "cache_vlm": ROOT / "data/cache/vlm",
    "reports": ROOT / "reports",
}

CATEGORIES = ["urgency", "scarcity", "social_proof", "guilt_wording", "other"]

LABEL_MAPPING = {
    "COUNTDOWN TIMER": "urgency",
    "LIMITED TIME MESSAGE": "urgency",
    "LOW STOCK MESSAGE": "scarcity",
    "HIGH DEMAND MESSAGE": "scarcity",
    "ACTIVITY MESSAGE": "social_proof",
    "DEFAULT CHOICE": "other",
    "NAGGING": "other",
    "DISGUISED ADS": "other",
    "GAMIFICATION": "other",
    "ATTENTION DISTRACTION": "other",
    "NO DP": None,
}

OCR_ENGINE = "easyocr"        # pip-only + GPU; tesseract needs an apt binary
OCR_MIN_CONFIDENCE = 0.4      # drop low-confidence easyocr blocks
OCR_MIN_CHARS = 3             # drop tiny fragments
OCR_MIN_ALPHA_CHARS = 2       # drop non-text garbage ("$$", "|-|")


AGGREGATION = "max_product"

VLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
# one-line swap target: "llava-hf/llava-v1.6-mistral-7b-hf"
VLM_LOAD_IN_4BIT = True
VLM_MAX_PIXELS = 1280 * 28 * 28   # ~1.0 MP cap on image tokens (Qwen processor)
VLM_MIN_PIXELS = 256 * 28 * 28
VLM_MAX_NEW_TOKENS = 200   # room for v3's one-sentence preamble + full JSON


PROMPT_VERSION = "v3"

# --- Evaluation ------------------------------------------------------------
THRESHOLD = 0.5               # score -> flag, both systems, no tuning
DEV_SUBSET_SIZE = 25          # prompt tuning allowed on these only
DEV_SUBSET_SEED = SEED
