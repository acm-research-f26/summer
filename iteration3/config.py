
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
IT2_ROOT = ROOT.parent / "iteration2"

SEED = 42

CATEGORIES = ["urgency", "scarcity", "social_proof", "guilt_wording", "other"]

# Potency weights, derived from Luguri & Strahilevitz (2021). L&S report an
# ordering, not cardinal values; these numbers interpolate that ordering onto
# [0, 1]. "other" gets 0.5 rather than the top L&S weight because in this
# corpus the bucket is mostly preselected defaults and nagging, not the
# hidden-information/obstruction patterns L&S found most potent; at 0.9,
# severity would reduce to an "other" detector (187/546 screens carry the
# flag). Frozen — never tuned on results.
POTENCY_WEIGHTS = {
    "urgency": 0.2,
    "scarcity": 0.3,
    "social_proof": 0.5,
    "guilt_wording": 0.6,
    "other": 0.5,
}

# Sensitivity check: the naive mapping that takes L&S's top tier at face
# value for "other". All results also run under this vector.
POTENCY_WEIGHTS_ALT = {**POTENCY_WEIGHTS, "other": 0.9}

SEVERITY_FORMULA = "noisy_or"  # severity = 1 - prod_c(1 - w_c * s_c)

DETECTION_THRESHOLD = 0.5  # same fixed cutoff as iteration 2, no tuning

# Iteration-2 cached artifacts. v4 is the calibrated prompt and must match
# iteration2/config.PROMPT_VERSION; v3 is the frozen iteration-2 prompt kept
# as the calibration baseline.
IT2_PROMPT_VERSION = "v4"
IT2_PROMPT_VERSION_BASELINE = "v3"
IT2_CACHE_OCR = IT2_ROOT / "data/cache/ocr"
IT2_CACHE_VLM = IT2_ROOT / "data/cache/vlm"
AGGREGATE_FILES = {
    ("text", "contextdp"): IT2_CACHE_OCR / "system_a_contextdp.json",
    ("text", "own"): IT2_CACHE_OCR / "system_a_own.json",
    ("vision", "contextdp"): IT2_CACHE_VLM / f"system_b_contextdp_{IT2_PROMPT_VERSION}.json",
    ("vision", "own"): IT2_CACHE_VLM / f"system_b_own_{IT2_PROMPT_VERSION}.json",
}
BASELINE_VISION_AGGREGATES = {
    "contextdp": IT2_CACHE_VLM / f"system_b_contextdp_{IT2_PROMPT_VERSION_BASELINE}.json",
    "own": IT2_CACHE_VLM / f"system_b_own_{IT2_PROMPT_VERSION_BASELINE}.json",
}
# per-image VLM cache key suffix, for reconstructing a missing aggregate
VLM_CACHE_SUFFIX = f"_{IT2_PROMPT_VERSION}_qwen2-vl-7b-instruct.json"

EXPECTED_SCREENS = {"contextdp": 501, "own": 45}
MAX_MISSING_VISION = 5  # a couple of CONTEXTDP screens have no cached VLM result

PATHS = {
    "data": ROOT / "data",
    "reports": ROOT / "reports",
    "corpus_scores": ROOT / "data/corpus_scores.csv",
    "stimuli": ROOT / "data/stimuli.csv",
    "responses": ROOT / "data/responses/responses.csv",
    "neutralized": ROOT / "data/neutralized",
}

# Behavioral experiment (built this iteration, run in the fall)
N_STIMULI = 10
N_SEVERITY_BINS = 5
STIMULUS_SEVERITY_COLUMN = "severity_vision"  # the score being validated
MIN_SEVERITY_SPREAD = 0.15  # picked-set range below this is flagged degenerate
VISUAL_ONLY_PREFIX = "FIVE_"  # own-set purely-visual screens (hard constraint)

SYNTH_DEFAULTS = {
    # 800: the planted effect was only borderline-detectable at n=300 under
    # the compressed v3 severity range
    "n_participants": 800,
    "effect_size": 0.5,
    "base_rate": 0.3,
    "noise_sd": 0.05,
}
