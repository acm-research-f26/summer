"""Synthetic participants with a known planted effect, pushed through the real
instrument logging path and analysis pipeline, must recover the planted
relationship; a null effect must not produce a spurious one."""

import numpy as np
import pandas as pd
import pytest

from config import STIMULUS_SEVERITY_COLUMN
from src.analysis import run_analysis
from src.synthetic_participants import generate


@pytest.fixture
def stimuli_csv(tmp_path):
    n = 8
    sev = np.linspace(0.05, 0.9, n)
    df = pd.DataFrame({
        "screen_id": [f"S{i}" for i in range(n)],
        "source_dataset": ["contextdp"] * n,
        "platform": ["web"] * n,
        "manipulated_path": [f"/img/S{i}.png" for i in range(n)],
        "neutralized_path": [f"/img/S{i}_neutral.png" for i in range(n)],
        "severity_vision": sev,
        "severity_text": np.clip(0.8 * sev + 0.05, 0, 1),
        "severity_vision_alt": np.clip(sev * 1.05, 0, 1),
        "top_category_vision": ["other"] * n,
        "gt_guilt_wording": [0] * n,
        "selection_reason": ["fixture"] * n,
    })
    path = tmp_path / "stimuli.csv"
    df.to_csv(path, index=False)
    return path


def run_pipeline(tmp_path, stimuli_csv, effect_size, seed):
    responses = tmp_path / f"responses_{effect_size}_{seed}.csv"
    out_dir = tmp_path / f"reports_{effect_size}_{seed}"
    stimuli = pd.read_csv(stimuli_csv)
    generate(stimuli, n_participants=300, effect_size=effect_size,
             base_rate=0.3, noise_sd=0.05, seed=seed, responses_path=responses)
    res = run_analysis(responses_path=responses, stimuli_path=stimuli_csv,
                       out_dir=out_dir)
    return res, out_dir


def test_recovers_planted_effect(tmp_path, stimuli_csv):
    planted = 0.5
    res, out_dir = run_pipeline(tmp_path, stimuli_csv, planted, seed=42)
    prim = res[STIMULUS_SEVERITY_COLUMN]
    assert prim["slope"] > 0
    assert prim["p_value"] < 0.01
    assert prim["r2"] > 0.5
    assert abs(prim["slope"] - planted) < 0.15  # ~2 standard errors

    # reports exist and are unmistakably labeled synthetic
    md = (out_dir / "severity_vs_behavior.md").read_text()
    assert "SYNTHETIC" in md
    assert "NOT HUMAN RESULTS" in md
    assert (out_dir / "severity_vs_behavior.png").exists()


def test_null_effect_not_spuriously_recovered(tmp_path, stimuli_csv):
    res, _ = run_pipeline(tmp_path, stimuli_csv, effect_size=0.0, seed=11)
    prim = res[STIMULUS_SEVERITY_COLUMN]
    assert abs(prim["slope"]) < 0.1
    assert prim["p_value"] > 0.05


def test_analysis_requires_enough_screens(tmp_path, stimuli_csv):
    stimuli = pd.read_csv(stimuli_csv).head(2)
    small = tmp_path / "small_stimuli.csv"
    stimuli.to_csv(small, index=False)
    responses = tmp_path / "small_responses.csv"
    generate(stimuli, n_participants=20, effect_size=0.5, base_rate=0.3,
             noise_sd=0.05, seed=42, responses_path=responses)
    with pytest.raises(RuntimeError):
        run_analysis(responses_path=responses, stimuli_path=small,
                     out_dir=tmp_path / "small_reports")
