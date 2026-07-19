
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS, SEED, STIMULUS_SEVERITY_COLUMN, SYNTH_DEFAULTS
from src.instrument.storage import append_response, assign_conditions, new_response_id


def generate(
    stimuli: pd.DataFrame,
    n_participants: int,
    effect_size: float,
    base_rate: float,
    noise_sd: float,
    seed: int = SEED,
    responses_path: Path | None = None,
) -> int:
    """Simulate participants whose behavior depends on severity by a known,
    planted amount, logged through the real instrument path.

    Response model per participant-screen:
        P(decision=1) = clip(base_rate
                             + effect_size * severity * 1[manipulated]
                             + participant_offset + trial_noise, 0, 1)
    so the planted per-screen delta is effect_size * severity, and regressing
    delta on severity should recover slope ~= effect_size. Every row is
    written via storage.append_response with source="synthetic".
    """
    rng = np.random.default_rng(seed)
    screen_ids = stimuli["screen_id"].tolist()
    sev = dict(zip(stimuli["screen_id"], stimuli[STIMULUS_SEVERITY_COLUMN]))
    manip = dict(zip(stimuli["screen_id"], stimuli["manipulated_path"]))
    neut = dict(zip(stimuli["screen_id"], stimuli["neutralized_path"]))

    n_rows = 0
    for i in range(n_participants):
        pid = f"synth_{i:04d}"
        offset = rng.normal(0.0, noise_sd)
        conditions = assign_conditions(pid, screen_ids, seed)
        for sid in screen_ids:
            cond = conditions[sid]
            p = base_rate + offset + rng.normal(0.0, noise_sd)
            if cond == "manipulated":
                p += effect_size * sev[sid]
            decision = int(rng.random() < min(max(p, 0.0), 1.0))
            append_response(
                {
                    "response_id": new_response_id(),
                    "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "participant_id": pid,
                    "source": "synthetic",
                    "screen_id": sid,
                    "condition": cond,
                    "image_shown": manip[sid] if cond == "manipulated" else neut[sid],
                    "decision": decision,
                    "rt_ms": int(max(rng.normal(2500, 600), 300)),
                },
                path=responses_path,
            )
            n_rows += 1
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=SYNTH_DEFAULTS["n_participants"])
    ap.add_argument("--effect-size", type=float, default=SYNTH_DEFAULTS["effect_size"])
    ap.add_argument("--base-rate", type=float, default=SYNTH_DEFAULTS["base_rate"])
    ap.add_argument("--noise-sd", type=float, default=SYNTH_DEFAULTS["noise_sd"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--stimuli", type=Path, default=PATHS["stimuli"])
    ap.add_argument("--out", type=Path, default=PATHS["responses"])
    args = ap.parse_args()

    stimuli = pd.read_csv(args.stimuli)
    n_rows = generate(
        stimuli,
        n_participants=args.n,
        effect_size=args.effect_size,
        base_rate=args.base_rate,
        noise_sd=args.noise_sd,
        seed=args.seed,
        responses_path=args.out,
    )
    print(
        f"[synthetic] logged {n_rows} SYNTHETIC responses "
        f"({args.n} participants x {len(stimuli)} screens, planted "
        f"effect_size={args.effect_size}) -> {args.out}"
    )
    print("[synthetic] every row carries source=synthetic — these are NOT human data")


if __name__ == "__main__":
    main()
