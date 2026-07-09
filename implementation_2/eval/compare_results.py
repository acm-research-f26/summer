#!/usr/bin/env python3
"""
Aggregate baseline vs ARP trajectory files into a comparison report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_runs(runs_dir: Path, mode: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob(f"*_{mode}.json"))]


def aggregate(data: list[dict]) -> dict:
    if not data:
        return {"runs": 0, "success_rate": 0, "avg_steps": 0, "avg_irrelevant_clicks": 0}
    n = len(data)
    return {
        "runs": n,
        "success_rate": sum(1 for d in data if d.get("success")) / n,
        "avg_steps": sum(d["steps"] for d in data) / n,
        "avg_irrelevant_clicks": sum(d.get("irrelevant_clicks", 0) for d in data) / n,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    runs_dir = root / "results" / "runs"
    if not runs_dir.exists():
        print(f"No runs found in {runs_dir}")
        sys.exit(1)

    baseline = aggregate(load_runs(runs_dir, "baseline"))
    arp = aggregate(load_runs(runs_dir, "arp"))

    def pct(b, a):
        return round(100 * (b - a) / b, 1) if b else 0.0

    report = {
        "baseline": baseline,
        "arp": arp,
        "improvement": {
            "steps_reduction_pct": pct(baseline["avg_steps"], arp["avg_steps"]),
            "irrelevant_clicks_reduction_pct": pct(
                baseline["avg_irrelevant_clicks"], arp["avg_irrelevant_clicks"]
            ),
        },
    }

    out = root / "results" / "comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
