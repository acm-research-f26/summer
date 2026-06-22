#!/usr/bin/env python3
"""Score RAVEN detection results against PhysioNet hypnogram annotations."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
HYPNOGRAM_ROOT = Path(__file__).resolve().parent.parent / 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'
EXPECTED_STAGES = {
    'kcomplex': {'Sleep stage 2'},
    'spindle': {'Sleep stage 2'},
    'deltawave': {'Sleep stage 3', 'Sleep stage 4'},
}
STAGE_ALIAS = {
    'Sleep stage 4': 'Sleep stage 3',
}


def load_json(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def normalize_stage(stage: str) -> str:
    stage = stage.strip()
    return STAGE_ALIAS.get(stage, stage)


def load_hypnogram(path: Path) -> List[Dict[str, object]]:
    try:
        import pyedflib
    except ImportError as exc:
        raise ImportError('pyedflib is required to load hypnogram EDF files') from exc

    with pyedflib.EdfReader(str(path)) as reader:
        starts, durations, labels = reader.readAnnotations()

    timeline: List[Dict[str, object]] = []
    for start, duration, label in zip(starts, durations, labels):
        if duration <= 0:
            continue
        timeline.append({
            'start': float(start),
            'end': float(start + duration),
            'label': normalize_stage(str(label)),
        })

    if not timeline:
        raise ValueError(f'No annotations found in hypnogram file: {path}')

    return sorted(timeline, key=lambda item: item['start'])


def find_hypnogram_file(result_name: str, hypnogram_root: Path) -> Optional[Path]:
    base_key = result_name.replace('-PSG', '')[:-1]
    candidates = [p for p in hypnogram_root.glob(f'{base_key}*Hypnogram.edf')]
    if len(candidates) == 1:
        return candidates[0]
    return None


def stage_at_time(timeline: Sequence[Dict[str, object]], t: float) -> Optional[str]:
    for stage in timeline:
        if stage['start'] <= t < stage['end']:
            return stage['label']
    if timeline and t >= timeline[-1]['end']:
        return timeline[-1]['label']
    return None


def event_overlaps_stages(event: Dict[str, object], timeline: Sequence[Dict[str, object]], expected_stages: set) -> bool:
    """Check if event overlaps with any expected stage."""
    event_start = float(event.get('start', 0.0))
    event_end = float(event.get('end', event_start))
    if event_end <= event_start:
        event_end = event_start + 0.1
    for stage in timeline:
        if stage['label'] in expected_stages:
            stage_start = stage['start']
            stage_end = stage['end']
            if event_start < stage_end and event_end > stage_start:
                return True
    return False


def event_proximity_score(event: Dict[str, object], timeline: Sequence[Dict[str, object]], expected_stages: set, proximity_window: float = 60.0) -> float:
    """Calculate score based on proximity to expected stages (0 to 1).
    
    Returns 1.0 if event overlaps with expected stage,
    decreases linearly with distance up to proximity_window seconds away.
    """
    event_start = float(event.get('start', 0.0))
    event_end = float(event.get('end', event_start))
    if event_end <= event_start:
        event_end = event_start + 0.1
    
    for stage in timeline:
        if stage['label'] in expected_stages:
            stage_start = stage['start']
            stage_end = stage['end']
            
            if event_start < stage_end and event_end > stage_start:
                return 1.0
            
            if event_end < stage_start:
                distance = stage_start - event_end
                if distance < proximity_window:
                    return 1.0 - (distance / proximity_window)
            elif event_start > stage_end:
                distance = event_start - stage_end
                if distance < proximity_window:
                    return 1.0 - (distance / proximity_window)
    
    return 0.0


def score_result_file(result_path: Path, hypnogram_path: Path) -> Dict[str, object]:
    results = load_json(result_path)
    timeline = load_hypnogram(hypnogram_path)

    feature_scores: Dict[str, object] = {}
    for feature, events in results.items():
        if not isinstance(events, list):
            raise ValueError(f'Feature {feature} in {result_path} is not a list')

        expected_stages = EXPECTED_STAGES.get(feature, set())
        total_events = len(events)
        matched_events = 0
        stage_counts: Dict[str, int] = {}

        for event in events:
            start = float(event.get('start', 0.0))
            stage = stage_at_time(timeline, start) or 'Unknown'
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            score_contribution = event_proximity_score(event, timeline, expected_stages, proximity_window=600.0)
            matched_events += score_contribution

        feature_score = None
        if total_events > 0:
            feature_score = matched_events / total_events

        feature_scores[feature] = {
            'total_events': total_events,
            'matched_events': matched_events,
            'score': feature_score,
            'stage_counts': stage_counts,
            'expected_stages': sorted(expected_stages),
            'note': f'Evaluated all detected events using overlap-based matching with expected stages',
        }

    valid_scores = [item['score'] for item in feature_scores.values() if item['score'] is not None]
    overall_score = mean(valid_scores) if valid_scores else None

    return {
        'result_path': str(result_path),
        'hypnogram_path': str(hypnogram_path),
        'overall_score': overall_score,
        'features': feature_scores,
    }


def summarize_scores(scores: Sequence[Dict[str, object]]) -> Dict[str, object]:
    valid_scores = [score['overall_score'] for score in scores if score['overall_score'] is not None]
    return {
        'file_count': len(scores),
        'mean_accuracy': mean(valid_scores) if valid_scores else None,
        'scored_files': len(valid_scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Score RAVEN results against PhysioNet hypnogram annotations')
    parser.add_argument('--results-dir', '-r', type=Path, default=RESULTS_ROOT, help='RAVEN results directory')
    parser.add_argument('--hypnogram-dir', '-g', type=Path, default=HYPNOGRAM_ROOT, help='PhysioNet hypnogram directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed stage counts for each file')
    args = parser.parse_args()

    if not args.results_dir.exists() or not args.results_dir.is_dir():
        print(f'Invalid results directory: {args.results_dir}')
        return 1
    if not args.hypnogram_dir.exists() or not args.hypnogram_dir.is_dir():
        print(f'Invalid hypnogram directory: {args.hypnogram_dir}')
        return 1

    scores: List[Dict[str, object]] = []
    missing: List[str] = []

    for result_dir in sorted(args.results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        result_file = result_dir / 'results.json'
        if not result_file.exists():
            missing.append(str(result_dir))
            continue

        hypnogram_file = find_hypnogram_file(result_dir.name, args.hypnogram_dir)
        if hypnogram_file is None:
            missing.append(str(result_dir.name))
            continue

        try:
            score = score_result_file(result_file, hypnogram_file)
            scores.append(score)
            if args.verbose:
                print(f"Scored {result_dir.name}: {score['overall_score']:.4f}")
                for feature, details in score['features'].items():
                    print(f"  {feature}: {details['matched_events']}/{details['total_events']} = {details['score']:.4f}" if details['score'] is not None else f"  {feature}: no events")
                    if args.verbose:
                        print(f"    expected stages: {details['expected_stages']}")
                        print(f"    stage counts: {details['stage_counts']}")
                print()
        except Exception as exc:
            print(f'Error scoring {result_dir.name}: {exc}')

    summary = summarize_scores(scores)
    print('---')
    print(f'Files scored: {len(scores)}')
    print(f'Overall mean accuracy: {summary["mean_accuracy"]:.4f}' if summary['mean_accuracy'] is not None else 'Overall mean accuracy: N/A')
    if missing:
        print(f'Missing or unmatched results: {len(missing)}')
        for item in missing:
            print(f'  {item}')

    return 0 if scores else 1


if __name__ == '__main__':
    raise SystemExit(main())
