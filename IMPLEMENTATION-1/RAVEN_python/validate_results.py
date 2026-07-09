#!/usr/bin/env python3
"""Validate RAVEN result JSON files."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_EVENT_KEYS = {'start', 'end', 'duration', 'confidence'}
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def load_results_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def validate_event(event: Any, path: Path, index: int, tolerance: float = 1e-6) -> None:
    if not isinstance(event, dict):
        raise ValueError(f"Event at index {index} in {path} is not an object")

    missing_keys = REQUIRED_EVENT_KEYS - event.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys {sorted(missing_keys)} in event {index} of {path}")

    start = event['start']
    end = event['end']
    duration = event['duration']
    confidence = event['confidence']

    if not isinstance(start, (int, float)):
        raise ValueError(f"Invalid start value in event {index} of {path}: {start}")
    if not isinstance(end, (int, float)):
        raise ValueError(f"Invalid end value in event {index} of {path}: {end}")
    if not isinstance(duration, (int, float)):
        raise ValueError(f"Invalid duration value in event {index} of {path}: {duration}")
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Invalid confidence value in event {index} of {path}: {confidence}")

    if start < 0:
        raise ValueError(f"Event {index} in {path} has negative start: {start}")
    if end <= start:
        raise ValueError(f"Event {index} in {path} has end <= start: {start} >= {end}")
    if duration <= 0:
        raise ValueError(f"Event {index} in {path} has non-positive duration: {duration}")

    if abs((end - start) - duration) > tolerance:
        raise ValueError(
            f"Event {index} in {path} has inconsistent duration: end-start={end - start} vs duration={duration}"
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Event {index} in {path} has confidence outside [0,1]: {confidence}"
        )


def validate_results_item(feature: str, events: Any, path: Path) -> None:
    if not isinstance(events, list):
        raise ValueError(f"Feature '{feature}' in {path} is not a list")

    previous_start: Optional[float] = None
    for index, event in enumerate(events):
        validate_event(event, path, index)
        start = event['start']
        if previous_start is not None and start < previous_start:
            raise ValueError(
                f"Events for feature '{feature}' in {path} are not sorted by start time at index {index}"
            )
        previous_start = start


def validate_results_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Results file is empty: {path}")

    data = load_results_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Root of {path} must be a JSON object")

    if not data:
        raise ValueError(f"Results file {path} contains no features")

    for feature, events in data.items():
        if not isinstance(feature, str):
            raise ValueError(f"Feature key is not a string in {path}: {feature}")
        validate_results_item(feature, events, path)


def find_results_files(root: Path = DEFAULT_RESULTS_DIR) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Results root folder not found: {root}")

    result_files = sorted(root.rglob('results.json'))
    if not result_files:
        raise FileNotFoundError(f"No results.json files found below: {root}")
    return result_files


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Validate RAVEN results JSON files')
    parser.add_argument(
        '--results-dir', '-r', type=Path, default=DEFAULT_RESULTS_DIR,
        help='Path to the RAVEN results folder'
    )
    args = parser.parse_args()

    result_files = find_results_files(args.results_dir)
    errors: List[str] = []

    for path in result_files:
        try:
            validate_results_file(path)
            print(f"OK: {path}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    print(f"\nValidated {len(result_files)} results.json files")
    if errors:
        print("Errors:")
        for error in errors:
            print(f" - {error}")
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
