#!/usr/bin/env python3
"""Regenerate all RAVEN results using full recordings (no truncation)."""

import json
from pathlib import Path
from RAVEN_python.main import RAVEN

# Find all PSG files
psg_dir = Path('physionet.org/files/sleep-edfx/1.0.0/sleep-cassette')
psg_files = sorted(psg_dir.glob('*PSG.edf'))

print(f"Found {len(psg_files)} PSG files")

results_dir = Path('RAVEN_python/results')
results_dir.mkdir(parents=True, exist_ok=True)

for i, psg_file in enumerate(psg_files, 1):
    result_name = psg_file.stem  # e.g., 'SC4001E0-PSG'
    result_dir = results_dir / result_name
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / 'results.json'
    
    print(f"[{i}/{len(psg_files)}] Analyzing {result_name}...")
    
    try:
        raven = RAVEN()
        if not raven.load_edf(str(psg_file)):
            print(f"  ERROR: Failed to load {psg_file}")
            continue
        
        # Run detectors
        raven.analyze()
        
        # Save results
        with open(result_file, 'w') as f:
            json.dump(raven.results, f, indent=2)
        
        print(f"  ✓ Saved to {result_file}")
        
        # Print summary
        for feature, events in raven.results.items():
            if isinstance(events, list):
                print(f"    {feature}: {len(events)} events")
    
    except Exception as e:
        print(f"  ERROR: {e}")

print("Done!")
