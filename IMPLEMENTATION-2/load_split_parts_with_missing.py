import mne
from pathlib import Path

def load_single_part(filepath):
    """Load a single FIF part, ignoring split file errors."""
    try:
        # Try normal load
        return mne.io.read_raw_fif(filepath, preload=True)
    except ValueError as e:
        if "Split raw file detected" in str(e):
            # Use the Raw class directly with on_split_missing='ignore'
            print(f"  Loading {Path(filepath).name} with split ignore...")
            return mne.io.Raw(fname=filepath, preload=True, on_split_missing='ignore')
        raise

# Your parts
parts = [
    '/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/data/young_adult.fif',
    '/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/data/young_adult-1.fif',
    '/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/data/young_adult-3.fif',
]

# Load first part
print("Loading first part...")
raw = load_single_part(parts[0])
print(f"  Duration: {raw.times[-1]:.2f}s")

# Load and append remaining parts
for part_path in parts[1:]:
    print(f"\nLoading: {Path(part_path).name}")
    raw_part = load_single_part(part_path)
    if raw_part is not None:
        # Check channels match
        if len(raw_part.ch_names) == len(raw.ch_names):
            raw.append(raw_part)
            print(f"  Appended! Total: {raw.times[-1]/60:.1f} min")
        else:
            print(f"  Skipping - channel mismatch")

# Save
output = '/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/data/young_adult_combined.fif'
raw.save(output, overwrite=True)
print(f"\n✅ Combined file saved: {output}")
print(f"Total duration: {raw.times[-1]/60:.1f} minutes")