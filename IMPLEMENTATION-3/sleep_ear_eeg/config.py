# config.py
import os
from pathlib import Path

# Get the project root directory (where this config file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# DataLad repository path (ds005178 is at the same level as sleep_ear_eeg)
DATALAD_REPO_PATH = PROJECT_ROOT.parent / "ds005178"

# If your ds005178 is in a different location, update this path
# DATALAD_REPO_PATH = Path("/path/to/your/ds005178")

# Use only session 001 for consistency
USE_SESSIONS = ["ses-001"]

# Subject list - will be auto-detected
SUBJECTS = None  # Will be populated in main.py

# Sampling parameters
SAMPLING_RATE = 250  # Hz
EPOCH_SECONDS = 30
EPOCH_SAMPLES = EPOCH_SECONDS * SAMPLING_RATE

# Frequency bands for feature extraction
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# AASM sleep stage mapping
# EESM23 uses: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=REM
STAGE_MAP = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
    5: 'REM'
}

# Minimum number of subjects needed
MIN_SUBJECTS = 3
TARGET_SUBJECTS = 10  # Use all available

