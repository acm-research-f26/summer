# utils/data_loader.py
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATALAD_REPO_PATH

def load_bids_ear_eeg(subject, session="ses-001", task="sleep"):
    """
    Load ear-EEG data from BIDS-formatted EESM23 dataset.
    """
    bids_root = DATALAD_REPO_PATH
    subject_path = Path(bids_root) / subject / session / "eeg"
    
    if not subject_path.exists():
        raise FileNotFoundError(f"Path not found: {subject_path}")
    
    # Find earEEG file
    ear_files = list(subject_path.glob(f"*{subject}_{session}_task-{task}_acq-earEEG_eeg.set"))
    if not ear_files:
        ear_files = list(subject_path.glob(f"*{subject}_{session}_*earEEG*.set"))
    
    if not ear_files:
        raise FileNotFoundError(f"No earEEG file found for {subject} {session}")
    
    ear_file = ear_files[0]
    print(f"Loading earEEG: {ear_file}")
    
    # Load using MNE
    raw = mne.io.read_raw_eeglab(ear_file, preload=True, verbose=False)
    
    # Find scoring file
    scoring_files = list(subject_path.glob(f"*{subject}_{session}_task-{task}_acq-scoring_events.tsv"))
    
    if not scoring_files:
        # Try PSGtrigger as fallback
        scoring_files = list(subject_path.glob(f"*{subject}_{session}_task-{task}_acq-PSGtrigger_events.tsv"))
    
    if not scoring_files:
        # Try any events file
        scoring_files = list(subject_path.glob(f"*{subject}_{session}_*events.tsv"))
    
    if not scoring_files:
        raise FileNotFoundError(f"No scoring file found for {subject} {session}")
    
    scoring_file = scoring_files[0]
    print(f"Found scoring file: {scoring_file}")
    
    # Load labels from scoring file
    labels = extract_labels_from_scoring_file(scoring_file)
    
    if labels is None:
        raise ValueError(f"Could not extract sleep stage labels from {scoring_file}")
    
    # Clean labels: map string labels to numeric values
    label_map = {
        'Wake': 0, 'W': 0,
        'N1': 1, 
        'N2': 2,
        'N3': 3,
        'REM': 5, 'R': 5
    }
    
    # Convert string labels to numeric
    numeric_labels = np.array([label_map.get(label, -1) for label in labels])
    
    # Remove any invalid labels
    valid_mask = numeric_labels >= 0
    numeric_labels = numeric_labels[valid_mask]
    
    if len(numeric_labels) == 0:
        raise ValueError(f"No valid sleep stage labels found in {scoring_file}")
    
    # Trim labels to match data length
    n_samples = len(raw)
    sfreq = raw.info['sfreq']
    n_epochs = int(n_samples // (30 * sfreq))
    
    if len(numeric_labels) > n_epochs:
        numeric_labels = numeric_labels[:n_epochs]
    elif len(numeric_labels) < n_epochs:
        # Pad if too short (shouldn't happen, but just in case)
        pad_len = n_epochs - len(numeric_labels)
        numeric_labels = np.concatenate([numeric_labels, np.zeros(pad_len, dtype=int)])
    
    print(f"Loaded {len(numeric_labels)} epochs from {subject} {session}")
    print(f"Stage distribution: {dict(zip(*np.unique(numeric_labels, return_counts=True)))}")
    
    return raw, numeric_labels

def extract_labels_from_scoring_file(file_path):
    """
    Extract sleep stage labels from a scoring events TSV file.
    The file has columns: onset, duration, scoring_idx, scoring
    """
    try:
        # Read as pandas DataFrame
        df = pd.read_csv(file_path, sep='\t')
        print(f"Reading labels from: {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have the expected columns
        if 'scoring' in df.columns:
            # Extract the scoring labels
            labels = df['scoring'].values.tolist()
            print(f"Extracted {len(labels)} labels from 'scoring' column")
            
            # Print unique labels to see what we have
            unique_labels = set(labels)
            print(f"Unique labels found: {sorted(unique_labels)}")
            
            return labels
        elif 'annotation' in df.columns:
            # Alternative: use annotation column
            labels = df['annotation'].values.tolist()
            print(f"Extracted {len(labels)} labels from 'annotation' column")
            return labels
        else:
            # Try to find any column with string values that look like sleep stages
            for col in df.columns:
                values = df[col].astype(str).values
                # Check if values look like sleep stages
                sleep_keywords = ['Wake', 'N1', 'N2', 'N3', 'REM', 'W', 'R']
                if any(any(kw in str(v) for kw in sleep_keywords) for v in values[:10]):
                    labels = values.tolist()
                    print(f"Extracted {len(labels)} labels from '{col}' column")
                    return labels
        
        return None
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_ear_eeg_channel(raw):
    """
    Find and return the best ear-EEG channel from the raw data.
    """
    # Common ear-EEG channel patterns in EESM23
    preferred_patterns = [
        'ELW-ERW', 'ERW-ELW',
        'ELW', 'ERW',
        'ELC', 'ERC',
        'ELC-ERC', 'ERC-ELC'
    ]
    
    available_channels = []
    
    for ch in raw.ch_names:
        ch_upper = ch.upper()
        for pattern in preferred_patterns:
            if pattern.upper() in ch_upper or ch_upper in pattern.upper():
                available_channels.append(ch)
                break
    
    if available_channels:
        chosen = available_channels[0]
        print(f"Selected ear-EEG channel: {chosen}")
        return chosen
    
    # If no preferred channel found, look for any channel with 'E' or 'ear'
    for ch in raw.ch_names:
        if 'E' in ch.upper() and ch not in ['ECG', 'EOG']:
            print(f"Using ear-related channel: {ch}")
            return ch
    
    # Fallback: use first channel
    print(f"No specific ear channel found. Using: {raw.ch_names[0]}")
    return raw.ch_names[0]