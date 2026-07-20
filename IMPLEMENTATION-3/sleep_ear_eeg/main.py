# main.py
import sys
import os

import inspect
if not hasattr(inspect, 'cleandoc'):
    import textwrap
    inspect.cleandoc = textwrap.dedent

# Add a workaround for the numpy issue
try:
    import numpy as np
except AttributeError as e:
    if "cleandoc" in str(e):
        print("Fixing numpy compatibility issue...")
        import inspect
        # Add cleandoc if it doesn't exist (backport for Python 3.10)
        if not hasattr(inspect, 'cleandoc'):
            import textwrap
            inspect.cleandoc = textwrap.dedent
        import numpy as np
    else:
        raise

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from utils.datalad_helper import get_available_subjects, get_data_size
from utils.data_loader import load_bids_ear_eeg, get_ear_eeg_channel
from utils.preprocessing import segment_and_extract_features
from utils.metrics import calculate_agreement_metrics
from models.sleep_stager import SleepStager
from config import STAGE_MAP, TARGET_SUBJECTS, DATALAD_REPO_PATH

def flatten_features(features_list):
    """Convert list of dictionaries to 2D feature array."""
    if not features_list:
        return np.array([])
    
    flat_features = []
    # Get the feature names from the first dictionary
    feature_names = list(features_list[0].keys())
    
    for epoch_feats in features_list:
        vals = [epoch_feats.get(name, 0.0) for name in feature_names]
        flat_features.append(vals)
    
    return np.array(flat_features)

def print_stage_distribution(y, title="Stage Distribution"):
    """Print the distribution of sleep stages"""
    if len(y) == 0:
        print("No data to display")
        return
    
    print(f"\n{title}:")
    total = len(y)
    unique_stages = sorted(np.unique(y))
    
    for stage in unique_stages:
        count = np.sum(y == stage)
        stage_name = STAGE_MAP.get(int(stage), f"Stage {stage}")
        print(f"  {stage_name}: {count:6d} epochs ({count/total*100:5.1f}%)")

def main():
    print("="*70)
    print("EESM23 SLEEP STAGING WITH EAR-EEG")
    print("="*70)
    
    # --- 1. Find available subjects ---
    print("\n[1] Scanning for available subjects...")
    
    # Get all available subjects with ses-001 data
    subjects = []
    for subject_dir in sorted((DATALAD_REPO_PATH).glob("sub-*")):
        if subject_dir.is_dir():
            subject = subject_dir.name
            # Check if ses-001 exists
            session_path = DATALAD_REPO_PATH / subject / "ses-001" / "eeg"
            if session_path.exists():
                # Check for earEEG file and scoring file
                ear_files = list(session_path.glob(f"*{subject}_ses-001_*earEEG*.set"))
                scoring_files = list(session_path.glob(f"*{subject}_ses-001_*scoring_events.tsv"))
                if ear_files and scoring_files:
                    subjects.append(subject)
    
    if not subjects:
        print("ERROR: No subjects found with complete ses-001 data.")
        print(f"Please check that {DATALAD_REPO_PATH} contains the EESM23 dataset.")
        return
    
    # Limit to TARGET_SUBJECTS if specified
    if TARGET_SUBJECTS and len(subjects) > TARGET_SUBJECTS:
        subjects = subjects[:TARGET_SUBJECTS]
    
    print(f"Using {len(subjects)} subjects with ses-001 data:")
    for i, subject in enumerate(subjects, 1):
        size = get_data_size(subject)
        print(f"  {i:2d}. {subject} (size: {size})")
    
    # --- 2. Load data from all subjects ---
    print("\n[2] Loading EEG data...")
    all_features = []
    all_labels = []
    subject_epoch_counts = {}
    failed_subjects = []
    
    for subject_idx, subject in enumerate(subjects, 1):
        subject_epochs = 0
        session = "ses-001"  # Only use first session
        
        try:
            print(f"\n  Processing {subject}, {session}...")
            
            # Load earEEG data and labels
            raw, labels = load_bids_ear_eeg(subject, session)
            
            # Get the best ear-EEG channel
            ear_channel = get_ear_eeg_channel(raw)
            print(f"Using channel: {ear_channel}")
            
            # Extract features from ear-EEG
            features_by_epoch = segment_and_extract_features(raw, ear_channel)
            
            if not features_by_epoch:
                print(f"    Warning: No features extracted for {subject}")
                failed_subjects.append(subject)
                continue
            
            X = flatten_features(features_by_epoch)
            
            # Match labels to features
            min_len = min(len(X), len(labels))
            if min_len > 0:
                X = X[:min_len]
                y = labels[:min_len]
                
                # Remove any labels that are not in STAGE_MAP
                valid_stages = list(STAGE_MAP.keys())
                valid_mask = np.isin(y, valid_stages)
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) > 0:
                    all_features.append(X)
                    all_labels.append(y)
                    subject_epochs = len(X)
                    subject_epoch_counts[subject] = subject_epochs
                    print(f"    Added {subject_epochs} epochs")
                    # Print stage distribution for this subject
                    unique, counts = np.unique(y, return_counts=True)
                    stage_dist = {STAGE_MAP.get(int(s), f"Stage{s}"): c for s, c in zip(unique, counts)}
                    print(f"    Stages: {stage_dist}")
                else:
                    print(f"    Warning: No valid epochs after cleaning for {subject}")
                    failed_subjects.append(subject)
            else:
                print(f"    Warning: No matching epochs for {subject}")
                failed_subjects.append(subject)
                
        except Exception as e:
            print(f"    Error processing {subject}: {e}")
            failed_subjects.append(subject)
            continue
    
    if not all_features:
        print("\nERROR: No data loaded. Please check the dataset structure.")
        print("Troubleshooting tips:")
        print("  1. Verify the dataset path: ", DATALAD_REPO_PATH)
        print("  2. Check that you have downloaded the data with: datalad get sub-*/ses-001/eeg/")
        print("  3. Check that scoring files exist and have the correct format")
        return
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"\n[3] Data loaded successfully!")
    print(f"  Successful subjects: {len(subjects) - len(failed_subjects)} of {len(subjects)}")
    if failed_subjects:
        print(f"  Failed subjects: {', '.join(failed_subjects)}")
    print(f"  Total epochs: {len(X)}")
    print(f"  Features per epoch: {X.shape[1]}")
    
    # Print subject breakdown
    print("\n  Epochs per subject:")
    for subject, count in subject_epoch_counts.items():
        print(f"    {subject}: {count} epochs")
    
    print_stage_distribution(y, "Overall Stage Distribution")
    
    # --- 4. Train-test split ---
    print("\n[4] Training model...")
    
    # Use stratified split to maintain stage distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} epochs")
    print(f"  Test set: {len(X_test)} epochs")
    
    # --- 5. Train model ---
    stager = SleepStager()
    stager.train(X_train, y_train)
    
    # --- 6. Evaluate ---
    print("\n[5] Evaluating model...")
    y_pred = stager.predict(X_test)
    metrics = calculate_agreement_metrics(y_test, y_pred)
    
    # --- 7. Print results ---
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"  F1 Score:      {metrics['f1_score']:.4f}")
    
    # Compare to paper
    paper_kappa = 0.76
    print(f"\n  Paper's reported κ: {paper_kappa}")
    if metrics['cohen_kappa'] >= paper_kappa * 0.95:
        print(f"  ✓ Model matches the paper's performance!")
    elif metrics['cohen_kappa'] >= paper_kappa * 0.85:
        print(f"  ~ Model close to paper's performance")
    else:
        print(f"  - Model below paper's performance. Consider:")
        print(f"    1. Adding more subjects")
        print(f"    2. Using more features (power spectral density)")
        print(f"    3. Trying a more complex model")
    
    # Stage-specific performance
    print("\nStage-Specific Performance:")
    unique_stages = sorted(np.unique(y_test))
    
    # Get confusion matrix
    cm = metrics['confusion_matrix']
    
    for idx, stage_idx in enumerate(unique_stages):
        stage_name = STAGE_MAP.get(stage_idx, f"Stage {stage_idx}")
        
        # Calculate per-class metrics
        tp = cm[idx, idx] if idx < len(cm) else 0
        fp = cm[:, idx].sum() - tp if idx < len(cm) else 0
        fn = cm[idx, :].sum() - tp if idx < len(cm) else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {stage_name:8s}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Confusion matrix
    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    stage_names = [STAGE_MAP.get(s, str(s)) for s in unique_stages]
    print("  " + " | ".join(stage_names))
    print("-" * 50)
    for i, row in enumerate(cm):
        if i < len(unique_stages):
            print(f"{stage_names[i]:8s}: " + " | ".join([f"{val:6d}" for val in row]))
    
    # --- 8. Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Subjects attempted:      {len(subjects)}")
    print(f"  Successful subjects:     {len(subjects) - len(failed_subjects)}")
    print(f"  Total epochs:            {len(X)}")
    print(f"  Training epochs:         {len(X_train)}")
    print(f"  Test epochs:             {len(X_test)}")
    print(f"  Accuracy:                {metrics['accuracy']:.4f}")
    print(f"  Cohen's Kappa:           {metrics['cohen_kappa']:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['accuracy', 'cohen_kappa', 'f1_score', 'subjects_used', 'total_epochs'],
        'value': [
            metrics['accuracy'], 
            metrics['cohen_kappa'], 
            metrics['f1_score'],
            len(subjects) - len(failed_subjects),
            len(X)
        ]
    })
    results_df.to_csv('results.csv', index=False)
    print(f"\n  Results saved to: results.csv")
    print("="*70)

if __name__ == "__main__":
    from config import DATALAD_REPO_PATH
    main()