# utils/datalad_helper.py
import subprocess
import os
from pathlib import Path
from config import DATALAD_REPO_PATH, USE_SESSIONS

def check_subject_data(subject, session="ses-001"):
    """Check if subject has complete data for a session"""
    subject_path = Path(DATALAD_REPO_PATH) / subject / session / "eeg"
    if not subject_path.exists():
        return False
    
    # Check for required files
    ear_files = list(subject_path.glob(f"*{subject}_{session}_task-sleep_acq-earEEG_eeg.set"))
    psg_files = list(subject_path.glob(f"*{subject}_{session}_task-sleep_acq-PSG_eeg.set"))
    scoring_files = list(subject_path.glob(f"*{subject}_{session}_task-sleep_acq-scoring_events.tsv"))
    
    return len(ear_files) > 0 and len(psg_files) > 0 and len(scoring_files) > 0

def get_available_subjects():
    """Get list of all subjects with complete data"""
    if not DATALAD_REPO_PATH.exists():
        print(f"Warning: DataLad repository not found at {DATALAD_REPO_PATH}")
        print("Please update DATALAD_REPO_PATH in config.py")
        return []
    
    subjects_path = DATALAD_REPO_PATH
    available = []
    
    for subject_dir in sorted(subjects_path.glob("sub-*")):
        if subject_dir.is_dir():
            subject = subject_dir.name
            # Check if data exists for any session
            for session in USE_SESSIONS:
                if check_subject_data(subject, session):
                    available.append(subject)
                    break
    
    print(f"Found {len(available)} subjects with complete data")
    return available

def get_session_list(subject):
    """Get available sessions for a subject"""
    subject_path = DATALAD_REPO_PATH / subject
    if subject_path.exists():
        sessions = []
        for d in subject_path.glob("ses-*"):
            if d.is_dir():
                sessions.append(d.name)
        return sorted(sessions)
    return []

def get_data_size(subject):
    """Get approximate size of a subject's data"""
    subject_path = DATALAD_REPO_PATH / subject
    if subject_path.exists():
        try:
            result = subprocess.run(
                ["du", "-sh", str(subject_path)],
                capture_output=True,
                text=True
            )
            return result.stdout.split()[0]
        except:
            return "Unknown"
    return "Unknown"

def get_available_sessions(subject):
    """Get all available sessions for a subject"""
    subject_path = DATALAD_REPO_PATH / subject
    if subject_path.exists():
        sessions = []
        for session_dir in subject_path.glob("ses-*"):
            if session_dir.is_dir():
                sessions.append(session_dir.name)
        return sorted(sessions)
    return []