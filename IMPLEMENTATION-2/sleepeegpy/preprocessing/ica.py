"""ICA preprocessing module (A2)."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import mne
from mne.preprocessing import ICA

from sleepeegpy.config.defaults import ICAConfig, DEFAULT_CONFIG

def apply_ica(
    raw: mne.io.Raw,
    config: Optional[ICAConfig] = None,
    picks: Optional[Union[str, list]] = None,
    components_to_remove: Optional[List[int]] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[mne.io.Raw, ICA, List[int], dict]:
    """
    Apply ICA for artifact removal (A2).
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data (preferably cleaned)
    config : ICAConfig, optional
        Configuration parameters
    picks : str or list, optional
        Channels to use for ICA
    components_to_remove : list, optional
        Indices of components to remove
    verbose : bool
        Whether to print progress
    **kwargs : dict
        Additional arguments for ICA
        
    Returns
    -------
    tuple (mne.io.Raw, ICA, list, dict)
        Reconstructed raw, ICA object, removed components, and info
    """
    if config is None:
        config = DEFAULT_CONFIG['ica']
    
    if verbose:
        print("Applying ICA...")
    
    # Create copy for ICA processing
    raw_ica = raw.copy()
    
    # Apply high-pass filter for ICA (recommended by MNE)
    if config.high_pass_ica is not None:
        if verbose:
            print(f"Applying high-pass filter for ICA: {config.high_pass_ica} Hz")
        raw_ica.filter(l_freq=config.high_pass_ica, h_freq=None, verbose=verbose)
    
    # Determine picks
    if picks is None:
        picks = mne.pick_types(raw_ica.info, eeg=True, exclude='bads')
    elif isinstance(picks, str):
        picks = mne.pick_channels_regexp(raw_ica.ch_names, picks)
    
    # Initialize ICA
    if verbose:
        print(f"Initializing ICA with {config.algorithm} algorithm, "
              f"{config.n_components} components")
    
    ica = ICA(
        n_components=config.n_components,
        method=config.algorithm,
        random_state=config.random_state,
        **kwargs
    )
    
    # Fit ICA
    if verbose:
        print("Fitting ICA...")
    ica.fit(raw_ica, picks=picks)
    
    # Remove components if specified
    if components_to_remove is None:
        components_to_remove = []
    
    if components_to_remove:
        if verbose:
            print(f"Removing {len(components_to_remove)} components: {components_to_remove}")
        raw_reconstructed = ica.apply(raw, exclude=components_to_remove)
    else:
        raw_reconstructed = raw.copy()
    
    ica_info = {
        'n_components': config.n_components,
        'algorithm': config.algorithm,
        'removed_components': components_to_remove,
        'picks': picks,
    }
    
    if verbose:
        print(f"ICA completed. {len(components_to_remove)} components removed.")
    
    return raw_reconstructed, ica, components_to_remove, ica_info

def find_artifact_components(
    ica: ICA,
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = None,
    eog_channel: Optional[str] = None,
    ecg_channel: Optional[str] = None,
    threshold: float = 3.0,
    verbose: bool = True
) -> Tuple[List[int], dict]:
    """
    Automatically find artifact components using EOG/ECG correlations.
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Original EEG data
    picks : str or list, optional
        Channels to use for detection
    eog_channel : str, optional
        EOG channel name
    ecg_channel : str, optional
        ECG channel name
    threshold : float
        Correlation threshold for artifact detection
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple (list, dict)
        List of artifact components and detection info
    """
    artifact_components = []
    detection_info = {}
    
    # Find EOG components
    if eog_channel is not None and eog_channel in raw.ch_names:
        if verbose:
            print(f"Finding EOG-related components using {eog_channel}")
        eog_indices, eog_scores = ica.find_bads_eog(
            raw, ch_name=eog_channel, threshold=threshold
        )
        artifact_components.extend(eog_indices)
        detection_info['eog_components'] = eog_indices
        detection_info['eog_scores'] = eog_scores
    
    # Find ECG components
    if ecg_channel is not None and ecg_channel in raw.ch_names:
        if verbose:
            print(f"Finding ECG-related components using {ecg_channel}")
        ecg_indices, ecg_scores = ica.find_bads_ecg(
            raw, ch_name=ecg_channel, threshold=threshold
        )
        artifact_components.extend(ecg_indices)
        detection_info['ecg_components'] = ecg_indices
        detection_info['ecg_scores'] = ecg_scores
    
    # Remove duplicates
    artifact_components = list(set(artifact_components))
    
    if verbose:
        print(f"Found {len(artifact_components)} artifact components: "
              f"{artifact_components}")
    
    return artifact_components, detection_info