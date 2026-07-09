"""Topographical visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
import mne

def plot_topographical_distribution(
    data: np.ndarray,
    info: mne.Info,
    title: str = '',
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    axes: Optional[plt.Axes] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot topographical distribution of EEG data.
    
    Parameters
    ----------
    data : np.ndarray
        Data to plot (n_channels,)
    info : mne.Info
        EEG info with channel positions
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color limits
    axes : plt.Axes, optional
        Axes to plot on
    show : bool
        Whether to show the plot
        
    Returns
    -------
    plt.Axes
        Axes with the plot
    """
    if axes is None:
        fig, axes = plt.subplots(figsize=(4, 3))
    
    # Determine color limits
    if vmin is None:
        vmin = np.percentile(data[~np.isnan(data)], 5)
    if vmax is None:
        vmax = np.percentile(data[~np.isnan(data)], 95)
    
    # Plot topomap
    im, _ = mne.viz.plot_topomap(
        data, info, axes=axes, show=False,
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    
    axes.set_title(title, fontsize=10)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return axes


def plot_stage_topographies(
    stage_psds: dict,
    info: mne.Info,
    band: Tuple[float, float],
    stage_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Plot topographies for different sleep stages.
    
    Parameters
    ----------
    stage_psds : dict
        Dictionary mapping stage to SpectrumArray
    info : mne.Info
        EEG info
    band : tuple
        Frequency band (fmin, fmax)
    stage_ids : list, optional
        Stages to include
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with topographies
    """
    if stage_ids is None:
        stage_ids = sorted([s for s in stage_psds.keys() if s != 'full'])
    
    n_stages = len(stage_ids)
    n_cols = min(4, n_stages)
    n_rows = int(np.ceil(n_stages / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Compute band power for each stage
    for idx, stage in enumerate(stage_ids):
        row = idx // n_cols
        col = idx % n_cols
        
        psd = stage_psds[stage]
        mask = (psd.freqs >= band[0]) & (psd.freqs <= band[1])
        band_power = np.mean(psd.data[:, mask], axis=1)
        
        # Plot topomap
        ax = axes[row, col]
        vmin = np.percentile(band_power, 5)
        vmax = np.percentile(band_power, 95)
        
        mne.viz.plot_topomap(
            band_power, info, axes=ax, show=False,
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        
        stage_name = STAGE_NAMES.get(stage, f'Stage {stage}')
        ax.set_title(f"{stage_name.upper()}\n{band[0]}-{band[1]} Hz", fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_stages, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig