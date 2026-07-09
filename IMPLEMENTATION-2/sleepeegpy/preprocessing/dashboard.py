"""Dashboard visualization module."""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Optional, Tuple
from sleepeegpy.utils.hypnogram import STAGE_NAMES

def generate_dashboard(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    epoch_length: float = 30.0,
    picks: Optional[list] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate dashboard for preprocessing assessment.
    
    Creates a 2x3 grid showing:
    - Preprocessing info
    - Topographical power distribution
    - Spectrogram with hypnogram
    - PSD by sleep stage
    """
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True)[:1]
    
    # Use Pz if available
    if 'Pz' in raw.ch_names:
        channel_idx = raw.ch_names.index('Pz')
    else:
        channel_idx = picks[0]
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel a: General info
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_info(ax_a, raw, hypnogram, epoch_length)
    
    # Panel b: Topography
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_topography(ax_b, raw)
    
    # Panel c: Spectrogram with hypnogram
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_spectrogram(ax_c, raw, hypnogram, epoch_length, channel_idx)
    
    # Panel d: PSD by stage
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_psd_by_stage(ax_d, raw, hypnogram, epoch_length, channel_idx)
    
    # Hide empty subplots
    for idx in [0, 2]:  # Hide bottom corners
        ax = fig.add_subplot(gs[1, idx])
        ax.axis('off')
    
    fig.suptitle('SleepEEGpy Dashboard - Preprocessing Summary', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig

def _plot_info(ax, raw, hypnogram, epoch_length):
    """Plot preprocessing information."""
    ax.axis('off')
    
    info_text = [
        f"EEG Info:",
        f"  Channels: {len(raw.ch_names)}",
        f"  Sampling rate: {raw.info['sfreq']} Hz",
        f"  Duration: {raw.times[-1]/60:.1f} min",
        f"  File type: .fif",
    ]
    
    if hypnogram is not None:
        info_text.append("")
        info_text.append("Sleep Stages:")
        for stage_name, stage_id in STAGE_NAMES.items():
            duration = np.sum(hypnogram == stage_id) * epoch_length / 60
            if duration > 0:
                info_text.append(f"  {stage_name}: {duration:.1f} min")
    
    y_pos = 0.9
    for line in info_text:
        ax.text(0.05, y_pos, line, transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
        y_pos -= 0.05

def _plot_topography(ax, raw):
    """Plot a simple topomap."""
    try:
        # Compute alpha band power (8-12 Hz) for topography
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=8, fmax=12, verbose=False)
        alpha_power = np.mean(psd, axis=1)
        
        vmin = np.percentile(alpha_power, 5)
        vmax = np.percentile(alpha_power, 95)
        
        mne.viz.plot_topomap(alpha_power, raw.info, axes=ax, show=False,
                           vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_title('Alpha Power (8-12 Hz)')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')

def _plot_spectrogram(ax, raw, hypnogram, epoch_length, channel):
    """Plot spectrogram with hypnogram overlay."""
    from scipy.signal import spectrogram
    
    # Get channel data
    data, times = raw[channel, :]
    data = data.flatten()
    
    # Compute spectrogram
    fs = raw.info['sfreq']
    nperseg = int(4 * fs)
    f, t, Sxx = spectrogram(data, fs, nperseg=nperseg, noverlap=nperseg//2)
    
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12),
                  shading='gouraud', cmap='viridis')
    ax.set_ylim(0, 30)
    
    # Overlay hypnogram
    if hypnogram is not None:
        stage_times = np.arange(len(hypnogram)) * epoch_length
        max_stage = max(hypnogram) if len(hypnogram) > 0 else 4
        if max_stage > 0:
            hyp_norm = (hypnogram / max_stage) * 30
            ax.plot(stage_times, hyp_norm, 'w-', linewidth=1, alpha=0.7)
            ax.fill_between(stage_times, 0, hyp_norm, alpha=0.2, color='white')
    
    ax.set_title('Spectrogram with Hypnogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

def _plot_psd_by_stage(ax, raw, hypnogram, epoch_length, channel):
    """Plot PSD by sleep stage."""
    if hypnogram is None:
        ax.text(0.5, 0.5, 'Hypnogram required', transform=ax.transAxes,
                ha='center', va='center')
        return
    
    from sleepeegpy.analysis.spectral import compute_psd
    
    try:
        stage_psds = compute_psd(raw, hypnogram, verbose=False)
        
        stage_colors = {'wake': 'blue', 'n1': 'orange', 'n2': 'green',
                       'n3': 'red', 'rem': 'purple'}
        
        for stage, psd in stage_psds.items():
            if stage == 'full':
                continue
            stage_name = STAGE_NAMES.get(stage, f'Stage{stage}')
            color = stage_colors.get(stage_name, 'gray')
            
            # Get power at the selected channel
            ch_idx = raw.ch_names.index(raw.ch_names[channel])
            data = psd.data[ch_idx, :]
            freqs = psd.freqs
            
            ax.plot(freqs, 10 * np.log10(data + 1e-12),
                   color=color, label=stage_name.upper(), linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('PSD by Sleep Stage')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                ha='center', va='center')