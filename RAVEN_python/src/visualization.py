# File: src/visualization.py
"""
Visualization utilities for RAVEN
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

class RAVENVisualizer:
    """Visualization tools for RAVEN"""
    
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs
        self.time = np.arange(len(signal)) / fs
    
    def plot_signal_with_events(self, events, event_type, title=None):
        """Plot signal with detected events"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot signal
        ax.plot(self.time, self.signal * 1e6, 'b-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel('Amplitude (μV)')
        
        # Plot events
        colors = {'kcomplex': 'red', 'spindle': 'green', 'deltawave': 'orange'}
        color = colors.get(event_type, 'red')
        
        legend_added = False
        for event in events:
            ax.axvspan(event['start'], event['end'], 
                      alpha=0.3, color=color,
                      label=event_type.capitalize() if not legend_added else "")
            if not legend_added:
                legend_added = True
        
        ax.set_xlabel('Time (seconds)')
        ax.set_title(title or f'{event_type.capitalize()} Detection Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_summary(self, results):
        """Plot summary of all detections"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # Plot signal in μV
        axes[0].plot(self.time, self.signal * 1e6, 'k-', linewidth=0.5)
        axes[0].set_ylabel('EEG (μV)')
        axes[0].set_title('Raw EEG Signal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot events
        event_types = ['kcomplex', 'spindle', 'deltawave']
        colors = ['red', 'green', 'orange']
        labels = ['K-Complexes', 'Spindles', 'Delta Waves']
        
        for i, (event_type, color, label) in enumerate(zip(event_types, colors, labels)):
            events = results.get(event_type, [])
            
            if events:
                for event in events:
                    axes[i+1].axvspan(event['start'], event['end'], 
                                     alpha=0.3, color=color)
            axes[i+1].set_ylabel(label)
            axes[i+1].set_ylim([0, 1])
            axes[i+1].text(0.02, 0.5, f'n={len(events)}', 
                          transform=axes[i+1].transAxes,
                          verticalalignment='center')
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        return fig