# File: main.py (in root directory)
"""
RAVEN: Sleep Microstructure Analysis - Main Entry Point
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Add src directory to Python path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import from src
from signal_processing import SignalProcessor
from k_complex_detection import KComplexDetector
from spindle_detection import SpindleDetector
from delta_wave_detection import DeltaWaveDetector
from visualization import RAVENVisualizer

import matplotlib.pyplot as plt
import numpy as np

class RAVEN:
    """Main RAVEN class"""
    
    def __init__(self, fs=256):
        self.fs = fs
        self.signal = None
        self.results = {}
        self.detectors = {
            'kcomplex': KComplexDetector(fs),
            'spindle': SpindleDetector(fs),
            'deltawave': DeltaWaveDetector(fs)
        }
    
    def load_edf(self, filepath):
        """Load EDF file"""
        try:
            import pyedflib
            with pyedflib.EdfReader(filepath) as f:
                signal_labels = f.getSignalLabels()
                
                # Find EEG channels
                eeg_channels = []
                for i, label in enumerate(signal_labels):
                    if any(x in label for x in ['C', 'EEG', 'F', 'P', 'O']):
                        eeg_channels.append(i)
                
                if not eeg_channels:
                    eeg_channels = [0]
                
                self.signal = f.readSignal(eeg_channels[0])
                self.fs = f.getSampleFrequency(eeg_channels[0])
                
                print(f"Loaded: {len(self.signal)} samples at {self.fs} Hz")
                print(f"Channel: {signal_labels[eeg_channels[0]]}")
                return True
        except ImportError:
            print("pyedflib not installed. Install: pip install pyedflib")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def load_csv(self, filepath, fs=256):
        """Load CSV file"""
        try:
            import pandas as pd
            data = pd.read_csv(filepath)
            if data.shape[1] >= 2:
                self.signal = data.iloc[:, 1].values
                self.fs = fs
                print(f"Loaded CSV: {len(self.signal)} samples at {self.fs} Hz")
                return True
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def analyze(self, features=['kcomplex', 'spindle', 'deltawave']):
        """Run analysis"""
        if self.signal is None:
            raise ValueError("No signal loaded")
        
        print("\n" + "="*50)
        print("RAVEN Analysis")
        print(f"Features: {features}")
        print(f"Samples: {len(self.signal)}")
        print("="*50 + "\n")
        
        for feature in features:
            if feature not in self.detectors:
                continue
            
            print(f"Detecting {feature.capitalize()}...")
            try:
                events = self.detectors[feature].detect(self.signal)
                self.results[feature] = events
                print(f"  ✓ Found {len(events)} events")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results[feature] = []
        
        return self.results
    
    def save_results(self, output_path):
        """Save to JSON"""
        results_serializable = {}
        for key, events in self.results.items():
            results_serializable[key] = []
            for event in events:
                event_serializable = {}
                for k, v in event.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        event_serializable[k] = v.tolist()
                    elif isinstance(v, (np.float32, np.float64)):
                        event_serializable[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        event_serializable[k] = int(v)
                    else:
                        event_serializable[k] = v
                results_serializable[key].append(event_serializable)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nSaved to: {output_path}")
    
    def visualize(self, output_dir='results'):
        """Generate plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        viz = RAVENVisualizer(self.signal, self.fs)
        
        for feature, events in self.results.items():
            if events:
                fig, _ = viz.plot_signal_with_events(
                    events, feature,
                    title=f'{feature.capitalize()} Detection'
                )
                fig.savefig(os.path.join(output_dir, f'{feature}.png'), dpi=150)
                plt.close(fig)
        
        if self.results:
            fig = viz.plot_summary(self.results)
            fig.savefig(os.path.join(output_dir, 'summary.png'), dpi=150)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='RAVEN: Sleep Microstructure Analysis')
    parser.add_argument('input', help='Input file (EDF or CSV)')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--features', '-f', nargs='+',
                       default=['kcomplex', 'spindle', 'deltawave'],
                       choices=['kcomplex', 'spindle', 'deltawave'],
                       help='Features to analyze')
    parser.add_argument('--fs', type=int, default=256, help='Sampling frequency (for CSV)')
    parser.add_argument('--format', choices=['edf', 'csv'], default='edf',
                       help='Input format')
    
    args = parser.parse_args()
    
    raven = RAVEN(fs=args.fs)
    
    if args.format == 'edf' or args.input.endswith('.edf'):
        success = raven.load_edf(args.input)
    elif args.format == 'csv' or args.input.endswith('.csv'):
        success = raven.load_csv(args.input, args.fs)
    else:
        print("Unsupported format")
        return
    
    if not success:
        print("Failed to load file")
        return
    
    raven.analyze(args.features)
    os.makedirs(args.output, exist_ok=True)
    raven.save_results(os.path.join(args.output, 'results.json'))
    raven.visualize(args.output)
    
    print("\n" + "="*50)
    print("✓ Analysis complete!")
    print(f"Results: {args.output}")
    print("="*50)

if __name__ == '__main__':
    main()