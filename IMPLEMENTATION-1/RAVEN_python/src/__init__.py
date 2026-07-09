"""
RAVEN - Sleep Microstructure Analysis Package
"""

__all__ = [
    'SignalProcessor',
    'KComplexDetector', 
    'SpindleDetector',
    'DeltaWaveDetector',
    'RAVENVisualizer'
]

# Lazy imports so modules are only loaded when needed

def __getattr__(name):
    if name == 'SignalProcessor':
        from .signal_processing import SignalProcessor
        return SignalProcessor
    elif name == 'KComplexDetector':
        from .k_complex_detection import KComplexDetector
        return KComplexDetector
    elif name == 'SpindleDetector':
        from .spindle_detection import SpindleDetector
        return SpindleDetector
    elif name == 'DeltaWaveDetector':
        from .delta_wave_detection import DeltaWaveDetector
        return DeltaWaveDetector
    elif name == 'RAVENVisualizer':
        from .visualization import RAVENVisualizer
        return RAVENVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
