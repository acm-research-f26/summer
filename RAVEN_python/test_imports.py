#!/usr/bin/env python
"""
Test script to verify all imports are working
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing imports...\n")
    
    all_ok = True
    
    # Test each dependency
    deps = ['numpy', 'scipy', 'matplotlib', 'sklearn', 'pywt', 'pyedflib']
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} imported successfully")
        except ImportError as e:
            print(f"✗ {dep} failed: {e}")
            all_ok = False
    
    # Test our modules
    print("\nTesting RAVEN modules...")
    
    try:
        from signal_processing import SignalProcessor
        print("✓ signal_processing imported successfully")
    except ImportError as e:
        print(f"✗ signal_processing failed: {e}")
        all_ok = False
    
    try:
        from k_complex_detection import KComplexDetector
        print("✓ k_complex_detection imported successfully")
    except ImportError as e:
        print(f"✗ k_complex_detection failed: {e}")
        all_ok = False
    
    try:
        from spindle_detection import SpindleDetector
        print("✓ spindle_detection imported successfully")
    except ImportError as e:
        print(f"✗ spindle_detection failed: {e}")
        all_ok = False
    
    try:
        from delta_wave_detection import DeltaWaveDetector
        print("✓ delta_wave_detection imported successfully")
    except ImportError as e:
        print(f"✗ delta_wave_detection failed: {e}")
        all_ok = False
    
    try:
        from visualization import RAVENVisualizer
        print("✓ visualization imported successfully")
    except ImportError as e:
        print(f"✗ visualization failed: {e}")
        all_ok = False
    
    print("\n" + "="*50)
    if all_ok:
        print("✓ All imports successful!")
        print("You can now run: python main.py data/sample.edf")
    else:
        print("✗ Some imports failed.")
        print("\nTry installing missing packages:")
        print("pip install numpy scipy matplotlib scikit-learn PyWavelets pyedflib")
    print("="*50)