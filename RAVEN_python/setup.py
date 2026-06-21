# File: setup.py
from setuptools import setup

setup(
    name="RAVEN",
    version="1.0.0",
    description="Automated sleep microstructure analysis from EEG",
    author="RAVEN Team",
    package_dir={'': 'src'},
    py_modules=[
        'signal_processing',
        'k_complex_detection',
        'spindle_detection',
        'delta_wave_detection',
        'visualization'
    ],
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.2.0',
        'PyWavelets>=1.4.0',
        'pyedflib>=0.1.36',
        'mne>=1.3.0',
        'PyQt5>=5.15.0',
        'tqdm>=4.65.0',
        'joblib>=1.2.0',
    ],
    python_requires='>=3.8',
)