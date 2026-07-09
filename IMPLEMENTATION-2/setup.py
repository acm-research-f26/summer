"""Setup configuration for SleepEEGpy."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")

setup(
    name="sleepeegpy",
    version="1.0.0",
    author="Rotem Falach, Gennadiy Belonosov, et al.",
    author_email="nirlab@tauex.tau.ac.il",
    description="SleepEEGpy: A Python-based software integration package for sleep EEG analysis",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NirLab-TAU/sleepeegpy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "mne>=1.5.0",
        "PyPREP>=0.5.0",
        "yasa>=0.6.0",
        "specparam>=1.1.0",
        "scikit-learn>=1.2.0",
        "jupyter>=1.0.0",
        "ipython>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "pre-commit>=3.0.0",
        ],
        "vis": [
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/NirLab-TAU/sleepeegpy/issues",
        "Source": "https://github.com/NirLab-TAU/sleepeegpy",
        "Documentation": "https://github.com/NirLab-TAU/sleepeegpy",
    },
)