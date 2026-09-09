"""figomeas — FIGO type & percent-intramural measurement from uterine fibroid MRI masks.

This package computes a measurement, not a diagnosis. The headline quantity is
``percent_intramural``; FIGO type is derived from it deterministically.
"""

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
