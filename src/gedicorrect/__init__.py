"""GEDICorrect public Python API."""

from .config import CORRECTION_CRITERIA, CORRECTION_MODES, CorrectionConfig
from .runner import CorrectionResult, run_correction

__version__ = "1.0.0"


def __getattr__(name):
    if name == "GEDICorrect":
        from .correct import GEDICorrect
        return GEDICorrect
    raise AttributeError(name)


__all__ = [
    "CORRECTION_CRITERIA",
    "CORRECTION_MODES",
    "CorrectionConfig",
    "CorrectionResult",
    "GEDICorrect",
    "run_correction",
]
