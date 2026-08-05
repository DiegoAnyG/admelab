"""
admelab
=======
Analog design & ADME/Tox laboratory.

Typical flow:
    1. generation   -> generate analogs from a lead molecule
    2. reactions    -> extensible reaction engine (esterification, amidation...)
    3. predict      -> RDKit descriptors/rules + ML prediction (ADMET-AI)
    4. toxicity     -> LD50 and acute-toxicity classification (GHS)
    5. naming/naming_smart -> IUPAC name (self-verified via OPSIN round-trip)
    6. ranking      -> top/bottom selection, filters and composite score
    7. viz          -> molecule grids and profile plots

Each module is independent; the notebooks orchestrate them.
"""

__version__ = "0.2.0"

# Lightweight package-level imports. Heavy modules (predict with ADMET-AI) are
# loaded lazily inside their functions to avoid penalizing startup.
from . import generation  # noqa: F401

__all__ = [
    "generation", "reactions", "esterification", "predict", "toxicity",
    "naming", "naming_smart", "ranking", "viz", "pipeline",
]
