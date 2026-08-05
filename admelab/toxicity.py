"""
admelab.toxicity
================
Interpretation of the acute toxicity (LD50) predicted by ADMET-AI.

ADMET-AI predicts the `LD50_Zhu` endpoint from the Therapeutics Data Commons.
The native TDC value is in units of -log10(LD50) with LD50 in mol/kg (higher
value = MORE toxic). Here we convert it to mg/kg (more intuitive) and classify
it into the GHS categories of acute oral toxicity.

Note: converting to mg/kg requires the molecular weight. If calibration with
reference molecules showed a different unit, just change `LD50_UNIT`.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# Assumed native unit of the ADMET-AI LD50 column.
#   "neglog10_mol/kg" -> value = -log10(LD50 in mol/kg)   (default, TDC)
#   "log10_mol/kg"    -> value =  log10(LD50 in mol/kg)
#   "mg/kg"           -> direct value in mg/kg
#
# Empirical calibration (rat oral LD50, mg/kg): with "neglog10_mol/kg" the model
# predicts acetaminophen ~2274 (exp. ~1944) and ibuprofen ~756 (exp. ~636),
# which confirms the unit and the scale of the GHS classification below.
LD50_UNIT = "neglog10_mol/kg"


# GHS categories of acute oral toxicity (LD50 cutoffs in mg/kg).
_GHS_BINS = [
    (5,     "GHS-1 (Fatal, very toxic)"),
    (50,    "GHS-2 (Fatal)"),
    (300,   "GHS-3 (Toxic)"),
    (2000,  "GHS-4 (Harmful)"),
    (5000,  "GHS-5 (Low toxicity)"),
    (math.inf, "Unclassified (practically non-toxic)"),
]


def ld50_to_mg_per_kg(value: float, mw: float, unit: str = LD50_UNIT) -> Optional[float]:
    """Convert the native LD50 value to mg/kg using the molecular weight."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if unit == "mg/kg":
        return float(value)
    if unit == "neglog10_mol/kg":
        mol_per_kg = 10 ** (-value)
    elif unit == "log10_mol/kg":
        mol_per_kg = 10 ** (value)
    else:
        raise ValueError(f"Unrecognized LD50 unit: {unit!r}")
    # mol/kg * g/mol = g/kg ; *1000 = mg/kg
    return mol_per_kg * mw * 1000.0


def ghs_category(ld50_mg_per_kg: Optional[float]) -> str:
    """GHS acute oral toxicity category from LD50 in mg/kg."""
    if ld50_mg_per_kg is None or math.isnan(ld50_mg_per_kg):
        return "unknown"
    for cutoff, label in _GHS_BINS:
        if ld50_mg_per_kg <= cutoff:
            return label
    return _GHS_BINS[-1][1]


def find_ld50_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the LD50 column in the ADMET-AI output (case-insensitive)."""
    candidates = [c for c in df.columns
                  if "ld50" in c.lower() and "percentile" not in c.lower()]
    return candidates[0] if candidates else None


def annotate_toxicity(
    df: pd.DataFrame,
    mw_column: str = "MW",
    unit: str = LD50_UNIT,
) -> pd.DataFrame:
    """Add interpreted toxicity columns to the predictions DataFrame.

    Requires: an LD50 column (from ADMET-AI) and a molecular-weight column.
    Adds:
      - LD50_native     : original model value
      - LD50_mg_per_kg  : converted to mg/kg
      - GHS_category    : GHS acute oral toxicity category
    """
    out = df.copy()
    ld50_col = find_ld50_column(out)
    if ld50_col is None or mw_column not in out.columns:
        out["LD50_native"] = float("nan")
        out["LD50_mg_per_kg"] = float("nan")
        out["GHS_category"] = "unknown"
        return out

    out["LD50_native"] = out[ld50_col]
    out["LD50_mg_per_kg"] = [
        ld50_to_mg_per_kg(v, mw, unit=unit)
        for v, mw in zip(out[ld50_col], out[mw_column])
    ]
    out["GHS_category"] = [ghs_category(v) for v in out["LD50_mg_per_kg"]]
    return out
