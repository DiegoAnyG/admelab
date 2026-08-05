"""
admelab.ranking
===============
Selection and prioritization of analogs.

Key ideas:
  - Objectives (`Objective`): for each property you decide whether to MAXIMIZE,
    MINIMIZE or approach a TARGET value, with a weight.
  - Desirability: each property is normalized to [0, 1] (simplified Derringer
    function) and combined into a composite `score` (weighted arithmetic or
    geometric mean).
  - Selection: `top`/`bottom` N by score or by any column, and range filters
    (e.g. 250 <= MW <= 400, LD50 >= 500 mg/kg).

Everything operates on the DataFrame produced by `predict` + `toxicity`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Objective:
    """An optimization criterion over a DataFrame column.

    goal:
      - "max": higher is better (e.g. QED, LD50_mg_per_kg, bioavailability)
      - "min": lower is better (e.g. Lipinski violations, number of alerts)
      - "target": closer to `target` is better (e.g. LogP ~ 2.5)
    low/high: bounds for normalization. If None, taken from the dataset itself.
    weight: relative importance (> 0).
    """
    column: str
    goal: str = "max"
    weight: float = 1.0
    target: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


def _desirability(series: pd.Series, obj: Objective) -> pd.Series:
    """Normalize a column to desirability in [0, 1] according to the objective."""
    x = pd.to_numeric(series, errors="coerce")
    lo = obj.low if obj.low is not None else np.nanmin(x.values)
    hi = obj.high if obj.high is not None else np.nanmax(x.values)
    rng = hi - lo

    if rng == 0 or np.isnan(rng):
        # No variability: neutral desirability.
        return pd.Series(np.full(len(x), 0.5), index=x.index)

    if obj.goal == "max":
        d = (x - lo) / rng
    elif obj.goal == "min":
        d = (hi - x) / rng
    elif obj.goal == "target":
        t = obj.target if obj.target is not None else (lo + hi) / 2
        # Normalized distance to the target, inverted.
        d = 1 - (x - t).abs() / max(abs(hi - t), abs(t - lo), 1e-9)
    else:
        raise ValueError(f"Unrecognized goal: {obj.goal!r}")

    return d.clip(0.0, 1.0).fillna(0.0)


def default_objectives(df: pd.DataFrame) -> list[Objective]:
    """Reasonable ADME objectives using only the columns present.

    Prioritizes: good druglikeness (high QED), low toxicity (high LD50), few
    rule violations and few structural alerts. Adds common ADMET-AI endpoints
    if present.
    """
    candidates = [
        Objective("QED", "max", 1.5),
        Objective("LD50_mg_per_kg", "max", 1.5),
        Objective("Lipinski_violations", "min", 1.0),
        Objective("n_alerts", "min", 1.0),
        Objective("LogP", "target", 1.0, target=2.5),
        # Frequent ADMET-AI endpoints (ignored if absent):
        Objective("Bioavailability_Ma", "max", 1.0),
        Objective("Solubility_AqSolDB", "max", 0.8),
        Objective("HIA_Hou", "max", 0.8),
        Objective("hERG", "min", 1.0),
        Objective("DILI", "min", 1.0),
        Objective("AMES", "min", 1.0),
    ]
    return [o for o in candidates if o.column in df.columns]


def add_composite_score(
    df: pd.DataFrame,
    objectives: Optional[Sequence[Objective]] = None,
    method: str = "weighted_mean",
    score_column: str = "score",
) -> pd.DataFrame:
    """Add a `score` column (0-100) and per-objective desirabilities.

    method:
      - "weighted_mean": weighted arithmetic mean of desirabilities.
      - "geometric": weighted geometric mean (penalizes weak points more).
    """
    out = df.copy()
    if objectives is None:
        objectives = default_objectives(out)
    objectives = [o for o in objectives if o.column in out.columns]
    if not objectives:
        out[score_column] = np.nan
        return out

    weights = np.array([o.weight for o in objectives], dtype=float)
    desis = []
    for obj in objectives:
        d = _desirability(out[obj.column], obj)
        out[f"d_{obj.column}"] = d.round(3)
        desis.append(d.values)
    D = np.vstack(desis)  # (n_obj, n_mol)

    if method == "weighted_mean":
        score = np.average(D, axis=0, weights=weights)
    elif method == "geometric":
        # Weighted geometric mean: exp( sum(w*ln d) / sum(w) ).
        eps = 1e-6
        score = np.exp(np.average(np.log(D + eps), axis=0, weights=weights))
    else:
        raise ValueError(f"Unrecognized method: {method!r}")

    out[score_column] = (100 * score).round(2)
    return out


def filter_range(
    df: pd.DataFrame,
    filters: dict[str, tuple[Optional[float], Optional[float]]],
) -> pd.DataFrame:
    """Filter by ranges: {column: (minimum, maximum)}. None = no bound.

    Example: filter_range(df, {"MW": (250, 400), "LD50_mg_per_kg": (500, None)})
    """
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in filters.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if lo is not None:
            mask &= x >= lo
        if hi is not None:
            mask &= x <= hi
    return df[mask].copy()


def top(
    df: pd.DataFrame,
    n: int = 10,
    by: str = "score",
    ascending: bool = False,
) -> pd.DataFrame:
    """Best N rows sorted by `by` (default, highest score first)."""
    if by not in df.columns:
        raise KeyError(f"Column {by!r} does not exist. Columns: {list(df.columns)[:10]}...")
    return df.sort_values(by, ascending=ascending, na_position="last").head(n).reset_index(drop=True)


def bottom(
    df: pd.DataFrame,
    n: int = 10,
    by: str = "score",
) -> pd.DataFrame:
    """Worst N rows by `by` (useful to discard or to analyze)."""
    return top(df, n=n, by=by, ascending=True)


def rank(
    df: pd.DataFrame,
    objectives: Optional[Sequence[Objective]] = None,
    method: str = "weighted_mean",
    filters: Optional[dict] = None,
    n: Optional[int] = None,
    which: str = "top",
) -> pd.DataFrame:
    """Full pipeline: (filters) -> composite score -> top/bottom selection.

    which: "top" or "bottom". n: how many to return (None = all, sorted).
    """
    work = df.copy()
    if filters:
        work = filter_range(work, filters)
    work = add_composite_score(work, objectives=objectives, method=method)
    work = work.sort_values("score", ascending=(which == "bottom"), na_position="last")
    work = work.reset_index(drop=True)
    if n is not None:
        work = work.head(n)
    return work
