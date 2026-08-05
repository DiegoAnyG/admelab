"""
admelab.pipeline
================
High-level function that chains the whole flow in a single call:

    generate analogs -> predict ADME/LD50 -> (toxicity) -> rank

Useful for the notebook's interactive panel and for quick experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from . import generation, predict, ranking, toxicity


@dataclass
class PipelineResult:
    ranked: pd.DataFrame                 # final DataFrame sorted by score
    generation: generation.GenerationResult
    n_generated: int = 0
    n_scored: int = 0


def run_pipeline(
    lead_smiles: str,
    methods: Sequence[str] = ("decoration",),
    use_ml: bool = True,
    # --- generation ---
    substituents: Optional[dict] = None,
    scope: str = "aromatic_ch",
    positions: Optional[Sequence[int]] = None,
    n_substitutions=(1,),
    max_decor: int = 300,
    max_brics: int = 60,
    # --- ranking ---
    objectives: Optional[Sequence[ranking.Objective]] = None,
    filters: Optional[dict] = None,
    score_method: str = "weighted_mean",
    include_lead: bool = True,
) -> PipelineResult:
    """Run the full flow and return a PipelineResult.

    Notable parameters:
      positions        -> the SITE of branching (atom indices; None = auto)
      n_substitutions  -> the NUMBER of substitutions (int or iterable, e.g. [1,2])
      substituents     -> substituent library {name: SMILES}
      objectives       -> selection criteria (max/min/target with weights)
      filters          -> ranges {column: (min, max)}
    """
    if substituents is None:
        substituents = generation.SMALL_SUBSTITUENTS

    gen = generation.generate(
        lead_smiles,
        methods=methods,
        dec_substituents=substituents,
        dec_scope=scope,
        dec_positions=positions,
        dec_n_substitutions=n_substitutions,
        dec_max_products=max_decor,
        brics_max_products=max_brics,
    )

    smiles = []
    if include_lead:
        smiles.append(generation.canonical_smiles(lead_smiles))
    smiles += gen.analogs

    df = predict.predict_batch(smiles, use_ml=use_ml)
    df = toxicity.annotate_toxicity(df)

    lead_canon = generation.canonical_smiles(lead_smiles)
    if "SMILES" in df.columns:
        df.insert(1, "is_lead", df["SMILES"] == lead_canon)

    ranked = ranking.rank(
        df, objectives=objectives, method=score_method,
        filters=filters, which="top",
    )
    return PipelineResult(
        ranked=ranked,
        generation=gen,
        n_generated=len(gen),
        n_scored=len(df),
    )
