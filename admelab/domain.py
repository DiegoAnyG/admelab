"""Applicability domain for the ADMET-AI predictions.

A model returns a number for any molecule you hand it, including molecules unlike anything it
was trained on, and the number looks equally confident either way. This module measures that
distance so an extrapolated prediction can be reported as one.

Why it matters here: ADMET-AI is trained on the TDC collections, where 1,2,5-oxadiazole
2-oxides (furoxans, benzofuroxans) are barely represented. Predictions for that series are
extrapolation unless shown otherwise, and saying so is a result, not a caveat.

The reference is the actual training data, downloaded from the same Harvard Dataverse files
TDC distributes. PyTDC is deliberately not a dependency: it pins numpy < 2, pandas < 3 and
rdkit < 2024.3, and pulls transformers and cellxgene-census, which would downgrade and break
the environment ADMET-AI runs in. The files are plain tables and requests is enough.

    from admelab.domain import applicability
    ad = applicability(["O=C(OCc1ccccc1)c1ccc2[n+]([O-])onc2c1"])
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ---------------------------------------------------------------------------
# Where the training data lives
# ---------------------------------------------------------------------------

_SERVER = "https://dataverse.harvard.edu/api/access/datafile/"

# Dataverse file ids, one per endpoint ADMET-AI has a model for. Taken from TDC's own metadata;
# they are stable identifiers, not paths, and there is no API that resolves a dataset name to
# one without installing PyTDC.
_ENDPOINT_FILE = {
    "AMES": 4259564,
    "BBB_Martins": 4259566,
    "Bioavailability_Ma": 4259567,
    "CYP1A2_Veith": 4259573,
    "CYP2C19_Veith": 4259576,
    "CYP2C9_Substrate_CarbonMangels": 4259584,
    "CYP2C9_Veith": 4259577,
    "CYP2D6_Substrate_CarbonMangels": 4259578,
    "CYP2D6_Veith": 4259580,
    "CYP3A4_Substrate_CarbonMangels": 4259581,
    "CYP3A4_Veith": 4259582,
    "Caco2_Wang": 4259569,
    "Carcinogens_Lagunin": 4259570,
    "Clearance_Hepatocyte_AZ": 4266187,
    "Clearance_Microsome_AZ": 4266186,
    "ClinTox": 4259572,
    "DILI": 4259585,
    "HIA_Hou": 4259591,
    "Half_Life_Obach": 4266799,
    "HydrationFreeEnergy_FreeSolv": 4259594,
    "LD50_Zhu": 4267146,
    "Lipophilicity_AstraZeneca": 4259595,
    "PAMPA_NCATS": 6695858,
    "PPBR_AZ": 6413140,
    "Pgp_Broccatelli": 4259597,
    "Skin_Reaction": 4259609,
    "Solubility_AqSolDB": 4259610,
    "VDss_Lombardo": 4267387,
    "hERG": 4259588,
}

# The twelve Tox21 assays share one table, each as a column that is empty where the compound
# was not measured, so each label has its own training set.
_TOX21_FILE = 4259612
_TOX21_LABELS = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)

ENDPOINTS = tuple(sorted(_ENDPOINT_FILE)) + _TOX21_LABELS

# Morgan radius 2 over 2048 bits: the fingerprint ADMET-AI's own literature uses for chemical
# space comparison, so the distances are comparable with what is published.
_RADIUS = 2
_NBITS = 2048

# The threshold is a percentile of the training set's own nearest-neighbour similarities, so it
# adapts to how tight each dataset is instead of being one arbitrary cut for all of them. That
# distribution is estimated on a sample, because the exact figure would cost a full 13k x 13k
# comparison per endpoint and shifts the percentile by far less than the decision needs.
_SAMPLE = 2000
_SEED = 42


def _cache_dir() -> Path:
    d = Path(os.environ.get("ADMELAB_CACHE", Path.home() / ".cache" / "admelab" / "tdc"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(file_id: int, name: str) -> pd.DataFrame:
    cached = _cache_dir() / f"{name}.tsv"
    if cached.exists():
        return pd.read_csv(cached, sep="\t")

    r = requests.get(f"{_SERVER}{file_id}", timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), sep="\t")
    df.to_csv(cached, sep="\t", index=False)
    return df


# TDC does not use one column name throughout: most tables call the structure "Drug", while
# LD50_Zhu and Tox21 call it "X".
_SMILES_COLUMNS = ("Drug", "X", "smiles", "SMILES")


def _smiles_column(df: pd.DataFrame) -> str:
    for c in _SMILES_COLUMNS:
        if c in df.columns:
            return c
    raise KeyError(f"No structure column in {list(df.columns)}")


def training_smiles(endpoint: str) -> list:
    """SMILES ADMET-AI was trained on for one endpoint. Cached after the first call."""
    if endpoint in _TOX21_LABELS:
        df = _download(_TOX21_FILE, "tox21")
        col = _smiles_column(df)
        return df.loc[df[endpoint].notna(), col].dropna().astype(str).tolist()

    if endpoint not in _ENDPOINT_FILE:
        raise KeyError(f"Unknown endpoint {endpoint!r}. See admelab.domain.ENDPOINTS.")

    df = _download(_ENDPOINT_FILE[endpoint], endpoint)
    return df[_smiles_column(df)].dropna().astype(str).tolist()


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=_RADIUS, fpSize=_NBITS)


def _fingerprints(smiles: Iterable[str]) -> list:
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(_GEN.GetFingerprint(mol))
    return fps


def _max_similarity(query_fps: Sequence, reference_fps: Sequence) -> np.ndarray:
    """Similarity of each query to its closest neighbour in the reference set."""
    ref = list(reference_fps)
    return np.array([max(DataStructs.BulkTanimotoSimilarity(q, ref)) for q in query_fps])


def _threshold(reference_fps: Sequence, percentile: float) -> float:
    """How isolated a training compound is allowed to be before it counts as an outlier.

    Each sampled compound is compared with the rest of the training set, its own perfect
    self-match excluded, and the requested percentile of those similarities is the line.
    """
    rng = np.random.default_rng(_SEED)
    ref = list(reference_fps)
    n = len(ref)
    idx = rng.choice(n, size=min(_SAMPLE, n), replace=False)

    nn = []
    for i in idx:
        sims = DataStructs.BulkTanimotoSimilarity(ref[i], ref)
        sims[i] = -1.0
        nn.append(max(sims))

    return float(np.percentile(nn, percentile))


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

def applicability(smiles: Sequence[str],
                  endpoints: Optional[Sequence[str]] = None,
                  percentile: float = 5.0) -> pd.DataFrame:
    """Per endpoint, how far each molecule sits from what the model was trained on.

    Returns one row per (SMILES, endpoint) with:
      similarity  Tanimoto to the closest training compound
      threshold   the percentile of the training set's own nearest-neighbour similarities
      in_domain   similarity >= threshold

    A molecule outside the domain is not a wrong prediction; it is a prediction the training
    data does not support, and should be reported as such rather than silently used.
    """
    endpoints = list(endpoints) if endpoints else list(ENDPOINTS)
    query_fps = _fingerprints(smiles)
    valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]

    rows = []
    for ep in endpoints:
        ref_fps = _fingerprints(training_smiles(ep))
        if not ref_fps:
            continue
        cut = _threshold(ref_fps, percentile)
        sims = _max_similarity(query_fps, ref_fps)
        for s, sim in zip(valid, sims):
            rows.append({
                "SMILES": s,
                "endpoint": ep,
                "similarity": round(float(sim), 4),
                "threshold": round(cut, 4),
                "in_domain": bool(sim >= cut),
                "n_train": len(ref_fps),
            })

    return pd.DataFrame(rows)


def summary(ad: pd.DataFrame) -> pd.DataFrame:
    """One row per molecule: how many endpoints support it, and how close it sits overall."""
    g = ad.groupby("SMILES")
    return pd.DataFrame({
        "endpoints": g.size(),
        "in_domain": g["in_domain"].sum(),
        "fraction_in_domain": (g["in_domain"].mean()).round(3),
        "median_similarity": g["similarity"].median().round(4),
        "max_similarity": g["similarity"].max().round(4),
    }).reset_index()
