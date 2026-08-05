"""
admelab.predict
===============
ADME property prediction in two layers:

Layer 1 (always, instant) - RDKit descriptors and rules:
    molecular weight, LogP, TPSA, H donors/acceptors, rotatable bonds,
    fraction sp3, QED, estimated solubility (ESOL), druglikeness rules
    (Lipinski/Veber/Egan/Ghose) and structural alerts (PAINS/Brenk).

Layer 2 (optional, ML) - ADMET-AI:
    ~41 ADMET endpoints trained on the Therapeutics Data Commons (absorption,
    BBB, protein binding, CYPs, clearance, hERG, AMES, DILI, LD50...). The model
    is loaded lazily (singleton) because it is expensive.

Main function: `predict_batch(smiles_list, use_ml=True)` -> DataFrame.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Structural-alert catalog (PAINS + Brenk), built once.
# ---------------------------------------------------------------------------
_ALERT_CATALOG: FilterCatalog | None = None


def _get_alert_catalog() -> FilterCatalog:
    global _ALERT_CATALOG
    if _ALERT_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _ALERT_CATALOG = FilterCatalog(params)
    return _ALERT_CATALOG


# ---------------------------------------------------------------------------
# Layer 1 - physicochemical descriptors
# ---------------------------------------------------------------------------
def physchem_descriptors(mol: Chem.Mol) -> dict:
    """Base physicochemical descriptors."""
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "MolMR": Crippen.MolMR(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "QED": QED.qed(mol),
    }


def esol_logS(mol: Chem.Mol, desc: dict | None = None) -> float:
    """Estimated aqueous solubility log S (mol/L), Delaney's ESOL model.

    logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RotB - 0.74*AP
    with AP = fraction of heavy atoms that are aromatic.
    """
    if desc is None:
        desc = physchem_descriptors(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = (aromatic_atoms / heavy) if heavy else 0.0
    return (0.16
            - 0.63 * desc["LogP"]
            - 0.0062 * desc["MW"]
            + 0.066 * desc["RotB"]
            - 0.74 * ap)


def druglikeness_rules(desc: dict) -> dict:
    """Druglikeness rules from the base descriptors."""
    mw, logp, hbd, hba = desc["MW"], desc["LogP"], desc["HBD"], desc["HBA"]
    tpsa, rotb, mr, heavy = desc["TPSA"], desc["RotB"], desc["MolMR"], desc["HeavyAtoms"]

    lipinski_viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber = (rotb <= 10) and (tpsa <= 140)
    egan = (tpsa <= 131.6) and (logp <= 5.88)
    ghose = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and (40 <= mr <= 130) and (20 <= heavy <= 70)
    lead_like = (mw <= 350) and (logp <= 3.5) and (rotb <= 7)

    return {
        "Lipinski_violations": lipinski_viol,
        "Lipinski_pass": lipinski_viol <= 1,
        "Veber_pass": bool(veber),
        "Egan_pass": bool(egan),
        "Ghose_pass": bool(ghose),
        "LeadLike_pass": bool(lead_like),
    }


def structural_alerts(mol: Chem.Mol) -> dict:
    """Count matches against PAINS/Brenk filters."""
    catalog = _get_alert_catalog()
    matches = catalog.GetMatches(mol)
    descriptions = [m.GetDescription() for m in matches]
    return {
        "n_alerts": len(descriptions),
        "alerts": "; ".join(descriptions[:5]),  # first 5 to avoid clutter
    }


def rdkit_profile(smiles: str) -> dict | None:
    """Full layer-1 profile for one SMILES. None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = physchem_descriptors(mol)
    row = {"SMILES": Chem.MolToSmiles(mol)}
    row.update(desc)
    row["ESOL_logS"] = esol_logS(mol, desc)
    row.update(druglikeness_rules(desc))
    row.update(structural_alerts(mol))
    return row


def rdkit_batch(smiles_list: Sequence[str]) -> pd.DataFrame:
    """RDKit profile for a list of SMILES (drops invalid ones)."""
    rows = []
    for smi in smiles_list:
        prof = rdkit_profile(smi)
        if prof is not None:
            rows.append(prof)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 2 - ADMET-AI (ML)
# ---------------------------------------------------------------------------
_ADMET_MODEL = None


def quiet_ml_logs() -> None:
    """Silence PyTorch Lightning verbosity / warnings during prediction.

    ADMET-AI internally uses a Lightning Trainer that prints progress bars and
    warnings on every prediction; this reduces them for clean notebook output.
    """
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    for name in ("lightning.pytorch", "pytorch_lightning", "lightning",
                 "lightning.pytorch.utilities.rank_zero",
                 "lightning.fabric.utilities.rank_zero"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Disable Lightning "Predicting ..." bars if the variable exists.
    import os
    os.environ.setdefault("PT_LIGHTNING_ENABLE_PROGRESS_BAR", "0")


def get_admet_model(quiet: bool = True):
    """Return the ADMET-AI model (lazy load, singleton)."""
    global _ADMET_MODEL
    if quiet:
        quiet_ml_logs()
    if _ADMET_MODEL is None:
        from admet_ai import ADMETModel  # lazy import: pulls torch/chemprop
        _ADMET_MODEL = ADMETModel()
    return _ADMET_MODEL


def predict_ml(smiles_list: Sequence[str]) -> pd.DataFrame:
    """ML prediction with ADMET-AI. DataFrame with SMILES in the 'SMILES' column."""
    model = get_admet_model()
    preds = model.predict(smiles=list(smiles_list))
    # ADMET-AI returns a DataFrame indexed by SMILES.
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    preds = preds.reset_index().rename(columns={"index": "SMILES"})
    if "SMILES" not in preds.columns:
        preds = preds.rename(columns={preds.columns[0]: "SMILES"})
    return preds


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
def predict_batch(
    smiles_list: Sequence[str],
    use_ml: bool = True,
    ml_prefix: str = "",
) -> pd.DataFrame:
    """Combined prediction (layer 1 always; layer 2 if `use_ml`).

    Returns a DataFrame with one row per valid molecule. ADMET-AI columns that
    duplicate layer-1 names are renamed with `ml_prefix` (default "", but name
    clashes are renamed to *_ml).
    """
    base = rdkit_batch(smiles_list)
    if base.empty or not use_ml:
        return base

    try:
        ml = predict_ml(base["SMILES"].tolist())
    except Exception as exc:  # if ADMET-AI fails, return at least layer 1
        print(f"[predict] Warning: ADMET-AI unavailable ({exc}). "
              "Returning RDKit descriptors only.")
        return base

    # Avoid name clashes on merge (except the SMILES key).
    overlap = (set(base.columns) & set(ml.columns)) - {"SMILES"}
    if overlap:
        ml = ml.rename(columns={c: f"{c}{ml_prefix}_ml" for c in overlap})

    merged = base.merge(ml, on="SMILES", how="left")
    return merged
