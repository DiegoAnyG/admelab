"""
admelab.naming
==============
IUPAC name of a molecule, with a hybrid strategy:

1) PubChem -> EXACT curated name if the molecule already exists in the database
   (structure search). Ideal for the lead and known analogs.
2) CACTUS  -> NCI structure->name resolver (cactus.nci.nih.gov) which generates
   the IUPAC name ALGORITHMICALLY. Works for NOVEL molecules not in any database.

Technical note: the initial plan considered STOUT (an ML SMILES->IUPAC model),
but STOUT-pypi requires tensorflow==2.10.1, which has no distribution for
Python 3.12 and would break the environment (RDKit/ADMET-AI). CACTUS covers the
same use case (naming novel molecules) without heavy dependencies and with
deterministic precision.

Both paths are web queries: used sparingly (a pause between calls) and the
module degrades gracefully without a connection.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence
from urllib.parse import quote

from rdkit import Chem

try:
    import requests
except ImportError:  # requests ships with the environment, but just in case
    requests = None


# ---------------------------------------------------------------------------
# PubChem (known molecules)
# ---------------------------------------------------------------------------
def name_from_pubchem(smiles: str) -> Optional[str]:
    """IUPAC name from PubChem by structure search. None if it does not exist
    or the network fails."""
    try:
        import pubchempy as pcp
    except ImportError:
        return None
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles", listkey_count=1)
    except Exception:
        return None
    if not compounds:
        return None
    name = getattr(compounds[0], "iupac_name", None)
    return name or None


# ---------------------------------------------------------------------------
# CACTUS (novel molecules; NCI algorithm)
# ---------------------------------------------------------------------------
CACTUS_URL = "https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"


def name_from_cactus(smiles: str, timeout: float = 20.0) -> Optional[str]:
    """IUPAC name from the NCI CACTUS resolver. None if it does not resolve."""
    if requests is None:
        return None
    url = CACTUS_URL.format(smiles=quote(smiles, safe=""))
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    text = resp.text.strip()
    # CACTUS may return an HTML error page with status 200 in some cases.
    if not text or "<html" in text.lower() or "page not found" in text.lower():
        return None
    # It may return several lines (synonyms); take the first.
    return text.splitlines()[0].strip() or None


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
def iupac_name(
    smiles: str,
    use_pubchem: bool = True,
    use_cactus: bool = True,
) -> dict:
    """Return {'iupac_name': str|None, 'name_source': 'pubchem'|'cactus'|None}."""
    if use_pubchem:
        name = name_from_pubchem(smiles)
        if name:
            return {"iupac_name": name, "name_source": "pubchem"}
    if use_cactus:
        name = name_from_cactus(smiles)
        if name:
            return {"iupac_name": name, "name_source": "cactus"}
    return {"iupac_name": None, "name_source": None}


def name_batch(
    smiles_list: Sequence[str],
    use_pubchem: bool = True,
    use_cactus: bool = True,
    pause: float = 0.30,
    progress: bool = True,
) -> "pd.DataFrame":
    """Name a list of SMILES. Returns DataFrame [SMILES, iupac_name, name_source].

    - `pause`: wait between web queries (courtesy to the services).
    - Recommended to name only the selected TOP molecules, not thousands.
    """
    import pandas as pd

    iterator = smiles_list
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(smiles_list, desc="Naming (IUPAC)")
        except ImportError:
            pass

    rows = []
    for smi in iterator:
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol) if mol else smi
        res = {"SMILES": canon, "iupac_name": None, "name_source": None}

        if use_pubchem:
            nm = name_from_pubchem(canon)
            if pause:
                time.sleep(pause)
            if nm:
                res.update(iupac_name=nm, name_source="pubchem")
                rows.append(res)
                continue
        if use_cactus:
            nm = name_from_cactus(canon)
            if pause:
                time.sleep(pause)
            if nm:
                res.update(iupac_name=nm, name_source="cactus")
        rows.append(res)
    return pd.DataFrame(rows)
