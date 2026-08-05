"""
admelab.reactions
=================
**Extensible** "one core x N partners" reaction engine.

Instead of manipulating text strings, each reaction is a SMARTS transformation
with atom maps, so that **stereochemistry is preserved** and products are
**validated by exact atomic formula**.

Adding a new reaction = adding a `Reaction` entry to the `REACTIONS` catalog
(or passing your own `Reaction` to the functions). Included out of the box:

  - `esterification` : carboxylic acid + alcohol/phenol -> ester (+ H2O)
  - `amidation`      : carboxylic acid + 1.deg/2.deg amine -> amide (+ H2O)

The alcohol-specialized layer (OH classification, Fischer viability,
regioselectivity, audits) lives in `admelab.esterification`, which reuses this
engine.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd

RDLogger.DisableLog("rdApp.*")


@dataclass
class Reaction:
    """A chemical transformation.

    smarts       : reaction SMARTS 'reactant1.reactant2>>product' with atom maps.
    byproduct    : formula of the eliminated byproduct (e.g. 'H2O' for
                   condensations); empty string if nothing is lost. Used to
                   validate the product formula.
    core_pattern : SMARTS the CORE must contain (input validation).
    """
    name: str
    smarts: str
    description: str = ""
    byproduct: str = "H2O"
    core_pattern: str = "[CX3](=[OX1])[OX2H1]"   # -COOH by default


# Reaction catalog. Extend it with new entries.
REACTIONS: dict[str, Reaction] = {
    "esterification": Reaction(
        name="esterification",
        smarts="[CX3:1](=[OX1:2])[OX2H1].[OX2H1:3][#6:4]>>[CX3:1](=[OX1:2])[O:3][#6:4]",
        description="Fischer esterification: carboxylic acid + alcohol/phenol -> ester + water.",
        byproduct="H2O",
    ),
    "amidation": Reaction(
        name="amidation",
        smarts="[CX3:1](=[OX1:2])[OX2H1].[NX3;H2,H1:3][#6:4]>>[CX3:1](=[OX1:2])[N:3][#6:4]",
        description="Amide formation: carboxylic acid + primary/secondary amine -> amide + water.",
        byproduct="H2O",
    ),
}

# Reusable alias for the esterification module (single source of the SMARTS).
ESTERIFICATION_SMARTS = REACTIONS["esterification"].smarts


def get_reaction(reaction: "str | Reaction") -> Reaction:
    """Accepts either a reaction name from the catalog or a Reaction object."""
    if isinstance(reaction, Reaction):
        return reaction
    if reaction in REACTIONS:
        return REACTIONS[reaction]
    raise KeyError(f"Unknown reaction: {reaction!r}. "
                   f"Available: {list(REACTIONS)} (or pass a Reaction object).")


# ---------------------------------------------------------------------------
# Formula utilities
# ---------------------------------------------------------------------------
def _atom_counts(mol: Chem.Mol) -> Counter:
    """Exact atomic count including hydrogens."""
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())


def _byproduct_counts(formula: str) -> Counter:
    """Atomic count of a simple byproduct given as a formula (e.g. 'H2O')."""
    import re
    c: Counter = Counter()
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if sym:
            c[sym] += int(num) if num else 1
    return c


def _n_stereocenters(mol: Optional[Chem.Mol]) -> int:
    if mol is None:
        return 0
    n_chiral = len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False,
                                             includeUnassigned=False))
    n_db = sum(1 for b in mol.GetBonds()
               if b.GetStereo() != Chem.BondStereo.STEREONONE)
    return n_chiral + n_db


# ---------------------------------------------------------------------------
# Reaction and validation
# ---------------------------------------------------------------------------
def run_reaction(
    core_smiles: str,
    partner_smiles: str,
    reaction: "str | Reaction" = "esterification",
    policy: str = "all",
) -> list[dict]:
    """Apply a reaction between a core and a partner.

    policy: "all" (all regiochemical products) or "first" (only the first one).
    Returns [{'smiles', 'inchikey', 'n_products'}].
    """
    rxn_def = get_reaction(reaction)
    core = Chem.MolFromSmiles(core_smiles)
    partner = Chem.MolFromSmiles(partner_smiles)
    if core is None or partner is None:
        return []

    rxn = AllChem.ReactionFromSmarts(rxn_def.smarts)
    uniq: dict[str, Chem.Mol] = {}
    for tup in rxn.RunReactants((core, partner)):
        prod = tup[0]
        try:
            Chem.SanitizeMol(prod)
            Chem.AssignStereochemistry(prod, cleanIt=True, force=True)
            uniq[Chem.MolToSmiles(prod)] = prod
        except Exception:
            continue

    items = list(uniq.items())
    if policy == "first" and items:
        items = items[:1]
    return [{"smiles": smi, "inchikey": Chem.MolToInchiKey(m), "n_products": len(uniq)}
            for smi, m in items]


def validate_product(
    core_smiles: str,
    partner_smiles: str,
    product_smiles: str,
    reaction: "str | Reaction" = "esterification",
) -> dict:
    """Validate a product: exact atomic formula (core + partner - byproduct),
    sanitization and preservation of stereocenters."""
    rxn_def = get_reaction(reaction)
    core = Chem.MolFromSmiles(core_smiles)
    partner = Chem.MolFromSmiles(partner_smiles)
    prod = Chem.MolFromSmiles(product_smiles) if product_smiles else None
    res = {"valid": False, "formula_ok": False, "stereo_ok": False,
           "formula": None, "reason": ""}
    if prod is None:
        res["reason"] = "product did not parse"; return res
    if core is None or partner is None:
        res["reason"] = "reactant did not parse"; return res

    expected = _atom_counts(core) + _atom_counts(partner) - _byproduct_counts(rxn_def.byproduct)
    expected = Counter({k: v for k, v in expected.items() if v > 0})
    res["formula"] = rdmd.CalcMolFormula(prod)
    res["formula_ok"] = (expected == _atom_counts(prod))
    res["stereo_ok"] = (_n_stereocenters(prod) >= _n_stereocenters(core) + _n_stereocenters(partner))

    reasons = []
    if not res["formula_ok"]:
        reasons.append(f"formula {res['formula']} != expected")
    if not res["stereo_ok"]:
        reasons.append("stereocenters lost")
    res["reason"] = "; ".join(reasons)
    res["valid"] = res["formula_ok"] and res["stereo_ok"]
    return res


def react_library(
    core_smiles: str,
    partners: "pd.DataFrame | Sequence[str]",
    reaction: "str | Reaction" = "esterification",
    smiles_col: str = "SMILES",
    name_col: Optional[str] = None,
    policy: str = "first",
    reference_col: Optional[str] = None,
    keep_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply a reaction of one core with a library of partners.

    Generic for any reaction in the catalog. Returns a DataFrame with one row
    per product and validation columns (plus audit columns if `reference_col`
    is given). For esterification with alcohol-aware logic use
    `admelab.esterification.esterify_library`.
    """
    rxn_def = get_reaction(reaction)
    if not isinstance(partners, pd.DataFrame):
        partners = pd.DataFrame({smiles_col: list(partners)})

    rows = []
    for _, r in partners.iterrows():
        smi = str(r[smiles_col]).strip()
        name = str(r[name_col]) if (name_col and name_col in partners.columns) else None
        base = {"name": name, "reactant_SMILES": smi, "reaction": rxn_def.name}
        for c in keep_cols:
            if c in partners.columns:
                base[c] = r[c]

        prods = run_reaction(core_smiles, smi, reaction=rxn_def, policy=policy)
        if not prods:
            rows.append({**base, "SMILES": None, "status": "ERROR: no product"})
            continue
        for p in prods:
            val = validate_product(core_smiles, smi, p["smiles"], reaction=rxn_def)
            m = Chem.MolFromSmiles(p["smiles"])
            row = {**base, "SMILES": p["smiles"], "InChIKey": p["inchikey"],
                   "n_products": p["n_products"], "formula": val["formula"],
                   "MW": round(Descriptors.MolWt(m), 2) if m else None,
                   "valid": val["valid"],
                   "status": "OK" if val["valid"] else f"REVIEW: {val['reason']}"}
            if reference_col and reference_col in partners.columns:
                ref = Chem.MolFromSmiles(str(r[reference_col]))
                rk = Chem.MolToInchiKey(ref) if ref is not None else None
                row["ref_SMILES"] = r[reference_col]
                if rk is None:
                    row["audit"] = "reference did not parse"
                elif rk == p["inchikey"]:
                    row["audit"] = "match"
                else:
                    same = rk.split("-")[0] == p["inchikey"].split("-")[0]
                    row["audit"] = ("MISMATCH (stereochemistry)" if same
                                    else "MISMATCH (skeleton/regiochemistry)")
            rows.append(row)
    return pd.DataFrame(rows)
