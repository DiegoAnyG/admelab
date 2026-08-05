"""
admelab.generation
===================
Generation of analogs from a lead molecule.

Two complementary strategies:

1) Position decoration (R-group)  -> `enumerate_decorations`
   Fine control over the SITE (which atoms) and the NUMBER (mono/di/tri...) of
   substitutions, using a medicinal-chemistry substituent library.

2) BRICS recombination            -> `brics_analogs`
   Fragments the lead along synthetically accessible bonds and recombines to
   explore a broader chemical space (less positional control).

Both return lists of canonical SMILES, deduplicated by InChIKey and excluding
the lead molecule itself.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS, Descriptors

# RDKit is very verbose about invalid intermediate molecules; we handle those
# ourselves with try/except, so we silence its logger.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Substituent library (common medicinal chemistry).
# The ATTACHMENT POINT is always atom index 0 of the fragment SMILES.
# ---------------------------------------------------------------------------
DEFAULT_SUBSTITUENTS: dict[str, str] = {
    "fluoro":          "F",
    "chloro":          "Cl",
    "bromo":           "Br",
    "methyl":          "C",
    "ethyl":           "CC",
    "isopropyl":       "C(C)C",
    "cyclopropyl":     "C1CC1",
    "trifluoromethyl": "C(F)(F)F",
    "hydroxy":         "O",
    "methoxy":         "OC",
    "trifluoromethoxy": "OC(F)(F)F",
    "amino":           "N",
    "dimethylamino":   "N(C)C",
    "cyano":           "C#N",
    "nitro":           "[N+](=O)[O-]",
    "carboxy":         "C(=O)O",
    "acetyl":          "C(C)=O",
    "carboxamide":     "C(N)=O",
    "sulfonamide":     "S(N)(=O)=O",
    "phenyl":          "c1ccccc1",
}

# "Small" subset useful for quick enumerations (avoids combinatorial explosion).
SMALL_SUBSTITUENTS: dict[str, str] = {
    k: DEFAULT_SUBSTITUENTS[k]
    for k in ("fluoro", "chloro", "methyl", "trifluoromethyl",
              "hydroxy", "methoxy", "amino", "cyano")
}


@dataclass
class GenerationResult:
    """Container for the results of a generation run."""
    lead_smiles: str
    analogs: list[str] = field(default_factory=list)      # canonical SMILES
    method: str = ""
    provenance: list[dict] = field(default_factory=list)  # metadata per analog

    def __len__(self) -> int:
        return len(self.analogs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def to_mol(smiles: str) -> Chem.Mol | None:
    """Parse SMILES to a sanitized Mol, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol


def canonical_smiles(mol_or_smiles) -> str | None:
    """Return canonical SMILES or None."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def inchikey(mol_or_smiles) -> str | None:
    """InChIKey for deduplication (ignores representation differences)."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def substituent_positions(
    mol: Chem.Mol,
    scope: str = "aromatic_ch",
) -> list[int]:
    """Return the indices of substitutable atoms (those with at least one H).

    scope:
      - "aromatic_ch": aromatic carbons with H (typical ring positions)
      - "aliphatic_ch": sp3 carbons with H
      - "any_c": any carbon with H
      - "hetero_h": heteroatoms (N, O, S) with H
      - "all_h": any heavy atom with H
    """
    idxs: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetTotalNumHs() < 1:
            continue
        z = atom.GetAtomicNum()
        arom = atom.GetIsAromatic()
        if scope == "aromatic_ch" and z == 6 and arom:
            idxs.append(atom.GetIdx())
        elif scope == "aliphatic_ch" and z == 6 and not arom:
            idxs.append(atom.GetIdx())
        elif scope == "any_c" and z == 6:
            idxs.append(atom.GetIdx())
        elif scope == "hetero_h" and z in (7, 8, 16):
            idxs.append(atom.GetIdx())
        elif scope == "all_h" and z > 1:
            idxs.append(atom.GetIdx())
    return idxs


def attach_substituent(core: Chem.Mol, atom_idx: int, sub_smiles: str) -> Chem.Mol | None:
    """Attach a substituent to atom `atom_idx` of the core, replacing one H.

    The substituent attachment point is its atom index 0. One H is released on
    both ends to respect valences. Returns None if the result is not chemically
    valid (e.g. no H available or valence exceeded).
    """
    frag = Chem.MolFromSmiles(sub_smiles)
    if frag is None:
        return None

    combo = Chem.RWMol(Chem.CombineMols(core, frag))
    n_core = core.GetNumAtoms()
    anchor = n_core  # atom 0 of the fragment in the combined mol

    for idx in (atom_idx, anchor):
        at = combo.GetAtomWithIdx(idx)
        h = at.GetTotalNumHs()
        if h < 1:
            return None
        at.SetNumExplicitHs(h - 1)
        at.SetNoImplicit(True)

    combo.AddBond(atom_idx, anchor, Chem.BondType.SINGLE)
    mol = combo.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


# ---------------------------------------------------------------------------
# Strategy 1: position decoration (R-group)
# ---------------------------------------------------------------------------
def enumerate_decorations(
    lead_smiles: str,
    substituents: dict[str, str] | None = None,
    positions: Sequence[int] | None = None,
    scope: str = "aromatic_ch",
    n_substitutions: int | Iterable[int] = 1,
    max_products: int = 2000,
    max_heavy_atoms: int | None = 50,
    allow_same_substituent_repeat: bool = True,
) -> GenerationResult:
    """Generate analogs by decorating positions of the lead molecule.

    Key parameters:
      positions          -> the SITE of branching. List of atom indices.
                            If None, detected automatically according to `scope`.
      n_substitutions    -> the NUMBER of simultaneous substitutions. An int
                            (e.g. 2 = disubstitution) or an iterable (e.g.
                            [1, 2] = mono and di).
      substituents       -> dict {name: fragment_SMILES}. Defaults to
                            SMALL_SUBSTITUENTS to avoid combinatorial explosion.
      scope              -> automatic position-detection criterion.
      max_products       -> cap on generated analogs (safety).
      max_heavy_atoms    -> discards products that are too large.

    Returns a GenerationResult with unique canonical SMILES.
    """
    if substituents is None:
        substituents = SMALL_SUBSTITUENTS

    lead = Chem.MolFromSmiles(lead_smiles)
    if lead is None:
        raise ValueError(f"Invalid lead SMILES: {lead_smiles!r}")
    lead_canon = Chem.MolToSmiles(lead)
    lead_key = Chem.MolToInchiKey(lead)

    if positions is None:
        positions = substituent_positions(lead, scope=scope)
    positions = list(positions)
    if not positions:
        raise ValueError(
            f"No substitutable positions found (scope={scope!r}). "
            "Try another scope or pass `positions` manually."
        )

    if isinstance(n_substitutions, int):
        n_values = [n_substitutions]
    else:
        n_values = list(n_substitutions)

    sub_items = list(substituents.items())  # [(name, smiles), ...]

    seen: set[str] = {lead_key}
    result = GenerationResult(lead_smiles=lead_canon, method="decoration")

    for n in n_values:
        if n < 1 or n > len(positions):
            continue
        # combinations of POSITIONS to decorate simultaneously
        for pos_combo in itertools.combinations(positions, n):
            # assignment of one substituent to each chosen position
            if allow_same_substituent_repeat:
                sub_assignments = itertools.product(sub_items, repeat=n)
            else:
                sub_assignments = itertools.permutations(sub_items, n)

            for assignment in sub_assignments:
                mol = Chem.Mol(lead)  # clean copy of the lead
                ok = True
                labels = []
                # Apply from higher to lower index so we don't invalidate the
                # indices of positions not yet processed (attach only ADDS atoms
                # at the end, so original core indices are preserved; still, we
                # keep this for conceptual robustness).
                for pos, (name, smi) in zip(pos_combo, assignment):
                    mol = attach_substituent(mol, pos, smi)
                    if mol is None:
                        ok = False
                        break
                    labels.append((pos, name))
                if not ok or mol is None:
                    continue

                if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
                    continue

                key = inchikey(mol)
                if key is None or key in seen:
                    continue
                seen.add(key)
                smi_canon = Chem.MolToSmiles(mol)
                result.analogs.append(smi_canon)
                result.provenance.append({
                    "smiles": smi_canon,
                    "method": "decoration",
                    "n_subs": n,
                    "sites": [p for p, _ in labels],
                    "substituents": [nm for _, nm in labels],
                })
                if len(result.analogs) >= max_products:
                    return result
    return result


# ---------------------------------------------------------------------------
# Strategy 2: BRICS recombination
# ---------------------------------------------------------------------------
def brics_analogs(
    lead_smiles: str,
    extra_fragments: Sequence[str] | None = None,
    max_products: int = 200,
    max_heavy_atoms: int | None = 50,
    seed: int = 0xC0FFEE,
) -> GenerationResult:
    """Generate analogs by BRICS decomposition/recombination.

    Fragments the lead along BRICS bonds and reassembles by combining those
    fragments (and optional `extra_fragments`). Explores rearrangements and
    new combinations of pieces.
    """
    lead = Chem.MolFromSmiles(lead_smiles)
    if lead is None:
        raise ValueError(f"Invalid lead SMILES: {lead_smiles!r}")
    lead_canon = Chem.MolToSmiles(lead)
    lead_key = Chem.MolToInchiKey(lead)

    frags = set(BRICS.BRICSDecompose(lead))
    if extra_fragments:
        frags.update(extra_fragments)

    frag_mols = []
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is not None:
            frag_mols.append(m)

    result = GenerationResult(lead_smiles=lead_canon, method="brics")
    if len(frag_mols) < 2:
        # Not enough fragments to recombine.
        return result

    seen: set[str] = {lead_key}
    # BRICSBuild uses the global random generator; we fix the seed so the
    # enumeration is reproducible (RDKit does not accept a `seed` kwarg).
    random.seed(seed)
    builder = BRICS.BRICSBuild(frag_mols)
    try:
        for i, mol in enumerate(builder):
            if len(result.analogs) >= max_products:
                break
            if mol is None:
                continue
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
                continue
            key = inchikey(mol)
            if key is None or key in seen:
                continue
            seen.add(key)
            smi_canon = Chem.MolToSmiles(mol)
            result.analogs.append(smi_canon)
            result.provenance.append({
                "smiles": smi_canon,
                "method": "brics",
            })
            # Hard safeguard against very long generators.
            if i > max_products * 50:
                break
    except Exception:
        pass
    return result


def generate(
    lead_smiles: str,
    methods: Sequence[str] = ("decoration",),
    **kwargs,
) -> GenerationResult:
    """Facade: combine strategies and return a unified GenerationResult.

    methods: any subset of {"decoration", "brics"}.
    kwargs are dispatched by prefix:
      - dec_*   -> enumerate_decorations (without the prefix)
      - brics_* -> brics_analogs (without the prefix)
    """
    lead = Chem.MolFromSmiles(lead_smiles)
    if lead is None:
        raise ValueError(f"Invalid lead SMILES: {lead_smiles!r}")
    combined = GenerationResult(lead_smiles=Chem.MolToSmiles(lead), method="+".join(methods))
    seen: set[str] = {Chem.MolToInchiKey(lead)}

    dec_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("dec_")}
    brics_kwargs = {k[6:]: v for k, v in kwargs.items() if k.startswith("brics_")}

    parts: list[GenerationResult] = []
    if "decoration" in methods:
        parts.append(enumerate_decorations(lead_smiles, **dec_kwargs))
    if "brics" in methods:
        parts.append(brics_analogs(lead_smiles, **brics_kwargs))

    for part in parts:
        for smi, prov in zip(part.analogs, part.provenance):
            key = inchikey(smi)
            if key is None or key in seen:
                continue
            seen.add(key)
            combined.analogs.append(smi)
            combined.provenance.append(prov)
    return combined
