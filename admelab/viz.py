"""
admelab.viz
===========
Visualization of molecules and ADME profiles.

  - `mol_grid`         : grid of structures with property legends.
  - `highlight_changes`: highlights what changed in an analog vs the lead.
  - `scatter`          : scatter of two properties (color by a third).
  - `property_hist`    : histograms of property distributions.
  - `radar`            : radar profile (desirabilities) of one or more molecules.

Intended for use inside the notebook (returns images/figures displayed in the
cell output).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
def _legend_from_row(row: pd.Series, fields: Sequence[str]) -> str:
    parts = []
    for f in fields:
        if f in row and pd.notna(row[f]):
            v = row[f]
            if isinstance(v, float):
                v = f"{v:.2f}"
            parts.append(f"{f}={v}")
    return " ".join(parts)


def mol_grid(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    legend_fields: Sequence[str] = ("score", "MW", "LD50_mg_per_kg", "QED"),
    n: int = 12,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (300, 220),
    legend_prefix: Optional[str] = None,
):
    """Grid of molecules with legends built from DataFrame columns.

    Returns a (PIL) image that the notebook displays directly.
    """
    sub = df.head(n)
    mols, legends = [], []
    for _, row in sub.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        mols.append(mol)
        legend = _legend_from_row(row, legend_fields)
        if legend_prefix and legend_prefix in row:
            legend = f"{row[legend_prefix]} | {legend}"
        legends.append(legend)
    if not mols:
        raise ValueError("No valid molecules to draw.")
    return Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
        legends=legends, useSVG=False,
    )


def highlight_changes(lead_smiles: str, analog_smiles: str, size=(420, 320)):
    """Draw the analog highlighting the atoms that are NOT part of the maximum
    common substructure with the lead (i.e. the introduced modification)."""
    lead = Chem.MolFromSmiles(lead_smiles)
    analog = Chem.MolFromSmiles(analog_smiles)
    if lead is None or analog is None:
        raise ValueError("Invalid SMILES.")

    mcs = rdFMCS.FindMCS([lead, analog], timeout=5)
    common = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    match = set(analog.GetSubstructMatch(common)) if common else set()
    changed = [a.GetIdx() for a in analog.GetAtoms() if a.GetIdx() not in match]

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, analog, highlightAtoms=changed,
        highlightAtomColors={i: (1.0, 0.6, 0.6) for i in changed},
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    try:
        from IPython.display import Image
        return Image(png)
    except ImportError:
        return png


# ---------------------------------------------------------------------------
# Property plots (matplotlib)
# ---------------------------------------------------------------------------
def scatter(
    df: pd.DataFrame,
    x: str = "MW",
    y: str = "LogP",
    color: Optional[str] = "score",
    highlight_lead: Optional[str] = None,
    lead_smiles: Optional[str] = None,
    figsize=(7, 5),
):
    """Scatter of two properties, colored by a third."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    c = df[color] if (color and color in df.columns) else None
    sc = ax.scatter(df[x], df[y], c=c, cmap="viridis", s=45,
                    edgecolor="k", linewidth=0.3, alpha=0.85)
    if c is not None:
        fig.colorbar(sc, ax=ax, label=color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def property_hist(
    df: pd.DataFrame,
    columns: Sequence[str] = ("MW", "LogP", "TPSA", "QED", "LD50_mg_per_kg"),
    figsize=(12, 7),
):
    """Histograms of the distributions of several properties."""
    import matplotlib.pyplot as plt
    cols = [c for c in columns if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(data, bins=20, color="#4c72b0", edgecolor="white")
        ax.set_title(col)
        ax.grid(alpha=0.2)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def radar(
    df: pd.DataFrame,
    rows: Sequence[int] = (0,),
    desirability_cols: Optional[Sequence[str]] = None,
    label_col: Optional[str] = None,
    figsize=(6, 6),
):
    """Radar of desirabilities (d_* columns) for one or more molecules.

    Requires ranking.add_composite_score to have been run (it creates
    'd_<property>' columns in [0,1]). Ideal to compare the profile of the tops.
    """
    import matplotlib.pyplot as plt
    if desirability_cols is None:
        desirability_cols = [c for c in df.columns if c.startswith("d_")]
    if not desirability_cols:
        raise ValueError("No desirability columns (d_*). "
                         "Run ranking.add_composite_score first.")

    labels = [c[2:] for c in desirability_cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i in rows:
        if i >= len(df):
            continue
        row = df.iloc[i]
        values = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in desirability_cols]
        values += values[:1]
        name = str(row[label_col]) if (label_col and label_col in df.columns) else f"#{i}"
        ax.plot(angles, values, linewidth=1.8, label=name)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("ADME desirability profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig
