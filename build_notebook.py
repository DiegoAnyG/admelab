"""Build ADME_Lab.ipynb with nbformat (guaranteed valid JSON)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ---------------------------------------------------------------- cover
md(r"""
# 🧪 ADME Lab — Analog design & ADME / Tox prediction

Starting from a **lead molecule** this notebook:

1. **Generates** analogs (*R-group* position decoration and/or *BRICS*
   recombination), with control over the **branching site** and the **number of
   substitutions**.
2. **Predicts** **ADME** properties in two layers: RDKit descriptors/rules
   (instant) and **ADMET-AI** (ML models: absorption, BBB, CYPs, hERG, DILI,
   solubility, clearance… and **LD50** acute toxicity).
3. **Interprets toxicity**: converts LD50 to mg/kg and classifies it into
   **GHS** categories.
4. **Names** the best molecules (**IUPAC** via provenance + PubChem → CACTUS).
5. **Ranks and selects** a *top* (or the *worst*) with a configurable
   **composite score** and **range filters**.
6. **Visualizes**: structure grid, property scatter, profile radar and
   distributions.

> Run the cells top to bottom. Each section has its **parameters** at the start,
> clearly marked for you to edit.
""")

# ---------------------------------------------------------------- setup
md("## 0 · Setup")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image, display

from admelab import generation, predict, toxicity, naming, naming_smart, ranking, viz, pipeline
predict.quiet_ml_logs()   # silence PyTorch Lightning verbosity
print("OPSIN (verified naming):", "available" if naming_smart.opsin_available() else "not found")

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
print("admelab ready  (the 1st ML prediction loads the ADMET-AI model)")
""")

# ---------------------------------------------------------------- lead
md(r"""
## 1 · Lead molecule

Define your base structure as **SMILES**. Default is **ibuprofen**.
""")
code(r"""
LEAD = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"   # <-- change to your lead molecule (SMILES)

lead_mol = Chem.MolFromSmiles(LEAD)
assert lead_mol is not None, "invalid SMILES"
LEAD = Chem.MolToSmiles(lead_mol)      # canonicalize
display(Draw.MolToImage(lead_mol, size=(340, 240)))

# Base profile (RDKit layer, instant)
profile = predict.rdkit_profile(LEAD)
pd.Series(profile).to_frame("lead")
""")

md(r"""
### Substitutable positions (the *branching site*)

The numbers are **atom indices**. You can set them manually in `POSITIONS`
(section 2) to direct substitution to specific positions.
""")
code(r"""
positions = generation.substituent_positions(lead_mol, scope="aromatic_ch")
print("Aromatic C-H positions detected:", positions)

mol = Chem.Mol(lead_mol)
for atom in mol.GetAtoms():
    atom.SetProp("atomNote", str(atom.GetIdx()))
d = rdMolDraw2D.MolDraw2DCairo(420, 300)
rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
d.FinishDrawing()
Image(d.GetDrawingText())
""")

# ---------------------------------------------------------------- generation
md(r"""
## 2 · Generate analogs

Two strategies (choose one or both in `METHODS`):

- **`decoration`** — decorates positions with a substituent library. You control
  **where** (`POSITIONS`/`SCOPE`) and **how many** simultaneous substitutions
  (`N_SUBSTITUTIONS`).
- **`brics`** — fragments the lead and recombines (explores a broader space).

Available substituent libraries: `generation.DEFAULT_SUBSTITUENTS` (20 groups)
and `generation.SMALL_SUBSTITUENTS` (8, faster).
""")
code(r"""
# ---------------- GENERATION PARAMETERS ----------------
METHODS          = ("decoration", "brics")          # ("decoration",) / ("brics",) / both
SUBSTITUENTS     = generation.DEFAULT_SUBSTITUENTS   # or generation.SMALL_SUBSTITUENTS
SCOPE            = "aromatic_ch"    # aromatic_ch | aliphatic_ch | any_c | hetero_h | all_h
POSITIONS        = None             # None = automatic; or a list e.g. [10, 12]
N_SUBSTITUTIONS  = [1, 2]           # mono and di-substitution (int or list)
MAX_DECOR        = 250              # cap of analogs per decoration
MAX_BRICS        = 60               # cap of analogs per BRICS
# -------------------------------------------------------

res = generation.generate(
    LEAD, methods=METHODS,
    dec_substituents=SUBSTITUENTS, dec_scope=SCOPE, dec_positions=POSITIONS,
    dec_n_substitutions=N_SUBSTITUTIONS, dec_max_products=MAX_DECOR,
    brics_max_products=MAX_BRICS,
)
analog_smiles = res.analogs
print(f"Unique analogs generated: {len(res)}")
prov = pd.DataFrame(res.provenance)
prov.head(8)
""")

# ---------------------------------------------------------------- prediction
md(r"""
## 3 · Predict ADME + LD50

`USE_ML=True` uses **ADMET-AI** (ML, ~41 endpoints + LD50; runs on your GPU).
`USE_ML=False` is instant but only gives RDKit descriptors/rules (no LD50).
""")
code(r"""
USE_ML = True   # ADMET-AI (ML). Set to False for an instant ML-free calculation.

all_smiles = [LEAD] + analog_smiles          # include the lead for comparison
df = predict.predict_batch(all_smiles, use_ml=USE_ML)
df = toxicity.annotate_toxicity(df)          # adds LD50_mg_per_kg and GHS_category
df.insert(1, "is_lead", df["SMILES"] == LEAD)

print("Rows:", len(df), " · Columns:", df.shape[1])
key_cols = ["SMILES","is_lead","MW","LogP","TPSA","QED",
            "LD50_mg_per_kg","GHS_category","Lipinski_violations","n_alerts"]
df[key_cols].head()
""")

# ---------------------------------------------------------------- ranking
md(r"""
## 4 · Rank and select (top / worst)

The **composite score** combines several **objectives** normalized to
desirability [0-1]. Each objective is
`Objective(column, "max"|"min"|"target", weight, target=…)`.
Tune `OBJECTIVES` and the `FILTERS` (ranges) to your criteria.
""")
code(r"""
# ---------------- OBJECTIVES AND FILTERS ----------------
OBJECTIVES = [
    ranking.Objective("QED",                "max",    1.5),
    ranking.Objective("LD50_mg_per_kg",     "max",    1.5),   # less toxic = better
    ranking.Objective("Lipinski_violations","min",    1.0),
    ranking.Objective("n_alerts",           "min",    1.0),
    ranking.Objective("LogP",               "target", 1.0, target=2.5),
    ranking.Objective("Bioavailability_Ma", "max",    1.0),   # ADMET-AI endpoint
    ranking.Objective("hERG",               "min",    1.0),   # cardiotoxicity
]
# Automatic alternative: OBJECTIVES = ranking.default_objectives(df)

FILTERS = {                       # acceptance ranges (None = no bound)
    "MW": (None, 500),
    "LD50_mg_per_kg": (300, None),
}
TOP_N = 10
# -------------------------------------------------------

ranked = ranking.rank(df, objectives=OBJECTIVES, filters=FILTERS, which="top")
print(f"Molecules after filtering: {len(ranked)} (of {len(df)})")
top = ranking.top(ranked, n=TOP_N, by="score")
top[["SMILES","score","MW","LogP","QED","LD50_mg_per_kg","GHS_category",
     "Lipinski_violations","n_alerts"]]
""")

md("### The *worst* in range (to discard or analyze)")
code(r"""
worst = ranking.bottom(ranked, n=5, by="score")
worst[["SMILES","score","MW","LogP","LD50_mg_per_kg","GHS_category"]]
""")

# ---------------------------------------------------------------- naming
md(r"""
## 5 · IUPAC name of the *top* (self-verified)

Cascade system with **round-trip verification** (OPSIN): each name is converted
back to a structure and checked to reconstruct the exact molecule (column
`iupac_verified`). Candidate sources:

- **provenance** — offline and instant, for the analogs we generated (we know
  the substituent and the position);
- **PubChem** (known molecules) and **CACTUS** (novel).

Requires the JRE + OPSIN from `tools/` (see `tools_setup.sh`). Without them, it
falls back to PubChem/CACTUS without verification.
""")
code(r"""
NAME_TOP = 10
top_named = naming_smart.name_dataframe(
    top.head(NAME_TOP), gen_result=res, lead_smiles=LEAD, progress=False)

n_ver = int(top_named["iupac_verified"].sum())
print(f"Verified by round-trip: {n_ver}/{len(top_named)}")
top_named[["iupac_name","name_source","iupac_verified","score",
           "LD50_mg_per_kg","GHS_category"]]
""")

# ---------------------------------------------------------------- viz
md("## 6 · Visualization")
md("**Grid of the best molecules** (with their score and key properties)")
code(r"""
viz.mol_grid(top, legend_fields=["score","MW","LD50_mg_per_kg","QED"],
             n=8, mols_per_row=4)
""")
md("**Scatter** of properties (color = score). The chemical space explored.")
code(r"""
viz.scatter(ranked, x="MW", y="LogP", color="score");
""")
md("**Radar** of desirability of the top 3 + property **distributions**")
code(r"""
viz.radar(ranked, rows=[0, 1, 2]);
""")
code(r"""
viz.property_hist(df, columns=["MW","LogP","TPSA","QED","LD50_mg_per_kg","ESOL_logS"]);
""")
md("**Structural change** of the top-1 vs the lead (new part in red)")
code(r"""
viz.highlight_changes(LEAD, top.iloc[0]["SMILES"])
""")

# ---------------------------------------------------------------- export
md("## 7 · Export results")
code(r"""
OUT = "adme_results.csv"
ranked.to_csv(OUT, index=False)
print(f"Saved {OUT}  ({len(ranked)} molecules, {ranked.shape[1]} columns)")
""")

# ---------------------------------------------------------------- panel
md(r"""
## 8 · (Optional) Interactive panel

Change the parameters with the controls and click **Run** to re-run the whole
flow (generate → predict → rank) and see the *top*. A convenience extra.
""")
code(r"""
import ipywidgets as W
from IPython.display import clear_output

w_lead  = W.Text(value=LEAD, description="Lead:", layout=W.Layout(width="60%"))
w_meth  = W.SelectMultiple(options=["decoration","brics"], value=("decoration",),
                           description="Methods:")
w_nsub  = W.SelectMultiple(options=[1,2,3], value=(1,), description="N subs:")
w_ml    = W.Checkbox(value=True, description="Use ADMET-AI (ML)")
w_topn  = W.IntSlider(value=8, min=3, max=20, description="Top N:")
w_ld50  = W.IntSlider(value=300, min=0, max=2000, step=50, description="LD50 min:")
out = W.Output()

@W.interact_manual(lead=w_lead, methods=w_meth, nsub=w_nsub,
                   use_ml=w_ml, top_n=w_topn, ld50_min=w_ld50)
def _run(lead, methods, nsub, use_ml, top_n, ld50_min):
    with out:
        clear_output()
    r = pipeline.run_pipeline(
        lead, methods=tuple(methods) or ("decoration",),
        use_ml=use_ml, substituents=generation.DEFAULT_SUBSTITUENTS,
        n_substitutions=list(nsub) or [1],
        filters={"LD50_mg_per_kg": (ld50_min, None)},
    )
    print(f"Generated {r.n_generated} · scored {r.n_scored}")
    cols = ["SMILES","score","MW","LogP","QED","LD50_mg_per_kg","GHS_category"]
    cols = [c for c in cols if c in r.ranked.columns]
    display(r.ranked.head(top_n)[cols])
""")

md(r"""
---
### Notes and limitations

- ADMET-AI predictions are **estimates** (models trained on TDC), useful to
  **prioritize**, not to decide definitively.
- **LD50** is predicted for acute oral toxicity (*Zhu* dataset); the mg/kg
  conversion uses the molecular weight and was calibrated with reference drugs.
- The **IUPAC name** is emitted with **structural verification** (OPSIN
  round-trip): a name with `iupac_verified=True` reconstructs the molecule
  *exactly*; those flagged `False` are unverified proposals. The provenance path
  guarantees structural correctness, not necessarily the single canonical
  *Preferred IUPAC Name*.
- Generating many analogs × ML can be slow; tune `MAX_DECOR`/`MAX_BRICS`.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (adme)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open("ADME_Lab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"ADME_Lab.ipynb written with {len(cells)} cells.")
