"""Build Reactions_Demo.ipynb (generic example, no private data)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# ⚗️ Reactions Demo — targeted reactions + ADME/Tox

**Self-contained** example (no external files) of the `admelab` reaction engine.
Starting from a **generic acid** (benzoic acid) and libraries of alcohols and
amines:

1. Generate **esters** and **amides** with real SMARTS reactions (stereochemistry
   preserved), validated by **exact atomic formula**.
2. Predict **ADME + LD50** of all products.
3. Assign **self-verified IUPAC names** via an OPSIN round-trip.
4. Rank with a composite score.

> The engine is **extensible**: adding a reaction = adding an entry to
> `admelab.reactions.REACTIONS`. Here we show `esterification` and `amidation`.
""")

md("## 0 · Setup")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from admelab import reactions as rx, esterification as est
from admelab import predict, toxicity, naming_smart, ranking, viz
predict.quiet_ml_logs()
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 200)
print("Available reactions:", list(rx.REACTIONS))
""")

md("## 1 · Acid core and libraries (all inline)")
code(r"""
ACID = "OC(=O)c1ccccc1"     # benzoic acid (replace with your core)

alcohols = pd.DataFrame({
    "name":  ["benzyl", "cyclohexyl", "1-butanol", "menthol", "2-phenylethanol"],
    "SMILES":["OCc1ccccc1","OC1CCCCC1","OCCCC","CC(C)C1CCC(C)CC1O","OCCc1ccccc1"],
})
amines = pd.DataFrame({
    "name":  ["benzylamine", "morpholine", "aniline", "piperidine"],
    "SMILES":["NCc1ccccc1","C1COCCN1","Nc1ccccc1","C1CCNCC1"],
})
Draw.MolToImage(Chem.MolFromSmiles(ACID), size=(240,180))
""")

md("## 2 · Reactions + validation")
code(r"""
# Esterification (specialized path, with OH classification and Fischer viability)
esters = est.esterify_library(ACID, alcohols, smiles_col="SMILES",
                              name_col="name", policy="preferred")
esters["class"] = "ester"

# Amidation (generic path of the reaction engine)
amides = rx.react_library(ACID, amines, reaction="amidation",
                          smiles_col="SMILES", name_col="name", policy="first")
amides["class"] = "amide"

prod = pd.concat([esters[["name","SMILES","valid","status","class"]],
                  amides[["name","SMILES","valid","status","class"]]],
                 ignore_index=True)
print(f"Products: {len(prod)}  |  valid: {int(prod['valid'].sum())}/{len(prod)}")
print(prod["class"].value_counts().to_string())
prod
""")

md("## 3 · ADME + LD50")
code(r"""
ok = prod[prod["valid"]].reset_index(drop=True)
df = predict.predict_batch(ok["SMILES"].tolist(), use_ml=True)
df = toxicity.annotate_toxicity(df)
df = df.merge(ok[["SMILES","name","class"]], on="SMILES", how="left")
df[["name","class","MW","LogP","QED","LD50_mg_per_kg","GHS_category"]].head(12)
""")

md("## 4 · Ranking")
code(r"""
OBJ = [
    ranking.Objective("QED","max",1.5),
    ranking.Objective("LD50_mg_per_kg","max",1.5),
    ranking.Objective("Lipinski_violations","min",1.0),
    ranking.Objective("n_alerts","min",1.0),
]
ranked = ranking.rank(df, objectives=OBJ, which="top")
ranked.head(10)[["name","class","score","MW","LogP","LD50_mg_per_kg","GHS_category"]]
""")

md("## 5 · Self-verified IUPAC names (esters)")
code(r"""
top_e = ranked[ranked["class"]=="ester"].head(6).merge(
    esters[["SMILES","alcohol_SMILES"]], on="SMILES", how="left")
named = naming_smart.name_esters_batch(
    top_e["SMILES"].tolist(), top_e["alcohol_SMILES"].tolist(),
    acid_smiles=ACID, progress=False)
top_e = top_e.merge(named, on="SMILES", how="left")
print(f"Verified: {int(top_e['iupac_verified'].sum())}/{len(top_e)}")
top_e[["name","iupac_name","name_source","iupac_verified","score"]]
""")

md("## 6 · Visualization")
code(r"""
viz.mol_grid(ranked, legend_fields=["score","LD50_mg_per_kg"],
             legend_prefix="name", n=9, mols_per_row=3)
""")

md(r"""
---
**To use your own system:** change `ACID` and the libraries above, or load an
`.xlsx`/`.csv` with `est.load_alcohols(...)`. For a new reaction, add a
`Reaction` to `admelab.reactions.REACTIONS` and call `rx.react_library(...)`.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (adme)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open("Reactions_Demo.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Reactions_Demo.ipynb written with {len(cells)} cells.")
