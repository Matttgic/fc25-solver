from flask import Flask, render_template, request
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary
import os

app = Flask(__name__)

# Extraction automatique si le CSV n’est pas encore présent
if os.path.exists("fc25-solver.zip") and not os.path.exists("player-data-full-2025-june.csv"):
    import zipfile
    with zipfile.ZipFile("fc25-solver.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Chargement des données
df = pd.read_csv("player-data-full-2025-june.csv")
df = df.dropna(subset=["positions", "name"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["value"] = df["value"].fillna(0)

# Formations disponibles
formations_dict = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
    "4-2-3-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
    "4-1-4-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1}
}

# Colonnes numériques disponibles comme critères
numeric_criteria = df.select_dtypes(include='number').columns.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    equipe = []
    selected_critere = None

    if request.method == "POST":
        try:
            critere = request.form.get("critere")
            selected_critere = critere
            budget = float(request.form.get("budget", "0")) * 1_000_000  # format M€
            formation_name = request.form.get("formation")
            formation = formations_dict.get(formation_name, formations_dict["4-4-2"])

            df_ = df[["name", "positions", "value", critere]].dropna()
            df_["value"] = pd.to_numeric(df_["value"], errors="coerce")
            df_ = df_.dropna(subset=["value", critere])
            df_["positions_list"] = df_["positions"].str.split(",")

            prob = LpProblem("Team_Selection", LpMaximize)
            player_vars = {i: LpVariable(f"player_{i}", cat=LpBinary) for i in df_.index}

            # Contrainte budget
            prob += lpSum([player_vars[i] * df_.loc[i, "value"] for i in df_.index]) <= budget

            # Contrainte total joueurs
            prob += lpSum([player_vars[i] for i in df_.index]) == sum(formation.values())

            # Contraintes par poste
            for poste, nb in formation.items():
                prob += lpSum([
                    player_vars[i]
                    for i in df_.index
                    if poste in [p.strip() for p in df_.loc[i, "positions_list"]]
                ]) >= nb

            # Fonction objectif : maximiser le critère
            prob += lpSum([player_vars[i] * df_.loc[i, critere] for i in df_.index])

            prob.solve()

            selected_players = [i for i in df_.index if player_vars[i].varValue == 1.0]
            equipe = df_.loc[selected_players].to_dict(orient="records")

        except Exception as e:
            print("Erreur lors de la génération de l'équipe:", e)

    return render_template(
        "index.html",
        criteres=numeric_criteria,
        formations=list(formations_dict.keys()),
        equipe=equipe,
        selected_critere=selected_critere
    )

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
