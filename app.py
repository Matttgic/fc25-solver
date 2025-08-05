from flask import Flask, render_template, request
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

app = Flask(__name__)

# Charger les données
DATA_PATH = "player-data-full-2025-june.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# Nettoyage de base
df = df.dropna(subset=["positions", "name"])
df["positions"] = df["positions"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "UNK")

# Colonnes numériques disponibles pour optimiser
numeric_criteria = df.select_dtypes(include=["number"]).columns.tolist()
default_critere = "sprint_speed" if "sprint_speed" in numeric_criteria else numeric_criteria[0]

# Toutes les formations possibles
formations = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
}

# Fonction pour associer un poste principal simplifié
def simplifie_poste(row):
    if "GK" in row["positions"]:
        return "GK"
    elif "DF" in row["positions"]:
        return "DF"
    elif "MF" in row["positions"]:
        return "MF"
    elif "FW" in row["positions"]:
        return "FW"
    return "UNK"

df["poste"] = df.apply(simplifie_poste, axis=1)
df["value"] = pd.to_numeric(df["value"], errors="coerce")

@app.route("/", methods=["GET", "POST"])
def index():
    equipe = []
    selected_critere = default_critere
    selected_formation = "4-4-2"
    budget = 500_000_000  # budget par défaut : 500M

    if request.method == "POST":
        try:
            selected_critere = request.form.get("critere", default_critere)
            selected_formation = request.form.get("formation", "4-4-2")
            budget = int(float(request.form.get("budget", "500")) * 1_000_000)

            poste_structure = formations.get(selected_formation, formations["4-4-2"])

            # Filtrer uniquement les joueurs valides pour le critère choisi
            df_valid = df.dropna(subset=[selected_critere, "value"])

            # Variables de décision
            joueurs = df_valid.index.tolist()
            x = LpVariable.dicts("x", joueurs, cat=LpBinary)

            prob = LpProblem("TeamSelection", LpMaximize)

            # Objectif : maximiser le score du critère
            prob += lpSum(df_valid.loc[i, selected_critere] * x[i] for i in joueurs)

            # Contraintes par poste
            for poste, nb in poste_structure.items():
                prob += lpSum(x[i] for i in joueurs if df_valid.loc[i, "poste"] == poste) == nb

            # Contraintes de budget
            prob += lpSum(df_valid.loc[i, "value"] * x[i] for i in joueurs) <= budget

            prob.solve()

            equipe = df_valid.loc[[i for i in joueurs if x[i].varValue == 1]].to_dict(orient="records")
        except Exception as e:
            print(f"Erreur : {e}")
            equipe = []

    return render_template("index.html",
                           criteres=numeric_criteria,
                           formations=list(formations.keys()),
                           equipe=equipe,
                           selected_critere=selected_critere,
                           selected_formation=selected_formation,
                           budget=budget // 1_000_000)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
