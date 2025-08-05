from flask import Flask, render_template, request
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import os

app = Flask(__name__)

# Chargement des données
DATA_PATH = "player-data-full-2025-june.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["positions", "name"])

# On transforme les colonnes numériques
numeric_criteria = df.select_dtypes(include='number').columns.tolist()

# Formations possibles
formations = {
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}
}

# Simplification du poste
def simplify_position(pos):
    if "GK" in pos:
        return "GK"
    elif any(x in pos for x in ["CB", "LB", "RB"]):
        return "DEF"
    elif any(x in pos for x in ["CM", "CDM", "CAM", "LM", "RM"]):
        return "MID"
    elif any(x in pos for x in ["ST", "CF", "RW", "LW"]):
        return "FWD"
    else:
        return "MID"

df["role"] = df["positions"].apply(simplify_position)

@app.route('/', methods=['GET', 'POST'])
def index():
    equipe = []
    selected_critere = None

    if request.method == 'POST':
        selected_critere = request.form['critere']
        budget_input = request.form['budget']
        formation_name = request.form['formation']

        # gestion format budget : "500" => 500M
        try:
            budget = float(budget_input) * 1_000_000
        except:
            budget = 500_000_000  # valeur par défaut

        formation = formations.get(formation_name, {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})

        model = LpProblem(name="team-builder", sense=LpMaximize)
        players = df.index.tolist()
        x = {i: LpVariable(name=f"x_{i}", cat="Binary") for i in players}

        # Objectif : maximiser le critère sélectionné
        model += lpSum(df.loc[i, selected_critere] * x[i] for i in players)

        # Contraintes : budget
        model += lpSum(df.loc[i, "value"] * x[i] for i in players) <= budget

        # Contraintes : formation
        for role, count in formation.items():
            model += lpSum(x[i] for i in players if df.loc[i, "role"] == role) == count

        # Contraintes : 11 joueurs
        model += lpSum(x[i] for i in players) == 11

        # Résolution
        model.solve()

        # Résultat
        selected = [i for i in players if x[i].value() == 1]
        equipe = df.loc[selected].to_dict(orient="records")

    return render_template('index.html',
                           criteres=numeric_criteria,
                           formations=list(formations),
                           equipe=equipe,
                           selected_critere=selected_critere)

# Pour Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
