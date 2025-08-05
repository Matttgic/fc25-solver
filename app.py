from flask import Flask, render_template, request
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

app = Flask(__name__)
DATA_PATH = "player-data-full-2025-june.csv"

# Chargement des données
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["positions", "name"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Formations possibles : dict(poste : nb joueurs)
formations = {
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
}

def poste_simplifie(positions):
    if isinstance(positions, str):
        if "GK" in positions:
            return "GK"
        elif any(p in positions for p in ["CB", "RB", "LB", "RWB", "LWB"]):
            return "DEF"
        elif any(p in positions for p in ["CDM", "CM", "CAM", "RM", "LM"]):
            return "MID"
        elif any(p in positions for p in ["ST", "CF", "RW", "LW"]):
            return "FWD"
    return "AUTRE"

df["poste_simplifie"] = df["positions"].apply(poste_simplifie)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", numeric_columns=numeric_columns, team=None, formations=formations)

@app.route("/build_team", methods=["POST"])
def build_team():
    budget = float(request.form["budget"]) * 1e6
    criterion = request.form["criterion"]
    formation = request.form["formation"]
    poste_requis = formations[formation]

    data = df.dropna(subset=["value", criterion])
    data = data[data["poste_simplifie"].isin(poste_requis.keys())]

    players = list(data.index)
    model = LpProblem("TeamSelection", LpMaximize)
    x = LpVariable.dicts("x", players, cat=LpBinary)

    # Objectif
    model += lpSum(data.loc[i, criterion] * x[i] for i in players)

    # Contraintes de budget
    model += lpSum(data.loc[i, "value"] * x[i] for i in players) <= budget

    # Contraintes de poste
    for poste, nombre in poste_requis.items():
        model += lpSum(x[i] for i in players if data.loc[i, "poste_simplifie"] == poste) == nombre

    # Total joueurs = 11
    model += lpSum(x[i] for i in players) == 11

    model.solve()

    selected_players = data.loc[[i for i in players if x[i].value() == 1]]
    selected_players = selected_players[["name", "club_name", "positions", "poste_simplifie", "value", criterion]]

    return render_template("index.html", numeric_columns=numeric_columns, team=selected_players, formations=formations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
