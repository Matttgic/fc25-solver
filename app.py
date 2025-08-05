import os
from flask import Flask, render_template, request
import pandas as pd
from pulp import LpMaximize, LpProblem, lpSum, LpVariable

# Config
DATA_PATH = "player-data-full-2025-june.csv"
PORT = int(os.environ.get("PORT", 5000))  # Render injecte le port ici

# App Flask
app = Flask(__name__)

# Chargement du CSV
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["position", "name"])

# Rendre toutes les colonnes disponibles comme critères
all_columns = df.columns.tolist()

# Route principale
@app.route("/", methods=["GET", "POST"])
def index():
    selected_criteria = []
    budget_millions = 500
    team = []

    if request.method == "POST":
        selected_criteria = request.form.getlist("criteria")
        budget_millions = float(request.form.get("budget", 500))
        team = generate_team(selected_criteria, budget_millions)

    return render_template("index.html",
                           columns=all_columns,
                           selected_criteria=selected_criteria,
                           budget=budget_millions,
                           team=team)

# Solver
def generate_team(criteria, budget_m):
    team_structure = {"GK": 1, "DF": 4, "MF": 4, "FW": 2}
    problem = LpProblem("Best_Team", LpMaximize)

    players = []
    for i, row in df.iterrows():
        if not all(col in row and pd.notna(row[col]) for col in criteria):
            continue
        player = {
            "id": i,
            "name": row["name"],
            "position": row["position"],
            "value": row.get("value_eur", 0) / 1e6,
            "score": sum(float(row[c]) for c in criteria)
        }
        players.append(player)

    player_vars = {p["id"]: LpVariable(f"player_{p['id']}", cat="Binary") for p in players}
    problem += lpSum(p["score"] * player_vars[p["id"]] for p in players), "TotalScore"
    problem += lpSum(p["value"] * player_vars[p["id"]] for p in players) <= budget_m, "BudgetConstraint"

    for pos, count in team_structure.items():
        problem += lpSum(player_vars[p["id"]] for p in players if p["position"] == pos) == count, f"{pos}_count"

    problem.solve()

    return [p for p in players if player_vars[p["id"]].value() == 1]

# Lancer l'app avec les bons paramètres pour Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
