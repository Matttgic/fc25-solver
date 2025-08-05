from flask import Flask, render_template, request import pandas as pd from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

app = Flask(name)

=== Chargement des données ===

DATA_PATH = "player-data-full-2025-june.csv" df = pd.read_csv(DATA_PATH)

Nettoyage de base

if "positions" in df.columns: df["positions"] = df["positions"].astype(str).str.split(',')

Ne garder que les joueurs avec un nom et une position

if "position" in df.columns: df = df.dropna(subset=["position", "name"])

=== Formations disponibles ===

formations_dict = { "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2}, "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3}, "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3}, "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1}, "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2}, "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1}, "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2}, "4-2-3-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1}, "3-2-3-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2}, } formations = list(formations_dict.keys())

=== Colonnes numériques sélectionnables ===

numeric_criteria = df.select_dtypes(include='number').columns.tolist()

@app.route('/', methods=['GET', 'POST']) def index(): selected_critere = request.form.get("critere") budget = float(request.form.get("budget", 100)) selected_formation = request.form.get("formation", "4-4-2")

if request.method == 'POST' and selected_critere:
    try:
        formation = formations_dict[selected_formation]
        result = solve_team(df, budget, selected_critere, formation)
        return render_template("index.html", criteres=numeric_criteria,
                               formations=formations,
                               equipe=result,
                               selected_critere=selected_critere,
                               selected_formation=selected_formation,
                               budget=budget)
    except Exception as e:
        return f"Erreur dans le solver : {str(e)}"

return render_template("index.html",
                       criteres=numeric_criteria,
                       formations=formations,
                       equipe=None,
                       selected_critere=None,
                       selected_formation="4-4-2",
                       budget=100)

def solve_team(df, budget, critere, formation): df_filtered = df.dropna(subset=[critere, "positions", "value", "name"]) players = df_filtered.copy()

# Initialisation du problème
prob = LpProblem("Team_Selection", LpMaximize)

x = LpVariable.dicts("player", players.index, cat=LpBinary)

# Objectif : maximiser le critère choisi
prob += lpSum([x[i] * players.loc[i, critere] for i in players.index])

# Contraintes de formation (par position)
for pos, count in formation.items():
    prob += lpSum([x[i] for i in players.index if pos in players.loc[i, "positions"]]) == count

# Budget en millions d’euros
prob += lpSum([x[i] * players.loc[i, "value"] for i in players.index]) <= budget

# Une seule fois chaque joueur
prob += lpSum([x[i] for i in players.index]) == sum(formation.values())

prob.solve()

selected = players.loc[[i for i in players.index if x[i].varValue == 1]]
return selected.to_dict(orient="records")

if name == 'main': app.run(debug=True, port=5000, host="0.0.0.0")

