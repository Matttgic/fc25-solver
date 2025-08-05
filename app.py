import pandas as pd
import pulp
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Chargement et préparation des données
DATA_PATH = "player-data-full-2025-june.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# On renomme les colonnes sensibles
df = df.rename(columns={"positions": "position", "value": "value_eur", "club_name": "club"})

# Filtrage : garder les lignes avec des infos essentielles
df = df.dropna(subset=["position", "name", "value_eur", "club"])

# Nettoyage des données
df = df[df["value_eur"].apply(lambda x: str(x).replace(".", "").isdigit())]
df["value_eur"] = df["value_eur"].astype(float)
df["position"] = df["position"].astype(str)

# Colonnes numériques filtrables
numeric_criteria = sorted([col for col in df.select_dtypes(include=["float", "int"]).columns if df[col].nunique() > 20 and not col.startswith("gk_")])

# Formations possibles
formations = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1}
}

# Utilitaires pour filtrer par rôle
def poste_est(role, poste):
    role = role.upper()
    poste = poste.upper()
    if poste == "GK":
        return role == "GK"
    if poste == "DF":
        return any(p in role for p in ["CB", "LB", "RB"])
    if poste == "MF":
        return any(p in role for p in ["CM", "CDM", "CAM", "LM", "RM"])
    if poste == "FW":
        return any(p in role for p in ["ST", "CF", "RW", "LW"])
    return False

# Optimisation
def generer_equipe(critere, budget_millions, formation):
    budget = budget_millions * 1_000_000
    compo = formations[formation]

    model = pulp.LpProblem("Equipe_Optimale", pulp.LpMaximize)
    joueurs = list(df.index)
    x = pulp.LpVariable.dicts("joueur", joueurs, 0, 1, pulp.LpBinary)

    # Objectif
    model += pulp.lpSum([x[i] * df.loc[i, critere] for i in joueurs if not pd.isna(df.loc[i, critere])]), "Score_Total"

    # Contraintes
    for poste, nb in compo.items():
        model += pulp.lpSum([x[i] for i in joueurs if poste_est(df.loc[i, "position"], poste)]) == nb

    model += pulp.lpSum([x[i] * df.loc[i, "value_eur"] for i in joueurs]) <= budget
    model += pulp.lpSum([x[i] for i in joueurs]) == 11

    model.solve()

    selection = []
    for i in joueurs:
        if x[i].varValue == 1:
            joueur = df.loc[i]
            role = joueur["position"]
            selection.append({
                "nom": joueur["name"],
                "club": joueur["club"],
                "poste": role,
                "valeur": int(joueur["value_eur"]),
                "role": [p for p in compo if poste_est(role, p)][0],
                "critere": joueur.get(critere, "—")
            })

    return selection

@app.route('/', methods=['GET', 'POST'])
def index():
    equipe = []
    selected_critere = None
    if request.method == 'POST':
        try:
            budget = float(request.form['budget'])  # en millions
            critere = request.form['critere']
            formation = request.form['formation']
            selected_critere = critere
            equipe = generer_equipe(critere, budget, formation)
        except Exception as e:
            print("Erreur :", e)
            equipe = []

    return render_template('index.html', criteres=numeric_criteria, formations=formations.keys(), equipe=equipe, selected_critere=selected_critere)

if __name__ == '__main__':
    app.run(debug=True)
