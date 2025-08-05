import os
import zipfile
import pandas as pd
from flask import Flask, render_template, request

# Décompression si nécessaire
if os.path.exists("fc25-solver.zip") and not os.path.exists("player-data-full-2025-june.csv"):
    with zipfile.ZipFile("fc25-solver.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Chargement des données
DATA_PATH = "player-data-full-2025-june.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# Nettoyage
df = df.dropna(subset=["positions", "name"])
df["positions"] = df["positions"].astype(str)

# App Flask
app = Flask(__name__)

# Fonction utilitaire pour sélectionner des joueurs par poste
def select_players(df, formation_dict):
    team = []
    used_names = set()
    for position, count in formation_dict.items():
        candidates = df[df["positions"].str.contains(position)]
        candidates = candidates[~candidates["name"].isin(used_names)]
        candidates = candidates.sort_values("overall_rating", ascending=False)
        selected = candidates.head(count)
        team.extend(selected.to_dict(orient="records"))
        used_names.update(selected["name"])
    return team

# Route principale
@app.route("/", methods=["GET", "POST"])
def index():
    formations = {
        "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
        "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
        "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
        "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
        "4-2-3-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
        "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
        "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2}
    }

    selected_formation = request.form.get("formation", "4-4-2")
    formation_dict = formations.get(selected_formation, formations["4-4-2"])
    team = select_players(df, formation_dict)

    return render_template("index.html",
                           team=team,
                           formations=formations,
                           selected_formation=selected_formation)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
