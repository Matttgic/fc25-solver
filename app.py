from flask import Flask, render_template, request
import pandas as pd
from solver import solve_team

app = Flask(__name__)

DATA_PATH = "player-data-full-2025-june.csv"
df = pd.read_csv(DATA_PATH)
all_columns = [col for col in df.columns if col not in ["image", "description"]]

FORMATION_OPTIONS = {
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        budget_million = float(request.form.get("budget", 500))
        budget = budget_million * 1_000_000
        formation_name = request.form.get("formation")
        selected_criteria = request.form.getlist("criteria")

        filters = {}
        for col in all_columns:
            val = request.form.get(col)
            if val:
                filters[col] = val

        formation = FORMATION_OPTIONS[formation_name]
        result = solve_team(df, formation, budget, selected_criteria, filters)

    return render_template("index.html",
                           criteria=all_columns,
                           formations=FORMATION_OPTIONS.keys(),
                           result=result)

if __name__ == "__main__":
    app.run(debug=True)
