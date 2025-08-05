import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")
    df = df[df["value"].apply(lambda x: str(x).replace(",", "").replace("€", "").replace("M", "").replace("K", "").replace(".", "").isdigit())]
    df["value"] = df["value"].replace("€", "", regex=True).replace(",", "", regex=True)
    df["value"] = df["value"].replace("M", "", regex=True).astype(float)
    df = df[df["value"] > 0]
    df = df[df["positions"].notna()]
    return df

df = load_data()

# --- UI ---
st.title("FC25 - Générateur d'équipe intelligente")

# Critère à maximiser
all_numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_critere = st.selectbox("Choisir le critère à maximiser", options=all_numeric_cols)

# Budget
budget = st.slider("Budget maximum (en millions €)", 50, 1000, 500, step=10)

# Bouton
generate = st.button("🎯 Construire l'équipe optimale")

# Formations dispo
formations = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
}
formation_choice = st.selectbox("Formation", list(formations.keys()))
structure = formations[formation_choice]

# Lancer la génération
if generate:
    try:
        df = df[df[selected_critere].notna()]
        df = df[df["positions"].notna()]
        df["value"] = df["value"].astype(float)
        df[selected_critere] = df[selected_critere].astype(float)

        players = df.to_dict(orient="records")
        prob = LpProblem("Team_Selection", LpMaximize)
        x = LpVariable.dicts("Player", range(len(players)), 0, 1, LpBinary)

        # Contraintes de budget
        prob += lpSum([x[i] * players[i]["value"] for i in range(len(players))]) <= budget

        # Contraintes de nombre de joueurs par poste
        for pos, required in structure.items():
            prob += lpSum([x[i] for i in range(len(players)) if pos in players[i]["positions"]]) >= required

        # Contraintes : 11 joueurs au total
        prob += lpSum([x[i] for i in range(len(players))]) == 11

        # Objectif : maximiser la somme des critères
        prob += lpSum([x[i] * players[i][selected_critere] for i in range(len(players))])

        # Résolution
        prob.solve()

        selected_players = [players[i] for i in range(len(players)) if x[i].varValue == 1]

        if selected_players:
            result_df = pd.DataFrame(selected_players)
            st.success("✅ Équipe générée avec succès !")
            st.dataframe(result_df[["name", "positions", "value", selected_critere]])
        else:
            st.error("Aucune équipe valide trouvée avec ces contraintes. Essaie d'augmenter le budget ou de choisir un autre critère.")
    except Exception as e:
        st.exception(f"Erreur pendant la génération : {e}") 
