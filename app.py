import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

# ------------------ Chargement des données ------------------

@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")

    # Nettoyage valeur marchande
    df["value"] = df["value"].astype(str).str.replace("€", "").str.replace("M", "").str.replace("K", "")
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    df = df.dropna(subset=["value"])
    df["value"] = df["value"].astype(float)

    # Conversion position principale
    df["main_position"] = df["positions"].astype(str).str.split(",").str[0]

    return df

df = load_data()

# ------------------ Configuration de l'app ------------------

st.set_page_config(page_title="FC25 Solver", layout="wide")
st.title("⚽ Générateur d’équipe FC25")

# Critères numériques disponibles
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
critere = st.selectbox("📊 Critère d’optimisation", numeric_cols)

# Budget max
min_budget = int(df["value"].min())
budget = st.slider("💰 Budget max (en M€)", min_value=min_budget, max_value=1000, value=500)

# Formations possibles
formations = {
    "4-4-2": {"GK":1, "DF":4, "MF":4, "FW":2},
    "3-4-3": {"GK":1, "DF":3, "MF":4, "FW":3},
    "4-3-3": {"GK":1, "DF":4, "MF":3, "FW":3},
    "5-3-2": {"GK":1, "DF":5, "MF":3, "FW":2},
    "5-4-1": {"GK":1, "DF":5, "MF":4, "FW":1},
    "3-5-2": {"GK":1, "DF":3, "MF":5, "FW":2}
}
formation_choice = st.selectbox("📐 Formation", list(formations.keys()))
structure = formations[formation_choice]

# ------------------ Optimisation ------------------

if st.button("✅ Construire l’équipe"):
    model = LpProblem("Optimisation_equipe", LpMaximize)

    joueurs = list(df.index)
    x = LpVariable.dicts("joueur", joueurs, cat=LpBinary)

    # Fonction à maximiser
    model += lpSum(x[i] * df.loc[i, critere] for i in joueurs)

    # Contraintes de poste (selon structure)
    for poste, count in structure.items():
        model += lpSum(x[i] for i in joueurs if poste in str(df.loc[i, "main_position"])) == count

    # Contrainte de budget
    model += lpSum(x[i] * df.loc[i, "value"] for i in joueurs) <= budget

    # Résolution
    model.solve()

    # Affichage équipe
    st.subheader("📋 Équipe sélectionnée")
    selected = df[[critere, "name", "main_position", "value", "club_name"]].copy()
    selected["selection"] = [x[i].varValue if i in x else 0 for i in selected.index]
    selected = selected[selected["selection"] == 1].sort_values("main_position")

    if not selected.empty:
        st.dataframe(selected.drop(columns=["selection"]), use_container_width=True)
    else:
        st.warning("❌ Aucune équipe ne peut être générée avec ces critères.")
