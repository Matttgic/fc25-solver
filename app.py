import pandas as pd
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]
    return df

df = load_data()

st.title("⚽ FC25 – Générateur d'équipe optimale")

# Critères disponibles : toutes les colonnes numériques
numeric_columns = df.select_dtypes(include='number').columns.tolist()
critere = st.selectbox("🎯 Critère à maximiser", numeric_columns)

# Budget (en millions d’euros)
budget = st.slider("💰 Budget max (en M€)", int(df["value"].min()), 1000, 500)

# Formations possibles
formations = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "4-2-3-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1}
}
formation = st.selectbox("📋 Formation", list(formations.keys()))
structure = formations[formation]

if st.button("🚀 Construire l'équipe"):
    model = LpProblem("Optimisation_Equipe", LpMaximize)
    players = df.index
    x = LpVariable.dicts("joueur", players, cat=LpBinary)

    # Contraintes
    model += lpSum(x[i] * df.loc[i, "value"] for i in players) <= budget

    for poste, nombre in structure.items():
        model += lpSum(x[i] for i in players if poste in df.loc[i, "positions"]) == nombre

    model += lpSum(x[i] * df.loc[i, critere] for i in players)

    model.solve()

    selected = [i for i in players if x[i].varValue == 1]

    if selected:
        st.success("✅ Équipe générée avec succès !")
        st.dataframe(df.loc[selected][["name", "positions", "value", critere]])
        total_value = df.loc[selected]["value"].sum()
        total_score = df.loc[selected][critere].sum()
        st.markdown(f"**💶 Coût total :** `{total_value:.1f} M€`")
        st.markdown(f"**📈 Score total ({critere}) :** `{total_score:.2f}`")
    else:
        st.error("❌ Aucune équipe possible avec ce budget et cette formation.")
