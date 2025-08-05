import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

# === Chargement des données ===
@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")
    df = df.dropna(subset=["positions", "name", "value", "overall_rating"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df

df = load_data()

st.title("FC25 - Générateur d’équipe optimale ⚽")

# === Sélection des critères ===
st.sidebar.header("Filtres et options")

critere = st.sidebar.selectbox("Critère à maximiser", df.select_dtypes(include='number').columns)
budget = st.sidebar.slider("Budget maximum (€)", 10, 1000, 300) * 1_000_000

# === Sélection de la formation ===
formations = {
    "4-4-2": {"GK":1, "DF":4, "MF":4, "FW":2},
    "4-3-3": {"GK":1, "DF":4, "MF":3, "FW":3},
    "3-4-3": {"GK":1, "DF":3, "MF":4, "FW":3},
    "5-4-1": {"GK":1, "DF":5, "MF":4, "FW":1},
    "4-2-3-1": {"GK":1, "DF":4, "MF":5, "FW":1},
}
formation_name = st.sidebar.selectbox("Formation", list(formations.keys()))
formation = formations[formation_name]

# === Marquage du poste principal
def poste_principal(positions):
    return positions.split(",")[0].strip().upper()

df["poste"] = df["positions"].apply(poste_principal)

# === Optimisation
model = LpProblem("Team_Selection", LpMaximize)
players = list(df.index)
x = LpVariable.dicts("player", players, cat=LpBinary)

# Objectif : maximiser le critère choisi
model += lpSum(x[i] * df.loc[i, critere] for i in players)

# Contraintes :
model += lpSum(x[i] * df.loc[i, "value"] for i in players) <= budget  # budget

# Contraintes par poste
for poste, nombre in formation.items():
    model += lpSum(x[i] for i in players if poste in df.loc[i, "poste"]) == nombre

# Total joueurs = 11
model += lpSum(x[i] for i in players) == 11

# Résolution
model.solve()

# === Affichage des résultats
selected_players = df[[x[i].varValue == 1 for i in players]]
if selected_players.empty:
    st.warning("Aucune équipe trouvée avec les contraintes actuelles.")
else:
    st.success("Équipe optimale générée avec succès !")
    st.write(f"**Total {critere} :** {selected_players[critere].sum():.2f}")
    st.write(f"**Valeur totale :** {selected_players['value'].sum():,.0f} €")
    st.dataframe(selected_players[["name", "poste", critere, "value", "club_name"]])
