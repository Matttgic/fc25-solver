import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

# === Chargement des données ===
@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")
    df = df.dropna(subset=["positions", "name", "value"])
    df = df[df["value"] > 0]  # On garde les joueurs ayant une valeur définie
    return df

df = load_data()

# === Interface utilisateur ===
st.title("FC25 - Générateur d’équipe")

# Budget en millions
budget_millions = st.slider("Budget total (en millions d'euros)", 50, 1000, 500)
budget = budget_millions * 1_000_000  # conversion

# Choix de la formation
formations = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
}
formation_choice = st.selectbox("Formation", list(formations.keys()))

# Colonnes numériques disponibles pour l’optimisation
numeric_cols = df.select_dtypes(include='number').columns.tolist()
excluded_cols = ['player_id', 'club_id', 'country_id', 'value', 'wage']
criteres_disponibles = [c for c in numeric_cols if c not in excluded_cols]

selected_critere = st.selectbox("Critère à maximiser", sorted(criteres_disponibles))

# Bouton pour construire l’équipe
if st.button("Construire l’équipe"):
    st.subheader("Résultat du solver")

    selected_df = df[["name", "positions", "value", selected_critere]].copy()
    selected_df = selected_df.dropna(subset=[selected_critere])
    
    # Poste principal uniquement
    selected_df["main_pos"] = selected_df["positions"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")
    pos_map = {
        "GK": "GK",
        "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF", "RCB": "DF", "LCB": "DF",
        "CM": "MF", "CAM": "MF", "CDM": "MF", "LM": "MF", "RM": "MF", "LCM": "MF", "RCM": "MF",
        "ST": "FW", "CF": "FW", "LF": "FW", "RF": "FW", "RW": "FW", "LW": "FW"
    }
    selected_df["pos_group"] = selected_df["main_pos"].map(pos_map)
    selected_df = selected_df[selected_df["pos_group"].isin(["GK", "DF", "MF", "FW"])]
    
    # Optimisation avec PuLP
    model = LpProblem("Team_Selection", LpMaximize)
    players = selected_df.index
    choices = LpVariable.dicts("Player", players, cat=LpBinary)

    # Objectif : maximiser la somme pondérée du critère choisi
    model += lpSum([choices[i] * selected_df.loc[i, selected_critere] for i in players])

    # Contraintes de poste
    formation = formations[formation_choice]
    for pos in formation:
        model += lpSum([choices[i] for i in players if selected_df.loc[i, "pos_group"] == pos]) == formation[pos]

    # Contrainte de budget
    model += lpSum([choices[i] * selected_df.loc[i, "value"] for i in players]) <= budget

    # Résolution
    model.solve()

    selected_players = selected_df[[choices[i].varValue == 1 for i in players]]

    if not selected_players.empty:
        st.success(f"Équipe générée avec succès ! Critère maximisé : `{selected_critere}`")
        st.write(selected_players[["name", "main_pos", "value", selected_critere]].reset_index(drop=True))
        total_value = selected_players["value"].sum()
        total_score = selected_players[selected_critere].sum()
        st.markdown(f"**Total du budget utilisé :** {total_value/1_000_000:.1f} M€")
        st.markdown(f"**Total du critère `{selected_critere}` :** {total_score:.2f}")
    else:
        st.error("Aucune équipe valide n’a pu être générée avec les contraintes actuelles.")
