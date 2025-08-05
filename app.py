import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PulpError

@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")

    # Nettoyage des colonnes monétaires
    for col in ["value", "wage", "release_clause"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("M", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("K", "", regex=False)
                .str.replace("nan", "")
                .replace("", 0)
                .astype(float)
                .fillna(0)
            )
            # Conversion K et M en millions
            if col == "wage":
                df[col] = df[col] / 1000  # salaire annuel estimé en millions

    # Hauteur : on convertit 1.80 → 180
    df["height_cm"] = df["height_cm"].apply(lambda x: float(x) * 100 if float(x) < 3 else float(x))

    # On garde les lignes avec valeurs valides
    df = df[df["value"] > 0]
    df = df[df["overall_rating"] > 0]

    return df

df = load_data()

st.set_page_config(page_title="FC25 - Générateur d’équipe", layout="wide")
st.title("⚽ Générateur d’équipe FC25")

# Formations possibles
formations_possibles = {
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "5-3-2": {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "5-4-1": {"GK": 1, "DF": 5, "MF": 4, "FW": 1},
    "4-5-1": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
}

# Critères numériques sélectionnables
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
critere = st.selectbox("📊 Critère d’optimisation", options=numeric_cols, index=numeric_cols.index("overall_rating") if "overall_rating" in numeric_cols else 0)

# Budget (en millions d’euros)
budget = st.slider("💰 Budget max (en M€)", min_value=1, max_value=1000, value=500)

# Choix formation
formation_nom = st.selectbox("📐 Formation", list(formations_possibles.keys()))
formation = formations_possibles[formation_nom]

# Bouton pour générer l’équipe
if st.button("🚀 Construire l'équipe"):
    df_filtered = df.dropna(subset=[critere, "value", "positions"])
    joueurs = df_filtered.index.tolist()

    # Variables de décision
    x = LpVariable.dicts("player", joueurs, 0, 1, LpBinary)

    model = LpProblem("Team_Selection", LpMaximize)
    try:
        model += lpSum(x[i] * df_filtered.loc[i, critere] for i in joueurs)

        # Contraintes : budget
        model += lpSum(x[i] * df_filtered.loc[i, "value"] for i in joueurs) <= budget

        # Contraintes de postes
        for poste, nombre in formation.items():
            model += lpSum(
                x[i]
                for i in joueurs
                if poste in str(df_filtered.loc[i, "positions"])
            ) == nombre

        model.solve()

        equipe = df_filtered[[critere, "name", "value", "positions"]].copy()
        equipe["selectionné"] = [x[i].varValue for i in joueurs]
        equipe = equipe[equipe["selectionné"] == 1.0]
        equipe = equipe.sort_values(by=critere, ascending=False)

        st.success("✅ Équipe construite avec succès !")
        st.write(equipe.drop(columns=["selectionné"]))

        total_score = equipe[critere].sum()
        total_value = equipe["value"].sum()
        st.metric("🎯 Score total", f"{total_score:.2f}")
        st.metric("💸 Coût total", f"{total_value:.2f} M€")

    except PulpError as e:
        st.error("❌ Erreur de résolution du modèle.")
        st.exception(e)
