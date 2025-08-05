import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

st.set_page_config(page_title="FC25 Solver", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("player-data-full-2025-june.csv")

    # Nettoyage de la colonne "value"
    df["value"] = df["value"].astype(str)
    df["value"] = df["value"].str.replace("€", "", regex=False)
    df["value"] = df["value"].str.replace("M", "", regex=False)
    df["value"] = df["value"].str.replace("K", "", regex=False)
    df["value"] = df["value"].str.replace(",", "", regex=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()]
    df = df[df["value"] > 0]

    # Nettoyage des positions
    df = df[df["positions"].notna()]
    return df

df = load_data()

# Liste des colonnes numériques filtrables
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
excluded = ['player_id', 'version', 'value', 'wage']
criteria_columns = [col for col in numeric_columns if col not in excluded]

st.title("FC25 • Team Builder")

# Barre latérale
st.sidebar.header("⚙️ Paramètres")
selected_critere = st.sidebar.selectbox("🧠 Critère d'optimisation", criteria_columns)
budget = st.sidebar.slider("💸 Budget maximal (en millions)", min_value=10, max_value=1000, step=10, value=500)
build_button = st.sidebar.button("⚽ Construire l'équipe")

# Formation par défaut (1-4-4-2)
formation = {"GK": 1, "DF": 4, "MF": 4, "FW": 2}

def get_position_group(pos_string):
    if "GK" in pos_string:
        return "GK"
    elif any(pos in pos_string for pos in ["CB", "LB", "RB", "LWB", "RWB"]):
        return "DF"
    elif any(pos in pos_string for pos in ["CM", "CDM", "CAM", "LM", "RM"]):
        return "MF"
    elif any(pos in pos_string for pos in ["ST", "CF", "LW", "RW"]):
        return "FW"
    return None

def construire_equipe(df, critere, budget):
    df["position_group"] = df["positions"].apply(get_position_group)
    df = df[df["position_group"].isin(formation.keys())]
    df = df.dropna(subset=[critere])

    model = LpProblem("Team_Selection", LpMaximize)

    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in df.index}
    model += lpSum(df.loc[i, critere] * x[i] for i in df.index)

    model += lpSum(x[i] for i in df.index) == sum(formation.values())
    model += lpSum(df.loc[i, "value"] * x[i] for i in df.index) <= budget

    for pos, count in formation.items():
        model += lpSum(x[i] for i in df[df["position_group"] == pos].index) == count

    model.solve()

    selected_players = df[[x[i].value() == 1 for i in df.index]]
    return selected_players

if build_button:
    try:
        equipe = construire_equipe(df.copy(), selected_critere, budget)
        if not equipe.empty:
            st.success("✅ Équipe générée avec succès !")
            st.dataframe(equipe[["full_name", "positions", selected_critere, "value"]].sort_values(by=selected_critere, ascending=False))
        else:
            st.warning("⚠️ Aucune équipe trouvée avec ces contraintes.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération : {e}")
else:
    st.info("Sélectionne un critère et un budget, puis clique sur « Construire l’équipe ».")
