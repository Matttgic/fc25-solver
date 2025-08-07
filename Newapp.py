import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

# --- Configuration de la page ---
st.set_page_config(page_title="FC25 Solver Pro", page_icon="⚽", layout="wide")

# --- CSS ---
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem; font-weight: bold;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 1rem;
}
.stButton>button { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ FC25 TEAM SOLVER PRO</h1>', unsafe_allow_html=True)

# --- Constantes ---
FORMATIONS = {
    "4-3-3": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "CAM": 1, "LW": 1, "RW": 1, "ST": 1},
    "4-4-2": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "LM": 1, "CM": 2, "RM": 1, "ST": 2},
    "3-5-2": {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CDM": 2, "CAM": 1, "ST": 2},
    "4-2-3-1": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 2, "CAM": 1, "LW": 1, "RW": 1, "ST": 1},
}

# --- Fonctions ---
@st.cache_data
def load_data(uploaded_file):
    """Charge et nettoie les données."""
    try:
        df = pd.read_csv(uploaded_file)
        if 'value' in df.columns:
            df['value_numeric'] = df['value'].astype(str).str.replace('[€,]', '', regex=True)
            df['value_numeric'] = pd.to_numeric(df['value_numeric'].str.replace('[MK]', '', regex=True), errors='coerce')
            df.loc[df['value_clean'].str.contains('K', na=False), 'value_numeric'] /= 1000
            df['value_numeric'] = df['value_numeric'].fillna(0)
        df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
        # Score combinant overall et potentiel
        df['score'] = df['overall_rating'] * 0.6 + df['potential'] * 0.4
        df['player_id'] = df.index
        return df.dropna(subset=['name', 'overall_rating', 'age', 'positions', 'value_numeric'])
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return None

def can_play_position(player_positions, required_position):
    """Vérifie si le poste est dans les 3 premiers postes du joueur."""
    if pd.isna(player_positions): return False
    return required_position in [p.strip() for p in str(player_positions).split(',')[:3]]

def solve_team(df, formation, budget, criteria, filters):
    """
    CORRIGÉ (Pb 1): Utilise un solveur d'optimisation linéaire pour trouver la meilleure équipe.
    """
    # 1. Filtrer les joueurs éligibles en amont
    candidate_df = df.copy()
    if 'age_range' in filters:
        min_age, max_age = filters['age_range']
        candidate_df = candidate_df[candidate_df['age'].between(min_age, max_age)]
    if 'potential_range' in filters:
        min_pot, max_pot = filters['potential_range']
        candidate_df = candidate_df[candidate_df['potential'].between(min_pot, max_pot)]
    if 'min_overall' in filters:
        candidate_df = candidate_df[candidate_df['overall_rating'] >= filters['min_overall']]

    # 2. Définir le problème d'optimisation
    prob = LpProblem("TeamBuilder", LpMaximize)
    
    # 3. Créer les variables : une pour chaque joueur possible pour chaque poste
    player_vars = {}
    positions_to_fill = FORMATIONS[formation]

    for position, count in positions_to_fill.items():
        # Pour chaque poste, trouver les joueurs qui peuvent y jouer
        eligible_players = candidate_df[candidate_df['positions'].apply(lambda x: can_play_position(x, position))]
        for p_idx in eligible_players.index:
            # Créer une variable binaire: 1 si le joueur est choisi pour ce poste, 0 sinon
            player_vars[(p_idx, position)] = LpVariable(f"player_{p_idx}_pos_{position}", cat=LpBinary)

    # 4. Définir l'objectif : Maximiser le critère choisi
    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, criteria] 
                   for (p_idx, pos) in player_vars), "Total_Score"

    # 5. Ajouter les contraintes
    # Contrainte de budget
    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, 'value_numeric'] 
                   for (p_idx, pos) in player_vars) <= budget, "Budget"

    # Contrainte de formation : remplir chaque poste avec le bon nombre de joueurs
    for position, count in positions_to_fill.items():
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx, pos) in player_vars if pos == position) == count, f"Formation_{position}"

    # Contrainte d'unicité : un joueur ne peut être sélectionné qu'une seule fois
    for p_idx in candidate_df.index:
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx_c, pos) in player_vars if p_idx_c == p_idx) <= 1, f"Uniqueness_{p_idx}"

    # 6. Résoudre le problème
    prob.solve()

    # 7. Extraire les résultats
    if LpProblem.status[prob.status] == 'Optimal':
        team = []
        for (p_idx, pos), var in player_vars.items():
            if var.varValue == 1:
                player_data = df.loc[p_idx]
                team.append({'player': player_data, 'position': pos})
        return team
    else:
        return None # Aucune solution trouvée

# --- Application Principale ---
def main():
    uploaded_file = st.file_uploader("📁 **Chargez votre base de données joueurs FC25 (CSV)**", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return
        st.success(f"✅ **{len(df):,} joueurs chargés avec succès !**")

        st.header("🏗️ **Constructeur d'Équipe Optimisé**")
        st.info("Le solveur trouvera la meilleure équipe possible en respectant vos contraintes de budget, de formation et de filtres.")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Configuration")
            formation = st.selectbox("📋 **Formation**", list(FORMATIONS.keys()))
            budget = st.number_input("💰 **Budget total (M€)**", min_value=1.0, value=100.0, step=10.0)
            
            st.subheader("Objectif d'Optimisation")
            criteria = st.selectbox("🎯 **Maximiser**", ["score", "overall_rating", "potential"],
                                    format_func=lambda x: {"score": "Score (Overall + Potentiel)", "overall_rating": "Overall Actuel", "potential": "Potentiel Futur"}[x])

            st.subheader("Filtres")
            age_range = st.slider("🎂 Âge", 16, 45, (18, 34))
            potential_range = st.slider("💎 **Potentiel**", 50, 100, (75, 99))
            min_overall = st.slider("⭐ Overall minimum", 40, 99, 70)
            
            filters = {
                'age_range': age_range, 'potential_range': potential_range, 'min_overall': min_overall
            }

            if st.button("🚀 **TROUVER LA MEILLEURE ÉQUIPE**", type="primary", use_container_width=True):
                with st.spinner("🧠 Le solveur analyse des milliers de combinaisons..."):
                    team = solve_team(df, formation, budget, criteria, filters)
                    st.session_state.team_results = team

        with col2:
            if 'team_results' in st.session_state:
                team = st.session_state.team_results
                if team is None:
                    st.error("❌ **Aucune solution trouvée.** Il est mathématiquement impossible de former une équipe avec ces filtres et ce budget. Essayez d'augmenter le budget ou d'élargir les filtres.")
                else:
                    st.success(f"✅ **Équipe optimale trouvée ! ({len(team)} joueurs)**")
                    
                    # CORRIGÉ (Pb 1): Affichage en liste/tableau simple
                    team_data = []
                    total_cost = 0
                    for p_data in team:
                        player = p_data['player']
                        cost = player['value_numeric']
                        total_cost += cost
                        team_data.append({
                            "Position": p_data['position'],
                            "Nom": player['name'],
                            "OVR": player['overall_rating'],
                            "POT": player['potential'],
                            "Âge": int(player['age']),
                            "Coût (M€)": f"{cost:.2f}"
                        })
                    
                    team_df = pd.DataFrame(team_data)
                    st.dataframe(team_df, use_container_width=True, hide_index=True)

                    # Métriques
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("💰 **Coût Total de l'Équipe**", f"€{total_cost:.2f}M", f"Budget: €{budget:.2f}M")
                    m_col2.metric(f"⭐ **Moyenne '{criteria.replace('_', ' ').title()}'**", f"{team_df[criteria.replace('score', 'OVR').replace('_rating','').upper()].mean():.1f}")
                    
                    # Export
                    csv = team_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger l'équipe en CSV",
                        data=csv,
                        file_name=f'equipe_{formation}_{budget}M.csv',
                        mime='text/csv',
                    )

if __name__ == "__main__":
    main()
