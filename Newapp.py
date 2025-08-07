import streamlit as st
import pandas as pd
import numpy as np
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
ALL_POSITIONS = sorted(list(set(pos for formation in FORMATIONS.values() for pos in formation.keys())))

# --- Fonctions ---
@st.cache_data
def load_data(uploaded_file):
    """Charge et nettoie les données."""
    try:
        df = pd.read_csv(uploaded_file)
        if 'value' in df.columns:
            value_str = df['value'].astype(str).str.replace('[€,]', '', regex=True).str.strip()
            is_million = value_str.str.endswith('M', na=False)
            is_thousand = value_str.str.endswith('K', na=False)
            numeric_part = pd.to_numeric(value_str.str.replace('[MK]', '', regex=True), errors='coerce')
            df['value_numeric'] = numeric_part
            df.loc[is_million, 'value_numeric'] *= 1
            df.loc[is_thousand, 'value_numeric'] /= 1000
        else:
            df['value_numeric'] = 0
        df['value_numeric'] = df['value_numeric'].fillna(0)
        
        df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
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
    """Solveur avec une approche pure pour une optimisation maximale."""
    # 1. Filtrer les joueurs uniquement sur la base des critères de l'utilisateur
    candidate_df = df.copy()
    if not filters.get('include_free_agents', True):
        candidate_df = candidate_df[candidate_df['value_numeric'] > 0]
    
    # Appliquer les filtres de l'utilisateur
    if 'age_range' in filters:
        candidate_df = candidate_df[candidate_df['age'].between(*filters['age_range'])]
    if 'potential_range' in filters:
        candidate_df = candidate_df[candidate_df['potential'].between(*filters['potential_range'])]
    if 'min_overall' in filters:
        candidate_df = candidate_df[candidate_df['overall_rating'] >= filters['min_overall']]
    
    if candidate_df.empty: 
        return None

    # 2. Lancer le solveur sur TOUS les joueurs éligibles
    prob = LpProblem("TeamBuilder", LpMaximize)
    player_vars = {}
    positions_to_fill = FORMATIONS[formation]

    for position, count in positions_to_fill.items():
        eligible_for_pos = candidate_df[candidate_df['positions'].apply(lambda x: can_play_position(x, position))]
        for p_idx in eligible_for_pos.index:
            player_vars[(p_idx, position)] = LpVariable(f"player_{p_idx}_pos_{position}", cat=LpBinary)

    # Définir l'objectif
    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, criteria] for (p_idx, pos) in player_vars), "Total_Score"
    
    # Définir les contraintes
    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, 'value_numeric'] for (p_idx, pos) in player_vars) <= budget, "Budget"
    for position, count in positions_to_fill.items():
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx, pos) in player_vars if pos == position) == count, f"Formation_{position}"
    for p_idx in candidate_df.index:
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx_c, pos) in player_vars if p_idx_c == p_idx) <= 1, f"Uniqueness_{p_idx}"

    # Résoudre
    prob.solve()
    
    if prob.status == 1: # 1 = Optimal
        team = [{'player': df.loc[p_idx], 'position': pos} for (p_idx, pos), var in player_vars.items() if var.varValue == 1]
        return team
    return None

def search_players(df, num_players, position, budget_per_player, filters):
    """Filtre et retourne les meilleurs joueurs selon des critères."""
    # ... (le code de cette fonction reste inchangé)
    return ...

# --- Application Principale ---
def main():
    uploaded_file = st.file_uploader("📁 **Chargez votre base de données joueurs FC25 (CSV)**", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return
        st.success(f"✅ **{len(df):,} joueurs chargés avec succès !**")

        tab1, tab2 = st.tabs(["🏗️ **Constructeur d'Équipe**", "🔍 **Recherche de Joueurs**"])

        with tab1:
            st.header("Constructeur d'Équipe Optimisé")
            st.info("Le solveur recherche la meilleure équipe mathématiquement possible. Soyez patient, la perfection prend du temps !")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Configuration")
                formation = st.selectbox("📋 Formation", list(FORMATIONS.keys()), key="t1_formation")
                budget = st.number_input("💰 Budget total (M€)", min_value=0.1, value=50.0, step=5.0, key="t1_budget")
                criteria = st.selectbox("🎯 Maximiser", ["overall_rating", "potential", "score"],
                                        format_func=lambda x: {"overall_rating": "Moyenne Générale", "potential": "Potentiel Moyen", "score": "Score Moyen (Overall+Potentiel)"}[x], key="t1_criteria")
                st.subheader("Filtres")
                age_range = st.slider("🎂 Âge", 16, 45, (16, 40), key="t1_age")
                potential_range = st.slider("💎 Potentiel", 40, 99, (40, 99), key="t1_potential")
                min_overall = st.slider("⭐ Overall minimum", 40, 99, 40, key="t1_overall")
                include_free_agents = st.checkbox("🆓 Inclure agents libres (€0)", value=True, key="t1_free_agents")
                
                filters = {'age_range': age_range, 'potential_range': potential_range, 'min_overall': min_overall, 'include_free_agents': include_free_agents}

                if st.button("🚀 TROUVER L'ÉQUIPE OPTIMALE", type="primary", use_container_width=True):
                    with st.spinner("🧠 Analyse de toutes les combinaisons possibles... Cette opération peut prendre plusieurs minutes."):
                        team = solve_team(df, formation, budget, criteria, filters)
                        st.session_state.team_results = team
            with col2:
                if 'team_results' in st.session_state:
                    team = st.session_state.team_results
                    if team is None:
                        st.error("❌ **Aucune solution trouvée.** Il est mathématiquement impossible de former une équipe avec ces filtres et ce budget. Essayez d'augmenter le budget ou d'élargir les filtres.")
                    else:
                        st.success(f"✅ **Équipe optimale trouvée ! ({len(team)} joueurs)**")
                        team_data, total_cost, total_score = [], 0, 0
                        for p_data in team:
                            player, cost = p_data['player'], p_data['player']['value_numeric']
                            total_cost += cost
                            total_score += player[criteria]
                            team_data.append({"Position": p_data['position'], "Nom": player['name'], "OVR": player['overall_rating'], "POT": player['potential'], "Âge": int(player['age']), "Coût (M€)": f"{cost:.2f}"})
                        
                        team_df = pd.DataFrame(team_data).sort_values(by="Position", key=lambda x: x.map({pos: i for i, pos in enumerate(FORMATIONS[formation].keys())}))
                        st.dataframe(team_df, use_container_width=True, hide_index=True)
                        m1, m2 = st.columns(2)
                        m1.metric("💰 Coût Total", f"€{total_cost:.2f}M", f"Budget: €{budget:.2f}M")
                        m2.metric(f"⭐ Moyenne '{criteria.replace('_', ' ').replace('rating', 'Générale').title()}'", f"{(total_score / len(team)):.1f}")
                        st.download_button("📥 Télécharger en CSV", team_df.to_csv(index=False).encode('utf-8'), f'equipe.csv', 'text/csv')

        # Le code pour l'onglet de recherche reste inchangé
        with tab2:
            # ... (code de l'onglet 2)
            pass

if __name__ == "__main__":
    main() 
