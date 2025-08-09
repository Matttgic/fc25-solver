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

def solve_team(df, formation, budget, criteria, filters, excluded_player_ids=None):
    """Solveur avec une approche pure pour une optimisation maximale."""
    # 1. Filtrer les joueurs uniquement sur la base des critères de l'utilisateur
    candidate_df = df.copy()
    if excluded_player_ids:
        candidate_df = candidate_df[~candidate_df['player_id'].isin(excluded_player_ids)]

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

def display_team_results(team, formation, budget, criteria, title):
    """Affiche les résultats d'une équipe."""
    st.subheader(title)
    if team is None:
        st.warning("❌ **Aucune solution trouvée pour cette équipe.**")
        return

    st.success(f"✅ **Équipe trouvée ! ({len(team)} joueurs)**")
    team_data, total_cost, total_score = [], 0, 0
    for p_data in team:
        player, cost = p_data['player'], p_data['player']['value_numeric']
        total_cost += cost
        total_score += player[criteria]
        team_data.append({
            "Position": p_data['position'], "Nom": player['name'], "OVR": player['overall_rating'],
            "POT": player['potential'], "Âge": int(player['age']), "Coût (M€)": f"{cost:.2f}"
        })

    team_df = pd.DataFrame(team_data).sort_values(
        by="Position", key=lambda x: x.map({pos: i for i, pos in enumerate(FORMATIONS[formation].keys())})
    )
    st.dataframe(team_df, use_container_width=True, hide_index=True)
    m1, m2 = st.columns(2)
    m1.metric("💰 Coût Total", f"€{total_cost:.2f}M", f"Budget: €{budget:.2f}M")
    m2.metric(f"⭐ Moyenne '{criteria.replace('_', ' ').replace('rating', 'Générale').title()}'", f"{(total_score / len(team)):.1f}")
    st.download_button(f"📥 Télécharger {title}", team_df.to_csv(index=False).encode('utf-8'), f'{title.replace(" ", "_").lower()}.csv', 'text/csv', key=title)

def search_players(df, positions_to_find, budget, criteria, filters):
    """Filtre et retourne les meilleurs joueurs pour les postes et quantités demandés."""
    candidate_df = df.copy()
    
    # Appliquer les filtres de base (âge, etc.)
    if 'age_range' in filters:
        candidate_df = candidate_df[candidate_df['age'].between(*filters['age_range'])]
    if 'potential_range' in filters:
        candidate_df = candidate_df[candidate_df['potential'].between(*filters['potential_range'])]
    if 'min_overall' in filters:
        candidate_df = candidate_df[candidate_df['overall_rating'] >= filters['min_overall']]
    
    # Filtrer par budget
    candidate_df = candidate_df[candidate_df['value_numeric'] <= budget]

    found_players = pd.DataFrame()

    for position, count in positions_to_find.items():
        if count == 0:
            continue
        
        # Trouver les joueurs éligibles pour le poste
        eligible_for_pos = candidate_df[candidate_df['positions'].apply(lambda x: can_play_position(x, position))]

        # Trier par le critère de l'utilisateur et prendre les meilleurs
        best_for_pos = eligible_for_pos.sort_values(by=criteria, ascending=False).head(count)
        best_for_pos['requested_position'] = position
        found_players = pd.concat([found_players, best_for_pos])

    return found_players.drop_duplicates(subset=['player_id'])

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
            st.info("Le solveur recherche la meilleure équipe mathématiquement possible, puis une deuxième équipe avec les joueurs restants.")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Configuration")
                formation = st.selectbox("📋 Formation", list(FORMATIONS.keys()), key="t1_formation")
                budget = st.number_input("💰 Budget total (M€)", min_value=0.1, value=50.0, step=5.0, key="t1_budget")
                criteria = st.selectbox("🎯 Maximiser", ["overall_rating", "potential", "score"],
                                        format_func=lambda x: {"overall_rating": "Moyenne Générale", "potential": "Potentiel Moyen", "score": "Score Moyen (Overall+Potentiel)"}[x], key="t1_criteria")
                
                with st.expander("Filtres avancés"):
                    age_range = st.slider("🎂 Âge", 16, 45, (16, 40), key="t1_age")
                    potential_range = st.slider("💎 Potentiel", 40, 99, (40, 99), key="t1_potential")
                    min_overall = st.slider("⭐ Overall minimum", 40, 99, 40, key="t1_overall")
                    include_free_agents = st.checkbox("🆓 Inclure agents libres (€0)", value=True, key="t1_free_agents")
                
                filters = {'age_range': age_range, 'potential_range': potential_range, 'min_overall': min_overall, 'include_free_agents': include_free_agents}

                if st.button("🚀 TROUVER LES ÉQUIPES OPTIMALES", type="primary", use_container_width=True):
                    with st.spinner("🧠 Recherche de la meilleure équipe..."):
                        team1 = solve_team(df, formation, budget, criteria, filters)
                        st.session_state.team1_results = team1
                        st.session_state.team2_results = None # Reset

                    if team1:
                        with st.spinner("🧠 Recherche de la deuxième meilleure équipe..."):
                            team1_ids = [p['player']['player_id'] for p in team1]
                            team1_cost = sum(p['player']['value_numeric'] for p in team1)
                            remaining_budget = budget - team1_cost

                            team2 = solve_team(df, formation, remaining_budget, criteria, filters, excluded_player_ids=team1_ids)
                            st.session_state.team2_results = team2

            with col2:
                if 'team1_results' in st.session_state:
                    team1 = st.session_state.team1_results
                    if team1 is None:
                        st.error("❌ **Aucune solution trouvée.** Il est mathématiquement impossible de former une équipe avec ces filtres et ce budget. Essayez d'augmenter le budget ou d'élargir les filtres.")
                    else:
                        display_team_results(team1, formation, budget, criteria, "🏆 Meilleure Équipe Possible")
                        
                        st.markdown("---")

                        team2 = st.session_state.get('team2_results')
                        display_team_results(team2, formation, budget, criteria, "🥈 Deuxième Meilleure Équipe (Nouveaux Joueurs)")


        with tab2:
            st.header("🔍 Recherche de Joueurs par Critères")
            st.info("Définissez vos besoins pour chaque poste, votre budget et vos critères de performance pour trouver les perles rares.")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Critères de Recherche")
                search_budget = st.slider("💰 Budget maximum par joueur (M€)", 0.1, 200.0, 20.0, 0.5, key="s_budget")
                search_criteria = st.selectbox("🎯 Trier par", ["overall_rating", "potential", "score"],
                                               format_func=lambda x: {"overall_rating": "Note Générale", "potential": "Potentiel", "score": "Score (Note+Potentiel)"}[x], key="s_criteria")

                with st.expander("Filtres de performance"):
                    search_age = st.slider("🎂 Âge", 16, 45, (16, 35), key="s_age")
                    search_potential = st.slider("💎 Potentiel", 40, 99, (60, 99), key="s_potential")
                    search_overall = st.slider("⭐ Overall minimum", 40, 99, 60, key="s_overall")
                
                search_filters = {'age_range': search_age, 'potential_range': search_potential, 'min_overall': search_overall}

            with col2:
                st.subheader("Quantité de joueurs par poste")
                
                cols_pos = st.columns(4)
                positions_to_find = {}
                # Diviser les postes pour un meilleur affichage
                split1 = ALL_POSITIONS[:len(ALL_POSITIONS)//2]
                split2 = ALL_POSITIONS[len(ALL_POSITIONS)//2:]
                
                with st.container():
                    c1, c2 = st.columns(2)
                    with c1:
                        for pos in split1:
                            positions_to_find[pos] = st.number_input(pos, min_value=0, max_value=10, value=0, key=f"s_{pos}")
                    with c2:
                        for pos in split2:
                            positions_to_find[pos] = st.number_input(pos, min_value=0, max_value=10, value=0, key=f"s_{pos}")

            if st.button("🔍 TROUVER DES JOUEURS", type="primary", use_container_width=True):
                total_players_requested = sum(positions_to_find.values())
                if total_players_requested == 0:
                    st.warning("Veuillez indiquer le nombre de joueurs que vous souhaitez trouver pour au moins un poste.")
                else:
                    with st.spinner("🕵️‍♂️ Recherche des joueurs correspondants..."):
                        found_players_df = search_players(df, positions_to_find, search_budget, search_criteria, search_filters)
                        st.session_state.found_players = found_players_df
            
            if 'found_players' in st.session_state:
                results = st.session_state.found_players
                st.subheader(f"Résultats de la Recherche ({len(results)} joueurs trouvés)")

                if results.empty:
                    st.warning("Aucun joueur ne correspond à tous vos critères. Essayez d'élargir votre recherche.")
                else:
                    display_df = results[['requested_position', 'name', 'age', 'overall_rating', 'potential', 'value_numeric', 'club_name']].copy()
                    display_df.rename(columns={
                        'requested_position': 'Poste Cherché', 'name': 'Nom', 'age': 'Âge',
                        'overall_rating': 'Note', 'potential': 'Potentiel', 'value_numeric': 'Valeur (M€)', 'club_name': 'Club'
                    }, inplace=True)
                    st.dataframe(display_df.sort_values(by=['Poste Cherché', search_criteria], ascending=[True, False]), use_container_width=True, hide_index=True)
                    st.download_button("📥 Télécharger les résultats", results.to_csv(index=False).encode('utf-8'), 'recherche_joueurs.csv', 'text/csv')

if __name__ == "__main__":
    main()
