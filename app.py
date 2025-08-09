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
        df.columns = df.columns.str.strip() # Nettoyer les noms de colonnes

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

def solve_team(df, formation, budget, criteria, filters, excluded_player_ids=None, num_solutions=1):
    """Solveur qui peut générer une ou plusieurs solutions optimales."""
    # 1. Filtrer les joueurs
    candidate_df = df.copy()
    if excluded_player_ids:
        candidate_df = candidate_df[~candidate_df['player_id'].isin(excluded_player_ids)]

    if not filters.get('include_free_agents', True):
        candidate_df = candidate_df[candidate_df['value_numeric'] > 0]
    
    if 'age_range' in filters:
        candidate_df = candidate_df[candidate_df['age'].between(*filters['age_range'])]
    if 'potential_range' in filters:
        candidate_df = candidate_df[candidate_df['potential'].between(*filters['potential_range'])]
    if 'min_overall' in filters:
        candidate_df = candidate_df[candidate_df['overall_rating'] >= filters['min_overall']]
    
    if filters.get('nationality') and filters['nationality'] != "Toutes":
        candidate_df = candidate_df[candidate_df['country_name'] == filters['nationality']]

    if candidate_df.empty:
        return None if num_solutions == 1 else []

    # 2. Définir le problème
    prob = LpProblem("TeamBuilder", LpMaximize)
    player_vars = {}
    positions_to_fill = FORMATIONS[formation]
    num_players_in_formation = sum(positions_to_fill.values())

    for position, count in positions_to_fill.items():
        eligible_for_pos = candidate_df[candidate_df['positions'].apply(lambda x: can_play_position(x, position))]
        for p_idx in eligible_for_pos.index:
            player_vars[(p_idx, position)] = LpVariable(f"player_{p_idx}_pos_{position}", cat=LpBinary)

    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, criteria] for (p_idx, pos) in player_vars), "Total_Score"
    prob += lpSum(player_vars[(p_idx, pos)] * candidate_df.loc[p_idx, 'value_numeric'] for (p_idx, pos) in player_vars) <= budget, "Budget"

    for position, count in positions_to_fill.items():
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx, pos) in player_vars if pos == position) == count, f"Formation_{position}"
    for p_idx in candidate_df.index:
        prob += lpSum(player_vars[(p_idx, pos)] for (p_idx_c, pos) in player_vars if p_idx_c == p_idx) <= 1, f"Uniqueness_{p_idx}"

    # 3. Résoudre et collecter les solutions
    solutions = []
    for i in range(num_solutions):
        prob.solve()

        if prob.status == 1: # Si une solution optimale est trouvée
            solution_vars = [var for var, val in player_vars.items() if val.varValue == 1]
            if not solution_vars:
                break # Plus de solution possible

            team = [{'player': df.loc[p_idx], 'position': pos} for (p_idx, pos) in solution_vars]
            solutions.append(team)

            # Ajouter une contrainte pour exclure cette solution exacte
            prob += lpSum(player_vars[var] for var in solution_vars) <= num_players_in_formation - 1
        else:
            break # Arrêter s'il n'y a plus de solution

    if num_solutions == 1:
        return solutions[0] if solutions else None
    else:
        return solutions

def display_team_results(team, formation, budget, criteria, title, chem_total=None, chem_indiv=None):
    """Affiche les résultats d'une équipe, y compris le collectif."""
    st.subheader(title)
    if team is None:
        st.warning("❌ **Aucune solution trouvée pour cette équipe.**")
        return

    st.success(f"✅ **Équipe trouvée ! ({len(team)} joueurs)**")
    team_data, total_cost, total_score = [], 0, 0

    # Utiliser une copie pour éviter de modifier l'original
    player_chem = chem_indiv.copy() if chem_indiv else {}

    for p_data in team:
        player, cost = p_data['player'], p_data['player']['value_numeric']
        total_cost += cost
        # 'criteria' peut être 'collectif', il faut donc le gérer
        stat_criteria = 'score' if criteria == 'collectif' else criteria
        total_score += player[stat_criteria]

        chem = player_chem.get(player['player_id'], 0)
        team_data.append({
            "Position": p_data['position'], "Nom": player['name'], "OVR": player['overall_rating'],
            "POT": player['potential'], "Collectif": f"♦ {chem}", "Âge": int(player['age']), "Coût (M€)": f"{cost:.2f}"
        })

    team_df = pd.DataFrame(team_data).sort_values(
        by="Position", key=lambda x: x.map({pos: i for i, pos in enumerate(FORMATIONS[formation].keys())})
    )
    st.dataframe(team_df, use_container_width=True, hide_index=True)

    # Affichage des métriques
    cols = st.columns(3)
    cols[0].metric("💰 Coût Total", f"€{total_cost:.2f}M", f"Budget: €{budget:.2f}M")
    avg_score_display = "Score" if criteria == "collectif" else criteria.replace('_', ' ').replace('rating', 'Générale').title()
    cols[1].metric(f"⭐ Moyenne '{avg_score_display}'", f"{(total_score / len(team)):.1f}")
    if chem_total is not None:
        cols[2].metric("💎 Collectif Total", f"{chem_total} / 33")

    st.download_button(f"📥 Télécharger {title}", team_df.to_csv(index=False).encode('utf-8'), f'{title.replace(" ", "_").lower()}.csv', 'text/csv', key=f"download_{title}")


def calculate_chemistry(team, formation):
    """Calcule le score de collectif d'une équipe selon un modèle simplifié."""
    if not team:
        return 0, {}

    team_df = pd.DataFrame([p['player'] for p in team])

    # Compter les occurrences
    nation_counts = team_df['country_name'].value_counts()
    league_counts = team_df['club_league_name'].value_counts()
    club_counts = team_df['club_name'].value_counts()

    # Définir les seuils de collectif
    chem_points = {}

    # NOUVEAU SYSTÈME DE CHIMIE (similaire à FC24)
    # Chaque joueur peut avoir de 0 à 3 points.
    NATION_THRESHOLDS = {3: 1, 5: 2, 8: 3}
    LEAGUE_THRESHOLDS = {3: 1, 5: 2, 8: 3}
    CLUB_THRESHOLDS = {2: 1, 4: 2, 7:3}

    for idx, player in team_df.iterrows():
        player_id = player['player_id']
        chem_points[player_id] = 0

        # Points de club
        player_club = player['club_name']
        club_count = club_counts.get(player_club, 0)
        for threshold, points in sorted(CLUB_THRESHOLDS.items(), reverse=True):
            if club_count >= threshold:
                chem_points[player_id] += points
                break

        # Points de ligue
        player_league = player['club_league_name']
        league_count = league_counts.get(player_league, 0)
        for threshold, points in sorted(LEAGUE_THRESHOLDS.items(), reverse=True):
            if league_count >= threshold:
                chem_points[player_id] += points
                break

        # Points de nation
        player_nation = player['country_name']
        nation_count = nation_counts.get(player_nation, 0)
        for threshold, points in sorted(NATION_THRESHOLDS.items(), reverse=True):
            if nation_count >= threshold:
                chem_points[player_id] += points
                break

    # Caper le score individuel à 3 et sommer
    total_chem = sum(min(3, p) for p in chem_points.values())

    # Bonus: vérifier si le joueur est à son poste (simplifié)
    # Pour l'instant, on ne le fait pas pour garder le modèle simple.

    return total_chem, {pid: min(3, p) for pid, p in chem_points.items()}


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

    if not filters.get('include_free_agents', True):
        candidate_df = candidate_df[candidate_df['value_numeric'] > 0]

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

        # --- Section: Constructeur d'Équipe ---
        st.header("🏗️ Constructeur d'Équipe Optimisé")
        st.info("Le solveur recherche la meilleure équipe mathématiquement possible, puis une deuxième équipe avec les joueurs restants.")

        # Préparer les listes pour les filtres
        unique_nationalities = df['country_name'].dropna().unique()
        nationalities = ["Toutes"] + sorted(unique_nationalities)

        team_builder_cols = st.columns([1, 2])
        with team_builder_cols[0]:
            st.subheader("Configuration")
            formation = st.selectbox("📋 Formation", list(FORMATIONS.keys()), key="t1_formation")
            budget = st.number_input("💰 Budget total (M€)", min_value=0.1, value=50.0, step=5.0, key="t1_budget")
            criteria = st.selectbox("🎯 Maximiser", ["score", "overall_rating", "potential", "collectif"],
                                    format_func=lambda x: {"score": "Score (Note+Potentiel)", "overall_rating": "Note Générale", "potential": "Potentiel", "collectif": "Collectif"}[x], key="t1_criteria")

            with st.expander("Filtres avancés"):
                age_range = st.slider("🎂 Âge", 16, 45, (16, 40), key="t1_age")
                potential_range = st.slider("💎 Potentiel", 40, 99, (40, 99), key="t1_potential")
                min_overall = st.slider("⭐ Overall minimum", 40, 99, 40, key="t1_overall")
                selected_nationality = st.selectbox("🌍 Nationalité", options=nationalities)
                include_free_agents = st.checkbox("🆓 Inclure agents libres (€0)", value=True, key="t1_free_agents")

            filters = {
                'age_range': age_range, 'potential_range': potential_range,
                'min_overall': min_overall, 'include_free_agents': include_free_agents,
                'nationality': selected_nationality
            }

            if st.button("🚀 TROUVER LES ÉQUIPES OPTIMALES", type="primary", use_container_width=True):
                team1 = None
                if criteria == 'collectif':
                    with st.spinner("🧠 Recherche des meilleures combinaisons pour le collectif (peut prendre un moment)..."):
                        # 1. Générer un pool de solutions très performantes
                        candidate_teams = solve_team(df, formation, budget, 'score', filters, num_solutions=50)

                        if candidate_teams:
                            # 2. Calculer le collectif pour chaque solution
                            best_chem_score = -1
                            best_chem_team = None
                            for team in candidate_teams:
                                chem_score, _ = calculate_chemistry(team, formation)
                                if chem_score > best_chem_score:
                                    best_chem_score = chem_score
                                    best_chem_team = team
                            team1 = best_chem_team
                else:
                    with st.spinner("🧠 Recherche de la meilleure équipe..."):
                        team1 = solve_team(df, formation, budget, criteria, filters)

                st.session_state.team1_results = team1
                st.session_state.team2_results = None # Reset
                
                if team1:
                    with st.spinner("🧠 Recherche de la deuxième meilleure équipe..."):
                        team1_ids = [p['player']['player_id'] for p in team1]
                        team1_cost = sum(p['player']['value_numeric'] for p in team1)
                        remaining_budget = budget - team1_cost
                        
                        # La deuxième équipe est toujours optimisée pour le score
                        team2 = solve_team(df, formation, remaining_budget, 'score', filters, excluded_player_ids=team1_ids)
                        st.session_state.team2_results = team2

        with team_builder_cols[1]:
            if 'team1_results' in st.session_state:
                team1 = st.session_state.team1_results
                if team1 is None:
                    st.error("❌ **Aucune solution trouvée.** Il est mathématiquement impossible de former une équipe avec ces filtres et ce budget. Essayez d'augmenter le budget ou d'élargir les filtres.")
                else:
                    chem1_total, chem1_indiv = calculate_chemistry(team1, formation)
                    display_team_results(team1, formation, budget, criteria, "🏆 Meilleure Équipe Possible", chem1_total, chem1_indiv)

                    st.markdown("---")

                    team2 = st.session_state.get('team2_results')
                    if team2:
                        chem2_total, chem2_indiv = calculate_chemistry(team2, formation)
                        display_team_results(team2, formation, budget, criteria, "🥈 Deuxième Meilleure Équipe (Nouveaux Joueurs)", chem2_total, chem2_indiv)
                    else:
                        st.info("Aucune deuxième équipe n'a pu être formée avec les joueurs et le budget restants.")

        st.divider()

        # --- Section: Recherche de Joueurs ---
        st.header("🔍 Recherche de Joueurs par Critères")
        st.info("Définissez vos besoins pour chaque poste, votre budget et vos critères de performance pour trouver les perles rares.")

        search_cols = st.columns([1, 2])
        with search_cols[0]:
            st.subheader("Critères de Recherche")
            search_budget = st.slider("💰 Budget maximum par joueur (M€)", 0.1, 200.0, 20.0, 0.5, key="s_budget")
            search_criteria = st.selectbox("🎯 Trier par", ["overall_rating", "potential", "score"],
                                           format_func=lambda x: {"overall_rating": "Note Générale", "potential": "Potentiel", "score": "Score (Note+Potentiel)"}[x], key="s_criteria")

            with st.expander("Filtres de performance"):
                search_age = st.slider("🎂 Âge", 16, 45, (16, 35), key="s_age")
                search_potential = st.slider("💎 Potentiel", 40, 99, (60, 99), key="s_potential")
                search_overall = st.slider("⭐ Overall minimum", 40, 99, 60, key="s_overall")
                search_free_agents = st.checkbox("🆓 Inclure agents libres (€0)", value=True, key="s_free_agents")
            
            search_filters = {'age_range': search_age, 'potential_range': search_potential, 'min_overall': search_overall, 'include_free_agents': search_free_agents}

        with search_cols[1]:
            st.subheader("Quantité de joueurs par poste")

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
                # 1. Trier les résultats avant de renommer les colonnes
                sorted_results = results.sort_values(by=['requested_position', search_criteria], ascending=[True, False])

                # 2. Définir les colonnes à afficher et s'assurer qu'elles existent
                cols_to_show = ['requested_position', 'name', 'age', 'overall_rating', 'potential', 'score', 'value_numeric', 'club_name']
                display_df = sorted_results[[col for col in cols_to_show if col in sorted_results.columns]].copy()

                # 3. Renommer les colonnes pour l'affichage
                display_df.rename(columns={
                    'requested_position': 'Poste Cherché', 'name': 'Nom', 'age': 'Âge',
                    'overall_rating': 'Note', 'potential': 'Potentiel', 'score': 'Score',
                    'value_numeric': 'Valeur (M€)', 'club_name': 'Club'
                }, inplace=True)

                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Télécharger les résultats", sorted_results.to_csv(index=False).encode('utf-8'), 'recherche_joueurs.csv', 'text/csv', key='download_search')

if __name__ == "__main__":
    main()
