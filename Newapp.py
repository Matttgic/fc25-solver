import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- Configuration de la page ---
st.set_page_config(page_title="FC25 Team Builder Pro", page_icon="⚽", layout="wide")

# --- CSS pour un design amélioré ---
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: bold;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.player-card {
    text-align: center;
    padding: 10px;
    border-radius: 12px;
    margin: 5px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 0.85rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.player-card:hover {
    transform: scale(1.05);
}
.stSelectbox > div > div > select {
    background-color: #f0f2f6;
}
.stButton>button {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ FC25 TEAM BUILDER PRO</h1>', unsafe_allow_html=True)

# --- Constantes ---
FORMATIONS = {
    "4-3-3": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "CAM": 1, "LW": 1, "RW": 1, "ST": 1}},
    "4-4-2": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "LM": 1, "CM": 2, "RM": 1, "ST": 2}},
    "3-5-2": {"positions": {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CDM": 2, "CAM": 1, "ST": 2}},
    "4-2-3-1": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 2, "CAM": 1, "LW": 1, "RW": 1, "ST": 1}},
}

POSITION_COMPATIBILITY = {
    "GK": ["GK"], "CB": ["CB"], "LB": ["LB", "LWB"], "RB": ["RB", "RWB"],
    "LWB": ["LWB", "LB", "LM"], "RWB": ["RWB", "RB", "RM"], "CDM": ["CDM", "CM"],
    "CM": ["CM", "CDM", "CAM"], "LM": ["LM", "LW", "CM"], "RM": ["RM", "RW", "CM"],
    "CAM": ["CAM", "CM", "ST"], "LW": ["LW", "LM", "ST"], "RW": ["RW", "RM", "ST"],
    "ST": ["ST", "CF", "LW", "RW"]
}

# Colonnes de statistiques à extraire
STATS_COLUMNS = [
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physicality'
]

# --- Fonctions ---
@st.cache_data
def load_data(uploaded_file):
    """Charge, nettoie et pré-calcule les données des joueurs."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Nettoyage de la valeur marchande
        if 'value' in df.columns:
            df['value_clean'] = df['value'].astype(str).str.replace('[€,]', '', regex=True)
            df['value_numeric'] = pd.to_numeric(df['value_clean'].str.replace('[MK]', '', regex=True), errors='coerce')
            df.loc[df['value_clean'].str.contains('K', na=False), 'value_numeric'] /= 1000
            df['value_numeric'] = df['value_numeric'].fillna(0)
        
        # Calculs supplémentaires
        df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
        df['potential_gap'] = df.get('potential', df['overall_rating']) - df['overall_rating']
        
        # S'assurer que les colonnes de stats existent
        for col in STATS_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan # Ajouter la colonne avec des NaN si elle manque
        
        if 'player_id' not in df.columns:
            df['player_id'] = range(len(df))
            
        return df.dropna(subset=['name', 'overall_rating', 'age'])
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        return None

@st.cache_data
def get_player_suggestions(_df, search_term=""):
    """Retourne les suggestions de joueurs pour l'autocomplétion."""
    if not search_term:
        top_players = _df.nlargest(100, 'overall_rating')
        return [f"{name} ({int(overall)})" for name, overall in zip(top_players['name'], top_players['overall_rating'])]
    
    filtered = _df[_df['name'].str.contains(search_term, case=False, na=False)]
    if filtered.empty:
        return ["Aucun joueur trouvé..."]
    
    filtered = filtered.nlargest(20, 'overall_rating')
    return [f"{name} ({int(overall)})" for name, overall in zip(filtered['name'], filtered['overall_rating'])]

def extract_player_name(selected_option):
    """Extrait le nom du joueur de l'option sélectionnée."""
    if not selected_option or selected_option == "Aucun joueur trouvé...":
        return ""
    return selected_option.split(' (')[0]

def can_play_position(player_positions, required_position):
    """Vérifie la compatibilité de la position du joueur."""
    if pd.isna(player_positions): return False
    player_pos_list = str(player_positions).split(',')
    compatible = POSITION_COMPATIBILITY.get(required_position, [required_position])
    return any(pos.strip() in compatible for pos in player_pos_list)

def get_filtered_players(df, position=None, exclude_ids=None, filters=None):
    """Filtre les joueurs selon une multitude de critères."""
    exclude_ids = exclude_ids or []
    result = df[~df['player_id'].isin(exclude_ids)].copy()
    
    if position:
        result = result[result['positions'].apply(lambda x: can_play_position(x, position))]
    
    if filters:
        # Appliquer les filtres de manière robuste
        if 'age_range' in filters:
            min_age, max_age = filters['age_range']
            result = result[result['age'].between(min_age, max_age)]
        if 'potential_range' in filters:
            min_pot, max_pot = filters['potential_range']
            result = result[result['potential'].between(min_pot, max_pot)]
        if 'max_budget' in filters:
            result = result[result['value_numeric'] <= filters['max_budget']]
        if 'min_overall' in filters:
            result = result[result['overall_rating'] >= filters['min_overall']]
        if not filters.get('include_free_agents', True):
            result = result[result['value_numeric'] > 0]
        if 'leagues' in filters and filters['leagues']:
            result = result[result.get('league_name', pd.Series(dtype='object')).isin(filters['leagues'])]
            
    return result

def optimize_team(df, formation, budget, filters):
    """Optimise une équipe en fonction de la formation, du budget et des filtres."""
    team = []
    used_ids = set()
    positions = FORMATIONS[formation]["positions"]

    for position, count in sorted(positions.items(), key=lambda x: -df['overall_rating'].max()):
        for _ in range(count):
            candidates = get_filtered_players(df, position, used_ids, {**filters, 'max_budget': budget})
            if candidates.empty: continue

            # Score composite pour le meilleur choix
            candidates['score'] = (
                candidates['overall_rating'] * 0.5 +
                candidates['potential'] * 0.3 +
                (40 - candidates['age']) * 0.2
            )
            
            best_player = candidates.loc[candidates['score'].idxmax()]
            cost = best_player['value_numeric']
            
            if budget >= cost:
                team.append({'player': best_player, 'position': position, 'cost': cost})
                budget -= cost
                used_ids.add(best_player['player_id'])

    return team, budget

def find_similar_players(df, target_name, budget, top_n=5):
    """Trouve des joueurs similaires à un joueur cible."""
    target_player = df[df['name'] == target_name].iloc[0]
    
    df_filtered = df[
        (df['value_numeric'] <= budget) & 
        (df['player_id'] != target_player['player_id']) &
        (df['age'].between(target_player['age'] - 3, target_player['age'] + 3))
    ].copy()
    
    # Score de similarité
    df_filtered['similarity'] = (
        100 - abs(df_filtered['overall_rating'] - target_player['overall_rating']) * 1.5 -
        abs(df_filtered['potential'] - target_player['potential']) * 1.5 -
        abs(df_filtered['age'] - target_player['age']) * 1.0
    )
    return df_filtered.nlargest(top_n, 'similarity')


def display_team_formation(players, formation):
    """Affiche la composition de l'équipe."""
    st.subheader(f"🏆 Composition en {formation}")
    
    # Organiser les joueurs par ligne
    lines = {"Attaquants": [], "Milieux": [], "Défenseurs": [], "Gardien": []}
    for p in players:
        pos = p['position']
        if pos in ["ST", "LW", "RW", "CF"]: lines["Attaquants"].append(p)
        elif pos in ["CM", "CAM", "CDM", "LM", "RM"]: lines["Milieux"].append(p)
        elif pos in ["CB", "LB", "RB", "LWB", "RWB"]: lines["Défenseurs"].append(p)
        else: lines["Gardien"].append(p)

    for line_name, line_players in lines.items():
        if not line_players: continue
        st.write(f"**{line_name}**")
        cols = st.columns(len(line_players) or 1)
        for i, player_data in enumerate(line_players):
            with cols[i]:
                p = player_data['player']
                st.markdown(f"""
                <div class="player-card">
                    <b>{p['name']}</b><br>
                    {player_data['position']} | {p['overall_rating']} OVR<br>
                    €{player_data['cost']:.1f}M | {int(p.get('age', 0))} ans
                </div>
                """, unsafe_allow_html=True)

# --- Application Principale ---
def main():
    uploaded_file = st.file_uploader("📁 **Chargez votre base de données joueurs FC25 (CSV)**", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        st.success(f"✅ **{len(df):,} joueurs chargés avec succès !**")
        
        tab1, tab2, tab3 = st.tabs(["🏗️ **Constructeur d'Équipe**", "🔍 **Recherche Avancée**", "👥 **Joueurs Similaires**"])

        # --- ONGLET 1: CONSTRUCTEUR ---
        with tab1:
            st.header("🏗️ **Constructeur d'équipe optimisé**")
            st.info("Créez automatiquement une équipe compétitive en fonction de votre budget, formation et critères.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Configuration")
                formation = st.selectbox("📋 **Formation**", list(FORMATIONS.keys()), key="builder_formation")
                budget = st.number_input("💰 **Budget total (M€)**", 10, 2000, 250, 25)
                
                st.subheader("Filtres")
                age_range = st.slider("🎂 Âge", 16, 45, (18, 34), key="builder_age")
                potential_range = st.slider("💎 **Potentiel**", 50, 100, (80, 99), key="builder_potential")
                min_overall = st.slider("⭐ Overall minimum", 40, 99, 75, key="builder_overall")
                include_free = st.checkbox("🆓 **Inclure agents libres (€0)**", value=True, key="builder_free_agents")
                
                filters = {
                    'age_range': age_range,
                    'potential_range': potential_range,
                    'min_overall': min_overall,
                    'include_free_agents': include_free
                }

                if st.button("🚀 **CONSTRUIRE MON ÉQUIPE**", type="primary"):
                    with st.spinner("⚡ Optimisation de l'équipe en cours..."):
                        team, remaining = optimize_team(df, formation, budget, filters)
                        st.session_state.team_results = {
                            'team': team, 'remaining_budget': remaining, 'formation': formation
                        }
            
            with col2:
                if 'team_results' in st.session_state:
                    res = st.session_state.team_results
                    team = res['team']
                    if not team:
                        st.error("❌ Aucun joueur ne correspond à ces critères. Essayez d'élargir vos filtres.")
                    else:
                        total_cost = sum(p['cost'] for p in team)
                        avg_overall = np.mean([p['player']['overall_rating'] for p in team])
                        avg_age = np.mean([p['player']['age'] for p in team])

                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("💰 Coût total", f"€{total_cost:.1f}M", f"€{res['remaining_budget']:.1f}M restant")
                        m_col2.metric("⭐ Overall moyen", f"{avg_overall:.1f}")
                        m_col3.metric("🎂 Âge moyen", f"{avg_age:.1f} ans")
                        
                        display_team_formation(team, res['formation'])

        # --- ONGLET 2: RECHERCHE AVANCÉE ---
        with tab2:
            st.header("🔍 **Recherche Avancée de Joueurs**")
            st.info("Trouvez des joueurs spécifiques avec des filtres précis et consultez leurs statistiques détaillées.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Paramètres de recherche")
                num_players = st.number_input("👥 **Nombre de joueurs à trouver**", 1, 100, 15)
                max_price = st.number_input("💰 **Prix max par joueur (M€)**", 0, 500, 50)
                
                st.subheader("Filtres")
                search_pos = st.selectbox("📍 Position", ["Toutes"] + list(POSITION_COMPATIBILITY.keys()), key="search_pos")
                search_age = st.slider("🎂 Âge", 16, 45, (18, 30), key="search_age")
                search_potential = st.slider("💎 **Potentiel**", 50, 100, (82, 99), key="search_potential")
                search_overall = st.slider("⭐ Overall min", 40, 99, 78, key="search_overall")
                
                search_filters = {
                    'age_range': search_age,
                    'potential_range': search_potential,
                    'min_overall': search_overall,
                    'max_budget': max_price,
                    'include_free_agents': True # Toujours inclure dans la recherche
                }

                if st.button("🔍 **LANCER LA RECHERCHE**"):
                    pos = None if search_pos == "Toutes" else search_pos
                    results = get_filtered_players(df, pos, filters=search_filters)
                    if not results.empty:
                        st.session_state.search_results = results.nlargest(num_players, 'overall_rating')
                    else:
                        st.warning("Aucun joueur trouvé avec ces critères.")
                        st.session_state.search_results = pd.DataFrame()

            with col2:
                if 'search_results' in st.session_state and not st.session_state.search_results.empty:
                    results = st.session_state.search_results
                    st.success(f"**{len(results)} joueurs trouvés !**")
                    
                    display_cols = ['name', 'age', 'overall_rating', 'potential', 'value_numeric', 'positions'] + STATS_COLUMNS
                    display_df = results[display_cols].copy()
                    display_df.rename(columns={
                        'name': 'Nom', 'age': 'Âge', 'overall_rating': 'OVR', 'potential': 'POT',
                        'value_numeric': 'Prix (M€)', 'positions': 'Positions',
                        'pace': 'VIT', 'shooting': 'TIR', 'passing': 'PAS', 
                        'dribbling': 'DRI', 'defending': 'DEF', 'physicality': 'PHY'
                    }, inplace=True)
                    display_df['Prix (M€)'] = display_df['Prix (M€)'].round(1)
                    
                    st.dataframe(display_df, use_container_width=True, height=500)

        # --- ONGLET 3: JOUEURS SIMILAIRES ---
        with tab3:
            st.header("👥 **Trouver des Joueurs Similaires**")
            st.info("Choisissez un joueur et un budget pour découvrir des alternatives au profil comparable.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Joueur Cible")
                search_filter = st.text_input("✍️ **Filtrez la liste par nom**", placeholder="Ex: Mbappé, Haaland...")
                suggestions = get_player_suggestions(df, search_filter)
                
                selected_player_option = st.selectbox(
                    "👤 **Choisissez un joueur**", options=suggestions, index=0
                )
                target_name = extract_player_name(selected_player_option)
                
                if target_name:
                    st.subheader("Budget pour Alternatives")
                    similar_budget = st.number_input("💰 **Budget max (M€)**", 1, 500, 100)
                    num_similar = st.slider("📊 **Nombre de résultats**", 3, 15, 5)

                    if st.button("FIND SIMILAR PLAYERS", type="primary"):
                        with st.spinner("Recherche d'alternatives..."):
                            similar_players = find_similar_players(df, target_name, similar_budget, num_similar)
                            if not similar_players.empty:
                                st.session_state.similar_players = similar_players
                                st.session_state.target_name = target_name
                            else:
                                st.warning(f"Aucun joueur similaire à '{target_name}' trouvé dans ce budget.")
                                st.session_state.similar_players = pd.DataFrame()
            
            with col2:
                if 'similar_players' in st.session_state and not st.session_state.similar_players.empty:
                    similar = st.session_state.similar_players
                    st.success(f"**Alternatives trouvées pour {st.session_state.target_name}**")
                    
                    display_similar = similar[['name', 'age', 'overall_rating', 'potential', 'value_numeric', 'similarity']].copy()
                    display_similar.rename(columns={
                        'name': 'Nom', 'age': 'Âge', 'overall_rating': 'OVR', 'potential': 'POT',
                        'value_numeric': 'Prix (M€)', 'similarity': 'Similarité (%)'
                    }, inplace=True)
                    display_similar['Prix (M€)'] = display_similar['Prix (M€)'].round(1)
                    display_similar['Similarité (%)'] = display_similar['Similarité (%)'].round(1)
                    
                    st.dataframe(display_similar, use_container_width=True)

                    # Graphique
                    fig = px.bar(display_similar, x='Nom', y='Similarité (%)',
                                 title=f"Score de similarité vs {st.session_state.target_name}",
                                 color='Similarité (%)', color_continuous_scale="Greens")
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main() 
