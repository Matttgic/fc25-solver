import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="FC25 Team Builder",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FC25 Team Builder")
st.markdown("Construisez votre équipe de rêve avec un budget donné !")

# Définition des formations tactiques
FORMATIONS = {
    "4-3-3": {
        "GK": 1, "CB": 2, "LB": 1, "RB": 1, 
        "CDM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1
    },
    "4-4-2": {
        "GK": 1, "CB": 2, "LB": 1, "RB": 1,
        "LM": 1, "CM": 2, "RM": 1, "ST": 2
    },
    "3-5-2": {
        "GK": 1, "CB": 3, "LWB": 1, "RWB": 1,
        "CDM": 1, "CM": 2, "ST": 2
    },
    "4-2-3-1": {
        "GK": 1, "CB": 2, "LB": 1, "RB": 1,
        "CDM": 2, "CAM": 3, "ST": 1
    },
    "3-4-3": {
        "GK": 1, "CB": 3, "LM": 1, "RM": 1,
        "CM": 2, "LW": 1, "RW": 1, "ST": 1
    }
}

# Mapping des positions compatibles
POSITION_COMPATIBILITY = {
    "GK": ["GK"],
    "CB": ["CB", "SW"],
    "LB": ["LB", "LWB", "LM"],
    "RB": ["RB", "RWB", "RM"],
    "LWB": ["LWB", "LB", "LM"],
    "RWB": ["RWB", "RB", "RM"],
    "CDM": ["CDM", "CM", "DM"],
    "CM": ["CM", "CDM", "CAM"],
    "LM": ["LM", "LW", "LB", "LWB"],
    "RM": ["RM", "RW", "RB", "RWB"],
    "CAM": ["CAM", "CM", "CF"],
    "LW": ["LW", "LM", "LF"],
    "RW": ["RW", "RM", "RF"],
    "ST": ["ST", "CF", "LF", "RF"]
}

@st.cache_data
def load_data(uploaded_file):
    """Charge et nettoie les données du CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Nettoyage des colonnes de prix
        if 'value' in df.columns:
            df['value_numeric'] = df['value'].str.replace('€', '').str.replace('M', '').str.replace('K', '')
            df['value_numeric'] = pd.to_numeric(df['value_numeric'], errors='coerce')
            # Conversion en millions
            df.loc[df['value'].str.contains('K', na=False), 'value_numeric'] /= 1000
        
        if 'wage' in df.columns:
            df['wage_numeric'] = df['wage'].str.replace('€', '').str.replace('K', '').str.replace('M', '')
            df['wage_numeric'] = pd.to_numeric(df['wage_numeric'], errors='coerce')
        
        # Nettoyage des positions
        if 'positions' in df.columns:
            df['positions_list'] = df['positions'].str.split(',')
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

def can_play_position(player_positions, required_position):
    """Vérifie si un joueur peut jouer à une position donnée"""
    if not player_positions or pd.isna(player_positions):
        return False
    
    player_pos_list = player_positions.split(',') if isinstance(player_positions, str) else player_positions
    compatible_positions = POSITION_COMPATIBILITY.get(required_position, [required_position])
    
    return any(pos.strip() in compatible_positions for pos in player_pos_list)

def get_players_for_position(df, position, exclude_ids=None):
    """Récupère les joueurs pouvant jouer à une position donnée"""
    exclude_ids = exclude_ids or []
    
    mask = df['positions'].apply(lambda x: can_play_position(x, position))
    available_players = df[mask & ~df['player_id'].isin(exclude_ids)]
    
    return available_players.copy()

def optimize_team_greedy(df, formation, budget, optimize_by):
    """Optimise l'équipe avec un algorithme glouton"""
    selected_players = []
    remaining_budget = budget
    used_player_ids = set()
    
    formation_requirements = FORMATIONS[formation].copy()
    
    # Trier les positions par difficulté de remplacement (GK en premier, puis positions rares)
    position_priority = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "LM", "RM", "CAM", "LW", "RW", "ST"]
    
    for position in position_priority:
        if position not in formation_requirements:
            continue
            
        needed_count = formation_requirements[position]
        
        for _ in range(needed_count):
            available_players = get_players_for_position(df, position, used_player_ids)
            
            if available_players.empty:
                continue
                
            # Filtrer par budget
            affordable_players = available_players[available_players['value_numeric'] <= remaining_budget]
            
            if affordable_players.empty:
                continue
            
            # Optimiser selon le critère choisi
            if optimize_by in affordable_players.columns:
                best_player = affordable_players.loc[affordable_players[optimize_by].idxmax()]
            else:
                best_player = affordable_players.iloc[0]
            
            selected_players.append({
                'player': best_player,
                'position': position,
                'cost': best_player['value_numeric']
            })
            
            remaining_budget -= best_player['value_numeric']
            used_player_ids.add(best_player['player_id'])
    
    return selected_players, remaining_budget

def display_team_formation(selected_players, formation):
    """Affiche l'équipe selon la formation choisie"""
    st.subheader(f"🏆 Équipe optimisée - Formation {formation}")
    
    # Organiser les joueurs par ligne
    formation_lines = {
        "4-3-3": [
            ["ST"],
            ["LW", "RW"],
            ["CM", "CM", "CDM"],
            ["LB", "CB", "CB", "RB"],
            ["GK"]
        ],
        "4-4-2": [
            ["ST", "ST"],
            ["LM", "CM", "CM", "RM"],
            ["LB", "CB", "CB", "RB"],
            ["GK"]
        ],
        "3-5-2": [
            ["ST", "ST"],
            ["CM", "CDM", "CM"],
            ["LWB", "RWB"],
            ["CB", "CB", "CB"],
            ["GK"]
        ],
        "4-2-3-1": [
            ["ST"],
            ["CAM", "CAM", "CAM"],
            ["CDM", "CDM"],
            ["LB", "CB", "CB", "RB"],
            ["GK"]
        ],
        "3-4-3": [
            ["LW", "ST", "RW"],
            ["CM", "CM"],
            ["LM", "RM"],
            ["CB", "CB", "CB"],
            ["GK"]
        ]
    }
    
    lines = formation_lines.get(formation, [])
    player_dict = {p['position']: p for p in selected_players}
    
    for line in lines:
        cols = st.columns(len(line))
        for i, pos in enumerate(line):
            with cols[i]:
                if pos in player_dict:
                    player_info = player_dict[pos]
                    player = player_info['player']
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px solid #1f77b4; border-radius: 10px; margin: 5px;">
                        <strong>{player['name']}</strong><br>
                        <small>{pos} | OVR: {player['overall_rating']}</small><br>
                        <small>€{player_info['cost']:.1f}M</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px dashed #ccc; border-radius: 10px; margin: 5px;">
                        <strong>-</strong><br>
                        <small>{pos}</small><br>
                        <small>Non trouvé</small>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    # Upload du fichier
    uploaded_file = st.file_uploader("📁 Chargez votre fichier CSV FC25", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Données chargées: {len(df)} joueurs")
            
            # Sidebar pour les paramètres
            with st.sidebar:
                st.header("⚙️ Paramètres")
                
                # Budget
                budget = st.number_input("💰 Budget (en millions €)", 
                                       min_value=1, max_value=2000, value=500, step=10)
                
                # Formation
                formation = st.selectbox("📋 Formation tactique", 
                                       options=list(FORMATIONS.keys()))
                
                # Critère d'optimisation
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                optimize_by = st.selectbox("🎯 Optimiser par", 
                                         options=['overall_rating', 'potential'] + numeric_columns)
                
                # Bouton d'optimisation
                if st.button("🚀 Optimiser l'équipe", type="primary"):
                    with st.spinner("Optimisation en cours..."):
                        selected_players, remaining_budget = optimize_team_greedy(
                            df, formation, budget, optimize_by
                        )
                        
                        st.session_state['team'] = selected_players
                        st.session_state['remaining_budget'] = remaining_budget
                        st.session_state['formation'] = formation
                        st.session_state['total_budget'] = budget
            
            # Affichage des résultats
            if 'team' in st.session_state:
                team = st.session_state['team']
                
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                
                total_cost = sum(p['cost'] for p in team)
                avg_overall = np.mean([p['player']['overall_rating'] for p in team])
                avg_potential = np.mean([p['player']['potential'] for p in team if 'potential' in p['player']])
                
                with col1:
                    st.metric("💰 Coût total", f"€{total_cost:.1f}M")
                with col2:
                    st.metric("💳 Budget restant", f"€{st.session_state['remaining_budget']:.1f}M")
                with col3:
                    st.metric("⭐ Overall moyen", f"{avg_overall:.1f}")
                with col4:
                    st.metric("🌟 Potentiel moyen", f"{avg_potential:.1f}")
                
                # Affichage de la formation
                display_team_formation(team, st.session_state['formation'])
                
                # Tableau détaillé
                st.subheader("📊 Détails de l'équipe")
                
                team_data = []
                for p in team:
                    player = p['player']
                    team_data.append({
                        'Position': p['position'],
                        'Nom': player['name'],
                        'Club': player.get('club_name', 'N/A'),
                        'Overall': player['overall_rating'],
                        'Potentiel': player.get('potential', 'N/A'),
                        'Valeur (€M)': f"{p['cost']:.1f}",
                        'Âge': 2025 - pd.to_datetime(player['dob'], errors='coerce').year if 'dob' in player else 'N/A'
                    })
                
                team_df = pd.DataFrame(team_data)
                st.dataframe(team_df, use_container_width=True)
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(team_df, x='Position', y='Overall', 
                               title="Overall par position")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(team_df, values='Valeur (€M)', names='Position',
                               title="Répartition du budget")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Aperçu des données
            with st.expander("👀 Aperçu des données"):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"Colonnes disponibles: {list(df.columns)}")

if __name__ == "__main__":
    main()
