import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="FC25 Ultimate Team Builder Pro",
    page_icon="⚽",
    layout="wide"
)

# CSS personnalisé pour un design pro
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.info-box {
    padding: 0.5rem;
    border-radius: 8px;
    background-color: #f0f2f6;
    border-left: 4px solid #FF6B35;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #333;
}
.mode-card {
    padding: 1rem;
    border-radius: 10px;
    border: 2px solid #ddd;
    margin: 0.5rem;
    text-align: center;
    transition: all 0.3s;
}
.mode-card:hover {
    border-color: #FF6B35;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
}
.player-card {
    text-align: center;
    padding: 10px;
    border-radius: 15px;
    margin: 5px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ FC25 ULTIMATE TEAM BUILDER PRO</h1>', unsafe_allow_html=True)
st.markdown("**🔥 L'outil le plus avancé pour construire votre équipe de rêve !**")

# Définition des formations tactiques avec bonus
FORMATIONS = {
    "4-3-3 (Attaque)": {
        "positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1},
        "bonus": {"attack": 15, "defense": 0, "creativity": 10}
    },
    "4-4-2 (Équilibré)": {
        "positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "LM": 1, "CM": 2, "RM": 1, "ST": 2},
        "bonus": {"attack": 5, "defense": 5, "creativity": 5}
    },
    "3-5-2 (Possession)": {
        "positions": {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CDM": 1, "CM": 2, "ST": 2},
        "bonus": {"attack": 0, "defense": 10, "creativity": 15}
    },
    "4-2-3-1 (Créatif)": {
        "positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 2, "CAM": 3, "ST": 1},
        "bonus": {"attack": 10, "defense": 0, "creativity": 20}
    },
    "5-3-2 (Défense)": {
        "positions": {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CM": 3, "ST": 2},
        "bonus": {"attack": -5, "defense": 20, "creativity": 0}
    },
    "3-4-3 (Intense)": {
        "positions": {"GK": 1, "CB": 3, "LM": 1, "RM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1},
        "bonus": {"attack": 20, "defense": -5, "creativity": 5}
    }
}

# Modes de jeu avancés
GAME_MODES = {
    "🚀 Ultimate Team": {
        "description": "Mode ultime avec chimie et synergies",
        "budget_multiplier": 1.0,
        "focus": "chimie",
        "constraints": {"max_same_nationality": 4, "min_different_leagues": 3}
    },
    "💎 Chasse aux Pépites": {
        "description": "Jeunes talents à fort potentiel",
        "budget_multiplier": 0.7,
        "focus": "potential",
        "constraints": {"max_age": 23, "min_potential_gap": 5}
    },
    "👑 Galactiques": {
        "description": "Les meilleurs joueurs absolus",
        "budget_multiplier": 3.0,
        "focus": "overall_rating",
        "constraints": {"min_overall": 85}
    },
    "💰 Mercato Réaliste": {
        "description": "Budget incluant salaires 3 ans",
        "budget_multiplier": 1.5,
        "focus": "value_for_money",
        "constraints": {"include_wages": True}
    },
    "⚖️ Qualité/Prix": {
        "description": "Meilleur rapport performance/coût",
        "budget_multiplier": 0.8,
        "focus": "efficiency",
        "constraints": {"max_price_per_overall": 15}
    }
}

# Mapping des positions compatibles (étendu)
POSITION_COMPATIBILITY = {
    "GK": ["GK"],
    "CB": ["CB", "SW", "CDM"],
    "LB": ["LB", "LWB", "LM", "CB"],
    "RB": ["RB", "RWB", "RM", "CB"],
    "LWB": ["LWB", "LB", "LM", "LW"],
    "RWB": ["RWB", "RB", "RM", "RW"],
    "CDM": ["CDM", "CM", "CB", "DM"],
    "CM": ["CM", "CDM", "CAM", "LM", "RM"],
    "LM": ["LM", "LW", "LB", "LWB", "CM"],
    "RM": ["RM", "RW", "RB", "RWB", "CM"],
    "CAM": ["CAM", "CM", "CF", "LW", "RW"],
    "LW": ["LW", "LM", "LF", "ST", "CAM"],
    "RW": ["RW", "RM", "RF", "ST", "CAM"],
    "ST": ["ST", "CF", "LF", "RF", "CAM"]
}

@st.cache_data
def load_data(uploaded_file):
    """Charge et nettoie les données du CSV avec traitement avancé"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Fallback si pas de colonne positions
        mask = pd.Series([True] * len(df))
    
    available_players = df[mask & ~df['player_id'].isin(exclude_ids)].copy()
    
    # Application des filtres
    if filters:
        if 'age_range' in filters:
            min_age, max_age = filters['age_range']
            available_players = available_players[
                (available_players['age'] >= min_age) & 
                (available_players['age'] <= max_age)
            ]
        
        if 'leagues' in filters and filters['leagues']:
            if 'league_name' in available_players.columns:
                available_players = available_players[
                    available_players['league_name'].isin(filters['leagues'])
                ]
        
        if 'nationalities' in filters and filters['nationalities']:
            if 'nationality' in available_players.columns:
                available_players = available_players[
                    available_players['nationality'].isin(filters['nationalities'])
                ]
    
    return available_players

def search_players_advanced(df, filters):
    """Recherche avancée de joueurs avec critères multiples"""
    result_df = df.copy()
    
    # Filtres par critères
    if filters.get('positions'):
        if 'positions' in df.columns:
            position_mask = df['positions'].apply(
                lambda x: any(can_play_position(x, pos) for pos in filters['positions'])
            )
            result_df = result_df[position_mask]
    
    if filters.get('min_overall'):
        result_df = result_df[result_df['overall_rating'] >= filters['min_overall']]
    
    if filters.get('max_overall'):
        result_df = result_df[result_df['overall_rating'] <= filters['max_overall']]
    
    if filters.get('age_range'):
        min_age, max_age = filters['age_range']
        result_df = result_df[(result_df['age'] >= min_age) & (result_df['age'] <= max_age)]
    
    if filters.get('max_value') and 'value_numeric' in result_df.columns:
        result_df = result_df[result_df['value_numeric'] <= filters['max_value']]
    
    if filters.get('leagues') and 'league_name' in result_df.columns:
        result_df = result_df[result_df['league_name'].isin(filters['leagues'])]
    
    if filters.get('nationalities') and 'nationality' in result_df.columns:
        result_df = result_df[result_df['nationality'].isin(filters['nationalities'])]
    
    if filters.get('clubs') and 'club_name' in result_df.columns:
        result_df = result_df[result_df['club_name'].isin(filters['clubs'])]
    
    if filters.get('max_players'):
        result_df = result_df.head(filters['max_players'])
    
    return result_df

def optimize_team_advanced(df, formation, budget, game_mode, optimization_weights, filters):
    """Optimisation avancée avec algorithme multicritère"""
    selected_players = []
    remaining_budget = budget
    used_player_ids = set()
    
    formation_data = FORMATIONS[formation]
    formation_requirements = formation_data["positions"].copy()
    mode_data = GAME_MODES[game_mode]
    
    # Calcul du score composite pour chaque joueur
    def calculate_composite_score(player):
        weights = optimization_weights
        score = 0
        
        # Overall rating
        score += player['overall_rating'] * weights['overall'] / 100
        
        # Potentiel
        if 'potential' in player and not pd.isna(player['potential']):
            score += player['potential'] * weights['potential'] / 100
        
        # Âge (inversé - plus jeune = mieux)
        age_score = max(0, 40 - player.get('age', 25)) / 40 * 100
        score += age_score * weights['age'] / 100
        
        # Efficacité prix
        if 'value_numeric' in player and player['value_numeric'] > 0:
            efficiency = player['overall_rating'] / np.log1p(player['value_numeric'])
            score += efficiency * weights.get('efficiency', 10) / 100
        
        return score
    
    # Tri des positions par priorité
    position_priority = ["GK", "CB", "CDM", "LB", "RB", "LWB", "RWB", "CM", "LM", "RM", "CAM", "LW", "RW", "ST"]
    
    for position in position_priority:
        if position not in formation_requirements:
            continue
            
        needed_count = formation_requirements[position]
        
        for _ in range(needed_count):
            available_players = get_players_for_position(df, position, used_player_ids, filters)
            
            if available_players.empty:
                continue
            
            # Filtrer par budget (incluant salaires si mode mercato)
            if mode_data["constraints"].get("include_wages", False) and 'wage_numeric' in available_players.columns:
                # Budget sur 3 ans incluant salaires
                total_cost = available_players['value_numeric'] + (available_players['wage_numeric'] * 3)
                affordable_players = available_players[total_cost <= remaining_budget]
            else:
                if 'value_numeric' in available_players.columns:
                    affordable_players = available_players[available_players['value_numeric'] <= remaining_budget]
                else:
                    affordable_players = available_players
            
            if affordable_players.empty:
                continue
            
            # Appliquer les contraintes du mode de jeu
            if mode_data["constraints"].get("min_overall"):
                affordable_players = affordable_players[
                    affordable_players['overall_rating'] >= mode_data["constraints"]["min_overall"]
                ]
            
            if mode_data["constraints"].get("max_age"):
                affordable_players = affordable_players[
                    affordable_players['age'] <= mode_data["constraints"]["max_age"]
                ]
            
            if affordable_players.empty:
                continue
            
            # Calcul des scores et sélection du meilleur
            affordable_players['composite_score'] = affordable_players.apply(calculate_composite_score, axis=1)
            best_player = affordable_players.loc[affordable_players['composite_score'].idxmax()]
            
            # Calcul du coût réel
            if mode_data["constraints"].get("include_wages", False) and 'wage_numeric' in best_player:
                real_cost = best_player.get('value_numeric', 0) + (best_player.get('wage_numeric', 0) * 3)
            else:
                real_cost = best_player.get('value_numeric', 0)
            
            selected_players.append({
                'player': best_player,
                'position': position,
                'cost': real_cost,
                'score': best_player['composite_score']
            })
            
            remaining_budget -= real_cost
            used_player_ids.add(best_player['player_id'])
    
    return selected_players, remaining_budget

def analyze_team_balance(selected_players):
    """Analyse l'équilibre de l'équipe"""
    if not selected_players:
        return []
    
    suggestions = []
    ages = [p['player'].get('age', 25) for p in selected_players]
    overalls = [p['player']['overall_rating'] for p in selected_players]
    
    # Analyse de l'âge
    avg_age = np.mean(ages)
    if avg_age > 30:
        suggestions.append("🧓 Équipe vieillissante - Pensez à rajeunir l'effectif")
    elif avg_age < 22:
        suggestions.append("👶 Équipe très jeune - Manque d'expérience possible")
    
    # Analyse de la régularité
    overall_std = np.std(overalls)
    if overall_std > 8:
        suggestions.append("⚖️ Gros écarts de niveau - Équipe déséquilibrée")
    
    # Analyse de l'attaque
    attack_power = calculate_attack_power(selected_players)
    if attack_power < 60:
        suggestions.append("⚽ Attaque faible - Manque de créativité offensive")
    
    # Analyse de la défense
    defense_power = calculate_defense_power(selected_players)
    if defense_power < 60:
        suggestions.append("🛡️ Défense fragile - Renforcez l'arrière-garde")
    
    # Analyse des nationalités
    nationalities = [p['player'].get('nationality', 'Unknown') for p in selected_players]
    nationality_counts = pd.Series(nationalities).value_counts()
    if len(nationality_counts) < 4:
        suggestions.append("🌍 Manque de diversité - Ajoutez des nationalités")
    
    return suggestions

def display_advanced_formation(selected_players, formation):
    """Affichage avancé de la formation avec stats"""
    st.subheader(f"🏆 Formation {formation}")
    
    # Lignes de formation personnalisées
    formation_layouts = {
        "4-3-3 (Attaque)": [["ST"], ["LW", "RW"], ["CM", "CDM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
        "4-4-2 (Équilibré)": [["ST", "ST"], ["LM", "CM", "CM", "RM"], ["LB", "CB", "CB", "RB"], ["GK"]],
        "3-5-2 (Possession)": [["ST", "ST"], ["CM", "CDM", "CM"], ["LWB", "RWB"], ["CB", "CB", "CB"], ["GK"]],
        "4-2-3-1 (Créatif)": [["ST"], ["CAM", "CAM", "CAM"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
        "5-3-2 (Défense)": [["ST", "ST"], ["CM", "CM", "CM"], ["LWB", "RWB"], ["CB", "CB", "CB"], ["GK"]],
        "3-4-3 (Intense)": [["LW", "ST", "RW"], ["CM", "CM"], ["LM", "RM"], ["CB", "CB", "CB"], ["GK"]]
    }
    
    lines = formation_layouts.get(formation, [])
    player_dict = {p['position']: p for p in selected_players}
    
    for line_idx, line in enumerate(lines):
        cols = st.columns(len(line))
        for i, pos in enumerate(line):
            with cols[i]:
                if pos in player_dict:
                    player_info = player_dict[pos]
                    player = player_info['player']
                    
                    # Carte joueur avancée
                    card_color = get_card_color(player['overall_rating'])
                    st.markdown(f"""
                    <div class="player-card" style="background: {card_color};">
                        <strong>{str(player['name'])[:15] if 'name' in player else 'Unknown'}</strong><br>
                        <small>{pos} | {player['overall_rating']} OVR</small><br>
                        <small>€{player_info['cost']:.1f}M | {player.get('age', 'N/A')} ans</small><br>
                        <small>{str(player.get('nationality', 'Unknown'))[:3]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px dashed #ccc; border-radius: 10px; margin: 5px;">
                        <strong>-</strong><br><small>{pos}</small>
                    </div>
                    """, unsafe_allow_html=True)

def get_card_color(overall):
    """Couleur de carte selon l'overall"""
    if overall >= 90: return "linear-gradient(135deg, #FFD700, #FFA500)"  # Or
    elif overall >= 85: return "linear-gradient(135deg, #C0C0C0, #808080)"  # Argent
    elif overall >= 80: return "linear-gradient(135deg, #CD7F32, #8B4513)"  # Bronze
    elif overall >= 75: return "linear-gradient(135deg, #4CAF50, #45a049)"  # Vert
    else: return "linear-gradient(135deg, #666, #444)"  # Gris

def generate_export_data(selected_players, team_stats, formation):
    """Génère les données d'export - VERSION CORRIGÉE"""
    
    # Fonction helper pour convertir les types numpy/pandas en types Python natifs
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    export_data = {
        "formation": formation,
        "team_stats": {},
        "players": [],
        "export_date": datetime.now().isoformat()
    }
    
    # Conversion des stats d'équipe
    for key, value in team_stats.items():
        export_data["team_stats"][key] = convert_to_serializable(value)
    
    # Conversion des données joueurs
    for p in selected_players:
        player = p['player']
        player_data = {
            "name": str(player.get('name', 'Unknown')) if not pd.isna(player.get('name')) else "Unknown",
            "position": str(p['position']),
            "overall": convert_to_serializable(player['overall_rating']),
            "potential": convert_to_serializable(player.get('potential', 0)),
            "age": convert_to_serializable(player.get('age', 0)),
            "nationality": str(player.get('nationality', '')) if not pd.isna(player.get('nationality')) else "Unknown",
            "club": str(player.get('club_name', '')) if not pd.isna(player.get('club_name')) else "Unknown",
            "value": convert_to_serializable(p['cost'])
        }
        export_data["players"].append(player_data)
    
    return export_data

def main():
    # Upload du fichier
    uploaded_file = st.file_uploader("📁 **Chargez votre base de données FC25**", type=['csv'])
    
    st.markdown('<div class="info-box">🔍 <strong>Comment ça marche:</strong> Téléchargez un fichier CSV contenant les données des joueurs FC25. Le fichier doit contenir au minimum les colonnes: name, overall_rating, positions.</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ **{len(df):,} joueurs chargés avec succès !**")
            
            # Interface principale avec tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎮 **Constructeur**", 
                "🔍 **Recherche Avancée**", 
                "👥 **Joueurs Similaires**", 
                "📊 **Analytics**", 
                "⚔️ **Comparaison**", 
                "📤 **Export**"
            ])
            
            with tab1:
                st.markdown('<div class="info-box">🎯 <strong>Constructeur d\'équipe:</strong> Configurez vos critères et laissez l\'IA optimiser votre équipe selon la formation et le mode de jeu choisis. Ajustez les poids pour privilégier certains aspects.</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 🎯 **Configuration**")
                    
                    # Mode de jeu
                    selected_mode = st.selectbox(
                        "🎮 **Mode de jeu**", 
                        list(GAME_MODES.keys()),
                        help="Chaque mode a ses propres contraintes et objectifs"
                    )
                    
                    mode_info = GAME_MODES[selected_mode]
                    st.info(f"📋 {mode_info['description']}")
                    
                    # Budget
                    base_budget = st.number_input(
                        "💰 **Budget de base (millions €)**", 
                        min_value=10, max_value=5000, value=500, step=25
                    )
                    
                    final_budget = base_budget * mode_info["budget_multiplier"]
                    st.metric("💳 Budget final", f"€{final_budget:.0f}M")
                    
                    # Formation
                    formation = st.selectbox("📋 **Formation tactique**", list(FORMATIONS.keys()))
                    
                    st.markdown('<div class="info-box">⚖️ <strong>Poids d\'optimisation:</strong> Ajustez ces curseurs pour privilégier certains critères. Plus le poids est élevé, plus ce critère sera important dans la sélection.</div>', unsafe_allow_html=True)
                    
                    # Poids d'optimisation
                    st.markdown("### ⚖️ **Critères d'optimisation**")
                    
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        weight_overall = st.slider("⭐ Overall", 0, 100, 40, 5)
                        weight_potential = st.slider("🌟 Potentiel", 0, 100, 30, 5)
                    with col_w2:
                        weight_age = st.slider("👶 Jeunesse", 0, 100, 20, 5)
                        weight_efficiency = st.slider("💰 Efficacité", 0, 100, 10, 5)
                    
                    optimization_weights = {
                        'overall': weight_overall,
                        'potential': weight_potential,
                        'age': weight_age,
                        'efficiency': weight_efficiency
                    }
                    
                    # Filtres avancés
                    with st.expander("🔍 **Filtres avancés**"):
                        st.markdown('<div class="info-box">🎛️ <strong>Filtres:</strong> Affinez votre recherche en limitant par âge, championnat ou nationalité. Laissez vide pour aucune restriction.</div>', unsafe_allow_html=True)
                        
                        # Âge
                        age_range = st.slider("🎂 Âge", 16, 40, (18, 35))
                        
                        # Ligues
                        if 'league_name' in df.columns:
                            leagues = st.multiselect(
                                "🏆 Championnats",
                                options=sorted(df['league_name'].dropna().unique()),
                                default=[]
                            )
                        else:
                            leagues = []
                        
                        # Nationalités
                        if 'nationality' in df.columns:
                            nationalities = st.multiselect(
                                "🌍 Nationalités",
                                options=sorted(df['nationality'].dropna().unique()),
                                default=[]
                            )
                        else:
                            nationalities = []
                    
                    filters = {
                        'age_range': age_range,
                        'leagues': leagues,
                        'nationalities': nationalities
                    }
                    
                    # Bouton d'optimisation
                    if st.button("🚀 **OPTIMISER L'ÉQUIPE**", type="primary", use_container_width=True):
                        with st.spinner("🔄 Optimisation en cours..."):
                            selected_players, remaining_budget = optimize_team_advanced(
                                df, formation, final_budget, selected_mode, optimization_weights, filters
                            )
                            
                            if selected_players:
                                team_stats = calculate_team_stats(selected_players, formation)
                                suggestions = analyze_team_balance(selected_players)
                                
                                st.session_state.update({
                                    'team': selected_players,
                                    'remaining_budget': remaining_budget,
                                    'formation': formation,
                                    'total_budget': final_budget,
                                    'team_stats': team_stats,
                                    'suggestions': suggestions,
                                    'mode': selected_mode
                                })
                                st.success("✅ **Équipe optimisée avec succès !**")
                            else:
                                st.error("❌ **Impossible de créer une équipe avec ces contraintes**")
                
                with col2:
                    # Affichage des résultats
                    if 'team' in st.session_state:
                        team = st.session_state['team']
                        team_stats = st.session_state['team_stats']
                        
                        # Métriques avancées
                        st.markdown("### 📊 **Statistiques d'équipe**")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric("💰 Coût", f"€{sum(p['cost'] for p in team):.0f}M", 
                                    f"€{st.session_state['remaining_budget']:.0f}M restant")
                        with col_m2:
                            st.metric("⭐ Overall", f"{team_stats['overall']:.1f}", 
                                    f"{team_stats['potential']:.1f} pot.")
                        with col_m3:
                            st.metric("🧪 Chimie", f"{team_stats['chemistry']:.0f}%", 
                                    f"{team_stats['age']:.1f} ans moy.")
                        with col_m4:
                            st.metric("⚔️ Attaque", f"{team_stats['attack']:.0f}", 
                                    f"{team_stats['defense']:.0f} déf.")
                        
                        # Graphique radar des stats
                        fig_radar = go.Figure()
                        
                        categories = ['Attaque', 'Défense', 'Créativité', 'Chimie', 'Expérience']
                        values = [
                            team_stats['attack'],
                            team_stats['defense'], 
                            team_stats['creativity'],
                            team_stats['chemistry'],
                            team_stats['experience']
                        ]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Votre équipe',
                            line_color='rgb(255, 107, 53)'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100])
                            ),
                            title="📈 Profil de l'équipe",
                            height=400
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Formation tactique
                        display_advanced_formation(team, st.session_state['formation'])
                        
                        # Suggestions d'amélioration
                        if st.session_state.get('suggestions'):
                            st.markdown("### 💡 **Suggestions d'amélioration**")
                            for suggestion in st.session_state['suggestions']:
                                st.info(suggestion)
                    else:
                        st.info("🎯 Configurez vos critères et cliquez sur 'OPTIMISER L'ÉQUIPE' pour commencer !")
            
            with tab2:
                st.markdown('<div class="info-box">🔍 <strong>Recherche Avancée:</strong> Trouvez exactement les joueurs que vous cherchez avec des critères précis. Vous pouvez rechercher de 1 à 11 joueurs selon vos besoins.</div>', unsafe_allow_html=True)
                
                st.markdown("### 🔍 **Recherche Avancée de Joueurs**")
                
                col_search1, col_search2 = st.columns([1, 2])
                
                with col_search1:
                    st.markdown("#### 🎛️ **Critères de recherche**")
                    
                    # Nombre de joueurs à chercher
                    max_players = st.slider("👥 Nombre de joueurs", 1, 11, 5)
                    
                    # Positions
                    if 'positions' in df.columns:
                        all_positions = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "LM", "RM", "CAM", "LW", "RW", "ST"]
                        selected_positions = st.multiselect(
                            "📍 Positions", 
                            options=all_positions,
                            default=[]
                        )
                    else:
                        selected_positions = []
                    
                    # Overall rating
                    col_overall1, col_overall2 = st.columns(2)
                    with col_overall1:
                        min_overall = st.number_input("⭐ Overall min", 40, 99, 70)
                    with col_overall2:
                        max_overall = st.number_input("⭐ Overall max", 40, 99, 95)
                    
                    # Âge
                    age_min, age_max = st.slider("🎂 Âge", 16, 40, (18, 35), key="search_age")
                    
                    # Budget max
                    if 'value_numeric' in df.columns:
                        max_value = st.number_input("💰 Valeur max (M€)", 0.0, 200.0, 50.0, 5.0)
                    else:
                        max_value = None
                    
                    # Autres filtres
                    if 'league_name' in df.columns:
                        search_leagues = st.multiselect(
                            "🏆 Championnats",
                            options=sorted(df['league_name'].dropna().unique()),
                            default=[]
                        )
                    else:
                        search_leagues = []
                    
                    if 'nationality' in df.columns:
                        search_nationalities = st.multiselect(
                            "🌍 Nationalités",
                            options=sorted(df['nationality'].dropna().unique()),
                            default=[]
                        )
                    else:
                        search_nationalities = []
                    
                    if 'club_name' in df.columns:
                        search_clubs = st.multiselect(
                            "🏟️ Clubs",
                            options=sorted(df['club_name'].dropna().unique()),
                            default=[]
                        )
                    else:
                        search_clubs = []
                    
                    # Bouton de recherche
                    if st.button("🔍 **RECHERCHER**", type="primary", use_container_width=True):
                        search_filters = {
                            'positions': selected_positions,
                            'min_overall': min_overall,
                            'max_overall': max_overall,
                            'age_range': (age_min, age_max),
                            'max_value': max_value,
                            'leagues': search_leagues,
                            'nationalities': search_nationalities,
                            'clubs': search_clubs,
                            'max_players': max_players
                        }
                        
                        with st.spinner("🔄 Recherche en cours..."):
                            search_results = search_players_advanced(df, search_filters)
                            st.session_state['search_results'] = search_results
                
                with col_search2:
                    if 'search_results' in st.session_state:
                        results = st.session_state['search_results']
                        
                        st.markdown(f"### 📊 **Résultats ({len(results)} joueurs)**")
                        
                        if not results.empty:
                            # Tableau des résultats
                            display_columns = ['name', 'overall_rating', 'age']
                            
                            if 'positions' in results.columns:
                                display_columns.append('positions')
                            if 'nationality' in results.columns:
                                display_columns.append('nationality')
                            if 'club_name' in results.columns:
                                display_columns.append('club_name')
                            if 'value_numeric' in results.columns:
                                display_columns.append('value_numeric')
                            
                            # Préparer les données pour l'affichage
                            display_data = results[display_columns].copy()
                            
                            # Renommer les colonnes pour l'affichage
                            column_names = {
                                'name': 'Nom',
                                'overall_rating': 'Overall',
                                'age': 'Âge',
                                'positions': 'Positions',
                                'nationality': 'Nationalité',
                                'club_name': 'Club',
                                'value_numeric': 'Valeur (M€)'
                            }
                            
                            display_data = display_data.rename(columns=column_names)
                            
                            st.dataframe(display_data, use_container_width=True, height=400)
                            
                            # Graphiques des résultats
                            col_graph1, col_graph2 = st.columns(2)
                            
                            with col_graph1:
                                # Distribution Overall
                                fig_overall = px.histogram(
                                    results, 
                                    x='overall_rating', 
                                    nbins=20,
                                    title="📊 Distribution Overall",
                                    labels={'overall_rating': 'Overall', 'count': 'Nombre'}
                                )
                                st.plotly_chart(fig_overall, use_container_width=True)
                            
                            with col_graph2:
                                # Distribution âge
                                fig_age = px.histogram(
                                    results, 
                                    x='age', 
                                    nbins=15,
                                    title="📊 Distribution Âge",
                                    labels={'age': 'Âge', 'count': 'Nombre'}
                                )
                                st.plotly_chart(fig_age, use_container_width=True)
                            
                            # Top 5 joueurs
                            st.markdown("#### 🌟 **Top 5 joueurs**")
                            top_players = results.nlargest(5, 'overall_rating')
                            
                            for i, (idx, player) in enumerate(top_players.iterrows()):
                                col_top = st.columns([1, 3, 1, 1, 1])
                                
                                with col_top[0]:
                                    st.write(f"**#{i+1}**")
                                with col_top[1]:
                                    st.write(f"**{player['name']}**")
                                with col_top[2]:
                                    st.write(f"{player['overall_rating']} OVR")
                                with col_top[3]:
                                    st.write(f"{player['age']} ans")
                                with col_top[4]:
                                    if 'value_numeric' in player:
                                        st.write(f"€{player['value_numeric']:.1f}M")
                        else:
                            st.warning("❌ Aucun joueur trouvé avec ces critères")
                    else:
                        st.info("🔍 Configurez vos critères et lancez une recherche !")
            
            with tab3:
                st.markdown('<div class="info-box">👥 <strong>Joueurs Similaires:</strong> Trouvez des joueurs ayant des caractéristiques similaires à un joueur de référence. L\'algorithme compare les stats, l\'âge et les capacités.</div>', unsafe_allow_html=True)
                
                st.markdown("### 👥 **Recherche de Joueurs Similaires**")
                
                col_sim1, col_sim2 = st.columns([1, 2])
                
                with col_sim1:
                    st.markdown("#### 🎯 **Joueur de référence**")
                    
                    # Recherche du joueur de référence
                    if 'name' in df.columns:
                        player_names = df['name'].dropna().tolist()
                        selected_player_name = st.selectbox(
                            "🔍 Chercher un joueur",
                            options=[''] + sorted(player_names),
                            format_func=lambda x: x if x else "-- Sélectionnez un joueur --"
                        )
                        
                        if selected_player_name:
                            reference_player = df[df['name'] == selected_player_name].iloc[0]
                            
                            # Affichage du joueur de référence
                            st.markdown("#### 📋 **Profil du joueur**")
                            
                            card_color = get_card_color(reference_player['overall_rating'])
                            st.markdown(f"""
                            <div class="player-card" style="background: {card_color};">
                                <strong>{reference_player['name']}</strong><br>
                                <small>{reference_player.get('positions', 'N/A')} | {reference_player['overall_rating']} OVR</small><br>
                                <small>{reference_player.get('age', 'N/A')} ans</small><br>
                                <small>{reference_player.get('nationality', 'N/A')}</small><br>
                                <small>{reference_player.get('club_name', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Paramètres de recherche
                            st.markdown("#### ⚙️ **Paramètres**")
                            
                            num_similar = st.slider("📊 Nombre de joueurs similaires", 5, 20, 10)
                            
                            # Filtres additionnels
                            with st.expander("🔍 Filtres additionnels"):
                                max_age_diff = st.slider("📅 Écart d'âge max", 1, 10, 3)
                                min_similarity = st.slider("🎯 Similarité min (%)", 50, 95, 70)
                                exclude_same_club = st.checkbox("🚫 Exclure le même club", True)
                            
                            if st.button("🔍 **TROUVER DES JOUEURS SIMILAIRES**", type="primary", use_container_width=True):
                                with st.spinner("🔄 Analyse des similarités..."):
                                    similar_players = find_similar_players(reference_player, df, num_similar * 2)
                                    
                                    # Application des filtres
                                    filtered_similar = []
                                    for sim_player in similar_players:
                                        player = sim_player['player']
                                        similarity = sim_player['similarity']
                                        
                                        # Filtres
                                        if similarity < min_similarity:
                                            continue
                                        
                                        age_diff = abs(player.get('age', 25) - reference_player.get('age', 25))
                                        if age_diff > max_age_diff:
                                            continue
                                        
                                        if exclude_same_club and player.get('club_name') == reference_player.get('club_name'):
                                            continue
                                        
                                        filtered_similar.append(sim_player)
                                    
                                    # Limiter au nombre demandé
                                    filtered_similar = filtered_similar[:num_similar]
                                    
                                    st.session_state['similar_players'] = filtered_similar
                                    st.session_state['reference_player'] = reference_player
                    else:
                        st.warning("❌ Colonne 'name' non trouvée dans les données")
                
                with col_sim2:
                    if 'similar_players' in st.session_state and 'reference_player' in st.session_state:
                        similar_players = st.session_state['similar_players']
                        reference_player = st.session_state['reference_player']
                        
                        st.markdown(f"### 🎯 **Joueurs similaires à {reference_player['name']}**")
                        
                        if similar_players:
                            # Tableau des joueurs similaires
                            similar_data = []
                            for sim_player in similar_players:
                                player = sim_player['player']
                                similar_data.append({
                                    'Nom': player.get('name', 'Unknown'),
                                    'Overall': player['overall_rating'],
                                    'Âge': player.get('age', 'N/A'),
                                    'Position': player.get('positions', 'N/A'),
                                    'Club': player.get('club_name', 'N/A'),
                                    'Nationalité': player.get('nationality', 'N/A'),
                                    'Similarité %': f"{sim_player['similarity']:.1f}%",
                                    'Valeur M€': f"{player.get('value_numeric', 0):.1f}"
                                })
                            
                            similar_df = pd.DataFrame(similar_data)
                            st.dataframe(similar_df, use_container_width=True, height=400)
                            
                            # Graphique de similarité
                            similarities = [s['similarity'] for s in similar_players]
                            names = [s['player'].get('name', 'Unknown') for s in similar_players]
                            
                            fig_sim = px.bar(
                                x=names,
                                y=similarities,
                                title="📊 Niveau de similarité",
                                labels={'x': 'Joueurs', 'y': 'Similarité (%)'}
                            )
                            fig_sim.update_xaxis(tickangle=45)
                            st.plotly_chart(fig_sim, use_container_width=True)
                            
                            # Comparaison détaillée Top 3
                            st.markdown("#### 🏆 **Top 3 - Comparaison détaillée**")
                            
                            top_3 = similar_players[:3]
                            comparison_cols = st.columns(4)
                            
                            # Joueur de référence
                            with comparison_cols[0]:
                                st.markdown("**🎯 Référence**")
                                ref_card_color = get_card_color(reference_player['overall_rating'])
                                st.markdown(f"""
                                <div class="player-card" style="background: {ref_card_color};">
                                    <strong>{reference_player['name'][:12]}</strong><br>
                                    <small>{reference_player['overall_rating']} OVR</small><br>
                                    <small>{reference_player.get('age', 'N/A')} ans</small><br>
                                    <small>{reference_player.get('club_name', 'N/A')[:10]}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Top 3 similaires
                            for i, sim_player in enumerate(top_3):
                                with comparison_cols[i + 1]:
                                    player = sim_player['player']
                                    similarity = sim_player['similarity']
                                    
                                    st.markdown(f"**#{i+1} - {similarity:.1f}%**")
                                    sim_card_color = get_card_color(player['overall_rating'])
                                    st.markdown(f"""
                                    <div class="player-card" style="background: {sim_card_color};">
                                        <strong>{str(player.get('name', 'Unknown'))[:12]}</strong><br>
                                        <small>{player['overall_rating']} OVR</small><br>
                                        <small>{player.get('age', 'N/A')} ans</small><br>
                                        <small>{str(player.get('club_name', 'N/A'))[:10]}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("❌ Aucun joueur similaire trouvé avec ces critères")
                    else:
                        st.info("🎯 Sélectionnez un joueur de référence pour commencer l'analyse !")
            
            with tab4:
                st.markdown('<div class="info-box">📊 <strong>Analytics:</strong> Analysez en détail votre équipe avec des statistiques avancées, graphiques et comparaisons par rapport à la moyenne du championnat.</div>', unsafe_allow_html=True)
                
                if 'team' in st.session_state:
                    team = st.session_state['team']
                    
                    st.markdown("### 📊 **Analytics avancées**")
                    
                    # Tableau détaillé
                    team_data = []
                    for p in team:
                        player = p['player']
                        team_data.append({
                            'Position': p['position'],
                            'Nom': str(player.get('name', 'Unknown')),
                            'Club': str(player.get('club_name', 'N/A')),
                            'Overall': player['overall_rating'],
                            'Potentiel': player.get('potential', 'N/A'),
                            'Âge': player.get('age', 'N/A'),
                            'Nationalité': str(player.get('nationality', 'N/A')),
                            'Valeur €M': f"{p['cost']:.1f}",
                            'Efficacité': f"{player.get('efficiency_score', 0):.2f}"
                        })
                    
                    team_df = pd.DataFrame(team_data)
                    st.dataframe(team_df, use_container_width=True, height=400)
                    
                    # Graphiques analytiques
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Distribution des âges
                        ages = [p['player'].get('age', 25) for p in team]
                        fig_age = px.histogram(
                            x=ages, 
                            nbins=10, 
                            title="📊 Distribution des âges",
                            labels={'x': 'Âge', 'y': 'Nombre de joueurs'}
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
                        
                        # Répartition par nationalité
                        nationalities = [p['player'].get('nationality', 'Unknown') for p in team]
                        nat_counts = pd.Series(nationalities).value_counts()
                        
                        fig_nat = px.pie(
                            values=nat_counts.values, 
                            names=nat_counts.index,
                            title="🌍 Répartition des nationalités"
                        )
                        st.plotly_chart(fig_nat, use_container_width=True)
                    
                    with col_g2:
                        # Overall par position
                        positions = [p['position'] for p in team]
                        overalls = [p['player']['overall_rating'] for p in team]
                        
                        fig_pos = px.bar(
                            x=positions, 
                            y=overalls,
                            title="⭐ Overall par position",
                            labels={'x': 'Position', 'y': 'Overall'}
                        )
                        st.plotly_chart(fig_pos, use_container_width=True)
                        
                        # Analyse de valeur
                        costs = [p['cost'] for p in team]
                        fig_value = px.bar(
                            x=positions,
                            y=costs,
                            title="💰 Coût par position",
                            labels={'x': 'Position', 'y': 'Valeur (M€)'}
                        )
                        st.plotly_chart(fig_value, use_container_width=True)
                    
                    # Analyse comparative
                    st.markdown("### 📈 **Analyse comparative**")
                    
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    
                    with col_comp1:
                        avg_overall_league = df['overall_rating'].mean()
                        team_avg_overall = np.mean([p['player']['overall_rating'] for p in team])
                        diff_overall = team_avg_overall - avg_overall_league
                        st.metric(
                            "📊 Niveau vs moyenne", 
                            f"{team_avg_overall:.1f}",
                            f"{diff_overall:+.1f} pts"
                        )
                    
                    with col_comp2:
                        avg_age_league = df['age'].mean()
                        team_avg_age = np.mean([p['player'].get('age', 25) for p in team])
                        diff_age = team_avg_age - avg_age_league
                        st.metric(
                            "👶 Âge vs moyenne",
                            f"{team_avg_age:.1f} ans",
                            f"{diff_age:+.1f} ans"
                        )
                    
                    with col_comp3:
                        total_value = sum(p['cost'] for p in team)
                        value_per_point = total_value / team_avg_overall
                        st.metric(
                            "💰 Coût par point",
                            f"€{value_per_point:.1f}M",
                            "Efficacité"
                        )
                    
                    # Heatmap des stats par position (si disponible)
                    if any(col in df.columns for col in ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical']):
                        st.markdown("### 🔥 **Heatmap des compétences**")
                        
                        stat_columns = []
                        for stat in ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical']:
                            if stat in df.columns:
                                stat_columns.append(stat)
                        
                        if stat_columns:
                            heatmap_data = []
                            for p in team:
                                player = p['player']
                                row = [p['position']]
                                for stat in stat_columns:
                                    row.append(player.get(stat, 50))
                                heatmap_data.append(row)
                            
                            heatmap_df = pd.DataFrame(heatmap_data, columns=['Position'] + stat_columns)
                            
                            # Créer heatmap avec plotly
                            fig_heatmap = px.imshow(
                                heatmap_df[stat_columns].values,
                                labels=dict(x="Compétences", y="Joueurs", color="Niveau"),
                                x=stat_columns,
                                y=heatmap_df['Position'],
                                title="🎯 Profil des compétences par position"
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("🎮 Créez d'abord une équipe dans l'onglet Constructeur !")
            
            with tab5:
                st.markdown('<div class="info-box">⚔️ <strong>Comparaison:</strong> Sauvegardez plusieurs équipes et comparez-les directement. Analysez les différences de performance, coût et style de jeu.</div>', unsafe_allow_html=True)
                
                st.markdown("### ⚔️ **Comparaison d'équipes**")
                
                # Sauvegarde d'équipes
                if 'team' in st.session_state:
                    team_name = st.text_input("💾 Nom de l'équipe à sauvegarder")
                    if st.button("💾 Sauvegarder cette équipe"):
                        if team_name:
                            if 'saved_teams' not in st.session_state:
                                st.session_state['saved_teams'] = {}
                            
                            st.session_state['saved_teams'][team_name] = {
                                'team': st.session_state['team'],
                                'stats': st.session_state['team_stats'],
                                'formation': st.session_state['formation'],
                                'mode': st.session_state['mode']
                            }
                            st.success(f"✅ Équipe '{team_name}' sauvegardée !")
                        else:
                            st.warning("⚠️ Veuillez entrer un nom pour l'équipe")
                
                # Affichage des équipes sauvegardées
                if st.session_state.get('saved_teams'):
                    st.markdown("#### 📚 **Équipes sauvegardées**")
                    
                    teams_to_compare = st.multiselect(
                        "Sélectionnez les équipes à comparer",
                        options=list(st.session_state['saved_teams'].keys()),
                        default=list(st.session_state['saved_teams'].keys())[:2]
                    )
                    
                    if len(teams_to_compare) >= 2:
                        # Tableau de comparaison
                        comparison_data = []
                        
                        for team_name in teams_to_compare:
                            team_data = st.session_state['saved_teams'][team_name]
                            stats = team_data['stats']
                            
                            comparison_data.append({
                                'Équipe': team_name,
                                'Formation': team_data['formation'],
                                'Mode': team_data['mode'],
                                'Overall': f"{stats['overall']:.1f}",
                                'Potentiel': f"{stats['potential']:.1f}",
                                'Âge moyen': f"{stats['age']:.1f}",
                                'Attaque': f"{stats['attack']:.0f}",
                                'Défense': f"{stats['defense']:.0f}",
                                'Chimie': f"{stats['chemistry']:.0f}%",
                                'Coût total': f"€{sum(p['cost'] for p in team_data['team']):.0f}M"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Graphique comparatif radar
                        fig_comp = go.Figure()
                        
                        categories = ['Overall', 'Attaque', 'Défense', 'Chimie', 'Créativité']
                        
                        colors = ['rgb(255, 107, 53)', 'rgb(53, 107, 255)', 'rgb(107, 255, 53)', 'rgb(255, 53, 107)']
                        
                        for i, team_name in enumerate(teams_to_compare[:4]):  # Max 4 équipes
                            team_data = st.session_state['saved_teams'][team_name]
                            stats = team_data['stats']
                            
                            values = [
                                stats['overall'],
                                stats['attack'],
                                stats['defense'],
                                stats['chemistry'],
                                stats['creativity']
                            ]
                            
                            fig_comp.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=team_name,
                                line_color=colors[i % len(colors)]
                            ))
                        
                        fig_comp.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            title="📊 Comparaison des équipes",
                            height=500
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Recommandation
                        if len(teams_to_compare) == 2:
                            team1_stats = st.session_state['saved_teams'][teams_to_compare[0]]['stats']
                            team2_stats = st.session_state['saved_teams'][teams_to_compare[1]]['stats']
                            
                            winner_categories = []
                            for category in ['overall', 'attack', 'defense', 'chemistry']:
                                if team1_stats[category] > team2_stats[category]:
                                    winner_categories.append(f"✅ {teams_to_compare[0]} domine en {category}")
                                elif team2_stats[category] > team1_stats[category]:
                                    winner_categories.append(f"✅ {teams_to_compare[1]} domine en {category}")
                            
                            st.markdown("#### 🏆 **Analyse comparative**")
                            for category in winner_categories:
                                st.write(category)
                                
                        # Analyse des différences de coût
                        st.markdown("#### 💰 **Analyse des coûts**")
                        
                        cost_data = []
                        for team_name in teams_to_compare:
                            team_data = st.session_state['saved_teams'][team_name]
                            total_cost = sum(p['cost'] for p in team_data['team'])
                            cost_data.append({'Équipe': team_name, 'Coût': total_cost})
                        
                        cost_df = pd.DataFrame(cost_data)
                        
                        fig_cost = px.bar(
                            cost_df,
                            x='Équipe',
                            y='Coût',
                            title="💰 Comparaison des coûts",
                            labels={'Coût': 'Coût total (M€)'}
                        )
                        st.plotly_chart(fig_cost, use_container_width=True)
                        
                else:
                    st.info("💾 Sauvegardez d'abord des équipes pour les comparer !")
            
            with tab6:
                st.markdown('<div class="info-box">📤 <strong>Export:</strong> Exportez votre équipe en différents formats (CSV, JSON) ou générez un résumé pour le partager sur les réseaux sociaux.</div>', unsafe_allow_html=True)
                
                st.markdown("### 📤 **Export et partage**")
                
                if 'team' in st.session_state:
                    team = st.session_state['team']
                    team_stats = st.session_state['team_stats']
                    formation = st.session_state['formation']
                    
                    # Export CSV et JSON
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        st.markdown("#### 📊 **Export CSV**")
                        
                        # Création des données CSV
                        csv_data = []
                        for p in team:
                            player = p['player']
                            
                            # Fonction de conversion sécurisée
                            def safe_convert(value, default='N/A'):
                                if pd.isna(value):
                                    return default
                                if isinstance(value, (np.integer, np.int64)):
                                    return int(value)
                                elif isinstance(value, (np.floating, np.float64)):
                                    return float(value)
                                else:
                                    return str(value)
                            
                            csv_data.append({
                                'Nom': safe_convert(player.get('name'), 'Unknown'),
                                'Position': safe_convert(p['position']),
                                'Overall': safe_convert(player['overall_rating'], 0),
                                'Potentiel': safe_convert(player.get('potential'), 0),
                                'Age': safe_convert(player.get('age'), 0),
                                'Nationalité': safe_convert(player.get('nationality'), 'Unknown'),
                                'Club': safe_convert(player.get('club_name'), 'Unknown'),
                                'Valeur_Millions': safe_convert(p['cost'], 0),
                                'Formation': formation
                            })
                        
                        csv_df = pd.DataFrame(csv_data)
                        csv_string = csv_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv_string,
                            file_name=f"equipe_fc25_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown('<div class="info-box">📊 Le fichier CSV contient tous les détails des joueurs et peut être ouvert dans Excel ou Google Sheets.</div>', unsafe_allow_html=True)
                    
                    with col_exp2:
                        st.markdown("#### 📋 **Export JSON**")
                        
                        export_data = generate_export_data(team, team_stats, formation)
                        json_string = json.dumps(export_data, indent=2, ensure_ascii=False)
                        
                        st.download_button(
                            label="📥 Télécharger JSON",
                            data=json_string,
                            file_name=f"equipe_fc25_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                        
                        st.markdown('<div class="info-box">📋 Le fichier JSON contient les données structurées et peut être importé dans d\'autres applications.</div>', unsafe_allow_html=True)
                    
                    # Résumé pour partage
                    st.markdown("#### 🔗 **Résumé pour partage**")
                    
                    share_text = f"""🏆 **MON ÉQUIPE FC25 ULTIMATE**

📋 **Formation:** {formation}
🎮 **Mode:** {st.session_state.get('mode', 'Standard')}
💰 **Budget utilisé:** €{sum(p['cost'] for p in team):.0f}M

📊 **Statistiques:**
⭐ Overall moyen: {team_stats['overall']:.1f}
⚔️ Attaque: {team_stats['attack']:.0f}/100
🛡️ Défense: {team_stats['defense']:.0f}/100
🧪 Chimie: {team_stats['chemistry']:.0f}%

👥 **Titulaires:**"""
                    
                    for p in team:
                        player_name = str(p['player'].get('name', 'Unknown'))
                        share_text += f"\n• {p['position']}: {player_name} ({p['player']['overall_rating']} OVR)"
                    
                    share_text += f"\n\n🔧 **Créé avec FC25 Ultimate Team Builder Pro**"
                    
                    st.text_area(
                        "Copier pour partager sur les réseaux sociaux:",
                        share_text,
                        height=300
                    )
                    
                    st.markdown('<div class="info-box">🔗 Copiez ce texte pour le partager sur Twitter, Discord, ou tout autre réseau social !</div>', unsafe_allow_html=True)
                    
                    # Statistiques détaillées
                    st.markdown("#### 📈 **Rapport détaillé**")
                    
                    with st.expander("📊 Voir le rapport complet"):
                        st.markdown(f"""
**RAPPORT D'ANALYSE D'ÉQUIPE**

**Configuration:**
- Formation: {formation}
- Mode de jeu: {st.session_state.get('mode', 'Standard')}  
- Budget total: €{st.session_state.get('total_budget', 0):.0f}M
- Budget utilisé: €{sum(p['cost'] for p in team):.0f}M
- Budget restant: €{st.session_state.get('remaining_budget', 0):.0f}M

**Statistiques d'équipe:**
- Overall moyen: {team_stats['overall']:.2f}
- Potentiel moyen: {team_stats['potential']:.2f}
- Âge moyen: {team_stats['age']:.1f} ans
- Puissance offensive: {team_stats['attack']:.0f}/100
- Solidité défensive: {team_stats['defense']:.0f}/100
- Chimie d'équipe: {team_stats['chemistry']:.0f}%
- Créativité: {team_stats['creativity']:.0f}/100
- Expérience: {team_stats['experience']:.0f}/100

**Répartition budgétaire:**
- Coût moyen par joueur: €{sum(p['cost'] for p in team)/len(team):.1f}M
- Joueur le plus cher: €{max(p['cost'] for p in team):.1f}M
- Efficacité (Overall/€): {team_stats['overall']/(sum(p['cost'] for p in team)/len(team)):.2f}

**Analyse:**
{chr(10).join(st.session_state.get('suggestions', ['Équipe bien équilibrée !']))}
                        """)
                        
                    # Graphique final - Distribution des coûts
                    st.markdown("#### 📊 **Visualisation finale**")
                    
                    costs = [p['cost'] for p in team]
                    positions = [p['position'] for p in team]
                    names = [str(p['player'].get('name', 'Unknown')) for p in team]
                    
                    fig_final = px.treemap(
                        names=names,
                        values=costs,
                        parents=[f"Position {pos}" for pos in positions],
                        title="💰 Répartition du budget par joueur"
                    )
                    st.plotly_chart(fig_final, use_container_width=True)
                    
                else:
                    st.info("🎮 Créez d'abord une équipe pour l'exporter !")
            
            # Aperçu des données
            with st.expander("👀 **Base de données - Aperçu**"):
                st.markdown('<div class="info-box">👀 <strong>Aperçu des données:</strong> Consultez ici un échantillon de votre base de données pour vérifier que le chargement s\'est bien passé.</div>', unsafe_allow_html=True)
                
                st.markdown(f"**📊 {len(df):,} joueurs dans la base**")
                
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("⭐ Overall max", int(df['overall_rating'].max()))
                with col_info2: 
                    if 'nationality' in df.columns:
                        st.metric("🌍 Nationalités", len(df['nationality'].unique()))
                    else:
                        st.metric("🌍 Nationalités", "N/A")
                with col_info3:
                    if 'league_name' in df.columns:
                        st.metric("🏆 Ligues", len(df['league_name'].unique()))
                    else:
                        st.metric("🏆 Ligues", "N/A")
                with col_info4:
                    st.metric("👶 Âge moyen", f"{df['age'].mean():.1f} ans")
                
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.checkbox("🔍 Afficher toutes les colonnes"):
                    st.write("**Colonnes disponibles:**", list(df.columns))
                    
                # Statistiques de la base
                st.markdown("#### 📈 **Statistiques de la base**")
                
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    # Distribution Overall
                    fig_db_overall = px.histogram(
                        df, 
                        x='overall_rating',
                        nbins=30,
                        title="📊 Distribution Overall (base complète)",
                        labels={'overall_rating': 'Overall', 'count': 'Nombre de joueurs'}
                    )
                    st.plotly_chart(fig_db_overall, use_container_width=True)
                
                with col_stats2:
                    # Distribution âge
                    fig_db_age = px.histogram(
                        df,
                        x='age',
                        nbins=25,
                        title="📊 Distribution Âge (base complète)",
                        labels={'age': 'Âge', 'count': 'Nombre de joueurs'}
                    )
                    st.plotly_chart(fig_db_age, use_container_width=True)
        else:
            st.error("❌ Erreur lors du chargement des données. Vérifiez le format de votre fichier CSV.")
    else:
        # Instructions d'utilisation
        st.markdown("## 🚀 **Comment utiliser FC25 Ultimate Team Builder Pro**")
        
        st.markdown("""
        ### 📋 **Prérequis**
        
        Pour utiliser cette application, vous devez disposer d'un fichier CSV contenant les données des joueurs FC25.
        
        **Colonnes requises minimales :**
        - `name` : Nom du joueur
        - `overall_rating` : Note globale du joueur (0-99)
        - `positions` : Positions jouables (ex: "ST,CF" ou "CB,CDM")
        
        **Colonnes optionnelles recommandées :**
        - `age` ou `dob` : Âge ou date de naissance
        - `potential` : Potentiel du joueur
        - `value` : Valeur marchande (ex: "50M", "2.5K")
        - `wage` : Salaire (ex: "200K", "50K")
        - `nationality` : Nationalité
        - `club_name` : Club actuel
        - `league_name` : Championnat
        - Stats spécifiques : `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physical`
        
        ### 🎯 **Fonctionnalités principales**
        
        1. **🎮 Constructeur d'équipe** : Optimisation automatique selon vos critères
        2. **🔍 Recherche avancée** : Trouvez des joueurs spécifiques
        3. **👥 Joueurs similaires** : Découvrez des alternatives à vos joueurs favoris
        4. **📊 Analytics** : Analyses détaillées de votre équipe
        5. **⚔️ Comparaison** : Comparez plusieurs équipes sauvegardées
        6. **📤 Export** : Exportez vos créations en CSV, JSON ou format partage
        
        ### 🎮 **Modes de jeu disponibles**
        
        - **🚀 Ultimate Team** : Mode complet avec chimie et synergies
        - **💎 Chasse aux Pépites** : Focus sur les jeunes talents
        - **👑 Galactiques** : Les meilleurs joueurs absolus
        - **💰 Mercato Réaliste** : Budget incluant les salaires
        - **⚖️ Qualité/Prix** : Meilleur rapport performance/coût
        
        ### 📊 **Formations disponibles**
        
        - **4-3-3 (Attaque)** : +15 Attaque, +10 Créativité
        - **4-4-2 (Équilibré)** : +5 partout
        - **3-5-2 (Possession)** : +10 Défense, +15 Créativité
        - **4-2-3-1 (Créatif)** : +10 Attaque, +20 Créativité
        - **5-3-2 (Défense)** : +20 Défense, -5 Attaque
        - **3-4-3 (Intense)** : +20 Attaque, -5 Défense
        
        **👆 Chargez votre fichier CSV ci-dessus pour commencer !**
        """)
        
        # Exemple de format CSV
        st.markdown("### 📝 **Exemple de format CSV**")
        
        example_data = {
            'name': ['Kylian Mbappé', 'Erling Haaland', 'Pedri', 'Virgil van Dijk'],
            'overall_rating': [91, 88, 85, 89],
            'potential': [95, 94, 91, 89],
            'age': [24, 23, 21, 31],
            'positions': ['LW,ST,RW', 'ST', 'CM,CAM', 'CB'],
            'nationality': ['France', 'Norway', 'Spain', 'Netherlands'],
            'club_name': ['Paris Saint-Germain', 'Manchester City', 'FC Barcelona', 'Liverpool'],
            'league_name': ['Ligue 1', 'Premier League', 'La Liga', 'Premier League'],
            'value': ['180M', '150M', '80M', '40M'],
            'wage': ['250K', '200K', '100K', '180K']
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**⚽ FC25 Ultimate Team Builder Pro** - Créé avec ❤️ pour les passionnés de football")

if __name__ == "__main__":
    main() Nettoyage des colonnes de prix
        if 'value' in df.columns:
            df['value_clean'] = df['value'].astype(str).str.replace('€', '').str.replace(',', '')
            df['value_numeric'] = pd.to_numeric(
                df['value_clean'].str.replace('M', '').str.replace('K', ''), 
                errors='coerce'
            )
            # Conversion en millions
            mask_k = df['value_clean'].str.contains('K', na=False)
            df.loc[mask_k, 'value_numeric'] /= 1000
        
        if 'wage' in df.columns:
            df['wage_clean'] = df['wage'].astype(str).str.replace('€', '').str.replace(',', '')
            df['wage_numeric'] = pd.to_numeric(
                df['wage_clean'].str.replace('K', '').str.replace('M', ''), 
                errors='coerce'
            )
            mask_k = df['wage_clean'].str.contains('K', na=False)
            df.loc[mask_k, 'wage_numeric'] /= 1000
        
        # Calculs avancés
        if 'dob' in df.columns:
            df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
        else:
            df['age'] = np.random.randint(18, 35, len(df))  # Fallback
            
        if 'potential' in df.columns and 'overall_rating' in df.columns:
            df['potential_gap'] = df['potential'] - df['overall_rating']
        
        if 'value_numeric' in df.columns and 'overall_rating' in df.columns:
            df['value_per_overall'] = df['value_numeric'] / df['overall_rating']
            df['efficiency_score'] = df['overall_rating'] / np.log1p(df['value_numeric'])
        
        # ID unique si pas présent
        if 'player_id' not in df.columns:
            df['player_id'] = range(len(df))
            
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

def calculate_player_similarity(player1, player2, df):
    """Calcule la similarité entre deux joueurs"""
    try:
        # Colonnes pour la similarité (stats principales)
        stat_columns = ['overall_rating', 'potential', 'age']
        
        # Ajouter les stats spécifiques si disponibles
        possible_stats = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical',
                         'diving', 'handling', 'kicking', 'reflexes', 'speed', 'positioning']
        
        for stat in possible_stats:
            if stat in df.columns:
                stat_columns.append(stat)
        
        # Extraire les valeurs pour les deux joueurs
        player1_stats = []
        player2_stats = []
        
        for col in stat_columns:
            if col in player1 and col in player2:
                val1 = pd.to_numeric(player1[col], errors='coerce')
                val2 = pd.to_numeric(player2[col], errors='coerce')
                if not pd.isna(val1) and not pd.isna(val2):
                    player1_stats.append(val1)
                    player2_stats.append(val2)
        
        if len(player1_stats) < 3:  # Au minimum 3 stats pour calculer la similarité
            return 0
        
        # Normalisation et calcul de similarité cosinus
        scaler = StandardScaler()
        stats_matrix = scaler.fit_transform([player1_stats, player2_stats])
        similarity = cosine_similarity([stats_matrix[0]], [stats_matrix[1]])[0][0]
        
        # Convertir en pourcentage
        return max(0, similarity * 100)
        
    except Exception as e:
        return 0

def find_similar_players(reference_player, df, top_n=10):
    """Trouve les joueurs les plus similaires à un joueur de référence"""
    similarities = []
    
    for idx, player in df.iterrows():
        if player['player_id'] == reference_player['player_id']:
            continue
            
        similarity = calculate_player_similarity(reference_player, player, df)
        similarities.append({
            'player': player,
            'similarity': similarity
        })
    
    # Trier par similarité décroissante
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_n]

def calculate_team_chemistry(selected_players):
    """Calcule la chimie d'équipe basée sur nationalités et clubs"""
    if not selected_players:
        return 0
    
    nationalities = [p['player'].get('nationality', 'Unknown') for p in selected_players]
    clubs = [p['player'].get('club_name', 'Unknown') for p in selected_players]
    
    # Bonus nationalités communes
    nationality_counts = pd.Series(nationalities).value_counts()
    nationality_bonus = sum([count * 2 for count in nationality_counts if count > 1])
    
    # Bonus clubs communs
    club_counts = pd.Series(clubs).value_counts()
    club_bonus = sum([count * 3 for count in club_counts if count > 1])
    
    # Bonus formation
    base_chemistry = 50
    total_chemistry = min(100, base_chemistry + nationality_bonus + club_bonus)
    
    return total_chemistry

def calculate_team_stats(selected_players, formation):
    """Calcule les statistiques avancées de l'équipe"""
    if not selected_players:
        return {}
    
    players_data = [p['player'] for p in selected_players]
    
    # Stats de base
    avg_overall = np.mean([p['overall_rating'] for p in players_data])
    avg_potential = np.mean([p.get('potential', p['overall_rating']) for p in players_data])
    avg_age = np.mean([p.get('age', 25) for p in players_data])
    
    # Stats avancées
    attack_power = calculate_attack_power(selected_players)
    defense_power = calculate_defense_power(selected_players)
    chemistry = calculate_team_chemistry(selected_players)
    
    # Bonus de formation
    formation_bonus = FORMATIONS[formation]["bonus"]
    attack_final = attack_power + formation_bonus["attack"]
    defense_final = defense_power + formation_bonus["defense"]
    creativity = avg_overall * 0.8 + formation_bonus["creativity"]
    
    return {
        "overall": avg_overall,
        "potential": avg_potential,
        "age": avg_age,
        "attack": max(0, min(100, attack_final)),
        "defense": max(0, min(100, defense_final)),
        "chemistry": chemistry,
        "creativity": max(0, min(100, creativity)),
        "experience": min(100, avg_age * 3)
    }

def calculate_attack_power(selected_players):
    """Calcule la puissance offensive"""
    attack_positions = ["ST", "LW", "RW", "CAM", "CF", "LF", "RF"]
    attack_players = [p for p in selected_players if p['position'] in attack_positions]
    
    if not attack_players:
        return 30
    
    attack_overall = np.mean([p['player']['overall_rating'] for p in attack_players])
    return min(100, attack_overall * 1.2)

def calculate_defense_power(selected_players):
    """Calcule la puissance défensive"""
    defense_positions = ["GK", "CB", "LB", "RB", "CDM", "LWB", "RWB"]
    defense_players = [p for p in selected_players if p['position'] in defense_positions]
    
    if not defense_players:
        return 30
    
    defense_overall = np.mean([p['player']['overall_rating'] for p in defense_players])
    return min(100, defense_overall * 1.1)

def can_play_position(player_positions, required_position):
    """Vérifie si un joueur peut jouer à une position donnée"""
    if not player_positions or pd.isna(player_positions):
        return False
    
    player_pos_list = str(player_positions).split(',')
    compatible_positions = POSITION_COMPATIBILITY.get(required_position, [required_position])
    
    return any(pos.strip() in compatible_positions for pos in player_pos_list)

def get_players_for_position(df, position, exclude_ids=None, filters=None):
    """Récupère les joueurs avec filtres avancés"""
    exclude_ids = exclude_ids or []
    
    # Filtre de position
    if 'positions' in df.columns:
        mask = df['positions'].apply(lambda x: can_play_position(x, position))
    else:
        #
