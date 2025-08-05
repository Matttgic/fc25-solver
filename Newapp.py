import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from io import StringIO

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
        
        # Nettoyage des colonnes de prix
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
        df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
        df['potential_gap'] = df['potential'] - df['overall_rating']
        df['value_per_overall'] = df['value_numeric'] / df['overall_rating']
        df['efficiency_score'] = df['overall_rating'] / np.log1p(df['value_numeric'])
        
        # ID unique si pas présent
        if 'player_id' not in df.columns:
            df['player_id'] = range(len(df))
            
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

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
    mask = df['positions'].apply(lambda x: can_play_position(x, position))
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
            available_players = available_players[
                available_players['league_name'].isin(filters['leagues'])
            ]
        
        if 'nationalities' in filters and filters['nationalities']:
            available_players = available_players[
                available_players['nationality'].isin(filters['nationalities'])
            ]
    
    return available_players

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
        if 'potential' in player:
            score += player['potential'] * weights['potential'] / 100
        
        # Âge (inversé - plus jeune = mieux)
        age_score = max(0, 40 - player.get('age', 25)) / 40 * 100
        score += age_score * weights['age'] / 100
        
        # Efficacité prix
        if player['value_numeric'] > 0:
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
            if mode_data["constraints"].get("include_wages", False):
                # Budget sur 3 ans incluant salaires
                total_cost = available_players['value_numeric'] + (available_players['wage_numeric'] * 3)
                affordable_players = available_players[total_cost <= remaining_budget]
            else:
                affordable_players = available_players[available_players['value_numeric'] <= remaining_budget]
            
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
            if mode_data["constraints"].get("include_wages", False):
                real_cost = best_player['value_numeric'] + (best_player['wage_numeric'] * 3)
            else:
                real_cost = best_player['value_numeric']
            
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
                        <strong>{player['name'][:15]}</strong><br>
                        <small>{pos} | {player['overall_rating']} OVR</small><br>
                        <small>€{player_info['cost']:.1f}M | {player.get('age', 'N/A')} ans</small><br>
                        <small>{player.get('nationality', 'Unknown')[:3]}</small>
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
    """Génère les données d'export"""
    export_data = {
        "formation": formation,
        "team_stats": team_stats,
        "players": [],
        "export_date": datetime.now().isoformat()
    }
    
    for p in selected_players:
        player_data = {
            "name": p['player']['name'],
            "position": p['position'],
            "overall": p['player']['overall_rating'],
            "potential": p['player'].get('potential', 0),
            "age": p['player'].get('age', 0),
            "nationality": p['player'].get('nationality', ''),
            "club": p['player'].get('club_name', ''),
            "value": p['cost']
        }
        export_data["players"].append(player_data)
    
    return export_data

def main():
    # Upload du fichier
    uploaded_file = st.file_uploader("📁 **Chargez votre base de données FC25**", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ **{len(df):,} joueurs chargés avec succès !**")
            
            # Interface principale avec tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🎮 **Constructeur**", "📊 **Analytics**", "⚔️ **Comparaison**", "📤 **Export**"])
            
            with tab1:
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
                            st.metric("⭐ Overall", f"{team_stats['overall']
