import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuration
st.set_page_config(page_title="FC25 Team Builder Pro", page_icon="⚽", layout="wide")

# CSS simplifié + style pour le selectbox
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.player-card {
    text-align: center;
    padding: 8px;
    border-radius: 10px;
    margin: 3px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 0.8rem;
}
.stSelectbox > div > div > select {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ FC25 TEAM BUILDER PRO</h1>', unsafe_allow_html=True)

# Formations simplifiées
FORMATIONS = {
    "4-3-3": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1}},
    "4-4-2": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "LM": 1, "CM": 2, "RM": 1, "ST": 2}},
    "3-5-2": {"positions": {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CDM": 1, "CM": 2, "ST": 2}},
    "4-2-3-1": {"positions": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CDM": 2, "CAM": 3, "ST": 1}}
}

# Compatibilité positions simplifiée
POSITION_COMPATIBILITY = {
    "GK": ["GK"], "CB": ["CB", "CDM"], "LB": ["LB", "LWB", "LM"], "RB": ["RB", "RWB", "RM"],
    "LWB": ["LWB", "LB", "LM"], "RWB": ["RWB", "RB", "RM"], "CDM": ["CDM", "CM", "CB"],
    "CM": ["CM", "CDM", "CAM"], "LM": ["LM", "LW", "LB"], "RM": ["RM", "RW", "RB"],
    "CAM": ["CAM", "CM", "LW", "RW"], "LW": ["LW", "LM", "ST"], "RW": ["RW", "RM", "ST"],
    "ST": ["ST", "CF", "LW", "RW"]
}

@st.cache_data
def load_data(uploaded_file):
    """Charge et nettoie les données"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Nettoyage prix
        if 'value' in df.columns:
            df['value_clean'] = df['value'].astype(str).str.replace('[€,]', '', regex=True)
            df['value_numeric'] = pd.to_numeric(df['value_clean'].str.replace('[MK]', '', regex=True), errors='coerce')
            df.loc[df['value_clean'].str.contains('K', na=False), 'value_numeric'] /= 1000
            df['value_numeric'] = df['value_numeric'].fillna(0)
        
        # Calculs
        df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
        df['potential_gap'] = df.get('potential', df['overall_rating']) - df['overall_rating']
        df['efficiency'] = df['overall_rating'] / (np.log1p(df['value_numeric']) + 1)
        
        if 'player_id' not in df.columns:
            df['player_id'] = range(len(df))
            
        return df
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None

@st.cache_data
def get_player_suggestions(df, search_term=""):
    """Retourne les suggestions de joueurs pour l'autocomplétion"""
    if not search_term:
        # Retourne les 100 joueurs les mieux notés par défaut
        top_players = df.nlargest(100, 'overall_rating')
        return [f"{name} ({overall} OVR)" 
                for name, overall in zip(top_players['name'], top_players['overall_rating'])]
    
    # Filtre par nom contenant le terme de recherche
    filtered = df[df['name'].str.contains(search_term, case=False, na=False)]
    if filtered.empty:
        return ["Aucun joueur trouvé..."]
    
    # Trie par overall décroissant et prend les 20 premiers
    filtered = filtered.nlargest(20, 'overall_rating')
    return [f"{name} ({overall} OVR)" 
            for name, overall in zip(filtered['name'], filtered['overall_rating'])]

def extract_player_name(selected_option):
    """Extrait le nom du joueur de l'option sélectionnée"""
    if not selected_option or selected_option == "Aucun joueur trouvé...":
        return ""
    # Supprime la partie " (XX OVR)" à la fin
    return selected_option.split(' (')[0]

def can_play_position(player_positions, required_position):
    """Vérifie compatibilité position"""
    if not player_positions or pd.isna(player_positions):
        return False
    player_pos_list = str(player_positions).split(',')
    compatible = POSITION_COMPATIBILITY.get(required_position, [required_position])
    return any(pos.strip() in compatible for pos in player_pos_list)

def get_filtered_players(df, position=None, exclude_ids=None, filters=None):
    """Filtre les joueurs selon critères"""
    exclude_ids = exclude_ids or []
    result = df[~df['player_id'].isin(exclude_ids)].copy()
    
    if position:
        result = result[result['positions'].apply(lambda x: can_play_position(x, position))]
    
    if filters:
        if filters.get('age_range'):
            min_age, max_age = filters['age_range']
            result = result[(result['age'] >= min_age) & (result['age'] <= max_age)]
        
        if filters.get('max_budget'):
            result = result[result['value_numeric'] <= filters['max_budget']]
        
        if filters.get('min_overall'):
            result = result[result['overall_rating'] >= filters['min_overall']]
        
        if not filters.get('include_free_agents', True):
            result = result[result['value_numeric'] > 0]
        
        if filters.get('leagues'):
            result = result[result.get('league_name', pd.Series(dtype='object')).isin(filters['leagues'])]
    
    return result

def optimize_team(df, formation, budget, filters):
    """Optimise l'équipe selon formation et budget"""
    selected_players = []
    remaining_budget = budget
    used_ids = set()
    
    positions = FORMATIONS[formation]["positions"]
    
    for position, count in positions.items():
        for _ in range(count):
            candidates = get_filtered_players(df, position, used_ids, {**filters, 'max_budget': remaining_budget})
            
            if candidates.empty:
                continue
            
            # Score composite
            candidates['score'] = (
                candidates['overall_rating'] * 0.4 +
                candidates['efficiency'] * 0.3 +
                (40 - candidates['age'].fillna(25)) * 0.2 +
                candidates.get('potential', candidates['overall_rating']) * 0.1
            )
            
            best = candidates.loc[candidates['score'].idxmax()]
            cost = best['value_numeric']
            
            selected_players.append({
                'player': best,
                'position': position,
                'cost': cost
            })
            
            remaining_budget -= cost
            used_ids.add(best['player_id'])
    
    return selected_players, remaining_budget

def find_similar_players(df, target_name, budget, top_n=5):
    """Trouve des joueurs similaires"""
    target = df[df['name'].str.contains(target_name, case=False, na=False)]
    
    if target.empty:
        return pd.DataFrame()
    
    target_player = target.iloc[0]
    
    # Critères de similarité
    df_filtered = df[
        (df['value_numeric'] <= budget) & 
        (df['player_id'] != target_player['player_id']) &
        (abs(df['age'] - target_player['age']) <= 5)
    ].copy()
    
    # Score de similarité
    df_filtered['similarity'] = (
        100 - abs(df_filtered['overall_rating'] - target_player['overall_rating']) * 2 -
        abs(df_filtered['age'] - target_player['age']) * 1 -
        abs(df_filtered.get('potential', df_filtered['overall_rating']) - target_player.get('potential', target_player['overall_rating'])) * 1.5
    )
    
    return df_filtered.nlargest(top_n, 'similarity')[['name', 'overall_rating', 'age', 'value_numeric', 'similarity', 'positions']]

def display_team_formation(players, formation):
    """Affiche la formation simplement"""
    st.subheader(f"🏆 Formation {formation}")
    
    player_dict = {p['position']: p for p in players}
    positions_order = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "LM", "RM", "CAM", "LW", "RW", "ST"]
    
    cols = st.columns(min(len(positions_order), 6))
    col_idx = 0
    
    for pos in positions_order:
        if pos in player_dict:
            with cols[col_idx % 6]:
                p = player_dict[pos]['player']
                st.markdown(f"""
                <div class="player-card">
                    <b>{p['name'][:12]}</b><br>
                    {pos} | {p['overall_rating']} OVR<br>
                    €{player_dict[pos]['cost']:.1f}M | {p.get('age', 'N/A')} ans
                </div>
                """, unsafe_allow_html=True)
                col_idx += 1

def main():
    # Upload fichier
    uploaded_file = st.file_uploader("📁 **Chargez votre base FC25 (CSV)**", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ **{len(df):,} joueurs chargés !**")
            
            # Tabs principales
            tab1, tab2, tab3, tab4 = st.tabs(["🏗️ **Constructeur**", "🔍 **Recherche**", "👥 **Similaires**", "📊 **Analytics**"])
            
            with tab1:
                st.markdown("### 🏗️ **Constructeur d'équipe optimisé**")
                st.info("💡 **Fonctionnalité :** Créez automatiquement une équipe complète selon votre formation, budget et critères. L'algorithme optimise le rapport qualité/prix.")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Configuration de base
                    formation = st.selectbox("📋 **Formation**", list(FORMATIONS.keys()))
                    budget = st.number_input("💰 **Budget (millions €)**", min_value=10, max_value=2000, value=200, step=25)
                    
                    # Filtres
                    st.markdown("#### 🔍 **Filtres**")
                    age_range = st.slider("🎂 Âge", 16, 40, (18, 35))
                    min_overall = st.slider("⭐ Overall minimum", 40, 99, 70)
                    include_free = st.checkbox("🆓 Inclure agents libres (€0)", value=True, help="Les joueurs sans club coûtent 0€")
                    
                    # Ligues (si disponible)
                    leagues = []
                    if 'league_name' in df.columns:
                        leagues = st.multiselect("🏆 Championnats", options=sorted(df['league_name'].dropna().unique()))
                    
                    filters = {
                        'age_range': age_range,
                        'min_overall': min_overall,
                        'include_free_agents': include_free,
                        'leagues': leagues
                    }
                    
                    if st.button("🚀 **CRÉER ÉQUIPE**", type="primary"):
                        with st.spinner("⚡ Optimisation..."):
                            team, remaining = optimize_team(df, formation, budget, filters)
                            
                            if team:
                                st.session_state.update({
                                    'team': team,
                                    'remaining_budget': remaining,
                                    'formation': formation,
                                    'total_spent': sum(p['cost'] for p in team)
                                })
                                st.success("✅ **Équipe créée !**")
                            else:
                                st.error("❌ **Impossible avec ces critères**")
                
                with col2:
                    if 'team' in st.session_state:
                        team = st.session_state['team']
                        
                        # Métriques
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("💰 Coût", f"€{st.session_state['total_spent']:.0f}M", 
                                     f"€{st.session_state['remaining_budget']:.0f}M restant")
                        with col_m2:
                            avg_overall = np.mean([p['player']['overall_rating'] for p in team])
                            st.metric("⭐ Overall", f"{avg_overall:.1f}")
                        with col_m3:
                            avg_age = np.mean([p['player'].get('age', 25) for p in team])
                            st.metric("👶 Âge moyen", f"{avg_age:.1f} ans")
                        
                        # Formation
                        display_team_formation(team, st.session_state['formation'])
            
            with tab2:
                st.markdown("### 🔍 **Recherche personnalisée**")
                st.info("💡 **Fonctionnalité :** Trouvez exactement le nombre de joueurs souhaité selon vos critères précis (position, budget, stats, etc.).")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Paramètres recherche
                    search_position = st.selectbox("📍 **Position**", ["Toutes"] + list(POSITION_COMPATIBILITY.keys()))
                    num_players = st.number_input("👥 **Nombre de joueurs**", min_value=1, max_value=50, value=10)
                    max_price = st.number_input("💰 **Prix max par joueur (€M)**", min_value=0, max_value=500, value=50)
                    
                    # Filtres avancés
                    min_overall_search = st.slider("⭐ Overall min", 40, 99, 75)
                    age_range_search = st.slider("🎂 Âge", 16, 40, (18, 32))
                    include_free_search = st.checkbox("🆓 Agents libres", value=True)
                    
                    search_filters = {
                        'age_range': age_range_search,
                        'min_overall': min_overall_search,
                        'max_budget': max_price,
                        'include_free_agents': include_free_search
                    }
                    
                    if st.button("🔍 **RECHERCHER**"):
                        pos = None if search_position == "Toutes" else search_position
                        results = get_filtered_players(df, pos, filters=search_filters)
                        
                        if not results.empty:
                            results = results.nlargest(num_players, 'overall_rating')
                            st.session_state['search_results'] = results
                        else:
                            st.warning("❌ Aucun joueur trouvé")
                
                with col2:
                    if 'search_results' in st.session_state:
                        results = st.session_state['search_results']
                        st.success(f"✅ **{len(results)} joueurs trouvés**")
                        
                        # Tableau résultats
                        display_data = results[['name', 'positions', 'overall_rating', 'age', 'value_numeric']].copy()
                        display_data.columns = ['Nom', 'Positions', 'Overall', 'Âge', 'Prix (€M)']
                        display_data['Prix (€M)'] = display_data['Prix (€M)'].round(1)
                        
                        st.dataframe(display_data, use_container_width=True, height=400)
                        
                        # Stats rapides
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric("💰 Prix total", f"€{results['value_numeric'].sum():.0f}M")
                            st.metric("⭐ Overall moyen", f"{results['overall_rating'].mean():.1f}")
                        with col_s2:
                            st.metric("👶 Âge moyen", f"{results['age'].mean():.1f} ans")
                            st.metric("💎 Joueur le plus cher", f"€{results['value_numeric'].max():.1f}M")
            
            with tab3:
                st.markdown("### 👥 **Joueurs similaires**")
                st.info("💡 **Fonctionnalité :** Sélectionnez un joueur dans la liste déroulante pour trouver des alternatives similaires dans votre budget (même style, âge proche, stats comparables).")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # ✨ NOUVELLE FONCTIONNALITÉ: Menu déroulant avec suggestions
                    st.markdown("#### 🎯 **Sélection du joueur**")
                    
                    # Champ de recherche pour filtrer
                    search_filter = st.text_input(
                        "🔍 **Filtrer par nom**", 
                        placeholder="Tapez pour filtrer la liste...",
                        help="Commencez à taper pour réduire la liste des joueurs"
                    )
                    
                    # Génère les suggestions basées sur le filtre
                    suggestions = get_player_suggestions(df, search_filter)
                    
                    # Menu déroulant avec suggestions
                    selected_player_option = st.selectbox(
                        "👤 **Choisir le joueur**",
                        options=suggestions,
                        index=0,
                        help="Sélectionnez un joueur dans la liste"
                    )
                    
                    # Extraction du nom réel du joueur
                    target_name = extract_player_name(selected_player_option)
                    
                    # Affichage du joueur sélectionné
                    if target_name and target_name != "":
                        player_info = df[df['name'].str.contains(target_name, case=False, na=False)]
                        if not player_info.empty:
                            player = player_info.iloc[0]
                            st.success(f"✅ **Joueur sélectionné:** {player['name']}")
                            
                            # Affichage des infos du joueur
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("⭐ Overall", f"{player['overall_rating']}")
                                st.metric("💰 Prix", f"€{player['value_numeric']:.1f}M")
                            with col_info2:
                                st.metric("👶 Âge", f"{player.get('age', 'N/A')} ans")
                                st.metric("📍 Positions", f"{player.get('positions', 'N/A')}")
                    
                    st.markdown("---")
                    
                    # Configuration de la recherche
                    similar_budget = st.number_input("💰 **Budget max pour alternatives (€M)**", min_value=1, max_value=500, value=100)
                    num_similar = st.slider("📊 **Nombre de résultats**", 3, 15, 5)
                    
                    if st.button("🔍 **TROUVER SIMILAIRES**", type="primary") and target_name:
                        with st.spinner("🔄 Recherche en cours..."):
                            similar_players = find_similar_players(df, target_name, similar_budget, num_similar)
                            
                            if not similar_players.empty:
                                st.session_state['similar_players'] = similar_players
                                st.session_state['target_name'] = target_name
                                st.success("✅ **Alternatives trouvées !**")
                            else:
                                st.warning(f"❌ Aucun joueur similaire à '{target_name}' trouvé dans ce budget")
                
                with col2:
                    if 'similar_players' in st.session_state:
                        similar = st.session_state['similar_players']
                        target = st.session_state['target_name']
                        
                        st.success(f"✅ **Joueurs similaires à {target}**")
                        
                        # Affichage résultats avec icônes et couleurs
                        display_similar = similar.copy()
                        display_similar.columns = ['Nom', 'Overall', 'Âge', 'Prix (€M)', 'Similarité %', 'Positions']
                        display_similar['Prix (€M)'] = display_similar['Prix (€M)'].round(1)
                        display_similar['Similarité %'] = display_similar['Similarité %'].round(1)
                        
                        # Style conditionnel pour le dataframe
                        def highlight_similarity(val):
                            if val >= 85:
                                return 'background-color: #d4edda; color: #155724'  # Vert
                            elif val >= 70:
                                return 'background-color: #fff3cd; color: #856404'  # Jaune
                            else:
                                return 'background-color: #f8d7da; color: #721c24'  # Rouge
                        
                        styled_df = display_similar.style.applymap(highlight_similarity, subset=['Similarité %'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Graphiques analytiques
                        col_g1, col_g2 = st.columns(2)
                        
                        with col_g1:
                            # Graphique similarité
                            fig_sim = px.bar(
                                x=display_similar['Nom'][:5],
                                y=display_similar['Similarité %'][:5],
                                title="📊 Score de similarité (Top 5)",
                                labels={'x': 'Joueur', 'y': 'Similarité %'},
                                color=display_similar['Similarité %'][:5],
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig_sim, use_container_width=True)
                        
                        with col_g2:
                            # Graphique Prix vs Overall
                            fig_scatter = px.scatter(
                                display_similar,
                                x='Prix (€M)',
                                y='Overall',
                                size='Similarité %',
                                hover_name='Nom',
                                title="💰 Prix vs Overall des alternatives",
                                color='Similarité %',
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Recommandation du meilleur choix
                        best_value = display_similar.loc[display_similar['Similarité %'].idxmax()]
                        st.markdown("#### 🏆 **Recommandation**")
                        st.success(f"🥇 **Meilleur choix:** {best_value['Nom']} - {best_value['Similarité %']:.0f}% de similarité pour €{best_value['Prix (€M)']}M")
            
            with tab4:
                st.markdown("### 📊 **Analytics et export**")
                st.info("💡 **Fonctionnalité :** Analysez vos données avec des graphiques avancés et exportez vos équipes en CSV/JSON.")
                
                if 'team' in st.session_state or 'search_results' in st.session_state:
                    
                    # Choix données à analyser
                    data_source = st.radio("📊 **Analyser :**", 
                                         ["Équipe créée", "Résultats recherche"] if 'search_results' in st.session_state else ["Équipe créée"])
                    
                    if data_source == "Équipe créée" and 'team' in st.session_state:
                        team_data = [p['player'] for p in st.session_state['team']]
                        analysis_df = pd.DataFrame(team_data)
                        title_suffix = "de l'équipe"
                    elif data_source == "Résultats recherche" and 'search_results' in st.session_state:
                        analysis_df = st.session_state['search_results']
                        title_suffix = "des résultats"
                    else:
                        st.info("🔍 Créez une équipe ou faites une recherche d'abord")
                        return
                    
                    # Graphiques
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Distribution âges
                        fig_age = px.histogram(
                            analysis_df, x='age', nbins=8,
                            title=f"📊 Distribution des âges {title_suffix}",
                            labels={'age': 'Âge', 'count': 'Nombre'}
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
                        
                        # Overall vs Prix
                        if 'value_numeric' in analysis_df.columns:
                            fig_scatter = px.scatter(
                                analysis_df, x='value_numeric', y='overall_rating',
                                title=f"💰 Overall vs Prix {title_suffix}",
                                labels={'value_numeric': 'Prix (€M)', 'overall_rating': 'Overall'}
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col_g2:
                        # Top joueurs par overall
                        top_players = analysis_df.nlargest(8, 'overall_rating')
                        fig_top = px.bar(
                            top_players, x='overall_rating', y='name',
                            orientation='h', title=f"🌟 Top joueurs {title_suffix}",
                            labels={'overall_rating': 'Overall', 'name': 'Joueur'}
                        )
                        st.plotly_chart(fig_top, use_container_width=True)
                        
                        # Nationalités
                        if 'nationality' in analysis_df.columns:
                            nat_counts = analysis_df['nationality'].value_counts().head(6)
                            fig_nat = px.pie(
                                values=nat_counts.values, names=nat_counts.index,
                                title=f"🌍 Nationalités {title_suffix}"
                            )
                            st.plotly_chart(fig_nat, use_container_width=True)
                    
                    # Export
                    st.markdown("#### 📤 **Export des données**")
                    
                    if st.button("📥 **Télécharger CSV**"):
                        csv_data = analysis_df[['name', 'overall_rating', 'age', 'value_numeric']].copy()
                        csv_data.columns = ['Nom', 'Overall', 'Age', 'Prix_Millions']
                        
                        csv_string = csv_data.to_csv(index=False)
                        st.download_button(
                            "💾 Cliquez pour télécharger",
                            csv_string,
                            f"fc25_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )
                
                else:
                    st.info("🔍 Créez une équipe ou effectuez une recherche pour voir les analytics !")
            
            # Aperçu base de données
            with st.expander("👀 **Aperçu de la base de données**"):
                col_db1, col_db2, col_db3 = st.columns(3)
                with col_db1:
                    st.metric("👥 Total joueurs", f"{len(df):,}")
                with col_db2:
                    st.metric("⭐ Overall max", int(df['overall_rating'].max()))
                with col_db3:
                    st.metric("💰 Joueur le + cher", f"€{df['value_numeric'].max():.0f}M")
                
                st.dataframe(df[['name', 'overall_rating', 'age', 'value_numeric', 'positions']].head(10), 
                           use_container_width=True)
    
    else:
        st.info("📁 **Chargez votre fichier CSV FC25 pour commencer !**")
        
        st.markdown("""
        ### 🎮 **Guide d'utilisation rapide**
        
        **🏗️ Constructeur :** Créez une équipe complète automatiquement selon votre formation et budget
        
        **🔍 Recherche :** Trouvez un nombre précis de joueurs selon vos critères
        
        **👥 Similaires :** Découvrez des alternatives à vos joueurs favoris
        
        **📊 Analytics :** Analysez vos données avec des graphiques et exportez en CSV/JSON
        
        **💡 Astuce :** Activez les "agents libres" pour inclure les joueurs gratuits dans vos recherches !
        """)

if __name__ == "__main__":
    main()
