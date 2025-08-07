import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="FC25 Team Builder Pro", page_icon="⚽", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Nettoyage prix
    df['value_clean'] = df['value'].astype(str).str.replace('[€,]', '', regex=True)
    df['value_numeric'] = pd.to_numeric(df['value_clean'].str.replace('[MK]', '', regex=True), errors='coerce')
    df.loc[df['value_clean'].str.contains('K', na=False), 'value_numeric'] /= 1000
    df['value_numeric'] = df['value_numeric'].fillna(0)

    df['age'] = 2025 - pd.to_datetime(df['dob'], errors='coerce').dt.year
    df['potential_gap'] = df.get('potential', df['overall_rating']) - df['overall_rating']
    df['efficiency'] = df['overall_rating'] / (np.log1p(df['value_numeric']) + 1)

    if 'player_id' not in df.columns:
        df['player_id'] = range(len(df))
    return df

def can_play_position(player_positions, required_position):
    POSITION_COMPATIBILITY = {
        "GK": ["GK"], "CB": ["CB", "CDM"], "LB": ["LB", "LWB", "LM"], "RB": ["RB", "RWB", "RM"],
        "LWB": ["LWB", "LB", "LM"], "RWB": ["RWB", "RB", "RM"], "CDM": ["CDM", "CM", "CB"],
        "CM": ["CM", "CDM", "CAM"], "LM": ["LM", "LW", "LB"], "RM": ["RM", "RW", "RB"],
        "CAM": ["CAM", "CM", "LW", "RW"], "LW": ["LW", "LM", "ST"], "RW": ["RW", "RM", "ST"],
        "ST": ["ST", "CF", "LW", "RW"]
    }
    if not player_positions or pd.isna(player_positions):
        return False
    player_pos_list = str(player_positions).split(',')
    compatible = POSITION_COMPATIBILITY.get(required_position, [required_position])
    return any(pos.strip() in compatible for pos in player_pos_list)

def get_filtered_players(df, position=None, exclude_ids=None, filters=None):
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

        if filters.get('leagues') and 'league_name' in df.columns:
            result = result[result['league_name'].isin(filters['leagues'])]

        # 🔥 Ajout de tous les filtres numériques et catégoriels dynamiques
        for key, val in filters.items():
            if key in df.columns:
                if isinstance(val, tuple) and len(val) == 2:
                    result = result[(result[key] >= val[0]) & (result[key] <= val[1])]
                elif isinstance(val, list) and val:
                    result = result[result[key].isin(val)]
    return result

def optimize_team(df, formation_dict, budget, filters):
    selected_players = []
    remaining_budget = budget
    used_ids = set()

    for position, count in formation_dict.items():
        for _ in range(count):
            candidates = get_filtered_players(df, position, exclude_ids=used_ids, filters={**filters, 'max_budget': remaining_budget})
            if candidates.empty:
                continue
            candidates['score'] = (
                candidates['overall_rating'] * 0.4 +
                candidates['efficiency'] * 0.3 +
                (40 - candidates['age'].fillna(25)) * 0.2 +
                candidates.get('potential', candidates['overall_rating']) * 0.1
            )
            best = candidates.loc[candidates['score'].idxmax()]
            selected_players.append(best)
            remaining_budget -= best['value_numeric']
            used_ids.add(best['player_id'])

    return selected_players

def main():
    st.title("⚽ FC25 TEAM BUILDER PRO")
    uploaded_file = st.file_uploader("📁 Chargez votre base FC25 (CSV)", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.success(f"{len(df)} joueurs chargés !")

        tab1, tab2 = st.tabs(["🏗️ Constructeur", "🔍 Recherche"])

        # FORMATIONS
        FORMATIONS = {
            "4-4-2": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "LM": 1, "RM": 1, "ST": 2},
            "4-3-3": {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 3, "LW": 1, "RW": 1, "ST": 1},
            "3-5-2": {"GK": 1, "CB": 3, "CDM": 1, "CM": 2, "CAM": 1, "ST": 2}
        }

        NUMERIC_FILTERS = [
            'height_cm', 'weight_kg', 'wage', 'value_numeric',
            'crossing', 'finishing', 'heading_accuracy', 'short_passing',
            'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
            'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
            'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
            'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
            'composure', 'defensive_awareness', 'standing_tackle', 'sliding_tackle'
        ]

        CATEGORICAL_FILTERS = [
            'preferred_foot', 'work_rate', 'body_type',
            'club_name', 'club_league_name', 'country_name'
        ]

        with tab1:
            st.subheader("🏗️ Générateur d'équipe intelligent")
            formation_choice = st.selectbox("📋 Formation", list(FORMATIONS.keys()))
            formation_dict = FORMATIONS[formation_choice]
            budget = st.number_input("💰 Budget (€M)", 10, 1000, 200, step=10)
            age_range = st.slider("🎂 Âge", 16, 45, (18, 32))
            min_overall = st.slider("⭐ Overall minimum", 40, 99, 70)
            include_free = st.checkbox("🆓 Inclure agents libres", value=True)

            leagues = []
            if 'league_name' in df.columns:
                leagues = st.multiselect("🏆 Ligues", options=sorted(df['league_name'].dropna().unique()))

            filters = {
                'age_range': age_range,
                'min_overall': min_overall,
                'include_free_agents': include_free,
                'leagues': leagues
            }

            # ✅ Filtres avancés
            with st.expander("⚙️ Filtres avancés personnalisés"):
                st.markdown("Affinez la sélection avec des critères spécifiques sur les attributs du joueur.")
                for col in NUMERIC_FILTERS:
                    if col in df.columns:
                        min_val, max_val = int(df[col].min()), int(df[col].max())
                        filters[col] = st.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, (min_val, max_val))
                for col in CATEGORICAL_FILTERS:
                    if col in df.columns:
                        options = sorted(df[col].dropna().unique())
                        selection = st.multiselect(f"{col.replace('_', ' ').title()}", options)
                        if selection:
                            filters[col] = selection

            if st.button("🚀 Construire l'équipe"):
                team = optimize_team(df, formation_dict, budget, filters)
                if team:
                    st.success("✅ Équipe générée avec succès !")
                    for player in team:
                        st.markdown(f"- {player['name']} ({player['positions']}) | {player['overall_rating']} OVR | €{player['value_numeric']:.1f}M")
                else:
                    st.error("❌ Aucun joueur trouvé pour cette configuration.")

        with tab2:
            st.subheader("🔍 Recherche personnalisée")
            search_position = st.selectbox("📌 Position", ["Toutes"] + sorted(df['positions'].dropna().unique()))
            number = st.slider("👥 Nombre de joueurs", 1, 50, 10)
            max_price = st.slider("💶 Prix maximum (€M)", 0, 500, 50)
            min_overall_search = st.slider("⭐ Overall minimum", 40, 99, 70)
            age_search = st.slider("🎂 Âge", 16, 45, (18, 35))
            include_free_search = st.checkbox("🆓 Inclure agents libres", value=True)

            filters_search = {
                'age_range': age_search,
                'min_overall': min_overall_search,
                'max_budget': max_price,
                'include_free_agents': include_free_search
            }

            with st.expander("⚙️ Filtres avancés personnalisés (recherche)"):
                for col in NUMERIC_FILTERS:
                    if col in df.columns:
                        min_val, max_val = int(df[col].min()), int(df[col].max())
                        filters_search[col] = st.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, (min_val, max_val))
                for col in CATEGORICAL_FILTERS:
                    if col in df.columns:
                        options = sorted(df[col].dropna().unique())
                        selection = st.multiselect(f"{col.replace('_', ' ').title()}", options)
                        if selection:
                            filters_search[col] = selection

            if st.button("🔎 Lancer la recherche"):
                pos = None if search_position == "Toutes" else search_position
                results = get_filtered_players(df, pos, filters=filters_search)
                if not results.empty:
                    results = results.nlargest(number, 'overall_rating')
                    st.dataframe(results[['name', 'positions', 'overall_rating', 'age', 'value_numeric']])
                else:
                    st.warning("Aucun joueur trouvé avec ces critères.")

if __name__ == "__main__":
    main() 
