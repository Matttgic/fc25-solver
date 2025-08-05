import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary

def get_position_group(positions):
    if 'GK' in positions:
        return 'GK'
    elif any(pos in positions for pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']):
        return 'DEF'
    elif any(pos in positions for pos in ['CM', 'CDM', 'CAM', 'LM', 'RM']):
        return 'MID'
    else:
        return 'FWD'

def solve_team(df, formation, budget, criteria, filters):
    df = df.copy()
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    df['main_position'] = df['positions'].apply(lambda p: get_position_group(p if isinstance(p, str) else ""))

    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col].astype(str).str.contains(val, case=False, na=False)]

    prob = LpProblem("Team_Selection", LpMaximize)
    players = list(df.index)
    x = LpVariable.dicts("player", players, 0, 1, LpBinary)

    def safe_score(i, c):
        try:
            return float(df.loc[i, c])
        except:
            return 0

    prob += lpSum([x[i] * sum(safe_score(i, c) for c in criteria) for i in players])
    prob += lpSum([x[i] * df.loc[i, 'value'] for i in players]) <= budget

    for role, count in formation.items():
        prob += lpSum([x[i] for i in players if df.loc[i, 'main_position'] == role]) == count

    prob += lpSum([x[i] for i in players]) == sum(formation.values())
    prob.solve()

    selected = df.loc[[i for i in players if x[i].varValue == 1]].copy()
    selected['selected_criteria_sum'] = selected[criteria].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    selected = selected.sort_values(by='main_position')
    return selected.to_dict(orient='records')
