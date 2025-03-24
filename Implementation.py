import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ✅ 1. Charger les données des actifs
assets_results = [
    {
        "Asset": "EURUSD",
        "Net Profit (%)": 13.174,
        "Compounding Annual Return (%)": 1.247,
        "Sharpe Ratio": -0.246,
        "Sortino Ratio": -0.185,
        "Win Rate (%)": 65,
        "Loss Rate (%)": 35,
        "Profit-Loss Ratio": 1.08,
        "Drawdown (%)": 10.8,
        "Beta": 0.013,
        "Alpha": -0.013,
        "Annual Standard Deviation": 0.049,
        "Information Ratio": -0.595
    },
    {
        "Asset": "BTCUSD",
        "Net Profit (%)": 367.415,
        "Compounding Annual Return (%)": 16.699,
        "Sharpe Ratio": 0.954,
        "Sortino Ratio": 1.036,
        "Win Rate (%)": 90,
        "Loss Rate (%)": 10,
        "Profit-Loss Ratio": 1.18,
        "Drawdown (%)": 16.2,
        "Beta": 0.054,
        "Alpha": 0.092,
        "Annual Standard Deviation": 0.101,
        "Information Ratio": 0.102
    },
    {
        "Asset": "ETHUSD",
        "Net Profit (%)": 234.741,
        "Compounding Annual Return (%)": 12.862,
        "Sharpe Ratio": 0.868,
        "Sortino Ratio": 0.872,
        "Win Rate (%)": 93,
        "Loss Rate (%)": 7,
        "Profit-Loss Ratio": 2.38,
        "Drawdown (%)": 10.9,
        "Beta": 0.064,
        "Alpha": 0.063,
        "Annual Standard Deviation": 0.079,
        "Information Ratio": -0.064
    },
    {
        "Asset": "XAUUSD",
        "Net Profit (%)": 50.433,
        "Compounding Annual Return (%)": 4.174,
        "Sharpe Ratio": 0.128,
        "Sortino Ratio": 0.115,
        "Win Rate (%)": 76,
        "Loss Rate (%)": 24,
        "Profit-Loss Ratio": 0.71,
        "Drawdown (%)": 20.7,
        "Beta": 0.004,
        "Alpha": 0.009,
        "Annual Standard Deviation": 0.076,
        "Information Ratio": -0.42
    },
    {
        "Asset": "IEF",
        "Net Profit (%)": 9.225,
        "Compounding Annual Return (%)": 0.888,
        "Sharpe Ratio": -0.349,
        "Sortino Ratio": -0.217,
        "Win Rate (%)": 69,
        "Loss Rate (%)": 31,
        "Profit-Loss Ratio": 1.09,
        "Drawdown (%)": 15.0,
        "Beta": -0.078,
        "Alpha": -0.009,
        "Annual Standard Deviation": 0.043,
        "Information Ratio": -0.574
    },
    {
        "Asset": "SPX500USD",
        "Net Profit (%)": 100.015,
        "Compounding Annual Return (%)": 7.189,
        "Sharpe Ratio": 0.36,
        "Sortino Ratio": 0.298,
        "Win Rate (%)": 81,
        "Loss Rate (%)": 19,
        "Profit-Loss Ratio": 1.14,
        "Drawdown (%)": 16.3,
        "Beta": 0.289,
        "Alpha": 0.008,
        "Annual Standard Deviation": 0.087,
        "Information Ratio": -0.37
    }
]
df = pd.DataFrame(assets_results)

# ✅ 2. Fonctions pour initialiser et ajuster les allocations
def reset_allocations():
    st.session_state.allocations = np.ones(len(df)) / len(df)

if "allocations" not in st.session_state:
    reset_allocations()

if st.sidebar.button("🔄 Réinitialiser les allocations"):
    reset_allocations()

def adjust_allocations(index, new_value):
    allocations = st.session_state.allocations
    allocations[index] = new_value
    remaining = 1 - new_value
    other_indices = [i for i in range(len(df)) if i != index]

    if remaining == 0:
        allocations[other_indices] = 0
    else:
        total_other = sum(allocations[other_indices])
        if total_other > 0:
            factor = remaining / total_other
            allocations[other_indices] *= factor
        else:
            allocations[other_indices] = remaining / len(other_indices)
    st.session_state.allocations = allocations

# ✅ 3. Sélection de toutes les colonnes numériques (sauf "Asset") comme features
all_features = [col for col in df.columns if col != "Asset"]
X = df[all_features]

# ✅ 4. Mise à l'échelle des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 5. Définition de la cible (y) : on inclut Beta, Alpha, etc.
#    Ex : On dit qu'un actif est "performant" si :
#       - Sharpe Ratio > médiane
#       - Sortino Ratio > médiane
#       - Win Rate (%) > médiane
#       - Profit-Loss Ratio > médiane
#       - Drawdown (%) < médiane
#       - Net Profit (%) > médiane
#       - Beta < médiane (optionnel)
#       - Alpha > médiane (optionnel)
#       - Annual Standard Deviation < médiane (optionnel)
#    Ajustez selon votre logique !

df_median = df.median(numeric_only=True)

sharpe_ok  = df["Sharpe Ratio"]        > df_median["Sharpe Ratio"]
sortino_ok = df["Sortino Ratio"]       > df_median["Sortino Ratio"]
winrate_ok = df["Win Rate (%)"]        > df_median["Win Rate (%)"]
plratio_ok = df["Profit-Loss Ratio"]   > df_median["Profit-Loss Ratio"]
drawdown_ok= df["Drawdown (%)"]        < df_median["Drawdown (%)"]
netprofit_ok = df["Net Profit (%)"]    > df_median["Net Profit (%)"]

# Exemples de conditions optionnelles :
beta_ok  = df["Beta"]                  < df_median["Beta"]
alpha_ok = df["Alpha"]                 > df_median["Alpha"]
std_ok   = df["Annual Standard Deviation"] < df_median["Annual Standard Deviation"]


all_conditions = [
    sharpe_ok, sortino_ok, winrate_ok, plratio_ok,
    drawdown_ok, netprofit_ok, beta_ok, alpha_ok, std_ok
]

# On compte combien de critères sont respectés pour chaque actif
criteria_sum = np.sum(all_conditions, axis=0)

# On dit qu'un actif est "performant" si >= 5 conditions sont respectées
threshold = 5
y = np.where(criteria_sum >= threshold, 1, 0)

unique_classes = np.unique(y)


if len(unique_classes) < 2:
    st.warning("Impossible d'entraîner le modèle : y n'a qu'une seule classe. Ajustez vos critères.")
else:
    # ✅ 6. Entraînement du modèle
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_scaled, y)

    # ✅ 7. Génération des allocations
    def predict_best_allocations():
        probabilities = model.predict_proba(X_scaled)[:, 1]
        allocations = probabilities / probabilities.sum()
        return allocations

    if st.sidebar.button("💡 Générer des allocations IA"):
        st.session_state.allocations = predict_best_allocations()

# ✅ 8. Interface utilisateur (sliders)
for i, asset in enumerate(df["Asset"]):
    new_value = st.sidebar.slider(
        f"{asset}",
        0.0, 1.0,
        st.session_state.allocations[i],
        step=0.05,
        key=f"slider_{i}"
    )
    if new_value != st.session_state.allocations[i]:
        adjust_allocations(i, new_value)

df["User Allocation"] = st.session_state.allocations

# ✅ 9. Calcul des performances du portefeuille
portfolio_metrics = {}
for col in df.columns:
    if col != "Asset" and pd.api.types.is_numeric_dtype(df[col]):
        portfolio_metrics[col] = np.dot(df[col], df["User Allocation"])

# ✅ 10. Affichage des performances
st.subheader("📊 Performance du Portefeuille Personnalisé")
st.dataframe(pd.DataFrame(portfolio_metrics, index=["Valeurs"]).T, width=1500)

# ✅ 11. Visualisation 3D (par ex. Allocation vs Sharpe Ratio vs Drawdown)
fig = go.Figure()
if "Net Profit (%)" in df.columns:
    net_profit = df["Net Profit (%)"]
    min_np = net_profit.min()
    max_np = net_profit.max()
else:
    net_profit = [0]*len(df)
    min_np, max_np = 0, 1

for i, asset in enumerate(df["Asset"]):
    fig.add_trace(go.Scatter3d(
        x=[df["User Allocation"][i]],
        y=[df["Sharpe Ratio"][i]],
        z=[df["Drawdown (%)"][i]],
        mode='markers',
        marker=dict(
            size=10,
            color=net_profit[i],
            cmin=min_np,
            cmax=max_np,
            colorscale='Plasma',
            showscale=False
        ),
        name=asset,
        text=(
            f"<b>{asset}</b><br>"
            f"Net Profit: {net_profit[i]}%<br>"
            f"Allocation: {df['User Allocation'][i]:.2f}<br>"
            f"Sharpe Ratio: {df['Sharpe Ratio'][i]:.3f}<br>"
            f"Drawdown: {df['Drawdown (%)'][i]}%"
        ),
        hoverinfo='text'
    ))

fig.update_layout(
    title="Graphique 3D (Beta, Alpha, etc. inclus)",
    template="plotly_dark",
    scene=dict(
        xaxis=dict(title="Allocation (%)"),
        yaxis=dict(title="Sharpe Ratio"),
        zaxis=dict(title="Drawdown (%)"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    height=700
)
st.plotly_chart(fig)

# ✅ 12. Jauge (Net Profit)
portfolio_net_profit = portfolio_metrics.get("Net Profit (%)", 0)
min_range_gauge = 0
max_range_gauge = 400

gauge_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=portfolio_net_profit,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio Net Profit (%)"},
        gauge=dict(
            axis=dict(range=[min_range_gauge, max_range_gauge]),
            bar=dict(color="white"),
            steps=[
                {'range': [0, 100],  'color': '#8B0000'},
                {'range': [100, 200], 'color': '#eb4034'},
                {'range': [200, 300], 'color': '#f0bd28'},
                {'range': [300, 400], 'color': '#3cb371'}
            ],
        )
    )
)
gauge_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(gauge_fig)
