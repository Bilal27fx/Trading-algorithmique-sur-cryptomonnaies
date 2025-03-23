import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ✅ 1. Charger les données des actifs
assets_results = [
    {"Asset": "EURUSD", "Net Profit (%)": 13.174, "Compounding Annual Return (%)": 1.247, "Sharpe Ratio": -0.246, "Sortino Ratio": -0.185, "Win Rate (%)": 65, "Loss Rate (%)": 35, "Profit-Loss Ratio": 1.08, "Drawdown (%)": 10.8, "Beta": 0.013, "Alpha": -0.013, "Annual Standard Deviation": 0.049, "Information Ratio": -0.595},
    {"Asset": "BTCUSD", "Net Profit (%)": 367.415, "Compounding Annual Return (%)": 16.699, "Sharpe Ratio": 0.954, "Sortino Ratio": 1.036, "Win Rate (%)": 90, "Loss Rate (%)": 10, "Profit-Loss Ratio": 1.18, "Drawdown (%)": 16.2, "Beta": 0.054, "Alpha": 0.092, "Annual Standard Deviation": 0.101, "Information Ratio": 0.102},
    {"Asset": "ETHUSD", "Net Profit (%)": 234.741, "Compounding Annual Return (%)": 12.862, "Sharpe Ratio": 0.868, "Sortino Ratio": 0.872, "Win Rate (%)": 93, "Loss Rate (%)": 7, "Profit-Loss Ratio": 2.38, "Drawdown (%)": 10.9, "Beta": 0.064, "Alpha": 0.063, "Annual Standard Deviation": 0.079, "Information Ratio": -0.064},
    {"Asset": "XAUUSD", "Net Profit (%)": 50.433, "Compounding Annual Return (%)": 4.174, "Sharpe Ratio": 0.128, "Sortino Ratio": 0.115, "Win Rate (%)": 76, "Loss Rate (%)": 24, "Profit-Loss Ratio": 0.71, "Drawdown (%)": 20.7, "Beta": 0.004, "Alpha": 0.009, "Annual Standard Deviation": 0.076, "Information Ratio": -0.42},
    {"Asset": "IEF", "Net Profit (%)": 9.225, "Compounding Annual Return (%)": 0.888, "Sharpe Ratio": -0.349, "Sortino Ratio": -0.217, "Win Rate (%)": 69, "Loss Rate (%)": 31, "Profit-Loss Ratio": 1.09, "Drawdown (%)": 15.0, "Beta": -0.078, "Alpha": -0.009, "Annual Standard Deviation": 0.043, "Information Ratio": -0.574},
    {"Asset": "SPX500USD", "Net Profit (%)": 100.015, "Compounding Annual Return (%)": 7.189, "Sharpe Ratio": 0.36, "Sortino Ratio": 0.298, "Win Rate (%)": 81, "Loss Rate (%)": 19, "Profit-Loss Ratio": 1.14, "Drawdown (%)": 16.3, "Beta": 0.289, "Alpha": 0.008, "Annual Standard Deviation": 0.087, "Information Ratio": -0.37}
]
df = pd.DataFrame(assets_results)

# ✅ Fonction pour initialiser et ajuster les allocations
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

# ✅ Optimisation avancée de l'IA
X = df[["Sharpe Ratio", "Sortino Ratio", "Win Rate (%)", "Loss Rate (%)", "Profit-Loss Ratio", "Drawdown (%)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.where(df["Sharpe Ratio"] > df["Sharpe Ratio"].median(), 1, 0)

# Utilisation de Gradient Boosting pour améliorer la précision
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_scaled, y)

def predict_best_allocations():
    probabilities = model.predict_proba(X_scaled)[:, 1]
    allocations = probabilities / probabilities.sum()
    return allocations

if st.sidebar.button("💡 Générer des allocations IA"):
    st.session_state.allocations = predict_best_allocations()

for i, asset in enumerate(df["Asset"]):
    new_value = st.sidebar.slider(f"{asset}", 0.0, 1.0, st.session_state.allocations[i], step=0.05, key=f"slider_{i}")
    if new_value != st.session_state.allocations[i]:
        adjust_allocations(i, new_value)

df["User Allocation"] = st.session_state.allocations

# ✅ Calcul des performances du portefeuille
portfolio_metrics = {metric: np.dot(df[metric], df["User Allocation"]) for metric in df.columns[1:]}

# ✅ Affichage des performances
st.subheader("📊 Performance du Portefeuille Personnalisé")
st.dataframe(pd.DataFrame(portfolio_metrics, index=["Valeurs"]).T, width=1500)
fig = go.Figure()

net_profit = df["Net Profit (%)"]
min_np = net_profit.min()
max_np = net_profit.max()

for i, asset in enumerate(df["Asset"]):
    fig.add_trace(go.Scatter3d(
        x=[df["User Allocation"][i]],
        y=[df["Sharpe Ratio"][i]],
        z=[df["Drawdown (%)"][i]],
        mode='markers',
        marker=dict(
            size=10,                   # Taille fixe pour tous les points
            color=net_profit[i],       # Couleur basée sur Net Profit
            cmin=min_np,               # Min global
            cmax=max_np,               # Max global
            colorscale='Plasma',       # Palette adaptée au fond sombre
            showscale=False            # Désactive la barre de couleurs
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
    title="Graphique 3D sans colorbar (taille fixe des points)",
    template="plotly_dark",  # Fond sombre
    scene=dict(
        xaxis=dict(
            title="Allocation (%)",
            backgroundcolor="black",
            gridcolor="gray",
            showbackground=True
        ),
        yaxis=dict(
            title="Sharpe Ratio",
            backgroundcolor="black",
            gridcolor="gray",
            showbackground=True
        ),
        zaxis=dict(
            title="Drawdown (%)",
            backgroundcolor="black",
            gridcolor="gray",
            showbackground=True
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    height=700
)

st.plotly_chart(fig)

portfolio_net_profit = portfolio_metrics["Net Profit (%)"]

# Définissez la plage de la jauge.
# Vous pouvez la rendre dynamique en fonction de la plage possible, ou la fixer.
min_range_gauge = 0
max_range_gauge = 400  # Ajustez selon vos valeurs

# Exemple d'amélioration des couleurs de la jauge
gauge_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=portfolio_net_profit,  # Valeur du Net Profit du portefeuille
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio Net Profit (%)"},
        gauge=dict(
            axis=dict(range=[min_range_gauge, max_range_gauge]),
            # Aiguille en blanc pour bien ressortir sur fond sombre
            bar=dict(color="white"),  
            # Palette du rouge (faible) au vert (élevé)
            steps=[
                {'range': [0, 100],  'color': '#8B0000'},   # Rouge foncé
                {'range': [100, 200], 'color': '#eb4034'}, # Rouge clair
                {'range': [200, 300], 'color': '#f0bd28'}, # Orange/jaune
                {'range': [300, 400], 'color': '#3cb371'}  # Vert
            ],
        )
    )
)

gauge_fig.update_layout(
    template="plotly_dark",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(gauge_fig)
