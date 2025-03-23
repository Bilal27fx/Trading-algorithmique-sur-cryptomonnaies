import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

# ✅ Entraînement d'un modèle RandomForest pour améliorer la prédiction
X = df[["Sharpe Ratio", "Sortino Ratio", "Win Rate (%)", "Loss Rate (%)", "Profit-Loss Ratio", "Drawdown (%)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.where(df["Sharpe Ratio"] > df["Sharpe Ratio"].median(), 1, 0)  # 1 = Bonne allocation, 0 = Mauvaise
model = RandomForestClassifier(n_estimators=100, random_state=42)
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

for i, asset in enumerate(df["Asset"]):
    fig.add_trace(go.Scatter3d(
        x=[df["User Allocation"][i]],
        y=[df["Sharpe Ratio"][i]],
        z=[df["Drawdown (%)"][i]],
        mode='markers',
        marker=dict(size=7),
        name=asset
    ))

fig.update_layout(
    title="📊 Allocation vs Sharpe Ratio vs Drawdown",
    scene=dict(
        xaxis_title="Allocation (%)",
        yaxis_title="Sharpe Ratio",
        zaxis_title="Drawdown (%)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[df["Sharpe Ratio"].min() - 0.2, df["Sharpe Ratio"].max() + 0.2]),
        zaxis=dict(range=[df["Drawdown (%)"].min() - 2, df["Drawdown (%)"].max() + 2])
    ),
    margin=dict(l=50, r=50, b=50, t=50),
    height=600
)

st.plotly_chart(fig)
