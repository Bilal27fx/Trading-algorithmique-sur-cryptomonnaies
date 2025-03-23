import streamlit as st
import pandas as pd
import plotly.express as px

# 📌 Définition des profils d'investissement
profiles = {
    "Conservateur": {
        "Allocation": {"EURUSD": 20, "BTCUSD": 5, "ETHUSD": 5, "XAUUSD": 30, "IEF": 20, "SPX500USD": 20},
        "Net Profit (%)": 69.72,
        "Sharpe Ratio": 0.0825,
        "Sortino Ratio": 0.1091,
        "Drawdown (%)": 15.98,
        "Volatility": 0.0676,
        "Beta": 0.0519,
    },
    "Normal": {
        "Allocation": {"EURUSD": 15, "BTCUSD": 20, "ETHUSD": 10, "XAUUSD": 15, "IEF": 10, "SPX500USD": 30},
        "Net Profit (%)": 137.42,
        "Sharpe Ratio": 0.333,
        "Sortino Ratio": 0.352,
        "Drawdown (%)": 15.44,
        "Volatility": 0.0772,
        "Beta": 0.0986,
    },
    "Agressif": {
        "Allocation": {"EURUSD": 10, "BTCUSD": 40, "ETHUSD": 20, "XAUUSD": 10, "IEF": 10, "SPX500USD": 10},
        "Net Profit (%)": 228.96,
        "Sharpe Ratio": 0.676,
        "Sortino Ratio": 0.690,
        "Drawdown (%)": 15.62,
        "Volatility": 0.0899,
        "Beta": 0.1215,
    },
}

# 📌 Interface Streamlit
st.set_page_config(page_title="Comparaison des Profils d'Investissement", layout="wide")
st.title("💼 Comparaison des Profils d'Investissement")

# 🎯 Sélecteur de profil
selected_profile = st.selectbox("Choisissez un profil", list(profiles.keys()))

# Séparation des allocations et des performances
allocation_data = profiles[selected_profile]["Allocation"]
performance_data = {k: v for k, v in profiles[selected_profile].items() if k != "Allocation"}
df_selected = pd.DataFrame(performance_data, index=["Valeur"]).T  # Conversion en DataFrame

# 📊 Mise en page
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📌 Répartition Optimale - {selected_profile}")
    fig_pie = px.pie(names=allocation_data.keys(), values=allocation_data.values(), title=f"Répartition de l'Allocation - {selected_profile}")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Performance du Profil")
    st.dataframe(df_selected.style.format("{:.2f}").set_caption(f"Détails du profil {selected_profile}"))
