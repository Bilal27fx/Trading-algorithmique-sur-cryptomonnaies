import pandas as pd
import mplfinance as mpf
import numpy as np

# Charger le fichier CSV
filename = "BTCUSDT_1d.csv"
df = pd.read_csv(filename)

# Convertir les timestamps en datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
df.set_index("Timestamp", inplace=True)

# Convertir en float
df["Open"] = df["Open"].astype(float)
df["High"] = df["High"].astype(float)
df["Low"] = df["Low"].astype(float)
df["Close"] = df["Close"].astype(float)

# 📌 Détection améliorée des Swing Highs et Swing Lows majeurs
def detect_major_swing_points(data, window=15, min_variation=0.01):
    highs = data["High"].rolling(window, center=True).max()
    lows = data["Low"].rolling(window, center=True).min()

    swing_highs = (data["High"] == highs) & ((highs - lows) > data["High"] * min_variation)
    swing_lows = (data["Low"] == lows) & ((highs - lows) > data["Low"] * min_variation)
    
    return swing_highs.fillna(False), swing_lows.fillna(False)  # ✅ Convertir en booléen

swing_highs, swing_lows = detect_major_swing_points(df, window=20, min_variation=0.02)

# Vérifier si le swing est réellement le plus haut/bas dans une période récente
def is_major_high(index, df, lookback=50):
    return df.loc[index, "High"] == df["High"].rolling(lookback).max()[index]

def is_major_low(index, df, lookback=50):
    return df.loc[index, "Low"] == df["Low"].rolling(lookback).min()[index]

major_swing_highs = [idx for idx in df[swing_highs].index if is_major_high(idx, df)]
major_swing_lows = [idx for idx in df[swing_lows].index if is_major_low(idx, df)]

# 🛑 Détection des prises de liquidité sur swings majeurs
liquidity_grabs_highs = df.index.isin(major_swing_highs) & (df["Close"] < df["High"])
liquidity_grabs_lows = df.index.isin(major_swing_lows) & (df["Close"] > df["Low"])

# 📌 Vérification de la réintégration après la prise de liquidité
def strong_reintegration(index, df, lookback=3):
    """ Vérifie qu'aucune clôture ne reste en dehors du niveau après la prise de liquidité """
    high_level = df.loc[index, "High"]
    low_level = df.loc[index, "Low"]

    # Vérifier que toutes les clôtures reviennent à l'intérieur du range après un lookback de X jours
    reintegrated_high = all(df.loc[index:index + pd.Timedelta(days=lookback), "Close"] < high_level)
    reintegrated_low = all(df.loc[index:index + pd.Timedelta(days=lookback), "Close"] > low_level)

    return reintegrated_high or reintegrated_low

# ✅ Vérifier la réintégration après liquidation
reintegration_check = pd.Series(df.index.map(lambda idx: strong_reintegration(idx, df)), index=df.index).fillna(False)

# **Validation finale :** Prise de liquidité **uniquement si une réintégration suit**
valid_liquidity_grabs_highs = liquidity_grabs_highs & reintegration_check
valid_liquidity_grabs_lows = liquidity_grabs_lows & reintegration_check

# 📌 Génération des annotations sous forme de tuples
lines = []

# 🔴 Prises de liquidité en rouge (seulement si réintégration validée)
for idx in df[valid_liquidity_grabs_highs].index:
    lines.append(((idx, df.loc[idx, "High"]), (idx, df.loc[idx, "High"] * 1.005), 'red'))
for idx in df[valid_liquidity_grabs_lows].index:
    lines.append(((idx, df.loc[idx, "Low"]), (idx, df.loc[idx, "Low"] * 0.995), 'red'))

# 📌 Convertir les annotations pour mplfinance
alines = [(line[0], line[1]) for line in lines]
colors = [line[2] for line in lines]

# 📊 Tracer le graphique en chandeliers avec les nouvelles annotations
mpf.plot(df, type="candle", style="charles", volume=False,
         title="Stratégie Prise de Liquidité + Réintégration BTC/USDT",
         ylabel="Prix (USDT)", figsize=(12, 6),
         alines=dict(alines=alines, colors=colors))
