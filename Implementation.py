import pandas as pd
import mplfinance as mpf

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

# Détection des hauts et bas majeurs (Swing Highs / Swing Lows)
def detect_swing_points(data, window=5):
    highs = data["High"].rolling(window, center=True).max()
    lows = data["Low"].rolling(window, center=True).min()
    
    swing_highs = (data["High"] == highs)
    swing_lows = (data["Low"] == lows)
    
    return swing_highs, swing_lows

swing_highs, swing_lows = detect_swing_points(df)

# Détection des prises de liquidité (cassures suivies de rejet)
liquidity_grabs_highs = (swing_highs & (df["Close"] < df["High"]))  # Rejet après un plus haut
liquidity_grabs_lows = (swing_lows & (df["Close"] > df["Low"]))  # Rejet après un plus bas

# Détection des cassures de structure (BOS)
bos_up = liquidity_grabs_highs.shift(1) & (df["Close"] > df["High"].shift(1))
bos_down = liquidity_grabs_lows.shift(1) & (df["Close"] < df["Low"].shift(1))

# Points d'entrée après le retest des prises de liquidité
entry_buy = bos_down.shift(1) & (df["Close"] > df["Low"].shift(2))  # Achat sur retest bas
entry_sell = bos_up.shift(1) & (df["Close"] < df["High"].shift(2))  # Vente sur retest haut

# Génération des annotations sous forme de liste de tuples
lines = []

# Prises de liquidité en rouge
for idx in df[liquidity_grabs_highs].index:
    lines.append(((idx, df.loc[idx, "High"]), (idx, df.loc[idx, "High"] * 1.005), "red"))
for idx in df[liquidity_grabs_lows].index:
    lines.append(((idx, df.loc[idx, "Low"]), (idx, df.loc[idx, "Low"] * 0.995), "red"))

# Cassures de structure en noir
for idx in df[bos_up].index:
    lines.append(((idx, df.loc[idx, "High"] * 1.005), (idx, df.loc[idx, "High"] * 1.01), "black"))
for idx in df[bos_down].index:
    lines.append(((idx, df.loc[idx, "Low"] * 0.995), (idx, df.loc[idx, "Low"] * 0.99), "black"))

# Points d’entrée en bleu
for idx in df[entry_buy].index:
    lines.append(((idx, df.loc[idx, "Low"] * 1.002), (idx, df.loc[idx, "Low"] * 1.005), "blue"))
for idx in df[entry_sell].index:
    lines.append(((idx, df.loc[idx, "High"] * 0.998), (idx, df.loc[idx, "High"] * 0.995), "blue"))

# Tracer le graphique en chandeliers avec annotations
mpf.plot(df, type="candle", style="charles", volume=False,
         title="Stratégie Prise de Liquidité BTC/USDT",
         ylabel="Prix (USDT)", figsize=(12, 6),
         alines=[(line[0], line[1]) for line in lines], colors=[line[2] for line in lines])
