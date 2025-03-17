import pandas as pd
import mplfinance as mpf

# Charger le fichier CSV
filename = "BTCUSDT_1d.csv"  # Assurez-vous que le fichier est dans le même dossier que le script
df = pd.read_csv(filename)

# Convertir le timestamp en format date
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

# Convertir les colonnes en type float
df["Open"] = df["Open"].astype(float)
df["High"] = df["High"].astype(float)
df["Low"] = df["Low"].astype(float)
df["Close"] = df["Close"].astype(float)

# Définir l'index sur la date
df.set_index("Timestamp", inplace=True)

# Affichage du graphique en chandeliers japonais
mpf.plot(df, type="candle", style="charles", volume=False,
         title="Évolution du prix BTC/USDT en Chandeliers Japonais",
         ylabel="Prix (USDT)", figsize=(12, 6))
