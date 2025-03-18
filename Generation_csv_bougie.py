import requests
import csv
import os

# Paramètres
symbol = "BTCUSDT"  # Remplace par la paire de ton choix (ETHUSDT, etc.)
interval = "1d"  # Choisir : '1m', '5m', '15m', '30m', '1h', '4h', '1d'
limit = 5000  # Nombre de bougies (max 1000)
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# Récupération des données depuis Binance
response = requests.get(url)
data = response.json()

# Déterminer le chemin du fichier (dans le même dossier que le script)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Récupère le dossier du script
filename = os.path.join(script_dir, f"{symbol}_{interval}.csv")  # Chemin du fichier

# Création du fichier CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Volume"])  # En-têtes
    
    for candle in data:
        writer.writerow([candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]])

print(f"Données sauvegardées dans {filename}")
