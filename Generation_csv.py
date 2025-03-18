import requests
import csv
import os
import time
import datetime

# Paramètres
symbol = "BTCUSDT"  # La paire de trading
interval = "15m"    # Intervalle des bougies
limit = 1000        # Limite maximale par requête (fixe par l'API Binance)
max_candles = 20000  # Nombre de bougies souhaité

# Initialisation
all_data = []
# Vous pouvez définir un startTime personnalisé (ici, début epoch). Pour récupérer depuis une date précise, convertissez-la en timestamp.
start_time = 0  

while len(all_data) < max_candles:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start_time}"
    response = requests.get(url)
    data = response.json()

    if not data:
        print("Plus de données disponibles.")
        break

    all_data.extend(data)

    # Si moins de 'limit' bougies ont été retournées, c'est la fin des données
    if len(data) < limit:
        print("Fin des données historiques.")
        break

    # Mettre à jour start_time avec le timestamp de la dernière bougie + 1 ms pour éviter la redondance
    last_candle_open_time = data[-1][0]
    start_time = last_candle_open_time + 1

    # Petite pause pour respecter les limites de l'API
    time.sleep(0.5)

print(f"Nombre de bougies récupérées : {len(all_data)}")

# Sauvegarde des données dans un fichier CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, f"{symbol}_{interval}.csv")

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    for candle in all_data:
        writer.writerow([candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]])

print(f"Données sauvegardées dans {filename}")
