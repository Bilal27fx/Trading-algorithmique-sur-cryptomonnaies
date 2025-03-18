import pandas as pd
from binance.client import Client
from binance.websockets import BinanceSocketManager
import numpy as np
import datetime

# On importe vos classes existantes
from Implementation import LiquidityGrabDetector  # Adaptez le nom de fichier
# (ou copiez directement vos classes dans ce script si nécessaire)

# --- Paramètres de connexion à l'API (utilisez une clé test si possible) ---
API_KEY = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"

class RealTimeTrader:
    """
    Classe qui gère la réception des bougies en temps réel et
    exécute la stratégie LiquidityGrabDetector dès qu'une bougie se ferme.
    """
    def __init__(self, symbol="BTCUSDT", interval="15m", r_percent=0.01):
        self.symbol = symbol
        self.interval = interval
        self.r_percent = r_percent
        
        # DataFrame pour stocker les bougies live
        self.df_live = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        # On initialise le client Binance
        self.client = Client(api_key=API_KEY, api_secret=API_SECRET)
        self.bsm = BinanceSocketManager(self.client)

    def process_message(self, msg):
        """
        Callback appelé à chaque réception d'un message kline.
        On y ajoute la nouvelle bougie quand elle est clôturée,
        puis on appelle la stratégie.
        """
        if msg['e'] != 'error':
            # Données de la bougie (kline)
            kline = msg['k']
            
            # Si la bougie vient juste de se fermer
            if kline['x']:  # 'x' = True si la bougie est clôturée
                open_time = kline['t']  # timestamp d'ouverture
                open_price = float(kline['o'])
                high_price = float(kline['h'])
                low_price = float(kline['l'])
                close_price = float(kline['c'])
                volume = float(kline['v'])
                
                # Conversion du timestamp en datetime
                dt = pd.to_datetime(open_time, unit='ms')
                
                # Ajout/Remplacement de la ligne correspondante dans df_live
                self.df_live.loc[dt, "Open"] = open_price
                self.df_live.loc[dt, "High"] = high_price
                self.df_live.loc[dt, "Low"] = low_price
                self.df_live.loc[dt, "Close"] = close_price
                self.df_live.loc[dt, "Volume"] = volume
                
                # Tri par index pour maintenir un historique correct
                self.df_live.sort_index(inplace=True)
                
                # Appel de la stratégie si on a suffisamment de bougies
                if len(self.df_live) > 50:  # par ex. on attend un certain nombre de bougies
                    self.run_strategy()

        else:
            print("Erreur dans le message :", msg)

    def run_strategy(self):
        """
        Exécute la détection LiquidityGrabDetector et simule les trades.
        """
        # Copie le DataFrame live
        df_copy = self.df_live.copy()
        
        # Il faut un index de type DatetimeIndex
        # => Il l'est déjà, on peut le renommer si besoin
        df_copy.index.name = "Timestamp"
        
        # Les colonnes doivent correspondre à ["Open", "High", "Low", "Close"] pour votre algo
        # C'est déjà le cas
        
        # Instancier le détecteur
        detector = LiquidityGrabDetector(df_copy, r_percent=self.r_percent)
        detector.run_detection(
            window=20, 
            min_variation=0.02, 
            lookback_swing=50, 
            lookback_reintegration=3
        )
        
        # Simuler les trades
        trades = detector.simulate_trades()
        if trades:
            # Filtrer uniquement les trades qui viennent de se déclencher sur la dernière bougie
            # Pour cela, on peut comparer le 'time' du trade à l'index max
            last_timestamp = df_copy.index.max()
            new_trades = [t for t in trades if t['time'] == last_timestamp]
            if new_trades:
                print("=== NOUVEAU(S) TRADE(S) DÉTECTÉ(S) ===")
                for nt in new_trades:
                    print(nt)
                # Ici, vous pouvez soit simuler un ordre, soit en envoyer un vrai via l'API

    def start(self):
        """
        Lance la connexion WebSocket pour récupérer les bougies.
        """
        # Démarre le kline socket pour la paire choisie
        self.conn_key = self.bsm.start_kline_socket(
            self.symbol, 
            self.process_message, 
            interval=self.interval
        )
        self.bsm.start()

        print(f"WebSocket démarré pour {self.symbol} en timeframe {self.interval}...")

def main():
    trader = RealTimeTrader(symbol="BTCUSDT", interval="15m", r_percent=0.01)
    trader.start()

if __name__ == "__main__":
    main()
