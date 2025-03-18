import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt  # Pour afficher le graphique

class MarketData:
    """
    Gère le chargement et la préparation des données depuis un fichier CSV.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.df = None

    def load_data(self):
        """
        Charge le CSV, convertit les timestamps et met en forme le DataFrame.
        """
        df = pd.read_csv(self.filename)

        # Convertir les timestamps en datetime et indexer
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Timestamp", inplace=True)

        # Convertir en float
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)

        self.df = df

    def get_data(self) -> pd.DataFrame:
        """
        Retourne le DataFrame chargé.
        """
        if self.df is None:
            raise ValueError("Les données ne sont pas encore chargées. Appelez load_data() d'abord.")
        return self.df


class LiquidityGrabDetector:
    """
    Implémente la détection des swings majeurs, des prises de liquidité
    et la simulation de trades en utilisant une méthodologie où l'entrée se fait
    immédiatement après la clôture de la bougie générant le signal, pendant la réintégration.
    Le risque est fixé à 1% du prix d'entrée.
    """
    def __init__(self, df: pd.DataFrame, r_percent=0.01):
        self.df = df.copy()
        self.major_swing_highs = []
        self.major_swing_lows = []
        self.valid_liquidity_grabs_highs = None
        self.valid_liquidity_grabs_lows = None
        self.r_percent = r_percent  # Risque fixé à 1%

    def detect_major_swing_points(self, window=15, min_variation=0.01):
        """
        Détecte les swings hauts et bas majeurs.
        Retourne deux Series booléennes : swing_highs et swing_lows.
        """
        highs = self.df["High"].rolling(window, center=True).max()
        lows = self.df["Low"].rolling(window, center=True).min()

        swing_highs = (self.df["High"] == highs) & ((highs - lows) > self.df["High"] * min_variation)
        swing_lows = (self.df["Low"] == lows) & ((highs - lows) > self.df["Low"] * min_variation)

        return swing_highs.fillna(False), swing_lows.fillna(False)

    def is_major_high(self, index, lookback=50) -> bool:
        """
        Vérifie si la bougie à 'index' est le plus haut sur 'lookback' périodes.
        """
        return self.df.loc[index, "High"] == self.df["High"].rolling(lookback).max()[index]

    def is_major_low(self, index, lookback=50) -> bool:
        """
        Vérifie si la bougie à 'index' est le plus bas sur 'lookback' périodes.
        """
        return self.df.loc[index, "Low"] == self.df["Low"].rolling(lookback).min()[index]

    def strong_reintegration(self, index, lookback=3) -> bool:
        """
        Vérifie qu'il y a réintégration (c'est-à-dire que la clôture reste à l'intérieur
        du range défini par la bougie de prise de liquidité pendant 'lookback' jours).
        """
        high_level = self.df.loc[index, "High"]
        low_level = self.df.loc[index, "Low"]

        reintegrated_high = all(self.df.loc[index:index + pd.Timedelta(days=lookback), "Close"] < high_level)
        reintegrated_low = all(self.df.loc[index:index + pd.Timedelta(days=lookback), "Close"] > low_level)

        return reintegrated_high or reintegrated_low

    def run_detection(self, window=20, min_variation=0.02, lookback_swing=50, lookback_reintegration=3):
        """
        Lance le processus de détection :
         - Détecte les swings majeurs
         - Valide la prise de liquidité en fonction de la réintégration
        """
        swing_highs, swing_lows = self.detect_major_swing_points(window=window, min_variation=min_variation)

        self.major_swing_highs = [
            idx for idx in self.df[swing_highs].index if self.is_major_high(idx, lookback=lookback_swing)
        ]
        self.major_swing_lows = [
            idx for idx in self.df[swing_lows].index if self.is_major_low(idx, lookback=lookback_swing)
        ]

        liquidity_grabs_highs = self.df.index.isin(self.major_swing_highs) & (self.df["Close"] < self.df["High"])
        liquidity_grabs_lows = self.df.index.isin(self.major_swing_lows) & (self.df["Close"] > self.df["Low"])

        reintegration_check = pd.Series(
            data=self.df.index.map(lambda idx: self.strong_reintegration(idx, lookback=lookback_reintegration)),
            index=self.df.index
        ).fillna(False)

        self.valid_liquidity_grabs_highs = liquidity_grabs_highs & reintegration_check
        self.valid_liquidity_grabs_lows = liquidity_grabs_lows & reintegration_check

    def get_alines_for_plot(self):
        """
        Génère les annotations (lignes) pour la visualisation avec mplfinance.
        """
        lines = []
        for idx in self.df[self.valid_liquidity_grabs_highs].index:
            lines.append(((idx, self.df.loc[idx, "High"]),
                          (idx, self.df.loc[idx, "High"] * 1.005), 'red'))
        for idx in self.df[self.valid_liquidity_grabs_lows].index:
            lines.append(((idx, self.df.loc[idx, "Low"]),
                          (idx, self.df.loc[idx, "Low"] * 0.995), 'blue'))
        alines = [(line[0], line[1]) for line in lines]
        colors = [line[2] for line in lines]
        return alines, colors

    def simulate_trade_outcome(self, signal_idx, entry, sl, tp, trade_type):
        """
        Parcourt les bougies postérieures pour déterminer si le TP ou le SL est atteint.
        On commence à partir de la bougie suivante (après le signal).
        En cas de double atteinte dans une même bougie, on considère le SL.
        """
        df_after = self.df[self.df.index > signal_idx]
        for idx, row in df_after.iterrows():
            if trade_type == 'long':
                if row["Low"] <= sl:
                    return 'loss'
                if row["High"] >= tp:
                    return 'win'
            elif trade_type == 'short':
                if row["High"] >= sl:
                    return 'loss'
                if row["Low"] <= tp:
                    return 'win'
        return 'open'

    def simulate_trades(self):
        """
        Pour chaque signal validé, simule le trade en prenant l'entrée immédiatement
        après la clôture de la bougie (pendant la réintégration) avec un risque fixe de 1%.
        Renvoie la liste des trades avec leurs paramètres et résultats.
        """
        trades = []
        # Trade LONG : liquidity grabs sur swing bas
        for idx in self.df[self.valid_liquidity_grabs_lows].index:
            candle = self.df.loc[idx]
            entry = candle["Close"]            # Entrée au prix de clôture
            sl = entry * (1 - self.r_percent)    # SL à -1%
            tp = entry * (1 + 2 * self.r_percent)  # TP à +2%
            outcome = self.simulate_trade_outcome(idx, entry, sl, tp, trade_type='long')
            trades.append({
                'time': idx,
                'type': 'long',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'r': self.r_percent,
                'outcome': outcome
            })
        # Trade SHORT : liquidity grabs sur swing haut
        for idx in self.df[self.valid_liquidity_grabs_highs].index:
            candle = self.df.loc[idx]
            entry = candle["Close"]            # Entrée au prix de clôture
            sl = entry * (1 + self.r_percent)    # SL à +1%
            tp = entry * (1 - 2 * self.r_percent)  # TP à -2%
            outcome = self.simulate_trade_outcome(idx, entry, sl, tp, trade_type='short')
            trades.append({
                'time': idx,
                'type': 'short',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'r': self.r_percent,
                'outcome': outcome
            })
        return trades


def main():
    # 1) Préparer les données
    data = MarketData(filename="BTCUSDT_15m.csv")
    data.load_data()
    df = data.get_data()

    # 2) Détecter les signaux de prise de liquidité
    detector = LiquidityGrabDetector(df, r_percent=0.01)
    detector.run_detection(window=20, min_variation=0.02, lookback_swing=50, lookback_reintegration=3)

    # 3) Tracer le graphique avec annotations
    alines, colors = detector.get_alines_for_plot()
    fig, axlist = mpf.plot(df,
                           type="candle",
                           style="charles",
                           volume=False,
                           title="Stratégie Prise de Liquidité + Réintégration BTC/USDT",
                           ylabel="Prix (USDT)",
                           figsize=(12, 6),
                           alines=dict(alines=alines, colors=colors),
                           returnfig=True)

    # 4) Simulation des trades sur les signaux détectés
    trades = detector.simulate_trades()
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    losses = sum(1 for t in trades if t['outcome'] == 'loss')
    opens = sum(1 for t in trades if t['outcome'] == 'open')
    
    # Calcul du profit net en % du capital (gain = +2%, perte = -1%)
    net_profit_percent = wins * 2 - losses * 1

    print("=== Résultats de la simulation ===")
    print(f"Nombre total de trades simulés : {total_trades}")
    print(f"Trades gagnants : {wins}")
    print(f"Trades perdants : {losses}")
    print(f"Trades non clôturés : {opens}")
    if total_trades > 0:
        print(f"Taux de réussite : {wins / total_trades * 100:.2f}%")
        print(f"Profit net en % du capital : {net_profit_percent:.2f}%")

    plt.show()

if __name__ == "__main__":
    main()
