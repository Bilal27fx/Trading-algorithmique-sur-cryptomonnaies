from AlgorithmImports import *
import numpy as np
import pandas as pd

class SingleAssetMomentum(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.lookback_period = 126
        self.volatility_period = 60
        self.target_volatility = 0.1
        self.leverage_cap = 4
        self.momentum_threshold = 0.02

        #  Changer cet actif à chaque test
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol


        # Stockage des prix passés pour calcul du momentum
        self.data = RollingWindow[float](self.lookback_period)
        self.ema = self.EMA(self.symbol, self.lookback_period, Resolution.Daily)
        self.rsi = self.RSI(self.symbol, 14, Resolution.Daily)

        self.results = []  # Stocke les performances
        self.last_rebalance = self.StartDate

    def OnData(self, data):
        if not data.ContainsKey(self.symbol) or not data[self.symbol]:
            return
        
        price = data[self.symbol].Close
        self.data.Add(price)

        if not self.data.IsReady or not self.ema.IsReady or not self.rsi.IsReady:
            return
        
        if (self.Time - self.last_rebalance) < timedelta(days=14):
            return
        self.last_rebalance = self.Time  

        # Calcul du momentum
        momentum = self.data[0] / self.data[self.lookback_period - 1] - 1

        # Calcul de la volatilité
        returns = np.array([self.data[i] / self.data[i+1] - 1 for i in range(self.volatility_period)])
        volatility = np.std(returns) * np.sqrt(252)

        if np.isnan(momentum) or np.isnan(volatility) or volatility == 0:
            return

        # Confirmation EMA et RSI
        ema_trend = price > self.ema.Current.Value
        rsi_confirmation = self.rsi.Current.Value > 50  

        # Calcul de la taille de position
        position_size = min(self.leverage_cap, self.target_volatility / volatility)

        # Gestion des positions
        if momentum > self.momentum_threshold and ema_trend and rsi_confirmation:
            self.SetHoldings(self.symbol, position_size)
        elif momentum < -self.momentum_threshold or not ema_trend:
            self.Liquidate(self.symbol)

