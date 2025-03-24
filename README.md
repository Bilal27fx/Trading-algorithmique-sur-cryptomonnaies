# Projet de Test de Stratégie Momentum et Construction de Portefeuille

## Objectif

Ce projet a pour objectif de tester une stratégie momentum sur six actifs différents. L'idée est d'évaluer la performance d'une approche basée sur le momentum et, par la suite, de créer un portefeuille avec une répartition particulière des pourcentages selon des critères définis.

## Description du Projet

Le projet se compose de deux grandes parties :

1. **Backtesting de la stratégie momentum sur QuantConnect**  
   Le fichier `strategie_momentum_quantconnect.py` contient le code utilisé pour implémenter la stratégie momentum sur la plateforme QuantConnect.  
   - La stratégie calcule le momentum à partir d'une fenêtre glissante de prix.
   - Elle utilise également des indicateurs tels que l'EMA et le RSI pour confirmer les signaux.
   - La gestion de position se fait avec un intervalle de rebalancement de 14 jours.

2. **Création et Visualisation du Portefeuille**  
   Après avoir testé la stratégie sur les différents actifs, une allocation de portefeuille est réalisée avec une répartition spécifique des pourcentages.  
   - Le portefeuille est visualisé via une interface web.
   - Pour lancer l'application de visualisation, utilisez la commande suivante dans votre terminal :

     ```bash
     streamlit run Implementation.py
     ```

## Structure des Fichiers

- `strategie_momentum_quantconnect.py` :  
  Contient le code de la stratégie momentum utilisée sur QuantConnect. Ce script implémente le calcul du momentum, l'utilisation des indicateurs EMA et RSI ainsi que la gestion des positions (entrée et liquidation).

- `Implementation.py` :  
  Script Streamlit permettant de visualiser et analyser le portefeuille construit à partir des tests de la stratégie sur les différents actifs.

- Autres fichiers et dossiers :  
  Vous pouvez trouver dans ce projet des fichiers supplémentaires (données, scripts d'analyse, etc.) utiles à la compréhension et à l'évaluation de la stratégie et du portefeuille.

## Prérequis

- **Python 3.x**  
- **Bibliothèques Python requises** :
  - `numpy`
  - `pandas`
  - `streamlit`
  - (Éventuellement d'autres bibliothèques spécifiques à votre environnement QuantConnect ou à l'analyse)

## Installation

1. Clonez le dépôt sur votre machine :

   ```bash
   git clone https://github.com/votre-utilisateur/votre-projet.git
   cd votre-projet
