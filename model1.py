# gas_predictor.py
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

class GasPricePredictor:
    def __init__(self):
        self.api_key = os.getenv("EIA_API_KEY")
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.last_update = None
        self.cache_duration = 3600  # 1 heure en secondes
        
        # Charger le modèle pré-entraîné s'il existe
        self.load_model()
    
    def fetch_gas_data(self, days=1000):
        """Récupère les données de gaz avec gestion d'erreurs"""
        if not self.api_key:
            raise ValueError("Clé API EIA manquante")
        
        url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
        params = {
            "api_key": self.api_key,
            "frequency": "daily",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": days
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data["response"]["data"]
            df = pd.DataFrame(records)

            df = df[["period", "value"]]
            df = df.rename(columns={"period": "date", "value": "Price"})
            df["date"] = pd.to_datetime(df["date"])
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

            df = df.dropna(subset=["Price"])
            df = df.groupby("date", as_index=False).mean()
            df = df.sort_values("date").reset_index(drop=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération des données: {e}")
    
    def add_features(self, df):
        """Ajoute toutes les features nécessaires"""
        # Features basées sur les prix
        for window in [3, 5, 10, 21]:
            df[f'sma_{window}'] = df['Price'].shift(1).rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = (df['Price'].shift(1) - df[f'sma_{window}']) / df[f'sma_{window}']
        
        # Volatilité
        df['return_1d'] = df['Price'].pct_change(1)
        for window in [5, 10, 21]:
            df[f'volatility_{window}'] = df['return_1d'].shift(1).rolling(window=window).std()
        
        # Prix décalés
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['Price'].shift(lag)
        
        # RSI simple
        df['rsi_14'] = self.calculate_rsi(df['Price'], 14)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)
    
    def train_model(self, retrain=False):
        """Entraîne le modèle (appelé si nécessaire)"""
        if self.model is not None and not retrain:
            return
        
        print("Entraînement du modèle...")
        
        # Récupérer les données
        df = self.fetch_gas_data(days=2000)
        df = self.add_features(df)
        
        # Créer la cible
        df['target'] = df['Price'].shift(-1)
        
        # Sélectionner les features
        self.feature_cols = [col for col in df.columns if col not in ['date', 'Price', 'target', 'return_1d']]
        
        # Nettoyer les données
        df = df.dropna(subset=['target'] + self.feature_cols)
        
        if len(df) < 100:
            raise ValueError("Données insuffisantes pour l'entraînement")
        
        # Préparer les données
        X = df[self.feature_cols]
        y = df['target']
        
        # Normalisation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraîner le modèle
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        
        # Sauvegarder le modèle
        self.save_model()
        
        print("Modèle entraîné avec succès!")
    
    def get_prediction(self):
        """Obtient la prédiction pour le lendemain"""
        # Vérifier si le modèle est chargé
        if self.model is None:
            self.train_model()
        
        # Récupérer les données récentes
        df = self.fetch_gas_data(days=100)
        df = self.add_features(df)
        
        # Nettoyer et préparer
        df = df.dropna(subset=self.feature_cols)
        
        if len(df) == 0:
            raise ValueError("Pas de données disponibles pour la prédiction")
        
        # Prédiction
        X_latest = df[self.feature_cols].iloc[[-1]]
        X_latest_scaled = self.scaler.transform(X_latest)
        
        predicted_price = self.model.predict(X_latest_scaled)[0]
        current_price = df['Price'].iloc[-1]
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        return {
            'current_price': round(current_price, 4),
            'predicted_price': round(predicted_price, 4),
            'price_change': round(price_change, 4),
            'price_change_pct': round(price_change_pct, 2),
            'confidence': 'Modéle entraîné avec succès',
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    
    

    def get_model_metrics(self):
        """Calcule les vraies métriques du modèle sur les données d'entraînement"""
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")

        # Charger les données utilisées pour l'entraînement
        df = self.fetch_gas_data(days=2000)
        df = self.add_features(df)
        df['target'] = df['Price'].shift(-1)
        df = df.dropna(subset=['target'] + self.feature_cols)

        if len(df) == 0:
            raise ValueError("Pas assez de données pour évaluer le modèle.")

        X = df[self.feature_cols]
        y = df['target']
        X_scaled = self.scaler.transform(X)

        y_pred = self.model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)

        return {
            'r2_score': round(r2, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'last_trained': self.last_update
        }

    
    def save_model(self):
        """Sauvegarde le modèle et le scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'last_update': datetime.now().isoformat()
        }
        
        with open('gas_price_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Charge le modèle sauvegardé"""
        try:
            with open('gas_price_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.last_update = model_data.get('last_update')
            
            print("Modèle chargé avec succès!")
            
        except FileNotFoundError:
            print("Aucun modèle sauvegardé trouvé. Entraînement nécessaire.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")