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
        self.api_key = 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe'
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
        """Features optimisées pour la prédiction financière"""
        # Rendements (plus prédictifs que les prix absolus)
        df['return_1d'] = df['Price'].pct_change(1).shift(1)
        df['return_2d'] = df['Price'].pct_change(2).shift(1)
        df['return_3d'] = df['Price'].pct_change(3).shift(1)
        df['return_5d'] = df['Price'].pct_change(5).shift(1)
        df['return_10d'] = df['Price'].pct_change(10).shift(1)
        
        # Volatilité réalisée (très importante en finance)
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['return_1d'].shift(1).rolling(window=window).std()
        
        # Moyennes mobiles sur les rendements
        for window in [3, 5, 10, 20]:
            df[f'ma_return_{window}d'] = df['return_1d'].shift(1).rolling(window=window).mean()
        
        # RSI amélioré
        df['rsi_14'] = self.calculate_rsi(df['Price'], 14).shift(1)
        df['rsi_7'] = self.calculate_rsi(df['Price'], 7).shift(1)
        
        # Bandes de Bollinger
        for window in [10, 20]:
            rolling_mean = df['Price'].shift(1).rolling(window=window).mean()
            rolling_std = df['Price'].shift(1).rolling(window=window).std()
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_position_{window}'] = (df['Price'].shift(1) - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Momentum features
        df['momentum_3d'] = (df['Price'].shift(1) - df['Price'].shift(4)) / df['Price'].shift(4)
        df['momentum_5d'] = (df['Price'].shift(1) - df['Price'].shift(6)) / df['Price'].shift(6)
        df['momentum_10d'] = (df['Price'].shift(1) - df['Price'].shift(11)) / df['Price'].shift(11)
        
        # Prix relatifs aux moyennes mobiles
        for window in [5, 10, 20]:
            sma = df['Price'].shift(1).rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = (df['Price'].shift(1) - sma) / sma
        
        # Tendance à court terme
        df['trend_5d'] = df['Price'].shift(1).rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
        df['trend_10d'] = df['Price'].shift(1).rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
        
        # Autocorrélation des rendements (mean reversion)
        df['return_autocorr_5d'] = df['return_1d'].shift(1).rolling(window=5).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
        
        # Features de saisonnalité
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Interactions importantes
        df['vol_momentum'] = df['volatility_10d'] * df['momentum_5d']
        df['rsi_momentum'] = df['rsi_14'] * df['momentum_3d']
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_model(self, retrain=False):
        """Entraîne le modèle avec une approche optimisée"""
        if self.model is not None and not retrain:
            return
        
        print("Entraînement du modèle optimisé...")
        
        # Récupérer plus de données pour un meilleur entraînement
        df = self.fetch_gas_data(days=3000)
        df = self.add_features(df)
        
        # Target: prédire le rendement plutôt que le prix absolu (plus stable)
        df['target'] = df['Price'].pct_change(1).shift(-1)
        
        # Sélectionner les features (exclure les colonnes non-features)
        self.feature_cols = [col for col in df.columns if col not in ['date', 'Price', 'target']]
        
        # Nettoyer les données
        df = df.dropna(subset=['target'] + self.feature_cols)
        
        if len(df) < 200:
            raise ValueError("Données insuffisantes pour l'entraînement")
        
        # Division temporelle: 70% train, 15% validation, 15% test
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Préparer les données
        X_train = train_df[self.feature_cols]
        y_train = train_df['target']
        X_val = val_df[self.feature_cols]
        y_val = val_df['target']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target']
        
        # Normalisation
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modèle optimisé pour les séries temporelles financières
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,  # Faible pour éviter l'overfitting
            learning_rate=0.03,  # Très faible pour un apprentissage stable
            subsample=0.7,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            objective="reg:squarederror",
            random_state=42,
            early_stopping_rounds=30,
            reg_alpha=0.1,  # Régularisation L1
            reg_lambda=0.1,  # Régularisation L2
            gamma=0.1  # Complexité minimale pour split
        )
        
        # Entraînement avec validation
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val), (X_test_scaled, y_test)],
            verbose=False
        )
        
        # Calcul des métriques sur le test set
        self.test_metrics = self.calculate_validation_metrics(X_test_scaled, y_test)
        self.save_model()
        
        print("Modèle entraîné avec succès!")
        print(f"Métriques test: R²={self.test_metrics['r2_score']:.4f}, MAE={self.test_metrics['mae']:.4f}")
    
    def calculate_validation_metrics(self, X_test_scaled, y_test):
        """Calcule les métriques sur les données de test"""
        y_pred = self.model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Vérifier et nettoyer les valeurs NaN/infinies
        if np.isnan(r2) or np.isinf(r2):
            r2 = 0.0
        if np.isnan(mae) or np.isinf(mae):
            mae = 0.0
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = 0.0
            
        return {
            'r2_score': round(float(r2), 4),
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4)
        }
    
    def get_prediction(self):
        """Obtient la prédiction pour le lendemain"""
        if self.model is None:
            self.train_model()
        
        # Récupérer les données récentes
        df = self.fetch_gas_data(days=150)
        df = self.add_features(df)
        
        # Nettoyer et préparer
        df = df.dropna(subset=self.feature_cols)
        
        if len(df) == 0:
            raise ValueError("Pas de données disponibles pour la prédiction")
        
        # Prédiction du rendement
        X_latest = df[self.feature_cols].iloc[[-1]]
        X_latest_scaled = self.scaler.transform(X_latest)
        
        predicted_return = self.model.predict(X_latest_scaled)[0]
        current_price = df['Price'].iloc[-1]
        
        # Convertir le rendement prédit en prix
        predicted_price = current_price * (1 + predicted_return)
        
        # Vérifier que les valeurs sont valides
        if pd.isna(predicted_price) or pd.isna(current_price):
            raise ValueError("Valeurs de prix invalides (NaN)")
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Vérifier et nettoyer les valeurs NaN/infinies
        if np.isnan(predicted_price) or np.isinf(predicted_price):
            predicted_price = current_price
            price_change = 0
            price_change_pct = 0
        
        return {
            'current_price': round(float(current_price), 4),
            'predicted_price': round(float(predicted_price), 4),
            'price_change': round(float(price_change), 4),
            'price_change_pct': round(float(price_change_pct), 2),
            'confidence': f'Modèle optimisé - R²: {self.test_metrics.get("r2_score", "N/A")}',
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    
    def get_model_metrics(self):
        """Retourne les métriques de test"""
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")

        # Retourner les métriques de test stockées
        if hasattr(self, 'test_metrics'):
            return {
                **self.test_metrics,
                'last_trained': self.last_update,
                'note': 'Métriques calculées sur données de test (holdout)'
            }
        else:
            return {
                'r2_score': 'N/A',
                'mae': 'N/A', 
                'rmse': 'N/A',
                'last_trained': self.last_update,
                'note': 'Métriques de test non disponibles'
            }
    
    def save_model(self):
        """Sauvegarde le modèle et le scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'last_update': datetime.now().isoformat(),
            'test_metrics': getattr(self, 'test_metrics', {})
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
            self.test_metrics = model_data.get('test_metrics', {})
            
            print("Modèle chargé avec succès!")
            
        except FileNotFoundError:
            print("Aucun modèle sauvegardé trouvé. Entraînement nécessaire.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")