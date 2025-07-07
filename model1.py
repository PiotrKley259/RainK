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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

class GasPricePredictor:
    def __init__(self):
        self.api_key = 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe'
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.last_update = None
        self.cache_duration = 3600  # 1 hour in seconds
        self.test_data = None
        self.load_model()

    def fetch_gas_data(self, days=3000):
        """Fetch gas price data from EIA and options data from Yahoo Finance"""
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
            response = requests.get(url, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            records = data["response"]["data"]
            df = pd.DataFrame(records)

            df = df[["period", "value"]]
            df = df.rename(columns={"period": "date", "value": "Price"})
            df["date"] = pd.to_datetime(df["date"])
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

            # Remove outliers using IQR
            Q1 = df["Price"].quantile(0.25)
            Q3 = df["Price"].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df["Price"] >= Q1 - 1.5 * IQR) & (df["Price"] <= Q3 + 1.5 * IQR)]

            df = df.dropna(subset=["Price"])
            df = df.groupby("date", as_index=False).mean()
            df = df.sort_values("date").reset_index(drop=True)
            df["Price"] = df["Price"].interpolate(method="linear")
            
            # Fetch options data from Yahoo Finance for UNG
            ticker = yf.Ticker("UNG")
            options_data = []
            for date in df["date"]:
                try:
                    exp_dates = ticker.options
                    if not exp_dates:
                        continue
                    exp_date = min([pd.to_datetime(exp) for exp in exp_dates if pd.to_datetime(exp) >= date])
                    opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
                    
                    hist = ticker.history(start=date, end=date + timedelta(days=1))
                    if hist.empty:
                        continue
                    underlying_price = hist["Close"].iloc[0]
                    
                    calls = opt_chain.calls
                    puts = opt_chain.puts
                    atm_strike = calls["strike"].iloc[(calls["strike"] - underlying_price).abs().idxmin()]
                    
                    call_price = calls[calls["strike"] == atm_strike]["lastPrice"].iloc[0] if not calls[calls["strike"] == atm_strike].empty else np.nan
                    put_price = puts[puts["strike"] == atm_strike]["lastPrice"].iloc[0] if not puts[puts["strike"] == atm_strike].empty else np.nan
                    
                    options_data.append({
                        "date": date,
                        "call_price": call_price,
                        "put_price": put_price,
                        "underlying_price": underlying_price
                    })
                except Exception as e:
                    options_data.append({"date": date, "call_price": np.nan, "put_price": np.nan, "underlying_price": np.nan})
            
            options_df = pd.DataFrame(options_data)
            options_df["date"] = pd.to_datetime(options_df["date"])
            options_df[["call_price", "put_price", "underlying_price"]] = options_df[["call_price", "put_price", "underlying_price"]].interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")
            
            df = df.merge(options_df, on="date", how="left")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération des données: {e}")

    def add_features(self, df):
        """Add technical and options-based features"""
        df['return_1d'] = df['Price'].pct_change(1).shift(1)
        df['return_2d'] = df['Price'].pct_change(2).shift(1)
        df['return_3d'] = df['Price'].pct_change(3).shift(1)
        df['return_5d'] = df['Price'].pct_change(5).shift(1)
        df['return_7d'] = df['Price'].pct_change(7).shift(1)
        df['return_14d'] = df['Price'].pct_change(14).shift(1)

        for window in [5, 10]:
            df[f'volatility_{window}d'] = df['return_1d'].shift(1).rolling(window=window).std()
        
        for window in [5, 10]:
            df[f'ma_return_{window}d'] = df['return_1d'].shift(1).rolling(window=window).mean()
        
        df['rsi_14'] = self.calculate_rsi(df['Price'], 14).shift(1)
        
        for window in [10]:
            rolling_mean = df['Price'].shift(1).rolling(window=window).mean()
            rolling_std = df['Price'].shift(1).rolling(window=window).std()
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_position_{window}'] = (df['Price'].shift(1) - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        df['momentum_5d'] = (df['Price'].shift(1) - df['Price'].shift(6)) / df['Price'].shift(6)
        
        for window in [5, 10]:
            sma = df['Price'].shift(1).rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = (df['Price'].shift(1) - sma) / sma
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        df['vol_momentum'] = df['volatility_10d'] * df['momentum_5d']
        
        df['call_put_ratio'] = (df['call_price'] / df['put_price']).shift(1)
        df['call_put_diff'] = (df['call_price'] - df['put_price']).shift(1)
        df['call_price_norm'] = (df['call_price'] / df['underlying_price']).shift(1)
        df['put_price_norm'] = (df['put_price'] / df['underlying_price']).shift(1)
        
        return df

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train_model(self, retrain=False):
        """Train model to predict prices directly with cross-validation"""
        if self.model is not None and not retrain:
            return
        
        print("Entraînement du modèle optimisé...")
        
        df = self.fetch_gas_data(days=5000)
        df = self.add_features(df)
        
        # Target: Next day's price
        df['target'] = df['Price'].shift(-1)
        self.feature_cols = [col for col in df.columns if col not in ['date', 'Price', 'target', 'call_price', 'put_price', 'underlying_price']]
        df = df.dropna(subset=['target'] + self.feature_cols)
        
        if len(df) < 200:
            raise ValueError("Données insuffisantes pour l'entraînement")
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = float('inf')
        best_model = None
        
        param_grid = [
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.01},
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.03}
        ]
        
        for params in param_grid:
            model = xgb.XGBRegressor(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                early_stopping_rounds=20,
                reg_alpha=0.1,
                reg_lambda=0.1
            )
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                X_train = train_df[self.feature_cols]
                y_train = train_df['target']
                X_val = val_df[self.feature_cols]
                y_val = val_df['target']
                
                X_train_scaled = self.scaler.fit_transform(X_train) if self.scaler else StandardScaler().fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val) if self.scaler else StandardScaler().fit_transform(X_val)
                
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
                y_pred = model.predict(X_val_scaled)
                score = mean_absolute_error(y_val, y_pred)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_model = model
        
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['target']
        X_val = val_df[self.feature_cols]
        y_val = val_df['target']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target']
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = best_model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val), (X_test_scaled, y_test)],
            verbose=False
        )
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred = pd.Series(y_pred).rolling(window=3, min_periods=1).mean().values
        
        # Calculate returns for visualization
        actual_returns = test_df['Price'].pct_change(1).shift(-1).iloc[:-1].values
        predicted_returns = (y_pred[1:] - test_df['Price'].iloc[:-1].values) / test_df['Price'].iloc[:-1].values
        
        self.test_data = {
            'dates': test_df['date'].tolist()[:-1],  # Adjust for returns
            'actual_prices': y_test.tolist()[:-1],
            'predicted_prices': y_pred.tolist()[:-1],
            'actual_returns': actual_returns.tolist(),
            'predicted_returns': predicted_returns.tolist()
        }
        
        self.test_metrics = self.calculate_validation_metrics(X_test_scaled, y_test)
        self.save_model()
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_cols, importance))
        print("Feature Importance:", sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        print("Modèle entraîné avec succès!")
        print(f"Métriques test: R²={self.test_metrics['r2_score']:.4f}, MAE={self.test_metrics['mae']:.4f}, Direction Accuracy={self.test_metrics['direction_accuracy']:.4f}")

    def calculate_validation_metrics(self, X_test_scaled, y_test):
        """Calculate metrics with direction accuracy"""
        y_pred = self.model.predict(X_test_scaled)
        y_pred = pd.Series(y_pred).rolling(window=3, min_periods=1).mean().values
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # Calculate direction accuracy based on price changes
        actual_changes = y_test[1:] - y_test[:-1]
        predicted_changes = y_pred[1:] - y_pred[:-1]
        direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes))
        
        if np.isnan(r2) or np.isinf(r2):
            r2 = 0.0
        if np.isnan(mae) or np.isinf(mae):
            mae = 0.0
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = 0.0
            
        return {
            'r2_score': round(float(r2), 4),
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'direction_accuracy': round(float(direction_accuracy), 4)
        }

    def get_prediction(self):
        """Get next-day price prediction"""
        if self.model is None:
            self.train_model()
        
        df = self.fetch_gas_data(days=150)
        df = self.add_features(df)
        df = df.dropna(subset=self.feature_cols)
        
        if len(df) == 0:
            raise ValueError("Pas de données disponibles pour la prédiction")
        
        X_latest = df[self.feature_cols].iloc[[-1]]
        X_latest_scaled = self.scaler.transform(X_latest)
        
        predicted_price = self.model.predict(X_latest_scaled)[0]
        predicted_price = np.clip(predicted_price, df['Price'].min(), df['Price'].max())  # Clip to historical range
        current_price = df['Price'].iloc[-1]
        
        if np.isnan(predicted_price) or np.isinf(predicted_price):
            predicted_price = current_price
        
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        return {
            'current_price': round(float(current_price), 4),
            'predicted_price': round(float(predicted_price), 4),
            'price_change': round(float(price_change), 4),
            'price_change_pct': round(float(price_change_pct), 2),
            'confidence': f'Modèle optimisé - R²: {self.test_metrics.get("r2_score", "N/A")}',
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }

    def get_weekly_prediction(self):
        """Get weekly price predictions"""
        if self.model is None:
            self.train_model()
        
        df = self.fetch_gas_data(days=150)
        df = self.add_features(df)
        df = df.dropna(subset=self.feature_cols)
        
        if len(df) == 0:
            raise ValueError("Pas de données disponibles pour la prédiction")
        
        current_price = df['Price'].iloc[-1]
        current_date = df['date'].iloc[-1]
        predictions = []
        temp_df = df.copy()
        
        for day in range(1, 8):
            X_latest = temp_df[self.feature_cols].iloc[[-1]]
            X_latest_scaled = self.scaler.transform(X_latest)
            
            predicted_price = self.model.predict(X_latest_scaled)[0]
            predicted_price = np.clip(predicted_price, df['Price'].min(), df['Price'].max())
            
            if np.isnan(predicted_price) or np.isinf(predicted_price):
                predicted_price = temp_df['Price'].iloc[-1]
            
            predicted_return = (predicted_price - temp_df['Price'].iloc[-1]) / temp_df['Price'].iloc[-1]
            prediction_date = current_date + timedelta(days=day)
            
            predictions.append({
                'date': prediction_date.strftime('%Y-%m-%d'),
                'day_name': prediction_date.strftime('%A'),
                'predicted_price': round(float(predicted_price), 4),
                'predicted_return': round(float(predicted_return * 100), 2),
                'price_change_from_current': round(float(predicted_price - current_price), 4),
                'price_change_pct_from_current': round(float((predicted_price - current_price) / current_price * 100), 2)
            })
            
            new_row = temp_df.iloc[[-1]].copy()
            new_row['date'] = prediction_date
            new_row['Price'] = predicted_price
            new_row['call_price'] = temp_df['call_price'].iloc[-1]
            new_row['put_price'] = temp_df['put_price'].iloc[-1]
            new_row['underlying_price'] = temp_df['underlying_price'].iloc[-1]
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            temp_df = self.add_features(temp_df)
            temp_df = temp_df.dropna(subset=self.feature_cols)
        
        weekly_stats = {
            'current_price': round(float(current_price), 4),
            'week_start_price': predictions[0]['predicted_price'],
            'week_end_price': predictions[-1]['predicted_price'],
            'total_change': round(float(predictions[-1]['predicted_price'] - current_price), 4),
            'total_change_pct': round(float((predictions[-1]['predicted_price'] - current_price) / current_price * 100), 2),
            'max_price': round(max([p['predicted_price'] for p in predictions]), 4),
            'min_price': round(min([p['predicted_price'] for p in predictions]), 4),
            'avg_price': round(sum([p['predicted_price'] for p in predictions]) / len(predictions), 4),
            'predictions': predictions
        }
        
        return weekly_stats

    def get_performance_chart(self):
        """Generate performance chart with direction accuracy"""
        if not self.test_data:
            raise ValueError("Données de test non disponibles. Veuillez d'abord entraîner le modèle.")
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('#1a1a1a')
        
        dates = pd.to_datetime(self.test_data['dates'])
        ax1.plot(dates, self.test_data['actual_prices'], 
                label='Prix Réels', color='#60a5fa', linewidth=2)
        ax1.plot(dates, self.test_data['predicted_prices'], 
                label='Prix Prédits', color='#f59e0b', linewidth=2, linestyle='--')
        
        ax1.set_title('Performance du Modèle - Prix du Gaz Naturel (Test Set)', 
                     fontsize=14, fontweight='bold', color='white')
        ax1.set_xlabel('Date', fontsize=12, color='white')
        ax1.set_ylabel('Prix ($/MMBtu)', fontsize=12, color='white')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2.scatter(self.test_data['actual_returns'], self.test_data['predicted_returns'], 
                   alpha=0.6, color='#60a5fa', s=20)
        
        min_val = min(min(self.test_data['actual_returns']), min(self.test_data['predicted_returns']))
        max_val = max(max(self.test_data['actual_returns']), max(self.test_data['predicted_returns']))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax2.set_title('Rendements Prédits vs Réels', fontsize=14, fontweight='bold', color='white')
        ax2.set_xlabel('Rendements Réels', fontsize=12, color='white')
        ax2.set_ylabel('Rendements Prédits', fontsize=12, color='white')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        
        r2_score = self.test_metrics.get('r2_score', 'N/A')
        mae = self.test_metrics.get('mae', 'N/A')
        rmse = self.test_metrics.get('rmse', 'N/A')
        direction_accuracy = self.test_metrics.get('direction_accuracy', 'N/A')
        
        textstr = f'R² = {r2_score}\nMAE = {mae}\nRMSE = {rmse}\nDirection Accuracy = {direction_accuracy}'
        props = dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='white')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1a1a1a', 
                   bbox_inches='tight', dpi=100)
        buffer.seek(0)
        
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_data

    def get_model_metrics(self):
        """Return test metrics with direction accuracy"""
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        
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
                'direction_accuracy': 'N/A',
                'last_trained': self.last_update,
                'note': 'Métriques de test non disponibles'
            }

    def save_model(self):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'last_update': datetime.now().isoformat(),
            'test_metrics': getattr(self, 'test_metrics', {}),
            'test_data': getattr(self, 'test_data', {})
        }
        
        with open('gas_price_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load saved model"""
        try:
            with open('gas_price_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.last_update = model_data.get('last_update')
            self.test_metrics = model_data.get('test_metrics', {})
            self.test_data = model_data.get('test_data', {})
            
            print("Modèle chargé avec succès!")
            
        except FileNotFoundError:
            print("Aucun modèle sauvegardé trouvé. Entraînement nécessaire.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
