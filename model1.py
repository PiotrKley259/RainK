import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

warnings.filterwarnings('ignore')

class GasPricePredictor:
    def __init__(self, api_key=None):
        # Configuration par défaut
        self.api_key = api_key or 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe'
        self.last_update = None
        self.cache_duration = 3600  # 1 hour in seconds
        
        # Cache global pour les données API
        self.global_data_cache = {}
        self.global_cache_timestamp = None
        
        # Cache par utilisateur pour les prédictions
        self.user_cache = {}  # {session_id: {'test_data': {...}, 'last_update': datetime}}

    def fetch_gas_data(self, days=5000):
        """Fetch gas price data from EIA API with better error handling"""
        if (self.global_cache_timestamp and 
            (datetime.now() - self.global_cache_timestamp).seconds < self.cache_duration and
            'gas_data' in self.global_data_cache):
            print("Utilisation des données globales en cache")
            return self.global_data_cache['gas_data']
        
        if not self.api_key:
            raise ValueError("Clé API EIA manquante")
        
        print(f"Récupération des données de prix du gaz naturel ({days} jours)...")
        
        url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
        params = {
            "api_key": self.api_key,
            "frequency": "daily",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": min(days, 5000)
        }

        try:
            print("Appel à l'API EIA...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'response' not in data or 'data' not in data['response']:
                raise ValueError("Format de réponse API invalide")
            
            records = data["response"]["data"]
            
            if not records:
                raise ValueError("Aucune donnée retournée par l'API")
            
            df = pd.DataFrame(records)
            print(f"Données récupérées: {len(df)} enregistrements")

            # Nettoyage des données
            df = df[["period", "value"]]
            df = df.rename(columns={"period": "date", "value": "Price"})
            df["date"] = pd.to_datetime(df["date"])
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

            # Supprimer les valeurs aberrantes
            Q1 = df["Price"].quantile(0.25)
            Q3 = df["Price"].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df["Price"] >= Q1 - 1.5 * IQR) & (df["Price"] <= Q3 + 1.5 * IQR)]

            # Nettoyage final
            df = df.dropna(subset=["Price"])
            df = df.groupby("date", as_index=False).mean()
            df = df.sort_values("date").reset_index(drop=True)
            
            # Interpolation des valeurs manquantes
            df["Price"] = df["Price"].interpolate(method="linear")
            
            # Mise en cache global
            self.global_data_cache['gas_data'] = df
            self.global_cache_timestamp = datetime.now()
            
            print(f"Données finales: {len(df)} enregistrements après nettoyage")
            return df
            
        except requests.exceptions.Timeout:
            raise Exception("Timeout lors de la récupération des données. Veuillez réessayer.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur réseau lors de la récupération des données: {str(e)}")
        except Exception as e:
            raise Exception(f"Erreur lors du traitement des données: {str(e)}")

    def estimate_bsm_parameters(self, df, window=20):
        """Estimate BSM parameters (drift and volatility) from historical data"""
        returns = df['Price'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)  # Annualized volatility
        drift = returns.mean() * 252  # Annualized drift
        return drift, volatility

    def simulate_bsm_paths(self, S0, T=7/252, n_steps=7, n_paths=1000, r=0.02, sigma=None):
        """Simulate price paths using Black-Scholes-Merton (Geometric Brownian Motion)"""
        if sigma is None:
            df = self.fetch_gas_data(days=200)
            _, sigma = self.estimate_bsm_parameters(df)
        
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            z = np.random.normal(0, 1, n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths

    def get_weekly_prediction(self, session_id):
        """Get weekly price predictions using BSM simulation for a specific user"""
        # Vérifier si les prédictions existent dans le cache utilisateur
        if (session_id in self.user_cache and 
            self.user_cache[session_id]['last_update'] and
            (datetime.now() - self.user_cache[session_id]['last_update']).seconds < self.cache_duration):
            print(f"Utilisation des prédictions en cache pour la session {session_id}")
            return self.user_cache[session_id]['weekly_stats']
        
        try:
            df = self.fetch_gas_data(days=200)
            if len(df) == 0:
                raise ValueError("Pas de données disponibles pour la prédiction")
            
            current_price = df['Price'].iloc[-1]
            current_date = df['date'].iloc[-1]
            drift, volatility = self.estimate_bsm_parameters(df)
            
            # Simulate 1000 price paths for the next 7 days
            n_paths = 1000
            paths = self.simulate_bsm_paths(
                S0=current_price,
                T=7/252,  # 7 days in years
                n_steps=7,
                n_paths=n_paths,
                r=0.02,  # Risk-free rate
                sigma=volatility
            )
            
            # Generate dates for the next 7 days
            dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(8)]
            day_names = [(current_date + timedelta(days=i)).strftime('%A') for i in range(8)]
            
            # Calculate statistics for each day
            predictions = []
            for day in range(1, 8):
                day_prices = paths[:, day]
                mean_price = np.mean(day_prices)
                predictions.append({
                    'date': dates[day],
                    'day_name': day_names[day],
                    'predicted_price': round(float(mean_price), 4),
                    'price_change_from_current': round(float(mean_price - current_price), 4),
                    'price_change_pct_from_current': round(float((mean_price - current_price) / current_price * 100), 2)
                })
            
            # Calculate distribution of final prices
            final_prices = paths[:, -1]
            price_distribution = {
                'mean': round(float(np.mean(final_prices)), 4),
                'std': round(float(np.std(final_prices)), 4),
                'percentiles': {
                    '5': round(float(np.percentile(final_prices, 5)), 4),
                    '25': round(float(np.percentile(final_prices, 25)), 4),
                    '50': round(float(np.percentile(final_prices, 50)), 4),
                    '75': round(float(np.percentile(final_prices, 75)), 4),
                    '95': round(float(np.percentile(final_prices, 95)), 4)
                }
            }
            
            weekly_stats = {
                'current_price': round(float(current_price), 4),
                'week_start_price': round(float(np.mean(paths[:, 1])), 4),
                'week_end_price': round(float(np.mean(paths[:, -1])), 4),
                'total_change': round(float(np.mean(paths[:, -1]) - current_price), 4),
                'total_change_pct': round(float((np.mean(paths[:, -1]) - current_price) / current_price * 100), 2),
                'max_price': round(float(np.max(paths[:, 1:])), 4),
                'min_price': round(float(np.min(paths[:, 1:])), 4),
                'avg_price': round(float(np.mean(paths[:, 1:])), 4),
                'volatility': round(float(volatility), 4),
                'predictions': predictions,
                'price_paths': paths.tolist(),
                'price_distribution': price_distribution
            }
            
            # Afficher les prédictions journalières pour vérification
            print("\nPrédictions journalières :")
            for pred in predictions:
                print(f"{pred['date']} ({pred['day_name']}): ${pred['predicted_price']}, Variation: {pred['price_change_pct_from_current']}%")
            
            # Stocker les données dans le cache utilisateur
            self.user_cache[session_id] = {
                'weekly_stats': weekly_stats,
                'test_data': {
                    'dates': dates[1:],
                    'price_paths': paths.tolist(),
                    'price_distribution': price_distribution
                },
                'last_update': datetime.now()
            }
            
            self.last_update = datetime.now().isoformat()
            return weekly_stats
        
        except Exception as e:
            print(f"Erreur lors de la prédiction hebdomadaire pour la session {session_id}: {str(e)}")
            raise

    def get_performance_chart(self, session_id):
        """Generate chart showing simulated price paths and distribution, styled like the example"""
        if session_id not in self.user_cache or not self.user_cache[session_id].get('test_data'):
            raise ValueError("Données de simulation non disponibles. Veuillez d'abord exécuter get_weekly_prediction.")
        
        try:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(12, 6))
            fig.patch.set_facecolor('#1a1a1a')
            ax = fig.add_subplot(111)
            
            # Récupérer les données de l'utilisateur
            test_data = self.user_cache[session_id]['test_data']
            dates = pd.to_datetime(test_data['dates'])
            paths = np.array(test_data['price_paths'])
            
            # Tracer les trajectoires (jusqu'à 100 pour la lisibilité)
            for i in range(min(100, len(paths))):
                ax.plot(dates, paths[i, 1:], color='#60a5fa', alpha=0.1, linewidth=1)
            
            # Tracer la moyenne des trajectoires
            mean_prices = np.mean(paths[:, 1:], axis=0)
            ax.plot(dates, mean_prices, color='#f59e0b', linewidth=3, label='Moyenne des Trajectoires')
            
            ax.set_title('Trajectoires Simulées des Prix du Gaz Naturel (BSM)', 
                        fontsize=16, fontweight='bold', color='white', pad=20)
            ax.set_xlabel('Date', fontsize=12, color='white')
            ax.set_ylabel('Prix ($/MMBtu)', fontsize=12, color='white')
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Ajouter un encadré avec les statistiques
            dist_stats = test_data['price_distribution']
            metrics_text = (f"Moyenne finale: ${dist_stats['mean']:.2f}\n"
                          f"Écart-type: ${dist_stats['std']:.2f}\n"
                          f"5e percentile: ${dist_stats['percentiles']['5']:.2f}\n"
                          f"95e percentile: ${dist_stats['percentiles']['95']:.2f}")
            
            props = dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', alpha=0.8, edgecolor='white')
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, color='white')
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1a1a1a', 
                       bbox_inches='tight', dpi=150, edgecolor='none')
            buffer.seek(0)
            
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            print(f"Erreur lors de la génération du graphique pour la session {session_id}: {str(e)}")
            raise

    def get_model_metrics(self, session_id):
        """Return comprehensive model metrics for a specific user"""
        if session_id not in self.user_cache or not self.user_cache[session_id].get('test_data'):
            return {
                'status': 'non_simulated',
                'message': 'Aucune simulation effectuée'
            }
        
        return {
            'status': 'simulated',
            'metrics': self.user_cache[session_id]['test_data'].get('price_distribution', {}),
            'last_simulated': self.last_update,
            'model_type': 'Black-Scholes-Merton (GBM)',
            'data_points': len(self.user_cache[session_id]['test_data'].get('dates', []))
        }

    def save_model_state(self, session_id):
        """Save model state as JSON for a specific user"""
        if session_id not in self.user_cache or not self.user_cache[session_id].get('test_data'):
            return None
        
        try:
            model_state = {
                'test_metrics': self.user_cache[session_id]['test_data'].get('price_distribution', {}),
                'last_update': self.last_update,
                'test_data': self.user_cache[session_id]['test_data'],
                'model_simulated': True
            }
            
            return json.dumps(model_state, indent=2)
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde pour la session {session_id}: {str(e)}")
            return None

    def get_status(self, session_id):
        """Get current predictor status for a specific user"""
        return {
            'model_simulated': session_id in self.user_cache and bool(self.user_cache[session_id].get('test_data')),
            'last_update': self.last_update,
            'cache_active': self.global_cache_timestamp is not None,
            'api_key_configured': bool(self.api_key)
        }
