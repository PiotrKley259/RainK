import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta, date
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
            current_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


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
        """Generate chart showing simulated price paths and distribution with improved white background styling"""
        if session_id not in self.user_cache or not self.user_cache[session_id].get('test_data'):
            raise ValueError("Données de simulation non disponibles. Veuillez d'abord exécuter get_weekly_prediction.")
        
        try:
            # Configuration du style moderne avec fond blanc
            plt.style.use('default')
            fig = plt.figure(figsize=(14, 8))
            fig.patch.set_facecolor('white')
            
            # Création d'un layout avec grille pour plus de sophistication
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1], 
                                hspace=0.3, wspace=0.3)
            
            # Graphique principal des trajectoires
            ax_main = fig.add_subplot(gs[0, 0])
            ax_main.set_facecolor('#fafafa')
            
            # Récupérer les données de l'utilisateur
            test_data = self.user_cache[session_id]['test_data']
            dates = pd.to_datetime(test_data['dates'])
            paths = np.array(test_data['price_paths'])
            
            # Tracer les trajectoires individuelles avec un dégradé de couleurs
            n_paths_to_show = min(150, len(paths))
            colors = plt.cm.Blues(np.linspace(0.3, 0.7, n_paths_to_show))
            
            for i in range(n_paths_to_show):
                ax_main.plot(dates, paths[i, 1:], color=colors[i], alpha=0.4, linewidth=0.8)
            
            # Calculer les percentiles pour la zone de confiance
            percentiles = np.percentile(paths[:, 1:], [10, 25, 75, 90], axis=0)
            
            # Zone de confiance 80% (10e-90e percentile)
            ax_main.fill_between(dates, percentiles[0], percentiles[3], 
                               alpha=0.2, color='#3b82f6', label='Confidence zone 80%')
            
            # Zone de confiance 50% (25e-75e percentile)
            ax_main.fill_between(dates, percentiles[1], percentiles[2], 
                               alpha=0.3, color='#1d4ed8', label='Confidence zone 50%')
            
            # Tracer la médiane des trajectoires
            median_prices = np.median(paths[:, 1:], axis=0)
            ax_main.plot(dates, median_prices, color='#dc2626', linewidth=3, 
                        label='Median of the trajectories', linestyle='-')
            
            # Tracer la moyenne des trajectoires
            mean_prices = np.mean(paths[:, 1:], axis=0)
            ax_main.plot(dates, mean_prices, color='#f59e0b', linewidth=3, 
                        label='Mean of trajectories', linestyle='--')
            
            # Styling du graphique principal
            ax_main.set_title('Simulated trajectories of Natural Gas Price\n(Modèle Black-Scholes-Merton)', 
                            fontsize=16, fontweight='bold', color='#1f2937', pad=20)
            ax_main.set_xlabel('Date', fontsize=12, color='#374151', fontweight='500')
            ax_main.set_ylabel('Price ($/MMBtu)', fontsize=12, color='#374151', fontweight='500')
            ax_main.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, shadow=True)
            
            # Grille moderne
            ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#9ca3af')
            ax_main.set_axisbelow(True)
            
            # Couleurs des ticks
            ax_main.tick_params(colors='#4b5563', labelsize=10)
            
            # Format des dates
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Histogramme des prix finaux
            ax_hist = fig.add_subplot(gs[0, 1])
            ax_hist.set_facecolor('#fafafa')
            
            final_prices = paths[:, -1]
            n_bins = 30
            counts, bins, patches = ax_hist.hist(final_prices, bins=n_bins, orientation='horizontal',
                                               alpha=0.7, color='#3b82f6', edgecolor='white', linewidth=0.5)
            
            # Colorer l'histogramme avec un dégradé
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.Blues(0.3 + 0.5 * i / len(patches)))
            
            ax_hist.set_xlabel('Frequency', fontsize=10, color='#374151')
            ax_hist.set_ylabel('Final Price ($/MMBtu)', fontsize=10, color='#374151')
            ax_hist.set_title('Distribution\n of Final Prices', fontsize=12, fontweight='bold', color='#1f2937')
            ax_hist.tick_params(colors='#4b5563', labelsize=9)
            ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#9ca3af')
            
            # Tableau des statistiques
            ax_stats = fig.add_subplot(gs[1, :])
            ax_stats.axis('off')
            
            # Préparer les données statistiques
            dist_stats = test_data['price_distribution']
            current_price = self.user_cache[session_id]['weekly_stats']['current_price']
            
            stats_data = [
                ['Metrics', 'Values', 'Variation vs Actual'],
                ['Actual Price', f"${current_price:.3f}", '-'],
                ['Final Mean Price', f"${dist_stats['mean']:.3f}", 
                 f"{((dist_stats['mean'] - current_price) / current_price * 100):+.2f}%"],
                ['Std', f"${dist_stats['std']:.3f}", '-'],
                ['5e Percentile', f"${dist_stats['percentiles']['5']:.3f}", 
                 f"{((dist_stats['percentiles']['5'] - current_price) / current_price * 100):+.2f}%"],
                ['95e Percentile', f"${dist_stats['percentiles']['95']:.3f}", 
                 f"{((dist_stats['percentiles']['95'] - current_price) / current_price * 100):+.2f}%"],
            ]
            
            # Créer le tableau
            table = ax_stats.table(cellText=stats_data[1:], colLabels=stats_data[0],
                                 cellLoc='center', loc='center', 
                                 colWidths=[0.25, 0.25, 0.25])
            
            # Style du tableau
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Colorer l'en-tête
            for i in range(len(stats_data[0])):
                table[(0, i)].set_facecolor('#3b82f6')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorer les cellules alternativement
            for i in range(1, len(stats_data)):
                for j in range(len(stats_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f8fafc')
                    else:
                        table[(i, j)].set_facecolor('white')
                    table[(i, j)].set_edgecolor('#e2e8f0')
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder le graphique
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='white', 
                       bbox_inches='tight', dpi=300, edgecolor='none')
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