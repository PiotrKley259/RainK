import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

warnings.filterwarnings('ignore')

class PairTradingAnalyzer:
    def __init__(self):
        self.cache_duration = 3600  # 1 hour in seconds
        self.user_cache = {}  # {session_id: {'analysis_data': {...}, 'last_update': datetime}}

    def validate_ticker(self, ticker):
        """Validate if ticker exists and has sufficient data"""
        try:
            # Clean up ticker input
            ticker = ticker.upper().strip()
            
            # Try to fetch a small amount of data to validate
            test_data = yf.download(ticker, period='5d', progress=False)
            
            if test_data.empty:
                return False, f"No data found for ticker '{ticker}'"
            
            # Get company info if available
            try:
                stock_info = yf.Ticker(ticker)
                company_name = stock_info.info.get('longName', ticker)
            except:
                company_name = ticker
                
            return True, company_name
            
        except Exception as e:
            return False, f"Invalid ticker '{ticker}': {str(e)}"

    def fetch_stock_data(self, tickers, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {tickers} over {period}")
            data = yf.download(tickers, period=period, progress=False)['Adj Close']
            
            if len(tickers) == 1:
                data = data.to_frame()
                data.columns = tickers
            
            # Remove any missing data
            data = data.dropna()
            
            if len(data) < 50:  # Minimum data points required
                raise ValueError("Insufficient data points for analysis")
                
            print(f"Successfully fetched {len(data)} data points")
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")

    def calculate_correlation_metrics(self, data, ticker1, ticker2):
        """Calculate comprehensive correlation metrics"""
        try:
            # Basic correlation
            correlation = data[ticker1].corr(data[ticker2])
            
            # Calculate returns
            returns1 = data[ticker1].pct_change().dropna()
            returns2 = data[ticker2].pct_change().dropna()
            
            # Rolling correlation (30-day window)
            rolling_corr = data[ticker1].rolling(window=30).corr(data[ticker2]).dropna()
            
            # Volatility measures
            volatility1 = returns1.std() * np.sqrt(252)  # Annualized
            volatility2 = returns2.std() * np.sqrt(252)
            
            # Beta calculation (ticker2 as market)
            X = returns2.values.reshape(-1, 1)
            y = returns1.values
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            alpha = model.intercept_
            r_squared = model.score(X, y)
            
            return {
                'correlation': round(float(correlation), 4),
                'rolling_correlation_mean': round(float(rolling_corr.mean()), 4),
                'rolling_correlation_std': round(float(rolling_corr.std()), 4),
                'volatility1': round(float(volatility1), 4),
                'volatility2': round(float(volatility2), 4),
                'beta': round(float(beta), 4),
                'alpha': round(float(alpha), 6),
                'r_squared': round(float(r_squared), 4),
                'rolling_correlation': rolling_corr.tolist()
            }
            
        except Exception as e:
            raise Exception(f"Error calculating correlation metrics: {str(e)}")

    def test_cointegration(self, data, ticker1, ticker2):
        """Test for cointegration using Augmented Dickey-Fuller test"""
        try:
            # Calculate the spread
            X = data[ticker2].values.reshape(-1, 1)
            y = data[ticker1].values
            model = LinearRegression().fit(X, y)
            spread = y - model.predict(X)
            
            # ADF test on the spread
            adf_result = adfuller(spread, maxlag=1)
            
            # Calculate half-life of mean reversion
            spread_lag = np.roll(spread, 1)[1:]
            spread_diff = np.diff(spread)
            model_halflife = LinearRegression().fit(spread_lag.reshape(-1, 1), spread_diff)
            half_life = -np.log(2) / model_halflife.coef_[0] if model_halflife.coef_[0] != 0 else np.inf
            
            return {
                'adf_statistic': round(float(adf_result[0]), 4),
                'adf_pvalue': round(float(adf_result[1]), 4),
                'adf_critical_values': {k: round(v, 4) for k, v in adf_result[4].items()},
                'is_cointegrated': adf_result[1] < 0.05,  # 5% significance level
                'spread': spread.tolist(),
                'half_life': round(float(half_life), 2) if half_life != np.inf else 'inf',
                'hedge_ratio': round(float(model.coef_[0]), 4)
            }
            
        except Exception as e:
            raise Exception(f"Error testing cointegration: {str(e)}")

    def analyze_pair_trading_opportunity(self, correlation_metrics, cointegration_results):
        """Analyze if pair trading is viable and provide recommendations"""
        try:
            score = 0
            recommendations = []
            
            # Correlation analysis
            corr = abs(correlation_metrics['correlation'])
            if corr > 0.7:
                score += 30
                recommendations.append(f"✓ Strong correlation ({corr:.3f}) - Good for pair trading")
            elif corr > 0.5:
                score += 20
                recommendations.append(f"⚠ Moderate correlation ({corr:.3f}) - Acceptable but monitor closely")
            else:
                score += 0
                recommendations.append(f"✗ Weak correlation ({corr:.3f}) - Not ideal for pair trading")
            
            # Cointegration analysis
            if cointegration_results['is_cointegrated']:
                score += 40
                recommendations.append("✓ Stocks are cointegrated - Excellent for pair trading")
            else:
                score += 10
                recommendations.append("⚠ No cointegration detected - Higher risk strategy")
            
            # Volatility analysis
            vol_ratio = max(correlation_metrics['volatility1'], correlation_metrics['volatility2']) / \
                       min(correlation_metrics['volatility1'], correlation_metrics['volatility2'])
            
            if vol_ratio < 2:
                score += 20
                recommendations.append(f"✓ Similar volatility levels (ratio: {vol_ratio:.2f})")
            else:
                score += 5
                recommendations.append(f"⚠ High volatility difference (ratio: {vol_ratio:.2f})")
            
            # Half-life analysis
            if cointegration_results['half_life'] != 'inf' and cointegration_results['half_life'] < 30:
                score += 10
                recommendations.append(f"✓ Fast mean reversion (half-life: {cointegration_results['half_life']} days)")
            elif cointegration_results['half_life'] != 'inf':
                score += 5
                recommendations.append(f"⚠ Slow mean reversion (half-life: {cointegration_results['half_life']} days)")
            
            # Overall assessment
            if score >= 80:
                overall_rating = "Excellent"
                trading_recommendation = "Highly recommended for pair trading strategy"
            elif score >= 60:
                overall_rating = "Good"
                trading_recommendation = "Good candidate for pair trading with careful monitoring"
            elif score >= 40:
                overall_rating = "Moderate"
                trading_recommendation = "Possible but requires careful risk management"
            else:
                overall_rating = "Poor"
                trading_recommendation = "Not recommended for pair trading"
            
            return {
                'score': score,
                'max_score': 100,
                'overall_rating': overall_rating,
                'trading_recommendation': trading_recommendation,
                'detailed_recommendations': recommendations
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing pair trading opportunity: {str(e)}")

    def perform_pair_analysis(self, session_id, ticker1, ticker2, period='1y'):
        """Perform comprehensive pair trading analysis"""
        # Check cache first
        cache_key = f"{ticker1}_{ticker2}_{period}"
        if (session_id in self.user_cache and 
            cache_key in self.user_cache[session_id] and
            (datetime.now() - self.user_cache[session_id][cache_key]['last_update']).seconds < self.cache_duration):
            print(f"Using cached analysis for {ticker1} vs {ticker2}")
            return self.user_cache[session_id][cache_key]['analysis_data']
        
        try:
            # Validate tickers
            valid1, name1 = self.validate_ticker(ticker1)
            valid2, name2 = self.validate_ticker(ticker2)
            
            if not valid1:
                raise ValueError(name1)
            if not valid2:
                raise ValueError(name2)
            
            # Fetch data
            tickers = [ticker1.upper(), ticker2.upper()]
            data = self.fetch_stock_data(tickers, period)
            
            # Ensure we have both tickers in data
            if ticker1.upper() not in data.columns or ticker2.upper() not in data.columns:
                raise ValueError("Missing data for one or both tickers")
            
            # Calculate metrics
            correlation_metrics = self.calculate_correlation_metrics(data, ticker1.upper(), ticker2.upper())
            cointegration_results = self.test_cointegration(data, ticker1.upper(), ticker2.upper())
            trading_analysis = self.analyze_pair_trading_opportunity(correlation_metrics, cointegration_results)
            
            # Prepare final results
            analysis_data = {
                'ticker1': ticker1.upper(),
                'ticker2': ticker2.upper(),
                'company1': name1,
                'company2': name2,
                'period': period,
                'data_points': len(data),
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'correlation_metrics': correlation_metrics,
                'cointegration_results': cointegration_results,
                'trading_analysis': trading_analysis,
                'price_data': {
                    'dates': [d.strftime('%Y-%m-%d') for d in data.index],
                    'prices1': data[ticker1.upper()].tolist(),
                    'prices2': data[ticker2.upper()].tolist()
                }
            }
            
            # Cache the results
            if session_id not in self.user_cache:
                self.user_cache[session_id] = {}
            
            self.user_cache[session_id][cache_key] = {
                'analysis_data': analysis_data,
                'last_update': datetime.now()
            }
            
            return analysis_data
            
        except Exception as e:
            raise Exception(f"Error performing pair analysis: {str(e)}")

    def generate_analysis_chart(self, session_id, ticker1, ticker2, period='1y'):
        """Generate comprehensive analysis charts"""
        try:
            # Get analysis data
            cache_key = f"{ticker1}_{ticker2}_{period}"
            if session_id not in self.user_cache or cache_key not in self.user_cache[session_id]:
                raise ValueError("Analysis data not available. Please run analysis first.")
            
            analysis_data = self.user_cache[session_id][cache_key]['analysis_data']
            
            # Setup the plot
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 12))
            fig.patch.set_facecolor('white')
            
            # Create subplot layout
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.3)
            
            # Prepare data
            dates = pd.to_datetime(analysis_data['price_data']['dates'])
            prices1 = np.array(analysis_data['price_data']['prices1'])
            prices2 = np.array(analysis_data['price_data']['prices2'])
            
            # Plot 1: Price comparison (normalized)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.set_facecolor('#fafafa')
            
            # Normalize prices to start at 100
            norm_prices1 = 100 * prices1 / prices1[0]
            norm_prices2 = 100 * prices2 / prices2[0]
            
            ax1.plot(dates, norm_prices1, label=f'{ticker1} ({analysis_data["company1"]})', 
                    color='#3b82f6', linewidth=2)
            ax1.plot(dates, norm_prices2, label=f'{ticker2} ({analysis_data["company2"]})', 
                    color='#ef4444', linewidth=2)
            
            ax1.set_title(f'Normalized Price Comparison: {ticker1} vs {ticker2}', 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
            ax1.legend(fontsize=11, loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='#4b5563', labelsize=10)
            
            # Plot 2: Rolling correlation
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_facecolor('#fafafa')
            
            rolling_corr = analysis_data['correlation_metrics']['rolling_correlation']
            corr_dates = dates[29:]  # Skip first 29 days for 30-day rolling
            
            ax2.plot(corr_dates, rolling_corr, color='#8b5cf6', linewidth=2)
            ax2.axhline(y=0.7, color='#10b981', linestyle='--', alpha=0.7, label='Strong correlation (0.7)')
            ax2.axhline(y=0.5, color='#f59e0b', linestyle='--', alpha=0.7, label='Moderate correlation (0.5)')
            ax2.set_title('30-Day Rolling Correlation', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Correlation', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='#4b5563', labelsize=9)
            
            # Plot 3: Spread analysis
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.set_facecolor('#fafafa')
            
            spread = analysis_data['cointegration_results']['spread']
            ax3.plot(dates, spread, color='#dc2626', linewidth=1.5, alpha=0.8)
            ax3.axhline(y=0, color='#374151', linestyle='-', alpha=0.5)
            
            # Add mean and standard deviation lines
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            ax3.axhline(y=spread_mean, color='#6366f1', linestyle='--', alpha=0.7, label='Mean')
            ax3.axhline(y=spread_mean + 2*spread_std, color='#f59e0b', linestyle=':', alpha=0.7, label='+2σ')
            ax3.axhline(y=spread_mean - 2*spread_std, color='#f59e0b', linestyle=':', alpha=0.7, label='-2σ')
            
            ax3.set_title('Price Spread Analysis', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Spread', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(colors='#4b5563', labelsize=9)
            
            # Plot 4: Summary statistics table
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            
            # Prepare summary data
            corr_metrics = analysis_data['correlation_metrics']
            coint_results = analysis_data['cointegration_results']
            trading_analysis = analysis_data['trading_analysis']
            
            summary_data = [
                ['Metric', 'Value', 'Interpretation'],
                ['Correlation', f"{corr_metrics['correlation']:.4f}", 
                 'Strong' if abs(corr_metrics['correlation']) > 0.7 else 'Moderate' if abs(corr_metrics['correlation']) > 0.5 else 'Weak'],
                ['Cointegrated', 'Yes' if coint_results['is_cointegrated'] else 'No', 
                 'Good for pairs trading' if coint_results['is_cointegrated'] else 'Higher risk'],
                ['ADF p-value', f"{coint_results['adf_pvalue']:.4f}", 
                 'Statistically significant' if coint_results['adf_pvalue'] < 0.05 else 'Not significant'],
                ['Half-life', f"{coint_results['half_life']}", 
                 'Fast reversion' if coint_results['half_life'] != 'inf' and float(coint_results['half_life']) < 30 else 'Slow reversion'],
                ['Overall Score', f"{trading_analysis['score']}/100", trading_analysis['overall_rating']],
                ['Recommendation', trading_analysis['trading_recommendation'][:50] + '...', '']
            ]
            
            # Create table
            table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                             cellLoc='left', loc='center', colWidths=[0.2, 0.2, 0.6])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            
            # Style the table
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#3b82f6')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(summary_data)):
                for j in range(len(summary_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f8fafc')
                    else:
                        table[(i, j)].set_facecolor('white')
                    table[(i, j)].set_edgecolor('#e2e8f0')
            
            # Format dates on x-axis
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='white', 
                       bbox_inches='tight', dpi=300, edgecolor='none')
            buffer.seek(0)
            
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            raise Exception(f"Error generating analysis chart: {str(e)}")

    def get_analysis_status(self, session_id):
        """Get current analysis status for a session"""
        if session_id not in self.user_cache:
            return {'has_analysis': False, 'analyses_count': 0}
        
        return {
            'has_analysis': len(self.user_cache[session_id]) > 0,
            'analyses_count': len(self.user_cache[session_id]),
            'cached_pairs': list(self.user_cache[session_id].keys())
        }