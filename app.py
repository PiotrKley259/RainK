import os
import pandas as pd
import io
from datetime import datetime
from model1 import GasPricePredictor
from flask_session import Session
from flask import Flask, render_template, jsonify, send_file, session, request
from model2 import PairTradingAnalyzer

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Instance du modèle
predictor = GasPricePredictor()
pair_analyzer = PairTradingAnalyzer()

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/model1")
def model1():
    # Générer ou récupérer l'identifiant de session
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    try:
        # Get weekly predictions and performance chart
        weekly_stats = predictor.get_weekly_prediction(session['sid'])
        metrics = predictor.get_model_metrics(session['sid'])
        chart_data = predictor.get_performance_chart(session['sid'])

        return render_template(
            "model1.html",
            weekly_predictions=weekly_stats['predictions'],
            weekly_summary={
                'week_start_price': weekly_stats['week_start_price'],
                'week_end_price': weekly_stats['week_end_price'],
                'total_change': weekly_stats['total_change'],
                'total_change_pct': weekly_stats['total_change_pct'],
                'max_price': weekly_stats['max_price'],
                'min_price': weekly_stats['min_price'],
                'avg_price': weekly_stats['avg_price'],
                'volatility': weekly_stats['volatility']
            },
            metrics=metrics,
            chart_data=chart_data,
            error=None
        )
    except Exception as e:
        return render_template(
            "model1.html",
            weekly_predictions=[],
            weekly_summary={},
            metrics={'status': 'non_simulated', 'message': 'Aucune simulation effectuée'},
            chart_data=None,
            error=f"Erreur lors du chargement des prédictions ou du graphique: {str(e)}"
        )

@app.route("/api/predict", methods=['GET'])
def predict_gas_price():
    """API endpoint pour obtenir la prédiction"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    try:
        weekly_stats = predictor.get_weekly_prediction(session['sid'])
        
        return jsonify({
            "success": True,
            "weekly_predictions": weekly_stats['predictions'],
            "weekly_summary": {
                "week_start_price": weekly_stats['week_start_price'],
                "week_end_price": weekly_stats['week_end_price'],
                "total_change": weekly_stats['total_change'],
                "total_change_pct": weekly_stats['total_change_pct'],
                "max_price": weekly_stats['max_price'],
                "min_price": weekly_stats['min_price'],
                "avg_price": weekly_stats['avg_price'],
                "volatility": weekly_stats['volatility'],
                "price_distribution": weekly_stats['price_distribution']
            },
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/model-performance", methods=['GET'])
def model_performance():
    """API endpoint pour les métriques du modèle"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    try:
        metrics = predictor.get_model_metrics(session['sid'])
        return jsonify({
            "success": True,
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/download-report", methods=['GET'])
def download_report():
    """API endpoint pour télécharger le rapport"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    try:
        weekly_stats = predictor.get_weekly_prediction(session['sid'])
        metrics = predictor.get_model_metrics(session['sid'])
        
        # Create CSV report
        output = io.StringIO()
        output.write('Natural Gas Price Prediction Report\n')
        output.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        output.write('Weekly Predictions\n')
        df = pd.DataFrame(weekly_stats['predictions'])
        df.to_csv(output, index=False)
        output.write('\nWeekly Summary\n')
        for key, value in weekly_stats.items():
            if key not in ['predictions', 'price_paths', 'price_distribution']:
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        output.write('\nPrice Distribution\n')
        for key, value in weekly_stats['price_distribution'].items():
            if key == 'percentiles':
                for p, v in value.items():
                    output.write(f'{p}th Percentile: {v}\n')
            else:
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        output.write('\nModel Performance Metrics\n')
        for key, value in metrics.items():
            if key == 'metrics':
                for k, v in value.items():
                    if k == 'percentiles':
                        for p, pv in v.items():
                            output.write(f'{p}th Percentile: {pv}\n')
                    else:
                        output.write(f'{k.replace("_", " ").title()}: {v}\n')
            else:
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        
        # Prepare CSV file for download
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='gas_price_prediction_report.csv'
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/model2")
def model2():
    """Model 2 - Pair Trading Analysis Selection Page"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    return render_template("model2.html")

@app.route("/model2/analysis")
def model2_analysis():
    """Model 2 - Pair Trading Analysis Results Page"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    ticker1 = request.args.get('ticker1', '').upper().strip()
    ticker2 = request.args.get('ticker2', '').upper().strip()
    period = request.args.get('period', '1y')
    
    if not ticker1 or not ticker2:
        return render_template(
            "model2_analysis.html",
            error="Please provide both ticker symbols",
            ticker1=ticker1,
            ticker2=ticker2
        )
    
    if ticker1 == ticker2:
        return render_template(
            "model2_analysis.html",
            error="Please select two different ticker symbols",
            ticker1=ticker1,
            ticker2=ticker2
        )
    
    try:
        # Perform pair trading analysis
        analysis_data = pair_analyzer.perform_pair_analysis(session['sid'], ticker1, ticker2, period)
        chart_data = pair_analyzer.generate_analysis_chart(session['sid'], ticker1, ticker2, period)
        
        return render_template(
            "model2_analysis.html",
            analysis_data=analysis_data,
            chart_data=chart_data,
            ticker1=ticker1,
            ticker2=ticker2,
            error=None
        )
        
    except Exception as e:
        return render_template(
            "model2_analysis.html",
            error=f"Error analyzing pair: {str(e)}",
            ticker1=ticker1,
            ticker2=ticker2,
            analysis_data=None,
            chart_data=None
        )

@app.route("/api/validate-ticker", methods=['POST'])
def validate_ticker():
    """API endpoint to validate a ticker symbol"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        
        if not ticker:
            return jsonify({
                "success": False,
                "error": "Ticker symbol is required"
            }), 400
        
        is_valid, result = pair_analyzer.validate_ticker(ticker)
        
        if is_valid:
            return jsonify({
                "success": True,
                "ticker": ticker.upper(),
                "company_name": result
            })
        else:
            return jsonify({
                "success": False,
                "error": result
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/pair-analysis", methods=['POST'])
def api_pair_analysis():
    """API endpoint for pair trading analysis"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    try:
        data = request.get_json()
        ticker1 = data.get('ticker1', '').upper().strip()
        ticker2 = data.get('ticker2', '').upper().strip()
        period = data.get('period', '1y')
        
        if not ticker1 or not ticker2:
            return jsonify({
                "success": False,
                "error": "Both ticker symbols are required"
            }), 400
        
        if ticker1 == ticker2:
            return jsonify({
                "success": False,
                "error": "Please select two different ticker symbols"
            }), 400
        
        analysis_data = pair_analyzer.perform_pair_analysis(session['sid'], ticker1, ticker2, period)
        
        return jsonify({
            "success": True,
            "analysis_data": analysis_data,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/download-pair-report", methods=['GET'])
def download_pair_report():
    """API endpoint to download pair trading analysis report"""
    if 'sid' not in session:
        session['sid'] = os.urandom(24).hex()
    
    ticker1 = request.args.get('ticker1', '').upper().strip()
    ticker2 = request.args.get('ticker2', '').upper().strip()
    period = request.args.get('period', '1y')
    
    try:
        # Get analysis data
        cache_key = f"{ticker1}_{ticker2}_{period}"
        if (session['sid'] not in pair_analyzer.user_cache or 
            cache_key not in pair_analyzer.user_cache[session['sid']]):
            return jsonify({
                "success": False,
                "error": "No analysis data available. Please run analysis first."
            }), 400
        
        analysis_data = pair_analyzer.user_cache[session['sid']][cache_key]['analysis_data']
        
        # Create CSV report
        output = io.StringIO()
        output.write(f'Pair Trading Analysis Report: {ticker1} vs {ticker2}\n')
        output.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        output.write(f'Analysis Period: {period}\n')
        output.write(f'Data Points: {analysis_data["data_points"]}\n')
        output.write(f'Date Range: {analysis_data["start_date"]} to {analysis_data["end_date"]}\n\n')
        
        # Company information
        output.write('Company Information\n')
        output.write(f'Ticker 1: {ticker1} - {analysis_data["company1"]}\n')
        output.write(f'Ticker 2: {ticker2} - {analysis_data["company2"]}\n\n')
        
        # Correlation metrics
        output.write('Correlation Metrics\n')
        corr_metrics = analysis_data['correlation_metrics']
        for key, value in corr_metrics.items():
            if key != 'rolling_correlation':
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        
        # Cointegration results
        output.write('\nCointegration Analysis\n')
        coint_results = analysis_data['cointegration_results']
        for key, value in coint_results.items():
            if key not in ['spread', 'adf_critical_values']:
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        
        # Trading analysis
        output.write('\nTrading Analysis\n')
        trading_analysis = analysis_data['trading_analysis']
        output.write(f'Overall Score: {trading_analysis["score"]}/{trading_analysis["max_score"]}\n')
        output.write(f'Rating: {trading_analysis["overall_rating"]}\n')
        output.write(f'Recommendation: {trading_analysis["trading_recommendation"]}\n\n')
        
        output.write('Detailed Recommendations:\n')
        for rec in trading_analysis['detailed_recommendations']:
            output.write(f'- {rec}\n')
        
        # Price data
        output.write(f'\nHistorical Price Data\n')
        price_df = pd.DataFrame({
            'Date': analysis_data['price_data']['dates'],
            f'{ticker1}_Price': analysis_data['price_data']['prices1'],
            f'{ticker2}_Price': analysis_data['price_data']['prices2']
        })
        price_df.to_csv(output, index=False)
        
        # Prepare file for download
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'pair_trading_analysis_{ticker1}_{ticker2}.csv'
        )
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)