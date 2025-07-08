from flask import Flask, render_template, jsonify, send_file, session
import os
import pandas as pd
import io
from datetime import datetime
from model1 import GasPricePredictor
from flask_session import Session

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Instance du modèle
predictor = GasPricePredictor()

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)