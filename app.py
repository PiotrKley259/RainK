from flask import Flask, render_template, jsonify, send_file
import os
import pandas as pd
import io
from datetime import datetime
from model1 import GasPricePredictor

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'OGEllO09BgxinM54A1nHEKTcU8juJuI5CvdUIvNe')

# Instance du modèle
predictor = GasPricePredictor()

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/model1")
def model1():
    try:
        # Get predictions and metrics
        prediction = predictor.get_prediction()
        weekly_stats = predictor.get_weekly_prediction()
        metrics = predictor.get_model_metrics()

        # Prepare data for template
        return render_template(
            "model1.html",
            current_price=prediction['current_price'],
            predicted_price=prediction['predicted_price'],
            price_change=prediction['price_change'],
            price_change_pct=prediction['price_change_pct'],
            prediction_date=prediction['prediction_date'],
            weekly_predictions=weekly_stats['predictions'],
            weekly_summary={
                'week_start_price': weekly_stats['week_start_price'],
                'week_end_price': weekly_stats['week_end_price'],
                'total_change': weekly_stats['total_change'],
                'total_change_pct': weekly_stats['total_change_pct'],
                'max_price': weekly_stats['max_price'],
                'min_price': weekly_stats['min_price'],
                'avg_price': weekly_stats['avg_price']
            },
            metrics=metrics,
            error=None
        )
    except Exception as e:
        # Handle errors gracefully
        return render_template(
            "model1.html",
            current_price="N/A",
            predicted_price="N/A",
            price_change="N/A",
            price_change_pct="N/A",
            prediction_date="N/A",
            weekly_predictions=[],
            weekly_summary={},
            metrics={'r2_score': 'N/A', 'mae': 'N/A', 'rmse': 'N/A', 'last_trained': 'N/A'},
            error=f"Error loading predictions: {str(e)}"
        )

@app.route("/api/predict", methods=['GET'])
def predict_gas_price():
    """API endpoint pour obtenir la prédiction"""
    try:
        result = predictor.get_prediction()
        weekly_stats = predictor.get_weekly_prediction()
        
        return jsonify({
            "success": True,
            "current_price": result['current_price'],
            "predicted_price": result['predicted_price'],
            "price_change": result['price_change'],
            "price_change_pct": result['price_change_pct'],
            "prediction_date": result['prediction_date'],
            "weekly_predictions": weekly_stats['predictions'],
            "weekly_summary": {
                "week_start_price": weekly_stats['week_start_price'],
                "week_end_price": weekly_stats['week_end_price'],
                "total_change": weekly_stats['total_change'],
                "total_change_pct": weekly_stats['total_change_pct'],
                "max_price": weekly_stats['max_price'],
                "min_price": weekly_stats['min_price'],
                "avg_price": weekly_stats['avg_price']
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
    try:
        metrics = predictor.get_model_metrics()
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
    try:
        weekly_stats = predictor.get_weekly_prediction()
        metrics = predictor.get_model_metrics()
        
        # Create CSV report
        output = io.StringIO()
        output.write('Natural Gas Price Prediction Report\n')
        output.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        output.write('Weekly Predictions\n')
        df = pd.DataFrame(weekly_stats['predictions'])
        df.to_csv(output, index=False)
        output.write('\nWeekly Summary\n')
        for key, value in weekly_stats.items():
            if key != 'predictions':
                output.write(f'{key.replace("_", " ").title()}: {value}\n')
        output.write('\nModel Performance Metrics\n')
        for key, value in metrics.items():
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