# app.py (version améliorée)
from flask import Flask, render_template, jsonify
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from model1 import GasPricePredictor  # Votre code de ML réorganisé

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
    return render_template("model1.html")

@app.route("/api/predict", methods=['GET'])
def predict_gas_price():
    """API endpoint pour obtenir la prédiction"""
    try:
        # Récupérer les données et faire la prédiction
        result = predictor.get_prediction()
        
        return jsonify({
            "success": True,
            "current_price": result['current_price'],
            "predicted_price": result['predicted_price'],
            "price_change": result['price_change'],
            "price_change_pct": result['price_change_pct'],
            "confidence": result['confidence'],
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False en production