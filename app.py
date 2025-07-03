from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests
import math
import datetime
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# Stockage des données utilisateur en mémoire
users = {}

# Simulated investment account
investment_account = {
    "balance": 0.0
}

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        firefly_url = request.form.get("firefly_url").rstrip("/")
        token = request.form.get("access_token")

        # Stockage temporaire en session + en mémoire
        session["user_id"] = "user1"
        users["user1"] = {
            "firefly_url": firefly_url,
            "token": token,
            "roundup_total": 0.0,
            "last_roundup_date": None
        }
        return redirect(url_for("dashboard"))
    
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    user_id = session.get("user_id")
    if not user_id or user_id not in users:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

def firefly_headers(user):
    return {
        "Authorization": f"Bearer {user['token']}",
        "Accept": "application/json"
    }

@app.route("/simulate-roundup")
def simulate_roundup():
    user_id = session.get("user_id")
    user = users.get(user_id)
    if not user:
        return jsonify({"error": "Utilisateur non connecté"}), 400

    today = datetime.date.today().isoformat()
    if user.get("last_roundup_date") == today:
        return jsonify({
            "message": "Déjà traité aujourd'hui",
            "total": user["roundup_total"]
        })

    last_week = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    resp = requests.get(
        f"{user['firefly_url']}/api/v1/transactions",
        headers=firefly_headers(user),
        params={"start": last_week, "type": "withdrawal"}
    )

    if resp.status_code != 200:
        return jsonify({"error": "Erreur Firefly", "details": resp.text}), 500

    transactions = resp.json()["data"]
    roundup_total = 0.0
    for tx in transactions:
        amount = abs(float(tx["attributes"]["amount"]))
        if amount >= 1:
            roundup = round(math.ceil(amount) - amount, 2)
            roundup_total += roundup

    user["roundup_total"] += roundup_total
    user["last_roundup_date"] = today

    return jsonify({
        "message": "Arrondis calculés",
        "today_roundup": roundup_total,
        "total": user["roundup_total"]
    })

@app.route("/transfer-roundup", methods=["POST"])
def transfer_roundup():
    user_id = session.get("user_id")
    user = users.get(user_id)
    if not user:
        return render_template("confirmation.html", message="Utilisateur non connecté", amount=0, balance=investment_account["balance"])

    source_account = request.form.get("sourceAccount")
    amount = user["roundup_total"]
    if amount <= 0:
        return render_template("confirmation.html", message="Aucun montant à transférer", amount=0, balance=investment_account["balance"])

    investment_account["balance"] += amount
    user["roundup_total"] = 0.0

    return render_template("confirmation.html",
                           message=f"Transfert manuel depuis « {source_account} » simulé avec succès.",
                           amount=amount,
                           balance=investment_account["balance"])

@app.route("/investment-balance")
def investment_balance():
    return jsonify({
        "balance": investment_account["balance"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
