from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests
import math
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

FIREFLY_INSTANCE_URL = os.environ.get("FIREFLY_INSTANCE_URL", "").rstrip("/")
FIREFLY_ADMIN_TOKEN = os.environ.get("FIREFLY_ADMIN_TOKEN")

# Stockage des données utilisateur en mémoire
users = {}

# Simulated investment account
investment_account = {
    "balance": 0.0
}

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Créer l'utilisateur sur Firefly III
        user_resp = requests.post(
            f"{FIREFLY_INSTANCE_URL}/api/v1/user",
            headers={"Authorization": f"Bearer {FIREFLY_ADMIN_TOKEN}", "Accept": "application/json"},
            json={"email": email, "password": password}
        )

        if user_resp.status_code != 200:
            return render_template("register.html", error="Erreur lors de la création du compte Firefly.")

        user_id = user_resp.json()["data"]["id"]

        # Créer un token API personnel pour cet utilisateur
        token_resp = requests.post(
            f"{FIREFLY_INSTANCE_URL}/api/v1/personal_tokens",
            headers={"Authorization": f"Bearer {FIREFLY_ADMIN_TOKEN}", "Accept": "application/json"},
            json={
                "name": f"Token {email}",
                "user_id": user_id,
                "permissions": ["*"]
            }
        )

        if token_resp.status_code != 200:
            return render_template("register.html", error="Erreur lors de la création du token API.")

        token = token_resp.json()["data"]["token"]

        # Stocker en session et mémoire
        session["user_id"] = email
        users[email] = {
            "firefly_url": f"{FIREFLY_INSTANCE_URL}/api/v1",
            "token": token,
            "roundup_total": 0.0,
            "last_roundup_date": None
        }

        return redirect(url_for("dashboard"))

    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    user_id = session.get("user_id")
    if not user_id or user_id not in users:
        return redirect(url_for("register"))
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
        f"{user['firefly_url']}/transactions",
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
