from flask import Flask, request, jsonify, render_template
import requests
import math
import datetime
import os

app = Flask(__name__)

# === CONFIGURATION SALT EDGE ===
SALT_EDGE_APP_ID = "kBga8tvYr2BVtHNSZ2RtSueZbg9ddqczAL57pJJn4gI"
SALT_EDGE_SECRET = "XO2PEbXmpjYrsnmojb5_T0BpT4LsiVAgvBfP81UntRI"
SALT_EDGE_API_URL = "https://www.saltedge.com/api/v5"

# Simulated user store (memory)
users = {}

def get_headers():
    return {
        "App-id": SALT_EDGE_APP_ID,
        "Secret": SALT_EDGE_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/connect", methods=["POST"])
def connect_bank():
    identifier = "user1"

    # Step 1: Try creating the customer
    customer_resp = requests.post(
        f"{SALT_EDGE_API_URL}/customers",
        headers=get_headers(),
        json={"data": {"identifier": identifier}}
    )

    if customer_resp.status_code == 200:
        customer_id = customer_resp.json()["data"]["id"]

    elif customer_resp.status_code == 409:
        # Already exists → get by identifier
        existing_resp = requests.get(
            f"{SALT_EDGE_API_URL}/customers",
            headers=get_headers(),
            params={"identifier": identifier}
        )
        if existing_resp.status_code != 200 or not existing_resp.json().get("data"):
            return jsonify({
                "error": "retrieving existing customer failed",
                "details": existing_resp.text
            }), 400
        customer_id = existing_resp.json()["data"][0]["id"]
    else:
        return jsonify({
            "error": "customer creation failed",
            "details": customer_resp.text
        }), 400

    # Step 2: Create Salt Edge Connect Session
    session_resp = requests.post(
        f"{SALT_EDGE_API_URL}/connect_sessions/create",
        headers=get_headers(),
        json={
            "data": {
                "customer_id": customer_id,
                "consent": {
                    "scopes": ["account_details", "transactions_details"]
                },
                "attempt": {
                    "return_to": "https://raink.onrender.com/callback"  # 🔁 redirection vers ton domaine Render
                }
            }
        }
    )

    if session_resp.status_code != 200:
        return jsonify({
            "error": "connect session failed",
            "details": session_resp.text
        }), 400

    connect_url = session_resp.json()["data"]["connect_url"]
    return jsonify({"connect_url": connect_url})

@app.route("/callback")
def callback():
    users["user1"] = {
        "roundup_total": 0.0,
        "last_roundup_date": None
    }
    return render_template("tirelire.html")

@app.route("/simulate-roundup")
def simulate_roundup():
    user = users.get("user1")
    if not user:
        return jsonify({"error": "Utilisateur non connecté"}), 400

    today = datetime.date.today().isoformat()
    if user.get("last_roundup_date") == today:
        return jsonify({
            "message": "Déjà traité aujourd'hui",
            "total": user["roundup_total"]
        })

    # 🔧 Exemple factice (remplace avec transactions Salt Edge si tu veux)
    fake_transactions = [
        {"amount": -2.75},
        {"amount": -9.20},
        {"amount": -1.99},
        {"amount": 1200.00}
    ]

    roundup_total = 0.0
    for tx in fake_transactions:
        amount = abs(float(tx["amount"]))
        if tx["amount"] < 0 and amount >= 1:
            roundup = round(math.ceil(amount) - amount, 2)
            roundup_total += roundup

    user["roundup_total"] += roundup_total
    user["last_roundup_date"] = today

    return jsonify({
        "message": "Round-up simulé",
        "today_roundup": roundup_total,
        "total": user["roundup_total"]
    })

@app.route("/reset", methods=["POST"])
def reset_customer():
    identifier = "user1"
    response = requests.delete(
        f"{SALT_EDGE_API_URL}/customers/{identifier}",
        headers=get_headers()
    )

    if response.status_code == 204:
        users.pop("user1", None)
        return jsonify({"message": f"✅ Customer '{identifier}' supprimé avec succès."})
    elif response.status_code == 404:
        return jsonify({"message": f"ℹ️ Customer '{identifier}' n'existe pas."})
    else:
        return jsonify({"error": "❌ Erreur de suppression", "details": response.text}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

