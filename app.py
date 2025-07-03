from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import csv
import io
import math
import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "supersecret"

investment_account = {"balance": 0.0}
users_data = {}

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        if not email:
            return render_template("login.html", error="Email requis")

        # Création automatique du compte si nouveau
        if email not in users_data:
            users_data[email] = {
                "roundups": {},
                "last_import": None
            }

        session["user_email"] = email
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    email = session.get("user_email")
    if not email or email not in users_data:
        return redirect(url_for("login"))

    user = users_data[email]
    return render_template("dashboard.html",
                           roundups=user["roundups"],
                           last_import=user["last_import"],
                           email=email)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    email = session.get("user_email")
    if not email or email not in users_data:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file:
            return render_template("upload.html", error="Aucun fichier sélectionné.")

        stream = io.StringIO(file.stream.read().decode("utf-8"))
        reader = csv.DictReader(stream)

        monthly_roundups = defaultdict(float)

        for row in reader:
            try:
                amount = abs(float(row.get("amount", 0)))
                date_str = row.get("date", "")
                if not date_str or amount == 0:
                    continue

                date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                key = date.strftime("%Y-%m")

                if amount >= 1:
                    roundup = round(math.ceil(amount) - amount, 2)
                    monthly_roundups[key] += roundup

            except Exception:
                continue  # Ignore bad lines

        users_data[email]["roundups"] = dict(monthly_roundups)
        users_data[email]["last_import"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return redirect(url_for("dashboard"))

    return render_template("upload.html")

@app.route("/transfer", methods=["POST"])
def transfer():
    email = session.get("user_email")
    if not email or email not in users_data:
        return redirect(url_for("login"))

    roundups = users_data[email].get("roundups", {})
    total = sum(roundups.values())
    investment_account["balance"] += total
    users_data[email]["roundups"] = {}

    return render_template("confirmation.html", amount=total, balance=investment_account["balance"])

@app.route("/investment-balance")
def investment_balance():
    return jsonify({"balance": investment_account["balance"]})

if __name__ == "__main__":
    app.run(debug=True)
