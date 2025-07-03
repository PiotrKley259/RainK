from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import csv
import io
import math
import datetime
from collections import defaultdict
import os

app = Flask(__name__)
app.secret_key = "supersecret"

# Simuler la "base de données" en mémoire
users_data = {}  # email -> user info
investment_accounts = {}  # email -> balance

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        password = request.form.get("password")
        dob = request.form.get("dob")

        if email in users_data:
            return render_template("register.html", error="Un compte existe déjà avec cet email.")

        users_data[email] = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": password,
            "dob": dob,
            "last_import": None,
            "roundups": {},
            "invested": 0.0
        }

        investment_accounts[email] = 0.0
        session["user_email"] = email

        return redirect(url_for("upload"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = users_data.get(email)
        if not user or user["password"] != password:
            return render_template("login.html", error="Email ou mot de passe incorrect.")

        session["user_email"] = email
        return redirect(url_for("dashboard"))

    return render_template("login.html")

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
        today = datetime.date.today()
        previous_month = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")

        for row in reader:
            try:
                amount = abs(float(row.get("Amount", 0)))
                date_str = row.get("Completed Date", "")
                if not date_str or amount == 0:
                    continue

                date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                key = date.strftime("%Y-%m")

                if key != previous_month:
                    continue  # ne garder que les transactions du mois précédent

                if amount >= 1:
                    roundup = round(math.ceil(amount) - amount, 2)
                    monthly_roundups[key] += roundup

            except Exception:
                continue

        total = sum(monthly_roundups.values())
        users_data[email]["roundups"] = dict(monthly_roundups)
        users_data[email]["invested"] += total
        investment_accounts[email] += total
        users_data[email]["last_import"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return redirect(url_for("dashboard"))

    return render_template("upload.html")

@app.route("/dashboard")
def dashboard():
    email = session.get("user_email")
    if not email or email not in users_data:
        return redirect(url_for("login"))

    user = users_data[email]
    return render_template("dashboard.html",
                           email=email,
                           first_name=user["first_name"],
                           last_import=user["last_import"],
                           invested=user["invested"],
                           roundups=user["roundups"])

@app.route("/investment-balance")
def investment_balance():
    email = session.get("user_email")
    if not email or email not in investment_accounts:
        return jsonify({"error": "Non connecté"}), 401

    return jsonify({"balance": investment_accounts[email]})

@app.route("/admin/users")
def admin_users():
    # Pour sécuriser, ajoute un mot de passe plus tard
    return render_template("admin_users.html", users=users_data)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
