from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/model1")
def model1():
    return render_template("model1.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)