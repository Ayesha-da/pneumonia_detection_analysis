
""" Flask application that runs the API and renders the html page(s) """
from flask import Flask, render_template, jsonify
# from flask_sqlalchemy import SQLAlchemy

# Spin up flask app
app = Flask(__name__)

# This route renders the homepage
@app.route("/")
def index():
    return render_template("index.html")

# You need this - this allows you to actually run the app
if __name__ == "__main__":
    app.run(debug=True)