from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/predict")
def predict():
    return jsonify({"result": 123.45})

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/data")
def get_data():
    return jsonify([
        {"power": 3.2, "soc": 58},
        {"power": 4.1, "soc": 60},
        {"power": 2.8, "soc": 55}
    ])