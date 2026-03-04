from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

data_log = []

@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.json

    data["timestamp"] = datetime.now().strftime("%H:%M:%S")

    # 🤖 AI 예측 더미
    current_power = data.get("power", 0)
    data["pred_1h"] = round(current_power * random.uniform(0.9, 1.1), 2)

    # ⚡ 에너지 모드 계산
    soc = data.get("soc", 50)

    if soc < 20:
        mode = 0
    elif soc < 60:
        mode = 1
    else:
        mode = 2

    data["mode"] = mode

    data_log.append(data)

    return jsonify({
        "mode": mode,
        "water_alert": 1 if data.get("soil", 50) < 40 else 0
    })


@app.route("/data")
def get_data():
    return jsonify(data_log[-50:])


@app.route("/predict")
def predict():
    if not data_log:
        return jsonify({"pred_1h": 0})

    latest = data_log[-1]
    return jsonify({"pred_1h": latest["pred_1h"]})


@app.route("/stats")
def stats():
    total_power = sum(d.get("power", 0) for d in data_log)

    return jsonify({
        "total_generation": round(total_power, 2),
        "carbon_reduction_g": round(total_power * 0.5, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)