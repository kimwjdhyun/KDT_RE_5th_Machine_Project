from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import os
import json
import random

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "sensor_log.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------------
# 유틸
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def estimate_soc_from_voltage(voltage: float) -> float:
    """
    임시 SOC 추정
    4직렬 18650 기준 예시
    필요시 팀 상황에 맞게 조정
    """
    v_min = 12.0
    v_max = 16.8
    soc = ((voltage - v_min) / (v_max - v_min)) * 100.0
    return round(clamp(soc, 0, 100), 1)


def calc_mode(soc: float) -> int:
    if soc < 20:
        return 0
    elif soc < 60:
        return 1
    return 2


def calc_water_alert(soil: float) -> int:
    return 1 if soil < 40 else 0


def predict_power_1h(latest_power: float) -> float:
    """
    임시 더미 예측
    나중에 AI 담당이 GRU/XGBoost 결과로 교체
    """
    return round(latest_power * random.uniform(0.9, 1.1), 2)


# ----------------------------
# 파일 저장/조회
# ----------------------------
def load_logs():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_logs(logs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def append_log(row: dict, max_keep: int = 300):
    logs = load_logs()
    logs.append(row)
    logs = logs[-max_keep:]
    save_logs(logs)
    return logs


# ----------------------------
# API
# ----------------------------
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "message": "공용 Flask 서버 정상 실행 중 ✅"
    })


@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.get_json(silent=True) or {}

    temp = float(data.get("temp", 0) or 0)
    hum = float(data.get("hum", 0) or 0)
    soil = float(data.get("soil", 0) or 0)
    light = float(data.get("light", 0) or 0)

    voltage = float(data.get("voltage", 0) or 0)
    current = float(data.get("current", 0) or 0)
    power = float(data.get("power", 0) or 0)

    pump = int(data.get("pump", 0) or 0)
    led = int(data.get("led", 0) or 0)

    incoming_soc = data.get("soc", None)
    soc = float(incoming_soc) if incoming_soc is not None else estimate_soc_from_voltage(voltage)

    pred_1h = predict_power_1h(power)
    mode = calc_mode(soc)
    water_alert = calc_water_alert(soil)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": round(temp, 1),
        "hum": round(hum, 1),
        "soil": round(soil, 1),
        "light": round(light, 0),
        "voltage": round(voltage, 2),
        "current": round(current, 1),
        "power": round(power, 2),
        "soc": round(soc, 1),
        "pump": pump,
        "led": led,
        "pred_1h": pred_1h,
        "mode": mode,
        "water_alert": water_alert
    }

    append_log(row)

    return jsonify({
        "status": "ok",
        "mode": mode,
        "water_alert": water_alert,
        "pred_1h": pred_1h
    })


@app.route("/data", methods=["GET"])
def get_data():
    logs = load_logs()
    return jsonify(logs[-50:])


@app.route("/latest", methods=["GET"])
def get_latest():
    logs = load_logs()
    if not logs:
        return jsonify({})
    return jsonify(logs[-1])


@app.route("/stats", methods=["GET"])
def stats():
    logs = load_logs()

    total_power = sum(float(d.get("power", 0) or 0) for d in logs)
    total_count = len(logs)

    avg_temp = round(
        sum(float(d.get("temp", 0) or 0) for d in logs) / total_count, 1
    ) if total_count else 0

    avg_hum = round(
        sum(float(d.get("hum", 0) or 0) for d in logs) / total_count, 1
    ) if total_count else 0

    return jsonify({
        "count": total_count,
        "total_generation": round(total_power, 2),
        "carbon_reduction_g": round(total_power * 0.5, 2),
        "avg_temp": avg_temp,
        "avg_hum": avg_hum
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)