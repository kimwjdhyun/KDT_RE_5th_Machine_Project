from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import os
import json
import random
import csv

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_FILE = os.path.join(DATA_DIR, "sensor_log.json")
CSV_FILE = os.path.join(DATA_DIR, "sensor_log.csv")

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
    현재 충전 회로 이슈가 있더라도,
    전압값이 들어오면 임시값으로만 사용 가능
    """
    v_min = 12.0
    v_max = 16.8
    soc = ((voltage - v_min) / (v_max - v_min)) * 100.0
    return round(clamp(soc, 0, 100), 1)


def calc_mode(soc: float) -> int:
    """
    모드 예시
    0: 절전
    1: 중간
    2: 정상
    """
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
# JSON 저장/조회
# ----------------------------
def load_logs():
    if not os.path.exists(JSON_FILE):
        return []
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_logs(logs):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def append_log_json(row: dict, max_keep: int = 300):
    logs = load_logs()
    logs.append(row)
    logs = logs[-max_keep:]
    save_logs(logs)
    return logs


# ----------------------------
# CSV 저장
# ----------------------------
def ensure_csv_exists():
    """
    CSV 파일이 없으면 헤더를 생성한다.
    XGBoost 학습용으로 바로 활용할 수 있게 컬럼을 명확히 맞춘다.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "temp",
                "hum",
                "soil",
                "light",
                "voltage",
                "current",
                "power",
                "soc",
                "pump",
                "led",
                "pred_1h",
                "mode",
                "water_alert"
            ])


def append_log_csv(row: dict):
    ensure_csv_exists()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["timestamp"],
            row["temp"],
            row["hum"],
            row["soil"],
            row["light"],
            row["voltage"],
            row["current"],
            row["power"],
            row["soc"],
            row["pump"],
            row["led"],
            row["pred_1h"],
            row["mode"],
            row["water_alert"]
        ])


def append_log(row: dict, max_keep: int = 300):
    """
    JSON + CSV 둘 다 저장
    - JSON: 대시보드 API용
    - CSV : XGBoost 학습용
    """
    logs = append_log_json(row, max_keep=max_keep)
    append_log_csv(row)
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
    """
    Arduino / ESP-01이 보내는 센서 데이터를 수신
    현재 권장 payload 예시:
    {
      "temp": 24.5,
      "hum": 55.1,
      "soil": 38,
      "light": 812,
      "voltage": 12.4,
      "current": 0.35,
      "power": 4.34,   # 없어도 됨
      "pump": 0,
      "led": 1
    }
    """
    data = request.get_json(silent=True) or {}

    temp = float(data.get("temp", 0) or 0)
    hum = float(data.get("hum", 0) or 0)
    soil = float(data.get("soil", 0) or 0)
    light = float(data.get("light", 0) or 0)

    voltage = float(data.get("voltage", 0) or 0)
    current = float(data.get("current", 0) or 0)

    # power가 안 오면 서버에서 자동 계산
    incoming_power = data.get("power", None)
    if incoming_power is None or incoming_power == "":
        power = voltage * current
    else:
        power = float(incoming_power)

    pump = int(data.get("pump", 0) or 0)
    led = int(data.get("led", 0) or 0)

    # SOC는 들어오면 사용, 없으면 전압 기반 임시 추정
    # 현재 SOC를 빼고 가고 싶으면 이 부분은 그대로 둬도 문제 없음
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
        "current": round(current, 3),
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
        "message": "sensor data saved",
        "mode": mode,
        "water_alert": water_alert,
        "pred_1h": pred_1h,
        "saved_row": row
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


@app.route("/csv-status", methods=["GET"])
def csv_status():
    """
    CSV 생성 여부와 저장 경로 확인용
    """
    return jsonify({
        "csv_exists": os.path.exists(CSV_FILE),
        "csv_file": CSV_FILE,
        "json_file": JSON_FILE
    })


if __name__ == "__main__":
    ensure_csv_exists()
    app.run(host="0.0.0.0", port=5000, debug=True)