from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import deque
import os
import json
import random
import csv
import time
import threading
import serial

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_FILE = os.path.join(DATA_DIR, "sensor_log.json")
CSV_FILE = os.path.join(DATA_DIR, "sensor_log.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# 시리얼 설정
# Windows: COM3, COM4 ...
# Mac/Linux: /dev/ttyUSB0, /dev/ttyACM0 ...
# ----------------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

latest_row = None
history_memory = deque(maxlen=300)

# ----------------------------
# 유틸
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def estimate_soc_from_battery_voltage(battery_voltage: float) -> float:
    v_min = 12.0
    v_max = 16.8
    soc = ((battery_voltage - v_min) / (v_max - v_min)) * 100.0
    return round(clamp(soc, 0, 100), 1)


def calc_mode(soc: float) -> int:
    if soc < 20:
        return 0
    elif soc < 60:
        return 1
    return 2


def calc_water_alert(soil: float) -> int:
    return 1 if soil < 40 else 0


def predict_power_1h(latest_solar_power: float) -> float:
    return round(latest_solar_power * random.uniform(0.9, 1.1), 2)


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
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "temperature",
                "humidity",
                "soil",
                "light",
                "solar_voltage",
                "solar_current",
                "solar_power",
                "battery_voltage",
                "battery_current",
                "battery_power",
                "pump",
                "led",
                "soc",
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
            row["temperature"],
            row["humidity"],
            row["soil"],
            row["light"],
            row["solar_voltage"],
            row["solar_current"],
            row["solar_power"],
            row["battery_voltage"],
            row["battery_current"],
            row["battery_power"],
            row["pump"],
            row["led"],
            row["soc"],
            row["pred_1h"],
            row["mode"],
            row["water_alert"]
        ])


def append_log(row: dict, max_keep: int = 300):
    logs = append_log_json(row, max_keep=max_keep)
    append_log_csv(row)
    history_memory.append(row)
    return logs


# ----------------------------
# 시리얼 데이터 처리
# ----------------------------
def build_row_from_serial(data: dict) -> dict:
    temperature = safe_float(data.get("temperature", 0), 0)
    humidity = safe_float(data.get("humidity", 0), 0)
    soil = safe_float(data.get("soil", 0), 0)
    light = safe_float(data.get("light", 0), 0)

    solar_voltage = safe_float(data.get("solar_voltage", 0), 0)
    solar_current = safe_float(data.get("solar_current", 0), 0)

    incoming_solar_power = data.get("solar_power", None)
    if incoming_solar_power is None or incoming_solar_power == "":
        solar_power = solar_voltage * solar_current
    else:
        solar_power = safe_float(incoming_solar_power, 0)

    battery_voltage = safe_float(data.get("battery_voltage", 0), 0)
    battery_current = safe_float(data.get("battery_current", 0), 0)

    incoming_battery_power = data.get("battery_power", None)
    if incoming_battery_power is None or incoming_battery_power == "":
        battery_power = battery_voltage * battery_current
    else:
        battery_power = safe_float(incoming_battery_power, 0)

    pump = safe_int(data.get("pump", 0), 0)
    led = safe_int(data.get("led", 0), 0)

    incoming_soc = data.get("soc", None)
    if incoming_soc is None or incoming_soc == "":
        soc = estimate_soc_from_battery_voltage(battery_voltage)
    else:
        soc = round(clamp(safe_float(incoming_soc, 0), 0, 100), 1)

    incoming_mode = data.get("mode", None)
    if incoming_mode is None or incoming_mode == "":
        mode = calc_mode(soc)
    else:
        mode = safe_int(incoming_mode, calc_mode(soc))

    incoming_water_alert = data.get("water_alert", None)
    if incoming_water_alert is None or incoming_water_alert == "":
        water_alert = calc_water_alert(soil)
    else:
        water_alert = safe_int(incoming_water_alert, calc_water_alert(soil))

    pred_1h = predict_power_1h(solar_power)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "soil": round(soil, 1),
        "light": round(light, 0),
        "solar_voltage": round(solar_voltage, 2),
        "solar_current": round(solar_current, 3),
        "solar_power": round(solar_power, 2),
        "battery_voltage": round(battery_voltage, 2),
        "battery_current": round(battery_current, 3),
        "battery_power": round(battery_power, 2),
        "pump": pump,
        "led": led,
        "soc": round(soc, 1),
        "pred_1h": pred_1h,
        "mode": mode,
        "water_alert": water_alert
    }
    return row


def serial_reader():
    global latest_row

    while True:
        try:
            print(f"[SERIAL] 연결 시도: {SERIAL_PORT}")
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                time.sleep(3)
                print("[SERIAL] 연결 성공")

                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not raw:
                        continue

                    print("[RAW]", raw)

                    if raw.startswith("[DEBUG]") or raw.startswith("[MASTER]") or raw.startswith("[INA219]"):
                        continue

                    if raw.startswith("{") and raw.endswith("}"):
                        try:
                            data = json.loads(raw)
                            row = build_row_from_serial(data)
                            latest_row = row
                            append_log(row)
                            print("[DATA 저장 완료]", row)
                        except json.JSONDecodeError:
                            print("[JSON 파싱 실패]", raw)
                    else:
                        print("[무시됨]", raw)

        except serial.SerialException as e:
            print("[SERIAL 오류]", e)
            time.sleep(3)
        except Exception as e:
            print("[예상치 못한 오류]", e)
            time.sleep(3)


# ----------------------------
# API
# ----------------------------
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "message": "Flask 서버 정상 실행 중 ✅",
        "serial_port": SERIAL_PORT
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "latest_exists": latest_row is not None,
        "history_count": len(history_memory),
        "serial_port": SERIAL_PORT
    })


@app.route("/latest", methods=["GET"])
@app.route("/api/latest", methods=["GET"])
def get_latest():
    if latest_row is not None:
        return jsonify(latest_row)

    logs = load_logs()
    if not logs:
        return jsonify({})
    return jsonify(logs[-1])


@app.route("/data", methods=["GET"])
@app.route("/api/history", methods=["GET"])
def get_data():
    if len(history_memory) > 0:
        return jsonify(list(history_memory)[-50:])

    logs = load_logs()
    return jsonify(logs[-50:])


@app.route("/stats", methods=["GET"])
@app.route("/api/stats", methods=["GET"])
def stats():
    logs = list(history_memory) if len(history_memory) > 0 else load_logs()
    total_count = len(logs)

    total_solar_generation = sum(float(d.get("solar_power", 0) or 0) for d in logs)

    avg_temperature = round(
        sum(float(d.get("temperature", 0) or 0) for d in logs) / total_count, 1
    ) if total_count else 0

    avg_humidity = round(
        sum(float(d.get("humidity", 0) or 0) for d in logs) / total_count, 1
    ) if total_count else 0

    avg_battery_voltage = round(
        sum(float(d.get("battery_voltage", 0) or 0) for d in logs) / total_count, 2
    ) if total_count else 0

    return jsonify({
        "count": total_count,
        "total_solar_generation": round(total_solar_generation, 2),
        "carbon_reduction_g": round(total_solar_generation * 0.5, 2),
        "avg_temperature": avg_temperature,
        "avg_humidity": avg_humidity,
        "avg_battery_voltage": avg_battery_voltage
    })


@app.route("/csv-status", methods=["GET"])
def csv_status():
    return jsonify({
        "csv_exists": os.path.exists(CSV_FILE),
        "csv_file": CSV_FILE,
        "json_file": JSON_FILE
    })


if __name__ == "__main__":
    ensure_csv_exists()

    loaded_logs = load_logs()[-300:]
    for row in loaded_logs:
        history_memory.append(row)

    if loaded_logs:
        latest_row = loaded_logs[-1]

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False)