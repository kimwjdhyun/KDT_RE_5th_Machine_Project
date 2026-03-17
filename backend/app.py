import os
import csv
import json
import time
import threading
import serial
import random
import numpy as np
import torch
import torch.nn as nn
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import deque


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_FILE = os.path.join(DATA_DIR, "sensor_log.json")
CSV_FILE = os.path.join(DATA_DIR, "sensor_log.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ★ 본인 포트로 변경
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

SEQ_LEN = 30
WATER_THRESHOLD = 0.5
SOIL_THRESHOLD = 40.0

latest_row = None
history_memory = deque(maxlen=300)


# ── 유틸 ───────────────────────────────────────
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


# ── GRU 모델 ───────────────────────────────────
# FEATURES: solar_power, solar_voltage, light, temperature, hour_sin, hour_cos, soc
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ── 급수 분류 모델 ─────────────────────────────
# FEATURES: soil, temperature, humidity
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ── 모델 로드 ──────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
gru_model = PowerPredictionGRU()
water_model = WaterClassifier()
scaler_gru = None
scaler_water = None
gru_ready = False
water_ready = False

if os.path.exists(os.path.join(MODELS_DIR, "gru.pth")) and os.path.exists(os.path.join(MODELS_DIR, "scaler_gru.pkl")):
    try:
        gru_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "gru.pth"), map_location="cpu"))
        gru_model.eval()
        scaler_gru = joblib.load(os.path.join(MODELS_DIR, "scaler_gru.pkl"))
        gru_ready = True
        print("✅ GRU 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ GRU 모델 로드 실패: {e}")
else:
    print("⚠️ GRU 모델 없음 → 더미 예측으로 동작")

if os.path.exists(os.path.join(MODELS_DIR, "water.pth")) and os.path.exists(os.path.join(MODELS_DIR, "scaler_water.pkl")):
    try:
        water_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "water.pth"), map_location="cpu"))
        water_model.eval()
        scaler_water = joblib.load(os.path.join(MODELS_DIR, "scaler_water.pkl"))
        water_ready = True
        print("✅ 급수 분류 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ 급수 모델 로드 실패: {e}")
else:
    print("⚠️ 급수 모델 없음 → 토양습도 40% 기준으로 동작")

recent_buffer = deque(maxlen=SEQ_LEN)


# ── AI 추론 ────────────────────────────────────
def predict_power(data: dict) -> float:
    # GRU 준비 안됐으면 더미 예측
    if not gru_ready or len(recent_buffer) < SEQ_LEN:
        return round(safe_float(data.get("solar_power", 0)) * random.uniform(0.9, 1.1), 2)

    try:
        hour = datetime.now().hour
        seq = []
        for row in recent_buffer:
            seq.append([
                safe_float(row.get("solar_power", 0)),
                safe_float(row.get("solar_voltage", 0)),
                safe_float(row.get("light", 0)),
                safe_float(row.get("temperature", 0)),
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                safe_float(row.get("soc", 100)),
            ])

        seq_np = np.array(seq, dtype=np.float32)
        seq_scaled = scaler_gru.transform(seq_np)
        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = gru_model(x).cpu().numpy()

        dummy = np.zeros((1, 7))
        dummy[0, 0] = pred_scaled[0, 0]
        pred_real = scaler_gru.inverse_transform(dummy)[0, 0]

        return round(max(0.0, float(pred_real)), 2)

    except Exception as e:
        print(f"GRU 추론 오류: {e}")
        return round(safe_float(data.get("solar_power", 0)) * random.uniform(0.9, 1.1), 2)


def predict_water(soil: float, temperature: float, humidity: float) -> int:
    if not water_ready:
        return 1 if soil < SOIL_THRESHOLD else 0

    try:
        features = np.array([[soil, temperature, humidity]], dtype=np.float32)
        features_scaled = scaler_water.transform(features)
        x = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = water_model(x).item()

        return 1 if prob >= WATER_THRESHOLD else 0

    except Exception as e:
        print(f"급수 추론 오류: {e}")
        return 1 if soil < SOIL_THRESHOLD else 0


def decide_mode(solar_power: float, pred_power: float, soc: float) -> int:
    if soc < 20:
        base = 0
    elif soc < 60:
        base = 1
    else:
        base = 2

    if pred_power < 1.0 and base > 0:
        base -= 1

    return base


# ── CSV / JSON 저장 ────────────────────────────
def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "temperature", "humidity", "soil", "light",
                "solar_voltage", "solar_current", "solar_power",
                "battery_voltage", "battery_current", "battery_power",
                "pump", "led", "soc", "pred_1h", "mode", "water_alert"
            ])


def append_log_csv(row: dict):
    ensure_csv_exists()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["timestamp"], row["temperature"], row["humidity"],
            row["soil"], row["light"], row["solar_voltage"],
            row["solar_current"], row["solar_power"], row["battery_voltage"],
            row["battery_current"], row["battery_power"], row["pump"],
            row["led"], row["soc"], row["pred_1h"],
            row["mode"], row["water_alert"]
        ])


def append_log_json(row: dict, max_keep: int = 300):
    logs = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(row)
    logs = logs[-max_keep:]
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def append_log(row: dict):
    append_log_json(row)
    append_log_csv(row)
    history_memory.append(row)


def load_logs():
    if not os.path.exists(JSON_FILE):
        return []
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ── 시리얼 데이터 처리 ─────────────────────────
def build_row(data: dict) -> dict:
    temperature = safe_float(data.get("temperature", 0))
    humidity = safe_float(data.get("humidity", 0))
    soil = safe_float(data.get("soil", 0))
    light = safe_float(data.get("light", 0))

    solar_voltage = safe_float(data.get("solar_voltage", 0))
    solar_current = safe_float(data.get("solar_current", 0))
    solar_power = safe_float(data.get("solar_power", solar_voltage * solar_current))

    battery_voltage = safe_float(data.get("battery_voltage", 0))
    battery_current = safe_float(data.get("battery_current", 0))
    battery_power = safe_float(data.get("battery_power", battery_voltage * battery_current))

    pump = safe_int(data.get("pump", 0))
    led = safe_int(data.get("led", 0))

    incoming_soc = data.get("soc", None)
    if incoming_soc is None or incoming_soc == "":
        v_min = 12.0
        v_max = 16.8
        soc = round(clamp((battery_voltage - v_min) / (v_max - v_min) * 100, 0, 100), 1)
    else:
        soc = round(clamp(safe_float(incoming_soc), 0, 100), 1)

    pred_1h = predict_power(data)
    water_alert = predict_water(soil, temperature, humidity)
    mode = decide_mode(solar_power, pred_1h, soc)

    return {
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
        "soc": soc,
        "pred_1h": pred_1h,
        "mode": mode,
        "water_alert": water_alert,
    }


# ── 시리얼 읽기 스레드 ─────────────────────────
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

                    if raw.startswith("[DEBUG]") or raw.startswith("[MASTER]") or raw.startswith("[INA3221]"):
                        continue

                    if raw.startswith("{") and raw.endswith("}"):
                        try:
                            data = json.loads(raw)
                            recent_buffer.append(data)
                            row = build_row(data)
                            latest_row = row
                            append_log(row)
                            print(
                                f"[저장] 온도:{row['temperature']}° 발전:{row['solar_power']}W "
                                f"mode:{row['mode']} water:{row['water_alert']} pred:{row['pred_1h']}W"
                            )
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


# ── API ────────────────────────────────────────
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "message": "AI Flask 서버 정상 실행 중 ✅",
        "serial_port": SERIAL_PORT,
        "gru_ready": gru_ready,
        "water_ready": water_ready,
        "buffer": f"{len(recent_buffer)}/{SEQ_LEN}"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "serial_port": SERIAL_PORT,
        "gru_ready": gru_ready,
        "water_ready": water_ready,
        "buffer": f"{len(recent_buffer)}/{SEQ_LEN}",
        "latest_exists": latest_row is not None,
        "history_count": len(history_memory),
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
@app.route("/api/data", methods=["GET"])
@app.route("/api/history", methods=["GET"])
def get_data():
    if len(history_memory) > 0:
        return jsonify(list(history_memory)[-50:])

    logs = load_logs()
    return jsonify(logs[-50:] if logs else [])


@app.route("/stats", methods=["GET"])
@app.route("/api/stats", methods=["GET"])
def stats():
    logs = list(history_memory) if len(history_memory) > 0 else load_logs()
    total = len(logs)

    if total == 0:
        return jsonify({
            "count": 0,
            "total_solar_generation": 0,
            "carbon_reduction_g": 0,
            "avg_temperature": 0,
            "avg_humidity": 0,
            "avg_battery_voltage": 0,
            "gru_ready": gru_ready,
            "water_ready": water_ready,
        })

    total_solar = sum(float(d.get("solar_power", 0) or 0) for d in logs)

    return jsonify({
        "count": total,
        "total_solar_generation": round(total_solar, 2),
        "carbon_reduction_g": round(total_solar * 0.5, 2),
        "avg_temperature": round(sum(float(d.get("temperature", 0) or 0) for d in logs) / total, 1),
        "avg_humidity": round(sum(float(d.get("humidity", 0) or 0) for d in logs) / total, 1),
        "avg_battery_voltage": round(sum(float(d.get("battery_voltage", 0) or 0) for d in logs) / total, 2),
        "gru_ready": gru_ready,
        "water_ready": water_ready,
    })


# ── 서버 실행 ──────────────────────────────────
# debug=False 필수
if __name__ == "__main__":
    ensure_csv_exists()

    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)[-300:]
            for row in loaded:
                history_memory.append(row)
            if loaded:
                latest_row = loaded[-1]
        except Exception:
            pass

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False)