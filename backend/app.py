import os
import csv
import json
import time
import threading
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import serial
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

JSON_FILE = os.path.join(DATA_DIR, "sensor_log.json")
CSV_FILE = os.path.join(DATA_DIR, "sensor_log.csv")
XGB_MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SERIAL_PORT = "COM3"
BAUD_RATE = 9600

latest_row = None
history_memory = deque(maxlen=300)

# 10초 간격 기준 약 2시간 보관
recent_raw_buffer = deque(maxlen=720)

# ----------------------------
# XGBoost 모델 로드
# ----------------------------
xgb_model = None
xgb_feature_cols = None
xgb_pred_step = 12
xgb_resample_rule = "5min"
xgb_ready = False


# ----------------------------
# 유틸
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_row(row: dict) -> dict:
    """
    예전 로그/현재 로그 모두 19컬럼 구조로 정규화
    """
    if not isinstance(row, dict):
        return None

    timestamp = row.get("timestamp")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    normalized = {
        "timestamp": timestamp,
        "temperature": round(safe_float(row.get("temperature", 0), 0), 1),
        "humidity": round(safe_float(row.get("humidity", 0), 0), 1),
        "soil_raw": safe_int(row.get("soil_raw", 0), 0),
        "soil": round(safe_float(row.get("soil", 0), 0), 1),
        "light": round(safe_float(row.get("light", 0), 0), 0),
        "solar_voltage": round(safe_float(row.get("solar_voltage", 0), 0), 2),
        "solar_current": round(safe_float(row.get("solar_current", 0), 0), 3),
        "solar_power": round(safe_float(row.get("solar_power", 0), 0), 2),
        "battery_voltage": round(safe_float(row.get("battery_voltage", 0), 0), 2),
        "battery_current": round(safe_float(row.get("battery_current", 0), 0), 3),
        "battery_power": round(safe_float(row.get("battery_power", 0), 0), 2),
        "pump": safe_int(row.get("pump", 0), 0),
        "led": safe_int(row.get("led", 0), 0),
        "led_brightness": safe_int(row.get("led_brightness", 0), 0),
        "soc": round(clamp(safe_float(row.get("soc", 0), 0), 0, 100), 1),
        "pred_1h": round(max(0.0, safe_float(row.get("pred_1h", 0), 0)), 2),
        "mode": safe_int(row.get("mode", 0), 0),
        "water_alert": safe_int(row.get("water_alert", 0), 0),
    }

    # solar_power / battery_power가 0 또는 비정상일 경우 재계산
    if normalized["solar_power"] <= 0:
        normalized["solar_power"] = round(
            max(0.0, normalized["solar_voltage"] * normalized["solar_current"]), 2
        )

    if normalized["battery_power"] == 0 and (
        normalized["battery_voltage"] != 0 or normalized["battery_current"] != 0
    ):
        normalized["battery_power"] = round(
            normalized["battery_voltage"] * normalized["battery_current"], 2
        )

    return normalized


def load_xgb_model():
    global xgb_model, xgb_feature_cols, xgb_pred_step, xgb_resample_rule, xgb_ready

    xgb_model = None
    xgb_feature_cols = None
    xgb_pred_step = 12
    xgb_resample_rule = "5min"
    xgb_ready = False

    if os.path.exists(XGB_MODEL_FILE):
        try:
            bundle = joblib.load(XGB_MODEL_FILE)

            if isinstance(bundle, dict):
                xgb_model = bundle.get("model")
                xgb_feature_cols = bundle.get("feature_cols")
                xgb_pred_step = bundle.get("pred_step", 12)
                xgb_resample_rule = bundle.get("resample_rule", "5min")
            else:
                xgb_model = bundle

            xgb_ready = xgb_model is not None and xgb_feature_cols is not None
            print("✅ XGBoost 모델 로드 완료" if xgb_ready else "⚠️ XGBoost 모델 형식 확인 필요")
        except Exception as e:
            print(f"⚠️ XGBoost 모델 로드 실패: {e}")
    else:
        print("⚠️ XGBoost 모델 없음 → pred_1h는 0.0으로 동작")


# ----------------------------
# JSON 저장/조회
# ----------------------------
def load_logs():
    if not os.path.exists(JSON_FILE):
        return []

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if not isinstance(logs, list):
            return []

        normalized_logs = []
        for row in logs:
            nrow = normalize_row(row)
            if nrow is not None:
                normalized_logs.append(nrow)

        return normalized_logs

    except Exception as e:
        print("[JSON 로드 실패]", e)
        return []


def save_logs(logs):
    try:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[JSON 저장 실패]", e)


def append_log_json(row: dict, max_keep: int = 300):
    logs = load_logs()
    logs.append(normalize_row(row))
    logs = logs[-max_keep:]
    save_logs(logs)
    return logs


# ----------------------------
# CSV 저장
# ----------------------------
CSV_COLUMNS = [
    "timestamp",
    "temperature",
    "humidity",
    "soil_raw",
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
    "led_brightness",
    "soc",
    "pred_1h",
    "mode",
    "water_alert",
]


def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_log_csv(row: dict):
    ensure_csv_exists()
    row = normalize_row(row)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row[col] for col in CSV_COLUMNS])


def append_log(row: dict, max_keep: int = 300):
    row = normalize_row(row)
    append_log_json(row, max_keep=max_keep)
    append_log_csv(row)
    history_memory.append(row)


# ----------------------------
# XGBoost 실시간 예측용 feature 생성
# ----------------------------
def build_live_feature_from_buffer():
    if not xgb_ready:
        return None

    # 버퍼가 너무 적으면 예측 안 함
    if len(recent_raw_buffer) < 180:
        return None

    try:
        df = pd.DataFrame(list(recent_raw_buffer))
        if df.empty or "timestamp" not in df.columns:
            return None

        # 누락 컬럼 보정
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        if df.empty:
            return None

        numeric_cols = [
            "temperature", "humidity", "soil_raw", "soil", "light",
            "solar_voltage", "solar_current", "solar_power",
            "battery_voltage", "battery_current", "battery_power",
            "pump", "led", "led_brightness", "soc", "mode", "water_alert"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 발전량 보정
        df["solar_power"] = df["solar_power"].fillna(df["solar_voltage"] * df["solar_current"])
        df["solar_power"] = df["solar_power"].clip(lower=0)

        # 전압/전류 기반 보정
        df["battery_power"] = df["battery_power"].fillna(df["battery_voltage"] * df["battery_current"])

        df = df.set_index("timestamp")
        df = df.resample(xgb_resample_rule).mean(numeric_only=True)
        df = df.dropna(how="all").reset_index()

        if len(df) < 8:
            return None

        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["power_lag_1"] = df["solar_power"].shift(1)
        df["power_lag_2"] = df["solar_power"].shift(2)
        df["power_lag_3"] = df["solar_power"].shift(3)

        df["recent_mean_power_3"] = df["solar_power"].rolling(window=3).mean()
        df["recent_mean_power_6"] = df["solar_power"].rolling(window=6).mean()

        df["power_trend_3"] = df["solar_power"] - df["recent_mean_power_3"]
        df["power_diff_1"] = df["solar_power"] - df["power_lag_1"]
        df["power_std_6"] = df["solar_power"].rolling(window=6).std()

        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return None

        latest = df.iloc[[-1]].copy()

        # 모델이 요구하는 컬럼이 없으면 0으로 보정
        for col in xgb_feature_cols:
            if col not in latest.columns:
                latest[col] = 0.0

        X_live = latest[xgb_feature_cols].copy()

        # 최종 숫자형 보정
        X_live = X_live.apply(pd.to_numeric, errors="coerce").fillna(0)

        return X_live

    except Exception as e:
        print("[XGB live feature 생성 실패]", e)
        return None


def predict_power_1h_xgb():
    if not xgb_ready:
        return 0.0

    try:
        X_live = build_live_feature_from_buffer()
        if X_live is None:
            return 0.0

        pred = xgb_model.predict(X_live)[0]
        pred = max(0.0, float(pred))
        return round(pred, 2)

    except Exception as e:
        print("[XGB 예측 실패]", e)
        return 0.0


# ----------------------------
# 시리얼 데이터 처리
# ----------------------------
def build_row_from_serial(data: dict) -> dict:
    temperature = safe_float(data.get("temperature", 0), 0)
    humidity = safe_float(data.get("humidity", 0), 0)

    soil_raw = safe_int(data.get("soil_raw", 0), 0)
    soil = safe_float(data.get("soil", 0), 0)
    light = safe_float(data.get("light", 0), 0)

    solar_voltage = safe_float(data.get("solar_voltage", 0), 0)
    solar_current = safe_float(data.get("solar_current", 0), 0)

    incoming_solar_power = data.get("solar_power", None)
    if incoming_solar_power in (None, ""):
        solar_power = solar_voltage * solar_current
    else:
        solar_power = safe_float(incoming_solar_power, 0)

    solar_power = max(0.0, solar_power)

    battery_voltage = safe_float(data.get("battery_voltage", 0), 0)
    battery_current = safe_float(data.get("battery_current", 0), 0)

    incoming_battery_power = data.get("battery_power", None)
    if incoming_battery_power in (None, ""):
        battery_power = battery_voltage * battery_current
    else:
        battery_power = safe_float(incoming_battery_power, 0)

    pump = safe_int(data.get("pump", 0), 0)
    led = safe_int(data.get("led", 0), 0)
    led_brightness = safe_int(data.get("led_brightness", 0), 0)

    incoming_soc = data.get("soc", None)
    soc = round(clamp(safe_float(incoming_soc, 0), 0, 100), 1) if incoming_soc not in (None, "") else 0.0

    mode = safe_int(data.get("mode", 0), 0)
    water_alert = safe_int(data.get("water_alert", 0), 0)

    pred_1h = predict_power_1h_xgb()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "soil_raw": soil_raw,
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
        "led_brightness": led_brightness,
        "soc": round(soc, 1),
        "pred_1h": round(pred_1h, 2),
        "mode": mode,
        "water_alert": water_alert,
    }
    return normalize_row(row)


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

                    if raw.startswith("[DEBUG]") or raw.startswith("[MASTER]") or raw.startswith("[INA"):
                        continue

                    if raw.startswith("{") and raw.endswith("}"):
                        try:
                            data = json.loads(raw)

                            # 예측용 버퍼에는 "현재 pred_1h 계산 전" 상태보다
                            # 현재 row를 넣는 편이 전체 구조상 일관적
                            row = build_row_from_serial(data)

                            latest_row = row
                            recent_raw_buffer.append(row)
                            append_log(row)

                            print("[DATA 저장 완료]", row)
                        except json.JSONDecodeError:
                            print("[JSON 파싱 실패]", raw)
                        except Exception as e:
                            print("[ROW 생성/저장 실패]", e)
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
        "serial_port": SERIAL_PORT,
        "xgb_ready": xgb_ready,
        "buffer_size": len(recent_raw_buffer)
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "latest_exists": latest_row is not None,
        "history_count": len(history_memory),
        "serial_port": SERIAL_PORT,
        "xgb_ready": xgb_ready,
        "buffer_size": len(recent_raw_buffer)
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
    total_count = len(logs)

    if total_count == 0:
        return jsonify({
            "count": 0,
            "total_solar_generation": 0,
            "carbon_reduction_g": 0,
            "avg_temperature": 0,
            "avg_humidity": 0,
            "avg_battery_voltage": 0,
            "xgb_ready": xgb_ready
        })

    total_solar_generation = sum(safe_float(d.get("solar_power", 0), 0) for d in logs)

    avg_temperature = round(
        sum(safe_float(d.get("temperature", 0), 0) for d in logs) / total_count, 1
    )

    avg_humidity = round(
        sum(safe_float(d.get("humidity", 0), 0) for d in logs) / total_count, 1
    )

    avg_battery_voltage = round(
        sum(safe_float(d.get("battery_voltage", 0), 0) for d in logs) / total_count, 2
    )

    return jsonify({
        "count": total_count,
        "total_solar_generation": round(total_solar_generation, 2),
        "carbon_reduction_g": round(total_solar_generation * 0.5, 2),
        "avg_temperature": avg_temperature,
        "avg_humidity": avg_humidity,
        "avg_battery_voltage": avg_battery_voltage,
        "xgb_ready": xgb_ready
    })


@app.route("/csv-status", methods=["GET"])
def csv_status():
    return jsonify({
        "csv_exists": os.path.exists(CSV_FILE),
        "csv_file": CSV_FILE,
        "json_file": JSON_FILE,
        "xgb_model_file": XGB_MODEL_FILE,
        "xgb_ready": xgb_ready
    })


# ----------------------------
# 시작
# ----------------------------
if __name__ == "__main__":
    ensure_csv_exists()
    load_xgb_model()

    loaded_logs = load_logs()[-300:]
    for row in loaded_logs:
        nrow = normalize_row(row)
        if nrow is not None:
            history_memory.append(nrow)
            recent_raw_buffer.append(nrow)

    if loaded_logs:
        latest_row = normalize_row(loaded_logs[-1])

    print(f"[INIT] history_memory={len(history_memory)}")
    print(f"[INIT] recent_raw_buffer={len(recent_raw_buffer)}")

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False)