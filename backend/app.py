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
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# React build 결과물 경로
FRONTEND_BUILD_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "frontend", "build")
)
FRONTEND_STATIC_DIR = os.path.join(FRONTEND_BUILD_DIR, "static")

JSON_FILE = os.path.join(DATA_DIR, "sensor_log.json")
CSV_FILE = os.path.join(DATA_DIR, "sensor_log.csv")
XGB_MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# Flask 앱 설정
# =========================================================
app = Flask(
    __name__,
    static_folder=FRONTEND_STATIC_DIR,
    static_url_path="/static"
)
CORS(app)

# =========================================================
# 시리얼 통신 설정
# =========================================================
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

# =========================================================
# 메모리 캐시
# =========================================================
latest_row = None
history_memory = deque(maxlen=300)

# 10초 간격 기준 1000개면 약 2시간 45분 정도
recent_raw_buffer = deque(maxlen=1000)

# =========================================================
# XGBoost 모델 관련 전역 변수
# =========================================================
xgb_model = None
xgb_feature_cols = None
xgb_pred_step = 12
xgb_resample_rule = "5min"
xgb_day_power_threshold = 0.05
xgb_feature_set_name = None
xgb_ready = False


# =========================================================
# 공통 유틸 함수
# =========================================================
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
    센서 row를 프로젝트 표준 19개 컬럼 구조로 맞춘다.
    JSON/CSV/메모리 저장 형식을 항상 동일하게 유지한다.
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


# =========================================================
# XGBoost 모델 로드
# =========================================================
def load_xgb_model():
    """
    train_xgboost.py에서 저장한 xgboost_model.pkl을 불러온다.
    모델뿐 아니라 feature_cols, resample_rule 같은 설정도 함께 읽는다.
    """
    global xgb_model, xgb_feature_cols, xgb_pred_step
    global xgb_resample_rule, xgb_day_power_threshold
    global xgb_feature_set_name, xgb_ready

    xgb_model = None
    xgb_feature_cols = None
    xgb_pred_step = 12
    xgb_resample_rule = "5min"
    xgb_day_power_threshold = 0.05
    xgb_feature_set_name = None
    xgb_ready = False

    if not os.path.exists(XGB_MODEL_FILE):
        print("⚠️ XGBoost 모델 없음 → pred_1h는 0.0으로 동작")
        return

    try:
        bundle = joblib.load(XGB_MODEL_FILE)

        if isinstance(bundle, dict):
            xgb_model = bundle.get("model")
            xgb_feature_cols = bundle.get("feature_cols")
            xgb_pred_step = bundle.get("pred_step", 12)
            xgb_resample_rule = bundle.get("resample_rule", "5min")
            xgb_day_power_threshold = bundle.get("day_power_threshold", 0.05)
            xgb_feature_set_name = bundle.get("feature_set_name")
        else:
            xgb_model = bundle

        xgb_ready = xgb_model is not None and xgb_feature_cols is not None

        if xgb_ready:
            print(f"✅ XGBoost 모델 로드 완료: {XGB_MODEL_FILE}")
            print(f"   - feature 개수: {len(xgb_feature_cols)}")
            print(f"   - pred_step: {xgb_pred_step}")
            print(f"   - resample_rule: {xgb_resample_rule}")
            print(f"   - feature_set_name: {xgb_feature_set_name}")
        else:
            print("⚠️ XGBoost 모델 형식 확인 필요")

    except Exception as e:
        print(f"⚠️ XGBoost 모델 로드 실패: {e}")


# =========================================================
# JSON 저장 / 조회
# =========================================================
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


# =========================================================
# CSV 저장
# =========================================================
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


def load_recent_csv_rows(max_rows=1000):
    """
    서버 시작 시 최근 CSV 데이터를 읽어
    recent_raw_buffer와 history_memory를 복원할 때 사용한다.
    """
    if not os.path.exists(CSV_FILE):
        return []

    try:
        try:
            df = pd.read_csv(CSV_FILE, on_bad_lines="skip", encoding="utf-8")
        except TypeError:
            df = pd.read_csv(CSV_FILE, error_bad_lines=False, encoding="utf-8")

        if df.empty:
            return []

        df = df.tail(max_rows).copy()
        rows = []

        for _, row in df.iterrows():
            nrow = normalize_row(row.to_dict())
            if nrow is not None:
                rows.append(nrow)

        return rows

    except Exception as e:
        print("[CSV 로드 실패]", e)
        return []


# =========================================================
# XGBoost 실시간 예측용 feature 생성
# =========================================================
def build_live_feature_from_buffer():
    """
    recent_raw_buffer에 쌓인 최근 원본 데이터를 이용해
    현재 시점 예측용 feature 1줄을 만든다.
    """
    if not xgb_ready:
        return None

    if len(recent_raw_buffer) < 200:
        return None

    try:
        df = pd.DataFrame(list(recent_raw_buffer))
        if df.empty or "timestamp" not in df.columns:
            return None

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

        df["solar_power"] = df["solar_power"].fillna(df["solar_voltage"] * df["solar_current"])
        df["solar_power"] = df["solar_power"].clip(lower=0)

        df["battery_power"] = df["battery_power"].fillna(df["battery_voltage"] * df["battery_current"])

        df = df.set_index("timestamp")
        df = df.resample(xgb_resample_rule).mean(numeric_only=True)
        df = df.dropna(how="all").reset_index()

        print(f"[XGB] resampled rows={len(df)}")

        if len(df) < 24:
            return None

        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["dayofweek"] = df["timestamp"].dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        for lag in [1, 2, 3, 6, 12]:
            df[f"power_lag_{lag}"] = df["solar_power"].shift(lag)
            df[f"light_lag_{lag}"] = df["light"].shift(lag)

        for window in [3, 6, 12, 24]:
            df[f"power_mean_{window}"] = df["solar_power"].rolling(window).mean()
            df[f"power_std_{window}"] = df["solar_power"].rolling(window).std()
            df[f"power_max_{window}"] = df["solar_power"].rolling(window).max()
            df[f"power_min_{window}"] = df["solar_power"].rolling(window).min()

        for window in [3, 6, 12]:
            df[f"light_mean_{window}"] = df["light"].rolling(window).mean()

        df["power_diff_1"] = df["solar_power"].diff(1)
        df["power_diff_3"] = df["solar_power"].diff(3)
        df["power_diff_6"] = df["solar_power"].diff(6)

        df["light_diff_1"] = df["light"].diff(1)
        df["light_diff_3"] = df["light"].diff(3)

        df["voltage_current_mul"] = df["solar_voltage"] * df["solar_current"]
        df["battery_eff_gap"] = df["battery_voltage"] - df["solar_voltage"]
        df["is_day_by_hour"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
        df["is_active_now"] = (df["solar_power"] > xgb_day_power_threshold).astype(int)

        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return None

        latest = df.iloc[[-1]].copy()

        for col in xgb_feature_cols:
            if col not in latest.columns:
                latest[col] = 0.0

        X_live = latest[xgb_feature_cols].copy()
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
        print(f"[XGB] pred_1h={pred:.2f}")
        return round(pred, 2)

    except Exception as e:
        print("[XGB 예측 실패]", e)
        return 0.0


# =========================================================
# 시리얼 데이터 처리
# =========================================================
def build_row_from_serial(data: dict) -> dict:
    """
    Arduino JSON 데이터를 표준 row 구조로 변환한다.
    이 시점에 XGBoost 예측값 pred_1h도 함께 계산한다.
    """
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
    """
    백그라운드 스레드에서 시리얼 포트를 계속 읽는다.
    """
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


# =========================================================
# React build 서빙
# =========================================================
@app.route("/", methods=["GET"])
def serve_dashboard():
    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_BUILD_DIR, "index.html")
    return (
        "React build 파일이 없습니다. frontend 폴더에서 npm run build를 먼저 실행하세요.",
        404,
    )


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/<path:path>", methods=["GET"])
def serve_react_assets(path):
    """
    React build 파일 및 라우팅 대응
    """
    requested_path = os.path.join(FRONTEND_BUILD_DIR, path)

    if os.path.exists(requested_path) and os.path.isfile(requested_path):
        return send_from_directory(FRONTEND_BUILD_DIR, path)

    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_BUILD_DIR, "index.html")

    return jsonify({"error": "frontend build not found"}), 404


# =========================================================
# API 엔드포인트
# =========================================================
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "message": "Flask 서버 정상 실행 중 ✅",
        "serial_port": SERIAL_PORT,
        "xgb_ready": xgb_ready,
        "buffer_size": len(recent_raw_buffer),
        "feature_set_name": xgb_feature_set_name,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "latest_exists": latest_row is not None,
        "history_count": len(history_memory),
        "serial_port": SERIAL_PORT,
        "xgb_ready": xgb_ready,
        "buffer_size": len(recent_raw_buffer),
        "feature_count": len(xgb_feature_cols) if xgb_feature_cols else 0,
        "resample_rule": xgb_resample_rule,
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


@app.route("/api/predict", methods=["GET"])
def get_predict():
    pred_1h = predict_power_1h_xgb()
    return jsonify({
        "xgb_ready": xgb_ready,
        "pred_1h": pred_1h,
        "feature_set_name": xgb_feature_set_name,
    })


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
            "avg_pred_1h": 0,
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

    avg_pred_1h = round(
        sum(safe_float(d.get("pred_1h", 0), 0) for d in logs) / total_count, 2
    )

    return jsonify({
        "count": total_count,
        "total_solar_generation": round(total_solar_generation, 2),
        "carbon_reduction_g": round(total_solar_generation * 0.5, 2),
        "avg_temperature": avg_temperature,
        "avg_humidity": avg_humidity,
        "avg_battery_voltage": avg_battery_voltage,
        "avg_pred_1h": avg_pred_1h,
        "xgb_ready": xgb_ready
    })


@app.route("/csv-status", methods=["GET"])
@app.route("/api/csv-status", methods=["GET"])
def csv_status():
    return jsonify({
        "csv_exists": os.path.exists(CSV_FILE),
        "csv_file": CSV_FILE,
        "json_file": JSON_FILE,
        "xgb_model_file": XGB_MODEL_FILE,
        "xgb_ready": xgb_ready,
        "feature_set_name": xgb_feature_set_name,
        "frontend_build_dir": FRONTEND_BUILD_DIR,
        "frontend_build_exists": os.path.exists(FRONTEND_BUILD_DIR),
    })


# =========================================================
# 서버 시작
# =========================================================
if __name__ == "__main__":
    ensure_csv_exists()
    load_xgb_model()

    loaded_logs = load_logs()[-300:]
    for row in loaded_logs:
        nrow = normalize_row(row)
        if nrow is not None:
            history_memory.append(nrow)

    recent_rows = load_recent_csv_rows(1000)
    for row in recent_rows:
        nrow = normalize_row(row)
        if nrow is not None:
            recent_raw_buffer.append(nrow)

    if loaded_logs:
        latest_row = normalize_row(loaded_logs[-1])
    elif recent_rows:
        latest_row = normalize_row(recent_rows[-1])

    print(f"[INIT] history_memory={len(history_memory)}")
    print(f"[INIT] recent_raw_buffer={len(recent_raw_buffer)}")
    print(f"[INIT] frontend_build_dir={FRONTEND_BUILD_DIR}")
    print(f"[INIT] frontend_build_exists={os.path.exists(FRONTEND_BUILD_DIR)}")

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False)