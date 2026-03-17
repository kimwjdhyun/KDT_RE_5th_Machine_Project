import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import deque


# ================================================
# Flask 앱 초기화
# ================================================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)


# ================================================
# 전역 설정값
#
# CSV_FILE  : 아두이노 데이터 저장 CSV 파일
# JSON_FILE : 아두이노 데이터 저장 JSON 파일 (app.py와 호환)
# SEQ_LEN   : GRU에 넣을 과거 데이터 개수 (최근 30개)
# ================================================
DATA_DIR        = "data"
CSV_FILE        = os.path.join(DATA_DIR, "sensor_log.csv")
JSON_FILE       = os.path.join(DATA_DIR, "sensor_log.json")
SEQ_LEN         = 30
WATER_THRESHOLD = 0.5
SOIL_THRESHOLD  = 40.0

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ================================================
# 키 이름 호환 헬퍼 함수
#
# 아두이노가 "temperature", "solar_power" 등으로 보내므로
# 두 키 이름 모두 허용
# ================================================
def get_val(data, *keys, default=0.0):
    for k in keys:
        if k in data and data[k] != "" and data[k] is not None:
            try:
                return float(data[k])
            except (TypeError, ValueError):
                continue
    return default


# ================================================
# GRU 발전량 예측 모델 클래스
#
# ★ train_gru.py의 PowerPredictionGRU와 완전히 동일해야 함!
# [input_size=7]
#   solar_power, solar_voltage, light, temperature,
#   hour_sin, hour_cos, soc
# ================================================
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


# ================================================
# 급수 분류 모델 클래스
# ★ train_water.py의 WaterClassifier와 완전히 동일해야 함!
# ================================================
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


# ================================================
# 모델 & 스케일러 로드
# 모델 파일 없으면 규칙 기반으로 대체 동작
# ================================================
gru_model    = PowerPredictionGRU()
water_model  = WaterClassifier()
scaler_gru   = None
scaler_water = None
gru_ready    = False
water_ready  = False

if os.path.exists("models/gru.pth") and os.path.exists("models/scaler_gru.pkl"):
    try:
        gru_model.load_state_dict(torch.load("models/gru.pth", map_location="cpu"))
        gru_model.eval()
        scaler_gru = joblib.load("models/scaler_gru.pkl")
        gru_ready  = True
        print("✅ GRU 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ GRU 모델 로드 실패: {e} → 규칙 기반으로 동작")
else:
    print("⚠️ GRU 모델 없음 → 규칙 기반으로 동작 (train_gru.py 실행 후 재시작)")

if os.path.exists("models/water.pth") and os.path.exists("models/scaler_water.pkl"):
    try:
        water_model.load_state_dict(torch.load("models/water.pth", map_location="cpu"))
        water_model.eval()
        scaler_water = joblib.load("models/scaler_water.pkl")
        water_ready  = True
        print("✅ 급수 분류 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ 급수 모델 로드 실패: {e} → 규칙 기반으로 동작")
else:
    print("⚠️ 급수 모델 없음 → 토양습도 40% 기준으로 동작")

recent_buffer = deque(maxlen=SEQ_LEN)


# ================================================
# CSV 저장 함수
# app.py와 동일한 컬럼 구조로 저장
# ================================================
def save_csv(row: dict):
    fieldnames = [
        "timestamp", "temperature", "humidity", "soil", "light",
        "solar_voltage", "solar_current", "solar_power",
        "battery_voltage", "battery_current", "battery_power",
        "pump", "led", "soc", "pred_1h", "mode", "water_alert"
    ]

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ================================================
# JSON 저장 함수 (app.py와 호환)
# 최근 300개만 유지
# ================================================
def save_json(row: dict, max_keep: int = 300):
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


# ================================================
# 에너지 모드 결정 함수
#
# SOC < 20% → mode 0: 긴급절전
# SOC < 60% → mode 1: 절약
# SOC ≥ 60% → mode 2: 풀가동
# GRU 예측 발전량 1W 미만이면 선제 강등
# ================================================
def decide_mode(power: float, pred_power, soc: int) -> int:
    if soc < 20:
        base = 0
    elif soc < 60:
        base = 1
    else:
        base = 2

    if pred_power is not None and pred_power < 1.0 and base > 0:
        base -= 1

    return base


# ================================================
# GRU 발전량 예측 함수
#
# 버퍼에 30개 쌓여야 추론 시작
# FEATURES 순서: solar_power, solar_voltage, light,
#                temperature, hour_sin, hour_cos, soc
# ★ train_gru.py의 FEATURES 순서와 완전히 같아야 함!
# ================================================
def predict_power(data: dict):
    if not gru_ready or len(recent_buffer) < SEQ_LEN:
        return None

    try:
        hour = datetime.now().hour

        seq = []
        for row in recent_buffer:
            seq.append([
                get_val(row, "solar_power"),
                get_val(row, "solar_voltage"),
                get_val(row, "light"),
                get_val(row, "temperature"),
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                get_val(row, "soc", default=100),
            ])

        seq_np     = np.array(seq, dtype=np.float32)
        seq_scaled = scaler_gru.transform(seq_np)

        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = gru_model(x).cpu().numpy()

        dummy = np.zeros((1, 7))
        dummy[0, 0] = pred_scaled[0, 0]
        pred_real = scaler_gru.inverse_transform(dummy)[0, 0]

        return max(0.0, float(pred_real))

    except Exception as e:
        print(f"GRU 추론 오류: {e}")
        return None


# ================================================
# 급수 분류 추론 함수
#
# 입력: soil, temperature, humidity
# ★ train_water.py의 FEATURES 순서와 동일해야 함!
# ================================================
def predict_water(soil: float, temperature: float, humidity: float) -> bool:
    if not water_ready:
        return soil < SOIL_THRESHOLD

    try:
        features = np.array([[soil, temperature, humidity]], dtype=np.float32)
        features_scaled = scaler_water.transform(features)
        x = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = water_model(x).item()

        return prob >= WATER_THRESHOLD

    except Exception as e:
        print(f"급수 추론 오류: {e}")
        return soil < SOIL_THRESHOLD


# ================================================
# Flask 라우트: /sensor (POST)
#
# 아두이노가 보내는 JSON 예시:
#   {"temperature":24.5, "humidity":58.0, "soil":47, "light":51,
#    "solar_voltage":12.34, "solar_current":0.32, "solar_power":2.5,
#    "battery_voltage":14.8, "battery_current":0.3, "battery_power":4.4,
#    "soc":80, "pump":0, "led":1}
# ================================================
@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400

    recent_buffer.append(data)

    # 값 추출
    solar_power      = get_val(data, "solar_power")
    solar_voltage    = get_val(data, "solar_voltage")
    solar_current    = get_val(data, "solar_current")
    battery_voltage  = get_val(data, "battery_voltage")
    battery_current  = get_val(data, "battery_current")
    battery_power    = get_val(data, "battery_power")
    temperature      = get_val(data, "temperature")
    humidity         = get_val(data, "humidity")
    soil             = get_val(data, "soil")
    light            = get_val(data, "light")
    soc              = int(get_val(data, "soc", default=100))
    pump             = int(get_val(data, "pump", default=0))
    led              = int(get_val(data, "led",  default=0))

    # AI 추론
    pred_power  = predict_power(data)
    water_alert = predict_water(soil, temperature, humidity)
    mode        = decide_mode(solar_power, pred_power, soc)

    # 저장할 row 구성
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature":     round(temperature,   1),
        "humidity":        round(humidity,       1),
        "soil":            round(soil,           1),
        "light":           round(light,          1),
        "solar_voltage":   round(solar_voltage,  2),
        "solar_current":   round(solar_current,  3),
        "solar_power":     round(solar_power,    2),
        "battery_voltage": round(battery_voltage,2),
        "battery_current": round(battery_current,3),
        "battery_power":   round(battery_power,  2),
        "pump":            pump,
        "led":             led,
        "soc":             soc,
        "pred_1h":         round(pred_power, 2) if pred_power is not None else 0.0,
        "mode":            mode,
        "water_alert":     1 if water_alert else 0,
    }

    # CSV + JSON 둘 다 저장
    save_csv(row)
    save_json(row)

    print(f"[수신] 토양:{soil}% 조도:{light}% 온도:{temperature}° 발전:{solar_power}W "
          f"→ mode:{mode} water:{water_alert} "
          f"pred:{f'{pred_power:.2f}W' if pred_power is not None else 'GRU대기중'}")

    return jsonify({
        "mode":        mode,
        "water_alert": water_alert,
        "pred_power":  round(pred_power, 2) if pred_power is not None else None
    })


# ================================================
# Flask 라우트: /data (GET)
# 최근 50개 데이터 반환
# ================================================
@app.route("/data", methods=["GET"])
def get_data():
    if not os.path.exists(JSON_FILE):
        return jsonify([])

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return jsonify(logs[-50:])
    except Exception:
        return jsonify([])


# ================================================
# Flask 라우트: /latest (GET)
# 가장 최근 데이터 1개 반환
# ================================================
@app.route("/latest", methods=["GET"])
def latest():
    if not os.path.exists(JSON_FILE):
        return jsonify({}), 404

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if not logs:
            return jsonify({}), 404
        return jsonify(logs[-1])
    except Exception:
        return jsonify({}), 404


# ================================================
# Flask 라우트: /stats (GET)
# 전체 데이터 통계 요약
# ================================================
@app.route("/stats", methods=["GET"])
def stats():
    if not os.path.exists(JSON_FILE):
        return jsonify({"count": 0})

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        return jsonify({"count": 0})

    total = len(logs)
    if total == 0:
        return jsonify({"count": 0})

    return jsonify({
        "count":        total,
        "avg_temp":     round(sum(float(r.get("temperature", 0)) for r in logs) / total, 1),
        "avg_soil":     round(sum(float(r.get("soil",        0)) for r in logs) / total, 1),
        "gru_ready":    gru_ready,
        "water_ready":  water_ready,
        "buffer_size":  len(recent_buffer),
    })


# ================================================
# Flask 라우트: /status (GET)
# 모델/버퍼 상태 확인
# ================================================
@app.route("/status", methods=["GET"])
def status():
    csv_rows = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            csv_rows = sum(1 for _ in f) - 1

    return jsonify({
        "gru_model":   "로드됨" if gru_ready   else "없음 (규칙 기반 동작 중)",
        "water_model": "로드됨" if water_ready  else "없음 (규칙 기반 동작 중)",
        "buffer":      f"{len(recent_buffer)}/{SEQ_LEN} "
                       f"(GRU 추론 {'가능' if len(recent_buffer) >= SEQ_LEN else '대기중'})",
        "csv_rows":    csv_rows,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)