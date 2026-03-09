# app_gru.py
# GRU 기반 스마트팜 Flask 서버
# 실제 데이터 없을 때는 규칙 기반으로 동작
# 실제 데이터 쌓이면 GRU 모델 자동으로 사용

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import deque

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────
CSV_FILE        = "farm_data.csv"
SEQ_LEN         = 30      # GRU 입력 시퀀스 길이
WATER_THRESHOLD = 0.5     # 급수 확률 임계값
SOIL_THRESHOLD  = 40.0    # 토양습도 임계값 (%)


# ─────────────────────────────────────────
# GRU 모델 정의
# train_gru.py 와 완전히 동일해야 함
# ─────────────────────────────────────────
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ─────────────────────────────────────────
# 급수 분류 모델 정의
# train_water.py 와 완전히 동일해야 함
# ─────────────────────────────────────────
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 모델 & 스케일러 로드
# 파일 없으면 규칙 기반으로 fallback
# ─────────────────────────────────────────
gru_model    = PowerPredictionGRU()
water_model  = WaterClassifier()
scaler_gru    = None
scaler_target = None   # 발전량 역변환용 (feature 스케일러와 별도)
scaler_water  = None
gru_ready     = False
water_ready   = False

# GRU 모델 로드
if os.path.exists("models/gru.pth") and os.path.exists("models/scaler_gru.pkl"):
    try:
        gru_model.load_state_dict(torch.load("models/gru.pth", map_location="cpu"))
        gru_model.eval()
        scaler_gru    = joblib.load("models/scaler_gru.pkl")
        scaler_target = joblib.load("models/scaler_gru_target.pkl")
        gru_ready     = True
        print("✅ GRU 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ GRU 모델 로드 실패: {e} → 규칙 기반으로 동작")
else:
    print("⚠️ GRU 모델 없음 → 규칙 기반으로 동작 (train_gru.py 실행 후 재시작)")

# 급수 분류 모델 로드
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

# 최근 30개 데이터 버퍼 (GRU 시퀀스용)
recent_buffer = deque(maxlen=SEQ_LEN)


# ─────────────────────────────────────────
# CSV 저장
# ─────────────────────────────────────────
def save_csv(data: dict, mode: int):
    fieldnames = ["timestamp", "temp", "hum", "power",
                  "voltage", "current", "soc", "light", "soil", "mode"]
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temp":    round(float(data.get("temp",    0)), 1),
            "hum":     round(float(data.get("hum",     0)), 1),
            "power":   round(float(data.get("power",   0)), 3),
            "voltage": round(float(data.get("voltage", 0)), 3),
            "current": round(float(data.get("current", 0)), 3),
            "soc":     round(float(data.get("soc",     0)), 1),
            "light":   round(float(data.get("light",   0)), 1),
            "soil":    round(float(data.get("soil",    0)), 1),
            "mode":    mode,
        })


# ─────────────────────────────────────────
# 에너지 모드 결정
# SOC 기반 규칙 + GRU 선제 강등
# ─────────────────────────────────────────
def decide_mode(soc: float, pred_power) -> int:
    if soc < 20:   base = 0   # 긴급절전
    elif soc < 60: base = 1   # 절약
    else:          base = 2   # 풀가동

    # GRU 예측 발전량이 낮으면 한 단계 선제 강등
    if pred_power is not None and pred_power < 1.0 and base > 0:
        base -= 1

    return base


# ─────────────────────────────────────────
# GRU 추론 — 다음 1시간 평균 발전량 예측
#
# feature 순서: train_gru.py의 FEATURES와 반드시 동일해야 함
#   ["power", "wind_power", "voltage", "light", "temp", "hour_sin", "hour_cos"]
# ─────────────────────────────────────────
def predict_power():
    if not gru_ready or len(recent_buffer) < SEQ_LEN:
        return None

    try:
        seq = []
        for row in recent_buffer:
            hour = datetime.now().hour
            seq.append([
                row.get("power",      0),
                row.get("wind_power", 0),
                row.get("voltage",    0),
                row.get("light",      0),
                row.get("temp",       0),
                np.sin(2 * np.pi * hour / 24),   # hour_sin
                np.cos(2 * np.pi * hour / 24),   # hour_cos
            ])

        seq_np     = np.array(seq, dtype=np.float32)           # (30, 7)
        seq_scaled = scaler_gru.transform(seq_np)              # 정규화
        x          = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 30, 7)

        with torch.no_grad():
            pred_scaled = gru_model(x).cpu().numpy()           # (1, 1) 정규화된 예측값

        # target_scaler로 역변환 → 실제 W 단위
        pred_real = scaler_target.inverse_transform(pred_scaled)[0, 0]
        return max(0.0, pred_real)

    except Exception as e:
        print(f"GRU 추론 오류: {e}")
        return None


# ─────────────────────────────────────────
# 급수 분류 추론
# 모델 없으면 토양습도 40% 기준 규칙 기반
# ─────────────────────────────────────────
def predict_water(soil: float, temp: float, hum: float) -> bool:
    if not water_ready:
        return soil < SOIL_THRESHOLD

    try:
        features        = np.array([[soil, temp, hum]], dtype=np.float32)
        features_scaled = scaler_water.transform(features)
        x               = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = water_model(x).item()

        return prob >= WATER_THRESHOLD

    except Exception as e:
        print(f"급수 추론 오류: {e}")
        return soil < SOIL_THRESHOLD


# ─────────────────────────────────────────
# /sensor — 아두이노 데이터 수신
# ─────────────────────────────────────────
@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400

    recent_buffer.append(data)

    pred_power  = predict_power()
    water_alert = predict_water(
        float(data.get("soil", 50)),
        float(data.get("temp", 25)),
        float(data.get("hum",  50))
    )

    soc  = float(data.get("soc", 50))
    mode = decide_mode(soc, pred_power)

    save_csv(data, mode)

    print(f"[수신] 토양:{data.get('soil')}% 조도:{data.get('light')}% "
          f"온도:{data.get('temp')}° → mode:{mode} water:{water_alert} "
          f"pred:{f'{pred_power:.2f}W' if pred_power is not None else 'GRU대기중'}")

    return jsonify({
        "mode":        mode,
        "water_alert": water_alert,
        "pred_power": round(pred_power, 2) if pred_power is not None else None
    })


# ─────────────────────────────────────────
# /data — 최근 50개 데이터 조회
# ─────────────────────────────────────────
@app.route("/data", methods=["GET"])
def get_data():
    if not os.path.exists(CSV_FILE):
        return jsonify([])
    rows = []
    with open(CSV_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return jsonify(rows[-50:])


# ─────────────────────────────────────────
# /stats — 통계
# ─────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    if not os.path.exists(CSV_FILE):
        return jsonify({"count": 0, "avg_temp": 0, "avg_soil": 0})
    rows = []
    with open(CSV_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return jsonify({"count": 0, "avg_temp": 0, "avg_soil": 0})

    return jsonify({
        "count":       len(rows),
        "avg_temp":    round(sum(float(r["temp"]) for r in rows) / len(rows), 1),
        "avg_soil":    round(sum(float(r["soil"]) for r in rows) / len(rows), 1),
        "gru_ready":   gru_ready,
        "water_ready": water_ready,
        "buffer_size": len(recent_buffer),
    })


# ─────────────────────────────────────────
# /status — 모델 상태 확인
# ─────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "gru_model":   "✅ 로드됨" if gru_ready   else "⚠️ 없음 (규칙 기반 동작 중)",
        "water_model": "✅ 로드됨" if water_ready  else "⚠️ 없음 (규칙 기반 동작 중)",
        "buffer":      f"{len(recent_buffer)}/{SEQ_LEN} (GRU 추론 {'가능' if len(recent_buffer) >= SEQ_LEN else '대기중'})",
        "csv_rows":    sum(1 for _ in open(CSV_FILE)) - 1 if os.path.exists(CSV_FILE) else 0,
    })


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)