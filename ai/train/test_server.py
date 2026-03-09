# ================================================
# test_server.py
# Flask 서버 없이 AI 추론만 단독 테스트
#
# 학습된 모델이 정상 동작하는지 확인하는 용도.
# 아두이노/Flask 없이도 실행 가능.
#
# 실행 전 필요한 파일:
#   models/gru.pth, models/water.pth
#   models/scaler_gru.pkl, models/scaler_gru_target.pkl
#   models/scaler_water.pkl
# ================================================

import torch
import torch.nn as nn
import numpy as np
import joblib
from collections import deque

# ── FEATURES: train_gru_v2.py와 순서까지 완전히 일치해야 함 ──
# 순서가 다르면 스케일러가 엉뚱한 열을 정규화해서 예측값이 틀어짐
FEATURES = ["power", "wind_power", "voltage", "light", "temp", "hour_sin", "hour_cos"]


# ── GRU 모델 클래스: train_gru_v2.py와 완전히 동일하게 유지 ──
# 클래스 구조가 다르면 저장된 .pth 가중치를 불러올 때 에러 발생
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ── 급수 분류 모델: train_water.py와 완전히 동일하게 유지 ──
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),   # 입력: 토양습도, 온도, 공기습도
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid() # 출력: 급수 필요 확률 (0~1)
        )
    def forward(self, x):
        return self.net(x)


# ── 모델 & 스케일러 로드 ──────────────────────────────
gru_model   = PowerPredictionGRU(input_size=len(FEATURES))
water_model = WaterClassifier()

# map_location="cpu": GPU 없는 환경에서도 실행 가능하도록 CPU 강제 지정
gru_model.load_state_dict(torch.load("models/gru.pth",    map_location="cpu"))
water_model.load_state_dict(torch.load("models/water.pth", map_location="cpu"))

# eval(): 추론 모드 전환 → Dropout 등 학습 전용 기능 비활성화 (추론 시 반드시 호출)
gru_model.eval(); water_model.eval()

f_scaler = joblib.load("models/scaler_gru.pkl")           # feature 정규화용
t_scaler = joblib.load("models/scaler_gru_target.pkl")    # 발전량 역변환용
w_scaler = joblib.load("models/scaler_water.pkl")

print("✅ 모델 & 스케일러 로드 완료")


# ── GRU 발전량 예측 테스트 ────────────────────────────
print("\n" + "=" * 45)
print("GRU 발전량 예측 테스트")
print("=" * 45)

# 버퍼에 60개(SEQ_LEN) 가짜 센서값 채우기
# 실제 서버에서는 아두이노가 5분마다 데이터를 보내면 여기에 쌓임
# deque(maxlen=60): 새 데이터 추가 시 오래된 것 자동 제거 (FIFO 구조)
buffer = deque(maxlen=60)
hour = 14   # 오후 2시 시나리오

for _ in range(60):
    # sin/cos 변환: 숫자 hour 대신 사용 (23시와 0시가 수치적으로도 가깝게 표현됨)
    buffer.append({
        "power":      8.0  + np.random.normal(0, 0.2),
        "wind_power": 2.5  + np.random.normal(0, 0.3),
        "voltage":    11.2 + np.random.normal(0, 0.1),
        "light":      85.0 + np.random.normal(0, 1.0),
        "temp":       24.5 + np.random.normal(0, 0.1),
        "hour_sin":   np.sin(2 * np.pi * hour / 24),
        "hour_cos":   np.cos(2 * np.pi * hour / 24),
    })

# 버퍼 → numpy 배열 → 정규화 → 3D 텐서
seq    = np.array([[r[f] for f in FEATURES] for r in buffer], dtype=np.float32)
seq_sc = f_scaler.transform(seq)
# unsqueeze(0): (60, 7) → (1, 60, 7) ← GRU는 3D 입력 필요 (배치, 시퀀스, 특성)
x      = torch.tensor(seq_sc).unsqueeze(0)

with torch.no_grad():
    pred_sc = gru_model(x).cpu().numpy()

# target 전용 스케일러로 역변환 → 실제 W 단위 복원
pred_w = max(0.0, t_scaler.inverse_transform(pred_sc)[0, 0])
print(f"상황: 오후 2시, 조도 85%, 태양광 약 8W + 풍력 약 2.5W")
print(f"→ 다음 1시간 평균 발전량 예측: {pred_w:.2f} W")


# ── 급수 분류 테스트 ──────────────────────────────────
print("\n" + "=" * 45)
print("급수 분류 테스트")
print("=" * 45)

scenarios = [
    {"soil": 65.0, "temp": 24.5, "hum": 60.0, "label": "정상 (촉촉)"},
    {"soil": 20.0, "temp": 26.0, "hum": 50.0, "label": "매우 건조 → 즉시 급수"},
]

for s in scenarios:
    feat   = np.array([[s["soil"], s["temp"], s["hum"]]], dtype=np.float32)
    feat_sc = w_scaler.transform(feat)
    with torch.no_grad():
        prob = water_model(torch.tensor(feat_sc)).item()

    result = "💧 급수 필요" if prob >= 0.5 else "✅ 정상"
    print(f"[{s['label']}] 토양:{s['soil']}%  → 확률:{prob:.3f}  {result}")


# ── 에너지 모드 결정 테스트 ───────────────────────────
print("\n" + "=" * 45)
print("에너지 모드 결정 테스트 (규칙 + GRU 선제 강등)")
print("=" * 45)

def decide_mode(soc, pred_power):
    """
    SOC 기반 기본 모드 결정 + GRU 예측으로 선제 강등.

    기본 규칙:
      SOC < 20%  → 0: 긴급절전
      SOC < 60%  → 1: 절약
      SOC >= 60% → 2: 풀가동

    선제 강등:
      예측 발전량 < 1W이면 한 단계 낮춤
      (예: SOC=70%라도 발전 예측이 낮으면 절약 모드로 미리 전환 → 과방전 방지)
    """
    mode = 0 if soc < 20 else (1 if soc < 60 else 2)
    if pred_power is not None and pred_power < 1.0 and mode > 0:
        mode -= 1
    return mode

labels = {0: "🔴 긴급절전", 1: "🟡 절약", 2: "🟢 풀가동"}

for soc, pred, desc in [
    (75.0, 8.0, "SOC 충분 + 발전 충분"),
    (65.0, 0.3, "SOC 높지만 발전 낮음 → 선제 강등"),
    (15.0, 5.0, "SOC 위험"),
]:
    mode = decide_mode(soc, pred)
    print(f"[{desc}]  SOC:{soc}%  예측:{pred}W  →  {labels[mode]}")

print("\n✅ 전체 추론 테스트 완료!")