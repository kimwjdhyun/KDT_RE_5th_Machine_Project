# ================================================
# train_water.py
# 급수 필요 여부 이진 분류 모델 학습
#
# [이 파일이 하는 일]
#   토양습도, 온도, 공기습도를 보고
#   "지금 물을 줘야 하나?" 를 AI가 판단하도록 학습
#
# [GRU(발전량 예측)와 비교]
#   GRU         : "발전량이 몇 W일까?" → 숫자 예측 → MSELoss
#   여기(급수)  : "물을 줘야 하나?" → 예/아니오 → BCELoss
#
# [실행 후 생성되는 파일]
#   models/water.pth          → 학습된 모델 가중치
#   models/scaler_water.pkl   → 입력값 정규화 기준
#   models/water_result.png   → 학습 결과 그래프
# ================================================

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("../../backend/models", exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── 하이퍼파라미터 ──
BATCH    = 32
EPOCHS   = 100
LR       = 0.001
PATIENCE = 15   # 검증 손실이 15번 연속 개선 없으면 학습 중단 (Early Stopping)

# 급수 판단 기준: 토양습도가 이 값 이하면 "물 필요" 라벨(1) 붙임
# 작물 종류에 따라 조정 가능 (예: 고추=45%, 상추=50%)
WATER_THRESHOLD = 40.0

FEATURES = ["soil", "temp", "hum"]  # 입력 변수 3개
TARGET   = "water_label"            # 출력: 0(정상) or 1(급수 필요)


# ================================================
# 급수 분류 모델 (WaterClassifier)
#
# [왜 GRU가 아닌 FC(Fully Connected) 레이어?]
#   GRU : 시계열(시간 흐름) 데이터에 특화
#         "과거 60분을 기억해야" 예측할 수 있을 때 사용
#
#   FC  : 지금 이 순간의 값만 보고 판단할 때 사용
#         토양습도 38% = 물 필요 → 과거 흐름 볼 필요 없음!
#         → 단순한 FC로 충분
#
# [모델 구조]
#   입력 3개 → 16개로 확장(패턴 학습) → 8개로 압축 → 1개(확률)
#
# [Sigmoid를 마지막에 붙이는 이유]
#   Linear 마지막 출력은 어떤 값이든 될 수 있음 (-1000 ~ 1000)
#   Sigmoid는 어떤 값이든 0~1 사이로 압축
#   → "급수 필요 확률"로 해석 가능
#   → 0.5 이상이면 급수, 미만이면 대기
#
# ★ app_gru.py의 WaterClassifier와 완전히 동일해야 함!
# ================================================
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),  # 입력 3개 → 16개 (특성 확장, 복잡한 패턴 학습)
            nn.ReLU(),         # 활성화 함수: 음수 → 0, 양수 → 그대로 (비선형성 추가)
            nn.Linear(16, 8),  # 16개 → 8개 (압축)
            nn.ReLU(),
            nn.Linear(8, 1),   # 8개 → 1개 (최종 값)
            nn.Sigmoid()       # 0~1 확률로 변환
        )

    def forward(self, x):
        return self.net(x)


# ================================================
# 학습 함수
#
# [BCELoss란?]
#   Binary Cross Entropy Loss = 이진 분류 전용 손실 함수
#
#   예측=0.9, 정답=1 → 손실 낮음 (잘 맞춤)
#   예측=0.1, 정답=1 → 손실 높음 (크게 틀림)
#   예측=0.5, 정답=1 → 손실 중간
#
#   MSELoss를 쓰면 안 되는 이유:
#   MSE는 (예측-정답)² 인데, 확률(0~1)과 라벨(0,1)의 거리를 제대로 못 잼
#   BCELoss가 이진 분류에 수학적으로 올바른 손실 함수
# ================================================
def train(model, train_loader, val_loader, device):
    criterion = nn.BCELoss()  # 이진 분류 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 단계 ──
        model.train()  # 학습 모드
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()  # 기울기 초기화
            loss.backward()        # 역전파 (기울기 계산)
            optimizer.step()       # 가중치 업데이트
            losses.append(loss.item())

        # ── 검증 단계 ──
        model.eval()  # 평가 모드 (dropout 비활성화 등)
        val_losses = []
        with torch.no_grad():  # 기울기 계산 안 함 (메모리 절약)
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

        t_loss = np.mean(losses)
        v_loss = np.mean(val_losses)

        if epoch % 10 == 0:
            print(f"   Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.4f}  Val: {v_loss:.4f}")

        # ── Early Stopping ──
        if v_loss < best_val:
            best_val   = v_loss
            best_state = model.state_dict()
            torch.save(best_state, "models/water.pth")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n   Early Stopping! (epoch {epoch})")
                break

    model.load_state_dict(best_state)  # 가장 좋았던 모델로 복원
    return model


# ================================================
# 메인 함수
# ================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ① 데이터 로드 및 이상값 제거
    df = pd.read_csv("data/sensor_log.csv")
    df = df.dropna()
    # 센서 오류로 범위 벗어난 값 제거 (between: 양 끝값 포함)
    df = df[df["soil"].between(0, 100)]
    df = df[df["temp"].between(0, 50)]
    df = df[df["hum"].between(0, 100)]

    # ② 급수 라벨 생성
    # 토양습도 40% 이하 → 1(급수 필요), 초과 → 0(정상)
    # generate_data.py에서 35% 이하면 급수했으니
    # 35~40% 구간 = "이미 말라가는 중" → 미리 물 주는 게 맞음
    df[TARGET] = (df["soil"] <= WATER_THRESHOLD).astype(float)

    # ③ 클래스 불균형 확인
    # 급수 필요(1)가 전체의 10% 미만이거나 90% 초과면 모델이 한쪽으로 치우침
    # 예) 95%가 정상 → "항상 정상" 예측해도 정확도 95% → 의미 없는 모델
    pos_ratio = df[TARGET].mean()
    print(f"데이터: {len(df)}행  |  급수필요(1) 비율: {pos_ratio:.1%}")
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        print("   ⚠️  클래스 불균형이 심함 → WATER_THRESHOLD 값 조정 권장")

    # ④ 학습/테스트 분리 → 정규화
    n        = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    test_df  = df.iloc[int(n * 0.8):]

    # MinMaxScaler: 0~1 사이로 정규화
    # 토양습도(0~100), 온도(0~50), 습도(0~100) 단위가 다름
    # → 정규화 안 하면 온도(작은 값)가 무시되고 토양습도만 학습함
    scaler  = MinMaxScaler()
    train_X = scaler.fit_transform(train_df[FEATURES]).astype(np.float32)
    test_X  = scaler.transform(test_df[FEATURES]).astype(np.float32)
    train_y = train_df[TARGET].values.reshape(-1, 1).astype(np.float32)
    test_y  = test_df[TARGET].values.reshape(-1, 1).astype(np.float32)

    joblib.dump(scaler, "models/scaler_water.pkl")
    print("스케일러 저장 완료")

    # ⑤ train에서 val 분리 (Early Stopping 판단용)
    val_size = int(len(train_X) * 0.2)
    X_tr, y_tr   = train_X[:-val_size], train_y[:-val_size]
    X_val, y_val = train_X[-val_size:], train_y[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(test_X)}")

    # ⑥ DataLoader 생성
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    # ⑦ 학습
    model = WaterClassifier().to(device)
    model = train(model, train_loader, val_loader, device)

    # ⑧ 테스트 데이터로 최종 평가
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(test_X).to(device)).cpu().numpy()

    # 확률 0.5 기준으로 0/1로 변환
    pred = (prob >= 0.5).astype(int)
    true = test_y.astype(int)

    acc = accuracy_score(true, pred)
    print(f"\n정확도: {acc:.4f} ({acc*100:.1f}%)")
    print("\n[분류 리포트]")
    # precision : 급수 명령 내렸을 때 실제로 필요했던 비율 (오작동 방지)
    # recall    : 실제 급수 필요한 상황에서 명령 내린 비율 (누락 방지)
    print(classification_report(true, pred, target_names=["정상(0)", "급수필요(1)"]))

    # ⑨ 시각화: 예측 확률 분포
    # 파란 막대(실제 정상)가 0 근처에 몰리고
    # 빨간 막대(실제 급수)가 1 근처에 몰리면 → 잘 학습된 것!
    plt.figure(figsize=(7, 4))
    plt.hist(prob[true.flatten() == 0], bins=30, alpha=0.6, label="실제: 정상(0)", color="steelblue")
    plt.hist(prob[true.flatten() == 1], bins=30, alpha=0.6, label="실제: 급수필요(1)", color="tomato")
    plt.axvline(0.5, color="black", linestyle="--", label="판단 기준 (0.5)")
    plt.xlabel("예측 확률"); plt.ylabel("샘플 수")
    plt.title("급수 분류 모델 - 예측 확률 분포")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../../backend/models/water_result.png", dpi=150)
    print("그래프 저장 → models/water_result.png")
    print("✅ 완료!")


if __name__ == "__main__":
    main()