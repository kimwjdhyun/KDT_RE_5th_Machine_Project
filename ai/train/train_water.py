# ================================================
# train_water.py
# 급수 필요 여부 이진 분류 모델 학습
#
# [입력] 토양습도, 온도, 공기습도 (3개)
# [출력] 급수 필요 확률 (0~1)
#        0.5 이상이면 → 급수 모터 ON
#        0.5 미만이면 → 대기
#
# GRU(발전량 예측)와 다른 점:
#   GRU    → 연속값 예측 (몇 W인가?)  → MSELoss
#   여기서 → 이진 분류 (줄까/말까?)   → BCELoss
#
# 실행하면 → models/water.pth, models/scaler_water.pkl 생성
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
os.makedirs("models", exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── 하이퍼파라미터 ──
BATCH    = 32
EPOCHS   = 100
LR       = 0.001
PATIENCE = 15    # 검증 손실 개선 없으면 조기 종료

# 급수 판단 기준: 토양습도 40% 이하면 물이 필요하다고 라벨링
# 실제 작물 종류에 따라 이 값을 조정하면 됨
WATER_THRESHOLD = 40.0

FEATURES = ["soil", "temp", "hum"]   # 입력 3개
TARGET   = "water_label"             # 라벨: 0(정상) or 1(급수 필요)


# ================================================
# 급수 분류 모델
#
# GRU처럼 복잡한 시계열 구조가 필요 없음.
# 토양습도/온도/습도 3개만 보고 급수 여부를 판단하면 되므로
# 단순한 FC(Fully Connected) 레이어만으로도 충분.
#
# 마지막에 Sigmoid를 붙이는 이유:
#   Sigmoid는 어떤 값이든 0~1 사이로 압축해줌
#   → 이걸 "급수 필요 확률"로 해석할 수 있음
#   → 0.5 이상이면 급수, 미만이면 대기
# ================================================
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),   # 3개 입력 → 16개로 확장해서 패턴 학습
            nn.Linear(16, 8), nn.ReLU(),   # 16 → 8로 압축
            nn.Linear(8, 1),               # 8 → 1개 값으로 최종 압축
            nn.Sigmoid()                   # 0~1 확률로 변환
        )
    def forward(self, x):
        return self.net(x)


# ================================================
# 학습
#
# BCELoss(Binary Cross Entropy):
#   이진 분류(0 or 1)의 표준 손실 함수.
#   예측이 정답에서 멀수록 큰 패널티 부여.
#   MSELoss는 연속값 예측용이라 여기선 사용 불가.
# ================================================
def train(model, train_loader, val_loader, device):
    # BCELoss: 0/1 분류 전용 손실 함수 (MSELoss는 연속값 예측용이라 여기선 사용 불가)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 ──
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())

        # ── 검증 ──
        model.eval()
        val_losses = []
        with torch.no_grad():
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

    model.load_state_dict(best_state)
    return model


# ================================================
# 메인
# ================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ① 데이터 로드
    df = pd.read_csv("farm_data.csv")
    df = df.dropna()
    df = df[df["soil"].between(0, 100)]
    df = df[df["temp"].between(0, 50)]
    df = df[df["hum"].between(0, 100)]

    # ② 라벨 생성
    # 토양습도 40% 이하면 급수 필요(1), 초과면 정상(0)
    # → generate_data.py에서 35% 이하면 급수했으므로
    #   실제 센서 기준으로는 35~40% 사이가 "이미 말라가는 중" 구간
    df[TARGET] = (df["soil"] <= WATER_THRESHOLD).astype(float)

    # 클래스 불균형 확인
    # 급수 필요(1)가 너무 적거나 많으면 모델이 한쪽으로 치우쳐서 학습될 수 있음
    pos_ratio = df[TARGET].mean()
    print(f"데이터: {len(df)}행  |  급수필요(1) 비율: {pos_ratio:.1%}")
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        print("   ⚠️  클래스 불균형이 심함 → WATER_THRESHOLD 값 조정 권장")

    # ③ 분리 → 정규화
    n        = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    test_df  = df.iloc[int(n * 0.8):]

    scaler   = MinMaxScaler()
    train_X  = scaler.fit_transform(train_df[FEATURES]).astype(np.float32)
    test_X   = scaler.transform(test_df[FEATURES]).astype(np.float32)
    train_y  = train_df[TARGET].values.reshape(-1, 1).astype(np.float32)
    test_y   = test_df[TARGET].values.reshape(-1, 1).astype(np.float32)

    joblib.dump(scaler, "models/scaler_water.pkl")
    print("스케일러 저장 완료")

    # ④ train/val 분리
    val_size = int(len(train_X) * 0.2)
    X_tr, y_tr   = train_X[:-val_size], train_y[:-val_size]
    X_val, y_val = train_X[-val_size:], train_y[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(test_X)}")

    # ⑤ DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader   = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    # ⑥ 학습
    model = WaterClassifier().to(device)
    model = train(model, train_loader, val_loader, device)

    # ⑦ 평가
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(test_X).to(device)).cpu().numpy()

    # 확률 0.5 기준으로 0/1 이진 변환
    pred = (prob >= 0.5).astype(int)
    true = test_y.astype(int)

    acc = accuracy_score(true, pred)
    print(f"\n정확도: {acc:.4f} ({acc*100:.1f}%)")
    print("\n[분류 리포트]")
    print(classification_report(true, pred, target_names=["정상(0)", "급수필요(1)"]))

    # ⑧ 시각화: 예측 확률 분포
    # 0에 몰린 막대와 1에 몰린 막대가 뚜렷하게 분리되면 잘 학습된 것
    plt.figure(figsize=(7, 4))
    plt.hist(prob[true.flatten() == 0], bins=30, alpha=0.6, label="실제: 정상(0)", color="steelblue")
    plt.hist(prob[true.flatten() == 1], bins=30, alpha=0.6, label="실제: 급수필요(1)", color="tomato")
    plt.axvline(0.5, color="black", linestyle="--", label="판단 기준 (0.5)")
    plt.xlabel("예측 확률"); plt.ylabel("샘플 수")
    plt.title("급수 분류 모델 - 예측 확률 분포")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/water_result.png", dpi=150)
    print("그래프 저장 → models/water_result.png")
    print("✅ 완료! 다음: python test_server.py")


if __name__ == "__main__":
    main()