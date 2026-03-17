# ================================================
# train_water.py
# 급수 필요 여부 이진 분류 모델 학습
#
# [이 파일이 하는 일]
#   토양습도, 온도, 공기습도를 보고
#   "지금 물을 줘야 하나?" 를 AI가 판단하도록 학습
#
# [실행 후 생성되는 파일]
#   models/water.pth
#   models/scaler_water.pkl
#   models/water_result.png
#
# ★ sensor_log.csv 컬럼명 기준으로 작성
#   soil        : 토양습도
#   temperature : 온도
#   humidity    : 공기습도
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
os.makedirs("models", exist_ok=True)  # ★ 경로 수정

random.seed(42); np.random.seed(42); torch.manual_seed(42)

BATCH    = 32
EPOCHS   = 100
LR       = 0.001
PATIENCE = 15

WATER_THRESHOLD = 40.0

# ★ sensor_log.csv 컬럼명에 맞게 수정
FEATURES = ["soil", "temperature", "humidity"]
TARGET   = "water_label"


# ================================================
# 급수 분류 모델 (WaterClassifier)
# ★ app_gru.py의 WaterClassifier와 완전히 동일해야 함!
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


def train(model, train_loader, val_loader, device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

        t_loss = np.mean(losses)
        v_loss = np.mean(val_losses)

        if epoch % 10 == 0:
            print(f"   Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.4f}  Val: {v_loss:.4f}")

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ★ sensor_log.csv 경로 (backend/ 폴더에서 실행 기준)
    df = pd.read_csv("data/sensor_log.csv")
    df = df.dropna()
    df = df[df["soil"].between(0, 100)]
    df = df[df["temperature"].between(0, 50)]
    df = df[df["humidity"].between(0, 100)]

    df[TARGET] = (df["soil"] <= WATER_THRESHOLD).astype(float)

    pos_ratio = df[TARGET].mean()
    print(f"데이터: {len(df)}행  |  급수필요(1) 비율: {pos_ratio:.1%}")
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        print("   ⚠️  클래스 불균형이 심함 → WATER_THRESHOLD 값 조정 권장")

    n        = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    test_df  = df.iloc[int(n * 0.8):]

    scaler  = MinMaxScaler()
    train_X = scaler.fit_transform(train_df[FEATURES]).astype(np.float32)
    test_X  = scaler.transform(test_df[FEATURES]).astype(np.float32)
    train_y = train_df[TARGET].values.reshape(-1, 1).astype(np.float32)
    test_y  = test_df[TARGET].values.reshape(-1, 1).astype(np.float32)

    joblib.dump(scaler, "models/scaler_water.pkl")
    print("스케일러 저장 완료")

    val_size = int(len(train_X) * 0.2)
    X_tr, y_tr   = train_X[:-val_size], train_y[:-val_size]
    X_val, y_val = train_X[-val_size:], train_y[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(test_X)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    model = WaterClassifier().to(device)
    model = train(model, train_loader, val_loader, device)

    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(test_X).to(device)).cpu().numpy()

    pred = (prob >= 0.5).astype(int)
    true = test_y.astype(int)

    acc = accuracy_score(true, pred)
    print(f"\n정확도: {acc:.4f} ({acc*100:.1f}%)")
    print("\n[분류 리포트]")
    print(classification_report(true, pred, target_names=["정상(0)", "급수필요(1)"]))

    plt.figure(figsize=(7, 4))
    plt.hist(prob[true.flatten() == 0], bins=30, alpha=0.6, label="실제: 정상(0)", color="steelblue")
    plt.hist(prob[true.flatten() == 1], bins=30, alpha=0.6, label="실제: 급수필요(1)", color="tomato")
    plt.axvline(0.5, color="black", linestyle="--", label="판단 기준 (0.5)")
    plt.xlabel("예측 확률"); plt.ylabel("샘플 수")
    plt.title("급수 분류 모델 - 예측 확률 분포")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/water_result.png", dpi=150)  # ★ 경로 수정
    print("그래프 저장 → models/water_result.png")
    print("✅ 완료!")


if __name__ == "__main__":
    main()