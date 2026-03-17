# ================================================
# train_gru.py
# GRU 발전량 예측 모델 학습
#
# [이 파일이 하는 일]
#   sensor_log.csv를 읽어서 GRU 딥러닝 모델을 학습
#   "과거 60분치 데이터를 보고 → 앞으로 1시간 평균 발전량" 예측
#
# [실행 후 생성되는 파일]
#   backend/models/gru.pth
#   backend/models/scaler_gru.pkl
#   backend/models/scaler_gru_target.pkl
#   backend/models/gru_result.png
#
# [입력 변수 7개]
#   solar_power   : 태양광 발전량
#   solar_voltage : 전압
#   light         : 조도
#   temperature   : 온도
#   hour_sin      : 시간 sin 인코딩
#   hour_cos      : 시간 cos 인코딩
#   soc           : 배터리 잔량
# ================================================

import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────
# 어느 폴더에서 실행해도 backend/models/ 에 저장됨
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "backend", "models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "backend", "data")
CSV_PATH   = os.path.join(DATA_DIR, "sensor_log.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

SEQ_LEN  = 60
PRED_LEN = 60
BATCH    = 32
EPOCHS   = 200
LR       = 0.001
PATIENCE = 20

FEATURES = [
    "solar_power",
    "solar_voltage",
    "light",
    "temperature",
    "hour_sin",
    "hour_cos",
    "soc",
]
TARGET = "solar_power"


# ── GRU 모델 ───────────────────────────────────
# ★ app.py의 PowerPredictionGRU와 완전히 동일해야 함!
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_sequences(X, y, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    X_list, y_list = [], []
    for i in range(len(X) - seq_len - pred_len):
        X_list.append(X[i : i + seq_len])
        y_list.append(y[i + seq_len : i + seq_len + pred_len].mean())
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32).reshape(-1, 1),
    )


def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, wait = float("inf"), None, 0
    train_hist, val_hist = [], []

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
        train_hist.append(t_loss); val_hist.append(v_loss)

        if epoch % 10 == 0:
            print(f"   Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.6f}  Val: {v_loss:.6f}")

        if v_loss < best_val:
            best_val   = v_loss
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(MODELS_DIR, "gru.pth"))
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n   Early Stopping! (epoch {epoch})")
                break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    print(f"CSV 경로: {CSV_PATH}")
    print(f"모델 저장: {MODELS_DIR}")

    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    df = df[df["solar_power"] >= 0]
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df = add_time_features(df)

    if "light" not in df.columns:
        df["light"] = 0.0
    if "soc" not in df.columns:
        df["soc"] = 100.0

    print(f"데이터: {len(df)}행")

    n        = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    test_df  = df.iloc[int(n * 0.8):]

    f_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()

    train_X = f_scaler.fit_transform(train_df[FEATURES])
    test_X  = f_scaler.transform(test_df[FEATURES])
    train_y = t_scaler.fit_transform(train_df[[TARGET]])
    test_y  = t_scaler.transform(test_df[[TARGET]])

    joblib.dump(f_scaler, os.path.join(MODELS_DIR, "scaler_gru.pkl"))
    joblib.dump(t_scaler, os.path.join(MODELS_DIR, "scaler_gru_target.pkl"))
    print("스케일러 저장 완료")

    X_all,  y_all  = create_sequences(train_X, train_y)
    X_test, y_test = create_sequences(test_X,  test_y)

    val_size = int(len(X_all) * 0.2)
    X_tr,  y_tr  = X_all[:-val_size], y_all[:-val_size]
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    model = PowerPredictionGRU(input_size=len(FEATURES)).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")
    model, train_hist, val_hist = train(model, train_loader, val_loader, device)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    pred_w = t_scaler.inverse_transform(pred)
    true_w = t_scaler.inverse_transform(y_test)

    mae  = mean_absolute_error(true_w, pred_w)
    rmse = math.sqrt(mean_squared_error(true_w, pred_w))
    r2   = r2_score(true_w, pred_w)
    print(f"\n MAE: {mae:.3f}W  RMSE: {rmse:.3f}W  R²: {r2:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(train_hist, label="Train"); axes[0].plot(val_hist, label="Val")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    n = min(200, len(pred_w))
    axes[1].plot(true_w[:n], label="실제", color="steelblue")
    axes[1].plot(pred_w[:n], label=f"예측 (MAE={mae:.2f}W)", color="tomato", linestyle="--")
    axes[1].set_title("실제 vs 예측 발전량"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "gru_result.png"), dpi=150)
    print(f"그래프 저장 → {MODELS_DIR}/gru_result.png")
    print("✅ 완료! 다음: python train_water.py")


if __name__ == "__main__":
    main()