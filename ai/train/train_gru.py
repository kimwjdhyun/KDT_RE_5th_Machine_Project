# ================================================
# train_gru.py
# GRU 발전량 예측 모델 학습
#
# [목표]
#   sensor_log.csv를 읽어서 GRU 모델을 학습
#   "과거 1시간(5분 단위 12개)을 보고 → 1시간 뒤 시점 발전량" 예측
#
# [XGBoost와 비교 기준 통일]
#   - 원본 10초 데이터 → 5분 평균 다운샘플링
#   - 입력 시퀀스 길이: 12 step (1시간)
#   - 예측 목표: 12 step 뒤 solar_power (1시간 뒤 시점값)
#
# [실행 후 생성 파일]
#   backend/models/gru.pth
#   backend/models/scaler_gru.pkl
#   backend/models/scaler_gru_target.pkl
#   backend/results/gru_prediction_result.csv
#   backend/results/gru_result.png
# ================================================

import os
import math
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "backend", "results")
CSV_PATH = os.path.join(DATA_DIR, "sensor_log.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 시드 고정 ──────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── 하이퍼파라미터 ─────────────────────────────
SEQ_LEN = 12        # 과거 1시간 (5분 * 12)
PRED_STEP = 12      # 1시간 뒤 시점값
BATCH = 32
EPOCHS = 200
LR = 0.001
PATIENCE = 20

FEATURES = [
    "solar_power",
    "solar_voltage",
    "solar_current",
    "light",
    "temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "soc",
]
TARGET = "solar_power"


# ── 유틸 함수 ──────────────────────────────────
def ensure_column(df, col_name, default_value=0):
    if col_name not in df.columns:
        df[col_name] = default_value
    return df


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_sequences(X, y, timestamps, seq_len=SEQ_LEN, pred_step=PRED_STEP):
    """
    X: feature array
    y: target array
    timestamps: 각 row의 timestamp
    입력: 과거 seq_len개
    타깃: 마지막 입력 시점으로부터 pred_step 뒤의 시점값
    """
    X_list, y_list, ts_list = [], [], []

    for i in range(len(X) - seq_len - pred_step + 1):
        x_seq = X[i : i + seq_len]
        y_target = y[i + seq_len - 1 + pred_step]
        target_ts = timestamps[i + seq_len - 1 + pred_step]

        X_list.append(x_seq)
        y_list.append(y_target)
        ts_list.append(target_ts)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32).reshape(-1, 1),
        np.array(ts_list),
    )


# ── GRU 모델 ───────────────────────────────────
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ── 학습 함수 ──────────────────────────────────
def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    wait = 0
    train_hist = []
    val_hist = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.6f}  Val: {v_loss:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(MODELS_DIR, "gru.pth"))
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\nEarly Stopping! (epoch {epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_hist, val_hist


# ── 메인 ───────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CSV 경로:", CSV_PATH)
    print("모델 저장:", MODELS_DIR)
    print("결과 저장:", RESULT_DIR)
    print("장치:", device)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

    # 1. CSV 읽기
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip", encoding="utf-8")
    except TypeError:
        df = pd.read_csv(CSV_PATH, error_bad_lines=False, encoding="utf-8")

    print(f"원본 데이터 수: {len(df)}")

    # 2. 필수 컬럼 보정
    needed_cols = [
        "timestamp",
        "solar_power",
        "solar_voltage",
        "solar_current",
        "light",
        "temperature",
        "humidity",
        "soc",
    ]

    optional_cols = [
        "battery_voltage",
        "battery_current",
        "battery_power",
        "soil",
        "soil_raw",
        "led_brightness",
    ]

    for col in needed_cols:
        df = ensure_column(df, col, 0)

    for col in optional_cols:
        df = ensure_column(df, col, 0)

    numeric_cols = needed_cols[1:] + optional_cols
    df = safe_numeric(df, numeric_cols)

    # 3. timestamp 처리 및 정렬
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. solar_power 보정
    df["solar_power"] = pd.to_numeric(df["solar_power"], errors="coerce")
    df["solar_power"] = df["solar_power"].fillna(df["solar_voltage"] * df["solar_current"])
    df["solar_power"] = df["solar_power"].clip(lower=0)

    # 5. 결측 보정
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    df = df.dropna(subset=["solar_power"]).copy()

    # 6. 5분 다운샘플링
    df = df.set_index("timestamp")
    df = df.resample("5min").mean(numeric_only=True)
    df = df.dropna(how="all").reset_index()

    print(f"5분 다운샘플링 후 데이터 수: {len(df)}")

    if len(df) < (SEQ_LEN + PRED_STEP + 30):
        raise ValueError("다운샘플링 후 데이터가 너무 적습니다. 더 수집한 뒤 다시 실행하세요.")

    # 7. 시간 파생변수
    df = add_time_features(df)

    # 8. 필요한 컬럼만 정리
    model_cols = ["timestamp"] + FEATURES + [TARGET]
    model_cols = list(dict.fromkeys(model_cols))  # 중복 제거
    df = df[model_cols].copy()

    df = safe_numeric(df, FEATURES + [TARGET])
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    print(f"전처리 후 데이터 수: {len(df)}")

    # 9. train / test 분리 (시간순)
    n = len(df)
    split_idx = int(n * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"학습 원본 구간: {len(train_df)}")
    print(f"테스트 원본 구간: {len(test_df)}")

    if len(train_df) < (SEQ_LEN + PRED_STEP + 10):
        raise ValueError("학습 구간 데이터가 너무 적습니다.")
    if len(test_df) < (SEQ_LEN + PRED_STEP + 5):
        raise ValueError("테스트 구간 데이터가 너무 적습니다.")

    # 10. 스케일링
    f_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()

    train_X_scaled = f_scaler.fit_transform(train_df[FEATURES])
    test_X_scaled = f_scaler.transform(test_df[FEATURES])

    train_y_scaled = t_scaler.fit_transform(train_df[[TARGET]])
    test_y_scaled = t_scaler.transform(test_df[[TARGET]])

    joblib.dump(f_scaler, os.path.join(MODELS_DIR, "scaler_gru.pkl"))
    joblib.dump(t_scaler, os.path.join(MODELS_DIR, "scaler_gru_target.pkl"))
    print("스케일러 저장 완료")

    # 11. 시퀀스 생성
    X_all, y_all, ts_all = create_sequences(
        train_X_scaled,
        train_y_scaled,
        train_df["timestamp"].values,
        seq_len=SEQ_LEN,
        pred_step=PRED_STEP,
    )

    X_test, y_test, ts_test = create_sequences(
        test_X_scaled,
        test_y_scaled,
        test_df["timestamp"].values,
        seq_len=SEQ_LEN,
        pred_step=PRED_STEP,
    )

    if len(X_all) == 0 or len(X_test) == 0:
        raise ValueError("시퀀스 생성 결과가 비었습니다. 데이터 길이를 확인하세요.")

    # 12. train / val 분리
    val_size = max(1, int(len(X_all) * 0.2))

    X_tr, y_tr = X_all[:-val_size], y_all[:-val_size]
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]

    print(f"시퀀스 기준 학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH,
        shuffle=False,
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH,
        shuffle=False,
    )

    # 13. 모델 학습
    model = PowerPredictionGRU(input_size=len(FEATURES)).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")

    model, train_hist, val_hist = train(model, train_loader, val_loader, device)

    # 14. 예측
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test).to(device)).cpu().numpy()

    pred_w = t_scaler.inverse_transform(pred_scaled).reshape(-1)
    true_w = t_scaler.inverse_transform(y_test).reshape(-1)

    # 15. 성능 평가
    mae = mean_absolute_error(true_w, pred_w)
    rmse = math.sqrt(mean_squared_error(true_w, pred_w))
    r2 = r2_score(true_w, pred_w)

    print("\n===== GRU Result =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # 추가 확인용 샘플 출력
    print("\n===== 실제값 / 예측값 샘플 10개 =====")
    for i in range(min(10, len(pred_w))):
        print(f"{i+1:2d}. true={true_w[i]:.4f}, pred={pred_w[i]:.4f}")

    # 16. 결과 저장
    result_df = pd.DataFrame({
        "timestamp": ts_test,
        "actual_power_1h": true_w,
        "pred_power_1h": pred_w,
    })

    result_path = os.path.join(RESULT_DIR, "gru_prediction_result.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n예측 결과 저장 → {result_path}")

    # 17. 그래프 저장
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_hist, label="Train")
    axes[0].plot(val_hist, label="Val")
    axes[0].set_title("GRU Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    plot_n = min(100, len(result_df))
    axes[1].plot(
        result_df["timestamp"].iloc[:plot_n],
        result_df["actual_power_1h"].iloc[:plot_n],
        label="Actual",
        linewidth=2,
    )
    axes[1].plot(
        result_df["timestamp"].iloc[:plot_n],
        result_df["pred_power_1h"].iloc[:plot_n],
        label=f"Pred (MAE={mae:.3f})",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_title("GRU: Actual vs Predicted Solar Power")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    plot_path = os.path.join(RESULT_DIR, "gru_result.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"그래프 저장 → {plot_path}")
    print("\n✅ GRU 학습 완료!")


if __name__ == "__main__":
    main()