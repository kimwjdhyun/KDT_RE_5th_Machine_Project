# ================================================
# train_gru.py
# GRU 발전량 예측 모델 학습
#
# [입력] 과거 60스텝(5시간)의 센서 데이터
# [출력] 다음 1시간 평균 발전량(W) 예측
#
# 실행하면 → models/gru.pth, scaler_gru.pkl, scaler_gru_target.pkl 생성
# .pth = 모델 가중치 (PyTorch 전용)
#        GRU가 수천 번 학습하면서 알아낸 "패턴"이 숫자로 저장된 파일
#        이게 없으면 모델은 그냥 빈 껍데기

# .pkl = 스케일러 (scikit-learn 전용)
#        학습 데이터의 최솟값/최댓값을 기억해두는 파일
#        예) power의 범위가 0~12W라는 걸 저장해둠
#        이게 없으면 새 데이터를 어떤 기준으로 정규화해야 할지 모름
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
os.makedirs("models", exist_ok=True)

# 재현성 고정: random, numpy, torch 세 가지 모두 시드 고정해야
# 매번 실행해도 같은 결과가 나옴 (하나라도 빠지면 결과가 달라질 수 있음)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── 하이퍼파라미터 ──────────────────────────
SEQ_LEN  = 60    # 입력 시퀀스 길이: 60스텝 = 5분 × 60 = 5시간치 데이터
PRED_LEN = 12    # 예측 구간: 12스텝 = 5분 × 12 = 1시간
                 # 단일 시점 값 대신 '1시간 평균'으로 예측 → 노이즈에 강건
BATCH    = 32
EPOCHS   = 200
LR       = 0.001
PATIENCE = 20    # 검증 손실이 20번 연속 개선 없으면 학습 조기 종료

# ── 입력 변수(Feature) ───────────────────────
# current 제거: current = power/voltage 라서 두 값이 있으면 중복
# minute_sin/cos 제거: 5분 단위 데이터에서 효과 미미
# hour → hour_sin/cos 교체: 23시와 0시가 수치적으로 가깝게 표현됨
#   (숫자 그대로 쓰면 23과 0의 거리가 23으로 멀어 보여서 GRU가 연속성을 못 파악)
FEATURES = [
    "power",        # 태양광 발전량 (W) ← 예측 대상이자 핵심 입력
    "wind_power",   # 풍력 발전량 (W)   ← 야간/흐린 날 태양광 보완
    "voltage",      # 발전 전압 (V)
    "light",        # 조도 (%)          ← 태양광 발전의 핵심 변수
    "temp",         # 온도 (°C)
    "hour_sin",     # 시간 sin 인코딩   ← 숫자 hour 대신 사용
    "hour_cos",     # 시간 cos 인코딩   ← sin+cos 쌍으로 시간 순환성 표현
]
TARGET = "power"


# ================================================
# GRU 모델
#
# GRU(Gated Recurrent Unit): 시계열 패턴을 기억하는 딥러닝 구조.
# 과거 데이터의 흐름을 내부 상태(hidden state)에 압축해서 다음 값을 예측.
#
# num_layers=1일 때 dropout은 레이어 "사이"에 적용되는데
# 레이어가 1개면 사이가 없어서 적용 불가 → 조건부 처리 필요
# ================================================
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,  # num_layers=1이면 dropout 불필요
            batch_first = True,   # 입력 형태: (배치, 시퀀스, 특성)
        )
        # GRU 마지막 출력(hidden_size=32) → 발전량 1개 값으로 압축
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])   # 마지막 시점 출력만 FC에 통과


# ================================================
# 시간 sin/cos 변환
#
# 왜 sin/cos로 변환하는가?
#   hour=23(밤 11시)과 hour=0(자정)은 실제로 1시간 차이인데
#   숫자 그대로 쓰면 GRU가 두 값의 거리를 23으로 인식함 (엄청 멀어 보임!)
#   sin/cos로 바꾸면 23시와 0시의 거리가 수치적으로도 매우 가까워짐.
#
#   sin만으론 안 되는 이유: 3시와 9시가 같은 sin값 → 구분 불가
#   sin+cos 쌍: 24시간 중 각 시각이 고유한 좌표 → 완벽하게 구분 가능
# ================================================
def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ================================================
# 슬라이딩 윈도우로 시퀀스 생성
#
# 예시 (seq_len=60, pred_len=12):
#   i=0: X=[0~59행 데이터], y=[60~71행 발전량 평균]
#   i=1: X=[1~60행 데이터], y=[61~72행 발전량 평균]
#   ...
#
# 단일 시점 예측 대신 '구간 평균' 예측:
#   "60분 뒤 그 순간 몇W?" 보다
#   "앞으로 60분 평균 몇W?" 가 노이즈에 강하고 에너지 계획에 더 실용적
# ================================================
def create_sequences(X, y, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    X_list, y_list = [], []
    for i in range(len(X) - seq_len - pred_len):
        X_list.append(X[i : i + seq_len])
        y_list.append(y[i + seq_len : i + seq_len + pred_len].mean())   # 구간 평균
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32).reshape(-1, 1),
    )


# ================================================
# 학습
# ================================================
def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()   # MSE: 연속값 예측의 표준 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, wait = float("inf"), None, 0
    train_hist, val_hist = [], []

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
        train_hist.append(t_loss); val_hist.append(v_loss)

        if epoch % 10 == 0:
            print(f"   Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.6f}  Val: {v_loss:.6f}")

        # ── Early Stopping: 검증 손실이 개선되지 않으면 조기 종료 ──
        if v_loss < best_val:
            best_val   = v_loss
            best_state = model.state_dict()
            torch.save(best_state, "models/gru.pth")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n   Early Stopping! (epoch {epoch})")
                break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


# ================================================
# 메인
# ================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ① 데이터 로드
    df = pd.read_csv("farm_data.csv")
    df = df.dropna()
    df = df[df["power"] >= 0]
    df["hour"]   = pd.to_datetime(df["timestamp"]).dt.hour
    df["minute"] = pd.to_datetime(df["timestamp"]).dt.minute
    df = add_time_features(df)

    # light 컬럼이 없으면 0으로 채움 (구버전 farm_data.csv 대비)
    if "light" not in df.columns:
        df["light"] = 0.0

    print(f"데이터: {len(df)}행")

    # ② 분리 → 정규화 (순서 중요!)
    # 잘못된 방법: 전체 정규화 후 분리 → 테스트 정보가 스케일러에 새어들어가 성능 부풀림
    # 올바른 방법: 분리 먼저 → train으로만 스케일러 학습 → test는 변환만
    n = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    test_df  = df.iloc[int(n * 0.8):]

    f_scaler = MinMaxScaler()
    t_scaler = MinMaxScaler()
    train_X = f_scaler.fit_transform(train_df[FEATURES])
    test_X  = f_scaler.transform(test_df[FEATURES])
    train_y = t_scaler.fit_transform(train_df[[TARGET]])
    test_y  = t_scaler.transform(test_df[[TARGET]])

    joblib.dump(f_scaler, "models/scaler_gru.pkl")
    joblib.dump(t_scaler, "models/scaler_gru_target.pkl")
    print("스케일러 저장 완료")

    # ③ 시퀀스 생성
    X_all, y_all = create_sequences(train_X, train_y)
    X_test, y_test = create_sequences(test_X, test_y)

    # train 안에서 val 분리
    # → val로 Early Stopping 판단, test는 최종 평가 전용으로만 사용
    val_size = int(len(X_all) * 0.2)
    X_tr, y_tr = X_all[:-val_size], y_all[:-val_size]
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(X_test)}")

    # ④ DataLoader
    # train은 shuffle=True: 매 배치마다 다른 순서 → 특정 패턴 과적합 방지
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader   = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    # ⑤ 모델 생성 & 학습
    model = PowerPredictionGRU(input_size=len(FEATURES)).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")

    model, train_hist, val_hist = train(model, train_loader, val_loader, device)

    # ⑥ 평가
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    # target_scaler로 바로 역변환 (발전량 전용 스케일러라 한 줄로 깔끔하게 처리)
    pred_w = t_scaler.inverse_transform(pred)
    true_w = t_scaler.inverse_transform(y_test)

    mae  = mean_absolute_error(true_w, pred_w)
    rmse = math.sqrt(mean_squared_error(true_w, pred_w))
    r2   = r2_score(true_w, pred_w)
    print(f"\n MAE: {mae:.3f}W  RMSE: {rmse:.3f}W  R²: {r2:.3f}")

    # ⑦ 시각화
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss 곡선: train/val이 같이 내려가면 정상, val만 올라가면 과적합 신호
    axes[0].plot(train_hist, label="Train"); axes[0].plot(val_hist, label="Val")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # 실제 vs 예측 비교
    n = min(200, len(pred_w))
    axes[1].plot(true_w[:n], label="실제", color="steelblue")
    axes[1].plot(pred_w[:n], label=f"예측 (MAE={mae:.2f}W)", color="tomato", linestyle="--")
    axes[1].set_title("실제 vs 예측 발전량"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/gru_result.png", dpi=150)
    print("그래프 저장 → models/gru_result.png")
    print("✅ 완료! 다음: python train_water.py")


if __name__ == "__main__":
    main()