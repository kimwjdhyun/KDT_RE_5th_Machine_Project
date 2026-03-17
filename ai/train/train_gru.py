# ================================================
# train_gru.py
# GRU 발전량 예측 모델 학습
#
# [이 파일이 하는 일]
#   farm_data.csv를 읽어서 GRU 딥러닝 모델을 학습
#   "과거 60분치 데이터를 보고 → 앞으로 1시간 평균 발전량" 예측
#
# [GRU가 뭔가?]
#   Gated Recurrent Unit = 시계열(순서 있는) 데이터 전문 딥러닝
#   LSTM의 단순화 버전, 속도 빠르고 성능 비슷
#   "과거 패턴을 기억해서 미래 예측"에 특화
#
# [실행 후 생성되는 파일]
#   models/gru.pth              → 학습된 모델 가중치
#   models/scaler_gru.pkl       → 입력값 정규화 기준 (0~1로 변환)
#   models/scaler_gru_target.pkl → 출력값 정규화 기준 (역변환에 사용)
#   models/gru_result.png       → 학습 결과 그래프
#
# [입력 변수 7개]
#   power     : 태양광 발전량
#   voltage   : 전압
#   light     : 조도
#   temp      : 온도
#   hour_sin  : 시간 sin 인코딩
#   hour_cos  : 시간 cos 인코딩
#   soc       : 배터리 잔량
#
# [wind_power는 왜 없나?]
#   풍력 발전기 제거됨 → 태양광만 사용하는 시스템
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

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
os.makedirs("models/", exist_ok=True)             # models 폴더 없으면 자동 생성

# 재현성 고정: 이 세 줄이 없으면 매번 다른 결과가 나옴
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── 하이퍼파라미터 ─────────────────────────────
# 하이퍼파라미터 = 사람이 직접 정해주는 설정값 (모델이 학습하는 값 아님)

SEQ_LEN  = 60   # 입력 시퀀스 길이: 60스텝 = 1분 × 60 = 과거 1시간치 데이터
PRED_LEN = 60   # 60개 = 60분(1시간) 평균 예측 -> 1시간 평균 발전량

BATCH    = 32   # 한 번에 학습하는 샘플 수 (메모리와 속도 트레이드오프)
EPOCHS   = 200  # 전체 데이터를 몇 번 반복 학습할지
LR       = 0.001 # Learning Rate: 가중치 업데이트 속도 (너무 크면 불안정, 너무 작으면 느림)
PATIENCE = 20   # Early Stopping: 검증 손실이 20번 연속 개선 없으면 학습 중단

# ── 입력 변수 ─────────────────────────────────
# GRU에 넣을 feature 목록 (순서 중요! app_gru.py와 동일해야 함)
FEATURES = [
    "power",     # 태양광 발전량(W) - 예측 대상이자 핵심 입력
    "voltage",   # 발전 전압(V) - 패널 상태 반영
    "light",     # 조도(%) - 태양광 발전의 가장 직접적 원인
    "temp",      # 온도(°C) - 패널 효율에 영향 (고온일수록 효율 감소)
    "hour_sin",  # 시간 sin 인코딩 - 시간 순환성 표현
    "hour_cos",  # 시간 cos 인코딩 - sin만으론 3시/9시 구분 불가
    "soc",       # 배터리 잔량도 발전량 예측에 영향
]
TARGET = "power"  # 예측할 값: 태양광 발전량


# ================================================
# GRU 모델 클래스
#
# [GRU 내부 동작 원리]
#   입력: (배치크기, 시퀀스길이, 특성수) = (32, 60, 6)
#   hidden state: 과거 패턴을 압축해서 기억하는 32차원 벡터
#   출력: 마지막 시점의 hidden state → FC → 발전량 1개 값
#
# [num_layers=1일 때 dropout 못 쓰는 이유]
#   dropout은 레이어 "사이"에 적용
#   레이어가 1개면 사이가 없어서 적용 불가 → 조건부 처리
#
# ★ app_gru.py의 PowerPredictionGRU와 완전히 동일해야 함!
#   다르면 저장한 모델 불러올 때 에러 발생
# ================================================
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,   # 입력 특성 수 (6개)
            hidden_size = hidden_size,  # hidden state 크기 (32)
            num_layers  = num_layers,   # GRU 레이어 수
            dropout     = dropout if num_layers > 1 else 0.0,  # 레이어 1개면 dropout 불필요
            batch_first = True,         # 입력 순서: (배치, 시퀀스, 특성) 형태로 받음
        )
        # GRU 출력(32차원) → 발전량 1개 값으로 압축
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        out, _ = self.gru(x)            # out: (배치, 시퀀스, hidden_size)
        return self.fc(out[:, -1, :])   # 마지막 시점(-1)의 출력만 FC에 통과


# ================================================
# 시간 sin/cos 변환 함수
#
# [왜 숫자 hour 대신 sin/cos를 쓰나?]
#   hour=23(밤 11시)과 hour=0(자정)은 실제로 1시간 차이인데
#   숫자 그대로 쓰면 GRU가 두 값의 거리를 23으로 인식 (엄청 멀어 보임!)
#
#   sin/cos로 바꾸면:
#   - 23시: sin=-0.26, cos=0.97
#   - 0시: sin=0, cos=1
#   수치적으로 매우 가까워짐!
#
#   sin만 쓰면 안 되는 이유:
#   - 3시와 9시가 같은 sin값 → 구분 불가
#   sin+cos 쌍: 24시간 중 각 시각이 고유한 좌표 → 완벽히 구분
# ================================================
def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ================================================
# 슬라이딩 윈도우로 시퀀스 생성 함수
#
# [슬라이딩 윈도우란?]
#   데이터를 창문처럼 일정 크기로 잘라서 샘플 만들기
#
#   예시 (SEQ_LEN=60, PRED_LEN=12):
#   i=0 : X=[0~59행], y=[60~71행 발전량 평균]   → "과거 60분 보고 앞 12분 예측"
#   i=1 : X=[1~60행], y=[61~72행 발전량 평균]   → 1분 뒤 시나리오
#   i=2 : X=[2~61행], y=[62~73행 발전량 평균]   → 2분 뒤 시나리오
#   ...
#
# [구간 평균 예측을 쓰는 이유]
#   "정확히 12분 뒤 그 순간 발전량" 예측 → 노이즈가 많아서 정확도 낮음
#   "앞으로 12분 평균 발전량" 예측 → 노이즈 평균화, 에너지 계획에 더 실용적
# ================================================
def create_sequences(X, y, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    X_list, y_list = [], []
    for i in range(len(X) - seq_len - pred_len):
        X_list.append(X[i : i + seq_len])
        # 정답: seq_len 이후 pred_len 구간의 발전량 평균
        y_list.append(y[i + seq_len : i + seq_len + pred_len].mean())
    return (
        np.array(X_list, dtype=np.float32),          # (샘플수, 60, 6)
        np.array(y_list, dtype=np.float32).reshape(-1, 1),  # (샘플수, 1)
    )


# ================================================
# 학습 함수
#
# [MSELoss란?]
#   Mean Squared Error = 평균 제곱 오차
#   예측값과 실제값의 차이를 제곱해서 평균
#   연속값 예측의 표준 손실 함수
#
# [Early Stopping이란?]
#   검증 손실이 PATIENCE번 연속으로 개선되지 않으면 학습 중단
#   과적합(train은 잘 되는데 새 데이터에선 못함) 방지
#
# [best_state를 저장하는 이유]
#   학습 중 가장 좋은 순간의 모델을 저장해두고
#   최종적으로 그 가중치를 사용
# ================================================
def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()  # 손실 함수: 연속값 예측용
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam: 가장 많이 쓰는 최적화기
    best_val, best_state, wait = float("inf"), None, 0
    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 단계 ──
        model.train()  # 학습 모드 (Dropout 등 활성화)
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()  # 이전 기울기 초기화 (누적되면 안 됨)
            loss.backward()        # 역전파: 기울기 계산
            optimizer.step()       # 가중치 업데이트
            losses.append(loss.item())

        # ── 검증 단계 ──
        model.eval()  # 평가 모드 (Dropout 비활성화)
        val_losses = []
        with torch.no_grad():  # 기울기 계산 안 함 (메모리/속도 절약)
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

        t_loss = np.mean(losses)
        v_loss = np.mean(val_losses)
        train_hist.append(t_loss); val_hist.append(v_loss)

        if epoch % 10 == 0:
            print(f"   Epoch [{epoch:3d}/{EPOCHS}]  Train: {t_loss:.6f}  Val: {v_loss:.6f}")

        # ── Early Stopping 판단 ──
        if v_loss < best_val:
            best_val   = v_loss
            best_state = model.state_dict()     # 최적 가중치 백업
            torch.save(best_state, "models//gru.pth")  # 파일로 저장
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n   Early Stopping! (epoch {epoch})")
                break

    model.load_state_dict(best_state)  # 최적 가중치 복원
    return model, train_hist, val_hist


# ================================================
# 메인 함수
# ================================================
def main():
    # GPU 있으면 GPU, 없으면 CPU 사용 (노트북은 대부분 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ① 데이터 로드 및 전처리
    df = pd.read_csv("data/sensor_log.csv")
    df = df.dropna()           # 결측값(NaN) 있는 행 제거
    df = df[df["power"] >= 0]  # 음수 발전량 제거 (이상값)
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour  # timestamp에서 시간 추출
    df = add_time_features(df)  # hour → hour_sin, hour_cos 변환

    # light 컬럼 없으면 0으로 채움 (구버전 CSV 대비)
    if "light" not in df.columns:
        df["light"] = 0.0

    print(f"데이터: {len(df)}행")

    # ② 학습/테스트 분리 → 정규화
    # [왜 분리 먼저 하고 정규화?]
    #   테스트 데이터 정보가 스케일러에 "새어들어가면" 성능이 부풀려짐
    #   올바른 순서: 분리 → train으로만 스케일러 학습 → test는 변환만
    n        = len(df)
    train_df = df.iloc[:int(n * 0.8)]  # 앞 80% = 학습용
    test_df  = df.iloc[int(n * 0.8):]  # 뒤 20% = 테스트용

    # MinMaxScaler: 값을 0~1 사이로 정규화
    # GRU는 입력값 크기 차이가 크면 학습이 불안정해짐 → 정규화 필수
    f_scaler = MinMaxScaler()  # feature(입력) 스케일러
    t_scaler = MinMaxScaler()  # target(출력) 스케일러 (역변환용으로 별도 관리)

    train_X = f_scaler.fit_transform(train_df[FEATURES])  # 학습: 기준값 계산 + 변환
    test_X  = f_scaler.transform(test_df[FEATURES])       # 테스트: 변환만
    train_y = t_scaler.fit_transform(train_df[[TARGET]])
    test_y  = t_scaler.transform(test_df[[TARGET]])

    # 스케일러 저장 (app_gru.py에서 추론 시 동일한 기준으로 변환해야 함)
    joblib.dump(f_scaler, "models//scaler_gru.pkl")
    joblib.dump(t_scaler, "models//scaler_gru_target.pkl")
    print("스케일러 저장 완료")

    # ③ 슬라이딩 윈도우로 시퀀스 생성
    X_all,  y_all  = create_sequences(train_X, train_y)
    X_test, y_test = create_sequences(test_X,  test_y)

    # train 안에서 val 분리 (val로 Early Stopping 판단)
    val_size = int(len(X_all) * 0.2)
    X_tr,  y_tr  = X_all[:-val_size], y_all[:-val_size]
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]
    print(f"학습: {len(X_tr)} / 검증: {len(X_val)} / 테스트: {len(X_test)}")

    # ④ DataLoader 생성
    # shuffle=True: 매 에포크마다 순서 섞기 → 특정 패턴 과적합 방지
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH)

    # ⑤ 모델 생성 및 학습
    model = PowerPredictionGRU(input_size=len(FEATURES)).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")
    model, train_hist, val_hist = train(model, train_loader, val_loader, device)

    # ⑥ 테스트 데이터로 최종 성능 평가
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    # 역정규화: 0~1 값을 실제 W 단위로 복원
    pred_w = t_scaler.inverse_transform(pred)
    true_w = t_scaler.inverse_transform(y_test)

    mae  = mean_absolute_error(true_w, pred_w)          # 평균 절대 오차 (W)
    rmse = math.sqrt(mean_squared_error(true_w, pred_w))# 평균 제곱근 오차 (W)
    r2   = r2_score(true_w, pred_w)                     # 결정계수 (1에 가까울수록 좋음)
    print(f"\n MAE: {mae:.3f}W  RMSE: {rmse:.3f}W  R²: {r2:.3f}")

    # ⑦ 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 왼쪽: Loss 곡선 (train/val이 같이 내려가면 정상, val만 오르면 과적합)
    axes[0].plot(train_hist, label="Train"); axes[0].plot(val_hist, label="Val")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # 오른쪽: 실제 vs 예측 비교 (점선이 실선을 잘 따라가면 성공)
    n = min(200, len(pred_w))
    axes[1].plot(true_w[:n], label="실제", color="steelblue")
    axes[1].plot(pred_w[:n], label=f"예측 (MAE={mae:.2f}W)", color="tomato", linestyle="--")
    axes[1].set_title("실제 vs 예측 발전량"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models//gru_result.png", dpi=150)
    print("그래프 저장 → models/gru_result.png")
    print("✅ 완료! 다음: python train_water.py")


if __name__ == "__main__":
    main()