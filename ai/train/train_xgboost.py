# ================================================
# train_xgboost.py
# XGBoost 발전량 예측 모델 학습
#
# [GRU와의 차이점]
#   GRU      : 과거 30스텝 시계열 흐름을 기억해서 예측
#              "지난 2시간 패턴을 보고 다음 1시간을 예측"
#   XGBoost  : 현재 시점 feature 값만 보고 예측
#              "지금 이 순간의 값들로만 예측"
#
#   → 발전량 예측은 시계열이라 이론상 GRU가 유리
#   → 실제로 비교해서 발표할 때 근거로 사용
#
# [비교 지표]
#   MAE, RMSE, R² 를 GRU와 동일한 기준으로 측정
#
# 실행하면 → models/xgboost.pkl, models/xgboost_result.png 생성
# ================================================

import os
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("models", exist_ok=True)

np.random.seed(42)

# ── 설정값 ──
PRED_LEN = 12   # 예측 구간: 12스텝 = 1시간 (GRU와 동일 기준)
TARGET   = "power"

# ── 입력 변수 ──
# GRU와 다른 점:
#   GRU는 hour_sin/cos로 시간 인코딩 → 시계열 순환성 학습
#   XGBoost는 hour 숫자 그대로 써도 됨 → 트리 기반이라 순서 상관없음
FEATURES = ["power", "wind_power", "voltage", "light", "temp", "hour"]


# ================================================
# 데이터 로드 & 전처리
# ================================================
print("=" * 45)
print("① 데이터 로드")
print("=" * 45)

df = pd.read_csv("farm_data.csv")
df = df.dropna()
df = df[df["power"] >= 0]
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

if "wind_power" not in df.columns:
    print("   ⚠️  wind_power 없음 → 0으로 채움")
    df["wind_power"] = 0.0

print(f"   데이터: {len(df)}행")


# ================================================
# 예측 목표 생성
#
# GRU와 동일한 기준으로 비교하기 위해
# "다음 12스텝(1시간)의 평균 발전량"을 예측 목표로 설정.
#
# rolling(PRED_LEN).mean().shift(-PRED_LEN):
#   각 시점에서 앞으로 12스텝의 평균을 미리 계산해서
#   현재 행의 정답 라벨로 붙이는 것.
# ================================================
df["target"] = df[TARGET].rolling(PRED_LEN).mean().shift(-PRED_LEN)
df = df.dropna(subset=["target"])   # 앞뒤 NaN 제거


# ================================================
# 분리 → 정규화
#
# GRU와 동일한 80/20 분리 방식 사용 (공정한 비교를 위해)
# ================================================
print("\n② 분리 → 정규화")

n        = len(df)
train_df = df.iloc[:int(n * 0.8)]
test_df  = df.iloc[int(n * 0.8):]

# feature 스케일러
f_scaler = MinMaxScaler()
train_X  = f_scaler.fit_transform(train_df[FEATURES])
test_X   = f_scaler.transform(test_df[FEATURES])

# target 스케일러
t_scaler = MinMaxScaler()
train_y  = t_scaler.fit_transform(train_df[["target"]]).ravel()
test_y   = t_scaler.transform(test_df[["target"]]).ravel()

joblib.dump(f_scaler, "models/scaler_xgb.pkl")
joblib.dump(t_scaler, "models/scaler_xgb_target.pkl")
print(f"   학습: {len(train_X)}개  |  테스트: {len(test_X)}개")


# ================================================
# XGBoost 학습
#
# 주요 파라미터:
#   n_estimators  : 트리 개수 (많을수록 정확하지만 느림)
#   max_depth     : 트리 깊이 (깊을수록 복잡한 패턴 학습, 과적합 위험)
#   learning_rate : 학습률 (낮을수록 안정적, 더 많은 트리 필요)
#   subsample     : 각 트리 학습 시 사용할 데이터 비율 (과적합 방지)
# ================================================
print("\n③ XGBoost 학습")

model = XGBRegressor(
    n_estimators  = 300,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    random_state   = 42,
    verbosity      = 0,    # 학습 로그 숨김
)
model.fit(train_X, train_y)
joblib.dump(model, "models/xgboost.pkl")
print("   모델 저장 → models/xgboost.pkl")


# ================================================
# 평가
# ================================================
print("\n④ 평가")

pred_scaled = model.predict(test_X).reshape(-1, 1)
pred_w = t_scaler.inverse_transform(pred_scaled).ravel()
true_w = t_scaler.inverse_transform(test_y.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(true_w, pred_w)
rmse = math.sqrt(mean_squared_error(true_w, pred_w))
r2   = r2_score(true_w, pred_w)

print(f"   MAE : {mae:.3f} W")
print(f"   RMSE: {rmse:.3f} W")
print(f"   R²  : {r2:.3f}")


# ================================================
# 시각화
# ================================================
print("\n⑤ 시각화")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 왼쪽: feature 중요도
# XGBoost는 어떤 feature가 예측에 얼마나 기여했는지 볼 수 있음
# → GRU는 이런 해석이 어렵기 때문에 XGBoost의 장점 중 하나
importances = model.feature_importances_
axes[0].barh(FEATURES, importances, color="steelblue")
axes[0].set_title("Feature 중요도")
axes[0].set_xlabel("중요도")
axes[0].grid(alpha=0.3)

# 오른쪽: 실제 vs 예측
n_plot = min(200, len(pred_w))
axes[1].plot(true_w[:n_plot], label="실제", color="steelblue")
axes[1].plot(pred_w[:n_plot], label=f"XGBoost 예측 (MAE={mae:.2f}W)",
             color="tomato", linestyle="--")
axes[1].set_title("실제 vs 예측 발전량")
axes[1].set_xlabel("시점")
axes[1].set_ylabel("발전량 (W)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("models/xgboost_result.png", dpi=150)
print("   그래프 저장 → models/xgboost_result.png")

# ================================================
# GRU vs XGBoost 비교 요약 출력
# (GRU 결과가 있으면 같이 출력)
# ================================================
print("\n" + "=" * 45)
print("GRU vs XGBoost 비교")
print("=" * 45)
print(f"   XGBoost  MAE: {mae:.3f}W  RMSE: {rmse:.3f}W  R²: {r2:.3f}")

# GRU 결과가 저장돼 있으면 같이 출력
if os.path.exists("models/gru_predictions.csv"):
    gru_df  = pd.read_csv("models/gru_predictions.csv")
    gru_mae = mean_absolute_error(
        gru_df["actual_next_1h_mean_power"],
        gru_df["pred_next_1h_mean_power"]
    )
    gru_r2 = r2_score(
        gru_df["actual_next_1h_mean_power"],
        gru_df["pred_next_1h_mean_power"]
    )
    print(f"   GRU      MAE: {gru_mae:.3f}W  R²: {gru_r2:.3f}")
    print(f"\n   → {'GRU' if gru_mae < mae else 'XGBoost'} 가 더 정확!")
else:
    print("   (GRU 결과 없음 → train_gru.py 먼저 실행하면 비교 가능)")

print("\n✅ XGBoost 학습 완료!")