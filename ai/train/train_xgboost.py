# ================================================
# train_xgboost.py
# XGBoost 발전량 예측 모델 (GRU 비교용)
#
#   - 5분 단위 다운샘플링
#   - lag / rolling feature 추가
#   - 발표용 그래프 3개 (전체, 마지막 50개, feature 중요도)
#
# [farm_data.csv 컬럼에 맞게 변경]
#   solar_voltage → voltage
#   solar_current → current
#   solar_power   → power
#   temperature   → temp
#   battery_*     → soc (INA219로 측정)
#
# [실행 후 생성되는 파일]
#   models/xgboost.pkl
#   models/scaler_xgb.pkl
#   models/scaler_xgb_target.pkl
#   models/xgboost_result.png
#   models/xgboost_result_last50.png
#   models/xgboost_feature_importance.png
# ================================================

import os, math
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

TARGET   = "power"
PRED_STEP = 12  # 5분 × 12 = 1시간 후 예측


# =========================
# 1. 데이터 로드
# =========================
print("① 데이터 로드")
df = pd.read_csv("farm_data.csv")
print(f"   원본 데이터: {len(df)}행")
print(f"   컬럼: {df.columns.tolist()}")

if len(df) < 30:
    raise ValueError("데이터가 너무 적습니다. 최소 30개 이상 필요합니다.")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.dropna()
df = df[df["power"] >= 0]


# =========================
# 2. 5분 단위 다운샘플링
#    1분 원본 데이터 → 5분 평균 데이터
# =========================
print("\n② 5분 단위 다운샘플링")
df = df.set_index("timestamp")
df = df.resample("5min").mean(numeric_only=True)
df = df.dropna().reset_index()
print(f"   다운샘플링 후: {len(df)}행")

if len(df) < 30:
    raise ValueError("5분 다운샘플링 후 데이터가 너무 적습니다.")


# =========================
# 3. 시간 파생변수
# =========================
df["hour"]     = df["timestamp"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# =========================
# 4. lag / rolling feature
#    1 step = 5분
# =========================
df["power_lag_1"] = df["power"].shift(1)   # 5분 전
df["power_lag_2"] = df["power"].shift(2)   # 10분 전
df["power_lag_3"] = df["power"].shift(3)   # 15분 전

df["recent_mean_power_3"] = df["power"].rolling(window=3).mean()  # 최근 15분 평균
df["recent_mean_power_6"] = df["power"].rolling(window=6).mean()  # 최근 30분 평균

df["power_trend_3"] = df["power"] - df["recent_mean_power_3"]  # 현재 vs 15분 평균 차이
df["power_diff_1"]  = df["power"] - df["power_lag_1"]          # 현재 vs 5분 전 차이
df["power_std_6"]   = df["power"].rolling(window=6).std()       # 최근 30분 변동성


# =========================
# 5. 예측 타깃 생성
#    1시간 후 = 12 step 뒤 (5분 × 12)
# =========================
df["target"] = df["power"].shift(-PRED_STEP)
df = df.dropna().reset_index(drop=True)
print(f"   전처리 후: {len(df)}행")

if len(df) < 20:
    raise ValueError("전처리 후 데이터가 너무 적습니다.")


# =========================
# 6. 입력 변수 선택
# =========================
# 기본 변수
feature_cols = [
    "voltage",            # 태양광 전압
    "power",              # 태양광 발전량
    "light",              # 조도
    "temp",               # 온도
    "hum",                # 습도
    "hour_sin",           # 시간 sin 인코딩
    "hour_cos",           # 시간 cos 인코딩
    "power_lag_1",        # 5분 전 발전량
    "power_lag_2",        # 10분 전 발전량
    "power_lag_3",        # 15분 전 발전량
    "recent_mean_power_3",# 최근 15분 평균
    "recent_mean_power_6",# 최근 30분 평균
    "power_trend_3",      # 현재 vs 15분 평균 추세
    "power_diff_1",       # 5분 간 변화량
    "power_std_6",        # 최근 30분 변동성
]

# 있으면 추가 (없어도 에러 안 남)
optional_cols = ["current", "soil", "soc"]
for col in optional_cols:
    if col in df.columns:
        feature_cols.append(col)

print(f"\n   입력 변수 {len(feature_cols)}개: {feature_cols}")


# =========================
# 7. 학습 / 테스트 분리 → 정규화
# =========================
print("\n③ 분리 → 정규화")
n        = len(df)
train_df = df.iloc[:int(n * 0.8)]
test_df  = df.iloc[int(n * 0.8):]

f_scaler = MinMaxScaler()
t_scaler = MinMaxScaler()
train_X  = f_scaler.fit_transform(train_df[feature_cols])
test_X   = f_scaler.transform(test_df[feature_cols])
train_y  = t_scaler.fit_transform(train_df[["target"]]).ravel()
test_y   = t_scaler.transform(test_df[["target"]]).ravel()

joblib.dump(f_scaler, "models/scaler_xgb.pkl")
joblib.dump(t_scaler, "models/scaler_xgb_target.pkl")
print(f"   학습: {len(train_X)}개  |  테스트: {len(test_X)}개")


# =========================
# 8. XGBoost 학습
# =========================
print("\n④ XGBoost 학습")
model = XGBRegressor(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9,
    random_state=42, verbosity=0,
)
model.fit(train_X, train_y)
joblib.dump(model, "models/xgboost.pkl")
print("   모델 저장 → models/xgboost.pkl")


# =========================
# 9. 예측 및 평가
# =========================
print("\n⑤ 평가")
pred_scaled = model.predict(test_X)
pred_w = t_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
true_w = t_scaler.inverse_transform(test_y.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(true_w, pred_w)
rmse = math.sqrt(mean_squared_error(true_w, pred_w))
r2   = r2_score(true_w, pred_w)

print(f"\n===== XGBoost Result =====")
print(f"원본 데이터 개수       : {len(pd.read_csv('farm_data.csv'))}")
print(f"5분 다운샘플링 후      : {len(df)}행")
print(f"학습 데이터 개수       : {len(train_X)}")
print(f"테스트 데이터 개수     : {len(test_X)}")
print(f"MAE  : {mae:.4f}W")
print(f"RMSE : {rmse:.4f}W")
print(f"R²   : {r2:.4f}")


# =========================
# 10. 예측 결과 저장
# =========================
result_df = pd.DataFrame({
    "timestamp":      test_df["timestamp"].values,
    "actual_power":   true_w,
    "pred_power":     pred_w,
})
result_df.to_csv("models/xgboost_prediction_result.csv", index=False)
print("\n예측 결과 저장 → models/xgboost_prediction_result.csv")


# =========================
# 11. 그래프 ① 전체 테스트 구간
# =========================
print("\n⑥ 시각화")
plt.figure(figsize=(14, 6))
plt.plot(result_df["timestamp"], result_df["actual_power"],
         label="실제 발전량", linewidth=2, color="steelblue")
plt.plot(result_df["timestamp"], result_df["pred_power"],
         label=f"XGBoost 예측 (MAE={mae:.2f}W)", linewidth=2,
         color="tomato", linestyle="--")
plt.title("XGBoost: 실제 vs 예측 발전량 (1시간 후)")
plt.xlabel("시간"); plt.ylabel("발전량 (W)")
plt.legend(); plt.xticks(rotation=45); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("models/xgboost_result.png", dpi=150)
print("   그래프 저장 → models/xgboost_result.png")


# =========================
# 12. 그래프 ② 발표용 마지막 50개 확대
# =========================
plot_df = result_df.tail(50)
plt.figure(figsize=(14, 6))
plt.plot(plot_df["timestamp"], plot_df["actual_power"],
         label="실제 발전량", linewidth=3, color="steelblue")
plt.plot(plot_df["timestamp"], plot_df["pred_power"],
         label=f"XGBoost 예측 (MAE={mae:.2f}W)", linewidth=3,
         color="tomato", linestyle="--")
plt.title("XGBoost 예측 (마지막 50개 - 발표용)")
plt.xlabel("시간"); plt.ylabel("발전량 (W)")
plt.legend(); plt.xticks(rotation=45); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("models/xgboost_result_last50.png", dpi=150)
print("   발표용 그래프 저장 → models/xgboost_result_last50.png")


# =========================
# 13. 그래프 ③ Feature 중요도
# =========================
importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
plt.title("XGBoost Feature 중요도")
plt.xlabel("Importance"); plt.gca().invert_yaxis()
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("models/xgboost_feature_importance.png", dpi=150)
print("   Feature 중요도 저장 → models/xgboost_feature_importance.png")


# =========================
# 14. GRU vs XGBoost 비교
# =========================
print("\n" + "=" * 45)
print("GRU vs XGBoost 비교")
print("=" * 45)
print(f"   XGBoost  MAE: {mae:.3f}W  RMSE: {rmse:.3f}W  R²: {r2:.3f}")
if os.path.exists("models/gru_predictions.csv"):
    gru_df  = pd.read_csv("models/gru_predictions.csv")
    gru_mae = mean_absolute_error(gru_df["actual"], gru_df["pred"])
    gru_r2  = r2_score(gru_df["actual"], gru_df["pred"])
    print(f"   GRU      MAE: {gru_mae:.3f}W  R²: {gru_r2:.3f}")
    print(f"\n   → {'GRU' if gru_mae < mae else 'XGBoost'} 가 더 정확!")
else:
    print("   (GRU 결과 없음 → train_gru.py 먼저 실행)")

print("\n✅ XGBoost 완료!")