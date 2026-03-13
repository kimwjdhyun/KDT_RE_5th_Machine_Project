"""
XGBoost 기반 1시간 후 태양광 발전량 예측 코드
-------------------------------------------------
[프로젝트 목적]
- 현재 센서 데이터(태양광 전압, 전류, 조도, 온도, 습도, 배터리 정보, 시간 정보)를 이용해
  1시간 후 태양광 발전량을 예측한다.

[현재 데이터 구조]
- Arduino/Flask는 10초 간격으로 원본 데이터를 저장한다.
- 모델 학습 시에는 5분 단위 평균값으로 다운샘플링한다.
- 따라서 1시간 후 예측 target은 12 step 뒤 값이다.
  (5분 × 12 = 60분)
"""

# =========================
# 0. 필요한 라이브러리
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# 1. 데이터 불러오기
# =========================
CSV_PATH = "../backend/data/sensor_log.csv"
df = pd.read_csv(CSV_PATH)
print("===== CSV 기본 확인 =====")
print(df.head())
print()
print("===== 컬럼명 확인 =====")
print(df.columns.tolist())
print()
print("===== 원본 데이터 개수 =====")
print(len(df))
print()

if len(df) < 30:
    raise ValueError("원본 데이터가 너무 적습니다. 최소 30개 이상 수집 후 실행하세요.")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# =========================
# 2. 5분 단위 다운샘플링
#    10초 원본 데이터 -> 5분 평균 데이터
# =========================
df = df.set_index("timestamp")
df = df.resample("5min").mean(numeric_only=True)
df = df.dropna().reset_index()
print("===== 5분 다운샘플 후 확인 =====")
print(df.head())
print()
print("===== 다운샘플 후 데이터 개수 =====")
print(len(df))
print()

if len(df) < 30:
    raise ValueError("5분 단위 다운샘플링 후 데이터가 너무 적습니다. 더 많은 데이터를 수집하세요.")


# =========================
# 3. 발전량(solar_power) 계산/보정
# =========================
if "solar_power" not in df.columns:
    df["solar_power"] = df["solar_voltage"] * df["solar_current"]
else:
    df["solar_power"] = df["solar_power"].fillna(df["solar_voltage"] * df["solar_current"])


# =========================
# 4. 시간 파생변수
# =========================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# =========================
# 5. lag / rolling feature
#    이제 1 step = 5분
# =========================
df["power_lag_1"] = df["solar_power"].shift(1)   # 5분 전
df["power_lag_2"] = df["solar_power"].shift(2)   # 10분 전
df["power_lag_3"] = df["solar_power"].shift(3)   # 15분 전

df["recent_mean_power_3"] = df["solar_power"].rolling(window=3).mean()  # 최근 15분 평균
df["recent_mean_power_6"] = df["solar_power"].rolling(window=6).mean()  # 최근 30분 평균

df["power_trend_3"] = df["solar_power"] - df["recent_mean_power_3"]
df["power_diff_1"] = df["solar_power"] - df["power_lag_1"]
df["power_std_6"] = df["solar_power"].rolling(window=6).std()


# =========================
# 6. 예측 타깃 생성
#    1시간 후 = 12 step 뒤 (5분 × 12)
# =========================
PRED_STEP = 12
df["target_power_1h"] = df["solar_power"].shift(-PRED_STEP)


# =========================
# 7. 결측치 제거
# =========================
df = df.dropna().reset_index(drop=True)

if len(df) < 20:
    raise ValueError("전처리 후 남은 데이터가 너무 적습니다. 더 많은 데이터를 수집하세요.")


# =========================
# 8. 입력 변수 선택
# =========================
feature_cols = [
    "solar_voltage",
    "solar_current",
    "light",
    "temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "solar_power",
    "power_lag_1",
    "power_lag_2",
    "power_lag_3",
    "recent_mean_power_3",
    "recent_mean_power_6",
    "power_trend_3",
    "power_diff_1",
    "power_std_6"
]

optional_cols = [
    "battery_voltage",
    "battery_current",
    "battery_power",
    "soil",
    "soc"
]

for col in optional_cols:
    if col in df.columns:
        feature_cols.append(col)

X = df[feature_cols]
y = df["target_power_1h"]


# =========================
# 9. 학습 / 테스트 분리
#    시계열이므로 시간 순서 유지
# =========================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("학습/테스트 데이터가 부족합니다. 더 많은 데이터를 수집하세요.")


# =========================
# 10. 모델 생성 및 학습
# =========================
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 11. 예측
# =========================
pred = model.predict(X_test)


# =========================
# 12. 성능 평가
# =========================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("===== XGBoost Result =====")
print(f"원본 데이터 개수           : {len(pd.read_csv(CSV_PATH))}")
print(f"5분 다운샘플링 후 개수     : {len(df)}")
print(f"학습 데이터 개수           : {len(X_train)}")
print(f"테스트 데이터 개수         : {len(X_test)}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# =========================
# 13. 예측 결과 저장
# =========================
result_df = pd.DataFrame({
    "timestamp": df.iloc[split_idx:]["timestamp"].values,
    "actual_power_1h": y_test.values,
    "pred_power_1h": pred
})

result_df.to_csv("xgboost_prediction_result.csv", index=False)
print("예측 결과 저장 완료: xgboost_prediction_result.csv")


# =========================
# 14. 전체 테스트 구간 그래프
# =========================
plt.figure(figsize=(14, 6))

plt.plot(
    result_df["timestamp"],
    result_df["actual_power_1h"],
    label="Actual Power",
    linewidth=2
)

plt.plot(
    result_df["timestamp"],
    result_df["pred_power_1h"],
    label="Predicted Power",
    linewidth=2
)

plt.title("XGBoost: Actual vs Predicted Solar Power (1 Hour Ahead)")
plt.xlabel("Timestamp")
plt.ylabel("Solar Power (W)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("xgboost_actual_vs_pred.png", dpi=300)
plt.show()

print("그래프 저장 완료: xgboost_actual_vs_pred.png")


# =========================
# 15. 발표용 확대 그래프
# =========================
plot_df = result_df.tail(50)

plt.figure(figsize=(14, 6))

plt.plot(
    plot_df["timestamp"],
    plot_df["actual_power_1h"],
    label="Actual Power",
    linewidth=3
)

plt.plot(
    plot_df["timestamp"],
    plot_df["pred_power_1h"],
    label="Predicted Power",
    linewidth=3
)

plt.title("XGBoost Prediction (Last 50 Samples)")
plt.xlabel("Timestamp")
plt.ylabel("Solar Power (W)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("xgboost_actual_vs_pred_last50.png", dpi=300)
plt.show()

print("발표용 확대 그래프 저장 완료: xgboost_actual_vs_pred_last50.png")


# =========================
# 16. 변수 중요도 그래프
# =========================
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("xgboost_feature_importance.png", dpi=300)
plt.show()

print("변수 중요도 그래프 저장 완료: xgboost_feature_importance.png")