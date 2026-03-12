"""
XGBoost 기반 1시간 후 태양광 발전량 예측 코드
-------------------------------------------------
[프로젝트 목적]
- 현재 센서 데이터(전압, 전류, 조도, 온도, 습도, 시간 정보)를 이용해
  1시간 후 태양광 발전량을 예측한다.

[가정]
- 데이터는 5분 간격으로 수집되었다.
- 따라서 1시간 후 값은 현재 시점 기준 12 step 뒤 데이터이다.
  (5분 × 12 = 60분)

[입력 데이터 예시 컬럼]
- timestamp        : 측정 시각
- solar_voltage    : 태양광 패널 전압(V)
- solar_current    : 태양광 패널 전류(A)
- light            : 조도
- temperature      : 온도
- humidity         : 습도
- battery_voltage  : 배터리 전압(선택 컬럼, 있으면 함께 사용)

[출력]
- xgboost_prediction_result.csv
- xgboost_actual_vs_pred.png
- xgboost_actual_vs_pred_last50.png
- xgboost_feature_importance.png
"""

# =========================
# 0. 필요한 라이브러리 불러오기
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# 1. 데이터 불러오기
# =========================
# 학습에 사용할 CSV 파일을 읽는다.
df = pd.read_csv("solar_data.csv")

# timestamp를 datetime 형식으로 변환한 뒤 시간 순으로 정렬한다.
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# =========================
# 2. 발전량(solar_power) 계산
# =========================
# 태양광 발전량 = 전압 × 전류
df["solar_power"] = df["solar_voltage"] * df["solar_current"]


# =========================
# 3. 시간 관련 파생변수 생성
# =========================
# 태양광 발전은 시간대 영향을 많이 받으므로 timestamp에서 시간 정보를 추출한다.
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

# 시간의 주기성(23시 다음이 0시)을 반영하기 위해 sin/cos 변환 사용
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# =========================
# 4. 과거 발전량 정보(lag, rolling) 생성
# =========================
# XGBoost는 표 데이터 기반 모델이므로,
# 최근 발전량 흐름을 반영할 수 있도록 과거 값/평균 값을 feature로 추가한다.
df["power_lag_1"] = df["solar_power"].shift(1)  # 직전 시점 발전량
df["power_lag_2"] = df["solar_power"].shift(2)  # 2 step 전 발전량
df["power_lag_3"] = df["solar_power"].shift(3)  # 3 step 전 발전량

df["recent_mean_power_3"] = df["solar_power"].rolling(window=3).mean()  # 최근 15분 평균
df["recent_mean_power_6"] = df["solar_power"].rolling(window=6).mean()  # 최근 30분 평균


# =========================
# 4-1. 성능 향상을 위한 추가 feature
# =========================
# 최근 평균과 비교했을 때 현재 발전량이 얼마나 높은지/낮은지
df["power_trend_3"] = df["solar_power"] - df["recent_mean_power_3"]

# 직전 시점 대비 발전량 변화량
df["power_diff_1"] = df["solar_power"] - df["power_lag_1"]

# 최근 6개 시점(30분) 발전량 변동성
df["power_std_6"] = df["solar_power"].rolling(window=6).std()


# =========================
# 5. 예측 타깃(target) 생성
# =========================
# 목표: 1시간 후 발전량 예측
# 수집 간격이 5분이므로, 1시간 후는 12 step 뒤 값이다.
PRED_STEP = 12
df["target_power_1h"] = df["solar_power"].shift(-PRED_STEP)


# =========================
# 6. 결측치 제거
# =========================
# lag, rolling, 미래 타깃 생성 과정에서 생기는 NaN 제거
df = df.dropna().reset_index(drop=True)


# =========================
# 7. 모델 입력 변수(feature) 선택
# =========================
feature_cols = [
    "solar_voltage",       # 현재 태양광 전압
    "solar_current",       # 현재 태양광 전류
    "light",               # 현재 조도
    "temperature",         # 현재 온도
    "humidity",            # 현재 습도
    "hour_sin",            # 시간 주기성 반영(sin)
    "hour_cos",            # 시간 주기성 반영(cos)
    "solar_power",         # 현재 발전량
    "power_lag_1",         # 직전 발전량
    "power_lag_2",         # 2 step 전 발전량
    "power_lag_3",         # 3 step 전 발전량
    "recent_mean_power_3", # 최근 15분 평균 발전량
    "recent_mean_power_6", # 최근 30분 평균 발전량
    "power_trend_3",       # 최근 평균 대비 현재 발전량 차이
    "power_diff_1",        # 직전 대비 변화량
    "power_std_6"          # 최근 30분 변동성
]

# battery_voltage 컬럼이 있으면 추가 입력 변수로 사용
if "battery_voltage" in df.columns:
    feature_cols.append("battery_voltage")

X = df[feature_cols]
y = df["target_power_1h"]


# =========================
# 8. 학습 / 테스트 데이터 분리
# =========================
# 시계열 데이터이므로 랜덤 셔플 대신 시간 순서대로 앞 80% / 뒤 20% 분리
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# =========================
# 9. XGBoost 모델 생성 및 학습
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
# 10. 테스트 데이터 예측
# =========================
pred = model.predict(X_test)


# =========================
# 11. 모델 성능 평가
# =========================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("===== XGBoost Result =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# =========================
# 12. 실제값 / 예측값 저장
# =========================
result_df = pd.DataFrame({
    "timestamp": df.iloc[split_idx:]["timestamp"].values,
    "actual_power_1h": y_test.values,
    "pred_power_1h": pred
})

result_df.to_csv("xgboost_prediction_result.csv", index=False)
print("예측 결과 저장 완료: xgboost_prediction_result.csv")


# =========================
# 13. 전체 테스트 구간 실제값 vs 예측값 그래프
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
# 14. 발표용 확대 그래프 (마지막 50개 데이터)
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
# 15. 변수 중요도(Feature Importance) 그래프
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