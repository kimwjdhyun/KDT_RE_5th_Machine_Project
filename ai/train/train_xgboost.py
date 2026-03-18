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

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── 경로 설정 ──────────────────────────────────
# train 폴더 어디서 실행해도 backend 기준으로 동작
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "backend", "results")
CSV_PATH = os.path.join(DATA_DIR, "sensor_log.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("CSV 경로:", CSV_PATH)
print("모델 저장:", MODELS_DIR)
print("결과 저장:", RESULT_DIR)


# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv(CSV_PATH)

print("\n===== CSV 기본 확인 =====")
print(df.head())
print("\n===== 컬럼명 확인 =====")
print(df.columns.tolist())
print(f"\n원본 데이터 개수: {len(df)}")

if len(df) < 30:
    raise ValueError("원본 데이터가 너무 적습니다. 최소 30개 이상 수집 후 실행하세요.")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df = df.sort_values("timestamp").reset_index(drop=True)


# =========================
# 2. 5분 단위 다운샘플링
# =========================
df = df.set_index("timestamp")
df = df.resample("5min").mean(numeric_only=True)
df = df.dropna().reset_index()

print(f"\n5분 다운샘플링 후: {len(df)}행")

if len(df) < 30:
    raise ValueError("5분 단위 다운샘플링 후 데이터가 너무 적습니다.")


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
# =========================
df["power_lag_1"] = df["solar_power"].shift(1)
df["power_lag_2"] = df["solar_power"].shift(2)
df["power_lag_3"] = df["solar_power"].shift(3)

df["recent_mean_power_3"] = df["solar_power"].rolling(window=3).mean()
df["recent_mean_power_6"] = df["solar_power"].rolling(window=6).mean()

df["power_trend_3"] = df["solar_power"] - df["recent_mean_power_3"]
df["power_diff_1"] = df["solar_power"] - df["power_lag_1"]
df["power_std_6"] = df["solar_power"].rolling(window=6).std()


# =========================
# 6. 예측 타깃 생성 (1시간 후)
# =========================
PRED_STEP = 12
df["target_power_1h"] = df["solar_power"].shift(-PRED_STEP)

df = df.dropna().reset_index(drop=True)
print(f"전처리 후: {len(df)}행")

if len(df) < 20:
    raise ValueError("전처리 후 남은 데이터가 너무 적습니다.")


# =========================
# 7. 입력 변수 선택
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
    "power_std_6",
]

optional_cols = [
    "battery_voltage",
    "battery_current",
    "battery_power",
    "soil",
    "soc",
]

for col in optional_cols:
    if col in df.columns:
        feature_cols.append(col)

X = df[feature_cols]
y = df["target_power_1h"]


# =========================
# 8. 학습 / 테스트 분리
# =========================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("학습/테스트 데이터가 부족합니다.")


# =========================
# 9. 모델 학습
# =========================
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# 모델 + 메타정보 저장
model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,
        "pred_step": PRED_STEP,
        "resample_rule": "5min"
    },
    model_path
)
print(f"\n모델 저장 완료 → {model_path}")


# =========================
# 10. 성능 평가
# =========================
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n===== XGBoost Result =====")
print(f"원본 데이터 개수       : {len(pd.read_csv(CSV_PATH))}")
print(f"5분 다운샘플링 후      : {len(df)}행")
print(f"학습 데이터 개수       : {len(X_train)}")
print(f"테스트 데이터 개수     : {len(X_test)}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# =========================
# 11. 예측 결과 저장
# =========================
result_df = pd.DataFrame({
    "timestamp": df.iloc[split_idx:]["timestamp"].values,
    "actual_power_1h": y_test.values,
    "pred_power_1h": pred
})

result_path = os.path.join(RESULT_DIR, "xgboost_prediction_result.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
print(f"\n예측 결과 저장 → {result_path}")


# =========================
# 12. 전체 테스트 구간 그래프
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

all_plot_path = os.path.join(RESULT_DIR, "xgboost_actual_vs_pred.png")
plt.savefig(all_plot_path, dpi=300)
plt.show()
print(f"그래프 저장 → {all_plot_path}")


# =========================
# 13. 발표용 확대 그래프 (마지막 50개)
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

last50_plot_path = os.path.join(RESULT_DIR, "xgboost_actual_vs_pred_last50.png")
plt.savefig(last50_plot_path, dpi=300)
plt.show()
print(f"발표용 그래프 저장 → {last50_plot_path}")


# =========================
# 14. Feature 중요도 그래프
# =========================
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

importance_path = os.path.join(RESULT_DIR, "xgboost_feature_importance.png")
plt.savefig(importance_path, dpi=300)
plt.show()
print(f"Feature 중요도 저장 → {importance_path}")

print("\n✅ XGBoost 완료!")