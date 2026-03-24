# ================================================
# train_xgboost.py
# XGBoost 기반 1시간 후 태양광 발전량 예측 모델 학습
#
# [이 파일이 하는 일]
# 1. sensor_log.csv를 읽는다.
# 2. 10초 간격 원본 데이터를 5분 평균으로 다운샘플링한다.
# 3. 시간/추세/변화량 feature를 만든다.
# 4. 여러 feature set과 파라미터 조합을 비교한다.
# 5. 가장 좋은 조합으로 최종 모델을 다시 학습한다.
# 6. 모델을 xgboost_model.pkl 로 저장한다.
#
# [핵심 예측 목표]
# - 현재 시점 기준 feature를 사용해
# - 1시간 뒤의 solar_power를 예측한다.
# - 5분 단위 데이터이므로 12 step 뒤 값을 target으로 사용한다.
# ================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------
# 경로 설정
# ------------------------------------------------
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

# ------------------------------------------------
# 실험 설정
# ------------------------------------------------
RESAMPLE_RULE = "5min"
PRED_STEP = 12
DAY_POWER_THRESHOLD = 0.05


# ------------------------------------------------
# 유틸 함수
# ------------------------------------------------
def ensure_column(df, col_name, default_value=0):
    if col_name not in df.columns:
        df[col_name] = default_value
    return df


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def unique_keep_order(seq):
    seen = set()
    result = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def calc_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metric_block(title, metrics_dict):
    print(f"\n===== {title} =====")
    for k, v in metrics_dict.items():
        if pd.isna(v):
            print(f"{k:<6}: NaN")
        else:
            print(f"{k:<6}: {v:.4f}")


# ------------------------------------------------
# 데이터 로드
# ------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip", encoding="utf-8")
except TypeError:
    df = pd.read_csv(CSV_PATH, error_bad_lines=False, encoding="utf-8")

print("\n===== 원본 데이터 확인 =====")
print(df.head())
print("\n컬럼:", df.columns.tolist())
print(f"원본 행 수: {len(df)}")

required_cols = [
    "timestamp",
    "temperature",
    "humidity",
    "soil",
    "light",
    "solar_voltage",
    "solar_current",
    "solar_power",
    "battery_voltage",
    "battery_current",
    "battery_power",
    "pump",
    "led",
    "soc",
    "mode",
    "water_alert",
]

optional_cols = [
    "soil_raw",
    "led_brightness",
    "pred_1h",
]

for col in required_cols:
    df = ensure_column(df, col, 0)

for col in optional_cols:
    df = ensure_column(df, col, 0)

numeric_cols = [
    "temperature",
    "humidity",
    "soil",
    "soil_raw",
    "light",
    "solar_voltage",
    "solar_current",
    "solar_power",
    "battery_voltage",
    "battery_current",
    "battery_power",
    "pump",
    "led",
    "led_brightness",
    "soc",
    "mode",
    "water_alert",
    "pred_1h",
]

df = safe_numeric(df, numeric_cols)

# timestamp 정리
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).copy()
df = df.sort_values("timestamp").reset_index(drop=True)

# solar_power 비어 있으면 전압 * 전류로 보정
df["solar_power"] = pd.to_numeric(df["solar_power"], errors="coerce")
df["solar_power"] = df["solar_power"].fillna(df["solar_voltage"] * df["solar_current"])
df["solar_power"] = df["solar_power"].clip(lower=0)

# 나머지 결측 보정
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

df = df.dropna(subset=["solar_power"]).copy()

# ------------------------------------------------
# 5분 다운샘플링
# ------------------------------------------------
df = df.set_index("timestamp")
df = df.resample(RESAMPLE_RULE).mean(numeric_only=True)
df = df.dropna(how="all").reset_index()

print(f"\n5분 다운샘플링 후 행 수: {len(df)}")

if len(df) < 80:
    raise ValueError("다운샘플링 후 데이터가 너무 적습니다.")


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["dayofweek"] = df["timestamp"].dt.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

# lag feature
for lag in [1, 2, 3, 6, 12]:
    df[f"power_lag_{lag}"] = df["solar_power"].shift(lag)
    df[f"light_lag_{lag}"] = df["light"].shift(lag)

# rolling feature
for window in [3, 6, 12, 24]:
    df[f"power_mean_{window}"] = df["solar_power"].rolling(window).mean()
    df[f"power_std_{window}"] = df["solar_power"].rolling(window).std()
    df[f"power_max_{window}"] = df["solar_power"].rolling(window).max()
    df[f"power_min_{window}"] = df["solar_power"].rolling(window).min()

for window in [3, 6, 12]:
    df[f"light_mean_{window}"] = df["light"].rolling(window).mean()

# 변화량 feature
df["power_diff_1"] = df["solar_power"].diff(1)
df["power_diff_3"] = df["solar_power"].diff(3)
df["power_diff_6"] = df["solar_power"].diff(6)

df["light_diff_1"] = df["light"].diff(1)
df["light_diff_3"] = df["light"].diff(3)

# 기타 파생 feature
df["voltage_current_mul"] = df["solar_voltage"] * df["solar_current"]
df["battery_eff_gap"] = df["battery_voltage"] - df["solar_voltage"]
df["is_day_by_hour"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
df["is_active_now"] = (df["solar_power"] > DAY_POWER_THRESHOLD).astype(int)

# 1시간 뒤 target 생성
df["target_power_1h"] = df["solar_power"].shift(-PRED_STEP)

df = df.dropna().reset_index(drop=True)
print(f"전처리 후 최종 행 수: {len(df)}")

if len(df) < 100:
    raise ValueError("전처리 후 데이터가 너무 적습니다.")


# ------------------------------------------------
# Feature Set 후보
# ------------------------------------------------
base_features = [
    "solar_voltage", "solar_current", "light",
    "temperature", "humidity", "soc",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "solar_power"
]

trend_features = [
    "power_lag_1", "power_lag_2", "power_lag_3", "power_lag_6", "power_lag_12",
    "power_mean_3", "power_mean_6", "power_mean_12", "power_mean_24",
    "power_std_3", "power_std_6", "power_std_12",
    "power_max_12", "power_min_12",
    "power_diff_1", "power_diff_3", "power_diff_6"
]

light_features = [
    "light_lag_1", "light_lag_2", "light_lag_3", "light_lag_6", "light_lag_12",
    "light_mean_3", "light_mean_6", "light_mean_12",
    "light_diff_1", "light_diff_3"
]

battery_features = [
    "battery_voltage", "battery_current", "battery_power",
    "battery_eff_gap"
]

aux_features = [
    "soil", "soil_raw", "led_brightness", "pump", "led",
    "mode", "water_alert",
    "voltage_current_mul",
    "is_day_by_hour", "is_active_now"
]

feature_sets = {
    "base_only": base_features,
    "base_trend": base_features + trend_features,
    "base_trend_light": base_features + trend_features + light_features,
    "base_trend_light_battery": base_features + trend_features + light_features + battery_features,
    "full": base_features + trend_features + light_features + battery_features + aux_features,
}

for name in feature_sets:
    feature_sets[name] = unique_keep_order([c for c in feature_sets[name] if c in df.columns])

print("\n===== Feature Set 후보 =====")
for name, cols in feature_sets.items():
    print(f"{name}: {len(cols)}개")


# ------------------------------------------------
# train / val / test 분리
# ------------------------------------------------
n_total = len(df)
train_end = int(n_total * 0.7)
val_end = int(n_total * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\n===== 데이터 분할 =====")
print(f"Train: {len(train_df)}")
print(f"Val  : {len(val_df)}")
print(f"Test : {len(test_df)}")

param_grid = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
]

# ------------------------------------------------
# 실험 비교
# ------------------------------------------------
experiment_rows = []
best_score = float("inf")
best_bundle = None

for fs_name, feature_cols in feature_sets.items():
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_power_1h"].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df["target_power_1h"].copy()

    for i, params in enumerate(param_grid, start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **params
        )
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_metrics = calc_metrics(y_val, val_pred)

        val_day_mask = y_val > DAY_POWER_THRESHOLD
        val_day_metrics = calc_metrics(y_val[val_day_mask], val_pred[val_day_mask])

        # 발전구간 MAE 우선 선택
        score = val_day_metrics["MAE"] if not pd.isna(val_day_metrics["MAE"]) else val_metrics["MAE"]

        row = {
            "feature_set": fs_name,
            "param_idx": i,
            "feature_count": len(feature_cols),
            "val_mae": val_metrics["MAE"],
            "val_rmse": val_metrics["RMSE"],
            "val_r2": val_metrics["R2"],
            "val_day_mae": val_day_metrics["MAE"],
            "val_day_rmse": val_day_metrics["RMSE"],
            "val_day_r2": val_day_metrics["R2"],
            "params": json.dumps(params, ensure_ascii=False),
        }
        experiment_rows.append(row)

        print(f"\n[실험] feature_set={fs_name}, param_idx={i}")
        print(f"전체 Val MAE={val_metrics['MAE']:.4f}, RMSE={val_metrics['RMSE']:.4f}, R2={val_metrics['R2']:.4f}")
        if not pd.isna(val_day_metrics["MAE"]):
            print(f"주간 Val MAE={val_day_metrics['MAE']:.4f}, RMSE={val_day_metrics['RMSE']:.4f}, R2={val_day_metrics['R2']:.4f}")

        if score < best_score:
            best_score = score
            best_bundle = {
                "feature_set_name": fs_name,
                "feature_cols": feature_cols,
                "params": params,
            }

if best_bundle is None:
    raise RuntimeError("최적 모델 선택 실패")

experiment_df = pd.DataFrame(experiment_rows).sort_values(["val_day_mae", "val_mae"], na_position="last")
experiment_path = os.path.join(RESULT_DIR, "xgboost_experiment_summary.csv")
experiment_df.to_csv(experiment_path, index=False, encoding="utf-8-sig")

print("\n===== 최고 조합 선택 =====")
print("Feature Set :", best_bundle["feature_set_name"])
print("Feature 수  :", len(best_bundle["feature_cols"]))
print("Params      :", best_bundle["params"])


# ------------------------------------------------
# 최종 모델 학습
# ------------------------------------------------
best_features = best_bundle["feature_cols"]
best_params = best_bundle["params"]

trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

X_trainval = trainval_df[best_features].copy()
y_trainval = trainval_df["target_power_1h"].copy()

X_test = test_df[best_features].copy()
y_test = test_df["target_power_1h"].copy()

best_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    **best_params
)
best_model.fit(X_trainval, y_trainval)

pred_test = best_model.predict(X_test)

overall_metrics = calc_metrics(y_test, pred_test)

day_mask = y_test > DAY_POWER_THRESHOLD
day_metrics = calc_metrics(y_test[day_mask], pred_test[day_mask])

hour_day_mask = (test_df["hour"] >= 6) & (test_df["hour"] <= 18)
hour_day_metrics = calc_metrics(y_test[hour_day_mask], pred_test[hour_day_mask])

print_metric_block("최종 전체 성능", overall_metrics)
print_metric_block("최종 발전구간 성능(actual > 0.05)", day_metrics)
print_metric_block("최종 시간기준 주간 성능(06~18시)", hour_day_metrics)


# ------------------------------------------------
# 모델 저장
# - 기존 파일명 유지
# ------------------------------------------------
model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
joblib.dump(
    {
        "model": best_model,
        "feature_cols": best_features,
        "pred_step": PRED_STEP,
        "resample_rule": RESAMPLE_RULE,
        "day_power_threshold": DAY_POWER_THRESHOLD,
        "best_params": best_params,
        "feature_set_name": best_bundle["feature_set_name"],
    },
    model_path
)
print(f"\n모델 저장 완료 → {model_path}")


# ------------------------------------------------
# 결과 저장
# ------------------------------------------------
result_df = pd.DataFrame({
    "timestamp": test_df["timestamp"].values,
    "actual_power_1h": y_test.values,
    "pred_power_1h": pred_test,
    "hour": test_df["hour"].values,
    "is_day_actual": (y_test.values > DAY_POWER_THRESHOLD).astype(int),
})

result_path = os.path.join(RESULT_DIR, "xgboost_prediction_result.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
print(f"예측 결과 저장 → {result_path}")

metrics_summary = pd.DataFrame([
    {
        "scope": "overall",
        "MAE": overall_metrics["MAE"],
        "RMSE": overall_metrics["RMSE"],
        "R2": overall_metrics["R2"],
    },
    {
        "scope": "active_power_gt_0.05",
        "MAE": day_metrics["MAE"],
        "RMSE": day_metrics["RMSE"],
        "R2": day_metrics["R2"],
    },
    {
        "scope": "hour_06_to_18",
        "MAE": hour_day_metrics["MAE"],
        "RMSE": hour_day_metrics["RMSE"],
        "R2": hour_day_metrics["R2"],
    },
])

metrics_path = os.path.join(RESULT_DIR, "xgboost_metrics_summary.csv")
metrics_summary.to_csv(metrics_path, index=False, encoding="utf-8-sig")
print(f"성능 요약 저장 → {metrics_path}")

# 전체 그래프
plt.figure(figsize=(14, 6))
plt.plot(result_df["timestamp"], result_df["actual_power_1h"], label="Actual", linewidth=2)
plt.plot(result_df["timestamp"], result_df["pred_power_1h"], label="Predicted", linewidth=2)
plt.title("XGBoost: Actual vs Predicted Solar Power (1 Hour Ahead)")
plt.xlabel("Timestamp")
plt.ylabel("Solar Power (W)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plot_path = os.path.join(RESULT_DIR, "xgboost_actual_vs_pred.png")
plt.savefig(plot_path, dpi=250)
plt.close()
print(f"전체 그래프 저장 → {plot_path}")

# 마지막 50개
plot_df = result_df.tail(50)
plt.figure(figsize=(14, 6))
plt.plot(plot_df["timestamp"], plot_df["actual_power_1h"], label="Actual", linewidth=3)
plt.plot(plot_df["timestamp"], plot_df["pred_power_1h"], label="Predicted", linewidth=3)
plt.title("XGBoost: Last 50 Samples")
plt.xlabel("Timestamp")
plt.ylabel("Solar Power (W)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

last50_path = os.path.join(RESULT_DIR, "xgboost_actual_vs_pred_last50.png")
plt.savefig(last50_path, dpi=250)
plt.close()
print(f"마지막 50개 그래프 저장 → {last50_path}")

# 발전구간 그래프
active_plot_df = result_df[result_df["actual_power_1h"] > DAY_POWER_THRESHOLD].copy()
if len(active_plot_df) > 0:
    active_plot_df = active_plot_df.tail(min(80, len(active_plot_df)))
    plt.figure(figsize=(14, 6))
    plt.plot(active_plot_df["timestamp"], active_plot_df["actual_power_1h"], label="Actual", linewidth=3)
    plt.plot(active_plot_df["timestamp"], active_plot_df["pred_power_1h"], label="Predicted", linewidth=3)
    plt.title("XGBoost: Active Generation Zone")
    plt.xlabel("Timestamp")
    plt.ylabel("Solar Power (W)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    active_plot_path = os.path.join(RESULT_DIR, "xgboost_active_zone.png")
    plt.savefig(active_plot_path, dpi=250)
    plt.close()
    print(f"발전구간 그래프 저장 → {active_plot_path}")

# Feature 중요도
importance_df = pd.DataFrame({
    "feature": best_features,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)

importance_path_csv = os.path.join(RESULT_DIR, "xgboost_feature_importance.csv")
importance_df.to_csv(importance_path_csv, index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 8))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

importance_plot_path = os.path.join(RESULT_DIR, "xgboost_feature_importance.png")
plt.savefig(importance_plot_path, dpi=250)
plt.close()
print(f"Feature 중요도 저장 → {importance_plot_path}")

print("\n✅ XGBoost 학습 완료!")