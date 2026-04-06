import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 설정
# =========================
CSV_PATH = r"C:\Users\dkreh\Desktop\KDT_RE_5th\9_Machine_Learning_project\backend\data\sensor_log.csv"   # 네 경로에 맞게 수정
TIME_COL = "timestamp"

RAW_TARGET_COL = "solar_power"   # 실제 발전량
PRED_COL = "pred_1h"             # 현재 시점에서 예측한 1시간 후 발전량

RESAMPLE_RULE = "5min"           # 학습/추론 기준
PRED_STEP = 12                   # 5분 * 12 = 60분 후

OUTPUT_DIR = "./backend/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 유틸
# =========================
def safe_read_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    if TIME_COL not in df.columns:
        raise ValueError(f"'{TIME_COL}' 컬럼이 없습니다.")

    if RAW_TARGET_COL not in df.columns:
        raise ValueError(f"'{RAW_TARGET_COL}' 컬럼이 없습니다.")

    if PRED_COL not in df.columns:
        raise ValueError(f"'{PRED_COL}' 컬럼이 없습니다.")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # timestamp 변환
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()

    # 숫자형 변환
    for col in [RAW_TARGET_COL, PRED_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 음수 제거 보정
    df[RAW_TARGET_COL] = df[RAW_TARGET_COL].clip(lower=0)
    df[PRED_COL] = df[PRED_COL].clip(lower=0)

    # 정렬 및 중복 제거
    df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL]).reset_index(drop=True)

    return df


def make_eval_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 10초 단위 로그를 5분 단위로 리샘플링한 뒤,
    pred_1h(현재 시점 예측값)와
    actual_1h(실제 1시간 뒤 solar_power)를 정렬해 평가용 테이블 생성
    """
    temp = df[[TIME_COL, RAW_TARGET_COL, PRED_COL]].copy()
    temp = temp.set_index(TIME_COL)

    # 5분 단위로 평균
    resampled = temp.resample(RESAMPLE_RULE).mean()

    # 실제 1시간 후 값
    resampled["actual_1h"] = resampled[RAW_TARGET_COL].shift(-PRED_STEP)

    # 현재 시점 예측값
    resampled["predicted_1h"] = resampled[PRED_COL]

    # 평가용 테이블
    eval_df = resampled[["predicted_1h", "actual_1h"]].dropna().copy()

    # 이상치 방지
    eval_df = eval_df[np.isfinite(eval_df["predicted_1h"])]
    eval_df = eval_df[np.isfinite(eval_df["actual_1h"])]

    return eval_df


def calc_metrics(eval_df: pd.DataFrame) -> dict:
    y_true = eval_df["actual_1h"].values
    y_pred = eval_df["predicted_1h"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    abs_error = np.abs(y_true - y_pred)

    metrics = {
        "rows": len(eval_df),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "max_error": float(abs_error.max()),
        "min_error": float(abs_error.min()),
        "mean_actual": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }
    return metrics


def analyze_timebands(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    시간대별 오차 확인용
    """
    temp = eval_df.copy()
    temp["hour"] = temp.index.hour
    temp["abs_error"] = np.abs(temp["actual_1h"] - temp["predicted_1h"])

    grouped = temp.groupby("hour").agg(
        samples=("abs_error", "count"),
        mae=("abs_error", "mean"),
        actual_mean=("actual_1h", "mean"),
        pred_mean=("predicted_1h", "mean"),
    )

    return grouped.reset_index()


def plot_actual_vs_pred(eval_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(14, 6))
    plt.plot(eval_df.index, eval_df["actual_1h"], label="Actual 1h Later")
    plt.plot(eval_df.index, eval_df["predicted_1h"], label="Predicted 1h Later")
    plt.title("Actual vs Predicted Solar Power (1 Hour Ahead)")
    plt.xlabel("Time")
    plt.ylabel("Solar Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error_distribution(eval_df: pd.DataFrame, save_path: str):
    errors = eval_df["actual_1h"] - eval_df["predicted_1h"]

    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=30)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hourly_mae(hourly_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_df["hour"], hourly_df["mae"], marker="o")
    plt.title("Hourly MAE")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.xticks(range(0, 24, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_report(metrics: dict, hourly_df: pd.DataFrame, save_path: str):
    lines = []
    lines.append("=== XGBoost 1시간 후 발전량 예측 검증 결과 ===")
    lines.append(f"평가 샘플 수: {metrics['rows']}")
    lines.append(f"MAE      : {metrics['mae']:.4f}")
    lines.append(f"RMSE     : {metrics['rmse']:.4f}")
    lines.append(f"R2 Score : {metrics['r2']:.4f}")
    lines.append(f"최대 오차 : {metrics['max_error']:.4f}")
    lines.append(f"최소 오차 : {metrics['min_error']:.4f}")
    lines.append(f"실제 평균 : {metrics['mean_actual']:.4f}")
    lines.append(f"예측 평균 : {metrics['mean_pred']:.4f}")
    lines.append("")
    lines.append("=== 시간대별 MAE ===")

    for _, row in hourly_df.iterrows():
        lines.append(
            f"{int(row['hour']):02d}시 | samples={int(row['samples'])} | "
            f"MAE={row['mae']:.4f} | actual_mean={row['actual_mean']:.4f} | pred_mean={row['pred_mean']:.4f}"
        )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("[1] CSV 로드")
    df = safe_read_csv(CSV_PATH)

    print("[2] 전처리")
    df = preprocess(df)

    print("[3] 평가용 테이블 생성")
    eval_df = make_eval_table(df)

    if len(eval_df) == 0:
        raise ValueError("평가 가능한 데이터가 없습니다. pred_1h / solar_power / timestamp를 확인하세요.")

    print(f"[INFO] 평가 샘플 수: {len(eval_df)}")

    print("[4] 성능 지표 계산")
    metrics = calc_metrics(eval_df)

    print("[5] 시간대별 오차 분석")
    hourly_df = analyze_timebands(eval_df)

    print("[6] 결과 저장")
    eval_csv_path = os.path.join(OUTPUT_DIR, "xgboost_eval_table.csv")
    hourly_csv_path = os.path.join(OUTPUT_DIR, "xgboost_hourly_mae.csv")
    report_txt_path = os.path.join(OUTPUT_DIR, "xgboost_validation_report.txt")

    plot1_path = os.path.join(OUTPUT_DIR, "xgboost_actual_vs_pred.png")
    plot2_path = os.path.join(OUTPUT_DIR, "xgboost_error_distribution.png")
    plot3_path = os.path.join(OUTPUT_DIR, "xgboost_hourly_mae.png")

    eval_df.to_csv(eval_csv_path, encoding="utf-8-sig")
    hourly_df.to_csv(hourly_csv_path, index=False, encoding="utf-8-sig")

    plot_actual_vs_pred(eval_df, plot1_path)
    plot_error_distribution(eval_df, plot2_path)
    plot_hourly_mae(hourly_df, plot3_path)

    save_report(metrics, hourly_df, report_txt_path)

    print("\n=== 검증 결과 ===")
    print(f"평가 샘플 수 : {metrics['rows']}")
    print(f"MAE          : {metrics['mae']:.4f}")
    print(f"RMSE         : {metrics['rmse']:.4f}")
    print(f"R2 Score     : {metrics['r2']:.4f}")
    print(f"최대 오차     : {metrics['max_error']:.4f}")
    print(f"실제 평균     : {metrics['mean_actual']:.4f}")
    print(f"예측 평균     : {metrics['mean_pred']:.4f}")

    print("\n=== 저장 파일 ===")
    print(eval_csv_path)
    print(hourly_csv_path)
    print(report_txt_path)
    print(plot1_path)
    print(plot2_path)
    print(plot3_path)


if __name__ == "__main__":
    main()