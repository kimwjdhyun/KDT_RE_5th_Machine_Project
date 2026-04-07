import csv
import os

# 원본 CSV 경로
CSV_PATH = r"C:\Users\dkreh\Desktop\KDT_RE_5th\9_Machine_Learning_project\backend\data\sensor_log.csv"

# 정리된 새 CSV 경로
FIXED_PATH = r"C:\Users\dkreh\Desktop\KDT_RE_5th\9_Machine_Learning_project\backend\data\sensor_log_fixed.csv"

# 최종 목표 19컬럼
TARGET_COLUMNS = [
    "timestamp",
    "temperature",
    "humidity",
    "soil_raw",
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
    "led_brightness",
    "soc",
    "pred_1h",
    "mode",
    "water_alert",
]

# 예전 17컬럼 형식
OLD_17_COLUMNS = [
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
    "pred_1h",
    "mode",
    "water_alert",
]

kept = 0
skipped = 0

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {CSV_PATH}")

with open(CSV_PATH, "r", encoding="utf-8") as rf, open(FIXED_PATH, "w", newline="", encoding="utf-8") as wf:
    reader = csv.reader(rf)
    writer = csv.DictWriter(wf, fieldnames=TARGET_COLUMNS)
    writer.writeheader()

    header = next(reader, None)
    print("원본 헤더:", header)

    for i, row in enumerate(reader, start=2):
        try:
            # 예전 17컬럼 줄
            if len(row) == 17:
                row_dict = dict(zip(OLD_17_COLUMNS, row))
                fixed = {
                    "timestamp": row_dict.get("timestamp", ""),
                    "temperature": row_dict.get("temperature", 0),
                    "humidity": row_dict.get("humidity", 0),
                    "soil_raw": 0,                 # 예전 파일엔 없던 값
                    "soil": row_dict.get("soil", 0),
                    "light": row_dict.get("light", 0),
                    "solar_voltage": row_dict.get("solar_voltage", 0),
                    "solar_current": row_dict.get("solar_current", 0),
                    "solar_power": row_dict.get("solar_power", 0),
                    "battery_voltage": row_dict.get("battery_voltage", 0),
                    "battery_current": row_dict.get("battery_current", 0),
                    "battery_power": row_dict.get("battery_power", 0),
                    "pump": row_dict.get("pump", 0),
                    "led": row_dict.get("led", 0),
                    "led_brightness": 0,           # 예전 파일엔 없던 값
                    "soc": row_dict.get("soc", 0),
                    "pred_1h": row_dict.get("pred_1h", 0),
                    "mode": row_dict.get("mode", 0),
                    "water_alert": row_dict.get("water_alert", 0),
                }
                writer.writerow(fixed)
                kept += 1

            # 현재 19컬럼 줄
            elif len(row) == 19:
                row_dict = dict(zip(TARGET_COLUMNS, row))
                writer.writerow(row_dict)
                kept += 1

            # 컬럼 수가 이상한 줄은 건너뜀
            else:
                skipped += 1
                print(f"[SKIP] line {i}: column count = {len(row)}")

        except Exception as e:
            skipped += 1
            print(f"[ERROR] line {i}: {e}")

print(f"\n완료")
print(f"정상 저장 행 수: {kept}")
print(f"건너뛴 행 수: {skipped}")
print(f"저장 파일: {FIXED_PATH}")