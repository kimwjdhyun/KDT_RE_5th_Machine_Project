# ================================================
# generate_data.py
# 가상 센서 데이터 생성기
#
# 실제 아두이노 센서가 아직 준비되지 않았을 때
# AI 학습용 데이터를 미리 만들어두는 파일.
# 실제 하드웨어가 준비되면 이 파일은 필요 없음.
#
# 실행하면 → farm_data.csv 생성
# ================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DAYS     = 14   # 2주치 데이터 생성
INTERVAL = 5    # 5분 단위 샘플링 → 총 4,032행

np.random.seed(42)   # 매번 같은 데이터가 나오도록 고정


# ────────────────────────────────────────
# 태양광 발전량 (W)
#
# 실제 태양광 패널은 정오(13시)에 최대 출력,
# 오전/오후로 갈수록 줄어드는 종 모양 패턴.
# 수학적으로는 가우시안(정규분포) 곡선으로 표현.
# cloud 값이 높을수록 구름이 많아 발전량 감소.
# ────────────────────────────────────────
def solar_power(hour, minute, cloud=0.0):
    t = hour + minute / 60.0

    if t < 6.0 or t > 19.0:
        return 0.0   # 오전 6시 이전, 오후 7시 이후는 야간 → 발전 없음

    # 가우시안 곡선: 13시가 꼭대기, spread(14)가 클수록 완만한 곡선
    base = 12.0 * np.exp(-((t - 13.0) ** 2) / 14.0)
    base *= (1.0 - cloud * 0.85)                      # 구름이 낄수록 최대 85% 감소
    return max(0.0, base + np.random.normal(0, 0.15)) # 구름 지나가는 노이즈


# ────────────────────────────────────────
# 풍력 발전량 (W)
#
# 소형 풍력 발전기는 풍속이 일정 수준(컷인 풍속 2m/s) 이상이어야 발전 시작.
# 발전량은 (풍속 - 컷인풍속)^2에 비례 → 풍속이 조금만 올라가도 빠르게 증가.
# 태양광과 달리 밤에도 발전 가능 → 야간 배터리 방전 보완 역할.
# ────────────────────────────────────────
def wind_power(hour, cloud=0.0):
    # 풍속 시뮬레이션: 하루에 두 번 피크(오전/오후), 흐린 날 바람이 약간 더 강함
    speed = (
        2.5
        + 1.2 * np.sin(2 * np.pi * hour / 24 * 2)
        + cloud * 0.8
        + np.random.normal(0, 0.7)
    )
    speed = max(0.0, speed)

    if speed > 2.0:
        # 컷인 풍속(2m/s) 초과 시 발전 시작
        # 예) 풍속 3m/s → (3-2)^2 × 2.5 = 2.5W
        #     풍속 4m/s → (4-2)^2 × 2.5 = 10W
        return max(0.0, (speed - 2.0) ** 2 * 2.5 + np.random.normal(0, 0.3))
    return 0.0   # 컷인 풍속 미달 → 발전 없음


# ────────────────────────────────────────
# 배터리 SOC 업데이트 (%)
#
# SOC(State of Charge) = 배터리 충전 상태.
# 태양광 + 풍력으로 충전, 아두이노·센서가 소비.
# 발전량 > 소비량이면 SOC 증가, 반대면 감소.
# ────────────────────────────────────────
def update_soc(s_power, w_power, prev_soc):
    capacity_wh = 14.8           # 18650 배터리 2개 직렬 = 14.8Wh
    dt_h        = INTERVAL / 60.0  # 5분 → 0.083시간
    consumption = 0.5              # 아두이노 + 센서 소비 전력(W)

    # (총발전 - 소비) × 시간 / 용량 = SOC 변화량
    delta = ((s_power + w_power - consumption) * dt_h / capacity_wh) * 100.0
    return float(np.clip(prev_soc + delta, 5.0, 100.0))   # 5~100% 범위 유지


# ────────────────────────────────────────
# 토양습도 업데이트 (%)
#
# 물을 안 주면 자연 증발로 조금씩 감소.
# 토양습도 35% 이하가 되면 급수 → 습도가 급격히 상승.
# ────────────────────────────────────────
def update_soil(prev_soil, was_watered=False):
    if was_watered:
        return min(90.0, prev_soil + np.random.uniform(30, 50))   # 급수 후 급상승
    return max(10.0, prev_soil - np.random.uniform(0.05, 0.15))   # 자연 건조


# ────────────────────────────────────────
# 메인: 전체 데이터 생성
# ────────────────────────────────────────
def generate():
    rows  = []
    start = datetime(2025, 3, 1)
    soc   = 60.0   # 초기 SOC 60%
    soil  = 75.0   # 초기 토양습도 75%
    steps = (DAYS * 24 * 60) // INTERVAL   # 4,032 스텝

    for step in range(steps):
        now    = start + timedelta(minutes=step * INTERVAL)
        h, m   = now.hour, now.minute

        # 4일마다 흐린 날 시뮬레이션 (day 3, 7, 11...)
        day_num = step * INTERVAL // (24 * 60)
        cloud   = 0.7 if day_num % 4 == 3 else np.random.uniform(0, 0.2)

        s_power = solar_power(h, m, cloud)
        w_power = wind_power(h, cloud)

        # 전압: 태양광 출력 기준 9~12V 범위
        voltage = 9.0 + (s_power / 12.0) * 3.0 + np.random.normal(0, 0.1)

        temp = 22.0 + 2.0 * np.sin(2 * np.pi * h / 24) + np.random.normal(0, 0.3)
        hum  = float(np.clip(
            55.0 + 10.0 * np.sin(2 * np.pi * (h - 6) / 24) + np.random.normal(0, 1.0),
            30.0, 90.0
        ))

        soc  = update_soc(s_power, w_power, soc)
        soil = update_soil(soil, was_watered=(soil < 35.0))

        # 에너지 모드: SOC 기준으로 3단계 결정
        # 0=긴급절전(SOC<20%), 1=절약(20~60%), 2=풀가동(60% 이상)
        mode = 0 if soc < 20 else (1 if soc < 60 else 2)

        rows.append({
            "timestamp":  now.strftime("%Y-%m-%d %H:%M:%S"),
            "temp":       round(temp, 1),
            "hum":        round(hum, 1),
            "power":      round(s_power, 3),
            "wind_power": round(w_power, 3),
            "voltage":    round(voltage, 3),
            "soc":        round(soc, 1),
            "soil":       round(soil, 1),
            "mode":       mode,
        })

    df = pd.DataFrame(rows)
    df.to_csv("farm_data.csv", index=False)
    print(f"✅ farm_data.csv 생성 완료 ({len(df)}행)")
    print(df.head(3))
    return df


if __name__ == "__main__":
    generate()