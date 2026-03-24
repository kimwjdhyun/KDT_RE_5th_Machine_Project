# ================================================
# generate_data.py
# 가상 센서 데이터 생성기
#
# [왜 이 파일이 필요한가?]
#   AI 모델(GRU, 급수 분류)을 학습시키려면 많은 데이터가 필요함
#   실제 아두이노 센서로 2주치 데이터를 모으려면 2주를 기다려야 함
#   그래서 실제와 비슷한 가짜 데이터를 수학적으로 만들어서 사용
#
# [실행하면 생성되는 것]
#   farm_data.csv → train_gru.py, train_water.py의 입력 파일
#
# [나중에 실제 센서 데이터로 교체하는 방법]
#   아두이노가 Flask로 보내는 데이터가 farm_data.csv에 쌓임
#   2주치 쌓이면 이 파일 안 쓰고 바로 학습 가능
#
# [현재 없는 것 - SOC]
#   soc 컬럼이 없음 → 나중에 배터리 연결 후 추가 예정
#   추가하려면: rows.append() 안에 "soc": round(soc, 1) 넣기
# ================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DAYS     = 7   # 생성할 데이터 기간 (7일치)
INTERVAL = 1   # 샘플링 간격 (분) - 1분마다 1개 행
               # 7일 × 24시간 × 60분 = 10,080행 생성됨

np.random.seed(42)  # 랜덤 시드 고정 → 실행할 때마다 같은 데이터 나옴


# ────────────────────────────────────────
# 태양광 발전량 계산 함수
#
# [실제 태양광 패널 원리]
#   태양이 가장 높이 뜨는 정오(13시)에 최대 발전
#   아침/저녁으로 갈수록 줄어드는 종 모양 곡선
#
# [수학적 모델링 - 가우시안 곡선]
#   base = 최대출력 × e^(-(시간-정오)² / 퍼짐)
#   정오에 가까울수록 1에 가까워져서 최대 출력
#   정오에서 멀수록 0에 가까워져서 출력 감소
#
# [cloud 파라미터]
#   0.0 = 맑은 날 (최대 발전)
#   0.7 = 흐린 날 (약 85% 감소)
#   np.random.normal(0, 0.15) = 구름이 지나가는 노이즈 효과
# ────────────────────────────────────────
def solar_power(hour, minute, cloud=0.0):
    t = hour + minute / 60.0  # 예) 13시 30분 → 13.5

    # 야간(6시 이전, 19시 이후)은 발전 없음
    if t < 6.0 or t > 19.0:
        return 0.0

    # 가우시안 곡선: 13시가 꼭대기, 14.0이 클수록 완만한 곡선
    # 12.0 = 패널 최대 출력(W) - 실제 패널 스펙에 맞게 조정 가능
    base = 12.0 * np.exp(-((t - 13.0) ** 2) / 14.0)

    # 흐린 날씨 반영: cloud=0.7이면 최대 85% 발전량 감소
    base *= (1.0 - cloud * 0.85)

    # 구름이 순간적으로 지나가는 효과 (랜덤 노이즈)
    return max(0.0, base + np.random.normal(0, 0.15))


# ────────────────────────────────────────
# 조도 계산 함수
#
# [실제 LDR 센서와의 관계]
#   LDR(빛 저항)은 밝을수록 저항 낮아짐 → 전압 높아짐 → 아날로그 값 커짐
#   아두이노에서 0~1023으로 읽어서 0~100%로 변환
#
# 태양광 발전량과 비슷한 패턴이지만 독립적으로 계산
# (빛이 있어도 패널 각도나 그늘로 발전량이 다를 수 있음)
# ────────────────────────────────────────
def calc_light(hour, minute, cloud=0.0):
    t = hour + minute / 60.0

    if t < 6.0 or t > 19.0:
        return 0.0  # 밤은 조도 0%

    # 최대 100%, 정오에 꼭대기인 가우시안 곡선
    base = 100.0 * np.exp(-((t - 13.0) ** 2) / 16.0)
    base *= (1.0 - cloud * 0.75)  # 흐린 날 조도 감소

    # clip: 0~100 범위 벗어나지 않게 제한
    return float(np.clip(base + np.random.normal(0, 2.0), 0.0, 100.0))


# ────────────────────────────────────────
# 배터리 SOC 업데이트 함수
# [주의: 현재 farm_data.csv에 soc 컬럼 없음!]
# [나중에 배터리 연결하면 이 함수 결과를 CSV에 포함시킬 것]
#
# [SOC 계산 원리]
#   SOC = State of Charge = 배터리 남은 용량 (%)
#   충전량 = 발전량 × 시간 / 배터리용량 × 100
#   소비량 = 아두이노 + 센서 소비전력 × 시간 / 배터리용량 × 100
#   SOC 변화 = 충전량 - 소비량
# ────────────────────────────────────────
def update_soc(s_power, prev_soc):
    capacity_wh = 37.0    # 18650 배터리 4개 직렬 기준 총 용량(Wh)
                          # 18650 1ro = 3.7V 2.5Ah 기준
    dt_h        = INTERVAL / 60.0  # 1분 → 0.0167시간
    consumption = 0.5     # 아두이노 + 센서 + ESP-01 소비전력 (W 추정치)

    # (발전 - 소비) × 시간 / 용량 = SOC 변화율
    delta = ((s_power - consumption) * dt_h / capacity_wh) * 100.0

    # clip: SOC는 5%~100% 범위로 유지 (0%면 배터리 완전방전, 실제론 BMS가 차단)
    return float(np.clip(prev_soc + delta, 5.0, 100.0))


# ────────────────────────────────────────
# 토양습도 업데이트 함수
#
# [실제 토양 건조 원리]
#   물을 안 주면 증발 + 식물 흡수로 조금씩 줄어듦
#   0.05~0.15% 씩 감소 (분당)
#
# [급수 시 변화]
#   토양습도 35% 이하가 되면 자동 급수
#   급수 후 30~50%씩 급격히 증가
# ────────────────────────────────────────
def update_soil(prev_soil, was_watered=False):
    if was_watered:
        # 급수 후 습도 급상승 (30~50% 랜덤)
        return min(90.0, prev_soil + np.random.uniform(30, 50))

    # 자연 건조: 분당 0.05~0.15% 감소
    return max(10.0, prev_soil - np.random.uniform(0.05, 0.15))


# ────────────────────────────────────────
# 전체 데이터 생성 메인 함수
# ────────────────────────────────────────
def generate():
    rows  = []
    start = datetime(2025, 3, 1)  # 데이터 시작 날짜 (아무 날짜나 OK)
    soc   = 60.0   # 초기 SOC 60% (배터리 절반 차있다고 가정)
    soil  = 75.0   # 초기 토양습도 75% (촉촉한 상태에서 시작)

    # 총 스텝 수 계산: 7일 × 1440분/일 ÷ 1분 = 10,080 스텝
    steps = (DAYS * 24 * 60) // INTERVAL

    for step in range(steps):
        now  = start + timedelta(minutes=step * INTERVAL)
        h, m = now.hour, now.minute

        # 몇 번째 날인지 계산 (4일마다 흐린 날 시뮬레이션)
        day_num = step * INTERVAL // (24 * 60)

        # cloud: 4일째마다 흐린 날(0.7), 나머지는 맑은 날 (0~0.2)
        cloud = 0.7 if day_num % 4 == 3 else np.random.uniform(0, 0.2)

        # 각 센서값 계산
        s_power = solar_power(h, m, cloud)  # 태양광 발전량 (W)
        light   = calc_light(h, m, cloud)   # 조도 (0~100%)

        # 전압: 발전량에 비례 (9V~12V 범위)
        # 발전 없으면 9V(최소), 최대 발전이면 12V(최대)
        voltage = 9.0 + (s_power / 12.0) * 3.0 + np.random.normal(0, 0.1)

        # 온도: 하루 중 사인 곡선 (새벽 낮고, 낮에 높고)
        temp = 22.0 + 2.0 * np.sin(2 * np.pi * h / 24) + np.random.normal(0, 0.3)

        # 습도: 새벽에 높고, 낮에 낮은 패턴
        hum = float(np.clip(
            55.0 + 10.0 * np.sin(2 * np.pi * (h - 6) / 24) + np.random.normal(0, 1.0),
            30.0, 90.0
        ))

        # SOC 업데이트 (발전량으로 충전, 소비전력으로 방전)
        soc = update_soc(s_power, soc)

        # 토양습도 업데이트 (35% 이하면 급수)
        soil = update_soil(soil, was_watered=(soil < 35.0))

        # 에너지 모드 결정 (SOC 기반)
        # 나중에 실제 배터리 연결하면 이 로직이 app_gru.py에서 처리됨
        mode = 0 if soc < 20 else (1 if soc < 60 else 2)

        rows.append({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "temp":      round(temp, 1),
            "hum":       round(hum, 1),
            "power":     round(s_power, 3),   # 태양광 발전량 (W)
            "voltage":   round(voltage, 3),   # 전압 (V)
            "soc":       round(soc, 1),       # 배터리 SOC (%)
                                              # ← 지금은 생성만 하고 GRU 학습에는 안 씀
            "light":     round(light, 1),     # 조도 (%)
            "soil":      round(soil, 1),      # 토양습도 (%)
            "mode":      mode,                # 에너지 모드 (0/1/2)
        })

    df = pd.DataFrame(rows)
    os.makedirs("../../backend", exist_ok=True)
    df.to_csv("farm_data.csv", index=False)
    print(f"✅ farm_data.csv 생성 완료 ({len(df)}행)")
    print(f"   컬럼: {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate()