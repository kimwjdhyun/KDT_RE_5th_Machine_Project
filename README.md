# 🌱 AI 스마트팜 제어 시스템

> AI 예측을 실제 제어 시스템 동작으로 연결한 에너지 관리 프로젝트

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST_API-lightgrey?style=flat-square)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![React](https://img.shields.io/badge/Frontend-React-61dafb?style=flat-square)](https://reactjs.org)

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 기간 | 2026.02.24 – 2026.03.27 (약 5주) |
| 팀 구성 | 3인 (팀장 · 웹/하드웨어/AI 담당) |
| 담당 역할 | Flask 백엔드 · XGBoost 모델 · React 대시보드 · 시스템 통합 |

---

## 🎯 문제 정의

기존 스마트팜은 배터리 부족 **이후** 대응하는 구조로, 재생에너지 특성상 안정적인 운영이 어려움.

> **핵심 문제: "에너지 부족을 미리 알 수 없다는 것"**

---

## 💥 목표

1시간 후 발전량을 예측해 에너지 부족 상황을 **사전에** 대응하는 시스템 구축

---

## 🧠 해결 전략

- 1시간 후 발전량 예측 (XGBoost)
- 배터리 SOC 기반 EMS 3단계 자동 전환 로직 구현
- AI 예측 결과를 실제 제어 로직과 연결

---

## ⚙️ 시스템 구조

```
[Arduino] → (시리얼 통신) → [Flask 서버]
     ↓                            ↓
센서 데이터 수집              REST API 제공
(온도·습도·조도·SOC)         /api/latest, /api/history

[Flask 서버] → (XGBoost 추론) → [1시간 후 발전량 예측]
                    ↓
     [React 대시보드] → 실시간 시각화 + AI 신뢰도 표시
                    ↓
     [EMS 제어 로직] → SOC 기준 3단계 자동 전환
     Emergency(<20%) / Eco(<60%) / Standard(≥60%)
```

---

## 🔄 AI → 제어 연결 로직

```python
# 예측값 감소 감지 시 자동 대응
if predicted_power < threshold:
    led_brightness = 40       # 100 → 40 감소
    pump_delay = True         # 펌프 작동 지연
    ems_mode = "Eco"          # 에너지 모드 전환

# 결과: 배터리 SOC 급락 방지
```

---

## 🛠 기술적 의사결정

### 1. 모델 전환 전략 — LSTM → GRU → XGBoost

7일치로 제한된 데이터 규모를 고려해, 복잡한 시계열 모델보다 Feature Engineering 기반 모델이 더 적합하다고 판단. 직접 비교 실험을 통해 XGBoost로 전환.

> 단순히 성능이 나빠서 바꾼 게 아니라, 매 단계마다 데이터와 모델의 적합성을 판단한 근거 있는 의사결정

### 2. Feature Engineering

- Lag / Rolling / 차분 기반 **52개 피처** 직접 설계
- Feature Importance 기반 유효성 검증

### 3. 시간 인코딩 — sin/cos 변환

- sin/cos 변환을 통해 하루 주기 패턴 학습

### 4. 데이터 처리 — 5분 다운샘플링 + shift(-12)

- 10초 단위 원본 데이터 → 5분 단위 다운샘플링
- `shift(-12)` 적용으로 1시간 후 예측 구조 설계

---

## 📈 모델 검증

실제 운영 환경에서 수집된 센서 로그 데이터 **2,463건** 기반 사후 검증

| 지표 | 값 |
|------|-----|
| MAE | 0.1804 |
| RMSE | 0.3758 |
| R² Score | 0.4249 |

> 7일치 소규모 데이터와 실외 환경의 광량 변화 특성상 R²가 제한적이며,
> 이를 인지하고 과대예측 경향을 분석·문서화했습니다.
> 예측 평균(0.2843W) > 실제 평균(0.2023W) → 과대예측 경향 존재

---

## 📊 주요 결과

- 1시간 후 발전량 예측 모델 구현
- 실시간 AI 기반 제어 시스템 구축
- 센서 데이터 **63,937행** 수집 및 활용
- ngrok 기반 실시간 데모 시연

---

## 🔧 트러블슈팅

### 모델 성능 부족
- **원인**: 7일치(2,143행) 데이터로 시계열 패턴 학습 불충분
- **과정**: LSTM → GRU → XGBoost 순으로 직접 비교 실험
- **결과**: MAE 0.129W → 0.105W (18.6% 개선, 훈련 단계 기준 / 최종 검증 MAE 0.1804)

### WiFi 모듈(ESP-01) 불안정
- **원인**: 모듈 발열로 회로 전력 불안정 → 센서값 튐 현상 반복
- **해결**: USB 시리얼 통신으로 전환 → 63,937행 안정적 수집

### 태양광 배터리 충전 불가
- **원인**: 평균 발전 전류 0.07A → CC 조건(0.1A) 미충족
- **해결**: 태양광 패널 병렬 연결을 통한 전류 증대 방안 제안 + EMS 소비 최적화 집중

### 풍력 발전 출력 거의 없음
- **원인**: EVA폼 블레이드로는 일반 바람에서 발전 불가
- **해결**: 블레이드 소재 변경(카본파이버 등) 방향 제시, 단기적으로 태양광 단독 구성으로 전환

---

## 🚧 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|-----------|
| 센서 데이터 중심 → 기상 변수 미반영 | 기상 API 연동 |
| 7일 데이터 → 계절성 반영 한계 | 장기 데이터 수집 |
| 급격한 환경 변화 대응 부족 | Feature Engineering 고도화 |

---

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | Python · Flask · REST API |
| AI/ML | XGBoost (발전량 예측) · PyTorch / GRU (초기 실험 후 전환) · Scikit-learn |
| Frontend | React |
| Data | Pandas · NumPy · Matplotlib |
| IoT | Arduino · INA3221 · DHT22 |

---

## 💡 핵심 인사이트

> AI 모델보다 중요한 것은 데이터 규모와 시스템 구조에 맞는 선택

> 예측 모델은 단독이 아니라 제어 시스템과 결합될 때 가치가 생긴다