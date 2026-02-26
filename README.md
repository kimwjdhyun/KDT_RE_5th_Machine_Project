# 🌱⚡ Renewable Energy AI Smart Farm  
### _에너지를 이해하는 작은 생태계_

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![Arduino](https://img.shields.io/badge/Arduino-UNO-green)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌍 Story

태양은 매일 떠오르지만,  
그 에너지는 항상 충분하지 않습니다.

맑은 날에는 넘치고,  
흐린 날과 밤에는 사라집니다.

이 프로젝트는 질문에서 시작했습니다.

> “에너지가 부족할 것을 **미리 안다면**,  
> 식물을 더 안정적으로 키울 수 있지 않을까?”

그래서 우리는  
☀️ 태양광 + 💨 풍력으로 전기를 만들고  
🧠 AI가 발전량을 예측하고  
🔋 배터리 상황을 고려해  
🌱 식물이 가장 잘 자랄 환경을 유지하는 시스템을 만들었습니다.

단순 자동화가 아닌,  
**예측 기반 에너지 최적화 스마트팜**입니다.

---

# 📌 Overview

| 항목 | 내용 |
|------|------|
| 프로젝트명 | 재생에너지 기반 AI 스마트팜 에너지 최적화 및 발전량 예측 시스템 |
| 팀 구성 | 이수빈 (AI) · 백동윤 (Hardware) · 김정현 (Web) |
| 핵심 기술 | PyTorch LSTM · Binary Classification · Flask · Arduino UNO · Chart.js |
| 목표 | 발전량 예측 + SOC 기반 에너지 모드 자동 전환 + 급수 알림 시스템 |

---

# 🏗 System Architecture
Solar + Wind
↓
Battery (BMS)
↓
Arduino (Sensors)
↓ WiFi
Flask Server
├─ LSTM Prediction
├─ Water Classifier
└─ Rule-based Mode Control
↓
Web Dashboard

---

## 📊 Data Pipeline
Arduino (5s JSON)
↓
Flask
↓
CSV 저장
↓
전처리
↓
모델 학습 / 추론

### 전처리
- 결측값 제거
- 이상값 클리핑
- MinMax 정규화
- 30-step Sliding Window
- 시간 순서 유지 분할

---

## 🌐 API

| Endpoint | Method | Description |
|-----------|--------|------------|
| `/sensor` | POST | 센서 데이터 수신 + AI 추론 |
| `/data` | GET | 최근 데이터 반환 |
| `/predict` | GET | 1시간 후 발전량 |
| `/stats` | GET | 운영 통계 |
| `/` | GET | 대시보드 |

---

## 📈 Dashboard

- 실시간 발전량
- SOC 충·방전 그래프
- 실제 vs 예측 발전량
- 토양습도 추이
- 에너지 모드 표시
- 급수 알림 및 에너지 경고

---

## 🛠 Tech Stack

### Hardware
- Arduino UNO
- ESP-01
- INA219
- DHT11
- LDR
- Soil Moisture Sensor
- 18650 Battery + BMS

### Backend
- Python
- Flask
- PyTorch
- pandas
- scikit-learn

### Frontend
- HTML/CSS
- JavaScript
- Chart.js

---

## 📅 Timeline

| Week | Task |
|------|------|
| 1 | 기획 및 환경 구성 |
| 2 | 하드웨어 구성 + 서버 연결 |
| 3 | 데이터 수집 + AI 학습 |
| 4 | 통합 + 대시보드 + 성능 평가 |

---

## 🎯 Expected Outcome

- 재생에너지 기반 자가 운영 스마트팜 구현
- LSTM + 분류 모델 동시 적용
- IoT → AI → Web 통합 파이프라인 구축
- 에너지 예측 기반 선제적 절전 제어

---