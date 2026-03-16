import os           # 파일/폴더 존재 확인, 경로 처리
import csv          # CSV 파일 읽기/쓰기 (farm_data.csv)
import numpy as np  # 배열 연산 (GRU 입력 데이터 가공)
import torch        # PyTorch (GRU, 급수 모델 실행)
import torch.nn as nn  # 신경망 레이어 정의
import joblib       # 스케일러(.pkl 파일) 불러오기
from flask import Flask, request, jsonify  # 웹 서버, 요청 처리, JSON 응답
from flask_cors import CORS                # CORS: 다른 출처(포트)에서 접근 허용
from datetime import datetime              # 현재 시각 (CSV 타임스탬프용)
from collections import deque             # deque: 최근 N개만 유지하는 큐 (버퍼용)


# ================================================
# Flask 앱 초기화
#
# Flask(__name__) : 현재 파일을 기준으로 Flask 앱 생성
# JSON_AS_ASCII=False : 응답 JSON에 한글이 깨지지 않게 UTF-8로 출력
# CORS(app) : Cross-Origin Resource Sharing 허용
#   → 브라우저에서 다른 IP/포트로 API 요청 가능하게 해줌
#   → 아두이노는 직접 HTTP 요청이라 사실상 필요 없지만
#     나중에 웹 대시보드 만들 때 필요
# ================================================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 JSON 응답 깨짐 방지
CORS(app)                            # 모든 출처에서 API 접근 허용


# ================================================
# 전역 설정값
#
# CSV_FILE        : 아두이노 데이터 저장 파일 이름
# SEQ_LEN         : GRU에 넣을 과거 데이터 개수
#                   30 = 최근 30개 데이터 (아두이노가 1분마다 보내면 = 30분치)
#                   train_gru.py의 SEQ_LEN과 맞출 필요 없음 (추론용이므로)
# WATER_THRESHOLD : 급수 AI가 출력한 확률이 이 값 이상이면 급수 명령
#                   0.5 = 50% 이상이면 "물 줘야 함"
# SOIL_THRESHOLD  : AI 모델 없을 때 규칙 기반 판단 기준
#                   40.0% 이하면 급수 명령
# ================================================
CSV_FILE        = "farm_data.csv"
SEQ_LEN         = 30     # GRU 입력 시퀀스 길이 (최근 30개 데이터)
WATER_THRESHOLD = 0.5    # 급수 AI 확률 임계값 (0.5 = 50%)
SOIL_THRESHOLD  = 40.0   # 규칙 기반 토양습도 임계값 (%)


# ================================================
# 키 이름 호환 헬퍼 함수
#
# [왜 필요한가?]
#   아두이노 코드가 "temperature", "solar_power", "solar_voltage" 등
#   긴 키 이름으로 전송함
#   기존 서버 코드는 "temp", "power", "voltage" 등 짧은 키로 받음
#   → 두 키 이름을 모두 허용해서 어떤 아두이노 코드와도 호환되게 함
#
# [우선순위]
#   새 키(solar_power 등) 먼저 확인 → 없으면 예전 키(power 등) 확인
# ================================================
def get_val(data, *keys, default=0.0):
    # keys 순서대로 찾아서 처음 발견된 값 반환
    for k in keys:
        if k in data and data[k] != "" and data[k] is not None:
            try:
                return float(data[k])
            except (TypeError, ValueError):
                continue
    return default


# ================================================
# GRU 발전량 예측 모델 클래스
#
# ★ 반드시 train_gru.py의 PowerPredictionGRU와 완전히 동일해야 함!
#   클래스 구조가 다르면 저장된 .pth 가중치를 불러올 때 에러 발생
#
# [구조 설명]
#   GRU 레이어  : 시계열 패턴을 기억하는 핵심 레이어
#                 과거 30개 데이터의 흐름을 hidden state(32차원)에 압축
#   FC 레이어   : GRU 출력(32차원) → 16 → 1 (발전량 예측값)
#   ReLU        : 음수 → 0 변환 (비선형성 추가)
#
# [input_size=7인 이유]
#   power, voltage, light, temp, hour_sin, hour_cos, soc → 7개
# ================================================
class PowerPredictionGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=1):
        super().__init__()
        # GRU: 시계열 데이터 처리
        # input_size=7  : 입력 특성 수 (power, voltage, light, temp, hour_sin, hour_cos, soc)
        # hidden_size=32: GRU 내부 기억 크기 (클수록 복잡한 패턴 학습, 느려짐)
        # num_layers=1  : GRU 레이어 1개 (보통 1~3개)
        # batch_first=True : 입력 형태를 (배치, 시퀀스, 특성) 순서로 받음
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # FC: GRU 출력을 발전량 1개 값으로 변환
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),  # 32차원 → 16차원
            nn.ReLU(),                   # 음수 제거
            nn.Linear(16, 1)             # 16차원 → 발전량 1개
        )

    def forward(self, x):
        # x의 형태: (배치크기, 시퀀스길이, 특성수) = (1, 30, 7)
        out, _ = self.gru(x)          # out: (1, 30, 32) - 모든 시점의 hidden state
        return self.fc(out[:, -1, :]) # 마지막 시점(-1)만 FC에 통과 → (1, 1)


# ================================================
# 급수 분류 모델 클래스
#
# ★ 반드시 train_water.py의 WaterClassifier와 완전히 동일해야 함!
#
# [구조 설명]
#   입력 3개 (토양습도, 온도, 공기습도)
#   → 16개로 확장 (패턴 학습)
#   → 8개로 압축
#   → 1개 (급수 필요 확률 0~1)
#
# [Sigmoid 마지막에 붙이는 이유]
#   Linear 출력은 어떤 값이든 가능 (-∞ ~ +∞)
#   Sigmoid = 어떤 값이든 0~1 사이로 압축
#   → "급수 필요 확률"로 해석 가능
#   → 0.5 이상이면 급수, 미만이면 대기
# ================================================
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),   # 입력 3개 → 16개
            nn.ReLU(),
            nn.Linear(16, 8),   # 16개 → 8개
            nn.ReLU(),
            nn.Linear(8, 1),    # 8개 → 1개 (확률값)
            nn.Sigmoid()        # 0~1 범위로 압축
        )

    def forward(self, x):
        return self.net(x)


# ================================================
# 모델 & 스케일러 로드
#
# [로드 전략]
#   모델 파일(.pth)이 있으면 → AI 모델 사용
#   없으면 → 규칙 기반(단순 조건문)으로 대체 동작
#   → 처음 실행 때 train_gru.py를 아직 안 돌렸어도 서버가 켜짐!
#
# [map_location="cpu"]
#   GPU 없는 환경(노트북)에서도 모델 로드 가능하게 CPU 강제 지정
#   GPU 있는 서버에서 학습한 모델을 노트북에서 실행할 때도 OK
#
# [eval() 호출 이유]
#   model.train() 상태 = 학습 모드 (Dropout, BatchNorm 등 활성화)
#   model.eval()  상태 = 추론 모드 (Dropout 비활성화, 결과 안정)
#   추론할 때 반드시 eval() 먼저 호출해야 정확한 결과 나옴!
# ================================================

# 모델 객체 생성 (가중치 없는 빈 껍데기 상태)
gru_model    = PowerPredictionGRU()
water_model  = WaterClassifier()

# 스케일러: 입력값을 0~1로 정규화하는 기준값 저장
scaler_gru   = None  # GRU 입력 정규화용
scaler_water = None  # 급수 모델 입력 정규화용

# 모델 준비 상태 플래그
gru_ready    = False  # GRU 모델 사용 가능 여부
water_ready  = False  # 급수 모델 사용 가능 여부

# ── GRU 모델 로드 시도 ──────────────────────────
# 두 파일 모두 있어야 정상 동작 (모델 + 스케일러)
if os.path.exists("models/gru.pth") and os.path.exists("models/scaler_gru.pkl"):
    try:
        # load_state_dict: 저장된 가중치(숫자들)를 빈 모델에 채워 넣음
        gru_model.load_state_dict(torch.load("models/gru.pth", map_location="cpu"))
        gru_model.eval()                                   # 추론 모드로 전환
        scaler_gru = joblib.load("models/scaler_gru.pkl")  # 스케일러 복원
        gru_ready  = True
        print("✅ GRU 모델 로드 완료")
    except Exception as e:
        # 파일이 있어도 구조가 다르거나 손상됐으면 실패
        print(f"⚠️ GRU 모델 로드 실패: {e} → 규칙 기반으로 동작")
else:
    print("⚠️ GRU 모델 없음 → 규칙 기반으로 동작 (train_gru.py 실행 후 재시작)")

# ── 급수 분류 모델 로드 시도 ────────────────────
if os.path.exists("models/water.pth") and os.path.exists("models/scaler_water.pkl"):
    try:
        water_model.load_state_dict(torch.load("models/water.pth", map_location="cpu"))
        water_model.eval()
        scaler_water = joblib.load("models/scaler_water.pkl")
        water_ready  = True
        print("✅ 급수 분류 모델 로드 완료")
    except Exception as e:
        print(f"⚠️ 급수 모델 로드 실패: {e} → 규칙 기반으로 동작")
else:
    print("⚠️ 급수 모델 없음 → 토양습도 40% 기준으로 동작")

# ── 최근 데이터 버퍼 ────────────────────────────
# deque(maxlen=SEQ_LEN) : 최대 30개까지만 저장하는 큐
#   새 데이터가 들어오면 가장 오래된 데이터를 자동으로 제거
#   아두이노가 1분마다 데이터를 보내면 → 30분치 최근 데이터를 항상 유지
#   GRU 추론 시 이 버퍼를 시퀀스로 사용
recent_buffer = deque(maxlen=SEQ_LEN)


# ================================================
# CSV 저장 함수
#
# 아두이노에서 받은 데이터를 farm_data.csv에 한 줄씩 추가
# 나중에 AI 재학습 시 이 파일을 train_gru.py에 넣으면 됨
#
# [DictWriter 사용 이유]
#   일반 writer는 순서대로 값만 쓰는데
#   DictWriter는 {"컬럼명": 값} 딕셔너리로 써서 컬럼 순서 보장
# ================================================
def save_csv(data: dict, mode: int):
    # CSV에 저장할 컬럼 목록 (순서대로)
    fieldnames = ["timestamp", "temp", "hum", "power",
                  "voltage", "current", "light", "soil", "soc", "mode"]

    # 파일이 없으면 헤더(첫 줄)를 먼저 써야 함
    file_exists = os.path.exists(CSV_FILE)

    # "a" = append 모드: 기존 내용 유지하고 맨 아래에 추가
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()   # 파일이 없었으면 헤더 한 번만 씀

        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # get_val(): 새 키(temperature 등) 먼저 확인, 없으면 예전 키(temp 등) 확인
            # round(값, 소수점자리): 저장 정밀도 설정
            "temp":    round(get_val(data, "temperature",   "temp"),    1),
            "hum":     round(get_val(data, "humidity",      "hum"),     1),
            "power":   round(get_val(data, "solar_power",   "power"),   3),
            "voltage": round(get_val(data, "solar_voltage", "voltage"), 3),
            "current": round(get_val(data, "solar_current", "current"), 3),
            "light":   round(get_val(data, "light"),                    1),
            "soil":    round(get_val(data, "soil"),                     1),
            "soc":     int(get_val(data, "soc", default=100)),
            "mode":    mode,
        })


# ================================================
# 에너지 모드 결정 함수
#
# [SOC 기반으로 변경]
#   SOC < 20% -> mode 0: 긴급절전 (배터리 거의 없음, 최소한만 사용)
#   SOC < 60% -> mode 1: 절약 (배터리 절반 이하, 아껴서 사용)
#   SOC > 60% -> mode 2: 풀가동 (배터리 충분, 마음껏 사용)
#
# [GRU 선제 강등]
#   지금 당장은 발전이 충분해도
#   GRU가 "앞으로 발전량이 1W 미만일 것" 예측하면
#   미리 한 단계 내려서 절전 시작
#   → 과방전 방지 (배터리가 갑자기 떨어지는 상황 예방)
# ================================================
def decide_mode(power: float, pred_power, soc: int) -> int:
    # soc 기본 모드 결정
    if soc < 20:
        base = 0   # SOC 20% 미만 -> 긴급절전
    elif soc < 60:
        base = 1   # SOC 60% 미만 -> 절약
    else:
        base = 2   # SOC 60% 이상 -> 풀가동

    # GRU 예측값이 있고, 앞으로 발전량이 1W 미만 예측되면 선제 강등
    # pred_power is not None: GRU 데이터가 아직 30개 미만이면 None이라 건너뜀
    # base > 0: 이미 긴급절전(0)이면 더 내릴 곳 없음
    if pred_power is not None and pred_power < 1.0 and base > 0:
        base -= 1  # 한 단계 강등 (2→1, 1→0)

    return base


# ================================================
# GRU 발전량 예측 함수
#
# [동작 조건]
#   gru_ready == True   : 모델이 로드됐을 때만 실행
#   len(recent_buffer) >= SEQ_LEN : 최근 30개 데이터가 모였을 때만 실행
#   → 처음 30분은 데이터 부족으로 None 반환 (규칙 기반으로 대체)
#
# [추론 과정]
#   버퍼의 30개 데이터 → numpy 배열 → 스케일러로 정규화 → 텐서 변환 → GRU 입력
#   → 예측값(정규화된 상태) → 역정규화 → 실제 W 단위
#
# [입력 변수 7개 순서]
#   power, voltage, light, temp, hour_sin, hour_cos, soc
#   ★ train_gru.py의 FEATURES 순서와 완전히 같아야 함!
#
# [dummy 역정규화 방법]
#   scaler_gru는 7개 특성 전체를 정규화한 스케일러
#   발전량만 역변환하려면 (1, 7) 크기 배열이 필요
#   0으로 채운 dummy를 만들고 index 0(power 위치)에만 예측값 넣어서 역변환
# ================================================
def predict_power(data: dict):
    # 조건 미충족 시 None 반환 (호출한 쪽에서 None 체크)
    if not gru_ready or len(recent_buffer) < SEQ_LEN:
        return None

    try:
        hour = datetime.now().hour  # 현재 시간 (0~23)

        # 버퍼의 30개 데이터를 7개 특성 순서대로 리스트로 변환
        # 순서: [power, voltage, light, temp, hour_sin, hour_cos, soc]
        # ★ 이 순서가 train_gru.py의 FEATURES 순서와 완전히 같아야 함!
        # get_val(): 새 키(solar_power 등) 먼저 확인, 없으면 예전 키(power 등) 확인
        seq = []
        for row in recent_buffer:
            seq.append([
                get_val(row, "solar_power",   "power"),
                get_val(row, "solar_voltage", "voltage"),
                get_val(row, "light"),
                get_val(row, "temperature",   "temp"),
                np.sin(2 * np.pi * hour / 24),   # hour_sin: 시간 순환성 표현
                np.cos(2 * np.pi * hour / 24),   # hour_cos: sin만으론 3시/9시 구분 불가
                get_val(row, "soc", default=100),
            ])

        seq_np     = np.array(seq, dtype=np.float32)   # 리스트 → numpy (30, 7)
        seq_scaled = scaler_gru.transform(seq_np)       # 정규화: 0~1 범위로 변환

        # unsqueeze(0): (30, 7) → (1, 30, 7)
        # GRU는 3D 입력 필요: (배치크기, 시퀀스길이, 특성수)
        # 지금은 1개만 추론하므로 배치크기=1
        x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

        # torch.no_grad(): 기울기 계산 안 함 (추론 시 메모리/속도 절약, 필수!)
        with torch.no_grad():
            pred_scaled = gru_model(x).cpu().numpy()  # 결과: (1, 1) numpy 배열

        # 역정규화: 정규화된 예측값 → 실제 W 단위로 복원
        # scaler_gru는 7개 특성용이라 (1, 7) 형태가 필요
        dummy = np.zeros((1, 7))          # 0으로 채운 (1, 7) 배열
        dummy[0, 0] = pred_scaled[0, 0]   # index 0 = power 위치에 예측값 넣기
        pred_real = scaler_gru.inverse_transform(dummy)[0, 0]  # 역변환 후 power 값 추출

        return max(0.0, float(pred_real))  # 발전량은 음수 불가 → 0 이상으로 제한

    except Exception as e:
        print(f"GRU 추론 오류: {e}")
        return None  # 오류 나면 None 반환 → 규칙 기반으로 대체


# ================================================
# 급수 분류 추론 함수
#
# [AI 모델 있을 때]
#   토양습도, 온도, 공기습도를 정규화 → 모델 입력
#   출력 확률 >= 0.5이면 True (급수 필요)
#
# [AI 모델 없을 때 (규칙 기반)]
#   단순하게 토양습도 40% 이하면 True
#
# [왜 규칙 기반 fallback이 필요한가?]
#   처음 서버 켰을 때 모델 파일이 없어도 동작해야 하기 때문
#   모델 학습 후 서버 재시작하면 자동으로 AI 모드로 전환됨
# ================================================
def predict_water(soil: float, temp: float, hum: float) -> bool:
    # AI 모델 없으면 단순 규칙으로 판단
    if not water_ready:
        return soil < SOIL_THRESHOLD  # 토양습도 40% 이하면 급수

    try:
        # 입력값을 (1, 3) 형태 배열로 만들기 (1개 샘플, 3개 특성)
        features = np.array([[soil, temp, hum]], dtype=np.float32)

        # 스케일러로 0~1 범위 정규화 (학습 때와 동일한 기준 적용)
        features_scaled = scaler_water.transform(features)

        # numpy → PyTorch 텐서로 변환
        x = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = water_model(x).item()  # .item(): 텐서 → 파이썬 float로 변환

        # 확률이 임계값 이상이면 급수 필요
        return prob >= WATER_THRESHOLD

    except Exception as e:
        print(f"급수 추론 오류: {e}")
        return soil < SOIL_THRESHOLD  # 오류 시 규칙 기반으로 대체


# ================================================
# Flask 라우트: /sensor (POST)
#
# [역할]
#   아두이노 A보드가 1분마다 이 주소로 센서 데이터 전송
#   받은 데이터로 AI 추론 후 명령을 응답으로 돌려줌
#
# [@app.route 데코레이터]
#   해당 URL 경로와 HTTP 메서드를 이 함수에 연결
#   methods=["POST"] : POST 요청만 받음 (GET은 무시)
#
# [아두이노가 보내는 JSON 예시 - 새 키]
#   {"temperature":24.5, "humidity":58.0, "soil":47, "light":51,
#    "solar_voltage":12.34, "solar_current":0.32, "solar_power":2.5,
#    "battery_voltage":14.8, "soc":80, "pump":0, "led":1}
#
# [아두이노가 보내는 JSON 예시 - 예전 키 (호환)]
#   {"temp":24.5, "hum":58.0, "soil":47, "light":51,
#    "voltage":12.34, "current":320.5, "power":2504.0,
#    "soc":80, "pump":0, "led":1}
#
# [Flask가 돌려주는 JSON 예시]
#   {"mode":1, "water_alert":true, "pred_power":6.2}
# ================================================
@app.route("/sensor", methods=["POST"])
def sensor():
    # request.get_json(): 요청 본문에서 JSON 파싱
    # 파싱 실패(빈 요청, 형식 오류)면 None 반환
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400  # 400 = Bad Request

    # 새 데이터를 버퍼에 추가 (오래된 것은 자동 제거)
    recent_buffer.append(data)

    # 데이터에서 각 값 추출
    # get_val(): 새 키(solar_power 등) 먼저 확인, 없으면 예전 키(power 등) 확인
    power = get_val(data, "solar_power", "power")
    soc   = int(get_val(data, "soc", default=100))  # 없으면 기본값 100 (완충 가정)

    # AI 추론 실행
    pred_power  = predict_power(data)  # GRU: 앞으로 발전량 예측 (없으면 None)
    water_alert = predict_water(       # 급수 AI: 물 줘야 하나?
        get_val(data, "soil",        default=50),
        get_val(data, "temperature", "temp", default=25),
        get_val(data, "humidity",    "hum",  default=50),
    )
    mode = decide_mode(power, pred_power, soc)  # SOC 기반 에너지 모드 결정 (0/1/2)

    # CSV에 저장 (나중에 AI 재학습 시 사용)
    save_csv(data, mode)

    # 콘솔에 수신 로그 출력 (디버깅 확인용)
    print(f"[수신] 토양:{data.get('soil')}% 조도:{data.get('light')}% "
          f"온도:{data.get('temperature', data.get('temp'))}° 발전:{power}W "
          f"→ mode:{mode} water:{water_alert} "
          f"pred:{f'{pred_power:.2f}W' if pred_power is not None else 'GRU대기중'}")

    # 아두이노에 JSON으로 응답
    # water_alert: True/False → 아두이노에서 1/0으로 파싱
    return jsonify({
        "mode":        mode,         # 에너지 모드 (0/1/2)
        "water_alert": water_alert,  # 급수 명령 (True/False)
        "pred_power":  round(pred_power, 2) if pred_power is not None else None
    })


# ================================================
# Flask 라우트: /data (GET)
#
# [역할]
#   저장된 CSV 파일에서 최근 50개 데이터를 JSON으로 반환
#   웹 대시보드에서 그래프 그릴 때 사용
#
# [접속 방법]
#   브라우저에서 http://서버IP:5000/data 입력
# ================================================
@app.route("/data", methods=["GET"])
def get_data():
    if not os.path.exists(CSV_FILE):
        return jsonify([])  # 파일 없으면 빈 배열 반환

    rows = []
    with open(CSV_FILE, "r") as f:
        # DictReader: CSV를 {"컬럼명": 값} 딕셔너리 형태로 읽음
        for row in csv.DictReader(f):
            rows.append(row)

    return jsonify(rows[-50:])  # 마지막 50개만 반환 (최신 데이터)


# ================================================
# Flask 라우트: /stats (GET)
#
# [역할]
#   전체 데이터의 통계 요약 반환
#   평균 온도, 평균 토양습도, 총 데이터 수 등
#
# [접속 방법]
#   http://서버IP:5000/stats
# ================================================
@app.route("/stats", methods=["GET"])
def stats():
    if not os.path.exists(CSV_FILE):
        return jsonify({"count": 0, "avg_temp": 0, "avg_soil": 0})

    rows = []
    with open(CSV_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return jsonify({"count": 0, "avg_temp": 0, "avg_soil": 0})

    return jsonify({
        "count":       len(rows),  # 총 저장된 데이터 수
        # generator expression: 리스트 만들지 않고 바로 합산 (메모리 절약)
        "avg_temp":    round(sum(float(r["temp"]) for r in rows) / len(rows), 1),
        "avg_soil":    round(sum(float(r["soil"]) for r in rows) / len(rows), 1),
        "gru_ready":   gru_ready,          # GRU 모델 로드 여부
        "water_ready": water_ready,        # 급수 모델 로드 여부
        "buffer_size": len(recent_buffer), # 현재 버퍼에 쌓인 데이터 수 (최대 30)
    })


# ================================================
# Flask 라우트: /status (GET)
#
# [역할]
#   모델 상태, 버퍼 상태, CSV 행 수를 한눈에 확인
#   "GRU가 로드됐나? 버퍼가 30개 찼나?" 빠르게 확인용
#
# [접속 방법]
#   http://서버IP:5000/status
#   → {"gru_model":"로드됨", "buffer":"25/30 (대기중)", ...}
# ================================================
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "gru_model":   "로드됨" if gru_ready   else "없음 (규칙 기반 동작 중)",
        "water_model": "로드됨" if water_ready  else "없음 (규칙 기반 동작 중)",
        # f-string으로 버퍼 현황 표시: "25/30 (GRU 추론 대기중)"
        "buffer":      f"{len(recent_buffer)}/{SEQ_LEN} "
                       f"(GRU 추론 {'가능' if len(recent_buffer) >= SEQ_LEN else '대기중'})",
        # 파일 있으면 줄 수 세기, 헤더 1줄 빼면 실제 데이터 수
        "csv_rows":    sum(1 for _ in open(CSV_FILE)) - 1
                       if os.path.exists(CSV_FILE) else 0,
    })


# ================================================
# Flask 라우트: /latest (GET)
#
# [역할]
#   가장 최근 데이터 1개를 JSON으로 반환
#   웹 대시보드에서 현재 상태 실시간 표시할 때 사용
# ================================================
@app.route("/latest", methods=["GET"])
def latest():
    if not os.path.exists(CSV_FILE):
        return jsonify({}), 404

    rows = []
    with open(CSV_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return jsonify({}), 404

    return jsonify(rows[-1])


# ================================================
# 서버 실행
#
# __name__ == "__main__" :
#   이 파일을 직접 실행할 때만 아래 코드 실행
#   import해서 쓸 때는 실행 안 됨
#
# app.run() 옵션:
#   host="0.0.0.0" : 모든 IP에서 접속 허용
#                    "127.0.0.1"이면 같은 컴퓨터에서만 접속 가능
#                    "0.0.0.0"이면 같은 WiFi의 아두이노도 접속 가능
#   port=5000      : 포트 번호 (아두이노의 SERVER_PORT와 동일해야 함!)
#   debug=True     : 코드 수정하면 서버 자동 재시작 (개발 편의용)
#                    실제 배포 시에는 False로 변경
# ================================================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)  # models 폴더 없으면 자동 생성
    app.run(host="0.0.0.0", port=5000, debug=True)