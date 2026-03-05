from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from datetime import datetime
import threading
import time

app = Flask(__name__)
CORS(app)

data_log = []

# --- 시뮬레이터 상태 ---
simulator_on = False
simulator_thread = None


def append_log(item: dict):
    """공통 로그 저장(최대 200개 유지)"""
    global data_log
    data_log.append(item)
    data_log = data_log[-200:]


def make_record(power: float, soc: int, soil: int):
    record = {
        "power": round(float(power), 2),
        "soc": int(soc),
        "soil": int(soil),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    # 🤖 AI 예측(더미): 현재 발전량 주변으로 살짝 흔들림
    record["pred_1h"] = round(record["power"] * random.uniform(0.9, 1.1), 2)

    # ⚡ 모드 계산
    if record["soc"] < 20:
        record["mode"] = 0
    elif record["soc"] < 60:
        record["mode"] = 1
    else:
        record["mode"] = 2

    return record


def simulator_loop():
    """5초마다 자동으로 가짜 센서 데이터를 만들어 data_log에 저장"""
    global simulator_on

    # 흐름이 예쁘게 보이도록 약간 '연속성' 있는 값으로 만들기
    power = 5.0
    soc = 80
    soil = 55

    while simulator_on:
        # 발전량: 약간씩 오르내리는 형태
        power = max(0, min(15, power + random.uniform(-1.2, 1.2)))
        # SOC: 서서히 떨어졌다가(발표용) 가끔 회복
        soc = max(5, min(100, soc + random.randint(-3, 2)))
        # 토양습도: 천천히 마르다가 가끔 상승(물 준 것처럼)
        soil = max(10, min(90, soil + random.randint(-4, 2)))

        # 가끔 극적인 장면(모드 전환) 만들기
        if random.random() < 0.08:
            soc = random.choice([15, 35, 75])  # 🔴/🟡/🟢 장면용

        rec = make_record(power, soc, soil)
        append_log(rec)
        time.sleep(5)


# --- 센서 수신 (실제 하드웨어가 붙으면 이 엔드포인트 사용) ---
@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.json or {}
    power = data.get("power", 0)
    soc = data.get("soc", 50)
    soil = data.get("soil", 50)

    rec = make_record(power, soc, soil)
    append_log(rec)

    return jsonify({
        "mode": rec["mode"],
        "water_alert": 1 if rec["soil"] < 40 else 0
    })


# --- 최근 데이터 ---
@app.route("/data")
def get_data():
    return jsonify(data_log[-50:])


# --- 최신 예측값 ---
@app.route("/predict")
def predict():
    if not data_log:
        return jsonify({"pred_1h": 0})
    return jsonify({"pred_1h": data_log[-1]["pred_1h"]})


# --- 통계/ESG ---
@app.route("/stats")
def stats():
    total_power = sum(d.get("power", 0) for d in data_log)
    return jsonify({
        "total_generation": round(total_power, 2),
        "carbon_reduction_g": round(total_power * 0.5, 2)
    })


# --- 고정값 카드용(선택): 현재는 더미 유지 ---
@app.route("/api/energy")
def energy():
    return jsonify({
        "power": 128.5,
        "soc": 76,
        "carbon_saved": 32.4
    })


# --- 시뮬레이터 제어 ---
@app.route("/sim/start", methods=["POST"])
def sim_start():
    global simulator_on, simulator_thread
    if simulator_on:
        return jsonify({"status": "already_running"})
    simulator_on = True
    simulator_thread = threading.Thread(target=simulator_loop, daemon=True)
    simulator_thread.start()
    return jsonify({"status": "started"})


@app.route("/sim/stop", methods=["POST"])
def sim_stop():
    global simulator_on
    simulator_on = False
    return jsonify({"status": "stopped"})


@app.route("/sim/status", methods=["GET"])
def sim_status():
    return jsonify({"running": simulator_on, "log_len": len(data_log)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)