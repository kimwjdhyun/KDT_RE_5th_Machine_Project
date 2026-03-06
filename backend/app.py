from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from datetime import datetime
import threading
import time

app = Flask(__name__)
CORS(app)

# ----------------------------
# In-memory log
# ----------------------------
data_log = []

# ----------------------------
# Demo mode worker
# ----------------------------
demo_running = False
demo_thread = None
demo_lock = threading.Lock()


def calc_mode(soc: float) -> int:
    if soc < 20:
        return 0
    elif soc < 60:
        return 1
    else:
        return 2


def append_row(row: dict):
    """Keep only last 300 rows to prevent runaway memory."""
    data_log.append(row)
    if len(data_log) > 300:
        del data_log[:-300]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def demo_worker():
    """
    Every 2 seconds generate a new sensor-like record.
    Keeps running while demo_running == True.
    """
    global demo_running

    # 시작값: 그럴듯한 범위로 초기화
    soc = 75.0            # %
    power = 120.0         # kW
    soil = 55.0           # %
    temp = 24.0           # °C
    hum = 55.0            # %
    light = 65.0          # %

    while True:
        with demo_lock:
            if not demo_running:
                break

        # ----------------------------
        # Power / Light relationship (대충이라도 연동 느낌)
        # ----------------------------
        # 조도는 천천히 출렁이고
        light += random.uniform(-4, 4)
        light = clamp(light, 0, 100)

        # 발전량은 조도 영향을 받는 것처럼
        power += (light - 55) * 0.25 + random.uniform(-6, 6)
        power = clamp(power, 0, 220)

        # SOC: 발전량이 높으면 조금 충전, 낮으면 방전 느낌
        soc += (power - 110) * 0.01 + random.uniform(-0.5, 0.3)
        soc = clamp(soc, 0, 100)

        # 토양습도: 서서히 감소하다가 가끔 급수 이벤트
        soil -= random.uniform(0.3, 1.2)
        if soil < 30 and random.random() < 0.35:
            soil += random.uniform(15, 25)
        soil = clamp(soil, 0, 100)

        # 온도/습도: 완만한 랜덤 워크
        temp += random.uniform(-0.25, 0.35)
        temp = clamp(temp, 16, 34)

        hum += random.uniform(-1.8, 1.2)
        hum = clamp(hum, 20, 95)

        # AI 예측 더미(현 power 기준 +- 10%)
        pred_1h = round(power * random.uniform(0.9, 1.1), 2)
        mode = calc_mode(soc)

        row = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "power": round(power, 2),
            "soc": round(soc, 1),
            "soil": round(soil, 1),

            # ✅ 추가된 3개 필드
            "temp": round(temp, 1),
            "hum": round(hum, 1),
            "light": round(light, 0),

            "pred_1h": pred_1h,
            "mode": mode
        }

        append_row(row)
        time.sleep(2)


# ----------------------------
# API
# ----------------------------
@app.route("/sensor", methods=["POST"])
def sensor():
    """
    Hardware/Arduino/ESP side should POST JSON here.

    Example JSON:
    {
      "power": 123.4,
      "soc": 76,
      "soil": 52,
      "temp": 25.3,
      "hum": 58.1,
      "light": 70
    }
    """
    data = request.json or {}
    data["timestamp"] = datetime.now().strftime("%H:%M:%S")

    power = float(data.get("power", 0) or 0)
    soc = float(data.get("soc", 50) or 50)
    soil = float(data.get("soil", 50) or 50)

    # ✅ 추가 필드도 수신 (없으면 기본값)
    temp = float(data.get("temp", 0) or 0)
    hum = float(data.get("hum", 0) or 0)
    light = float(data.get("light", 0) or 0)

    # 🤖 AI 예측 더미
    data["pred_1h"] = round(power * random.uniform(0.9, 1.1), 2)

    # ⚡ 에너지 모드 계산
    data["mode"] = calc_mode(soc)

    # ✅ 값 표준화(프론트 표기 안정성)
    data["power"] = round(power, 2)
    data["soc"] = round(soc, 1)
    data["soil"] = round(soil, 1)
    data["temp"] = round(temp, 1)
    data["hum"] = round(hum, 1)
    data["light"] = round(light, 0)

    append_row(data)

    return jsonify({
        "mode": data["mode"],
        "water_alert": 1 if soil < 40 else 0
    })


@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(data_log[-50:])


@app.route("/stats", methods=["GET"])
def stats():
    total_power = sum(float(d.get("power", 0) or 0) for d in data_log)
    return jsonify({
        "total_generation": round(total_power, 2),
        "carbon_reduction_g": round(total_power * 0.5, 2)
    })


# ----------------------------
# Demo endpoints
# ----------------------------
@app.route("/sim/start", methods=["POST"])
def sim_start():
    global demo_running, demo_thread

    with demo_lock:
        if demo_running:
            return jsonify({"status": "ok", "message": "데모 모드가 이미 실행 중입니다 🎬"})
        demo_running = True

    demo_thread = threading.Thread(target=demo_worker, daemon=True)
    demo_thread.start()

    return jsonify({"status": "ok", "message": "데모 모드 시작! 2초마다 데이터가 생성됩니다 🎬"})


@app.route("/sim/stop", methods=["POST"])
def sim_stop():
    global demo_running
    with demo_lock:
        demo_running = False
    return jsonify({"status": "ok", "message": "데모 모드 정지! 🧊"})


@app.route("/sim/status", methods=["GET"])
def sim_status():
    with demo_lock:
        running = demo_running
    return jsonify({"running": running, "log_size": len(data_log)})


# (선택) 빠른 확인용
@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "백엔드 살아있음 ✅"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)