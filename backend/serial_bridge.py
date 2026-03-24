# serial_bridge.py
# backend/ 폴더에 저장하고 실행

import serial
import requests
import json
import time

SERIAL_PORT = "COM3"
BAUD_RATE   = 9600
SERVER_URL  = "http://localhost:5000/sensor"

print(f"[브릿지] {SERIAL_PORT} 연결 중...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("[브릿지] 시작! Flask로 전송 중...")

while True:
    try:
        line = ser.readline().decode("utf-8").strip()

        if line.startswith("{"):
            data = json.loads(line)
            res  = requests.post(SERVER_URL, json=data, timeout=3)
            print(f"[전송] {data.get('temperature')}° 발전:{data.get('solar_power')}W → {res.status_code}")

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"[오류] {e}")
        time.sleep(1)