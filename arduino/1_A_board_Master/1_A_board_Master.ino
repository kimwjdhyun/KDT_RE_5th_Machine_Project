#include <DHT.h>
#include <Wire.h>
#include <SoftwareSerial.h>

#define DHT_PIN         4
#define DHT_TYPE        DHT11
#define LDR_PIN         A1
#define SOIL_PIN        A2

// SoftwareSerial(rx, tx)
// UNO D2 <- ESP TX
// UNO D3 -> ESP RX  (분압 권장)
#define ESP_RX_PIN      2
#define ESP_TX_PIN      3

#define WIFI_SSID       "spreatics_gusan_cctv"
#define WIFI_PASS       "spreatics*"

// 실제 Flask 서버 IP로 수정
#define SERVER_IP       "192.168.201.138"
#define SERVER_PORT     "5000"

// 5분 전송 권장
#define SEND_INTERVAL   300000UL

#define LIGHT_THRESHOLD 30
#define SLAVE_ADDR      8

struct PowerData {
  int voltage_x100;      // V * 100
  int current_mA_x10;    // mA * 10
  int power_mW_x10;      // mW * 10
  int pumpState;
  int ledState;
  int soc;
};

DHT dht(DHT_PIN, DHT_TYPE);
SoftwareSerial espSerial(ESP_RX_PIN, ESP_TX_PIN);

unsigned long lastSendTime  = 0;
unsigned long lastPrintTime = 0;

int       energyMode = 2;
bool      waterAlert = false;
bool      wifiOK     = false;
PowerData powerData;


// ----------------------------
// 센서 계산
// ----------------------------
int calcLight() {
  return map(analogRead(LDR_PIN), 0, 1023, 0, 100);
}

int calcSoil() {
  int raw = constrain(analogRead(SOIL_PIN), 300, 1023);
  return map(raw, 300, 1023, 100, 0);
}


// ----------------------------
// 슬레이브(B)에서 전력 데이터 수신
// ----------------------------
void receiveFromB() {
  Wire.requestFrom(SLAVE_ADDR, (int)sizeof(PowerData));

  uint8_t *ptr = (uint8_t*)&powerData;
  int i = 0;

  while (Wire.available() && i < (int)sizeof(PowerData)) {
    ptr[i++] = Wire.read();
  }

  float voltage_V = powerData.voltage_x100 / 100.0;
  float current_A = powerData.current_mA_x10 / 10000.0; // (mA*10) -> A
  float power_W   = powerData.power_mW_x10 / 10000.0;   // (mW*10) -> W

  Serial.print("[B수신] 전압: ");
  Serial.print(voltage_V, 2);
  Serial.print(" V / 전류: ");
  Serial.print(current_A, 3);
  Serial.print(" A / 전력: ");
  Serial.print(power_W, 3);
  Serial.println(" W");
}


// ----------------------------
// 슬레이브(B)로 제어 명령 전송
// ----------------------------
void sendToB() {
  byte cmd = (energyMode << 1) | (waterAlert ? 1 : 0);

  Wire.beginTransmission(SLAVE_ADDR);
  Wire.write(cmd);
  Wire.endTransmission();

  Serial.print("[B전송] mode: ");
  Serial.print(energyMode);
  Serial.print(" / water: ");
  Serial.println(waterAlert ? "true" : "false");
}


// ----------------------------
// ESP-01에 AT 명령 전송
// ----------------------------
void sendAT(String cmd, int waitMs) {
  espSerial.println(cmd);
  delay(waitMs);

  while (espSerial.available()) {
    Serial.write(espSerial.read());
  }
}


// ----------------------------
// WiFi 연결
// ----------------------------
bool connectWifi() {
  Serial.println("[WiFi] 연결 중...");

  sendAT("AT+RST", 3000);
  sendAT("AT+CWMODE=1", 1000);
  sendAT("AT+CIPMUX=0", 1000);

  sendAT("AT+CWJAP=\"" + String(WIFI_SSID) + "\",\"" + String(WIFI_PASS) + "\"", 8000);

  espSerial.println("AT+CIFSR");
  delay(2000);

  String res = "";
  unsigned long t = millis();
  while (millis() - t < 2000) {
    if (espSerial.available()) {
      res += (char)espSerial.read();
    }
  }

  Serial.println(res);

  if (res.indexOf("STAIP") != -1) {
    Serial.println("[WiFi] 연결 성공!");
    return true;
  }

  Serial.println("[WiFi] 연결 실패");
  return false;
}


// ----------------------------
// 서버 전송
// ----------------------------
void sendToServer(float temp, float hum, int soil, int light) {
  if (!wifiOK) {
    Serial.println("[전송 스킵] WiFi 미연결");
    return;
  }

  // 슬레이브에서 받은 값을 서버 기준 단위로 변환
  float voltage_V = powerData.voltage_x100 / 100.0;
  float current_A = powerData.current_mA_x10 / 10000.0; // A 기준
  float power_W   = powerData.power_mW_x10 / 10000.0;   // W 기준

  String json = "{";
  json += "\"temp\":"    + String(isnan(temp) ? 0.0 : temp, 1) + ",";
  json += "\"hum\":"     + String(isnan(hum)  ? 0.0 : hum,  1) + ",";
  json += "\"soil\":"    + String(soil) + ",";
  json += "\"light\":"   + String(light) + ",";
  json += "\"voltage\":" + String(voltage_V, 2) + ",";
  json += "\"current\":" + String(current_A, 3) + ",";
  json += "\"power\":"   + String(power_W, 3) + ",";
  json += "\"soc\":"     + String(powerData.soc) + ",";
  json += "\"pump\":"    + String(powerData.pumpState) + ",";
  json += "\"led\":"     + String(powerData.ledState);
  json += "}";

  String req  = "POST /sensor HTTP/1.1\r\n";
  req += "Host: " + String(SERVER_IP) + ":" + String(SERVER_PORT) + "\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Connection: close\r\n";
  req += "Content-Length: " + String(json.length()) + "\r\n";
  req += "\r\n";
  req += json;

  Serial.println("[전송 JSON] " + json);

  // 이전 연결 정리
  sendAT("AT+CIPCLOSE", 300);

  // TCP 연결 시작
  sendAT("AT+CIPSTART=\"TCP\",\"" + String(SERVER_IP) + "\"," + SERVER_PORT, 2000);

  // 전송 길이 알림
  sendAT("AT+CIPSEND=" + String(req.length()), 1000);

  // 실제 요청 전송
  espSerial.print(req);

  String res = "";
  unsigned long t = millis();
  while (millis() - t < 5000) {
    while (espSerial.available()) {
      res += (char)espSerial.read();
    }
  }

  Serial.println("[서버응답]");
  Serial.println(res);

  // 서버 응답에서 mode 추출
  int idx = res.indexOf("\"mode\":");
  if (idx != -1) {
    energyMode = res.substring(idx + 7, idx + 8).toInt();
  }

  // 서버 응답에서 water_alert 추출
  idx = res.indexOf("\"water_alert\":");
  if (idx != -1) {
    waterAlert = res.substring(idx + 14, idx + 15).toInt() == 1;
  }

  sendAT("AT+CIPCLOSE", 300);

  Serial.print("[응답 반영] mode: ");
  Serial.print(energyMode);
  Serial.print(" / water: ");
  Serial.println(waterAlert ? "true" : "false");
}


void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  dht.begin();
  Wire.begin();

  memset(&powerData, 0, sizeof(PowerData));

  Serial.println("[MASTER] 시작!");
  wifiOK = connectWifi();
}


void loop() {
  float temp  = dht.readTemperature();
  float hum   = dht.readHumidity();
  int   light = calcLight();
  int   soil  = calcSoil();

  if (millis() - lastPrintTime >= 10000) {
    lastPrintTime = millis();

    Serial.print("온도: ");
    Serial.print(isnan(temp) ? 0 : temp);

    Serial.print(" / 습도: ");
    Serial.print(isnan(hum) ? 0 : hum);

    Serial.print(" / 조도: ");
    Serial.print(light);

    Serial.print(" / 토양: ");
    Serial.println(soil);
  }

  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    // 슬레이브에서 전력/상태값 수신
    receiveFromB();

    // 서버로 전송
    sendToServer(temp, hum, soil, light);

    // 서버 응답 기준으로 슬레이브에 제어 명령 전송
    sendToB();
  }
}