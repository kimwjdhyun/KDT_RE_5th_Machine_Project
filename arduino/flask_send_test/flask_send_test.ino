// =====================================================
// flask_send_test.ino
// Flask 서버 전송 테스트 코드
// 센서값을 Flask 서버에 보내고 응답 확인
// =====================================================
// [핀 배치]
//   D2  → ESP-01 TX
//   D3  → ESP-01 RX
//   D4  → DHT11 DATA
//   A1  → LDR
//   A2  → 토양습도 AOUT
//   A4  → LCD SDA
//   A5  → LCD SCL
// =====================================================

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

// ─────────────────────────────────────────
// 핀 설정
// ─────────────────────────────────────────
#define DHT_PIN      4
#define DHT_TYPE     DHT11
#define LDR_PIN      A1
#define SOIL_PIN     A2
#define ESP_TX       2
#define ESP_RX       3

// ─────────────────────────────────────────
// 설정값 
// ─────────────────────────────────────────
#define WIFI_SSID     "spreatics_gusan_cctv"
#define WIFI_PASS     "spreatics*"
#define SERVER_IP     "192.168.201.106"   // Flask 켜진 PC IP
#define SERVER_PORT   "5000"
#define SEND_INTERVAL  5000               // 전송 주기 (5초)
#define LCD_INTERVAL   1000               // LCD 갱신 주기 (1초)
#define LIGHT_THRESHOLD 30                // 조도 임계값 (%)

// ─────────────────────────────────────────
// 객체
// ─────────────────────────────────────────
DHT               dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
SoftwareSerial    espSerial(ESP_TX, ESP_RX);

// ─────────────────────────────────────────
// 전역 변수
// ─────────────────────────────────────────
unsigned long lastSendTime = 0;
unsigned long lastLcdTime  = 0;
int           energyMode   = 2;
bool          waterAlert   = false;
bool          wifiOK       = false;


// =====================================================
// 센서 계산
// =====================================================
int calcLight() {
  return map(analogRead(LDR_PIN), 0, 1023, 0, 100);
}

int calcSoil() {
  int raw = constrain(analogRead(SOIL_PIN), 300, 1023);
  return map(raw, 300, 1023, 100, 0);
}


// =====================================================
// LCD 표시
// 줄1: T:22C H:55%
// 줄2: S:47% L:51% DAY
// =====================================================
void updateLCD(float temp, float hum, int soil, int light) {
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(isnan(temp) ? 0 : (int)temp);
  lcd.print("C H:");
  lcd.print(isnan(hum) ? 0 : (int)hum);
  lcd.print("%   ");

  lcd.setCursor(0, 1);
  lcd.print("S:");
  lcd.print(soil);
  lcd.print("% L:");
  lcd.print(light);
  lcd.print("% ");
  lcd.print(light >= LIGHT_THRESHOLD ? "DAY" : "NGT");
}


// =====================================================
// AT 명령 전송
// =====================================================
void sendAT(String cmd, int waitMs) {
  espSerial.println(cmd);
  delay(waitMs);
  while (espSerial.available()) {
    Serial.write(espSerial.read());
  }
}


// =====================================================
// Flask 서버에 JSON POST 전송
// 전송: {temp, hum, soil, light}
// 수신: {mode, water_alert}
// =====================================================
void sendToServer(float temp, float hum, int soil, int light) {
  if (!wifiOK) {
    Serial.println("[전송 스킵] WiFi 미연결");
    return;
  }

  // JSON 조립
  String json = "{";
  json += "\"temp\":"  + String(isnan(temp) ? 0.0 : temp, 1) + ",";
  json += "\"hum\":"   + String(isnan(hum)  ? 0.0 : hum,  1) + ",";
  json += "\"soil\":"  + String(soil) + ",";
  json += "\"light\":" + String(light);
  json += "}";

  // HTTP POST 요청 조립
  String req  = "POST /sensor HTTP/1.1\r\n";
  req += "Host: " + String(SERVER_IP) + "\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Content-Length: " + String(json.length()) + "\r\n";
  req += "Connection: close\r\n\r\n";
  req += json;

  Serial.println("[전송] " + json);

  // TCP 연결 → 전송 → 응답 수신
  sendAT("AT+CIPSTART=\"TCP\",\"" + String(SERVER_IP) + "\"," + SERVER_PORT, 1000);
  sendAT("AT+CIPSEND=" + String(req.length()), 500);
  espSerial.print(req);

  // 응답 읽기 (3초)
  String res = "";
  unsigned long t = millis();
  while (millis() - t < 3000) {
    if (espSerial.available()) res += (char)espSerial.read();
  }

  // mode 파싱
  int idx = res.indexOf("\"mode\":");
  if (idx != -1) {
    energyMode = res.substring(idx + 7, idx + 8).toInt();
  }

  // water_alert 파싱
  idx = res.indexOf("\"water_alert\":");
  if (idx != -1) {
    waterAlert = res.substring(idx + 14, idx + 18).indexOf("true") != -1;
  }

  sendAT("AT+CIPCLOSE", 300);

  // 결과 출력
  Serial.print("[응답] mode:");
  Serial.print(energyMode);
  Serial.print(" water_alert:");
  Serial.println(waterAlert ? "true" : "false");

  // LCD에 서버 응답 잠깐 표시
  lcd.setCursor(0, 1);
  lcd.print("M:");
  lcd.print(energyMode);
  lcd.print(energyMode == 0 ? " 긴급" : energyMode == 1 ? " 절약" : " 풀가");
  lcd.print(waterAlert ? " W!" : "    ");
  delay(1500);
}


// =====================================================
// WiFi 연결
// =====================================================
bool connectWifi() {
  Serial.println("[WiFi] 연결 중...");
  lcd.setCursor(0, 1);
  lcd.print("WiFi 연결중...  ");

  sendAT("AT+RST",      3000);
  sendAT("AT+CWMODE=1", 1000);
  sendAT("AT+CWJAP=\"" + String(WIFI_SSID) + "\",\"" + String(WIFI_PASS) + "\"", 8000);

  // IP 확인
  espSerial.println("AT+CIFSR");
  delay(2000);
  String res = "";
  unsigned long t = millis();
  while (millis() - t < 2000) {
    if (espSerial.available()) res += (char)espSerial.read();
  }

  if (res.indexOf("STAIP") != -1) {
    Serial.println("[WiFi] 연결 성공!");
    lcd.setCursor(0, 1);
    lcd.print("WiFi OK!        ");
    delay(1500);
    return true;
  } else {
    Serial.println("[WiFi] 연결 실패");
    lcd.setCursor(0, 1);
    lcd.print("WiFi FAIL       ");
    delay(1500);
    return false;
  }
}


// =====================================================
// setup
// =====================================================
void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  dht.begin();
  Wire.begin();
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("SmartFarm v1.0  ");
  lcd.setCursor(0, 1);
  lcd.print("Starting...     ");
  delay(1000);

  wifiOK = connectWifi();
  lcd.clear();
  Serial.println("[시스템] 시작 완료!");
}


// =====================================================
// loop
// =====================================================
void loop() {
  float temp  = dht.readTemperature();
  float hum   = dht.readHumidity();
  int   light = calcLight();
  int   soil  = calcSoil();

  // 시리얼 출력
  Serial.print("온도:"); Serial.print(isnan(temp) ? 0 : temp);
  Serial.print(" 습도:"); Serial.print(isnan(hum) ? 0 : hum);
  Serial.print(" 조도:"); Serial.print(light);
  Serial.print(" 토양:"); Serial.println(soil);

  // LCD 갱신 (1초마다)
  if (millis() - lastLcdTime >= LCD_INTERVAL) {
    lastLcdTime = millis();
    updateLCD(temp, hum, soil, light);
  }

  // 서버 전송 (5초마다)
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();
    sendToServer(temp, hum, soil, light);
  }
}
