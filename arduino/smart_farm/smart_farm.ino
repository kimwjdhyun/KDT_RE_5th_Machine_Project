// =====================================================
// 센서: DHT11 + 토양습도 + LDR + LCD + ESP-01
// =====================================================
// [핀 배치]
//   D2  → ESP-01 TX
//   D3  → ESP-01 RX
//   D4  → DHT11 DATA (10kΩ 풀업저항 필요)
//   A1  → LDR (10kΩ 풀다운저항 필요)
//   A2  → 토양습도 AOUT
//   A4  → LCD SDA (I2C)
//   A5  → LCD SCL (I2C)
// =====================================================
// [라이브러리 설치]
//   DHT sensor library
//   LiquidCrystal I2C
// =====================================================

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

// ─────────────────────────────────────────
// 핀 설정
// ─────────────────────────────────────────
#define DHT_PIN     4
#define DHT_TYPE    DHT11
#define LDR_PIN     A1
#define SOIL_PIN    A2
#define ESP_TX      2    // ESP TX → 아두이노 D2
#define ESP_RX      3    // ESP RX → 아두이노 D3

// ─────────────────────────────────────────
// 상수
// ─────────────────────────────────────────
#define SEND_INTERVAL  5000   // 서버 전송 주기 (5초)
#define LCD_INTERVAL   1000   // LCD 갱신 주기 (1초)
#define LIGHT_THRESHOLD 30    // 조도 임계값 (%) — 이하면 밤
#define WIFI_SSID      "spreatics_gusan_cctv"      // WiFi
#define WIFI_PASS      "spreatics*"  // 비밀번호
#define SERVER_IP      "192.168.201.104"      // Flask 서버 PC IP
#define SERVER_PORT    "5000"

// ─────────────────────────────────────────
// 객체 선언
// ─────────────────────────────────────────
DHT               dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);  // 안 되면 0x3F 로 변경
SoftwareSerial    espSerial(ESP_TX, ESP_RX);

// ─────────────────────────────────────────
// 전역 변수
// ─────────────────────────────────────────
unsigned long lastSendTime = 0;
unsigned long lastLcdTime  = 0;
int           energyMode   = 2;
bool          waterAlert   = false;


// =====================================================
// 센서 계산 함수
// =====================================================

// 조도 (0~100%)
int calcLight() {
  return map(analogRead(LDR_PIN), 0, 1023, 0, 100);
}

// 토양습도 (0~100%)
int calcSoil() {
  int raw = constrain(analogRead(SOIL_PIN), 300, 1023);
  return map(raw, 300, 1023, 100, 0);
}


// =====================================================
// LCD 표시
// 줄1: 온도 + 습도
// 줄2: 토양 + 조도 + 모드
// =====================================================
void updateLCD(float temp, float hum, int soil, int light) {
  // ── 줄 1: 온도 + 습도 ──
  lcd.setCursor(0, 0);
  lcd.print("T:");
  if (isnan(temp)) {
    lcd.print("--");
  } else {
    lcd.print((int)temp);
  }
  lcd.print("C ");
  lcd.print("H:");
  if (isnan(hum)) {
    lcd.print("--");
  } else {
    lcd.print((int)hum);
  }
  lcd.print("%  ");

  // ── 줄 2: 토양 + 조도 + 낮밤 ──
  lcd.setCursor(0, 1);
  lcd.print("S:");
  lcd.print(soil);
  lcd.print("% ");
  lcd.print("L:");
  lcd.print(light);
  lcd.print("% ");
  if (light >= LIGHT_THRESHOLD) {
    lcd.print("DAY");
  } else {
    lcd.print("NGT");
  }
}


// =====================================================
// ESP-01 AT 명령 전송
// =====================================================
void sendAT(String cmd, int waitMs) {
  espSerial.println(cmd);
  delay(waitMs);
  while (espSerial.available()) {
    Serial.write(espSerial.read());
  }
}


// =====================================================
// Flask 서버에 JSON 전송
// =====================================================
void sendToServer(float temp, float hum, int soil, int light) {
  String json = "{";
  json += "\"temp\":"  + String(isnan(temp) ? 0 : temp, 1) + ",";
  json += "\"hum\":"   + String(isnan(hum)  ? 0 : hum,  1) + ",";
  json += "\"soil\":"  + String(soil)  + ",";
  json += "\"light\":" + String(light);
  json += "}";

  String httpReq  = "POST /sensor HTTP/1.1\r\n";
  httpReq += "Host: " + String(SERVER_IP) + "\r\n";
  httpReq += "Content-Type: application/json\r\n";
  httpReq += "Content-Length: " + String(json.length()) + "\r\n";
  httpReq += "Connection: close\r\n\r\n";
  httpReq += json;

  sendAT("AT+CIPSTART=\"TCP\",\"" + String(SERVER_IP) + "\"," + SERVER_PORT, 1000);
  sendAT("AT+CIPSEND=" + String(httpReq.length()), 500);
  espSerial.print(httpReq);

  // 응답 읽기
  String response = "";
  unsigned long start = millis();
  while (millis() - start < 3000) {
    if (espSerial.available()) response += (char)espSerial.read();
  }

  // mode 파싱
  int idx = response.indexOf("\"mode\":");
  if (idx != -1) energyMode = response.substring(idx + 7, idx + 8).toInt();

  // water_alert 파싱
  idx = response.indexOf("\"water_alert\":");
  if (idx != -1) waterAlert = response.substring(idx + 14, idx + 18).indexOf("true") != -1;

  sendAT("AT+CIPCLOSE", 300);

  Serial.print("[서버] mode:"); Serial.print(energyMode);
  Serial.print(" water:"); Serial.println(waterAlert);
}


// =====================================================
// setup
// =====================================================
void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  // 센서 초기화
  dht.begin();
  Wire.begin();
  lcd.init();
  lcd.backlight();

  // 시작 화면
  lcd.setCursor(0, 0);
  lcd.print("SmartFarm v1.0");
  lcd.setCursor(0, 1);
  lcd.print("Connecting...");

  // ESP-01 WiFi 연결
  delay(1000);
  sendAT("AT+RST",    3000);
  sendAT("AT+CWMODE=1", 1000);
  sendAT("AT+CWJAP=\"" + String(WIFI_SSID) + "\",\"" + String(WIFI_PASS) + "\"", 8000);

  // WiFi 결과 LCD 표시
  lcd.setCursor(0, 1);
  lcd.print("WiFi OK!       ");
  delay(1500);
  lcd.clear();

  Serial.println("시작 완료!");
}


// =====================================================
// loop
// =====================================================
void loop() {
  // ── 센서 읽기 ──
  float temp  = dht.readTemperature();
  float hum   = dht.readHumidity();
  int   light = calcLight();
  int   soil  = calcSoil();

  // ── 시리얼 디버그 출력 ──
  Serial.print("온도:"); Serial.print(isnan(temp) ? 0 : temp);
  Serial.print(" 습도:"); Serial.print(isnan(hum) ? 0 : hum);
  Serial.print(" 조도:"); Serial.print(light);
  Serial.print(" 토양:"); Serial.println(soil);

  // ── LCD 1초마다 갱신 ──
  if (millis() - lastLcdTime >= LCD_INTERVAL) {
    lastLcdTime = millis();
    updateLCD(temp, hum, soil, light);
  }

  // ── 서버 전송 5초마다 ──
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();
    sendToServer(temp, hum, soil, light);
  }
}
